"""Feature vector generation and FAISS index management.

Converts song attributes (audio features, tags, listening stats) into
dense feature vectors and maintains a FAISS index for similarity search.
"""

from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from src.data.db import get_project_root, load_config
from src.data.models import Song
from src.features.mood import build_tag_vocabulary, parse_tags


def build_feature_vectors(session: Session) -> dict[str, np.ndarray]:
    """Build feature vectors for all songs from audio features, tags, and stats.

    Feature composition:
    - Audio features (12 dims, normalized): danceability, energy, valence, tempo,
      acousticness, instrumentalness, speechiness, liveness, loudness, key, mode,
      time_signature — filled with 0.5 defaults if unavailable
    - Tag vector (N dims, sparse->dense via vocabulary)
    - Listening stats (4 dims): log(total_plays+1), recent_ratio, staleness, liked

    Returns dict mapping spotify_id -> numpy array.
    """
    vocabulary = build_tag_vocabulary(session, min_count=2)
    vocab_size = len(vocabulary)

    songs = session.query(Song).all()
    vectors: dict[str, np.ndarray] = {}

    for song in songs:
        features: list[float] = []

        # Audio features (12 dims, normalized 0-1)
        audio_attrs = [
            "danceability", "energy", "valence", "acousticness",
            "instrumentalness", "speechiness", "liveness",
        ]
        for attr in audio_attrs:
            val = getattr(song, attr, None)
            features.append(val if val is not None else 0.5)

        # Tempo: normalize to 0-1 range (assume 60-200 BPM)
        tempo = song.tempo
        if tempo is not None:
            features.append(max(0.0, min(1.0, (tempo - 60) / 140)))
        else:
            features.append(0.5)

        # Loudness: normalize to 0-1 (assume -60 to 0 dB)
        loudness = song.loudness
        if loudness is not None:
            features.append(max(0.0, min(1.0, (loudness + 60) / 60)))
        else:
            features.append(0.5)

        # Key: normalize to 0-1 (0-11)
        key = song.key
        features.append(key / 11.0 if key is not None else 0.5)

        # Mode: 0 or 1
        mode = song.mode
        features.append(float(mode) if mode is not None else 0.5)

        # Time signature: normalize (assume 3-7)
        ts = song.time_signature
        if ts is not None:
            features.append(max(0.0, min(1.0, (ts - 3) / 4)))
        else:
            features.append(0.5)

        # Tag vector (vocab_size dims)
        tags = parse_tags(song.lastfm_tags)
        tag_vec = [0.0] * vocab_size
        for tag in tags:
            if tag in vocabulary:
                tag_vec[vocabulary[tag]] = 1.0
        features.extend(tag_vec)

        # Listening stats (4 dims)
        total = song.total_plays or 0
        recent = song.recent_plays_30d or 0
        staleness = song.staleness_score if song.staleness_score is not None else 0.5
        liked = 1.0 if song.in_liked_songs else 0.0

        features.append(min(np.log2(1 + total) / 10.0, 1.0))  # log plays normalized
        features.append(min(recent / max(total, 1) * 5, 1.0))  # recent ratio
        features.append(staleness)
        features.append(liked)

        vectors[song.spotify_id] = np.array(features, dtype=np.float32)

    return vectors


def build_faiss_index(vectors: dict[str, np.ndarray], save_path: str | None = None):
    """Build a FAISS index from feature vectors and optionally save to disk.

    Returns (index, id_list) where id_list maps FAISS row -> spotify_id.
    """
    import faiss

    if not vectors:
        return None, []

    id_list = list(vectors.keys())
    matrix = np.stack([vectors[sid] for sid in id_list]).astype(np.float32)
    dim = matrix.shape[1]

    # L2 normalized for cosine similarity via inner product
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(path))
        # Save id list alongside
        id_path = path.with_suffix(".ids.npy")
        np.save(str(id_path), np.array(id_list))

    return index, id_list


def load_faiss_index(path: str):
    """Load a previously saved FAISS index from disk.

    Returns (index, id_list).
    """
    import faiss

    index = faiss.read_index(path)
    id_path = Path(path).with_suffix(".ids.npy")
    id_list = np.load(str(id_path), allow_pickle=True).tolist()
    return index, id_list


def find_similar(
    index, id_list: list[str], query_vector: np.ndarray, k: int = 10
) -> list[tuple[str, float]]:
    """Find k most similar songs to query_vector.

    Returns list of (spotify_id, similarity_score) pairs, highest first.
    """
    import faiss

    query = query_vector.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(query)
    distances, indices = index.search(query, k)

    results: list[tuple[str, float]] = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(id_list):
            continue
        results.append((id_list[idx], float(dist)))

    return results

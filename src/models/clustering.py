"""HDBSCAN + UMAP clustering.

Clusters songs by feature similarity using UMAP for dimensionality reduction
and HDBSCAN for density-based clustering.
"""

import numpy as np
from rich.console import Console
from sqlalchemy.orm import Session

from src.data.db import load_config
from src.data.models import Cluster, Song

console = Console()


def reduce_dimensions(
    vectors: dict[str, np.ndarray], n_components: int = 10
) -> tuple[dict[str, np.ndarray], object]:
    """Reduce feature vectors with UMAP.

    Returns (reduced_vectors dict, umap_model).
    """
    import umap

    if not vectors:
        return {}, None

    id_list = list(vectors.keys())
    matrix = np.stack([vectors[sid] for sid in id_list]).astype(np.float32)

    # Clamp n_components to available dimensions and samples
    n_components = min(n_components, matrix.shape[1], matrix.shape[0] - 1)
    if n_components < 2:
        n_components = 2

    reducer = umap.UMAP(
        n_components=n_components,
        metric="cosine",
        n_neighbors=min(15, matrix.shape[0] - 1),
        min_dist=0.1,
        random_state=42,
    )
    reduced = reducer.fit_transform(matrix)

    reduced_vectors = {sid: reduced[i] for i, sid in enumerate(id_list)}
    return reduced_vectors, reducer


def cluster_songs(
    reduced_vectors: dict[str, np.ndarray],
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
) -> tuple[list[int], list[str], object]:
    """Cluster reduced vectors with HDBSCAN.

    Returns (labels, id_list, clusterer).
    Labels of -1 indicate noise (unclustered).
    """
    import hdbscan

    config = load_config()
    cluster_cfg = config.get("clustering", {})
    if min_cluster_size is None:
        min_cluster_size = cluster_cfg.get("min_cluster_size", 10)
    if min_samples is None:
        min_samples = cluster_cfg.get("min_samples", 5)

    if not reduced_vectors:
        return [], [], None

    id_list = list(reduced_vectors.keys())
    matrix = np.stack([reduced_vectors[sid] for sid in id_list]).astype(np.float32)

    # Ensure min_cluster_size doesn't exceed sample count
    min_cluster_size = min(min_cluster_size, max(2, matrix.shape[0] // 2))
    min_samples = min(min_samples, min_cluster_size)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(matrix).tolist()

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = labels.count(-1)
    console.print(
        f"Found [green]{n_clusters}[/green] clusters, "
        f"[yellow]{n_noise}[/yellow] noise points"
    )

    return labels, id_list, clusterer


def assign_clusters(
    session: Session, spotify_ids: list[str], labels: list[int]
) -> int:
    """Assign cluster labels to songs in the database.

    Creates Cluster records for new clusters.
    Returns number of songs assigned.
    """
    if not spotify_ids or not labels:
        return 0

    # Find unique cluster IDs (excluding noise = -1)
    unique_labels = sorted(set(labels) - {-1})

    # Create or get Cluster records
    cluster_map: dict[int, int] = {}
    for label in unique_labels:
        existing = session.query(Cluster).filter_by(name=f"Cluster {label}").first()
        if existing:
            cluster_map[label] = existing.id
        else:
            cluster = Cluster(name=f"Cluster {label}")
            session.add(cluster)
            session.flush()
            cluster_map[label] = cluster.id

    # Assign songs to clusters
    assigned = 0
    for sid, label in zip(spotify_ids, labels):
        song = session.get(Song, sid)
        if song is None:
            continue
        if label == -1:
            song.cluster_id = None
        else:
            song.cluster_id = cluster_map[label]
        assigned += 1

    session.flush()
    console.print(f"Assigned [green]{assigned}[/green] songs to clusters")
    return assigned


def get_cluster_summary(session: Session) -> list[dict]:
    """Get summary of all clusters with song counts and representative tracks."""
    clusters = session.query(Cluster).all()
    summaries: list[dict] = []

    for cluster in clusters:
        songs = (
            session.query(Song)
            .filter(Song.cluster_id == cluster.id)
            .order_by(Song.total_plays.desc())
            .all()
        )

        if not songs:
            continue

        # Collect common tags
        all_tags: dict[str, int] = {}
        for song in songs:
            if song.lastfm_tags:
                for tag in song.lastfm_tags.split(","):
                    tag = tag.strip().lower()
                    if tag:
                        all_tags[tag] = all_tags.get(tag, 0) + 1

        top_tags = sorted(all_tags, key=all_tags.get, reverse=True)[:5]

        summaries.append({
            "id": cluster.id,
            "name": cluster.name,
            "description": cluster.description,
            "song_count": len(songs),
            "top_tags": top_tags,
            "top_songs": [
                {"title": s.title, "artist": s.artist, "plays": s.total_plays}
                for s in songs[:5]
            ],
        })

    return summaries

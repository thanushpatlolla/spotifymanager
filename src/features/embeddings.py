"""Sprint 2: Feature vector generation and FAISS index management.

Converts song attributes (audio features, tags, listening stats) into
dense feature vectors and maintains a FAISS index for similarity search.
"""


def build_feature_vectors(session) -> dict:
    """Build feature vectors for all songs from audio features, tags, and stats.

    Returns dict mapping spotify_id -> numpy array.
    """
    raise NotImplementedError("Sprint 2")


def build_faiss_index(vectors: dict, save_path: str | None = None):
    """Build a FAISS index from feature vectors and optionally save to disk."""
    raise NotImplementedError("Sprint 2")


def load_faiss_index(path: str):
    """Load a previously saved FAISS index from disk."""
    raise NotImplementedError("Sprint 2")


def find_similar(index, query_vector, k: int = 10) -> list[tuple[str, float]]:
    """Find k most similar songs to query_vector. Returns (spotify_id, distance) pairs."""
    raise NotImplementedError("Sprint 2")

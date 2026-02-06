"""Sprint 5: HDBSCAN + UMAP clustering.

Clusters songs by feature similarity using UMAP for dimensionality reduction
and HDBSCAN for density-based clustering.
"""


def reduce_dimensions(vectors: dict, n_components: int = 10):
    """Reduce feature vectors with UMAP. Returns (reduced_vectors, umap_model)."""
    raise NotImplementedError("Sprint 5")


def cluster_songs(reduced_vectors, min_cluster_size: int = 10, min_samples: int = 5):
    """Cluster reduced vectors with HDBSCAN. Returns (labels, clusterer)."""
    raise NotImplementedError("Sprint 5")


def assign_clusters(session, spotify_ids: list[str], labels: list[int]):
    """Assign cluster labels to songs in the database."""
    raise NotImplementedError("Sprint 5")


def get_cluster_summary(session) -> list[dict]:
    """Get summary of all clusters with song counts and representative tracks."""
    raise NotImplementedError("Sprint 5")

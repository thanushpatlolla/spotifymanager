"""Sprint 5: Reclustering script.

Rebuilds HDBSCAN clusters and generates LLM names.

Usage:
    uv run python scripts/recluster.py
"""

from src.data.db import get_session, init_db


def main():
    """Rebuild song clusters using HDBSCAN + UMAP and name them with Claude."""
    raise NotImplementedError("Sprint 5 — reclustering")


if __name__ == "__main__":
    main()

"""Sprint 5: LLM-powered cluster naming.

Uses Claude to generate descriptive names and descriptions for
song clusters based on their shared characteristics.
"""


def name_cluster(client, songs: list[dict]) -> dict:
    """Generate a name and description for a cluster of songs.

    Args:
        client: Anthropic client.
        songs: List of song dicts with title, artist, tags.

    Returns dict with 'name' and 'description'.
    """
    raise NotImplementedError("Sprint 5")


def name_all_clusters(session, client) -> int:
    """Name all unnamed clusters. Returns count of clusters named."""
    raise NotImplementedError("Sprint 5")

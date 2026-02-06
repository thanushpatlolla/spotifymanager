"""Sprint 5: LLM-powered discovery context.

Uses Claude to generate context and reasoning for music recommendations,
helping the user understand why songs are suggested.
"""


def generate_discovery_context(client, candidate: dict, user_profile: dict) -> str:
    """Generate a natural language explanation for why a song is recommended.

    Args:
        client: Anthropic client.
        candidate: Discovery candidate with title, artist, tags, score.
        user_profile: Summary of user's taste clusters and preferences.

    Returns explanation string.
    """
    raise NotImplementedError("Sprint 5")


def summarize_playlist_reasoning(client, songs: list[dict], mood: dict) -> str:
    """Generate a summary of why the current playlist was composed this way."""
    raise NotImplementedError("Sprint 5")

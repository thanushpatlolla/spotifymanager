"""LLM-powered discovery context.

Uses Claude to generate context and reasoning for music recommendations,
helping the user understand why songs are suggested.
"""

from src.llm.client import prompt_claude

DISCOVERY_SYSTEM = """\
You are a music recommendation assistant. Explain in 1-2 concise sentences \
why a song might appeal to the listener based on their taste profile. \
Be specific about musical qualities, not generic."""

PLAYLIST_SYSTEM = """\
You are a music curator. Summarize in 2-3 sentences why this playlist was \
composed this way, referencing the mood and song selection. Be concise."""


def generate_discovery_context(client, candidate: dict, user_profile: dict) -> str:
    """Generate a natural language explanation for why a song is recommended.

    Args:
        client: Anthropic client.
        candidate: Discovery candidate with title, artist, tags, score.
        user_profile: Summary of user's taste clusters and preferences.

    Returns explanation string.
    """
    top_tags = ", ".join(user_profile.get("top_tags", [])[:10])
    top_artists = ", ".join(user_profile.get("top_artists", [])[:5])
    clusters = ", ".join(user_profile.get("cluster_names", [])[:5])

    user_prompt = (
        f"Recommended song: {candidate['artist']} — {candidate['title']}\n"
        f"Song tags: {candidate.get('tags', 'unknown')}\n"
        f"Match score: {candidate.get('score', 'N/A')}\n\n"
        f"Listener profile:\n"
        f"- Favorite tags: {top_tags}\n"
        f"- Top artists: {top_artists}\n"
        f"- Taste clusters: {clusters}\n\n"
        f"Why might they enjoy this song?"
    )

    return prompt_claude(client, DISCOVERY_SYSTEM, user_prompt, max_tokens=150)


def summarize_playlist_reasoning(client, songs: list[dict], mood: dict) -> str:
    """Generate a summary of why the current playlist was composed this way."""
    song_sample = songs[:10]
    song_lines = [f"- {s['artist']} — {s['title']}" for s in song_sample]

    mood_tags = ", ".join(mood.get("top_tags", [])[:5]) if mood else "unknown"

    user_prompt = (
        f"Current mood tags: {mood_tags}\n"
        f"Playlist ({len(songs)} songs), sample:\n"
        + "\n".join(song_lines)
        + f"\n\nWhy was this playlist composed this way?"
    )

    return prompt_claude(client, PLAYLIST_SYSTEM, user_prompt, max_tokens=200)

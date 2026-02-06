"""Sprint 3: Dynamic playlist generation.

Generates and updates the managed Spotify playlist based on
current scores, mood, and playlist constraints.
"""


def generate_playlist(session, sp) -> list[str]:
    """Generate a new playlist track list based on current scores and mood.

    Returns list of spotify_ids to include.
    """
    raise NotImplementedError("Sprint 3")


def update_spotify_playlist(sp, playlist_id: str, track_ids: list[str]):
    """Replace the contents of a Spotify playlist with the given tracks."""
    raise NotImplementedError("Sprint 3")


def refresh_playlist(session, sp) -> str:
    """Full refresh: score, generate, update. Returns playlist URL."""
    raise NotImplementedError("Sprint 3")

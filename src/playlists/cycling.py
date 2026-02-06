"""Sprint 3: Song cycling logic.

Ensures variety by cycling songs in and out of the playlist
based on play history and staleness.
"""


def get_songs_to_cycle_out(session, playlist_track_ids: list[str]) -> list[str]:
    """Identify songs that should be removed from the playlist (overplayed, stale).

    Returns list of spotify_ids to remove.
    """
    raise NotImplementedError("Sprint 3")


def get_songs_to_cycle_in(session, current_track_ids: list[str], count: int) -> list[str]:
    """Select songs to add to the playlist as replacements.

    Considers freshness, score, and variety.
    Returns list of spotify_ids to add.
    """
    raise NotImplementedError("Sprint 3")

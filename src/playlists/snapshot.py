"""Sprint 3: Playlist snapshot management.

Saves and restores playlist states for history tracking and rollback.
"""


def save_snapshot(session, playlist_id: str, track_ids: list[str]) -> int:
    """Save the current playlist state as a snapshot. Returns snapshot ID."""
    raise NotImplementedError("Sprint 3")


def get_latest_snapshot(session, playlist_id: str) -> list[str] | None:
    """Get the most recent snapshot's track list, or None."""
    raise NotImplementedError("Sprint 3")


def diff_snapshots(old_tracks: list[str], new_tracks: list[str]) -> dict:
    """Compute difference between two snapshots. Returns {added, removed, kept}."""
    raise NotImplementedError("Sprint 3")

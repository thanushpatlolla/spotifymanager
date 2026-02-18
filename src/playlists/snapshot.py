"""Playlist snapshot management — save, load, and diff playlist states."""

import json

from sqlalchemy.orm import Session

from src.data.models import PlaylistSnapshot


def save_snapshot(session: Session, playlist_id: str, track_ids: list[str]) -> int:
    """Save the current playlist state as a snapshot. Returns snapshot ID."""
    snapshot = PlaylistSnapshot(
        playlist_id=playlist_id,
        track_ids=json.dumps(track_ids),
    )
    session.add(snapshot)
    session.flush()
    return snapshot.id


def get_latest_snapshot(session: Session, playlist_id: str) -> list[str] | None:
    """Get the most recent snapshot's track list, or None if no snapshots exist."""
    snapshot = (
        session.query(PlaylistSnapshot)
        .filter(PlaylistSnapshot.playlist_id == playlist_id)
        .order_by(PlaylistSnapshot.created_at.desc())
        .first()
    )
    if snapshot is None:
        return None
    return json.loads(snapshot.track_ids)


def diff_snapshots(
    old_tracks: list[str], new_tracks: list[str]
) -> dict[str, list[str]]:
    """Compute difference between two track lists.

    Returns dict with keys: added, removed, kept.
    """
    old_set = set(old_tracks)
    new_set = set(new_tracks)
    return {
        "added": [t for t in new_tracks if t not in old_set],
        "removed": [t for t in old_tracks if t not in new_set],
        "kept": [t for t in new_tracks if t in old_set],
    }

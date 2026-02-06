"""Compute listening statistics from ListenEvents."""

from datetime import datetime, timedelta

from rich.console import Console
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.data.models import ListenEvent, Song

console = Console()


def update_listening_stats(session: Session) -> int:
    """Update total_plays, recent_plays_30d, last_played, and staleness_score for all songs.

    Staleness score: 0.0 (just played) to 1.0 (not played in 180+ days).
    Songs with no listen events get staleness_score = None.

    Returns count of songs updated.
    """
    now = datetime.utcnow()
    thirty_days_ago = now - timedelta(days=30)
    staleness_max_days = 180.0

    # Get aggregated stats per song in one query
    stats = (
        session.query(
            ListenEvent.song_id,
            func.count(ListenEvent.id).label("total_plays"),
            func.max(ListenEvent.listened_at).label("last_played"),
            func.count(
                func.nullif(ListenEvent.listened_at < thirty_days_ago, True)
            ).label("recent_plays"),
        )
        .group_by(ListenEvent.song_id)
        .all()
    )

    stats_map = {}
    for row in stats:
        stats_map[row.song_id] = {
            "total_plays": row.total_plays,
            "last_played": row.last_played,
            "recent_plays": row.recent_plays,
        }

    # Also compute recent_plays_30d correctly with a separate query
    recent_counts = (
        session.query(
            ListenEvent.song_id,
            func.count(ListenEvent.id).label("recent"),
        )
        .filter(ListenEvent.listened_at >= thirty_days_ago)
        .group_by(ListenEvent.song_id)
        .all()
    )
    recent_map = {row.song_id: row.recent for row in recent_counts}

    songs = session.query(Song).all()
    updated = 0

    for song in songs:
        s = stats_map.get(song.spotify_id)
        if s is None:
            continue

        song.total_plays = s["total_plays"]
        song.last_played = s["last_played"]
        song.recent_plays_30d = recent_map.get(song.spotify_id, 0)

        # Compute staleness
        if s["last_played"]:
            days_since = (now - s["last_played"]).days
            song.staleness_score = min(days_since / staleness_max_days, 1.0)
        updated += 1

    session.flush()
    console.print(f"Updated listening stats for [green]{updated}[/green] songs")
    return updated

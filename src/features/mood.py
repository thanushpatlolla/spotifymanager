"""Tag-based mood vector computation from recent listening history.

Uses Last.fm tags as a sparse vector space to represent mood.
Songs are projected into this space, and recent listening is
time-decay weighted to compute a current mood vector.
"""

import math
from datetime import datetime, timedelta

from sqlalchemy import func
from sqlalchemy.orm import Session

from src.data.db import load_config
from src.data.models import ListenEvent, Song


def parse_tags(tag_string: str | None) -> list[str]:
    """Split comma-separated tag string into cleaned list."""
    if not tag_string:
        return []
    return [t.strip().lower() for t in tag_string.split(",") if t.strip()]


def build_tag_vocabulary(session: Session, min_count: int | None = None) -> dict[str, int]:
    """Build tag->index mapping, filtering tags appearing fewer than min_count times.

    Returns dict mapping tag string to integer index.
    """
    if min_count is None:
        config = load_config()
        min_count = config.get("mood", {}).get("min_tag_count", 2)

    songs = session.query(Song.lastfm_tags).filter(Song.lastfm_tags.isnot(None)).all()

    tag_counts: dict[str, int] = {}
    for (tags_str,) in songs:
        for tag in parse_tags(tags_str):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    vocabulary: dict[str, int] = {}
    idx = 0
    for tag in sorted(tag_counts):
        if tag_counts[tag] >= min_count:
            vocabulary[tag] = idx
            idx += 1

    return vocabulary


def song_to_tag_vector(song: Song, vocabulary: dict[str, int]) -> dict[int, float]:
    """Convert a song's tags to a sparse vector {index: 1.0}."""
    tags = parse_tags(song.lastfm_tags)
    vector: dict[int, float] = {}
    for tag in tags:
        if tag in vocabulary:
            vector[vocabulary[tag]] = 1.0
    return vector


def _l2_normalize(vector: dict[int, float]) -> dict[int, float]:
    """L2-normalize a sparse vector in place."""
    magnitude = math.sqrt(sum(v * v for v in vector.values()))
    if magnitude == 0:
        return vector
    return {k: v / magnitude for k, v in vector.items()}


def compute_mood_vector(
    session: Session, window_hours: int | None = None
) -> tuple[dict[int, float] | None, dict[str, int]]:
    """Compute mood vector from recent listening, with fallback to top played songs.

    Returns (mood_vector, vocabulary) where mood_vector is a sparse dict
    {tag_index: weight} or None if no songs have tags.
    """
    config = load_config()
    mood_cfg = config.get("mood", {})
    if window_hours is None:
        window_hours = mood_cfg.get("window_hours", 4)
    decay_factor = mood_cfg.get("decay_factor", 0.85)
    fallback_top_n = mood_cfg.get("fallback_top_n", 50)

    vocabulary = build_tag_vocabulary(session)
    if not vocabulary:
        return None, vocabulary

    now = datetime.utcnow()
    cutoff = now - timedelta(hours=window_hours)

    # Get recent listen events, most recent first
    recent_events = (
        session.query(ListenEvent)
        .filter(ListenEvent.listened_at >= cutoff)
        .order_by(ListenEvent.listened_at.desc())
        .all()
    )

    # Deduplicate: keep order of first appearance (most recent first)
    seen: set[str] = set()
    unique_song_ids: list[str] = []
    for event in recent_events:
        if event.song_id not in seen:
            seen.add(event.song_id)
            unique_song_ids.append(event.song_id)

    if unique_song_ids:
        songs = {s.spotify_id: s for s in session.query(Song).filter(Song.spotify_id.in_(unique_song_ids)).all()}
        mood_vector: dict[int, float] = {}
        for position, sid in enumerate(unique_song_ids):
            song = songs.get(sid)
            if not song:
                continue
            weight = decay_factor ** position
            tag_vec = song_to_tag_vector(song, vocabulary)
            for idx, val in tag_vec.items():
                mood_vector[idx] = mood_vector.get(idx, 0.0) + weight * val

        if mood_vector:
            return _l2_normalize(mood_vector), vocabulary

    # Fallback: top N most-played songs with equal weights
    top_songs = (
        session.query(Song)
        .filter(Song.total_plays > 0, Song.lastfm_tags.isnot(None))
        .order_by(Song.total_plays.desc())
        .limit(fallback_top_n)
        .all()
    )

    if not top_songs:
        return None, vocabulary

    mood_vector = {}
    for song in top_songs:
        tag_vec = song_to_tag_vector(song, vocabulary)
        for idx, val in tag_vec.items():
            mood_vector[idx] = mood_vector.get(idx, 0.0) + val

    if mood_vector:
        return _l2_normalize(mood_vector), vocabulary

    return None, vocabulary


def compute_song_mood_similarity(
    song: Song, mood_vector: dict[int, float], vocabulary: dict[str, int]
) -> float:
    """Compute cosine similarity between a song's tags and the mood vector.

    Returns 0.0-1.0. Songs with no tags get the configured default (0.5).
    """
    config = load_config()
    no_tag_default = config.get("mood", {}).get("no_tag_default", 0.5)

    song_vec = song_to_tag_vector(song, vocabulary)
    if not song_vec:
        return no_tag_default

    # Both vectors should be normalized for cosine similarity
    song_vec = _l2_normalize(song_vec)

    dot = sum(song_vec.get(k, 0.0) * mood_vector.get(k, 0.0) for k in song_vec)
    return max(0.0, min(1.0, dot))

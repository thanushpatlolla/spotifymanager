"""Heuristic taste scoring and combined score computation.

Combines taste (play history, liked status, recency), mood similarity,
and freshness into a single 0-1 score per song.
"""

import math
from datetime import datetime, timedelta

from sqlalchemy import func
from sqlalchemy.orm import Session

from src.data.db import load_config
from src.data.models import Song
from src.features.mood import compute_mood_vector, compute_song_mood_similarity


def compute_taste_score(song: Song, play_count_p95: float) -> float:
    """Compute heuristic taste score (0.0-1.0) for a single song.

    Components:
    - Base: 0.5
    - Play count: +0.15 * min(log2(1+plays)/log2(1+p95), 1.0)
      OR +0.25 new song bonus if 0 plays
    - Liked: +0.10
    - In playlist: +0.05
    - Recency: +0.10 * min(recent_30d/total * 5, 1.0) if actively played
    - Abandoned: -0.10 if 10+ plays, 0 recent, 90+ days since last play
    """
    score = 0.5

    plays = song.total_plays or 0
    recent = song.recent_plays_30d or 0

    if plays == 0:
        # New song bonus
        score += 0.25
    else:
        # Play count component
        if play_count_p95 > 0:
            ratio = math.log2(1 + plays) / math.log2(1 + play_count_p95)
            score += 0.15 * min(ratio, 1.0)

    # Liked bonus
    if song.in_liked_songs:
        score += 0.10

    # Source playlist bonus (in any user playlist)
    if song.source_playlist:
        score += 0.05

    # Recency bonus — only if the song has been played
    if plays > 0 and recent > 0:
        recency_ratio = (recent / plays) * 5
        score += 0.10 * min(recency_ratio, 1.0)

    # Abandoned penalty
    if plays >= 10 and recent == 0 and song.last_played:
        days_since = (datetime.utcnow() - song.last_played).days
        if days_since >= 90:
            score -= 0.10

    return max(0.0, min(1.0, score))


def _compute_p95_plays(session: Session) -> float:
    """Get 95th percentile of play counts for normalization."""
    songs = (
        session.query(Song.total_plays)
        .filter(Song.total_plays > 0)
        .order_by(Song.total_plays.asc())
        .all()
    )
    if not songs:
        return 1.0
    counts = [s.total_plays for s in songs]
    idx = int(len(counts) * 0.95)
    idx = min(idx, len(counts) - 1)
    return float(counts[idx])


def compute_song_scores(
    session: Session,
    mood_vector: dict[int, float] | None = None,
    vocabulary: dict[str, int] | None = None,
) -> dict[str, float]:
    """Compute combined scores for all songs.

    Score = taste_weight * taste + mood_weight * mood_sim + freshness_weight * freshness

    If mood_vector is None, it will be computed. If still None (no tags),
    mood component is replaced by distributing weight to taste and freshness.
    """
    config = load_config()
    scoring_cfg = config.get("scoring", {})
    taste_w = scoring_cfg.get("taste_weight", 0.5)
    mood_w = scoring_cfg.get("mood_weight", 0.3)
    freshness_w = scoring_cfg.get("freshness_weight", 0.2)

    # Compute mood vector if not provided
    if mood_vector is None or vocabulary is None:
        mood_vector, vocabulary = compute_mood_vector(session)

    # If no mood available, redistribute weight
    if mood_vector is None:
        total = taste_w + freshness_w
        if total > 0:
            taste_w = taste_w / total
            freshness_w = freshness_w / total
        mood_w = 0.0

    p95 = _compute_p95_plays(session)
    songs = session.query(Song).all()
    scores: dict[str, float] = {}

    for song in songs:
        taste = compute_taste_score(song, p95)

        # Mood similarity
        if mood_vector is not None and vocabulary is not None:
            mood_sim = compute_song_mood_similarity(song, mood_vector, vocabulary)
        else:
            mood_sim = 0.0

        # Freshness — staleness_score is high when song hasn't been played recently
        # which means it's "fresh" for the playlist
        if song.staleness_score is not None:
            freshness = song.staleness_score
        elif (song.total_plays or 0) == 0:
            freshness = 0.8  # Never-played songs are fairly fresh
        else:
            freshness = 0.5  # Default

        combined = taste_w * taste + mood_w * mood_sim + freshness_w * freshness
        scores[song.spotify_id] = max(0.0, min(1.0, combined))

    return scores


def rank_songs(scores: dict[str, float], limit: int = 100) -> list[str]:
    """Rank songs by score descending and return top N spotify_ids."""
    sorted_ids = sorted(scores, key=scores.get, reverse=True)
    return sorted_ids[:limit]

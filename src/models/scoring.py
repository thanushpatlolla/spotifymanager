"""Sprint 2: Combined scoring — taste x mood x freshness.

Computes a final score for each song by combining taste model predictions,
mood similarity, and freshness/staleness metrics.
"""


def compute_song_scores(session, mood_vector: list[float] | None = None) -> dict[str, float]:
    """Compute combined scores for all songs.

    Score = (taste_weight * taste_score) + (mood_weight * mood_sim) + (freshness_weight * freshness)
    Weights are read from config.yaml.

    Returns dict mapping spotify_id -> score.
    """
    raise NotImplementedError("Sprint 2")


def rank_songs(scores: dict[str, float], limit: int = 30) -> list[str]:
    """Rank songs by score and return top N spotify_ids."""
    raise NotImplementedError("Sprint 2")

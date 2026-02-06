"""Sprint 2: Mood vector computation from recent listening history.

Analyzes the user's recent listening patterns to derive a mood vector
that can be used for scoring and playlist generation.
"""


def compute_mood_vector(session, window_hours: int = 4) -> list[float]:
    """Compute a mood vector from recent listen events within the given window.

    Returns a normalized vector representing current listening mood.
    """
    raise NotImplementedError("Sprint 2")


def mood_similarity(mood_a: list[float], mood_b: list[float]) -> float:
    """Compute cosine similarity between two mood vectors."""
    raise NotImplementedError("Sprint 2")

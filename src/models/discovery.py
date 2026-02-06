"""Sprint 4: Multi-source music discovery.

Finds new music from Last.fm similar tracks, ListenBrainz recommendations,
and other sources. Evaluates candidates for inclusion in the library.
"""


def discover_from_lastfm(network, session, limit: int = 50) -> list[dict]:
    """Find discovery candidates from Last.fm similar tracks.

    Returns list of candidate dicts with title, artist, source, score.
    """
    raise NotImplementedError("Sprint 4")


def discover_from_listenbrainz(session, limit: int = 50) -> list[dict]:
    """Find discovery candidates from ListenBrainz recommendations."""
    raise NotImplementedError("Sprint 4")


def evaluate_candidates(session, candidates: list[dict]) -> list[dict]:
    """Score and rank discovery candidates against the user's taste profile."""
    raise NotImplementedError("Sprint 4")


def save_candidates(session, candidates: list[dict]) -> int:
    """Save evaluated candidates to the database. Returns count saved."""
    raise NotImplementedError("Sprint 4")

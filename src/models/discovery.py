"""Multi-source music discovery.

Finds new music from Last.fm similar tracks and other sources.
Evaluates candidates for inclusion in the library.
"""

import time

import httpx
from rich.console import Console
from sqlalchemy.orm import Session

from src.data.db import load_config
from src.data.models import DiscoveryCandidate, Song

console = Console()


def discover_from_lastfm(network, session: Session, limit: int = 50) -> list[dict]:
    """Find discovery candidates from Last.fm similar tracks.

    Samples top-played songs from the library, fetches their similar tracks,
    and filters out songs already in the library.

    Returns list of candidate dicts with title, artist, source, score.
    """
    # Get top played songs to use as seeds
    seeds = (
        session.query(Song)
        .filter(Song.total_plays > 0, Song.lastfm_tags.isnot(None))
        .order_by(Song.total_plays.desc())
        .limit(20)
        .all()
    )

    if not seeds:
        console.print("[yellow]No seed songs found for Last.fm discovery.[/yellow]")
        return []

    existing_ids = {s.spotify_id for s in session.query(Song.spotify_id).all()}
    existing_artists_titles = {
        (s.artist.lower(), s.title.lower())
        for s in session.query(Song.artist, Song.title).all()
    }

    candidates: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for seed in seeds:
        if len(candidates) >= limit:
            break

        primary_artist = seed.artist.split(",")[0].strip()
        try:
            track = network.get_track(primary_artist, seed.title)
            similar = track.get_similar(limit=10)
        except Exception:
            time.sleep(0.25)
            continue

        for item in similar:
            if len(candidates) >= limit:
                break

            try:
                artist_name = str(item.item.artist)
                title_name = str(item.item.title)
            except Exception:
                continue

            key = (artist_name.lower(), title_name.lower())
            if key in seen or key in existing_artists_titles:
                continue
            seen.add(key)

            match_score = float(item.match) if hasattr(item, "match") and item.match else 0.5

            candidates.append({
                "title": title_name,
                "artist": artist_name,
                "source": f"lastfm_similar:{seed.spotify_id}",
                "score": match_score,
            })

        time.sleep(0.25)

    console.print(f"Found [green]{len(candidates)}[/green] candidates from Last.fm")
    return candidates


def discover_from_listenbrainz(session: Session, limit: int = 50) -> list[dict]:
    """Find discovery candidates from ListenBrainz recommendations.

    Uses the ListenBrainz public API (no auth required for basic endpoints).
    """
    config = load_config()
    username = config.get("lastfm", {}).get("username", "")

    if not username:
        console.print("[yellow]No username configured for ListenBrainz discovery.[/yellow]")
        return []

    existing_artists_titles = {
        (s.artist.lower(), s.title.lower())
        for s in session.query(Song.artist, Song.title).all()
    }

    candidates: list[dict] = []
    url = f"https://api.listenbrainz.org/1/user/{username}/recommendation/recordings"

    try:
        resp = httpx.get(url, params={"count": limit}, timeout=15)
        if resp.status_code != 200:
            console.print(f"[yellow]ListenBrainz API returned {resp.status_code}[/yellow]")
            return []

        data = resp.json()
        recs = data.get("payload", {}).get("mbids", [])

        for rec in recs:
            if len(candidates) >= limit:
                break

            artist = rec.get("artist_name", "")
            title = rec.get("recording_name", "")

            if not artist or not title:
                continue

            key = (artist.lower(), title.lower())
            if key in existing_artists_titles:
                continue

            candidates.append({
                "title": title,
                "artist": artist,
                "source": "listenbrainz",
                "score": rec.get("score", 0.5),
            })

    except httpx.HTTPError:
        console.print("[yellow]Failed to reach ListenBrainz API.[/yellow]")

    console.print(f"Found [green]{len(candidates)}[/green] candidates from ListenBrainz")
    return candidates


def evaluate_candidates(
    session: Session, candidates: list[dict]
) -> list[dict]:
    """Score and rank discovery candidates against the user's library.

    Boosts score for candidates whose artist already appears in the library
    or whose inferred tags overlap with top-played tag profile.
    """
    if not candidates:
        return []

    # Get artists in library for affinity boost
    library_artists = {
        s.artist.lower()
        for s in session.query(Song.artist).all()
    }

    config = load_config()
    min_score = config.get("discovery", {}).get("min_score", 0.5)

    evaluated: list[dict] = []
    for c in candidates:
        score = c.get("score", 0.5)

        # Boost if artist is partially known (e.g. a different song by same artist)
        artist_lower = c["artist"].lower()
        if any(artist_lower in la or la in artist_lower for la in library_artists):
            score = min(score + 0.15, 1.0)

        c["score"] = score
        if score >= min_score:
            evaluated.append(c)

    evaluated.sort(key=lambda x: x["score"], reverse=True)
    return evaluated


def save_candidates(session: Session, candidates: list[dict]) -> int:
    """Save evaluated candidates to the database. Returns count saved."""
    saved = 0
    for c in candidates:
        # Skip if already exists
        existing = (
            session.query(DiscoveryCandidate)
            .filter_by(title=c["title"], artist=c["artist"])
            .first()
        )
        if existing:
            existing.score = c.get("score", existing.score)
            continue

        candidate = DiscoveryCandidate(
            spotify_id="",  # Will be resolved when added to playlist
            title=c["title"],
            artist=c["artist"],
            source=c.get("source", "unknown"),
            score=c.get("score"),
        )
        session.add(candidate)
        saved += 1

    session.flush()
    console.print(f"Saved [green]{saved}[/green] new discovery candidates")
    return saved

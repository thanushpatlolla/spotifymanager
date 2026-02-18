"""Track cycling logic — swap songs in and out of the playlist.

Ensures variety by removing lowest-scored tracks and replacing them
with weighted-random selections from the library.
"""

import random
from collections import Counter

from sqlalchemy.orm import Session

from src.data.db import load_config
from src.data.models import Song


def get_songs_to_cycle_out(
    session: Session,
    playlist_ids: list[str],
    scores: dict[str, float],
    cycle_fraction: float | None = None,
    min_keep: int | None = None,
) -> list[str]:
    """Identify lowest-scored songs to remove from the playlist.

    Removes approximately cycle_fraction of the playlist,
    but always keeps at least min_keep songs.
    """
    config = load_config()
    cycling_cfg = config.get("cycling", {})
    if cycle_fraction is None:
        cycle_fraction = cycling_cfg.get("cycle_fraction", 0.30)
    if min_keep is None:
        min_keep = cycling_cfg.get("min_keep", 20)

    if not playlist_ids:
        return []

    # Sort playlist songs by score ascending (worst first)
    scored = [(sid, scores.get(sid, 0.0)) for sid in playlist_ids]
    scored.sort(key=lambda x: x[1])

    n_to_remove = int(len(playlist_ids) * cycle_fraction)
    # Ensure we keep at least min_keep
    max_removable = max(0, len(playlist_ids) - min_keep)
    n_to_remove = min(n_to_remove, max_removable)

    return [sid for sid, _ in scored[:n_to_remove]]


def get_songs_to_cycle_in(
    session: Session,
    current_ids: list[str],
    scores: dict[str, float],
    count: int,
) -> list[str]:
    """Select songs to add to the playlist using weighted random sampling.

    Uses scores as sampling weights. Applies soft artist diversity:
    halves weight if artist already has 3+ tracks in the playlist.
    """
    current_set = set(current_ids)

    # Get candidates: all songs NOT in current playlist
    candidates = (
        session.query(Song.spotify_id, Song.artist)
        .filter(Song.spotify_id.notin_(current_set) if current_set else True)
        .all()
    )

    if not candidates:
        return []

    # Count artists already in playlist
    playlist_songs = (
        session.query(Song.spotify_id, Song.artist)
        .filter(Song.spotify_id.in_(current_ids))
        .all()
    ) if current_ids else []
    artist_counts = Counter(s.artist for s in playlist_songs)

    # Build weighted candidate list
    weighted: list[tuple[str, float]] = []
    for sid, artist in candidates:
        w = scores.get(sid, 0.0)
        if w <= 0:
            continue
        # Soft artist diversity: halve weight if artist is overrepresented
        if artist_counts.get(artist, 0) >= 3:
            w *= 0.5
        weighted.append((sid, w))

    if not weighted:
        return []

    # Weighted random sampling without replacement
    ids = [w[0] for w in weighted]
    weights = [w[1] for w in weighted]
    count = min(count, len(ids))

    selected: list[str] = []
    remaining_ids = list(ids)
    remaining_weights = list(weights)

    for _ in range(count):
        if not remaining_ids:
            break
        chosen = random.choices(remaining_ids, weights=remaining_weights, k=1)[0]
        selected.append(chosen)
        idx = remaining_ids.index(chosen)
        remaining_ids.pop(idx)
        remaining_weights.pop(idx)

    return selected

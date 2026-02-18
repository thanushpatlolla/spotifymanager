"""Dynamic playlist generation and refresh orchestration.

Handles both fresh generation and cycling-based updates,
manages Spotify API calls, and saves snapshots.
"""

import random
from collections import Counter

from rich.console import Console
from sqlalchemy.orm import Session

from src.data.db import load_config
from src.data.models import Song
from src.data.spotify_client import chunked, get_or_create_playlist
from src.features.mood import compute_mood_vector
from src.models.scoring import compute_song_scores
from src.playlists.cycling import get_songs_to_cycle_in, get_songs_to_cycle_out
from src.playlists.snapshot import diff_snapshots, get_latest_snapshot, save_snapshot

console = Console()


def _apply_artist_spacing(track_ids: list[str], session: Session) -> list[str]:
    """Reorder tracks so the same artist doesn't appear back-to-back.

    Simple greedy approach: iterate and swap adjacent same-artist tracks
    with the next different-artist track found.
    """
    if len(track_ids) <= 1:
        return track_ids

    songs = {
        s.spotify_id: s.artist
        for s in session.query(Song.spotify_id, Song.artist)
        .filter(Song.spotify_id.in_(track_ids))
        .all()
    }

    result = list(track_ids)
    for i in range(len(result) - 1):
        if songs.get(result[i]) == songs.get(result[i + 1]):
            # Find next track with different artist to swap
            for j in range(i + 2, len(result)):
                if songs.get(result[j]) != songs.get(result[i]):
                    result[i + 1], result[j] = result[j], result[i + 1]
                    break

    return result


def _weighted_random_select(
    scores: dict[str, float], count: int, session: Session
) -> list[str]:
    """Select tracks using score^2 weighted random sampling."""
    ids = list(scores.keys())
    weights = [scores[sid] ** 2 for sid in ids]
    total_weight = sum(weights)

    if total_weight == 0 or not ids:
        return ids[:count]

    selected: list[str] = []
    remaining_ids = list(ids)
    remaining_weights = list(weights)

    for _ in range(min(count, len(ids))):
        if not remaining_ids:
            break
        chosen = random.choices(remaining_ids, weights=remaining_weights, k=1)[0]
        selected.append(chosen)
        idx = remaining_ids.index(chosen)
        remaining_ids.pop(idx)
        remaining_weights.pop(idx)

    return selected


def generate_playlist(session: Session, sp) -> list[str]:
    """Generate a full fresh playlist from scratch.

    1. Compute mood vector
    2. Score all songs
    3. Weighted random select 90 tracks (or max_size - discovery_slots)
    4. Add discovery slots (placeholder — just fill from top unselected)
    5. Apply artist spacing
    """
    config = load_config()
    max_size = config.get("playlist", {}).get("max_size", 100)
    discovery_slots = config.get("discovery", {}).get("slots", 10)
    main_slots = max_size - discovery_slots

    mood_vector, vocabulary = compute_mood_vector(session)
    scores = compute_song_scores(session, mood_vector, vocabulary)

    if not scores:
        console.print("[red]No songs in library to generate playlist from.[/red]")
        return []

    # Weighted random selection for main slots
    selected = _weighted_random_select(scores, main_slots, session)

    # Discovery slots: fill from highest-scored unselected songs
    selected_set = set(selected)
    remaining = {sid: sc for sid, sc in scores.items() if sid not in selected_set}
    if remaining:
        discovery = _weighted_random_select(remaining, discovery_slots, session)
        selected.extend(discovery)

    # Apply artist spacing
    selected = _apply_artist_spacing(selected, session)

    console.print(f"Generated playlist with [green]{len(selected)}[/green] tracks")
    return selected


def update_spotify_playlist(sp, playlist_id: str, track_ids: list[str]):
    """Replace the contents of a Spotify playlist with the given tracks.

    Clears existing tracks, then adds new ones in batches of 100.
    """
    # Clear playlist
    sp.playlist_replace_items(playlist_id, [])

    # Add tracks in batches
    uris = [f"spotify:track:{tid}" for tid in track_ids]
    for chunk in chunked(uris, 100):
        sp.playlist_add_items(playlist_id, chunk)


def refresh_playlist(session: Session, sp) -> str:
    """Full refresh: auto-detect fresh vs cycling mode, update Spotify.

    - Has snapshot? -> cycling mode (keep ~70%, swap ~30%)
    - No snapshot? -> fresh generation
    - Saves snapshot after update

    Returns playlist URL.
    """
    config = load_config()
    playlist_name = config.get("playlist", {}).get("name", "spotifymanager generated")
    playlist_id = get_or_create_playlist(sp, playlist_name)

    previous = get_latest_snapshot(session, playlist_id)

    if previous is not None:
        # Cycling mode
        console.print("[cyan]Cycling mode[/cyan] — updating existing playlist")

        mood_vector, vocabulary = compute_mood_vector(session)
        scores = compute_song_scores(session, mood_vector, vocabulary)

        to_remove = get_songs_to_cycle_out(session, previous, scores)
        remove_set = set(to_remove)
        kept = [sid for sid in previous if sid not in remove_set]

        to_add = get_songs_to_cycle_in(session, kept, scores, len(to_remove))
        new_tracks = kept + to_add

        # Apply artist spacing
        new_tracks = _apply_artist_spacing(new_tracks, session)

        diff = diff_snapshots(previous, new_tracks)
        console.print(
            f"  Kept: [green]{len(diff['kept'])}[/green]  "
            f"Added: [green]{len(diff['added'])}[/green]  "
            f"Removed: [red]{len(diff['removed'])}[/red]"
        )
    else:
        # Fresh generation
        console.print("[cyan]Fresh generation[/cyan] — building new playlist")
        new_tracks = generate_playlist(session, sp)

    if not new_tracks:
        console.print("[red]No tracks to add to playlist.[/red]")
        return ""

    update_spotify_playlist(sp, playlist_id, new_tracks)
    save_snapshot(session, playlist_id, new_tracks)

    url = f"https://open.spotify.com/playlist/{playlist_id}"
    console.print(f"Playlist updated: [link={url}]{url}[/link]")
    return url

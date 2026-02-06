"""Ingestion pipeline orchestrator — imports data from Spotify and Last.fm into the database."""

from datetime import datetime

import spotipy
from rich.console import Console
from rich.progress import Progress
from sqlalchemy.orm import Session

from src.data.db import load_config
from src.data.lastfm_client import (
    build_song_lookup,
    fetch_all_scrobbles,
    fetch_lastfm_tags,
    match_lastfm_to_spotify,
)
from src.data.models import ListenEvent, Song
from src.data.spotify_client import (
    get_all_playlist_tracks,
    get_all_user_playlists,
    get_liked_songs,
    try_fetch_audio_features,
)

console = Console()


def _upsert_track(session: Session, track_data: dict, **extra_fields) -> Song:
    """Insert or update a Song from Spotify track data."""
    track = track_data["track"]
    spotify_id = track["id"]

    song = session.get(Song, spotify_id)
    if song is None:
        song = Song(
            spotify_id=spotify_id,
            title=track["name"],
            artist=", ".join(a["name"] for a in track["artists"]),
            album=track.get("album", {}).get("name"),
            duration_ms=track.get("duration_ms"),
            popularity=track.get("popularity"),
            added_at=_parse_added_at(track_data.get("added_at")),
            **extra_fields,
        )
        session.add(song)
    else:
        # Update fields that may have changed
        for key, value in extra_fields.items():
            if value is not None:
                setattr(song, key, value)
    return song


def _parse_added_at(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def ingest_playlists(sp: spotipy.Spotify, session: Session) -> int:
    """Ingest all tracks from all user-owned playlists. Returns count of songs added."""
    playlists = get_all_user_playlists(sp)
    console.print(f"Found [bold]{len(playlists)}[/bold] user playlists")
    count = 0

    with Progress() as progress:
        task = progress.add_task("Importing playlists...", total=len(playlists))
        for pl in playlists:
            tracks = get_all_playlist_tracks(sp, pl["id"])
            for item in tracks:
                _upsert_track(session, item, source_playlist=pl["name"])
                count += 1
            progress.advance(task)

    session.flush()
    console.print(f"Ingested [green]{count}[/green] tracks from playlists")
    return count


def ingest_liked_songs(sp: spotipy.Spotify, session: Session) -> int:
    """Ingest liked/saved songs, marking in_liked_songs=True. Returns count."""
    console.print("Fetching liked songs...")
    tracks = get_liked_songs(sp)
    count = 0

    for item in tracks:
        _upsert_track(session, item, in_liked_songs=True)
        count += 1

    session.flush()
    console.print(f"Ingested [green]{count}[/green] liked songs")
    return count


def ingest_audio_features(sp: spotipy.Spotify, session: Session) -> bool:
    """Attempt to fetch and store audio features for all songs.

    Returns True if successful, False if the API returned 403 (deprecated).
    """
    songs = session.query(Song).filter(Song.danceability.is_(None)).all()
    if not songs:
        console.print("No songs need audio features")
        return True

    track_ids = [s.spotify_id for s in songs]
    console.print(f"Fetching audio features for {len(track_ids)} tracks...")

    features = try_fetch_audio_features(sp, track_ids)
    if features is None:
        console.print(
            "[yellow]Audio features API returned 403 (deprecated). Skipping.[/yellow]"
        )
        return False

    feature_keys = [
        "danceability", "energy", "valence", "tempo", "acousticness",
        "instrumentalness", "speechiness", "liveness", "loudness",
        "key", "mode", "time_signature",
    ]

    applied = 0
    for song, feat in zip(songs, features):
        if feat is None:
            continue
        for key in feature_keys:
            if key in feat:
                setattr(song, key, feat[key])
        applied += 1

    session.flush()
    console.print(f"Applied audio features to [green]{applied}[/green] tracks")
    return True


def ingest_scrobbles(
    network, session: Session, username: str
) -> int:
    """Ingest Last.fm scrobble history, matching to Spotify tracks.

    Uses a match cache so only ~2-5k unique (artist, title) pairs need fuzzy matching,
    not the full 50k+ scrobble history.

    Returns count of matched listen events.
    """
    # Build lookup from existing songs
    all_songs = session.query(Song).all()
    lookup = build_song_lookup(all_songs)
    console.print(f"Built lookup with {len(lookup)} songs")

    # Match cache: (artist_lower, title_lower) -> spotify_id | None
    match_cache: dict[tuple[str, str], str | None] = {}
    matched_count = 0
    total_count = 0

    console.print("Fetching scrobble history (this may take a while)...")

    for scrobble in fetch_all_scrobbles(network, username):
        total_count += 1
        cache_key = (scrobble["artist"].lower().strip(), scrobble["title"].lower().strip())

        if cache_key not in match_cache:
            match_cache[cache_key] = match_lastfm_to_spotify(
                scrobble["artist"], scrobble["title"], lookup
            )

        spotify_id = match_cache[cache_key]
        if spotify_id is None:
            continue

        # Check for duplicate listen event
        existing = (
            session.query(ListenEvent)
            .filter_by(song_id=spotify_id, listened_at=scrobble["timestamp"])
            .first()
        )
        if existing:
            continue

        event = ListenEvent(
            song_id=spotify_id,
            listened_at=scrobble["timestamp"],
            source="lastfm",
        )
        session.add(event)
        matched_count += 1

        # Periodic flush to avoid memory buildup
        if matched_count % 1000 == 0:
            session.flush()
            console.print(f"  ...processed {total_count} scrobbles, {matched_count} matched")

    session.flush()
    unique_matched = sum(1 for v in match_cache.values() if v is not None)
    console.print(
        f"Processed [bold]{total_count}[/bold] scrobbles, "
        f"matched [green]{matched_count}[/green] events "
        f"({unique_matched} unique tracks from {len(match_cache)} unique scrobble pairs)"
    )
    return matched_count


def ingest_lastfm_tags(network, session: Session) -> int:
    """Fetch Last.fm tags for all songs without tags.

    This is slow (~30 min for 8k songs at 4 req/s). Shows progress.
    Returns count of songs tagged.
    """
    songs = session.query(Song).filter(Song.lastfm_tags.is_(None)).all()
    if not songs:
        console.print("All songs already have tags")
        return 0

    console.print(f"Fetching Last.fm tags for {len(songs)} songs (this will take a while)...")
    tagged = 0

    with Progress() as progress:
        task = progress.add_task("Fetching tags...", total=len(songs))
        for song in songs:
            # Use first artist only for tag lookup
            primary_artist = song.artist.split(",")[0].strip()
            tags = fetch_lastfm_tags(network, primary_artist, song.title)

            if tags:
                song.lastfm_tags = ", ".join(tags)
                tagged += 1

            progress.advance(task)

    session.flush()
    console.print(f"Tagged [green]{tagged}[/green] / {len(songs)} songs")
    return tagged

"""Last.fm API wrapper using pylast."""

import os
import time
from datetime import datetime
from typing import Iterator

import pylast
from thefuzz import fuzz

from src.data.db import load_config, load_env


def get_lastfm_client() -> pylast.LastFMNetwork:
    """Create an authenticated Last.fm client."""
    load_env()
    return pylast.LastFMNetwork(
        api_key=os.environ["LASTFM_API_KEY"],
        api_secret=os.environ["LASTFM_API_SECRET"],
    )


def fetch_all_scrobbles(
    network: pylast.LastFMNetwork, username: str
) -> Iterator[dict]:
    """Generator that yields all scrobbles for a user, paginated.

    Yields dicts with keys: artist, title, timestamp, album.
    """
    user = network.get_user(username)
    page = 1
    limit = 200

    while True:
        try:
            scrobbles = user.get_recent_tracks(limit=limit, page=page)
        except Exception:
            # Rate limit or transient error — wait and retry once
            time.sleep(2)
            try:
                scrobbles = user.get_recent_tracks(limit=limit, page=page)
            except Exception:
                break

        if not scrobbles:
            break

        for track in scrobbles:
            # Skip "now playing" entries (no timestamp)
            if track.timestamp is None:
                continue
            yield {
                "artist": str(track.track.artist),
                "title": str(track.track.title),
                "album": str(track.album) if track.album else None,
                "timestamp": datetime.fromtimestamp(int(track.timestamp)),
            }

        if len(scrobbles) < limit:
            break
        page += 1
        time.sleep(0.25)  # Rate limiting


def build_song_lookup(songs: list) -> dict[tuple[str, str], str]:
    """Build a (artist_lower, title_lower) -> spotify_id lookup dict from Song objects."""
    lookup = {}
    for song in songs:
        key = (song.artist.lower().strip(), song.title.lower().strip())
        lookup[key] = song.spotify_id
    return lookup


def match_lastfm_to_spotify(
    scrobble_artist: str,
    scrobble_title: str,
    lookup: dict[tuple[str, str], str],
    threshold: int = 75,
) -> str | None:
    """Match a Last.fm scrobble to a Spotify track.

    Two-tier matching:
    1. Exact match on (artist, title) — O(1) lookup
    2. Fuzzy match with weighted scoring — 60% title, 40% artist

    Returns spotify_id or None.
    """
    artist_lower = scrobble_artist.lower().strip()
    title_lower = scrobble_title.lower().strip()

    # Tier 1: Exact match
    exact = lookup.get((artist_lower, title_lower))
    if exact:
        return exact

    # Tier 2: Fuzzy match
    best_score = 0
    best_id = None

    for (lib_artist, lib_title), spotify_id in lookup.items():
        title_score = fuzz.ratio(title_lower, lib_title)
        artist_score = fuzz.ratio(artist_lower, lib_artist)
        combined = (title_score * 0.6) + (artist_score * 0.4)

        if combined > best_score:
            best_score = combined
            best_id = spotify_id

    if best_score >= threshold:
        return best_id
    return None


def fetch_lastfm_tags(
    network: pylast.LastFMNetwork,
    artist: str,
    title: str,
    max_tags: int = 5,
) -> list[str]:
    """Fetch Last.fm tags for a track, falling back to artist tags.

    Returns list of tag names, respecting rate limits.
    """
    tags = []

    # Try track tags first
    try:
        track = network.get_track(artist, title)
        top_tags = track.get_top_tags(limit=max_tags)
        tags = [str(tag.item) for tag in top_tags if tag.weight and int(tag.weight) > 0]
    except Exception:
        pass

    # Fall back to artist tags if no track tags
    if not tags:
        try:
            artist_obj = network.get_artist(artist)
            top_tags = artist_obj.get_top_tags(limit=max_tags)
            tags = [str(tag.item) for tag in top_tags if tag.weight and int(tag.weight) > 0]
        except Exception:
            pass

    time.sleep(0.25)  # Rate limit: ~4 req/s
    return tags

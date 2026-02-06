"""Spotify API wrapper using spotipy."""

import os
from typing import Iterator

import spotipy
from spotipy.oauth2 import SpotifyOAuth

from src.data.db import load_config, load_env

SCOPES = (
    "user-library-read "
    "playlist-read-private "
    "playlist-modify-public "
    "playlist-modify-private"
)


def get_spotify_client() -> spotipy.Spotify:
    """Create an authenticated Spotify client using Authorization Code Flow."""
    load_env()
    return spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            client_id=os.environ["SPOTIPY_CLIENT_ID"],
            client_secret=os.environ["SPOTIPY_CLIENT_SECRET"],
            redirect_uri=os.environ["SPOTIPY_REDIRECT_URI"],
            scope=SCOPES,
        )
    )


def chunked(lst: list, size: int) -> Iterator[list]:
    """Yield successive chunks of `size` from `lst`."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def get_all_user_playlists(sp: spotipy.Spotify) -> list[dict]:
    """Get all playlists owned by the current user (paginated)."""
    user_id = sp.me()["id"]
    playlists = []
    offset = 0
    while True:
        batch = sp.current_user_playlists(limit=50, offset=offset)
        for item in batch["items"]:
            if item["owner"]["id"] == user_id:
                playlists.append(item)
        if batch["next"] is None:
            break
        offset += 50
    return playlists


def get_all_playlist_tracks(sp: spotipy.Spotify, playlist_id: str) -> list[dict]:
    """Get all tracks from a playlist (paginated at 100/page)."""
    tracks = []
    fields = "items(added_at,track(id,name,artists,album,duration_ms,popularity)),next"
    offset = 0
    while True:
        batch = sp.playlist_items(playlist_id, fields=fields, limit=100, offset=offset)
        for item in batch["items"]:
            if item.get("track") and item["track"].get("id"):
                tracks.append(item)
        if batch["next"] is None:
            break
        offset += 100
    return tracks


def get_liked_songs(sp: spotipy.Spotify) -> list[dict]:
    """Get all liked/saved songs (paginated at 50/page)."""
    tracks = []
    offset = 0
    while True:
        batch = sp.current_user_saved_tracks(limit=50, offset=offset)
        for item in batch["items"]:
            if item.get("track") and item["track"].get("id"):
                tracks.append(item)
        if batch["next"] is None:
            break
        offset += 50
    return tracks


def try_fetch_audio_features(
    sp: spotipy.Spotify, track_ids: list[str]
) -> list[dict | None] | None:
    """Attempt to fetch audio features. Returns None on 403 (deprecated API).

    Batches at 100 IDs per request (Spotify API limit).
    """
    all_features = []
    for chunk in chunked(track_ids, 100):
        try:
            features = sp.audio_features(chunk)
            all_features.extend(features)
        except spotipy.SpotifyException as e:
            if e.http_status == 403:
                return None
            raise
    return all_features


def get_or_create_playlist(
    sp: spotipy.Spotify, name: str, public: bool = False
) -> str:
    """Find an existing playlist by name, or create one. Returns playlist ID."""
    config = load_config()
    playlist_name = name or config.get("playlist", {}).get("name", "spotifymanager generated")

    user_id = sp.me()["id"]
    playlists = get_all_user_playlists(sp)
    for pl in playlists:
        if pl["name"] == playlist_name:
            return pl["id"]

    result = sp.user_playlist_create(user_id, playlist_name, public=public)
    return result["id"]

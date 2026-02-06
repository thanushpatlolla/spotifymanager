"""Tests for Last.fm to Spotify fuzzy matching logic."""

import pytest

from src.data.lastfm_client import build_song_lookup, match_lastfm_to_spotify


class FakeSong:
    """Minimal song object for testing."""

    def __init__(self, spotify_id: str, artist: str, title: str):
        self.spotify_id = spotify_id
        self.artist = artist
        self.title = title


@pytest.fixture
def lookup():
    """Build a test lookup from fake songs."""
    songs = [
        FakeSong("id1", "Radiohead", "Karma Police"),
        FakeSong("id2", "Kendrick Lamar", "HUMBLE."),
        FakeSong("id3", "The National", "Bloodbuzz Ohio"),
        FakeSong("id4", "Bon Iver", "Skinny Love"),
        FakeSong("id5", "LCD Soundsystem", "All My Friends"),
    ]
    return build_song_lookup(songs)


def test_exact_match(lookup):
    """Exact artist + title match should return the correct spotify_id."""
    result = match_lastfm_to_spotify("Radiohead", "Karma Police", lookup)
    assert result == "id1"


def test_exact_match_case_insensitive(lookup):
    """Matching should be case-insensitive."""
    result = match_lastfm_to_spotify("radiohead", "karma police", lookup)
    assert result == "id1"


def test_exact_match_with_whitespace(lookup):
    """Matching should handle leading/trailing whitespace."""
    result = match_lastfm_to_spotify("  Radiohead  ", "  Karma Police  ", lookup)
    assert result == "id1"


def test_fuzzy_match_slight_difference(lookup):
    """Slight title differences should still match (e.g., missing punctuation)."""
    result = match_lastfm_to_spotify("Kendrick Lamar", "HUMBLE", lookup)
    assert result == "id2"


def test_fuzzy_match_artist_variation(lookup):
    """Artist name variations should fuzzy match."""
    result = match_lastfm_to_spotify("The National", "Blood Buzz Ohio", lookup)
    assert result == "id3"


def test_no_match_completely_different(lookup):
    """Completely different track should return None."""
    result = match_lastfm_to_spotify("Unknown Artist", "Unknown Song XYZ", lookup)
    assert result is None


def test_no_match_below_threshold(lookup):
    """Partial matches below threshold should return None."""
    result = match_lastfm_to_spotify("Bon Iver", "Totally Different Song", lookup, threshold=90)
    assert result is None


def test_build_song_lookup():
    """build_song_lookup creates correct (artist, title) -> id mapping."""
    songs = [
        FakeSong("a1", "Artist One", "Song One"),
        FakeSong("a2", "Artist Two", "Song Two"),
    ]
    lookup = build_song_lookup(songs)
    assert lookup[("artist one", "song one")] == "a1"
    assert lookup[("artist two", "song two")] == "a2"
    assert len(lookup) == 2


def test_empty_lookup():
    """Matching against empty lookup returns None."""
    result = match_lastfm_to_spotify("Any Artist", "Any Song", {})
    assert result is None

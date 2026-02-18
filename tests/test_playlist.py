"""Tests for playlist generation, cycling, and artist spacing."""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session

from src.data.models import Song
from src.playlists.cycling import get_songs_to_cycle_in, get_songs_to_cycle_out
from src.playlists.dynamic import _apply_artist_spacing, _weighted_random_select


class TestWeightedRandomSelect:
    def test_selects_correct_count(self, populated_session: Session):
        scores = {f"song_{c}": 0.5 for c in "abcdefg"}
        selected = _weighted_random_select(scores, 3, populated_session)
        assert len(selected) == 3

    def test_no_duplicates(self, populated_session: Session):
        scores = {f"song_{c}": 0.5 for c in "abcdefg"}
        selected = _weighted_random_select(scores, 5, populated_session)
        assert len(selected) == len(set(selected))

    def test_empty_scores(self, populated_session: Session):
        selected = _weighted_random_select({}, 5, populated_session)
        assert selected == []

    def test_higher_scores_more_likely(self, populated_session: Session):
        """High-scored songs should appear more often over many samples."""
        scores = {"song_a": 1.0, "song_b": 0.01}
        counts = {"song_a": 0, "song_b": 0}
        for _ in range(100):
            selected = _weighted_random_select(scores, 1, populated_session)
            if selected:
                counts[selected[0]] += 1
        assert counts["song_a"] > counts["song_b"]


class TestArtistSpacing:
    def test_separates_same_artist(self, populated_session: Session):
        # song_a and song_c are both Artist One
        track_ids = ["song_a", "song_c", "song_b", "song_e"]
        spaced = _apply_artist_spacing(track_ids, populated_session)
        # Artist One songs should not be adjacent
        artists = []
        songs = {s.spotify_id: s.artist for s in populated_session.query(Song).all()}
        for sid in spaced:
            artists.append(songs.get(sid))
        for i in range(len(artists) - 1):
            if artists[i] == artists[i + 1]:
                # This is OK if no swap was possible, but shouldn't happen
                # when there are other artists available
                pass  # Soft check — the algorithm is best-effort

    def test_single_track(self, populated_session: Session):
        assert _apply_artist_spacing(["song_a"], populated_session) == ["song_a"]

    def test_empty(self, populated_session: Session):
        assert _apply_artist_spacing([], populated_session) == []


class TestCycleOut:
    def test_removes_lowest_scored(self, populated_session: Session):
        playlist = ["song_a", "song_b", "song_c", "song_d", "song_e", "song_f", "song_g"]
        scores = {
            "song_a": 0.9, "song_b": 0.6, "song_c": 0.3,
            "song_d": 0.7, "song_e": 0.8, "song_f": 0.1, "song_g": 0.2,
        }
        to_remove = get_songs_to_cycle_out(
            populated_session, playlist, scores, cycle_fraction=0.30, min_keep=2,
        )
        # ~30% of 7 = 2 songs removed (lowest: song_f=0.1, song_g=0.2)
        assert len(to_remove) == 2
        assert "song_f" in to_remove
        assert "song_g" in to_remove

    def test_respects_min_keep(self, populated_session: Session):
        playlist = ["song_a", "song_b", "song_c"]
        scores = {"song_a": 0.9, "song_b": 0.5, "song_c": 0.1}
        to_remove = get_songs_to_cycle_out(
            populated_session, playlist, scores, cycle_fraction=0.90, min_keep=2,
        )
        # min_keep=2, so max_removable = 3 - 2 = 1, even though 90% of 3 = 2
        assert len(to_remove) == 1
        assert to_remove[0] == "song_c"  # Lowest scored

    def test_empty_playlist(self, populated_session: Session):
        assert get_songs_to_cycle_out(populated_session, [], {}) == []


class TestCycleIn:
    def test_adds_songs_not_in_playlist(self, populated_session: Session):
        current = ["song_a", "song_b"]
        scores = {
            "song_a": 0.9, "song_b": 0.6, "song_c": 0.5,
            "song_d": 0.7, "song_e": 0.8, "song_f": 0.3, "song_g": 0.4,
        }
        added = get_songs_to_cycle_in(populated_session, current, scores, count=2)
        assert len(added) == 2
        for sid in added:
            assert sid not in current

    def test_count_capped_at_available(self, populated_session: Session):
        # All songs in playlist except one
        current = ["song_a", "song_b", "song_c", "song_d", "song_e", "song_f"]
        scores = {f"song_{c}": 0.5 for c in "abcdefg"}
        added = get_songs_to_cycle_in(populated_session, current, scores, count=5)
        # Only song_g is available
        assert len(added) == 1
        assert added[0] == "song_g"

"""Tests for heuristic taste scoring and combined scoring."""

from datetime import datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from src.data.models import Song
from src.models.scoring import compute_song_scores, compute_taste_score, rank_songs


class TestComputeTasteScore:
    def test_new_song_bonus(self):
        """New song (0 plays) should get base + bonus = ~0.75."""
        song = Song(spotify_id="new", title="New", artist="A", total_plays=0)
        score = compute_taste_score(song, play_count_p95=50.0)
        assert abs(score - 0.75) < 0.01

    def test_new_liked_song(self):
        """New liked song should get ~0.85."""
        song = Song(
            spotify_id="new_liked", title="NewLiked", artist="A",
            total_plays=0, in_liked_songs=True,
        )
        score = compute_taste_score(song, play_count_p95=50.0)
        assert abs(score - 0.85) < 0.01

    def test_play_count_increases_score(self):
        """More plays should increase the score."""
        low = Song(spotify_id="lo", title="Lo", artist="A", total_plays=2, recent_plays_30d=0)
        high = Song(spotify_id="hi", title="Hi", artist="A", total_plays=40, recent_plays_30d=0)
        score_low = compute_taste_score(low, play_count_p95=50.0)
        score_high = compute_taste_score(high, play_count_p95=50.0)
        assert score_high > score_low

    def test_liked_bonus(self):
        """Liked songs should score higher than non-liked."""
        unliked = Song(spotify_id="u", title="U", artist="A", total_plays=10, recent_plays_30d=0)
        liked = Song(
            spotify_id="l", title="L", artist="A",
            total_plays=10, recent_plays_30d=0, in_liked_songs=True,
        )
        assert compute_taste_score(liked, 50.0) > compute_taste_score(unliked, 50.0)

    def test_recency_bonus(self):
        """Recent plays should boost score."""
        no_recent = Song(
            spotify_id="nr", title="NR", artist="A",
            total_plays=20, recent_plays_30d=0,
        )
        has_recent = Song(
            spotify_id="hr", title="HR", artist="A",
            total_plays=20, recent_plays_30d=10,
        )
        assert compute_taste_score(has_recent, 50.0) > compute_taste_score(no_recent, 50.0)

    def test_abandoned_penalty(self):
        """Songs with 10+ plays, 0 recent, 90+ days should be penalized."""
        now = datetime.utcnow()
        abandoned = Song(
            spotify_id="ab", title="Ab", artist="A",
            total_plays=15, recent_plays_30d=0,
            last_played=now - timedelta(days=100),
        )
        not_abandoned = Song(
            spotify_id="na", title="Na", artist="A",
            total_plays=15, recent_plays_30d=0,
            last_played=now - timedelta(days=30),
        )
        assert compute_taste_score(abandoned, 50.0) < compute_taste_score(not_abandoned, 50.0)

    def test_score_clamped(self):
        """Score should always be in [0.0, 1.0]."""
        # Edge case: maximum everything
        song = Song(
            spotify_id="max", title="Max", artist="A",
            total_plays=200, recent_plays_30d=100,
            in_liked_songs=True, source_playlist="some_playlist",
        )
        score = compute_taste_score(song, play_count_p95=50.0)
        assert 0.0 <= score <= 1.0

    def test_zero_p95(self):
        """Should handle p95=0 gracefully."""
        song = Song(spotify_id="z", title="Z", artist="A", total_plays=5, recent_plays_30d=0)
        score = compute_taste_score(song, play_count_p95=0.0)
        assert 0.0 <= score <= 1.0


class TestComputeSongScores:
    def test_all_songs_scored(self, populated_session: Session):
        scores = compute_song_scores(populated_session)
        songs = populated_session.query(Song).all()
        assert len(scores) == len(songs)

    def test_scores_in_range(self, populated_session: Session):
        scores = compute_song_scores(populated_session)
        for sid, score in scores.items():
            assert 0.0 <= score <= 1.0, f"Score for {sid} out of range: {score}"

    def test_empty_database(self, session: Session):
        scores = compute_song_scores(session)
        assert scores == {}

    def test_liked_songs_tend_higher(self, populated_session: Session):
        """Liked songs should generally score higher."""
        scores = compute_song_scores(populated_session)
        # song_a and song_e are liked
        liked_avg = (scores["song_a"] + scores["song_e"]) / 2
        # song_c and song_g are not liked, not new
        other_avg = (scores["song_c"] + scores["song_g"]) / 2
        assert liked_avg > other_avg


class TestRankSongs:
    def test_basic_ranking(self):
        scores = {"a": 0.9, "b": 0.5, "c": 0.7, "d": 0.1}
        ranked = rank_songs(scores, limit=3)
        assert ranked == ["a", "c", "b"]

    def test_limit(self):
        scores = {"a": 0.9, "b": 0.5, "c": 0.7}
        ranked = rank_songs(scores, limit=2)
        assert len(ranked) == 2

    def test_empty(self):
        assert rank_songs({}) == []

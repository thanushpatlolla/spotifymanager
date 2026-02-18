"""Tests for tag-based mood vector computation."""

from datetime import datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from src.data.models import ListenEvent, Song
from src.features.mood import (
    build_tag_vocabulary,
    compute_mood_vector,
    compute_song_mood_similarity,
    parse_tags,
    song_to_tag_vector,
)


class TestParseTags:
    def test_basic(self):
        assert parse_tags("rock, pop, indie") == ["rock", "pop", "indie"]

    def test_whitespace(self):
        assert parse_tags("  rock ,  pop  , indie  ") == ["rock", "pop", "indie"]

    def test_case_normalization(self):
        assert parse_tags("Rock, POP, Indie") == ["rock", "pop", "indie"]

    def test_empty_string(self):
        assert parse_tags("") == []

    def test_none(self):
        assert parse_tags(None) == []

    def test_single_tag(self):
        assert parse_tags("rock") == ["rock"]

    def test_empty_segments(self):
        assert parse_tags("rock,,pop") == ["rock", "pop"]


class TestBuildTagVocabulary:
    def test_filters_rare_tags(self, populated_session: Session):
        vocab = build_tag_vocabulary(populated_session, min_count=2)
        # "rock" appears in song_a, song_c, song_e -> count 3
        assert "rock" in vocab
        # "alternative" appears in song_a, song_e -> count 2
        assert "alternative" in vocab
        # "pop" appears only in song_f -> count 1, filtered out
        assert "pop" not in vocab

    def test_min_count_1_includes_all(self, populated_session: Session):
        vocab = build_tag_vocabulary(populated_session, min_count=1)
        assert "pop" in vocab
        assert "ambient" in vocab

    def test_empty_database(self, session: Session):
        vocab = build_tag_vocabulary(session, min_count=1)
        assert vocab == {}

    def test_indices_are_unique(self, populated_session: Session):
        vocab = build_tag_vocabulary(populated_session, min_count=1)
        indices = list(vocab.values())
        assert len(indices) == len(set(indices))


class TestSongToTagVector:
    def test_basic(self, populated_session: Session):
        vocab = build_tag_vocabulary(populated_session, min_count=1)
        song = populated_session.get(Song,"song_a")
        vec = song_to_tag_vector(song, vocab)
        # song_a has "rock, alternative"
        assert vocab["rock"] in vec
        assert vocab["alternative"] in vec
        assert all(v == 1.0 for v in vec.values())

    def test_no_tags(self, populated_session: Session):
        vocab = build_tag_vocabulary(populated_session, min_count=1)
        song = populated_session.get(Song,"song_g")
        vec = song_to_tag_vector(song, vocab)
        assert vec == {}

    def test_tags_not_in_vocab(self, session: Session):
        song = Song(spotify_id="x", title="X", artist="X", lastfm_tags="unknown_tag")
        session.add(song)
        session.flush()
        vec = song_to_tag_vector(song, {})
        assert vec == {}


class TestComputeMoodVector:
    def test_returns_vector_and_vocab(self, populated_session: Session):
        mood, vocab = compute_mood_vector(populated_session, window_hours=4)
        assert mood is not None
        assert len(vocab) > 0
        # Vector should be L2-normalized (magnitude ~1.0)
        import math
        magnitude = math.sqrt(sum(v * v for v in mood.values()))
        assert abs(magnitude - 1.0) < 0.01

    def test_no_recent_listening_falls_back(self, populated_session: Session):
        # Use a 0-hour window so no recent events qualify
        mood, vocab = compute_mood_vector(populated_session, window_hours=0)
        # Should fall back to top played songs
        assert mood is not None

    def test_empty_database(self, session: Session):
        mood, vocab = compute_mood_vector(session, window_hours=4)
        assert mood is None
        assert vocab == {}


class TestComputeSongMoodSimilarity:
    def test_similar_song(self, populated_session: Session):
        mood, vocab = compute_mood_vector(populated_session, window_hours=4)
        assert mood is not None
        # song_a has "rock, alternative" — should be similar to mood
        # (since song_a was recently played)
        song_a = populated_session.get(Song,"song_a")
        sim = compute_song_mood_similarity(song_a, mood, vocab)
        assert 0.0 <= sim <= 1.0
        assert sim > 0.3  # Should have decent similarity

    def test_no_tags_returns_default(self, populated_session: Session):
        mood, vocab = compute_mood_vector(populated_session, window_hours=4)
        assert mood is not None
        song_g = populated_session.get(Song,"song_g")
        sim = compute_song_mood_similarity(song_g, mood, vocab)
        assert sim == 0.5  # Default for no-tag songs

    def test_similarity_range(self, populated_session: Session):
        mood, vocab = compute_mood_vector(populated_session, window_hours=4)
        assert mood is not None
        for song in populated_session.query(Song).all():
            sim = compute_song_mood_similarity(song, mood, vocab)
            assert 0.0 <= sim <= 1.0

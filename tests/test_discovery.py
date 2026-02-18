"""Tests for multi-source music discovery."""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session

from src.data.models import DiscoveryCandidate, Song
from src.models.discovery import evaluate_candidates, save_candidates


class TestEvaluateCandidates:
    def test_filters_below_threshold(self, populated_session: Session):
        candidates = [
            {"title": "Low", "artist": "Unknown", "score": 0.1},
            {"title": "High", "artist": "Unknown", "score": 0.8},
        ]
        evaluated = evaluate_candidates(populated_session, candidates)
        # min_score=0.5 from config, so only High passes
        assert len(evaluated) == 1
        assert evaluated[0]["title"] == "High"

    def test_boosts_known_artist(self, populated_session: Session):
        candidates = [
            {"title": "New Song", "artist": "Artist One", "score": 0.4},
        ]
        evaluated = evaluate_candidates(populated_session, candidates)
        # Artist One is in library, so score gets +0.15 -> 0.55, passes threshold
        assert len(evaluated) == 1
        assert evaluated[0]["score"] > 0.4

    def test_sorted_by_score(self, populated_session: Session):
        candidates = [
            {"title": "A", "artist": "X", "score": 0.6},
            {"title": "B", "artist": "X", "score": 0.9},
            {"title": "C", "artist": "X", "score": 0.7},
        ]
        evaluated = evaluate_candidates(populated_session, candidates)
        scores = [c["score"] for c in evaluated]
        assert scores == sorted(scores, reverse=True)

    def test_empty_candidates(self, populated_session: Session):
        assert evaluate_candidates(populated_session, []) == []


class TestSaveCandidates:
    def test_saves_new_candidates(self, session: Session):
        candidates = [
            {"title": "New Song", "artist": "New Artist", "source": "lastfm", "score": 0.7},
            {"title": "Another", "artist": "Another Artist", "source": "lb", "score": 0.6},
        ]
        saved = save_candidates(session, candidates)
        assert saved == 2
        assert session.query(DiscoveryCandidate).count() == 2

    def test_skips_duplicates(self, session: Session):
        candidates = [{"title": "Song", "artist": "Art", "source": "test", "score": 0.5}]
        save_candidates(session, candidates)
        # Save again — should update, not duplicate
        saved = save_candidates(session, candidates)
        assert saved == 0
        assert session.query(DiscoveryCandidate).count() == 1

    def test_empty_candidates(self, session: Session):
        assert save_candidates(session, []) == 0

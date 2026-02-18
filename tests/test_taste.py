"""Tests for taste model data preparation."""

import numpy as np
import pytest
from sqlalchemy.orm import Session

from src.models.taste import prepare_training_data


class TestPrepareTrainingData:
    def test_returns_correct_shapes(self, populated_session: Session):
        X, y, feature_names = prepare_training_data(populated_session)
        assert X.shape[0] == 7  # 7 songs in populated_session
        assert X.shape[1] == len(feature_names)
        assert y.shape[0] == 7

    def test_labels_are_binary(self, populated_session: Session):
        X, y, feature_names = prepare_training_data(populated_session)
        assert set(y.tolist()).issubset({0, 1})

    def test_has_both_classes(self, populated_session: Session):
        X, y, feature_names = prepare_training_data(populated_session)
        # song_d has 0 plays (negative), song_f has 100 plays (positive)
        assert 0 in y
        assert 1 in y

    def test_feature_names_include_audio_and_tags(self, populated_session: Session):
        X, y, feature_names = prepare_training_data(populated_session)
        assert "danceability" in feature_names
        assert "liked" in feature_names
        # Should have tag features
        tag_features = [f for f in feature_names if f.startswith("tag_")]
        assert len(tag_features) > 0

    def test_values_in_range(self, populated_session: Session):
        X, y, feature_names = prepare_training_data(populated_session)
        assert np.all(X >= 0.0)
        assert np.all(X <= 1.5)  # Allow slight overshoot for normalized features

    def test_empty_database(self, session: Session):
        X, y, feature_names = prepare_training_data(session)
        assert len(X) == 0

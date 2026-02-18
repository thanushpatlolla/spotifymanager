"""Tests for feature vector generation."""

import numpy as np
import pytest
from sqlalchemy.orm import Session

from src.features.embeddings import build_feature_vectors


class TestBuildFeatureVectors:
    def test_returns_vectors_for_all_songs(self, populated_session: Session):
        vectors = build_feature_vectors(populated_session)
        # populated_session has 7 songs
        assert len(vectors) == 7

    def test_vectors_are_numpy(self, populated_session: Session):
        vectors = build_feature_vectors(populated_session)
        for sid, vec in vectors.items():
            assert isinstance(vec, np.ndarray)
            assert vec.dtype == np.float32

    def test_vectors_same_dimension(self, populated_session: Session):
        vectors = build_feature_vectors(populated_session)
        dims = {vec.shape[0] for vec in vectors.values()}
        assert len(dims) == 1  # All same dimension

    def test_vectors_have_expected_components(self, populated_session: Session):
        vectors = build_feature_vectors(populated_session)
        dim = next(iter(vectors.values())).shape[0]
        # At least 12 audio + some tags + 4 stats = 16+
        assert dim >= 16

    def test_empty_database(self, session: Session):
        vectors = build_feature_vectors(session)
        assert vectors == {}

    def test_values_in_reasonable_range(self, populated_session: Session):
        vectors = build_feature_vectors(populated_session)
        for sid, vec in vectors.items():
            assert np.all(vec >= 0.0), f"Negative values for {sid}"
            assert np.all(vec <= 1.5), f"Values too large for {sid}"

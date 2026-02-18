"""Tests for clustering (unit tests that don't require hdbscan/umap)."""

import pytest
from sqlalchemy.orm import Session

from src.data.models import Cluster, Song
from src.models.clustering import assign_clusters, get_cluster_summary


class TestAssignClusters:
    def test_assigns_songs(self, populated_session: Session):
        ids = ["song_a", "song_b", "song_c"]
        labels = [0, 0, 1]
        assigned = assign_clusters(populated_session, ids, labels)
        assert assigned == 3

        song_a = populated_session.get(Song, "song_a")
        song_b = populated_session.get(Song, "song_b")
        assert song_a.cluster_id == song_b.cluster_id  # Both cluster 0

        song_c = populated_session.get(Song, "song_c")
        assert song_c.cluster_id != song_a.cluster_id  # Different cluster

    def test_noise_label_unassigns(self, populated_session: Session):
        ids = ["song_a", "song_b"]
        labels = [0, -1]  # song_b is noise
        assign_clusters(populated_session, ids, labels)

        song_b = populated_session.get(Song, "song_b")
        assert song_b.cluster_id is None

    def test_empty_input(self, populated_session: Session):
        assert assign_clusters(populated_session, [], []) == 0

    def test_creates_cluster_records(self, populated_session: Session):
        ids = ["song_a", "song_b"]
        labels = [0, 1]
        assign_clusters(populated_session, ids, labels)
        clusters = populated_session.query(Cluster).all()
        assert len(clusters) >= 2


class TestGetClusterSummary:
    def test_returns_summaries(self, populated_session: Session):
        # Assign some clusters first
        ids = ["song_a", "song_b", "song_c", "song_e"]
        labels = [0, 0, 1, 1]
        assign_clusters(populated_session, ids, labels)

        summaries = get_cluster_summary(populated_session)
        assert len(summaries) == 2

        for s in summaries:
            assert "id" in s
            assert "name" in s
            assert "song_count" in s
            assert s["song_count"] >= 1
            assert "top_tags" in s
            assert "top_songs" in s

    def test_empty_database(self, session: Session):
        assert get_cluster_summary(session) == []

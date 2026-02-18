"""Tests for playlist snapshot save/load/diff."""

import pytest
from sqlalchemy.orm import Session

from src.playlists.snapshot import diff_snapshots, get_latest_snapshot, save_snapshot


class TestSaveAndLoad:
    def test_round_trip(self, session: Session):
        tracks = ["id_1", "id_2", "id_3"]
        snap_id = save_snapshot(session, "playlist_abc", tracks)
        assert snap_id > 0

        loaded = get_latest_snapshot(session, "playlist_abc")
        assert loaded == tracks

    def test_latest_snapshot(self, session: Session):
        save_snapshot(session, "pl1", ["a", "b"])
        save_snapshot(session, "pl1", ["c", "d", "e"])

        loaded = get_latest_snapshot(session, "pl1")
        assert loaded == ["c", "d", "e"]

    def test_no_snapshot(self, session: Session):
        assert get_latest_snapshot(session, "nonexistent") is None

    def test_different_playlists(self, session: Session):
        save_snapshot(session, "pl1", ["a"])
        save_snapshot(session, "pl2", ["b"])

        assert get_latest_snapshot(session, "pl1") == ["a"]
        assert get_latest_snapshot(session, "pl2") == ["b"]

    def test_empty_track_list(self, session: Session):
        save_snapshot(session, "pl_empty", [])
        assert get_latest_snapshot(session, "pl_empty") == []


class TestDiffSnapshots:
    def test_basic_diff(self):
        old = ["a", "b", "c"]
        new = ["b", "c", "d"]
        diff = diff_snapshots(old, new)
        assert diff["added"] == ["d"]
        assert diff["removed"] == ["a"]
        assert set(diff["kept"]) == {"b", "c"}

    def test_no_changes(self):
        tracks = ["a", "b", "c"]
        diff = diff_snapshots(tracks, tracks)
        assert diff["added"] == []
        assert diff["removed"] == []
        assert diff["kept"] == tracks

    def test_complete_replacement(self):
        diff = diff_snapshots(["a", "b"], ["c", "d"])
        assert diff["added"] == ["c", "d"]
        assert diff["removed"] == ["a", "b"]
        assert diff["kept"] == []

    def test_empty_old(self):
        diff = diff_snapshots([], ["a", "b"])
        assert diff["added"] == ["a", "b"]
        assert diff["removed"] == []

    def test_empty_new(self):
        diff = diff_snapshots(["a", "b"], [])
        assert diff["removed"] == ["a", "b"]
        assert diff["added"] == []

    def test_preserves_order(self):
        diff = diff_snapshots(["a", "c"], ["c", "b", "a"])
        # "added" preserves new list order for items not in old
        assert diff["added"] == ["b"]
        # "kept" preserves new list order for items in old
        assert diff["kept"] == ["c", "a"]

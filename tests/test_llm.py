"""Tests for LLM modules (mocked — no API calls)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.llm.cluster_naming import name_cluster
from src.llm.discovery_context import generate_discovery_context, summarize_playlist_reasoning


class TestNameCluster:
    def test_parses_json_response(self):
        mock_client = MagicMock()
        response_json = json.dumps({"name": "Chill Vibes", "description": "Relaxing ambient tracks"})

        with patch("src.llm.cluster_naming.prompt_claude", return_value=response_json):
            result = name_cluster(mock_client, [
                {"title": "Song A", "artist": "Artist A", "tags": "chill, ambient"},
            ])

        assert result["name"] == "Chill Vibes"
        assert result["description"] == "Relaxing ambient tracks"

    def test_handles_non_json_response(self):
        mock_client = MagicMock()

        with patch("src.llm.cluster_naming.prompt_claude", return_value="Energetic Rock Anthems"):
            result = name_cluster(mock_client, [
                {"title": "Song A", "artist": "Artist A", "tags": "rock"},
            ])

        assert "name" in result
        assert len(result["name"]) > 0


class TestDiscoveryContext:
    def test_returns_string(self):
        mock_client = MagicMock()

        with patch("src.llm.discovery_context.prompt_claude", return_value="Great match because..."):
            result = generate_discovery_context(
                mock_client,
                candidate={"title": "Song", "artist": "Art", "tags": "rock", "score": 0.8},
                user_profile={"top_tags": ["rock"], "top_artists": ["Art"], "cluster_names": ["Rock"]},
            )

        assert isinstance(result, str)
        assert len(result) > 0


class TestPlaylistReasoning:
    def test_returns_string(self):
        mock_client = MagicMock()

        with patch("src.llm.discovery_context.prompt_claude", return_value="This playlist blends..."):
            result = summarize_playlist_reasoning(
                mock_client,
                songs=[{"title": "Song", "artist": "Art"}],
                mood={"top_tags": ["rock"]},
            )

        assert isinstance(result, str)

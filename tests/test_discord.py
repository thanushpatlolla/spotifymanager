"""Tests for Discord bot (basic structure, no live Discord connection)."""

import pytest

from src.interface.discord_bot import SpotifyManagerBot


class TestDiscordBotSetup:
    def test_bot_instantiation(self):
        """Bot can be instantiated without errors."""
        bot = SpotifyManagerBot()
        assert bot.tree is not None

    def test_bot_has_intents(self):
        """Bot is configured with default intents."""
        bot = SpotifyManagerBot()
        assert bot.intents is not None

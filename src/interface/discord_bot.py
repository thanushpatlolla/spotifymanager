"""Sprint 6: Discord bot interface.

Provides a conversational interface for managing playlists,
viewing stats, and triggering refreshes via Discord.
"""


async def start_bot():
    """Initialize and start the Discord bot."""
    raise NotImplementedError("Sprint 6")


async def handle_refresh(interaction):
    """Handle /refresh command — regenerate playlist."""
    raise NotImplementedError("Sprint 6")


async def handle_stats(interaction):
    """Handle /stats command — show library statistics."""
    raise NotImplementedError("Sprint 6")


async def handle_discover(interaction):
    """Handle /discover command — find new music."""
    raise NotImplementedError("Sprint 6")

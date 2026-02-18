"""Discord bot interface.

Provides a conversational interface for managing playlists,
viewing stats, and triggering refreshes via Discord.
"""

import os

import discord
from discord import app_commands
from rich.console import Console
from sqlalchemy import func

from src.data.db import get_session, load_env
from src.data.models import ListenEvent, Song
from src.data.spotify_client import get_spotify_client
from src.features.listening_stats import update_listening_stats

console = Console()


class SpotifyManagerBot(discord.Client):
    """Discord bot with slash commands for spotifymanager."""

    def __init__(self):
        intents = discord.Intents.default()
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        await self.tree.sync()

    async def on_ready(self):
        console.print(f"Discord bot logged in as [green]{self.user}[/green]")


bot = SpotifyManagerBot()


@bot.tree.command(name="refresh", description="Regenerate your Spotify playlist")
async def handle_refresh(interaction: discord.Interaction):
    """Handle /refresh command — regenerate playlist."""
    await interaction.response.defer(thinking=True)

    try:
        from src.playlists.dynamic import refresh_playlist

        load_env()
        sp = get_spotify_client()

        with get_session() as session:
            update_listening_stats(session)
            url = refresh_playlist(session, sp)

        if url:
            await interaction.followup.send(f"Playlist refreshed! {url}")
        else:
            await interaction.followup.send("No tracks available to generate playlist.")
    except Exception as e:
        await interaction.followup.send(f"Error refreshing playlist: {e}")


@bot.tree.command(name="stats", description="Show library statistics")
async def handle_stats(interaction: discord.Interaction):
    """Handle /stats command — show library statistics."""
    await interaction.response.defer(thinking=True)

    try:
        with get_session() as session:
            total_songs = session.query(func.count(Song.spotify_id)).scalar()
            liked_songs = session.query(func.count(Song.spotify_id)).filter(Song.in_liked_songs).scalar()
            with_tags = session.query(func.count(Song.spotify_id)).filter(Song.lastfm_tags.isnot(None)).scalar()
            total_events = session.query(func.count(ListenEvent.id)).scalar()

            top_played = (
                session.query(Song)
                .filter(Song.total_plays > 0)
                .order_by(Song.total_plays.desc())
                .limit(5)
                .all()
            )

            lines = [
                f"**Library Stats**",
                f"Songs: {total_songs} | Liked: {liked_songs} | Tagged: {with_tags}",
                f"Listen events: {total_events}",
            ]

            if top_played:
                lines.append("\n**Top 5 Most Played:**")
                for i, song in enumerate(top_played, 1):
                    lines.append(f"{i}. {song.artist} — {song.title} ({song.total_plays} plays)")

        await interaction.followup.send("\n".join(lines))
    except Exception as e:
        await interaction.followup.send(f"Error fetching stats: {e}")


@bot.tree.command(name="discover", description="Find new music recommendations")
async def handle_discover(interaction: discord.Interaction):
    """Handle /discover command — find new music."""
    await interaction.response.defer(thinking=True)

    try:
        from src.data.lastfm_client import get_lastfm_client
        from src.models.discovery import (
            discover_from_lastfm,
            evaluate_candidates,
            save_candidates,
        )

        load_env()
        network = get_lastfm_client()

        with get_session() as session:
            candidates = discover_from_lastfm(network, session, limit=20)
            evaluated = evaluate_candidates(session, candidates)
            saved = save_candidates(session, evaluated)

            if evaluated:
                lines = [f"**Found {len(evaluated)} recommendations** ({saved} new):"]
                for c in evaluated[:10]:
                    lines.append(f"- {c['artist']} — {c['title']} (score: {c['score']:.2f})")
                await interaction.followup.send("\n".join(lines))
            else:
                await interaction.followup.send("No new recommendations found.")
    except Exception as e:
        await interaction.followup.send(f"Error discovering music: {e}")


async def start_bot():
    """Initialize and start the Discord bot."""
    load_env()
    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        console.print("[red]DISCORD_BOT_TOKEN not set in .env[/red]")
        return
    await bot.start(token)

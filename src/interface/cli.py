"""Click CLI for spotifymanager."""

import click
from rich.console import Console
from rich.table import Table
from sqlalchemy import func

from src.data.db import get_session, init_db
from src.data.models import ListenEvent, Song

console = Console()


@click.group()
def cli():
    """spotifymanager — AI-powered Spotify library manager."""
    pass


@cli.command()
def init():
    """Initialize the database (create tables)."""
    init_db()
    console.print("[green]Database initialized successfully.[/green]")


@cli.command()
def stats():
    """Show library statistics."""
    with get_session() as session:
        total_songs = session.query(func.count(Song.spotify_id)).scalar()
        liked_songs = session.query(func.count(Song.spotify_id)).filter(Song.in_liked_songs).scalar()
        with_features = (
            session.query(func.count(Song.spotify_id))
            .filter(Song.danceability.isnot(None))
            .scalar()
        )
        with_tags = (
            session.query(func.count(Song.spotify_id))
            .filter(Song.lastfm_tags.isnot(None))
            .scalar()
        )
        total_events = session.query(func.count(ListenEvent.id)).scalar()

        table = Table(title="spotifymanager Library Stats")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        table.add_row("Total songs", str(total_songs))
        table.add_row("Liked songs", str(liked_songs))
        table.add_row("With audio features", str(with_features))
        table.add_row("With Last.fm tags", str(with_tags))
        table.add_row("Total listen events", str(total_events))

        console.print(table)

        # Top 10 most played
        if total_events > 0:
            top_played = (
                session.query(Song)
                .filter(Song.total_plays > 0)
                .order_by(Song.total_plays.desc())
                .limit(10)
                .all()
            )
            if top_played:
                console.print()
                top_table = Table(title="Top 10 Most Played")
                top_table.add_column("#", justify="right")
                top_table.add_column("Song")
                top_table.add_column("Artist")
                top_table.add_column("Plays", justify="right")

                for i, song in enumerate(top_played, 1):
                    top_table.add_row(str(i), song.title, song.artist, str(song.total_plays))

                console.print(top_table)


@cli.command()
def refresh():
    """Refresh playlist with current mood and taste (Sprint 3)."""
    console.print("[yellow]Not yet implemented — coming in Sprint 3.[/yellow]")


@cli.command()
def discover():
    """Find new music from external sources (Sprint 4)."""
    console.print("[yellow]Not yet implemented — coming in Sprint 4.[/yellow]")


if __name__ == "__main__":
    cli()

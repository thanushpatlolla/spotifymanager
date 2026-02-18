"""Click CLI for spotifymanager."""

import click
from rich.console import Console
from rich.table import Table
from sqlalchemy import func

from src.data.db import get_session, init_db, load_env
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
def scores():
    """Show top 20 scored songs (diagnostic)."""
    from src.features.mood import compute_mood_vector
    from src.models.scoring import compute_song_scores, compute_taste_score, _compute_p95_plays

    with get_session() as session:
        mood_vector, vocabulary = compute_mood_vector(session)
        all_scores = compute_song_scores(session, mood_vector, vocabulary)

        if not all_scores:
            console.print("[yellow]No songs found to score.[/yellow]")
            return

        p95 = _compute_p95_plays(session)
        top_ids = sorted(all_scores, key=all_scores.get, reverse=True)[:20]
        songs = {
            s.spotify_id: s
            for s in session.query(Song).filter(Song.spotify_id.in_(top_ids)).all()
        }

        table = Table(title="Top 20 Scored Songs")
        table.add_column("#", justify="right")
        table.add_column("Song")
        table.add_column("Artist")
        table.add_column("Plays", justify="right")
        table.add_column("Taste", justify="right")
        table.add_column("Score", justify="right")

        for i, sid in enumerate(top_ids, 1):
            song = songs.get(sid)
            if not song:
                continue
            taste = compute_taste_score(song, p95)
            table.add_row(
                str(i),
                song.title[:40],
                song.artist[:25],
                str(song.total_plays or 0),
                f"{taste:.3f}",
                f"{all_scores[sid]:.3f}",
            )

        console.print(table)
        console.print(f"\nTotal songs scored: [green]{len(all_scores)}[/green]")
        if mood_vector:
            console.print(f"Mood vector dimensions: [green]{len(mood_vector)}[/green]")
        else:
            console.print("[yellow]No mood vector (no tags or recent listening)[/yellow]")


@cli.command()
def refresh():
    """Refresh playlist with current mood and taste."""
    from src.data.spotify_client import get_spotify_client
    from src.features.listening_stats import update_listening_stats
    from src.playlists.dynamic import refresh_playlist

    load_env()
    sp = get_spotify_client()

    with get_session() as session:
        console.print("[bold]Updating listening stats...[/bold]")
        update_listening_stats(session)

        console.print("[bold]Refreshing playlist...[/bold]")
        url = refresh_playlist(session, sp)

        if url:
            console.print(f"\n[bold green]Done![/bold green] {url}")


@cli.command()
def discover():
    """Find new music from external sources (Sprint 4)."""
    console.print("[yellow]Not yet implemented — coming in Sprint 4.[/yellow]")


if __name__ == "__main__":
    cli()

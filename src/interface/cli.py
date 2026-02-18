"""Click CLI for spotifymanager."""

import asyncio

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
    from src.models.scoring import _compute_p95_plays, compute_song_scores, compute_taste_score

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
@click.option("--limit", default=50, help="Max candidates to fetch per source.")
def discover(limit):
    """Find new music from Last.fm and ListenBrainz."""
    from src.data.lastfm_client import get_lastfm_client
    from src.models.discovery import (
        discover_from_lastfm,
        discover_from_listenbrainz,
        evaluate_candidates,
        save_candidates,
    )

    load_env()
    network = get_lastfm_client()

    with get_session() as session:
        console.print("[bold]Discovering from Last.fm...[/bold]")
        lastfm_candidates = discover_from_lastfm(network, session, limit=limit)

        console.print("[bold]Discovering from ListenBrainz...[/bold]")
        lb_candidates = discover_from_listenbrainz(session, limit=limit)

        all_candidates = lastfm_candidates + lb_candidates
        console.print(f"Total raw candidates: {len(all_candidates)}")

        evaluated = evaluate_candidates(session, all_candidates)
        saved = save_candidates(session, evaluated)

        if evaluated:
            table = Table(title=f"Top Discovery Candidates ({saved} new)")
            table.add_column("#", justify="right")
            table.add_column("Song")
            table.add_column("Artist")
            table.add_column("Source")
            table.add_column("Score", justify="right")

            for i, c in enumerate(evaluated[:20], 1):
                table.add_row(
                    str(i),
                    c["title"][:40],
                    c["artist"][:25],
                    c["source"].split(":")[0],
                    f"{c['score']:.3f}",
                )
            console.print(table)
        else:
            console.print("[yellow]No candidates met the minimum score threshold.[/yellow]")


@cli.command()
def cluster():
    """Cluster songs by feature similarity (requires sprint5 deps)."""
    from src.features.embeddings import build_feature_vectors
    from src.models.clustering import (
        assign_clusters,
        cluster_songs,
        get_cluster_summary,
        reduce_dimensions,
    )

    with get_session() as session:
        console.print("[bold]Building feature vectors...[/bold]")
        vectors = build_feature_vectors(session)
        if not vectors:
            console.print("[yellow]No songs to cluster.[/yellow]")
            return

        console.print(f"Built vectors for [green]{len(vectors)}[/green] songs "
                       f"({len(next(iter(vectors.values())))} dims)")

        console.print("[bold]Reducing dimensions with UMAP...[/bold]")
        reduced, _ = reduce_dimensions(vectors)

        console.print("[bold]Clustering with HDBSCAN...[/bold]")
        labels, id_list, _ = cluster_songs(reduced)

        console.print("[bold]Assigning clusters...[/bold]")
        assign_clusters(session, id_list, labels)

        summaries = get_cluster_summary(session)
        if summaries:
            table = Table(title="Song Clusters")
            table.add_column("ID", justify="right")
            table.add_column("Name")
            table.add_column("Songs", justify="right")
            table.add_column("Top Tags")
            table.add_column("Top Song")

            for s in summaries:
                top_song = s["top_songs"][0] if s["top_songs"] else {}
                table.add_row(
                    str(s["id"]),
                    s["name"] or "Unnamed",
                    str(s["song_count"]),
                    ", ".join(s["top_tags"][:3]),
                    f"{top_song.get('artist', '')} — {top_song.get('title', '')}",
                )
            console.print(table)


@cli.command(name="name-clusters")
def name_clusters():
    """Use Claude to name song clusters (requires ANTHROPIC_API_KEY)."""
    from src.llm.client import get_claude_client
    from src.llm.cluster_naming import name_all_clusters

    load_env()
    client = get_claude_client()

    with get_session() as session:
        named = name_all_clusters(session, client)
        if named == 0:
            console.print("[yellow]No unnamed clusters found.[/yellow]")


@cli.command()
def train():
    """Train the LightGBM taste model (requires sprint5 deps)."""
    from src.models.taste import prepare_training_data, save_model, train_taste_model

    with get_session() as session:
        console.print("[bold]Preparing training data...[/bold]")
        X, y, feature_names = prepare_training_data(session)

        if len(X) == 0:
            console.print("[yellow]No songs in library to train on.[/yellow]")
            return

        console.print(f"Training on [green]{len(X)}[/green] songs, "
                       f"[green]{len(feature_names)}[/green] features")
        console.print(f"Positive: {int(y.sum())}, Negative: {len(y) - int(y.sum())}")

        model, metrics = train_taste_model(X, y, feature_names)
        if model:
            save_model(model)
            if metrics.get("top_features"):
                console.print(f"Top features: {', '.join(metrics['top_features'][:5])}")


@cli.command()
def bot():
    """Start the Discord bot (requires DISCORD_BOT_TOKEN)."""
    from src.interface.discord_bot import start_bot

    load_env()
    console.print("[bold]Starting Discord bot...[/bold]")
    asyncio.run(start_bot())


if __name__ == "__main__":
    cli()

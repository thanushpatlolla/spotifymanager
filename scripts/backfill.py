"""One-time backfill script — imports all data from Spotify and Last.fm.

Usage:
    uv run python scripts/backfill.py [--skip-audio-features] [--skip-tags] [--skip-scrobbles]
"""

import argparse
import sys
import time

from rich.console import Console

from src.data.db import get_session, init_db, load_config, load_env
from src.data.ingest import (
    ingest_audio_features,
    ingest_lastfm_tags,
    ingest_liked_songs,
    ingest_playlists,
    ingest_scrobbles,
)
from src.data.lastfm_client import get_lastfm_client
from src.data.models import ListenEvent, Song
from src.data.spotify_client import get_spotify_client
from src.features.listening_stats import update_listening_stats

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Backfill spotifymanager database")
    parser.add_argument("--skip-audio-features", action="store_true", help="Skip audio features fetch")
    parser.add_argument("--skip-tags", action="store_true", help="Skip Last.fm tag fetch")
    parser.add_argument("--skip-scrobbles", action="store_true", help="Skip scrobble history import")
    args = parser.parse_args()

    start_time = time.time()
    config = load_config()

    console.rule("[bold blue]spotifymanager Backfill")

    # Step 1: Initialize database
    console.print("\n[bold]Step 1/8:[/bold] Initializing database...")
    init_db()
    console.print("[green]Done.[/green]")

    # Step 2: Authenticate Spotify
    console.print("\n[bold]Step 2/8:[/bold] Authenticating with Spotify...")
    load_env()
    sp = get_spotify_client()
    user = sp.me()
    console.print(f"[green]Authenticated as:[/green] {user['display_name']} ({user['id']})")

    # Step 3: Ingest playlists
    console.print("\n[bold]Step 3/8:[/bold] Importing playlists...")
    with get_session() as session:
        playlist_count = ingest_playlists(sp, session)

    # Step 4: Ingest liked songs
    console.print("\n[bold]Step 4/8:[/bold] Importing liked songs...")
    with get_session() as session:
        liked_count = ingest_liked_songs(sp, session)

    # Step 5: Audio features
    audio_ok = False
    if args.skip_audio_features:
        console.print("\n[bold]Step 5/8:[/bold] [dim]Skipping audio features (--skip-audio-features)[/dim]")
    else:
        console.print("\n[bold]Step 5/8:[/bold] Fetching audio features...")
        with get_session() as session:
            audio_ok = ingest_audio_features(sp, session)

    # Step 6: Scrobble history
    scrobble_count = 0
    if args.skip_scrobbles:
        console.print("\n[bold]Step 6/8:[/bold] [dim]Skipping scrobbles (--skip-scrobbles)[/dim]")
    else:
        console.print("\n[bold]Step 6/8:[/bold] Importing scrobble history...")
        lastfm_username = config.get("lastfm", {}).get("username", "QCD7")
        network = get_lastfm_client()
        with get_session() as session:
            scrobble_count = ingest_scrobbles(network, session, lastfm_username)

    # Step 7: Last.fm tags
    tag_count = 0
    if args.skip_tags:
        console.print("\n[bold]Step 7/8:[/bold] [dim]Skipping tags (--skip-tags)[/dim]")
    else:
        console.print("\n[bold]Step 7/8:[/bold] Fetching Last.fm tags...")
        if "network" not in dir():
            network = get_lastfm_client()
        with get_session() as session:
            tag_count = ingest_lastfm_tags(network, session)

    # Step 8: Compute listening stats
    console.print("\n[bold]Step 8/8:[/bold] Computing listening stats...")
    with get_session() as session:
        update_listening_stats(session)

    # Summary
    elapsed = time.time() - start_time
    console.rule("[bold green]Backfill Complete")

    with get_session() as session:
        total_songs = session.query(Song).count()
        total_events = session.query(ListenEvent).count()

    console.print(f"\n  Total songs in DB:       [bold]{total_songs}[/bold]")
    console.print(f"  Playlist tracks imported: [bold]{playlist_count}[/bold]")
    console.print(f"  Liked songs imported:     [bold]{liked_count}[/bold]")
    console.print(f"  Audio features:           [bold]{'yes' if audio_ok else 'unavailable (403)'}[/bold]")
    console.print(f"  Scrobble events matched:  [bold]{scrobble_count}[/bold]")
    console.print(f"  Songs tagged:             [bold]{tag_count}[/bold]")
    console.print(f"  Total listen events:      [bold]{total_events}[/bold]")
    console.print(f"  Time elapsed:             [bold]{elapsed:.1f}s[/bold]\n")


if __name__ == "__main__":
    main()

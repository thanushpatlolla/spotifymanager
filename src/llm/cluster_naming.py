"""LLM-powered cluster naming.

Uses Claude to generate descriptive names and descriptions for
song clusters based on their shared characteristics.
"""

import json

from rich.console import Console
from sqlalchemy.orm import Session

from src.data.models import Cluster, Song
from src.llm.client import prompt_claude

console = Console()

SYSTEM_PROMPT = """\
You are a music curator. Given a list of songs in a cluster with their artists and tags, \
generate a short, evocative name (2-4 words) and a one-sentence description for the cluster. \
Respond in JSON: {"name": "...", "description": "..."}"""


def name_cluster(client, songs: list[dict]) -> dict:
    """Generate a name and description for a cluster of songs.

    Args:
        client: Anthropic client.
        songs: List of song dicts with title, artist, tags.

    Returns dict with 'name' and 'description'.
    """
    song_lines = []
    for s in songs[:15]:  # Limit context size
        tags = s.get("tags", "")
        song_lines.append(f"- {s['artist']} — {s['title']} [{tags}]")
    song_list = "\n".join(song_lines)

    user_prompt = f"Name this cluster of {len(songs)} songs:\n\n{song_list}"
    response = prompt_claude(client, SYSTEM_PROMPT, user_prompt, max_tokens=200)

    try:
        result = json.loads(response)
        return {"name": result.get("name", "Unnamed"), "description": result.get("description", "")}
    except (json.JSONDecodeError, KeyError):
        # Try to extract from non-JSON response
        return {"name": response[:50].strip(), "description": ""}


def name_all_clusters(session: Session, client) -> int:
    """Name all unnamed clusters. Returns count of clusters named."""
    clusters = (
        session.query(Cluster)
        .filter(Cluster.description.is_(None))
        .all()
    )

    if not clusters:
        console.print("All clusters already named.")
        return 0

    named = 0
    for cluster in clusters:
        songs = (
            session.query(Song)
            .filter(Song.cluster_id == cluster.id)
            .order_by(Song.total_plays.desc())
            .limit(15)
            .all()
        )

        if not songs:
            continue

        song_dicts = [
            {"title": s.title, "artist": s.artist, "tags": s.lastfm_tags or ""}
            for s in songs
        ]

        result = name_cluster(client, song_dicts)
        cluster.name = result["name"]
        cluster.description = result["description"]
        named += 1

        console.print(f"  Cluster {cluster.id}: [green]{result['name']}[/green]")

    session.flush()
    console.print(f"Named [green]{named}[/green] clusters")
    return named

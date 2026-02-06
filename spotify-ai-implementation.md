# Spotify AI — Implementation Plan

## Summary of Decisions

| Decision | Choice |
|----------|--------|
| Priority | Dynamic playlist → Discovery → Mood adaptation → Clustering |
| Interface | CLI first, Telegram bot later (VPS ~$5/mo) |
| Hosting | Laptop batch mode (→ VPS when adding bot) |
| Cycling | Tunable parameter, one-at-a-time approval |
| Budget | $5–20/month |
| Listening device | Phone (system outputs playlists, no live playback control) |
| Last.fm | A few years of scrobbles (solid training data) |
| Genre coverage | Most of 8k songs already genre-tagged (great for clustering) |
| Discovery | Open to new sources beyond Last.fm/RYM |
| Approval | Interactive — one song at a time |

---

## Project Structure

```
spotify-ai/
├── pyproject.toml              # Dependencies (use Poetry or pip)
├── config.yaml                 # All tunables (cycling rate, playlist sizes, etc.)
├── .env                        # API keys (Spotify, Last.fm, Claude, Telegram)
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── spotify_client.py   # Spotify API wrapper (auth, playlists, audio features)
│   │   ├── lastfm_client.py    # Last.fm scrobble history, similar tracks/artists
│   │   ├── models.py           # SQLAlchemy/dataclass models (SongProfile, ListenEvent, etc.)
│   │   ├── db.py               # Database connection, migrations
│   │   └── ingest.py           # Backfill + ongoing ingestion pipelines
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── embeddings.py       # Build song feature vectors (audio features + tags + embeddings)
│   │   ├── listening_stats.py  # Compute play counts, skip rates, staleness from scrobbles
│   │   └── mood.py             # Current mood vector from recent listening
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── clustering.py       # HDBSCAN clustering, cluster management
│   │   ├── taste.py            # Long-term preference model (LightGBM scorer)
│   │   ├── scoring.py          # Combined scoring: taste × mood × (1-staleness) + exploration
│   │   └── discovery.py        # Multi-source candidate generation + ranking
│   │
│   ├── playlists/
│   │   ├── __init__.py
│   │   ├── dynamic.py          # Dynamic playlist generation and refresh
│   │   ├── cycling.py          # Song cycling logic (propose additions/removals)
│   │   └── snapshot.py         # Periodic playlist snapshots (replacing manual dated playlists)
│   │
│   ├── interface/
│   │   ├── __init__.py
│   │   ├── cli.py              # Click-based CLI (Phase 1 interface)
│   │   └── telegram_bot.py     # Telegram bot (Phase 2 interface)
│   │
│   └── llm/
│       ├── __init__.py
│       ├── client.py           # Claude API wrapper
│       ├── cluster_naming.py   # Name clusters from track lists
│       └── discovery_context.py # Contextualize recommendations in natural language
│
├── data/
│   ├── spotify_ai.db           # SQLite database
│   └── faiss_index/            # Vector similarity index
│
├── scripts/
│   ├── backfill.py             # One-time: pull all existing data
│   ├── train_taste_model.py    # Batch: retrain preference model
│   └── recluster.py            # Batch: re-run clustering
│
└── tests/
    └── ...
```

---

## Database Schema

Using SQLite with SQLAlchemy. This is the core data model.

```python
# src/data/models.py

from sqlalchemy import Column, String, Float, Integer, DateTime, Boolean, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Song(Base):
    """A song in your library."""
    __tablename__ = "songs"

    spotify_id = Column(String, primary_key=True)        # Spotify track ID
    title = Column(String, nullable=False)
    artist = Column(String, nullable=False)
    album = Column(String)
    duration_ms = Column(Integer)
    release_date = Column(String)
    date_added = Column(DateTime)                         # When you first added it

    # Spotify audio features (fetched from API)
    danceability = Column(Float)
    energy = Column(Float)
    valence = Column(Float)
    acousticness = Column(Float)
    instrumentalness = Column(Float)
    speechiness = Column(Float)
    tempo = Column(Float)
    key = Column(Integer)
    mode = Column(Integer)
    loudness = Column(Float)
    liveness = Column(Float)

    # Genre/style tags (JSON dict: {"indie rock": 0.8, "midwest emo": 0.6})
    genre_tags = Column(JSON, default=dict)

    # Embedding (stored as JSON list of floats; loaded into numpy at runtime)
    embedding = Column(JSON)

    # Cluster assignment
    cluster_id = Column(Integer, ForeignKey("clusters.id"), nullable=True)

    # Computed stats (updated periodically from listen events)
    total_plays = Column(Integer, default=0)
    recent_plays_30d = Column(Integer, default=0)
    skip_rate = Column(Float, default=0.0)           # fraction of plays skipped <30s
    avg_listen_fraction = Column(Float, default=1.0)  # how much of the song you hear
    staleness_score = Column(Float, default=0.0)

    # User signals
    in_liked_songs = Column(Boolean, default=False)
    in_best_playlist = Column(Boolean, default=False)
    user_rating = Column(Float, nullable=True)        # explicit 1-5 rating if given
    is_approved = Column(Boolean, default=True)       # False = pending trial song
    is_blacklisted = Column(Boolean, default=False)   # Explicitly rejected

    # Playlist membership (JSON list of playlist names)
    playlist_memberships = Column(JSON, default=list)


class ListenEvent(Base):
    """A single scrobble / listen event."""
    __tablename__ = "listen_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    spotify_id = Column(String, ForeignKey("songs.spotify_id"))
    timestamp = Column(DateTime, nullable=False)
    source = Column(String, default="lastfm")         # "lastfm" or "spotify_recent"
    listened_ms = Column(Integer, nullable=True)       # How long they listened
    skipped = Column(Boolean, default=False)


class Cluster(Base):
    """A genre/mood cluster."""
    __tablename__ = "clusters"

    id = Column(Integer, primary_key=True)
    name = Column(String)                              # LLM-generated name
    description = Column(String)                       # LLM-generated description
    centroid = Column(JSON)                            # Cluster centroid embedding
    song_count = Column(Integer, default=0)
    spotify_playlist_id = Column(String, nullable=True) # Corresponding Spotify playlist
    parent_cluster_id = Column(Integer, nullable=True)  # For hierarchy


class DiscoveryCandidate(Base):
    """A song discovered but not yet in your library."""
    __tablename__ = "discovery_candidates"

    spotify_id = Column(String, primary_key=True)
    title = Column(String)
    artist = Column(String)
    source = Column(String)                            # "spotify_recs", "lastfm_similar", "rym", etc.
    discovered_date = Column(DateTime)
    predicted_score = Column(Float)                    # Taste model prediction
    status = Column(String, default="pending")         # "pending", "trial", "approved", "rejected"
    trial_plays = Column(Integer, default=0)
    trial_skips = Column(Integer, default=0)


class PlaylistSnapshot(Base):
    """Frozen snapshot of the dynamic playlist (replaces manual dated playlists)."""
    __tablename__ = "playlist_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime)
    name = Column(String)                              # e.g., "2026-02"
    spotify_playlist_id = Column(String)
    song_ids = Column(JSON)                            # List of spotify_ids at snapshot time
```

---

## config.yaml — All Tunables

This is important. Every magic number should live here so you can experiment.

```yaml
# config.yaml

spotify:
  playlist_name: "AI Mix"              # Name of the dynamic playlist on Spotify
  snapshot_prefix: "AI Snapshot"       # Prefix for frozen dated playlists

playlist:
  target_size: 100                     # Songs in the dynamic playlist
  core_fraction: 0.70                  # % from high-scoring library songs
  exploration_fraction: 0.20           # % cycled in from dump (known but not recent)
  discovery_fraction: 0.10             # % from new discovery candidates (trial songs)
  max_change_per_refresh: 0.15         # Max % of playlist replaced per refresh (stability)
  min_artist_spacing: 4                # Min songs between same artist
  snapshot_interval_days: 30           # How often to snapshot the dynamic playlist

cycling:
  aggressiveness: 5                    # 1-10 scale (tunable via CLI)
  staleness_lambda: 0.05              # Decay rate for staleness calculation
  staleness_window_days: 30            # Window for recent play counting
  cooldown_days: 60                    # How long a "tired" song is suppressed
  min_plays_for_staleness: 5           # Don't penalize songs you've barely heard
  blacklist_threshold: 5               # Consecutive rejections before suggesting removal from dump

scoring:
  taste_weight: 0.4
  mood_weight: 0.4
  freshness_weight: 0.2               # (1 - staleness) weight
  exploration_noise: 0.1              # Epsilon for epsilon-greedy exploration

mood:
  window_size: 20                      # Number of recent songs for mood vector
  decay_rate: 0.85                     # Exponential decay (most recent weighted highest)

discovery:
  candidates_per_refresh: 30           # How many new candidates to fetch per cycle
  trial_listens_before_decision: 3     # How many times to try a song before asking for verdict
  sources:
    spotify_recs: true
    lastfm_similar: true
    lastfm_friends: true
    listenbrain: true

taste_model:
  retrain_interval_days: 14
  model_type: "lightgbm"               # "lightgbm" or "neural"

clustering:
  min_cluster_size: 15
  min_samples: 5
  recluster_interval_days: 60
```

---

## Sprint-by-Sprint Implementation

### Sprint 1: Data Foundation (Week 1–2)

**Goal:** All your existing Spotify data and Last.fm history in a local database, queryable.

**Tasks:**

1. **Set up the project scaffolding.** Poetry/pip, SQLAlchemy, the config system. Use `python-dotenv` for secrets.

2. **Spotify API auth.** Use `spotipy` with the Authorization Code Flow (you need user scopes: `playlist-read-private`, `playlist-modify-private`, `playlist-modify-public`, `user-library-read`, `user-read-recently-played`). The auth flow will open a browser for you to approve, then cache the token.

```python
# src/data/spotify_client.py — core auth setup
import spotipy
from spotipy.oauth2 import SpotifyOAuth

def get_spotify_client() -> spotipy.Spotify:
    return spotipy.Spotify(auth_manager=SpotifyOAuth(
        scope=" ".join([
            "playlist-read-private",
            "playlist-modify-private",
            "playlist-modify-public",
            "user-library-read",
            "user-read-recently-played",
            "user-top-read",
        ]),
        redirect_uri="http://127.0.0.1:8888/callback",
        # SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET from .env
    ))
```

3. **Backfill playlists.** Pull every track from every playlist. Spotify paginates at 100 items — handle it. Store track metadata + which playlists each song belongs to.

```python
# scripts/backfill.py — playlist ingestion (sketch)
def backfill_playlists(sp: spotipy.Spotify, db: Session):
    playlists = get_all_user_playlists(sp)  # handle pagination
    for pl in playlists:
        tracks = get_all_playlist_tracks(sp, pl["id"])  # handle pagination
        for item in tracks:
            track = item["track"]
            if not track or not track["id"]:
                continue
            song = get_or_create_song(db, track)
            song.playlist_memberships = list(
                set(song.playlist_memberships or []) | {pl["name"]}
            )
    db.commit()
```

4. **Backfill audio features.** Batch fetch in groups of 100 (API limit).

```python
def backfill_audio_features(sp: spotipy.Spotify, db: Session):
    songs = db.query(Song).filter(Song.danceability.is_(None)).all()
    for batch in chunked(songs, 100):
        ids = [s.spotify_id for s in batch]
        features = sp.audio_features(ids)
        for song, feat in zip(batch, features):
            if feat:
                song.danceability = feat["danceability"]
                song.energy = feat["energy"]
                # ... etc
    db.commit()
```

5. **Backfill Last.fm scrobbles.** Use `pylast` to paginate through your full history. Map Last.fm tracks to Spotify IDs (this requires fuzzy matching on title + artist — use `thefuzz` library for this). Store as `ListenEvent` rows.

```python
# src/data/lastfm_client.py
import pylast

def get_lastfm_client() -> pylast.LastFMNetwork:
    return pylast.LastFMNetwork(
        api_key=os.getenv("LASTFM_API_KEY"),
        api_secret=os.getenv("LASTFM_API_SECRET"),
    )

def fetch_all_scrobbles(network: pylast.LastFMNetwork, username: str):
    """Generator that yields all scrobbles, handling pagination."""
    user = network.get_user(username)
    # pylast's get_recent_tracks handles pagination with `limit` and `page`
    # Fetch in pages of 200, going back to the beginning
    ...
```

6. **Compute listening stats.** From the scrobble data, compute per-song: total plays, recent plays (30d), estimated skip rate, last played date. Update `Song` rows.

**Deliverable:** Run `python scripts/backfill.py` and get a SQLite database with ~8k songs, audio features, playlist memberships, and listening history.

---

### Sprint 2: Feature Vectors & Scoring (Week 3–4)

**Goal:** Every song has a feature vector. You can score and rank your library.

**Tasks:**

1. **Build feature vectors.** Combine audio features + genre tags into a single vector per song.

```python
# src/features/embeddings.py
import numpy as np
from sklearn.preprocessing import StandardScaler

AUDIO_FEATURE_KEYS = [
    "danceability", "energy", "valence", "acousticness",
    "instrumentalness", "speechiness", "tempo", "loudness", "liveness"
]

def build_audio_feature_vector(song: Song) -> np.ndarray:
    """9-dimensional audio feature vector."""
    return np.array([getattr(song, k) for k in AUDIO_FEATURE_KEYS])

def build_genre_vector(song: Song, all_tags: list[str]) -> np.ndarray:
    """Multi-hot genre vector across all known tags."""
    tags = song.genre_tags or {}
    return np.array([tags.get(t, 0.0) for t in all_tags])

def build_combined_embedding(song: Song, all_tags: list[str],
                              audio_scaler: StandardScaler) -> np.ndarray:
    """Combined, normalized feature vector."""
    audio = audio_scaler.transform(
        build_audio_feature_vector(song).reshape(1, -1)
    )[0]
    genre = build_genre_vector(song, all_tags)
    return np.concatenate([audio, genre])
```

2. **Enrich genre tags.** Spotify's artist genres are coarse. Supplement with Last.fm tags per track:

```python
def fetch_lastfm_tags(network, artist: str, title: str) -> dict[str, float]:
    """Fetch top tags for a track from Last.fm. Returns {tag: weight}."""
    try:
        track = network.get_track(artist, title)
        top_tags = track.get_top_tags(limit=10)
        return {tag.item.name.lower(): float(tag.weight) for tag in top_tags}
    except pylast.WSError:
        # Fall back to artist tags
        try:
            artist_obj = network.get_artist(artist)
            top_tags = artist_obj.get_top_tags(limit=10)
            return {tag.item.name.lower(): float(tag.weight) for tag in top_tags}
        except:
            return {}
```

Note: fetching tags for 8k songs will take a while (rate limits). Budget ~2-3 hours with polite rate limiting. Cache aggressively.

3. **Build FAISS index for similarity search.**

```python
# src/features/embeddings.py
import faiss

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a cosine similarity index."""
    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index
```

4. **Implement the mood vector.**

```python
# src/features/mood.py
import numpy as np
from datetime import datetime, timedelta

def compute_mood_vector(
    recent_songs: list[Song],
    timestamps: list[datetime],
    decay_rate: float = 0.85,
) -> np.ndarray:
    """
    Weighted average of recent song embeddings.
    Most recent song has weight 1.0, each prior song decays by decay_rate.
    """
    if not recent_songs:
        return None

    # Sort by timestamp descending (most recent first)
    paired = sorted(zip(timestamps, recent_songs), key=lambda x: x[0], reverse=True)

    vectors = []
    weights = []
    for i, (ts, song) in enumerate(paired):
        if song.embedding is not None:
            vectors.append(np.array(song.embedding))
            weights.append(decay_rate ** i)

    if not vectors:
        return None

    weights = np.array(weights) / sum(weights)
    return np.average(vectors, axis=0, weights=weights)
```

5. **Implement the scoring function.**

```python
# src/models/scoring.py
import numpy as np
from src.features.mood import compute_mood_vector

def score_song(
    song: Song,
    mood_vector: np.ndarray | None,
    config: dict,
) -> float:
    """
    Score a song for dynamic playlist inclusion.
    Higher = more likely to be included.
    """
    # Taste score: based on historical signals
    taste = compute_taste_score(song)

    # Mood similarity: cosine similarity to current mood vector
    if mood_vector is not None and song.embedding is not None:
        song_emb = np.array(song.embedding)
        mood_sim = np.dot(song_emb, mood_vector) / (
            np.linalg.norm(song_emb) * np.linalg.norm(mood_vector) + 1e-8
        )
        mood_sim = (mood_sim + 1) / 2  # Normalize from [-1,1] to [0,1]
    else:
        mood_sim = 0.5  # Neutral if no mood data

    # Freshness: inverse of staleness
    freshness = 1.0 - song.staleness_score

    # Weighted combination
    w = config["scoring"]
    score = (
        w["taste_weight"] * taste
        + w["mood_weight"] * mood_sim
        + w["freshness_weight"] * freshness
    )

    # Exploration noise
    noise = np.random.uniform(0, w["exploration_noise"])
    return score + noise


def compute_taste_score(song: Song) -> float:
    """
    Heuristic taste score before we have a trained model.
    Upgrade to ML model in Sprint 5.
    """
    score = 0.5  # Base

    # Positive signals
    if song.in_best_playlist:
        score += 0.25
    if song.in_liked_songs:
        score += 0.15
    if song.user_rating:
        score += (song.user_rating - 3) * 0.1  # -0.2 to +0.2

    # Play count signal (logarithmic — diminishing returns)
    if song.total_plays > 0:
        score += min(0.15, 0.05 * np.log1p(song.total_plays))

    # Negative signals
    if song.skip_rate > 0.5:
        score -= 0.2
    if song.is_blacklisted:
        score = 0.0

    return np.clip(score, 0.0, 1.0)
```

**Deliverable:** You can run `python -c "from src.models.scoring import ...; ..."` and get a ranked list of your 8k songs scored for "what should I listen to right now."

---

### Sprint 3: Dynamic Playlist (Week 5–6)

**Goal:** A Spotify playlist that auto-refreshes with intelligently selected songs.

**Tasks:**

1. **Playlist generation logic.**

```python
# src/playlists/dynamic.py

def generate_dynamic_playlist(
    db: Session,
    sp: spotipy.Spotify,
    mood_vector: np.ndarray,
    config: dict,
) -> list[str]:
    """Generate ordered list of spotify_ids for the dynamic playlist."""
    pl_config = config["playlist"]
    target = pl_config["target_size"]

    # Score all eligible songs
    all_songs = db.query(Song).filter(
        Song.is_blacklisted == False,
        Song.is_approved == True,
    ).all()

    scored = [(song, score_song(song, mood_vector, config)) for song in all_songs]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Core: top-scoring songs from library
    core_n = int(target * pl_config["core_fraction"])
    core = [s for s, _ in scored[:core_n]]

    # Exploration: songs from dump not played recently
    # (moderate taste score, low recent plays = good exploration candidates)
    exploration_pool = [
        s for s, sc in scored
        if s not in core
        and s.recent_plays_30d <= 2
        and sc > 0.3  # Don't explore songs you probably dislike
    ]
    exploration_n = int(target * pl_config["exploration_fraction"])
    exploration = random.sample(
        exploration_pool, min(exploration_n, len(exploration_pool))
    )

    # Discovery: trial songs from discovery candidates
    trial_songs = db.query(Song).filter(
        Song.is_approved == False,
        Song.is_blacklisted == False,
    ).order_by(Song.date_added.desc()).limit(
        int(target * pl_config["discovery_fraction"])
    ).all()

    # Combine and order intelligently
    playlist_songs = core + exploration + list(trial_songs)
    playlist_songs = enforce_artist_spacing(playlist_songs, config)
    playlist_songs = smooth_energy_transitions(playlist_songs)

    return [s.spotify_id for s in playlist_songs]


def push_to_spotify(
    sp: spotipy.Spotify,
    playlist_id: str,
    track_ids: list[str],
    current_ids: list[str],
    max_change_fraction: float,
):
    """
    Update Spotify playlist, respecting the stability constraint.
    Only changes up to max_change_fraction of the playlist per refresh.
    """
    max_changes = int(len(current_ids) * max_change_fraction)

    to_remove = [t for t in current_ids if t not in track_ids][:max_changes // 2]
    to_add = [t for t in track_ids if t not in current_ids][:max_changes // 2]

    if to_remove:
        sp.playlist_remove_all_occurrences_of_items(
            playlist_id,
            [f"spotify:track:{t}" for t in to_remove]
        )
    if to_add:
        sp.playlist_add_items(
            playlist_id,
            [f"spotify:track:{t}" for t in to_add]
        )

    return {"removed": len(to_remove), "added": len(to_add)}
```

2. **Staleness computation.**

```python
# src/features/listening_stats.py

def update_staleness(db: Session, config: dict):
    """Recompute staleness scores for all songs."""
    window = config["cycling"]["staleness_window_days"]
    lam = config["cycling"]["staleness_lambda"]
    min_plays = config["cycling"]["min_plays_for_staleness"]

    cutoff = datetime.utcnow() - timedelta(days=window)
    songs = db.query(Song).all()

    for song in songs:
        recent = db.query(ListenEvent).filter(
            ListenEvent.spotify_id == song.spotify_id,
            ListenEvent.timestamp >= cutoff,
        ).count()

        song.recent_plays_30d = recent
        if recent >= min_plays:
            song.staleness_score = 1.0 - np.exp(-lam * recent)
        else:
            song.staleness_score = 0.0

    db.commit()
```

3. **Snapshot logic** (auto-saves dated playlists).

```python
# src/playlists/snapshot.py

def maybe_snapshot(db: Session, sp: spotipy.Spotify, config: dict):
    """Create a frozen copy of the dynamic playlist if it's time."""
    last = db.query(PlaylistSnapshot).order_by(
        PlaylistSnapshot.date.desc()
    ).first()

    interval = timedelta(days=config["playlist"]["snapshot_interval_days"])
    if last and datetime.utcnow() - last.date < interval:
        return None

    # Get current dynamic playlist contents
    dynamic_id = get_or_create_dynamic_playlist(sp, config)
    tracks = get_all_playlist_tracks(sp, dynamic_id)
    track_ids = [t["track"]["id"] for t in tracks if t["track"]]

    # Create frozen copy on Spotify
    name = f"{config['spotify']['snapshot_prefix']} {datetime.utcnow().strftime('%Y-%m')}"
    snapshot_pl = sp.user_playlist_create(sp.me()["id"], name, public=False)

    # Add tracks in batches of 100
    for batch in chunked(track_ids, 100):
        sp.playlist_add_items(snapshot_pl["id"], batch)

    # Record in DB
    snap = PlaylistSnapshot(
        date=datetime.utcnow(),
        name=name,
        spotify_playlist_id=snapshot_pl["id"],
        song_ids=track_ids,
    )
    db.add(snap)
    db.commit()

    return snap
```

4. **CLI for refresh.**

```python
# src/interface/cli.py
import click

@click.group()
def cli():
    """Spotify AI — your music, managed intelligently."""
    pass

@cli.command()
def refresh():
    """Refresh the dynamic playlist based on current mood and scores."""
    db = get_db_session()
    sp = get_spotify_client()
    config = load_config()

    # Update listening stats
    ingest_recent_scrobbles(db, sp)
    update_staleness(db, config)

    # Compute current mood
    recent = get_recent_listens(db, n=config["mood"]["window_size"])
    mood = compute_mood_vector(
        [r.song for r in recent],
        [r.timestamp for r in recent],
        decay_rate=config["mood"]["decay_rate"],
    )

    # Generate and push playlist
    track_ids = generate_dynamic_playlist(db, sp, mood, config)
    playlist_id = get_or_create_dynamic_playlist(sp, config)
    current_ids = get_current_playlist_track_ids(sp, playlist_id)
    result = push_to_spotify(sp, playlist_id, track_ids, current_ids,
                              config["playlist"]["max_change_per_refresh"])

    click.echo(f"Playlist refreshed: +{result['added']} / -{result['removed']}")

    # Maybe snapshot
    snap = maybe_snapshot(db, sp, config)
    if snap:
        click.echo(f"Snapshot saved: {snap.name}")

@cli.command()
@click.argument("value", type=int)
def cycling(value):
    """Set cycling aggressiveness (1-10)."""
    config = load_config()
    config["cycling"]["aggressiveness"] = max(1, min(10, value))
    save_config(config)
    click.echo(f"Cycling aggressiveness set to {value}/10")

@cli.command()
@click.option("--text", "-t", help="Describe your mood in words")
def mood(text):
    """Show current detected mood, or set it explicitly."""
    if text:
        # Use LLM to translate text → target embedding
        # (Sprint 6 feature)
        click.echo(f"Mood set to: {text}")
    else:
        # Show detected mood from recent listening
        db = get_db_session()
        recent = get_recent_listens(db, n=20)
        mood_vec = compute_mood_vector(...)
        nearest_cluster = find_nearest_cluster(mood_vec)
        click.echo(f"Current mood: {nearest_cluster.name}")
        click.echo(f"Based on last {len(recent)} songs")

@cli.command()
def next():
    """Get the next song suggestion for approval."""
    db = get_db_session()
    # Find top pending item (cycling candidate or discovery candidate)
    candidate = get_next_candidate(db)
    if candidate:
        click.echo(f"\n🎵 {candidate.title} — {candidate.artist}")
        click.echo(f"   Source: {candidate.source}")
        click.echo(f"   Predicted score: {candidate.predicted_score:.2f}")
        action = click.prompt("(a)pprove / (r)eject / (s)kip / (l)isten first", type=str)
        handle_candidate_action(db, candidate, action)
    else:
        click.echo("No pending suggestions. Run 'discover' to find new music.")

@cli.command()
@click.option("--n", default=20, help="Number of candidates to fetch")
def discover(n):
    """Fetch new discovery candidates from all sources."""
    ...

@cli.command()
def stats():
    """Show listening stats and system health."""
    ...
```

**Deliverable:** `spotify-ai refresh` creates/updates a dynamic playlist on your Spotify that you can just play on your phone. `spotify-ai next` lets you approve/reject song suggestions.

---

### Sprint 4: Discovery Engine (Week 7–8)

**Goal:** Multi-source discovery pipeline that finds new songs and presents them for approval.

**Tasks:**

1. **Spotify Recommendations API integration.**

```python
# src/models/discovery.py

def discover_from_spotify(
    sp: spotipy.Spotify,
    db: Session,
    mood_vector: np.ndarray,
    n: int = 20,
) -> list[dict]:
    """Get recommendations seeded by current mood and top tracks."""
    # Find songs closest to mood vector as seeds
    seed_songs = get_mood_aligned_songs(db, mood_vector, n=3)
    seed_ids = [s.spotify_id for s in seed_songs]

    # Also get seed genres from current top clusters
    seed_genres = get_top_cluster_genres(db, mood_vector, n=2)

    recs = sp.recommendations(
        seed_tracks=seed_ids[:3],         # Max 5 seeds total
        seed_genres=seed_genres[:2],
        limit=n,
    )

    candidates = []
    for track in recs["tracks"]:
        if not db.query(Song).get(track["id"]):  # Not already in library
            candidates.append({
                "spotify_id": track["id"],
                "title": track["name"],
                "artist": track["artists"][0]["name"],
                "source": "spotify_recs",
            })
    return candidates
```

2. **Last.fm similar tracks/artists.**

```python
def discover_from_lastfm(
    lastfm: pylast.LastFMNetwork,
    db: Session,
    n: int = 20,
) -> list[dict]:
    """Find songs similar to your recent favorites via Last.fm."""
    # Get your top tracks from last month
    recent_favorites = db.query(Song).order_by(
        Song.recent_plays_30d.desc()
    ).limit(10).all()

    candidates = []
    for song in recent_favorites:
        try:
            track = lastfm.get_track(song.artist, song.title)
            similar = track.get_similar(limit=5)
            for sim_track, score in similar:
                # Try to find on Spotify
                results = sp.search(
                    q=f"track:{sim_track.title} artist:{sim_track.get_artist().name}",
                    type="track", limit=1
                )
                if results["tracks"]["items"]:
                    t = results["tracks"]["items"][0]
                    if not db.query(Song).get(t["id"]):
                        candidates.append({
                            "spotify_id": t["id"],
                            "title": t["name"],
                            "artist": t["artists"][0]["name"],
                            "source": "lastfm_similar",
                            "similar_to": song.title,
                        })
        except pylast.WSError:
            continue

    return candidates[:n]
```

3. **Last.fm friends' listening.**

```python
def discover_from_friends(
    lastfm: pylast.LastFMNetwork,
    friends: list[str],  # Last.fm usernames
    db: Session,
    n: int = 20,
) -> list[dict]:
    """Songs your friends are listening to that you haven't heard."""
    candidates = []
    for friend_name in friends:
        friend = lastfm.get_user(friend_name)
        recent = friend.get_recent_tracks(limit=50)
        for played in recent:
            # Check if it's in your library
            # Search Spotify, deduplicate, etc.
            ...
    return candidates[:n]
```

4. **ListenBrainz collaborative filtering** (new source for you).

```python
def discover_from_listenbrainz(username: str, n: int = 20) -> list[dict]:
    """Get recommendations from ListenBrainz's open-source rec engine."""
    import requests
    resp = requests.get(
        f"https://api.listenbrainz.org/1/cf/recommendation/user/{username}/recording",
        headers={"Authorization": f"Token {os.getenv('LISTENBRAINZ_TOKEN')}"}
    )
    # Parse and map to Spotify IDs...
```

5. **Candidate scoring and deduplication.**

```python
def aggregate_candidates(
    all_candidates: list[dict],
    db: Session,
    sp: spotipy.Spotify,
    taste_model,  # trained model or heuristic scorer
) -> list[DiscoveryCandidate]:
    """Deduplicate, score, and store discovery candidates."""
    # Deduplicate by spotify_id
    seen = set()
    unique = []
    for c in all_candidates:
        if c["spotify_id"] not in seen:
            seen.add(c["spotify_id"])
            unique.append(c)

    # Fetch audio features and predict taste scores
    ids = [c["spotify_id"] for c in unique]
    features = sp.audio_features(ids)

    scored_candidates = []
    for candidate, feat in zip(unique, features):
        if feat:
            predicted = taste_model.predict(feat)  # or heuristic
            dc = DiscoveryCandidate(
                spotify_id=candidate["spotify_id"],
                title=candidate["title"],
                artist=candidate["artist"],
                source=candidate["source"],
                discovered_date=datetime.utcnow(),
                predicted_score=predicted,
                status="pending",
            )
            scored_candidates.append(dc)

    # Store in DB, sorted by predicted score
    scored_candidates.sort(key=lambda x: x.predicted_score, reverse=True)
    for dc in scored_candidates:
        db.merge(dc)
    db.commit()

    return scored_candidates
```

**Deliverable:** `spotify-ai discover` pulls 30+ candidates from multiple sources, scores them, and queues them for your review via `spotify-ai next`.

---

### Sprint 5: Taste Model & Clustering (Week 9–10)

**Goal:** Replace heuristic scoring with a trained ML model. Auto-cluster your library.

**Tasks:**

1. **Train the taste model.** By now you have weeks of feedback data from approvals, rejections, and listening behavior.

```python
# scripts/train_taste_model.py
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

def train_taste_model(db: Session):
    """Train a LightGBM model to predict song preference."""
    songs = db.query(Song).filter(Song.embedding.isnot(None)).all()

    X = []  # Feature vectors
    y = []  # Target: preference score

    for song in songs:
        features = np.concatenate([
            np.array(song.embedding),
            np.array([
                song.total_plays,
                song.skip_rate,
                song.avg_listen_fraction,
                float(song.in_best_playlist),
                float(song.in_liked_songs),
                song.user_rating or 3.0,
            ])
        ])
        X.append(features)

        # Continuous target derived from signals
        target = compute_preference_target(song)
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    # Train with time-aware cross-validation
    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=10,  # Conservative — small dataset
    )

    # Use TimeSeriesSplit to avoid leaking future data
    cv = TimeSeriesSplit(n_splits=3)
    for train_idx, val_idx in cv.split(X):
        model.fit(
            X[train_idx], y[train_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            # early_stopping_rounds handled internally by lgb
        )

    # Save model
    model.booster_.save_model("data/taste_model.txt")

    # Feature importance (useful for understanding what drives your taste)
    importance = model.feature_importances_
    ...

    return model


def compute_preference_target(song: Song) -> float:
    """
    Derive a 0-1 preference score from all available signals.
    This is the ground truth for training.
    """
    score = 0.0
    weights_total = 0.0

    if song.user_rating is not None:
        score += (song.user_rating / 5.0) * 3.0  # Explicit ratings weighted heavily
        weights_total += 3.0

    if song.in_best_playlist:
        score += 1.0 * 2.0
        weights_total += 2.0
    elif song.in_liked_songs:
        score += 0.8 * 1.5
        weights_total += 1.5

    if song.total_plays > 0:
        play_signal = min(1.0, np.log1p(song.total_plays) / np.log1p(50))
        score += play_signal * 1.0
        weights_total += 1.0

    if song.skip_rate > 0:
        score += (1.0 - song.skip_rate) * 1.0
        weights_total += 1.0

    if song.is_blacklisted:
        return 0.0

    return score / weights_total if weights_total > 0 else 0.5
```

2. **HDBSCAN clustering.**

```python
# src/models/clustering.py
import hdbscan
import umap

def cluster_library(db: Session, config: dict):
    """Cluster all songs and assign cluster IDs."""
    songs = db.query(Song).filter(Song.embedding.isnot(None)).all()
    embeddings = np.array([s.embedding for s in songs])

    # Reduce dimensionality before clustering (helps HDBSCAN)
    reducer = umap.UMAP(n_components=20, metric="cosine", random_state=42)
    reduced = reducer.fit_transform(embeddings)

    # Cluster
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config["clustering"]["min_cluster_size"],
        min_samples=config["clustering"]["min_samples"],
        metric="euclidean",
    )
    labels = clusterer.fit_predict(reduced)

    # Assign clusters
    for song, label in zip(songs, labels):
        song.cluster_id = int(label) if label >= 0 else None  # -1 = noise

    db.commit()

    # Compute centroids
    unique_labels = set(labels) - {-1}
    for label in unique_labels:
        mask = labels == label
        centroid = embeddings[mask].mean(axis=0)
        cluster = Cluster(
            id=int(label),
            centroid=centroid.tolist(),
            song_count=int(mask.sum()),
        )
        db.merge(cluster)

    db.commit()
    return clusterer, reducer
```

3. **LLM-powered cluster naming.**

```python
# src/llm/cluster_naming.py
import anthropic

def name_clusters(db: Session):
    """Use Claude to name each cluster based on its songs."""
    client = anthropic.Anthropic()
    clusters = db.query(Cluster).all()

    for cluster in clusters:
        songs = db.query(Song).filter(Song.cluster_id == cluster.id).all()

        # Build a representative sample
        sample = random.sample(songs, min(30, len(songs)))
        song_list = "\n".join(
            f"- {s.artist} — {s.title}" for s in sample
        )

        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": f"""Here are songs from a music cluster. Give me:
1. A short, evocative playlist name (2-5 words)
2. A one-sentence description of the vibe/genre

Songs:
{song_list}

Respond as JSON: {{"name": "...", "description": "..."}}"""
            }],
        )

        result = json.loads(response.content[0].text)
        cluster.name = result["name"]
        cluster.description = result["description"]

    db.commit()
```

4. **UMAP visualization** (optional but very satisfying).

```python
# scripts/visualize_clusters.py
import matplotlib.pyplot as plt

def plot_library(db: Session):
    songs = db.query(Song).filter(Song.embedding.isnot(None)).all()
    embeddings = np.array([s.embedding for s in songs])
    labels = np.array([s.cluster_id or -1 for s in songs])

    reducer_2d = umap.UMAP(n_components=2, metric="cosine")
    coords = reducer_2d.fit_transform(embeddings)

    plt.figure(figsize=(16, 12))
    scatter = plt.scatter(
        coords[:, 0], coords[:, 1],
        c=labels, cmap="tab20", s=3, alpha=0.6
    )
    # Add cluster name labels at centroids
    for cluster in db.query(Cluster).all():
        mask = labels == cluster.id
        if mask.any():
            cx, cy = coords[mask].mean(axis=0)
            plt.annotate(cluster.name, (cx, cy), fontsize=8, weight="bold")

    plt.title("Your Music Library")
    plt.savefig("data/library_map.png", dpi=150)
```

**Deliverable:** Trained taste model improves scoring. Library is auto-clustered with LLM-named clusters. You can see your entire library as a 2D map.

---

### Sprint 6: Telegram Bot (Week 11–12)

**Goal:** Move the CLI interaction to a Telegram bot for mobile-friendly approval workflow.

**Deploy on VPS:** Set up a Hetzner VPS (~$4.50/month for CX22). The bot runs there, the DB syncs there (or runs there entirely since the ML workloads are light enough for 8k songs).

```python
# src/interface/telegram_bot.py
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, MessageHandler
)

AUTHORIZED_USER_ID = int(os.getenv("TELEGRAM_USER_ID"))

def auth_required(func):
    """Only respond to your Telegram account."""
    async def wrapper(update: Update, context):
        if update.effective_user.id != AUTHORIZED_USER_ID:
            return
        return await func(update, context)
    return wrapper

@auth_required
async def cmd_refresh(update: Update, context):
    """Refresh the dynamic playlist."""
    await update.message.reply_text("🔄 Refreshing playlist...")
    result = run_refresh()  # Calls the same logic as CLI
    await update.message.reply_text(
        f"✅ Playlist updated: +{result['added']} / -{result['removed']}"
    )

@auth_required
async def cmd_next(update: Update, context):
    """Present the next song suggestion for approval."""
    candidate = get_next_candidate()
    if not candidate:
        await update.message.reply_text("No pending suggestions. Use /discover first.")
        return

    # Build rich message
    text = (
        f"🎵 *{candidate.title}*\n"
        f"by {candidate.artist}\n\n"
        f"Source: {candidate.source}\n"
        f"Predicted match: {candidate.predicted_score:.0%}"
    )

    # Spotify link for easy listening
    spotify_url = f"https://open.spotify.com/track/{candidate.spotify_id}"
    text += f"\n\n[Listen on Spotify]({spotify_url})"

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Approve", callback_data=f"approve:{candidate.spotify_id}"),
            InlineKeyboardButton("❌ Reject", callback_data=f"reject:{candidate.spotify_id}"),
        ],
        [
            InlineKeyboardButton("⏭ Skip for now", callback_data=f"skip:{candidate.spotify_id}"),
        ]
    ])

    await update.message.reply_text(text, reply_markup=keyboard, parse_mode="Markdown")

@auth_required
async def handle_approval(update: Update, context):
    """Handle button presses for song approval."""
    query = update.callback_query
    await query.answer()

    action, spotify_id = query.data.split(":")
    db = get_db_session()

    if action == "approve":
        approve_candidate(db, spotify_id)
        await query.edit_message_text(f"✅ Added to your library!")
        # Automatically ask next
        await cmd_next(update, context)

    elif action == "reject":
        reject_candidate(db, spotify_id)
        await query.edit_message_text(f"❌ Rejected. Won't suggest again.")
        await cmd_next(update, context)

    elif action == "skip":
        await query.edit_message_text(f"⏭ Skipped for later.")

@auth_required
async def cmd_mood(update: Update, context):
    """Set mood via natural language."""
    text = " ".join(context.args) if context.args else None
    if text:
        # Use LLM to interpret mood and adjust scoring
        mood_embedding = interpret_mood_text(text)
        set_explicit_mood(mood_embedding)
        await update.message.reply_text(f"🎭 Mood set: {text}\nRefreshing playlist...")
        result = run_refresh(override_mood=mood_embedding)
        await update.message.reply_text(f"✅ Playlist adjusted.")
    else:
        current = get_current_mood_description()
        await update.message.reply_text(f"🎭 Current mood: {current}")

@auth_required
async def cmd_cycling(update: Update, context):
    """Adjust cycling aggressiveness."""
    if context.args:
        value = int(context.args[0])
        set_cycling_aggressiveness(value)
        await update.message.reply_text(f"🔄 Cycling set to {value}/10")
    else:
        current = get_config()["cycling"]["aggressiveness"]
        await update.message.reply_text(f"🔄 Current cycling: {current}/10")

def main():
    app = Application.builder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()

    app.add_handler(CommandHandler("refresh", cmd_refresh))
    app.add_handler(CommandHandler("next", cmd_next))
    app.add_handler(CommandHandler("mood", cmd_mood))
    app.add_handler(CommandHandler("discover", cmd_discover))
    app.add_handler(CommandHandler("cycling", cmd_cycling))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CallbackQueryHandler(handle_approval))

    app.run_polling()
```

**Bot commands summary:**
```
/refresh         — Refresh the dynamic playlist
/next            — Get next song suggestion (approve/reject)
/discover        — Fetch new discovery candidates
/mood [text]     — Show or set current mood
/cycling [1-10]  — Adjust cycling aggressiveness
/stats           — Listening stats and system health
/playlist        — Link to current dynamic playlist
/snapshot        — Force a playlist snapshot now
```

**Deliverable:** A Telegram bot on your phone that pings you with song suggestions. Tap approve/reject. Tell it your mood in natural language. One-command playlist refresh.

---

## Scheduling

Once on VPS, set up cron jobs for autonomous operation:

```cron
# Ingest new scrobbles every 15 minutes
*/15 * * * * cd /app && python -m src.tasks.ingest_scrobbles

# Refresh dynamic playlist twice daily
0 8,20 * * * cd /app && python -m src.tasks.refresh_playlist

# Fetch discovery candidates daily
0 12 * * * cd /app && python -m src.tasks.discover

# Retrain taste model biweekly
0 3 * * 0 cd /app && python scripts/train_taste_model.py  # Every Sunday 3am

# Recluster monthly
0 4 1 * * cd /app && python scripts/recluster.py
```

---

## Cost Estimate

| Item | Monthly Cost |
|------|-------------|
| Hetzner VPS (CX22, 2 vCPU, 4GB RAM) | ~$5 |
| Claude API (cluster naming, mood interpretation, ~500 calls/mo) | ~$2–5 |
| Domain name (optional) | ~$1 |
| **Total** | **~$8–11/month** |

Spotify API: free. Last.fm API: free. ListenBrainz API: free. FAISS: local. LightGBM: local.

---

## Upgrade Path (Post-MVP)

After the 6 sprints, these are natural extensions in priority order:

1. **CLAP audio embeddings** — Replace Spotify's 9-dim audio features with 512-dim CLAP embeddings for much richer song representations. Run locally or on VPS.

2. **Thompson Sampling for explore/exploit** — Replace epsilon-greedy with proper Bayesian bandit for smarter exploration.

3. **Web dashboard** — Streamlit app showing your library UMAP map, listening trends, cluster health, model confidence. Run on same VPS.

4. **Proactive suggestions** — Bot pings you when it finds a high-confidence discovery candidate or detects a mood shift, rather than waiting for you to ask.

5. **Lyric embeddings** — Fetch lyrics from Genius, embed with a sentence transformer, add to song vectors. Captures thematic similarity that audio misses.

6. **RYM integration** — Scrape (carefully) RYM charts and genre pages for curated discovery. Particularly valuable for finding critically acclaimed albums in genres you already like.

7. **"Adventure mode"** playlist — Deliberately high-novelty, pushing your boundaries. Separate from the comfort of the dynamic playlist.

8. **Social features** — When a Last.fm friend listens to something the model predicts you'd love, get a ping: "Your friend X is listening to Y — 92% predicted match for you."

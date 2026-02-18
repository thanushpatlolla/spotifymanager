"""Shared test fixtures for spotifymanager tests."""

from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.data.models import Base, ListenEvent, Song


@pytest.fixture
def session():
    """Create an in-memory SQLite database with empty tables."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    sess = SessionLocal()
    yield sess
    sess.close()


@pytest.fixture
def populated_session(session: Session):
    """Session pre-loaded with a diverse set of songs and listen events.

    Songs:
    - song_a: 50 plays, liked, has tags "rock, alternative", last played 2 days ago
    - song_b: 20 plays, has tags "electronic, dance", last played 10 days ago
    - song_c: 5 plays, has tags "rock, indie", last played 60 days ago
    - song_d: 0 plays, new song, has tags "electronic, ambient"
    - song_e: 15 plays, liked, has tags "rock, alternative, indie", last played 1 day ago
    - song_f: 100 plays, no recent, has tags "pop", last played 120 days ago (abandoned)
    - song_g: 3 plays, no tags, last played 5 days ago
    """
    now = datetime.utcnow()

    songs = [
        Song(
            spotify_id="song_a", title="Alpha", artist="Artist One",
            in_liked_songs=True, lastfm_tags="rock, alternative",
            total_plays=50, recent_plays_30d=10, last_played=now - timedelta(days=2),
            staleness_score=2 / 180,
        ),
        Song(
            spotify_id="song_b", title="Beta", artist="Artist Two",
            lastfm_tags="electronic, dance",
            total_plays=20, recent_plays_30d=3, last_played=now - timedelta(days=10),
            staleness_score=10 / 180,
        ),
        Song(
            spotify_id="song_c", title="Gamma", artist="Artist One",
            lastfm_tags="rock, indie",
            total_plays=5, recent_plays_30d=0, last_played=now - timedelta(days=60),
            staleness_score=60 / 180,
        ),
        Song(
            spotify_id="song_d", title="Delta", artist="Artist Three",
            lastfm_tags="electronic, ambient",
            total_plays=0, recent_plays_30d=0,
        ),
        Song(
            spotify_id="song_e", title="Epsilon", artist="Artist Two",
            in_liked_songs=True, lastfm_tags="rock, alternative, indie",
            total_plays=15, recent_plays_30d=8, last_played=now - timedelta(days=1),
            staleness_score=1 / 180,
        ),
        Song(
            spotify_id="song_f", title="Zeta", artist="Artist Four",
            lastfm_tags="pop",
            total_plays=100, recent_plays_30d=0, last_played=now - timedelta(days=120),
            staleness_score=120 / 180,
        ),
        Song(
            spotify_id="song_g", title="Eta", artist="Artist Five",
            total_plays=3, recent_plays_30d=1, last_played=now - timedelta(days=5),
            staleness_score=5 / 180,
        ),
    ]
    session.add_all(songs)
    session.flush()

    # Add listen events for recent listening (within 4 hours for mood)
    recent_events = [
        ListenEvent(song_id="song_a", listened_at=now - timedelta(minutes=30)),
        ListenEvent(song_id="song_e", listened_at=now - timedelta(minutes=60)),
        ListenEvent(song_id="song_a", listened_at=now - timedelta(minutes=90)),
        ListenEvent(song_id="song_b", listened_at=now - timedelta(hours=2)),
    ]
    session.add_all(recent_events)
    session.flush()

    return session

"""Tests for database models — creation, relationships, and defaults."""

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.data.models import Base, Cluster, ListenEvent, Song


@pytest.fixture
def session():
    """Create an in-memory SQLite database and session for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


def test_create_tables(session: Session):
    """All tables should be created without errors."""
    tables = Base.metadata.tables.keys()
    assert "songs" in tables
    assert "listen_events" in tables
    assert "clusters" in tables
    assert "discovery_candidates" in tables
    assert "playlist_snapshots" in tables


def test_song_creation(session: Session):
    """Song can be created with required fields and sensible defaults."""
    song = Song(
        spotify_id="test123",
        title="Test Song",
        artist="Test Artist",
    )
    session.add(song)
    session.commit()

    retrieved = session.get(Song, "test123")
    assert retrieved is not None
    assert retrieved.title == "Test Song"
    assert retrieved.artist == "Test Artist"
    assert retrieved.total_plays == 0
    assert retrieved.recent_plays_30d == 0
    assert retrieved.in_liked_songs is False
    assert retrieved.danceability is None
    assert retrieved.lastfm_tags is None


def test_song_with_audio_features(session: Session):
    """Song can store audio features when available."""
    song = Song(
        spotify_id="feat123",
        title="Featured Song",
        artist="Featured Artist",
        danceability=0.8,
        energy=0.6,
        valence=0.7,
        tempo=120.0,
    )
    session.add(song)
    session.commit()

    retrieved = session.get(Song, "feat123")
    assert retrieved.danceability == 0.8
    assert retrieved.energy == 0.6
    assert retrieved.valence == 0.7
    assert retrieved.tempo == 120.0


def test_listen_event_relationship(session: Session):
    """ListenEvent correctly references Song via relationship."""
    song = Song(spotify_id="rel123", title="Rel Song", artist="Rel Artist")
    session.add(song)
    session.flush()

    event = ListenEvent(
        song_id="rel123",
        listened_at=datetime(2024, 1, 15, 10, 30),
        source="lastfm",
    )
    session.add(event)
    session.commit()

    assert len(song.listen_events) == 1
    assert song.listen_events[0].listened_at == datetime(2024, 1, 15, 10, 30)
    assert event.song.title == "Rel Song"


def test_multiple_listen_events(session: Session):
    """Multiple listen events can be associated with one song."""
    song = Song(spotify_id="multi123", title="Multi Song", artist="Multi Artist")
    session.add(song)
    session.flush()

    for i in range(5):
        event = ListenEvent(
            song_id="multi123",
            listened_at=datetime(2024, 1, i + 1),
            source="lastfm",
        )
        session.add(event)
    session.commit()

    assert len(song.listen_events) == 5


def test_cluster_relationship(session: Session):
    """Song can be assigned to a cluster."""
    cluster = Cluster(name="Chill Vibes")
    session.add(cluster)
    session.flush()

    song = Song(
        spotify_id="clust123",
        title="Chill Song",
        artist="Chill Artist",
        cluster_id=cluster.id,
    )
    session.add(song)
    session.commit()

    assert song.cluster.name == "Chill Vibes"
    assert len(cluster.songs) == 1
    assert cluster.songs[0].spotify_id == "clust123"


def test_song_repr(session: Session):
    """Song __repr__ is readable."""
    song = Song(spotify_id="repr123", title="My Song", artist="My Artist")
    assert repr(song) == "<Song My Artist - My Song>"

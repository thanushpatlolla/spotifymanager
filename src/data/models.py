"""SQLAlchemy 2.0 ORM models for spotifymanager."""

from datetime import datetime
from typing import Optional

from sqlalchemy import ForeignKey, Index, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Song(Base):
    __tablename__ = "songs"

    spotify_id: Mapped[str] = mapped_column(String(62), primary_key=True)
    title: Mapped[str] = mapped_column(String(500))
    artist: Mapped[str] = mapped_column(String(500))
    album: Mapped[Optional[str]] = mapped_column(String(500))
    duration_ms: Mapped[Optional[int]] = mapped_column()
    popularity: Mapped[Optional[int]] = mapped_column()
    added_at: Mapped[Optional[datetime]] = mapped_column()

    # Source tracking
    in_liked_songs: Mapped[bool] = mapped_column(default=False)
    source_playlist: Mapped[Optional[str]] = mapped_column(String(500))

    # Audio features (optional — deprecated API, may be unavailable)
    danceability: Mapped[Optional[float]] = mapped_column()
    energy: Mapped[Optional[float]] = mapped_column()
    valence: Mapped[Optional[float]] = mapped_column()
    tempo: Mapped[Optional[float]] = mapped_column()
    acousticness: Mapped[Optional[float]] = mapped_column()
    instrumentalness: Mapped[Optional[float]] = mapped_column()
    speechiness: Mapped[Optional[float]] = mapped_column()
    liveness: Mapped[Optional[float]] = mapped_column()
    loudness: Mapped[Optional[float]] = mapped_column()
    key: Mapped[Optional[int]] = mapped_column()
    mode: Mapped[Optional[int]] = mapped_column()
    time_signature: Mapped[Optional[int]] = mapped_column()

    # Last.fm tags (comma-separated)
    lastfm_tags: Mapped[Optional[str]] = mapped_column(Text)

    # Computed stats
    total_plays: Mapped[int] = mapped_column(default=0)
    recent_plays_30d: Mapped[int] = mapped_column(default=0)
    last_played: Mapped[Optional[datetime]] = mapped_column()
    staleness_score: Mapped[Optional[float]] = mapped_column()

    # Cluster assignment (Sprint 5)
    cluster_id: Mapped[Optional[int]] = mapped_column(ForeignKey("clusters.id"))

    # Relationships
    listen_events: Mapped[list["ListenEvent"]] = relationship(back_populates="song")
    cluster: Mapped[Optional["Cluster"]] = relationship(back_populates="songs")

    __table_args__ = (
        Index("ix_songs_artist", "artist"),
        Index("ix_songs_total_plays", "total_plays"),
    )

    def __repr__(self) -> str:
        return f"<Song {self.artist} - {self.title}>"


class ListenEvent(Base):
    __tablename__ = "listen_events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    song_id: Mapped[str] = mapped_column(ForeignKey("songs.spotify_id"), index=True)
    listened_at: Mapped[datetime] = mapped_column(index=True)
    source: Mapped[str] = mapped_column(String(50), default="lastfm")

    song: Mapped["Song"] = relationship(back_populates="listen_events")

    __table_args__ = (Index("ix_listen_events_song_date", "song_id", "listened_at"),)

    def __repr__(self) -> str:
        return f"<ListenEvent {self.song_id} @ {self.listened_at}>"


class Cluster(Base):
    """Song cluster from HDBSCAN (Sprint 5)."""

    __tablename__ = "clusters"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[Optional[str]] = mapped_column(String(200))
    description: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    songs: Mapped[list["Song"]] = relationship(back_populates="cluster")

    def __repr__(self) -> str:
        return f"<Cluster {self.id}: {self.name}>"


class DiscoveryCandidate(Base):
    """Candidate song for discovery (Sprint 4)."""

    __tablename__ = "discovery_candidates"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    spotify_id: Mapped[str] = mapped_column(String(62), index=True)
    title: Mapped[str] = mapped_column(String(500))
    artist: Mapped[str] = mapped_column(String(500))
    source: Mapped[str] = mapped_column(String(100))
    score: Mapped[Optional[float]] = mapped_column()
    added_to_playlist: Mapped[bool] = mapped_column(default=False)
    discovered_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<DiscoveryCandidate {self.artist} - {self.title}>"


class PlaylistSnapshot(Base):
    """Snapshot of generated playlist state (Sprint 3)."""

    __tablename__ = "playlist_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    playlist_id: Mapped[str] = mapped_column(String(62))
    track_ids: Mapped[str] = mapped_column(Text)  # JSON array of spotify_ids
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<PlaylistSnapshot {self.playlist_id} @ {self.created_at}>"

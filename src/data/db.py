"""Database connection, config loading, and session management."""

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.data.models import Base

_engine = None
_SessionFactory = None


def get_project_root() -> Path:
    """Find project root by walking up to pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not find project root (no pyproject.toml found)")


def load_config() -> dict[str, Any]:
    """Load config.yaml from project root."""
    root = get_project_root()
    config_path = root / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_db_path() -> Path:
    """Get absolute path to the database file."""
    config = load_config()
    db_relative = config.get("database", {}).get("path", "data/spotifymanager.db")
    return get_project_root() / db_relative


def _init_engine():
    """Initialize the SQLAlchemy engine and session factory."""
    global _engine, _SessionFactory
    if _engine is None:
        db_path = get_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(f"sqlite:///{db_path}", echo=False)
        _SessionFactory = sessionmaker(bind=_engine)


@contextmanager
def get_session():
    """Context manager that yields a SQLAlchemy session with auto-commit/rollback."""
    _init_engine()
    session: Session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db():
    """Create all tables in the database."""
    _init_engine()
    Base.metadata.create_all(_engine)


def load_env():
    """Load .env file from project root."""
    env_path = get_project_root() / ".env"
    load_dotenv(env_path)

"""LightGBM taste model.

Trains a taste model that predicts how much the user will enjoy a song
based on audio features, tags, and listening history.
"""

import pickle
from pathlib import Path

import numpy as np
from rich.console import Console
from sqlalchemy.orm import Session

from src.data.db import get_project_root
from src.data.models import Song
from src.features.mood import build_tag_vocabulary, parse_tags

console = Console()

MODEL_DIR = "data/models"


def prepare_training_data(session: Session) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Prepare features and labels for taste model training.

    Positive examples (label=1): songs with above-median play counts.
    Negative examples (label=0): songs in library but below-median or never played.

    Returns (X, y, feature_names).
    """
    vocabulary = build_tag_vocabulary(session, min_count=2)
    vocab_tags = sorted(vocabulary.keys(), key=lambda t: vocabulary[t])

    songs = session.query(Song).all()
    if not songs:
        return np.array([]), np.array([]), []

    # Compute median play count for threshold
    play_counts = [s.total_plays or 0 for s in songs]
    median_plays = sorted(play_counts)[len(play_counts) // 2]
    # Use at least 3 to avoid labeling everything positive
    threshold = max(median_plays, 3)

    feature_names = [
        "danceability", "energy", "valence", "acousticness",
        "instrumentalness", "speechiness", "liveness",
        "tempo_norm", "loudness_norm", "key_norm", "mode", "time_signature_norm",
        "log_plays", "recent_ratio", "staleness", "liked", "popularity_norm",
    ] + [f"tag_{t}" for t in vocab_tags]

    X_rows: list[list[float]] = []
    y_rows: list[int] = []

    for song in songs:
        row: list[float] = []

        # Audio features
        for attr in ["danceability", "energy", "valence", "acousticness",
                      "instrumentalness", "speechiness", "liveness"]:
            val = getattr(song, attr, None)
            row.append(val if val is not None else 0.5)

        tempo = song.tempo
        row.append(max(0.0, min(1.0, (tempo - 60) / 140)) if tempo is not None else 0.5)

        loudness = song.loudness
        row.append(max(0.0, min(1.0, (loudness + 60) / 60)) if loudness is not None else 0.5)

        key = song.key
        row.append(key / 11.0 if key is not None else 0.5)

        mode = song.mode
        row.append(float(mode) if mode is not None else 0.5)

        ts = song.time_signature
        row.append(max(0.0, min(1.0, (ts - 3) / 4)) if ts is not None else 0.5)

        # Listening stats
        total = song.total_plays or 0
        recent = song.recent_plays_30d or 0
        staleness = song.staleness_score if song.staleness_score is not None else 0.5
        liked = 1.0 if song.in_liked_songs else 0.0
        popularity = (song.popularity or 50) / 100.0

        row.append(min(np.log2(1 + total) / 10.0, 1.0))
        row.append(min(recent / max(total, 1) * 5, 1.0))
        row.append(staleness)
        row.append(liked)
        row.append(popularity)

        # Tag vector
        tags = parse_tags(song.lastfm_tags)
        for tag in vocab_tags:
            row.append(1.0 if tag in tags else 0.0)

        X_rows.append(row)
        y_rows.append(1 if total >= threshold else 0)

    return np.array(X_rows, dtype=np.float32), np.array(y_rows), feature_names


def train_taste_model(
    X: np.ndarray, y: np.ndarray, feature_names: list[str]
) -> tuple[object, dict]:
    """Train a LightGBM classifier. Returns (model, metrics).

    Metrics dict contains accuracy, auc, and feature_importance.
    """
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split

    if len(X) < 10:
        console.print("[yellow]Not enough data to train taste model.[/yellow]")
        return None, {}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None,
    )

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    valid_data = lgb.Dataset(X_test, label=y_test, feature_name=feature_names, reference=train_data)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    callbacks = [lgb.log_evaluation(period=0)]
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[valid_data],
        callbacks=callbacks,
    )

    # Compute metrics
    y_pred = model.predict(X_test)
    from sklearn.metrics import accuracy_score, roc_auc_score
    accuracy = accuracy_score(y_test, (y_pred > 0.5).astype(int))

    try:
        auc = roc_auc_score(y_test, y_pred)
    except ValueError:
        auc = 0.0  # Single class in test set

    importance = dict(zip(feature_names, model.feature_importance(importance_type="gain").tolist()))
    top_features = sorted(importance, key=importance.get, reverse=True)[:10]

    metrics = {"accuracy": accuracy, "auc": auc, "top_features": top_features}
    console.print(f"Taste model: accuracy={accuracy:.3f}, AUC={auc:.3f}")
    return model, metrics


def predict_taste(model, features: np.ndarray) -> float:
    """Predict taste score for a song (0.0 to 1.0).

    Features should match the format from prepare_training_data.
    """
    if model is None:
        return 0.5
    pred = model.predict(features.reshape(1, -1))
    return float(pred[0])


def save_model(model, path: str | None = None):
    """Save trained model to disk."""
    if model is None:
        return

    if path is None:
        root = get_project_root()
        path = str(root / MODEL_DIR / "taste_model.pkl")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, "wb") as f:
        pickle.dump(model, f)

    console.print(f"Model saved to [green]{path}[/green]")


def load_model(path: str | None = None):
    """Load trained model from disk. Returns None if not found."""
    if path is None:
        root = get_project_root()
        path = str(root / MODEL_DIR / "taste_model.pkl")

    p = Path(path)
    if not p.exists():
        return None

    with open(p, "rb") as f:
        return pickle.load(f)

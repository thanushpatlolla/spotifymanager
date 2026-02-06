"""Sprint 5: LightGBM taste model.

Trains a taste model that predicts how much the user will enjoy a song
based on audio features, tags, and listening history.
"""


def prepare_training_data(session) -> tuple:
    """Prepare features and labels for taste model training.

    Positive examples: songs with high play counts.
    Negative examples: songs in library but rarely/never played.

    Returns (X, y, feature_names).
    """
    raise NotImplementedError("Sprint 5")


def train_taste_model(X, y, feature_names: list[str]):
    """Train a LightGBM classifier. Returns (model, metrics)."""
    raise NotImplementedError("Sprint 5")


def predict_taste(model, features) -> float:
    """Predict taste score for a song (0.0 to 1.0)."""
    raise NotImplementedError("Sprint 5")


def save_model(model, path: str):
    """Save trained model to disk."""
    raise NotImplementedError("Sprint 5")


def load_model(path: str):
    """Load trained model from disk."""
    raise NotImplementedError("Sprint 5")

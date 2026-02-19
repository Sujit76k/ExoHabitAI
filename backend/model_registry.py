# backend/model_registry.py

import joblib
from backend.config import MODEL_PATH

# Global cached model (loaded once)
_model = None


def get_model():
    """
    Loads ML model only once and reuses it.
    This prevents reloading model on every API request.
    """
    global _model

    if _model is None:
        print("✅ Loading ML model from registry...")
        _model = joblib.load(MODEL_PATH)

    return _model

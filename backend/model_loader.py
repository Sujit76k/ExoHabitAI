# backend/model_loader.py

import joblib
from backend.config import MODEL_PATH

_model = None

def load_model():
    """
    Load ML model only once (singleton pattern)
    """
    global _model
    if _model is None:
        print("✅ Loading ML model...")
        _model = joblib.load(MODEL_PATH)
    return _model

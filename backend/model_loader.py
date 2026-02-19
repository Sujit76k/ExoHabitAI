# ======================================================
# 🚀 ExoHabitAI — Production Model Loader
# Thread-safe singleton ML model loader
# ======================================================

import os
import joblib
import threading

from backend.config import MODEL_PATH


# ======================================================
# 🧠 GLOBAL MODEL CACHE
# ======================================================

_model = None
_model_lock = threading.Lock()


# ======================================================
# ✅ INTERNAL LOAD FUNCTION
# ======================================================

def _load_from_disk():
    """
    Safely load model from disk.
    Raises clear errors if file missing or corrupted.
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"❌ Model file not found at: {MODEL_PATH}\n"
            f"Run training pipeline first."
        )

    try:
        print("🚀 Loading ML model from disk...")
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully")
        return model

    except Exception as e:
        raise RuntimeError(
            f"❌ Failed to load model: {str(e)}"
        )


# ======================================================
# ⭐ PUBLIC SINGLETON LOADER
# ======================================================

def load_model():
    """
    Returns cached model instance.

    Production features:
    - Thread-safe
    - Lazy loaded
    - Loaded only once
    """

    global _model

    # Fast path (already loaded)
    if _model is not None:
        return _model

    # Thread-safe loading
    with _model_lock:
        if _model is None:
            _model = _load_from_disk()

    return _model


# ======================================================
# 🧪 OPTIONAL — FORCE RELOAD (DEV MODE)
# ======================================================

def reload_model():
    """
    Force reload model from disk.
    Useful during development when retraining.
    """

    global _model

    with _model_lock:
        print("♻️ Reloading ML model...")
        _model = _load_from_disk()

    return _model
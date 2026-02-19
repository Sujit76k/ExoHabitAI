# ======================================================
# 🚀 ExoHabitAI — Production Model Registry
# Centralized ML model access layer
# ======================================================

import os
import joblib
import threading

from backend.config import MODEL_PATH


# ======================================================
# 🧠 GLOBAL MODEL STORAGE
# ======================================================

_model = None
_model_lock = threading.Lock()


# ======================================================
# 🔧 INTERNAL LOADER
# ======================================================

def _load_model_from_disk():
    """
    Safely load ML model from disk.
    Provides clear production-level error messages.
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"❌ Model not found at: {MODEL_PATH}\n"
            f"Run training pipeline before starting API."
        )

    try:
        print("🚀 Loading ML model (registry)...")
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully")
        return model

    except Exception as e:
        raise RuntimeError(f"❌ Failed to load model: {str(e)}")


# ======================================================
# ⭐ PUBLIC ACCESS FUNCTION
# ======================================================

def get_model():
    """
    Returns cached ML model instance.

    Features:
    - Singleton pattern
    - Thread-safe loading
    - Used across API routes & services
    """

    global _model

    # Fast return if already loaded
    if _model is not None:
        return _model

    # Thread-safe lazy load
    with _model_lock:
        if _model is None:
            _model = _load_model_from_disk()

    return _model


# ======================================================
# 🔁 OPTIONAL — FORCE RELOAD (DEV / HOT SWAP)
# ======================================================

def reload_model():
    """
    Force reload model from disk.
    Useful after retraining without restarting server.
    """

    global _model

    with _model_lock:
        print("♻️ Reloading model from registry...")
        _model = _load_model_from_disk()

    return _model
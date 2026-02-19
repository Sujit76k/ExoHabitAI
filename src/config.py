"""
=====================================================
🚀 ExoHabitAI — Global Configuration (Production Ready)
Centralized paths + ML settings
=====================================================
"""

import os

# =====================================================
# 🌌 BASE DIRECTORIES
# =====================================================

# Absolute project root (safe for any OS)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# =====================================================
# 📂 DATA PATHS
# =====================================================

RAW_DATA_PATH = os.path.join(
    RAW_DIR,
    "PS_2026.01.19_01.24.31.csv"
)

CLEANED_DATA_PATH = os.path.join(
    PROCESSED_DIR,
    "cleaned_exoplanets.csv"
)

FEATURE_ENGINEERED_PATH = os.path.join(
    PROCESSED_DIR,
    "feature_engineered_exoplanets.csv"
)

RANKED_DATA_PATH = os.path.join(
    PROCESSED_DIR,
    "ranked_exoplanets.csv"
)

# =====================================================
# 🤖 MODEL PATHS
# =====================================================

BASELINE_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "baseline_model.pkl"
)

BEST_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "week4_best_model.pkl"
)

METRICS_PATH = os.path.join(
    MODELS_DIR,
    "metrics.json"
)

# =====================================================
# 🧠 MACHINE LEARNING SETTINGS
# =====================================================

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Optional: future hyperparameter tuning
CV_FOLDS = 5

# =====================================================
# 🌍 API / DASHBOARD SETTINGS
# =====================================================

API_HOST = "127.0.0.1"
API_PORT = 5000

# Toggle debug mode globally
DEBUG = True

# =====================================================
# ⭐ SAFETY CHECK (Auto-create folders)
# =====================================================

REQUIRED_DIRS = [
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    MODELS_DIR,
    REPORTS_DIR,
]

for d in REQUIRED_DIRS:
    os.makedirs(d, exist_ok=True)
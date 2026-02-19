import os

# ======================================================
# 🚀 BASE PROJECT PATHS
# ======================================================

# Absolute root directory of ExoHabitAI project
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

# ======================================================
# 📁 DATA PATHS
# ======================================================

DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

CLEANED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "cleaned_exoplanets.csv")
ENGINEERED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "feature_engineered_exoplanets.csv")
RANKED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "ranked_exoplanets.csv")

# ======================================================
# 🤖 MODEL PATHS
# ======================================================

MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "week4_best_model.pkl")
BASELINE_MODEL_PATH = os.path.join(MODEL_DIR, "baseline_model.pkl")

# ======================================================
# 📊 REPORTS / FIGURES
# ======================================================

REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# ======================================================
# 🌐 API CONFIG
# ======================================================

API_HOST = "127.0.0.1"
API_PORT = 5000
DEBUG_MODE = True

# ======================================================
# 🧠 FRONTEND CONFIG
# ======================================================

FRONTEND_ORIGIN = "*"

# ======================================================
# 🔐 FUTURE DEPLOYMENT FLAGS
# ======================================================

ENVIRONMENT = os.getenv("EXOHABITAI_ENV", "development")
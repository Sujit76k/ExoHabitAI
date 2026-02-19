import os
import pandas as pd

from backend.config import RANKED_DATA_PATH


# =====================================================
# 🚀 SAFE DATA LOADER
# =====================================================

def load_ranked_dataset():
    """
    Loads ranked exoplanet dataset safely.
    Prevents crashes if file missing.
    """

    if not os.path.exists(RANKED_DATA_PATH):
        raise FileNotFoundError(
            f"Ranked dataset not found at: {RANKED_DATA_PATH}"
        )

    df = pd.read_csv(RANKED_DATA_PATH)

    if df.empty:
        raise ValueError("Ranked dataset is empty")

    return df


# =====================================================
# ⭐ LEVEL-100 RANKING SERVICE
# =====================================================

def get_ranked_planets(limit: int = 20):
    """
    Returns top ranked exoplanets.

    Features:
    - Safe loading
    - Automatic sorting
    - Column validation
    - Production-ready output
    """

    try:
        df = load_ranked_dataset()

        # --------------------------------------------------
        # Ensure important columns exist
        # --------------------------------------------------
        required_cols = ["prediction", "habitability_score"]

        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # --------------------------------------------------
        # Sort by habitability score (highest first)
        # --------------------------------------------------
        df = df.sort_values(
            by="habitability_score",
            ascending=False
        )

        # --------------------------------------------------
        # Limit results
        # --------------------------------------------------
        df = df.head(limit)

        # --------------------------------------------------
        # Convert safely to JSON format
        # --------------------------------------------------
        return df.to_dict(orient="records")

    except Exception as e:
        return {
            "error": "Ranking service failed",
            "message": str(e)
        }
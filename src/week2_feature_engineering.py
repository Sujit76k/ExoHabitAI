import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import ensure_dir_exists

CLEANED_PATH = os.path.join("data", "processed", "cleaned_exoplanets.csv")
ENGINEERED_PATH = os.path.join("data", "processed", "feature_engineered_exoplanets.csv")
FIG_DIR = os.path.join("reports", "figures")


# ======================================================
# ⭐ SAFE SCORE FUNCTION
# ======================================================
def safe_score(series, ideal, scale):
    """
    Creates a score 0..1 based on distance from ideal.
    Always returns pandas Series.
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    score = 1 - (np.abs(series - ideal) / scale)
    score = np.clip(score, 0, 1)
    return pd.Series(score, index=series.index)


# ======================================================
# ⭐ SAFE HSI (API + TRAINING COMPATIBLE)
# ======================================================
def create_hsi(df: pd.DataFrame) -> pd.Series:
    """
    Habitability Score Index (0 to 1)
    SAFE version — works even if columns missing.
    """

    radius = df["pl_rade"] if "pl_rade" in df.columns else pd.Series(0, index=df.index)
    temp   = df["pl_eqt"]  if "pl_eqt"  in df.columns else pd.Series(0, index=df.index)

    radius_score = safe_score(radius, ideal=1.0, scale=1.5)
    temp_score   = safe_score(temp, ideal=288, scale=200)

    hsi = (radius_score.fillna(0) + temp_score.fillna(0)) / 2
    return hsi


# ======================================================
# ⭐ SAFE SCI (API + TRAINING COMPATIBLE)
# ======================================================
def create_sci(df: pd.DataFrame) -> pd.Series:
    """
    Stellar Compatibility Index (0 to 1)
    SAFE version — works even if columns missing.
    """

    teff = df["st_teff"] if "st_teff" in df.columns else pd.Series(0, index=df.index)
    mass = df["st_mass"] if "st_mass" in df.columns else pd.Series(0, index=df.index)
    rad  = df["st_rad"]  if "st_rad"  in df.columns else pd.Series(0, index=df.index)

    teff_score = safe_score(teff, ideal=5778, scale=2500)
    mass_score = safe_score(mass, ideal=1.0, scale=1.0)
    rad_score  = safe_score(rad,  ideal=1.0, scale=1.0)

    sci = (teff_score.fillna(0) + mass_score.fillna(0) + rad_score.fillna(0)) / 3
    return sci


# ======================================================
# ⭐ API HELPER — USED BY BACKEND PREDICT ROUTE
# ======================================================
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply SAME feature engineering used during training.
    SAFE for API prediction with partial input.
    """

    df = df.copy()

    # Add engineered features
    df["HSI"] = create_hsi(df)
    df["SCI"] = create_sci(df)

    # Add missing training columns expected by model
    required_defaults = [
        "ast_flag",
        "cb_flag",
        "dec"
    ]

    for col in required_defaults:
        if col not in df.columns:
            df[col] = 0

    return df


# ======================================================
# ⭐ PLOT CORRELATION HEATMAP (WEEK 2 REPORT)
# ======================================================
def plot_correlation_heatmap(df: pd.DataFrame, save_path: str):
    numeric_df = df.select_dtypes(include=["number"])
    corr = numeric_df.corr()

    plt.figure(figsize=(12, 8))
    plt.imshow(corr, aspect="auto")
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ======================================================
# ⭐ WEEK 2 SCRIPT ENTRYPOINT
# ======================================================
def main():
    print("✅ WEEK 2 FEATURE ENGINEERING STARTED...")

    ensure_dir_exists(os.path.dirname(ENGINEERED_PATH))
    ensure_dir_exists(FIG_DIR)

    df = pd.read_csv(CLEANED_PATH)

    print("✅ Creating engineered features: HSI and SCI...")
    df["HSI"] = create_hsi(df)
    df["SCI"] = create_sci(df)

    # Create baseline habitability label if missing
    if "habitability" not in df.columns:
        df["habitability"] = (df["HSI"] >= 0.60).astype(int)

    df.to_csv(ENGINEERED_PATH, index=False)
    print(f"✅ Feature engineered dataset saved: {ENGINEERED_PATH}")

    print("✅ Saving correlation heatmap...")
    plot_correlation_heatmap(df, os.path.join(FIG_DIR, "correlation_heatmap.png"))

    print("🎉 WEEK 2 FEATURE ENGINEERING COMPLETED!")


if __name__ == "__main__":
    main()

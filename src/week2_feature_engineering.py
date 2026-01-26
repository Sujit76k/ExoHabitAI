import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import ensure_dir_exists

CLEANED_PATH = os.path.join("data", "processed", "cleaned_exoplanets.csv")
ENGINEERED_PATH = os.path.join("data", "processed", "feature_engineered_exoplanets.csv")
FIG_DIR = os.path.join("reports", "figures")


def safe_score(series, ideal, scale):
    """
    Creates a score 0..1 based on distance from ideal.
    score = 1 - (abs(x-ideal)/scale)
    """
    score = 1 - (np.abs(series - ideal) / scale)
    score = np.clip(score, 0, 1)
    return score


def create_hsi(df: pd.DataFrame) -> pd.Series:
    """
    Habitability Score Index (0 to 1)
    Based on radius + equilibrium temperature (if available).
    """
    radius = df.get("pl_rade", np.nan)
    temp = df.get("pl_eqt", np.nan)

    # Earth ideals
    radius_score = safe_score(radius, ideal=1.0, scale=1.5)  # allow super-earth
    temp_score = safe_score(temp, ideal=288, scale=200)      # wider safe margin

    # Average score (ignore missing by filling with 0)
    hsi = (radius_score.fillna(0) + temp_score.fillna(0)) / 2
    return hsi


def create_sci(df: pd.DataFrame) -> pd.Series:
    """
    Stellar Compatibility Index (0 to 1)
    Based on star temp + mass + radius (if available).
    """
    teff = df.get("st_teff", np.nan)
    mass = df.get("st_mass", np.nan)
    rad = df.get("st_rad", np.nan)

    teff_score = safe_score(teff, ideal=5778, scale=2500)  # Sun-like range
    mass_score = safe_score(mass, ideal=1.0, scale=1.0)
    rad_score = safe_score(rad, ideal=1.0, scale=1.0)

    sci = (teff_score.fillna(0) + mass_score.fillna(0) + rad_score.fillna(0)) / 3
    return sci


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


def main():
    print("✅ WEEK 2 FEATURE ENGINEERING STARTED...")

    ensure_dir_exists(os.path.dirname(ENGINEERED_PATH))
    ensure_dir_exists(FIG_DIR)

    df = pd.read_csv(CLEANED_PATH)

    # Feature Engineering
    print("✅ Creating engineered features: HSI and SCI...")
    df["HSI"] = create_hsi(df)
    df["SCI"] = create_sci(df)

    # If habitability target not present, create a baseline label from HSI
    if "habitability" not in df.columns:
        df["habitability"] = (df["HSI"] >= 0.60).astype(int)

    # Save engineered dataset
    df.to_csv(ENGINEERED_PATH, index=False)
    print(f"✅ Feature engineered dataset saved: {ENGINEERED_PATH}")

    # Plots
    print("✅ Saving correlation heatmap...")
    plot_correlation_heatmap(df, os.path.join(FIG_DIR, "correlation_heatmap.png"))

    print("🎉 WEEK 2 FEATURE ENGINEERING COMPLETED!")


if __name__ == "__main__":
    main()

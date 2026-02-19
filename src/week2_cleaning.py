"""
=====================================================
🚀 ExoHabitAI — WEEK 2 CLEANING (Production Version)
Handles:
✔ NASA dataset cleaning
✔ API preprocessing
✔ Outlier clipping
✔ Visualization
=====================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_raw_data
from src.preprocessing import fix_duplicate_columns, basic_cleaning
from src.utils import ensure_dir_exists, log


CLEANED_PATH = os.path.join("data", "processed", "cleaned_exoplanets.csv")
FIG_DIR = os.path.join("reports", "figures")


# =====================================================
# ⭐ API CLEANING FUNCTION (BACKEND SAFE)
# =====================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight cleaning used during API prediction.

    Must remain FAST and CONSISTENT with training pipeline.
    """

    df = df.copy()

    # ⭐ FAST numeric conversion (vectorized)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass

    # Replace infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill numeric NaN with 0 (safe fallback)
    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].fillna(0)

    return df


# =====================================================
# 📊 VISUALIZATION
# =====================================================

def plot_missing_values(df: pd.DataFrame, save_path: str):
    try:
        missing_pct = (df.isna().mean() * 100).sort_values(ascending=False).head(25)

        if missing_pct.empty:
            return

        plt.figure(figsize=(10, 5))
        plt.bar(missing_pct.index.astype(str), missing_pct.values)
        plt.xticks(rotation=90)
        plt.title("Top 25 Columns Missing Percentage")
        plt.ylabel("Missing %")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    except Exception as e:
        log(f"Plot missing values failed: {e}", "WARNING")


def plot_boxplots(df_before, df_after, col, save_path):
    try:
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.boxplot(df_before[col].dropna(), vert=True)
        plt.title(f"Before Clip\n{col}")

        plt.subplot(1, 2, 2)
        plt.boxplot(df_after[col].dropna(), vert=True)
        plt.title(f"After Clip\n{col}")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    except Exception as e:
        log(f"Boxplot failed for {col}: {e}", "WARNING")


# =====================================================
# 📉 OUTLIER HANDLING
# =====================================================

def iqr_clip_outliers(df: pd.DataFrame, cols: list, factor: float = 1.5):
    df = df.copy()

    for col in cols:

        if col not in df.columns:
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        if pd.isna(IQR) or IQR == 0:
            continue

        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR

        df[col] = df[col].clip(lower, upper)

    return df


# =====================================================
# 🚀 MAIN WEEK 2 CLEANING PIPELINE
# =====================================================

def main():

    log("WEEK 2 CLEANING STARTED")

    ensure_dir_exists(os.path.dirname(CLEANED_PATH))
    ensure_dir_exists(FIG_DIR)

    log("Loading raw dataset...")
    df = load_raw_data()

    log("Fixing duplicate columns...")
    df = fix_duplicate_columns(df)

    log("Removing duplicates...")
    df = basic_cleaning(df)

    # ===============================
    # Missing Value Visualization
    # ===============================
    plot_missing_values(df, os.path.join(FIG_DIR, "missing_values_top25.png"))

    # ===============================
    # Numeric Filling (Vectorized)
    # ===============================
    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        df[cat_cols] = df[cat_cols].fillna("Unknown")

    # ===============================
    # Outlier Clipping
    # ===============================
    log("Applying IQR clipping...")

    df_before = df.copy()

    important_numeric_cols = [
        "pl_rade",
        "pl_bmasse",
        "pl_orbper",
        "pl_eqt",
        "st_teff",
        "st_mass",
        "st_rad",
    ]

    df = iqr_clip_outliers(df, important_numeric_cols)

    for c in ["pl_rade", "pl_eqt", "pl_orbper"]:
        if c in df.columns:
            plot_boxplots(
                df_before,
                df,
                c,
                os.path.join(FIG_DIR, f"boxplot_{c}.png"),
            )

    # ===============================
    # Save Dataset
    # ===============================
    df.to_csv(CLEANED_PATH, index=False)

    log(f"Cleaned dataset saved → {CLEANED_PATH}")
    log("WEEK 2 CLEANING COMPLETED", "SUCCESS")


if __name__ == "__main__":
    main()
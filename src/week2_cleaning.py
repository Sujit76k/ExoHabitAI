import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_raw_data
from src.preprocessing import fix_duplicate_columns, basic_cleaning
from src.utils import ensure_dir_exists


CLEANED_PATH = os.path.join("data", "processed", "cleaned_exoplanets.csv")
FIG_DIR = os.path.join("reports", "figures")

# =====================================================
# ⭐ API CLEANING FUNCTION (USED BY BACKEND)
# =====================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight cleaning for API prediction.
    Must match training preprocessing behavior.
    """

    df = df.copy()

    # Convert numeric safely
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # Replace infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill missing numeric with 0 (simple safe rule)
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    return df


def plot_missing_values(df: pd.DataFrame, save_path: str):
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False).head(25)

    plt.figure(figsize=(10, 5))
    plt.bar(missing_pct.index.astype(str), missing_pct.values)
    plt.xticks(rotation=90)
    plt.title("Top 25 Columns Missing Percentage")
    plt.ylabel("Missing %")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def iqr_clip_outliers(df: pd.DataFrame, cols: list, factor: float = 1.5) -> pd.DataFrame:
    """
    Clips outliers using IQR method.
    """
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


def plot_boxplots(df_before: pd.DataFrame, df_after: pd.DataFrame, col: str, save_path: str):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.boxplot(df_before[col].dropna(), vert=True)
    plt.title(f"Before Outlier Clip\n{col}")

    plt.subplot(1, 2, 2)
    plt.boxplot(df_after[col].dropna(), vert=True)
    plt.title(f"After Outlier Clip\n{col}")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    print("✅ WEEK 2 CLEANING STARTED...")

    ensure_dir_exists(os.path.dirname(CLEANED_PATH))
    ensure_dir_exists(FIG_DIR)

    print("✅ Loading raw dataset...")
    df = load_raw_data()

    print("✅ Fixing duplicate columns...")
    df = fix_duplicate_columns(df)

    print("✅ Basic cleaning (duplicates removal)...")
    df = basic_cleaning(df)

    # Plot missing values
    print("✅ Plotting missing values...")
    plot_missing_values(df, os.path.join(FIG_DIR, "missing_values_top25.png"))

    # Fill missing numeric with median
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill missing categorical with "Unknown"
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")

    # Outlier handling (clip)
    print("✅ Handling outliers using IQR clipping...")
    df_before = df.copy()

    important_numeric_cols = [
        "pl_rade", "pl_bmasse", "pl_orbper", "pl_eqt",
        "st_teff", "st_mass", "st_rad"
    ]

    df = iqr_clip_outliers(df, important_numeric_cols)

    # Save boxplots for common important columns if present
    for c in ["pl_rade", "pl_eqt", "pl_orbper"]:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            plot_boxplots(
                df_before, df, c,
                os.path.join(FIG_DIR, f"boxplot_{c}.png")
            )

    # Save cleaned dataset
    df.to_csv(CLEANED_PATH, index=False)
    print(f"✅ Cleaned dataset saved: {CLEANED_PATH}")

    print("🎉 WEEK 2 CLEANING COMPLETED!")


if __name__ == "__main__":
    main()

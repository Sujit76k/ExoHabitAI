"""
=====================================================
🚀 ExoHabitAI — Exploratory Data Analysis Module
Professional EDA summary generator
=====================================================
"""

import os
import logging
import pandas as pd

# -----------------------------------------------------
# LOGGER
# -----------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------
# INTERNAL HELPERS
# -----------------------------------------------------

def _get_missing_summary(df: pd.DataFrame, top_n: int = 20) -> pd.Series:
    """
    Returns missing value percentage sorted descending.
    """
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    return missing_pct.head(top_n)


def _get_numeric_summary(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Safe numeric description (handles large datasets).
    """
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        return pd.DataFrame()

    return numeric_df.describe().T.head(top_n)


def _get_categorical_summary(df: pd.DataFrame, top_n: int = 20) -> pd.Series:
    """
    Top categorical columns by uniqueness.
    """
    cat_df = df.select_dtypes(include=["object"])
    if cat_df.empty:
        return pd.Series(dtype="object")

    return cat_df.nunique().sort_values(ascending=False).head(top_n)


# -----------------------------------------------------
# MAIN FUNCTION
# -----------------------------------------------------

def save_eda_summary(df: pd.DataFrame, output_path: str):
    """
    Generate structured EDA report.

    Output includes:
    ✔ dataset shape
    ✔ missing value analysis
    ✔ numeric statistics
    ✔ categorical uniqueness
    ✔ memory usage

    Saves a readable professional .txt file.
    """

    logger.info("📊 Generating EDA summary...")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = []

    # -------------------------------------------------
    # BASIC INFO
    # -------------------------------------------------

    lines.append("=================================================")
    lines.append("🚀 EXOHABITAI — EDA SUMMARY REPORT")
    lines.append("=================================================")

    lines.append(f"\nRows: {df.shape[0]}")
    lines.append(f"Columns: {df.shape[1]}")
    lines.append(f"Memory Usage (MB): {round(df.memory_usage(deep=True).sum() / 1e6, 2)}")

    # -------------------------------------------------
    # COLUMN TYPES
    # -------------------------------------------------

    lines.append("\n--- Column Types ---")
    dtype_counts = df.dtypes.value_counts()
    lines.append(str(dtype_counts))

    # -------------------------------------------------
    # MISSING VALUES
    # -------------------------------------------------

    lines.append("\n--- Missing Values % (Top 20) ---")
    missing_summary = _get_missing_summary(df)
    lines.append(str(missing_summary))

    # -------------------------------------------------
    # NUMERIC STATS
    # -------------------------------------------------

    lines.append("\n--- Numeric Feature Statistics ---")
    numeric_summary = _get_numeric_summary(df)
    if numeric_summary.empty:
        lines.append("No numeric columns found.")
    else:
        lines.append(str(numeric_summary))

    # -------------------------------------------------
    # CATEGORICAL STATS
    # -------------------------------------------------

    lines.append("\n--- Categorical Feature Uniqueness ---")
    cat_summary = _get_categorical_summary(df)
    if cat_summary.empty:
        lines.append("No categorical columns found.")
    else:
        lines.append(str(cat_summary))

    # -------------------------------------------------
    # SAVE FILE
    # -------------------------------------------------

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"✅ EDA report saved at: {output_path}")
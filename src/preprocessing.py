"""
=====================================================
🚀 ExoHabitAI — Data Preprocessing Module
Production-ready preprocessing utilities
=====================================================
"""

import pandas as pd


# -----------------------------------------------------
# COLUMN UTILITIES
# -----------------------------------------------------

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names for safety.

    - strip spaces
    - lower case
    - replace spaces with underscore
    """

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.lower()
    )

    return df


# -----------------------------------------------------
# DUPLICATE COLUMN FIX
# -----------------------------------------------------

def fix_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all column names are unique.

    Example:
    ['a','a','b'] -> ['a','a_1','b']
    """

    df = df.copy()
    cols = pd.Series(df.columns)

    for dup in cols[cols.duplicated()].unique():
        dup_idx = cols[cols == dup].index.tolist()
        for i, idx in enumerate(dup_idx):
            if i == 0:
                continue
            cols[idx] = f"{dup}_{i}"

    df.columns = cols
    return df


# -----------------------------------------------------
# BASIC CLEANING
# -----------------------------------------------------

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic dataset cleanup.
    """

    df = df.copy()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")

    return df


# -----------------------------------------------------
# SAFE NUMERIC CONVERSION
# -----------------------------------------------------

def force_numeric_conversion(df: pd.DataFrame, ignore_cols=None) -> pd.DataFrame:
    """
    Convert columns into numeric safely.

    ignore_cols:
        columns that should NOT be converted
        (e.g., pl_name, hostname)
    """

    if ignore_cols is None:
        ignore_cols = []

    df = df.copy()

    for col in df.columns:

        if col in ignore_cols:
            continue

        # Only attempt conversion on object columns
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# -----------------------------------------------------
# FULL PREPROCESS ENTRY (RECOMMENDED)
# -----------------------------------------------------

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-call preprocessing function.
    Use this inside ML pipeline scripts.
    """

    if df is None or len(df) == 0:
        raise ValueError("❌ Empty dataframe provided to preprocessing.")

    df = normalize_column_names(df)
    df = fix_duplicate_columns(df)
    df = basic_cleaning(df)

    return df


# -----------------------------------------------------
# FEATURE SELECTION (NUMERIC ONLY)
# -----------------------------------------------------

def select_numeric_features(df: pd.DataFrame, target_col: str = "habitability"):
    """
    Extract numeric features and target column.

    Returns:
        X (numeric features)
        y (target)
    """

    if target_col not in df.columns:
        raise ValueError(
            f"❌ Target column '{target_col}' not found in dataset."
        )

    df = fix_duplicate_columns(df)

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Keep numeric columns only
    X = X.select_dtypes(include=["number"])

    # Remove columns that are entirely NaN
    X = X.dropna(axis=1, how="all")

    if X.shape[1] == 0:
        raise ValueError(
            "❌ No numeric features found after preprocessing.\n"
            "Dataset loaded but no usable numeric columns exist."
        )

    return X, y
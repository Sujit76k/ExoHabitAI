"""
=====================================================
🚀 ExoHabitAI — Habitability Target Builder
Production-grade target generation for ML training
=====================================================
"""

import numpy as np
import pandas as pd


# -----------------------------------------------------
# COLUMN FINDER (SAFE)
# -----------------------------------------------------

def find_first_existing_col(df: pd.DataFrame, candidates):
    """
    Return first column that exists inside dataframe.
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


# -----------------------------------------------------
# SAFE TARGET BALANCER
# -----------------------------------------------------

def ensure_class_balance(df: pd.DataFrame, col="habitability"):
    """
    Prevent single-class datasets.
    If only one class exists -> auto create soft split.
    """

    if col not in df.columns:
        return df

    unique_vals = df[col].nunique()

    # If model will fail due to only one class
    if unique_vals < 2:
        print("⚠️ Only one class detected. Applying quantile fallback target.")

        numeric_df = df.select_dtypes(include=["number"]).copy()

        if numeric_df.shape[1] > 0:
            score = numeric_df.fillna(numeric_df.median()).sum(axis=1)
            threshold = score.quantile(0.80)
            df[col] = np.where(score >= threshold, 1, 0)

    return df


# -----------------------------------------------------
# MAIN TARGET BUILDER
# -----------------------------------------------------

def build_habitability_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    LEVEL-100 Target Builder

    Strategy:
    1️⃣ Scientific rule-based labeling (if columns exist)
    2️⃣ Fallback quantile scoring
    3️⃣ Auto class-balance protection
    """

    df = df.copy()

    # Avoid overriding if already exists
    if "habitability" not in df.columns:
        df["habitability"] = 0

    # Common NASA column candidates
    radius_candidates = ["pl_rade", "pl_radj"]
    temp_candidates = ["pl_eqt"]
    star_temp_candidates = ["st_teff"]

    r_col = find_first_existing_col(df, radius_candidates)
    t_col = find_first_existing_col(df, temp_candidates)
    st_col = find_first_existing_col(df, star_temp_candidates)

    # -------------------------------------------------
    # ⭐ SCIENTIFIC TARGET (PRIMARY LOGIC)
    # -------------------------------------------------

    if r_col is not None and t_col is not None:

        # Wider safe astrophysics ranges
        temp_ok = (df[t_col] >= 150) & (df[t_col] <= 400)
        radius_ok = (df[r_col] >= 0.5) & (df[r_col] <= 2.5)

        # Optional stellar filtering
        if st_col is not None:
            star_ok = (df[st_col] >= 3500) & (df[st_col] <= 8000)
        else:
            star_ok = True

        df["habitability"] = np.where(temp_ok & radius_ok & star_ok, 1, 0)

    # -------------------------------------------------
    # ⭐ FALLBACK TARGET (NO KEY FEATURES)
    # -------------------------------------------------

    else:
        numeric_df = df.select_dtypes(include=["number"]).copy()

        if numeric_df.shape[1] == 0:
            print("⚠️ No numeric features available. Target not created.")
            return df

        print("⚠️ Using fallback quantile-based habitability target.")

        score = numeric_df.fillna(numeric_df.median()).sum(axis=1)

        # Slightly softer threshold for better class balance
        threshold = score.quantile(0.85)

        df["habitability"] = np.where(score >= threshold, 1, 0)

    # -------------------------------------------------
    # ⭐ CLASS BALANCE PROTECTION
    # -------------------------------------------------

    df = ensure_class_balance(df, "habitability")

    # -------------------------------------------------
    # ⭐ DEBUG INFO (VERY IMPORTANT FOR ML)
    # -------------------------------------------------

    try:
        counts = df["habitability"].value_counts().to_dict()
        print(f"✅ Habitability distribution: {counts}")
    except Exception:
        pass

    return df
"""
=====================================================
🚀 ExoHabitAI — Feature Engineering Module
Scientific & Production-ready feature creation
=====================================================
"""

import numpy as np
import pandas as pd


# -----------------------------------------------------
# SAFE SCORE FUNCTION
# -----------------------------------------------------

def safe_score(series, ideal: float, scale: float):
    """
    Create normalized 0..1 score based on distance from ideal value.

    score = 1 - (abs(x - ideal) / scale)

    Works safely even if column missing.
    """
    if isinstance(series, (int, float)):
        series = pd.Series([series])

    score = 1 - (np.abs(series - ideal) / scale)
    return np.clip(score, 0, 1)


# -----------------------------------------------------
# HABITABILITY SCORE INDEX (HSI)
# -----------------------------------------------------

def create_hsi(df: pd.DataFrame) -> pd.Series:
    """
    Habitability Score Index
    Based on:
        - planet radius
        - equilibrium temperature
    """

    radius = df["pl_rade"] if "pl_rade" in df.columns else pd.Series(0, index=df.index)
    temp = df["pl_eqt"] if "pl_eqt" in df.columns else pd.Series(0, index=df.index)

    radius_score = safe_score(radius, ideal=1.0, scale=1.5)
    temp_score = safe_score(temp, ideal=288, scale=200)

    return (radius_score.fillna(0) + temp_score.fillna(0)) / 2


# -----------------------------------------------------
# STELLAR COMPATIBILITY INDEX (SCI)
# -----------------------------------------------------

def create_sci(df: pd.DataFrame) -> pd.Series:
    """
    Stellar Compatibility Index
    Measures similarity to Sun-like stars.
    """

    teff = df["st_teff"] if "st_teff" in df.columns else pd.Series(0, index=df.index)
    mass = df["st_mass"] if "st_mass" in df.columns else pd.Series(0, index=df.index)
    rad = df["st_rad"] if "st_rad" in df.columns else pd.Series(0, index=df.index)

    teff_score = safe_score(teff, ideal=5778, scale=2500)
    mass_score = safe_score(mass, ideal=1.0, scale=1.0)
    rad_score = safe_score(rad, ideal=1.0, scale=1.0)

    return (teff_score.fillna(0) + mass_score.fillna(0) + rad_score.fillna(0)) / 3


# -----------------------------------------------------
# EXTRA SCIENTIFIC FEATURES (LEVEL-UP)
# -----------------------------------------------------

def create_log_features(df: pd.DataFrame):
    """
    Add log-transformed features for skewed astronomy values.
    """
    df = df.copy()

    log_cols = ["pl_orbper", "pl_bmasse", "pl_rade"]

    for col in log_cols:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    return df


def create_ratio_features(df: pd.DataFrame):
    """
    Create physics-inspired ratios.
    """

    df = df.copy()

    if "pl_rade" in df.columns and "st_rad" in df.columns:
        df["planet_star_radius_ratio"] = df["pl_rade"] / (df["st_rad"] + 1e-6)

    if "st_mass" in df.columns and "st_rad" in df.columns:
        df["stellar_density_proxy"] = df["st_mass"] / (df["st_rad"] + 1e-6)

    return df


# -----------------------------------------------------
# MAIN FEATURE ENGINEERING PIPELINE
# -----------------------------------------------------

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    MASTER FEATURE ENGINEERING FUNCTION

    This function must be identical during:
        - Training
        - API Prediction
        - Batch Processing
    """

    df = df.copy()

    # ---------------------------------------------
    # Scientific Engineered Features
    # ---------------------------------------------

    df["HSI"] = create_hsi(df)
    df["SCI"] = create_sci(df)

    # ---------------------------------------------
    # Advanced Features
    # ---------------------------------------------

    df = create_log_features(df)
    df = create_ratio_features(df)

    # ---------------------------------------------
    # SAFE DEFAULT FLAGS (needed by model training)
    # ---------------------------------------------

    default_cols = ["ast_flag", "cb_flag", "dec"]

    for col in default_cols:
        if col not in df.columns:
            df[col] = 0

    return df
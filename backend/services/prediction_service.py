# backend/services/prediction_service.py

import pandas as pd
import math

from backend.model_registry import get_model
from src.week2_cleaning import clean_data
from src.week2_feature_engineering import add_engineered_features


# =====================================================
# 🚀 FEATURE ALIGNMENT ENGINE
# =====================================================

def align_features_to_model(df: pd.DataFrame, model):

    final_model = model

    if hasattr(model, "named_steps"):
        final_model = list(model.named_steps.values())[-1]

    try:
        expected_cols = model.feature_names_in_
    except AttributeError:
        expected_cols = getattr(final_model, "feature_names_in_", [])

    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    if len(expected_cols) > 0:
        df = df[expected_cols]

    return df


# =====================================================
# 🌌 QUANTUM ORBITAL STABILITY INDEX (NEW)
# =====================================================

def orbital_stability_score(data: dict):
    """
    Estimates orbital stability based on period.
    Earth-like ~365 days = ideal
    """

    orb = float(data.get("pl_orbper", 0) or 0)

    if orb == 0:
        return 0.5  # neutral

    ideal = 365
    scale = 600

    score = 1 - abs(orb - ideal) / scale
    return max(0.0, min(1.0, score))


# =====================================================
# 🧠 QUANTUM FUSION ENGINE
# =====================================================

def quantum_neural_fusion(model_prob, hsi, sci, orbit_score):
    """
    NASA MODE Fusion Formula

    Weights tuned for astrophysics logic:
        AI Model       = 35%
        HSI Physics    = 30%
        SCI Stellar    = 20%
        Orbit Stability= 15%
    """

    model_prob = max(0, min(1, model_prob))
    hsi = max(0, min(1, hsi))
    sci = max(0, min(1, sci))
    orbit_score = max(0, min(1, orbit_score))

    base_score = (
        0.35 * model_prob +
        0.30 * hsi +
        0.20 * sci +
        0.15 * orbit_score
    )

    # --------------------------------------------------
    # ⭐ QUANTUM BOOST CURVE (NON-LINEAR ENHANCEMENT)
    # --------------------------------------------------
    # Boost high-quality planets more aggressively
    quantum_score = math.pow(base_score, 0.85)

    return round(float(quantum_score), 4)


# =====================================================
# 🚀 FINAL QUANTUM HABITABILITY ENGINE
# =====================================================

def predict_planet(data: dict):

    model = get_model()

    if not data:
        raise ValueError("Empty input data provided")

    # --------------------------------------------------
    # Build dataframe
    # --------------------------------------------------
    df = pd.DataFrame([data])

    df = clean_data(df)
    df = add_engineered_features(df)

    # --------------------------------------------------
    # Extract science features BEFORE alignment
    # --------------------------------------------------
    hsi = float(df["HSI"].iloc[0]) if "HSI" in df.columns else 0.0
    sci = float(df["SCI"].iloc[0]) if "SCI" in df.columns else 0.0

    orbit_score = orbital_stability_score(data)

    # --------------------------------------------------
    # Align to ML model schema
    # --------------------------------------------------
    df_model = align_features_to_model(df.copy(), model)

    # --------------------------------------------------
    # ML Prediction
    # --------------------------------------------------
    prediction = int(model.predict(df_model)[0])
    model_prob = float(model.predict_proba(df_model)[0][1])

    # --------------------------------------------------
    # 🌌 QUANTUM HABITABILITY SCORE
    # --------------------------------------------------
    final_score = quantum_neural_fusion(
        model_prob,
        hsi,
        sci,
        orbit_score
    )

    # Dynamic threshold (Quantum Decision Layer)
    prediction = 1 if final_score >= 0.58 else 0

    # --------------------------------------------------
    # Dashboard Response
    # --------------------------------------------------
    return {
        "prediction": prediction,
        "habitability_score": final_score,
        "insights": {
            "model_probability": round(model_prob, 4),
            "HSI": round(hsi, 4),
            "SCI": round(sci, 4),
            "orbit_stability": round(orbit_score, 4),
        },
    }
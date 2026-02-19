import pandas as pd

from backend.model_registry import get_model
from src.week2_cleaning import clean_data
from src.week2_feature_engineering import add_engineered_features


# =====================================================
# 🚀 FEATURE ALIGNMENT ENGINE (CRITICAL FOR PIPELINES)
# =====================================================

def align_features_to_model(df: pd.DataFrame, model):
    """
    Ensures dataframe columns match training pipeline.

    - Adds missing columns
    - Removes extra columns
    - Keeps correct order
    """

    try:
        expected_cols = model.feature_names_in_
    except AttributeError:
        # If using sklearn pipeline
        expected_cols = model.named_steps["model"].feature_names_in_

    # Add missing features
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    # Keep only expected columns
    df = df[expected_cols]

    return df


# =====================================================
# ⭐ LEVEL-100 SCIENTIFIC PREDICTION SERVICE
# =====================================================

def predict_planet(data: dict):
    """
    Main AI prediction service.

    Steps:
    1. Convert JSON → DataFrame
    2. Apply cleaning pipeline
    3. Apply feature engineering (HSI / SCI)
    4. Align features with trained model
    5. Run prediction
    """

    # --------------------------------------------------
    # Load model (singleton from registry)
    # --------------------------------------------------
    model = get_model()

    if not data:
        raise ValueError("Empty input data provided")

    # --------------------------------------------------
    # Convert input to dataframe
    # --------------------------------------------------
    df = pd.DataFrame([data])

    # --------------------------------------------------
    # Apply SAME TRAINING PIPELINE
    # --------------------------------------------------
    df = clean_data(df)
    df = add_engineered_features(df)

    # --------------------------------------------------
    # VERY IMPORTANT:
    # Align dataframe to model schema
    # --------------------------------------------------
    df = align_features_to_model(df, model)

    # --------------------------------------------------
    # Predict
    # --------------------------------------------------
    prediction = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "habitability_score": float(prob)
    }
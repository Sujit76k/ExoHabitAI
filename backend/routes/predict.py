from flask import Blueprint, request, jsonify
import pandas as pd

from backend.model_loader import load_model
from src.week2_cleaning import clean_data
from src.week2_feature_engineering import add_engineered_features


predict_bp = Blueprint("predict", __name__)

model = load_model()


# =====================================================
# ⭐ PRO FEATURE ALIGNMENT FUNCTION
# =====================================================
def align_features_to_model(df, model):
    """
    Ensures dataframe matches training feature schema.
    Adds missing columns and keeps correct order.
    """

    # Try to read feature names from trained pipeline
    try:
        expected_cols = model.feature_names_in_
    except AttributeError:
        # fallback if pipeline stores inside final estimator
        expected_cols = model.named_steps["model"].feature_names_in_

    # Add missing columns
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    # Keep ONLY expected columns in correct order
    df = df[expected_cols]

    return df


# =====================================================
# ⭐ PRO PREDICT ROUTE
# =====================================================
@predict_bp.route("/predict", methods=["POST"])
def predict():

    """
    Predict Exoplanet Habitability
    ---
    tags:
      - Prediction
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            pl_rade:
              type: number
              example: 1.2
            pl_eqt:
              type: number
              example: 290
    responses:
      200:
        description: Prediction result
        schema:
          type: object
          properties:
            prediction:
              type: integer
            habitability_score:
              type: number
    """

    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    # Convert to dataframe
    df = pd.DataFrame([data])

    # Apply same pipeline as training
    df = clean_data(df)
    df = add_engineered_features(df)

    # ⭐ ALIGN FEATURES TO MODEL (PRO FIX)
    df = align_features_to_model(df, model)

    # Predict
    prediction = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return jsonify({
        "prediction": int(prediction),
        "habitability_score": float(prob)
    })

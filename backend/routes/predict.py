from flask import Blueprint, request, jsonify
import pandas as pd

from backend.model_registry import get_model
from src.week2_cleaning import clean_data
from src.week2_feature_engineering import add_engineered_features

predict_bp = Blueprint("predict", __name__)


# =====================================================
# 🚀 SCIENTIFIC INPUT VALIDATION
# =====================================================

VALIDATION_RULES = {
    "pl_rade": (0.1, 20),
    "pl_eqt": (50, 2000),
    "pl_orbper": (0, 5000),
    "st_teff": (2000, 10000),
    "st_mass": (0.1, 5),
    "st_rad": (0.1, 10),
}


def validate_inputs(data: dict):
    errors = []

    for key, (min_v, max_v) in VALIDATION_RULES.items():
        if key in data:
            try:
                val = float(data[key])
                if val < min_v or val > max_v:
                    errors.append(f"{key} outside scientific range [{min_v},{max_v}]")
            except Exception:
                errors.append(f"{key} must be numeric")

    return errors


# =====================================================
# ⭐ FEATURE ALIGNMENT
# =====================================================

def align_features_to_model(df, model):
    """
    Align dataframe columns with model training schema.
    """

    # Extract final estimator if pipeline
    final_model = model
    if hasattr(model, "named_steps"):
        final_model = list(model.named_steps.values())[-1]

    # Get expected feature order
    try:
        expected_cols = model.feature_names_in_
    except AttributeError:
        expected_cols = getattr(final_model, "feature_names_in_", [])

    # Add missing columns
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    # Keep only expected columns
    if len(expected_cols) > 0:
        df = df[expected_cols]

    return df


# =====================================================
# 🚀 FINAL PRO PREDICT ROUTE
# =====================================================

@predict_bp.route("/predict", methods=["POST"])
def predict():
    """
    Predict Exoplanet Habitability
    ---
    tags:
      - Prediction
    consumes:
      - application/json
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
            pl_orbper:
              type: number
              example: 365
            st_teff:
              type: number
              example: 5778
            st_mass:
              type: number
              example: 1.0
            st_rad:
              type: number
              example: 1.0
    responses:
      200:
        description: Prediction result
    """

    try:
        model = get_model()

        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        # --------------------------------------------------
        # 🧪 Scientific Validation
        # --------------------------------------------------
        errors = validate_inputs(data)
        if errors:
            return jsonify({
                "status": "invalid_input",
                "errors": errors
            }), 400

        # --------------------------------------------------
        # 🧠 Build dataframe
        # --------------------------------------------------
        df = pd.DataFrame([data])

        # Apply same pipeline as training
        df = clean_data(df)
        df = add_engineered_features(df)

        # Align with model features
        df = align_features_to_model(df, model)

        # --------------------------------------------------
        # 🤖 Predict
        # --------------------------------------------------
        prediction = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        # --------------------------------------------------
        # 📊 Include engineered insights (for dashboard)
        # --------------------------------------------------
        extra = {}
        if "HSI" in df.columns:
            extra["HSI"] = float(df["HSI"].iloc[0])
        if "SCI" in df.columns:
            extra["SCI"] = float(df["SCI"].iloc[0])

        return jsonify({
            "status": "success",
            "prediction": int(prediction),
            "habitability_score": float(prob),
            "insights": extra,
            "model": "ExoHabitAI-Week4-RandomForest"
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
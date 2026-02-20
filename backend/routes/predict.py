from flask import Blueprint, request, jsonify

from backend.services.prediction_service import predict_planet

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
                    errors.append(
                        f"{key} outside scientific range [{min_v},{max_v}]"
                    )
            except Exception:
                errors.append(f"{key} must be numeric")

    return errors


# =====================================================
# 🚀 FINAL ADAPTIVE NEURAL PREDICT ROUTE
# =====================================================

@predict_bp.route("/predict", methods=["POST"])
def predict():
    """
    Adaptive Neural Habitability Prediction API
    """

    try:
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
        # 🧠 CALL AI SERVICE (REAL SCORING ENGINE)
        # --------------------------------------------------
        result = predict_planet(data)

        # --------------------------------------------------
        # 🚀 RESPONSE TO DASHBOARD
        # --------------------------------------------------
        return jsonify({
            "status": "success",
            "prediction": result["prediction"],
            "habitability_score": result["habitability_score"],
            "insights": result.get("insights", {}),
            "model": "ExoHabitAI-AdaptiveNeural"
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
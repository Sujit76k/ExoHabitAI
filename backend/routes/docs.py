from flask import Blueprint, jsonify

docs_bp = Blueprint("docs", __name__)


@docs_bp.route("/docs", methods=["GET"])
def docs():
    """
    Simple professional API documentation endpoint.
    """

    return jsonify({
        "project": "ExoHabitAI - Exoplanet Habitability Prediction API",
        "endpoints": [
            {
                "path": "/",
                "method": "GET",
                "description": "API health check"
            },
            {
                "path": "/predict",
                "method": "POST",
                "description": "Predict exoplanet habitability",
                "example_body": {
                    "pl_rade": 1.2,
                    "pl_eqt": 290
                }
            },
            {
                "path": "/rank",
                "method": "GET",
                "description": "Get ranked exoplanets"
            }
        ]
    })

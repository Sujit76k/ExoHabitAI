from flask import Blueprint, jsonify

docs_bp = Blueprint("docs", __name__)


@docs_bp.route("/docs", methods=["GET"])
def docs():
    """
    🚀 ExoHabitAI Professional API Documentation
    Lightweight JSON docs used by dashboard + developers.
    """

    return jsonify({
        "project": "🚀 ExoHabitAI — Exoplanet Habitability Prediction API",
        "version": "1.0.0",
        "description": "Scientific AI system for predicting exoplanet habitability using machine learning and astrophysics-based features.",

        "base_url": "http://127.0.0.1:5000",

        "endpoints": [

            # ===========================
            # HEALTH CHECK
            # ===========================
            {
                "name": "Health Check",
                "path": "/",
                "method": "GET",
                "description": "Verify API server is running.",
                "response_example": "ExoHabitAI API Running"
            },

            # ===========================
            # PREDICT
            # ===========================
            {
                "name": "Habitability Prediction",
                "path": "/predict",
                "method": "POST",
                "description": "Predict habitability using planetary + stellar parameters.",
                "body_schema": {
                    "pl_rade": "Planet radius (Earth = 1)",
                    "pl_eqt": "Equilibrium temperature (Kelvin)",
                    "pl_orbper": "Orbital period (days)",
                    "st_teff": "Star temperature (Kelvin)",
                    "st_mass": "Star mass (Sun = 1)",
                    "st_rad": "Star radius (Sun = 1)"
                },
                "example_request": {
                    "pl_rade": 1.2,
                    "pl_eqt": 290,
                    "pl_orbper": 365,
                    "st_teff": 5778,
                    "st_mass": 1.0,
                    "st_rad": 1.0
                },
                "example_response": {
                    "prediction": 1,
                    "habitability_score": 0.83
                }
            },

            # ===========================
            # RANKING
            # ===========================
            {
                "name": "Ranked Exoplanets",
                "path": "/rank",
                "method": "GET",
                "description": "Returns ranked exoplanets dataset with AI scores."
            },

            # ===========================
            # DATASET STATS
            # ===========================
            {
                "name": "Dataset Statistics",
                "path": "/stats",
                "method": "GET",
                "description": "Returns dataset analytics used by dashboard.",
                "response_fields": [
                    "total_planets",
                    "habitable_count",
                    "avg_score"
                ]
            },

            # ===========================
            # FEATURE IMPORTANCE
            # ===========================
            {
                "name": "Model Feature Importance",
                "path": "/importance",
                "method": "GET",
                "description": "Returns ML feature importance values from RandomForest model."
            },

            # ===========================
            # SWAGGER UI
            # ===========================
            {
                "name": "Swagger UI",
                "path": "/apidocs",
                "method": "GET",
                "description": "Interactive API documentation powered by Flasgger."
            }
        ],

        "scientific_notes": {
            "radius_range": "0.1 — 20 Earth radii",
            "temperature_range": "50 — 2000 Kelvin",
            "stellar_temp_range": "2000 — 10000 Kelvin"
        },

        "author": "ExoHabitAI AI Research System"
    })
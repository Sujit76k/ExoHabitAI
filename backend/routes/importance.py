from flask import Blueprint, jsonify
from backend.model_registry import get_model

importance_bp = Blueprint("importance", __name__)


@importance_bp.route("/importance", methods=["GET"])
def importance():
    """
    🚀 Returns model feature importance in production-ready format.
    Safe for pipelines and multiple model types.
    """

    try:
        model = get_model()

        if model is None:
            return jsonify({
                "status": "error",
                "message": "Model not loaded"
            }), 500

        # --------------------------------------------------
        # 🧠 Handle sklearn pipelines
        # --------------------------------------------------
        final_model = model
        if hasattr(model, "named_steps"):
            # last step usually the estimator
            final_model = list(model.named_steps.values())[-1]

        # --------------------------------------------------
        # 📊 Extract feature importance safely
        # --------------------------------------------------
        if not hasattr(final_model, "feature_importances_"):
            return jsonify({
                "status": "ok",
                "importance": [],
                "message": "Model does not support feature importance"
            })

        importances = final_model.feature_importances_

        # Try to get feature names from model
        feature_names = []

        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
        else:
            # fallback generic names
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        # --------------------------------------------------
        # 📈 Build structured response
        # --------------------------------------------------
        importance_data = [
            {
                "feature": name,
                "importance": float(score)
            }
            for name, score in zip(feature_names, importances)
        ]

        # sort descending (professional API behavior)
        importance_data = sorted(
            importance_data,
            key=lambda x: x["importance"],
            reverse=True
        )

        return jsonify({
            "status": "success",
            "count": len(importance_data),
            "importance": importance_data
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
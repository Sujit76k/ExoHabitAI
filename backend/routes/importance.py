from flask import Blueprint, jsonify
from backend.model_registry import get_model

importance_bp = Blueprint("importance", __name__)

@importance_bp.route("/importance")
def importance():
    model = get_model()
    return jsonify({"importance": model.feature_importances_.tolist()})

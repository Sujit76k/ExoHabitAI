from flask import Blueprint, jsonify
import pandas as pd
from backend.config import RANKED_DATA_PATH

rank_bp = Blueprint("rank", __name__)

@rank_bp.route("/rank", methods=["GET"])
def rank():
    df = pd.read_csv(RANKED_DATA_PATH)
    return jsonify(df.head(20).to_dict(orient="records"))

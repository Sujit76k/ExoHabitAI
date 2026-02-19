from flask import Blueprint, jsonify
import pandas as pd

rank_bp = Blueprint("rank", __name__)

@rank_bp.route("/rank", methods=["GET"])
def rank():
    df = pd.read_csv("../data/processed/ranked_exoplanets.csv")
    return jsonify(df.head(20).to_dict(orient="records"))

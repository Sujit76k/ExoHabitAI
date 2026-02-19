from flask import Blueprint, request, jsonify
import pandas as pd
from backend.model_loader import load_model

predict_bp = Blueprint("predict", __name__)

model = load_model()

@predict_bp.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])

    prediction = model.predict(df.select_dtypes("number"))[0]
    prob = model.predict_proba(df.select_dtypes("number"))[0][1]

    return jsonify({
        "prediction": int(prediction),
        "habitability_score": float(prob)
    })

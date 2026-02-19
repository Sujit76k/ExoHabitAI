from flask import Blueprint, jsonify, request
import pandas as pd
import os

from backend.config import RANKED_DATA_PATH

rank_bp = Blueprint("rank", __name__)


# =====================================================
# 🚀 PROFESSIONAL RANK ROUTE
# =====================================================

@rank_bp.route("/rank", methods=["GET"])
def rank():
    """
    🌍 Returns ranked exoplanets dataset.
    Supports:
    - limit
    - sort
    - order
    """

    try:
        # --------------------------------------------------
        # 📂 Validate file exists
        # --------------------------------------------------
        if not os.path.exists(RANKED_DATA_PATH):
            return jsonify({
                "status": "error",
                "message": "ranked_exoplanets.csv not found. Run Week4 pipeline."
            }), 500

        df = pd.read_csv(RANKED_DATA_PATH)

        # --------------------------------------------------
        # 🧭 Query Parameters (PRO FEATURE)
        # --------------------------------------------------
        limit = request.args.get("limit", default=20, type=int)
        sort_col = request.args.get("sort", default="habitability_score", type=str)
        order = request.args.get("order", default="desc", type=str)

        # Safety limit
        limit = max(1, min(limit, 200))

        # --------------------------------------------------
        # 📊 Sorting Logic
        # --------------------------------------------------
        if sort_col in df.columns:
            df = df.sort_values(
                by=sort_col,
                ascending=(order.lower() == "asc")
            )

        # --------------------------------------------------
        # ✂️ Slice result
        # --------------------------------------------------
        result_df = df.head(limit)

        # --------------------------------------------------
        # 📈 Metadata (VERY IMPORTANT FOR DASHBOARD)
        # --------------------------------------------------
        metadata = {
            "total_rows": int(len(df)),
            "returned_rows": int(len(result_df)),
        }

        if "habitability_score" in df.columns:
            metadata["avg_score"] = float(df["habitability_score"].mean())

        if "prediction" in df.columns:
            metadata["habitable_count"] = int((df["prediction"] == 1).sum())

        # --------------------------------------------------
        # 🚀 Response
        # --------------------------------------------------
        return jsonify({
            "status": "success",
            "metadata": metadata,
            "data": result_df.to_dict(orient="records")
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
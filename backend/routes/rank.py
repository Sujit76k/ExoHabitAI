from flask import Blueprint, jsonify, request

import pandas as pd
import numpy as np
import os

from backend.config import RANKED_DATA_PATH

rank_bp = Blueprint("rank", __name__)

# =====================================================
# 🚀 JSON SAFE SANITIZER (REAL FIX)
# =====================================================

def sanitize_dataframe_for_json(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert invalid numeric values into JSON-safe format.

    Fixes:
    - NaN
    - Infinity
    - -Infinity
    """

    # Replace infinities → NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Convert NaN → None (valid JSON)
    df = df.where(pd.notnull(df), None)

    return df


# =====================================================
# 🚀 PROFESSIONAL RANK ROUTE (UPDATED)
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

        # --------------------------------------------------
        # 🚀 PERFORMANCE BOOST (NO BREAKING CHANGE)
        # Load only columns needed by dashboard
        # --------------------------------------------------
        FRONTEND_COLUMNS = [
            "pl_name",
            "habitability_score",
            "prediction"
        ]

        df = pd.read_csv(
            RANKED_DATA_PATH,
            usecols=lambda c: c in FRONTEND_COLUMNS
        )

        # --------------------------------------------------
        # 🧭 Query Parameters
        # --------------------------------------------------
        limit = request.args.get("limit", default=20, type=int)
        sort_col = request.args.get("sort", default="habitability_score", type=str)
        order = request.args.get("order", default="desc", type=str)

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
        # 🔥 REAL JSON SAFE FIX
        # --------------------------------------------------
        result_df = sanitize_dataframe_for_json(result_df)

        # --------------------------------------------------
        # 📈 Metadata (FOR DASHBOARD)
        # --------------------------------------------------
        metadata = {
            "total_rows": int(len(df)),
            "returned_rows": int(len(result_df)),
        }

        if "habitability_score" in df.columns:
            clean_scores = df["habitability_score"].replace([np.inf, -np.inf], np.nan)
            metadata["avg_score"] = float(clean_scores.mean())

        if "prediction" in df.columns:
            metadata["habitable_count"] = int((df["prediction"] == 1).sum())

        # --------------------------------------------------
        # 🚀 FINAL RESPONSE
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
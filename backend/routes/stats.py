from flask import Blueprint, jsonify
import pandas as pd
import os

stats_bp = Blueprint("stats", __name__)

# =====================================================
# 🚀 ABSOLUTE SAFE PATH (PRODUCTION READY)
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

RANKED_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "ranked_exoplanets.csv"
)


# =====================================================
# ⭐ LEVEL-100 SCIENTIFIC ANALYTICS STATS ROUTE
# =====================================================

@stats_bp.route("/stats", methods=["GET"])
def stats():
    """
    🌌 ExoHabitAI Scientific Analytics Endpoint

    Returns:
    - dataset health
    - totals
    - averages
    - score distribution
    - min/max values
    """

    try:

        # --------------------------------------------------
        # 📂 Dataset Safety Check
        # --------------------------------------------------
        if not os.path.exists(RANKED_PATH):
            return jsonify({
                "status": "warning",
                "message": "Ranked dataset not found",
                "total_planets": 0,
                "habitable_count": 0,
                "avg_score": 0
            })

        df = pd.read_csv(RANKED_PATH)

        # --------------------------------------------------
        # 📊 BASIC METRICS
        # --------------------------------------------------
        total_planets = int(len(df))

        habitable_count = 0
        avg_score = 0
        min_score = 0
        max_score = 0

        if "prediction" in df.columns:
            habitable_count = int((df["prediction"] == 1).sum())

        if "habitability_score" in df.columns:
            avg_score = float(df["habitability_score"].mean())
            min_score = float(df["habitability_score"].min())
            max_score = float(df["habitability_score"].max())

        # --------------------------------------------------
        # 📈 DISTRIBUTION DATA (FOR CHARTS)
        # --------------------------------------------------
        distribution = {}

        if "habitability_score" in df.columns:
            # Create histogram bins for frontend graphs
            bins = [0, 0.25, 0.5, 0.75, 1.0]
            labels = ["Very Low", "Low", "Medium", "High"]

            df["score_band"] = pd.cut(
                df["habitability_score"],
                bins=bins,
                labels=labels,
                include_lowest=True
            )

            distribution = (
                df["score_band"]
                .value_counts()
                .sort_index()
                .to_dict()
            )

        # --------------------------------------------------
        # 🌍 OPTIONAL EXTRA SCIENTIFIC INSIGHTS
        # --------------------------------------------------
        feature_means = {}

        important_features = [
            "pl_rade",
            "pl_eqt",
            "st_teff",
            "st_mass",
            "st_rad"
        ]

        for col in important_features:
            if col in df.columns:
                feature_means[col] = float(df[col].mean())

        # --------------------------------------------------
        # 🚀 FINAL RESPONSE (DASHBOARD READY)
        # --------------------------------------------------
        return jsonify({
            "status": "success",
            "dataset_health": "ok",
            "total_planets": total_planets,
            "habitable_count": habitable_count,
            "avg_score": avg_score,
            "min_score": min_score,
            "max_score": max_score,
            "distribution": distribution,
            "feature_means": feature_means
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Stats calculation failed",
            "details": str(e)
        }), 500
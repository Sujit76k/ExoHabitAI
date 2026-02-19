# ======================================================
# 🚀 ExoHabitAI — Scientific Analytics Dashboard (Streamlit)
# Production-ready version
# ======================================================

import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ======================================================
# 📁 PATH SETUP (SAFE PRODUCTION STYLE)
# ======================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "ranked_exoplanets.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "week4_best_model.pkl")

# ======================================================
# ⚡ FAST DATA LOADER
# ======================================================

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


# ======================================================
# 🚀 PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="ExoHabitAI Dashboard",
    layout="wide",
)

st.title("🚀 ExoHabitAI — Scientific Analytics Dashboard")

# ======================================================
# 📊 LOAD DATA
# ======================================================

df = load_data()

if df.empty:
    st.error("❌ Ranked dataset not found. Run ML pipeline first.")
    st.stop()

# ======================================================
# ⭐ METRICS PANEL
# ======================================================

total_planets = len(df)

habitable_count = 0
avg_score = 0

if "prediction" in df.columns:
    habitable_count = int((df["prediction"] == 1).sum())

if "habitability_score" in df.columns:
    avg_score = float(df["habitability_score"].mean())

col1, col2, col3 = st.columns(3)

col1.metric("🪐 Total Planets", total_planets)
col2.metric("🌍 Habitable Planets", habitable_count)
col3.metric("⭐ Avg Habitability Score", f"{avg_score:.2f}")

st.divider()

# ======================================================
# 🔎 INTERACTIVE FILTER
# ======================================================

st.subheader("🔎 Dataset Explorer")

score_filter = st.slider(
    "Minimum Habitability Score",
    0.0,
    1.0,
    0.0,
    0.05,
)

filtered_df = df.copy()

if "habitability_score" in filtered_df.columns:
    filtered_df = filtered_df[
        filtered_df["habitability_score"] >= score_filter
    ]

st.dataframe(filtered_df.head(50), use_container_width=True)

st.divider()

# ======================================================
# 📈 HABITABILITY DISTRIBUTION
# ======================================================

st.subheader("📈 Habitability Score Distribution")

if "habitability_score" in df.columns:

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(
        df["habitability_score"],
        bins=30,
    )

    ax.set_xlabel("Habitability Score")
    ax.set_ylabel("Planet Count")

    st.pyplot(fig)

else:
    st.warning("Habitability score column not found.")

# ======================================================
# 🌌 PREDICTION BALANCE
# ======================================================

st.subheader("🌌 Prediction Balance")

if "prediction" in df.columns:

    fig2, ax2 = plt.subplots()

    counts = df["prediction"].value_counts()

    ax2.bar(counts.index.astype(str), counts.values)

    ax2.set_xlabel("Prediction")
    ax2.set_ylabel("Count")

    st.pyplot(fig2)

# ======================================================
# 🧠 FEATURE IMPORTANCE (IF MODEL SUPPORTS)
# ======================================================

st.subheader("🧠 Feature Importance")

model = load_model()

if model is None:
    st.info("Model not found — feature importance unavailable.")
else:
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_

            fig3, ax3 = plt.subplots()

            ax3.bar(range(len(importances)), importances)

            ax3.set_title("Model Feature Importance")

            st.pyplot(fig3)

        elif hasattr(model, "named_steps"):
            final_model = model.named_steps.get("model", None)

            if hasattr(final_model, "feature_importances_"):
                importances = final_model.feature_importances_

                fig4, ax4 = plt.subplots()

                ax4.bar(range(len(importances)), importances)

                st.pyplot(fig4)
            else:
                st.info("Feature importance not supported by this model.")

        else:
            st.info("Feature importance not available.")

    except Exception as e:
        st.warning(f"Feature importance error: {str(e)}")

# ======================================================
# 📊 RAW DATA DOWNLOAD
# ======================================================

st.subheader("⬇️ Export Data")

st.download_button(
    label="Download Ranked Dataset",
    data=df.to_csv(index=False),
    file_name="ranked_exoplanets.csv",
    mime="text/csv",
)

st.success("✅ ExoHabitAI Scientific Dashboard Ready")
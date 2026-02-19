"""
=====================================================
🚀 ExoHabitAI — Production Training Pipeline
Milestone-Ready ML Training Script
=====================================================
"""

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.config import (
    PROCESSED_DATA_PATH,
    MODEL_PATH,
    METRICS_PATH,
    TEST_SIZE,
    RANDOM_STATE,
)

from src.data_loader import load_raw_data
from src.preprocessing import basic_cleaning, select_numeric_features
from src.target_builder import build_habitability_target
from src.pipeline import build_baseline_pipeline
from src.evaluate_model import evaluate_classification_model
from src.utils import ensure_dir_exists, save_json
from src.eda import save_eda_summary


# -----------------------------------------------------
# REPORT PATHS
# -----------------------------------------------------

REPORT_DIR = "reports"
REPORT_EDA_PATH = os.path.join(REPORT_DIR, "eda_summary.txt")
REPORT_MODEL_PATH = os.path.join(REPORT_DIR, "model_report.txt")
FEATURE_SCHEMA_PATH = os.path.join(REPORT_DIR, "feature_schema.json")


# -----------------------------------------------------
# TRAINING PIPELINE
# -----------------------------------------------------

def main():

    print("\n🚀 Starting ExoHabitAI Training Pipeline\n")

    # -------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------

    print("✅ Loading raw dataset...")
    df = load_raw_data()

    if df.empty:
        raise RuntimeError("❌ Dataset loaded but empty.")

    # -------------------------------------------------
    # BASIC CLEANING
    # -------------------------------------------------

    print("✅ Cleaning dataset...")
    df = basic_cleaning(df)

    # -------------------------------------------------
    # BUILD TARGET
    # -------------------------------------------------

    print("✅ Creating habitability target...")
    df = build_habitability_target(df)

    # -------------------------------------------------
    # SAVE EDA REPORT
    # -------------------------------------------------

    ensure_dir_exists(REPORT_DIR)
    save_eda_summary(df, REPORT_EDA_PATH)
    print(f"📊 EDA report saved → {REPORT_EDA_PATH}")

    # -------------------------------------------------
    # SAVE PROCESSED DATASET
    # -------------------------------------------------

    ensure_dir_exists(os.path.dirname(PROCESSED_DATA_PATH))
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"💾 Processed dataset saved → {PROCESSED_DATA_PATH}")

    # -------------------------------------------------
    # FEATURE SELECTION
    # -------------------------------------------------

    print("✅ Selecting numeric features...")
    X, y = select_numeric_features(df, target_col="habitability")

    print("📐 Feature Matrix Shape:", X.shape)
    print("🎯 Target Distribution:\n", y.value_counts())

    # Prevent training failure
    if len(y.unique()) < 2:
        raise RuntimeError(
            "❌ Only one target class detected.\n"
            "Adjust thresholds inside target_builder.py"
        )

    # -------------------------------------------------
    # SAVE FEATURE SCHEMA (VERY IMPORTANT FOR API)
    # -------------------------------------------------

    save_json({"features": list(X.columns)}, FEATURE_SCHEMA_PATH)
    print(f"🧠 Feature schema saved → {FEATURE_SCHEMA_PATH}")

    # -------------------------------------------------
    # TRAIN TEST SPLIT
    # -------------------------------------------------

    print("✅ Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # -------------------------------------------------
    # BUILD MODEL
    # -------------------------------------------------

    print("🤖 Building ML pipeline...")
    model = build_baseline_pipeline()

    # -------------------------------------------------
    # TRAIN MODEL
    # -------------------------------------------------

    print("⚡ Training model...")
    model.fit(X_train, y_train)

    # -------------------------------------------------
    # EVALUATE MODEL
    # -------------------------------------------------

    print("📊 Evaluating model...")
    metrics = evaluate_classification_model(model, X_test, y_test)

    save_json(metrics, METRICS_PATH)
    print(f"📈 Metrics saved → {METRICS_PATH}")

    # -------------------------------------------------
    # SAVE MODEL
    # -------------------------------------------------

    ensure_dir_exists(os.path.dirname(MODEL_PATH))
    joblib.dump(model, MODEL_PATH)
    print(f"💾 Model saved → {MODEL_PATH}")

    # -------------------------------------------------
    # SAVE TEXT REPORT
    # -------------------------------------------------

    report_text = classification_report(
        y_test,
        model.predict(X_test),
        zero_division=0
    )

    with open(REPORT_MODEL_PATH, "w", encoding="utf-8") as f:
        f.write("===== MODEL CLASSIFICATION REPORT =====\n")
        f.write(report_text)

    print(f"🧾 Model report saved → {REPORT_MODEL_PATH}")

    print("\n🎉 TRAINING COMPLETE — MODEL READY FOR API\n")


# -----------------------------------------------------
# ENTRYPOINT
# -----------------------------------------------------

if __name__ == "__main__":
    main()
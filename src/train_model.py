import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.config import (
    PROCESSED_DATA_PATH,
    MODEL_PATH,
    METRICS_PATH,
    TEST_SIZE,
    RANDOM_STATE
)
from src.data_loader import load_raw_data
from src.preprocessing import basic_cleaning, select_numeric_features
from src.target_builder import build_habitability_target
from src.pipeline import build_baseline_pipeline
from src.evaluate_model import evaluate_classification_model
from src.utils import ensure_dir_exists, save_json
from src.eda import save_eda_summary

REPORT_EDA_PATH = os.path.join("reports", "eda_summary.txt")
REPORT_MODEL_PATH = os.path.join("reports", "model_report.txt")


def main():
    print("✅ Loading raw dataset...")
    df = load_raw_data()

    print("✅ Basic cleaning...")
    df = basic_cleaning(df)

    print("✅ Creating target label (habitability)...")
    df = build_habitability_target(df)

    # Save EDA report
    save_eda_summary(df, REPORT_EDA_PATH)
    print(f"✅ EDA report saved: {REPORT_EDA_PATH}")

    # Save processed dataset
    ensure_dir_exists(os.path.dirname(PROCESSED_DATA_PATH))
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"✅ Processed dataset saved: {PROCESSED_DATA_PATH}")

    print("✅ Selecting features...")
    X, y = select_numeric_features(df, target_col="habitability")
    print("✅ X shape:", X.shape)
    print("✅ Target distribution:\n", y.value_counts())


    if len(y.unique()) < 2:
        print("\n⚠️ WARNING: Target contains only 1 class (all 0 or all 1).")
        print("This means your habitability rule is too strict or dataset lacks required columns.")
        print("Try widening thresholds in target_builder.py")

    print("✅ Train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if len(y.unique()) > 1 else None
    )

    print("✅ Building baseline pipeline...")
    model = build_baseline_pipeline()

    print("✅ Training model...")
    model.fit(X_train, y_train)

    print("✅ Evaluating model...")
    metrics = evaluate_classification_model(model, X_test, y_test)

    # Save metrics
    save_json(metrics, METRICS_PATH)
    print(f"✅ Metrics saved: {METRICS_PATH}")

    # Save model
    ensure_dir_exists(os.path.dirname(MODEL_PATH))
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved: {MODEL_PATH}")

    # Save text report
    ensure_dir_exists("reports")
    report_text = classification_report(y_test, model.predict(X_test), zero_division=0)
    with open(REPORT_MODEL_PATH, "w", encoding="utf-8") as f:
        f.write("===== MODEL CLASSIFICATION REPORT =====\n")
        f.write(report_text)

    print(f"✅ Model report saved: {REPORT_MODEL_PATH}")

    print("\n🎉 Milestone-1 baseline pipeline complete!")


if __name__ == "__main__":
    main()

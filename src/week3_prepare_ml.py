"""
=====================================================
🚀 ExoHabitAI — WEEK 3 ML DATASET PREPARATION
Production-Level ML Pipeline
=====================================================
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from src.utils import ensure_dir_exists, log


ENGINEERED_PATH = os.path.join("data", "processed", "feature_engineered_exoplanets.csv")
MODEL_PATH = os.path.join("models", "week3_pipeline_model.pkl")
REPORT_PATH = os.path.join("reports", "week3_model_report.txt")
FIG_DIR = os.path.join("reports", "figures")


# ======================================================
# 📊 CONFUSION MATRIX PLOT
# ======================================================

def plot_confusion_matrix(cm, save_path: str):

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, aspect="auto")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["Not Habitable", "Habitable"])
    plt.yticks([0, 1], ["Not Habitable", "Habitable"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ======================================================
# ⭐ FEATURE SELECTION (CORRELATION BASED)
# ======================================================

def correlation_feature_selection(df: pd.DataFrame, target_col: str, top_n: int = 25):

    numeric_df = df.select_dtypes(include=["number"]).copy()

    if target_col not in numeric_df.columns:
        log("Target not numeric — skipping correlation selection", "WARNING")
        return []

    corrs = numeric_df.corr()[target_col].abs().sort_values(ascending=False)
    corrs = corrs.drop(target_col, errors="ignore")

    return corrs.head(top_n).index.tolist()


# ======================================================
# 🚀 MAIN PIPELINE
# ======================================================

def main():

    log("WEEK 3 ML DATASET PREPARATION STARTED")

    ensure_dir_exists("models")
    ensure_dir_exists("reports")
    ensure_dir_exists(FIG_DIR)

    df = pd.read_csv(ENGINEERED_PATH)

    target_col = "habitability"

    if target_col not in df.columns:
        raise ValueError("❌ Target column 'habitability' not found. Run Week 2 first!")

    # ==================================================
    # FEATURE SELECTION
    # ==================================================

    log("Selecting important features using correlation...")

    top_features = correlation_feature_selection(
        df,
        target_col=target_col,
        top_n=25
    )

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    selected_cols = list(set(top_features + categorical_cols))

    if len(selected_cols) == 0:
        raise ValueError("❌ No usable features selected.")

    X = df[selected_cols]
    y = df[target_col]

    # ==================================================
    # TRAIN TEST SPLIT
    # ==================================================

    log("Splitting dataset into train/test (80:20)...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    # ==================================================
    # PREPROCESSING PIPELINES
    # ==================================================

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    log(f"Numeric columns: {len(numeric_cols)}")
    log(f"Categorical columns: {len(categorical_cols)}")

    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipeline, numeric_cols))

    if categorical_cols:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    # ==================================================
    # MODEL
    # ==================================================

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
        max_depth=None,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # ==================================================
    # TRAIN
    # ==================================================

    log("Training ML pipeline (RandomForest)...")
    pipeline.fit(X_train, y_train)

    # ==================================================
    # EVALUATION
    # ==================================================

    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    log(f"Accuracy: {acc:.4f}")

    # ==================================================
    # SAVE MODEL
    # ==================================================

    joblib.dump(pipeline, MODEL_PATH)
    log(f"Model pipeline saved → {MODEL_PATH}")

    # ==================================================
    # SAVE REPORT
    # ==================================================

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("===== WEEK 3 MODEL REPORT =====\n")
        f.write(f"Accuracy: {acc}\n\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    log(f"Report saved → {REPORT_PATH}")

    plot_confusion_matrix(
        cm,
        os.path.join(FIG_DIR, "confusion_matrix_week3.png")
    )

    log("Confusion matrix saved.")
    log("WEEK 3 ML DATASET PREPARATION COMPLETED", "SUCCESS")


if __name__ == "__main__":
    main()
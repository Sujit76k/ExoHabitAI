"""
=====================================================
🚀 ExoHabitAI — WEEK 4 MODEL COMPARISON ENGINE
Production-Level Auto Model Selection
=====================================================
"""

import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src.utils import ensure_dir_exists, log


DATA_PATH = "data/processed/feature_engineered_exoplanets.csv"
MODEL_PATH = "models/week4_best_model.pkl"
RANKED_PATH = "data/processed/ranked_exoplanets.csv"


# ======================================================
# LOAD DATASET
# ======================================================

log("Loading feature engineered dataset...")

df = pd.read_csv(DATA_PATH)

target = "habitability"

if target not in df.columns:
    raise ValueError("❌ Target column 'habitability' missing.")

# Keep numeric only
X = df.select_dtypes(include="number").drop(columns=[target], errors="ignore")
y = df[target]

log(f"Dataset shape: {df.shape}")
log(f"Feature count: {X.shape[1]}")


# ======================================================
# TRAIN TEST SPLIT
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y if y.nunique() > 1 else None,
)

# ======================================================
# MODELS TO COMPARE
# ======================================================

models = {

    "LogisticRegression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LogisticRegression(max_iter=3000, class_weight="balanced"))
    ]),

    "RandomForest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ))
    ]),

    "GradientBoosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingClassifier())
    ])
}


# ======================================================
# TRAIN & EVALUATE MODELS
# ======================================================

results = []

for name, pipeline in models.items():

    log(f"Training {name}...")

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)

    print("\n==============================")
    print(f"MODEL: {name}")
    print("==============================")
    print(classification_report(y_test, y_pred, zero_division=0))

    results.append((name, auc, pipeline))


# ======================================================
# SELECT BEST MODEL AUTOMATICALLY
# ======================================================

results_sorted = sorted(results, key=lambda x: x[1], reverse=True)

best_name, best_auc, best_model = results_sorted[0]

log(f"Best model selected → {best_name} (ROC-AUC={best_auc:.4f})")

ensure_dir_exists("models")
joblib.dump(best_model, MODEL_PATH)

log(f"Best model saved → {MODEL_PATH}")


# ======================================================
# CREATE RANKED PLANETS FILE
# ======================================================

log("Creating ranked planets dataset...")

df_rank = df.copy()

df_rank["habitability_score"] = best_model.predict_proba(X)[:, 1]

# Prediction column
df_rank["prediction"] = (df_rank["habitability_score"] >= 0.5).astype(int)

df_rank = df_rank.sort_values("habitability_score", ascending=False)

ensure_dir_exists("data/processed")
df_rank.to_csv(RANKED_PATH, index=False)

log(f"Ranked dataset saved → {RANKED_PATH}")

log("🎉 WEEK 4 COMPLETE — MODEL + RANKING READY", "SUCCESS")
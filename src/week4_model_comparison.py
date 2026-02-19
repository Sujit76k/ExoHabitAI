import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


DATA_PATH = "data/processed/feature_engineered_exoplanets.csv"

print("✅ Loading dataset...")
df = pd.read_csv(DATA_PATH)

target = "habitability"

# Keep numeric only for now
X = df.select_dtypes(include="number").drop(columns=[target])
y = df[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ⭐ Add Imputer to handle NaN automatically
models = {
    "LogisticRegression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LogisticRegression(max_iter=2000))
    ]),
    "RandomForest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42
        ))
    ])
}

results = []

for name, pipeline in models.items():
    print(f"\n🚀 Training {name}...")

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:,1]

    auc = roc_auc_score(y_test, y_prob)

    print(classification_report(y_test, y_pred))

    results.append((name, auc, pipeline))

# ⭐ Select best model automatically
best_model = sorted(results, key=lambda x: x[1], reverse=True)[0][2]

print("\n✅ Saving best model...")
joblib.dump(best_model, "models/week4_best_model.pkl")

# ⭐ Create Ranking File
print("✅ Creating ranked planets file...")

df["habitability_score"] = best_model.predict_proba(X)[:,1]
df = df.sort_values("habitability_score", ascending=False)

df.to_csv("data/processed/ranked_exoplanets.csv", index=False)

print("🎉 WEEK 4 COMPLETE — MODEL + RANKING READY")

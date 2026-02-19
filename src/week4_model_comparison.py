import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

DATA_PATH = "data/processed/feature_engineered_exoplanets.csv"

df = pd.read_csv(DATA_PATH)

target = "habitability"
X = df.select_dtypes(include="number").drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced")
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    auc = roc_auc_score(y_test, y_prob)
    print(classification_report(y_test, y_pred))

    results.append((name, auc, model))

# Select best model
best_model = sorted(results, key=lambda x: x[1], reverse=True)[0][2]

joblib.dump(best_model, "models/week4_best_model.pkl")

# ⭐ Planet Ranking
df["habitability_score"] = best_model.predict_proba(X)[:,1]
df = df.sort_values("habitability_score", ascending=False)

df.to_csv("data/processed/ranked_exoplanets.csv", index=False)

print("✅ Week 4 Complete: Model + Ranking Saved")

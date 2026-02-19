# ▶️ HOW TO RUN ExoHabitAI PROJECT

Follow these commands step-by-step to run the full AI system.

---

## ✅ 1️⃣ Clone Project
Clone repository and open project folder.

git clone <YOUR_REPO_LINK>
cd ExoHabitAI

---

## ✅ 2️⃣ Create Virtual Environment
Create isolated Python environment.

python -m venv venv

Activate environment:

Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

---

## ✅ 3️⃣ Install Requirements
Install all backend and ML dependencies.

pip install -r requirements.txt

---

## ✅ 4️⃣ Run Data Pipeline
Generate cleaned dataset, engineered features, trained model and ranking file.

python -m src.week2_cleaning
python -m src.week2_feature_engineering
python -m src.week3_ml_pipeline
python -m src.week4_model_comparison

---

## ✅ 5️⃣ Start Backend API
Run Flask AI server.

python -m backend.app

API runs at:
http://127.0.0.1:5000

Swagger Docs:
http://127.0.0.1:5000/apidocs

---

## ✅ 6️⃣ Open Frontend Dashboard
Open AI interface in browser.

frontend/index.html

OR use Live Server:
http://127.0.0.1:5500/frontend/index.html

---

## ✅ 7️⃣ Test Prediction
Enter planetary inputs → Click **Predict Habitability** → View AI result.

---

# ⚠️ Quick Fixes

Run backend ONLY using:
python -m backend.app

If model missing:
python -m src.week4_model_comparison

---

# 🎉 DONE
Backend + ML + Dashboard now fully running.

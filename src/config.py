import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "PS_2026.01.19_01.24.31.csv")


PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_exoplanets.csv")

MODEL_PATH = os.path.join(BASE_DIR, "models", "baseline_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")

RANDOM_STATE = 42
TEST_SIZE = 0.2

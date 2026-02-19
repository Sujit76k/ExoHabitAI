import pandas as pd
from backend.config import RANKED_DATA_PATH

def get_ranked_planets():
    df = pd.read_csv(RANKED_DATA_PATH)
    return df.head(20).to_dict(orient="records")

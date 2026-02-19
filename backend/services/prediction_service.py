import pandas as pd
from backend.model_registry import get_model

def predict_planet(data):
    model = get_model()
    df = pd.DataFrame([data])

    pred = model.predict(df.select_dtypes("number"))[0]
    prob = model.predict_proba(df.select_dtypes("number"))[0][1]

    return int(pred), float(prob)

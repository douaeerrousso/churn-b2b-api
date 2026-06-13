from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

model = joblib.load("tabnet_churn_model.pkl")

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    X = df.astype(np.float32).values
    X = np.nan_to_num(X)

    pred = model.predict(X)[0]

    return {"churn_prediction": int(pred)}

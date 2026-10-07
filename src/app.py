"""FastAPI service that serves predictions from model/model.joblib.

Run:  uvicorn src.app:app --reload
"""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = Path("model") / "model.joblib"
APP_VERSION = "1.1.0"
app = FastAPI(title="Diabetes Progression API", version=APP_VERSION)
model = joblib.load(MODEL_PATH)


class Patient(BaseModel):
    """Input schema. FastAPI rejects bad input automatically (HTTP 422)."""
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


@app.get("/health")
def health():
    return {"status": "ok", "version": APP_VERSION}


@app.post("/predict")
def predict(patient: Patient):
    row = pd.DataFrame([patient.model_dump()])
    prediction = model.predict(row)[0]
    return {"prediction": round(float(prediction), 2)}

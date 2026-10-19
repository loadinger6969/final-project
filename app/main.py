"""FastAPI service: predict if a red wine is "good" from its lab measurements.

Run locally: uvicorn app.main:app --reload
Then open http://127.0.0.1:8000/docs for the interactive page.
"""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "ml" / "registry" / "model.pkl"

_bundle = joblib.load(MODEL_PATH)
_model = _bundle["model"]
FEATURES = _bundle["features"]  # column order the model expects

app = FastAPI(title="Wine Quality Predictor", version="1.0")


class WineInput(BaseModel):
    # the JSON keys keep the original dataset names (with spaces)
    model_config = ConfigDict(populate_by_name=True)

    fixed_acidity: float = Field(alias="fixed acidity", ge=4.0, le=16.0)
    volatile_acidity: float = Field(alias="volatile acidity", ge=0.0, le=2.0)
    citric_acid: float = Field(alias="citric acid", ge=0.0, le=1.0)
    residual_sugar: float = Field(alias="residual sugar", ge=0.0, le=16.0)
    chlorides: float = Field(ge=0.0, le=1.0)
    free_sulfur_dioxide: float = Field(alias="free sulfur dioxide", ge=0, le=100)
    total_sulfur_dioxide: float = Field(alias="total sulfur dioxide", ge=0, le=300)
    density: float = Field(ge=0.9, le=1.1)
    pH: float = Field(ge=2.5, le=4.0)
    sulphates: float = Field(ge=0.2, le=2.0)
    alcohol: float = Field(ge=8.0, le=15.0)


@app.get("/")
def root():
    return {"status": "ok", "task": "predict good red wine", "model": "random forest"}


@app.get("/sample")
def sample():
    # a valid input you can copy into /predict
    return {
        "fixed acidity": 7.4,
        "volatile acidity": 0.7,
        "citric acid": 0.0,
        "residual sugar": 1.9,
        "chlorides": 0.076,
        "free sulfur dioxide": 11.0,
        "total sulfur dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
    }


@app.post("/predict")
def predict(wine: WineInput):
    row = wine.model_dump(by_alias=True)
    # the derived feature added in dataset v2
    row["alcohol_density_ratio"] = row["alcohol"] / row["density"]
    X = pd.DataFrame([row])[FEATURES]

    proba = _model.predict_proba(X)[0]
    label = int(_model.predict(X)[0])
    return {
        "prediction": "good" if label == 1 else "not good",
        "good_probability": round(float(proba[1]), 3),
    }

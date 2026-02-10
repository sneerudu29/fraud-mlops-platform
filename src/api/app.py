from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import joblib
import json
import numpy as np


app = FastAPI(title="Fraud MLOps Platform")

# ---- Load model + metadata ONCE when the API starts ----
MODEL_PATH = "artifacts/model/model.joblib"
META_PATH = "artifacts/model/metadata.json"

try:
    model = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        metadata = json.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load model or metadata: {e}")

threshold = float(metadata["threshold"])
feature_columns = metadata["feature_columns"]


# ---- Request schema (what the API expects) ----

class PredictRequest(BaseModel):
    features: Dict[str, float] # must be same length as feature_columns

# ---- Response schema (optional, but nice) ----
class PredictResponse(BaseModel):
    fraud_probability: float
    prediction: int

class BatchPredictRequest(BaseModel):
    records: List[Dict[str, float]]


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/schema")
def schema():
    return {
        "model": "LogisticRegression",
        "threshold": threshold,
        "num_features": len(feature_columns),
        "feature_columns": feature_columns
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    missing = [c for c in feature_columns if c not in req.features]
    extra = [k for k in req.features.keys() if k not in feature_columns]

    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing[:5]} ... total={len(missing)}")
    if extra:
        raise HTTPException(status_code=400, detail=f"Unexpected features: {extra[:5]} ... total={len(extra)}")

    # Create a 1-row DataFrame with correct column order (fixes sklearn warning too)
    import pandas as pd
    X = pd.DataFrame([[req.features[c] for c in feature_columns]], columns=feature_columns)


    prob = float(model.predict_proba(X)[0, 1])
    pred = int(prob >= threshold)

    return PredictResponse(fraud_probability=prob, prediction=pred)

@app.post("/predict_batch")
def predict_batch(req: BatchPredictRequest):
    rows = []

    for record in req.records:
        rows.append([record[c] for c in feature_columns])

    import pandas as pd
    X = pd.DataFrame(rows, columns=feature_columns)

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    return {
        "count": len(probs),
        "results": [
            {"fraud_probability": float(p), "prediction": int(d)}
            for p, d in zip(probs, preds)
        ]
    }

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
    features: list[float]  # must be same length as feature_columns


# ---- Response schema (optional, but nice) ----
class PredictResponse(BaseModel):
    fraud_probability: float
    prediction: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Validate input length
    if len(req.features) != len(feature_columns):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(feature_columns)} features, got {len(req.features)}"
        )

    # Convert list -> 2D array for sklearn (1 row, N columns)
    X = np.array(req.features, dtype=float).reshape(1, -1)

    # Predict fraud probability
    prob = float(model.predict_proba(X)[0, 1])

    # Apply threshold
    pred = int(prob >= threshold)

    return PredictResponse(fraud_probability=prob, prediction=pred)

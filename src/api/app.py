from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import joblib
import json
import hashlib
import time
from src.api.observability import record_request, log_prediction, prometheus_metrics, uptime_seconds



app = FastAPI(title="Fraud MLOps Platform")

def compute_schema_hash(feature_columns: list[str]) -> str:
    s = "|".join(feature_columns).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


# ---- Load model + metadata ONCE when the API starts ----
MODEL_PATH = "artifacts/model/model.joblib"
META_PATH = "artifacts/model/metadata.json"

try:
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load model or metadata: {e}")


threshold = float(metadata["threshold"])
feature_columns = metadata["feature_columns"]
# --- Fail fast if metadata is inconsistent ---
if int(metadata.get("num_features", len(feature_columns))) != len(feature_columns):
    raise RuntimeError("Metadata num_features does not match feature_columns length")

expected_hash = metadata.get("schema_hash")
actual_hash = compute_schema_hash(feature_columns)

if expected_hash and expected_hash != actual_hash:
    raise RuntimeError("Schema hash mismatch: metadata schema_hash != computed schema hash")



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
    return {
        "status": "ok",
        "model_version": metadata.get("model_version"),
        "schema_hash": metadata.get("schema_hash"),
        "uptime_seconds": uptime_seconds(),
    }


@app.get("/schema")
def schema():
    return {
        "model_version": metadata.get("model_version"),
        "trained_at": metadata.get("trained_at"),
        "git_commit": metadata.get("git_commit"),
        "threshold": threshold,
        "num_features": len(feature_columns),
        "schema_hash": metadata.get("schema_hash"),
        "feature_columns": feature_columns,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.time()
    is_error = False
    try:
        missing = [c for c in feature_columns if c not in req.features]
        extra = [k for k in req.features.keys() if k not in feature_columns]

        if missing:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing[:5]} ... total={len(missing)}")
        if extra:
            raise HTTPException(status_code=400, detail=f"Unexpected features: {extra[:5]} ... total={len(extra)}")

        import pandas as pd
        X = pd.DataFrame([[req.features[c] for c in feature_columns]], columns=feature_columns)

        prob = float(model.predict_proba(X)[0, 1])
        pred = int(prob >= threshold)

        # audit log
        log_prediction({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_version": metadata.get("model_version"),
            "schema_hash": metadata.get("schema_hash"),
            "fraud_probability": prob,
            "prediction": pred
        })

        return PredictResponse(fraud_probability=prob, prediction=pred)

    except HTTPException:
        is_error = True
        raise
    finally:
        latency_ms = (time.time() - start) * 1000
        # pred might not exist if error
        record_request(latency_ms=latency_ms, is_error=is_error, is_fraud=(locals().get("pred", 0) == 1))


@app.post("/predict_batch")
def predict_batch(req: BatchPredictRequest):
    rows = []
    for i, record in enumerate(req.records):
        missing = [c for c in feature_columns if c not in record]
        extra = [k for k in record.keys() if k not in feature_columns]

        if missing:
            raise HTTPException(status_code=400, detail=f"Record {i}: missing features {missing[:5]}... total={len(missing)}")
        if extra:
            raise HTTPException(status_code=400, detail=f"Record {i}: unexpected features {extra[:5]}... total={len(extra)}")

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
    
@app.get("/metrics")
def metrics():
    return prometheus_metrics()


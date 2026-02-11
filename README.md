# Fraud MLOps Platform

Step-by-step build from beginner ML baseline → API serving → MLflow tracking → drift monitoring → retraining trigger.

## Current status
- [ ] Baseline model training + evaluation (PR-AUC)
- [ ] Save model artifact
- [ ] FastAPI inference service
- [ ] MLflow tracking
- [ ] Drift detection (Evidently)
- [ ] CI/CD (GitHub Actions)

## Run with Docker

### Build
docker build -t fraud-mlops-api:latest .

### Run (mount trained model artifacts)
docker run --rm -p 8000:8000 ^
  -v "%cd%/artifacts/model:/app/artifacts/model" ^
  fraud-mlops-api:latest

Open:
- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/schema


# Fraud MLOps Platform

Production-style end-to-end ML system for fraud detection.

This project demonstrates the full ML lifecycle:

Model Training → API Serving → Drift Monitoring → Automated Retraining → CI/CD

Designed to simulate real-world MLOps responsibilities in financial systems.

---

## 🔍 Problem

Fraud detection models degrade over time due to:

- **Data Drift** – Feature distributions shift
- **Concept Drift** – Fraud strategies evolve
- **Prior Probability Shift** – Fraud rate changes
- **Label Lag** – Production labels arrive late

This platform detects drift using PSI and automatically triggers retraining via GitHub Actions.

---

## 🧠 Model

- Algorithm: Logistic Regression
- Metric: **PR-AUC** (better than accuracy for imbalanced data)
- Threshold-based fraud classification
- Model artifact saved with:
  - `model_version`
  - `trained_at`
  - `git_commit`
  - `schema_hash`
  - `num_features`

---

## 🏗 Architecture

T# Fraud MLOps Platform

Production-style end-to-end ML system for fraud detection.

This project demonstrates the full ML lifecycle:

Model Training → API Serving → Drift Monitoring → Automated Retraining → CI/CD

Designed to simulate real-world MLOps responsibilities in financial systems.

---

## 🔍 Problem

Fraud detection models degrade over time due to:

- **Data Drift** – Feature distributions shift
- **Concept Drift** – Fraud strategies evolve
- **Prior Probability Shift** – Fraud rate changes
- **Label Lag** – Production labels arrive late

This platform detects drift using PSI and automatically triggers retraining via GitHub Actions.

---

## 🧠 Model

- Algorithm: Logistic Regression
- Metric: **PR-AUC** (better than accuracy for imbalanced data)
- Threshold-based fraud classification
- Model artifact saved with:
  - `model_version`
  - `trained_at`
  - `git_commit`
  - `schema_hash`
  - `num_features`

---

## 🏗 Architecture

Training → Model Artifact → FastAPI Service
↓
Drift Monitoring (PSI)
↓
Retrain Trigger (CI)
↓
Updated Model + Drift Report


---

## 🚀 Features Implemented

✔ Baseline model training + PR-AUC evaluation  
✔ Model artifact saving (joblib + metadata)  
✔ Fail-fast schema validation using schema hash  
✔ FastAPI inference service (`/predict`)  
✔ Batch prediction endpoint (`/predict_batch`)  
✔ Schema discovery endpoint (`/schema`)  
✔ Drift detection (Feature PSI + Prediction PSI)  
✔ Automated retraining trigger  
✔ GitHub Actions CI pipeline  
✔ Dockerized API service  

---

## 📊 API Endpoints

### `GET /health`
Health check

### `GET /schema`
Returns:
- model_version
- trained_at
- git_commit
- threshold
- schema_hash
- feature_columns

### `POST /predict`
Single transaction fraud prediction

### `POST /predict_batch`
Batch transaction scoring

---

## 🧪 Testing from Terminal

### Single Prediction

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @payload.json
Batch Prediction
curl -X POST http://127.0.0.1:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d @batch_payload.json

Negative Test (Missing Feature)

If "Amount" is removed from payload:

Expected response:

400 Bad Request
Missing features: ['Amount']


This demonstrates fail-fast schema validation.

### Run with Docker
Build
docker build -t fraud-mlops-api:latest .

Run
docker run --rm -p 8000:8000 \
  -v "$(pwd)/artifacts/model:/app/artifacts/model" \
  fraud-mlops-api:latest


### Open:

http://127.0.0.1:8000/docs

http://127.0.0.1:8000/schema

### CI/CD Pipeline

GitHub Actions workflow performs:

Train baseline model

Prepare reference + simulated current dataset

Run drift detection

Trigger retraining if drift threshold exceeded

Upload:

drift_report.json

model.joblib

metadata.json

### Drift Monitoring

Drift detection uses:

Feature-level PSI

Prediction-level PSI

Threshold-based retrain trigger

PSI Interpretation:

< 0.1 → Low drift

0.1 – 0.25 → Moderate drift

0.25 → High drift

### Training-Serving Skew Protection

This project prevents silent schema mismatch by:

Saving feature order during training

Computing schema hash

Validating metadata consistency at API startup

Rejecting requests with missing or extra features

Fail-fast > Silent incorrect predictions.

### Why PR-AUC Instead of Accuracy?

Fraud detection is highly imbalanced.

Accuracy can be misleading (99% normal → 99% accuracy).

PR-AUC focuses on:

Precision (quality of fraud flags)

Recall (coverage of actual fraud)

### Tech Stack

Python

Scikit-learn

FastAPI

Docker

GitHub Actions

Pandas

Joblib

PSI-based Drift Detection

### Future Improvements

MLflow experiment tracking + model registry

Cost-sensitive threshold optimization

Canary deployment strategy

Real-time streaming scoring

S3 model artifact registry
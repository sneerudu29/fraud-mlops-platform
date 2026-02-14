import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import argparse


MODEL_PATH = "artifacts/model/model.joblib"
META_PATH = "artifacts/model/metadata.json"

REFERENCE_PATH = "data/raw/creditcard.csv"  # baseline source for now
CURRENT_PATH = "data/raw/creditcard_current_drifted.csv"

REPORT_DIR = "artifacts/monitoring"
os.makedirs(REPORT_DIR, exist_ok=True)


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10, eps: float = 1e-6) -> float:
    """
    Population Stability Index (PSI):
    - expected: reference distribution
    - actual: current distribution
    Higher PSI => more drift
    """
    # Use quantile bins from expected to make stable buckets
    quantiles = np.linspace(0, 1, buckets + 1)
    breaks = np.unique(np.quantile(expected, quantiles))

    # If feature is constant or breaks collapse, no drift measurable
    if len(breaks) < 3:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=breaks)
    act_counts, _ = np.histogram(actual, bins=breaks)

    exp_perc = exp_counts / max(exp_counts.sum(), 1)
    act_perc = act_counts / max(act_counts.sum(), 1)

    exp_perc = np.clip(exp_perc, eps, 1)
    act_perc = np.clip(act_perc, eps, 1)

    return float(np.sum((act_perc - exp_perc) * np.log(act_perc / exp_perc)))


def load_model_and_meta():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return model, meta


def load_features(csv_path: str, feature_columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])
    # enforce exact training feature order
    df = df[feature_columns]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--current", required=True)
    ap.add_argument("--out", default="artifacts/monitoring/drift_report.json")
    args = ap.parse_args()

    REFERENCE_PATH = args.reference
    CURRENT_PATH = args.current
    out_path = args.out
    model, meta = load_model_and_meta()
    feature_columns = meta["feature_columns"]
    threshold = float(meta["threshold"])

    ref = load_features(REFERENCE_PATH, feature_columns)
    cur = load_features(CURRENT_PATH, feature_columns)

    # Take a stable subset to keep runtime small (you can increase later)
    ref_sample = ref.sample(n=min(20000, len(ref)), random_state=42)
    cur_sample = cur.sample(n=min(20000, len(cur)), random_state=99)

    # ---- 1) Feature drift (PSI per feature) ----
    feature_psi = {}
    for col in feature_columns:
        feature_psi[col] = psi(ref_sample[col].to_numpy(), cur_sample[col].to_numpy(), buckets=10)

    # Summary stats
    avg_feature_psi = float(np.mean(list(feature_psi.values())))
    max_feature_psi = float(np.max(list(feature_psi.values())))
    top5 = sorted(feature_psi.items(), key=lambda x: x[1], reverse=True)[:5]

    # ---- 2) Prediction drift ----
    ref_probs = model.predict_proba(ref_sample)[:, 1]
    cur_probs = model.predict_proba(cur_sample)[:, 1]

    pred_psi = psi(ref_probs, cur_probs, buckets=10)

    # ---- 3) Simple trigger logic ----
    # PSI rough interpretation:
    # <0.1 low drift, 0.1-0.25 moderate, >0.25 high
    trigger_retrain = (avg_feature_psi > 0.10) or (pred_psi > 0.10)

    # Also track how many predictions flip above threshold
    ref_rate = float(np.mean(ref_probs >= threshold))
    cur_rate = float(np.mean(cur_probs >= threshold))
    rate_delta = float(cur_rate - ref_rate)

    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "reference_path": REFERENCE_PATH,
        "current_path": CURRENT_PATH,
        "avg_feature_psi": avg_feature_psi,
        "max_feature_psi": max_feature_psi,
        "top5_feature_psi": [{"feature": k, "psi": v} for k, v in top5],
        "prediction_psi": pred_psi,
        "threshold": threshold,
        "ref_flag_rate": ref_rate,
        "cur_flag_rate": cur_rate,
        "flag_rate_delta": rate_delta,
        "trigger_retrain": trigger_retrain,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("Saved drift report to:", out_path)
    print("Avg feature PSI:", round(avg_feature_psi, 4))
    print("Prediction PSI:", round(pred_psi, 4))
    print("Trigger retrain:", trigger_retrain)
    print("Top-5 drifted features:", report["top5_feature_psi"])


if __name__ == "__main__":
    main()

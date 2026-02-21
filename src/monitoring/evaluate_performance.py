import argparse
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def utc_now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_artifacts(model_path: str, meta_path: str):
    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Labeled CSV containing Class column")
    ap.add_argument("--model", default="artifacts/model/model.joblib")
    ap.add_argument("--meta", default="artifacts/model/metadata.json")
    ap.add_argument("--out", default="artifacts/reports/performance_report.json")
    ap.add_argument("--min_pr_auc", type=float, default=0.70)
    ap.add_argument("--min_recall", type=float, default=0.70)
    ap.add_argument("--min_precision", type=float, default=0.20)
    args = ap.parse_args()

    model, meta = load_artifacts(args.model, args.meta)

    df = pd.read_csv(args.data)
    if "Class" not in df.columns:
        raise ValueError("Dataset must contain 'Class' label column")

    y_true = df["Class"].astype(int)
    X = df.drop(columns=["Class"])

    feature_columns = meta["feature_columns"]
    threshold = float(meta["threshold"])

    # Enforce same training schema
    missing = [c for c in feature_columns if c not in X.columns]
    extra = [c for c in X.columns if c not in feature_columns]
    if missing:
        raise ValueError(f"Missing feature columns in data: {missing[:10]} (total={len(missing)})")
    if extra:
        # not fatal, but we force exact order/selection
        X = X[feature_columns]
    else:
        X = X[feature_columns]

    # Predict
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Metrics
    pr_auc = float(average_precision_score(y_true, y_prob))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fraud_rate = float(np.mean(y_true))

    report = {
        "timestamp": utc_now_z(),
        "model_version": meta.get("model_version"),
        "git_commit": meta.get("git_commit"),
        "threshold": threshold,
        "schema_hash": meta.get("schema_hash"),
        "data_path": args.data,
        "rows": int(len(df)),
        "fraud_rate": fraud_rate,
        "metrics": {
            "pr_auc": pr_auc,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        },
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        },
        "gates": {
            "min_pr_auc": args.min_pr_auc,
            "min_recall": args.min_recall,
            "min_precision": args.min_precision,
        },
    }

    # Save report
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Saved:", args.out)
    print("PR-AUC:", round(pr_auc, 4))
    print("Precision:", round(float(precision), 4))
    print("Recall:", round(float(recall), 4))
    print("F1:", round(float(f1), 4))

    # Quality gates: fail if performance is below thresholds
    failed = []
    if pr_auc < args.min_pr_auc:
        failed.append(f"pr_auc {pr_auc:.4f} < {args.min_pr_auc:.4f}")
    if float(recall) < args.min_recall:
        failed.append(f"recall {float(recall):.4f} < {args.min_recall:.4f}")
    if float(precision) < args.min_precision:
        failed.append(f"precision {float(precision):.4f} < {args.min_precision:.4f}")

    if failed:
        print("\n❌ Performance gate failed:")
        for x in failed:
            print(" -", x)
        raise SystemExit(2)

    print("\n✅ Performance gate passed.")


if __name__ == "__main__":
    main()
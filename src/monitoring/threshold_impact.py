import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, confusion_matrix


def cost_from_confusion(y_true, y_prob, threshold, fraud_loss=500, review_cost=10):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = (fn * fraud_loss) + (fp * review_cost)
    return total_cost, tn, fp, fn, tp, y_pred


def metrics(y_true, y_prob, y_pred):
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "flag_rate": float(np.mean(y_pred)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--label_col", default="Class")
    p.add_argument("--model_path", default="artifacts/model/model.joblib")
    p.add_argument("--metadata_path", default="artifacts/model/metadata.json")
    p.add_argument("--fraud_loss", type=float, default=500.0)
    p.add_argument("--review_cost", type=float, default=10.0)
    p.add_argument("--baseline_threshold", type=float, default=0.5)
    args = p.parse_args()

    df = pd.read_csv(args.data)
    if args.label_col not in df.columns:
        raise ValueError(f"Missing label column: {args.label_col}")

    meta = json.loads(Path(args.metadata_path).read_text())
    feature_cols = meta["feature_columns"]
    opt_threshold = float(meta["threshold"])

    X = df[feature_cols]
    y_true = df[args.label_col].astype(int).values

    model = joblib.load(args.model_path)
    y_prob = model.predict_proba(X)[:, 1]

    out = {
        "fraud_loss": args.fraud_loss,
        "review_cost": args.review_cost,
        "baseline_threshold": args.baseline_threshold,
        "optimized_threshold": opt_threshold,
    }

    # baseline
    base_cost, tn, fp, fn, tp, y_pred_base = cost_from_confusion(
        y_true, y_prob, args.baseline_threshold, args.fraud_loss, args.review_cost
    )
    out["baseline"] = {
        **metrics(y_true, y_prob, y_pred_base),
        "cost": float(base_cost),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    # optimized
    opt_cost, tn, fp, fn, tp, y_pred_opt = cost_from_confusion(
        y_true, y_prob, opt_threshold, args.fraud_loss, args.review_cost
    )
    out["optimized"] = {
        **metrics(y_true, y_prob, y_pred_opt),
        "cost": float(opt_cost),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    Path("reports").mkdir(parents=True, exist_ok=True)
    report_path = Path("reports/threshold_impact.json")
    report_path.write_text(json.dumps(out, indent=2))

    print("Saved:", report_path)
    print("Baseline cost:", out["baseline"]["cost"])
    print("Optimized cost:", out["optimized"]["cost"])
    print("Optimized threshold:", opt_threshold)


if __name__ == "__main__":
    main()
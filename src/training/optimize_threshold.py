import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def compute_cost(y_true, y_prob, threshold, fraud_loss=500, review_cost=10):
    """
    fraud_loss: cost of missing a fraud (FN)
    review_cost: cost of reviewing a normal transaction flagged as fraud (FP)
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = (fn * fraud_loss) + (fp * review_cost)
    return total_cost, fp, fn, tp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to labeled CSV")
    parser.add_argument("--model_path", default="artifacts/model/model.joblib")
    parser.add_argument("--metadata_path", default="artifacts/model/metadata.json")
    parser.add_argument("--label_col", default="Class", help="Label column name")
    parser.add_argument("--fraud_loss", type=float, default=500.0)
    parser.add_argument("--review_cost", type=float, default=10.0)
    parser.add_argument("--min_t", type=float, default=0.0001)
    parser.add_argument("--max_t", type=float, default=0.50)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--max_flag_rate", type=float, default=None,
                        help="Optional constraint: max fraction flagged as fraud, e.g. 0.05")
    args = parser.parse_args()

    data_path = Path(args.data)
    model_path = Path(args.model_path)
    meta_path = Path(args.metadata_path)

    df = pd.read_csv(data_path)

    if args.label_col not in df.columns:
        raise ValueError(f"Label column '{args.label_col}' not found. Columns: {list(df.columns)}")

    y_true = df[args.label_col].astype(int).values

    # Load metadata to get correct feature order
    meta = json.loads(meta_path.read_text())
    feature_cols = meta["feature_columns"]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required feature columns: {missing[:5]} ... total={len(missing)}")

    X = df[feature_cols]

    model = joblib.load(model_path)
    y_prob = model.predict_proba(X)[:, 1]

    thresholds = np.linspace(args.min_t, args.max_t, args.steps)

    best_t = None
    best_cost = float("inf")
    best_stats = None

    for t in thresholds:
        # optional operational constraint: don't flag too many
        if args.max_flag_rate is not None:
            flag_rate = float((y_prob >= t).mean())
            if flag_rate > args.max_flag_rate:
                continue

        cost, fp, fn, tp = compute_cost(
            y_true, y_prob, t,
            fraud_loss=args.fraud_loss,
            review_cost=args.review_cost
        )

        if cost < best_cost:
            best_cost = cost
            best_t = float(t)
            best_stats = {
                "threshold": best_t,
                "min_expected_cost": float(best_cost),
                "fraud_loss": float(args.fraud_loss),
                "review_cost": float(args.review_cost),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "flag_rate": float((y_prob >= t).mean()),
            }

    if best_t is None:
        raise RuntimeError("No threshold satisfied constraints (try increasing max_flag_rate or adjusting range).")

    # Write threshold into metadata.json (THIS is the main outcome)
    meta["threshold"] = best_t
    meta["threshold_policy"] = {
        "type": "cost_sensitive",
        "fraud_loss": float(args.fraud_loss),
        "review_cost": float(args.review_cost),
        "max_flag_rate": args.max_flag_rate,
        "search": {"min_t": args.min_t, "max_t": args.max_t, "steps": args.steps},
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    # Save an optimization report
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "threshold_optimization.json"
    report_path.write_text(json.dumps(best_stats, indent=2))

    print(f"Optimal Threshold: {best_t:.4f}")
    print(f"Minimum Expected Cost: {best_cost:.1f}")
    print(f"Saved: {report_path}")
    print(f"Updated: {meta_path} (threshold={best_t:.4f})")


if __name__ == "__main__":
    main()
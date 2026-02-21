import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix
import argparse


def compute_cost(y_true, y_prob, threshold, fraud_loss=500, review_cost=10):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    total_cost = (fn * fraud_loss) + (fp * review_cost)

    return total_cost, fp, fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="artifacts/model/model.joblib")
    parser.add_argument("--meta", default="artifacts/model/metadata.json")
    parser.add_argument("--fraud_loss", type=float, default=500)
    parser.add_argument("--review_cost", type=float, default=10)
    args = parser.parse_args()

    # Load model + metadata
    model = joblib.load(args.model)
    with open(args.meta, "r") as f:
        meta = json.load(f)

    df = pd.read_csv(args.data)
    y_true = df["Class"]
    X = df[meta["feature_columns"]]

    y_prob = model.predict_proba(X)[:, 1]

    thresholds = np.linspace(0.01, 0.99, 100)

    best_threshold = None
    best_cost = float("inf")

    for t in thresholds:
        cost, fp, fn = compute_cost(
            y_true,
            y_prob,
            threshold=t,
            fraud_loss=args.fraud_loss,
            review_cost=args.review_cost,
        )

        if cost < best_cost:
            best_cost = cost
            best_threshold = t

    print("\nOptimal Threshold:", round(best_threshold, 4))
    print("Minimum Expected Cost:", round(best_cost, 2))

    # Save into metadata
    meta["threshold"] = float(best_threshold)
    meta["business_cost"] = {
        "fraud_loss": args.fraud_loss,
        "review_cost": args.review_cost,
        "expected_cost": best_cost,
    }

    with open(args.meta, "w") as f:
        json.dump(meta, f, indent=2)

    print("\nMetadata updated with optimized threshold.")


if __name__ == "__main__":
    main()
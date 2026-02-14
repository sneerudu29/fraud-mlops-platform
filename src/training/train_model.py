import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to CSV with Class label")
    ap.add_argument("--outdir", required=True, help="Output directory for model artifacts")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    df = pd.read_csv(args.data)

    y = df["Class"].astype(int)
    X = df.drop(columns=["Class"])

    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    y_probs = model.predict_proba(X_test)[:, 1]
    pr_auc = float(average_precision_score(y_test, y_probs))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, outdir / "model.joblib")

    meta = {
        "feature_columns": feature_columns,
        "threshold": args.threshold,
        "pr_auc": pr_auc
    }
    with open(outdir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved model to:", outdir)
    print("PR-AUC:", round(pr_auc, 4))


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path
import hashlib
import subprocess
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def compute_schema_hash(feature_columns: list[str]) -> str:
    s = "|".join(feature_columns).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to CSV with Class label")
    ap.add_argument("--outdir", required=True, help="Output directory for model artifacts")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max_iter", type=int, default=2000)
    ap.add_argument("--cv", type=int, default=3)
    args = ap.parse_args()

    df = pd.read_csv(args.data)

    y = df["Class"].astype(int)
    X = df.drop(columns=["Class"])
    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base_model = LogisticRegression(max_iter=args.max_iter)

    # calibrated probabilities -> better thresholding realism
    model = CalibratedClassifierCV(base_model, method="sigmoid", cv=args.cv)
    model.fit(X_train, y_train)

    y_probs = model.predict_proba(X_test)[:, 1]
    pr_auc = float(average_precision_score(y_test, y_probs))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, outdir / "model.joblib")

    trained_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    git_commit = get_git_commit()
    schema_hash = compute_schema_hash(feature_columns)

    meta = {
        "model_version": trained_at,
        "trained_at": trained_at,
        "git_commit": git_commit,
        "num_features": len(feature_columns),
        "schema_hash": schema_hash,
        "feature_columns": feature_columns,
        "threshold": args.threshold,
        "pr_auc": pr_auc,
        "calibration": {"method": "sigmoid", "cv": args.cv},
    }

    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2))

    print("Saved model to:", outdir)
    print("PR-AUC:", round(pr_auc, 4))


if __name__ == "__main__":
    main()
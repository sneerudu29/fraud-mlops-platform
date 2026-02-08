import json
import joblib
import pandas as pd

# Load trained model
model = joblib.load("artifacts/model/model.joblib")

# Load metadata
with open("artifacts/model/metadata.json") as f:
    metadata = json.load(f)
    
print("Loaded threshold:", metadata["threshold"])
print("Number of features:", len(metadata["feature_columns"]))

# Load data
df = pd.read_csv("data/raw/creditcard.csv")
X = df.drop(columns=["Class"])

# Ensure feature order matches training
X = X[metadata["feature_columns"]]

# Take a few samples
sample = X.head(5)

# Predict probabilities
probs = model.predict_proba(sample)[:, 1]

# Apply threshold
preds = (probs >= metadata["threshold"]).astype(int)

print("Probabilities:", probs)
print("Predictions:", preds)

# Repeat prediction to confirm stability
probs_2 = model.predict_proba(sample)[:, 1]

print("Predictions stable:", (probs == probs_2).all())


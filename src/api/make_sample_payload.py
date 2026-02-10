import json
import pandas as pd

# Load metadata
with open("artifacts/model/metadata.json") as f:
    metadata = json.load(f)

cols = metadata["feature_columns"]

# Load data and take one real row
df = pd.read_csv("data/raw/creditcard.csv")
X = df.drop(columns=["Class"])
X = X[cols]

row = X.iloc[0].to_dict()   # real transaction features

payload = {"features": row}

print(json.dumps(payload, indent=2))

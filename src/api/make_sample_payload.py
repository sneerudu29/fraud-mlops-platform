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

rows = [X.iloc[0].to_dict(), X.iloc[1].to_dict()] #  real transaction features
payload = {"records": rows}
print(json.dumps(payload, indent=2))
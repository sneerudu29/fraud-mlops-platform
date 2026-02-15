import pandas as pd
from pathlib import Path

RAW = Path("data/raw/creditcard.csv")
OUT = Path("data/sample/creditcard_sample.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW)
df.columns = [c.strip() for c in df.columns]

label_col = "Class"
print("RAW columns (last 5):", df.columns.tolist()[-5:])

if label_col not in df.columns:
    raise ValueError(f"RAW file missing '{label_col}'. Found: {df.columns.tolist()}")

fraud = df[df[label_col] == 1]
normal = df[df[label_col] == 0]

fraud_s = fraud.sample(n=min(len(fraud), 2000), random_state=42)
normal_s = normal.sample(n=min(len(normal), 2000), random_state=42)

sample_df = pd.concat([fraud_s, normal_s], ignore_index=True)
sample_df = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

assert label_col in sample_df.columns, "BUG: label got dropped before saving!"

sample_df.to_csv(OUT, index=False)

print("Wrote:", OUT)
print("Sample columns (last 5):", sample_df.columns.tolist()[-5:])
print("Rows:", len(sample_df))
print("Fraud count:", int(sample_df[label_col].sum()))
print("Fraud rate:", round(sample_df[label_col].mean(), 6))

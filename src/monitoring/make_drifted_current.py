import pandas as pd
import numpy as np

df = pd.read_csv("data/raw/creditcard.csv")

# simulate drift: increase Amount distribution
df["Amount"] = df["Amount"] * 50

df.to_csv("data/raw/creditcard_current_drifted.csv", index=False)
print("Wrote data/raw/creditcard_current_drifted.csv")

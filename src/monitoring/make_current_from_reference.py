import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--amount_multiplier", type=float, default=1.0)
    args = ap.parse_args()

    df = pd.read_csv(args.infile)

    if "Amount" in df.columns:
        df["Amount"] = df["Amount"] * args.amount_multiplier

    df.to_csv(args.outfile, index=False)
    print("Wrote:", args.outfile)

if __name__ == "__main__":
    main()

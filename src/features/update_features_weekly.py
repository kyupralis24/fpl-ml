# src/features/update_features_weekly.py
import argparse
import os
import pandas as pd
import numpy as np

RAW_DIR = "data/raw/current"
FEATURES_PATH = "data/processed/features.csv"

ROLL_COLS = ["total_points","minutes","goals_scored","assists","clean_sheets","bps"]

def make_rollings(df):
    # sort by GW and compute rolling means on PRIOR weeks (shift(1))
    df = df.sort_values(["element","GW"])
    for col in ROLL_COLS:
        df[f"roll3_{col}"] = (
            df.groupby("element")[col]
              .apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
              .reset_index(level=0, drop=True)
        )
    # Fill NaNs for early weeks
    roll_cols = [f"roll3_{c}" for c in ROLL_COLS]
    df[roll_cols] = df[roll_cols].fillna(0)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gw", type=int, required=True, help="Gameweek to append (e.g., 1)")
    args = parser.parse_args()

    gw_file = os.path.join(RAW_DIR, f"gw{args.gw}_player_stats.csv")
    if not os.path.exists(gw_file):
        raise FileNotFoundError(f"Missing {gw_file}. Run fetch_gw.py first.")

    new_gw = pd.read_csv(gw_file)

    # Minimal type safety
    for c in ["GW","element","value","minutes","total_points"]:
        if c in new_gw.columns:
            new_gw[c] = pd.to_numeric(new_gw[c], errors="coerce")

    # Load existing features (if any), append, drop duplicates per (element, GW)
    if os.path.exists(FEATURES_PATH):
        base = pd.read_csv(FEATURES_PATH)
        # harmonize key columns if needed
        needed = set(new_gw.columns)
        for col in needed - set(base.columns):
            base[col] = np.nan
        for col in set(base.columns) - set(new_gw.columns):
            new_gw[col] = np.nan
        combined = pd.concat([base[base.columns], new_gw[base.columns]], ignore_index=True)
        combined = combined.drop_duplicates(subset=["element","GW"], keep="last")
    else:
        combined = new_gw

    combined = make_rollings(combined)
    os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
    combined.to_csv(FEATURES_PATH, index=False)
    print(f"✅ Updated features with GW{args.gw} → {FEATURES_PATH}")
    print(f"Rows: {len(combined)}, Cols: {len(combined.columns)}")

if __name__ == "__main__":
    main()
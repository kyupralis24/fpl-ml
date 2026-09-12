"""Build leakage-free player form features and next-gameweek labels."""
import argparse
import os
import numpy as np
import pandas as pd

RAW_DIR = "data/raw/current"
FEATURES_PATH = "data/processed/features.csv"
ROLL_COLS = ["total_points", "minutes", "goals_scored", "assists", "clean_sheets", "bps"]
TARGET_FIXTURE_COLS = ["fixture_count", "fixture_home_count", "fixture_difficulty"]


def prior_rolling(df, column, window, aggregation="mean"):
    """Return a per-player rolling value using only earlier gameweeks."""
    return df.groupby("element")[column].transform(
        lambda s: getattr(s.shift(1).rolling(window, min_periods=1), aggregation)()
    )


def add_features(df):
    df = df.sort_values(["element", "GW"]).copy()
    for col in ROLL_COLS:
        df[f"roll3_{col}"] = prior_rolling(df, col, 3)
    for window in (3, 5):
        df[f"avg_points_last{window}"] = prior_rolling(df, "total_points", window)
        df[f"avg_minutes_last{window}"] = prior_rolling(df, "minutes", window)
        df[f"goals_last{window}"] = prior_rolling(df, "goals_scored", window, "sum")
        df[f"std_points_last{window}"] = df.groupby("element")["total_points"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=2).std()
        )
    df["ema_points"] = df.groupby("element")["total_points"].transform(
        lambda s: s.shift(1).ewm(span=3, adjust=False).mean()
    )
    # A GW-t snapshot predicts aggregate FPL points in GW t+1.
    next_gw = df.groupby("element")["GW"].shift(-1)
    df["next_total_points"] = df.groupby("element")["total_points"].shift(-1)
    df.loc[next_gw.ne(df["GW"] + 1), "next_total_points"] = np.nan
    # These schedule fields are taken from the target GW, not the completed GW.
    for col in TARGET_FIXTURE_COLS:
        df[f"target_{col}"] = df.groupby("element")[col].shift(-1) if col in df else 0.0
    generated = [c for c in df if c.startswith(("roll", "avg_", "goals_last", "std_", "ema_"))]
    df[generated] = df[generated].fillna(0.0)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gw", type=int, required=True, help="Gameweek just fetched")
    parser.add_argument("--reset", action="store_true", help="Rebuild from raw GW CSVs")
    args = parser.parse_args()
    files = sorted(
        (f for f in os.listdir(RAW_DIR) if f.startswith("gw") and f.endswith("_player_stats.csv")),
        key=lambda f: int(f.split("_")[0][2:]),
    )
    if not files:
        raise FileNotFoundError(f"No player-stat files found in {RAW_DIR}. Run fetch_gw.py first.")
    frames = [pd.read_csv(os.path.join(RAW_DIR, f)) for f in files]
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(["element", "GW"], keep="last")
    numeric = combined.select_dtypes(include=[np.number]).columns
    combined[numeric] = combined[numeric].fillna(0)
    features = add_features(combined)
    os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
    features.to_csv(FEATURES_PATH, index=False)
    print(f"✅ Rebuilt {FEATURES_PATH}: {len(features)} rows, {features['next_total_points'].notna().sum()} next-GW labels")


if __name__ == "__main__":
    main()

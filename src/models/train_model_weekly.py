"""Train the production next-gameweek model using only pre-deadline features."""
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

FEATURES_PATH = "data/processed/features.csv"
MODEL_PATH = "models/next_gw_model.pkl"


def feature_columns(df):
    form = [c for c in df if c.startswith(("roll", "avg_", "goals_last", "std_", "ema_", "target_fixture"))]
    return ["position_id", "value", *form]


def make_model():
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                             min_child_samples=30, random_state=42, verbosity=-1)
    except ImportError:
        return RandomForestRegressor(n_estimators=400, min_samples_leaf=8, random_state=42, n_jobs=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_gw", type=int, required=True, help="First GW not available at the deadline")
    args = parser.parse_args()
    df = pd.read_csv(FEATURES_PATH)
    cols = [c for c in feature_columns(df) if c in df]
    train = df[(df["GW"] < args.target_gw) & df["next_total_points"].notna()].copy()
    if train.empty:
        raise ValueError("No labelled next-GW rows available. Fetch at least two completed gameweeks.")
    X = train[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    model = make_model()
    model.fit(X, train["next_total_points"].astype(float))
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": model, "feature_columns": cols, "trained_through_gw": args.target_gw - 1}, MODEL_PATH)
    print(f"✅ Trained next-GW model on {len(train)} labelled rows through GW{args.target_gw - 1}")


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

RAW_DATA_PATH = "data/raw/2024-25_merged_gw.csv"
PROCESSED_DATA_PATH = "data/processed/features.csv"


class CustomFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        # Use 'name' instead of 'player_id'
        player_col = "name"

        # Rolling average points over last 3 matches
        if "total_points" in X.columns:
            X["points_last_3"] = (
                X.groupby(player_col)["total_points"]
                .rolling(3, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

        # Goals + Assists combined
        if "goals_scored" in X.columns and "assists" in X.columns:
            X["goal_contributions"] = X["goals_scored"] + X["assists"]

        # Minutes played percentage (assuming 90 min match)
        if "minutes" in X.columns:
            X["minutes_pct"] = X["minutes"] / 90.0

        # Home/Away flag
        if "was_home" in X.columns:
            X["is_home"] = X["was_home"].astype(int)

        # Fill missing values
        X = X.fillna(0)

        return X


def main():
    # Load raw historical data
    df = pd.read_csv(RAW_DATA_PATH, on_bad_lines="skip")
    print(f"Loaded data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Apply custom features
    cf = CustomFeatures()
    df_features = cf.fit_transform(df)

    # Save processed dataset
    df_features.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"✅ Features saved to {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()
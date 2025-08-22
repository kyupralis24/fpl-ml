# src/models/train_model_weekly.py
import argparse
import os
import pandas as pd
import numpy as np

MODEL_PATH = "models/LightGBM_model.pkl"

def choose_model():
    try:
        import lightgbm as lgb
        return "lgb"
    except Exception:
        return "rf"

def train_and_save(df_train, feature_cols, y_col):
    model_choice = choose_model()
    if model_choice == "lgb":
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.9,
            random_state=42
        )
    else:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=400, max_depth=None, random_state=42, n_jobs=-1
        )

    X = df_train[feature_cols]
    y = df_train[y_col].astype(float)

    model.fit(X, y)
    import joblib, os
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model trained on {len(df_train)} rows and saved to {MODEL_PATH}")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_gw", type=int, required=True,
                        help="Train on all rows with GW < target_gw (e.g., 2 for training after GW1)")
    args = parser.parse_args()
    target_gw = args.target_gw

    FEAT_PATH = "data/processed/features.csv"
    if not os.path.exists(FEAT_PATH):
        raise FileNotFoundError("Missing data/processed/features.csv. Run update_features_weekly.py first.")

    df = pd.read_csv(FEAT_PATH)

    # Training set: rows strictly before target_gw
    train_df = df[df["GW"] < target_gw].copy()
    if train_df.empty:
        raise ValueError(f"No training rows found for GW < {target_gw}")

    # Build feature columns (numeric only, excluding target & GW/leakage)
    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = {"total_points","GW","team_h_score","team_a_score","fixture_id","opponent_team","element","team_id","position_id"}
    feature_cols = [c for c in num_cols if c not in drop_cols]

    print(f"Using {len(feature_cols)} numeric features.")

    _ = train_and_save(train_df, feature_cols, y_col="total_points")

if __name__ == "__main__":
    main()
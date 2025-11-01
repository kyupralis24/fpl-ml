# src/models/train_model_weekly.py
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import joblib

MODEL_PATH = "models/LightGBM_model.pkl"
STACKED_MODEL_PATH = "models/stacked_model.pkl"

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
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model trained on {len(df_train)} rows and saved to {MODEL_PATH}")
    return model

def train_stacked_model(df_train, feature_cols, y_col):
    """Train a stacking ensemble model with LGBM, RF, KNN as base models and Ridge as meta-learner."""
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        raise ImportError("lightgbm is required for stacked model. Install with: pip install lightgbm")
    
    X = df_train[feature_cols]
    y = df_train[y_col].astype(float)
    
    # Split for validation (optional, but good for evaluation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    base_models = [
        ("lgbm", LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42)),
        ("rf", RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)),
        ("knn", KNeighborsRegressor(n_neighbors=5))
    ]
    
    stacked = StackingRegressor(estimators=base_models, final_estimator=Ridge(random_state=42))
    stacked.fit(X_train, y_train)
    
    os.makedirs(os.path.dirname(STACKED_MODEL_PATH), exist_ok=True)
    joblib.dump(stacked, STACKED_MODEL_PATH)
    print(f"✅ Stacked model trained on {len(X_train)} rows and saved to {STACKED_MODEL_PATH}")
    print(f"   Test set size: {len(X_test)} rows")
    
    # Optional: print test score
    test_score = stacked.score(X_test, y_test)
    print(f"   Test R² score: {test_score:.4f}")
    
    return stacked

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

    # Train both single model (legacy) and stacked model
    _ = train_and_save(train_df, feature_cols, y_col="total_points")
    _ = train_stacked_model(train_df, feature_cols, y_col="total_points")

if __name__ == "__main__":
    main()
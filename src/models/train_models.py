# src/models/train_models.py

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
import joblib

FEATURES_PATH = "data/processed/features.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    df = pd.read_csv(FEATURES_PATH)
    return df

def split_by_gw(df, split_ratio=0.6):
    total_gw = df['round'].max()
    split_gw = int(total_gw * split_ratio)
    train = df[df['round'] <= split_gw]
    test = df[df['round'] > split_gw]
    return train, test

def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"{name} MAE: {mae:.3f}")
    joblib.dump(model, f"{MODEL_DIR}/{name}_model.pkl")
    return mae

def main():
    df = load_data()
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} features")

    train_df, test_df = split_by_gw(df, split_ratio=0.6)

    target = "total_points"
    features = [c for c in df.select_dtypes(include=[np.number]).columns if c != target and c != 'round']

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    models = {
        "LightGBM": lgb.LGBMRegressor(n_estimators=200),
        "RandomForest": RandomForestRegressor(n_estimators=100),
        "KNN": KNeighborsRegressor(n_neighbors=10)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        mae = evaluate_model(name, model, X_test, y_test)
        results[name] = mae

    print("Performance summary (MAE):")
    for k, v in results.items():
        print(f" - {k}: {v:.3f}")

if __name__ == "__main__":
    main()
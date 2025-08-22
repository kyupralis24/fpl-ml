# src/models/predict_next_gw.py
import argparse
import os
import pandas as pd
import numpy as np
import joblib

FEATURES_PATH = "data/processed/features.csv"
MODEL_PATH = "models/LightGBM_model.pkl"
PRED_DIR = "data/predictions"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_gw", type=int, required=False, help="GW to predict (e.g., 2)")
    args = parser.parse_args()

    df = pd.read_csv(FEATURES_PATH)
    current_max = int(df["GW"].max())
    target_gw = args.target_gw if args.target_gw else current_max + 1

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Train it first with train_model_weekly.py")

    model = joblib.load(MODEL_PATH)

    # Use the latest known row per player (typically GW target_gw-1)
    latest_gw = target_gw - 1
    inf_df = df[df["GW"] == latest_gw].copy()
    if inf_df.empty:
        # fallback: use each player's last available GW
        inf_df = df.sort_values("GW").groupby("element").tail(1)

    # Numeric features consistent with training
    num_cols = inf_df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = {"total_points","GW","team_h_score","team_a_score","fixture_id","opponent_team","element","team_id","position_id"}
    feature_cols = [c for c in num_cols if c not in drop_cols]

    preds = model.predict(inf_df[feature_cols])

    out = inf_df[["name","team","position","value"]].copy()
    out["pred_points"] = preds
    out = out.sort_values("pred_points", ascending=False).reset_index(drop=True)

    os.makedirs(PRED_DIR, exist_ok=True)
    out_path = os.path.join(PRED_DIR, f"predictions_gw{target_gw}.csv")
    out.to_csv(out_path, index=False)

    print(f"Current last GW in features: {current_max}")
    print(out.head(15))
    print(f"✅ Predictions for GW{target_gw} saved to {out_path}")

if __name__ == "__main__":
    main()
"""Create deadline-safe predictions for the target gameweek."""
import argparse
import os
import joblib
import numpy as np
import pandas as pd
import requests

FEATURES_PATH = "data/processed/features.csv"
MODEL_PATH = "models/next_gw_model.pkl"
PRED_DIR = "data/predictions"
API_BASE = "https://fantasy.premierleague.com/api"


def target_fixture_features(target_gw):
    fixtures = pd.DataFrame(requests.get(f"{API_BASE}/fixtures/?event={target_gw}", timeout=20).json())
    rows = []
    for team_id in range(1, 21):
        fs = fixtures[(fixtures.team_h == team_id) | (fixtures.team_a == team_id)] if not fixtures.empty else fixtures
        homes = fs[fs.team_h == team_id] if not fs.empty else fs
        difficulty = [r.team_h_difficulty if r.team_h == team_id else r.team_a_difficulty for _, r in fs.iterrows()]
        rows.append({"team_id": team_id, "target_fixture_count": len(fs),
                     "target_fixture_home_count": len(homes),
                     "target_fixture_difficulty": np.mean(difficulty) if difficulty else 0.0})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_gw", type=int, help="GW to predict; defaults to next completed GW")
    parser.add_argument("--offline", action="store_true", help="Use a saved raw fixture snapshot if present")
    args = parser.parse_args()
    df = pd.read_csv(FEATURES_PATH)
    target_gw = args.target_gw or int(df.GW.max()) + 1
    artifact = joblib.load(MODEL_PATH)
    if not isinstance(artifact, dict):
        raise ValueError("Old model artifact found. Re-run train_model_weekly.py.")
    latest = df[df.GW == target_gw - 1].copy()
    if latest.empty:
        latest = df.sort_values("GW").groupby("element").tail(1).copy()
    snapshot = f"data/raw/current/gw{target_gw}_fixtures.csv"
    if args.offline and os.path.exists(snapshot):
        fixture = pd.read_csv(snapshot).rename(columns={c: f"target_{c}" for c in ["fixture_count", "fixture_home_count", "fixture_difficulty"]})
        fixture = fixture.rename(columns={"target_team_id": "team_id"})
    else:
        fixture = target_fixture_features(target_gw)
    latest = latest.drop(columns=[c for c in latest if c.startswith("target_fixture_")], errors="ignore")
    latest = latest.merge(fixture, on="team_id", how="left")
    cols = artifact["feature_columns"]
    X = latest.reindex(columns=cols, fill_value=0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    latest["pred_points"] = artifact["model"].predict(X).clip(min=0)
    out = latest[["element", "name", "team", "position", "value", "pred_points"]].rename(columns={"element": "player_id"})
    os.makedirs(PRED_DIR, exist_ok=True)
    path = f"{PRED_DIR}/predictions_gw{target_gw}.csv"
    out.sort_values("pred_points", ascending=False).to_csv(path, index=False)
    print(f"✅ Saved {len(out)} deadline-safe predictions to {path}")


if __name__ == "__main__":
    main()

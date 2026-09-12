# src/optimization/select_squad.py

import argparse
import os
import re
import pandas as pd
import pulp

BUDGET = 100.0
POSITION_LIMITS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_TEAM = 3


def normalize_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize position labels to {GK, DEF, MID, FWD}."""
    pos_map = {"GKP": "GK", "GK": "GK", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}
    df = df.copy()
    df["position"] = df["position"].map(lambda x: pos_map.get(str(x).upper(), str(x).upper()))
    return df


def scale_values_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Auto-detect price scale:
      - If any value > 20, assume tenths (e.g., 92 -> 9.2) and divide by 10.
      - Else leave as-is.
    """
    df = df.copy()
    if (df["value"].max() > 20) or (df["value"].median() > 20):
        df["value"] = df["value"] / 10.0
    return df


def main():
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred",
        type=str,
        default="data/predictions/predictions_gw2.csv",
        help="Path to predictions CSV",
    )
    args = parser.parse_args()
    pred_file = args.pred

    # Extract GW number from filename
    m = re.search(r"gw(\d+)", pred_file, re.IGNORECASE)
    gw = m.group(1) if m else "unknown"

    # Load predictions
    df = pd.read_csv(pred_file)

    # Basic column checks
    required_cols = {"name", "team", "position", "value", "pred_points"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Prediction file missing required columns: {sorted(missing)}")

    # Normalize positions & value scale
    df = normalize_positions(df)
    df = scale_values_if_needed(df)

    # Create ILP problem
    prob = pulp.LpProblem("FPL_Squad_Selection", pulp.LpMaximize)

    # Decision variables
    player_vars = pulp.LpVariable.dicts("player", df.index, cat="Binary")
    starter_vars = pulp.LpVariable.dicts("starter", df.index, cat="Binary")
    captain_vars = pulp.LpVariable.dicts("captain", df.index, cat="Binary")

    # Score only the starting XI; the captain receives one additional score.
    prob += pulp.lpSum(
        starter_vars[i] * df.loc[i, "pred_points"]
        + captain_vars[i] * df.loc[i, "pred_points"]
        for i in df.index
    )

    # Budget
    prob += pulp.lpSum(player_vars[i] * df.loc[i, "value"] for i in df.index) <= BUDGET

    # Position constraints (exact)
    for pos, limit in POSITION_LIMITS.items():
        prob += pulp.lpSum(player_vars[i] for i in df.index if df.loc[i, "position"] == pos) == limit

    # Max 3 per team
    for team in df["team"].unique():
        prob += pulp.lpSum(player_vars[i] for i in df.index if df.loc[i, "team"] == team) <= MAX_PER_TEAM

    # Exactly 15 players
    prob += pulp.lpSum(player_vars[i] for i in df.index) == 15

    # Valid FPL starting XI: 1 GK, 3--5 DEF, 2--5 MID, and 1--3 FWD.
    prob += pulp.lpSum(starter_vars[i] for i in df.index) == 11
    starter_limits = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
    for pos, (minimum, maximum) in starter_limits.items():
        starters_at_position = pulp.lpSum(starter_vars[i] for i in df.index if df.loc[i, "position"] == pos)
        prob += starters_at_position >= minimum
        prob += starters_at_position <= maximum

    # Exactly 1 captain
    prob += pulp.lpSum(captain_vars[i] for i in df.index) == 1

    # Captain must be selected
    for i in df.index:
        prob += starter_vars[i] <= player_vars[i]
        prob += captain_vars[i] <= player_vars[i]
        prob += captain_vars[i] <= starter_vars[i]

    # Solve
    status = prob.solve(pulp.PULP_CBC_CMD(msg=True))
    status_str = pulp.LpStatus[status]

    if status_str != "Optimal":
        raise RuntimeError(f"❌ Optimization status: {status_str}. Check constraints/data.")

    # Collect solution
    chosen = df[["name", "team", "position", "value", "pred_points"]].copy()
    chosen["selected"] = [player_vars[i].value() for i in df.index]
    chosen["starter"] = [starter_vars[i].value() for i in df.index]
    chosen["captain"] = [captain_vars[i].value() for i in df.index]

    squad = chosen[chosen["selected"] == 1].sort_values("pred_points", ascending=False)

    # Safety: ensure exactly 15 returned
    if len(squad) != 15:
        raise RuntimeError(f"❌ Solver returned {len(squad)} players, expected 15.")

    # Pretty output
    print(f"\n📊 Optimal Squad for GW{gw}:\n")
    print(squad[["name", "team", "position", "value", "pred_points", "starter", "captain"]])

    total_cost = round(squad["value"].sum(), 2)
    total_points = round(squad.loc[squad["starter"] == 1, "pred_points"].sum() + squad.loc[squad["captain"] == 1, "pred_points"].sum(), 2)

    print("\n💰 Total Cost: ", total_cost)
    print("⭐ Total Predicted Points (with captaincy): ", total_points)

    # Save with GW in filename (no overwrite of other weeks)
    os.makedirs("data/predictions", exist_ok=True)
    out_path = f"data/predictions/optimal_squad_gw{gw}.csv"
    squad.to_csv(out_path, index=False)
    print(f"\n✅ Squad saved to {out_path}")


if __name__ == "__main__":
    main()

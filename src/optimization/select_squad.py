import pandas as pd
import pulp

PRED_FILE = "data/predictions/predictions_gw22.csv"  # Update with latest prediction file
BUDGET = 100.0
POSITION_LIMITS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_TEAM = 3

def main():
    # Load predictions
    df = pd.read_csv(PRED_FILE)

    # Convert value to millions if needed (assumes input in tenths of a million)
    df["value"] = df["value"] / 10.0  

    if "value" not in df.columns:
        raise ValueError("Prediction file must contain 'value' column with player cost (in millions)")

    # Create ILP problem
    prob = pulp.LpProblem("FPL_Squad_Selection", pulp.LpMaximize)

    # Decision variables: 1 if player selected, 0 otherwise
    player_vars = pulp.LpVariable.dicts("player", df.index, cat="Binary")

    # Captain variables: 1 if player is captain, 0 otherwise
    captain_vars = pulp.LpVariable.dicts("captain", df.index, cat="Binary")

    # Objective: maximize predicted points (captain gets an extra x1 multiplier)
    prob += pulp.lpSum(
        player_vars[i] * df.loc[i, "pred_points"] +  # base points
        captain_vars[i] * df.loc[i, "pred_points"]   # extra for captain
        for i in df.index
    )

    # Budget constraint
    prob += pulp.lpSum(player_vars[i] * df.loc[i, "value"] for i in df.index) <= BUDGET

    # Position constraints
    for pos, limit in POSITION_LIMITS.items():
        prob += pulp.lpSum(player_vars[i] for i in df.index if df.loc[i, "position"] == pos) == limit

    # Max players per team constraint
    for team in df["team"].unique():
        prob += pulp.lpSum(player_vars[i] for i in df.index if df.loc[i, "team"] == team) <= MAX_PER_TEAM

    # Exactly 15 players
    prob += pulp.lpSum(player_vars[i] for i in df.index) == 15

    # Exactly 1 captain
    prob += pulp.lpSum(captain_vars[i] for i in df.index) == 1

    # Captain must be a selected player
    for i in df.index:
        prob += captain_vars[i] <= player_vars[i]

    # Solve
    prob.solve()

    # Output chosen squad
    chosen = df[["name", "team", "position", "value", "pred_points"]].copy()
    chosen["selected"] = [player_vars[i].value() for i in df.index]
    chosen["captain"] = [captain_vars[i].value() for i in df.index]

    squad = chosen[chosen["selected"] == 1].sort_values("pred_points", ascending=False)

    print("\nOptimal Squad:\n", squad)
    print("\nTotal Cost: ", squad["value"].sum())
    print("Total Predicted Points (with captaincy): ", 
          squad["pred_points"].sum() + squad.loc[squad["captain"] == 1, "pred_points"].sum())

    squad.to_csv("data/predictions/optimal_squad.csv", index=False)
    print("\n✅ Squad saved to data/predictions/optimal_squad.csv")

if __name__ == "__main__":
    main()
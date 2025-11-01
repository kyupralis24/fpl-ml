import pandas as pd
import argparse
import os
from sklearn.metrics import mean_absolute_error, r2_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gw", type=int, required=True, help="Gameweek number")
    args = parser.parse_args()
    gw = args.gw

    pred_path = f"data/predictions/predictions_gw{gw}.csv"
    actual_path = f"data/raw/gw{gw}_actual_points.csv"

    if not os.path.exists(pred_path) or not os.path.exists(actual_path):
        raise FileNotFoundError(f"Missing files for GW{gw}. Check paths:\n{pred_path}\n{actual_path}")

    # Load both datasets
    pred = pd.read_csv(pred_path)
    actual = pd.read_csv(actual_path)

    # Merge on 'name' and 'team'
    merged = pred.merge(
        actual[["name", "team", "total_points"]],
        on=["name", "team"],
        how="inner",
        suffixes=("_pred", "_actual")
    )

    if merged.empty:
        raise ValueError(f"No matching players found between predictions and actuals for GW{gw}")

    # Compute metrics
    mae = mean_absolute_error(merged["total_points"], merged["pred_points"])
    r2 = r2_score(merged["total_points"], merged["pred_points"])

    print(f"\n📊 Evaluation Results for GW{gw}:")
    print(f" - Mean Absolute Error (MAE): {mae:.3f}")
    print(f" - R² Score: {r2:.3f}")
    print(f" - Players compared: {len(merged)}")

    # Save merged data for record-keeping
    os.makedirs("data/evaluation", exist_ok=True)
    merged.to_csv(f"data/evaluation/eval_gw{gw}.csv", index=False)
    print(f"\n✅ Evaluation data saved to data/evaluation/eval_gw{gw}.csv")

if __name__ == "__main__":
    main()
"""Evaluate the exact weekly workflow: train through t-1, predict t."""
import argparse
import os
import sys
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))
from train_model_weekly import feature_columns, make_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_gw", type=int, default=3)
    parser.add_argument("--end_gw", type=int)
    args = parser.parse_args()
    df = pd.read_csv("data/processed/features.csv")
    cols = [c for c in feature_columns(df) if c in df]
    end = args.end_gw or int(df.GW.max())
    records = []
    for target_gw in range(args.start_gw, end + 1):
        train = df[(df.GW < target_gw - 1) & df.next_total_points.notna()]
        test = df[df.GW == target_gw - 1].dropna(subset=["next_total_points"])
        if train.empty or test.empty:
            continue
        model = make_model().fit(train[cols].fillna(0), train.next_total_points)
        pred = model.predict(test[cols].fillna(0))
        records.append({"GW": target_gw, "players": len(test),
                        "MAE": mean_absolute_error(test.next_total_points, pred),
                        "RMSE": mean_squared_error(test.next_total_points, pred) ** 0.5,
                        "Spearman": pd.Series(pred).corr(test.next_total_points.reset_index(drop=True), method="spearman")})
    out = pd.DataFrame(records)
    if out.empty:
        raise ValueError("Not enough consecutive, labelled gameweeks for walk-forward validation.")
    os.makedirs("data/evaluation", exist_ok=True)
    out.to_csv("data/evaluation/walk_forward_metrics.csv", index=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("✅ Saved data/evaluation/walk_forward_metrics.csv")


if __name__ == "__main__":
    main()

import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def main():
    eval_dir = "data/evaluation"
    if not os.path.exists(eval_dir):
        print("❌ No evaluation directory found. Run evaluate_model_weekly.py first.")
        return

    # Find all evaluation files
    eval_files = sorted(glob.glob(os.path.join(eval_dir, "eval_gw*.csv")))

    if not eval_files:
        print("⚠️ No evaluation files found in data/evaluation/. Run evaluations first.")
        return

    # Initialize list to collect metrics
    records = []

    for file in eval_files:
        gw = int(file.split("eval_gw")[-1].split(".csv")[0])
        df = pd.read_csv(file)

        if "total_points" not in df.columns or "pred_points" not in df.columns:
            print(f"Skipping {file} - missing required columns")
            continue

        # Calculate metrics
        mae = (df["pred_points"] - df["total_points"]).abs().mean()
        ss_res = ((df["pred_points"] - df["total_points"]) ** 2).sum()
        ss_tot = ((df["total_points"] - df["total_points"].mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0

        records.append({"GW": gw, "MAE": mae, "R2": r2})

    if not records:
        print("⚠️ No valid evaluation data to plot.")
        return

    results = pd.DataFrame(records).sort_values("GW")

    # Plot MAE and R² trends
    plt.figure(figsize=(10, 6))
    plt.plot(results["GW"], results["MAE"], marker="o", label="MAE (Lower is Better)")
    plt.plot(results["GW"], results["R2"], marker="s", label="R² Score (Higher is Better)")
    plt.title("📊 Model Performance Across Gameweeks")
    plt.xlabel("Gameweek")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save and show
    os.makedirs("data/plots", exist_ok=True)
    plt.savefig("data/plots/model_performance_trend.png")
    plt.show()

    print("\n✅ Saved performance plot to data/plots/model_performance_trend.png")

if __name__ == "__main__":
    main()
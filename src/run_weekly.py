"""Run the complete post-gameweek FPL pipeline with one command."""
import argparse
import subprocess
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
API = "https://fantasy.premierleague.com/api/bootstrap-static/"


def latest_finished_gw():
    events = requests.get(API, timeout=20).json()["events"]
    finished = [event["id"] for event in events if event.get("finished")]
    if not finished:
        raise ValueError("FPL has no finished gameweek yet. Pass --completed-gw explicitly if needed.")
    return max(finished)


def run(*args):
    command = [sys.executable, *args]
    print("\n▶", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch a completed GW, evaluate it if possible, and create next-GW recommendations."
    )
    parser.add_argument("--completed-gw", type=int, help="Completed GW; auto-detected from FPL by default")
    parser.add_argument("--validate", action="store_true", help="Also run the walk-forward backtest after the update")
    parser.add_argument("--no-optimize", action="store_true", help="Create predictions but skip squad/XI optimisation")
    args = parser.parse_args()

    completed = args.completed_gw or latest_finished_gw()
    target = completed + 1
    print(f"Running weekly update: completed GW{completed} → recommendations for GW{target}")

    run("src/ingest/fetch_gw.py", "--gw", str(completed))

    # Evaluation is available only if a prediction was previously saved for this GW.
    prediction = ROOT / "data" / "predictions" / f"predictions_gw{completed}.csv"
    if prediction.exists():
        run("src/evaluate/evaluate_model_weekly.py", "--gw", str(completed))
    else:
        print(f"\nℹ No saved GW{completed} prediction to evaluate; continuing.")

    run("src/features/update_features_weekly.py", "--gw", str(completed))
    run("src/models/train_model_weekly.py", "--target_gw", str(target))
    run("src/models/predict_next_gw.py", "--target_gw", str(target))
    if not args.no_optimize:
        run("src/optimization/select_squad.py", "--pred", f"data/predictions/predictions_gw{target}.csv")
    if args.validate:
        run("src/evaluate/walk_forward_validate.py", "--start_gw", "4")

    print(f"\n✅ Done. Review data/predictions/predictions_gw{target}.csv")


if __name__ == "__main__":
    main()

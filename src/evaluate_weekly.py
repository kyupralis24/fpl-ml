"""Fetch official results and evaluate the saved prediction in one command."""
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
        raise ValueError("No finished FPL gameweek found. Pass --gw once official points are available.")
    return max(finished)


def run(*args):
    command = [sys.executable, *args]
    print("\n▶", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch a finished GW, evaluate its prediction, and refresh the performance chart."
    )
    parser.add_argument("--gw", type=int, help="GW to evaluate; latest finished GW by default")
    args = parser.parse_args()
    gw = args.gw or latest_finished_gw()
    prediction = ROOT / "data" / "predictions" / f"predictions_gw{gw}.csv"
    if not prediction.exists():
        raise FileNotFoundError(
            f"No saved prediction for GW{gw}: {prediction}. "
            "Run the weekly pipeline before that GW's deadline."
        )
    print(f"Evaluating saved GW{gw} prediction against official FPL results.")
    run("src/ingest/fetch_gw.py", "--gw", str(gw))
    run("src/evaluate/evaluate_model_weekly.py", "--gw", str(gw))
    run("src/visualize/model_performance.py", "--no-show")
    print(f"\n✅ Evaluation saved to data/evaluation/eval_gw{gw}.csv")
    print("✅ Performance chart refreshed at data/plots/model_performance_trend.png")


if __name__ == "__main__":
    main()

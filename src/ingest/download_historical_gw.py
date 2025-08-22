# src/ingest/download_historical_gw.py
import os
import requests
import pandas as pd

SEASON = "2024-25"  # adjust if needed
URL = (
    f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
    f"master/data/{SEASON}/gws/merged_gw.csv"
)
OUT_PATH = f"data/raw/{SEASON}_merged_gw.csv"

def download():
    os.makedirs("data/raw", exist_ok=True)
    print(f"Downloading historical gameweek data for {SEASON}...")
    r = requests.get(URL)
    r.raise_for_status()
    with open(OUT_PATH, "wb") as f:
        f.write(r.content)
    print(f"✅ Saved historical data to {OUT_PATH}")

if __name__ == "__main__":
    download()
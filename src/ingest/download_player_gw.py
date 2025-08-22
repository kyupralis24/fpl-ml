# src/ingest/download_player_gw.py
import requests
import pandas as pd

BASE_URL = "https://fantasy.premierleague.com/api"

all_gw_data = []

# Premier League seasons usually have 38 GWs
for gw in range(1, 39):
    print(f"Fetching GW {gw}...")
    r = requests.get(f"{BASE_URL}/event/{gw}/live/")
    if r.status_code != 200:
        print(f"GW {gw} not available yet.")
        continue
    gw_data = r.json()["elements"]
    
    for player in gw_data:
        stats = player["stats"]
        stats["player_id"] = player["id"]
        stats["round"] = gw
        all_gw_data.append(stats)

df = pd.DataFrame(all_gw_data)
df.to_csv("data/raw/player_gw.csv", index=False)
print("✅ player_gw.csv saved with", len(df), "rows")
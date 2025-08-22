import requests
import pandas as pd
import os

def fetch_bootstrap():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()

def fetch_player_history(player_id):
    url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)

    bootstrap = fetch_bootstrap()
    players = pd.DataFrame(bootstrap['elements'])
    players.to_csv("data/raw/players.csv", index=False)

    all_histories = []
    for pid in players['id']:
        hist = fetch_player_history(pid)
        gw_df = pd.DataFrame(hist['history'])
        gw_df['player_id'] = pid
        all_histories.append(gw_df)

    history_df = pd.concat(all_histories, ignore_index=True)
    history_df.to_csv("data/raw/player_gw.csv", index=False)

    print("✅ Download complete: players.csv & player_gw.csv saved.")
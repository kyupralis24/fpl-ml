# src/ingest/fetch_gw.py
import argparse
import os
import time
import requests
import pandas as pd

API_BASE = "https://fantasy.premierleague.com/api"
RAW_DIR = "data/raw/current"

def get_json(url, retries=3, sleep=1.0):
    for i in range(retries):
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            return r.json()
        time.sleep(sleep)
    r.raise_for_status()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gw", type=int, required=True, help="Gameweek number to fetch (e.g., 1)")
    args = parser.parse_args()
    gw = args.gw

    os.makedirs(RAW_DIR, exist_ok=True)

    # Pull bootstrap (players/teams) + live GW stats + fixtures
    bootstrap = get_json(f"{API_BASE}/bootstrap-static/")
    live = get_json(f"{API_BASE}/event/{gw}/live/")
    fixtures = get_json(f"{API_BASE}/fixtures/?event={gw}")

    players = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"])
    types = pd.DataFrame(bootstrap["element_types"])
    fixtures_df = pd.DataFrame(fixtures) if fixtures else pd.DataFrame()

    # Keep handy maps
    type_map = dict(zip(types["id"], types["singular_name_short"]))  # 1->GK,2->DEF,3->MID,4->FWD
    team_name_map = dict(zip(teams["id"], teams["name"]))
    team_short_map = dict(zip(teams["id"], teams["short_name"]))

    # Team strength (can be useful features)
    team_strength = teams.set_index("id")[[
        "strength_overall_home","strength_overall_away",
        "strength_attack_home","strength_attack_away",
        "strength_defence_home","strength_defence_away"
    ]]

    # Build player lookup
    players_min = players[[
        "id","web_name","first_name","second_name","team","now_cost","element_type"
    ]].rename(columns={"id":"element","team":"team_id","element_type":"position_id"})
    players_min["position"] = players_min["position_id"].map(type_map)
    players_min["team_name"] = players_min["team_id"].map(team_name_map)
    players_min["team"] = players_min["team_id"].map(team_short_map)

    # Flatten live stats
    rows = []
    elements = live.get("elements", [])
    for el in elements:
        element_id = el.get("id")
        stats = el.get("stats", {}) or {}
        explain = el.get("explain", []) or []

        # Defaults
        was_home = None
        opponent_team = None
        fixture_id = None
        team_h_score = None
        team_a_score = None
        kickoff_time = None

        # If exactly one fixture, we can enrich with home/away/opponent/scores
        if len(explain) == 1:
            fixture_id = explain[0].get("fixture")
            was_home = explain[0].get("was_home")
            if fixtures_df.shape[0] > 0 and fixture_id in set(fixtures_df["id"].tolist()):
                fx = fixtures_df.loc[fixtures_df["id"] == fixture_id].iloc[0]
                team_h = fx["team_h"]
                team_a = fx["team_a"]
                team_h_score = fx.get("team_h_score")
                team_a_score = fx.get("team_a_score")
                kickoff_time = fx.get("kickoff_time")
                # Find player's team id
                p_row = players_min.loc[players_min["element"] == element_id]
                if not p_row.empty:
                    p_team = int(p_row.iloc[0]["team_id"])
                    if was_home is True:
                        opponent_team = int(team_a) if p_team == int(team_h) else int(team_h)
                    elif was_home is False:
                        opponent_team = int(team_h) if p_team == int(team_a) else int(team_a)

        row = {
            "element": element_id,
            "GW": gw,
            # totals across the gameweek (if DGW, FPL sums them here)
            "minutes": stats.get("minutes"),
            "goals_scored": stats.get("goals_scored"),
            "assists": stats.get("assists"),
            "clean_sheets": stats.get("clean_sheets"),
            "goals_conceded": stats.get("goals_conceded"),
            "saves": stats.get("saves"),
            "bps": stats.get("bps"),
            "bonus": stats.get("bonus"),
            "yellow_cards": stats.get("yellow_cards"),
            "red_cards": stats.get("red_cards"),
            "penalties_saved": stats.get("penalties_saved"),
            "penalties_missed": stats.get("penalties_missed"),
            "total_points": stats.get("total_points"),
            "was_home": was_home,
            "opponent_team": opponent_team,
            "fixture_id": fixture_id,
            "team_h_score": team_h_score,
            "team_a_score": team_a_score,
            "kickoff_time": kickoff_time,
        }
        rows.append(row)

    gw_stats = pd.DataFrame(rows)
    # Join identity/meta + price (keep FPL raw: e.g., 92 == £9.2m)
    df = gw_stats.merge(players_min, on="element", how="left")
    df = df.merge(team_strength, left_on="team_id", right_index=True, how="left")

    # Rename price to 'value' for compatibility with your pipeline
    df = df.rename(columns={"now_cost":"value", "web_name":"name"})
    # Keep nice order
    keep_cols = [
        "name","position","team","element","team_id","position_id","GW","value",
        "minutes","goals_scored","assists","clean_sheets","goals_conceded","saves",
        "bps","bonus","yellow_cards","red_cards","penalties_saved","penalties_missed",
        "total_points","was_home","opponent_team","fixture_id","team_h_score","team_a_score","kickoff_time",
        "strength_overall_home","strength_overall_away","strength_attack_home","strength_attack_away",
        "strength_defence_home","strength_defence_away"
    ]
    df = df[keep_cols]

    out_path = os.path.join(RAW_DIR, f"gw{gw}_player_stats.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Saved GW{gw} player stats to {out_path}")

if __name__ == "__main__":
    main()
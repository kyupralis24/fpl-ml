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

    # Fixture information is known before the deadline.  Persist a small,
    # score-free schedule snapshot so it can be used as a *next-GW* feature.
    # This also handles double and blank gameweeks without relying on the live
    # endpoint's ``explain`` payload.
    fixture_features = []
    for team_id in teams["id"]:
        team_fixtures = (
            fixtures_df[(fixtures_df["team_h"] == team_id) | (fixtures_df["team_a"] == team_id)]
            if not fixtures_df.empty else fixtures_df
        )
        home = team_fixtures[team_fixtures["team_h"] == team_id] if not team_fixtures.empty else team_fixtures
        difficulties = [
            row["team_h_difficulty"] if row["team_h"] == team_id else row["team_a_difficulty"]
            for _, row in team_fixtures.iterrows()
        ]
        fixture_features.append({
            "GW": gw,
            "team_id": team_id,
            "fixture_count": len(team_fixtures),
            "fixture_home_count": len(home),
            "fixture_difficulty": sum(difficulties) / len(difficulties) if difficulties else 0.0,
        })
    schedule = pd.DataFrame(fixture_features)
    schedule_path = os.path.join(RAW_DIR, f"gw{gw}_fixtures.csv")
    schedule.to_csv(schedule_path, index=False)

    # Flatten live stats
    rows = []
    elements = live.get("elements", [])
    if not elements:
        raise ValueError(
            f"No live player stats for GW{gw}. That gameweek has not started "
            "or the FPL live endpoint returned an empty elements list."
        )
    for el in elements:
        element_id = el.get("id")
        stats = el.get("stats", {}) or {}
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
        }
        rows.append(row)

    gw_stats = pd.DataFrame(rows)
    # Join identity/meta + price (keep FPL raw: e.g., 92 == £9.2m)
    df = gw_stats.merge(players_min, on="element", how="left")
    df = df.merge(team_strength, left_on="team_id", right_index=True, how="left")

    # Rename price to 'value' for compatibility with your pipeline
    df = df.rename(columns={"now_cost":"value", "web_name":"name"})
    df = df.merge(schedule, on=["GW", "team_id"], how="left")
    # Keep nice order
    keep_cols = [
        "name","position","team","element","team_id","position_id","GW","value",
        "minutes","goals_scored","assists","clean_sheets","goals_conceded","saves",
        "bps","bonus","yellow_cards","red_cards","penalties_saved","penalties_missed",
        "total_points","fixture_count","fixture_home_count","fixture_difficulty",
        "strength_overall_home","strength_overall_away","strength_attack_home","strength_attack_away",
        "strength_defence_home","strength_defence_away"
    ]
    df = df[keep_cols]

    out_path = os.path.join(RAW_DIR, f"gw{gw}_player_stats.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Saved GW{gw} player stats to {out_path}")
    print(f"✅ Saved GW{gw} fixture snapshot to {schedule_path}")

    # After you save the full GW data
    actual = df[["element", "name", "team", "position_id", "total_points"]].copy()
    actual.rename(columns={"element": "player_id"}, inplace=True)
    actual_path = f"data/raw/gw{gw}_actual_points.csv"
    os.makedirs(os.path.dirname(actual_path), exist_ok=True)
    actual.to_csv(actual_path, index=False)
    print(f"✅ Saved actual GW{gw} points for evaluation to {actual_path}")

if __name__ == "__main__":
    main()

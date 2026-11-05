"""
check_player.py  --  see a player's history next to his projection before you decide on an override.
Edit NAMES, then run it. Run make_projections.py first so projections_2026_27.csv is current.
"""
import pandas as pd

NAMES = ["Stephen Curry", "Domantas Sabonis", "LaMelo Ball", "Jayson Tatum", "Walker Kessler"]  # <-- edit this list

WEIGHTS = {"PTS": 1, "REB": 1.2, "AST": 1.5, "STL": 3, "BLK": 3, "TOV": -1}
raw = pd.read_csv("players_all_seasons.csv")
raw["FPTS_pg"] = sum(raw[s] * w for s, w in WEIGHTS.items())
proj = pd.read_csv("projections_2026_27.csv")

for name in NAMES:
    print("=" * 70)
    print(name)
    h = raw[raw["PLAYER_NAME"] == name].sort_values("SEASON")
    if h.empty:
        print("  not found in players_all_seasons.csv (check the spelling)")
        continue
    print(h[["SEASON", "TEAM_ABBREVIATION", "AGE", "GP", "MIN", "FPTS_pg"]].tail(6).round(1).to_string(index=False))
    p = proj[proj["PLAYER_NAME"] == name]
    if p.empty:
        print("  not in projections_2026_27.csv")
    else:
        print("  PROJECTION:")
        print(p[["proj_min", "proj_fpts_pg", "proj_gp", "proj_total"]].round(1).to_string(index=False))

top = proj.sort_values("proj_total", ascending=False).head(50)
print("=" * 70)
print("Average projected games, top 50 by total:", round(top["proj_gp"].mean(), 1))
print("(in the backtest, players who averaged 32+ minutes actually played about 62 games on average)")
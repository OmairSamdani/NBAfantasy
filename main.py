import pandas as pd

LEAGUE_TEAMS = 12   # <-- change to your league size
ACTIVE_SPOTS = 10   # <-- starting lineup slots per team (not bench)
ONLY_PLAYED_THIS_SEASON = False  # True hides players with no 2026-27 games yet
TOP_N = 60

proj = pd.read_csv("projections_2026_27.csv")
if ONLY_PLAYED_THIS_SEASON:
    proj = proj[proj["gp_cur"] > 0].copy()

# replacement level = per-game production of the last starter-level player in your league
repl_rank = LEAGUE_TEAMS * ACTIVE_SPOTS
by_pg = proj.sort_values("proj_fpts_pg", ascending=False).reset_index(drop=True)
repl_pg = by_pg.loc[min(repl_rank, len(by_pg) - 1), "proj_fpts_pg"]

# a game he misses is filled by a replacement player, so it costs (his pg - replacement pg), not all of his pg
proj["vor"] = (proj["proj_fpts_pg"] - repl_pg) * proj["proj_gp"]
proj["rank_total"] = proj["proj_total"].rank(ascending=False).astype(int)
proj = proj.sort_values("vor", ascending=False).reset_index(drop=True)
proj["rank_vor"] = proj.index + 1

print(f"Replacement level: {repl_pg:.1f} FPTS/game (the #{repl_rank} player by per-game production)\n")
cols = ["rank_vor", "rank_total", "PLAYER_NAME", "TEAM_ABBREVIATION", "proj_min",
        "proj_fpts_pg", "proj_gp", "proj_total", "vor"]
print(proj[cols].head(TOP_N).round(1).to_string(index=False))
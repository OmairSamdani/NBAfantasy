import time
import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats

seasons = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]
frames = []

for s in seasons:
    d = leaguedashplayerstats.LeagueDashPlayerStats(
        season=s, per_mode_detailed="PerGame", timeout=60
    ).get_data_frames()[0]
    d["SEASON"] = s
    frames.append(d)
    print("pulled", s, len(d))
    time.sleep(2)  # don't hammer the API

pd.concat(frames).to_csv("players_all_seasons.csv", index=False)
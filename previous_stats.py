import time
import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats

seasons = [
    "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
    "2025-26", "2026-27",
]
frames = []

for s in seasons:
    for attempt in range(3):  # retry up to 3 times if the API times out
        try:
            d = leaguedashplayerstats.LeagueDashPlayerStats(
                season=s, per_mode_detailed="PerGame", timeout=60
            ).get_data_frames()[0]
            d["SEASON"] = s
            frames.append(d)
            print("pulled", s, len(d))
            break
        except Exception as e:
            print("retrying", s, "-", e)
            time.sleep(5)
    time.sleep(2)

pd.concat(frames).to_csv("players_all_seasons.csv", index=False)
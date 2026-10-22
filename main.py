import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats

stats = leaguedashplayerstats.LeagueDashPlayerStats(
    season="2025-26", per_mode_detailed="PerGame", timeout=60
)
df = stats.get_data_frames()[0]

print(df.head())
print(df.columns.tolist())

df.to_csv("players_2025_26.csv", index=False)
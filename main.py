import pandas as pd

df = pd.read_csv("players_2025_26.csv")

# Keep players with real playing time
df = df[(df["MIN"] > 20) & (df["GP"] > 20)].copy()

# League's scoring weights
weights = {
    "PTS": 1,
    "REB": 1.2,
    "AST": 1.5,
    "STL": 3,
    "BLK": 3,
    "TOV": -1,
}

# Fantasy points per game = sum of (stat * weight)
df["FPTS_pg"] = 0
for stat, w in weights.items():
    df["FPTS_pg"] += df[stat] * w

# Fantasy points total (all games played)
df["FPTS_total"] = df["FPTS_pg"] * df["GP"]

top = df.sort_values("FPTS_pg", ascending=False)
print(top[["PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "FPTS_pg", "FPTS_total"]].head(30).to_string(index=False))
import pandas as pd

raw = pd.read_csv("players_all_seasons.csv")

weights = {"PTS": 1, "REB": 1.2, "AST": 1.5, "STL": 3, "BLK": 3, "TOV": -1}
raw["FPTS_pg"] = sum(raw[s] * w for s, w in weights.items())
raw = raw[raw["MIN"] > 0].copy()
raw["FPTS_pm"] = raw["FPTS_pg"] / raw["MIN"]
raw["TOT_MIN"] = raw["MIN"] * raw["GP"]

# 2026-27 is unfinished, so leave it out of the curve
raw = raw[raw["SEASON"] != "2026-27"].copy()
raw["year"] = raw["SEASON"].str[:4].astype(int)

# pair each player-season with that same player's NEXT season
a = raw[["PLAYER_ID", "year", "AGE", "FPTS_pm", "TOT_MIN"]]
b = a.copy()
b["year"] = b["year"] - 1  # shift back one year so it lines up with "this" season
pairs = a.merge(b, on=["PLAYER_ID", "year"], suffixes=("", "_next"))

# only trust seasons with a real sample of minutes
pairs = pairs[(pairs["TOT_MIN"] >= 500) & (pairs["TOT_MIN_next"] >= 500)].copy()

# % change in production per minute, weighted by the smaller minutes sample
pairs["pct"] = pairs["FPTS_pm_next"] / pairs["FPTS_pm"] - 1
pairs["w"] = pairs[["TOT_MIN", "TOT_MIN_next"]].min(axis=1)
pairs["wpct"] = pairs["pct"] * pairs["w"]
pairs["age"] = pairs["AGE"].clip(19, 36)

g = pairs.groupby("age").agg(wpct=("wpct", "sum"), w=("w", "sum"), n=("pct", "size"))
g["pct_change"] = g["wpct"] / g["w"]
g["smooth"] = g["pct_change"].rolling(3, center=True, min_periods=1).mean()

curve = g[["n", "pct_change", "smooth"]].round(3).reset_index()
print(curve.to_string(index=False))
curve.to_csv("age_curve.csv", index=False)
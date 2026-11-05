import pandas as pd

df = pd.read_csv("players_all_seasons.csv")
df = df[(df["MIN"] > 15) & (df["GP"] > 20)].copy()

weights = {"PTS": 1, "REB": 1.2, "AST": 1.5, "STL": 3, "BLK": 3, "TOV": -1}
df["FPTS_pg"] = sum(df[s] * w for s, w in weights.items())

wide = df.pivot_table(
    index=["PLAYER_ID", "PLAYER_NAME"], columns="SEASON", values="FPTS_pg"
).reset_index()

def weighted_avg(row, cols, w):
    pairs = [(row[c], wt) for c, wt in zip(cols, w) if pd.notna(row[c])]
    if not pairs:
        return None
    return sum(v * wt for v, wt in pairs) / sum(wt for _, wt in pairs)

def backtest(target, prior, w):
    t = wide.dropna(subset=[target, prior[0]]).copy()
    t["pred"] = t.apply(lambda r: weighted_avg(r, prior, w), axis=1)
    return (t["pred"] - t[target]).abs().mean()

tests = [
    ("2025-26", ["2024-25", "2023-24", "2022-23"]),
    ("2024-25", ["2023-24", "2022-23", "2021-22"]),
]
weight_sets = {
    "last only": [1, 0, 0],
    "70/20/10": [0.7, 0.2, 0.1],
    "50/30/20": [0.5, 0.3, 0.2],
    "equal": [1/3, 1/3, 1/3],
}

for name, w in weight_sets.items():
    errs = [backtest(t, p, w) for t, p in tests]
    print(name, [round(e, 2) for e in errs], "avg", round(sum(errs) / len(errs), 2))
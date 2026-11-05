import numpy as np
import pandas as pd

WEIGHTS = {"PTS": 1, "REB": 1.2, "AST": 1.5, "STL": 3, "BLK": 3, "TOV": -1}
K_SHRINK = 400
SEASON_W = {1: 0.7, 2: 0.2, 3: 0.1}  # most recent season played = 1
GP_OWN_WEIGHT = 0.6
SEASON_GAMES = 82
TARGETS = [2021, 2022, 2023, 2024, 2025]  # 2021 means the 2021-22 season

raw = pd.read_csv("players_all_seasons.csv")
raw["FPTS_pg"] = sum(raw[s] * w for s, w in WEIGHTS.items())
raw = raw[raw["MIN"] > 0].copy()
raw["FPTS_pm"] = raw["FPTS_pg"] / raw["MIN"]
raw["TOT_MIN"] = raw["MIN"] * raw["GP"]
raw["year"] = raw["SEASON"].str[:4].astype(int)
raw["gp_frac"] = raw["GP"] / raw.groupby("year")["GP"].transform("max")


# ---------- age curve (built only from data before the target season) ----------
def build_curve(data):
    a = data[["PLAYER_ID", "year", "AGE", "FPTS_pm", "TOT_MIN"]]
    b = a.copy()
    b["year"] = b["year"] - 1
    p = a.merge(b, on=["PLAYER_ID", "year"], suffixes=("", "_next"))
    p = p[(p["TOT_MIN"] >= 500) & (p["TOT_MIN_next"] >= 500)].copy()
    p["pct"] = p["FPTS_pm_next"] / p["FPTS_pm"] - 1
    p["w"] = p[["TOT_MIN", "TOT_MIN_next"]].min(axis=1)
    p["wpct"] = p["pct"] * p["w"]
    p["age"] = p["AGE"].clip(19, 36).round().astype(int)
    g = p.groupby("age")[["wpct", "w"]].sum()
    smooth = (g["wpct"] / g["w"]).rolling(3, center=True, min_periods=1).mean()
    return {int(k): v for k, v in smooth.items()}


def age_factor(curve, age, years_back):
    f = 1.0
    a = int(round(age))
    for i in range(years_back):
        f *= 1 + curve.get(min(max(a + i, 19), 36), 0.0)
    return f


def rate_for(h, curve):
    d = h.copy()
    if curve:
        d["adj"] = [age_factor(curve, a, b) for a, b in zip(d["AGE"], d["back"])]
    else:
        d["adj"] = 1.0
    d["w"] = d["rank"].map(SEASON_W) * d["TOT_MIN"]
    d["wr"] = d["FPTS_pm"] * d["adj"] * d["w"]
    g = d.groupby("PLAYER_ID").agg(wr=("wr", "sum"), w=("w", "sum"), tm=("TOT_MIN", "sum"))
    lg = g["wr"].sum() / g["w"].sum()
    return ((g["wr"] / g["w"]) * g["tm"] + lg * K_SHRINK) / (g["tm"] + K_SHRINK)


# ---------- games-played model (same as make_projections.py) ----------
def games_features(d):
    age_c = (d["AGE"] - 27) / 10
    return np.column_stack([np.ones(len(d)), d["gp_frac"], d["gp_prev"],
                            d["MIN"] / 10, age_c, age_c ** 2])


def build_games_table(rows):
    a = rows[["PLAYER_ID", "year", "AGE", "MIN", "TOT_MIN", "gp_frac"]].copy()
    nxt = a[["PLAYER_ID", "year", "gp_frac"]].rename(columns={"gp_frac": "y"})
    nxt["year"] = nxt["year"] - 1
    prv = a[["PLAYER_ID", "year", "gp_frac"]].rename(columns={"gp_frac": "gp_prev"})
    prv["year"] = prv["year"] + 1
    d = a.merge(nxt, on=["PLAYER_ID", "year"]).merge(prv, on=["PLAYER_ID", "year"], how="left")
    d["gp_prev"] = d["gp_prev"].fillna(d["gp_frac"])
    return d[d["TOT_MIN"] >= 500].copy()


# ---------- project season T using only earlier data ----------
def project(T):
    hist = raw[raw["year"] < T]
    last = hist[hist["year"] == T - 1]
    pool = last.loc[last["TOT_MIN"] >= 1000, "PLAYER_ID"].unique()

    h = hist[hist["PLAYER_ID"].isin(pool)].copy()
    h["rank"] = h.groupby("PLAYER_ID")["year"].rank(ascending=False, method="first").astype(int)
    h = h[h["rank"] <= 3].copy()
    h["back"] = T - h["year"]

    out = pd.DataFrame(index=pd.Index(sorted(pool), name="PLAYER_ID"))
    out["rate_noage"] = rate_for(h, {})
    out["rate_age"] = rate_for(h, build_curve(hist))

    l1 = h[h["rank"] == 1].set_index("PLAYER_ID")
    out["base_min"] = l1["MIN"]
    out["last_pg"] = l1["FPTS_pg"]
    out["last_gp"] = l1["GP"]

    gp_hist = h.groupby("PLAYER_ID")["GP"].mean()
    lg_gp = h.loc[h["MIN"] > 15, "GP"].mean()
    out["gp_old"] = GP_OWN_WEIGHT * gp_hist + (1 - GP_OWN_WEIGHT) * lg_gp

    gt = build_games_table(hist)
    beta = np.linalg.lstsq(games_features(gt), gt["y"], rcond=None)[0]
    lr = l1[["AGE", "MIN", "gp_frac"]].copy()
    prev = h[h["rank"] == 2].set_index("PLAYER_ID")["gp_frac"]
    lr["gp_prev"] = prev.reindex(lr.index).fillna(lr["gp_frac"])
    pf = np.clip(games_features(lr) @ beta, 0.25, 0.95)
    out["gp_new"] = pd.Series(pf * SEASON_GAMES, index=lr.index)
    return out


def evaluate(T):
    p = project(T)
    act = raw[raw["year"] == T].set_index("PLAYER_ID")[["FPTS_pg", "GP", "MIN"]]
    act.columns = ["act_pg", "act_gp", "act_min"]
    d = p.join(act, how="inner")
    d["act_total"] = d["act_pg"] * d["act_gp"]

    methods = {
        "last": (d["last_pg"], d["last_gp"]),
        "old": (d["rate_noage"] * d["base_min"], d["gp_old"]),
        "old+games": (d["rate_noage"] * d["base_min"], d["gp_new"]),
        "old+age": (d["rate_age"] * d["base_min"], d["gp_old"]),
        "old+age+games": (d["rate_age"] * d["base_min"], d["gp_new"]),
    }
    pgmask = (d["act_gp"] >= 20) & (d["act_min"] > 15)
    rows = []
    for name, (pg, gp) in methods.items():
        total = pg * gp
        top = total.sort_values(ascending=False).head(30).index
        rows.append(dict(
            season=f"{T}-{str(T + 1)[2:]}", method=name, n=len(d),
            pg_miss=(pg - d["act_pg"])[pgmask].abs().mean(),
            gp_miss=(gp - d["act_gp"]).abs().mean(),
            total_miss=(total - d["act_total"]).abs().mean(),
            rank_corr=total.rank().corr(d["act_total"].rank()),
            top30=d.loc[top, "act_total"].mean(),
        ))
    perfect = d["act_total"].nlargest(30).mean()
    return rows, perfect


all_rows, perfects = [], []
for T in TARGETS:
    try:
        r, perfect = evaluate(T)
        all_rows += r
        perfects.append(perfect)
    except Exception as e:
        print(f"skipped {T}-{str(T + 1)[2:]}: {e}")

if not all_rows:
    raise SystemExit("No seasons could be tested. Is players_all_seasons.csv back to 2015-16?")

res = pd.DataFrame(all_rows)
order = ["last", "old", "old+games", "old+age", "old+age+games"]

print("=== AVERAGE ACROSS", res["season"].nunique(), "SEASONS (lower is better, except rank_corr and top30) ===")
avg = res.groupby("method")[["pg_miss", "gp_miss", "total_miss", "rank_corr", "top30"]].mean().reindex(order)
print(avg.round(2).to_string())
print(f"(a perfect draft of the real top 30 would average {np.mean(perfects):.0f})")

print("\n=== TOTAL MISS BY SEASON (FPTS, lower is better) ===")
print(res.pivot(index="method", columns="season", values="total_miss").reindex(order).round(0).to_string())

print("\n=== PER-GAME MISS BY SEASON (FPTS/game, lower is better) ===")
print(res.pivot(index="method", columns="season", values="pg_miss").reindex(order).round(2).to_string())

# ---- bias check: does each games method run high or low for different minute tiers? ----
parts = []
for T in TARGETS:
    p = project(T)
    act = raw[raw["year"] == T].set_index("PLAYER_ID")[["GP"]].rename(columns={"GP": "act_gp"})
    parts.append(p.join(act, how="inner"))
b = pd.concat(parts)
b["tier"] = pd.cut(b["base_min"], [0, 20, 28, 32, 60], labels=["under 20", "20-28", "28-32", "32+"])
tiers = b.groupby("tier", observed=True).agg(
    n=("act_gp", "size"), actual=("act_gp", "mean"), model=("gp_new", "mean"),
    simple=("gp_old", "mean"), last_year=("last_gp", "mean"))
print("\n=== AVERAGE GAMES PLAYED BY LAST-SEASON MINUTES (does the model run low for stars?) ===")
print(tiers.round(1).to_string())
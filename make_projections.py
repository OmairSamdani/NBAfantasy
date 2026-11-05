"""
make_projections.py  ->  writes projections_2026_27.csv

Inputs (all in the project folder):
  players_all_seasons.csv   (required)  from previous_stats.py
  age_curve.csv             (optional)  from age_curve.py
  draft_history.csv         (optional)  from pull_draft.py  (needed for the rookie prior)
  minutes_overrides.csv     (optional)  your manual minutes/games, columns: PLAYER_NAME,minutes,games

How a projection is built:
  projected FPTS/game = FPTS-per-minute rate x projected minutes
  projected total     = FPTS/game x projected games

  Veterans: rate comes from their last 3 seasons (age-adjusted, recency-weighted,
            shrunk toward league average), then updated with 2026-27 production so far.
  Rookies:  no history, so the rate starts from what past rookies drafted at a similar
            pick produced, then gets updated with 2026-27 production so far.
  Games:    learned from history (last two seasons' games, minutes, age).
  Minutes:  last season's minutes (rookies: draft-pick average), blended with 2026-27
            minutes so far. Minutes are trusted faster than production.
"""
import os
import numpy as np
import pandas as pd

# ===================== settings you can tweak =====================
CUR = "2026-27"
CUR_YEAR = int(CUR[:4])
WEIGHTS = {"PTS": 1, "REB": 1.2, "AST": 1.5, "STL": 3, "BLK": 3, "TOV": -1}
SEASON_W = {1: 0.7, 2: 0.2, 3: 0.1}  # most recent season played = 1
USE_AGE_CURVE = True  # set False to switch the age adjustment off
K_SHRINK = 400  # minutes of "league average" mixed into a veteran's history
K_VET = 2000  # minutes the veteran prior is worth vs. this season's minutes
K_ROOKIE = 1000  # same idea for rookies (their prior is weaker, so lower)
G_MIN = 8  # after this many games, current minutes count 50/50 with the prior
ROOKIE_MIN_MPG = 12  # new players must average this many minutes to be projected
ROOKIE_MIN_GP = 3  # ...and have played at least this many games
GP_OWN_WEIGHT = 0.6  # fallback only: weight on own games history (rest = league avg)
USE_GAMES_MODEL = True  # learn games played from history (last 2 seasons, minutes, age)
SEASON_GAMES = 82

# ===================== load + basic stats =====================
raw = pd.read_csv("players_all_seasons.csv")
raw["FPTS_pg"] = sum(raw[s] * w for s, w in WEIGHTS.items())
raw = raw[raw["MIN"] > 0].copy()
raw["FPTS_pm"] = raw["FPTS_pg"] / raw["MIN"]
raw["TOT_MIN"] = raw["MIN"] * raw["GP"]
raw["year"] = raw["SEASON"].str[:4].astype(int)
# share of the season played (handles shortened seasons like 2019-20 and 2020-21)
raw["gp_frac"] = raw["GP"] / raw.groupby("year")["GP"].transform("max")

past = raw[raw["year"] < CUR_YEAR].copy()
cur = raw[raw["year"] == CUR_YEAR].copy()
if cur.empty:
    print(f"NOTE: no {CUR} rows found. Run previous_stats.py again.")

latest = (raw.sort_values("SEASON").groupby("PLAYER_ID").last()
          [["PLAYER_NAME", "TEAM_ABBREVIATION"]].reset_index())

# ===================== age curve =====================
curve = {}
if USE_AGE_CURVE and os.path.exists("age_curve.csv"):
    ac = pd.read_csv("age_curve.csv")
    curve = {int(a): v for a, v in zip(ac["age"], ac["smooth"])}
elif USE_AGE_CURVE:
    print("NOTE: age_curve.csv not found, skipping age adjustment.")


def age_factor(age, years_back):
    """Move a past season's per-minute rate forward to this season's age."""
    f = 1.0
    a = int(round(age))
    for i in range(years_back):
        f *= 1 + curve.get(min(max(a + i, 19), 36), 0.0)
    return f


# ===================== games-played model =====================
def games_features(d):
    age_c = (d["AGE"] - 27) / 10
    return np.column_stack([np.ones(len(d)), d["gp_frac"], d["gp_prev"],
                            d["MIN"] / 10, age_c, age_c ** 2])


def build_games_table(rows):
    """One row per player-season that has a following season (the target)."""
    a = rows[["PLAYER_ID", "year", "AGE", "MIN", "TOT_MIN", "gp_frac"]].copy()
    nxt = a[["PLAYER_ID", "year", "gp_frac"]].rename(columns={"gp_frac": "y"})
    nxt["year"] = nxt["year"] - 1
    prv = a[["PLAYER_ID", "year", "gp_frac"]].rename(columns={"gp_frac": "gp_prev"})
    prv["year"] = prv["year"] + 1
    d = a.merge(nxt, on=["PLAYER_ID", "year"]).merge(prv, on=["PLAYER_ID", "year"], how="left")
    d["gp_prev"] = d["gp_prev"].fillna(d["gp_frac"])
    return d[d["TOT_MIN"] >= 500].copy()


beta = None
if USE_GAMES_MODEL:
    gt = build_games_table(past)
    if len(gt) >= 300:
        test_year = CUR_YEAR - 2  # predicts the last completed season
        tr, te = gt[gt["year"] < test_year], gt[gt["year"] == test_year]
        if len(tr) >= 200 and len(te) >= 50:
            b_ = np.linalg.lstsq(games_features(tr), tr["y"], rcond=None)[0]
            p_ = np.clip(games_features(te) @ b_, 0.25, 0.95)
            err_model = (p_ - te["y"]).abs().mean() * SEASON_GAMES
            err_naive = (te["gp_frac"] - te["y"]).abs().mean() * SEASON_GAMES
            print(f"\nGames model check on {test_year + 1}-{str(test_year + 2)[2:]}: "
                  f"avg miss {err_model:.1f} games (just repeating last year: {err_naive:.1f})")
        beta = np.linalg.lstsq(games_features(gt), gt["y"], rcond=None)[0]
    else:
        print("NOTE: not enough history for the games model, using the simple fallback.")

# ===================== veterans =====================
# who counts as a veteran: played in the last 2 completed seasons, OR is playing this
# season and has any NBA history (this catches players returning after time away)
recent_ids = set(past.loc[past["year"] >= CUR_YEAR - 2, "PLAYER_ID"])
returning_ids = set(cur["PLAYER_ID"]) & set(past["PLAYER_ID"])
vet_pool = recent_ids | returning_ids

# use each player's last 3 seasons THEY PLAYED (rank 1 = most recent)
h = past[past["PLAYER_ID"].isin(vet_pool)].copy()
h["rank"] = h.groupby("PLAYER_ID")["year"].rank(ascending=False, method="first").astype(int)
h = h[h["rank"] <= 3].copy()
h["back"] = CUR_YEAR - h["year"]  # calendar years back, used for the age adjustment
h["adj"] = [age_factor(a, b) for a, b in zip(h["AGE"], h["back"])]
h["w"] = h["rank"].map(SEASON_W) * h["TOT_MIN"]
h["wr"] = h["FPTS_pm"] * h["adj"] * h["w"]

vet = h.groupby("PLAYER_ID").agg(
    wr=("wr", "sum"), w=("w", "sum"), tm=("TOT_MIN", "sum"), gp_hist=("GP", "mean")
).reset_index()
vet["rate_hist"] = vet["wr"] / vet["w"]
lg_rate = vet["wr"].sum() / vet["w"].sum()
vet["rate_prior"] = (vet["rate_hist"] * vet["tm"] + lg_rate * K_SHRINK) / (vet["tm"] + K_SHRINK)

# baseline minutes = minutes per game in the most recent season they played
last_min = h[h["rank"] == 1].set_index("PLAYER_ID")["MIN"]
vet["base_min"] = vet["PLAYER_ID"].map(last_min).fillna(10)

# this season so far
cur_small = cur[["PLAYER_ID", "GP", "MIN", "FPTS_pm", "TOT_MIN"]].rename(
    columns={"GP": "gp_cur", "MIN": "min_cur", "FPTS_pm": "rate_cur", "TOT_MIN": "tm_cur"})
vet = vet.merge(cur_small, on="PLAYER_ID", how="left")
vet["gp_cur"] = vet["gp_cur"].fillna(0)
vet["tm_cur"] = vet["tm_cur"].fillna(0)
vet["rate_cur"] = vet["rate_cur"].fillna(0)

# update the rate with 2026-27 production (more minutes played = more weight)
vet["rate"] = (vet["rate_prior"] * K_VET + vet["rate_cur"] * vet["tm_cur"]) / (K_VET + vet["tm_cur"])

# update minutes with 2026-27 minutes (trusted faster than production)
w_min = vet["gp_cur"] / (vet["gp_cur"] + G_MIN)
vet["proj_min"] = (1 - w_min) * vet["base_min"] + w_min * vet["min_cur"].fillna(vet["base_min"])

# games played: own history pulled toward the league-typical rotation player
lg_gp = h.loc[h["MIN"] > 15, "GP"].mean()
vet["proj_gp"] = GP_OWN_WEIGHT * vet["gp_hist"] + (1 - GP_OWN_WEIGHT) * lg_gp  # fallback
if beta is not None:
    last_rows = h[h["rank"] == 1].copy()
    prev_rows = h[h["rank"] == 2][["PLAYER_ID", "gp_frac"]].rename(columns={"gp_frac": "gp_prev"})
    last_rows = last_rows.merge(prev_rows, on="PLAYER_ID", how="left")
    last_rows["gp_prev"] = last_rows["gp_prev"].fillna(last_rows["gp_frac"])
    pf = np.clip(games_features(last_rows) @ beta, 0.25, 0.95)
    gp_pred = dict(zip(last_rows["PLAYER_ID"], pf * SEASON_GAMES))
    vet["proj_gp"] = vet["PLAYER_ID"].map(gp_pred).fillna(vet["proj_gp"])

vet = vet.merge(latest, on="PLAYER_ID", how="left")
vet["type"] = "vet"
vet["bucket"] = ""

# ===================== rookie prior (from draft pick) =====================
dr = None
if os.path.exists("draft_history.csv"):
    d = pd.read_csv("draft_history.csv")
    need = {"PERSON_ID", "SEASON", "OVERALL_PICK"}
    if need.issubset(d.columns):
        d = d.rename(columns={"PERSON_ID": "PLAYER_ID"})
        d["OVERALL_PICK"] = pd.to_numeric(d["OVERALL_PICK"], errors="coerce")
        d["draft_year"] = pd.to_numeric(d["SEASON"], errors="coerce")
        dr = (d.sort_values("draft_year").drop_duplicates("PLAYER_ID", keep="last")
        [["PLAYER_ID", "OVERALL_PICK", "draft_year"]])
    else:
        print("NOTE: draft_history.csv is missing columns:", need - set(d.columns))
else:
    print("NOTE: draft_history.csv not found. Rookies get an average-rookie prior.")


def bucket(pick):
    if pd.isna(pick):
        return "undrafted"
    if pick <= 5:
        return "1-5"
    if pick <= 14:
        return "6-14"
    if pick <= 30:
        return "15-30"
    return "31+"


# what past rookies actually did in their first season
first = raw.groupby("PLAYER_ID")["year"].min().rename("first_year").reset_index()
rk = past.merge(first, on="PLAYER_ID")
rk = rk[(rk["year"] == rk["first_year"]) & (rk["year"] > raw["year"].min())].copy()
if dr is not None:
    rk = rk.merge(dr, on="PLAYER_ID", how="left")
    rk = rk[(rk["draft_year"] == rk["year"]) | rk["draft_year"].isna()].copy()
else:
    rk["OVERALL_PICK"] = np.nan
rk["bucket"] = rk["OVERALL_PICK"].map(bucket)

rk_q = rk[rk["TOT_MIN"] >= 200].copy()
rk_q["wr"] = rk_q["FPTS_pm"] * rk_q["TOT_MIN"]
labels = ["1-5", "6-14", "15-30", "31+", "undrafted"]
if len(rk_q) > 0:
    pt = rk_q.groupby("bucket").agg(
        n=("PLAYER_ID", "size"), wr=("wr", "sum"), tm=("TOT_MIN", "sum"), mpg=("MIN", "mean"))
    pt["rate"] = pt["wr"] / pt["tm"]
    all_rate = rk_q["wr"].sum() / rk_q["TOT_MIN"].sum()
    all_mpg = rk_q["MIN"].mean()
else:
    pt = pd.DataFrame(columns=["n", "rate", "mpg"])
    all_rate, all_mpg = lg_rate, 15.0
pt = pt.reindex(labels)
pt["n"] = pt["n"].fillna(0).astype(int)
pt["rate"] = pt["rate"].fillna(all_rate)
pt["mpg"] = pt["mpg"].fillna(all_mpg)
print("\nRookie prior by draft pick (past first-year players):")
print(pt[["n", "rate", "mpg"]].round(3).to_string())

# ===================== current rookies / new players =====================
new = cur[~cur["PLAYER_ID"].isin(set(vet["PLAYER_ID"]))].copy()
new = new[(new["MIN"] >= ROOKIE_MIN_MPG) & (new["GP"] >= ROOKIE_MIN_GP)].copy()
if dr is not None:
    new = new.merge(dr[["PLAYER_ID", "OVERALL_PICK"]], on="PLAYER_ID", how="left")
else:
    new["OVERALL_PICK"] = np.nan
new["bucket"] = new["OVERALL_PICK"].map(bucket)
new["rate_prior"] = new["bucket"].map(pt["rate"])
new["base_min"] = new["bucket"].map(pt["mpg"])
new["gp_cur"] = new["GP"]
new["rate"] = (new["rate_prior"] * K_ROOKIE + new["FPTS_pm"] * new["TOT_MIN"]) / (K_ROOKIE + new["TOT_MIN"])
w_new = new["GP"] / (new["GP"] + G_MIN)
new["proj_min"] = (1 - w_new) * new["base_min"] + w_new * new["MIN"]
new["proj_gp"] = lg_gp
new["type"] = "rookie"

# ===================== combine =====================
keep = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "type", "bucket",
        "gp_cur", "rate", "proj_min", "proj_gp"]
allp = pd.concat([vet[keep], new[keep]], ignore_index=True)

# your manual overrides (minutes and/or games)
if os.path.exists("minutes_overrides.csv"):
    ov = pd.read_csv("minutes_overrides.csv")
    ov["PLAYER_NAME"] = ov["PLAYER_NAME"].astype(str).str.strip()
    if "minutes" in ov.columns:
        m = dict(zip(ov["PLAYER_NAME"], ov["minutes"]))
        allp["proj_min"] = allp["PLAYER_NAME"].map(m).fillna(allp["proj_min"])
    if "games" in ov.columns:
        g = dict(zip(ov["PLAYER_NAME"], ov["games"]))
        allp["proj_gp"] = allp["PLAYER_NAME"].map(g).fillna(allp["proj_gp"])
    missing = set(ov["PLAYER_NAME"]) - set(allp["PLAYER_NAME"])
    if missing:
        print("\nWARNING: these override names matched nobody (check spelling):", sorted(missing))

allp["proj_fpts_pg"] = allp["rate"] * allp["proj_min"]
allp["proj_total"] = allp["proj_fpts_pg"] * allp["proj_gp"]

out_cols = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "type", "gp_cur",
            "proj_min", "proj_fpts_pg", "proj_gp", "proj_total"]
allp[out_cols].to_csv("projections_2026_27.csv", index=False)

# ===================== sanity checks =====================
print("\nTop rookies / new players by projected FPTS per game:")
r = allp[allp["type"] == "rookie"].sort_values("proj_fpts_pg", ascending=False)
print(r[["PLAYER_NAME", "TEAM_ABBREVIATION", "bucket", "gp_cur", "proj_min", "proj_fpts_pg"]]
      .head(15).round(1).to_string(index=False))

print("\nProjected minutes per team, counting only players who have played this season")
print("(rough check: should land near 240 to ~300. Far above = too many players projected big minutes)")
act = allp[allp["gp_cur"] > 0]
print(act.groupby("TEAM_ABBREVIATION")["proj_min"].sum().round().sort_values().to_string())

print("\nSaved projections_2026_27.csv with", len(allp), "players")
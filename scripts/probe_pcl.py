"""
Probe: does adding PCL park exposure help?

Static IL/PCL mapping based on ~2015-2020 modal org affiliates.
PCL is historically power-inflating due to elevation/dry-air parks
(Reno, Albuquerque, Las Vegas, El Paso, Colorado Springs).

If ISO's translation coefficient is partly PCL park noise, adding
PCL_share (fraction of pre-debut AAA PA at a PCL-affiliated org)
should either help predict MLB wRC+ directly OR attenuate the ISO
coefficient and stabilize other coefficients.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RAW_AAA = ROOT / "data" / "raw" / "aaa"
DATA = ROOT / "data" / "processed" / "matched_players.csv"
OUTCOME = "mlb_wRC+"
BASE_SKILL = ["aaa_BB%_zseason", "aaa_K%_zseason", "aaa_ISO_zseason", "aaa_BABIP_zseason", "age_at_debut"]

# Modal 2015-2020 org → AAA league (International League vs Pacific Coast League)
ORG_TO_LEAGUE = {
    "ATL": "IL", "BAL": "IL", "BOS": "IL", "CHW": "IL", "CIN": "IL",
    "CLE": "IL", "DET": "IL", "MIN": "IL", "NYY": "IL", "PHI": "IL",
    "PIT": "IL", "TBR": "IL", "TOR": "IL", "WSN": "IL",   # WSN pre-2019
    "ARI": "PCL", "CHC": "PCL", "COL": "PCL", "HOU": "PCL", "KCR": "PCL",
    "LAA": "PCL", "LAD": "PCL", "MIA": "PCL", "MIL": "PCL", "NYM": "PCL",
    "OAK": "PCL", "SDP": "PCL", "SEA": "PCL", "SFG": "PCL", "STL": "PCL",
    "TEX": "PCL",
}

# Load raw AAA and compute per-player PCL PA share (pre-debut only, joined with matched cohort)
matched = pd.read_csv(DATA)
matched["playerid"] = matched["playerid"].astype(str)
debut = dict(zip(matched["playerid"], matched["mlb_debut_year"]))

aaa = pd.concat([pd.read_csv(p) for p in RAW_AAA.glob("*.csv")], ignore_index=True)
aaa["PlayerId"] = aaa["PlayerId"].astype(str)
aaa["Season"] = pd.to_numeric(aaa["Season"], errors="coerce").astype("Int64")
aaa["PA"] = pd.to_numeric(aaa["PA"], errors="coerce")


def team_league(row):
    if pd.isna(row["Team"]):
        return None
    t = str(row["Team"]).split("/")[0]
    return ORG_TO_LEAGUE.get(t)


aaa["league"] = aaa.apply(team_league, axis=1)
aaa["debut_year"] = aaa["PlayerId"].map(debut)
predebut = aaa[aaa["debut_year"].notna() & (aaa["Season"] < aaa["debut_year"])].copy()

pcl_share = predebut.groupby("PlayerId").apply(
    lambda g: (g.loc[g["league"] == "PCL", "PA"].sum()) / max(g["PA"].sum(), 1)
).rename("pcl_pa_share").reset_index()
pcl_share = pcl_share.rename(columns={"PlayerId": "playerid"})

df = matched.merge(pcl_share, on="playerid", how="left")
df["pcl_pa_share"] = df["pcl_pa_share"].fillna(0.0)

print("PCL share distribution in cohort (fraction of pre-debut PA at PCL orgs):")
print(f"  mean {df['pcl_pa_share'].mean():.3f}, sd {df['pcl_pa_share'].std():.3f}, "
      f"min {df['pcl_pa_share'].min():.2f}, max {df['pcl_pa_share'].max():.2f}")
print(f"  players 100% PCL: {(df['pcl_pa_share'] > 0.95).sum()}")
print(f"  players 100% IL:  {(df['pcl_pa_share'] < 0.05).sum()}")
print(f"  players mixed:    {((df['pcl_pa_share'] >= 0.05) & (df['pcl_pa_share'] <= 0.95)).sum()}")

r = df[["pcl_pa_share", OUTCOME]].corr().iloc[0, 1]
print(f"\ncorr(pcl_pa_share, mlb_wRC+) = {r:+.4f}")

r_iso = df[["pcl_pa_share", "aaa_ISO_zseason"]].corr().iloc[0, 1]
print(f"corr(pcl_pa_share, aaa_ISO_zseason) = {r_iso:+.4f}   "
      f"(if PCL inflates ISO, this should be positive)")


def per_repeat_cv(predictors):
    sub = df[[OUTCOME] + predictors].dropna()
    X = sub[predictors].values
    y = sub[OUTCOME].values
    r2s = []
    for rep in range(10):
        kf = KFold(n_splits=5, shuffle=True, random_state=42 + rep)
        pipe = Pipeline([("scaler", StandardScaler()), ("lin", LinearRegression())])
        oof = cross_val_predict(pipe, X, y, cv=kf)
        r2s.append(r2_score(y, oof))
    return len(sub), np.array(r2s)


COMPETITORS = {
    "Baseline: raw AAA wRC+":              ["aaa_wRC+"],
    "Skill only (current)":                BASE_SKILL,
    "Skill + PCL share":                   BASE_SKILL + ["pcl_pa_share"],
    "Skill + PCL share + AAA wRC+":        BASE_SKILL + ["pcl_pa_share", "aaa_wRC+"],
}

print("\n=== Head-to-head (paired) ===")
results = {}
for name, preds in COMPETITORS.items():
    n, r2s = per_repeat_cv(preds)
    results[name] = r2s
    print(f"  {name:<45}  N={n}  CV-R² {r2s.mean():.4f} ± {r2s.std():.4f}")

base = results["Baseline: raw AAA wRC+"]
print("\nPaired t vs baseline:")
for name in list(COMPETITORS)[1:]:
    d = results[name] - base
    wins = int(np.sum(d > 0))
    t, p = sp_stats.ttest_1samp(d, 0.0)
    print(f"  {name:<45}  ΔR² {d.mean():+.4f} ± {d.std():.4f}   wins {wins}/10   t={t:+.2f}  p={p:.4f}")

# Also check if adding PCL changes the ISO coefficient meaningfully
print("\n=== Does adding PCL attenuate the ISO coefficient? (in-sample OLS) ===")
import statsmodels.api as sm

def std_ols(preds):
    sub = df[[OUTCOME] + preds].dropna()
    X = sub[preds].values
    Xz = (X - X.mean(0)) / X.std(0)
    y = sub[OUTCOME].values
    m = sm.OLS(y, sm.add_constant(Xz)).fit()
    return dict(zip(preds, m.params[1:])), dict(zip(preds, m.pvalues[1:])), m.rsquared

c1, p1, r1 = std_ols(BASE_SKILL)
c2, p2, r2 = std_ols(BASE_SKILL + ["pcl_pa_share"])
print(f"  {'predictor':<28} {'skill only':>10} {'with PCL':>10} {'p (w/PCL)':>10}")
for k in BASE_SKILL:
    print(f"  {k:<28} {c1[k]:+10.3f} {c2[k]:+10.3f} {p2[k]:>10.4f}")
print(f"  {'pcl_pa_share':<28} {'—':>10} {c2.get('pcl_pa_share', 0):+10.3f} {p2.get('pcl_pa_share', 0):>10.4f}")
print(f"  R²: skill only = {r1:.4f}, with PCL = {r2:.4f}")

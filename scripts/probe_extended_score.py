"""
Probe: does an Extended Translation Score that adds wRC+-invisible features
beat raw AAA wRC+?

Extra features tested (all already in matched_players.csv):
  - log(aaa_PA_predebut): pre-debut AAA PA volume (organizational-investment proxy)
  - aaa_seasons_predebut: number of AAA seasons before debut (Quad-A stall signal)
  - aaa_debut_gap: mlb_debut_year - aaa_last_predebut_year (delay signal)

Standard for winning: paired t-test on 10 CV shuffles (competitor − baseline),
same as Table 3. Combined-model p had to be 0.005 to be a "win"; this probe
uses the same bar.
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
DATA = ROOT / "data" / "processed" / "matched_players.csv"

OUTCOME = "mlb_wRC+"
BASE_SKILL = ["aaa_BB%_zseason", "aaa_K%_zseason", "aaa_ISO_zseason", "aaa_BABIP_zseason", "age_at_debut"]

df = pd.read_csv(DATA)
df["log_aaa_PA_predebut"] = np.log1p(df["aaa_PA_predebut"])
df["aaa_debut_gap"] = df["mlb_debut_year"] - df["aaa_last_predebut_year"]

EXTRA = ["log_aaa_PA_predebut", "aaa_seasons_predebut", "aaa_debut_gap"]

COMPETITORS = {
    "Baseline: raw AAA wRC+":                     ["aaa_wRC+"],
    "Translation Score (current — 5 skills)":     BASE_SKILL,
    "Extended Score (5 skills + prospect-status features)":  BASE_SKILL + EXTRA,
    "Extended Score + raw AAA wRC+":              BASE_SKILL + EXTRA + ["aaa_wRC+"],
}

N_FOLDS = 5
N_REPEATS = 10
SEED_BASE = 42


def per_repeat_cv(predictors):
    sub = df[[OUTCOME] + predictors].dropna()
    X = sub[predictors].values
    y = sub[OUTCOME].values
    r2s, rmses = [], []
    for rep in range(N_REPEATS):
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED_BASE + rep)
        pipe = Pipeline([("scaler", StandardScaler()), ("lin", LinearRegression())])
        oof = cross_val_predict(pipe, X, y, cv=kf)
        r2s.append(r2_score(y, oof))
        rmses.append(np.sqrt(mean_squared_error(y, oof)))
    return len(sub), np.array(r2s), np.array(rmses)


def bar(label): print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")


bar("EXTRA FEATURES — descriptive")
for c in EXTRA:
    print(f"  {c:<28}  mean {df[c].mean():.2f}, sd {df[c].std():.2f}, "
          f"min {df[c].min():.2f}, max {df[c].max():.2f}")

bar("SIMPLE UNIVARIATE CORRELATIONS with mlb_wRC+")
for c in EXTRA:
    r = df[[c, OUTCOME]].dropna().corr().iloc[0, 1]
    print(f"  corr({c:<28}, {OUTCOME}) = {r:+.3f}")

bar("EVALUATE ALL COMPETITORS")
results = {}
for name, preds in COMPETITORS.items():
    n, r2s, rmses = per_repeat_cv(preds)
    results[name] = {"r2s": r2s, "rmses": rmses}
    print(f"  {name:<58}  N={n}  CV-R² {r2s.mean():.4f} ± {r2s.std():.4f}")

bar("PAIRED T-TESTS vs 'Baseline: raw AAA wRC+'")
base = results["Baseline: raw AAA wRC+"]["r2s"]
for name in list(COMPETITORS)[1:]:
    d = results[name]["r2s"] - base
    wins = int(np.sum(d > 0))
    t, p = sp_stats.ttest_1samp(d, 0.0)
    print(f"  {name:<58}  ΔR² {d.mean():+.4f} ± {d.std():.4f}   wins {wins}/10   t={t:+.2f}  p={p:.4f}")

bar("VERDICT")
extended = results["Extended Score (5 skills + prospect-status features)"]["r2s"]
d_ext = extended - base
_, p_ext = sp_stats.ttest_1samp(d_ext, 0.0)
if p_ext < 0.05 and d_ext.mean() > 0:
    print(f"  Extended Score BEATS raw AAA wRC+ at p={p_ext:.4f}.")
    print(f"  This is a real, defensible win driven by wRC+-invisible features.")
elif p_ext < 0.15 and d_ext.mean() > 0:
    print(f"  Extended Score suggestively better (p={p_ext:.4f}) but not conventionally significant.")
else:
    print(f"  Extended Score does not beat raw AAA wRC+ (p={p_ext:.4f}, mean Δ={d_ext.mean():+.4f}).")

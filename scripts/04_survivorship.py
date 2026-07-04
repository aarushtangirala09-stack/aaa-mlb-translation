"""
Step 4 — survivorship bias correction, corrected pipeline.

Two corrections:

A) Sensitivity analysis (floor). Impute assumed MLB wRC+ for non-survivors
   at three levels (pessimistic/neutral/optimistic) and refit the skill
   regression under each. Tests whether Step 2 conclusions hold up.

B) Inverse probability weighting (stretch). Fit a logistic selection model
   predicting "reached MLB with 200+ PA" from AAA stats on all 5,245 AAA
   players. Weight cohort observations by 1/p(reach MLB) and refit OLS.
   Compare coefficients to the unweighted skill model.

Predictors after peer-review fixes:
   - within-season z-scored BB%/K%/ISO/BABIP for era adjustment
   - age variable defined at ALL-AAA level for selection (last-AAA-age) and at
     COHORT level for outcome (age at MLB debut)
   - outcome regression uses pre-debut aggregates only
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
RAW_AAA = ROOT / "data" / "raw" / "aaa"
MATCHED = ROOT / "data" / "processed" / "matched_players.csv"
PROCESSED = ROOT / "data" / "processed"
FIGS = ROOT / "figures"

sns.set_theme(style="whitegrid", context="talk")

STATS = ["BB%", "K%", "ISO", "BABIP"]

# Selection model predictors: computed on ALL AAA seasons per player
SEL_PRED = [f"aaa_{s}_zseason_all" for s in STATS] + ["age_last_aaa"]
# Outcome model predictors: pre-debut aggregates from the matched dataset
OUT_PRED = [f"aaa_{s}_zseason" for s in STATS] + ["age_at_debut"]

PRETTY = {
    "aaa_BB%_zseason_all": "AAA BB% (season-adj, all-AAA)",
    "aaa_K%_zseason_all": "AAA K% (season-adj, all-AAA)",
    "aaa_ISO_zseason_all": "AAA ISO (season-adj, all-AAA)",
    "aaa_BABIP_zseason_all": "AAA BABIP (season-adj, all-AAA)",
    "age_last_aaa": "Age (last AAA season)",
    "aaa_BB%_zseason": "AAA BB% (season-adj, pre-debut)",
    "aaa_K%_zseason": "AAA K% (season-adj, pre-debut)",
    "aaa_ISO_zseason": "AAA ISO (season-adj, pre-debut)",
    "aaa_BABIP_zseason": "AAA BABIP (season-adj, pre-debut)",
    "age_at_debut": "Age at MLB debut",
}
OUTCOME = "mlb_wRC+"
MIN_PA_FOR_SEASON_MEAN = 100


def header(label: str) -> None:
    print(f"\n{'=' * 76}\n{label}\n{'=' * 76}")


def pa_weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = v.notna() & w.notna() & (w > 0)
    if not mask.any():
        return np.nan
    return float(np.average(v[mask], weights=w[mask]))


def within_season_z(df: pd.DataFrame, stat: str, min_pa: int = MIN_PA_FOR_SEASON_MEAN) -> pd.Series:
    result = pd.Series(np.nan, index=df.index, dtype=float)
    for season, g in df.groupby("Season"):
        eligible = g[(g["PA"].astype(float) >= min_pa) & g[stat].notna()]
        if len(eligible) < 30:
            continue
        vals = eligible[stat].values.astype(float)
        w = eligible["PA"].values.astype(float)
        m = float(np.average(vals, weights=w))
        v = float(np.average((vals - m) ** 2, weights=w))
        sd = float(np.sqrt(v))
        if sd <= 0:
            continue
        result.loc[g.index] = (g[stat].astype(float) - m) / sd
    return result


def build_selection_sample() -> pd.DataFrame:
    """One row per AAA player, ALL AAA seasons aggregated, with selection predictors."""
    csvs = sorted(RAW_AAA.glob("*.csv"))
    df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    df["PA"] = pd.to_numeric(df["PA"], errors="coerce")
    df["Season"] = pd.to_numeric(df["Season"], errors="coerce").astype("Int64")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["PlayerId"] = df["PlayerId"].astype(str).str.strip()
    for s in STATS:
        df[s] = pd.to_numeric(df[s], errors="coerce")
        df[f"z_{s}"] = within_season_z(df, s)

    rows = []
    for pid, g in df.groupby("PlayerId"):
        last_idx = g["Season"].idxmax()
        row = {
            "playerid": pid,
            "aaa_PA_all": int(g["PA"].sum(skipna=True)),
            "age_last_aaa": float(g.loc[last_idx, "Age"]),
        }
        for s in STATS:
            row[f"aaa_{s}_zseason_all"] = pa_weighted_mean(g[f"z_{s}"], g["PA"])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    header("LOAD")
    matched = pd.read_csv(MATCHED)
    matched["playerid"] = matched["playerid"].astype(str)
    print(f"  cohort (already-matched, corrected): {len(matched):,}")

    all_aaa = build_selection_sample()
    print(f"  all AAA players (full): {len(all_aaa):,}")

    all_aaa["in_cohort"] = all_aaa["playerid"].isin(set(matched["playerid"]))
    print(f"  in cohort:    {all_aaa['in_cohort'].sum():,}")
    print(f"  out of cohort: {(~all_aaa['in_cohort']).sum():,}")

    # ---------------------------------------------------------------------
    # Selection model
    # ---------------------------------------------------------------------
    header("SELECTION MODEL  —  P(reach MLB w/ 200+ PA | AAA stats)")
    sel = all_aaa.dropna(subset=SEL_PRED + ["aaa_PA_all"]).copy()
    sel = sel[sel["aaa_PA_all"] >= 50].copy()
    print(f"  AAA players in selection sample (>=50 AAA PA): {len(sel):,}")
    print(f"    reached MLB+200PA: {sel['in_cohort'].sum():,} "
          f"({sel['in_cohort'].mean()*100:.1f}%)")

    Xs = sel[SEL_PRED].values
    ys = sel["in_cohort"].astype(int).values
    logit = LogisticRegression(max_iter=2000).fit(Xs, ys)
    sel["p_reach"] = logit.predict_proba(Xs)[:, 1]
    auc = roc_auc_score(ys, sel["p_reach"])
    print(f"  selection-model AUC: {auc:.3f}")
    print(f"  logistic coefficients (raw scales):")
    for name, c in zip(SEL_PRED, logit.coef_[0]):
        print(f"    {PRETTY[name]:<34} {c:+8.3f}")
    print(f"    intercept                          {logit.intercept_[0]:+8.3f}")

    coh_sel = sel[sel["in_cohort"]].copy()
    coh_sel["ipw"] = 1.0 / coh_sel["p_reach"]
    cap = coh_sel["ipw"].quantile(0.99)
    coh_sel["ipw_trunc"] = np.minimum(coh_sel["ipw"], cap)
    print(f"\n  IPW weights (truncated at 99th pct = {cap:.1f}):")
    print(f"    median {coh_sel['ipw_trunc'].median():.2f}, "
          f"mean {coh_sel['ipw_trunc'].mean():.2f}, "
          f"max {coh_sel['ipw_trunc'].max():.2f}")

    # Merge weights back into the pre-debut cohort features (matched)
    coh_out = matched.merge(coh_sel[["playerid", "ipw_trunc"]], on="playerid", how="inner")
    print(f"  cohort with both pre-debut features AND IPW weights: {len(coh_out)}")

    coh_out = coh_out.dropna(subset=OUT_PRED + [OUTCOME])
    print(f"  cohort usable for outcome regression: {len(coh_out)}")

    # ---------------------------------------------------------------------
    # Unweighted vs IPW-weighted skill model
    # ---------------------------------------------------------------------
    header("UNWEIGHTED SKILL MODEL (reference)")
    X = coh_out[OUT_PRED].values
    Xz = (X - X.mean(0)) / X.std(0)
    y = coh_out[OUTCOME].values
    Xc = sm.add_constant(Xz)
    m_ols = sm.OLS(y, Xc).fit()
    print(f"  N = {len(coh_out)}   R² = {m_ols.rsquared:.3f}")
    for name, c, p in zip(OUT_PRED, m_ols.params[1:], m_ols.pvalues[1:]):
        print(f"    {PRETTY[name]:<34} {c:+7.2f}   p={p:.4f}")

    header("IPW-WEIGHTED SKILL MODEL")
    w = coh_out["ipw_trunc"].values
    m_ipw = sm.WLS(y, Xc, weights=w).fit()
    print(f"  weighted N = {w.sum():.1f}  R² (weighted) = {m_ipw.rsquared:.3f}")
    print(f"  Coefficient comparison OLS → IPW:")
    print(f"    {'predictor':<34} {'OLS':>8} {'IPW':>8} {'Δ':>8}")
    rows = []
    for i, name in enumerate(OUT_PRED):
        c0 = m_ols.params[i+1]
        c1 = m_ipw.params[i+1]
        delta_pct = (c1 - c0) / abs(c0) * 100 if c0 != 0 else np.nan
        rows.append({
            "predictor": name, "OLS_coef": c0, "IPW_coef": c1,
            "delta_abs": c1 - c0, "delta_pct": delta_pct,
            "OLS_pval": m_ols.pvalues[i+1], "IPW_pval": m_ipw.pvalues[i+1],
        })
        print(f"    {PRETTY[name]:<34} {c0:+8.2f} {c1:+8.2f} {c1-c0:+8.2f}   ({delta_pct:+.1f}%)")
    pd.DataFrame(rows).to_csv(PROCESSED / "survivorship_corrected_coefs.csv", index=False)

    # ---------------------------------------------------------------------
    # Sensitivity analysis
    # ---------------------------------------------------------------------
    header("SENSITIVITY — impute non-cohort MLB wRC+ at 3 scenarios")
    non_sel = sel[~sel["in_cohort"]].dropna(subset=SEL_PRED).copy()
    print(f"  non-cohort AAA players in sensitivity: {len(non_sel):,}")

    # For sensitivity we must use variables defined on BOTH populations. Use
    # the all-AAA selection predictors (SEL_PRED) — same variable definition
    # for cohort and non-cohort.
    coh_sens_source = coh_sel.dropna(subset=SEL_PRED + ["playerid"]).merge(
        matched[["playerid", OUTCOME]], on="playerid", how="inner"
    )
    combined_X = pd.concat([coh_sens_source[SEL_PRED], non_sel[SEL_PRED]], ignore_index=True)
    mu = combined_X.mean(0)
    sd = combined_X.std(0)
    Xz_full = (combined_X - mu) / sd

    sensitivity_rows = []
    for assumed_y in [50, 80, 100]:
        y_full = np.concatenate([
            coh_sens_source[OUTCOME].values,
            np.full(len(non_sel), assumed_y, dtype=float),
        ])
        Xc_full = sm.add_constant(Xz_full.values)
        m_s = sm.OLS(y_full, Xc_full).fit()
        print(f"\n  assume non-cohort MLB wRC+ = {assumed_y}")
        print(f"    R² = {m_s.rsquared:.3f}   N = {len(y_full):,}")
        for i, name in enumerate(SEL_PRED):
            c = m_s.params[i+1]
            p = m_s.pvalues[i+1]
            print(f"      {PRETTY[name]:<34} {c:+7.2f}   p={p:.4f}")
            sensitivity_rows.append({
                "scenario": f"impute_{assumed_y}",
                "predictor": name, "coef": c, "p_value": p,
            })
    pd.DataFrame(sensitivity_rows).to_csv(
        PROCESSED / "sensitivity_coefficients.csv", index=False)

    # ---------------------------------------------------------------------
    # Figures
    # ---------------------------------------------------------------------
    header("FIGURES")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, grp, color in [
        ("Did NOT reach MLB+200PA", sel[~sel["in_cohort"]], "#d62728"),
        ("Reached MLB+200PA", sel[sel["in_cohort"]], "#2ca02c"),
    ]:
        ax.hist(grp["p_reach"], bins=30, alpha=0.55,
                label=f"{label} (n={len(grp):,})",
                color=color, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Predicted P(reach MLB w/ 200+ PA)")
    ax.set_ylabel("Number of AAA players")
    ax.set_title(f"Selection-model separation  (AUC = {auc:.3f})")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGS / "07_selection_probabilities.png", dpi=150)
    plt.close(fig)

    # Coefficient robustness — use IPW & sensitivity coefficients where names
    # align. Because SEL_PRED != OUT_PRED (different age variables), we show
    # each in its own group and label clearly.
    rows_for_plot = [
        ("Unweighted OLS (outcome-model preds)", [m_ols.params[i+1] for i in range(len(OUT_PRED))], OUT_PRED),
        ("IPW-weighted OLS",                     [m_ipw.params[i+1] for i in range(len(OUT_PRED))], OUT_PRED),
    ]
    for scen in [50, 80, 100]:
        scen_coefs = [r["coef"] for r in sensitivity_rows if r["scenario"] == f"impute_{scen}"]
        rows_for_plot.append((f"Sensitivity: non-cohort = {scen}", scen_coefs, SEL_PRED))

    # Show only the four skill z-stats (they have the same z-scored meaning in
    # both predictor sets), and note the age variable separately in the caption.
    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.15
    positions = np.arange(len(STATS))
    palette = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#2ca02c"]
    stat_labels = [f"AAA {s}" for s in STATS]
    for i, (label, coefs, preds) in enumerate(rows_for_plot):
        idx = [preds.index(f"aaa_{s}_zseason") if f"aaa_{s}_zseason" in preds else preds.index(f"aaa_{s}_zseason_all") for s in STATS]
        vals = [coefs[j] for j in idx]
        ax.bar(positions + (i - 2) * width, vals, width, label=label,
               color=palette[i], edgecolor="black", linewidth=0.6, alpha=0.9)
    ax.set_xticks(positions)
    ax.set_xticklabels(stat_labels, rotation=15, ha="right")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Standardized coefficient on MLB wRC+")
    ax.set_title("Skill-stat coefficients across survivorship corrections\n(age variable differs across specifications; not shown)",
                 fontsize=12)
    ax.legend(fontsize=9, loc="lower right", ncol=1)
    fig.tight_layout()
    fig.savefig(FIGS / "08_coefficient_robustness.png", dpi=150)
    plt.close(fig)

    header("SUMMARY")
    print("  Unweighted vs IPW key shifts (outcome-model predictors):")
    for r in rows:
        print(f"    {PRETTY[r['predictor']]:<34} OLS {r['OLS_coef']:+6.2f}  →  IPW {r['IPW_coef']:+6.2f}   ({r['delta_pct']:+.1f}%)")


if __name__ == "__main__":
    main()

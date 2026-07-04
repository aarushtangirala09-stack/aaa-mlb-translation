"""
Step 3 — build the AAA Translation Score and test it head-to-head against
raw AAA wRC+ on cross-validated MLB wRC+ prediction.

Competitors:
  1. Baseline: AAA wRC+ alone
  2. Translation Score: BB% + K% + ISO + BABIP + age (Step 2 skill model)
  3. Purified: same as 2 but drops BABIP (insignificant in Step 2)
  4. Skill + wRC+: skill predictors plus raw AAA wRC+ (does wRC+ add info?)

Honest comparison via 5-fold cross-validation (10 repeats for stability),
saves predicted-vs-actual scatters and a competitor bar chart.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "matched_players.csv"
FIGS = ROOT / "figures"
OUT = ROOT / "data" / "processed"
FIGS.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")

OUTCOME = "mlb_wRC+"

# Predictors after the peer-review fixes:
#   - within-season z-scored BB%/K%/ISO/BABIP (era/juice adjustment)
#   - age_at_debut (not age in final AAA season — removes reverse-causal contamination)
#   - all upstream AAA aggregates restricted to pre-debut seasons (no temporal leakage)
COMPETITORS = {
    "Baseline: raw AAA wRC+":
        ["aaa_wRC+"],
    "Translation Score (season-adj skills + age at debut)":
        ["aaa_BB%_zseason", "aaa_K%_zseason", "aaa_ISO_zseason", "aaa_BABIP_zseason", "age_at_debut"],
    "Purified Score (drops BABIP)":
        ["aaa_BB%_zseason", "aaa_K%_zseason", "aaa_ISO_zseason", "age_at_debut"],
    "Translation Score + wRC+":
        ["aaa_BB%_zseason", "aaa_K%_zseason", "aaa_ISO_zseason", "aaa_BABIP_zseason", "age_at_debut", "aaa_wRC+"],
}

N_FOLDS = 5
N_REPEATS = 10
SEED_BASE = 42


def header(label: str) -> None:
    print(f"\n{'=' * 76}\n{label}\n{'=' * 76}")


def fit_pipeline(predictors: list[str]) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("lin", LinearRegression())])


def evaluate(df: pd.DataFrame, predictors: list[str]) -> dict:
    sub = df[[OUTCOME] + predictors].dropna()
    X = sub[predictors].values
    y = sub[OUTCOME].values

    # In-sample fit on ALL data
    pipe = fit_pipeline(predictors)
    pipe.fit(X, y)
    in_sample_pred = pipe.predict(X)
    in_r2 = r2_score(y, in_sample_pred)
    in_rmse = float(np.sqrt(mean_squared_error(y, in_sample_pred)))

    # Repeated 5-fold OOF. Same seeds across models → repeats are paired,
    # so we can later do paired-difference tests.
    oof_r2s, oof_rmses = [], []
    for rep in range(N_REPEATS):
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED_BASE + rep)
        oof_pred = cross_val_predict(fit_pipeline(predictors), X, y, cv=kf)
        oof_r2s.append(r2_score(y, oof_pred))
        oof_rmses.append(float(np.sqrt(mean_squared_error(y, oof_pred))))

    return {
        "predictors": predictors,
        "n_used": len(sub),
        "pipeline_fit_all_data": pipe,
        "in_sample_R2": in_r2,
        "in_sample_RMSE": in_rmse,
        "cv_R2_mean": float(np.mean(oof_r2s)),
        "cv_R2_sd": float(np.std(oof_r2s)),
        "cv_RMSE_mean": float(np.mean(oof_rmses)),
        "cv_RMSE_sd": float(np.std(oof_rmses)),
        "oof_r2_per_repeat": np.array(oof_r2s),
        "oof_rmse_per_repeat": np.array(oof_rmses),
        "oof_pred_last_split": oof_pred,
        "y": y,
        "sub": sub,
    }


def paired_diff_report(results: dict, baseline_name: str, competitor_names: list[str]) -> None:
    """Report paired per-repeat differences (competitor − baseline).

    Reports mean ± SD of paired differences, wins/10, and a paired t-test
    (H0: mean difference = 0) on the 10 paired per-repeat R² differences.
    """
    base_r2 = results[baseline_name]["oof_r2_per_repeat"]
    base_rmse = results[baseline_name]["oof_rmse_per_repeat"]
    print(f"\nPaired per-repeat differences vs '{baseline_name}':")
    print(f"  ({N_REPEATS} repeats, same fold assignment across models; positive R² / negative RMSE = competitor wins)")
    print(f"  Formal test: two-tailed paired t on 10 paired R² differences (df=9); p<0.05 = significant.")
    print(f"  {'competitor':<55} {'Δ CV-R²':>16} {'wins/10':>8} {'t (df=9)':>10} {'p':>8}")
    for name in competitor_names:
        c_r2 = results[name]["oof_r2_per_repeat"]
        c_rmse = results[name]["oof_rmse_per_repeat"]
        d_r2 = c_r2 - base_r2
        d_rmse = c_rmse - base_rmse
        wins = int(np.sum(d_r2 > 0))
        t_stat, p_val = sp_stats.ttest_1samp(d_r2, popmean=0.0)
        print(f"  {name:<55} {d_r2.mean():+7.4f}±{d_r2.std():.4f}  {wins}/10   "
              f"{t_stat:+7.2f}  {p_val:.4f}")
        # Also report RMSE difference (kept as a secondary line)
        print(f"    (ΔRMSE = {d_rmse.mean():+.3f} ± {d_rmse.std():.3f} wRC+ pts)")


def make_translation_score(df: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    """Fit on ALL data, write a per-player score to the dataset."""
    sub = df[predictors + [OUTCOME]].dropna()
    X = sub[predictors].values
    y = sub[OUTCOME].values

    pipe = fit_pipeline(predictors)
    pipe.fit(X, y)

    out = df.copy()
    pred = np.full(len(df), np.nan)
    mask = df[predictors].notna().all(axis=1).values
    if mask.any():
        pred[mask] = pipe.predict(df.loc[mask, predictors].values)
    out["aaa_translation_score"] = pred

    # Also show the published coefficient table for the score
    scaler: StandardScaler = pipe.named_steps["scaler"]
    lin: LinearRegression = pipe.named_steps["lin"]
    intercept = lin.intercept_
    raw_coefs = lin.coef_ / scaler.scale_
    raw_intercept = intercept - float(np.dot(lin.coef_, scaler.mean_ / scaler.scale_))

    print("\n--- AAA Translation Score formula (fit on the full corrected cohort) ---")
    print(f"  intercept (raw-units form): {raw_intercept:+.3f}")
    print(f"  intercept (z-scored form):  {intercept:+.3f}")
    print(f"  Per 1-SD increase in each AAA predictor, predicted MLB wRC+ change:")
    for name, coef_z, coef_raw in zip(predictors, lin.coef_, raw_coefs):
        print(f"    {name:<22}  z-coef {coef_z:+7.3f}    raw-unit coef {coef_raw:+10.4f}")

    return out, pipe


def plot_competitor_bars(results: dict) -> None:
    names = list(results.keys())
    cv_r2 = [results[n]["cv_R2_mean"] for n in names]
    cv_r2_sd = [results[n]["cv_R2_sd"] for n in names]
    cv_rmse = [results[n]["cv_RMSE_mean"] for n in names]
    cv_rmse_sd = [results[n]["cv_RMSE_sd"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    short_names = [n.split(":")[0].split("(")[0].strip() for n in names]

    colors = ["#888888", "#1f77b4", "#2ca02c", "#9467bd"]
    axes[0].barh(short_names, cv_r2, xerr=cv_r2_sd, color=colors, alpha=0.85, edgecolor="black")
    axes[0].set_xlabel("Cross-validated R²  (higher = better)")
    axes[0].set_title("Head-to-head: out-of-fold R²")
    for i, (v, s) in enumerate(zip(cv_r2, cv_r2_sd)):
        axes[0].text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=10)

    axes[1].barh(short_names, cv_rmse, xerr=cv_rmse_sd, color=colors, alpha=0.85, edgecolor="black")
    axes[1].set_xlabel("Cross-validated RMSE  (lower = better)")
    axes[1].set_title("Head-to-head: out-of-fold RMSE (wRC+ points)")
    for i, (v, s) in enumerate(zip(cv_rmse, cv_rmse_sd)):
        axes[1].text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=10)

    fig.suptitle("Step 3 — Translation Score vs. raw AAA wRC+", y=1.02, fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGS / "05_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pred_vs_actual(results: dict) -> None:
    base = results["Baseline: raw AAA wRC+"]
    score = results["Translation Score (season-adj skills + age at debut)"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharex=True, sharey=True)
    for ax, label, r in zip(
        axes,
        ["Baseline — raw AAA wRC+", "Translation Score — disaggregated"],
        [base, score],
    ):
        y = r["y"]; p = r["oof_pred_last_split"]
        ax.scatter(p, y, alpha=0.4, s=20, color="#1f77b4", edgecolor="none")
        lo = min(p.min(), y.min())
        hi = max(p.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color="black", lw=1.5, ls="--", label="perfect prediction")
        ax.set_xlabel("Predicted MLB wRC+ (out-of-fold)")
        ax.set_ylabel("Actual MLB wRC+")
        ax.set_title(f"{label}\nCV R² = {r['cv_R2_mean']:.3f}   CV RMSE = {r['cv_RMSE_mean']:.2f}")
        ax.legend(loc="upper left", fontsize=11)
    fig.suptitle("Out-of-fold predictions: head-to-head", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "06_predicted_vs_actual_OOF.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    header("LOAD")
    df = pd.read_csv(DATA)
    print(f"  rows: {len(df)}")
    print(f"  outcome ({OUTCOME}): mean {df[OUTCOME].mean():.1f}, sd {df[OUTCOME].std():.1f}")

    results = {}
    for name, predictors in COMPETITORS.items():
        header(f"EVALUATE — {name}")
        print(f"  predictors: {predictors}")
        res = evaluate(df, predictors)
        results[name] = res
        print(f"  N used:          {res['n_used']}")
        print(f"  in-sample R²:    {res['in_sample_R2']:.3f}")
        print(f"  in-sample RMSE:  {res['in_sample_RMSE']:.2f}")
        print(f"  CV R² (mean ± sd):    {res['cv_R2_mean']:.3f} ± {res['cv_R2_sd']:.3f}")
        print(f"  CV RMSE (mean ± sd):  {res['cv_RMSE_mean']:.2f} ± {res['cv_RMSE_sd']:.2f}")

    header("PAIRED-DIFFERENCE TESTS (competitor − baseline; per-repeat, 10 paired repeats)")
    base_name = "Baseline: raw AAA wRC+"
    paired_diff_report(
        results, base_name,
        [
            "Translation Score (season-adj skills + age at debut)",
            "Purified Score (drops BABIP)",
            "Translation Score + wRC+",
        ],
    )
    print("\nHonest verdict conventions:")
    print("  wins/10 = 10 → consistent competitor win  |  0 → consistent baseline win")
    print("  wins/10 near 5, and |Δ CV-R²| smaller than its SD → null/tie at this power")

    header("FIGURES")
    plot_competitor_bars(results)
    plot_pred_vs_actual(results)
    for p in sorted(FIGS.glob("*.png")):
        print(f"  wrote {p.relative_to(ROOT)}")

    header("WRITE PER-PLAYER TRANSLATION SCORE")
    score_predictors = COMPETITORS["Translation Score (season-adj skills + age at debut)"]
    scored_df, pipe = make_translation_score(df, score_predictors)
    out_csv = OUT / "matched_players_with_score.csv"
    scored_df.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv.relative_to(ROOT)}")
    print(f"  Translation Score summary:")
    print(f"    mean {scored_df['aaa_translation_score'].mean():.1f}, "
          f"sd {scored_df['aaa_translation_score'].std():.1f}, "
          f"min {scored_df['aaa_translation_score'].min():.1f}, "
          f"max {scored_df['aaa_translation_score'].max():.1f}")

    header("SAVE COMPETITOR METRICS TABLE")
    summary_rows = []
    for name, r in results.items():
        summary_rows.append({
            "competitor": name,
            "n_used": r["n_used"],
            "in_sample_R2": r["in_sample_R2"],
            "in_sample_RMSE": r["in_sample_RMSE"],
            "cv_R2_mean": r["cv_R2_mean"],
            "cv_R2_sd": r["cv_R2_sd"],
            "cv_RMSE_mean": r["cv_RMSE_mean"],
            "cv_RMSE_sd": r["cv_RMSE_sd"],
        })
    pd.DataFrame(summary_rows).to_csv(OUT / "step3_model_comparison.csv", index=False)
    print(f"  wrote data/processed/step3_model_comparison.csv")


if __name__ == "__main__":
    main()

"""
Step 2 — predictive regression.

Fits three OLS models (baseline / skill / kitchen-sink) plus ridge/lasso
robustness, runs diagnostics (correlations, VIF, residuals), saves headline
figures to ./figures/, and writes coefficient table to data/processed/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "matched_players.csv"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)
OUT = ROOT / "data" / "processed"

sns.set_theme(style="whitegrid", context="talk")

OUTCOME = "mlb_wRC+"

# Primary specification: within-season z-scored predictors + age-at-debut.
# All fixes from the peer-review pass live here:
#   - "aaa_<STAT>_zseason" = PA-weighted average of within-season z-scores, using
#     the full AAA population as reference (era/juice adjustment)
#   - "age_at_debut" = age at MLB debut, computed from birth year (removes
#     reverse-causal contamination that "age in final AAA season" carried)
#   - All AAA aggregates upstream are restricted to pre-debut seasons only
#     (no temporal leakage)
SKILL_PREDICTORS = [
    "aaa_BB%_zseason", "aaa_K%_zseason", "aaa_ISO_zseason",
    "aaa_BABIP_zseason", "age_at_debut",
]
KITCHEN_SINK = SKILL_PREDICTORS + ["aaa_wOBA", "aaa_AVG", "aaa_OBP", "aaa_SLG"]
BASELINE = ["aaa_wRC+"]

PRETTY = {
    "aaa_BB%_zseason": "AAA BB% (season-adj)",
    "aaa_K%_zseason": "AAA K% (season-adj)",
    "aaa_ISO_zseason": "AAA ISO (season-adj)",
    "aaa_BABIP_zseason": "AAA BABIP (season-adj)",
    "age_at_debut": "Age at MLB debut",
    "aaa_BB%": "AAA BB% (raw)",
    "aaa_K%": "AAA K% (raw)",
    "aaa_ISO": "AAA ISO (raw)",
    "aaa_BABIP": "AAA BABIP (raw)",
    "aaa_wOBA": "AAA wOBA",
    "aaa_AVG": "AAA AVG",
    "aaa_OBP": "AAA OBP",
    "aaa_SLG": "AAA SLG",
    "aaa_wRC+": "AAA wRC+",
}


def header(label: str) -> None:
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")


def standardize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Z-score each predictor so coefficients are directly comparable."""
    scaler = StandardScaler()
    z = pd.DataFrame(scaler.fit_transform(df[cols]), columns=cols, index=df.index)
    return z


def fit_ols(df: pd.DataFrame, y_col: str, x_cols: list[str], label: str):
    sub = df[[y_col] + x_cols].dropna()
    y = sub[y_col]
    X_raw = sub[x_cols]
    X_std = standardize(sub, x_cols)
    X = sm.add_constant(X_std)
    model = sm.OLS(y, X).fit()
    print(f"\n--- {label}  (N used = {len(sub)}) ---")
    print(f"  R²       = {model.rsquared:.3f}")
    print(f"  adj R²   = {model.rsquared_adj:.3f}")
    print(f"  RMSE     = {np.sqrt(model.mse_resid):.2f} wRC+ points")
    print("  Standardized coefficients (per 1-SD change in predictor):")
    coefs = model.params.drop("const")
    ses = model.bse.drop("const")
    pvals = model.pvalues.drop("const")
    for name in coefs.index:
        sig = "***" if pvals[name] < 0.001 else "**" if pvals[name] < 0.01 else "*" if pvals[name] < 0.05 else ""
        print(f"    {PRETTY.get(name, name):<28} {coefs[name]:+7.2f}  ±{1.96*ses[name]:.2f}  (p={pvals[name]:.4f}) {sig}")
    return model, X_std, y, sub


def compute_vif(X: pd.DataFrame, label: str) -> pd.DataFrame:
    Xc = sm.add_constant(X)
    rows = []
    for i, col in enumerate(Xc.columns):
        if col == "const":
            continue
        rows.append({"predictor": col, "VIF": variance_inflation_factor(Xc.values, i)})
    vif = pd.DataFrame(rows)
    print(f"\n--- VIF ({label}) — values >5 are concerning, >10 are severe ---")
    for _, r in vif.iterrows():
        flag = "  ←⚠ severe" if r["VIF"] > 10 else "  ←⚠ moderate" if r["VIF"] > 5 else ""
        print(f"    {PRETTY.get(r['predictor'], r['predictor']):<28} VIF = {r['VIF']:5.2f}{flag}")
    return vif


def ridge_lasso(df: pd.DataFrame, y_col: str, x_cols: list[str]):
    sub = df[[y_col] + x_cols].dropna()
    y = sub[y_col].values
    X = standardize(sub, x_cols).values

    ridge = RidgeCV(alphas=np.logspace(-3, 3, 50)).fit(X, y)
    lasso = LassoCV(alphas=np.logspace(-3, 1, 50), cv=5, max_iter=20000).fit(X, y)

    print("\n--- Regularized regressions on the skill-model predictors ---")
    print(f"  RidgeCV  best alpha = {ridge.alpha_:.4f}")
    print(f"  LassoCV  best alpha = {lasso.alpha_:.4f}")
    for i, name in enumerate(x_cols):
        print(f"    {PRETTY.get(name, name):<28} ridge {ridge.coef_[i]:+6.2f}   lasso {lasso.coef_[i]:+6.2f}")
    return ridge, lasso


def plot_correlation_heatmap(df: pd.DataFrame, cols: list[str]) -> None:
    sub = df[cols].dropna()
    sub = sub.rename(columns={c: PRETTY.get(c, c) for c in sub.columns})
    corr = sub.corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, ax=ax, cbar_kws={"shrink": 0.75})
    ax.set_title("AAA predictor correlations (collinearity diagnostic)")
    fig.tight_layout()
    fig.savefig(FIGS / "01_correlation_heatmap.png", dpi=150)
    plt.close(fig)


def plot_aaa_vs_mlb_scatter(df: pd.DataFrame) -> None:
    sub = df[["aaa_wRC+", "mlb_wRC+", "mlb_PA"]].dropna()
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.scatter(sub["aaa_wRC+"], sub["mlb_wRC+"], alpha=0.4, s=18, color="#1f77b4", edgecolor="none")
    m, b = np.polyfit(sub["aaa_wRC+"], sub["mlb_wRC+"], 1)
    xs = np.linspace(sub["aaa_wRC+"].min(), sub["aaa_wRC+"].max(), 100)
    ax.plot(xs, m*xs + b, color="black", lw=2, label=f"OLS fit  (slope = {m:.2f})")
    ax.axhline(100, color="grey", ls="--", lw=1, alpha=0.6)
    ax.axvline(100, color="grey", ls="--", lw=1, alpha=0.6)
    r = sub["aaa_wRC+"].corr(sub["mlb_wRC+"])
    ax.set_xlabel("AAA wRC+ (career, PA-weighted)")
    ax.set_ylabel("MLB wRC+ (career, PA-weighted)")
    ax.set_title(f"AAA → MLB wRC+ — raw relationship  (N={len(sub)}, r={r:.2f})")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGS / "02_aaa_vs_mlb_wrcplus_scatter.png", dpi=150)
    plt.close(fig)


def plot_coefficients(model, x_cols: list[str], title: str, filename: str) -> None:
    coefs = model.params.drop("const")
    ses = model.bse.drop("const")
    order = coefs.abs().sort_values().index.tolist()
    coefs, ses = coefs[order], ses[order]
    labels = [PRETTY.get(c, c) for c in coefs.index]

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in coefs.values]
    ax.barh(labels, coefs.values, xerr=1.96*ses.values, color=colors,
            edgecolor="black", alpha=0.85, error_kw={"lw": 1.2})
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Standardized coefficient\n(predicted Δ in MLB wRC+ per 1-SD of AAA predictor)")
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGS / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(model, y, label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fitted = model.fittedvalues
    resid = model.resid
    axes[0].scatter(fitted, resid, alpha=0.4, s=18)
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set_xlabel("Fitted MLB wRC+")
    axes[0].set_ylabel("Residual")
    axes[0].set_title(f"Residuals vs fitted — {label}")
    sm.qqplot(resid, line="s", ax=axes[1])
    axes[1].set_title("Q-Q plot of residuals (normality)")
    fig.tight_layout()
    fig.savefig(FIGS / "04_residual_diagnostics.png", dpi=150)
    plt.close(fig)


def main() -> None:
    header("LOAD")
    df = pd.read_csv(DATA)
    print(f"  rows in matched dataset: {len(df)}")
    print(f"  outcome ({OUTCOME}) — mean {df[OUTCOME].mean():.1f}, sd {df[OUTCOME].std():.1f}")
    print(f"  predictors used:")
    for c in BASELINE + SKILL_PREDICTORS + ["aaa_wOBA", "aaa_AVG", "aaa_OBP", "aaa_SLG"]:
        nm = df[c].notna().sum()
        print(f"    {PRETTY.get(c, c):<28} non-null {nm}/{len(df)}")

    header("MODEL 1 — BASELINE: MLB wRC+ ~ AAA wRC+ alone")
    m1, _, _, _ = fit_ols(df, OUTCOME, BASELINE, "Baseline (raw AAA wRC+)")

    header("MODEL 2 — SKILL MODEL: MLB wRC+ ~ BB% + K% + ISO + BABIP + age")
    m2, X2, y2, sub2 = fit_ols(df, OUTCOME, SKILL_PREDICTORS, "Skill-disaggregated")
    compute_vif(X2, "skill model")

    header("MODEL 3 — KITCHEN SINK: add wOBA + AVG + OBP + SLG")
    m3, X3, _, _ = fit_ols(df, OUTCOME, KITCHEN_SINK, "Kitchen sink")
    compute_vif(X3, "kitchen sink")

    header("ROBUSTNESS — RIDGE / LASSO on skill model")
    ridge, lasso = ridge_lasso(df, OUTCOME, SKILL_PREDICTORS)

    header("FIGURES")
    plot_correlation_heatmap(df, KITCHEN_SINK + ["aaa_wRC+"])
    plot_aaa_vs_mlb_scatter(df)
    plot_coefficients(m2, SKILL_PREDICTORS,
                      "Which AAA skills predict MLB wRC+?  (Skill model, standardized OLS)",
                      "03_skill_coefficients.png")
    plot_residuals(m2, y2, "skill model")
    for p in sorted(FIGS.glob("*.png")):
        print(f"  wrote {p.relative_to(ROOT)}")

    header("SAVE COEFFICIENTS")
    out = OUT / "regression_coefficients.csv"
    rows = []
    for label, mdl, xs in [("baseline", m1, BASELINE), ("skill", m2, SKILL_PREDICTORS), ("kitchen_sink", m3, KITCHEN_SINK)]:
        for name in xs:
            rows.append({
                "model": label,
                "predictor": name,
                "std_coef": mdl.params[name],
                "std_error": mdl.bse[name],
                "p_value": mdl.pvalues[name],
                "ci_low": mdl.params[name] - 1.96 * mdl.bse[name],
                "ci_high": mdl.params[name] + 1.96 * mdl.bse[name],
            })
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  wrote {out.relative_to(ROOT)}")

    header("MODEL COMPARISON SUMMARY")
    print(f"  baseline    R² = {m1.rsquared:.3f}   adj R² = {m1.rsquared_adj:.3f}")
    print(f"  skill       R² = {m2.rsquared:.3f}   adj R² = {m2.rsquared_adj:.3f}")
    print(f"  kitchen     R² = {m3.rsquared:.3f}   adj R² = {m3.rsquared_adj:.3f}")


if __name__ == "__main__":
    main()

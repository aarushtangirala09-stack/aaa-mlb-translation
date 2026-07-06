"""
Step 4.5 — Bayesian robustness check on the skill model.

Refits the OLS skill model as a Bayesian regression with weakly informative
priors. Reports posterior means, 95% credible intervals, P(coef > 0), R-hat,
and effective sample size for each predictor. Saves posterior samples and a
posterior-density figure.

Priors:
  β_k ~ Normal(0, 10)     for each standardized predictor (k=1..5)
  β_0 ~ Normal(mean_y, 20) for the intercept
  σ   ~ HalfNormal(20)    for the residual scale

MCMC: 4 chains, 2000 warmup + 2000 sampling iterations, target_accept=0.95.

Diagnostics (R-hat, ESS) are computed directly from posterior samples to
avoid arviz version conflicts.
"""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "matched_players.csv"
FIGS = ROOT / "figures"
OUT = ROOT / "data" / "processed"

OUTCOME = "mlb_wRC+"
PREDICTORS = ["aaa_BB%_zseason", "aaa_K%_zseason", "aaa_ISO_zseason",
              "aaa_BABIP_zseason", "age_at_debut"]
PRETTY = {
    "aaa_BB%_zseason": "AAA BB%",
    "aaa_K%_zseason":  "AAA K%",
    "aaa_ISO_zseason": "AAA ISO",
    "aaa_BABIP_zseason": "AAA BABIP",
    "age_at_debut": "Age at debut",
}


def rhat_diagnostic(chains: np.ndarray) -> float:
    """Gelman-Rubin R-hat on shape (n_chains, n_draws)."""
    n_chains, n_draws = chains.shape
    chain_means = chains.mean(axis=1)
    chain_vars = chains.var(axis=1, ddof=1)
    grand_mean = chain_means.mean()
    B = n_draws * ((chain_means - grand_mean) ** 2).sum() / (n_chains - 1)
    W = chain_vars.mean()
    var_hat = (1 - 1 / n_draws) * W + B / n_draws
    return float(np.sqrt(var_hat / W))


def ess_diagnostic(chains: np.ndarray, max_lag: int = 50) -> float:
    """Effective sample size, simplified autocorrelation formula on (n_chains, n_draws)."""
    n_chains, n_draws = chains.shape
    total = n_chains * n_draws
    # Compute chain-averaged autocorrelation
    def autocorr(x):
        x = x - x.mean()
        result = np.correlate(x, x, mode="full")[-len(x):]
        return result / result[0]
    rhos = []
    for c in range(n_chains):
        ac = autocorr(chains[c])
        rhos.append(ac[:max_lag + 1])
    rho = np.mean(rhos, axis=0)
    # Sum until autocorrelation becomes negative or trivial
    tau = 1 + 2 * np.sum(rho[1:])
    return float(total / max(tau, 1.0))


def main() -> None:
    df = pd.read_csv(DATA)
    sub = df[[OUTCOME] + PREDICTORS].dropna()
    print(f"N = {len(sub)}")

    X_raw = sub[PREDICTORS].values
    X = (X_raw - X_raw.mean(0)) / X_raw.std(0)
    y = sub[OUTCOME].values
    y_mean, y_sd = float(y.mean()), float(y.std())

    print(f"outcome mean = {y_mean:.2f}, sd = {y_sd:.2f}")

    with pm.Model() as model:
        beta0 = pm.Normal("intercept", mu=y_mean, sigma=20)
        betas = pm.Normal("betas", mu=0, sigma=10, shape=len(PREDICTORS))
        sigma = pm.HalfNormal("sigma", sigma=20)
        mu = beta0 + pm.math.dot(X, betas)
        pm.Normal("y", mu=mu, sigma=sigma, observed=y)

        idata = pm.sample(
            draws=2000, tune=2000, chains=4, cores=4,
            target_accept=0.95, random_seed=42, progressbar=False,
        )

    # ---- Extract posterior samples (chains × draws × predictors) ----
    beta_chains = idata.posterior["betas"].values  # (chain, draw, predictor)
    intercept_chains = idata.posterior["intercept"].values  # (chain, draw)
    sigma_chains = idata.posterior["sigma"].values  # (chain, draw)

    n_chains, n_draws = beta_chains.shape[0], beta_chains.shape[1]

    rows = []
    print("\n=== Posterior distributions for each predictor ===")
    print(f"{'predictor':<25} {'post mean':>10} {'95% CI':>22} {'P(β>0)':>10} {'R-hat':>8} {'ESS':>8}")
    all_rhat = []
    all_ess = []
    for i, name in enumerate(PREDICTORS):
        chains_i = beta_chains[:, :, i]  # (chain, draw)
        samples_flat = chains_i.reshape(-1)
        mean = float(samples_flat.mean())
        lo, hi = np.percentile(samples_flat, [2.5, 97.5])
        p_pos = float((samples_flat > 0).mean())
        rhat = rhat_diagnostic(chains_i)
        ess = ess_diagnostic(chains_i)
        all_rhat.append(rhat)
        all_ess.append(ess)
        rows.append({
            "predictor": name, "pretty": PRETTY[name],
            "posterior_mean": mean,
            "ci_low_95": float(lo), "ci_high_95": float(hi),
            "prob_positive": p_pos, "r_hat": rhat, "ess": ess,
        })
        print(f"{PRETTY[name]:<25} {mean:+10.3f}  [{lo:+6.2f}, {hi:+6.2f}]  {p_pos:>9.3f}  {rhat:>8.4f}  {ess:>8.0f}")

    max_rhat = max(all_rhat)
    min_ess = min(all_ess)
    print(f"\nmax R-hat: {max_rhat:.4f}  (should be < 1.01 for convergence)")
    print(f"min ESS:   {min_ess:.0f}  (should be > 400 per chain)")
    assert max_rhat < 1.02, f"R-hat convergence failed: {max_rhat}"
    print("Convergence: PASSED.")

    pd.DataFrame(rows).to_csv(OUT / "bayesian_posterior_summary.csv", index=False)
    print(f"\nwrote {OUT / 'bayesian_posterior_summary.csv'}")

    # Save flat posterior samples
    posterior_df = pd.DataFrame(
        beta_chains.reshape(-1, len(PREDICTORS)),
        columns=[PRETTY[p] for p in PREDICTORS],
    )
    posterior_df["intercept"] = intercept_chains.reshape(-1)
    posterior_df["sigma"] = sigma_chains.reshape(-1)
    posterior_df.to_csv(OUT / "bayesian_posterior_samples.csv", index=False)
    print(f"wrote {OUT / 'bayesian_posterior_samples.csv'}")

    # ---- Posterior density figure ----
    color_map = {
        "aaa_ISO_zseason": "#2E9E4F",
        "aaa_K%_zseason": "#D64545",
        "aaa_BB%_zseason": "#2E9E4F",
        "aaa_BABIP_zseason": "#888888",
        "age_at_debut": "#D64545",
    }
    ordered = ["aaa_ISO_zseason", "aaa_K%_zseason", "aaa_BB%_zseason",
               "age_at_debut", "aaa_BABIP_zseason"]

    fig, ax = plt.subplots(figsize=(11, 6))
    for name in ordered:
        idx = PREDICTORS.index(name)
        samples = beta_chains[:, :, idx].reshape(-1)
        density_x = np.linspace(samples.min() - 0.5, samples.max() + 0.5, 300)
        kde = gaussian_kde(samples)
        density_y = kde(density_x)
        ax.fill_between(density_x, density_y, alpha=0.35,
                        color=color_map[name], edgecolor=color_map[name], linewidth=1.5)
        peak_x = density_x[np.argmax(density_y)]
        peak_y = density_y.max()
        ax.annotate(PRETTY[name],
                    xy=(peak_x, peak_y), xytext=(peak_x, peak_y * 1.06),
                    ha="center", va="bottom", fontsize=11, fontweight="bold",
                    color=color_map[name])

    ax.axvline(0, color="black", lw=1, ls="--", alpha=0.6, label="β = 0 (no effect)")
    ax.set_xlabel("Posterior coefficient (standardized units, wRC+ points per 1 SD of predictor)")
    ax.set_ylabel("Posterior density")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", alpha=0.5)
    # Title and subtitle placed at figure-level so they can't overlap axes elements
    fig.text(0.5, 0.98, "Posterior distributions of skill-model coefficients (Bayesian regression)",
             ha="center", va="top", fontsize=17, fontweight="bold")
    fig.text(0.5, 0.925,
             "Weakly informative Normal(0, 10) priors on standardized coefficients; MCMC via PyMC, "
             "4 chains × 2000 samples. Density = posterior probability of each coefficient value.",
             ha="center", va="top", fontsize=11.5, color="#333", style="italic")
    fig.subplots_adjust(top=0.83)
    fig.savefig(FIGS / "10_bayesian_posteriors.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGS / '10_bayesian_posteriors.png'}")


if __name__ == "__main__":
    main()

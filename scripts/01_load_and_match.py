"""
Step 1 — load, match, filter cohort. Corrected version.

Three methodological fixes vs the original build:
  #1. AAA aggregates restricted to seasons BEFORE MLB debut (no temporal leakage).
  #2. Age computed at MLB debut, not in the final AAA season (removes reverse-causality
      from post-demotion AAA seasons).
  #3. Predictors additionally computed as WITHIN-SEASON z-scores using the full AAA
      population as the reference distribution (era/juice adjustment; the biggest
      remaining confound, IL vs PCL park differences, requires team→league mapping
      and is deferred to future work).

Both raw (aaa_<STAT>) and within-season-adjusted (aaa_<STAT>_zseason) predictors
are written to the output CSV. Downstream scripts consume the _zseason versions
as the primary specification.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_AAA = ROOT / "data" / "raw" / "aaa"
RAW_MLB = ROOT / "data" / "raw" / "mlb"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

MLB_PA_THRESHOLD = 200
MIN_PA_FOR_SEASON_MEAN = 100  # eligibility for computing season-level reference dist

PLAYERID_CANDIDATES = ["playerid", "PlayerId", "IDfg", "playerId", "player_id"]

AAA_RATE_STATS = ["BB%", "K%", "ISO", "BABIP", "AVG", "OBP", "SLG", "wOBA"]
SEASON_ZSCORE_STATS = ["BB%", "K%", "ISO", "BABIP"]  # get within-season standardization


def load_concat(folder: Path, label: str) -> pd.DataFrame:
    csvs = sorted(folder.glob("*.csv"))
    if not csvs:
        sys.exit(f"\nNo CSVs found in {folder}.\nSee data/EXPORT_INSTRUCTIONS.md.")
    frames = []
    for p in csvs:
        df = pd.read_csv(p)
        df["__src_file"] = p.name
        frames.append(df)
        print(f"  {label}: loaded {p.name} ({len(df):,} rows)")
    return pd.concat(frames, ignore_index=True)


def find_playerid_col(df: pd.DataFrame) -> str:
    for c in PLAYERID_CANDIDATES:
        if c in df.columns:
            return c
    sys.exit(
        f"\nCould not find a FG player id column. "
        f"Looked for one of: {PLAYERID_CANDIDATES}."
    )


def strip_pct(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        return pd.to_numeric(
            series.astype(str).str.replace("%", "", regex=False).str.strip(),
            errors="coerce",
        )
    return pd.to_numeric(series, errors="coerce")


def pa_weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = v.notna() & w.notna() & (w > 0)
    if not mask.any():
        return np.nan
    return float(np.average(v[mask], weights=w[mask]))


def within_season_zscore(df: pd.DataFrame, stat: str, min_pa: int = MIN_PA_FOR_SEASON_MEAN) -> pd.Series:
    """PA-weighted z-score of `stat` within each Season.

    Season mean and SD computed on players with PA >= min_pa to avoid small-sample noise.
    Returns a Series aligned to df.index; NaN where no valid season stats.
    """
    result = pd.Series(np.nan, index=df.index, dtype=float)
    for season, g in df.groupby("Season"):
        eligible = g[(g["PA"].astype(float) >= min_pa) & g[stat].notna()]
        if len(eligible) < 30:
            continue
        vals = eligible[stat].values.astype(float)
        weights = eligible["PA"].values.astype(float)
        m = float(np.average(vals, weights=weights))
        var = float(np.average((vals - m) ** 2, weights=weights))
        sd = float(np.sqrt(var))
        if sd <= 0:
            continue
        result.loc[g.index] = (g[stat].astype(float) - m) / sd
    return result


def aggregate_mlb(df: pd.DataFrame, pid_col: str) -> pd.DataFrame:
    if "wRC+" in df.columns:
        df["wRC+"] = pd.to_numeric(df["wRC+"], errors="coerce")
    df["PA"] = pd.to_numeric(df["PA"], errors="coerce")
    df[pid_col] = df[pid_col].astype(str).str.strip()

    rows = []
    for pid, g in df.groupby(pid_col):
        rows.append(
            {
                "playerid": pid,
                "mlb_Name": g["Name"].mode().iat[0] if "Name" in g.columns else "",
                "mlb_PA": int(g["PA"].sum(skipna=True)),
                "mlb_seasons": int(g["Season"].nunique()),
                "mlb_wRC+": pa_weighted_mean(g["wRC+"], g["PA"]),
                "mlb_debut_year": int(g["Season"].min()),
            }
        )
    return pd.DataFrame(rows)


def compute_birth_years(aaa: pd.DataFrame, pid_col: str) -> dict:
    """Estimate birth year per player as median of (Season - Age) across their AAA seasons."""
    aaa = aaa.copy()
    aaa["Age"] = pd.to_numeric(aaa["Age"], errors="coerce")
    aaa["Season"] = pd.to_numeric(aaa["Season"], errors="coerce")
    aaa["__by"] = aaa["Season"] - aaa["Age"]
    return aaa.groupby(pid_col)["__by"].median().to_dict()


def prepare_aaa_all(aaa_all: pd.DataFrame, pid_col: str) -> pd.DataFrame:
    """Clean AAA table + add within-season z-score columns computed on the full AAA pop."""
    for c in AAA_RATE_STATS + ["wRC+", "Age"]:
        if c in aaa_all.columns:
            aaa_all[c] = strip_pct(aaa_all[c])
    aaa_all["PA"] = pd.to_numeric(aaa_all["PA"], errors="coerce")
    aaa_all["Season"] = pd.to_numeric(aaa_all["Season"], errors="coerce").astype("Int64")
    aaa_all[pid_col] = aaa_all[pid_col].astype(str).str.strip()

    for s in SEASON_ZSCORE_STATS:
        aaa_all[f"z_{s}"] = within_season_zscore(aaa_all, s)

    return aaa_all


def aggregate_aaa_predebut(
    aaa_all: pd.DataFrame,
    pid_col: str,
    debut_years: dict,
    birth_years: dict,
) -> pd.DataFrame:
    """Per-player AAA aggregates using only pre-debut seasons."""
    aaa_all = aaa_all.copy()
    aaa_all["debut_year"] = aaa_all[pid_col].map(debut_years)
    predebut_mask = aaa_all["debut_year"].notna() & (
        aaa_all["Season"].astype(float) < aaa_all["debut_year"].astype(float)
    )
    aaa_pre = aaa_all[predebut_mask].copy()

    rows = []
    for pid, g in aaa_pre.groupby(pid_col):
        by = birth_years.get(pid, np.nan)
        dy = debut_years.get(pid, np.nan)
        row = {
            "playerid": pid,
            "Name": g["Name"].mode().iat[0] if "Name" in g.columns else "",
            "aaa_seasons_predebut": int(g["Season"].nunique()),
            "aaa_PA_predebut": int(g["PA"].sum(skipna=True)),
            "aaa_first_year": int(g["Season"].min()),
            "aaa_last_predebut_year": int(g["Season"].max()),
            "birth_year_est": by,
            "age_at_debut": float(dy - by) if (pd.notna(dy) and pd.notna(by)) else np.nan,
        }
        for c in AAA_RATE_STATS + ["wRC+"]:
            row[f"aaa_{c}"] = pa_weighted_mean(g[c], g["PA"])
        for c in SEASON_ZSCORE_STATS:
            row[f"aaa_{c}_zseason"] = pa_weighted_mean(g[f"z_{c}"], g["PA"])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    print(f"\n=== Loading AAA CSVs from {RAW_AAA} ===")
    aaa_raw = load_concat(RAW_AAA, "AAA")
    aaa_pid = find_playerid_col(aaa_raw)
    print(f"  AAA player-season rows: {len(aaa_raw):,}  (id col: {aaa_pid})")

    print(f"\n=== Loading MLB CSVs from {RAW_MLB} ===")
    mlb_raw = load_concat(RAW_MLB, "MLB")
    mlb_pid = find_playerid_col(mlb_raw)
    print(f"  MLB player-season rows: {len(mlb_raw):,}  (id col: {mlb_pid})")

    print("\n=== Cleaning + season z-scoring AAA (full population reference) ===")
    aaa_all = prepare_aaa_all(aaa_raw, aaa_pid)
    for s in SEASON_ZSCORE_STATS:
        m = aaa_all[f"z_{s}"].mean()
        sd = aaa_all[f"z_{s}"].std()
        print(f"  z_{s:<6}  mean {m:+.3f}  sd {sd:.3f}   (should be ~0, ~1)")

    print("\n=== Aggregating MLB per player ===")
    mlb = aggregate_mlb(mlb_raw, mlb_pid)
    debut_years = {pid: int(y) for pid, y in mlb.set_index("playerid")["mlb_debut_year"].items()}
    print(f"  MLB unique players: {len(mlb):,}")

    print("\n=== Estimating birth years from AAA data ===")
    birth_years = compute_birth_years(aaa_raw, aaa_pid)
    print(f"  birth_year estimated for {len(birth_years):,} AAA players")

    print("\n=== Aggregating AAA (PRE-DEBUT seasons only) ===")
    aaa = aggregate_aaa_predebut(aaa_all, aaa_pid, debut_years, birth_years)
    print(f"  AAA players with any pre-debut AAA data: {len(aaa):,}")

    print("\n=== Matching AAA + MLB, applying 200+ MLB PA filter ===")
    matched = aaa.merge(mlb, on="playerid", how="inner")
    print(f"  cohort players in both files with pre-debut AAA: {len(matched):,}")
    final = matched[matched["mlb_PA"] >= MLB_PA_THRESHOLD].copy()
    print(f"  ==> FINAL cohort (200+ MLB PA): N = {len(final):,}")

    out = PROCESSED / "matched_players.csv"
    final.to_csv(out, index=False)
    print(f"\nWrote {out}")

    print("\n--- Cohort descriptives (corrected pipeline) ---")
    print(f"  MLB PA          — median {final['mlb_PA'].median():.0f}, p90 {final['mlb_PA'].quantile(0.9):.0f}, max {final['mlb_PA'].max()}")
    print(f"  MLB wRC+        — mean {final['mlb_wRC+'].mean():.1f}, median {final['mlb_wRC+'].median():.1f}, sd {final['mlb_wRC+'].std():.1f}")
    print(f"  AAA wRC+        — mean {final['aaa_wRC+'].mean():.1f}, median {final['aaa_wRC+'].median():.1f}, sd {final['aaa_wRC+'].std():.1f}")
    print(f"  age at debut    — mean {final['age_at_debut'].mean():.2f}, median {final['age_at_debut'].median():.1f}, sd {final['age_at_debut'].std():.2f}")
    print(f"  AAA seasons/player pre-debut  — median {final['aaa_seasons_predebut'].median():.0f}, max {final['aaa_seasons_predebut'].max()}")
    print(f"  AAA PA/player pre-debut       — median {final['aaa_PA_predebut'].median():.0f}")
    print(f"\n  Within-season standardized predictors (cohort mean should be >0 since cohort = successful subset):")
    for c in SEASON_ZSCORE_STATS:
        col = f"aaa_{c}_zseason"
        print(f"    {col:<30}  mean {final[col].mean():+.3f}   sd {final[col].std():.3f}")

    n = len(final)
    print("\n--- Go / no-go ---")
    if n >= 200:
        print(f"  GO. N = {n} supports the predictive analysis (after tightening the temporal filter).")
    elif n >= 80:
        print(f"  MARGINAL (N = {n}).")
    else:
        print(f"  NO-GO at current filters (N = {n}).")


if __name__ == "__main__":
    main()

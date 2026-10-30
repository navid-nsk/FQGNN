"""Vessel-clustered (MMSI-clustered) bootstrap statistics for the paper's headline comparisons.

Rationale: a gap-level bootstrap understates variance because many gaps share a
vessel. Here every bootstrap resample draws unique MMSIs with replacement (cluster
bootstrap); all gaps of a drawn vessel enter with that vessel's multiplicity.

Inputs (read-only):
  task_alpha/runs/fqnative2_5seed/seed{0..4}/per_gap.csv      (FQGNN vs GCD, 774 rows)
  task_alpha/runs/gap_sweep/our_processed/gap_6h/hermite/seed0/per_gap.csv (774 rows)

Alignment: all six per-gap files list the SAME 774 test gaps in the SAME row order
(asserted per row on the key (mmsi, gap_min, sinuosity, gcd_med) with NaN-sentinel
fill). The analysis set throughout is the 770-gap subset that EXCLUDES the 4 gaps with
NaN sinuosity (they cannot be curvature-bucketed) -- the subset on which the bucketed paper
tables are computed.  The run ends with sanity flags comparing the pooled FQ-vs-GCD ratio and
the Hermite ratio with reference values (see --pooled_ref).

Output: <FQGNN_FIGDATA>/clustered_stats.csv (tidy) + printed table.

Run from repo root:
  python -m task_alpha.analysis.clustered_stats
or directly:
  python task_alpha/analysis/clustered_stats.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]  # repository root
FQ_DIR_DEFAULT = REPO / "task_alpha" / "runs" / "fqnative2_5seed"
HERMITE_CSV = (REPO / "task_alpha" / "runs" / "gap_sweep" / "our_processed"
               / "gap_6h" / "hermite" / "seed0" / "per_gap.csv")
_FIGDATA = os.environ.get("FQGNN_FIGDATA", "").strip()        # env FQGNN_FIGDATA
OUT_CSV = (Path(_FIGDATA) if _FIGDATA else REPO / "paper_figures_tables" / "figdata") / "clustered_stats.csv"

N_SEEDS = 5
B = 10000
BOOT_SEED = 0
KEY_COLS = ["mmsi", "gap_min", "sinuosity", "gcd_med"]
NAN_SENTINEL = -999.0


# ----------------------------------------------------------------------------- load

def load_inputs(fq_dir: Path):
    fq = [pd.read_csv(fq_dir / f"seed{s}" / "per_gap.csv") for s in range(N_SEEDS)]
    herm = pd.read_csv(HERMITE_CSV)
    return fq, herm


def verify_alignment(fq: list[pd.DataFrame], herm: pd.DataFrame) -> int:
    """Assert all files list the same gaps in the same row order. Returns n aligned."""
    ref = fq[0][KEY_COLS].fillna(NAN_SENTINEL).to_numpy()
    for s in range(1, N_SEEDS):
        cand = fq[s][KEY_COLS].fillna(NAN_SENTINEL).to_numpy()
        if ref.shape != cand.shape or not (ref == cand).all():
            raise AssertionError(f"FQ seed{s} per_gap rows are not aligned with seed0")
    hval = herm[KEY_COLS].fillna(NAN_SENTINEL).to_numpy()
    if ref.shape != hval.shape or not (ref == hval).all():
        raise AssertionError("Hermite per_gap rows are not aligned with FQ seed0")
    return len(ref)


# ------------------------------------------------------------------------- bootstrap

def make_vessel_draws(mmsi: np.ndarray, rng_seed: int, n_boot: int):
    """Precompute cluster structure + one shared draw matrix (common random numbers).

    Returns (groups, draws): groups[j] = row indices of vessel j;
    draws[b] = vessel indices (with replacement) for bootstrap replicate b.
    """
    vessels = np.sort(np.unique(mmsi))
    groups = [np.flatnonzero(mmsi == v) for v in vessels]
    rng = np.random.default_rng(rng_seed)
    draws = rng.integers(0, len(vessels), size=(n_boot, len(vessels)))
    return groups, draws


def cluster_bootstrap(metric_fns: dict, groups, draws) -> dict:
    """metric_fns: name -> fn(row_index_array) -> float. Returns name -> (B,) array."""
    out = {name: np.empty(len(draws)) for name in metric_fns}
    for b, draw in enumerate(draws):
        idx = np.concatenate([groups[j] for j in draw])
        for name, fn in metric_fns.items():
            out[name][b] = fn(idx)
    return out


def pct_ci(samples: np.ndarray, lo=2.5, hi=97.5):
    return float(np.percentile(samples, lo)), float(np.percentile(samples, hi))


# --------------------------------------------------------------------------- metrics

def ratio_of_medians(model: np.ndarray, gcd: np.ndarray):
    return lambda idx: np.median(model[idx]) / np.median(gcd[idx])


def win_rate(win: np.ndarray):
    return lambda idx: win[idx].mean()


def median_diff(model: np.ndarray, gcd: np.ndarray):
    return lambda idx: np.median(model[idx] - gcd[idx])


# ------------------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run_dir", default=str(FQ_DIR_DEFAULT),
                    help="FQGNN 5-seed run dir with seed{0..4}/per_gap.csv "
                         "(default: the fqnative2_5seed fd-model run)")
    ap.add_argument("--pooled_ref", type=float, default=0.6331,
                    help="sanity reference for the pooled FQ-vs-GCD ratio (default 0.6331, the "
                         "fd-model headline; pass the matching reference when --run_dir changes, "
                         "e.g. 0.5720 for runs/fqnative_sobolev_5seed)")
    args = ap.parse_args()
    fq_dir = Path(args.run_dir)
    if not fq_dir.is_absolute():
        fq_dir = REPO / fq_dir
    print(f"FQ run dir: {fq_dir}")
    fq, herm = load_inputs(fq_dir)
    n_total = verify_alignment(fq, herm)

    # Analysis set: the paper's 770-gap subset (drop 4 NaN-sinuosity gaps).
    keep = fq[0]["sinuosity"].notna().to_numpy()
    n_dropped = int((~keep).sum())
    fq = [d.loc[keep].reset_index(drop=True) for d in fq]
    herm = herm.loc[keep].reset_index(drop=True)
    n_gaps = len(herm)

    mmsi = fq[0]["mmsi"].to_numpy()
    gcd = fq[0]["gcd_med"].to_numpy()
    herm_med = herm["model_med"].to_numpy()
    fq_meds = [d["model_med"].to_numpy() for d in fq]
    fq_pooled = np.mean(np.stack(fq_meds, axis=0), axis=0)

    # win columns: verify they equal (model_med < gcd_med)
    for s, d in enumerate(fq):
        recomputed = (d["model_med"].to_numpy() < gcd).astype(int)
        if not (recomputed == d["win"].to_numpy()).all():
            print(f"WARNING: FQ seed{s} 'win' column != (model_med < gcd_med); "
                  f"using the stored column for per-seed win rates.")
    win_seeds = [d["win"].to_numpy().astype(float) for d in fq]
    win_pooled = (fq_pooled < gcd).astype(float)      # recomputed for pooled model_med
    win_fq_vs_herm = {"pooled": (fq_pooled < herm_med).astype(float)}
    for s in range(N_SEEDS):
        win_fq_vs_herm[f"seed{s}"] = (fq_meds[s] < herm_med).astype(float)

    groups, draws = make_vessel_draws(mmsi, BOOT_SEED, B)
    n_vessels = len(groups)

    rows = []

    def add(comparison, metric, estimate, ci, seed_lbl, n_g=n_gaps, n_v=n_vessels):
        lo, hi = (np.nan, np.nan) if ci is None else ci
        rows.append(dict(comparison=comparison, metric=metric,
                         estimate=round(float(estimate), 4),
                         ci_lo=round(lo, 4) if np.isfinite(lo) else np.nan,
                         ci_hi=round(hi, 4) if np.isfinite(hi) else np.nan,
                         n_gaps=n_g, n_vessels=n_v, seed_or_pooled=seed_lbl))

    # ---- Section A: FQGNN vs GCD, per seed and pooled -------------------------
    configs = [(f"seed{s}", fq_meds[s], win_seeds[s]) for s in range(N_SEEDS)]
    configs.append(("pooled", fq_pooled, win_pooled))
    for lbl, model, win in configs:
        fns = {
            "ratio_of_medians": ratio_of_medians(model, gcd),
            "win_rate": win_rate(win),
            "median_paired_diff": median_diff(model, gcd),
        }
        boot = cluster_bootstrap(fns, groups, draws)
        all_idx = np.arange(n_gaps)
        for m, fn in fns.items():
            add("fqgnn_vs_gcd", m, fn(all_idx), pct_ci(boot[m]), lbl)

    # ---- Section B: FQGNN vs Hermite (paired, same 770 gaps) -------------------
    herm_ratio_fn = ratio_of_medians(herm_med, gcd)
    for lbl, model in [("pooled", fq_pooled)] + [(f"seed{s}", fq_meds[s])
                                                 for s in range(N_SEEDS)]:
        fq_ratio_fn = ratio_of_medians(model, gcd)
        fns = {
            "ratio_diff_fq_minus_hermite":
                lambda idx, f=fq_ratio_fn, h=herm_ratio_fn: f(idx) - h(idx),
            "win_rate_fq_over_hermite": win_rate(win_fq_vs_herm[lbl]),
        }
        boot = cluster_bootstrap(fns, groups, draws)
        all_idx = np.arange(n_gaps)
        for m, fn in fns.items():
            add("fqgnn_vs_hermite", m, fn(all_idx), pct_ci(boot[m]), lbl)

    # Vessel-level Wilcoxon signed-rank: per-vessel MEDIAN error, FQ pooled vs Hermite
    v_fq = np.array([np.median(fq_pooled[g]) for g in groups])
    v_he = np.array([np.median(herm_med[g]) for g in groups])
    w = wilcoxon(v_fq, v_he)
    add("fqgnn_vs_hermite", "wilcoxon_vessel_stat", w.statistic, None, "pooled")
    add("fqgnn_vs_hermite", "wilcoxon_vessel_p", w.pvalue, None, "pooled")
    add("fqgnn_vs_hermite", "vessel_median_err_diff_median",
        np.median(v_fq - v_he), None, "pooled")

    # ---- Section C: cluster diagnostics ----------------------------------------
    sizes = np.array([len(g) for g in groups])
    top5_share = np.sort(sizes)[::-1][:5].sum() / n_gaps
    add("cluster_diagnostics", "n_vessels", n_vessels, None, "all")
    add("cluster_diagnostics", "gaps_per_vessel_median", np.median(sizes), None, "all")
    add("cluster_diagnostics", "gaps_per_vessel_max", sizes.max(), None, "all")
    add("cluster_diagnostics", "top5_vessel_gap_share", top5_share, None, "all")
    add("cluster_diagnostics", "n_gaps_dropped_nan_sinuosity", n_dropped, None, "all",
        n_g=n_total)

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    # ---- report -----------------------------------------------------------------
    pd.set_option("display.width", 160)
    print(f"\nAligned gaps across all 6 per-gap files: {n_total} "
          f"(analysis set {n_gaps} after dropping {n_dropped} NaN-sinuosity gaps "
          f"to match the paper's headline subset)")
    print(f"Vessel clusters: {n_vessels} | B = {B} cluster-bootstrap replicates, "
          f"seed {BOOT_SEED} (shared draws across comparisons)\n")
    print(out.to_string(index=False))
    print(f"\nWrote {OUT_CSV}")

    # sanity flags vs known references
    pooled_ratio = out.query(
        "comparison=='fqgnn_vs_gcd' and metric=='ratio_of_medians' "
        "and seed_or_pooled=='pooled'")["estimate"].iloc[0]
    if abs(pooled_ratio - args.pooled_ref) > 0.02:
        print(f"SANITY FLAG: pooled ratio {pooled_ratio} far from reference {args.pooled_ref}")
    herm_ratio = herm_ratio_fn(np.arange(n_gaps))
    if abs(herm_ratio - 0.9579) > 0.02:
        print(f"SANITY FLAG: Hermite ratio {herm_ratio:.4f} far from reference 0.9579")
    return 0


if __name__ == "__main__":
    sys.exit(main())

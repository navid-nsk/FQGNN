"""Paired per-gap statistics for the init-table vs trained-table eval CSVs of _diag_init_table.

For every (family, seed, gamma) with both tables' per-gap files present:
  * paired per-gap deltas  d_i = model_med_init(i) - model_med_trained(i)
  * fraction of gaps materially changed (|d| > 0.05 NM) and improved
  * overall ratio diff + a VESSEL-CLUSTERED bootstrap CI on
    ratio_init - ratio_trained  (mmsi clusters from the row-aligned per_gap.csv of the opm run)

Reads runs/conform_baseline/init_table_anomaly/pergap_*.csv (written by _diag_init_table) and
prints one table row per (family, seed, gamma).

Run: python -m task_alpha._diag_init_table_stats
"""
from __future__ import annotations

import csv
import glob
import os
import re

import numpy as np

OUT = "task_alpha/runs/conform_baseline/init_table_anomaly"
MMSI = [int(r["mmsi"]) for r in csv.DictReader(
    open("task_alpha/runs/fqnative_opm_5seed/seed0/per_gap.csv"))]


def _load(p):
    rows = list(csv.DictReader(open(p)))
    return (np.array([float(r["model_med"]) for r in rows]),
            np.array([float(r["gcd_med"]) for r in rows]),
            np.array([float(r["sinuosity"]) for r in rows]))


def clustered_ratio_diff_ci(fq_a, fq_b, gc, mmsi, n_boot=4000, seed=20260719):
    """Bootstrap CI of median(fq_a)/median(gc) - median(fq_b)/median(gc), resampling VESSELS."""
    mmsi = np.asarray(mmsi)
    uniq = np.unique(mmsi)
    idx_by = {u: np.where(mmsi == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[u] for u in pick])
        g = np.median(gc[idx])
        ds.append(np.median(fq_a[idx]) / g - np.median(fq_b[idx]) / g)
    ds = np.array(ds)
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def main():
    pat = re.compile(r"pergap_(\w+)_s(\d+)_g([\w.]+)_(trained|init)\.csv$")
    conds = {}
    for p in sorted(glob.glob(f"{OUT}/pergap_*.csv")):
        m = pat.search(os.path.basename(p))
        if m:
            conds.setdefault((m.group(1), int(m.group(2)), m.group(3)), {})[m.group(4)] = p
    print(f"{'family':10s} {'s':>2s} {'gamma':>5s} | {'r_tr':>6s} {'r_in':>6s} {'diff':>7s} "
          f"| {'med(d)':>8s} {'%|d|>.05':>8s} {'%better':>7s} | clustered 95% CI")
    for (fam, seed, gtag), d in sorted(conds.items()):
        if "trained" not in d or "init" not in d:
            continue
        ft, gc, sn = _load(d["trained"])
        fi, gc2, _ = _load(d["init"])
        assert np.allclose(gc, gc2)
        rt = float(np.median(ft) / np.median(gc))
        ri = float(np.median(fi) / np.median(gc))
        dd = fi - ft
        chg = np.abs(dd) > 0.05
        pb = 100 * float(np.mean(dd[chg] < 0)) if chg.any() else float("nan")
        lo, hi = clustered_ratio_diff_ci(fi, ft, gc, MMSI)
        print(f"{fam:10s} {seed:2d} {gtag:>5s} | {rt:.4f} {ri:.4f} {ri - rt:+.4f} "
              f"| {np.median(dd):+.5f} {100 * float(np.mean(chg)):7.1f}% {pb:6.1f}% "
              f"| [{lo:+.4f}, {hi:+.4f}]{'  EXCLUDES 0' if lo * hi > 0 else ''}")


if __name__ == "__main__":
    main()

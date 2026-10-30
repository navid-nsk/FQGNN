"""Per-cell vessel-clustered bootstrap: FQGNN (Op-M stack, CLEAN exact-table re-evals) vs the
cubic-Hermite classical control, on ALL 8 (track, gap) cells.

Extends clustered_stats.py (same B=10000 vessel bootstrap, same percentile CIs, same pooled
semantics: per-gap FQ error = mean over the 5 seeds' model_med) from the single 6h headline
cell to the full landscape.  Alignment is by gap_id (the common_gaps per-gap schema) for the
sweep cells and by the KEY_COLS positional check for the headline cell (whose runner-native
per_gap.csv has no gap_id column).  Analysis set per cell = finite-sinuosity gaps only,
matching every reported bucket number (common_gaps.bucket_rows).

Inputs (all local):
  FQ    : runs/gap_sweep_sobolev{,_fd}_opm/<track>/gap_<G>h/fqgnn/seed{0..4}/per_gap_clean.csv
          (our/6h: runs/fqnative_opmstack_5seed_clean/seed{0..4}/per_gap.csv, staged copies)
  Hermite: runs/gap_sweep/<track>/gap_<G>h/hermite/seed0/per_gap.csv (deterministic)

Run:  python task_alpha/analysis/clustered_stats_cells.py
Out:  <FQGNN_FIGDATA>/clustered_stats_cells.csv
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "task_alpha" / "runs"
_FIGDATA = os.environ.get("FQGNN_FIGDATA", "").strip()        # env FQGNN_FIGDATA
OUT_CSV = (Path(_FIGDATA) if _FIGDATA else REPO / "paper_figures_tables" / "figdata") / "clustered_stats_cells.csv"

N_SEEDS = 5
B = 10000
BOOT_SEED = 0
KEY_COLS = ["mmsi", "gap_min", "sinuosity", "gcd_med"]
NAN_SENTINEL = -999.0

CELLS = [
    # (track, gap_h, fq_dir or None-for-headline)
    ("our_processed", 3, RUNS / "gap_sweep_sobolev_opm" / "our_processed" / "gap_3h" / "fqgnn"),
    ("our_processed", 6, None),
    ("our_processed", 12, RUNS / "gap_sweep_sobolev_opm" / "our_processed" / "gap_12h" / "fqgnn"),
    ("our_processed", 24, RUNS / "gap_sweep_sobolev_opm" / "our_processed" / "gap_24h" / "fqgnn"),
    ("standard_process", 2, RUNS / "gap_sweep_sobolev_fd_opm" / "standard_process" / "gap_2h" / "fqgnn"),
    ("standard_process", 6, RUNS / "gap_sweep_sobolev_fd_opm" / "standard_process" / "gap_6h" / "fqgnn"),
    ("standard_process", 12, RUNS / "gap_sweep_sobolev_fd_opm" / "standard_process" / "gap_12h" / "fqgnn"),
    ("standard_process", 24, RUNS / "gap_sweep_sobolev_fd_opm" / "standard_process" / "gap_24h" / "fqgnn"),
]
HEADLINE_DIR = RUNS / "fqnative_opmstack_5seed_clean"


def load_cell(track: str, gap: int, fq_dir):
    herm = pd.read_csv(RUNS / "gap_sweep" / track / f"gap_{gap}h" / "hermite" / "seed0"
                       / "per_gap.csv")
    if fq_dir is None:                                     # headline: positional (no gap_id col)
        fq = [pd.read_csv(HEADLINE_DIR / f"seed{s}" / "per_gap.csv") for s in range(N_SEEDS)]
        ref = fq[0][KEY_COLS].fillna(NAN_SENTINEL).to_numpy()
        for s in range(1, N_SEEDS):
            assert (fq[s][KEY_COLS].fillna(NAN_SENTINEL).to_numpy() == ref).all(), \
                f"headline seed{s} misaligned"
        assert (herm[KEY_COLS].fillna(NAN_SENTINEL).to_numpy() == ref).all(), \
            "headline vs hermite misaligned"
        base = fq[0][["mmsi", "sinuosity", "gcd_med"]].copy()
        for s in range(N_SEEDS):
            base[f"fq{s}"] = fq[s]["model_med"].to_numpy()
        base["herm"] = herm["model_med"].to_numpy()
    else:                                                  # sweep cells: join on gap_id
        fq = [pd.read_csv(fq_dir / f"seed{s}" / "per_gap_clean.csv") for s in range(N_SEEDS)]
        base = fq[0][["gap_id", "mmsi", "sinuosity", "gcd_med"]].copy()
        n0 = len(base)
        for s in range(1, N_SEEDS):
            assert (fq[s]["gap_id"].to_numpy() == base["gap_id"].to_numpy()).all(), \
                f"{fq_dir} seed{s} gap_id set differs"
        for s in range(N_SEEDS):
            base[f"fq{s}"] = fq[s]["model_med"].to_numpy()
        hm = herm[["gap_id", "mmsi", "model_med"]].rename(
            columns={"mmsi": "mmsi_h", "model_med": "herm"})
        base = base.merge(hm, on="gap_id", how="inner")
        assert (base["mmsi"] == base["mmsi_h"]).all(), "mmsi mismatch after gap_id join"
        base = base.drop(columns=["mmsi_h"])
        if len(base) != n0:
            print(f"  [note] {track}/gap_{gap}h: inner join kept {len(base)}/{n0} FQ gaps")
    base = base[np.isfinite(pd.to_numeric(base["sinuosity"], errors="coerce"))]
    return base.reset_index(drop=True)


def main():
    rows = []
    for track, gap, fq_dir in CELLS:
        cell = f"{'our' if track == 'our_processed' else 'std'}_{gap}h"
        d = load_cell(track, gap, fq_dir)
        n_gaps = len(d)
        mmsi = d["mmsi"].to_numpy()
        gcd = d["gcd_med"].to_numpy()
        herm = d["herm"].to_numpy()
        fq_meds = [d[f"fq{s}"].to_numpy() for s in range(N_SEEDS)]
        fq_pool = np.mean(np.stack(fq_meds), axis=0)

        vessels = np.sort(np.unique(mmsi))
        groups = [np.flatnonzero(mmsi == v) for v in vessels]
        rng = np.random.default_rng(BOOT_SEED)
        draws = rng.integers(0, len(vessels), size=(B, len(vessels)))
        n_v = len(vessels)

        def ratio(model, idx):
            return np.median(model[idx]) / np.median(gcd[idx])

        def add(metric, est, ci, lbl):
            lo, hi = (np.nan, np.nan) if ci is None else ci
            est = float(est) if metric.endswith("_p") else round(float(est), 4)  # p exact
            rows.append(dict(cell=cell, metric=metric, estimate=est,
                             ci_lo=round(lo, 4) if np.isfinite(lo) else np.nan,
                             ci_hi=round(hi, 4) if np.isfinite(hi) else np.nan,
                             n_gaps=n_gaps, n_vessels=n_v, seed_or_pooled=lbl))

        all_idx = np.arange(n_gaps)
        add("herm_ratio", ratio(herm, all_idx), None, "point")
        for lbl, model in [("pooled", fq_pool)] + [(f"seed{s}", fq_meds[s])
                                                   for s in range(N_SEEDS)]:
            diff_b = np.empty(B)
            win_b = np.empty(B)
            win_arr = (model < herm).astype(float)
            for b, draw in enumerate(draws):
                idx = np.concatenate([groups[j] for j in draw])
                diff_b[b] = ratio(model, idx) - ratio(herm, idx)
                win_b[b] = win_arr[idx].mean()
            est = ratio(model, all_idx) - ratio(herm, all_idx)
            add("ratio_diff_fq_minus_hermite", est,
                (float(np.percentile(diff_b, 2.5)), float(np.percentile(diff_b, 97.5))), lbl)
            if lbl == "pooled":
                add("fq_ratio", ratio(model, all_idx), None, "pooled")
                add("win_rate_fq_over_hermite", win_arr.mean(),
                    (float(np.percentile(win_b, 2.5)), float(np.percentile(win_b, 97.5))),
                    "pooled")
        v_fq = np.array([np.median(fq_pool[g]) for g in groups])
        v_he = np.array([np.median(herm[g]) for g in groups])
        w = wilcoxon(v_fq, v_he)
        add("wilcoxon_vessel_p", w.pvalue, None, "pooled")

        pd_rows = [r for r in rows if r["cell"] == cell and r["metric"] ==
                   "ratio_diff_fq_minus_hermite" and r["seed_or_pooled"] == "pooled"][0]
        verdict = ("FQ SIGNIF" if pd_rows["ci_hi"] < 0 else
                   "HERM SIGNIF" if pd_rows["ci_lo"] > 0 else "parity")
        print(f"[{cell}] n={n_gaps} vessels={n_v} | ratio_diff {pd_rows['estimate']:+.4f} "
              f"[{pd_rows['ci_lo']:+.4f},{pd_rows['ci_hi']:+.4f}] -> {verdict} | "
              f"wilcoxon p={w.pvalue:.3e}", flush=True)

    # Holm correction of the 8 per-cell vessel-level Wilcoxon tests (exact p's, cummax rule)
    praw = [(r["cell"], r["estimate"], r["n_gaps"], r["n_vessels"])
            for r in rows if r["metric"] == "wilcoxon_vessel_p"]
    order = np.argsort([p for _, p, _, _ in praw])
    m = len(praw)
    adj, run = {}, 0.0
    for rank, oi in enumerate(order):
        cell_o, p_o, _, _ = praw[oi]
        run = max(run, min(1.0, (m - rank) * p_o))
        adj[cell_o] = run
    for cell_o, p_o, ng, nv in praw:
        rows.append(dict(cell=cell_o, metric="wilcoxon_vessel_p_holm", estimate=adj[cell_o],
                         ci_lo=np.nan, ci_hi=np.nan, n_gaps=ng, n_vessels=nv,
                         seed_or_pooled="pooled"))
    print("Holm-adjusted (8 cells): max adjusted p = "
          f"{max(adj.values()):.3e} ({max(adj, key=adj.get)})")

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV} ({len(out)} rows)")


if __name__ == "__main__":
    main()

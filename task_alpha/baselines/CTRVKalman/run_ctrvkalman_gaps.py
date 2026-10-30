"""Runner: two-sided IMM-CTRV Kalman baseline, across the two-track gap-sweep cells.

Process noise (sigma_a_ct, sigma_om_ct -- how much the CT mode trusts a manoeuvre) is chosen ONCE via
a light grid search on the our_processed/gap_6h VALIDATION split (--tune), then FROZEN and reused for
every track/gap cell -- exactly how any deployed Kalman filter's noise is set, never re-tuned per test
condition. Measurement noise (r_var, AIS position accuracy) and the CV-mode noise are textbook
defaults, not tuned at all.

Deterministic given the frozen hyperparameters -- ONE evaluation pass per cell (seed0), matching the
GCD/BF-BiLSTM "deterministic single-seed reference" convention (no learned weights, no random init).
Output layout identical to every other gap-sweep model:
  task_alpha/runs/gap_sweep/<track>/gap_<G>h/ctrv_kalman/seed0/{per_gap.csv, test_buckets.csv, epochs.csv}
  task_alpha/runs/gap_sweep/<track>/gap_<G>h/ctrv_kalman/{all_seeds_epochs.csv, all_seeds_test_buckets.csv,
                                                            summary.csv}

Run (CPU only, no GPU needed):
  python -m task_alpha.baselines.CTRVKalman.run_ctrvkalman_gaps --tune
  python -m task_alpha.baselines.CTRVKalman.run_ctrvkalman_gaps --all
  python -m task_alpha.baselines.CTRVKalman.run_ctrvkalman_gaps --track our_processed --gap 12
  python -m task_alpha.baselines.CTRVKalman.run_ctrvkalman_gaps --smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time

# Many rapid small linalg calls (np.linalg.inv/solve on tiny 2x2/4x4 matrices, thousands of times
# per graph) can deadlock a BLAS backend's internal thread pool (the process then hangs at 0% CPU).
# Forcing single-threaded BLAS avoids this and must be set BEFORE numpy imports its backend.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))          # repository root
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np

from task_alpha.baselines.common_gaps import (anchor_states_enu, bucket_rows, enu_pred_to_latlon,
                                              gap_row, model_errors_latlon, write_buckets, write_epochs,
                                              write_per_gap)
from task_alpha.baselines.CTRVKalman.ctrv_imm import ctrv_kalman_predict
from task_alpha.data import collate_graphs_interp  # noqa: F401  (registers RunGraph for unpickling)

DATA_ROOT = "task_alpha/data"
SWEEP_ROOT = "task_alpha/runs/gap_sweep"
MODEL_NAME = "ctrv_kalman"
TRACK_GAPS = {"our_processed": (3, 6, 12, 24), "standard_process": (2, 6, 12, 24)}
HP_PATH = os.path.join(HERE, "tuned_hp.json")

# textbook defaults, never tuned: AIS position accuracy (~25 m std), CV-mode process noise (near-straight prior)
DEFAULT_HP = dict(sigma_a_cv=0.05, sigma_om_cv=0.005, sigma_a_ct=0.3, sigma_om_ct=0.03, r_var=25.0 ** 2)
GRID_SIGMA_A_CT = (0.1, 0.3, 0.6)          # m/s^2 -- manoeuvre longitudinal-accel std
GRID_SIGMA_OM_CT = (0.01, 0.03, 0.08)      # rad/s -- manoeuvre turn-rate std
TUNE_TRACK, TUNE_GAP = "our_processed", 6


def load_pkl(track, gap, split):
    with open(f"{DATA_ROOT}/{track}/gap_{gap}h/{split}.pkl", "rb") as f:
        return pickle.load(f)


def load_hp():
    if os.path.exists(HP_PATH):
        with open(HP_PATH) as f:
            return json.load(f)
    print(f"[ctrv_kalman] WARNING: {HP_PATH} not found -- using untuned DEFAULT_HP "
          f"(run --tune first)", flush=True)
    return dict(DEFAULT_HP)


def _predict_graph(g, hp):
    before, after, held_t, hidx, ref = anchor_states_enu(g)
    if len(before["t"]) == 0 or len(after["t"]) == 0 or len(held_t) == 0:
        return None
    try:
        px, py = ctrv_kalman_predict(before, after, held_t, hp)
    except np.linalg.LinAlgError:
        return None                                  # degenerate covariance on this one graph -- skip it
    if not (np.all(np.isfinite(px)) and np.all(np.isfinite(py))):
        return None                                  # defensive: a non-finite prediction is not scored
    plat, plon = enu_pred_to_latlon(px, py, ref)
    return model_errors_latlon(g, plat, plon)


def _score_ratio(graphs, hp):
    """Overall median-error ratio vs GCD over a set of graphs (for the tuning grid)."""
    fq, gc = [], []
    from task_alpha.metrics import gcd_errors
    for g in graphs:
        err = _predict_graph(g, hp)
        if err is None or len(err) == 0:
            continue
        fq.append(float(np.median(err))); gc.append(float(np.median(gcd_errors(g))))
    if not fq:
        return float("nan")
    return float(np.median(fq) / np.median(gc))


def tune():
    va = load_pkl(TUNE_TRACK, TUNE_GAP, "val")
    print(f"[ctrv_kalman][tune] {TUNE_TRACK}/gap_{TUNE_GAP}h val: {len(va)} graphs, "
          f"grid {len(GRID_SIGMA_A_CT)}x{len(GRID_SIGMA_OM_CT)}", flush=True)
    best_ratio, best_hp = float("inf"), None
    rows = []
    for sa in GRID_SIGMA_A_CT:
        for som in GRID_SIGMA_OM_CT:
            hp = dict(DEFAULT_HP, sigma_a_ct=sa, sigma_om_ct=som)
            ratio = _score_ratio(va, hp)
            rows.append((sa, som, ratio))
            print(f"  sigma_a_ct={sa:.2f}  sigma_om_ct={som:.3f}  ->  val ratio {ratio:.4f}", flush=True)
            if ratio < best_ratio:
                best_ratio, best_hp = ratio, hp
    print(f"[ctrv_kalman][tune] BEST: sigma_a_ct={best_hp['sigma_a_ct']} "
          f"sigma_om_ct={best_hp['sigma_om_ct']}  (val ratio {best_ratio:.4f})", flush=True)
    with open(HP_PATH, "w") as f:
        json.dump(best_hp, f, indent=2)
    print(f"[ctrv_kalman][tune] saved -> {HP_PATH}", flush=True)
    return best_hp


def run_cell(track, gap, hp):
    te = load_pkl(track, gap, "test")
    n = len(te)
    print(f"[ctrv_kalman] {track}/gap_{gap}h: {n} test graphs", flush=True)
    rows, skipped = [], 0
    t0 = time.time()
    for i, g in enumerate(te):
        err = _predict_graph(g, hp)
        if err is None:
            skipped += 1; continue
        if len(err):
            rows.append(gap_row(i, g, err))
        if (i + 1) % 500 == 0 or (i + 1) == n:
            print(f"[ctrv_kalman] {track}/gap_{gap}h: {i + 1}/{n} graphs "
                  f"({time.time() - t0:.0f}s elapsed)", flush=True)
    if skipped:
        print(f"[ctrv_kalman] {track}/gap_{gap}h: skipped {skipped} graphs (no anchors on one side, "
              f"degenerate covariance, or non-finite prediction)", flush=True)
    brows = bucket_rows(rows)

    cdir = f"{SWEEP_ROOT}/{track}/gap_{gap}h/{MODEL_NAME}"
    sdir = f"{cdir}/seed0"
    os.makedirs(sdir, exist_ok=True)
    write_per_gap(f"{sdir}/per_gap.csv", rows)
    write_buckets(f"{sdir}/test_buckets.csv", brows)
    ov = brows[0]
    write_epochs(f"{sdir}/epochs.csv", [{"epoch": 0, "train_loss": "", "val_ratio": ov["ratio"],
                                         "val_model_med": ov["model_med"], "val_gcd_med": ov["gcd_med"],
                                         "note": f"deterministic given frozen hp={json.dumps(hp)}"}])
    _rebuild_aggregates(cdir)
    print(f"[ctrv_kalman][TEST] {track}/gap_{gap}h  n={ov['n']}  GCD {ov['gcd_med']}  "
          f"model {ov['model_med']}  ratio {ov['ratio']}  win {ov['win_pct']}%", flush=True)
    for r in brows[1:]:
        print(f"   {r['bucket']:<11} n={r['n']:4d}  GCD {r['gcd_med']}  model {r['model_med']}  "
              f"ratio {r['ratio']}", flush=True)
    return ov


def _rebuild_aggregates(cdir):
    """Deterministic single-seed cell: all_seeds_*.csv + summary.csv mirror the seed0 files exactly
    (n=1 'seed', STD=0), matching the GCD/BF-BiLSTM 'deterministic single-seed reference' convention."""
    sdir = f"{cdir}/seed0"
    with open(f"{sdir}/epochs.csv", newline="") as f:
        ep = list(csv.DictReader(f))
    with open(f"{sdir}/test_buckets.csv", newline="") as f:
        bk = list(csv.DictReader(f))
    all_ep = [{"seed": 0, **r} for r in ep]
    all_bk = [{"seed": 0, **r} for r in bk]
    with open(f"{cdir}/all_seeds_epochs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_ep[0].keys())); w.writeheader(); w.writerows(all_ep)
    with open(f"{cdir}/all_seeds_test_buckets.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_bk[0].keys())); w.writeheader(); w.writerows(all_bk)
    ov = bk[0]
    summary = [{"seed": 0, "n_gaps": int(ov["n"]), "gcd_med": float(ov["gcd_med"]),
               "model_med": float(ov["model_med"]), "ratio": float(ov["ratio"]),
               "win_pct": float(ov["win_pct"]), "epochs_trained": 0, "best_val": float(ov["ratio"])}]
    with open(f"{cdir}/summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
        wtr = csv.writer(f)
        wtr.writerow(["MEAN", "", summary[0]["gcd_med"], round(summary[0]["model_med"], 3),
                      round(summary[0]["ratio"], 4), round(summary[0]["win_pct"], 1), "", ""])
        wtr.writerow(["STD", "", "", 0.0, 0.0, 0.0, "", ""])


def main():
    ap = argparse.ArgumentParser(description="IMM-CTRV Kalman baseline (vanilla, deterministic given frozen hp)")
    ap.add_argument("--tune", action="store_true", help="grid-search process noise on our_processed/gap_6h val, "
                                                        "save tuned_hp.json, then exit")
    ap.add_argument("--all", action="store_true", help="run all 8 (track, gap) cells")
    ap.add_argument("--track"); ap.add_argument("--gap", type=int)
    ap.add_argument("--smoke", action="store_true", help="data/_smoke/gap_6h only")
    args = ap.parse_args()

    if args.tune:
        tune(); return

    hp = load_hp()
    print(f"[ctrv_kalman] using hp = {hp}", flush=True)

    if args.smoke:
        cells = [("_smoke", 6)]
    elif args.all:
        cells = [(t, g) for t in ("our_processed", "standard_process") for g in TRACK_GAPS[t]]
    elif args.track and args.gap:
        cells = [(args.track, args.gap)]
    else:
        ap.error("choose --tune, --all, --smoke, or --track/--gap")

    for track, gap in cells:
        run_cell(track, gap, hp)


if __name__ == "__main__":
    main()

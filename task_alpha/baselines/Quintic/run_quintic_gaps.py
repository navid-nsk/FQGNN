"""Runner: quintic (min-jerk) Hermite baseline, across the two-track gap-sweep cells.

Deterministic -- ONE evaluation pass per cell (seed0), matching the GCD/BF-BiLSTM/Hermite
"deterministic single-seed reference" convention (no training, no random init, so repeated seeds
would be identical). Output layout is IDENTICAL to every other gap-sweep model, so
assemble_gap_sweep.py reads it like any other model ("quintic" is in its MODELS tuple):
  task_alpha/runs/gap_sweep/<track>/gap_<G>h/quintic/seed0/{per_gap.csv, test_buckets.csv, epochs.csv}
  task_alpha/runs/gap_sweep/<track>/gap_<G>h/quintic/{all_seeds_epochs.csv, all_seeds_test_buckets.csv,
                                                       summary.csv}

Run (any env with numpy/torch for unpickling RunGraph; CPU only, no GPU needed):
  python -m task_alpha.baselines.Quintic.run_quintic_gaps --all
  python -m task_alpha.baselines.Quintic.run_quintic_gaps --track our_processed --gap 12
  python -m task_alpha.baselines.Quintic.run_quintic_gaps --smoke
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))          # repository root
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from task_alpha.baselines.common_gaps import (anchor_states_enu, bucket_rows, enu_pred_to_latlon,
                                              gap_row, model_errors_latlon, write_buckets, write_epochs,
                                              write_per_gap)
from task_alpha.baselines.Quintic.quintic_model import quintic_predict
from task_alpha.data import collate_graphs_interp  # noqa: F401  (registers RunGraph for unpickling)

DATA_ROOT = "task_alpha/data"
SWEEP_ROOT = "task_alpha/runs/gap_sweep"
MODEL_NAME = "quintic"
TRACK_GAPS = {"our_processed": (3, 6, 12, 24), "standard_process": (2, 6, 12, 24)}


def load_test(track, gap):
    with open(f"{DATA_ROOT}/{track}/gap_{gap}h/test.pkl", "rb") as f:
        return pickle.load(f)


def run_cell(track, gap):
    te = load_test(track, gap)
    print(f"[quintic] {track}/gap_{gap}h: {len(te)} test graphs", flush=True)
    rows, skipped = [], 0
    for i, g in enumerate(te):
        before, after, held_t, hidx, ref = anchor_states_enu(g)
        if len(before["t"]) == 0 or len(after["t"]) == 0 or len(held_t) == 0:
            skipped += 1
            continue
        px, py = quintic_predict(before, after, held_t)
        plat, plon = enu_pred_to_latlon(px, py, ref)
        err = model_errors_latlon(g, plat, plon)
        if len(err):
            rows.append(gap_row(i, g, err))
    if skipped:
        print(f"[quintic] {track}/gap_{gap}h: skipped {skipped} graphs (no anchors on one side)", flush=True)
    brows = bucket_rows(rows)

    cdir = f"{SWEEP_ROOT}/{track}/gap_{gap}h/{MODEL_NAME}"
    sdir = f"{cdir}/seed0"
    os.makedirs(sdir, exist_ok=True)
    write_per_gap(f"{sdir}/per_gap.csv", rows)
    write_buckets(f"{sdir}/test_buckets.csv", brows)
    ov = brows[0]
    write_epochs(f"{sdir}/epochs.csv", [{"epoch": 0, "train_loss": "", "val_ratio": ov["ratio"],
                                         "val_model_med": ov["model_med"], "val_gcd_med": ov["gcd_med"],
                                         "note": "deterministic closed-form -- no training, no seed"}])
    _rebuild_aggregates(cdir)
    print(f"[quintic][TEST] {track}/gap_{gap}h  n={ov['n']}  GCD {ov['gcd_med']}  "
          f"model {ov['model_med']}  ratio {ov['ratio']}  win {ov['win_pct']}%", flush=True)
    for r in brows[1:]:
        print(f"   {r['bucket']:<11} n={r['n']:4d}  GCD {r['gcd_med']}  model {r['model_med']}  "
              f"ratio {r['ratio']}", flush=True)
    return ov


def _rebuild_aggregates(cdir):
    """Deterministic single-seed cell: all_seeds_*.csv + summary.csv mirror the seed0 files exactly
    (n=1 'seed', STD=0), matching the GCD/BF-BiLSTM/Hermite 'deterministic single-seed reference'
    convention used elsewhere in the gap-sweep (no repeated randomness to average over)."""
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
    ap = argparse.ArgumentParser(description="Quintic (min-jerk) Hermite baseline (vanilla, deterministic)")
    ap.add_argument("--all", action="store_true", help="run all 8 (track, gap) cells")
    ap.add_argument("--track"); ap.add_argument("--gap", type=int)
    ap.add_argument("--smoke", action="store_true", help="data/_smoke/gap_6h only")
    args = ap.parse_args()

    if args.smoke:
        cells = [("_smoke", 6)]
    elif args.all:
        cells = [(t, g) for t in ("our_processed", "standard_process") for g in TRACK_GAPS[t]]
    elif args.track and args.gap:
        cells = [(args.track, args.gap)]
    else:
        ap.error("choose --all, --smoke, or --track/--gap")

    for track, gap in cells:
        run_cell(track, gap)


if __name__ == "__main__":
    main()

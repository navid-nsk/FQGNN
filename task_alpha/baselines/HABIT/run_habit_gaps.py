"""Runner: HABIT (EDBT 2026) across the two-track gap-sweep cells.

Deterministic -- ONE evaluation pass per cell (seed0), matching the GCD / Hermite / BF-BiLSTM
"deterministic single-seed reference" convention: HABIT has no training and no random initialisation,
so repeated seeds would produce byte-identical files.  Output layout is identical to every other
gap-sweep model so the assembler picks it up unchanged:
  task_alpha/runs/gap_sweep/<track>/gap_<G>h/habit/seed0/{per_gap.csv, test_buckets.csv, epochs.csv}
  task_alpha/runs/gap_sweep/<track>/gap_<G>h/habit/{all_seeds_*.csv, summary.csv}
plus a habit_diag.csv per cell recording graph size, A*-failure rate and route length -- the A* failure
rate is REPORTED, never silently absorbed into the score.

  python -m task_alpha.baselines.HABIT.run_habit_gaps --all --resolution 6
  python -m task_alpha.baselines.HABIT.run_habit_gaps --track our_processed --gap 12
  python -m task_alpha.baselines.HABIT.run_habit_gaps --smoke
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import pickle
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from task_alpha.baselines.common_gaps import (bucket_rows, gap_row, model_errors_latlon,  # noqa: E402
                                              write_buckets, write_epochs, write_per_gap)
from task_alpha.baselines.HABIT.habit_model import (HabitAdapter, build_corpus_csv,  # noqa: E402
                                                    predict_graph)
from task_alpha.data import collate_graphs_interp  # noqa: F401  (registers RunGraph)

DATA_ROOT = "task_alpha/data"
SWEEP_ROOT = "task_alpha/runs/gap_sweep"
MODEL_NAME = "habit"
TRACK_GAPS = {"our_processed": (3, 6, 12, 24), "standard_process": (2, 6, 12, 24)}
SCRATCH = os.path.join(HERE, "_scratch")

SCHEMA = {"id": "TRIP", "lon": "LON", "lat": "LAT", "t": "TIMESTAMP",
          "sog": "SPEED", "cog": "COURSE", "trip": "TRIP"}
MAX_GAP = 25            # vendor default; verified non-binding on our data (identical edges for 25..200)


def _load(track, gap, split):
    with open(f"{DATA_ROOT}/{track}/gap_{gap}h/{split}.pkl", "rb") as f:
        return pickle.load(f)


def corpus_path(track, gap):
    return os.path.join(SCRATCH, f"corpus_{track}_{gap}h.csv")


def build_corpus(track, gap):
    """Phase A, run in its OWN process: load train+val+test, write the observed-ping corpus, exit.

    The interpolated standard_process RunGraphs occupy ~45 GB. `del` + `gc.collect()` frees the Python
    objects but CPython does not return those arenas to the OS, so the working set stays ~48 GB and
    DuckDB then allocates on top of it, driving a 64 GB machine ~50 GB into swap. Only process exit
    actually returns the memory, so corpus building is isolated in a subprocess.
    """
    os.makedirs(SCRATCH, exist_ok=True)
    tr, va, te = _load(track, gap, "train"), _load(track, gap, "val"), _load(track, gap, "test")
    n = build_corpus_csv(list(tr) + list(va) + list(te), corpus_path(track, gap))  # OBSERVED pings only
    print(f"[{MODEL_NAME}] corpus {track}/gap_{gap}h: {n} observed pings", flush=True)
    return n


def run_cell(track, gap, resolution, max_gap=MAX_GAP):
    os.makedirs(SCRATCH, exist_ok=True)
    csv_path = corpus_path(track, gap)
    t0 = time.time()
    if not os.path.exists(csv_path):
        subprocess.check_call([sys.executable, "-u", "-m", "task_alpha.baselines.HABIT.run_habit_gaps",
                               "--build-corpus", "--track", track, "--gap", str(gap)])
    nrows = sum(1 for _ in open(csv_path)) - 1        # header

    te = _load(track, gap, "test")                    # only the test split is needed from here on
    gc.collect()

    imp = HabitAdapter(resolution=resolution, max_gap=max_gap, schema=SCHEMA)
    nn, ne = imp.data_load(csv_path)
    print(f"[{MODEL_NAME}] {track}/gap_{gap}h  corpus {nrows} pings  graph {nn} nodes / {ne} edges "
          f"(res {resolution}, max_gap {max_gap})", flush=True)

    rows, nfb, routes, detours, skipped = [], 0, [], [], 0
    for i, g in enumerate(te):
        held = g.mask.numpy().astype(bool)
        if held.sum() == 0 or not held[1:-1].any():
            skipped += 1
            continue
        plat, plon, fb, nv, det = predict_graph(imp, g)
        nfb += int(fb)
        if not fb:
            routes.append(nv); detours.append(det)
        err = model_errors_latlon(g, plat, plon)
        if len(err):
            rows.append(gap_row(i, g, err))

    brows = bucket_rows(rows)
    ov = brows[0]
    fail_pct = 100.0 * nfb / max(len(te), 1)
    med_route = float(np.median(routes)) if routes else float("nan")
    # detour = route arclength / anchor chord.  A* SUCCEEDS on these, so they are invisible in
    # fail_pct; a >3x detour means HABIT had no historical corridor and wandered.
    det = np.array(detours) if detours else np.array([np.nan])
    med_detour = float(np.nanmedian(det))
    p99_detour = float(np.nanpercentile(det, 99))
    absurd_pct = 100.0 * float(np.mean(det > 3.0))
    dt = time.time() - t0

    cdir = f"{SWEEP_ROOT}/{track}/gap_{gap}h/{MODEL_NAME}"
    sdir = f"{cdir}/seed0"
    os.makedirs(sdir, exist_ok=True)
    write_per_gap(f"{sdir}/per_gap.csv", rows)
    write_buckets(f"{sdir}/test_buckets.csv", brows)
    write_epochs(f"{sdir}/epochs.csv", [{
        "epoch": 0, "train_loss": "", "val_ratio": ov["ratio"], "val_model_med": ov["model_med"],
        "val_gcd_med": ov["gcd_med"],
        "note": f"deterministic route lookup -- no training, no seed; res={resolution} "
                f"max_gap={max_gap} astar_fail={fail_pct:.2f}% med_route_cells={med_route:.0f} "
                f"med_detour={med_detour:.2f}x detour_gt3x={absurd_pct:.2f}%"}])
    with open(f"{cdir}/habit_diag.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track", "gap_h", "resolution", "max_gap", "corpus_pings", "nodes", "edges",
                    "n_test", "astar_fail_pct", "median_route_cells", "median_detour",
                    "p99_detour", "detour_gt3x_pct", "seconds"])
        w.writerow([track, gap, resolution, max_gap, nrows, nn, ne, len(te),
                    round(fail_pct, 3), round(med_route, 1), round(med_detour, 3),
                    round(p99_detour, 2), round(absurd_pct, 3), round(dt, 1)])
    _rebuild_aggregates(cdir)

    print(f"[{MODEL_NAME}][TEST] {track}/gap_{gap}h  n={ov['n']}  GCD {ov['gcd_med']}  "
          f"model {ov['model_med']}  ratio {ov['ratio']}  win {ov['win_pct']}%  "
          f"| A*fail {fail_pct:.1f}%  route {med_route:.0f} cells  detour {med_detour:.2f}x "
          f"(>3x: {absurd_pct:.2f}%)  ({dt:.0f}s)", flush=True)
    if skipped:
        print(f"[{MODEL_NAME}] skipped {skipped} graphs (no two-sided anchors)", flush=True)
    for r in brows[1:]:
        print(f"   {r['bucket']:<11} n={r['n']:4d}  GCD {r['gcd_med']}  model {r['model_med']}  "
              f"ratio {r['ratio']}", flush=True)
    return ov


def _rebuild_aggregates(cdir):
    """Deterministic single-seed cell: all_seeds_*.csv + summary.csv mirror seed0 exactly (STD = 0),
    the convention already used by GCD / Hermite / BF-BiLSTM in this sweep."""
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
    ap = argparse.ArgumentParser(description="HABIT H3-route baseline (vanilla, deterministic)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--track"); ap.add_argument("--gap", type=int)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resolution", type=int, default=6, help="H3 resolution (see smoke_habit.py scan)")
    ap.add_argument("--max-gap", type=int, default=MAX_GAP, help="25 = vendor default")
    ap.add_argument("--build-corpus", action="store_true",
                    help="internal: phase A -- write the corpus CSV and exit (frees ~45 GB on exit)")
    a = ap.parse_args()

    if a.build_corpus:
        if not (a.track and a.gap):
            ap.error("--build-corpus needs --track and --gap")
        build_corpus(a.track, a.gap)
        return

    if a.smoke:
        cells = [("_smoke", 6)]
    elif a.all:
        cells = [(t, g) for t in ("our_processed", "standard_process") for g in TRACK_GAPS[t]]
    elif a.track and a.gap:
        cells = [(a.track, a.gap)]
    else:
        ap.error("choose --all, --smoke, or --track/--gap")

    for track, gap in cells:
        run_cell(track, gap, a.resolution, a.max_gap)


if __name__ == "__main__":
    main()

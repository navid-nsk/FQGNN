"""Smoke test for the HABIT adapter: is it reasonable, and which (resolution, max_gap) let it FUNCTION?

HABIT's two structural knobs are dataset properties, not tuning dials:
  * `resolution` sets the H3 cell size, i.e. the coordinate-quantisation FLOOR of the method
    (predictions are per-cell median lat/lon, so error can never beat ~half a cell).
  * `max_gap` is the maximum H3 grid distance an edge may span.  Two consecutive pings farther apart
    than `max_gap` cells produce NO edge.  Our median inter-ping spacing is ~11 min; at 12 kn that is
    ~4 km, which at the vanilla resolution 9 (~0.17 km edge) is ~23 cells -- right at the vanilla
    `max_gap=25` cliff.  Defaults tuned on dense European coastal AIS will simply shatter the graph.

We therefore scan the grid and pick the setting where HABIT WORKS BEST (highest A* success, lowest
error), i.e. we hand the baseline its strongest honest configuration.  We are not tuning it to lose.

  python -m task_alpha.baselines.HABIT.smoke_habit
"""
from __future__ import annotations

import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from task_alpha.baselines.common_gaps import gap_row, model_errors_latlon  # noqa: E402
from task_alpha.baselines.HABIT.habit_model import (HabitAdapter, build_corpus_csv,  # noqa: E402
                                                    predict_graph)
from task_alpha.data import collate_graphs_interp  # noqa: F401  (registers RunGraph)

SCHEMA = {"id": "TRIP", "lon": "LON", "lat": "LAT", "t": "TIMESTAMP",
          "sog": "SPEED", "cog": "COURSE", "trip": "TRIP"}
SCRATCH = os.environ.get("SCRATCH", os.path.join(HERE, "_scratch"))

# H3 average edge length (km) -- the quantisation floor, reported so the choice is auditable.
MAX_FAIL_PCT = 5.0   # HABIT must genuinely route on >=95% of gaps to be "working"

EDGE_KM = {4: 22.6, 5: 8.54, 6: 3.23, 7: 1.22, 8: 0.461, 9: 0.174, 10: 0.065}


def med(xs):
    return float(np.median(xs)) if len(xs) else float("nan")


def evaluate(imp, test):
    rows, nfb, ncells = [], 0, []
    for i, g in enumerate(test):
        plat, plon, fb, nv, _det = predict_graph(imp, g)
        nfb += int(fb)
        if not fb:
            ncells.append(nv)
        err = model_errors_latlon(g, plat, plon)
        if len(err):
            rows.append(gap_row(i, g, err))
    return rows, nfb, ncells


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="our_processed")
    ap.add_argument("--gap", type=int, default=6)
    ap.add_argument("--n-train", type=int, default=800, help="0 = full train split")
    ap.add_argument("--n-test", type=int, default=80, help="0 = full test split")
    ap.add_argument("--res", type=int, nargs="+", default=[5, 6, 7])
    ap.add_argument("--max-gap", type=int, nargs="+", default=[25, 50, 100, 200])
    a = ap.parse_args()
    track, gap = a.track, a.gap

    def _load(split):
        with open(f"task_alpha/data/{track}/gap_{gap}h/{split}.pkl", "rb") as f:
            return pickle.load(f)

    tr = _load("train"); va = _load("val"); te = _load("test")
    if a.n_train:
        tr = tr[:a.n_train]; va = va[:max(1, a.n_train // 8)]
    if a.n_test:
        te = te[:a.n_test]

    os.makedirs(SCRATCH, exist_ok=True)
    csv_path = os.path.join(SCRATCH, f"habit_corpus_{track}_{gap}h_{len(tr)}.csv")
    # Corpus = OBSERVED pings of train + val + the test graphs themselves.  Held pings are excluded
    # everywhere, so no label leaks; the test graphs' observed context is information every model
    # already receives at inference time, and HABIT is a lookup index, not a trained model.
    nrows = build_corpus_csv(list(tr) + list(va) + list(te), csv_path)
    print(f"corpus: {nrows} observed pings from {len(tr)} train + {len(va)} val + {len(te)} test graphs\n",
          flush=True)

    print(f"{'res':>3} {'cell_km':>7} {'max_gap':>7} | {'nodes':>7} {'edges':>7} | "
          f"{'A*fail':>6} {'route':>6} | {'GCD_NM':>7} {'HABIT_NM':>8} {'ratio':>6} {'win%':>5} | {'sec':>5}")
    print("-" * 106)

    best = None
    for res in a.res:
        for mg in a.max_gap:
            t0 = time.time()
            imp = HabitAdapter(resolution=res, max_gap=mg, schema=SCHEMA)
            try:
                nn, ne = imp.data_load(csv_path)
            except Exception as e:  # noqa: BLE001
                print(f"{res:>3} {EDGE_KM[res]:>7.2f} {mg:>7} | FAILED: {type(e).__name__}: {e}", flush=True)
                continue
            rows, nfb, ncells = evaluate(imp, te)
            dt = time.time() - t0
            gm, mm = med([r["_gm"] for r in rows]), med([r["_fm"] for r in rows])
            win = 100.0 * np.mean([r["win"] for r in rows])
            fb = 100.0 * nfb / max(len(te), 1)
            rt = med(ncells) if ncells else float("nan")   # 2 => degenerate anchor-to-anchor teleport
            print(f"{res:>3} {EDGE_KM[res]:>7.2f} {mg:>7} | {nn:>7} {ne:>7} | {fb:>5.1f}% {rt:>6.0f} | "
                  f"{gm:>7.2f} {mm:>8.2f} {mm/gm:>6.3f} {win:>5.1f} | {dt:>5.1f}", flush=True)
            if rt <= 3:
                print("      ^^ DEGENERATE: A* is jumping anchor->anchor; HABIT == great-circle chord", flush=True)
                continue
            # An A* failure falls back to the great-circle chord, which BEATS HABIT here -- so ranking by
            # error alone would reward a shattered graph.  Require HABIT to actually route (fail <= 5%),
            # then take the lowest error within that: its strongest honest configuration.
            if fb <= MAX_FAIL_PCT and (best is None or mm < best[0]):
                best = (mm, res, mg, fb)

    if best:
        print(f"\nbest-for-HABIT: resolution={best[1]}  max_gap={best[2]}  "
              f"(median {best[0]:.2f} NM at A*fail={best[3]:.1f}% <= {MAX_FAIL_PCT}%)")
    else:
        print(f"\nno configuration routed on >= {100-MAX_FAIL_PCT}% of gaps")


if __name__ == "__main__":
    main()

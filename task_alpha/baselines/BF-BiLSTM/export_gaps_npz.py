"""BF-BiLSTM on the FQGNN gaps -- STAGE A (gpubench env): export the 774 test gaps to a plain npz.

The bfbilstm conda env (Python 3.9, TF-CPU 2.11) has NO torch, so it cannot load RunGraphs; the
gpubench env cannot load the Keras 2.6 .h5.  Cross-env protocol:
  A (this, gpubench) : test RunGraphs -> observed pre-gap / post-gap series + held times -> ONE npz
  B (bfbilstm env)   : predict_gaps_bf.py -- load model_dblstm.h5, rigid 12/24/12 grid, predict
  C (gpubench)       : eval_gaps_bf.py -- interpolate the 24 preds onto held t_sec, FQGNN-format CSVs

Per gap we export ONLY leak-safe information: OBSERVED (t, lat, lon) before and after the gap
(truth at observed nodes is legitimate model input) and the HELD t_sec values (query times only --
no held lat/lon truth ever leaves this stage).

Run (any cwd -- chdirs to the repository root itself):
  python task_alpha/baselines/BF-BiLSTM/export_gaps_npz.py
  add --limit 25 for the smoke subset (FIRST 25 test graphs)
Output: task_alpha/baselines/BF-BiLSTM/runs_gaps/test_gaps.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))          # repository root
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from task_alpha.baselines.common_gaps import load_gaps, to_series    # noqa: E402

OUT_DIR = os.path.join(HERE, "runs_gaps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="export only the FIRST N test graphs (smoke)")
    cli = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    _, _, te = load_gaps()
    if cli.limit is not None:
        te = te[:cli.limit]
        print(f"[limit] exporting FIRST {len(te)} test graphs only", flush=True)

    pre_t, pre_lat, pre_lon, pre_off = [], [], [], [0]
    post_t, post_lat, post_lon, post_off = [], [], [], [0]
    held_t, held_off = [], [0]
    gap_ids = []
    n_short_pre = n_short_post = 0
    for i, g in enumerate(te):
        s = to_series(g)
        h = s["held"]
        hidx = np.where(h)[0]
        f, l = int(hidx[0]), int(hidx[-1])                            # held is contiguous
        pre, post = slice(0, f), slice(l + 1, len(h))                 # everything else is observed
        if f < 2:
            n_short_pre += 1
        if (len(h) - l - 1) < 2:
            n_short_post += 1
        pre_t.append(s["t_sec"][pre]); pre_lat.append(s["lat_true"][pre]); pre_lon.append(s["lon_true"][pre])
        post_t.append(s["t_sec"][post]); post_lat.append(s["lat_true"][post]); post_lon.append(s["lon_true"][post])
        held_t.append(s["t_sec"][h])
        pre_off.append(pre_off[-1] + f)
        post_off.append(post_off[-1] + (len(h) - l - 1))
        held_off.append(held_off[-1] + int(h.sum()))
        gap_ids.append(i)

    out = os.path.join(OUT_DIR, "test_gaps.npz")
    np.savez(out,
             pre_t=np.concatenate(pre_t), pre_lat=np.concatenate(pre_lat),
             pre_lon=np.concatenate(pre_lon), pre_off=np.array(pre_off, np.int64),
             post_t=np.concatenate(post_t), post_lat=np.concatenate(post_lat),
             post_lon=np.concatenate(post_lon), post_off=np.array(post_off, np.int64),
             held_t=np.concatenate(held_t), held_off=np.array(held_off, np.int64),
             gap_id=np.array(gap_ids, np.int64))
    npre = np.diff(pre_off); npost = np.diff(post_off); nheld = np.diff(held_off)
    print(f"[export] {len(gap_ids)} gaps -> {out}", flush=True)
    print(f"  pre  pts: min {npre.min()} / med {int(np.median(npre))} / max {npre.max()} "
          f"({n_short_pre} sides with <2 pts)", flush=True)
    print(f"  post pts: min {npost.min()} / med {int(np.median(npost))} / max {npost.max()} "
          f"({n_short_post} sides with <2 pts)", flush=True)
    print(f"  held pts: min {nheld.min()} / med {int(np.median(nheld))} / max {nheld.max()}", flush=True)


if __name__ == "__main__":
    main()

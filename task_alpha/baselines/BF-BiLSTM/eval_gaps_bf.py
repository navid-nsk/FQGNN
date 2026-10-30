"""BF-BiLSTM on the FQGNN gaps -- STAGE C (gpubench env): score Stage-B predictions, FQGNN CSVs.

Loads preds_bf.npz (24 regular-grid predictions per gap, Stage B) + test_gaps.npz (held query times,
Stage A), linearly interpolates each gap's predicted (lat, lon) onto the held t_sec values, and
scores with the SAME machinery as every other baseline (common_gaps: model_errors_latlon ->
gap_row -> bucket_rows).  Zero-shot frozen model, deterministic -> single pass, seed=0, no variance.

Outputs (task_alpha/baselines/BF-BiLSTM/runs_gaps/, FQGNN format):
  per_gap.csv, test_buckets.csv, all_seeds_test_buckets.csv, summary.csv, README.txt

Run (any cwd -- chdirs to the repository root itself):
  python task_alpha/baselines/BF-BiLSTM/eval_gaps_bf.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))          # repository root
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from task_alpha.baselines.common_gaps import (                        # noqa: E402
    bucket_rows, gap_row, load_gaps, model_errors_latlon, summary_row, write_aggregates,
    write_buckets, write_per_gap)

RUN_DIR = os.path.join(HERE, "runs_gaps")

README = """BF-BiLSTM zero-shot baseline on the FQGNN test gaps -- CAVEATS
================================================================
* ZERO-SHOT FROZEN model: the authors' pretrained model_dblstm.h5 (Mediterranean FISHING vessels,
  5-min AIS), loaded UNCHANGED and applied to our Arctic gaps. No training, no fine-tuning.
* RIGID 12/24/12 regular-grid protocol baked into the weights: observed pre/post tracks are
  linearly resampled to 12 anchor points each; the model predicts 24 gap points on the regular
  grid between the last pre point and the first post point; those 24 are linearly interpolated
  onto the actual held timestamps for scoring.
* DETERMINISTIC -> single pass, no seed variance (summary.csv has one row, seed=0).
* Fresh MinMaxScaler fit on OUR observed (pre/post) positions; uses lat/lon only (drops sog/cog).
"""


def main():
    gd = np.load(os.path.join(RUN_DIR, "test_gaps.npz"))
    pd_ = np.load(os.path.join(RUN_DIR, "preds_bf.npz"))
    gap_id = pd_["gap_id"]
    assert np.array_equal(gap_id, gd["gap_id"]), "test_gaps.npz / preds_bf.npz gap_id mismatch"
    held_t, held_off = gd["held_t"], gd["held_off"]
    grid_t, pred_lat, pred_lon = pd_["grid_t"], pd_["pred_lat"], pd_["pred_lon"]

    _, _, te = load_gaps()
    print(f"[stage C] scoring {len(gap_id)} gaps "
          f"(degenerate sides from stage B: pre={int(pd_['n_deg_pre'])} "
          f"post={int(pd_['n_deg_post'])} empty={int(pd_['n_empty'])})", flush=True)

    rows = []
    for k in range(len(gap_id)):
        gid = int(gap_id[k])
        g = te[gid]
        th = held_t[int(held_off[k]):int(held_off[k + 1])]            # held query times (stage A)
        la = np.interp(th, grid_t[k], pred_lat[k])                    # 24 grid preds -> held times
        lo = np.interp(th, grid_t[k], pred_lon[k])
        rows.append(gap_row(gid, g, model_errors_latlon(g, la, lo)))

    write_per_gap(os.path.join(RUN_DIR, "per_gap.csv"), rows)
    brows = bucket_rows(rows)
    write_buckets(os.path.join(RUN_DIR, "test_buckets.csv"), brows)
    sr = summary_row(0, brows, 0, "zero-shot")                        # single deterministic pass
    write_aggregates(RUN_DIR, [], [{"seed": 0, **r} for r in brows], [sr])
    with open(os.path.join(RUN_DIR, "README.txt"), "w") as f:
        f.write(README)

    ov = brows[0]
    print(f"[TEST] {ov['n']} gaps OVERALL ratio {ov['ratio']} "
          f"(GCD {ov['gcd_med']} model {ov['model_med']} win {ov['win_pct']}%)", flush=True)
    for r in brows[1:]:
        print(f"   {r['bucket']:<11} n={r['n']:4d}  GCD {r['gcd_med']}  model {r['model_med']}  "
              f"ratio {r['ratio']}  win {r['win_pct']}%", flush=True)
    print(f"saved per_gap.csv / test_buckets.csv / all_seeds_test_buckets.csv / summary.csv / "
          f"README.txt under {RUN_DIR}", flush=True)


if __name__ == "__main__":
    main()

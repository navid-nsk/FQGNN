"""Stage 3 (gpubench env): score MultitaskAIS predictions into the standard gap-sweep CSVs.

Reads predictions.npz (stage 2) + test_index.pkl/meta.json (stage 1) + the cell's test RunGraphs,
decodes argmax lat/lon bins to bin-CENTER coordinates, applies the a-priori aggregation rule
(MEAN over the 16 samples per held step -- fixed before any scoring, never best-of-sample), and
emits per_gap.csv / test_buckets.csv / epochs.csv via common_gaps -- identical formats to every
other gap-sweep model -- into
  task_alpha/runs/gap_sweep/<track>/gap_<G>h/multitaskais/seed<k>/
plus cell aggregates (all_seeds_epochs.csv, all_seeds_test_buckets.csv, summary.csv +MEAN/STD)
rebuilt from disk, matching the MHGIN schema.

Longitude averaging happens in the SHIFTED (contiguous) frame and is rotated back afterwards, so
date-line cells average correctly (antimeridian-safe, as in the HABIT adapter).

  python -m task_alpha.baselines.MTAIS.mtais_score --track standard_process --gap 24 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from task_alpha.baselines.common_gaps import (bucket_rows, gap_row, model_errors_latlon,  # noqa: E402
                                              summary_row, write_aggregates, write_buckets,
                                              write_epochs, write_per_gap)
from task_alpha.baselines.MTAIS.mtais_export import (EXPORT_ROOT, decode_lat_bins,  # noqa: E402
                                                     decode_lon_bins_shifted, wrap180)
from task_alpha.data import collate_graphs_interp  # noqa: F401,E402  (registers RunGraph)

DATA_ROOT = "task_alpha/data"
SWEEP_ROOT = "task_alpha/runs/gap_sweep"
MODEL_NAME = "multitaskais"


def decode_mean_latlon(bins, roi):
    """[n_held, 2, S] argmax (lat,lon) bins -> mean bin-center (lat, lon) degrees per held step.
    A-priori rule: MEAN over the S samples per step (lon averaged in the shifted frame)."""
    lat = decode_lat_bins(bins[:, 0, :], roi).mean(axis=1)
    lon_s = decode_lon_bins_shifted(bins[:, 1, :], roi).mean(axis=1)
    lon = wrap180(lon_s + roi["lon_cut"])
    return lat, lon


def epochs_rows_from_loss_curve(sdir):
    """MHGIN epochs.csv schema from the stage-2 loss curve (there is no validation: the vendor
    trains a fixed budget with no early stopping and no val-based selection)."""
    path = os.path.join(sdir, "loss_curve.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        out.append({"epoch": int(r["epoch"]),
                    "train_loss": round(-float(r["train_ll_per_t"]), 5),   # loss = -ELBO/t
                    "val_ratio": "", "val_model_med": "", "val_gcd_med": "",
                    "note": f"steps={r['steps']};elbo_per_t={r['train_ll_per_t']};"
                            f"no-val-selection(vendor has none)"})
    return out


def rebuild_aggregates(cdir):
    """all_seeds_*.csv + summary.csv from whatever seed dirs exist on disk (resume-friendly)."""
    all_ep, all_bk, summ = [], [], []
    for sdir in sorted(glob.glob(os.path.join(cdir, "seed*"))):
        seed = int(os.path.basename(sdir).replace("seed", ""))
        tb = os.path.join(sdir, "test_buckets.csv")
        if not os.path.exists(tb):
            continue
        with open(tb, newline="") as f:
            bk = list(csv.DictReader(f))
        epp = os.path.join(sdir, "epochs.csv")
        ep = []
        if os.path.exists(epp):
            with open(epp, newline="") as f:
                ep = list(csv.DictReader(f))
        all_ep += [{"seed": seed, **r} for r in ep]
        all_bk += [{"seed": seed, **r} for r in bk]
        ov = bk[0]
        summ.append({"seed": seed, "n_gaps": int(ov["n"]), "gcd_med": float(ov["gcd_med"]),
                     "model_med": float(ov["model_med"]), "ratio": float(ov["ratio"]),
                     "win_pct": float(ov["win_pct"]), "epochs_trained": len(ep),
                     "best_val": ""})
    if summ:
        write_aggregates(cdir, all_ep, all_bk, summ)
    return summ


def score_seed(track, gap, seed, suffix="", model_name=MODEL_NAME):
    t0 = time.time()
    export_dir = f"{EXPORT_ROOT}/{track}/gap_{gap}h{suffix}"
    cdir = f"{SWEEP_ROOT}/{track}/gap_{gap}h/{model_name}"
    sdir = f"{cdir}/seed{seed}"
    with open(f"{export_dir}/meta.json") as f:
        meta = json.load(f)
    roi = meta["roi"]
    with open(f"{export_dir}/test_index.pkl", "rb") as f:
        entries = pickle.load(f)
    with open(f"{DATA_ROOT}/{track}/gap_{gap}h/test.pkl", "rb") as f:
        te = pickle.load(f)
    preds = np.load(f"{sdir}/predictions.npz")

    rows, n_skip = [], 0
    for e in entries:
        key = f"g{e['gap_id']}"
        if key not in preds.files:
            n_skip += 1
            continue
        bins = preds[key]                       # [n_held, 2, S] int16
        g = te[e["gap_id"]]
        assert int(g.mmsi) == e["mmsi"], f"test_index/test.pkl misalignment at gap {e['gap_id']}"
        assert bins.shape[0] == int(g.mask.sum()) == e["n_held"]
        plat, plon = decode_mean_latlon(bins, roi)
        err = model_errors_latlon(g, plat, plon)
        if len(err):
            rows.append(gap_row(e["gap_id"], g, err))
    if not rows:
        raise RuntimeError("no scored gaps -- empty predictions.npz?")

    brows = bucket_rows(rows)
    os.makedirs(sdir, exist_ok=True)
    write_per_gap(f"{sdir}/per_gap.csv", rows)
    write_buckets(f"{sdir}/test_buckets.csv", brows)
    write_epochs(f"{sdir}/epochs.csv", epochs_rows_from_loss_curve(sdir))
    rebuild_aggregates(cdir)

    ov = brows[0]
    print(f"[{model_name}][TEST] {track}/gap_{gap}h s{seed}  n={ov['n']}  GCD {ov['gcd_med']}  "
          f"model {ov['model_med']}  ratio {ov['ratio']}  win {ov['win_pct']}%  "
          f"(quant floor {meta['quant_floor_nm']:.2f} NM; ONE-SIDED reference; {n_skip} unscored; "
          f"{time.time()-t0:.0f}s)", flush=True)
    for r in brows[1:]:
        print(f"   {r['bucket']:<11} n={r['n']:4d}  GCD {r['gcd_med']}  model {r['model_med']}  "
              f"ratio {r['ratio']}", flush=True)
    return ov


def main():
    ap = argparse.ArgumentParser(description="MultitaskAIS stage-3 scorer (gpubench env)")
    ap.add_argument("--track", required=True)
    ap.add_argument("--gap", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--suffix", default="", help="export suffix, e.g. _smoke")
    ap.add_argument("--model-name", default=MODEL_NAME, help="output folder name")
    a = ap.parse_args()
    score_seed(a.track, a.gap, a.seed, a.suffix, a.model_name)


if __name__ == "__main__":
    main()

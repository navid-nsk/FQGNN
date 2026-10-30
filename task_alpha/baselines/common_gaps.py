"""Shared evaluation base for ALL Task-Alpha baselines (CSDI / SAITS / Bi-GRU / BF-BiLSTM).

Purpose: every baseline is trained/evaluated on the EXACT SAME gaps as the FQGNN 5-seed run
(run_full_faithful_5seed.py) and emits the EXACT SAME CSV formats, so per-gap rows and curvature
buckets are directly comparable (row-aligned with the FQGNN's per_gap.csv; identical GCD column).

Dataset  = build_dataset_raw(years=(2016,2017,2018), months=(7,8), gap_minutes=360, max_gap_len=500,
           min_span_factor=1.5, n_anchor=8, clean_only=True, n_mmsi=None, max_files_per_part=None,
           seed=0)  +  vessel-disjoint _split  ->  5350 train / 635 val / 774 test RunGraphs.
Metrics  = task_alpha.metrics.gcd_errors (GCD reference), haversine NM vs true_lonlat at mask.
Sinuosity= task_alpha.experiment_compare._sinuosity.  Buckets = (0, .02, .05, .10, .25, inf).

A baseline runner only needs to produce, per test graph, predicted (lat, lon) at the HELD nodes
(in original node order).  Everything else (per-gap rows, buckets, aggregates) is here.

Run any baseline from the repository root:  python -m task_alpha.baselines.<runner>
"""
from __future__ import annotations

import csv
import os

import numpy as np

from task_alpha.data import POS_OUT_SCALE, enu_to_lonlat, haversine_nm
from task_alpha.data_raw import build_dataset_raw
from task_alpha.experiment_compare import _sinuosity
from task_alpha.metrics import gcd_errors
from task_alpha.train_stream import _split

KNOTS_TO_MPS = 0.5144444444444445

BUCKET_EDGES = (0.0, 0.02, 0.05, 0.10, 0.25, 1e9)
DATASET_KW = dict(years=(2016, 2017, 2018), months=(7, 8), gap_minutes=360, max_gap_len=500,
                  min_span_factor=1.5, n_anchor=8, clean_only=True, n_mmsi=None,
                  max_files_per_part=None, seed=0)


# ------------------------------------------------------------------------------------------------ #
#  DATA: the FQGNN gaps, identical split                                                             #
# ------------------------------------------------------------------------------------------------ #
def load_gaps():
    """Same call + filter + split as run_full_faithful_5seed.main().  Returns (train, val, test)."""
    allg = build_dataset_raw(**DATASET_KW)
    allg = [g for g in allg if int(g.mask.sum()) > 0 and _split(g.mmsi) is not None]
    tr = [g for g in allg if _split(g.mmsi) == "train"]
    va = [g for g in allg if _split(g.mmsi) == "val"]
    te = [g for g in allg if _split(g.mmsi) == "test"]
    print(f"[common_gaps] {len(tr)} train / {len(va)} val / {len(te)} test", flush=True)
    return tr, va, te


def to_series(g, crop_ctx: int = 64):
    """RunGraph -> plain sequence view for time-series baselines (same information for every model):
      t_sec   [n]  absolute-scale seconds (reconstructed: coord t_norm * dur; Δt exact)
      lat,lon [n]  degrees   -- TRUTH on observed nodes; on HELD nodes the model must not read them
      sog     [n]  knots     (0 on held -- zeroed at build, no leakage)
      cog_sin/cog_cos [n]    (0 on held)
      held    [n]  bool mask (the contiguous gap)
      node_idx[n]  index into the ORIGINAL graph nodes (for writing predictions back)
    Cropped to `crop_ctx` observed pings on EACH side of the gap (generous two-sided context,
    comparable to the FQGNN's receptive field; metrics are unaffected -- they use the full graph)."""
    x = g.x.numpy(); coord = g.coord.numpy(); held = g.mask.numpy().astype(bool)
    tr_ll = g.true_lonlat.numpy()
    hidx = np.where(held)[0]
    si, last = int(hidx[0]), int(hidx[-1])
    span_norm = max(coord[last, 0] - coord[si, 0], 1e-9)
    dur_sec = (g.gap_minutes * 60.0) / span_norm          # recover absolute time scale
    t_sec = coord[:, 0] * dur_sec

    lo = max(0, si - crop_ctx); hi = min(len(held), last + 1 + crop_ctx)
    sl = slice(lo, hi)
    h = held[sl]
    lat_t = tr_ll[sl, 0].astype(np.float64); lon_t = tr_ll[sl, 1].astype(np.float64)
    return dict(
        t_sec=t_sec[sl].astype(np.float64),
        # TRUTH (training targets / metrics only -- NEVER model input at held nodes):
        lat_true=lat_t, lon_true=lon_t,
        # LEAK-SAFE MODEL INPUT (held nodes zeroed -- use ONLY these as network input):
        lat_in=np.where(h, 0.0, lat_t), lon_in=np.where(h, 0.0, lon_t),
        sog_in=(x[sl, 2] * 20.0).astype(np.float64),       # already zeroed at held in RunGraph.x
        cog_sin=x[sl, 3].astype(np.float64), cog_cos=x[sl, 4].astype(np.float64),
        held=h, node_idx=np.arange(lo, hi),
    )


def fit_latlon_stats(graphs, crop_ctx: int = 64):
    """z-score stats for (lat, lon, sog) over OBSERVED nodes of the given graphs (fit on TRAIN only)."""
    la, lo, sg = [], [], []
    for g in graphs:
        s = to_series(g, crop_ctx)
        obs = ~s["held"]
        la.append(s["lat_true"][obs]); lo.append(s["lon_true"][obs]); sg.append(s["sog_in"][obs])
    la, lo, sg = np.concatenate(la), np.concatenate(lo), np.concatenate(sg)
    mean = np.array([la.mean(), lo.mean(), sg.mean()]); std = np.array([la.std(), lo.std(), sg.std()])
    std[std < 1e-6] = 1.0
    return mean, std


def anchor_states_enu(g):
    """Two-sided anchor kinematics in real ENU METRES, reconstructed leak-safely from RunGraph fields
    (only non-held/observed node indices are read -- g.x is all-zeroed at held nodes by construction).
    Position/speed/course are read from g.x (the model's own leak-safe input), so every classical
    baseline sees EXACTLY what the neural models see, no more no less.

    Returns (before, after, held_t, hidx, ref):
      before, after  dicts of aligned arrays {t (sec), x, y (ENU metres), vx, vy (m/s)} in the
                     run's original chronological (ascending-time) node order.
      held_t         [n_held] absolute-scale seconds of every held (query) node.
      hidx           [n_held] the held node indices (original order) -- for writing predictions back.
      ref            (lat0, lon0) ENU reference, for the final ENU -> lat/lon conversion.
    """
    x = g.x.numpy(); coord = g.coord.numpy(); held = g.mask.numpy().astype(bool)
    hidx = np.where(held)[0]
    si, last = int(hidx[0]), int(hidx[-1])
    span_norm = max(coord[last, 0] - coord[si, 0], 1e-9)
    dur_sec = (g.gap_minutes * 60.0) / span_norm           # recover absolute time scale (to_series pattern)
    t_sec = coord[:, 0] * dur_sec

    ex_m = x[:, 0] * POS_OUT_SCALE; ey_m = x[:, 1] * POS_OUT_SCALE
    sog_mps = x[:, 2] * 20.0 * KNOTS_TO_MPS                 # x[:,2] = sog/20 (knots), zeroed at held
    vx = sog_mps * x[:, 3]; vy = sog_mps * x[:, 4]          # x[:,3]=sin(cog), x[:,4]=cos(cog)

    def _pack(idx):
        return dict(t=t_sec[idx].astype(np.float64), x=ex_m[idx].astype(np.float64),
                    y=ey_m[idx].astype(np.float64), vx=vx[idx].astype(np.float64),
                    vy=vy[idx].astype(np.float64))

    n = len(held)
    before = _pack(np.arange(0, si))
    after = _pack(np.arange(last + 1, n))
    return before, after, t_sec[hidx].astype(np.float64), hidx, g.ref


def enu_pred_to_latlon(pred_x_m, pred_y_m, ref):
    """Real ENU metres -> (lat, lon) via the graph's own reference, ready for model_errors_latlon."""
    lat0, lon0 = ref
    return enu_to_lonlat(np.asarray(pred_x_m, float), np.asarray(pred_y_m, float), lat0, lon0)


# ------------------------------------------------------------------------------------------------ #
#  METRICS: identical to the FQGNN run                                                               #
# ------------------------------------------------------------------------------------------------ #
def model_errors_latlon(g, pred_lat: np.ndarray, pred_lon: np.ndarray) -> np.ndarray:
    """Per-held-node great-circle NM error of predicted (lat, lon) [n_held] in HELD-node order."""
    m = g.mask.numpy().astype(bool)
    tr = g.true_lonlat.numpy()
    return haversine_nm(tr[m, 0], tr[m, 1], np.asarray(pred_lat, float), np.asarray(pred_lon, float))


def gap_row(gap_id: int, g, model_err_nm: np.ndarray) -> dict:
    """One per_gap.csv row -- identical semantics to run_full_faithful_5seed (median over held points)."""
    fm = float(np.median(model_err_nm)); gm = float(np.median(gcd_errors(g))); si = _sinuosity(g)
    return {"gap_id": gap_id, "mmsi": int(g.mmsi), "gap_min": round(float(g.gap_minutes), 1),
            "sinuosity": round(float(si), 4) if np.isfinite(si) else "",
            "gcd_med": round(gm, 3), "model_med": round(fm, 3), "win": int(fm < gm),
            "_sin_raw": si, "_fm": fm, "_gm": gm}          # raw fields for bucketing (stripped on write)


def bucket_rows(rows: list[dict]) -> list[dict]:
    """test_buckets.csv rows from per-gap rows -- EXACT bucket logic of run_full_faithful_5seed
    (median-of-per-gap-medians, ratio of bucket medians, win%; finite-sinuosity gaps only)."""
    fq = np.array([r["_fm"] for r in rows]); gc = np.array([r["_gm"] for r in rows])
    sn = np.array([r["_sin_raw"] for r in rows], dtype=float)
    keep = np.isfinite(sn); fqk, gck, snk = fq[keep], gc[keep], sn[keep]
    out = [{"bucket": "OVERALL", "n": int(len(fqk)), "gcd_med": round(float(np.median(gck)), 3),
            "model_med": round(float(np.median(fqk)), 3),
            "ratio": round(float(np.median(fqk) / np.median(gck)), 4),
            "win_pct": round(100 * float(np.mean(fqk < gck)), 1)}]
    for lo, hi in zip(BUCKET_EDGES[:-1], BUCKET_EDGES[1:]):
        sel = (snk >= lo) & (snk < hi)
        if sel.sum() < 3:
            continue
        lab = f"{lo:.2f}-{hi:.2f}" if hi < 1e9 else f">{lo:.2f}"
        out.append({"bucket": lab, "n": int(sel.sum()),
                    "gcd_med": round(float(np.median(gck[sel])), 3),
                    "model_med": round(float(np.median(fqk[sel])), 3),
                    "ratio": round(float(np.median(fqk[sel]) / np.median(gck[sel])), 4),
                    "win_pct": round(100 * float(np.mean(fqk[sel] < gck[sel])), 1)})
    return out


# ------------------------------------------------------------------------------------------------ #
#  WRITERS: identical file formats to runs/full_faithful_5seed                                       #
# ------------------------------------------------------------------------------------------------ #
PER_GAP_FIELDS = ["gap_id", "mmsi", "gap_min", "sinuosity", "gcd_med", "model_med", "win"]
BUCKET_FIELDS = ["bucket", "n", "gcd_med", "model_med", "ratio", "win_pct"]


def write_per_gap(path: str, rows: list[dict]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PER_GAP_FIELDS, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def write_buckets(path: str, brows: list[dict]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BUCKET_FIELDS); w.writeheader(); w.writerows(brows)


def write_epochs(path: str, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def write_aggregates(out_dir: str, all_ep: list[dict], all_bk: list[dict], summary: list[dict]):
    """all_seeds_epochs.csv / all_seeds_test_buckets.csv / summary.csv (+MEAN/STD) -- FQGNN format."""
    if all_ep:
        with open(f"{out_dir}/all_seeds_epochs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_ep[0].keys())); w.writeheader(); w.writerows(all_ep)
    if all_bk:
        with open(f"{out_dir}/all_seeds_test_buckets.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["seed"] + BUCKET_FIELDS); w.writeheader(); w.writerows(all_bk)
    if summary:
        with open(f"{out_dir}/summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
        rat = np.array([s["ratio"] for s in summary], float)
        mm = np.array([s["model_med"] for s in summary], float)
        wp = np.array([s["win_pct"] for s in summary], float)
        with open(f"{out_dir}/summary.csv", "a", newline="") as f:
            w = csv.writer(f)
            w.writerow(["MEAN", "", summary[0]["gcd_med"], round(float(mm.mean()), 3),
                        round(float(rat.mean()), 4), round(float(wp.mean()), 1), "", ""])
            w.writerow(["STD", "", "", round(float(mm.std(ddof=0)), 3),
                        round(float(rat.std(ddof=0)), 4), round(float(wp.std(ddof=0)), 1), "", ""])


def summary_row(seed, brows: list[dict], n_epochs: int, best_val) -> dict:
    ov = brows[0]
    return {"seed": seed, "n_gaps": ov["n"], "gcd_med": ov["gcd_med"], "model_med": ov["model_med"],
            "ratio": ov["ratio"], "win_pct": ov["win_pct"], "epochs_trained": n_epochs,
            "best_val": best_val}


# ------------------------------------------------------------------------------------------------ #
#  SELF-TEST: bucket GCD column must EXACTLY reproduce the FQGNN run's table                         #
# ------------------------------------------------------------------------------------------------ #
if __name__ == "__main__":
    tr, va, te = load_gaps()
    rows = []
    for i, g in enumerate(te):
        ge = gcd_errors(g)
        rows.append(gap_row(i, g, ge))                    # model == GCD -> model_med must equal gcd_med
    br = bucket_rows(rows)
    print("\nself-test (model=GCD; gcd_med column must match runs/full_faithful/test_buckets.csv):")
    for r in br:
        print(f"  {r['bucket']:<11} n={r['n']:4d}  gcd_med={r['gcd_med']}  ratio={r['ratio']} "
              f"(must be 1.0)  win={r['win_pct']} (must be 0.0)")
    exp = {"OVERALL": (770, 2.857), "0.00-0.02": (225, 0.893), "0.02-0.05": (74, 3.542),
           "0.05-0.10": (75, 5.103), "0.10-0.25": (105, 6.385), ">0.25": (291, 3.376)}
    ok = all(r["n"] == exp[r["bucket"]][0] and abs(r["gcd_med"] - exp[r["bucket"]][1]) < 1e-6
             and r["ratio"] == 1.0 for r in br if r["bucket"] in exp)
    print("MATCH vs FQGNN table:", "PASS" if ok else "FAIL")
    # sequence view sanity
    s = to_series(te[0])
    leak = float(np.abs(s["lat_in"][s["held"]]).max() + np.abs(s["lon_in"][s["held"]]).max())
    print(f"series sanity: n={len(s['t_sec'])} held={int(s['held'].sum())} "
          f"dt_med={np.median(np.diff(s['t_sec'])):.0f}s "
          f"lat_rng=({s['lat_true'].min():.2f},{s['lat_true'].max():.2f}) "
          f"held-input-leak={leak} (must be 0.0)")
    ns = np.array([len(to_series(g)["t_sec"]) for g in te])
    print(f"series lengths: min {ns.min()} / med {int(np.median(ns))} / max {ns.max()}")

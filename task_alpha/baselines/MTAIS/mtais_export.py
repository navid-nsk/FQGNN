"""Stage 1 (gpubench env): export Task-Alpha gap datasets to MultitaskAIS vendor format.

Per cell (track, gap) this writes, under task_alpha/baselines/MTAIS/_export/<track>/gap_<G>h[suffix]/:
  train.pkl      {voyage_id: np.ndarray [T,9]}  vendor columns LAT,LON,SOG,COG,HEADING,ROT,NAV_STT,
                 TIMESTAMP,MMSI with LAT/LON/SOG/COG normalized to [0,1) exactly as the vendor's
                 dataset_preprocessing.py does ((v-min)/range, sog/30 clipped, cog/360).
  valid.pkl      same format (exported for completeness; vendor training uses no validation).
  mean.pkl       mean four-hot vector of the TRAIN voyages (vendor calculate_AIS_mean.py semantics:
                 all pings, no subsampling) -- feeds generative_bias_init.
  test_index.pkl per test gap: the PRE-GAP observed pings (vendor format), held count/timestamps,
                 held node ids, mmsi, gap_minutes.  NO post-gap pings are exported for the test
                 sequences: the model is structurally one-sided (forward LSTM), post-gap data cannot
                 influence gap samples, and dropping them removes the vendor's own one-step
                 boundary-input leak at the end of the replacement window.
  meta.json      ROI, antimeridian cut, bin counts/resolutions, quantization floor, caps, counts.

LEAK SAFETY: only OBSERVED pings are ever written (held nodes excluded everywhere), using
common_gaps.to_series truth fields on observed indices and the HABIT observed-segments approach
(each gapped graph contributes its pre-gap and post-gap pings as SEPARATE voyages, so no exported
sequence steps across a gap).  Asserted in test_mtais_adapter.py.

CAPS: identical to run_gap_sweep.load_dataset -- TRAIN capped at 8000 graphs with fixed rng
20260702, VAL at 1000 with rng 20260703 (verbatim reimplementation; importing run_gap_sweep would
drag in torch/CUDA model modules).  TEST is never capped.

ANTIMERIDIAN: Arctic AIS spans +/-180.  Longitudes are rotated by a per-cell cut chosen in the
middle of the largest EMPTY circular arc of observed longitudes, so the occupied arc is contiguous
in the shifted frame; the four-hot lon bins live in that frame and predictions are rotated back
(wrap180(shifted + cut)) at decode time.  Round-trip asserted in test_mtais_adapter.py.

BIN RESOLUTION: lat_bins + lon_bins is fixed at 600 per cell (vendor Brittany: 500 -> data_dim
602), split between lat and lon proportionally to the ROI's METRIC extent (lon scaled by
cos(mid_lat)) so bins are roughly square on the ground.  SOG 1 kn / 30 bins and COG 5 deg / 72
bins are the vendor defaults.  The bin-center quantization floor (half bin diagonal, NM) is
recorded in meta.json -- MultitaskAIS cannot beat that floor by construction.

Pickles use protocol 4 (py3.7-loadable).  numpy 2.x pickles reference numpy._core; the TF1-side
loader (mtais_tf1_run._np_compat_load) maps that to numpy.core for numpy 1.16.

  python -m task_alpha.baselines.MTAIS.mtais_export --track standard_process --gap 24
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from task_alpha.baselines.common_gaps import to_series  # noqa: E402
from task_alpha.data import collate_graphs_interp  # noqa: F401,E402  (registers RunGraph)

DATA_ROOT = "task_alpha/data"
EXPORT_ROOT = "task_alpha/baselines/MTAIS/_export"

LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI = range(9)  # vendor column order
SPEED_MAX = 30.0                       # vendor knots ceiling (flags_config SPEED_MAX)
SOG_BINS, COG_BINS = 30, 72            # vendor: 1 kn and 5 deg resolutions
TARGET_LATLON_BINS = 600               # lat_bins + lon_bins per cell (vendor Brittany: 500)
MIN_LATLON_BINS = 40                   # never squeeze an axis below this
ROI_PAD_DEG = 0.02                     # keeps min/max strictly interior
MIN_SEG = 4                            # min pings per exported train/val voyage
CLIP_HI = 0.999999                     # normalized values strictly < 1 (vendor int(v*bins) indexing)
TRAIN_CAP, VAL_CAP = 8000, 1000        # run_gap_sweep.load_dataset matched budget (verbatim)
PKL_PROTOCOL = 4                       # py3.7-compatible


# ------------------------------------------------------------------------------------------------ #
#  dataset IO + caps (verbatim semantics of run_gap_sweep.load_dataset)                              #
# ------------------------------------------------------------------------------------------------ #
def _load(track, gap, split):
    with open(f"{DATA_ROOT}/{track}/gap_{gap}h/{split}.pkl", "rb") as f:
        return pickle.load(f)


def apply_caps(tr, va):
    """EXACT cap logic of run_gap_sweep.load_dataset (fixed rngs 20260702 / 20260703)."""
    if len(tr) > TRAIN_CAP:
        idx = np.random.default_rng(20260702).choice(len(tr), size=TRAIN_CAP, replace=False)
        tr = [tr[i] for i in sorted(idx)]
        print(f"[mtais-export] TRAIN capped to {TRAIN_CAP} (matched budget; fixed rng 20260702)", flush=True)
    if len(va) > VAL_CAP:
        idx = np.random.default_rng(20260703).choice(len(va), size=VAL_CAP, replace=False)
        va = [va[i] for i in sorted(idx)]
        print(f"[mtais-export] VAL capped to {VAL_CAP} (fixed rng 20260703)", flush=True)
    return tr, va


def observed_segments(held):
    """Index runs of OBSERVED pings, split at every held block (HABIT habit_model._observed_segments
    logic, reimplemented here to avoid importing the h3/duckdb stack)."""
    segs, start = [], None
    for k, h in enumerate(held):
        if not h and start is None:
            start = k
        elif h and start is not None:
            segs.append((start, k))
            start = None
    if start is not None:
        segs.append((start, len(held)))
    return segs


# ------------------------------------------------------------------------------------------------ #
#  ROI / antimeridian / binning                                                                      #
# ------------------------------------------------------------------------------------------------ #
def wrap180(x):
    return ((np.asarray(x, dtype=np.float64) + 180.0) % 360.0) - 180.0


def circular_cut(lons_deg, nbins=3600):
    """Rotation offset (deg in [0,360)) placed mid-way through the largest EMPTY circular arc of the
    observed longitudes, so the occupied arc is contiguous in the shifted frame."""
    h = np.zeros(nbins, dtype=np.int64)
    idx = (np.asarray(lons_deg, np.float64) % 360.0 / 360.0 * nbins).astype(np.int64) % nbins
    np.add.at(h, idx, 1)
    empty = (h == 0)
    if not empty.any():
        return 0.0
    # longest circular run of empty bins: double the array
    e2 = np.concatenate([empty, empty])
    best_len, best_start, run, start = 0, 0, 0, 0
    for i, v in enumerate(e2):
        if v:
            if run == 0:
                start = i
            run += 1
            if run > best_len and start < nbins:
                best_len, best_start = run, start
        else:
            run = 0
    best_len = min(best_len, nbins)
    center = (best_start + best_len / 2.0) % nbins
    return float(center * 360.0 / nbins)


def build_roi(lat_obs, lon_obs):
    """ROI + bin layout from OBSERVED lat/lon (degrees) of capped train + capped val + test."""
    cut = circular_cut(lon_obs)
    s = (np.asarray(lon_obs, np.float64) - cut) % 360.0
    lat_lo = float(np.min(lat_obs) - ROI_PAD_DEG)
    lat_hi = float(np.max(lat_obs) + ROI_PAD_DEG)
    lon_lo = float(np.min(s) - ROI_PAD_DEG)
    lon_hi = float(np.max(s) + ROI_PAD_DEG)
    mid_lat = 0.5 * (lat_lo + lat_hi)
    lat_nm = (lat_hi - lat_lo) * 60.0
    lon_nm = (lon_hi - lon_lo) * 60.0 * math.cos(math.radians(mid_lat))
    lat_bins = int(round(TARGET_LATLON_BINS * lat_nm / max(lat_nm + lon_nm, 1e-9)))
    lat_bins = min(max(lat_bins, MIN_LATLON_BINS), TARGET_LATLON_BINS - MIN_LATLON_BINS)
    lon_bins = TARGET_LATLON_BINS - lat_bins
    lat_reso = (lat_hi - lat_lo) / lat_bins
    lon_reso = (lon_hi - lon_lo) / lon_bins
    qfloor = 0.5 * math.hypot(lat_reso * 60.0, lon_reso * 60.0 * math.cos(math.radians(mid_lat)))
    return dict(lon_cut=cut, lat_lo=lat_lo, lat_hi=lat_hi, lon_lo=lon_lo, lon_hi=lon_hi,
                lat_bins=lat_bins, lon_bins=lon_bins, sog_bins=SOG_BINS, cog_bins=COG_BINS,
                data_dim=lat_bins + lon_bins + SOG_BINS + COG_BINS,
                lat_reso_deg=lat_reso, lon_reso_deg=lon_reso,
                sog_reso_kn=SPEED_MAX / SOG_BINS, cog_reso_deg=360.0 / COG_BINS,
                mid_lat_deg=mid_lat, quant_floor_nm=qfloor)


def norm_lat(lat_deg, roi):
    v = (np.asarray(lat_deg, np.float64) - roi["lat_lo"]) / (roi["lat_hi"] - roi["lat_lo"])
    return np.clip(v, 0.0, CLIP_HI)


def norm_lon(lon_deg, roi):
    s = (np.asarray(lon_deg, np.float64) - roi["lon_cut"]) % 360.0
    v = (s - roi["lon_lo"]) / (roi["lon_hi"] - roi["lon_lo"])
    return np.clip(v, 0.0, CLIP_HI)


def decode_lat_bins(bins, roi):
    """Bin index -> bin-CENTER latitude (deg)."""
    return roi["lat_lo"] + (np.asarray(bins, np.float64) + 0.5) * roi["lat_reso_deg"]


def decode_lon_bins_shifted(bins, roi):
    """Bin index -> bin-CENTER longitude in the SHIFTED (contiguous) frame (deg).  Callers average
    in this frame, then rotate back with wrap180(x + roi['lon_cut'])."""
    return roi["lon_lo"] + (np.asarray(bins, np.float64) + 0.5) * roi["lon_reso_deg"]


# ------------------------------------------------------------------------------------------------ #
#  voyage building                                                                                   #
# ------------------------------------------------------------------------------------------------ #
def _cog_deg(sin_c, cos_c):
    return np.degrees(np.arctan2(sin_c, cos_c)) % 360.0


def rows_from_series(s, sl, mmsi, roi):
    """Vendor-format [m,9] rows from a slice of a to_series view -- OBSERVED indices only."""
    lat = s["lat_true"][sl]
    m = len(lat)
    arr = np.zeros((m, 9), dtype=np.float64)
    arr[:, LAT] = norm_lat(lat, roi)
    arr[:, LON] = norm_lon(s["lon_true"][sl], roi)
    arr[:, SOG] = np.clip(np.clip(s["sog_in"][sl], 0.0, SPEED_MAX) / SPEED_MAX, 0.0, CLIP_HI)
    arr[:, COG] = np.clip(_cog_deg(s["cog_sin"][sl], s["cog_cos"][sl]) / 360.0, 0.0, CLIP_HI)
    arr[:, TIMESTAMP] = s["t_sec"][sl]
    arr[:, MMSI] = float(mmsi)
    return arr


def build_voyages(graphs, roi, min_seg=MIN_SEG):
    """{voyage_id: [T,9]} from the OBSERVED segments of the given graphs (held pings excluded;
    pre-gap and post-gap pings of a gapped graph are separate voyages -- no voyage spans a gap)."""
    out, vid, n_pings = {}, 0, 0
    for g in graphs:
        s = to_series(g, crop_ctx=10 ** 9)
        for lo, hi in observed_segments(s["held"]):
            if hi - lo < min_seg:
                continue
            out[vid] = rows_from_series(s, slice(lo, hi), int(g.mmsi), roi)
            n_pings += hi - lo
            vid += 1
    return out, n_pings


def mean_four_hot(voyages, roi):
    """Vendor calculate_AIS_mean.py semantics: mean four-hot over ALL pings of the train voyages."""
    D = roi["data_dim"]
    lb, ob, sb, cb = roi["lat_bins"], roi["lon_bins"], roi["sog_bins"], roi["cog_bins"]
    counts = np.zeros(D, dtype=np.int64)
    total = 0
    for arr in voyages.values():
        il = (arr[:, LAT] * lb).astype(np.int64)
        io = (arr[:, LON] * ob).astype(np.int64) + lb
        isg = (arr[:, SOG] * sb).astype(np.int64) + lb + ob
        ic = (arr[:, COG] * cb).astype(np.int64) + lb + ob + sb
        for ix in (il, io, isg, ic):
            np.add.at(counts, ix, 1)
        total += len(arr)
    return counts.astype(np.float64) / max(total, 1), total


def build_test_index(te, roi):
    """Per test gap: pre-gap observed pings + held bookkeeping.  gap_id = index in the test list
    (the common_gaps convention, so per_gap.csv rows are row-aligned with every other model)."""
    entries, n_skip = [], 0
    for i, g in enumerate(te):
        s = to_series(g, crop_ctx=10 ** 9)
        held = s["held"]
        hidx = np.where(held)[0]
        skip = (len(hidx) == 0) or (int(hidx[0]) < 1)
        si = int(hidx[0]) if len(hidx) else 0
        pre = rows_from_series(s, slice(0, si), int(g.mmsi), roi) if si else np.zeros((0, 9))
        entries.append({
            "gap_id": i, "mmsi": int(g.mmsi), "gap_minutes": float(g.gap_minutes),
            "n_pre": si, "n_held": int(len(hidx)),
            "node_idx": hidx.astype(np.int64),
            "t_held": s["t_sec"][held].astype(np.float64),   # for the record; the four-hot has no
            "pre": pre,                                       # time channel (count-based placement)
            "skip": bool(skip),
        })
        n_skip += int(skip)
    return entries, n_skip


# ------------------------------------------------------------------------------------------------ #
#  main                                                                                              #
# ------------------------------------------------------------------------------------------------ #
def export_cell(track, gap, suffix="", train_segments_cap=0):
    t0 = time.time()
    out_dir = f"{EXPORT_ROOT}/{track}/gap_{gap}h{suffix}"
    os.makedirs(out_dir, exist_ok=True)

    tr, va, te = _load(track, gap, "train"), _load(track, gap, "val"), _load(track, gap, "test")
    print(f"[mtais-export] {track}/gap_{gap}h: {len(tr)} train / {len(va)} val / {len(te)} test "
          f"RunGraphs", flush=True)
    tr, va = apply_caps(tr, va)

    # ROI from observed pings of everything the model will touch (observed data only -- leak-safe)
    lat_all, lon_all = [], []
    for g in list(tr) + list(va) + list(te):
        s = to_series(g, crop_ctx=10 ** 9)
        obs = ~s["held"]
        lat_all.append(s["lat_true"][obs])
        lon_all.append(s["lon_true"][obs])
    lat_all = np.concatenate(lat_all)
    lon_all = np.concatenate(lon_all)
    roi = build_roi(lat_all, lon_all)
    del lat_all, lon_all
    print(f"[mtais-export] ROI lat [{roi['lat_lo']:.3f},{roi['lat_hi']:.3f}] "
          f"lon(shifted) [{roi['lon_lo']:.3f},{roi['lon_hi']:.3f}] cut {roi['lon_cut']:.2f}deg | "
          f"bins lat {roi['lat_bins']} lon {roi['lon_bins']} data_dim {roi['data_dim']} | "
          f"quant floor {roi['quant_floor_nm']:.3f} NM", flush=True)

    tr_voy, tr_pings = build_voyages(tr, roi)
    if train_segments_cap and len(tr_voy) > train_segments_cap:
        keep = sorted(tr_voy.keys())[:train_segments_cap]
        tr_voy = {k: tr_voy[k] for k in keep}
        tr_pings = int(sum(len(v) for v in tr_voy.values()))
        print(f"[mtais-export] SMOKE: train voyages capped to {train_segments_cap}", flush=True)
    va_voy, va_pings = build_voyages(va, roi)
    mean, n_mean = mean_four_hot(tr_voy, roi)
    entries, n_skip = build_test_index(te, roi)

    with open(f"{out_dir}/train.pkl", "wb") as f:
        pickle.dump(tr_voy, f, protocol=PKL_PROTOCOL)
    with open(f"{out_dir}/valid.pkl", "wb") as f:
        pickle.dump(va_voy, f, protocol=PKL_PROTOCOL)
    with open(f"{out_dir}/mean.pkl", "wb") as f:
        pickle.dump(mean, f, protocol=PKL_PROTOCOL)
    with open(f"{out_dir}/test_index.pkl", "wb") as f:
        pickle.dump(entries, f, protocol=PKL_PROTOCOL)

    meta = {
        "track": track, "gap_h": gap, "suffix": suffix,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "caps": {"train_cap": TRAIN_CAP, "train_rng": 20260702, "val_cap": VAL_CAP,
                 "val_rng": 20260703, "train_segments_cap": train_segments_cap,
                 "source": "verbatim run_gap_sweep.load_dataset logic"},
        "roi": roi,
        "normalization": "lat=(lat-lat_lo)/range, lon=((lon-cut)%360-lon_lo)/range, "
                         "sog=clip(sog,0,30)/30, cog=(cog%360)/360; all clipped to [0, 0.999999]",
        "antimeridian": {"lon_cut_deg": roi["lon_cut"],
                         "occupied_arc_deg": roi["lon_hi"] - roi["lon_lo"],
                         "decode": "lon = wrap180(shifted_deg + lon_cut)"},
        "quant_floor_nm": roi["quant_floor_nm"],
        "quant_floor_def": "half bin diagonal: 0.5*hypot(lat_reso*60, lon_reso*60*cos(mid_lat)) NM;"
                           " predictions are bin CENTERS, so this floor is unbeatable by design",
        "min_seg": MIN_SEG,
        "counts": {"n_train_graphs": len(tr), "n_val_graphs": len(va), "n_test_graphs": len(te),
                   "n_train_voyages": len(tr_voy), "n_val_voyages": len(va_voy),
                   "n_train_pings": int(tr_pings), "n_val_pings": int(va_pings),
                   "n_mean_pings": int(n_mean), "n_test_skipped": int(n_skip)},
        "test_protocol": "pre-gap observed pings ONLY per test sequence (one-sided forward model; "
                         "post-gap data cannot influence gap samples and exporting it would "
                         "reintroduce the vendor's one-step boundary-input leak)",
        "held_placement": "by COUNT: the four-hot encoding has no time channel, so the k-th sampled "
                          "step in the gap window is scored against the k-th held node "
                          "(same caveat class as MH-GIN's max_delta cadence mismatch)",
        "pickle_protocol": PKL_PROTOCOL,
    }
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)

    print(f"[mtais-export] {track}/gap_{gap}h{suffix}: {len(tr_voy)} train voyages "
          f"({tr_pings} pings), {len(va_voy)} val voyages, {len(entries)} test gaps "
          f"({n_skip} skipped), data_dim {roi['data_dim']}  ({time.time()-t0:.0f}s)", flush=True)
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="MultitaskAIS stage-1 exporter (gpubench env)")
    ap.add_argument("--track", required=True)
    ap.add_argument("--gap", type=int, required=True)
    ap.add_argument("--suffix", default="", help="e.g. _smoke")
    ap.add_argument("--train-segments-cap", type=int, default=0, help="smoke: keep first N voyages")
    a = ap.parse_args()
    export_cell(a.track, a.gap, a.suffix, a.train_segments_cap)


if __name__ == "__main__":
    main()

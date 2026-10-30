"""
task_alpha.data_standard  --  STANDARD preprocessing track for the gap-length sweep.

This track deliberately uses the field-standard AIS product: the
SAIS_<YEAR>_CLASS_A_B_interptime10m.parquet files (10-minute regular grid, already cleaned the
standard way).  We do NOT re-clean beyond de-duplication + sorting -- that is the point of the
track: "what everyone else's preprocessing gives you".  Contrast with task_alpha.data_raw (our
track), which reads raw message-level pings and subsamples real observations.

SOURCE (READ-ONLY -- NEVER written):
  <paths.STATIC_DIR>/SAIS_<YEAR>_CLASS_A_B_interptime10m.parquet   (env FQGNN_STATIC_DIR)
  columns: mmsi int64, time timestamp[us], latitude, longitude, sog, cog (10-min cadence, lat>=60).

Pipeline:
  load_standard      -> month/Arctic/valid-MMSI filter, dedup (mmsi,time), sort.  Nothing else.
  build_runs_standard-> per MMSI, split at coverage gaps > 3 h (the interp product's own run
                        boundary), keep runs with >= 30 points that are MOVING (median sog >= 1 kn,
                        the same spirit as build_runs_raw's moving-run filter: parked vessels make
                        the imputation task trivial and pollute the metric).
  graphs_standard    -> task_alpha.data.make_run_graph UNCHANGED (max_gap_len=500, n_anchor=8,
                        min_span_factor=1.5), one held-out gap per run, seed stable per run
                        (md5 of "mmsi:run-start-epoch-seconds" -> independent of processing order).
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import pyarrow.dataset as pds

from .data import make_run_graph

from .paths import STATIC_DIR as SAIS_DIR  # env FQGNN_STATIC_DIR (see task_alpha/paths.py)


def interp_path(year: int) -> str:
    return os.path.join(SAIS_DIR, f"SAIS_{year}_CLASS_A_B_interptime10m.parquet")


def available_years(years) -> list[int]:
    """Which of the requested interp10m years exist on disk (missing ones are recorded upstream)."""
    return [int(y) for y in years if os.path.exists(interp_path(int(y)))]


def load_standard(years=(2016, 2017, 2018), months=(7, 8), min_lat: float = 60.0,
                  verbose: bool = False) -> pd.DataFrame:
    """Read the interp10m parquets (READ-ONLY) for `years`, keep `months`, Arctic lat>=min_lat,
    valid 9-digit MMSI (invalid ones can never be split by train_stream._split anyway), then the
    ONLY sanity ops of this track: drop duplicate (mmsi, time) rows and sort.  The product is
    already 10-min regular and standard-cleaned -- no further cleaning by design."""
    months = sorted(int(m) for m in months)
    frames = []
    for y in available_years(years):
        # coarse predicate pushdown: contiguous time range covering the requested months (exact
        # month filter applied after), plus the Arctic filter -- keeps the pandas side small.
        lo = datetime(y, months[0], 1)
        hi = datetime(y + 1, 1, 1) if months[-1] == 12 else datetime(y, months[-1] + 1, 1)
        dset = pds.dataset(interp_path(y), format="parquet")
        filt = (pds.field("time") >= lo) & (pds.field("time") < hi) & \
               (pds.field("latitude") >= min_lat)
        d = dset.to_table(filter=filt).to_pandas()
        d = d[d["time"].dt.month.isin(months)
              & d["mmsi"].between(100_000_000, 999_999_999)
              & d["longitude"].between(-180, 180)]
        if verbose:
            print(f"  interp10m {y}: {len(d):>9} rows, {d['mmsi'].nunique():>5} vessels", flush=True)
        if len(d):
            frames.append(d)
    if not frames:
        raise FileNotFoundError(f"no interp10m data for years={tuple(years)} months={tuple(months)}")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(["mmsi", "time"]).sort_values(["mmsi", "time"]).reset_index(drop=True)
    return df


def build_runs_standard(df: pd.DataFrame, coverage_gap_hours: float = 3.0, min_len: int = 30,
                        min_sog: float = 1.0) -> List[pd.DataFrame]:
    """Per MMSI: split at coverage gaps > coverage_gap_hours (the interp product's own run
    boundary), keep runs with >= min_len points and MOVING (median sog >= min_sog kn)."""
    runs = []
    for _, g in df.groupby("mmsi"):
        g = g.sort_values("time")
        t = g["time"].to_numpy().astype("datetime64[s]").astype("int64")
        brk = np.where(np.diff(t) > coverage_gap_hours * 3600)[0]
        for idx in np.split(np.arange(len(g)), brk + 1):
            if len(idx) < min_len:
                continue
            seg = g.iloc[idx]
            if float(seg["sog"].median()) < min_sog:
                continue
            runs.append(seg.reset_index(drop=True))
    return runs


def run_seed(run: pd.DataFrame) -> int:
    """Stable per-run seed: md5 of 'mmsi:run-start-epoch-seconds' -- deterministic and independent
    of how the years/months were chunked while loading."""
    mmsi = int(run["mmsi"].iloc[0])
    t0 = int(run["time"].iloc[0].to_datetime64().astype("datetime64[s]").astype("int64"))
    return int(hashlib.md5(f"{mmsi}:{t0}".encode()).hexdigest(), 16) % (2 ** 31)


def graphs_standard(runs: List[pd.DataFrame], gap_minutes: float, max_gap_len: int = 500,
                    n_anchor: int = 8, min_span_factor: float = 1.5, thin_k: int = 1) -> list:
    """make_run_graph UNCHANGED on every run (one held-out gap per run); keep graphs with >=1 held
    node.

    thin_k (de-circularization): keep every k-th point of the 10-min grid
    BEFORE building the graph, so consecutive truth points are >= k*10 min apart and straddle REAL
    path curvature instead of the linear-interpolation microstructure between raw pings (the truth
    between real pings in the interp product is linear; at 10-min spacing the task partly reduces to
    reproducing linear fill).  The per-run grid offset is stable (run_seed % k) so the retained-grid
    phase varies across runs but is deterministic.  Anchors AND held targets share the coarse
    cadence -- same principle as the raw track's thinning."""
    graphs = []
    for run in runs:
        seed = run_seed(run)
        if thin_k > 1:
            run = run.iloc[(seed % thin_k)::thin_k].reset_index(drop=True)
            if len(run) < 15:                                # too short after thinning
                continue
        g = make_run_graph(run, gap_minutes=gap_minutes, max_gap_len=max_gap_len,
                           n_anchor=n_anchor, min_span_factor=min_span_factor, seed=seed)
        if g is not None and int(g.mask.sum()) > 0:
            graphs.append(g)
    return graphs

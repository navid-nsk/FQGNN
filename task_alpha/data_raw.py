"""
task_alpha.data_raw  --  RAW AIS pipeline (the study's task; data.py keeps the interpolated loader).

WHY: the *_interptime10m.parquet files are LINEARLY INTERPOLATED to 10-min spacing, so the held-out
"truth" between real pings is itself a straight line, and scoring against it asks "does linear
interp reproduce linearly-interpolated data?" (circular; the GCD median error is ~0.25 NM and
cannot be beaten).  Here we read the RAW message-level pings, so the true path genuinely CURVES
through a gap and linear interp has a real error that grows with the gap length.

SOURCE (READ-ONLY, never written):  paths.RAW_DIR (env FQGNN_RAW_DIR; see task_alpha/paths.py)
  * m123/year=Y/month=M/  Class A position reports (msg 1/2/3), 2011-2019
  * m18 /year=Y/month=M/  Class B position reports (msg 18),     2011-2019
  * m27 /year=Y/month=M/  Class A long-range position (msg 27),  2013-2019
  * combined/year=Y/...   already-combined positions,            2020-2021
Class A = m123 + m27 ; Class B = m18.  Combining all three = Class A + B (as required).

We keep positions REAL: runs are split at >1 h coverage gaps, then THINNED to a target cadence by
keeping, per time-bin, the real ping NEAREST the bin centre (subsampling observations -- NOT
interpolation).  The held-out gap is then a contiguous middle window of real pings.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from typing import List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from .data import haversine_nm, make_run_graph
from .paths import CACHE_ROOT, RAW_DIR, STATIC_DIR

# Input and cache locations come from task_alpha/paths.py (environment variables FQGNN_DATA_ROOT,
# FQGNN_RAW_DIR, FQGNN_STATIC_DIR and FQGNN_CACHE_ROOT; the defaults lie inside the repository).
CACHE_DIR = CACHE_ROOT
POS_COLS = ["mmsi", "time", "latitude", "longitude", "sog", "cog", "vclass"]

_CLEAN_CACHE = {}


def _clean_whitelist(year):
    """Set of VALIDATED MMSIs for `year` from clean/static_clean_<year>.parquet (None if absent).
    Restricting raw positions to this set removes ~93% of glitch mass (bad/spoofed MMSIs)."""
    if year not in _CLEAN_CACHE:
        p = os.path.join(STATIC_DIR, "clean", f"static_clean_{year}.parquet")
        if os.path.exists(p):
            mm = pq.read_table(p, columns=["mmsi"]).to_pandas()["mmsi"].dropna()
            _CLEAN_CACHE[year] = set(int(x) for x in mm.unique())
        else:
            _CLEAN_CACHE[year] = None
    return _CLEAN_CACHE[year]


def _ts_seconds(s) -> np.ndarray:
    """datetime -> int64 seconds since epoch, dtype-agnostic (raw ts is datetime64[ms])."""
    return pd.to_datetime(s).values.astype("datetime64[s]").astype("int64")


def _read_part(mt: str, year: int, month: int, vclass: str, max_files: int) -> pd.DataFrame | None:
    """Read up to max_files part-files of one (message-type, year, month) partition, standardized."""
    fs = sorted(glob.glob(os.path.join(RAW_DIR, mt, f"year={year}", f"month={month:02d}", "*.parquet")))
    if not fs:
        return None
    cols = ["mmsi", "latitude", "longitude", "sog", "cog", "ts"]
    parts = []
    for f in fs[:max_files]:
        avail = set(pq.ParquetFile(f).schema_arrow.names)
        use = [c for c in cols if c in avail]
        d = pq.read_table(f, columns=use).to_pandas()
        d = d.rename(columns={"ts": "time"})
        if "cog" not in d:
            d["cog"] = np.nan
        d["vclass"] = vclass
        parts.append(d[["mmsi", "time", "latitude", "longitude", "sog", "cog", "vclass"]])
    return pd.concat(parts, ignore_index=True) if parts else None


def load_raw(years=(2013,), months=(6, 7), n_mmsi=150, min_lat: float = 60.0,
             max_files_per_part=2, seed: int = 0, clean_only: bool = True) -> pd.DataFrame:
    """Combine m123/m18/m27 (<=2019) or `combined` (2020-21) for the given year/month partitions;
    Arctic-filter, clean, de-duplicate, and keep a random sample of n_mmsi vessels.
    n_mmsi=None -> keep ALL vessels; max_files_per_part=None -> read ALL part files (= use ALL data).
    clean_only=True -> restrict to the VALIDATED MMSI whitelist clean/static_clean_<year>.parquet
    (removes ~93% of glitches)."""
    frames = []
    for y in years:
        for mo in months:
            if y <= 2019:
                for mt, vc in [("m123", "A"), ("m18", "B"), ("m27", "A")]:
                    d = _read_part(mt, y, mo, vc, max_files_per_part)
                    if d is not None:
                        frames.append(d)
            else:
                d = _read_part("combined", y, mo, "AB", max_files_per_part)
                if d is not None:
                    frames.append(d)
    if not frames:
        raise FileNotFoundError(f"no raw partitions found for years={years} months={months}")
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    df = df[(df.latitude >= min_lat) & (df.latitude <= 90) & df.longitude.between(-180, 180)
            & (df.sog >= 0) & (df.sog < 60) & (df.mmsi != 0)]
    df = df.dropna(subset=["time", "latitude", "longitude", "mmsi"])
    df["cog"] = df["cog"].where(df["cog"].between(0, 360), other=np.nan)   # invalid course -> NaN (scrubbed later)
    if clean_only:                                          # restrict to VALIDATED vessels (glitch removal at source)
        wl = set()
        for y in years:
            w = _clean_whitelist(int(y))
            if w:
                wl |= w
        if wl:
            df = df[df["mmsi"].isin(wl)]
    df = df.drop_duplicates(["mmsi", "time"]).sort_values(["mmsi", "time"]).reset_index(drop=True)
    if n_mmsi is None:                                   # keep ALL vessels (use all data)
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    vessels = df["mmsi"].unique()
    if len(vessels) > n_mmsi:
        vessels = rng.choice(vessels, size=n_mmsi, replace=False)
    return df[df["mmsi"].isin(vessels)].reset_index(drop=True)


def _thin(seg: pd.DataFrame, thin_sec: int) -> pd.DataFrame:
    """Greedy subsample of REAL pings with GUARANTEED monotone spacing >= thin_sec (keep a ping only
    if >= thin_sec since the last kept one).  Subsampling -- never interpolates/averages positions."""
    seg = seg.sort_values("time")
    t = _ts_seconds(seg["time"])
    keep = []; last = -1e18
    for i in range(len(t)):
        if t[i] - last >= thin_sec:
            keep.append(i); last = t[i]
    return seg.iloc[keep].reset_index(drop=True)


def build_runs_raw(df: pd.DataFrame, gap_hours: float = 1.0, thin_sec: int = 180, min_len: int = 30,
                   min_span_h: float = 3.0, min_sog: float = 1.0, max_speed_kn: float = 60.0
                   ) -> List[pd.DataFrame]:
    """Per vessel: split at >gap_hours coverage gaps, thin to thin_sec, keep MOVING runs (span >=
    min_span_h, median sog > min_sog, >=12 thinned pts) with no physically-impossible step (> max_speed_kn)."""
    runs = []
    for _, g in df.groupby("mmsi"):
        g = g.sort_values("time")
        t = _ts_seconds(g["time"])
        brk = np.where(np.diff(t) > gap_hours * 3600)[0]
        for s in np.split(np.arange(len(g)), brk + 1):
            if len(s) < min_len:
                continue
            seg = _thin(g.iloc[s], thin_sec)
            if len(seg) < 12:
                continue
            tt = _ts_seconds(seg["time"])
            if (tt[-1] - tt[0]) < min_span_h * 3600 or seg["sog"].median() < min_sog:
                continue
            la, lo = seg.latitude.values, seg.longitude.values
            step = haversine_nm(la[:-1], lo[:-1], la[1:], lo[1:])
            dt_h = np.maximum(np.diff(tt) / 3600.0, 1e-6)
            if (step / dt_h).max() > max_speed_kn:        # implausible jump -> bad record, drop run
                continue
            runs.append(seg.reset_index(drop=True))
    return runs


def _cache_key(n_mmsi, gap_minutes, k_temporal, thin_sec, min_lat, max_files_per_part, seed,
               run_filters, mk, clean_only=True) -> str:
    """Short hash of EVERY build-affecting parameter (years/months excluded -- they pick partitions).
    Any change to these invalidates the cache and forces a rebuild, so a cache hit is always correct."""
    d = dict(n_mmsi=n_mmsi, gap_minutes=gap_minutes, k_temporal=k_temporal, thin_sec=thin_sec,
             min_lat=min_lat, max_files=max_files_per_part, seed=seed, clean_only=clean_only,
             rf=sorted((run_filters or {}).items()), mk=sorted(mk.items()))
    return hashlib.md5(json.dumps(d, default=str, sort_keys=True).encode()).hexdigest()[:12]


def build_multi_horizon(years, months, gaps=(240, 360, 480), n_mmsi=None, max_files_per_part=None,
                        thin_sec=180, min_lat=60.0, seed=0, clean_only=True, run_filters=None,
                        cache_dir=None, verbose=True, max_gap_len=500, n_anchor=8,
                        min_span_factor=1.5, k_temporal=6):
    """Build + cache MULTIPLE gap horizons reading each (year,month) parquet ONCE (whitelist-filtered,
    runs built once) -> avoids the 3x parquet read. Per-partition caching => RESUMABLE (re-run skips
    cached partitions). Keys match build_dataset_raw(gap_minutes=gm, max_gap_len, min_span_factor,
    n_anchor, thin_sec, clean_only, ...) so later loads hit the cache."""
    mk = dict(max_gap_len=max_gap_len, n_anchor=n_anchor, min_span_factor=min_span_factor)
    keys = {gm: _cache_key(n_mmsi, gm, k_temporal, thin_sec, min_lat, max_files_per_part, seed,
                           run_filters, mk, clean_only=clean_only) for gm in gaps}
    if cache_dir is None:
        cache_dir = CACHE_DIR                          # portable default (see header)
    os.makedirs(cache_dir, exist_ok=True)
    totals = {gm: 0 for gm in gaps}
    for y in years:
        for mo in months:
            paths = {gm: os.path.join(cache_dir, f"part_{keys[gm]}_{y}_{int(mo):02d}.pt") for gm in gaps}
            if all(os.path.exists(p) for p in paths.values()):
                for gm in gaps:
                    totals[gm] += len(torch.load(paths[gm], weights_only=False))
                if verbose:
                    print(f"  {y}-{mo:02d}: all horizons cached", flush=True)
                continue
            pseed = seed + int(y) * 100 + int(mo)
            try:
                df = load_raw(years=[y], months=[mo], n_mmsi=n_mmsi, min_lat=min_lat,
                              max_files_per_part=max_files_per_part, seed=pseed, clean_only=clean_only)
            except FileNotFoundError:
                for gm in gaps:
                    if not os.path.exists(paths[gm]):
                        torch.save([], paths[gm])
                continue
            runs = build_runs_raw(df, thin_sec=thin_sec, **(run_filters or {}))   # read+runs ONCE
            cnt = {}
            for gm in gaps:
                if os.path.exists(paths[gm]):
                    continue
                part = [g for i, run in enumerate(runs)
                        for g in [make_run_graph(run, gap_minutes=gm, k_temporal=k_temporal,
                                                 seed=pseed + i, **mk)] if g is not None and int(g.mask.sum()) > 0]
                torch.save(part, paths[gm]); cnt[gm] = len(part); totals[gm] += len(part)
            if verbose:
                print(f"  {y}-{mo:02d}: runs {len(runs):>4} -> " +
                      " ".join(f"{gm//60}h:{cnt.get(gm, 'cached')}" for gm in gaps) +
                      f" | cum " + " ".join(f"{gm//60}h:{totals[gm]}" for gm in gaps), flush=True)
            del df, runs
    print("DONE multi-horizon: " + " ".join(f"{gm//60}h={totals[gm]}" for gm in gaps) +
          f" graphs | keys " + " ".join(f"{gm//60}h:{keys[gm]}" for gm in gaps), flush=True)
    return keys, totals


def build_dataset_raw(years=(2013,), months=(6, 7), n_mmsi: int = 150, gap_minutes: float = 90.0,
                      k_temporal: int = 6, thin_sec: int = 180, min_lat: float = 60.0,
                      max_files_per_part: int = 2, seed: int = 0,
                      run_filters: dict | None = None, verbose: bool = False,
                      cache: bool = True, cache_dir: str = None,
                      clean_only: bool = True, **mk) -> List:
    """End-to-end RAW pipeline, STREAMED + CACHED per (year, month) partition.  The build is
    deterministic given the config+seed, so the FIRST run of a partition reads the parquet, builds the
    graphs and writes them to `cache_dir/part_<key>_<year>_<month>.pt`; EVERY later run loads that file
    in a fraction of a second instead of re-reading/rebuilding.  Memory stays bounded (one partition at
    a time).  `run_filters` -> build_runs_raw; extra kwargs (max_gap_len/n_anchor/min_anchor/
    min_span_factor) -> make_run_graph.  Set cache=False to force a rebuild."""
    key = _cache_key(n_mmsi, gap_minutes, k_temporal, thin_sec, min_lat, max_files_per_part, seed,
                     run_filters, mk, clean_only=clean_only)
    if cache_dir is None:
        cache_dir = CACHE_DIR                          # portable default (see header)
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
    graphs = []
    for y in years:
        for mo in months:
            cpath = os.path.join(cache_dir, f"part_{key}_{y}_{int(mo):02d}.pt")
            if cache and os.path.exists(cpath):                       # CACHE HIT -> load (seconds)
                part = torch.load(cpath, weights_only=False)
                graphs.extend(part)
                if verbose:
                    print(f"  {y}-{mo:02d}: CACHED +{len(part):>4} (total {len(graphs)})", flush=True)
                continue
            pseed = seed + int(y) * 100 + int(mo)          # reproducible, partition-specific vessel sample
            try:
                df = load_raw(years=[y], months=[mo], n_mmsi=n_mmsi, min_lat=min_lat,
                              max_files_per_part=max_files_per_part, seed=pseed, clean_only=clean_only)
            except FileNotFoundError:
                if cache and not os.path.exists(cpath):    # NEVER overwrite an existing cache file
                    torch.save([], cpath)                  # cache the empty partition too
                if verbose:
                    print(f"  {y}-{mo:02d}: (no data)", flush=True)
                continue
            runs = build_runs_raw(df, thin_sec=thin_sec, **(run_filters or {}))
            part = []
            for i, run in enumerate(runs):                  # per-partition LOCAL seed -> partition-independent
                g = make_run_graph(run, gap_minutes=gap_minutes, k_temporal=k_temporal,
                                   seed=pseed + i, **mk)
                if g is not None and int(g.mask.sum()) > 0:
                    part.append(g)
            if cache and not os.path.exists(cpath):        # NEVER overwrite an existing cache file
                torch.save(part, cpath)
            graphs.extend(part)
            if verbose:
                print(f"  {y}-{mo:02d}: rows {len(df):>8} runs {len(runs):>4} -> +{len(part):>4} "
                      f"graphs (total {len(graphs)})", flush=True)
            del df, runs
    return graphs

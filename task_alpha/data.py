"""
task_alpha.data  --  Task Alpha (Reconstruction-24h) data pipeline.

Vessel-trajectory gap-filling / imputation:
  * SOURCE (READ-ONLY): SAIS_<YEAR>_CLASS_A_B_interptime10m.parquet in paths.STATIC_DIR
    columns: mmsi, time, latitude, longitude, sog, cog  (10-min interpolated, Arctic lat>=60).
  * RUNS: per-MMSI tracks split into continuous runs at gaps > 3 h.
  * TASK: hold out a contiguous middle SEGMENT of a run (the "gap"), keep ANCHORS on BOTH sides;
    predict the held-out lat/lon (two-sided interpolation).
  * GEOMETRY: a local ENU tangent plane (metres) for the model; lat/lon kept for the great-circle
    (Haversine) ADE/FDE metric in nautical miles.

NEVER writes to the source.  All outputs are torch tensors / a list of per-run graphs.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from .paths import STATIC_DIR as SAIS_DIR  # env FQGNN_STATIC_DIR (see task_alpha/paths.py)
NM_PER_M = 1.0 / 1852.0


# ----------------------------------------------------------------------------------------
# geometry: local ENU tangent (metres) <-> lat/lon ; great-circle distance in NM
# ----------------------------------------------------------------------------------------
def lonlat_to_enu(lat, lon, lat0, lon0):
    """Equirectangular local tangent (metres) about (lat0, lon0).  Valid for short runs."""
    x = np.radians(lon - lon0) * math.cos(math.radians(lat0)) * 6371000.0
    y = np.radians(lat - lat0) * 6371000.0
    return x, y


def enu_to_lonlat(x, y, lat0, lon0):
    lat = lat0 + np.degrees(y / 6371000.0)
    lon = lon0 + np.degrees(x / (6371000.0 * math.cos(math.radians(lat0))))
    return lat, lon


def haversine_nm(lat1, lon1, lat2, lon2):
    """Great-circle distance in nautical miles (vectorized)."""
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a)) * NM_PER_M


# ----------------------------------------------------------------------------------------
# load + runs
# ----------------------------------------------------------------------------------------
def load_sais(year: int = 2013, n_mmsi: int = 60, min_lat: float = 60.0,
              max_row_groups: int = 12, seed: int = 0) -> pd.DataFrame:
    """
    Read a SUBSET of the SAIS parquet (READ-ONLY): the first `max_row_groups` row groups, filter to
    lat>=min_lat and mmsi!=0, then keep a random sample of `n_mmsi` distinct vessels.
    """
    path = os.path.join(SAIS_DIR, f"SAIS_{year}_CLASS_A_B_interptime10m.parquet")
    pf = pq.ParquetFile(path)
    ng = min(max_row_groups, pf.num_row_groups)
    df = pf.read_row_groups(list(range(ng))).to_pandas()
    df = df[(df["mmsi"] != 0) & (df["latitude"] >= min_lat)].copy()
    rng = np.random.default_rng(seed)
    vessels = df["mmsi"].unique()
    if len(vessels) > n_mmsi:
        vessels = rng.choice(vessels, size=n_mmsi, replace=False)
    df = df[df["mmsi"].isin(vessels)].sort_values(["mmsi", "time"]).reset_index(drop=True)
    return df


def build_runs(df: pd.DataFrame, gap_hours: float = 3.0, min_len: int = 24,
               max_step_nm: float = 10.0) -> List[pd.DataFrame]:
    """Split each MMSI track into continuous runs at time gaps > gap_hours; keep runs with >= min_len
    points (a 24-pt run = ~4 h at 10-min spacing) AND no implausible jump (any consecutive
    great-circle step > max_step_nm NM => bad AIS record, drop the whole run)."""
    runs = []
    for _, g in df.groupby("mmsi"):
        g = g.sort_values("time")
        dt_h = g["time"].diff().dt.total_seconds().fillna(0.0) / 3600.0
        breaks = (dt_h > gap_hours).cumsum()
        for _, run in g.groupby(breaks):
            if len(run) < min_len:
                continue
            la = run["latitude"].to_numpy(); lo = run["longitude"].to_numpy()
            step = haversine_nm(la[:-1], lo[:-1], la[1:], lo[1:])
            if step.max() > max_step_nm:
                continue
            runs.append(run.reset_index(drop=True))
    return runs


# ----------------------------------------------------------------------------------------
# the gap-filling task graph
# ----------------------------------------------------------------------------------------
POS_OUT_SCALE = 1852.0 * 20.0   # ENU-position normalization (~20 NM units) for the ABSOLUTE target/feature


@dataclass
class RunGraph:
    x: torch.Tensor          # [N, F] node features: ENU position + sog + sin/cos cog -- ALL ZEROED for held-out
                             #        nodes (no in-gap leakage), only is_held=1 marks them
    coord: torch.Tensor      # [N, 2] the FQ-lift input = TIME coordinate (t_norm, dt_to_nearest_anchor_norm) -- known for
                             #        held-out, contains NO position info (no GCD/linear-interp leakage)
    y_pos: torch.Tensor      # [N, 2] TARGET = the TRUE ABSOLUTE ENU position (POS_OUT_SCALE units). NOT a residual over GCD.
    edge_index: torch.Tensor # [2, E] temporal kNN (two-sided)
    edge_dist: torch.Tensor  # [E] |Δt| in hours
    anchor_edge_index: torch.Tensor  # [2, A] (anchor_src -> held_dst): each held node to its nearest n_anchor//2
                             #          anchors BEFORE + n_anchor//2 AFTER the gap -- guaranteed TWO-SIDED
    mask: torch.Tensor       # [N] bool, held-out vertices (loss + metric only here)
    linterp_lonlat: torch.Tensor  # [N, 2] (lat, lon) of the linear-interp guess -- BASELINE ONLY (never fed to the model)
    true_lonlat: torch.Tensor     # [N, 2] (lat, lon) truth
    t_since_anchor: torch.Tensor  # [N] minutes since the last kept anchor (metric bucketing / gap-depth)
    ref: tuple               # (lat0, lon0) ENU reference
    mmsi: int                # vessel id -> VESSEL-DISJOINT train/val split (no cross-split vessel leakage)
    gap_minutes: float       # realized held-window duration (min) -> time-bucketed reporting


def make_run_graph(run: pd.DataFrame, gap_minutes: float = 90.0, min_anchor: int = 6,
                   k_temporal: int = 6, max_gap_len: int = 40, n_anchor: int = 8,
                   min_span_factor: float = 2.0, seed: int = 0) -> RunGraph | None:
    """Hold out a contiguous TIME WINDOW of `gap_minutes` at a random (jittered) valid position with
    >= min_anchor real pings on BOTH sides.  The model predicts the held-out vertices' ABSOLUTE
    positions from the ANCHOR pings -- held-out nodes carry NO observation (position+velocity zeroed);
    the linear-interp is computed as a baseline only, never fed to the model."""
    n = len(run)
    rng = np.random.default_rng(seed)
    lat = run["latitude"].to_numpy(); lon = run["longitude"].to_numpy()
    sog = run["sog"].to_numpy(); cog = run["cog"].to_numpy()
    t = run["time"].to_numpy().astype("datetime64[s]").astype("int64").astype(float)  # -> SECONDS (dtype-agnostic)
    mmsi = int(run["mmsi"].iloc[0]) if "mmsi" in run.columns else -1
    dur = max(t[-1] - t[0], 1e-6)
    W = gap_minutes * 60.0
    if dur < W * min_span_factor:                     # run must fit the gap + anchors on both sides
        return None
    lat0, lon0 = float(lat.mean()), float(lon.mean())
    ex, ey = lonlat_to_enu(lat, lon, lat0, lon0)

    # ---- pick a TIME-window gap at a random valid start (jittered -> not always dead-centre) ----
    lo_i, hi_i = min_anchor, n - min_anchor - 2
    if hi_i <= lo_i:
        return None
    si = int(rng.integers(lo_i, hi_i + 1))            # first held index (>= min_anchor anchors before)
    last = si
    while last + 1 < n and t[last + 1] <= t[si] + W:  # extend to a real elapsed window of W seconds
        last += 1
    held = np.zeros(n, dtype=bool); held[si:last + 1] = True
    glen = int(held.sum())
    if glen < 2 or glen > max_gap_len or (n - (last + 1)) < min_anchor:
        return None
    aL, aR = si - 1, last + 1                          # bracketing anchors
    gap_real_min = (t[last] - t[si]) / 60.0

    # ---- linear-interp guess: BASELINE ONLY (computed, never fed to the model) -- vectorized ----
    li_x, li_y = ex.copy(), ey.copy()
    w = (t[si:last + 1] - t[aL]) / max(t[aR] - t[aL], 1e-6)
    li_x[si:last + 1] = (1 - w) * ex[aL] + w * ex[aR]
    li_y[si:last + 1] = (1 - w) * ey[aL] + w * ey[aR]
    li_lat, li_lon = enu_to_lonlat(li_x, li_y, lat0, lon0)

    # ---- FQ-lift TIME coordinate (NO position): (t_norm, dt_to_nearest_kept-anchor_norm) ----
    anchor_t = t[~held]                                # sorted (subset of sorted t)
    t_norm = (t - t[0]) / dur
    idx = np.searchsorted(anchor_t, t)                 # nearest anchor = one of the two bracketing (vectorized)
    il = np.clip(idx - 1, 0, len(anchor_t) - 1); ir = np.clip(idx, 0, len(anchor_t) - 1)
    dt_near = np.minimum(np.abs(t - anchor_t[il]), np.abs(t - anchor_t[ir])) / dur   # 0 at anchors, grows mid-gap
    coord = np.nan_to_num(np.stack([t_norm, dt_near], axis=1))

    # ---- features: ALL observation channels (position + speed + course) ZEROED for held-out nodes
    #      -> no in-gap position OR velocity leakage; only is_held marks them. NaN-scrubbed. ----
    z = lambda arr: np.where(held, 0.0, arr)
    feat = np.stack([z(ex / POS_OUT_SCALE), z(ey / POS_OUT_SCALE), z(sog / 20.0),
                     z(np.sin(np.radians(cog))), z(np.cos(np.radians(cog))),
                     held.astype(float)], axis=1)
    feat = np.nan_to_num(feat)                         # any NaN cog on an anchor -> 0 (no NaN poison)

    # ---- ABSOLUTE target: the TRUE ENU position (NOT a residual) ----
    y_pos = np.stack([ex, ey], axis=1) / POS_OUT_SCALE

    # temporal kNN edges (two-sided) -- vectorized band (src=i+off, dst=i); identical set/order/values
    offs = np.array([o for o in range(-k_temporal, k_temporal + 1) if o != 0])
    de = np.repeat(np.arange(n), len(offs)); se = de + np.tile(offs, n)
    valid = (se >= 0) & (se < n); se = se[valid]; de = de[valid]
    dd = (np.abs(t[se] - t[de]) / 3600.0)
    edge_index = np.stack([se, de], axis=0)

    # ---- TWO-SIDED anchor edges: n_anchor//2 nearest anchors BEFORE the gap + n_anchor//2 AFTER,
    #      so every held node can interpolate ACROSS the gap (not a one-sided hull). ----
    before = np.arange(0, si); after = np.arange(last + 1, n); nb = max(1, n_anchor // 2)
    asrc, adst = [], []
    for i in range(si, last + 1):
        bs = before[np.argsort(np.abs(t[before] - t[i]))[:nb]]
        af = after[np.argsort(np.abs(t[after] - t[i]))[:nb]]
        for j in np.concatenate([bs, af]):
            asrc.append(int(j)); adst.append(int(i))
    anchor_edge_index = np.stack([asrc, adst], axis=0) if asrc else np.zeros((2, 0), dtype=int)

    tsa = np.zeros(n); last_a = t[0]
    for i in range(n):
        if not held[i]:
            last_a = t[i]
        tsa[i] = (t[i] - last_a) / 60.0

    f32 = lambda a: torch.tensor(a, dtype=torch.float32)
    return RunGraph(
        x=f32(feat), coord=f32(coord), y_pos=f32(y_pos),
        edge_index=torch.tensor(edge_index, dtype=torch.long), edge_dist=f32(dd),
        anchor_edge_index=torch.tensor(anchor_edge_index, dtype=torch.long),
        mask=torch.tensor(held), linterp_lonlat=f32(np.stack([li_lat, li_lon], 1)),
        true_lonlat=f32(np.stack([lat, lon], 1)), t_since_anchor=f32(tsa), ref=(lat0, lon0),
        mmsi=mmsi, gap_minutes=float(gap_real_min))


def collate_graphs(graphs: List[RunGraph]):
    """Merge run-graphs into ONE disjoint-union graph (edges offset per graph -> no cross-graph
    leakage) so the shared quasicrystal amplitude c is computed ONCE per batch, not per graph.
    Returns (x, coord, y_pos, mask, edge_index, edge_dist)."""
    xs, coords, ys, masks, eis, eds = [], [], [], [], [], []
    off = 0
    for g in graphs:
        xs.append(g.x); coords.append(g.coord); ys.append(g.y_pos); masks.append(g.mask)
        eis.append(g.edge_index + off); eds.append(g.edge_dist)
        off += g.x.shape[0]
    return (torch.cat(xs), torch.cat(coords), torch.cat(ys), torch.cat(masks),
            torch.cat(eis, dim=1), torch.cat(eds))


def collate_graphs_interp(graphs: List[RunGraph]):
    """Like collate_graphs but ALSO returns the offset anchor-edge index (for the Op-D-interpolation
    readout). Returns (x, coord, y_pos, mask, edge_index, edge_dist, anchor_edge_index)."""
    xs, coords, ys, masks, eis, eds, aeis = [], [], [], [], [], [], []
    off = 0
    for g in graphs:
        xs.append(g.x); coords.append(g.coord); ys.append(g.y_pos); masks.append(g.mask)
        eis.append(g.edge_index + off); eds.append(g.edge_dist)
        if g.anchor_edge_index.numel():
            aeis.append(g.anchor_edge_index + off)
        off += g.x.shape[0]
    aei = torch.cat(aeis, dim=1) if aeis else torch.zeros((2, 0), dtype=torch.long)
    return (torch.cat(xs), torch.cat(coords), torch.cat(ys), torch.cat(masks),
            torch.cat(eis, dim=1), torch.cat(eds), aei)


def build_dataset(year: int = 2013, n_mmsi: int = 60, gap_minutes: float = 90.0,
                  k_temporal: int = 6, seed: int = 0, **load_kw) -> List[RunGraph]:
    """End-to-end on the 10-min interpolated source (a circular task; the study uses data_raw.build_dataset_raw)."""
    df = load_sais(year=year, n_mmsi=n_mmsi, seed=seed, **load_kw)
    runs = build_runs(df)
    graphs = []
    for i, run in enumerate(runs):
        g = make_run_graph(run, gap_minutes=gap_minutes, k_temporal=k_temporal, seed=seed + i)
        if g is not None and int(g.mask.sum()) > 0:
            graphs.append(g)
    return graphs

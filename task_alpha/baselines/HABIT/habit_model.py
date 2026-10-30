"""HABIT (EDBT 2026, Spiliopoulos et al.) adapted to the Task-Alpha two-sided gap benchmark.

VANILLA-FAITHFULNESS CONTRACT
-----------------------------
The METHOD is used unchanged: build an H3 cell-transition graph from historical AIS with the
authors' *verbatim* DuckDB aggregation query (`HabitImputer.prepare_query`), weight edges by
(100 - min(transitions,100))/100, then A* between the cells of the two gap anchors and project
cells -> coordinates with the authors' per-cell MEDIAN lat/lon ("data-driven projection",
`fill_gap_weighted`).  Nothing about the routing, the weighting or the projection is altered.

Three changes, all NECESSARY and none of which help HABIT beat anyone:
  (1) PERFORMANCE.  Vanilla `data_load` does `for n in nodes: df[df.h3 == n]` -- an O(|nodes| x |rows|)
      pandas scan that never terminates on our ~7e5-row corpus.  Replaced by an O(|rows|) groupby
      lookup that yields *identical* node attributes.
  (2) UPSTREAM ENDPOINT DEFECTS (not reproduced here; they would cripple HABIT and misrepresent it):
        - imputer.py:187  the destination nearest-neighbour query passes `origin` instead of
          `destination`, so an off-graph destination snaps to the ORIGIN's neighbour.
        - imputer.py:192-197  when an endpoint is off-graph its cell is APPENDED to the end of the
          path (`cells + [origin]`), corrupting the polyline order.
      We instead route between the nearest graph nodes and then attach the *true* anchor
      coordinates as the polyline's first/last vertices, which is the authors' stated intent.
  (3) TIME PARAMETERISATION.  HABIT returns a geometry (a route), not timestamped positions.  Our
      metric needs (lat,lon) at each held timestamp, so the route polyline is sampled at constant
      speed: held time -> fraction of elapsed gap time -> that fraction of route arclength.  This is
      the neutral choice (a vessel traversing its route); no model-specific information is used.

Leak safety: the corpus and the anchors are built ONLY from observed (non-held) pings.  Held
lat/lon are never read.  Corpus vessels are the vessel-disjoint train (+val) split, plus -- with
`corpus="all_observed"` -- the *observed* pings of the test graphs themselves, which is exactly the
two-sided context every other model already sees at inference time.

A* can fail (disconnected graph, sparse Arctic traffic).  Failures are NOT hidden: the fraction is
reported, and those gaps fall back to a great-circle route between the anchors (what HABIT
degenerates to when it knows no route).
"""
from __future__ import annotations

import math
import re
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR = os.path.abspath(os.path.join(_HERE, "..", "HABIT-main", "habit", "src"))
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

import networkx as nx  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

import h3.api.numpy_int as h3  # noqa: E402
from habit.imputer import HabitImputer  # noqa: E402  (vendor, unmodified)

from task_alpha.baselines.common_gaps import to_series  # noqa: E402

EARTH_NM = 3440.065  # mean earth radius in nautical miles


# ---------------------------------------------------------------------------------------------- #
#  corpus                                                                                          #
# ---------------------------------------------------------------------------------------------- #
def _cog_deg(sin_c: np.ndarray, cos_c: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(sin_c, cos_c)) % 360.0


def _observed_segments(held: np.ndarray):
    """Index runs of OBSERVED pings, split at every held block.  A gap is a discontinuity in the
    observed record, so the pings either side of it are two distinct segments -- never one voyage."""
    segs, start = [], None
    for k, h in enumerate(held):
        if not h and start is None:
            start = k
        elif h and start is not None:
            segs.append((start, k)); start = None
    if start is not None:
        segs.append((start, len(held)))
    return segs


def build_corpus_csv(graphs, path: str) -> int:
    """Write the historical-AIS corpus HABIT indexes: OBSERVED pings only (held nodes excluded).

    TRIP is unique per (graph, observed segment).  This matters: the aggregation SQL builds edges from
    `lag(h3) over (partition by TRIP order by TIMESTAMP)`, so emitting a gapped graph as ONE trip would
    make the last pre-gap ping and the first post-gap ping "consecutive" and mint a single edge that
    jumps clean across the gap.  A* would take that edge and HABIT would collapse to the great-circle
    chord (a direct 2-cell path with a ratio of exactly 1.000).  That edge corresponds to
    no observed transition; splitting at held blocks removes it and lets `max_gap` do its real job of
    rejecting implausibly long within-segment jumps.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols, trip = [], 0
    for g in graphs:
        s = to_series(g, crop_ctx=10 ** 9)
        mmsi = int(g.mmsi)
        for lo, hi in _observed_segments(s["held"]):
            if hi - lo < 2:
                continue
            sl = slice(lo, hi)
            m = hi - lo
            cols.append(np.column_stack([
                np.full(m, trip, dtype=np.float64), np.full(m, mmsi, dtype=np.float64),
                s["lat_true"][sl], s["lon_true"][sl], s["sog_in"][sl],
                _cog_deg(s["cog_sin"][sl], s["cog_cos"][sl]), s["t_sec"][sl]]))
            trip += 1
    if not cols:
        raise RuntimeError("HABIT: no observed segments in corpus")
    import pandas as pd

    arr = np.concatenate(cols, axis=0)
    del cols
    # pandas' C writer, not a per-ping f-string loop and not np.savetxt (both format row-by-row in
    # Python: ~25-27 min on the 10.7M-row standard_process corpus, vs seconds here).
    df = pd.DataFrame({
        "TRIP": arr[:, 0].astype(np.int64), "MMSI": arr[:, 1].astype(np.int64),
        "LAT": arr[:, 2].round(6), "LON": arr[:, 3].round(6), "SPEED": arr[:, 4].round(3),
        "COURSE": arr[:, 5].round(2), "TIMESTAMP": arr[:, 6].round(1)})
    n = len(df)
    del arr
    df.to_csv(path, index=False)
    del df
    return n


# ---------------------------------------------------------------------------------------------- #
#  adapter                                                                                         #
# ---------------------------------------------------------------------------------------------- #
class HabitAdapter(HabitImputer):
    """HabitImputer with an O(N) graph build and the endpoint handling of (2).  Routing unchanged."""

    def prepare_query(self) -> str:
        """Vendor SQL verbatim, except `approx_quantile(x, 0.5)` -> `median(x)`.

        Same statistic (the per-cell median the authors' own comment asks for), computed EXACTLY rather
        than by T-Digest.  This changes memory use only, not the model: DuckDB keeps one T-Digest state
        per group, and the interpolated standard_process corpus has 265,846 cells x 4 quantiles, whose
        combined state exceeds a 64 GB machine (`select max(mlat) from nodes` does not return, with
        RSS > 47 GiB, while the identical query with `median` finishes in seconds).  `our_processed` has
        only 102k cells and fits either way.  Exact median differs from the T-Digest
        approximation by far less than one H3 cell, so HABIT's predictions are unaffected either way.
        """
        return re.sub(r"approx_quantile\(([^,]+),0\.5\)", r"median(\1)", super().prepare_query())

    def data_load(self, path: str, memory_limit: str = "12GB", tmp_dir: str | None = None):
        import duckdb

        self.path = path.replace("\\", "/")
        self.query = self.prepare_query()
        self.G = nx.Graph()

        # Vanilla sets `SET memory_limit='60GB'`.  We keep an explicit limit (a machine-appropriate one)
        # AND give DuckDB a temp directory so the windowed aggregation SPILLS to disk instead of raising
        # `OutOfMemoryException` on the large standard_process corpora.  Without temp_directory a
        # memory_limit turns an out-of-core query into a hard failure.
        duckdb.sql("INSTALL h3 FROM community;LOAD h3;")
        duckdb.sql(f"SET memory_limit='{memory_limit}';")
        tmp = (tmp_dir or os.path.join(os.path.dirname(self.path), "_duckdb_tmp")).replace("\\", "/")
        os.makedirs(tmp, exist_ok=True)
        duckdb.sql(f"SET temp_directory='{tmp}';")
        # Safe: mlon/mlat come from the `nodes` CTE and are therefore CONSTANT across all rows sharing an
        # h3, so the groupby().first() below is order-independent (as is vanilla's filtered.iloc[0]).
        duckdb.sql("SET preserve_insertion_order=false;")   # cuts peak memory
        df = duckdb.sql(self.query).df()
        if len(df) == 0:
            raise RuntimeError("HABIT: empty aggregation -- no cell transitions in corpus")

        df["h3"] = df["h3"].astype("int64")
        df["lag_h3"] = df["lag_h3"].astype("int64")

        # --- (1) O(N) replacement for vanilla's O(|nodes| x |rows|) `df[df.h3 == n]` scan ---
        med = df.groupby("h3")[["mlon", "mlat"]].first()
        med_lon = med["mlon"].to_dict(); med_lat = med["mlat"].to_dict()

        # REPRODUCIBILITY: insert nodes and edges in a canonical (sorted) order.
        # HABIT's weights are traffic counts, so on our corpus nearly every edge has the same weight
        # (0.99 = one transition) and equal-cost shortest paths abound.  `nx.astar_path` breaks those ties
        # by heap push order, which follows networkx's adjacency insertion order -- which would follow DuckDB's
        # row order, which is NOT stable across runs (multi-threaded scan + preserve_insertion_order=false).
        # Without a canonical order the method is deterministic in COST but not in PATH: re-running a
        # cell can move the 12h/24h medians by ~1-2%.  Sorting changes only the insertion order, never
        # the graph's content.
        nodes = sorted(set(df["h3"].tolist()) | set(df["lag_h3"].tolist()))
        for nid in nodes:
            p1 = h3.cell_to_latlng(nid)[::-1]                     # (lon, lat) centroid -- vanilla
            self.G.add_node(nid, location=p1,
                            mlon=med_lon.get(nid, p1[0]), mlat=med_lat.get(nid, p1[1]))

        d = df["distance"]
        keep = df[d.notna() & (d < self.max_gap)]
        ev = keep[["lag_h3", "h3", "transitions", "cog"]].astype("int64").values
        ev = ev[np.lexsort((ev[:, 1], ev[:, 0]))]                 # canonical order by (lag_h3, h3)
        for lag_h3, h3c, w, cog in ev:
            self.G.add_edge(int(lag_h3), int(h3c),
                            weight=(100.0 - min(float(w), 100.0)) / 100.0, cog=int(cog))

        self._node_list = list(self.G.nodes())
        pos = [self.G.nodes[n]["location"] for n in self._node_list]
        self.btree = cKDTree(pos)
        return len(self.G.nodes()), len(self.G.edges())

    # --- (2) endpoint handling: correct NN query per endpoint; true anchors as polyline ends ---
    def _nearest_node(self, lon: float, lat: float, res: int) -> int:
        cell = h3.latlng_to_cell(lat, lon, res)
        if cell in self.G:
            return cell
        _, ix = self.btree.query((lon, lat), 1)
        return self._node_list[int(ix)]

    def route_lonlat(self, o_lon, o_lat, d_lon, d_lat) -> np.ndarray | None:
        """A* route as a (lon,lat) polyline: [true origin] + per-cell MEDIAN coords + [true dest].
        Returns None when the graph has no path (reported, never silently hidden)."""
        if len(self.G) == 0:
            return None
        res = h3.get_resolution(self._node_list[0])
        u = self._nearest_node(o_lon, o_lat, res)
        v = self._nearest_node(d_lon, d_lat, res)
        try:
            cells = nx.astar_path(self.G, u, v)          # vanilla routing (weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
        mid = [(self.G.nodes[c]["mlon"], self.G.nodes[c]["mlat"]) for c in cells]
        pts = [(o_lon, o_lat)] + mid + [(d_lon, d_lat)]
        out = [pts[0]]
        for p in pts[1:]:
            if abs(p[0] - out[-1][0]) > 1e-12 or abs(p[1] - out[-1][1]) > 1e-12:
                out.append(p)
        return np.asarray(out, dtype=float)


# ---------------------------------------------------------------------------------------------- #
#  (3) route -> timestamped positions                                                              #
# ---------------------------------------------------------------------------------------------- #
def _hav_nm(lon1, lat1, lon2, lat2) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_NM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _wrap180(x):
    return ((x + 180.0) % 360.0) - 180.0


def resample_route_to_times(poly: np.ndarray, t_o: float, t_d: float, held_t: np.ndarray):
    """Constant-speed traversal: held time -> elapsed fraction -> that fraction of route arclength.

    Longitude is UNWRAPPED into a continuous frame before interpolating and re-wrapped afterwards.
    Interpolating raw degrees would sweep the long way round whenever a segment straddles +/-180 deg,
    while `_hav_nm` (whose sin(dl/2) is periodic) measures the short way -- an inconsistency that
    fabricates ~2000 NM errors on the Arctic date-line gaps that exist in this dataset.
    """
    lon = np.degrees(np.unwrap(np.radians(poly[:, 0])))
    lat = poly[:, 1]
    seg = _hav_nm(lon[:-1], lat[:-1], lon[1:], lat[1:])
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    denom = max(t_d - t_o, 1e-9)
    f = np.clip((np.asarray(held_t, float) - t_o) / denom, 0.0, 1.0)
    if total <= 1e-12:
        return np.full(len(f), lat[0]), np.full(len(f), _wrap180(lon[0]))
    s = f * total
    j = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1)
    w = np.where(seg[j] > 1e-12, (s - cum[j]) / np.maximum(seg[j], 1e-12), 0.0)
    lat_p = lat[j] + w * (lat[j + 1] - lat[j])
    lon_p = _wrap180(lon[j] + w * (lon[j + 1] - lon[j]))
    return lat_p, lon_p


def gc_route(o_lon, o_lat, d_lon, d_lat, held_t, t_o, t_d):
    """Fallback when HABIT knows no route: the great-circle chord (what it degenerates to)."""
    poly = np.array([[o_lon, o_lat], [d_lon, d_lat]], float)
    return resample_route_to_times(poly, t_o, t_d, held_t)


def route_length_nm(poly: np.ndarray) -> float:
    lon = np.degrees(np.unwrap(np.radians(poly[:, 0])))
    return float(_hav_nm(lon[:-1], poly[:-1, 1], lon[1:], poly[1:, 1]).sum())


def predict_graph(imp: HabitAdapter, g):
    """-> (pred_lat[n_held], pred_lon[n_held], used_fallback: bool, n_route_vertices: int, detour: float).

    `n_route_vertices` is a health diagnostic: a value of 2 means A* found a single edge straight from
    anchor to anchor, i.e. HABIT is reproducing the great-circle chord rather than routing.

    `detour` = route arclength / anchor chord.  HABIT's edge weight measures traffic UNPOPULARITY and
    carries no distance term, so where no historical corridor exists A* still succeeds but returns a
    hugely circuitous connected path (measured: 84x the chord on one Bering-Strait gap).  Those cases are
    NOT counted as A* failures, so the failure rate alone understates how often HABIT lacks a usable
    route; the detour factor is what exposes them.  Held lat/lon are never read."""
    s = to_series(g, crop_ctx=10 ** 9)
    held = s["held"]
    hi = np.where(held)[0]
    b, a = int(hi[0]) - 1, int(hi[-1]) + 1          # bracketing observed anchors (two-sided)
    o_lon, o_lat = float(s["lon_true"][b]), float(s["lat_true"][b])
    d_lon, d_lat = float(s["lon_true"][a]), float(s["lat_true"][a])
    t_o, t_d = float(s["t_sec"][b]), float(s["t_sec"][a])
    ht = s["t_sec"][held]

    poly = imp.route_lonlat(o_lon, o_lat, d_lon, d_lat)
    if poly is None or len(poly) < 2:
        lat, lon = gc_route(o_lon, o_lat, d_lon, d_lat, ht, t_o, t_d)
        return lat, lon, True, 0, 1.0
    lat, lon = resample_route_to_times(poly, t_o, t_d, ht)
    chord = float(_hav_nm(o_lon, o_lat, d_lon, d_lat))
    detour = route_length_nm(poly) / max(chord, 1e-9)
    return lat, lon, False, len(poly), detour

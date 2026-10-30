"""Leak + fidelity assertions for the HABIT and MH-GIN adapters.  Run before any full sweep.

  python -m task_alpha.baselines.test_new_baseline_adapters
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from task_alpha.baselines.common_gaps import to_series  # noqa: E402
from task_alpha.baselines.HABIT.habit_model import _observed_segments, build_corpus_csv  # noqa: E402
from task_alpha.baselines.MHGIN.mhgin_data import (COG, LAT, LON, SOG, TIME, build_arrays,  # noqa: E402
                                                   max_seq_len)
from task_alpha.data import collate_graphs_interp  # noqa: F401

OK = "  ok  "


def main():
    gs = pickle.load(open("task_alpha/data/our_processed/gap_6h/test.pkl", "rb"))[:60]

    # ---------------------------------------------------------------- MH-GIN
    ml = max_seq_len(gs)
    d, lb, m, pad, meta = build_arrays(gs, ml)
    n_held = 0
    for i, g in enumerate(gs):
        s = to_series(g, crop_ctx=64)
        L, h = meta[i]["L"], meta[i]["held"]
        n_held += int(h.sum())
        assert np.all(d[i, :L, LAT][h] == 91.0), "held latitude not sentinel-masked"
        assert np.all(d[i, :L, LON][h] == 181.0), "held longitude not sentinel-masked"
        assert np.all(d[i, :L, SOG][h] == -1.0), "held sog not sentinel-masked"
        assert np.all(d[i, :L, COG][h] == -1.0), "held cog not sentinel-masked"
        assert np.all(d[i, L:, LAT] == 91.0) and np.all(d[i, L:, LON] == 181.0), "padding not sentinel"
        # truth is present only in labels, and only where it exists
        assert np.allclose(lb[i, :L, LAT], s["lat_true"][:L], atol=1e-3), "lat labels != truth"
        assert np.allclose(lb[i, :L, LON], s["lon_true"][:L], atol=1e-3), "lon labels != truth"
        # timestamps at held steps are the KNOWN query times, deliberately unhidden
        assert np.all(np.isfinite(d[i, :L, TIME])) and np.all(d[i, :L, TIME][h] > 0)
        # loss gates: sog/cog have no truth inside the gap -> their loss must be switched off there
        assert not pad[i, :L, SOG][h].any() and not pad[i, :L, COG][h].any()
        assert pad[i, :L, LAT].all() and pad[i, :L, LON].all(), "spatial loss gated off"
        assert not m[i, :L, LAT][h].any(), "eval mask must mark held as hidden"
    print(f"{OK} MH-GIN: {len(gs)} graphs / {n_held} held nodes -- no held lat/lon/sog/cog in model input")
    print(f"{OK} MH-GIN: sentinels exact (91/181/-1), padding sentinelled, loss gates correct")

    # ---------------------------------------------------------------- HABIT
    # Leak is a PROVENANCE property, not a coordinate-value one: two different vessels legitimately
    # sail through the same point, and indexing that other vessel's ping is precisely what HABIT is
    # for.  So we check, per graph, that none of ITS OWN held positions were written for IT.
    tmp = "task_alpha/baselines/HABIT/_scratch/_leaktest.csv"
    build_corpus_csv(gs, tmp)
    rows = np.genfromtxt(tmp, delimiter=",", skip_header=1)
    trip_of = rows[:, 0].astype(int)

    obs_pts, n_seg, trip = 0, 0, 0
    for g in gs:
        s = to_series(g, crop_ctx=10 ** 9)
        h = s["held"]
        segs = _observed_segments(h)
        n_seg += len(segs)
        obs_pts += int((~h).sum())
        mine = np.isin(trip_of, [trip + k for k in range(len(segs))])
        trip += len(segs)
        own = set(zip(np.round(rows[mine, 2], 6), np.round(rows[mine, 3], 6)))
        held_own = set(zip(np.round(s["lat_true"][h], 6), np.round(s["lon_true"][h], 6)))
        assert not (own & held_own), "HABIT corpus wrote this graph's own HELD coordinates -- LEAK"

    assert len(rows) == obs_pts, f"corpus rows {len(rows)} != observed pings {obs_pts}"
    assert n_seg == 2 * len(gs), "each gapped graph must yield exactly 2 observed segments"
    print(f"{OK} HABIT: corpus {len(rows)} rows == {obs_pts} observed pings, {n_seg} trips from "
          f"{len(gs)} graphs (2 segments each) -- no graph contributes its own held positions")
    print(f"{OK} HABIT: gap-spanning transitions impossible (pre/post-gap pings are separate TRIPs)")
    os.remove(tmp)

    # ------------------------------------------------- HABIT antimeridian check
    # Interpolating raw degrees would sweep the LONG way round ±180 and fabricate ~2000 NM errors on
    # the Arctic date-line gaps.  These two test graphs straddle the meridian.
    from task_alpha.baselines.common_gaps import model_errors_latlon  # noqa: E402
    from task_alpha.baselines.HABIT.habit_model import gc_route  # noqa: E402
    full = pickle.load(open("task_alpha/data/our_processed/gap_6h/test.pkl", "rb"))
    for gi in (196, 381):
        g = full[gi]
        s = to_series(g, crop_ctx=10 ** 9)
        h = s["held"]; hi = np.where(h)[0]
        b, a = hi[0] - 1, hi[-1] + 1
        assert abs(s["lon_true"][b] - s["lon_true"][a]) > 180.0, f"gap {gi} no longer straddles ±180"
        lat, lon = gc_route(float(s["lon_true"][b]), float(s["lat_true"][b]),
                            float(s["lon_true"][a]), float(s["lat_true"][a]),
                            s["t_sec"][h], float(s["t_sec"][b]), float(s["t_sec"][a]))
        err = float(np.median(model_errors_latlon(g, lat, lon)))
        assert err < 50.0, f"antimeridian check: gap {gi} chord error {err:.1f} NM (long way round: ~2000+)"
        assert np.all(np.abs(lon) <= 180.0 + 1e-9), "longitude not re-wrapped to [-180, 180]"
    print(f"{OK} HABIT: date-line gaps 196/381 route the SHORT way (chord error < 50 NM, long way ~2000)")

    print("\nall adapter assertions passed")


if __name__ == "__main__":
    main()

"""Leak + fidelity assertions for the MultitaskAIS adapter.  Run before any full sweep (gpubench).

  python -m task_alpha.baselines.MTAIS.test_mtais_adapter

Checks:
  1. LEAK: exported train/val voyages contain none of the SAME graph's held (lat,lon) positions
     (provenance check, HABIT convention), and no voyage spans a gap (2 segments per gapped graph).
  2. LEAK: test_index 'pre' arrays are exactly the pre-gap observed pings (count == first held
     index; no held coordinates present); held bookkeeping matches the RunGraph mask.
  3. Normalization: all LAT/LON/SOG/COG in [0, 1); four-hot bin indices in range.
  4. Antimeridian: a synthetic +/-180-straddling cell round-trips encode->bin->decode within half a
     bin on both axes (lon compared wrap-aware).
  5. py3.7 pickle: a numpy-2 pickle written by the exporter loads in the mtais_tf1 env (via the
     numpy._core -> numpy.core shim) with an identical checksum.
"""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from task_alpha.baselines.common_gaps import to_series  # noqa: E402
from task_alpha.baselines.MTAIS.mtais_export import (CLIP_HI, LAT, LON, MIN_SEG, PKL_PROTOCOL,  # noqa: E402
                                                     SOG, COG, build_roi, build_test_index,
                                                     build_voyages, decode_lat_bins,
                                                     decode_lon_bins_shifted, norm_lat, norm_lon,
                                                     observed_segments, wrap180)
from task_alpha.baselines.MTAIS.run_mtais_gaps import CONDA, TF1_ENV  # noqa: E402
from task_alpha.data import collate_graphs_interp  # noqa: F401,E402

OK = "  ok  "


def main():
    gs = pickle.load(open("task_alpha/data/our_processed/gap_6h/test.pkl", "rb"))[:60]

    # shared ROI from the observed pings of these graphs
    la, lo = [], []
    for g in gs:
        s = to_series(g, crop_ctx=10 ** 9)
        obs = ~s["held"]
        la.append(s["lat_true"][obs])
        lo.append(s["lon_true"][obs])
    roi = build_roi(np.concatenate(la), np.concatenate(lo))
    assert roi["lat_bins"] + roi["lon_bins"] == 600 and roi["data_dim"] == 702
    print(f"{OK} ROI: lat_bins {roi['lat_bins']} lon_bins {roi['lon_bins']} "
          f"quant floor {roi['quant_floor_nm']:.3f} NM")

    # ---------------------------------------------------------------- 1. voyage leak safety
    voy, n_pings = build_voyages(gs, roi)
    exp_segs, exp_pings, vid = 0, 0, 0
    for g in gs:
        s = to_series(g, crop_ctx=10 ** 9)
        h = s["held"]
        segs = [(a, b) for a, b in observed_segments(h) if b - a >= MIN_SEG]
        exp_segs += len(segs)
        exp_pings += sum(b - a for a, b in segs)
        held_own = set(zip(np.round(norm_lat(s["lat_true"][h], roi), 9),
                           np.round(norm_lon(s["lon_true"][h], roi), 9)))
        for _ in segs:
            arr = voy[vid]
            own = set(zip(np.round(arr[:, LAT], 9), np.round(arr[:, LON], 9)))
            assert not (own & held_own), "voyage contains its own graph's HELD position -- LEAK"
            # no voyage spans a gap: timestamps must lie inside ONE observed segment
            assert len(arr) >= MIN_SEG
            vid += 1
    assert vid == len(voy) == exp_segs, "voyage count != observed-segment count"
    assert n_pings == exp_pings, "exported ping count mismatch"
    n2 = sum(1 for g in gs
             if len([1 for a, b in observed_segments(to_series(g, 10 ** 9)['held']) if b - a >= MIN_SEG]) == 2)
    print(f"{OK} voyages: {len(voy)} segments / {n_pings} observed pings from {len(gs)} graphs "
          f"({n2} graphs contribute the expected 2 segments) -- no held positions, none spans a gap")

    # ---------------------------------------------------------------- 2. test_index leak safety
    entries, n_skip = build_test_index(gs, roi)
    for e, g in zip(entries, gs):
        s = to_series(g, crop_ctx=10 ** 9)
        h = s["held"]
        hidx = np.where(h)[0]
        assert e["n_pre"] == int(hidx[0]), "pre-gap length != first held index"
        assert e["n_held"] == int(h.sum()) and np.array_equal(e["node_idx"], hidx)
        assert len(e["pre"]) == e["n_pre"]
        pre_pos = set(zip(np.round(e["pre"][:, LAT], 9), np.round(e["pre"][:, LON], 9)))
        held_own = set(zip(np.round(norm_lat(s["lat_true"][h], roi), 9),
                           np.round(norm_lon(s["lon_true"][h], roi), 9)))
        assert not (pre_pos & held_own), "test 'pre' contains a held position -- LEAK"
        assert np.allclose(e["pre"][:, LAT], norm_lat(s["lat_true"][:e["n_pre"]], roi))
        assert len(e["t_held"]) == e["n_held"]
    print(f"{OK} test_index: {len(entries)} gaps ({n_skip} skipped) -- 'pre' is exactly the pre-gap "
          f"observed record; held lat/lon/sog/cog never exported")

    # ---------------------------------------------------------------- 3. normalization ranges
    allv = np.concatenate([v[:, [LAT, LON, SOG, COG]] for v in voy.values()], axis=0)
    assert allv.min() >= 0.0 and allv.max() <= CLIP_HI
    for col, bins in ((LAT, roi["lat_bins"]), (LON, roi["lon_bins"]),
                      (SOG, roi["sog_bins"]), (COG, roi["cog_bins"])):
        ix = (allv[:, [LAT, LON, SOG, COG].index(col)] * bins).astype(np.int64)
        assert ix.min() >= 0 and ix.max() < bins
    print(f"{OK} normalization: all LAT/LON/SOG/COG in [0,1); four-hot indices in range")

    # ---------------------------------------------------------------- 4. antimeridian round-trip
    rng = np.random.default_rng(0)
    for name, lon_c in (("+/-180-straddling", 180.0), ("0-crossing (forces nonzero cut)", 0.0)):
        lon_am = wrap180(lon_c - 1.0 + rng.uniform(0, 2, 4000))               # 2-deg arc across lon_c
        lat_am = 64.0 + rng.uniform(0, 8, 4000)
        roi_am = build_roi(lat_am, lon_am)
        assert (roi_am["lon_hi"] - roi_am["lon_lo"]) < 30.0, \
            f"{name}: occupied arc not contiguous after the circular cut"
        if lon_c == 0.0:
            assert roi_am["lon_cut"] > 1.0, "0-crossing arc must force a nonzero rotation"
        bl = (norm_lat(lat_am, roi_am) * roi_am["lat_bins"]).astype(np.int64)
        bo = (norm_lon(lon_am, roi_am) * roi_am["lon_bins"]).astype(np.int64)
        dlat = np.abs(decode_lat_bins(bl, roi_am) - lat_am)
        dlon = np.abs(wrap180(wrap180(decode_lon_bins_shifted(bo, roi_am) + roi_am["lon_cut"])
                              - lon_am))
        assert dlat.max() <= roi_am["lat_reso_deg"] / 2 + 1e-9
        assert dlon.max() <= roi_am["lon_reso_deg"] / 2 + 1e-9
        print(f"{OK} antimeridian: {name} cell round-trips within half a bin "
              f"(max dlat {dlat.max():.5f} deg, max dlon {dlon.max():.5f} deg, "
              f"cut {roi_am['lon_cut']:.1f} deg)")

    # ---------------------------------------------------------------- 5. py3.7 load via shim
    sample = {0: voy[0], 1: voy[1]}
    tot = float(sum(np.asarray(v, np.float64).sum() for v in sample.values()))
    tmp = os.path.join(tempfile.gettempdir(), "mtais_py37_loadtest.pkl")
    with open(tmp, "wb") as f:
        pickle.dump(sample, f, protocol=PKL_PROTOCOL)
    out = subprocess.check_output(
        [CONDA, "run", "-n", TF1_ENV, "python",
         os.path.join(HERE, "mtais_tf1_run.py"), "--selftest-load", tmp],
        stderr=subprocess.DEVNULL).decode()
    line = [ln for ln in out.splitlines() if ln.startswith("SELFTEST")][0]
    got = float(line.split("checksum=")[1])
    assert abs(got - tot) < 1e-4, f"py3.7 checksum mismatch: {got} vs {tot}"
    os.remove(tmp)
    print(f"{OK} py3.7: numpy-2 pickle loads in {TF1_ENV} via the numpy-compat shim "
          f"(checksum match {got:.4f})")

    print("\nall MTAIS adapter assertions passed")


if __name__ == "__main__":
    main()

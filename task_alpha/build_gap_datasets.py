"""
task_alpha.build_gap_datasets  --  organized dataset folders for the FQGNN gap-length sweep.

TWO preprocessing tracks (each leaf: train.pkl / val.pkl / test.pkl = pickled lists of RunGraph,
plus manifest.json):

  task_alpha/data/
    our_processed/     gap_3h/ gap_6h/ gap_12h/ gap_24h/    (raw-ping pipeline, data_raw)
    standard_process/  gap_2h/ gap_6h/ gap_12h/ gap_24h/    (interp10m product, data_standard)

our_processed:
  * gap_6h  = the paper dataset -- common_gaps.load_gaps() (cache hit; 5350/635/774).
  * others  = build_dataset_raw with the SAME config except gap_minutes (cache-warmed once via
    build_multi_horizon so each raw parquet partition is read ONCE for all other horizons), then the
    exact load_gaps filter (mask.sum()>0, _split(mmsi) not None) + vessel-disjoint split.
standard_process:
  * data_standard.load_standard (interp10m, dedup+sort ONLY) -> runs (coverage gap > 3 h,
    >=30 pts, median sog >= 1 kn) -> data.make_run_graph UNCHANGED (max_gap_len=500, n_anchor=8,
    min_span_factor=1.5, stable per-run seed) -> _split(mmsi).

IDEMPOTENT: a leaf folder whose train.pkl AND manifest.json already exist is skipped -- rerunning
the script IS the resume.  Sources are READ-ONLY throughout.

Run from the repository root:  python -m task_alpha.build_gap_datasets
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import torch

from task_alpha.baselines.common_gaps import BUCKET_EDGES, DATASET_KW, load_gaps
from task_alpha.data import RunGraph  # noqa: F401  (unpickling target)
from task_alpha.data_raw import build_dataset_raw, build_multi_horizon
from task_alpha.data_standard import (available_years, build_runs_standard, graphs_standard,
                                      load_standard)
from task_alpha.experiment_compare import _sinuosity
from task_alpha.metrics import gcd_errors
from task_alpha.train_stream import _split

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(ROOT, "data")
YEARS, MONTHS = (2016, 2017, 2018), (7, 8)
CMD = "python -m task_alpha.build_gap_datasets"

OUR_GAPS = {"gap_3h": 180, "gap_6h": 360, "gap_12h": 720, "gap_24h": 1440}
STD_GAPS = {"gap_2h": 120, "gap_6h": 360, "gap_12h": 720, "gap_24h": 1440}

STD_RUN_CFG = dict(coverage_gap_hours=3.0, min_len=30, min_sog=1.0)
STD_GRAPH_CFG = dict(max_gap_len=500, n_anchor=8, min_span_factor=1.5)
# De-circularization: thin the 10-min grid so consecutive truth points
# straddle REAL curvature (>= 2-6x the ~10-min raw ping cadence), not the linear-interp fill.
# gap 2h -> every 2nd point (20 min, ~6 held), 6h -> every 3rd (30 min, ~12 held),
# 12h/24h -> every 6th (60 min, ~12/~24 held).  Anchors and held targets share the coarse cadence.
STD_THIN_K = {"gap_2h": 2, "gap_6h": 3, "gap_12h": 6, "gap_24h": 6}


# ------------------------------------------------------------------------------------------------ #
#  helpers                                                                                           #
# ------------------------------------------------------------------------------------------------ #
def leaf_dir(track: str, leaf: str) -> str:
    return os.path.join(OUT_ROOT, track, leaf)


def leaf_done(track: str, leaf: str) -> bool:
    d = leaf_dir(track, leaf)
    return os.path.exists(os.path.join(d, "train.pkl")) and \
        os.path.exists(os.path.join(d, "manifest.json"))


def split_graphs(graphs):
    """EXACT load_gaps filter + vessel-disjoint split (train_stream._split, md5-based)."""
    keep = [g for g in graphs if int(g.mask.sum()) > 0 and _split(g.mmsi) is not None]
    return ([g for g in keep if _split(g.mmsi) == "train"],
            [g for g in keep if _split(g.mmsi) == "val"],
            [g for g in keep if _split(g.mmsi) == "test"])


def gcd_median_test(te) -> float:
    """GCD reference on test: median over gaps of the per-gap median GCD error (NM)."""
    med = [float(np.median(gcd_errors(g))) for g in te]
    return float(np.median(med)) if med else float("nan")


def sinuosity_buckets(te) -> dict:
    """Test-gap counts per sinuosity bucket (finite only) + count of non-finite sinuosities."""
    sn = np.array([_sinuosity(g) for g in te], dtype=float)
    fin = sn[np.isfinite(sn)]
    out = {}
    for lo, hi in zip(BUCKET_EDGES[:-1], BUCKET_EDGES[1:]):
        lab = f"{lo:.2f}-{hi:.2f}" if hi < 1e9 else f">{lo:.2f}"
        out[lab] = int(((fin >= lo) & (fin < hi)).sum())
    out["nonfinite"] = int(len(sn) - len(fin))
    return out


def save_leaf(track: str, leaf: str, gap_minutes: int, tr, va, te, builder: dict, notes: dict):
    d = leaf_dir(track, leaf)
    os.makedirs(d, exist_ok=True)
    for name, gl in (("train", tr), ("val", va), ("test", te)):
        with open(os.path.join(d, f"{name}.pkl"), "wb") as f:
            pickle.dump(gl, f, protocol=4)
    manifest = {
        "track": track, "gap_minutes": gap_minutes,
        "years": list(YEARS), "months": list(MONTHS),
        "builder": builder,
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "n_vessels": {"train": len({g.mmsi for g in tr}), "val": len({g.mmsi for g in va}),
                      "test": len({g.mmsi for g in te})},
        "gcd_median_nm_test": round(gcd_median_test(te), 4),
        "sinuosity_buckets_test": sinuosity_buckets(te),
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "cmd": CMD,
        **notes,
    }
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[saved] {track}/{leaf}: {len(tr)}/{len(va)}/{len(te)} "
          f"gcd_med={manifest['gcd_median_nm_test']}", flush=True)


# ------------------------------------------------------------------------------------------------ #
#  track 1: our_processed (raw-ping pipeline)                                                        #
# ------------------------------------------------------------------------------------------------ #
def build_our_track():
    pending = {leaf: gm for leaf, gm in OUR_GAPS.items() if not leaf_done("our_processed", leaf)}
    if not pending:
        print("[our_processed] all leaves present -- skipping", flush=True)
        return
    # cache-warm the other horizons (everything except the already-cached 360) in ONE parquet pass
    new_gaps = tuple(gm for leaf, gm in pending.items() if leaf != "gap_6h")
    if new_gaps:
        print(f"[our_processed] warming cache for gaps {new_gaps} (one raw-parquet pass)", flush=True)
        build_multi_horizon(YEARS, MONTHS, gaps=new_gaps, n_mmsi=None, max_files_per_part=None,
                            thin_sec=180, min_lat=60.0, seed=0, clean_only=True, run_filters=None,
                            verbose=True, max_gap_len=500, n_anchor=8, min_span_factor=1.5,
                            k_temporal=6)
    for leaf, gm in pending.items():
        if leaf == "gap_6h":
            print("[our_processed/gap_6h] materializing the EXISTING dataset via load_gaps()", flush=True)
            tr, va, te = load_gaps()                      # cache hit; do NOT rebuild
            builder = dict(DATASET_KW, fn="common_gaps.load_gaps -> build_dataset_raw")
        else:
            kw = dict(DATASET_KW, gap_minutes=gm)
            print(f"[our_processed/{leaf}] build_dataset_raw gap_minutes={gm} (cached parts)", flush=True)
            tr, va, te = split_graphs(build_dataset_raw(**kw))
            builder = dict(kw, fn="data_raw.build_dataset_raw")
        save_leaf("our_processed", leaf, gm, tr, va, te, builder,
                  notes={"source": "raw message-level parquet (m123/m18/m27), clean whitelist",
                         "split": "train_stream._split md5 vessel-disjoint 80/10/10"})
        del tr, va, te


# ------------------------------------------------------------------------------------------------ #
#  track 2: standard_process (interp10m product)                                                     #
# ------------------------------------------------------------------------------------------------ #
def build_std_track():
    pending = {leaf: gm for leaf, gm in STD_GAPS.items() if not leaf_done("standard_process", leaf)}
    if not pending:
        print("[standard_process] all leaves present -- skipping", flush=True)
        return
    years_ok = available_years(YEARS)
    missing_years = [y for y in YEARS if y not in years_ok]
    if missing_years:
        print(f"[standard_process] WARNING: interp10m missing for years {missing_years} -- "
              f"using {years_ok}", flush=True)
    acc = {leaf: [] for leaf in pending}
    run_stats = {"n_runs": 0, "median_dt_s": [], "run_len_pts": []}
    for y in years_ok:                                   # per-year to bound memory; a >3 h coverage
        print(f"[standard_process] loading interp10m {y} months={MONTHS}", flush=True)
        df = load_standard(years=(y,), months=MONTHS, verbose=True)   # gap always splits years
        runs = build_runs_standard(df, **STD_RUN_CFG)
        del df
        for r in runs:
            t = r["time"].to_numpy().astype("datetime64[s]").astype("int64")
            run_stats["median_dt_s"].append(float(np.median(np.diff(t))))
            run_stats["run_len_pts"].append(len(r))
        run_stats["n_runs"] += len(runs)
        print(f"  {y}: {len(runs)} moving runs", flush=True)
        for leaf, gm in pending.items():
            gs = graphs_standard(runs, gap_minutes=gm, thin_k=STD_THIN_K[leaf], **STD_GRAPH_CFG)
            acc[leaf].extend(gs)
            print(f"    {leaf} (G={gm}m, thin_k={STD_THIN_K[leaf]}): +{len(gs)} graphs "
                  f"(cum {len(acc[leaf])})", flush=True)
        del runs
    stats_note = {
        "n_runs_total": run_stats["n_runs"],
        "run_dt_median_s": float(np.median(run_stats["median_dt_s"])) if run_stats["median_dt_s"] else None,
        "run_len_pts_median": float(np.median(run_stats["run_len_pts"])) if run_stats["run_len_pts"] else None,
        "years_used": years_ok, "years_missing": missing_years,
    }
    for leaf, gm in pending.items():
        tr, va, te = split_graphs(acc.pop(leaf))
        builder = dict(fn="data_standard.load_standard -> build_runs_standard -> data.make_run_graph",
                       months=list(MONTHS), min_lat=60.0, dedup="(mmsi,time)", **STD_RUN_CFG,
                       gap_minutes=gm, **STD_GRAPH_CFG, k_temporal=6, min_anchor=6,
                       thin_k=STD_THIN_K[leaf], effective_cadence_min=10 * STD_THIN_K[leaf],
                       thin_reason="de-circularization: truth points must straddle real curvature, "
                                   "not the 10-min linear-interp fill",
                       seed="md5(mmsi:run_start_epoch_s) % 2^31 (stable per run)")
        save_leaf("standard_process", leaf, gm, tr, va, te, builder,
                  notes={"source": "SAIS_<Y>_CLASS_A_B_interptime10m.parquet (10-min product; "
                                   "no re-cleaning beyond dedup+sort by design)",
                         "split": "train_stream._split md5 vessel-disjoint 80/10/10",
                         "run_stats": stats_note})
        del tr, va, te


# ------------------------------------------------------------------------------------------------ #
#  verification                                                                                      #
# ------------------------------------------------------------------------------------------------ #
def _load(track, leaf, name):
    with open(os.path.join(leaf_dir(track, leaf), f"{name}.pkl"), "rb") as f:
        return pickle.load(f)


def _recon_dt_median(g) -> float:
    """Median inter-node dt (s) reconstructed from coord t_norm + realized gap duration."""
    coord = g.coord.numpy(); held = g.mask.numpy().astype(bool)
    hi = np.where(held)[0]
    span = max(float(coord[hi[-1], 0] - coord[hi[0], 0]), 1e-12)
    dur = g.gap_minutes * 60.0 / span
    return float(np.median(np.diff(coord[:, 0] * dur)))


def verify():
    print("\n" + "=" * 100 + "\nVERIFICATION\n" + "=" * 100, flush=True)
    ok_all = True
    rows, trend = [], {}
    rng = np.random.default_rng(0)
    for track, gaps in (("our_processed", OUR_GAPS), ("standard_process", STD_GAPS)):
        trend[track] = []
        for leaf, gm in gaps.items():
            if not leaf_done(track, leaf):
                print(f"  MISSING {track}/{leaf}"); ok_all = False
                continue
            tr, va, te = (_load(track, leaf, n) for n in ("train", "val", "test"))
            vs = [{g.mmsi for g in s} for s in (tr, va, te)]
            overlap = len(vs[0] & vs[1]) + len(vs[0] & vs[2]) + len(vs[1] & vs[2])
            heldok = all(int(g.mask.sum()) > 0 for g in te)
            allg = tr + va + te
            samp = [allg[i] for i in rng.choice(len(allg), size=min(20, len(allg)), replace=False)]
            nan_free = all(not (torch.isnan(g.x).any() or torch.isnan(g.coord).any()
                                or torch.isnan(g.true_lonlat).any()) for g in samp)
            gmed = gcd_median_test(te)
            bk = sinuosity_buckets(te)
            ok = overlap == 0 and heldok and nan_free
            ok_all &= ok
            trend[track].append((gm, gmed))
            rows.append(dict(track=track, leaf=leaf, gm=gm, n_tr=len(tr), n_va=len(va),
                             n_te=len(te), n_te_vessels=len(vs[2]), gcd_med=round(gmed, 3),
                             buckets=bk, overlap=overlap, heldok=heldok, nan_free=nan_free))
            # standard-track cadence + held-window-span sanity
            if track == "standard_process" and te:
                dts = [_recon_dt_median(g) for g in
                       (te[i] for i in rng.choice(len(te), size=min(10, len(te)), replace=False))]
                med_dt = float(np.median(dts))
                span_min = float(te[0].gap_minutes)
                cad_ok = abs(med_dt - 600.0) < 5.0
                span_ok = (gm < 240) or (span_min >= 120.0)     # a sampled >=6h gap spans >=2 h
                ok_all &= cad_ok and span_ok
                print(f"  [std sanity] {leaf}: median dt {med_dt:.1f}s (~600 required: "
                      f"{'PASS' if cad_ok else 'FAIL'}); sampled held window {span_min:.0f} min "
                      f"({'PASS' if span_ok else 'FAIL'})", flush=True)
            del tr, va, te, allg

    # 1. the 6h oracle
    r6 = next((r for r in rows if r["track"] == "our_processed" and r["leaf"] == "gap_6h"), None)
    if r6:
        c_ok = (r6["n_tr"], r6["n_va"], r6["n_te"]) == (5350, 635, 774)
        g_ok = abs(r6["gcd_med"] - 2.857) < 1e-3
        ok_all &= c_ok and g_ok
        print(f"\n[1] our/gap_6h counts {r6['n_tr']}/{r6['n_va']}/{r6['n_te']} "
              f"(want 5350/635/774): {'PASS' if c_ok else 'FAIL'}; "
              f"gcd_med {r6['gcd_med']} (want 2.857): {'PASS' if g_ok else 'FAIL'}", flush=True)
    else:
        ok_all = False; print("\n[1] our/gap_6h MISSING: FAIL")

    print("[2] per-dataset split-overlap / test-mask / NaN checks: "
          + ("PASS (see table)" if all(r["overlap"] == 0 and r["heldok"] and r["nan_free"]
                                       for r in rows) else "FAIL (see table)"), flush=True)
    # 3. gcd trend
    for track, tl in trend.items():
        tl = sorted(tl)
        mono = all(b[1] > a[1] for a, b in zip(tl, tl[1:]))
        ok_all &= mono
        print(f"[3] {track} gcd_med trend: "
              + " -> ".join(f"{gm/60:.0f}h:{v:.3f}" for gm, v in tl)
              + f"  monotone-increasing: {'PASS' if mono else 'FAIL'}", flush=True)
    # 5. pickle round-trip
    try:
        te = _load("our_processed", "gap_6h", "test")
        g = te[0]
        rt = (isinstance(te, list) and type(g).__name__ == "RunGraph"
              and isinstance(g.x, torch.Tensor) and isinstance(g.mask, torch.Tensor)
              and isinstance(g.ref, tuple) and isinstance(g.mmsi, int))
        del te
    except Exception as e:                                # noqa: BLE001
        rt = False; print("    round-trip error:", e)
    ok_all &= rt
    print(f"[5] pickle round-trip (types intact): {'PASS' if rt else 'FAIL'}", flush=True)

    # final table
    print("\n" + "-" * 118)
    print(f"{'track':<17}{'gap':<9}{'n_train':>8}{'n_val':>7}{'n_test':>8}{'te_vess':>9}"
          f"{'gcd_med':>9}  {'sinuosity buckets [0,.02,.05,.10,.25,inf] + nonfin':<45}")
    print("-" * 118)
    for r in rows:
        b = r["buckets"]
        bs = "/".join(str(b[k]) for k in list(b)[:-1]) + f" (nf {b['nonfinite']})"
        print(f"{r['track']:<17}{r['leaf']:<9}{r['n_tr']:>8}{r['n_va']:>7}{r['n_te']:>8}"
              f"{r['n_te_vessels']:>9}{r['gcd_med']:>9}  {bs:<45}")
    print("-" * 118)
    # disk usage
    total = 0
    for track in ("our_processed", "standard_process"):
        for leaf in os.listdir(os.path.join(OUT_ROOT, track)) if os.path.isdir(
                os.path.join(OUT_ROOT, track)) else []:
            d = leaf_dir(track, leaf)
            sz = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))
            total += sz
            print(f"  {track}/{leaf}: {sz / 1e6:,.1f} MB")
    print(f"  TOTAL task_alpha/data: {total / 1e9:.2f} GB")
    print(f"\nVERIFY OVERALL: {'ALL PASS' if ok_all else 'FAILURES PRESENT'}", flush=True)
    return ok_all


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    if only in ("all", "our"):
        build_our_track()
    if only in ("all", "std"):
        build_std_track()
    if only in ("all", "verify"):
        verify()

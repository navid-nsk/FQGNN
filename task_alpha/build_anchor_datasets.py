"""
task_alpha.build_anchor_datasets  --  ANCHOR-SPARSITY datasets (Fig 6b).

The paper dataset (our_processed/gap_6h) uses n_anchor=8 = 4 anchors PER SIDE of the gap.  This
script builds the two sparsity variants on the SAME our-track 6h config -- the ONLY changed knob
is n_anchor (aps = anchors per side = n_anchor // 2):

  task_alpha/data/our_processed/
    gap_6h_aps2/   n_anchor = 4    (2 anchors per side)
    gap_6h_aps8/   n_anchor = 16   (8 anchors per side)

SAME-GAPS GUARANTEE (verified, not assumed): make_run_graph's gap placement is a single RNG draw
(si = rng.integers(...)) plus filters, ALL independent of n_anchor; n_anchor only enters the
anchor_edge_index block.  With the identical seed pipeline (pseed = seed + y*100 + mo,
make_run_graph seed = pseed + i), the variant graphs are IDENTICAL to the paper dataset
(same 5350/635/774 graphs, same mmsi/gap_minutes/mask/x/coord/truth) EXCEPT anchor_edge_index.
verify() checks exactly that, plus the gcd_median == 2.857 oracle on test (OVERALL semantics:
median over the 770 finite-sinuosity test gaps of the per-gap median GCD error).

CACHE: build_dataset_raw caches per (config-hash, year, month); a new n_anchor = a new key = a
fresh raw-parquet pass.  To read each parquet partition ONCE for BOTH variants, warm_cache_anchor
mirrors data_raw.build_multi_horizon but iterates n_anchor instead of gap horizons (identical
keys/seeds, so the later build_dataset_raw calls are pure cache hits).  Per-partition caching =>
RESUMABLE.

IDEMPOTENT: a leaf whose train.pkl AND manifest.json exist is skipped -- rerunning IS the resume.
Sources (raw parquets, task_alpha/{data.py,data_raw.py}) are READ-ONLY throughout.

Run from the repository root (gpubench env):
  python -m task_alpha.build_anchor_datasets            # build + verify
  python -m task_alpha.build_anchor_datasets build      # build only
  python -m task_alpha.build_anchor_datasets verify     # verify only
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import torch

from task_alpha.baselines.common_gaps import DATASET_KW
from task_alpha.build_gap_datasets import (YEARS, MONTHS, leaf_dir, leaf_done, save_leaf,
                                           split_graphs)
from task_alpha.data import RunGraph  # noqa: F401  (unpickling target)
from task_alpha.data_raw import (_cache_key, build_dataset_raw, build_runs_raw, load_raw,
                                 make_run_graph)
from task_alpha.experiment_compare import _sinuosity
from task_alpha.metrics import gcd_errors

CMD = "python -m task_alpha.build_anchor_datasets"
TRACK = "our_processed"
REF_LEAF = "gap_6h"                                   # the paper dataset (aps4, n_anchor=8)
APS_N = {2: 4, 8: 16}                                 # aps -> n_anchor (aps4/n_anchor=8 exists)
GAP_MINUTES = 360
ORACLE_GCD, ORACLE_N = 2.857, 770                     # OVERALL row of the paper table (finite sinuosity)
# the shared raw-graph cache dir -- the same location build_dataset_raw resolves (single source)
from task_alpha.data_raw import CACHE_DIR  # noqa: E402


def aps_leaf(aps: int) -> str:
    return f"gap_6h_aps{aps}"


def dataset_kw(n_anchor: int) -> dict:
    """The paper config (common_gaps.DATASET_KW) with ONLY n_anchor changed."""
    return dict(DATASET_KW, n_anchor=n_anchor)


# ------------------------------------------------------------------------------------------------ #
#  cache warm: ONE parquet pass per (year, month) for ALL pending n_anchor variants                  #
#  (mirrors data_raw.build_multi_horizon exactly, but iterates n_anchor at fixed gap_minutes=360)    #
# ------------------------------------------------------------------------------------------------ #
def _variant_key(n_anchor: int) -> str:
    """EXACT _cache_key of build_dataset_raw(**dataset_kw(n_anchor)): non-named kwargs -> mk."""
    kw = dataset_kw(n_anchor)
    mk = dict(max_gap_len=kw["max_gap_len"], n_anchor=kw["n_anchor"],
              min_span_factor=kw["min_span_factor"])
    return _cache_key(kw["n_mmsi"], kw["gap_minutes"], 6, 180, 60.0, kw["max_files_per_part"],
                      kw["seed"], None, mk, clean_only=kw["clean_only"])


def warm_cache_anchor(n_anchors: tuple[int, ...]):
    keys = {na: _variant_key(na) for na in n_anchors}
    os.makedirs(CACHE_DIR, exist_ok=True)
    totals = {na: 0 for na in n_anchors}
    for y in YEARS:
        for mo in MONTHS:
            paths = {na: os.path.join(CACHE_DIR, f"part_{keys[na]}_{y}_{int(mo):02d}.pt")
                     for na in n_anchors}
            if all(os.path.exists(p) for p in paths.values()):
                for na in n_anchors:
                    totals[na] += len(torch.load(paths[na], weights_only=False))
                print(f"  {y}-{mo:02d}: all n_anchor variants cached", flush=True)
                continue
            pseed = DATASET_KW["seed"] + int(y) * 100 + int(mo)     # EXACT build_dataset_raw seed
            try:
                df = load_raw(years=[y], months=[mo], n_mmsi=None, min_lat=60.0,
                              max_files_per_part=None, seed=pseed, clean_only=True)
            except FileNotFoundError:
                for na in n_anchors:
                    if not os.path.exists(paths[na]):
                        torch.save([], paths[na])
                print(f"  {y}-{mo:02d}: (no data)", flush=True)
                continue
            runs = build_runs_raw(df, thin_sec=180)                  # read + runs ONCE per partition
            cnt = {}
            for na in n_anchors:
                if os.path.exists(paths[na]):                        # NEVER overwrite a cache file
                    continue
                part = [g for i, run in enumerate(runs)
                        for g in [make_run_graph(run, gap_minutes=GAP_MINUTES, k_temporal=6,
                                                 seed=pseed + i, max_gap_len=500, n_anchor=na,
                                                 min_span_factor=1.5)]
                        if g is not None and int(g.mask.sum()) > 0]
                torch.save(part, paths[na]); cnt[na] = len(part); totals[na] += len(part)
            print(f"  {y}-{mo:02d}: rows {len(df):>8} runs {len(runs):>4} -> " +
                  " ".join(f"na{na}:{cnt.get(na, 'cached')}" for na in n_anchors), flush=True)
            del df, runs
    print("DONE anchor cache warm: " + " ".join(f"na{na}={totals[na]} (key {keys[na]})"
                                                for na in n_anchors), flush=True)


# ------------------------------------------------------------------------------------------------ #
#  build the leaves                                                                                  #
# ------------------------------------------------------------------------------------------------ #
def _load(leaf: str, name: str):
    with open(os.path.join(leaf_dir(TRACK, leaf), f"{name}.pkl"), "rb") as f:
        return pickle.load(f)


def _quick_align_check(aps: int, tr, va, te) -> bool:
    """Pre-save gate vs the paper dataset: same counts + per-graph mmsi/gap_minutes (order-aligned)."""
    if not leaf_done(TRACK, REF_LEAF):
        print(f"  [gate] reference {TRACK}/{REF_LEAF} missing -- cannot verify same-gaps", flush=True)
        return False
    ok = True
    for name, new in (("train", tr), ("val", va), ("test", te)):
        ref = _load(REF_LEAF, name)
        if len(ref) != len(new):
            print(f"  [gate aps{aps}] {name}: count {len(new)} != ref {len(ref)}", flush=True)
            ok = False; continue
        mm = all(a.mmsi == b.mmsi for a, b in zip(ref, new))
        gm = all(abs(a.gap_minutes - b.gap_minutes) < 1e-9 for a, b in zip(ref, new))
        if not (mm and gm):
            print(f"  [gate aps{aps}] {name}: mmsi-aligned={mm} gap_minutes-aligned={gm}", flush=True)
            ok = False
        del ref
    return ok


def build():
    pending = [aps for aps in APS_N if not leaf_done(TRACK, aps_leaf(aps))]
    if not pending:
        print("[anchor datasets] all leaves present -- skipping build", flush=True)
        return
    print(f"[anchor datasets] building aps={pending} (n_anchor={[APS_N[a] for a in pending]}); "
          f"warming cache (one raw-parquet pass for all variants)", flush=True)
    warm_cache_anchor(tuple(APS_N[a] for a in pending))
    for aps in pending:
        na = APS_N[aps]
        kw = dataset_kw(na)
        print(f"[{TRACK}/{aps_leaf(aps)}] build_dataset_raw n_anchor={na} (cached parts)", flush=True)
        tr, va, te = split_graphs(build_dataset_raw(**kw, verbose=True))
        if not _quick_align_check(aps, tr, va, te):
            print(f"[{TRACK}/{aps_leaf(aps)}] SAME-GAPS GATE FAILED -- NOT saving", flush=True)
            raise SystemExit(1)
        save_leaf(TRACK, aps_leaf(aps), GAP_MINUTES, tr, va, te,
                  builder=dict(kw, fn="data_raw.build_dataset_raw"),
                  notes={"cmd": CMD, "n_anchor": na, "anchors_per_side": aps,
                         "source": "raw message-level parquet (m123/m18/m27), clean whitelist",
                         "split": "train_stream._split md5 vessel-disjoint 80/10/10",
                         "same_gaps_as": f"{TRACK}/{REF_LEAF} (identical graphs except "
                                         f"anchor_edge_index; verified by the build gate + verify())"})
        del tr, va, te


# ------------------------------------------------------------------------------------------------ #
#  verification: variant graphs == paper graphs EXCEPT anchor_edge_index; oracle gcd 2.857           #
# ------------------------------------------------------------------------------------------------ #
def _expected_anchor_edges(g, nb: int) -> int:
    """make_run_graph anchor-block arithmetic: every held node gets min(nb, n_before) +
    min(nb, n_after) anchor edges (same pools for all held nodes)."""
    held = g.mask.numpy().astype(bool); n = len(held)
    hi = np.where(held)[0]
    return int(hi.size) * (min(nb, int(hi[0])) + min(nb, n - int(hi[-1]) - 1))


def _edge_pairs(g) -> set:
    a = g.anchor_edge_index.numpy()
    return set(zip(a[0].tolist(), a[1].tolist()))


def oracle_gcd_overall(te):
    """OVERALL-row semantics of the paper table (common_gaps.bucket_rows): median over the
    finite-sinuosity test gaps of the per-gap median GCD error."""
    per = [float(np.median(gcd_errors(g))) for g in te if np.isfinite(_sinuosity(g))]
    return len(per), float(np.median(per))


def verify() -> bool:
    print("\n" + "=" * 100 + "\nANCHOR-SPARSITY VERIFICATION (vs the paper dataset "
          f"{TRACK}/{REF_LEAF} = aps4/n_anchor=8)\n" + "=" * 100, flush=True)
    ok_all = True
    rng = np.random.default_rng(0)
    ref = {n: _load(REF_LEAF, n) for n in ("train", "val", "test")}
    ref_counts = tuple(len(ref[n]) for n in ("train", "val", "test"))
    c_ok = ref_counts == (5350, 635, 774)
    ok_all &= c_ok
    print(f"[ref] {REF_LEAF} counts {ref_counts} (want (5350, 635, 774)): "
          f"{'PASS' if c_ok else 'FAIL'}", flush=True)

    for aps, na in APS_N.items():
        leaf = aps_leaf(aps)
        if not leaf_done(TRACK, leaf):
            print(f"[aps{aps}] MISSING {TRACK}/{leaf}: FAIL", flush=True); ok_all = False
            continue
        nb = na // 2
        new = {n: _load(leaf, n) for n in ("train", "val", "test")}
        counts = tuple(len(new[n]) for n in ("train", "val", "test"))
        c_ok = counts == ref_counts
        # 1. per-graph identity of EVERYTHING except anchor_edge_index (all graphs, all splits)
        ident = True; edge_cnt_ok = True; n_edges_new = n_edges_ref = 0
        for n in ("train", "val", "test"):
            for a, b in zip(ref[n], new[n]):
                ident &= (a.mmsi == b.mmsi and abs(a.gap_minutes - b.gap_minutes) < 1e-9
                          and a.ref == b.ref
                          and torch.equal(a.mask, b.mask) and torch.equal(a.x, b.x)
                          and torch.equal(a.coord, b.coord) and torch.equal(a.y_pos, b.y_pos)
                          and torch.equal(a.true_lonlat, b.true_lonlat)
                          and torch.equal(a.edge_index, b.edge_index))
                edge_cnt_ok &= (b.anchor_edge_index.shape[1] == _expected_anchor_edges(b, nb))
                n_edges_new += int(b.anchor_edge_index.shape[1])
                n_edges_ref += int(a.anchor_edge_index.shape[1])
        # 2. nearest-anchor NESTING on a random sample: aps2 edges subset of aps4, aps4 subset of aps8
        te_r, te_n = ref["test"], new["test"]
        sub_ok = True
        for i in rng.choice(len(te_r), size=min(50, len(te_r)), replace=False):
            small, big = (_edge_pairs(te_n[i]), _edge_pairs(te_r[i])) if na < 8 else \
                         (_edge_pairs(te_r[i]), _edge_pairs(te_n[i]))
            sub_ok &= small.issubset(big)
        # 3. the oracle
        n_fin, gmed = oracle_gcd_overall(new["test"])
        o_ok = n_fin == ORACLE_N and abs(gmed - ORACLE_GCD) < 1e-3
        ok = c_ok and ident and edge_cnt_ok and sub_ok and o_ok
        ok_all &= ok
        print(f"[aps{aps}] (n_anchor={na}, {nb}/side) counts {counts}: {'PASS' if c_ok else 'FAIL'}"
              f" | graphs identical except anchors: {'PASS' if ident else 'FAIL'}"
              f" | anchor-edge count == held*(min({nb},before)+min({nb},after)): "
              f"{'PASS' if edge_cnt_ok else 'FAIL'}"
              f" | nearest-anchor nesting vs aps4 (50 test graphs): {'PASS' if sub_ok else 'FAIL'}",
              flush=True)
        print(f"        oracle gcd_med {gmed:.3f} on {n_fin} finite-sinuosity test gaps "
              f"(want {ORACLE_GCD} on {ORACLE_N}): {'PASS' if o_ok else 'FAIL'}"
              f" | total anchor edges {n_edges_new} vs ref {n_edges_ref} "
              f"(ratio {n_edges_new / max(n_edges_ref, 1):.3f})", flush=True)
        del new

    print(f"\nANCHOR VERIFY OVERALL: {'ALL PASS' if ok_all else 'FAILURES PRESENT'}", flush=True)
    return ok_all


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    if only in ("all", "build"):
        build()
    if only in ("all", "verify"):
        sys.exit(0 if verify() else 1)

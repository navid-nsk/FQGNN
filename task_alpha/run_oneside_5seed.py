"""ONE-SIDED-CONTEXT ablation, 5 seeds, on FQDeepFaithful (Op-K readout); Fig 2e uses run_oneside_native_5seed.py.

Identical to run_full_faithful_5seed (same model construction, recipe, seeds, output formats); the
ONLY change is at the DATA level (the model file is never touched): every RunGraph in train/val/test
is replaced by a one-sided copy built by the pure function `one_sided(g)` below.

DEFINITION.  The honest "one-sided context" is: the model may use ONLY
PRE-gap observed nodes to predict the held nodes.  In this model the held-node prediction is the
Op-K readout over the pooled embeddings h[anchors] + the raw anchor positions/velocities, and h is
produced by 3 layers of message passing -- so post-gap information must be kept out of BOTH the held
nodes AND the pre-gap nodes at ANY hop depth.  The surgical edit that guarantees this for any number
of layers:
  (a) anchor_edge_index: keep ONLY columns whose anchor src is BEFORE the gap (src < gap_start).
      The Op-K readout, its pooled conditioning h[anc], and its boundary velocities then see
      pre-gap anchors only.
  (b) edge_index (temporal kNN): SEVER every edge that crosses the gap boundary, i.e. every edge
      with exactly one endpoint in the post-gap segment (index > gap_end).  This includes the
      prescribed post-gap-observed -> held edges and their reverses (held -> post), AND ALSO the
      direct post-observed <-> pre-observed band edges that exist when the gap is shorter than the
      kNN band (k_temporal=6): without severing those, post-gap data would reach the pre-gap
      anchors' embeddings in one hop and leak into the readout pooling.  Severing ALL crossing
      edges gives the checkable invariant "zero temporal edges connect the post-gap side to the
      rest"; the held/pre side then provably contains no post-gap information at any depth.
      (Dropping held->post / pre->post is a no-op for the held predictions -- post embeddings are
      never read -- but keeps the invariant symmetric and checkable.)
  Post-gap nodes REMAIN in the graph (features, targets, mask unchanged) with their own internal
  edges -- they are simply disconnected from the pre+held side.

IMPORTANT metric note: GCD (linterp_lonlat) stays TWO-SIDED -- it is the fixed reference baseline,
computed at dataset build time and never fed to the model; we ablate the MODEL's context only.  So
`ratio` compares a one-sided model against the two-sided GCD reference (that asymmetry is the point
of the panel).

Outputs (task_alpha/runs/oneside_5seed/): the standard formats via baselines.common_gaps --
  seed<k>/epochs.csv, seed<k>/per_gap.csv, seed<k>/test_buckets.csv, seed<k>/best.pt,
  all_seeds_epochs.csv, all_seeds_test_buckets.csv, summary.csv (+ MEAN/STD rows).

Run (from the repository root):
  smoke : python -m task_alpha.run_oneside_5seed --smoke     (200/60/60 graphs, 2 ep, seed 0 only)
  real  : python -m task_alpha.run_oneside_5seed             (full data, 5 seeds)
"""
from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import math
import os
import random
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch

from .baselines.common_gaps import (bucket_rows, gap_row, load_gaps, summary_row,
                                    write_aggregates, write_buckets, write_epochs, write_per_gap)
from .data import RunGraph, collate_graphs_interp
from .diag_training import GROUPS, group_of
from .fqdeep_faithful import FQDeepFaithful
from .meltin_proof import _bvec, _eval
from .metrics import model_errors

DEV = "cuda"
OUT = "task_alpha/runs/oneside_5seed"
SEEDS = (0, 1, 2, 3, 4)


# ------------------------------------------------------------------------------------------------ #
#  the DATA-level surgery (pure function; never mutates g, never touches the model)                  #
# ------------------------------------------------------------------------------------------------ #
def one_sided(g: RunGraph) -> RunGraph:
    """One-sided copy of g per the header definition: (a) pre-gap anchor edges only; (b) sever every
    temporal edge with exactly one endpoint in the post-gap segment.  Everything else unchanged."""
    hidx = torch.where(g.mask)[0]
    gap_start, gap_end = int(hidx[0]), int(hidx[-1])     # held is a CONTIGUOUS block [start, end]
    # (a) anchor edges: keep src strictly BEFORE the gap
    aei = g.anchor_edge_index[:, g.anchor_edge_index[0] < gap_start]
    # (b) temporal kNN: sever ALL gap-crossing edges (exactly one endpoint after the gap)
    src, dst = g.edge_index[0], g.edge_index[1]
    keep = ~((src > gap_end) ^ (dst > gap_end))
    return dataclasses.replace(g, edge_index=g.edge_index[:, keep], edge_dist=g.edge_dist[keep],
                               anchor_edge_index=aei)


def verify_one_sided(g: RunGraph, go: RunGraph) -> None:
    """Unit test on one graph: edge counts before/after + NO post-gap->held (or any gap-crossing)
    edge remains + anchor edges are pre-gap only + kept edges are exactly the non-crossing set."""
    held = g.mask.bool()
    hidx = torch.where(held)[0]; gap_start, gap_end = int(hidx[0]), int(hidx[-1])
    assert bool(held[gap_start:gap_end + 1].all()), "held block not contiguous"
    s0, d0 = g.edge_index; s1, d1 = go.edge_index
    post = lambda i: i > gap_end                          # post-gap segment (all observed)
    pre = lambda i: i < gap_start                          # pre-gap segment (all observed)
    n_cross0 = int((post(s0) ^ post(d0)).sum())
    checks = {
        "post_obs->held": int((post(s1) & held[d1]).sum()),
        "held->post_obs": int((held[s1] & post(d1)).sum()),
        "post_obs->pre_obs": int((post(s1) & pre(d1)).sum()),
        "pre_obs->post_obs": int((pre(s1) & post(d1)).sum()),
        "anchor_src_not_pre": int((go.anchor_edge_index[0] >= gap_start).sum()),
    }
    a0, a1 = g.anchor_edge_index.shape[1], go.anchor_edge_index.shape[1]
    print(f"[one_sided unit test] mmsi {g.mmsi} | N={g.x.shape[0]} held={int(held.sum())} "
          f"gap=[{gap_start},{gap_end}]", flush=True)
    print(f"  temporal edges : {s0.shape[0]} -> {s1.shape[0]} (dropped {s0.shape[0] - s1.shape[0]}; "
          f"gap-crossing before={n_cross0})", flush=True)
    print(f"  anchor edges   : {a0} -> {a1} (dropped {a0 - a1}; pre-side kept)", flush=True)
    print(f"  leak counts (all MUST be 0): {checks}", flush=True)
    assert all(v == 0 for v in checks.values()), f"one_sided leak check FAILED: {checks}"
    assert s1.shape[0] == s0.shape[0] - n_cross0, "kept-edge count != non-crossing count"
    assert go.edge_dist.shape[0] == s1.shape[0], "edge_dist not aligned with edge_index"
    # unchanged fields must be the SAME objects/values (pure surgery, no side effects on g)
    assert go.x is g.x and go.mask is g.mask and go.linterp_lonlat is g.linterp_lonlat
    print("  PASS: no post-gap edge reaches the held/pre side; GCD reference untouched", flush=True)


# ------------------------------------------------------------------------------------------------ #
#  one seed: EXACT run_full_faithful_5seed recipe on the one-sided graphs                            #
# ------------------------------------------------------------------------------------------------ #
def run_seed(seed: int, tr, va, te, out: str, max_ep: int = 60):
    sdir = f"{out}/seed{seed}"
    os.makedirs(sdir, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    m = FQDeepFaithful(n=6, n_freqs=48, width=96, init_scale=1.84, seed=seed, in_feat=6, layers=3,
                       jacobi_R=4).to(DEV)
    m.use_cuda = True
    with torch.no_grad():
        m.s_raw.fill_(math.log(math.expm1(2.5)))
    b0 = collate_graphs_interp(tr[:4])
    m(b0[0].to(DEV), b0[1].to(DEV), b0[4].to(DEV), b0[6].to(DEV), _bvec(tr[:4]))
    init = {n: p.detach().clone() for n, p in m.named_parameters()}
    opt = torch.optim.Adam(m.parameters(), lr=5e-4)

    db = tr[:64]; dx, dco, dyp, dmk, dei, _, dae = collate_graphs_interp(db)
    DX, DCO, DEI, DAE, DBV = dx.to(DEV), dco.to(DEV), dei.to(DEV), dae.to(DEV), _bvec(db)
    DMK, DYT = dmk.to(DEV), dyp.to(DEV)

    def telemetry():
        m.train(); opt.zero_grad()
        (((m(DX, DCO, DEI, DAE, DBV)[DMK] - DYT[DMK]) ** 2).mean()).backward()
        gsq = {g: 0.0 for g in GROUPS}; dsq = {g: 0.0 for g in GROUPS}; psq = {g: 0.0 for g in GROUPS}
        for n, p in m.named_parameters():
            g = group_of(n)
            if g not in GROUPS:
                continue
            if p.grad is not None:
                gsq[g] += float(p.grad.norm()) ** 2
            dsq[g] += float((p.detach() - init[n]).norm()) ** 2; psq[g] += float(init[n].norm()) ** 2
        opt.zero_grad()
        with torch.no_grad():
            node, L, xi, c, s = m._feats(DX, DCO); h = m.enc(node); P = m._jacobi(xi)
            fa = float((m.theta[0](h[DEI[0]]) @ P.t()).abs().mean())
            fo = float((m.theta_out(h.mean(0, keepdim=True)) @ P.t() - 1).abs().mean())
        gn = {g: gsq[g] ** 0.5 for g in GROUPS}
        rc = {g: dsq[g] ** 0.5 / max(psq[g] ** 0.5, 1e-9) for g in GROUPS}
        return gn, rc, fa, fo

    rows = []
    best, best_state, pctr, bs = 1e9, None, 0, 64
    for ep in range(max_ep):
        m.train(); order = list(range(len(tr))); random.Random(seed * 1000 + ep).shuffle(order)
        ep_loss, ep_nb = 0.0, 0
        for i in range(0, len(order), bs):
            b = [tr[j] for j in order[i:i + bs]]; x, co, yp, mk, ei, _, ae = collate_graphs_interp(b)
            pred = m(x.to(DEV), co.to(DEV), ei.to(DEV), ae.to(DEV), _bvec(b))
            L = ((pred[mk.to(DEV)] - yp.to(DEV)[mk.to(DEV)]) ** 2).mean()
            opt.zero_grad()
            if torch.isfinite(L):
                L.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
                ep_loss += float(L.detach()); ep_nb += 1
        fq, gc, _ = _eval(m, va)                          # val is one-sided too (consistent ablation)
        vm, vg = float(np.median(fq)), float(np.median(gc)); vr = vm / vg
        gn, rc, fa, fo = telemetry()
        rows.append({"epoch": ep, "train_loss": round(ep_loss / max(ep_nb, 1), 6),
                     "val_ratio": round(vr, 4), "val_model_med": round(vm, 4), "val_gcd_med": round(vg, 4),
                     "agg_filter_mag": round(fa, 5), "readout_cond_dev": round(fo, 5),
                     **{f"grad_{g}": round(gn[g], 6) for g in GROUPS},
                     **{f"chg_{g}": round(rc[g], 6) for g in GROUPS}})
        write_epochs(f"{sdir}/epochs.csv", rows)
        if vr < best - 1e-4:
            best, best_state, pctr = vr, copy.deepcopy(m.state_dict()), 0
            torch.save({"state_dict": best_state}, f"{sdir}/best.pt")
        else:
            pctr += 1
        print(f"[seed {seed}] ep {ep:2d}: train {ep_loss / max(ep_nb, 1):.5f} | val {vr:.4f} "
              f"(best {best:.4f}) | aggfilt {fa:.3f} readout_dev {fo:.4f}", flush=True)
        if pctr >= 8:
            print(f"[seed {seed}] early stop", flush=True); break

    m.load_state_dict(best_state)
    # ---- TEST: per-gap + buckets via the SHARED common_gaps machinery (standard formats).
    #      Model predictions on ONE-SIDED graphs; GCD column = the unchanged TWO-SIDED reference. ----
    m.eval(); pg = []
    with torch.no_grad():
        for s0 in range(0, len(te), 64):
            b = te[s0:s0 + 64]; x, coord, yp, mask, ei, ed, aei = collate_graphs_interp(b)
            pred = m(x.to(DEV), coord.to(DEV), ei.to(DEV), aei.to(DEV), _bvec(b)).float().cpu(); off = 0
            for j, g in enumerate(b):
                ng = g.x.shape[0]; gp = pred[off:off + ng]; off += ng
                fe = model_errors(gp, g)
                if len(fe):
                    pg.append(gap_row(s0 + j, g, fe))
    brows = bucket_rows(pg)
    write_per_gap(f"{sdir}/per_gap.csv", pg)
    write_buckets(f"{sdir}/test_buckets.csv", brows)
    ov = brows[0]
    print(f"\n[seed {seed}][TEST one-sided] {ov['n']} gaps OVERALL ratio_med {ov['ratio']} "
          f"(GCD {ov['gcd_med']} model {ov['model_med']} win {ov['win_pct']}%)", flush=True)
    for r in brows[1:]:
        print(f"   {r['bucket']:<11} n={r['n']:4d}  GCD {r['gcd_med']:.2f}  "
              f"model {r['model_med']:.2f}  ratio {r['ratio']:.3f}", flush=True)

    best_val = min(r["val_ratio"] for r in rows)
    del m, opt, init, best_state
    torch.cuda.empty_cache()
    return rows, brows, best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="200/60/60 graphs, 2 epochs, seed 0 only, _smoke out dir")
    args = ap.parse_args()
    out = OUT + ("_smoke" if args.smoke else "")
    os.makedirs(out, exist_ok=True)

    tr, va, te = load_gaps()                              # data/split built ONCE (seed 0), shared
    if args.smoke:
        tr, va, te = tr[:200], va[:60], te[:60]
    seeds = (0,) if args.smoke else SEEDS
    max_ep = 2 if args.smoke else 60

    # ---- the one-sided surgery, applied ONCE at the data level (train + val + test) ----
    verify_one_sided(te[0], one_sided(te[0]))             # unit test on one graph (prints + asserts)
    tr = [one_sided(g) for g in tr]
    va = [one_sided(g) for g in va]
    te = [one_sided(g) for g in te]
    # dataset-wide invariant: zero gap-crossing temporal edges + zero post-gap anchor edges remain
    for split, gs in (("train", tr), ("val", va), ("test", te)):
        nx = sum(int(((g.edge_index[0] > int(torch.where(g.mask)[0][-1]))
                      ^ (g.edge_index[1] > int(torch.where(g.mask)[0][-1]))).sum()) for g in gs)
        na = sum(int((g.anchor_edge_index[0] >= int(torch.where(g.mask)[0][0])).sum()) for g in gs)
        assert nx == 0 and na == 0, f"{split}: crossing={nx} post-anchor={na} (must be 0)"
        print(f"[one_sided] {split}: {len(gs)} graphs, 0 gap-crossing edges, 0 post-gap anchor edges",
              flush=True)

    all_ep, all_bk, summary = [], [], []
    for seed in seeds:
        print(f"\n================ SEED {seed} (one-sided) ================", flush=True)
        rows, brows, best_val = run_seed(seed, tr, va, te, out, max_ep=max_ep)
        for r in rows:
            all_ep.append({"seed": seed, **r})
        for r in brows:
            all_bk.append({"seed": seed, **r})
        summary.append(summary_row(seed, brows, len(rows), best_val))
        write_aggregates(out, all_ep, all_bk, summary)    # crash-safe after EVERY seed (+MEAN/STD)

    print("\n=== ONE-SIDED 5-SEED SUMMARY (TEST, OVERALL) ===", flush=True)
    rat = np.array([s["ratio"] for s in summary], float)
    mm = np.array([s["model_med"] for s in summary], float)
    wp = np.array([s["win_pct"] for s in summary], float)
    for s in summary:
        print(f"  seed {s['seed']}: ratio {s['ratio']} | model_med {s['model_med']} "
              f"| GCD {s['gcd_med']} | win {s['win_pct']}%", flush=True)
    print(f"  ratio     = {rat.mean():.4f} +/- {rat.std(ddof=0):.4f}", flush=True)
    print(f"  model_med = {mm.mean():.3f} +/- {mm.std(ddof=0):.3f} NM", flush=True)
    print(f"  win_pct   = {wp.mean():.1f} +/- {wp.std(ddof=0):.1f} %", flush=True)
    print(f"\nsaved under {out}/seed*/ + all_seeds_epochs.csv, all_seeds_test_buckets.csv, summary.csv",
          flush=True)


if __name__ == "__main__":
    main()

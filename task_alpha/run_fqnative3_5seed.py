"""Anisotropic Op-H, FIVE SEEDS -- FQDeepNative with the ANISOTROPIC chord-frame metric + kinematics-
conditioned metric head (fq_oph3/fqdeep_native3), protocol identical to run_full_faithful_5seed.py
(the Op-K 5-seed reference) and to run_fqnative2_5seed.py (the isotropic Op-H reference).  Per seed:
train, ablation ladder on the best checkpoint (metric frozen / iso-tie / kinematics zeroed / eta
zeroed), full test with buckets.  Aggregates: all_seeds_*.csv + summary.csv with MEAN/STD rows
(columns 0-7 same schema as the references; extra ablation columns appended).

Run: python -m task_alpha.run_fqnative3_5seed
Out: task_alpha/runs/fqnative3_5seed/
"""
from __future__ import annotations

import copy
import csv
import math
import os
import random
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch

from .data import collate_graphs_interp
from .data_raw import build_dataset_raw
from .experiment_compare import _sinuosity
from .fqdeep_native3 import FQDeepNative
from .meltin_proof import _bvec, _eval
from .metrics import gcd_errors, model_errors
from .train_stream import _split

DEV = "cuda"
OUT = "task_alpha/runs/fqnative3_5seed"
REF = "task_alpha/runs/full_faithful_5seed"       # the recorded Op-K-Hermite results
REF2 = "task_alpha/runs/fqnative2_5seed"          # the isotropic-metric Op-H results
SEEDS = (0, 1, 2, 3, 4)


def run_seed(seed: int, tr, va, te):
    sdir = f"{OUT}/seed{seed}"
    os.makedirs(sdir, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    m = FQDeepNative(n=6, n_freqs=48, width=96, init_scale=1.84, seed=seed, in_feat=6, layers=3,
                     jacobi_R=4, oph_weighted=True, oph_aniso=True, oph_kin=True).to(DEV)
    m.use_cuda = True
    with torch.no_grad():
        m.s_raw.fill_(math.log(math.expm1(2.5)))
    # warmup forward BEFORE Adam construction: self.enc is nn.LazyLinear -- its parameters only
    # materialize on the first forward, and Adam captures m.parameters() at construction.  Same
    # order as run_full_faithful_5seed.py; skipping it would silently leave enc untrained.
    b0 = collate_graphs_interp(tr[:4])
    m(b0[0].to(DEV), b0[1].to(DEV), b0[4].to(DEV), b0[6].to(DEV), _bvec(tr[:4]))
    opt = torch.optim.Adam(m.parameters(), lr=5e-4)

    fields = ["epoch", "train_loss", "val_ratio", "val_model_med", "val_gcd_med", "gate_softplus_rho"]
    rows = []
    best, best_state, pctr, bs = 1e9, None, 0, 64
    for ep in range(60):
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
        fq, gc, _ = _eval(m, va)
        vm, vg = float(np.median(fq)), float(np.median(gc)); vr = vm / vg
        rows.append({"epoch": ep, "train_loss": round(ep_loss / max(ep_nb, 1), 6),
                     "val_ratio": round(vr, 4), "val_model_med": round(vm, 4),
                     "val_gcd_med": round(vg, 4),
                     "gate_softplus_rho": round(float(torch.nn.functional.softplus(m.gate_rho)), 4)})
        with open(f"{sdir}/epochs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
        if vr < best - 1e-4:
            best, best_state, pctr = vr, copy.deepcopy(m.state_dict()), 0
            torch.save({"state_dict": best_state}, f"{sdir}/best.pt")
        else:
            pctr += 1
        print(f"[fqnative3 seed {seed}] ep {ep:2d}: train {ep_loss / max(ep_nb, 1):.5f} | "
              f"val {vr:.4f} (best {best:.4f})", flush=True)
        if pctr >= 8:
            print(f"[fqnative3 seed {seed}] early stop", flush=True); break

    m.load_state_dict(best_state)
    # ---- ABLATION LADDER on the best checkpoint (metric channel, anisotropy, kinematics, eta) ----
    fqA, gcA, _ = _eval(m, va)
    r_full = float(np.median(fqA) / np.median(gcA))
    m.oph_w_const = True
    fqB, gcB, _ = _eval(m, va)
    r_woff = float(np.median(fqB) / np.median(gcB))
    m.oph_w_const = False
    m.oph_iso_tie = True                              # anisotropy OFF: along profile for both
    fqI, gcI, _ = _eval(m, va)
    r_iso = float(np.median(fqI) / np.median(gcI))
    m.oph_iso_tie = False
    m.oph_kin_zero = True                             # kinematics input zeroed: pool-only head
    fqK, gcK, _ = _eval(m, va)
    r_kin = float(np.median(fqK) / np.median(gcK))
    m.oph_kin_zero = False
    wsave = m.eta_head[-1].weight.detach().clone(); bsave = m.eta_head[-1].bias.detach().clone()
    with torch.no_grad():
        m.eta_head[-1].weight.zero_(); m.eta_head[-1].bias.zero_()
    fqC, gcC, _ = _eval(m, va)
    r_etaoff = float(np.median(fqC) / np.median(gcC))
    with torch.no_grad():
        m.eta_head[-1].weight.copy_(wsave); m.eta_head[-1].bias.copy_(bsave)
    print(f"\n[ABLATION seed {seed}] val full = {r_full:.4f} | METRIC frozen(const) = {r_woff:.4f} "
          f"(GNN metric channel: {r_woff - r_full:+.4f}) | ISO-tie = {r_iso:.4f} "
          f"(anisotropy: {r_iso - r_full:+.4f}) | KIN zeroed = {r_kin:.4f} "
          f"(kinematics: {r_kin - r_full:+.4f}) | eta zeroed = {r_etaoff:.4f} "
          f"(eta channel: {r_etaoff - r_full:+.4f})", flush=True)
    # ---- TEST (identical evaluation to run_full_faithful_5seed.py) ----
    m.eval(); fq, gc, sn, pg = [], [], [], []
    with torch.no_grad():
        for s0 in range(0, len(te), 64):
            b = te[s0:s0 + 64]; x, coord, yp, mask, ei, ed, aei = collate_graphs_interp(b)
            pred = m(x.to(DEV), coord.to(DEV), ei.to(DEV), aei.to(DEV), _bvec(b)).float().cpu(); off = 0
            for g in b:
                ng = g.x.shape[0]; gp = pred[off:off + ng]; off += ng; fe = model_errors(gp, g)
                if len(fe):
                    fm = float(np.median(fe)); gm = float(np.median(gcd_errors(g))); si = _sinuosity(g)
                    fq.append(fm); gc.append(gm); sn.append(si)
                    pg.append({"mmsi": int(g.mmsi), "gap_min": round(float(g.gap_minutes), 1),
                               "sinuosity": round(float(si), 4) if np.isfinite(si) else "",
                               "gcd_med": round(gm, 3), "model_med": round(fm, 3), "win": int(fm < gm)})
    fq, gc, sn = np.array(fq), np.array(gc), np.array(sn)
    with open(f"{sdir}/per_gap.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pg[0].keys())); w.writeheader(); w.writerows(pg)
    keep = np.isfinite(sn); fqk, gck, snk = fq[keep], gc[keep], sn[keep]
    brows = [{"bucket": "OVERALL", "n": len(fqk), "gcd_med": round(float(np.median(gck)), 3),
              "model_med": round(float(np.median(fqk)), 3),
              "ratio": round(float(np.median(fqk) / np.median(gck)), 4),
              "win_pct": round(100 * float(np.mean(fqk < gck)), 1)}]
    print(f"\n[fqnative3 seed {seed}][TEST] {len(fqk)} gaps OVERALL ratio {brows[0]['ratio']} "
          f"(GCD {brows[0]['gcd_med']} model {brows[0]['model_med']} win {brows[0]['win_pct']}%)",
          flush=True)
    for lo, hi in zip((0.0, 0.02, 0.05, 0.10, 0.25), (0.02, 0.05, 0.10, 0.25, 1e9)):
        sel = (snk >= lo) & (snk < hi)
        if sel.sum() < 3:
            continue
        lab = f"{lo:.2f}-{hi:.2f}" if hi < 1e9 else f">{lo:.2f}"
        rr = float(np.median(fqk[sel]) / np.median(gck[sel]))
        brows.append({"bucket": lab, "n": int(sel.sum()),
                      "gcd_med": round(float(np.median(gck[sel])), 3),
                      "model_med": round(float(np.median(fqk[sel])), 3), "ratio": round(rr, 4),
                      "win_pct": round(100 * float(np.mean(fqk[sel] < gck[sel])), 1)})
        print(f"   {lab:<11} n={int(sel.sum()):4d}  GCD {np.median(gck[sel]):.2f}  "
              f"model {np.median(fqk[sel]):.2f}  ratio {rr:.3f}", flush=True)
    with open(f"{sdir}/test_buckets.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(brows[0].keys())); w.writeheader(); w.writerows(brows)
    del m, opt, best_state
    torch.cuda.empty_cache()
    return rows, brows, {"val_full": r_full, "val_metric_frozen": r_woff, "val_iso_tie": r_iso,
                         "val_kin_zero": r_kin, "val_eta_zero": r_etaoff}


def _load_ref(path):
    """bucket->row dict from seed0/test_buckets.csv + (MEAN, STD) rows from summary.csv."""
    try:
        with open(f"{path}/seed0/test_buckets.csv", newline="") as f:
            ref = {r["bucket"]: r for r in csv.DictReader(f)}
        with open(f"{path}/summary.csv", newline="") as f:
            summ = list(csv.reader(f))
        mean = next((r for r in summ if r and r[0] == "MEAN"), None)
        std = next((r for r in summ if r and r[0] == "STD"), None)
        return ref, mean, std
    except FileNotFoundError:
        return None, None, None


def compare_with_refs(brows):
    """Side-by-side vs the Op-K-Hermite AND the isotropic Op-H results (same protocol/test gaps)."""
    print("\n========= Op-H anisotropic (aniso+kin)  vs  Op-H-W (iso)  vs  Op-K-Hermite =========", flush=True)
    refk, meank, stdk = _load_ref(REF)
    ref2, mean2, std2 = _load_ref(REF2)
    if refk is None and ref2 is None:
        print("  (no reference results found)", flush=True); return
    print(f"{'bucket':<11} {'n':>5} | {'aniso':>8} {'win%':>7} | {'iso':>8} {'win%':>7} "
          f"| {'OpK ratio':>9} {'OpK win%':>8}", flush=True)
    for r in brows:
        k = refk.get(r["bucket"]) if refk else None
        v2 = ref2.get(r["bucket"]) if ref2 else None
        print(f"{r['bucket']:<11} {r['n']:>5} | {r['ratio']:>8} {r['win_pct']:>7} | "
              f"{(v2['ratio'] if v2 else '-'):>8} {(v2['win_pct'] if v2 else '-'):>7} | "
              f"{(k['ratio'] if k else '-'):>9} {(k['win_pct'] if k else '-'):>8}", flush=True)
    if mean2 and std2:
        print(f"\nOp-H-W 5-seed OVERALL: ratio {mean2[4]} +/- {std2[4]}  win {mean2[5]} +/- {std2[5]} %",
              flush=True)
    if meank and stdk:
        print(f"Op-K    5-seed OVERALL: ratio {meank[4]} +/- {stdk[4]}  win {meank[5]} +/- {stdk[5]} %",
              flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    allg = build_dataset_raw(years=(2016, 2017, 2018), months=(7, 8), gap_minutes=360,
                             max_gap_len=500, min_span_factor=1.5, n_anchor=8, clean_only=True,
                             n_mmsi=None, max_files_per_part=None, seed=0)
    allg = [g for g in allg if int(g.mask.sum()) > 0 and _split(g.mmsi) is not None]
    tr = [g for g in allg if _split(g.mmsi) == "train"]
    va = [g for g in allg if _split(g.mmsi) == "val"]
    te = [g for g in allg if _split(g.mmsi) == "test"]
    print(f"{len(tr)} train / {len(va)} val / {len(te)} test", flush=True)

    all_ep, all_bk, summary = [], [], []
    for seed in SEEDS:
        print(f"\n================ SEED {seed} ================", flush=True)
        rows, brows, abl = run_seed(seed, tr, va, te)
        for r in rows:
            all_ep.append({"seed": seed, **r})
        for r in brows:
            all_bk.append({"seed": seed, **r})
        ov = brows[0]
        summary.append({"seed": seed, "n_gaps": ov["n"], "gcd_med": ov["gcd_med"],
                        "model_med": ov["model_med"], "ratio": ov["ratio"], "win_pct": ov["win_pct"],
                        "epochs_trained": len(rows),
                        "best_val_ratio": min(r["val_ratio"] for r in rows),
                        "abl_metric_frozen_val": round(abl["val_metric_frozen"], 4),
                        "abl_iso_tie_val": round(abl["val_iso_tie"], 4),
                        "abl_kin_zero_val": round(abl["val_kin_zero"], 4),
                        "abl_eta_zero_val": round(abl["val_eta_zero"], 4)})
        with open(f"{OUT}/all_seeds_epochs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_ep[0].keys())); w.writeheader(); w.writerows(all_ep)
        with open(f"{OUT}/all_seeds_test_buckets.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_bk[0].keys())); w.writeheader(); w.writerows(all_bk)
        with open(f"{OUT}/summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)

    rat = np.array([s["ratio"] for s in summary], float)
    mm = np.array([s["model_med"] for s in summary], float)
    wp = np.array([s["win_pct"] for s in summary], float)
    n_pad = len(summary[0]) - 6                       # columns after win_pct, blank in MEAN/STD rows
    with open(f"{OUT}/summary.csv", "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(["MEAN", "", summary[0]["gcd_med"], round(float(mm.mean()), 3),
                    round(float(rat.mean()), 4), round(float(wp.mean()), 1)] + [""] * n_pad)
        w.writerow(["STD", "", "", round(float(mm.std(ddof=0)), 3),
                    round(float(rat.std(ddof=0)), 4), round(float(wp.std(ddof=0)), 1)] + [""] * n_pad)
    print(f"\n=== Op-H anisotropic 5-SEED: ratio {rat.mean():.4f} +/- {rat.std(ddof=0):.4f} | "
          f"win {wp.mean():.1f} +/- {wp.std(ddof=0):.1f}% | "
          f"vs Op-H-W 0.6331 +/- 0.0104 (win 66.2 +/- 1.1%) | "
          f"vs Op-K 0.7366 +/- 0.0195 (win 64.9 +/- 0.8%) ===", flush=True)
    compare_with_refs(brows)


if __name__ == "__main__":
    main()

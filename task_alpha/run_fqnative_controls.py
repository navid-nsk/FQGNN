"""CONTROL SUITE -- matched-everything controls for the FQ frequency set and
the GNN conditioning channel.  Training/eval protocol is run_fqnative_blend_5seed.py VERBATIM
(same data, splits, optimizer, schedule, early stop, test-at-best-val, CSV schema); the ONLY
change is the variant flag on FQDeepNative:

  --variant grid     freq_source="grid"     periodic-comb control (regular lattice, matched M+range)
  --variant gauss    freq_source="gauss"    Gaussian-matched control (the RFF analogue, matched moments)
  --variant uniform  freq_source="uniform"  density-matched control (uniform on the FQ bounding box)
  --variant shuffle  freq_source="shuffle"  shuffled-FQ control (exact marginals, module coupling destroyed)
  --variant learned  freq_source="learned"  free-frequency control (nn.Parameter init at the FQ set)
  --variant mlp      pool_source="raw"      no-GNN control (readout conditioning from raw anchor rows)

Run: python -m task_alpha.run_fqnative_controls --variant gauss [--dmu blend] [--seeds 0,1,2,3,4]
Out: task_alpha/runs/controls/<variant>_<dmu>/
"""
from __future__ import annotations

import argparse
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
from .fqdeep_native import FQDeepNative
from .meltin_proof import _bvec
from .metrics import gcd_errors, model_errors
from .train_stream import _split

DEV = "cuda"
VARIANTS = ("grid", "gauss", "uniform", "shuffle", "learned", "mlp")


def _gm(b):
    return torch.tensor([float(g.gap_minutes) for g in b], device=DEV)


def _eval(m, graphs, bs=64):
    m.eval(); fq, gc, sn = [], [], []
    with torch.no_grad():
        for s in range(0, len(graphs), bs):
            b = graphs[s:s + bs]
            x, coord, yp, mask, ei, ed, aei = collate_graphs_interp(b)
            pred = m(x.to(DEV), coord.to(DEV), ei.to(DEV), aei.to(DEV), _bvec(b),
                     gap_minutes=_gm(b)).float().cpu()
            off = 0
            for g in b:
                ng = g.x.shape[0]; gp = pred[off:off + ng]; off += ng
                fe = model_errors(gp, g)
                if len(fe):
                    fq.append(np.median(fe)); gc.append(np.median(gcd_errors(g)))
                    sn.append(_sinuosity(g))
    return np.array(fq), np.array(gc), np.array(sn)


def run_seed(variant: str, dmu: str, out: str, seed: int, tr, va, te):
    sdir = f"{out}/seed{seed}"
    os.makedirs(sdir, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    kw = dict(pool_source="raw") if variant == "mlp" else dict(freq_source=variant)
    m = FQDeepNative(n=6, n_freqs=48, width=96, init_scale=1.84, seed=seed, in_feat=6, layers=3,
                     jacobi_R=4, oph_weighted=True, dmu_source=dmu, **kw).to(DEV)
    m.use_cuda = True
    with torch.no_grad():
        m.s_raw.fill_(math.log(math.expm1(2.5)))
    b0 = collate_graphs_interp(tr[:4])
    m(b0[0].to(DEV), b0[1].to(DEV), b0[4].to(DEV), b0[6].to(DEV), _bvec(tr[:4]),
      gap_minutes=_gm(tr[:4]))
    opt = torch.optim.Adam(m.parameters(), lr=5e-4)

    fields = ["epoch", "train_loss", "val_ratio", "val_model_med", "val_gcd_med"]
    rows = []
    best, best_state, pctr, bs = 1e9, None, 0, 64
    for ep in range(60):
        m.train(); order = list(range(len(tr))); random.Random(seed * 1000 + ep).shuffle(order)
        ep_loss, ep_nb = 0.0, 0
        for i in range(0, len(order), bs):
            b = [tr[j] for j in order[i:i + bs]]; x, co, yp, mk, ei, _, ae = collate_graphs_interp(b)
            pred = m(x.to(DEV), co.to(DEV), ei.to(DEV), ae.to(DEV), _bvec(b), gap_minutes=_gm(b))
            L = ((pred[mk.to(DEV)] - yp.to(DEV)[mk.to(DEV)]) ** 2).mean()
            opt.zero_grad()
            if torch.isfinite(L):
                L.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
                ep_loss += float(L.detach()); ep_nb += 1
        fq, gc, _ = _eval(m, va)
        vm, vg = float(np.median(fq)), float(np.median(gc)); vr = vm / vg
        rows.append({"epoch": ep, "train_loss": round(ep_loss / max(ep_nb, 1), 6),
                     "val_ratio": round(vr, 4), "val_model_med": round(vm, 4),
                     "val_gcd_med": round(vg, 4)})
        with open(f"{sdir}/epochs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
        if vr < best - 1e-4:
            best, best_state, pctr = vr, copy.deepcopy(m.state_dict()), 0
            torch.save({"state_dict": best_state}, f"{sdir}/best.pt")
        else:
            pctr += 1
        print(f"[ctrl:{variant} seed {seed}] ep {ep:2d}: train {ep_loss / max(ep_nb, 1):.5f} | "
              f"val {vr:.4f} (best {best:.4f})", flush=True)
        if pctr >= 8:
            print(f"[ctrl:{variant} seed {seed}] early stop", flush=True); break

    m.load_state_dict(best_state)
    # ---- TEST (identical evaluation to run_fqnative_blend_5seed.py) ----
    m.eval(); fq, gc, sn, pg = [], [], [], []
    with torch.no_grad():
        for s0 in range(0, len(te), 64):
            b = te[s0:s0 + 64]; x, coord, yp, mask, ei, ed, aei = collate_graphs_interp(b)
            pred = m(x.to(DEV), coord.to(DEV), ei.to(DEV), aei.to(DEV), _bvec(b),
                     gap_minutes=_gm(b)).float().cpu(); off = 0
            for g in b:
                ng = g.x.shape[0]; gp = pred[off:off + ng]; off += ng; fe = model_errors(gp, g)
                if len(fe):
                    fm = float(np.median(fe)); gmed = float(np.median(gcd_errors(g))); si = _sinuosity(g)
                    fq.append(fm); gc.append(gmed); sn.append(si)
                    pg.append({"mmsi": int(g.mmsi), "gap_min": round(float(g.gap_minutes), 1),
                               "sinuosity": round(float(si), 4) if np.isfinite(si) else "",
                               "gcd_med": round(gmed, 3), "model_med": round(fm, 3),
                               "win": int(fm < gmed)})
    fq, gc, sn = np.array(fq), np.array(gc), np.array(sn)
    with open(f"{sdir}/per_gap.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pg[0].keys())); w.writeheader(); w.writerows(pg)
    keep = np.isfinite(sn); fqk, gck, snk = fq[keep], gc[keep], sn[keep]
    brows = [{"bucket": "OVERALL", "n": len(fqk), "gcd_med": round(float(np.median(gck)), 3),
              "model_med": round(float(np.median(fqk)), 3),
              "ratio": round(float(np.median(fqk) / np.median(gck)), 4),
              "win_pct": round(100 * float(np.mean(fqk < gck)), 1)}]
    print(f"\n[ctrl:{variant} seed {seed}][TEST] {len(fqk)} gaps OVERALL ratio {brows[0]['ratio']} "
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
    return rows, brows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--dmu", default="blend", choices=("blend", "fd"))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    out = f"task_alpha/runs/controls/{a.variant}_{a.dmu}"
    os.makedirs(out, exist_ok=True)

    allg = build_dataset_raw(years=(2016, 2017, 2018), months=(7, 8), gap_minutes=360,
                             max_gap_len=500, min_span_factor=1.5, n_anchor=8, clean_only=True,
                             n_mmsi=None, max_files_per_part=None, seed=0)
    allg = [g for g in allg if int(g.mask.sum()) > 0 and _split(g.mmsi) is not None]
    tr = [g for g in allg if _split(g.mmsi) == "train"]
    va = [g for g in allg if _split(g.mmsi) == "val"]
    te = [g for g in allg if _split(g.mmsi) == "test"]
    print(f"[ctrl:{a.variant} dmu={a.dmu}] {len(tr)} train / {len(va)} val / {len(te)} test",
          flush=True)

    all_ep, all_bk, summary = [], [], []
    for seed in seeds:
        print(f"\n================ {a.variant.upper()} SEED {seed} ================", flush=True)
        rows, brows = run_seed(a.variant, a.dmu, out, seed, tr, va, te)
        for r in rows:
            all_ep.append({"seed": seed, **r})
        for r in brows:
            all_bk.append({"seed": seed, **r})
        ov = brows[0]
        summary.append({"variant": a.variant, "dmu": a.dmu, "seed": seed, "n_gaps": ov["n"],
                        "gcd_med": ov["gcd_med"], "model_med": ov["model_med"],
                        "ratio": ov["ratio"], "win_pct": ov["win_pct"],
                        "epochs_trained": len(rows),
                        "best_val_ratio": min(r["val_ratio"] for r in rows)})
        with open(f"{out}/all_seeds_epochs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_ep[0].keys())); w.writeheader(); w.writerows(all_ep)
        with open(f"{out}/all_seeds_test_buckets.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_bk[0].keys())); w.writeheader(); w.writerows(all_bk)
        with open(f"{out}/summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)

    rat = np.array([s["ratio"] for s in summary], float)
    wp = np.array([s["win_pct"] for s in summary], float)
    print(f"\n=== CONTROL {a.variant} ({a.dmu}) {len(seeds)}-seed: ratio {rat.mean():.4f} +/- "
          f"{rat.std(ddof=0):.4f} | win {wp.mean():.1f} +/- {wp.std(ddof=0):.1f}% ===", flush=True)
    print("=== references: FQ fd-Dmu 0.6331 +/- 0.0104 | cubic Hermite 0.5724 ===", flush=True)


if __name__ == "__main__":
    main()

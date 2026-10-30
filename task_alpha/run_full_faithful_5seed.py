"""5-seed run of FQDeepFaithful on FULL 6h data.  Does NOT change the model:
identical construction to run_full_faithful.py (n=6, n_freqs=48, width=96, init_scale=1.84,
layers=3, jacobi_R=4, s_raw init 2.5, Adam lr 5e-4, clip 1.0, bs 64, patience 8) -- ONLY the
seed varies (model init + batch order).  Data/split built ONCE (seed 0), shared by all seeds.

Outputs (task_alpha/runs/full_faithful_5seed/):
  seed<k>/epochs.csv        per-epoch TRAIN + VAL performance: train_loss, val_ratio,
                            val_model_med, val_gcd_med + telemetry (agg_filter_mag,
                            readout_cond_dev, grad_<group>, chg_<group>)
  seed<k>/per_gap.csv       per-gap TEST: mmsi, gap_min, sinuosity, gcd_med, model_med, win
  seed<k>/test_buckets.csv  TEST per curvature bucket: bucket, n, gcd_med, model_med, ratio, win%
  seed<k>/best.pt           best-val checkpoint
  all_seeds_test_buckets.csv  every seed's bucket rows (seed column) -- for visualisations
  all_seeds_epochs.csv        every seed's per-epoch rows (seed column) -- for visualisations
  summary.csv                 per-seed OVERALL test row + mean/std across seeds
Run: python -m task_alpha.run_full_faithful_5seed
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
from .diag_training import GROUPS, group_of
from .experiment_compare import _sinuosity
from .fqdeep_faithful import FQDeepFaithful
from .meltin_proof import _bvec, _eval
from .metrics import gcd_errors, model_errors
from .train_stream import _split

DEV = "cuda"
OUT = "task_alpha/runs/full_faithful_5seed"
SEEDS = (0, 1, 2, 3, 4)


def run_seed(seed: int, tr, va, te):
    """One full train+test at `seed`.  Model construction identical to run_full_faithful.py."""
    sdir = f"{OUT}/seed{seed}"
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

    fields = ["epoch", "train_loss", "val_ratio", "val_model_med", "val_gcd_med",
              "agg_filter_mag", "readout_cond_dev"] + \
             [f"grad_{g}" for g in GROUPS] + [f"chg_{g}" for g in GROUPS]
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
        gn, rc, fa, fo = telemetry()
        rows.append({"epoch": ep, "train_loss": round(ep_loss / max(ep_nb, 1), 6),
                     "val_ratio": round(vr, 4), "val_model_med": round(vm, 4), "val_gcd_med": round(vg, 4),
                     "agg_filter_mag": round(fa, 5), "readout_cond_dev": round(fo, 5),
                     **{f"grad_{g}": round(gn[g], 6) for g in GROUPS},
                     **{f"chg_{g}": round(rc[g], 6) for g in GROUPS}})
        with open(f"{sdir}/epochs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
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
    # ---- TEST: per-gap + buckets (identical evaluation to run_full_faithful.py) ----
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
    print(f"\n[seed {seed}][TEST] {len(fqk)} gaps OVERALL ratio_med {brows[0]['ratio']} "
          f"(GCD {brows[0]['gcd_med']} model {brows[0]['model_med']} win {brows[0]['win_pct']}%)", flush=True)
    for lo, hi in zip((0.0, 0.02, 0.05, 0.10, 0.25), (0.02, 0.05, 0.10, 0.25, 1e9)):
        sel = (snk >= lo) & (snk < hi)
        if sel.sum() < 3:
            continue
        lab = f"{lo:.2f}-{hi:.2f}" if hi < 1e9 else f">{lo:.2f}"
        rr = float(np.median(fqk[sel]) / np.median(gck[sel]))
        brows.append({"bucket": lab, "n": int(sel.sum()), "gcd_med": round(float(np.median(gck[sel])), 3),
                      "model_med": round(float(np.median(fqk[sel])), 3), "ratio": round(rr, 4),
                      "win_pct": round(100 * float(np.mean(fqk[sel] < gck[sel])), 1)})
        print(f"   {lab:<11} n={int(sel.sum()):4d}  GCD {np.median(gck[sel]):.2f}  "
              f"model {np.median(fqk[sel]):.2f}  ratio {rr:.3f}", flush=True)
    with open(f"{sdir}/test_buckets.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(brows[0].keys())); w.writeheader(); w.writerows(brows)

    del m, opt, init, best_state
    torch.cuda.empty_cache()
    return rows, brows


def main():
    os.makedirs(OUT, exist_ok=True)
    allg = build_dataset_raw(years=(2016, 2017, 2018), months=(7, 8), gap_minutes=360, max_gap_len=500,
                             min_span_factor=1.5, n_anchor=8, clean_only=True, n_mmsi=None,
                             max_files_per_part=None, seed=0)
    allg = [g for g in allg if int(g.mask.sum()) > 0 and _split(g.mmsi) is not None]
    tr = [g for g in allg if _split(g.mmsi) == "train"]
    va = [g for g in allg if _split(g.mmsi) == "val"]
    te = [g for g in allg if _split(g.mmsi) == "test"]
    print(f"{len(tr)} train / {len(va)} val / {len(te)} test", flush=True)

    all_ep, all_bk, summary = [], [], []
    for seed in SEEDS:
        print(f"\n================ SEED {seed} ================", flush=True)
        rows, brows = run_seed(seed, tr, va, te)
        for r in rows:
            all_ep.append({"seed": seed, **r})
        for r in brows:
            all_bk.append({"seed": seed, **r})
        ov = brows[0]
        summary.append({"seed": seed, "n_gaps": ov["n"], "gcd_med": ov["gcd_med"],
                        "model_med": ov["model_med"], "ratio": ov["ratio"], "win_pct": ov["win_pct"],
                        "epochs_trained": len(rows),
                        "best_val_ratio": min(r["val_ratio"] for r in rows)})
        # write aggregates after EVERY seed (crash-safe partial results)
        with open(f"{OUT}/all_seeds_epochs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_ep[0].keys())); w.writeheader(); w.writerows(all_ep)
        with open(f"{OUT}/all_seeds_test_buckets.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_bk[0].keys())); w.writeheader(); w.writerows(all_bk)
        with open(f"{OUT}/summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)

    # ---- mean/std rows across seeds ----
    rat = np.array([s["ratio"] for s in summary], float)
    mm = np.array([s["model_med"] for s in summary], float)
    wp = np.array([s["win_pct"] for s in summary], float)
    with open(f"{OUT}/summary.csv", "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(["MEAN", "", round(float(summary[0]["gcd_med"]), 3), round(float(mm.mean()), 3),
                    round(float(rat.mean()), 4), round(float(wp.mean()), 1), "", ""])
        w.writerow(["STD", "", "", round(float(mm.std(ddof=0)), 3),
                    round(float(rat.std(ddof=0)), 4), round(float(wp.std(ddof=0)), 1), "", ""])

    print("\n=== 5-SEED SUMMARY (TEST, OVERALL) ===", flush=True)
    for s in summary:
        print(f"  seed {s['seed']}: ratio {s['ratio']} | model_med {s['model_med']} "
              f"| GCD {s['gcd_med']} | win {s['win_pct']}%", flush=True)
    print(f"  ratio     = {rat.mean():.4f} +/- {rat.std(ddof=0):.4f}", flush=True)
    print(f"  model_med = {mm.mean():.3f} +/- {mm.std(ddof=0):.3f} NM", flush=True)
    print(f"  win_pct   = {wp.mean():.1f} +/- {wp.std(ddof=0):.1f} %", flush=True)
    print(f"\nsaved per-seed epochs/per_gap/test_buckets/best.pt under {OUT}/seed*/ "
          f"+ all_seeds_epochs.csv, all_seeds_test_buckets.csv, summary.csv", flush=True)


if __name__ == "__main__":
    main()

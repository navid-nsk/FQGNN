"""CONFORMANCE BASELINE HARNESS: fixed 6h test subsample on which a trained checkpoint and the
closed-form readout are evaluated, so that the numerical effect of a code change can be
measured exactly (ratios and raw predictions, compared against a saved baseline).

Saves (first run) / compares (later runs) under task_alpha/runs/conform_baseline/:
  subsample_ids.json   the 200 fixed gap indices (deterministic, seed 20260718)
  summary.json         overall ratios (trained ckpt + closed form)
  per_gap.csv          per-gap medians for both
  preds_trained.pt     raw predictions of the trained checkpoint (exact-diff detection)
  preds_closed.pt      raw predictions of the closed form

Run: python -m task_alpha._conform_subsample            (first run: writes baseline)
     python -m task_alpha._conform_subsample --compare  (after a code change: compares vs baseline)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch

from .data import collate_graphs_interp
from .experiment_compare import _sinuosity
from .fq_opk import boundary_velocity
from .fq_oph import oph_readout
from .fqdeep_native import FQDeepNative, boundary_velocity_blend
from .meltin_proof import _bvec
from .metrics import gcd_errors, model_errors

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "task_alpha/runs/conform_baseline"
CK = "task_alpha/runs/fqnative_sobolev_5seed/seed0/best.pt"
N_SUB = 200


def _gm(b):
    return torch.tensor([float(g.gap_minutes) for g in b], device=DEV)


def load_subsample():
    with open("task_alpha/data/our_processed/gap_6h/test.pkl", "rb") as f:
        te = pickle.load(f)
    os.makedirs(OUT, exist_ok=True)
    idp = f"{OUT}/subsample_ids.json"
    if os.path.exists(idp):
        ids = json.load(open(idp))
    else:
        rng = np.random.default_rng(20260718)
        ids = sorted(rng.choice(len(te), size=N_SUB, replace=False).tolist())
        json.dump(ids, open(idp, "w"))
    return [te[i] for i in ids], ids


def build_trained():
    m = FQDeepNative(n=6, n_freqs=48, width=96, init_scale=1.84, seed=0, in_feat=6, layers=3,
                     jacobi_R=4, oph_weighted=True, dmu_source="blend",
                     metric_backbone="sobolev2").to(DEV)
    m.use_cuda = True
    with torch.no_grad():
        m.s_raw.fill_(math.log(math.expm1(2.5)))
    return m


def eval_trained(sub):
    m = build_trained()
    b0 = collate_graphs_interp(sub[:4])
    m(b0[0].to(DEV), b0[1].to(DEV), b0[4].to(DEV), b0[6].to(DEV), _bvec(sub[:4]),
      gap_minutes=_gm(sub[:4]))
    m.load_state_dict(torch.load(CK, map_location=DEV)["state_dict"])
    m.eval()
    fq, gc, preds = [], [], []
    with torch.no_grad():
        for s0 in range(0, len(sub), 64):
            b = sub[s0:s0 + 64]
            x, coord, yp, mask, ei, ed, aei = collate_graphs_interp(b)
            pred = m(x.to(DEV), coord.to(DEV), ei.to(DEV), aei.to(DEV), _bvec(b),
                     gap_minutes=_gm(b)).float().cpu()
            preds.append(pred)
            off = 0
            for g in b:
                ng = g.x.shape[0]; gp = pred[off:off + ng]; off += ng
                fe = model_errors(gp, g)
                if len(fe):
                    fq.append(float(np.median(fe))); gc.append(float(np.median(gcd_errors(g))))
    return np.array(fq), np.array(gc), torch.cat(preds)


def eval_closed(sub):
    m = build_trained()
    m.eval()
    with torch.no_grad():
        _, xi, c, s = m.quasicrystal()
    om_all = (2 * math.pi * s * xi[:, 0]).detach()
    W2 = 1.0 / om_all.abs().clamp_min(1e-6) ** 2
    eta0 = torch.zeros(m.M, 2, 2, device=DEV)
    fq, gc, preds = [], [], []
    with torch.no_grad():
        for s0 in range(0, len(sub), 64):
            b = sub[s0:s0 + 64]
            x, coord, yp, mask, ei, ed, aei = collate_graphs_interp(b)
            x, coord, aei = x.to(DEV), coord.to(DEV), aei.to(DEV)
            batch = torch.cat([torch.full((g.x.shape[0],), i, dtype=torch.long)
                               for i, g in enumerate(b)]).to(DEV)
            gm = _gm(b)
            out = x[:, :2].clone()
            held_all = x[:, 5] > 0.5
            asrc, adst = aei[0], aei[1]
            gid_adst = batch[adst]
            for gid in torch.unique(batch):
                held = torch.where(held_all & (batch == gid))[0]
                anc = torch.unique(asrc[gid_adst == gid])
                if held.numel() == 0 or anc.numel() < 2:
                    continue
                at, ay = coord[anc, 0], x[anc, :2]
                v = boundary_velocity(at, ay).to(ay.dtype)
                v = boundary_velocity_blend(at, ay, x, anc, coord, held, float(gm[int(gid)]), v)
                out[held] = oph_readout(at, ay, v, coord[held, 0], om_all, eta0,
                                        gate_rho=None, min_phase=0.25, use_vel=True,
                                        w=W2).to(out.dtype)
            pred = out.float().cpu()
            preds.append(pred)
            off = 0
            for g in b:
                ng = g.x.shape[0]; gp = pred[off:off + ng]; off += ng
                fe = model_errors(gp, g)
                if len(fe):
                    fq.append(float(np.median(fe))); gc.append(float(np.median(gcd_errors(g))))
    return np.array(fq), np.array(gc), torch.cat(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true",
                    help="compare current code against the saved baseline")
    ap.add_argument("--tag", default="", help="label for this comparison")
    a = ap.parse_args()
    sub, ids = load_subsample()
    ft, gt, pt = eval_trained(sub)
    fc, gcl, pc = eval_closed(sub)
    rt = float(np.median(ft) / np.median(gt))
    rc = float(np.median(fc) / np.median(gcl))
    print(f"subsample n={len(ft)} | trained-ckpt ratio {rt:.4f} | closed-form ratio {rc:.4f}")

    if not a.compare:
        json.dump({"n": len(ft), "ratio_trained": rt, "ratio_closed": rc},
                  open(f"{OUT}/summary.json", "w"), indent=1)
        with open(f"{OUT}/per_gap.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["idx", "trained_med", "closed_med", "gcd_med"])
            for i, (a_, b_, c_) in enumerate(zip(ft, fc, gt)):
                w.writerow([ids[i] if i < len(ids) else i, round(a_, 4), round(b_, 4), round(c_, 4)])
        torch.save(pt, f"{OUT}/preds_trained.pt")
        torch.save(pc, f"{OUT}/preds_closed.pt")
        print(f"BASELINE SAVED under {OUT}/")
    else:
        base = json.load(open(f"{OUT}/summary.json"))
        bpt = torch.load(f"{OUT}/preds_trained.pt")
        bpc = torch.load(f"{OUT}/preds_closed.pt")
        dt = float((pt - bpt).abs().max()) if pt.shape == bpt.shape else float("nan")
        dc = float((pc - bpc).abs().max()) if pc.shape == bpc.shape else float("nan")
        print(f"[{a.tag}] vs baseline: trained ratio {base['ratio_trained']:.4f} -> {rt:.4f} "
              f"({rt - base['ratio_trained']:+.4f}) | closed {base['ratio_closed']:.4f} -> {rc:.4f} "
              f"({rc - base['ratio_closed']:+.4f})")
        print(f"[{a.tag}] max |pred diff|: trained {dt:.6f} | closed {dc:.6f}")
        with open(f"{OUT}/comparison_log.csv", "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([a.tag, round(rt, 4), round(rc, 4), round(dt, 6), round(dc, 6)])


if __name__ == "__main__":
    main()

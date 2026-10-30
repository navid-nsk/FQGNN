"""DIAGNOSTIC (eval-only, no training, no model change): closed-form Op-H with SWEPT FIXED metric.

Hypothesis (the Hermite-unification proposition): with metric w_k = |omega_k|^-2 the weighted
min-norm solve minimizes  sum omega^4 |c_k|^2  ~  integral |r''|^2  -- the BENDING ENERGY -- over
the FQ comb, subject to the same four boundary constraints (position + velocity at the two
bracketing anchors) the cubic Hermite matches.  The minimum-bending interpolant under exactly those
constraints IS the cubic Hermite, so the dense-comb limit of our operator at (w = omega^-2, eta = 0)
is cubic Hermite: the classical baseline is a DEGENERATE SPECIAL CASE of Op-H, not a rival.

This script evaluates the CLOSED FORM (eta = 0, no GNN, untrained init frequencies, s = 2.5,
blend jet) on the 6h our-track test cell for metric profiles:
    const | |om|^-1 (~H^1) | |om|^-2 (~H^2, cubic-equivalent) | |om|^-3 (~H^3, quintic-like)
    | |c_LY| (Lee-Yang amplitude profile)
Ratios are relative to GCD (= 1.0); compare with the cubic Hermite baseline on the same cell.
Reads task_alpha/data/our_processed/gap_6h/test.pkl; prints the results (no files written).

Run: python -m task_alpha._diag_sobolev_metric
"""
from __future__ import annotations

import math
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
from .metrics import gcd_errors, model_errors

DEV = "cuda" if torch.cuda.is_available() else "cpu"

with open("task_alpha/data/our_processed/gap_6h/test.pkl", "rb") as f:
    TE = pickle.load(f)
print(f"{len(TE)} test graphs | closed-form Op-H metric sweep (eta=0, untrained, blend jet)\n")

m = FQDeepNative(n=6, n_freqs=48, width=96, init_scale=1.84, seed=0, in_feat=6, layers=3,
                 jacobi_R=4, oph_weighted=True, dmu_source="blend").to(DEV)
with torch.no_grad():
    m.s_raw.fill_(math.log(math.expm1(2.5)))
m.eval()

with torch.no_grad():
    _, xi, c, s = m.quasicrystal()
om_all = (2 * math.pi * s * xi[:, 0]).detach()
eta0 = torch.zeros(m.M, 2, 2, device=DEV)

PROFILES = {
    "const": None,
    "|om|^-1": 1.0 / om_all.abs().clamp_min(1e-6),
    "|om|^-2 (bending/cubic)": 1.0 / om_all.abs().clamp_min(1e-6) ** 2,
    "|om|^-3 (jerk/quintic)": 1.0 / om_all.abs().clamp_min(1e-6) ** 3,
    "|c_LY| amplitudes": c.abs().detach(),
}


def closed_form_eval(w_prof):
    fq, gc, sn = [], [], []
    with torch.no_grad():
        for s0 in range(0, len(TE), 64):
            b = TE[s0:s0 + 64]
            x, coord, yp, mask, ei, ed, aei = collate_graphs_interp(b)
            x, coord, aei = x.to(DEV), coord.to(DEV), aei.to(DEV)
            batch = torch.cat([torch.full((g.x.shape[0],), i, dtype=torch.long)
                               for i, g in enumerate(b)]).to(DEV)
            gm = torch.tensor([float(g.gap_minutes) for g in b], device=DEV)
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
                v_obs = boundary_velocity(at, ay).to(ay.dtype)
                v_obs = boundary_velocity_blend(at, ay, x, anc, coord, held,
                                                float(gm[int(gid)]), v_obs)
                out[held] = oph_readout(at, ay, v_obs, coord[held, 0], om_all, eta0,
                                        gate_rho=None, min_phase=0.25, use_vel=True,
                                        w=w_prof).to(out.dtype)
            pred = out.float().cpu()
            off = 0
            for g in b:
                ng = g.x.shape[0]; gp = pred[off:off + ng]; off += ng
                fe = model_errors(gp, g)
                if len(fe):
                    fq.append(np.median(fe)); gc.append(np.median(gcd_errors(g)))
                    sn.append(_sinuosity(g))
    fq, gc, sn = np.array(fq), np.array(gc), np.array(sn)
    keep = np.isfinite(sn)
    return fq[keep], gc[keep], sn[keep]


print(f"{'metric profile':<26} {'ratio':>7} {'model_med':>10} {'win%':>6}")
res = {}
for name, prof in PROFILES.items():
    fqk, gck, snk = closed_form_eval(prof)
    r = float(np.median(fqk) / np.median(gck))
    res[name] = (fqk, gck, snk)
    print(f"{name:<26} {r:7.4f} {np.median(fqk):10.3f} {100 * np.mean(fqk < gck):6.1f}"
          f"   (n={len(fqk)})")

print("\nreferences: cubic Hermite 0.5724 | trained blend FQGNN 0.5958 | GCD 1.0")
fqk, gck, snk = res["|om|^-2 (bending/cubic)"]
print("\n|om|^-2 closed form by sinuosity bucket (vs Hermite seed0 buckets "
      "0.7306/0.4571/0.4264/0.4446/0.6713):")
for lo, hi in zip((0.0, 0.02, 0.05, 0.10, 0.25), (0.02, 0.05, 0.10, 0.25, 1e9)):
    sel = (snk >= lo) & (snk < hi)
    if sel.sum() >= 3:
        lab = f"{lo:.2f}-{hi:.2f}" if hi < 1e9 else f">{lo:.2f}"
        print(f"  {lab:<11} n={int(sel.sum()):4d}  ratio {np.median(fqk[sel]) / np.median(gck[sel]):.4f}")

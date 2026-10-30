"""DIAGNOSTIC (eval-only): closed-form Op-H (eta=0, untrained, s=2.5, w=|omega|^-2) with the
MATCHED per-track jet (blend on our_processed, fd on standard_process) on every gap-sweep cell --
the direct test of "does training add anything beyond the built-in prior".
Reads task_alpha/data/<track>/gap_<G>h/test.pkl of every cell plus the summary.csv of the trained
sobolev runs and of the Hermite cells; prints one table row per cell (no files written).
Run: python -m task_alpha._diag_closed_vs_trained
"""
from __future__ import annotations

import csv
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
from .metrics import gcd_errors, model_errors

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CELLS = [("our_processed", g, "blend") for g in (3, 6, 12, 24)] + \
        [("standard_process", g, "fd") for g in (2, 6, 12, 24)]

m = FQDeepNative(n=6, n_freqs=48, width=96, init_scale=1.84, seed=0, in_feat=6, layers=3,
                 jacobi_R=4, oph_weighted=True, dmu_source="blend").to(DEV)
with torch.no_grad():
    m.s_raw.fill_(math.log(math.expm1(2.5)))
m.eval()
with torch.no_grad():
    _, xi, c, s = m.quasicrystal()
om_all = (2 * math.pi * s * xi[:, 0]).detach()
W2 = 1.0 / om_all.abs().clamp_min(1e-6) ** 2
eta0 = torch.zeros(m.M, 2, 2, device=DEV)


def closed(track, gap, jet):
    with open(f"task_alpha/data/{track}/gap_{gap}h/test.pkl", "rb") as f:
        te = pickle.load(f)
    fq, gc, sn = [], [], []
    with torch.no_grad():
        for s0 in range(0, len(te), 64):
            b = te[s0:s0 + 64]
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
                v = boundary_velocity(at, ay).to(ay.dtype)
                if jet == "blend":
                    v = boundary_velocity_blend(at, ay, x, anc, coord, held,
                                                float(gm[int(gid)]), v)
                out[held] = oph_readout(at, ay, v, coord[held, 0], om_all, eta0,
                                        gate_rho=None, min_phase=0.25, use_vel=True,
                                        w=W2).to(out.dtype)
            pred = out.float().cpu()
            off = 0
            for g in b:
                ng = g.x.shape[0]; gp = pred[off:off + ng]; off += ng
                fe = model_errors(gp, g)
                if len(fe):
                    fq.append(np.median(fe)); gc.append(np.median(gcd_errors(g)))
                    sn.append(_sinuosity(g))
    fq, gc, sn = np.array(fq), np.array(gc), np.array(sn)
    k = np.isfinite(sn)
    return float(np.median(fq[k]) / np.median(gc[k]))


def trained(track, gap):
    if track == "our_processed" and gap == 6:
        p = "task_alpha/runs/fqnative_sobolev_5seed/summary.csv"
    else:
        root = "gap_sweep_sobolev" if track == "our_processed" else "gap_sweep_sobolev_fd"
        p = f"task_alpha/runs/{root}/{track}/gap_{gap}h/fqgnn/summary.csv"
    rows = [r for r in csv.DictReader(open(p)) if r.get("seed", "").isdigit()]
    return float(np.mean([float(r["ratio"]) for r in rows]))


def hermite(track, gap):
    p = f"task_alpha/runs/gap_sweep/{track}/gap_{gap}h/hermite/summary.csv"
    return float(next(csv.DictReader(open(p)))["ratio"])


print(f"{'cell':<12} {'jet':<6} {'closed':>7} {'trained':>8} {'delta':>8} {'hermite':>8}")
for track, gap, jet in CELLS:
    cl = closed(track, gap, jet)
    tr = trained(track, gap)
    h = hermite(track, gap)
    tag = ("our" if track.startswith("our") else "std") + f"-{gap}h"
    print(f"{tag:<12} {jet:<6} {cl:>7.4f} {tr:>8.4f} {tr - cl:>+8.4f} {h:>8.4f}", flush=True)

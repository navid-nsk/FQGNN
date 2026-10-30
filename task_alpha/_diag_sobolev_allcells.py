"""DIAGNOSTIC (eval-only): closed-form Op-H at the bending metric w=|omega|^-2 on ALL 8 gap-sweep
cells, side by side with the cubic Hermite baseline's stored per-cell results.
Reads task_alpha/data/<track>/gap_<G>h/test.pkl and the Hermite summary.csv of each cell; prints a
per-cell table (no files written).
Run: python -m task_alpha._diag_sobolev_allcells
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
TRACK_GAPS = {"our_processed": (3, 6, 12, 24), "standard_process": (2, 6, 12, 24)}

m = FQDeepNative(n=6, n_freqs=48, width=96, init_scale=1.84, seed=0, in_feat=6, layers=3,
                 jacobi_R=4, oph_weighted=True, dmu_source="blend").to(DEV)
with torch.no_grad():
    m.s_raw.fill_(math.log(math.expm1(2.5)))
m.eval()
with torch.no_grad():
    _, xi, c, s = m.quasicrystal()
om_all = (2 * math.pi * s * xi[:, 0]).detach()
W_PROF = 1.0 / om_all.abs().clamp_min(1e-6) ** 2
eta0 = torch.zeros(m.M, 2, 2, device=DEV)


def eval_cell(graphs):
    fq, gc, sn = [], [], []
    with torch.no_grad():
        for s0 in range(0, len(graphs), 64):
            b = graphs[s0:s0 + 64]
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
                                        w=W_PROF).to(out.dtype)
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
    fqk, gck = fq[keep], gc[keep]
    return (float(np.median(fqk) / np.median(gck)), float(np.median(fqk)),
            float(100 * np.mean(fqk < gck)), int(len(fqk)))


def hermite_ref(track, gap):
    p = f"task_alpha/runs/gap_sweep/{track}/gap_{gap}h/hermite/summary.csv"
    if not os.path.exists(p):
        return None
    with open(p) as f:
        for row in csv.DictReader(f):
            return float(row["ratio"])
    return None


print(f"{'cell':<26} {'FQ w=om^-2 (closed)':>20} {'Hermite':>9} {'delta':>8} {'win%':>6} {'n':>5}")
for track, gaps in TRACK_GAPS.items():
    for gap in gaps:
        path = f"task_alpha/data/{track}/gap_{gap}h/test.pkl"
        if not os.path.exists(path):
            print(f"{track}/gap_{gap}h: no test.pkl, skipped")
            continue
        with open(path, "rb") as f:
            te = pickle.load(f)
        r, med, win, n = eval_cell(te)
        h = hermite_ref(track, gap)
        d = "" if h is None else f"{r - h:+8.4f}"
        print(f"{track + '/gap_' + str(gap) + 'h':<26} {r:>20.4f} "
              f"{h if h is not None else float('nan'):>9.4f} {d:>8} {win:>6.1f} {n:>5}")

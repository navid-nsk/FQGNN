"""DIAGNOSTIC (eval-only): decompose the std-2h error gap between the fd-native and the
sobolev-blend configurations -- jet source x metric backbone.
Closed-form Op-H (eta=0, untrained, s=2.5) on standard_process/gap_2h test:
    {fd, blend} x {uniform, |om|^-2 (bending)}
Ratios are relative to GCD (= 1.0); compare with the trained models and the cubic Hermite
baseline on the same cell.
Also reports the fd-vs-instrument jet disagreement statistics at the bracketing anchors.
Reads task_alpha/data/standard_process/gap_2h/test.pkl; prints the results (no files written).
Run: python -m task_alpha._diag_std2h_decompose
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

with open("task_alpha/data/standard_process/gap_2h/test.pkl", "rb") as f:
    TE = pickle.load(f)
print(f"{len(TE)} std-2h test graphs | closed-form decomposition (jet x metric)\n")

m = FQDeepNative(n=6, n_freqs=48, width=96, init_scale=1.84, seed=0, in_feat=6, layers=3,
                 jacobi_R=4, oph_weighted=True, dmu_source="blend").to(DEV)
with torch.no_grad():
    m.s_raw.fill_(math.log(math.expm1(2.5)))
m.eval()
with torch.no_grad():
    _, xi, c, s = m.quasicrystal()
om_all = (2 * math.pi * s * xi[:, 0]).detach()
eta0 = torch.zeros(m.M, 2, 2, device=DEV)
W2 = 1.0 / om_all.abs().clamp_min(1e-6) ** 2

# jet disagreement stats (bracketing anchors, sog>0.1kn where instrument exists)
dis, spd = [], []


def eval_cfg(jet: str, w_prof):
    fq, gc, sn = [], [], []
    collect = jet == "blend" and w_prof is not None and not dis   # collect stats once
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
                v_fd = boundary_velocity(at, ay).to(ay.dtype)
                if jet == "blend":
                    v_use = boundary_velocity_blend(at, ay, x, anc, coord, held,
                                                    float(gm[int(gid)]), v_fd)
                    if collect:
                        nz = (v_fd.norm(dim=-1) > 0) & (v_use.norm(dim=-1) > 0)
                        if bool(nz.any()):
                            rel = ((v_use[nz] - v_fd[nz]).norm(dim=-1)
                                   / v_fd[nz].norm(dim=-1).clamp_min(1e-9))
                            dis.extend(rel.tolist())
                            spd.extend((x[anc, 2][nz] * 20.0).tolist())
                else:
                    v_use = v_fd
                out[held] = oph_readout(at, ay, v_use, coord[held, 0], om_all, eta0,
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
    k = np.isfinite(sn)
    return float(np.median(fq[k]) / np.median(gc[k])), float(100 * np.mean(fq[k] < gc[k])), int(k.sum())


print(f"{'jet':<7} {'metric':<16} {'ratio':>8} {'win%':>6}")
for jet in ("blend", "fd"):
    for wname, wp in (("|om|^-2", W2), ("uniform", None)):
        r, w, n = eval_cfg(jet, wp)
        print(f"{jet:<7} {wname:<16} {r:>8.4f} {w:>6.1f}   (n={n})")

print("\nreferences: trained fd-native 0.6245 | Hermite 0.8959 | trained sobolev-blend 0.955 | GCD 1.0")
if dis:
    d = np.array(dis)
    print(f"\njet disagreement |v_inst - v_fd|/|v_fd| at bracketing anchors (n={len(d)}):")
    print(f"  median {np.median(d):.3f} | mean {d.mean():.3f} | >25% rel diff: {100*np.mean(d>0.25):.1f}%"
          f" | >50%: {100*np.mean(d>0.5):.1f}%")

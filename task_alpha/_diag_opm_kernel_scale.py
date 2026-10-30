"""Companion of _diag_opm_lambda_oracle: measure the boundary-conditioned kernel scale
K~(tau_j, tau_j) on real gaps.

The soft block's dimensionless dial is lambda_j * K~(tau_j,tau_j) (the Sherman-Morrison
response is lambda K~ / (1 + lambda K~)).  The init kappa_lambda = softplus(-10) fixes
lambda in ABSOLUTE units; this script measures K~'s scale so the init's dimensionless operating
point -- and its distance to the oracle's -- can be read off.  Measurement only: reads the fixed
200-gap subsample (_conform_subsample) and runs/fqnative_opm_5seed/seed0/best.pt and prints
the scale statistics.

Run (repo root, env FQ_AMP_MODE=akkv_torus FQ_AMP_NMC=262144 FQ_READOUT_MODE=opm):
    python -m task_alpha._diag_opm_kernel_scale
"""
from __future__ import annotations

import math
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch

from ._conform_subsample import _gm, build_trained, load_subsample
from .data import collate_graphs_interp
from .fq_opk import boundary_velocity
from .fqdeep_native import boundary_velocity_blend
from .meltin_proof import _bvec

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CK = "task_alpha/runs/fqnative_opm_5seed/seed0/best.pt"
PI = math.pi


def main():
    assert os.environ.get("FQ_READOUT_MODE") == "opm"
    sub, _ = load_subsample()
    m = build_trained()
    b0 = collate_graphs_interp(sub[:4])
    m(b0[0].to(DEV), b0[1].to(DEV), b0[4].to(DEV), b0[6].to(DEV), _bvec(sub[:4]),
      gap_minutes=_gm(sub[:4]))
    m.load_state_dict(torch.load(CK, map_location=DEV)["state_dict"])
    m.eval()

    kdiags, r0s = [], []
    with torch.no_grad():
        _, xi, c, s = m.quasicrystal()
        om_all = (2 * PI * s * xi[:, 0]).detach()
        P = m._jacobi(xi)
        for s0 in range(0, len(sub), 64):
            b = sub[s0:s0 + 64]
            x, coord, yp, mask, ei, ed, aei = collate_graphs_interp(b)
            x, coord, aei, ei = x.to(DEV), coord.to(DEV), aei.to(DEV), ei.to(DEV)
            batch = torch.cat([torch.full((g.x.shape[0],), i, dtype=torch.long)
                               for i, g in enumerate(b)]).to(DEV)
            gm = _gm(b)
            held_all = x[:, 5] > 0.5
            asrc, adst = aei[0], aei[1]
            gid_adst = batch[adst]
            h, xi_, _, _, Pj, xi0 = m.embed(x, coord, ei)
            for gid in torch.unique(batch):
                held = torch.where(held_all & (batch == gid))[0]
                anc = torch.unique(asrc[gid_adst == gid])
                if held.numel() == 0 or anc.numel() < 2:
                    continue
                at, ay = coord[anc, 0], x[anc, :2]
                v = boundary_velocity(at, ay).to(ay.dtype)
                v = boundary_velocity_blend(at, ay, x, anc, coord, held, float(gm[int(gid)]), v)
                pool = h[anc].mean(0)
                filt = m.w_head(pool) @ Pj.t()
                w = torch.nn.functional.softplus(filt) + 1e-3
                w = w * (1.0 / om_all.abs().clamp_min(1e-6) ** 2)
                # mirror opm_readout's system assembly (gamma law skipped: scale question only)
                dt = ay.dtype
                ts_i = torch.argsort(at)
                ts, ys, vs = at[ts_i], ay[ts_i], v[ts_i]
                gapi = int(torch.argmax(ts[1:] - ts[:-1]))
                a_, b_ = ts[gapi], ts[gapi + 1]
                ba = (b_ - a_).clamp_min(1e-9)
                keep = om_all.abs() * ba > 0.25
                if int(keep.sum()) < 2:
                    continue
                omk = om_all[keep]
                wk = w[keep].clamp_min(1e-6); wk = wk / wk.max()
                w2 = torch.cat([wk, wk])
                t0 = 0.5 * (a_ + b_)
                slope = (ys[gapi + 1] - ys[gapi]) / ba

                def vrow(tau):
                    return torch.cat([torch.cos(omk * tau), torch.sin(omk * tau)])

                def drow(tau):
                    return torch.cat([-omk * torch.sin(omk * tau), omk * torch.cos(omk * tau)])

                om_ref = omk.abs().mean().clamp_min(1e-9)
                rows = [vrow(a_ - t0), vrow(b_ - t0)]
                for v_end, tau_end in ((vs[gapi], a_ - t0), (vs[gapi + 1], b_ - t0)):
                    if bool((v_end != 0).any()):
                        rows.append(drow(tau_end) / om_ref)
                B = torch.stack(rows) * w2.unsqueeze(0)
                Bp = torch.linalg.pinv(B, rcond=1e-6)
                ext = [j for j in range(ts.shape[0]) if j not in (gapi, gapi + 1)]
                if not ext:
                    continue
                tau_e = (ts - t0)[torch.tensor(ext, device=ts.device)]
                V = torch.cat([torch.cos(tau_e.unsqueeze(-1) * omk),
                               torch.sin(tau_e.unsqueeze(-1) * omk)], dim=-1) * w2.unsqueeze(0)
                PVT = V.t() - Bp @ (B @ V.t())
                G1 = V @ PVT                              # G at lambda = 1: diag = K~(tau_j,tau_j)
                kdiags.append(G1.diagonal().float().cpu().numpy())
                # r0 scale: detrended residual minus base prediction at exterior anchors
                from .fq_opm import _detrend_sorted
                rho_t = _detrend_sorted(ts, ys, gapi)[0]
                eta = m.eta_head(pool).view(m.M, 2, 2)
                etak = eta[keep]
                z = torch.cat([etak[:, 0, :], etak[:, 1, :]], dim=0)
                d_ = torch.stack([torch.zeros(2, device=DEV), torch.zeros(2, device=DEV)] +
                                 [(v_end - slope) / om_ref for v_end, tau_end in
                                  ((vs[gapi], 0), (vs[gapi + 1], 0)) if bool((v_end != 0).any())])
                u0 = Bp @ d_
                cfree = z - Bp @ (B @ z)
                r0 = rho_t[torch.tensor(ext, device=ts.device)] - V @ (u0 + cfree)
                r0s.append(r0.norm(dim=-1).float().cpu().numpy())

    kd = np.concatenate(kdiags)
    r0 = np.concatenate(r0s)
    print(f"K~(tau_j,tau_j) over {kd.size} exterior anchors ({len(kdiags)} gaps):")
    print(f"  median {np.median(kd):.4e} | mean {kd.mean():.4e} | P10 {np.percentile(kd,10):.4e} "
          f"| P90 {np.percentile(kd,90):.4e}")
    print(f"  => lambda for lambda*K~ = 1 (response half-max): median {1/np.median(kd):.3g}")
    print(f"  => init kappa_lambda = 4.541e-5 puts the dimensionless dial at "
          f"{4.541e-5*np.median(kd):.3e}")
    print(f"|r0_j| (residual the soft block acts on): median {np.median(r0):.4e} | "
          f"P90 {np.percentile(r0,90):.4e}  (position units, 20 NM = 1)")
    print(f"plug-in lambda = 1/median|r0|^2 = {1/np.median(r0)**2:.3g}")


if __name__ == "__main__":
    main()

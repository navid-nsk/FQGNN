"""
task_alpha.fq_interp  --  FQTrajectoryInterpolator with a Dmu-VELOCITY-EXTRAPOLATION readout.

A convex-combination-of-anchor-positions readout is NOT used: a convex combination of anchor
positions is bounded by their hull, so it cannot bend with the trajectory and cannot beat linear
interpolation (GCD) on curved gaps.

This readout EXTRAPOLATES with the faithful Dmu velocity channel (the derivative measure on the shared
quasicrystal spectrum, amplitudes (2 pi i xi) c_k).  For each held vertex i and each two-sided anchor
neighbour j, a first-order prediction
        p_ij = pos_j + v_j * (t_i - t_j)
is formed from the OBSERVED anchor position pos_j and a Dmu-derived velocity v_j (a linear readout of
the Dmu features at j), then blended:
        p_i = sum_j softmax_j( s_k * K_OpD(dt_ij) + <q_i, k_j> ) * p_ij .
Blending a forward extrapolation from the left anchors and a backward extrapolation from the right
anchors is Hermite-like (uses end positions AND velocities) -> it captures curvature and is NOT
hull-bounded.  EVERY primitive is load-bearing: Op-A'/QFE + Op-B + Op-C build the node embedding h
that drives the attention weights; Op-D's twisted-Eberlein kernel sets the kernel weights; the Dmu
velocity channel drives the extrapolation; the output is anchored to OBSERVED anchor positions.
Inherits the shared quasicrystal, all primitives, and the certificate ledger from FQTrajectoryImputer.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .fq_imputer import FQTrajectoryImputer
from .fq_opk import opk_interp

PI = math.pi


class FQTrajectoryInterpolator(FQTrajectoryImputer):
    def __init__(self, n: int = 6, n_freqs: int = 24, k_max: int = 3, width: int = 64,
                 ac_dim: int = 8, init_scale: float = 2.0, seed: int = 0, in_feat: int = 6,
                 att_dim: int = 32):
        super().__init__(n=n, n_freqs=n_freqs, k_max=k_max, width=width, ac_dim=ac_dim,
                         init_scale=init_scale, seed=seed, in_feat=in_feat)
        g = torch.Generator().manual_seed(seed + 1)
        # Op-D SCALAR kernel taps for the interpolation weights (one per shared frequency)
        self.g_k_cos = nn.Parameter(0.1 * torch.randn(self.M, generator=g))
        self.g_k_sin = nn.Parameter(0.1 * torch.randn(self.M, generator=g))
        self.kernel_scale = nn.Parameter(torch.tensor(1.0))     # Op-D kernel weight vs attention
        self.att_dim = att_dim
        self.Wq = nn.Linear(width, att_dim)                     # query / key on the FQ node embedding
        self.Wk = nn.Linear(width, att_dim)
        # Dmu-VELOCITY head: the derivative measure (4M features) -> a 2-D ENU velocity per node
        # (pos-units per NORMALIZED-time unit).  This is the faithful extrapolation term.
        self.vel_head = nn.Linear(4 * self.M, 2)
        self.readout = nn.Identity()                            # the regressor readout is REPLACED
        # --- Op-K (CPD/universal-kriging FQ interpolant) ADD-ON readout (see task_alpha/fq_opk.py) ---
        self.use_opk = False                                    # toggle: Op-K readout vs the softmax blend
        self.opk_ridge = 0.1                                    # relative Tikhonov ridge (× K(0)); conditioning

    def _opk_readout(self, x, coord, anchor_edge_index, batch, xi, c, s):
        """Op-K (CPD/universal-kriging FQ interpolant) readout: per gap, fit the affine drift + FQ kernel
        through the two-sided anchors and evaluate at the held nodes.  held = x[:,5]>0.5 (is_held feature);
        anchor positions = x[anchors,:2] (observed, kept).  Affine-reproducing => no straight-gap floor."""
        held_all = x[:, 5] > 0.5
        asrc, adst = anchor_edge_index[0], anchor_edge_index[1]
        gid_of_adst = batch[adst]                                  # graph id for each anchor edge
        pred = x[:, :2].clone()                                    # default identity for anchors
        for gid in torch.unique(batch):
            held = torch.where(held_all & (batch == gid))[0]
            if held.numel() == 0:
                continue
            anc = torch.unique(asrc[gid_of_adst == gid])           # this gap's two-sided anchors
            if anc.numel() < 2:
                continue
            p = opk_interp(coord[anc], coord[anc, 0], x[anc, :2], coord[held], coord[held, 0],
                           xi, c, s, ridge=self.opk_ridge)
            pred[held] = p.to(pred.dtype)
        return pred

    def forward(self, x, coord, edge_index, anchor_edge_index, edge_dist=None, extensive=False, batch=None):
        L, xi, c, s = self.quasicrystal()
        qfe, ac = self.qfe(coord, xi, c, s)                     # Op-A'
        theta, pUri = self.secular_features(coord, L)           # Op-B + Op-C
        vel = self.velocity(coord, xi, c, s)                    # D-mu velocity channel [N, 4M]
        node = torch.cat([qfe, ac, theta, pUri, vel, x], dim=-1)
        m = self.msg(node)

        # ---- Op-D twisted-Eberlein message passing -> FQ node embedding h ----
        N = coord.shape[0]
        if self.use_cuda:                                       # CUDA Op-D (fused conv + scatter + /deg)
            from fqcuda.fq_opd_cuda import opd_cuda
            agg = opd_cuda(m, coord, xi, c, s, self.g_cos, self.g_sin, edge_index, intensive=True)
            K = agg                                             # placeholder (telemetry only)
        else:
            src, dst = edge_index[0], edge_index[1]
            delta = coord[dst] - coord[src]
            phk = 2 * PI * s * (delta @ xi.t())
            Gc = c.real.unsqueeze(-1) * self.g_cos
            Gs = c.imag.unsqueeze(-1) * self.g_sin
            K = torch.cos(phk) @ Gc - torch.sin(phk) @ Gs
            agg = torch.zeros(N, m.shape[1], device=m.device, dtype=m.dtype)
            agg.index_add_(0, dst, m[src] * K)
            deg = torch.zeros(N, 1, device=m.device, dtype=m.dtype)
            deg.index_add_(0, dst, torch.ones(dst.shape[0], 1, device=m.device, dtype=m.dtype))
            agg = agg / deg.clamp_min(1.0)
        h = self.act(self.norm(m + self.update(torch.cat([m, agg], dim=-1))))   # [N, W]

        # ---- Dmu velocity per node (ENU pos-units per normalized-time unit) ----
        v = self.vel_head(vel)                                  # [N, 2]

        # ---- Op-K ADD-ON readout: all upstream FQ primitives (Op-A/B/C/D, Dmu, h) are still computed
        #      above as the FQ representation; Op-K replaces only the final softmax blend. ----
        if self.use_opk and batch is not None:
            pred = self._opk_readout(x, coord, anchor_edge_index, batch, xi, c, s)
            self._last = {"Op-A_QFE": qfe, "Op-B_eigenphase": theta, "Op-C_secular_value": pUri,
                          "Dmu_velocity": vel, "Op-D_aggregate": agg, "message": m, "readout": pred,
                          "opk": True}
            return pred

        # ---- Dmu-velocity-EXTRAPOLATION readout over two-sided anchor edges ----
        asrc, adst = anchor_edge_index[0], anchor_edge_index[1]
        dt = (coord[adst, 0] - coord[asrc, 0]).unsqueeze(-1)    # normalized time gap anchor->held [A,1]
        dts = coord[adst] - coord[asrc]
        pha = 2 * PI * s * (dts @ xi.t())                      # [A, M]
        Kk = torch.cos(pha) @ (c.real * self.g_k_cos) - torch.sin(pha) @ (c.imag * self.g_k_sin)  # [A] Op-D
        q = self.Wq(h)[adst]; k = self.Wk(h)[asrc]
        att = (q * k).sum(-1) / math.sqrt(self.att_dim)
        logit = (self.kernel_scale * Kk + att).clamp(-20.0, 20.0)

        # scatter-softmax over each held node's anchor neighbours (dtype-aligned for fp32/fp64)
        maxv = torch.full((N,), float("-inf"), device=logit.device, dtype=logit.dtype).scatter_reduce(
            0, adst, logit, reduce="amax", include_self=True)
        e = (logit - maxv[adst]).exp()
        denom = torch.zeros(N, device=logit.device, dtype=logit.dtype).index_add(0, adst, e)
        w = e / denom[adst].clamp_min(1e-9)                     # [A]

        # first-order (position + Dmu-velocity * dt) extrapolation from each anchor, then blend
        p_ij = x[asrc, :2] + v[asrc] * dt                      # [A, 2]  pos_j + v_j * (t_i - t_j)
        interp = torch.zeros(N, 2, device=x.device, dtype=x.dtype)
        interp.index_add_(0, adst, w.unsqueeze(-1) * p_ij)
        recv = torch.zeros(N, device=x.device, dtype=w.dtype).index_add(0, adst, torch.ones_like(w)) > 0
        pred = torch.where(recv.unsqueeze(-1), interp, x[:, :2])   # held -> extrapolated; else identity

        self._last = {"Op-A_QFE": qfe, "Op-A_ac": ac, "Op-B_eigenphase": theta,
                      "Op-C_secular_value": pUri, "Dmu_velocity": vel, "Dmu_v_enu": v,
                      "Op-D_kernel": K, "Op-D_aggregate": agg, "message": m,
                      "Op-D_interp_kernel": Kk, "interp_weight": w, "readout": pred}
        return pred

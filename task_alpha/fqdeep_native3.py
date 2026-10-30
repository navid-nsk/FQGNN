"""task_alpha.fqdeep_native3 -- FQDeepNative with the ANISOTROPIC chord-frame spectral metric
(fq_oph3.py) + boundary-kinematics conditioning of the metric head.

The message passing is INHERITED VERBATIM (embed() from fqdeep_faithful).  The readout differs
from fqdeep_native.py (oph_weighted=True, flat backbone, finite-difference jet) in two ways:

  1. ANISOTROPIC METRIC: the metric head emits TWO Jacobi profiles --
     along-chord and cross-chord -- and fq_oph3 solves each chord-frame component in its own
     weighted min-norm metric.  Physically: a turn is cross-chord
     spectral energy, a speed change is along-chord -- one shared profile cannot price them
     separately.  Every exactness theorem holds componentwise (see the fq_oph3.py selftests).

  2. KINEMATICS-CONDITIONED METRIC HEAD: the boundary kinematics carry identifiable information
     about the per-gap metric, so the head input is [pooled anchor embedding | 12 leak-safe boundary
     kinematics] (gap duration, chord speed, same-side boundary-velocity mismatches decomposed in
     the chord frame, sentinel flags).  All 12 derive from anchor times/positions and
     fq_opk.boundary_velocity output only -- held positions never read.

INIT: both profiles start uniform (bias[0] = bias[R+1] = 1, small weights)
=> at init the anisotropic and the isotropic solve coincide with the unweighted closed form.
Ablation switches: oph_w_const (metric frozen -> uniform metric, the GNN-channel ablation),
oph_iso_tie (cross profile dropped -> isotropic solve), oph_kin_zero (kinematics zeroed ->
pool-only head).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .fq_opk import boundary_velocity
from .fq_oph3 import oph_readout
from .fqdeep_faithful import FQDeepFaithful

PI = math.pi
KIN_DIM = 12


def boundary_kin(at: torch.Tensor, ay: torch.Tensor, av: torch.Tensor) -> torch.Tensor:
    """12 leak-safe boundary-kinematics features for the metric head.
    Inputs are the SAME tensors oph_readout sees: anchor times, anchor positions, and same-side
    finite-difference velocities (fq_opk.boundary_velocity output, gap segment excluded; exact-zero
    row = the no-same-side-segment sentinel).  Held positions never read.  Features are
    dimensionless or log-squashed so they sit O(1) next to the pooled embedding."""
    dt = ay.dtype
    at = at.to(dt)
    idx = torch.argsort(at)
    ts, ys, vs = at[idx], ay[idx], av[idx].to(dt)
    gapi = int(torch.argmax(ts[1:] - ts[:-1]))
    a, b = ts[gapi], ts[gapi + 1]
    pa, pb = ys[gapi], ys[gapi + 1]
    va, vb = vs[gapi], vs[gapi + 1]
    ba = (b - a).clamp_min(1e-9)
    slope = (pb - pa) / ba
    sn = slope.norm()
    if float(sn) > 1e-9:
        u_al = slope / sn
    else:
        u_al = torch.tensor([1.0, 0.0], device=ay.device, dtype=dt)
    u_cr = torch.stack([-u_al[1], u_al[0]])
    sa = float((va != 0).any())            # has-velocity flags (0 = sentinel side)
    sb = float((vb != 0).any())
    da, db = va - slope, vb - slope        # boundary-velocity mismatch vs the chord
    den = sn + 1e-6
    zero = torch.zeros((), device=ay.device, dtype=dt)
    cosab = (va @ vb) / (va.norm() * vb.norm() + 1e-9) if sa and sb else zero
    nrows = sa + sb
    dbar = (sa * da.norm() + sb * db.norm()) / (nrows * den) if nrows > 0 else zero
    feats = torch.stack([
        torch.log1p(ba), torch.log1p(sn),
        torch.tensor(sa, device=ay.device, dtype=dt), torch.tensor(sb, device=ay.device, dtype=dt),
        torch.tanh(da @ u_al / den) * sa, torch.tanh(da @ u_cr / den) * sa,
        torch.tanh(db @ u_al / den) * sb, torch.tanh(db @ u_cr / den) * sb,
        torch.tanh(da.norm() / den) * sa, torch.tanh(db.norm() / den) * sb,
        cosab, torch.tanh(dbar)])
    return feats                           # [KIN_DIM]


class FQDeepNative(FQDeepFaithful):
    def __init__(self, *args, oph_min_phase: float = 0.25, oph_gate: bool = True,
                 oph_weighted: bool = False, oph_aniso: bool = True, oph_kin: bool = True, **kw):
        super().__init__(*args, **kw)
        W, M = self.enc.out_features, self.M          # LazyLinear stores out_features at construction
        # eta head: pooled anchor embedding -> null-space coordinates [M, cos/sin, x/y].
        # Same init philosophy as theta_out: start AT the closed-form operator (eta ~ 0).
        self.eta_head = nn.Sequential(nn.Linear(W, W), nn.GELU(), nn.Linear(W, M * 4))
        nn.init.normal_(self.eta_head[-1].weight, std=0.02)
        nn.init.zeros_(self.eta_head[-1].bias)
        # curvature-gate sharpness (see fq_oph3.py docstring, incl. the parallel-tangent blind
        # spot).  softplus(rho) init 4.0 -> a 25% relative boundary-velocity mismatch gives
        # g ~ tanh(1) ~ 0.76.
        self.gate_rho = nn.Parameter(torch.tensor(math.log(math.expm1(4.0))))
        self.oph_min_phase = oph_min_phase
        self.oph_gate = oph_gate
        # ---- metric head: TWO Jacobi profiles (along-chord, cross-chord) on the module's
        # variety coordinate s_k, conditioned on [pool | boundary kinematics].  Same (*) Jacobi
        # mechanism as fqdeep_native.py; softplus keeps each metric > 0
        # and B c = d holds for ANY metrics under the rank/rcond caveats of fq_oph3.py,
        # so learning cannot break the theorems.  INIT: bias[0] = bias[R+1] = 1, small weights -> both profiles uniform ->
        # aniso == iso == the unweighted closed form at init.
        self.oph_weighted = oph_weighted
        self.oph_aniso = oph_aniso
        self.oph_kin = oph_kin
        n_prof = 2 if oph_aniso else 1
        in_dim = W + (KIN_DIM if oph_kin else 0)
        self.w_head = nn.Sequential(nn.Linear(in_dim, W), nn.GELU(), nn.Linear(W, n_prof * (self.R + 1)))
        nn.init.normal_(self.w_head[-1].weight, std=0.02)
        with torch.no_grad():
            self.w_head[-1].bias.zero_()
            for p in range(n_prof):
                self.w_head[-1].bias[p * (self.R + 1)] = 1.0
        # ablation switches (runner sets them on the best checkpoint; all default OFF)
        self.oph_w_const = False    # True -> ignore w_head entirely (uniform metric, GNN channel OFF)
        self.oph_iso_tie = False    # True -> drop the cross profile (isotropic solve)
        self.oph_kin_zero = False   # True -> zero the kinematics input (pool-only head)

    def forward(self, x, coord, edge_index, anchor_edge_index, batch, edge_dist=None):
        """FQ-DEEP message passing (inherited embed, verbatim) + the anisotropic Op-H readout."""
        h, xi, c, s, P, xi0 = self.embed(x, coord, edge_index)

        om_all = 2 * PI * s * xi0                              # [M] prescribed module time-frequencies
        asrc, adst = anchor_edge_index[0], anchor_edge_index[1]
        out = x[:, :2].clone()
        held_all = x[:, 5] > 0.5
        gid_adst = batch[adst]
        for gid in torch.unique(batch):
            held = torch.where(held_all & (batch == gid))[0]
            anc = torch.unique(asrc[gid_adst == gid])
            if held.numel() == 0 or anc.numel() < 2:
                continue
            at, ay = coord[anc, 0], x[anc, :2]                 # anchor times / REAL positions
            v_obs = boundary_velocity(at, ay).to(h.dtype)      # Dmu observed boundary velocity rows
            pool = h[anc].mean(0)
            eta = self.eta_head(pool).view(self.M, 2, 2)       # learned null-space coordinates
            w = wc = None
            if self.oph_weighted and not self.oph_w_const:
                if self.oph_kin:
                    kin = boundary_kin(at, ay, v_obs).to(pool.dtype)
                    if self.oph_kin_zero:
                        kin = torch.zeros_like(kin)
                    head_in = torch.cat([pool, kin])
                else:
                    head_in = pool
                n_prof = 2 if self.oph_aniso else 1
                filt = self.w_head(head_in).view(n_prof, self.R + 1) @ P.t()   # [n_prof, M]
                w = torch.nn.functional.softplus(filt[0]) + 1e-3               # along-chord metric
                if self.oph_aniso and not self.oph_iso_tie:
                    wc = torch.nn.functional.softplus(filt[1]) + 1e-3          # cross-chord metric
            out[held] = oph_readout(
                at, ay, v_obs, coord[held, 0], om_all, eta,
                gate_rho=self.gate_rho if self.oph_gate else None,
                min_phase=self.oph_min_phase, use_vel=self.use["dmu"],
                w=w, w_cross=wc).to(out.dtype)
        return out

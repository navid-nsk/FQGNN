"""task_alpha.run_ablation_ladder -- the ABLATION LADDER for the FQGNN paper (Figure-4b panel).

7 controlled degradations of the faithful FQ operator stack, each = the FULL FQDeepFaithful minus
ONE named ingredient, trained with the EXACT run_full_faithful_5seed recipe (same data/split, same
seeds 0-4, same Adam 5e-4 / clip 1.0 / bs 64 / <=60 ep / patience 8 / shuffles seed*1000+ep, same
s_raw init 2.5, same warmup, same telemetry columns, same output formats).  The model source
fqdeep_faithful.py is READ-ONLY: every ablation lives HERE via subclassing / constructor switches.

Variants (see VARIANT_DESC):
  no_dmu       Dmu velocity pathway removed at ALL 3 consumption sites (node vel features,
               message K^mu, readout Hermite rows -> position-only Op-K).
  rbf_kernel   Op-D twisted-Eberlein kernel -> generic RBF exp(-|delta|^2/2 ell^2), ONE learnable
               ell, at BOTH kernel sites (per-edge message kernel + readout Gram).
  decoupled_U  secular coupling U -> I (SecularUnitary.coupled=False), A frozen: decoupled
               amplitudes + factorized Op-B/C features + decoupled FQ-phase score.
  frozen_L     quasicrystal generator frozen at init: grassmann.log_scale + s_raw
               requires_grad=False (xi = L^T k inherits the freeze; rho envelope still learns).
  free_amps    variety-fixed c_k -> free complex nn.Parameter initialized to the faithful values
               (the "RFF-ify the amplitudes" control).
  free_amps_randinit   free amplitudes with a RANDOM magnitude-matched init (separates the variety
               VALUES from mere parameter freedom).
  d1_lift      lift collapses to scalar time: coord -> (t_norm, 0) (the periodic / d=1 ablation).
  none         the unablated full model through the same scaffolding (scaffolding guard).

Outputs per variant under task_alpha/runs/ablation_ladder/<variant>/ (or .../_smoke/<variant>/):
  seed<k>/{epochs.csv, per_gap.csv, test_buckets.csv, best.pt}
  all_seeds_epochs.csv, all_seeds_test_buckets.csv, summary.csv (+MEAN/STD)
identical formats to runs/full_faithful_5seed (per_gap via common_gaps.gap_row -> has gap_id col).

Run (from the repository root, gpubench env):
  smoke, all variants:  python -m task_alpha.run_ablation_ladder --variant all --smoke
  real, one variant:    python -m task_alpha.run_ablation_ladder --variant no_dmu --seeds 5
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import random
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn

from .baselines.common_gaps import (bucket_rows, gap_row, load_gaps, summary_row,
                                    write_aggregates, write_buckets, write_per_gap)
from .data import collate_graphs_interp
from .diag_training import GROUPS, group_of
from .fq_imputer import FQTrajectoryImputer
from .fq_opk import boundary_velocity, opk_interp
from .fqdeep_faithful import FQDeepFaithful
from .meltin_proof import _bvec, _eval
from .metrics import gcd_errors, model_errors

DEV = "cuda"
PI = math.pi
OUT_ROOT = "task_alpha/runs/ablation_ladder"
# EXACT model construction of run_full_faithful_5seed.run_seed (only `seed` varies per run)
MODEL_KW = dict(n=6, n_freqs=48, width=96, init_scale=1.84, in_feat=6, layers=3, jacobi_R=4)
VARIANTS = ("none", "no_dmu", "rbf_kernel", "decoupled_U", "frozen_L", "free_amps", "d1_lift",
            "free_amps_randinit")

VARIANT_DESC = {
    "none": "FULL model, no ablation (scaffolding guard: must train exactly like the 5-seed run)",
    "no_dmu": "Dmu OFF at all 3 sites: use['dmu']=False (node vel features -> 0, enc dim unchanged), "
              "use['vel']=False (message K^mu dead), readout -> position-only Op-K (opk_interp on the "
              "time axis; same FQ kernel/conditioning/ridge/affine drift, NO Hermite velocity rows)",
    "rbf_kernel": "Op-D twisted-Eberlein kernel -> generic RBF exp(-|d|^2/2 ell^2), ONE learnable ell "
                  "(+1 param) at BOTH sites: per-edge message kernel (K^mu -> RBF time-derivative, Dmu "
                  "path stays alive) and the readout Gram (RBF Hermite kriging, same saddle/ridge). "
                  "theta/theta_out conditioning heads become kernel-inert (no amplitudes to condition)",
    "decoupled_U": "coupling -> identity: SecularUnitary.coupled=False (U == I exactly), A_real/A_imag "
                   "frozen.  Decouples every consumer of U: amplitudes c_k (only pure single-coordinate "
                   "harmonics survive the log p_U cumulant), Op-B/C node features (theta=arg z, "
                   "p=prod(1-z_j)), and the per-layer FQ-phase attention score (M=diag(z))",
    "frozen_L": "quasicrystal generator frozen at init: grassmann.log_scale + s_raw requires_grad=False "
                "(s frozen AT the recipe init 2.5; xi=L^T k inherits the freeze).  rho (QFE envelope) "
                "still learns -- it is not the generator.  Everything else learns",
    "free_amps": "amplitudes cut loose from the variety: c_k -> free complex nn.Parameter (2M=+96 "
                 "params) initialized to the faithful log-p_U cumulant values; all downstream users "
                 "(QFE, Dmu features, kernel, readout |c_k|^2) consume the free c.  U still learns via "
                 "the score branch, L via the frequencies xi",
    "d1_lift": "lift -> scalar time: coord=(t_norm, 0) inside _feats/embed (removes the "
               "dt_to_nearest_anchor lift axis).  Equivalent to restricting every phase inner product "
               "to L[:,0]; Jacobi variety coordinate s_k=||xi_k|| untouched; readout uses coord[:,0] "
               "only, so it is unchanged by construction (the periodic / d=1 ablation)",
    "free_amps_randinit": "free amplitudes with RANDOM init (magnitude-matched complex Gaussian; "
                          "support mask also freed).  Disambiguates free_amps: variety VALUES vs "
                          "mere parameter freedom",
}


# ================================================================================================ #
#  RBF Hermite kriging (readout replacement for rbf_kernel)                                        #
# ================================================================================================ #
def rbf_hermite(at, ay, av, qt, ell, ridge: float = 0.1, ridge_v: float = 0.1):
    """Generic-RBF Hermite kriging: IDENTICAL bordered saddle structure to fq_opk.opk_hermite
    (position rows + velocity rows + affine [1,t] drift + relative Tikhonov ridge), with the FQ
    kernel K = sum_k w_k cos(omega_k (s-t)) replaced by K = exp(-(s-t)^2 / 2 ell^2).
    K(0)=1 and d1d2K(0)=1/ell^2 set the ridge scales (same relative convention as opk_hermite:
    ridge*K0 / ridge_v*Kdd0).  d = s-t: d2K = (d/ell^2) K, d1d2K = (1/ell^2 - d^2/ell^4) K
    (sign convention matches opk_hermite's +sin blocks).  Affine reproduction is preserved."""
    dt = ay.dtype
    at = at.to(dt); qt = qt.to(dt)
    M = at.shape[0]
    l2 = (ell * ell).to(dt)

    def blocks(a, b):                                       # a[Na], b[Nb] -> K00, d2K, d1d2K
        d = a.unsqueeze(1) - b.unsqueeze(0)                 # [Na, Nb]
        K = torch.exp(-(d * d) / (2.0 * l2))
        return K, (d / l2) * K, (1.0 / l2 - (d * d) / (l2 * l2)) * K

    K00, K01, K11 = blocks(at, at)
    eyeM = torch.eye(M, device=at.device, dtype=dt)
    A00 = K00 + ridge * eyeM                                # K(0) = 1  -> ridge * K0 = ridge
    A11 = K11 + ridge_v * (1.0 / l2) * eyeM                 # d1d2K(0) = 1/ell^2
    one = torch.ones_like(at); zero = torch.zeros_like(at)
    P = torch.stack([one, at], -1)                          # [M,2] drift values
    Pd = torch.stack([zero, one], -1)                       # [M,2] drift derivatives
    z2 = torch.zeros(2, 2, device=at.device, dtype=dt)
    row0 = torch.cat([A00, K01, P], dim=1)
    row1 = torch.cat([K01.t(), A11, Pd], dim=1)
    row2 = torch.cat([P.t(), Pd.t(), z2], dim=1)
    Sys = torch.cat([row0, row1, row2], dim=0)              # [2M+2, 2M+2]
    rhs = torch.cat([ay.to(dt), av.to(dt), torch.zeros(2, 2, device=at.device, dtype=dt)], dim=0)
    sol = torch.linalg.solve(Sys, rhs)
    c, e, d0 = sol[:M], sol[M:2 * M], sol[2 * M:]
    qK00, qK01, _ = blocks(qt, at)
    Pq = torch.stack([torch.ones_like(qt), qt], -1)
    return Pq @ d0 + qK00 @ c + qK01 @ e


# ================================================================================================ #
#  Shared readout scaffolding: VERBATIM copy of FQDeepFaithful.forward with ONLY the per-gap        #
#  interpolant call factored out (so variants swap the readout without editing the model source).   #
# ================================================================================================ #
def _readout_forward(model, x, coord, edge_index, anchor_edge_index, batch, readout_fn):
    h, xi, c, s, P, xi0 = model.embed(x, coord, edge_index)
    asrc, adst = anchor_edge_index[0], anchor_edge_index[1]
    out = x[:, :2].clone()
    held_all = x[:, 5] > 0.5
    gid_adst = batch[adst]
    for gid in torch.unique(batch):
        held = torch.where(held_all & (batch == gid))[0]
        anc = torch.unique(asrc[gid_adst == gid])
        if held.numel() == 0 or anc.numel() < 2:
            continue
        at, ay = coord[anc, 0], x[anc, :2]                  # anchor times / REAL positions
        if model.use["cond"]:                               # data-conditioned amplitudes (unchanged)
            filt = model.theta_out(h[anc].mean(0)) @ P.t()
            cw = filt.to(c.dtype) * c
        else:
            cw = c
        out[held] = readout_fn(model, at, ay, coord[held, 0], xi0, cw, s).to(out.dtype)
    return out


# ================================================================================================ #
#  The 7 ablation variants (subclasses / constructor switches ONLY -- model file unchanged)         #
# ================================================================================================ #
class FQNoDmu(FQDeepFaithful):
    """no_dmu -- see VARIANT_DESC.  Sites (1)+(2) via the model's own self.use switches; site (3)
    (readout Hermite velocity rows) via the factored forward: position-only opk_interp on the time
    axis with the SAME FQ kernel, conditioning, ridge and affine drift (boundary_velocity never
    computed -> the Dmu pathway is fully dead)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.use["dmu"] = False                             # node-feature velocity channel -> zeros
        self.use["vel"] = False                             # message-passing K^mu -> None
        self._abl_hits = {"readout": 0}

    def forward(self, x, coord, edge_index, anchor_edge_index, batch, edge_dist=None):
        def readout(mdl, at, ay, qt, xi0, cw, s):
            mdl._abl_hits["readout"] += 1
            return opk_interp(at.unsqueeze(-1), at, ay, qt.unsqueeze(-1), qt,
                              xi0.unsqueeze(-1), cw, s, ridge=mdl.opk_ridge)
        return _readout_forward(self, x, coord, edge_index, anchor_edge_index, batch, readout)


class FQRBFKernel(FQDeepFaithful):
    """rbf_kernel -- see VARIANT_DESC.  The per-edge kernel swap lives in _kernel (delta stashed by
    the embed wrapper: the SAME delta the base embed computes); the readout swap lives in the
    factored forward (rbf_hermite keeps the Hermite velocity rows so the Dmu pathway stays alive --
    orthogonal to no_dmu).  ONE learnable ell shared by both sites (the analog of the one shared
    spectrum).  cr/ci (theta Jacobi filters) and theta_out become kernel-inert -> grad-dead."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.rbf_log_ell = nn.Parameter(torch.tensor(math.log(math.expm1(0.25))))  # ell init 0.25
        self._abl_hits = {"msg_kernel": 0, "readout": 0}

    def _ell(self):
        return torch.nn.functional.softplus(self.rbf_log_ell) + 1e-3

    def embed(self, x, coord, edge_index):
        src, dst = edge_index[0], edge_index[1]
        self._rbf_delta = coord[dst] - coord[src]           # [E,2] = the base embed's delta, stashed
        return super().embed(x, coord, edge_index)

    def _kernel(self, cr, ci, phk, xi0=None):               # cr/ci/phk ignored: no amplitudes in RBF
        self._abl_hits["msg_kernel"] += 1
        d = self._rbf_delta
        l2 = self._ell() ** 2
        K = torch.exp(-(d * d).sum(-1) / (2.0 * l2))        # [E]
        if xi0 is None:
            return K
        Km = -(d[:, 0] / l2) * K                            # d/d(delta_t) RBF: the generic Dmu channel
        return K, Km

    def forward(self, x, coord, edge_index, anchor_edge_index, batch, edge_dist=None):
        def readout(mdl, at, ay, qt, xi0, cw, s):           # cw/xi0/s unused: RBF Gram replaces Op-D
            mdl._abl_hits["readout"] += 1
            v_obs = boundary_velocity(at, ay).to(ay.dtype)
            return rbf_hermite(at, ay, v_obs, qt, mdl._ell(), ridge=mdl.opk_ridge, ridge_v=mdl.opk_ridge)
        return _readout_forward(self, x, coord, edge_index, anchor_edge_index, batch, readout)


class FQDecoupledU(FQDeepFaithful):
    """decoupled_U -- see VARIANT_DESC.  SecularUnitary provides the switch: coupled=False
    -> unitary() returns EXACTLY I (fqmath.lee_yang: 'if U were diagonal the eigenphases decouple
    per coordinate -- the ablation, never the model').  A_real/A_imag frozen (they are disconnected
    from the graph anyway once coupled=False)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.secular.coupled = False
        self.secular.A_real.requires_grad_(False)
        self.secular.A_imag.requires_grad_(False)


class FQFreeAmps(FQDeepFaithful):
    """free_amps -- see VARIANT_DESC.  quasicrystal() is the single chokepoint through which every
    consumer (QFE, Dmu features, message kernel via _amp, readout cw) receives c, so overriding it
    replaces the amplitudes everywhere at once.  Initialized to the faithful variety-fixed values
    computed at construction (before any training) -> identical function at init, free thereafter."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        with torch.no_grad():
            _, _, c0, _ = FQTrajectoryImputer.quasicrystal(self)   # the faithful c at init
        self.c_free_re = nn.Parameter(c0.real.clone().float())
        self.c_free_im = nn.Parameter(c0.imag.clone().float())

    def quasicrystal(self):
        L, xi, c, s = super().quasicrystal()                # keeps L/xi/s handling byte-identical
        cf = torch.complex(self.c_free_re, self.c_free_im).to(c.dtype)
        return L, xi, cf, s


class FQFreeAmpsRand(FQFreeAmps):
    """free_amps_randinit -- same free amplitudes as free_amps but RANDOM init (complex Gaussian,
    L2-magnitude-matched to the faithful c; the variety SUPPORT mask is also freed since extinct
    stems get nonzero draws).  Separates the variety INIT from the amplitude CONSTRAINT: if this
    degrades where free_amps did not, the variety VALUES matter, not just the parameter freedom.
    Uses a local torch.Generator so the global RNG stream (LazyLinear init order) is unaffected."""

    def __init__(self, *a, seed=0, **k):
        super().__init__(*a, seed=seed, **k)
        with torch.no_grad():
            g = torch.Generator().manual_seed(int(seed) + 7777)
            c0 = torch.complex(self.c_free_re, self.c_free_im)
            scale = c0.abs().pow(2).sum().sqrt()
            re = torch.randn(c0.shape, generator=g)
            im = torch.randn(c0.shape, generator=g)
            cr = torch.complex(re, im)
            cr = cr * (scale / cr.abs().pow(2).sum().sqrt().clamp(min=1e-12))
            self.c_free_re.copy_(cr.real.float())
            self.c_free_im.copy_(cr.imag.float())


class FQD1Lift(FQDeepFaithful):
    """d1_lift -- see VARIANT_DESC.  Collapse applied inside _feats AND embed so every consumer of
    the lift (QFE phases, Op-B/C torus lift, Dmu features, edge deltas/phk) sees coord=(t_norm,0);
    idempotent (embed passes the collapsed coord to _feats which collapses again -> no-op)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._abl_hits = {"collapse": 0}
        self._col1_max = 0.0

    def _collapse(self, coord):
        self._abl_hits["collapse"] += 1
        c2 = torch.stack([coord[:, 0], torch.zeros_like(coord[:, 0])], dim=-1)
        self._col1_max = max(self._col1_max, float(c2[:, 1].abs().max()))
        return c2

    def _feats(self, x, coord):
        return super()._feats(x, self._collapse(coord))

    def embed(self, x, coord, edge_index):
        return super().embed(x, self._collapse(coord), edge_index)


def build_model(variant: str, seed: int):
    """EXACT construction order of run_full_faithful_5seed.run_seed (construct -> use_cuda -> s_raw
    init 2.5), then the variant's post-construction switches.  Subclass __init__s add no RNG draws,
    so the torch RNG stream at LazyLinear materialization matches the 5-seed recipe exactly."""
    cls = {"none": FQDeepFaithful, "frozen_L": FQDeepFaithful, "no_dmu": FQNoDmu,
           "rbf_kernel": FQRBFKernel, "decoupled_U": FQDecoupledU,
           "free_amps": FQFreeAmps, "d1_lift": FQD1Lift,
           "free_amps_randinit": FQFreeAmpsRand}[variant]
    m = cls(**MODEL_KW, seed=seed).to(DEV)
    m.use_cuda = True
    with torch.no_grad():
        m.s_raw.fill_(math.log(math.expm1(2.5)))
    if variant == "frozen_L":                               # freeze AT the recipe init (s = 2.5)
        m.grassmann.log_scale.requires_grad_(False)         # the only learnable dof of L
        m.s_raw.requires_grad_(False)                       # the band scale
    return m


# ================================================================================================ #
#  Sanity: frozen/replaced params, param-count diff vs full, and per-variant bite evidence          #
# ================================================================================================ #
def param_table(m):
    return {n: (tuple(p.shape), p.numel(), bool(p.requires_grad)) for n, p in m.named_parameters()}


def full_param_table(tr):
    """Reference table of the FULL (unablated) model, LazyLinear materialized via the recipe warmup."""
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    ref = build_model("none", 0)
    b0 = collate_graphs_interp(tr[:4])
    ref(b0[0].to(DEV), b0[1].to(DEV), b0[4].to(DEV), b0[6].to(DEV), _bvec(tr[:4]))
    t = param_table(ref)
    del ref
    torch.cuda.empty_cache()
    return t


def sanity_report(variant, m, ref_table, b0, tr):
    """Prints (a) the 1-line param diff vs the full model, (b) frozen/added params, (c) concrete
    per-variant bite evidence from one throwaway forward+backward on the warmup batch (grads are
    cleared afterwards; param values unchanged -> the training recipe is unaffected)."""
    vt = param_table(m)
    full_all = sum(v[1] for v in ref_table.values()); full_tr = sum(v[1] for v in ref_table.values() if v[2])
    var_all = sum(v[1] for v in vt.values()); var_tr = sum(v[1] for v in vt.values() if v[2])
    added = sorted(k for k in vt if k not in ref_table)
    frozen = sorted(k for k in vt if k in ref_table and ref_table[k][2] and not vt[k][2])
    print(f"[sanity:{variant}] {VARIANT_DESC[variant]}", flush=True)
    print(f"[sanity:{variant}] PARAM DIFF: full {full_all} ({full_tr} trainable) -> variant {var_all} "
          f"({var_tr} trainable) | added: {added if added else '-'} | frozen: {frozen if frozen else '-'}",
          flush=True)

    # ---- bite evidence: one throwaway backward ----
    x, co = b0[0].to(DEV), b0[1].to(DEV)
    ei, ae = b0[4].to(DEV), b0[6].to(DEV)
    yp, mk = b0[2].to(DEV), b0[3].to(DEV)
    m.zero_grad(set_to_none=True)
    pred = m(x, co, ei, ae, _bvec(tr[:4]))
    ((pred[mk] - yp[mk]) ** 2).mean().backward()
    msgs = []
    if variant == "none":
        msgs.append("full model through the ladder scaffolding (no switch active); "
                    f"loss finite={bool(torch.isfinite(pred).all())}")
    elif variant == "no_dmu":
        with torch.no_grad():
            node = m._feats(x, co)[0]
        M, acd = m.M, m.ac_head.out_features
        lo = 2 * M + acd + m.n + 2
        vmax = float(node[:, lo:lo + 4 * M].abs().max())
        msgs.append(f"vel-feature slice max|.| = {vmax:.2e} (must be 0.0)")
        msgs.append(f"use['dmu']={m.use['dmu']} use['vel']={m.use['vel']} (must be False/False)")
        msgs.append(f"position-only readout hits = {m._abl_hits['readout']} (must be >0; Hermite rows gone)")
    elif variant == "rbf_kernel":
        dead_t = all(p.grad is None or float(p.grad.abs().max()) == 0.0 for p in m.theta.parameters())
        dead_o = all(p.grad is None or float(p.grad.abs().max()) == 0.0 for p in m.theta_out.parameters())
        ge = m.rbf_log_ell.grad
        msgs.append(f"RBF code-path hits: msg_kernel={m._abl_hits['msg_kernel']} "
                    f"readout={m._abl_hits['readout']} (both must be >0)")
        msgs.append(f"theta heads grad-dead={dead_t}, theta_out grad-dead={dead_o} "
                    f"(kernel-inert, must be True/True)")
        msgs.append(f"ell={float(m._ell()):.4f}, grad(ell)={'None' if ge is None else f'{float(ge):.3e}'} "
                    f"(must be non-None: ell learns)")
    elif variant == "decoupled_U":
        with torch.no_grad():
            U = m.secular.unitary()
            dI = float((U - torch.eye(m.n, dtype=U.dtype, device=U.device)).abs().max())
            _, _, c, _ = m.quasicrystal()
            act = int((c.abs() > 1e-9).sum())
        a_dead = m.secular.A_real.grad is None and m.secular.A_imag.grad is None
        msgs.append(f"max|U - I| = {dI:.2e} (must be 0.0)")
        msgs.append(f"active amplitudes {act}/{m.M} (decoupled cumulant: only pure m*e_i harmonics survive)")
        msgs.append(f"secular A grads None (frozen) = {a_dead} (must be True)")
    elif variant == "frozen_L":
        gl, sr = m.grassmann.log_scale, m.s_raw
        s_val = float(torch.nn.functional.softplus(sr))
        msgs.append(f"grassmann.log_scale: requires_grad={gl.requires_grad}, grad={gl.grad} (False/None)")
        msgs.append(f"s_raw: requires_grad={sr.requires_grad}, grad={sr.grad}, s={s_val:.3f} "
                    f"(False/None, frozen at 2.5)")
    elif variant == "free_amps":
        with torch.no_grad():
            _, _, cf, _ = m.quasicrystal()
            _, _, c_faith, _ = FQTrajectoryImputer.quasicrystal(m)
            dmax = float((cf - c_faith).abs().max())
        g_ok = m.c_free_re.grad is not None and m.c_free_im.grad is not None
        msgs.append(f"max|c_free - c_faithful(init)| = {dmax:.2e} (0.0 at init, free thereafter)")
        msgs.append(f"grads flow to c_free (re&im) = {g_ok} (must be True)")
    elif variant == "free_amps_randinit":
        with torch.no_grad():
            _, _, cf, _ = m.quasicrystal()
            _, _, c_faith, _ = FQTrajectoryImputer.quasicrystal(m)
            dmax = float((cf - c_faith).abs().max())
            e_ratio = float(cf.abs().pow(2).sum() / c_faith.abs().pow(2).sum().clamp(min=1e-12))
            n_extinct_pop = int(((c_faith.abs() == 0) & (cf.abs() > 0)).sum())
        g_ok = m.c_free_re.grad is not None and m.c_free_im.grad is not None
        msgs.append(f"max|c_free - c_faithful| = {dmax:.2e} (must be LARGE: random init)")
        msgs.append(f"energy ratio ||c_rand||^2/||c_faith||^2 = {e_ratio:.3f} (must be ~1: matched)")
        msgs.append(f"extinct stems populated = {n_extinct_pop} (support mask freed)")
        msgs.append(f"grads flow to c_free (re&im) = {g_ok} (must be True)")
    elif variant == "d1_lift":
        msgs.append(f"coord-collapse hits = {m._abl_hits['collapse']} (must be >0)")
        msgs.append(f"max|coord[:,1]| seen by the lift = {m._col1_max:.2e} (must be 0.0)")
    for msg in msgs:
        print(f"[sanity:{variant}] BITE: {msg}", flush=True)
    m.zero_grad(set_to_none=True)


# ================================================================================================ #
#  run_seed: EXACT copy of run_full_faithful_5seed.run_seed (recipe unchanged; only the model class  #
#  and the output writers differ -- writers produce the identical formats via common_gaps)           #
# ================================================================================================ #
def run_seed(variant, seed, tr, va, te, out_dir, n_epochs, ref_table):
    sdir = f"{out_dir}/seed{seed}"
    os.makedirs(sdir, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    m = build_model(variant, seed)
    b0 = collate_graphs_interp(tr[:4])
    m(b0[0].to(DEV), b0[1].to(DEV), b0[4].to(DEV), b0[6].to(DEV), _bvec(tr[:4]))   # LazyLinear warmup
    sanity_report(variant, m, ref_table, b0, tr)
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
            if n in init:
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
    for ep in range(n_epochs):
        t0 = time.time()
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
        t_tr = time.time() - t0
        fq, gc, _ = _eval(m, va)
        t_va = time.time() - t0 - t_tr
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
        print(f"[{variant} seed {seed}] ep {ep:2d}: train {ep_loss / max(ep_nb, 1):.5f} | val {vr:.4f} "
              f"(best {best:.4f}) | aggfilt {fa:.3f} readout_dev {fo:.4f} "
              f"| {t_tr:.1f}s train + {t_va:.1f}s val", flush=True)
        if torch.cuda.is_available():                        # defragment between epochs: allocator
            torch.cuda.empty_cache()                         # fragmentation otherwise slows epochs, then OOM
        if pctr >= 8:
            print(f"[{variant} seed {seed}] early stop", flush=True); break

    m.load_state_dict(best_state)
    # ---- TEST: identical evaluation; per_gap/buckets via the shared common_gaps writers ----
    m.eval(); pg = []
    gi = 0
    with torch.no_grad():
        for s0 in range(0, len(te), 64):
            b = te[s0:s0 + 64]; x, coord, yp, mask, ei, ed, aei = collate_graphs_interp(b)
            pred = m(x.to(DEV), coord.to(DEV), ei.to(DEV), aei.to(DEV), _bvec(b)).float().cpu(); off = 0
            for g in b:
                ng = g.x.shape[0]; gp = pred[off:off + ng]; off += ng; fe = model_errors(gp, g)
                if len(fe):
                    pg.append(gap_row(gi, g, fe))
                gi += 1
    brows = bucket_rows(pg)
    write_per_gap(f"{sdir}/per_gap.csv", pg)
    write_buckets(f"{sdir}/test_buckets.csv", brows)
    ov = brows[0]
    print(f"\n[{variant} seed {seed}][TEST] {ov['n']} gaps OVERALL ratio_med {ov['ratio']} "
          f"(GCD {ov['gcd_med']} model {ov['model_med']} win {ov['win_pct']}%)", flush=True)
    for r in brows[1:]:
        print(f"   {r['bucket']:<11} n={r['n']:4d}  GCD {r['gcd_med']:.2f}  "
              f"model {r['model_med']:.2f}  ratio {r['ratio']:.3f}", flush=True)

    del m, opt, init, best_state
    torch.cuda.empty_cache()
    return rows, brows


def main():
    ap = argparse.ArgumentParser(description="FQGNN ablation ladder (Figure-4b)")
    ap.add_argument("--variant", required=True,
                    help=f"one of {VARIANTS}, a comma-list, or 'all'")
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds (0..N-1); recipe default 5")
    ap.add_argument("--smoke", action="store_true",
                    help="200/60/60 graphs, 2 epochs, 1 seed, outputs under _smoke/")
    args = ap.parse_args()
    variants = list(VARIANTS) if args.variant == "all" else [v.strip() for v in args.variant.split(",")]
    for v in variants:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v!r}; choose from {VARIANTS}")

    tr, va, te = load_gaps()                                # the EXACT 5350/635/774 gaps + split
    if args.smoke:
        tr, va, te = tr[:200], va[:60], te[:60]
        seeds, n_epochs, root = (0,), 2, f"{OUT_ROOT}/_smoke"
        print(f"[smoke] {len(tr)}/{len(va)}/{len(te)} graphs, {n_epochs} epochs, seeds {seeds}", flush=True)
    else:
        seeds, n_epochs, root = tuple(range(args.seeds)), 60, OUT_ROOT

    ref_table = full_param_table(tr)                        # full-model param reference (once)

    for variant in variants:
        out_dir = f"{root}/{variant}"
        os.makedirs(out_dir, exist_ok=True)
        all_ep, all_bk, summary = [], [], []
        for seed in seeds:
            print(f"\n================ ABLATION {variant} | SEED {seed} ================", flush=True)
            rows, brows = run_seed(variant, seed, tr, va, te, out_dir, n_epochs, ref_table)
            for r in rows:
                all_ep.append({"seed": seed, **r})
            for r in brows:
                all_bk.append({"seed": seed, **r})
            summary.append(summary_row(seed, brows, len(rows), min(r["val_ratio"] for r in rows)))
            write_aggregates(out_dir, all_ep, all_bk, summary)   # crash-safe after EVERY seed

        rat = np.array([s["ratio"] for s in summary], float)
        wp = np.array([s["win_pct"] for s in summary], float)
        print(f"\n=== {variant}: {len(seeds)}-SEED SUMMARY (TEST, OVERALL) ===", flush=True)
        for s in summary:
            print(f"  seed {s['seed']}: ratio {s['ratio']} | model_med {s['model_med']} "
                  f"| GCD {s['gcd_med']} | win {s['win_pct']}%", flush=True)
        print(f"  ratio   = {rat.mean():.4f} +/- {rat.std(ddof=0):.4f}", flush=True)
        print(f"  win_pct = {wp.mean():.1f} +/- {wp.std(ddof=0):.1f} %", flush=True)
        print(f"saved under {out_dir}/", flush=True)


if __name__ == "__main__":
    main()

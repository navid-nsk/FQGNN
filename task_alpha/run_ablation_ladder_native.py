"""task_alpha.run_ablation_ladder_native -- the ABLATION LADDER on the FQDeepNative (Op-H-W) base.

The ladder of task_alpha/run_ablation_ladder.py applied to FQDeepNative
(fqdeep_native.py: Op-H weighted min-norm boundary-Hermite readout, oph_weighted=True), so each
rung = the FULL Op-H-W model minus ONE named ingredient.  Training protocol is EXACTLY
run_fqnative2_5seed.py (== the run_full_faithful_5seed recipe): same data/split (load_gaps ->
5350/635/774), seeds 0-4, ctor FQDeepNative(n=6, n_freqs=48, width=96, init_scale=1.84, seed=seed,
in_feat=6, layers=3, jacobi_R=4, oph_weighted=True), use_cuda=True, s_raw init 2.5, LazyLinear
warmup forward BEFORE Adam(lr 5e-4), clip 1.0, bs 64, <=60 epochs, patience 8, shuffles
seed*1000+ep.  Output formats are byte-compatible with the Op-K ladder (per-seed epochs.csv /
per_gap.csv / test_buckets.csv / best.pt + all_seeds_*.csv + summary.csv via common_gaps writers).
The model sources fqdeep_native.py / fq_oph.py / fqdeep_faithful.py are READ-ONLY: every ablation
lives HERE via subclassing / constructor switches.

THE FULL ("none") RUNG IS NOT TRAINED HERE: the unablated FQDeepNative 5-seed run is
task_alpha/runs/fqnative2_5seed/ (run_fqnative2_5seed.py) -- the figure assembler reads the full
rung FROM THERE.
("none" is therefore not a runnable variant of this script; it is only used internally as the
param-count reference.)

Semantic changes vs the Op-K ladder (documented per class, summarized here):
  no_dmu       REDEFINED for the native readout: use['dmu']=False zeroes the node vel features AND
               (via fqdeep_native.forward's use_vel=self.use['dmu']) drops the boundary-Hermite
               VELOCITY rows of Op-H; use['vel']=False kills the message K^mu -- same 3 sites as
               in the Op-K ladder, but site 3 needs no readout swap (the flag is already plumbed).
               Side effect that IS part of the Dmu pathway: with no velocity rows the curvature
               gate has no mismatch evidence -> g = 1 (gate inert; see fq_oph.py).
  rbf_kernel   The Op-K ladder swaps BOTH kernel sites (per-edge message kernel + Op-K readout
               Gram).  UNDER OP-H THERE IS NO READOUT GRAM -- the readout is not kernel-based (the
               reconstruction IS an FQ exponential sum), so site 2 is N/A and the readout stays
               Op-H-W unchanged.  This rung therefore isolates the MESSAGE-kernel ablation only.
  no_metric    native-only rung: oph_w_const=True for the WHOLE run -> w_head ignored (frozen), the
               readout is the UNIFORM-metric closed form + eta.  Removes the learned
               spectral-metric GNN channel.
  no_eta       native-only rung: eta_head[-1] weight+bias zeroed at init and the WHOLE eta_head
               frozen -> eta == 0 all run: the readout is the pure (weighted) min-norm
               boundary-Hermite interpolant.  Removes the learned null-space maneuver channel.
  (decoupled_U, frozen_L, free_amps, free_amps_randinit, d1_lift are pre-readout ablations --
   same override bodies as in the Op-K ladder, subclassing FQDeepNative.)

NOTE on inherited-but-inert parts under the native readout: theta_out (the Op-K amplitude
conditioning head) exists on every FQDeepNative but is never called by the native forward -> it is
grad-dead in ALL rungs including the full model; the epochs.csv column readout_cond_dev is kept
for schema compatibility but is a constant-at-init diagnostic here.

Outputs per variant under task_alpha/runs/ablation_ladder_native/<variant>/ (or .../_smoke/<v>/):
  seed<k>/{epochs.csv, per_gap.csv, test_buckets.csv, best.pt}
  all_seeds_epochs.csv, all_seeds_test_buckets.csv, summary.csv (+MEAN/STD)
identical formats to runs/ablation_ladder (per_gap via common_gaps.gap_row -> has gap_id col).
With FQ_LADDER_SOBOLEV=1 the root is task_alpha/runs/ablation_ladder_native_sobolev/ (full rung
read from runs/fqnative_sobolev_5seed), and with FQ_READOUT_MODE=opm as well it is
task_alpha/runs/ablation_ladder_native_sobolev_opm/ (full rung: runs/fqnative_opmstack_5seed).

Run (from the repository root, gpubench env):
  verify (CPU, no data): python -m task_alpha.run_ablation_ladder_native --verify
  smoke, all variants:   python -m task_alpha.run_ablation_ladder_native --variant all --smoke
  real, one variant:     python -m task_alpha.run_ablation_ladder_native --variant no_dmu --seeds 5
  resumable queue:       python -m task_alpha.run_ablation_ladder_native --queue
                         (all variants; (variant, seed) pairs with an existing seed<k>/test_buckets.csv
                          are skipped and their aggregates rebuilt from the saved CSVs)
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
from .fqdeep_native import FQDeepNative
from .meltin_proof import _bvec, _eval


def _eval_ladder(m, graphs, bs=64):
    """meltin_proof._eval mirrored verbatim + the capability-gated blend gap_minutes kwarg
    (sobolev mode); byte-identical behavior for fd models (kwarg dict is empty)."""
    import numpy as _np
    from .experiment_compare import _sinuosity as _sin
    from .metrics import gcd_errors as _ge, model_errors as _me
    m.eval(); fq, gc, sn = [], [], []
    with torch.no_grad():
        for s in range(0, len(graphs), bs):
            b = graphs[s:s + bs]
            x, coord, yp, mask, ei, ed, aei = collate_graphs_interp(b)
            pred = m(x.to(DEV), coord.to(DEV), ei.to(DEV), aei.to(DEV), _bvec(b),
                     **_blend_kw(m, b)).float().cpu()
            off = 0
            for g in b:
                ng = g.x.shape[0]; gp = pred[off:off + ng]; off += ng
                fe = _me(gp, g)
                if len(fe):
                    fq.append(_np.median(fe)); gc.append(_np.median(_ge(g)))
                    sn.append(_sin(g))
    return _np.array(fq), _np.array(gc), _np.array(sn)
from .metrics import model_errors

DEV = "cuda"
PI = math.pi
OUT_ROOT = "task_alpha/runs/ablation_ladder_native"
FULL_RUN = "task_alpha/runs/fqnative2_5seed"     # the unablated full-model 5-seed (NOT trained here)
# EXACT model construction of run_fqnative2_5seed.run_seed (only `seed` varies per run)
MODEL_KW = dict(n=6, n_freqs=48, width=96, init_scale=1.84, in_feat=6, layers=3, jacobi_R=4,
                oph_weighted=True)

# ---- SOBOLEV mode (FQ_LADDER_SOBOLEV=1): the blend-Dmu
# + omega^-2 bending-metric config (FQDeepNative(..., dmu_source="blend", metric_backbone=
# "sobolev2"); 5-seed reference runs/fqnative_sobolev_5seed).  The flag
#   (a) adds the two ctor kwargs to EVERY rung via MODEL_KW (subclass switches unchanged),
#   (b) plumbs gap_minutes into every model forward via _blend_kw (the run_gap_sweep pattern,
#       capability-gated on dmu_source -- default fd models get {} -> byte-identical behavior),
#   (c) redirects OUT_ROOT/FULL_RUN to the _sobolev siblings so the fd ladder results at
#       runs/ablation_ladder_native are never overwritten.
# NO rung logic or training-protocol change of any kind.
SOBOLEV = os.environ.get("FQ_LADDER_SOBOLEV") == "1"
if SOBOLEV:
    OUT_ROOT = "task_alpha/runs/ablation_ladder_native_sobolev"
    FULL_RUN = "task_alpha/runs/fqnative_sobolev_5seed"
    MODEL_KW = dict(MODEL_KW, dmu_source="blend", metric_backbone="sobolev2")

# ---- OP-M STACK (FQ_READOUT_MODE=opm).  The readout switch lives in
# FQDeepNative (env-read at construction; MODEL_KW passes no readout kwarg, and the opm_* params
# are created LAST in __init__ so the RNG stream of every other parameter of each rung is the
# same).  Here we ONLY repoint the roots: the sobolev Op-H ladder at ablation_ladder_native_sobolev
# is never overwritten, and the full-rung reference becomes the Op-M stack five-seed run.  Default
# and sobolev-non-opm invocations use the roots set above.
OPM = os.environ.get("FQ_READOUT_MODE") == "opm"
if OPM and SOBOLEV:
    OUT_ROOT += "_opm"
    FULL_RUN = "task_alpha/runs/fqnative_opmstack_5seed"

VARIANTS = ("no_dmu", "rbf_kernel", "decoupled_U", "frozen_L", "free_amps", "d1_lift",
            "free_amps_randinit", "no_metric", "no_eta")


def _blend_kw(m, b):
    """gap_minutes kwarg iff the model's Dmu source is the blend jet (run_gap_sweep pattern;
    FQDeepNative-only kwarg).  Capability-gated: default fd models return {} -> with
    FQ_LADDER_SOBOLEV unset every forward call below receives no extra kwarg."""
    if getattr(m, "dmu_source", "fd") != "blend":
        return {}
    return {"gap_minutes": torch.tensor([float(g.gap_minutes) for g in b])}

VARIANT_DESC = {
    "none": "FULL model -- NOT retrained by this script; the assembler reads the full rung from "
            f"{FULL_RUN}/ (used internally only as the param-count reference)",
    "no_dmu": "Dmu OFF at all 3 native sites: use['dmu']=False (node vel features -> 0, enc dim "
              "unchanged; ALSO plumbed by fqdeep_native.forward as use_vel into oph_readout -> the "
              "boundary-Hermite VELOCITY rows drop, value rows only), use['vel']=False (message "
              "K^mu dead).  With no velocity rows the curvature gate has no evidence -> g=1 (inert)",
    "rbf_kernel": "Op-D twisted-Eberlein MESSAGE kernel -> generic RBF exp(-|d|^2/2 ell^2), ONE "
                  "learnable ell (+1 param).  READOUT SITE IS N/A UNDER OP-H (no readout Gram "
                  "exists -- the Op-H reconstruction IS the FQ exponential sum); readout stays "
                  "Op-H-W.  theta (per-layer Jacobi filter) heads become kernel-inert (grad-dead)",
    "decoupled_U": "coupling -> identity: SecularUnitary.coupled=False (U == I exactly), A_real/A_imag "
                   "frozen.  Decouples every consumer of U: amplitudes c_k (only pure single-coordinate "
                   "harmonics survive the log p_U cumulant), Op-B/C node features (theta=arg z, "
                   "p=prod(1-z_j)), and the per-layer FQ-phase attention score (M=diag(z))",
    "frozen_L": "quasicrystal generator frozen at init: grassmann.log_scale + s_raw requires_grad=False "
                "(s frozen AT the recipe init 2.5; xi=L^T k inherits the freeze, and so do the Op-H "
                "prescribed frequencies omega_k = 2 pi s xi_t).  rho (QFE envelope) still learns -- "
                "it is not the generator.  Everything else learns",
    "free_amps": "amplitudes cut loose from the variety: c_k -> free complex nn.Parameter (2M=+96 "
                 "params) initialized to the faithful log-p_U cumulant values; all remaining consumers "
                 "(QFE, Dmu features, message kernel) consume the free c.  NOTE: under Op-H the "
                 "READOUT does not consume c (that Op-K site is gone); U still learns via the score "
                 "branch, L via the frequencies xi",
    "d1_lift": "lift -> scalar time: coord=(t_norm, 0) inside _feats/embed (removes the "
               "dt_to_nearest_anchor lift axis).  Equivalent to restricting every phase inner product "
               "to L[:,0]; Jacobi variety coordinate s_k=||xi_k|| untouched; the Op-H readout uses "
               "coord[:,0] (times) only, so it is unchanged by construction (the periodic / d=1 "
               "ablation)",
    "free_amps_randinit": "free amplitudes with RANDOM init (magnitude-matched complex Gaussian; "
                          "support mask also freed).  Disambiguates free_amps: variety VALUES vs "
                          "mere parameter freedom",
    "no_metric": "learned SPECTRAL METRIC channel removed: oph_w_const=True for "
                 "the whole run (w_head ignored + frozen) -> the readout is the UNIFORM-metric "
                 "min-norm closed form + eta.  This is the GNN channel (Jacobi filter on the "
                 "module) trained OUT for the full run, not just flipped at eval",
    "no_eta": "learned NULL-SPACE maneuver channel removed: eta_head[-1] "
              "weight+bias zeroed at init and ALL eta_head params requires_grad=False -> eta == 0 "
              "for the whole run; the readout is the pure (weighted) min-norm boundary-Hermite FQ "
              "interpolant.  The curvature gate scales only the eta term, so it is inert here "
              "(gate_rho gets exactly-zero grads)",
}


# ================================================================================================ #
#  The ablation variants (subclasses / constructor switches ONLY -- model files unchanged).         #
#  All subclass FQDeepNative directly: forward is FQDeepNative.forward (Op-H-W readout) and the     #
#  overridden methods (embed/_feats/_kernel/quasicrystal) are inherited by FQDeepNative from        #
#  FQDeepFaithful unmodified, so each override body is the same as in the Op-K ladder.              #
#  No subclass __init__ draws from the global torch RNG -> the LazyLinear materialization stream    #
#  matches the run_fqnative2_5seed recipe exactly.                                                  #
# ================================================================================================ #
class FQNativeNoDmu(FQDeepNative):
    """no_dmu -- see VARIANT_DESC.  NATIVE mapping of the 3 sites of the Op-K ladder's FQNoDmu:
      (1) node vel features -> 0:        self.use['dmu']=False, consumed by the inherited _feats;
      (2) message-passing K^mu -> dead:  self.use['vel']=False, consumed by the inherited embed;
      (3) readout velocity rows -> gone: fqdeep_native.forward passes use_vel=self.use['dmu'] into
          oph_readout, so the boundary-Hermite VELOCITY rows drop automatically -- unlike the Op-K
          ladder, NO forward/readout override is needed (the flag is already plumbed).
    Documented side effect (part of the Dmu pathway, fq_oph.py): with zero included velocity rows
    the curvature gate has no mismatch evidence and returns g = 1 (gate inert)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.use["dmu"] = False                             # node-feature velocity channel -> zeros
        self.use["vel"] = False                             # message-passing K^mu -> None


class FQNativeRBFKernel(FQDeepNative):
    """rbf_kernel -- see VARIANT_DESC.  SEMANTIC DIFFERENCE vs the Op-K ladder, stated plainly:
    the Op-K ladder's FQRBFKernel replaces the kernel at BOTH sites (per-edge message kernel + the
    Op-K readout Gram).  Under the native readout SITE 2 DOES NOT EXIST: Op-H has no Gram/kriging
    smoother to re-kernel -- its output IS an FQ exponential sum on the module, so an 'RBF Op-H'
    would delete the readout rather than swap its kernel.  Implemented here: the MESSAGE-kernel swap
    only (override body identical to FQRBFKernel: delta stashed by the embed wrapper -- the SAME
    delta the base embed computes -- and _kernel replaced); forward stays FQDeepNative.forward
    (Op-H-W readout unchanged).  ONE learnable ell.  The per-layer theta Jacobi-filter heads become
    kernel-inert -> grad-dead (theta_out is grad-dead in ALL native rungs; see module docstring)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.rbf_log_ell = nn.Parameter(torch.tensor(math.log(math.expm1(0.25))))  # ell init 0.25
        self._abl_hits = {"msg_kernel": 0}

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


class FQNativeDecoupledU(FQDeepNative):
    """decoupled_U -- see VARIANT_DESC.  Override body identical to FQDecoupledU.  SecularUnitary
    provides the switch: coupled=False -> unitary() returns EXACTLY I (fqmath.lee_yang:
    'if U were diagonal the eigenphases decouple per coordinate -- the ablation, never the model').
    A_real/A_imag frozen (they are disconnected from the graph anyway once coupled=False)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.secular.coupled = False
        self.secular.A_real.requires_grad_(False)
        self.secular.A_imag.requires_grad_(False)


class FQNativeFreeAmps(FQDeepNative):
    """free_amps -- see VARIANT_DESC.  Override body identical to FQFreeAmps.  quasicrystal() is
    the single chokepoint through which every remaining consumer (QFE, Dmu features, message kernel
    via _amp) receives c, so overriding it replaces the amplitudes everywhere at once.  (The Op-K
    readout consumer 'cw' of the Op-K ladder does not exist under Op-H.)  Initialized to the
    faithful variety-fixed values computed at construction (before any training) -> identical
    function at init, free thereafter."""

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


class FQNativeFreeAmpsRand(FQNativeFreeAmps):
    """free_amps_randinit -- see VARIANT_DESC.  Override body identical to FQFreeAmpsRand: same
    free amplitudes as free_amps but RANDOM init (complex Gaussian, L2-magnitude-matched to the
    faithful c; the variety SUPPORT mask is also freed since extinct stems get nonzero draws).
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


class FQNativeD1Lift(FQDeepNative):
    """d1_lift -- see VARIANT_DESC.  Override body identical to FQD1Lift.  Collapse applied inside
    _feats AND embed so every consumer of the lift (QFE phases, Op-B/C torus lift, Dmu features,
    edge deltas/phk) sees coord=(t_norm,0); idempotent (embed passes the collapsed coord to _feats
    which collapses again -> no-op).  The Op-H readout reads coord[:,0] (times) only, so it is
    unchanged by construction -- as in the Op-K ladder."""

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


class FQNativeNoMetric(FQDeepNative):
    """no_metric -- native-only rung, see VARIANT_DESC.  oph_w_const=True from construction: the
    inherited forward then passes w=None into oph_readout -> the UNIFORM-metric min-norm closed
    form (+ eta channel, which still learns).  w_head is never called, so it is grad-dead; it is
    ALSO explicitly frozen here so the param table states the ablation.  This trains the GNN
    spectral-metric channel OUT for the whole run (vs run_fqnative2_5seed's eval-only flip)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.oph_w_const = True                             # uniform metric for the WHOLE run
        for p in self.w_head.parameters():                  # unused anyway; frozen for explicitness
            p.requires_grad_(False)


class FQNativeNoEta(FQDeepNative):
    """no_eta -- native-only rung, see VARIANT_DESC.  eta_head's LAST layer weight AND bias are
    zeroed at init -> eta == 0 identically (exact, input-independent), and ALL eta_head parameters
    are frozen (requires_grad=False) so nothing can revive the channel; grads that reach the zeroed
    frozen layer are discarded by autograd (leaf excluded), and the upstream eta_head[0] grads are
    exactly zero anyway (chain through a zero matrix).  The readout is then the pure (weighted)
    min-norm boundary-Hermite interpolant: c = c_part.  The curvature gate multiplies only the
    (now zero) eta term, so gate_rho receives exactly-zero grads (inert, part of this channel)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        with torch.no_grad():
            self.eta_head[-1].weight.zero_()
            self.eta_head[-1].bias.zero_()
        for p in self.eta_head.parameters():
            p.requires_grad_(False)


CLASSES = {"none": FQDeepNative, "frozen_L": FQDeepNative, "no_dmu": FQNativeNoDmu,
           "rbf_kernel": FQNativeRBFKernel, "decoupled_U": FQNativeDecoupledU,
           "free_amps": FQNativeFreeAmps, "d1_lift": FQNativeD1Lift,
           "free_amps_randinit": FQNativeFreeAmpsRand, "no_metric": FQNativeNoMetric,
           "no_eta": FQNativeNoEta}


def build_model(variant: str, seed: int, dev: str = DEV):
    """EXACT construction order of run_fqnative2_5seed.run_seed (construct -> use_cuda -> s_raw
    init 2.5), then the variant's post-construction switches.  Subclass __init__s add no RNG draws,
    so the torch RNG stream at LazyLinear materialization matches the 5-seed recipe exactly.
    `dev` exists only for the CPU --verify path; training always uses the default DEV."""
    m = CLASSES[variant](**MODEL_KW, seed=seed).to(dev)
    m.use_cuda = dev == "cuda"
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
    """Reference table of the FULL (unablated) FQDeepNative, LazyLinear materialized via the recipe
    warmup.  This is the ONLY use of the 'none' variant in this script (the full rung's trained
    results live at FULL_RUN and are not reproduced here)."""
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    ref = build_model("none", 0)
    b0 = collate_graphs_interp(tr[:4])
    ref(b0[0].to(DEV), b0[1].to(DEV), b0[4].to(DEV), b0[6].to(DEV), _bvec(tr[:4]),
        **_blend_kw(ref, tr[:4]))
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
    pred = m(x, co, ei, ae, _bvec(tr[:4]), **_blend_kw(m, tr[:4]))
    ((pred[mk] - yp[mk]) ** 2).mean().backward()
    msgs = []
    if variant == "no_dmu":
        with torch.no_grad():
            node = m._feats(x, co)[0]
        M, acd = m.M, m.ac_head.out_features
        lo = 2 * M + acd + m.n + 2
        vmax = float(node[:, lo:lo + 4 * M].abs().max())
        msgs.append(f"vel-feature slice max|.| = {vmax:.2e} (must be 0.0)")
        msgs.append(f"use['dmu']={m.use['dmu']} use['vel']={m.use['vel']} (must be False/False; "
                    f"forward passes use_vel=use['dmu'] -> Op-H velocity rows dropped, gate g=1)")
    elif variant == "rbf_kernel":
        dead_t = all(p.grad is None or float(p.grad.abs().max()) == 0.0 for p in m.theta.parameters())
        ge = m.rbf_log_ell.grad
        g_eta = m.eta_head[-1].weight.grad
        msgs.append(f"RBF code-path hits: msg_kernel={m._abl_hits['msg_kernel']} (must be >0); "
                    f"readout site N/A under Op-H (no Gram; readout stays Op-H-W)")
        msgs.append(f"theta heads grad-dead={dead_t} (kernel-inert, must be True)")
        msgs.append(f"ell={float(m._ell()):.4f}, grad(ell)={'None' if ge is None else f'{float(ge):.3e}'} "
                    f"(must be non-None: ell learns)")
        msgs.append(f"Op-H readout alive: eta_head grad non-None = {g_eta is not None} (must be True)")
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
                    f"(False/None, frozen at 2.5; Op-H frequencies omega=2 pi s xi_t inherit the freeze)")
    elif variant == "free_amps":
        with torch.no_grad():
            _, _, cf, _ = m.quasicrystal()
            _, _, c_faith, _ = FQTrajectoryImputer.quasicrystal(m)
            dmax = float((cf - c_faith).abs().max())
        g_ok = m.c_free_re.grad is not None and m.c_free_im.grad is not None
        msgs.append(f"max|c_free - c_faithful(init)| = {dmax:.2e} (0.0 at init, free thereafter)")
        msgs.append(f"grads flow to c_free (re&im) = {g_ok} (must be True; via QFE/Dmu feats + msg kernel)")
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
    elif variant == "no_metric":
        w_frozen = all(not p.requires_grad for p in m.w_head.parameters())
        w_dead = all(p.grad is None for p in m.w_head.parameters())
        g_eta = m.eta_head[-1].weight.grad
        msgs.append(f"oph_w_const={m.oph_w_const} (must be True: uniform metric for the whole run)")
        msgs.append(f"w_head frozen={w_frozen}, grad-dead={w_dead} (must be True/True)")
        msgs.append(f"eta channel still learns: eta_head grad non-None = {g_eta is not None} (must be True)")
    elif variant == "no_eta":
        e_frozen = all(not p.requires_grad for p in m.eta_head.parameters())
        wmax = float(m.eta_head[-1].weight.abs().max()); bmax = float(m.eta_head[-1].bias.abs().max())
        gr = m.gate_rho.grad
        gr_zero = gr is None or float(gr.abs().max()) == 0.0
        gw = m.w_head[-1].weight.grad
        msgs.append(f"eta_head ALL params requires_grad=False = {e_frozen} (must be True)")
        msgs.append(f"eta_head[-1] max|weight| = {wmax:.1e}, max|bias| = {bmax:.1e} "
                    f"(must be 0.0 -> eta == 0 identically)")
        msgs.append(f"gate_rho grad zero-or-None = {gr_zero} (gate scales only the eta term -> inert)")
        msgs.append(f"metric channel still learns: w_head grad non-None = {gw is not None} (must be True)")
    for msg in msgs:
        print(f"[sanity:{variant}] BITE: {msg}", flush=True)
    m.zero_grad(set_to_none=True)


# ================================================================================================ #
#  run_seed: EXACT copy of the Op-K ladder run_seed (itself the run_full_faithful_5seed recipe ==   #
#  the run_fqnative2_5seed protocol); only the model classes differ.  Writers via common_gaps.      #
# ================================================================================================ #
def run_seed(variant, seed, tr, va, te, out_dir, n_epochs, ref_table):
    sdir = f"{out_dir}/seed{seed}"
    os.makedirs(sdir, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    m = build_model(variant, seed)
    b0 = collate_graphs_interp(tr[:4])
    m(b0[0].to(DEV), b0[1].to(DEV), b0[4].to(DEV), b0[6].to(DEV), _bvec(tr[:4]),
      **_blend_kw(m, tr[:4]))   # LazyLinear warmup
    sanity_report(variant, m, ref_table, b0, tr)
    init = {n: p.detach().clone() for n, p in m.named_parameters()}
    opt = torch.optim.Adam(m.parameters(), lr=5e-4)

    db = tr[:64]; dx, dco, dyp, dmk, dei, _, dae = collate_graphs_interp(db)
    DX, DCO, DEI, DAE, DBV = dx.to(DEV), dco.to(DEV), dei.to(DEV), dae.to(DEV), _bvec(db)
    DMK, DYT = dmk.to(DEV), dyp.to(DEV)

    DKW = _blend_kw(m, db)

    def telemetry():
        m.train(); opt.zero_grad()
        (((m(DX, DCO, DEI, DAE, DBV, **DKW)[DMK] - DYT[DMK]) ** 2).mean()).backward()
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
            pred = m(x.to(DEV), co.to(DEV), ei.to(DEV), ae.to(DEV), _bvec(b), **_blend_kw(m, b))
            L = ((pred[mk.to(DEV)] - yp.to(DEV)[mk.to(DEV)]) ** 2).mean()
            opt.zero_grad()
            if torch.isfinite(L):
                L.backward()
                gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                # finite-grad guard: with a non-finite grad norm, clip_grad_norm_
                # rescales EVERY grad to NaN and the step would poison the model (NaN params ->
                # non-finite B~ -> a failing pinv at the next forward).  Skipping the step leaves
                # finite-grad behavior BYTE-IDENTICAL (clip_coef unchanged).
                if torch.isfinite(gn):
                    opt.step()
                ep_loss += float(L.detach()); ep_nb += 1
        t_tr = time.time() - t0
        fq, gc, _ = _eval_ladder(m, va)
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
            pred = m(x.to(DEV), coord.to(DEV), ei.to(DEV), aei.to(DEV), _bvec(b),
                     **_blend_kw(m, b)).float().cpu(); off = 0
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


def _load_seed_csvs(sdir):
    """Rebuild (rows, brows) for the aggregate writers from a COMPLETED seed dir (queue resume).
    test_buckets.csv is written last in run_seed, so its presence marks seed completion."""
    with open(f"{sdir}/epochs.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["val_ratio"] = float(r["val_ratio"])              # summary_row needs the numeric min
    with open(f"{sdir}/test_buckets.csv", newline="") as f:
        brows = list(csv.DictReader(f))
    return rows, brows


# ================================================================================================ #
#  --verify: synthetic-batch smoke of EVERY variant class (CPU, no dataset, no training)            #
# ================================================================================================ #
def verify():
    """Constructs + forwards + backwards every variant on the _wiring_test3 synthetic one-gap graph
    (N=20, 8 anchors left / 4 held / 8 anchors right).  Asserts finiteness, gradient flow, and the
    variant-defining invariants (no_metric == w_const output; no_eta frozen; etc.).  CPU-only."""
    print("[verify] synthetic-batch smoke of every native variant class (CPU, no dataset)", flush=True)
    N = 20
    tt = torch.cat([torch.arange(8.0) * 0.2, 1.4 + torch.arange(4.0) * 0.5,
                    3.4 + torch.arange(8.0) * 0.2])
    truth = torch.stack([torch.sin(0.7 * tt), 0.5 * tt + 0.3 * torch.cos(tt)], dim=-1)
    x = torch.zeros(N, 6)
    x[:, :2] = truth
    held = torch.zeros(N, dtype=torch.bool); held[8:12] = True
    x[held, :2] = 0.0                                       # held positions blanked (leak guard)
    x[:, 5] = held.float()
    coord = torch.zeros(N, 2); coord[:, 0] = tt; coord[:, 1] = 0.1
    src = torch.arange(N - 1)
    ei = torch.stack([torch.cat([src, src + 1]), torch.cat([src + 1, src])])
    anc = torch.where(~held)[0]
    ae = torch.stack([anc.repeat_interleave(4), torch.where(held)[0].repeat(len(anc))])
    batch = torch.zeros(N, dtype=torch.long)

    n_pass = 0
    fails = []
    for v in VARIANTS:
        try:
            torch.manual_seed(0); np.random.seed(0); random.seed(0)
            m = build_model(v, seed=0, dev="cpu")
            # synthetic graph has no RunGraph.gap_minutes -> synthesize the blend kwarg (1 graph,
            # a nominal 6h gap); capability-gated exactly like _blend_kw
            vkw = {"gap_minutes": torch.tensor([360.0])} \
                if getattr(m, "dmu_source", "fd") == "blend" else {}
            pred = m(x, coord, ei, ae, batch, **vkw)
            assert torch.isfinite(pred).all(), "non-finite forward"
            loss = ((pred[held] - truth[held]) ** 2).mean()
            assert torch.isfinite(loss), "non-finite loss"
            loss.backward()
            gs = [p.grad for p in m.parameters() if p.grad is not None]
            assert gs, "no gradients produced"
            assert all(torch.isfinite(g).all() for g in gs), "non-finite gradients"
            extra = f"grads on {len(gs)} tensors"
            if v == "no_metric":
                torch.manual_seed(0); np.random.seed(0); random.seed(0)
                full = build_model("none", seed=0, dev="cpu")
                full.oph_w_const = True
                pf = full(x, coord, ei, ae, batch, **vkw)
                d = float((pred.detach()[held] - pf.detach()[held]).abs().max())
                assert d < 1e-6, f"no_metric output != full-model w_const output (max diff {d:.2e})"
                assert all(p.grad is None for p in m.w_head.parameters()), "w_head got grads"
                extra += f" | == w_const output (max diff {d:.1e}) | w_head grad-dead"
            elif v == "no_eta":
                assert all(not p.requires_grad for p in m.eta_head.parameters()), \
                    "eta_head params not all requires_grad=False"
                assert float(m.eta_head[-1].weight.abs().max()) == 0.0, "eta weight not zero"
                assert float(m.eta_head[-1].bias.abs().max()) == 0.0, "eta bias not zero"
                gr = m.gate_rho.grad
                assert gr is None or float(gr.abs().max()) == 0.0, "gate_rho grad nonzero"
                extra += " | eta_head frozen+zeroed, gate_rho grad zero-or-None"
            elif v == "no_dmu":
                assert m.use["dmu"] is False and m.use["vel"] is False, "use flags not off"
                with torch.no_grad():
                    node = m._feats(x, coord)[0]
                M, acd = m.M, m.ac_head.out_features
                lo = 2 * M + acd + m.n + 2
                vmax = float(node[:, lo:lo + 4 * M].abs().max())
                assert vmax == 0.0, f"vel feature slice nonzero ({vmax:.2e})"
                extra += " | vel feats 0, flags off (readout rows drop via use_vel)"
            elif v == "rbf_kernel":
                assert m._abl_hits["msg_kernel"] > 0, "RBF message kernel never hit"
                assert m.rbf_log_ell.grad is not None, "ell got no grad"
                assert m.eta_head[-1].weight.grad is not None, "Op-H readout dead"
                extra += f" | msg_kernel hits {m._abl_hits['msg_kernel']}, ell learns, Op-H alive"
            elif v == "frozen_L":
                assert not m.grassmann.log_scale.requires_grad, "log_scale not frozen"
                assert not m.s_raw.requires_grad, "s_raw not frozen"
                extra += " | L generator + s frozen"
            elif v == "d1_lift":
                assert m._abl_hits["collapse"] > 0, "collapse never hit"
                assert m._col1_max == 0.0, "lift axis not zeroed"
                extra += f" | collapse hits {m._abl_hits['collapse']}, col1 max 0.0"
            elif v == "decoupled_U":
                with torch.no_grad():
                    U = m.secular.unitary()
                    dI = float((U - torch.eye(m.n, dtype=U.dtype)).abs().max())
                assert dI == 0.0, f"U != I (max diff {dI:.2e})"
                assert m.secular.A_real.grad is None and m.secular.A_imag.grad is None
                extra += " | U == I exactly, A frozen"
            elif v in ("free_amps", "free_amps_randinit"):
                assert m.c_free_re.grad is not None and m.c_free_im.grad is not None, \
                    "no grads to c_free"
                extra += " | grads flow to c_free"
            print(f"[verify] {v:<19} PASS  ({extra})", flush=True)
            n_pass += 1
        except Exception as e:                              # noqa: BLE001 -- report every variant
            print(f"[verify] {v:<19} FAIL  ({type(e).__name__}: {e})", flush=True)
            fails.append(v)
        finally:
            try:
                del m
            except NameError:
                pass
    print(f"[verify] {n_pass}/{len(VARIANTS)} variants PASS"
          + (f"; FAILED: {fails}" if fails else ""), flush=True)
    if fails:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description="FQGNN ablation ladder: FQDeepNative (Op-H-W) base")
    ap.add_argument("--variant", default=None,
                    help=f"one of {VARIANTS}, a comma-list, or 'all'")
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds (0..N-1); recipe default 5")
    ap.add_argument("--smoke", action="store_true",
                    help="200/60/60 graphs, 2 epochs, 1 seed, outputs under _smoke/")
    ap.add_argument("--queue", action="store_true",
                    help="run ALL variants, resumable: skip (variant, seed) pairs whose "
                         "seed<k>/test_buckets.csv already exists (aggregates rebuilt from disk)")
    ap.add_argument("--verify", action="store_true",
                    help="synthetic-batch smoke of every variant class (CPU, no dataset/training)")
    args = ap.parse_args()
    if SOBOLEV:
        print(f"[sobolev] FQ_LADDER_SOBOLEV=1: ctor +dmu_source='blend', metric_backbone="
              f"'sobolev2'; gap_minutes plumbed via _blend_kw; OUT_ROOT={OUT_ROOT}; "
              f"full rung reference {FULL_RUN}", flush=True)
    if OPM and SOBOLEV:
        print(f"[opm] FQ_READOUT_MODE=opm stack: OUT_ROOT={OUT_ROOT} | full rung {FULL_RUN} | "
              f"lam={os.environ.get('FQ_OPM_LAM', 'head')} "
              f"gamma={os.environ.get('FQ_OPM_GAMMA', '(raw dial)')} "
              f"amp={os.environ.get('FQ_AMP_MODE', 'secular')}/"
              f"nmc={os.environ.get('FQ_AMP_NMC', '')}/refresh={os.environ.get('FQ_AMP_REFRESH', '0')} "
              f"| gamma'-law constants (VBAR/MBAR/BETA2) are GLOBAL BY DESIGN: same law in every "
              f"cell/rung, no per-cell tuning", flush=True)
    if args.verify:
        verify()
        return
    if args.queue:
        variants = list(VARIANTS)
    elif args.variant:
        variants = list(VARIANTS) if args.variant == "all" else \
            [v.strip() for v in args.variant.split(",")]
    else:
        ap.error("one of --variant, --queue, --verify is required")
    for v in variants:
        if v == "none":
            raise SystemExit(f"the full ('none') rung is NOT retrained by this script -- the "
                             f"unablated FQDeepNative 5-seed lives at {FULL_RUN}/ (the assembler "
                             f"reads the full rung from there)")
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
            sdir = f"{out_dir}/seed{seed}"
            if args.queue and os.path.exists(f"{sdir}/test_buckets.csv"):
                print(f"[queue] {variant} seed {seed}: completed outputs found -> skip (resume)",
                      flush=True)
                rows, brows = _load_seed_csvs(sdir)
            else:
                print(f"\n================ ABLATION {variant} | SEED {seed} ================", flush=True)
                rows, brows = run_seed(variant, seed, tr, va, te, out_dir, n_epochs, ref_table)
            for r in rows:
                all_ep.append({"seed": seed, **r})
            for r in brows:
                all_bk.append({"seed": seed, **r})
            summary.append(summary_row(seed, brows, len(rows), min(r["val_ratio"] for r in rows)))
            write_aggregates(out_dir, all_ep, all_bk, summary)   # crash-safe after EVERY seed

        rat = np.array([s["ratio"] for s in summary], float)
        wp = np.array([float(s["win_pct"]) for s in summary], float)
        print(f"\n=== {variant}: {len(seeds)}-SEED SUMMARY (TEST, OVERALL) ===", flush=True)
        for s in summary:
            print(f"  seed {s['seed']}: ratio {s['ratio']} | model_med {s['model_med']} "
                  f"| GCD {s['gcd_med']} | win {s['win_pct']}%", flush=True)
        print(f"  ratio   = {rat.mean():.4f} +/- {rat.std(ddof=0):.4f}", flush=True)
        print(f"  win_pct = {wp.mean():.1f} +/- {wp.std(ddof=0):.1f} %", flush=True)
        print(f"saved under {out_dir}/", flush=True)


if __name__ == "__main__":
    main()

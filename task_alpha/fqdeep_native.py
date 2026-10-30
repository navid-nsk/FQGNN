"""task_alpha.fqdeep_native -- FQDeepFaithful with the FQ-NATIVE Op-H readout (fq_oph.py).

The message passing is INHERITED VERBATIM (embed() from fqdeep_faithful -- the FQ-DEEP stack:
Op-A lift, Op-B/C FQ-phase attention, Op-D data-conditioned twisted-Eberlein aggregation,
Dmu).  ONLY the readout changes:

  Op-K-Hermite (base class)         ->  Op-H (this class)
  kriging with an FQ kernel:            the reconstruction IS an FQ exponential sum:
  s(t) = sum_i c_i K_FQ(t,t_i) + d0     p(t) = A(t) + sum_k [a_k cos(w_k tau) + b_k sin(w_k tau)]
         + d1 t,  (2M+2)x(2M+2)                w_k = 2 pi s <xi_k, e_t>  from the learned module
         saddle solve, Tikhonov ridge          Lambda' = {L^T k}; A(t) = the Lev-Olevskii affine atom;
                                               boundary-Hermite (confluent Vandermonde) constraints;
                                               learned NULL-SPACE maneuver coordinates eta.

Instead of conditioning kernel amplitudes (the base class's theta_out Jacobi filter), the network
predicts the null-space coordinates eta of the boundary-constrained FQ coefficient space --
UNSCALED free coordinates (no |c_k| rescaling: with a median |c_k| of about 0.005 it would
suppress the learned branch).  eta head initialized ~0: at init the model IS the closed-form
min-norm boundary-Hermite FQ interpolant (chord + velocity match, exact under fq_oph.py's
rank/rcond caveats + minimal FQ energy) -- learning adds maneuver energy on top, just as the base
class's theta_out is initialized at the fixed Op-K-Hermite.

Information sources are IDENTICAL to the base forward: pooled ANCHOR embeddings (same pool as
theta_out), observed boundary velocities from same-side finite differences (fq_opk.boundary_velocity),
held lat/lon never read.  With the default freq_source/pool_source, only the readout differs from the base class.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn

from .fq_opk import boundary_velocity
from .fq_oph import oph_readout
from .fq_opm import (LAM_NUM_MAX, _sigma2_eps, chord_residual_features, dmu_velocity_batched,
                     gamma_prime, opm_c2_lambda, opm_readout, opm_readout_batched,
                     pad_sort_gaps)
from .fqdeep_faithful import FQDeepFaithful

PI = math.pi


POS_NM = 20.0                    # position normalization scale (NM) -- data.py contract


def boundary_velocity_blend(at, ay, x_feats, anc, coord, held, gap_minutes, v_fd,
                            sog_floor_kn: float = 0.1):
    """BLENDED Dmu boundary velocities: the INSTRUMENT velocity (sog/cog, the AIS transceiver's
    GPS-derived speed/course at each observed anchor) where a speed reading exists, else the
    same-side finite-difference velocity.  Leak-safe by construction: sog/cog are read from the
    model's own zeroed-at-held input features at OBSERVED anchor nodes only.  Units: converted to
    (position units)/(coord-time units) to match the finite-difference velocities exactly, via the
    per-graph absolute-time recovery dur_h = gap_minutes/60/span (the anchor_states_enu pattern).
    Rationale: the Hermite rows of the readout are only as good as the boundary velocities, and the
    instrument reading is a direct measurement at the anchor where the finite difference is a
    secant average over the neighbouring segment."""
    si, last = int(held[0]), int(held[-1])
    span = (coord[last, 0] - coord[si, 0]).clamp_min(1e-9)
    dur_h = (gap_minutes / 60.0) / span                     # hours per coord-time unit
    sog_kn = x_feats[anc, 2] * 20.0                         # x[:,2] = sog/20 (knots)
    v_pos = sog_kn * dur_h / POS_NM                         # position units per coord unit
    v_iv = torch.stack([v_pos * x_feats[anc, 3], v_pos * x_feats[anc, 4]], dim=-1)
    use_iv = (sog_kn > sog_floor_kn).unsqueeze(-1)          # observed speed -> instrument velocity
    return torch.where(use_iv, v_iv.to(v_fd.dtype), v_fd)


def _variant_frequencies(xi_ref: torch.Tensor, kind: str, seed: int) -> torch.Tensor:
    """MATCHED non-FQ frequency sets for the control suite: same count M, same
    norm range/statistics as the FQ module Lambda' -- only the SET STRUCTURE changes.  All variants
    are reduced to one canonical +-representative per pair (first nonzero coordinate positive), like
    `enumerate_spectrum_combos` -- for a real exponential sum the +- sign only flips the sin
    coefficient, so folding loses nothing and prevents rank-halving accidental +- duplicates.
    Finally NORM-RANK ALIGNED to xi_ref (k-th smallest-norm variant frequency takes the index slot of
    the k-th smallest-norm FQ frequency) so the index-paired amplitude-vs-||xi|| profile carried by
    c_k and the Jacobi variety coordinate s_k are preserved across variants."""
    M, D = xi_ref.shape
    g = torch.Generator().manual_seed(seed + 991)
    r = xi_ref.norm(dim=1)
    if kind == "grid":
        # periodic-comb control: a regular 2-D lattice, spacing set so ~M canonical (half-plane)
        # points fall inside the FQ set's max norm -- the periodic counterpart of the aperiodic module
        a = float(r.max()) * math.sqrt(math.pi / (2 * M))
        rng = int(math.ceil(float(r.max()) / a)) + 2
        pts = [(i * a, j * a) for i in range(-rng, rng + 1) for j in range(-rng, rng + 1)
               if (i or j) and (i > 0 or (i == 0 and j > 0))]
        pts.sort(key=lambda p: p[0] * p[0] + p[1] * p[1])
        xi = torch.tensor(pts[:M], dtype=xi_ref.dtype)
    elif kind == "gauss":
        # Gaussian-matched control (the RFF analogue): first two moments matched to the canonical
        # FQ representatives
        mu = xi_ref.mean(0)
        X = xi_ref - mu
        C = X.t() @ X / max(M - 1, 1) + 1e-9 * torch.eye(D, dtype=xi_ref.dtype)
        xi = mu + torch.randn(M, D, generator=g, dtype=xi_ref.dtype) @ torch.linalg.cholesky(C).t()
    elif kind == "uniform":
        # density-matched control: uniform on the bounding box of the FQ set (Poisson comb)
        lo, hi = xi_ref.min(0).values, xi_ref.max(0).values
        xi = lo + (hi - lo) * torch.rand(M, D, generator=g, dtype=xi_ref.dtype)
    elif kind == "shuffle":
        # shuffled-FQ control: each coordinate column independently permuted -- exact marginals,
        # module (rank-n cross-coordinate) coupling destroyed
        xi = torch.stack([xi_ref[torch.randperm(M, generator=g), d] for d in range(D)], dim=1)
    else:
        raise ValueError(kind)
    flip = (xi[:, 0] < 0) | ((xi[:, 0] == 0) & (xi[:, 1] < 0))
    xi = torch.where(flip.unsqueeze(1), -xi, xi)
    out = torch.empty_like(xi)
    out[r.argsort()] = xi[xi.norm(dim=1).argsort()]
    return out


class FQDeepNative(FQDeepFaithful):
    @classmethod
    def production(cls, seed: int):
        """THE paper operator: weighted spectral metric ON, omega^-2 Sobolev
        backbone, blend jet on the raw-AIS track.  The bare ctor defaults
        (flat/fd/unweighted) are the ablation settings and are NOT the paper model --
        construct the paper model through this factory (or pass all three flags explicitly)."""
        return cls(n=6, n_freqs=48, width=96, init_scale=1.84, seed=seed, in_feat=6, layers=3,
                   jacobi_R=4, oph_weighted=True, dmu_source="blend", metric_backbone="sobolev2")

    def __init__(self, *args, oph_min_phase: float = 0.25, oph_gate: bool = True,
                 oph_weighted: bool = False, dmu_source: str = "fd", freq_source: str = "fq",
                 pool_source: str = "gnn", metric_backbone: str = "flat",
                 readout_mode: str | None = None, **kw):
        assert dmu_source in ("fd", "blend"), dmu_source
        assert freq_source in ("fq", "grid", "gauss", "uniform", "shuffle", "learned"), freq_source
        assert pool_source in ("gnn", "raw"), pool_source
        # metric_backbone="sobolev2": the spectral metric is a
        # learned multiplicative correction on the FIXED Sobolev/bending backbone |omega|^-2,
        # w_k = (softplus(filt_k)+1e-3) * |om_k|^-2.  Rationale: minimizing ||W^-1 c||^2 with
        # w = |om|^-2 minimizes sum om^4 |c_k|^2 ~ integral |r''|^2 -- the BENDING ENERGY -- over the
        # FQ comb, whose dense-comb limit under the same four boundary constraints IS the cubic
        # Hermite interpolant (min-bending classical result): the cubic Hermite baseline is a
        # DEGENERATE SPECIAL CASE of this operator.  _diag_sobolev_metric.py compares the closed
        # form (eta=0, untrained, blend jet) with cubic Hermite.  The exactness properties of
        # fq_oph.py hold for ANY positive metric, so the backbone cannot break
        # the theorems.  "flat" = no backbone (the learned metric alone).
        assert metric_backbone in ("flat", "sobolev2"), metric_backbone
        self.metric_backbone = metric_backbone
        super().__init__(*args, **kw)
        # Dmu boundary-velocity source for the readout's Hermite rows: "fd" = same-side finite
        # differences (default) | "blend" = instrument sog/cog where a
        # speed reading exists, finite differences elsewhere (boundary_velocity_blend above).
        # "blend" requires the forward pass to receive gap_minutes (per-graph absolute durations).
        self.dmu_source = dmu_source
        W, M = self.enc.out_features, self.M          # LazyLinear stores out_features at construction
        # eta head: pooled anchor embedding -> null-space coordinates [M, cos/sin, x/y].
        # Same init philosophy as theta_out: start AT the closed-form operator (eta ~ 0).
        self.eta_head = nn.Sequential(nn.Linear(W, W), nn.GELU(), nn.Linear(W, M * 4))
        nn.init.normal_(self.eta_head[-1].weight, std=0.02)
        nn.init.zeros_(self.eta_head[-1].bias)
        # curvature-gate sharpness.  The gate is an implementation-level, leak-safe proxy for
        # "the FQ sum only switches on for genuine curvature" (see the fq_oph.py docstring, incl.
        # the parallel-tangent blind spot).  softplus(rho) init 4.0 -> a 25% relative boundary-velocity
        # mismatch gives g ~ tanh(1) ~ 0.76.
        self.gate_rho = nn.Parameter(torch.tensor(math.log(math.expm1(4.0))))
        self.oph_min_phase = oph_min_phase
        self.oph_gate = oph_gate
        # ---- learned SPECTRAL METRIC head (the GNN channel) ----
        # Same Jacobi-filter mechanism as the message-passing synthesis (*) and as Op-K's theta_out
        # conditioning: filt = w_head(pool h) @ P.T -> [M] over the
        # module's variety coordinate s_k; metric w_k = softplus(filt_k) + 1e-3 > 0.
        # INIT: bias[0]=1, small weights -> filt ~ 1 -> w ~ const x backbone -> the readout IS the flat (UNWEIGHTED) or sobolev2
        # closed form at init -- deliberately NOT the raw Lee-Yang amplitude profile, which as a
        # FIXED metric is nearly rank-1 (two dominant modes); in Op-K, theta_out LEARNS to rebalance that
        # profile.  B c = d holds identically for any w > 0 (under fq_oph.py's rank/rcond caveats),
        # so the metric cannot break the theorems (see the weighted-metric selftests of fq_oph.py).
        self.oph_weighted = oph_weighted
        self.w_head = nn.Sequential(nn.Linear(W, W), nn.GELU(), nn.Linear(W, self.R + 1))
        nn.init.normal_(self.w_head[-1].weight, std=0.02)
        with torch.no_grad():
            self.w_head[-1].bias.zero_(); self.w_head[-1].bias[0] = 1.0
        self.oph_w_const = False    # ablation switch: True -> ignore w_head (uniform, or fixed |om|^-2 under sobolev2), GNN channel OFF
        # ---- control suite: frequency-set + pool-source controls ----
        # freq_source != "fq" swaps the FQ module Lambda' for a MATCHED variant set at the single
        # quasicrystal() choke point -- it propagates to EVERY consumer (QFE features, Dmu velocity
        # channel, message-passing phases phk, readout omega_k) with the architecture, amplitudes,
        # learned metric and protocol byte-identical.  The learnable band scale s multiplies every
        # variant comb exactly as it does the FQ set (matched global-scale dof; the frozen
        # sqrt-prime chart means the FQ directions are fixed too).  "learned" makes the frequencies
        # a free nn.Parameter initialized at the FQ set (2M dof vs FQ's 1 -- the upper-freedom control).
        self.freq_source = freq_source
        if freq_source != "fq":
            with torch.no_grad():
                xi_ref = self.spectrum.frequencies(self.grassmann()).float()
            if freq_source == "learned":
                self.xi_var = nn.Parameter(xi_ref.clone())
            else:
                self.register_buffer("xi_var", _variant_frequencies(xi_ref, freq_source, seed=kw.get("seed", 0)))
        # pool_source="raw" = the no-GNN control: the readout conditioning (eta head + spectral
        # metric) is driven by an MLP pool over the RAW anchor rows [x ; coord] instead of the
        # message-passing embedding h -- same information at the anchors, no graph channel.  embed()
        # is skipped entirely (h unused), so the model contains NO message passing at all.
        self.pool_source = pool_source
        if pool_source == "raw":
            in_f = kw.get("in_feat", 6)
            self.raw_pool = nn.Sequential(nn.Linear(in_f + 2, W), nn.GELU(), nn.Linear(W, W))
            # embed() never runs in this control, so materialize the unused LazyLinear encoder
            # (bundle dim = msg.in_features from the base ctor) or optimizer construction fails
            self.enc = nn.Linear(self.msg.in_features, W)
        # ---- Op-M readout switch (DEFAULT OFF) ----
        # readout_mode "oph" (default) = the Op-H path (no Op-M parameters are created, so
        # construction RNG, state_dict and forward are exactly those of the Op-H model).
        # "opm" = the measured-spectrum multi-anchor operator (fq_opm.py) with its
        # two dials at their degenerate init: gamma_raw = 0
        # EXACTLY (identity parameterization -- a sigmoid could not reach 0) and precision scale
        # kappa_raw = -10 (softplus ~ 4.5e-5), so at init the model IS the Op-H model.
        # Env switches: FQ_READOUT_MODE=oph|opm; FQ_OPM_LAM=head|off|<float>
        # (off -> lam=None, the exact-zero bit-identity path); FQ_OPM_GAMMA=<float> (eval-time
        # gamma override for the zero-shot gamma grid); FQ_OPM_TIER=C|O (exterior-channel gate).
        # This block sits LAST in __init__ so the RNG stream of every other parameter is
        # the same as in the Op-H model.
        self.readout_mode = readout_mode if readout_mode is not None else \
            os.environ.get("FQ_READOUT_MODE", "oph")
        assert self.readout_mode in ("oph", "opm"), self.readout_mode
        if self.readout_mode == "opm":
            self.opm_gamma_raw = nn.Parameter(torch.tensor(0.0))       # metric-law dial, init 0
            # ---- further Op-M switches (ALL default-off: the default env gives the "head"
            # lambda mode and the learned gamma dial) ----
            # FQ_OPM_LAM also accepts the measured-units lambda modes: "c1" = measured-units head
            #   lambda_j = mu_j/sigma^2_{eps,j}, mu_j = softplus(kappa_raw)*softplus(theta.z_j),
            #   kappa_raw init softplus^-1(0.05) (the ~6-decade zero-signal plateau shrinks to
            #   ~1; the identity with the Op-H model at init becomes APPROXIMATE -- the trade);
            #   "c2" = the closed-form
            #   lambda-hat (fq_opm.opm_c2_lambda, no learnable lambda parameters); "c3" =
            #   spike-and-slab trust gate lambda_j = pi_j/sigma^2_{eps,j}, pi_j =
            #   sigmoid(theta.z_j) (a sigmoid is admissible HERE: exact zero stays attainable
            #   through Lambda_eff = g_M*lambda).
            # FQ_OPM_Z="c4" swaps the scale-free z_j entries for absolute observables (see
            #   _opm_lambda; construction-level: head input W+4; train-from-scratch only).
            # FQ_OPM_GAMMA="hatp" = the CELL-CONSTANT law gamma' =
            #   clamp(1-(FQ_OPM_VBAR+FQ_OPM_BETA2)/FQ_OPM_MBAR, 0, 1) -- cell medians/moment
            #   fit on TRAINING gaps; a fixed scalar downstream, so loop/batched are identical.
            self.opm_lam_mode = os.environ.get("FQ_OPM_LAM", "head")
            self.opm_z_mode = os.environ.get("FQ_OPM_Z", "std")
            assert self.opm_z_mode in ("std", "c4"), self.opm_z_mode
            k0 = math.log(math.expm1(0.05)) if self.opm_lam_mode == "c1" else -10.0
            self.opm_kappa_raw = nn.Parameter(torch.tensor(k0))        # lambda precision scale
            # lambda-head: theta_lambda . z_j, z_j = [|tau_j|/T ; SNR of rho~_j ;
            # heading-consistency ; h_j (the anchor's OWN embedding, NOT pooled)] (z_mode std)
            self.opm_lam_head = nn.Linear(W + (4 if self.opm_z_mode == "c4" else 3), 1)
            nn.init.normal_(self.opm_lam_head.weight, std=0.02)
            nn.init.zeros_(self.opm_lam_head.bias)
            self.opm_rho_m = nn.Parameter(torch.tensor(math.log(math.expm1(4.0))))  # Tier O gate
            self.opm_tier = os.environ.get("FQ_OPM_TIER", "C")
            assert self.opm_tier in ("C", "O"), self.opm_tier
            _ge = os.environ.get("FQ_OPM_GAMMA", "")
            # <float> = fixed eval-time gamma (zero-shot grid); "hat" = the
            # diagnostic Ledoit-Wolf gamma-hat law (closed-form, per gap, no training);
            # "hatp" = the cell-constant law (see block comment above)
            if _ge == "hatp":
                self.opm_gamma_env = gamma_prime(float(os.environ["FQ_OPM_VBAR"]),
                                                 float(os.environ["FQ_OPM_MBAR"]),
                                                 float(os.environ.get("FQ_OPM_BETA2", "0.0")))
            else:
                self.opm_gamma_env = None if _ge == "" else ("hat" if _ge == "hat" else float(_ge))

    def load_state_dict(self, state_dict, strict: bool = True, **kw):
        """Checkpoint compatibility for the Op-M switch: Op-H checkpoints carry no
        opm_* parameters, so in readout_mode='opm' exactly those keys may be missing (they keep
        their degenerate init -- gamma_raw = 0, kappa_raw = -10: the loaded model IS the
        checkpointed Op-H model).  Any OTHER mismatch still raises.  With the
        switch off this override is a plain pass-through."""
        if strict and getattr(self, "readout_mode", "oph") == "opm":
            opm_keys = {k for k in self.state_dict() if k.startswith("opm_")}
            if opm_keys - set(state_dict.keys()):
                ret = super().load_state_dict(state_dict, strict=False, **kw)
                bad = set(ret.missing_keys) - opm_keys
                assert not bad and not ret.unexpected_keys, \
                    f"load_state_dict: missing {sorted(bad)} unexpected {ret.unexpected_keys}"
                return ret
        return super().load_state_dict(state_dict, strict=strict, **kw)

    def _opm_lambda(self, at, ay, h_anc, x_anc, om_all=None):
        """Per-anchor soft precisions (the g_M gate is applied INSIDE opm_readout
        where the per-gap gate g lives).  Default mode "head":
        lambda_j = softplus(kappa_raw) * softplus(theta_lambda . z_j).
        Further modes (env FQ_OPM_LAM / FQ_OPM_Z, default-off):
          "c1"  lambda_j = mu_j / sigma^2_{eps,j}, mu_j = softplus(kappa_raw) *
                softplus(theta.z_j): the measured-units reparameterization -- mu is the
                DIMENSIONLESS trust, mu = 1 is the Bayes plug-in scale, response O(1) for
                mu in [0.1, 10].
          "c3"  lambda_j = Lambda_big * pi_j, Lambda_big = 1/sigma^2_{eps,j}, pi_j =
                sigmoid(theta.z_j): spike-and-slab -- the decades live in Lambda_big, the head
                answers only "is the exterior waveform gap-relevant?"; a sigmoid is admissible
                here (exact zero via the Lambda_eff = g_M*lambda gate).
          z_mode "c4": absolute observables [log sigma^2_{eps,j} ; log(sigma^2/sigma^2_flr)
                (absolute SNR, not a within-gap rank) ; per-side anchor count ; the O4
                value-vs-increment coherence h_hat of fq_opm.opm_c2_lambda] + h_j.
        Observable features are detached (measurements); gradients reach the GNN through h_j
        and the head through theta_lambda / kappa_raw.  Leak-safe: observed anchor rows only."""
        rho_t, tau, T, slope = chord_residual_features(at.detach(), ay.detach())
        rn = rho_t.norm(dim=-1)
        sig2 = None
        if self.opm_z_mode == "c4":
            assert om_all is not None, "z_mode c4 needs om_all (the O4 coherence feature)"
            sig2, s2flr = _sigma2_eps(rho_t)
            _, hh, _, _ = opm_c2_lambda(at, ay, om_all, min_phase=self.oph_min_phase,
                                        return_parts=True)
            side = tau < 0
            nside = torch.where(side, side.sum().to(rn.dtype), (~side).sum().to(rn.dtype))
            z = torch.cat([torch.log(sig2).unsqueeze(-1),
                           torch.log(sig2 / s2flr).unsqueeze(-1),
                           nside.unsqueeze(-1), hh.expand(rn.shape).unsqueeze(-1),
                           h_anc], dim=-1)
        else:
            snr = torch.log1p(rn / rn.median().clamp_min(1e-9))      # rho~ vs local noise floor
            u = slope / slope.norm().clamp_min(1e-9)
            headc = x_anc[:, 3].detach() * u[0] + x_anc[:, 4].detach() * u[1]   # cog . chord dir
            z = torch.cat([(tau.abs() / T).unsqueeze(-1), snr.unsqueeze(-1),
                           headc.unsqueeze(-1), h_anc], dim=-1)
        if self.opm_lam_mode == "c1":
            if sig2 is None:
                sig2 = _sigma2_eps(rho_t)[0]
            return (torch.nn.functional.softplus(self.opm_kappa_raw) *
                    torch.nn.functional.softplus(self.opm_lam_head(z)).squeeze(-1)
                    / sig2).clamp_max(LAM_NUM_MAX)     # numerical-validity ceiling (fq_opm)
        if self.opm_lam_mode == "c3":
            if sig2 is None:
                sig2 = _sigma2_eps(rho_t)[0]
            return (torch.sigmoid(self.opm_lam_head(z)).squeeze(-1)
                    / sig2).clamp_max(LAM_NUM_MAX)     # numerical-validity ceiling (fq_opm)
        lam = torch.nn.functional.softplus(self.opm_kappa_raw) * \
            torch.nn.functional.softplus(self.opm_lam_head(z)).squeeze(-1)
        return lam

    def quasicrystal(self):
        L, xi, c, s = super().quasicrystal()
        if self.freq_source != "fq":
            xi = self.xi_var.to(xi.dtype)
        return L, xi, c, s

    def forward(self, x, coord, edge_index, anchor_edge_index, batch, edge_dist=None,
                gap_minutes=None):
        """FQ-DEEP message passing (inherited embed, verbatim) + the Op-H FQ-native readout."""
        if self.pool_source == "raw":
            # no-GNN control: spectral objects only (no message passing), raw-anchor pool below
            _, xi, c, s = self.quasicrystal()
            P, xi0 = self._jacobi(xi), xi[:, 0]
            h = self.raw_pool(torch.cat([x, coord], dim=-1))
        else:
            h, xi, c, s, P, xi0 = self.embed(x, coord, edge_index)

        om_all = 2 * PI * s * xi0                              # [M] prescribed module time-frequencies
        asrc, adst = anchor_edge_index[0], anchor_edge_index[1]
        out = x[:, :2].clone()
        held_all = x[:, 5] > 0.5
        gid_adst = batch[adst]
        if self.readout_mode == "opm" and os.environ.get("FQ_OPM_BATCHED", "0") == "1":
            # ---- FQ_OPM_BATCHED=1: identical per-gap semantics, ONE batched solve (fq_opm) ----
            return self._forward_opm_batched(x, coord, batch, asrc, gid_adst, held_all, h,
                                             om_all, xi, P, gap_minutes, out)
        for gid in torch.unique(batch):
            held = torch.where(held_all & (batch == gid))[0]
            anc = torch.unique(asrc[gid_adst == gid])
            if held.numel() == 0 or anc.numel() < 2:
                continue
            at, ay = coord[anc, 0], x[anc, :2]                 # anchor times / REAL positions
            v_obs = boundary_velocity(at, ay).to(h.dtype)      # Dmu observed boundary velocity rows
            if self.dmu_source == "blend":
                assert gap_minutes is not None, "dmu_source='blend' needs gap_minutes in forward()"
                v_obs = boundary_velocity_blend(at, ay, x, anc, coord, held,
                                                float(gap_minutes[int(gid)]), v_obs)
            pool = h[anc].mean(0)
            eta = self.eta_head(pool).view(self.M, 2, 2)       # learned null-space coordinates
            if self.oph_weighted and not self.oph_w_const:
                filt = self.w_head(pool) @ P.t()               # [M] Jacobi filter on the module
                w = torch.nn.functional.softplus(filt) + 1e-3  # positive spectral metric
                if self.metric_backbone == "sobolev2":
                    w = w * (1.0 / om_all.abs().clamp_min(1e-6) ** 2)   # learned x bending backbone
            elif self.oph_weighted and self.metric_backbone == "sobolev2":
                # metric-frozen ablation under the backbone: the FIXED bending metric (the exact
                # closed form that contains cubic Hermite) -- isolates the LEARNED correction
                w = 1.0 / om_all.abs().clamp_min(1e-6) ** 2
            else:
                w = None                                       # unweighted / ablation: uniform metric
            if self.readout_mode == "opm":
                # ---- Op-M (fq_opm.py): identical inputs + the two dials ----
                if self.opm_lam_mode == "off":
                    lam_vec = None                     # Lambda = 0: the exact-parity code path
                elif self.opm_lam_mode in ("head", "c1", "c3"):
                    lam_vec = self._opm_lambda(at, ay, h[anc], x[anc], om_all)
                elif self.opm_lam_mode == "c2":        # c2: closed-form lambda-hat
                    lam_vec = opm_c2_lambda(at, ay, om_all, min_phase=self.oph_min_phase)
                else:                                  # fixed per-anchor precision (zero-shot)
                    lam_vec = torch.full((int(anc.numel()),), float(self.opm_lam_mode),
                                         device=ay.device, dtype=ay.dtype)
                if self.opm_gamma_env is None:
                    gam = self.opm_gamma_raw
                elif self.opm_gamma_env == "hat":      # diagnostic gamma-hat law
                    gam = "hat"
                else:
                    gam = torch.tensor(self.opm_gamma_env, device=ay.device, dtype=ay.dtype)
                out[held] = opm_readout(
                    at, ay, v_obs, coord[held, 0], om_all, eta,
                    gate_rho=self.gate_rho if self.oph_gate else None,
                    min_phase=self.oph_min_phase, use_vel=self.use["dmu"], w=w,
                    lam=lam_vec, gamma=gam, s_all=xi.norm(dim=1).detach(), P_jac=P,
                    tier=self.opm_tier,
                    gate_rho_m=self.opm_rho_m if self.opm_tier == "O" else None).to(out.dtype)
            else:
                out[held] = oph_readout(
                    at, ay, v_obs, coord[held, 0], om_all, eta,
                    gate_rho=self.gate_rho if self.oph_gate else None,
                    min_phase=self.oph_min_phase, use_vel=self.use["dmu"], w=w).to(out.dtype)
        return out

    def _forward_opm_batched(self, x, coord, batch, asrc, gid_adst, held_all, h,
                             om_all, xi, P, gap_minutes, out):
        """FQ_OPM_BATCHED=1 path: per-gap semantics IDENTICAL to the loop in forward() (the
        default), but every per-gap solve runs inside ONE padded
        batched fq_opm.opm_readout_batched call, the eta/metric heads run once on the stacked
        pools, and the Dmu finite-difference/blend velocities come from the batched
        sorted-domain replicas of boundary_velocity / boundary_velocity_blend (identical
        elementwise arithmetic).  Only the cheap per-gap index bookkeeping stays a Python loop.
        The predictions agree with the loop to fp32 round-off."""
        ats, ays, qts, helds, lams, pools, auxs, h0, h1, gids = [], [], [], [], [], [], [], [], [], []
        need_aux = self.dmu_source == "blend"
        for gid in torch.unique(batch).tolist():
            held = torch.where(held_all & (batch == gid))[0]
            anc = torch.unique(asrc[gid_adst == gid])
            if held.numel() == 0 or anc.numel() < 2:
                continue
            at, ay = coord[anc, 0], x[anc, :2]                 # anchor times / REAL positions
            ats.append(at); ays.append(ay); qts.append(coord[held, 0]); helds.append(held)
            pools.append(h[anc].mean(0))
            gids.append(gid)
            if need_aux:
                auxs.append(x[anc, 2:5]); h0.append(held[0]); h1.append(held[-1])
            if self.opm_lam_mode in ("head", "c1", "c3"):
                lams.append(self._opm_lambda(at, ay, h[anc], x[anc], om_all))
            elif self.opm_lam_mode == "c2":            # c2: closed-form lambda-hat
                lams.append(opm_c2_lambda(at, ay, om_all, min_phase=self.oph_min_phase))
            elif self.opm_lam_mode != "off":
                lams.append(torch.full((int(anc.numel()),), float(self.opm_lam_mode),
                                       device=ay.device, dtype=ay.dtype))
        if not helds:
            return out
        lam_in = lams if self.opm_lam_mode != "off" else None
        ts, ys, aux, lam_s, smask, gapi, Ma = pad_sort_gaps(
            ats, ays, auxs if need_aux else None, lam_in)
        vs = dmu_velocity_batched(ts, ys, smask, gapi, Ma).to(h.dtype)
        if need_aux:                                           # boundary_velocity_blend, batched
            assert gap_minutes is not None, "dmu_source='blend' needs gap_minutes in forward()"
            i0, i1 = torch.stack(h0), torch.stack(h1)
            span = (coord[i1, 0] - coord[i0, 0]).clamp_min(1e-9)
            # device-safe: callers may pass gap_minutes on CPU (ladder harness) or on DEV
            gm = gap_minutes.to(x.device)[torch.tensor(gids, device=x.device)]
            # fp64 division then fp32 cast = bit-exact replica of the loop's python-float
            # (gap_minutes / 60.0) scalar path in boundary_velocity_blend
            dur_h = (gm.double() / 60.0).to(span.dtype) / span     # hours per coord-time unit
            sog_kn = aux[:, :, 0] * 20.0                       # x[:,2] = sog/20 (knots)
            v_pos = sog_kn * dur_h.unsqueeze(1) / POS_NM       # position units per coord unit
            v_iv = torch.stack([v_pos * aux[:, :, 1], v_pos * aux[:, :, 2]], dim=-1)
            use_iv = (sog_kn > 0.1).unsqueeze(-1)              # observed speed -> instrument v
            vs = torch.where(use_iv, v_iv.to(vs.dtype), vs)
        G = len(helds)
        pool = torch.stack(pools)                              # [G, W]
        eta_all = self.eta_head(pool).view(G, self.M, 2, 2)
        if self.oph_weighted and not self.oph_w_const:
            filt = self.w_head(pool) @ P.t()                   # [G, M]
            w_all = torch.nn.functional.softplus(filt) + 1e-3
            if self.metric_backbone == "sobolev2":
                w_all = w_all * (1.0 / om_all.abs().clamp_min(1e-6) ** 2).unsqueeze(0)
        elif self.oph_weighted and self.metric_backbone == "sobolev2":
            w_all = (1.0 / om_all.abs().clamp_min(1e-6) ** 2).unsqueeze(0).expand(G, -1)
        else:
            w_all = None                                       # unweighted / ablation: uniform metric
        if self.opm_gamma_env is None:
            gam = self.opm_gamma_raw
        elif self.opm_gamma_env == "hat":
            gam = "hat"
        else:
            gam = torch.tensor(self.opm_gamma_env, device=x.device, dtype=out.dtype)
        qt_cat = torch.cat(qts)
        qn = torch.tensor([int(q.numel()) for q in qts], device=x.device)
        qgap = torch.repeat_interleave(torch.arange(G, device=x.device), qn)
        pred = opm_readout_batched(
            ts, ys, vs, smask, gapi, Ma, qt_cat, qgap, om_all, eta_all,
            gate_rho=self.gate_rho if self.oph_gate else None,
            min_phase=self.oph_min_phase, use_vel=self.use["dmu"], w_all=w_all,
            lam_s=lam_s, gamma=gam, s_all=xi.norm(dim=1).detach(), P_jac=P,
            tier=self.opm_tier,
            gate_rho_m=self.opm_rho_m if self.opm_tier == "O" else None)
        o = 0
        for held in helds:
            n = int(held.numel())
            out[held] = pred[o:o + n].to(out.dtype)
            o += n
        return out

"""
task_alpha.fq_imputer  --  THE faithful Task Alpha model (the full FQ-primitive showcase).

GOAL (not an applied GNN): demonstrate EVERY invented FQ primitive, faithfully, on one task -- the
precondition for claiming generalization.  So the model is built around ONE shared Lee-Yang
quasicrystal and routes every primitive through it:

    ONE geometry:  L in Gr_+(2,n) with Q-INDEPENDENT minors (frozen sqrt-prime ratios -> a genuine
                   d=2 aperiodic lift, NOT a product of 1-D FQs), and the secular variety U=exp(iH).
    ONE spectrum:  Lambda' = {L^T k : var(k)<d}, with the VARIETY-FIXED cumulant amplitude c_k
                   (fqmath.akkv_amplitudes, the log-p_U cumulant -- fixed by the variety, NOT free).

    Op-A' / QFE   : synthesize each vertex's 2-D position on (Lambda', c) [+ e^{rho.x} envelope + ac head].
    Op-B          : coupled eigenphases theta_j(Phi_L(pos)) of M=diag(z)U.
    Op-C          : secular value p_U(Phi_L(pos)) = variety membership of the position.
    Op-D          : TWISTED EBERLEIN convolution -- synthesize the held-out vertex from its two-sided
                    anchors, kernel K(delta)=Re sum_k c_k g_k e^{2pi i s<xi_k,delta>} (conjugate-
                    reflected on the reverse edge => translation-invariant).
    D-mu velocity : the SOG/COG channel as the DERIVATIVE measure on the SAME Lambda', amplitudes x 2pi i xi.
    ledger        : the amoeba/Parseval certificate + rho_pp are computed on the SAME (L,U,xi,c).

Every primitive is load-bearing and reads the SAME quasicrystal -- the FQ is one operator/basis,
shared, not a stack of random FQ layers.  The only plain Linear maps are the message/fusion/readout
interfaces (linear readouts on FQ features are admissible); the single task nonlinearity lives in
the labeled non-spectral update head.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn

from fqmath.grassmann import PositiveGrassmann
from fqmath.lee_yang import SecularUnitary
from fqmath.spectrum import FQSpectrum

PI = math.pi


class FQTrajectoryImputer(nn.Module):
    def __init__(self, n: int = 6, n_freqs: int = 24, k_max: int = 3, width: int = 64,
                 ac_dim: int = 8, init_scale: float = 2.0, seed: int = 0, in_feat: int = 6,
                 tap_mode: str | None = None):
        super().__init__()
        self.n = n
        self.use_cuda = False   # route Op-B/Op-C, QFE, Dmu, Op-D through the fqcuda kernels when True
        # ---- Op-D tap semantics of THIS base forward ----
        # "legacy" (default): TWO independent tap tables
        #   g_cos/g_sin, K = cos(phi)@(Re c * g_cos) - sin(phi)@(Im c * g_sin) -- NOT a multiplier
        #   of c_k (the amplitude's complex phase is not carried rigidly).
        # "multiplier": ONE real finitely-supported tap table ghat (:= g_cos, REUSED so
        #   checkpoints map key-for-key; g_sin stays registered but is INERT -- parameter shapes,
        #   init RNG draws and state_dict keys are identical across modes), applied to BOTH
        #   quadratures:  K = Re sum_k c_k * ghat_k * e^{i phi_k}
        #                   = sum_k ghat_k * (Re(c_k) cos(phi_k) - Im(c_k) sin(phi_k)),
        #   i.e. the true spectral-multiplier semantics (ghat acts as a Fourier multiplier on c_k).
        # Resolution: ctor arg wins; else env FQ_TAP_MODE; else "legacy".
        tap_mode = tap_mode if tap_mode is not None else os.environ.get("FQ_TAP_MODE", "legacy")
        if tap_mode not in ("legacy", "multiplier"):
            raise ValueError(f"tap_mode must be 'legacy' or 'multiplier', got {tap_mode!r}")
        self.tap_mode = tap_mode
        # ---- ONE shared quasicrystal geometry (genuine d=2, Q-independent) + variety ----
        self.grassmann = PositiveGrassmann(n=n, d=2, init_scale=init_scale, seed=seed,
                                           incommensurate_init=True, freeze_incommensurate=True)
        self.secular = SecularUnitary(n=n, coupled=True, seed=seed)
        self.spectrum = FQSpectrum(self.grassmann, k_max=k_max, max_freqs=n_freqs, seed=seed,
                                   amplitude="secular", secular=self.secular)
        self.M = self.spectrum.M
        # ---- learnable FQ knobs (all in-family) ----
        self.s_raw = nn.Parameter(torch.tensor(0.0))            # band scale s = softplus(s_raw)
        self.rho = nn.Parameter(torch.zeros(2))                 # QFE real-exponential envelope e^{rho.x}
        g = torch.Generator().manual_seed(seed)
        self.g_cos = nn.Parameter(0.1 * torch.randn(self.M, width, generator=g))  # Op-D cos taps (= ghat in multiplier mode)
        self.g_sin = nn.Parameter(0.1 * torch.randn(self.M, width, generator=g))  # Op-D sin taps (INERT in multiplier mode)
        # ---- non-spectral interfaces (admissible: linear on FQ features) ----
        self.ac_head = nn.Linear(2, ac_dim)                     # ac residual (out-of-family, labeled)
        bundle = 2 * self.M + ac_dim + n + 2 + 4 * self.M + in_feat   # qfe + ac + eig + pU + vel + raw feat
        self.msg = nn.Linear(bundle, width)                     # message projection
        self.update = nn.Linear(2 * width, width)               # update fusion [m, agg] -> width
        self.norm = nn.LayerNorm(width)
        self.act = nn.LeakyReLU(0.1)                            # the ONE task nonlinearity (update head)
        self.readout = nn.Linear(width, 2)                      # output interface -> ENU residual (NM)

    # ---- the shared quasicrystal (computed once per forward) ----
    def quasicrystal(self):
        L = self.grassmann()                                   # [n, 2]
        xi = self.spectrum.frequencies(L)                      # [M, 2]
        c = self.spectrum.amplitudes(L)                        # [M] complex (variety-fixed cumulant)
        s = torch.nn.functional.softplus(self.s_raw) + 1e-3
        dt = self.s_raw.dtype                                  # model real dtype -> dtype-align (fp32/fp64 support)
        cdt = torch.complex128 if dt == torch.float64 else torch.complex64
        return L.to(dt), xi.to(dt), c.to(cdt), s.to(dt)

    # ---- Op-A' / QFE position synthesis ----
    def qfe(self, pos, xi, c, s):
        if self.use_cuda:                                      # CUDA fused QFE (Op-A')
            from fqcuda.fq_phase_cuda import qfe_cuda
            return qfe_cuda(pos, xi, c, s, self.rho), torch.tanh(self.ac_head(pos))
        ph = 2 * PI * s * (pos @ xi.t())                       # [N, M]
        cr, ci = c.real.unsqueeze(0), c.imag.unsqueeze(0)
        re = cr * torch.cos(ph) - ci * torch.sin(ph)           # Re(c e^{i ph})
        im = cr * torch.sin(ph) + ci * torch.cos(ph)           # Im(c e^{i ph})
        env = torch.exp((pos @ self.rho).clamp(-3.0, 3.0)).unsqueeze(-1)   # e^{rho.x}, bounded for stability
        qfe = torch.cat([re, im], dim=-1) * env                # [N, 2M]
        ac = torch.tanh(self.ac_head(pos))                     # [N, ac_dim] ac residual
        return qfe, ac

    # ---- D-mu velocity channel (derivative measure on the SAME spectrum) ----
    def velocity(self, pos, xi, c, s):
        if self.use_cuda:                                      # CUDA Dmu velocity
            from fqcuda.fq_phase_cuda import velocity_cuda
            return velocity_cuda(pos, xi, c, s)
        ph = 2 * PI * s * (pos @ xi.t())                       # [N, M]
        cos, sin = torch.cos(ph), torch.sin(ph)
        outs = []
        for d in range(2):
            cv = (2 * PI) * xi[:, d].to(c.dtype) * (1j * c)     # (2 pi i xi_{k,d}) c_k
            cr, ci = cv.real.unsqueeze(0), cv.imag.unsqueeze(0)
            outs += [cr * cos - ci * sin, cr * sin + ci * cos]
        return torch.cat(outs, dim=-1)                         # [N, 4M]

    # ---- Op-B eigenphases + Op-C secular value (variety membership) ----
    def secular_features(self, pos, L):
        cdt = torch.complex128 if pos.dtype == torch.float64 else torch.complex64   # dtype-aware (fp32/fp64)
        U = self.secular.unitary().to(cdt)
        z = torch.exp(2j * PI * (pos @ L.t()).to(cdt))               # Phi_L(pos), [N, n]
        Mop = z.unsqueeze(-1) * U                                    # diag(z) U, [N, n, n]
        if self.use_cuda:                                           # CUDA Op-B eigenphases + Op-C det (fused)
            from fqcuda.fq_eig_cuda import OpBEig
            theta, pU = OpBEig.apply(Mop)
            return theta.to(pos.dtype), torch.stack([pU.real, pU.imag], dim=-1).to(pos.dtype)
        I = torch.eye(self.n, dtype=Mop.dtype, device=Mop.device)
        pU = torch.linalg.det(I - Mop)                              # Op-C: p_U(Phi_L(pos)) [N] (GPU, fast)
        # Op-B eigenphases: torch.linalg.eigvals on many tiny complex matrices is ~100x slower on GPU
        # than CPU (a known bad path), so compute on CPU and move back (autograd flows through .to()).
        ev = torch.linalg.eigvals(Mop.cpu()).to(Mop.device)         # [N, n]
        theta, _ = torch.sort(torch.angle(ev), dim=-1)            # eigenphases [N, n]
        return theta.to(pos.dtype), torch.stack([pU.real, pU.imag], dim=-1).to(pos.dtype)

    def forward(self, x, pos, edge_index, edge_dist=None, extensive=False):
        L, xi, c, s = self.quasicrystal()
        qfe, ac = self.qfe(pos, xi, c, s)                         # Op-A'
        theta, pUri = self.secular_features(pos, L)               # Op-B + Op-C
        vel = self.velocity(pos, xi, c, s)                        # D-mu velocity channel
        node = torch.cat([qfe, ac, theta, pUri, vel, x], dim=-1)  # the FQ feature bundle
        m = self.msg(node)                                       # [N, W] (linear on FQ features)

        # ---- Op-D twisted Eberlein convolution over two-sided anchors ----
        src, dst = edge_index[0], edge_index[1]
        delta = pos[dst] - pos[src]                              # [E, 2]
        phk = 2 * PI * s * (delta @ xi.t())                     # [E, M]
        if self.tap_mode == "multiplier":
            # TRUE multiplier form -- the SINGLE tap ghat (= g_cos; g_sin inert)
            # multiplies c_k, carrying its complex phase rigidly:  K = Re sum_k c_k ghat_k e^{i phi}
            # = sum_k ghat_k (Re c_k cos phi - Im c_k sin phi)  -- spectral-multiplier semantics.
            # CUDA: even with use_cuda=True, multiplier-mode Op-D runs on this torch path --
            # fq_opd.cu implements only the legacy two-tap contraction, and the base path is not
            # used by FQDeepFaithful/Native (they override embed and never read these taps); the
            # unfused matmul + index_add path costs little at base-path scale.
            Gc = c.real.unsqueeze(-1) * self.g_cos              # [M, W]  Re(c_k) * ghat_k
            Gs = c.imag.unsqueeze(-1) * self.g_cos              # [M, W]  Im(c_k) * ghat_k (SAME tap)
        else:
            # legacy (default): two independent tap tables.
            Gc = c.real.unsqueeze(-1) * self.g_cos              # [M, W]
            Gs = c.imag.unsqueeze(-1) * self.g_sin              # [M, W]
        K = torch.cos(phk) @ Gc - torch.sin(phk) @ Gs           # [E, W]  Re sum_k c_k g_k e^{i<xi,delta>}
        weighted = m[src] * K                                   # [E, W]
        N = pos.shape[0]
        agg = torch.zeros(N, m.shape[1], device=m.device, dtype=m.dtype)
        agg.index_add_(0, dst, weighted)
        if not extensive:                                       # intensive (mean) vs extensive (sum)
            deg = torch.zeros(N, 1, device=m.device, dtype=m.dtype)
            deg.index_add_(0, dst, torch.ones(dst.shape[0], 1, device=m.device, dtype=m.dtype))
            agg = agg / deg.clamp_min(1.0)

        # ---- non-spectral update head (the one task nonlinearity) ----
        h = self.act(self.norm(m + self.update(torch.cat([m, agg], dim=-1))))
        out = self.readout(h)                                  # ENU residual (NM)
        # per-primitive forward outputs (for the telemetry forward.csv)
        self._last = {"Op-A_QFE": qfe, "Op-A_ac": ac, "Op-B_eigenphase": theta,
                      "Op-C_secular_value": pUri, "Dmu_velocity": vel, "Op-D_kernel": K,
                      "Op-D_aggregate": agg, "message": m, "readout": out}
        return out

    @torch.no_grad()
    def primitive_stats(self):
        """Forward-output statistics of EACH primitive (for forward.csv)."""
        rows = []
        for name, t in getattr(self, "_last", {}).items():
            tf = t.detach().float()
            fin = torch.isfinite(tf)
            vals = tf[fin] if fin.any() else tf.nan_to_num()
            rows.append(dict(module=name, kind="primitive",
                             shape="x".join(map(str, tuple(tf.shape))),
                             out_mean=float(vals.mean()) if vals.numel() else 0.0,
                             out_std=float(vals.std()) if vals.numel() > 1 else 0.0,
                             out_l2=float(tf.norm()),
                             out_min=float(vals.min()) if vals.numel() else 0.0,
                             out_max=float(vals.max()) if vals.numel() else 0.0,
                             frac_finite=float(fin.float().mean())))
        return rows

    # ---- faithfulness ledger (for telemetry) ----
    @torch.no_grad()
    def diagnostics(self):
        from fqmath.genericity import lee_yang_certificate
        from fqmath.membership import parseval_residual
        from fqmath.diffraction import bragg_participation
        L, xi, c, s = self.quasicrystal()
        # telemetry runs on CPU copies: the certificate helpers mix devices on CUDA models
        L, xi, c = L.detach().cpu(), xi.detach().cpu(), c.detach().cpu()
        U = self.secular.unitary().detach().cpu()
        cert = lee_yang_certificate(U, L[:, 0])
        # certify BOTH rays of the pair (the Lee-Yang condition concerns the ray family at d=2;
        # a single positive ray is a partial check) -- report the worst-case margins
        cert_n = lee_yang_certificate(U, -L[:, 0])
        return dict(amoeba_margin=min(cert["amoeba_margin"], cert_n["amoeba_margin"]),
                    coupling_margin=min(cert["coupling_margin"], cert_n["coupling_margin"]),
                    eigenphase_gap=min(cert["eigenphase_gap"], cert_n["eigenphase_gap"]),
                    lee_yang_ok=int(cert["ok"] and cert_n["ok"]),
                    parseval_residual=float(parseval_residual(xi, c)),
                    bragg_participation=float(bragg_participation(c)),
                    min_minor=float(self.grassmann.min_minor(L)),
                    log_minor_barrier=float(self.grassmann.log_minor_barrier(L)))

    def min_minor(self):
        return self.grassmann.min_minor()

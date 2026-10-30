"""
fqops.op_conv_aggregate  --  genuine quasicrystal-convolution aggregation (Op-D).

Contrast: FQSpectralAggregate weights edges by exp(-gamma*dist) -> a per-channel-gated
distance-decay MEAN; there the FQ measure mu = sum_k c_k delta_{xi_k} never convolves the neighbour signal.

Faithful Op-D (Eberlein/Fourier-Bohr pairing; AKKV/LT inverse transform): the
real-space kernel of the FQ measure is  phi_mu(Delta) = sum_k c_k cos(2 pi s <xi_k, Delta>), and
aggregation is convolution against it on the spatial graph:

    agg_i = ( sum_{j in N(i)} phi_mu(x_i - x_j) * (m_ij (*) gate) ) / sum_{j} |phi_mu(x_i - x_j)| .

(xi_k = L^T k the aperiodic spectrum, c_k linear in the minors of L in Gr_+(2,n); s = learnable spatial
frequency scale so the kernel finds the right scale w.r.t. the centroid displacement). This is the ONLY
difference from FQSpectralAggregate: the edge weight is the FQ kernel, not a generic decay -> isolates
the effect of the convolution kernel.
"""
from __future__ import annotations

import itertools
import math
import torch
import torch.nn as nn

from fqmath.grassmann import PositiveGrassmann
from fqmath.spectrum import FQSpectrum


class FQConvAggregate(nn.Module):
    def __init__(self, width: int, n: int = 12, k_max: int = 2, max_freqs: int | None = None,
                 init_scale: float = 1.0, seed: int = 0):
        super().__init__()
        self.width = width
        mf = max_freqs if max_freqs is not None else width
        # NOTE: simple aggregator (the faithful Op-D is SecularCoupledConvAggregate; the
        # geometry-only ablation is FQConvAggregateStable).  Uses the geometry-only (free) amplitude.
        self.grassmann_g = PositiveGrassmann(n=n, d=1, init_scale=init_scale, seed=seed)
        self.spectrum_g = FQSpectrum(self.grassmann_g, k_max=k_max, max_freqs=max(mf, width), seed=seed,
                                     amplitude="free")
        # the FQ CONVOLUTION KERNEL: a d=2 aperiodic comb over the spatial displacement
        self.grassmann_k = PositiveGrassmann(n=n, d=2, init_scale=init_scale, seed=seed + 101)
        self.spectrum_k = FQSpectrum(self.grassmann_k, k_max=max(k_max, 3), max_freqs=64, seed=seed + 101,
                                     amplitude="free")
        self.freq_scale_raw = nn.Parameter(torch.tensor(0.0))   # softplus -> learnable spatial frequency scale

    def channel_gate(self) -> torch.Tensor:
        L = self.grassmann_g()
        c = self.spectrum_g.amplitudes(L).real               # signed real (free amplitude)
        g = torch.nn.functional.softplus(c[: self.width])    # channel multiplier (a gate, not the FQ amplitude)
        return g / g.mean().clamp_min(1e-6)

    def kernel(self, delta: torch.Tensor) -> torch.Tensor:
        """delta : [E, 2] receiver-sender centroid displacement -> [E] FQ kernel phi_mu(delta)."""
        L = self.grassmann_k()
        xi = self.spectrum_k.frequencies(L)                  # [M, 2]
        c = self.spectrum_k.amplitudes(L).real               # [M] signed real (free amplitude)
        s = torch.nn.functional.softplus(self.freq_scale_raw) + 1e-3
        a = 2 * math.pi * s * (delta @ xi.t())               # [E, M]
        return (torch.cos(a) * c.unsqueeze(0)).sum(-1)       # [E]  phi_mu(delta)

    def forward(self, messages: torch.Tensor, edge_index: torch.Tensor, num_nodes: int,
                pos: torch.Tensor) -> torch.Tensor:
        """messages : [E, W] ; edge_index : [2, E] (row0=src/sender, row1=dst/receiver) ; pos : [N, 2]."""
        src, dst = edge_index[0], edge_index[1]
        delta = pos[dst] - pos[src]                          # [E, 2]
        K = self.kernel(delta)                               # [E]  (signed FQ convolution weight)
        m = messages * self.channel_gate().unsqueeze(0)      # [E, W] amplitude-gated message
        m = m * K.unsqueeze(-1)                              # convolve against the FQ measure
        agg = torch.zeros(num_nodes, self.width, device=m.device, dtype=m.dtype)
        agg.index_add_(0, dst, m)
        wsum = torch.zeros(num_nodes, 1, device=m.device, dtype=m.dtype)
        wsum.index_add_(0, dst, K.abs().unsqueeze(-1))
        return agg / wsum.clamp_min(1e-6)

    def min_minor(self) -> torch.Tensor:
        return torch.minimum(self.grassmann_g.min_minor(), self.grassmann_k.min_minor())


class FQConvAggregateStable(nn.Module):
    """GEOMETRY-ONLY QUASICRYSTAL ABLATION (a baseline -- NOT the faithful FQ Op-D; the
    faithful operator is SecularCoupledConvAggregate).  Eberlein convolution = per-channel Fourier
    multiplier on the APERIODIC Gr_+ spectrum Lambda' = L^T k, but with a FREE (geometry-only)
    amplitude -- there is NO variety pinning it, so it is the ablation that isolates "does the
    aperiodic GEOMETRY alone help" from "does the Lee-Yang COUPLING + variety-fixed amplitude help".

    Design: amplitude is SIGNED (no softplus positivity); pooling has an
    intensive/EXTENSIVE switch (Gong-et-al); frequencies stay on Lambda' (no off-spectrum leak).

        agg_i = sum_k c_k * g_k (.) [ sum_{j in N(i)} a_ij cos(2 pi s <xi_k, x_i-x_j>) m_ij ].
    """
    def __init__(self, width: int, n: int = 12, k_max: int = 3, max_freqs: int = 48,
                 init_scale: float = 1.0, seed: int = 0, disp_scale: float = 1.0, d: int = 2,
                 extensive: bool = False):
        super().__init__()
        self.width = width; self.d = d        # d=2 spatial; d=3 SPATIOTEMPORAL (Dx,Dy,Dt) -- same operator
        self.extensive = bool(extensive)      # sum (extensive) vs 1/deg (intensive) pooling
        self.grassmann_k = PositiveGrassmann(n=n, d=d, init_scale=init_scale, seed=seed + 101)
        # amplitude="free": the labeled geometry-only ablation amplitude (NOT variety-fixed).
        self.spectrum_k = FQSpectrum(self.grassmann_k, k_max=k_max, max_freqs=max_freqs, seed=seed + 101,
                                     amplitude="free")
        self.M = self.spectrum_k.M
        # init s so the median phase ~1 rad given the graph displacement scale + |xi| at init
        with torch.no_grad():
            xi0 = self.spectrum_k.frequencies(self.grassmann_k())
            xi_rms = float(xi0.pow(2).sum(-1).sqrt().mean().clamp_min(1e-6))
        s0 = 1.0 / (2 * math.pi * max(xi_rms * max(disp_scale, 1e-6), 1e-6))
        self.freq_scale_raw = nn.Parameter(torch.tensor(math.log(math.expm1(min(max(s0, 1e-3), 1e3)))))
        # per-(channel, frequency) taps -> genuine rank-M multiplier (BernNet/Specformer-grade)
        self.g = nn.Parameter(0.1 * torch.randn(self.M, width))

    def _amp(self, L):
        # SIGNED geometry-only amplitude (free, NOT variety-fixed; NO softplus positivity).
        c = self.spectrum_k.amplitudes(L).real               # [M] real signed (free mode has Im=0)
        return c / c.abs().sum().clamp_min(1e-6)             # unit total |mass|

    def forward(self, messages, edge_index, num_nodes, pos):
        src, dst = edge_index[0], edge_index[1]
        L = self.grassmann_k()
        xi = self.spectrum_k.frequencies(L)                  # [M,2]
        w = self._amp(L)                                     # [M] signed, unit |mass|
        s = torch.nn.functional.softplus(self.freq_scale_raw) + 1e-3
        delta = pos[dst] - pos[src]                          # [E,2]
        cosA = torch.cos(2 * math.pi * s * (delta @ xi.t())) # [E,M]
        if self.extensive:
            a = torch.ones(dst.shape[0], 1, device=messages.device, dtype=messages.dtype)  # SUM pooling
        else:
            deg = torch.zeros(num_nodes, 1, device=messages.device, dtype=messages.dtype)
            deg.index_add_(0, dst, torch.ones(dst.shape[0], 1, device=messages.device, dtype=messages.dtype))
            a = (1.0 / deg.clamp_min(1.0))[dst]              # [E,1] intensive (mean)
        # EFFICIENT (exact) Op-D: reassociate the freq-sum to the edge level so the [E,M,W] tensor is
        # never built -> O(E*M + E*W) not O(E*M*W) (far less memory, mathematically identical):
        #   agg[n,w] = sum_{j->n} a_j msg[j,w] * (cosA @ G)[j,w],  G[m,w] = w_m * g[m,w].
        G = w.unsqueeze(-1) * self.g                         # [M,W]  amplitude-filtered taps
        Kc = cosA @ G                                        # [E,W]  per-edge spectral response
        weighted = (a * messages) * Kc                       # [E,W]
        agg = torch.zeros(num_nodes, self.width, device=messages.device, dtype=messages.dtype)
        agg.index_add_(0, dst, weighted)                     # [N,W]
        return agg

    def min_minor(self) -> torch.Tensor:
        return self.grassmann_k.min_minor()


class RFFConvAggregate(nn.Module):
    """ABLATION BASELINE for 'is it just Fourier features?': IDENTICAL convolution machinery to
    FQConvAggregateStable (degree-norm, multi-tap g_k, freq-scale, Bochner-positive amps) BUT the
    frequencies xi_k are FREE learnable params (learnable_freq=True) or FIXED random Gaussian (=classic
    RFF, learnable_freq=False) -- NOT xi_k=L^T k from the positive Grassmannian, and amplitudes are FREE
    (not linear in minors). If FQ beats this, the FQ STRUCTURE (aperiodic spectrum) is the edge, not merely
    having frequencies."""
    def __init__(self, width: int, n_freqs: int = 48, init_scale: float = 1.0, seed: int = 0,
                 disp_scale: float = 1.0, learnable_freq: bool = True):
        super().__init__()
        self.width = width; self.M = n_freqs; self.learnable_freq = learnable_freq
        g = torch.Generator().manual_seed(seed)
        xi0 = init_scale * torch.randn(self.M, 2, generator=g)
        if learnable_freq:
            self.xi = nn.Parameter(xi0)
        else:
            self.register_buffer("xi", xi0)
        self.amp_raw = nn.Parameter(0.1 * torch.randn(self.M, generator=g))
        self.g = nn.Parameter(0.1 * torch.randn(self.M, width, generator=g))
        xi_rms = float(xi0.pow(2).sum(-1).sqrt().mean().clamp_min(1e-6))
        s0 = 1.0 / (2 * math.pi * max(xi_rms * max(disp_scale, 1e-6), 1e-6))
        self.freq_scale_raw = nn.Parameter(torch.tensor(math.log(math.expm1(min(max(s0, 1e-3), 1e3)))))

    def forward(self, messages, edge_index, num_nodes, pos):
        src, dst = edge_index[0], edge_index[1]
        w = torch.nn.functional.softplus(self.amp_raw); w = w / w.sum().clamp_min(1e-6)
        s = torch.nn.functional.softplus(self.freq_scale_raw) + 1e-3
        delta = pos[dst] - pos[src]
        cosA = torch.cos(2 * math.pi * s * (delta @ self.xi.t()))
        deg = torch.zeros(num_nodes, 1, device=messages.device, dtype=messages.dtype)
        deg.index_add_(0, dst, torch.ones(dst.shape[0], 1, device=messages.device, dtype=messages.dtype))
        a = (1.0 / deg.clamp_min(1.0))[dst]
        G = w.unsqueeze(-1) * self.g; Kc = cosA @ G
        agg = torch.zeros(num_nodes, self.width, device=messages.device, dtype=messages.dtype)
        agg.index_add_(0, dst, (a * messages) * Kc)
        return agg

    def min_minor(self):
        return torch.tensor(1.0)


class SecularCoupledConvAggregate(nn.Module):
    """FAITHFUL Op-D with all three FQ ingredients: (1) Eberlein convolution; (2) a learnable unitary
    U=exp(iH) (H Hermitian -> strict Lee-Yang for ANY param); (3) VARIETY-FIXED amplitudes on the subset
    atoms xi_S=L^T 1_S, derived from the secular polynomial of U (see secular_amplitudes_complex).
    The amplitudes stay COMPLEX (sign + phase, NOT |.|) and the operator uses BOTH cos and sin taps, so it
    can realize Re sum_S a_S e^{i<xi_S,.>}; a magnitude-only, cos-only form lacks the odd/sin subspace and
    cannot represent such fields in general.  At U=I the coupling vanishes (u_offdiag=0) and the secular
    polynomial factorizes, det(diag(z)I-I)=prod(z_j-1) (the decoupled comb)."""
    def __init__(self, width: int, n: int = 6, d: int = 2, init_scale: float = 1.0, seed: int = 0,
                 disp_scale: float = 1.0, u_init_scale: float = 0.3, extensive: bool = False):
        super().__init__()
        self.width = width; self.n = n; self.d = d
        # intensive (1/deg mean) vs EXTENSIVE (sum) pooling (Gong-et-al). Sum is strictly
        # better for extensive / long-range targets (mean washes out the long-range signal).
        self.extensive = bool(extensive)
        self.grassmann_k = PositiveGrassmann(n=n, d=d, init_scale=init_scale, seed=seed + 101)
        self.subsets = [S for r in range(n + 1) for S in itertools.combinations(range(n), r)]   # 2^n atoms
        combos = torch.zeros(len(self.subsets), n)
        for i, S in enumerate(self.subsets):
            if S: combos[i, list(S)] = 1.0
        self.register_buffer("combos", combos)                          # [2^n, n] subset indicators (var=0)
        self._combos_int = [tuple(int(v) for v in row) for row in combos]
        self.M = len(self.subsets)
        self.register_buffer("signs", torch.tensor([(-1.0) ** (n - len(S)) for S in self.subsets]))   # [M] (for telemetry)
        g = torch.Generator().manual_seed(seed)
        self.A_re = nn.Parameter(u_init_scale * torch.randn(n, n, generator=g))
        self.A_im = nn.Parameter(u_init_scale * torch.randn(n, n, generator=g))
        self.g_cos = nn.Parameter(0.1 * torch.randn(self.M, width, generator=g))   # cos-channel taps
        self.g_sin = nn.Parameter(0.1 * torch.randn(self.M, width, generator=g))   # sin-channel taps (odd subspace)
        with torch.no_grad():
            xi0 = self.combos @ self.grassmann_k()
            xi_rms = float(xi0.pow(2).sum(-1).sqrt().mean().clamp_min(1e-6))
        s0 = 1.0 / (2 * math.pi * max(xi_rms * max(disp_scale, 1e-6), 1e-6))
        self.freq_scale_raw = nn.Parameter(torch.tensor(math.log(math.expm1(min(max(s0, 1e-3), 1e3)))))

    def U(self):
        A = torch.complex(self.A_re, self.A_im); H = 0.5 * (A + A.conj().transpose(-2, -1))
        return torch.linalg.matrix_exp(1j * H.to(torch.complex64))

    def u_offdiag_energy(self):
        U = self.U(); return (U - torch.diag_embed(torch.diagonal(U))).abs().pow(2).sum().sqrt()

    def secular_amplitudes_complex(self):
        """The VARIETY-FIXED Fourier-Bohr amplitude
            c_k = ||xi_k|| * c_P(k) ,   c_P(k) = [z^k] log p_U   (the log-p_U cumulant, fqmath.akkv_amplitudes)
        NOT a_S = (-1)^{n-|S|} det(U[S,S]) -- det(U_S) is only the COEFFICIENT of p_U in its multilinear
        expansion, not the diffraction mass.  The subset
        atoms k in {0,1}^n are var(k)=0, so the cumulant is EXACT here.  Complex, unit total mass."""
        from fqmath.akkv_amplitudes import secular_pullback_amplitudes
        L = self.grassmann_k()
        U = self.U()
        c, _mask = secular_pullback_amplitudes(L, U, self._combos_int)   # [M] complex (var=0 exact)
        return c / c.abs().sum().clamp_min(1e-6)                          # [M] complex, unit total mass

    def secular_amplitudes(self):
        """|a_S| magnitude envelope (telemetry / U=I -> uniform 1/M check). Use *_complex in forward."""
        return self.secular_amplitudes_complex().abs().to(self.g_cos.dtype)

    def _phase(self, coords):                                           # 2 pi s <xi_S, coords>  -> [N, M]
        L = self.grassmann_k(); xi = self.combos @ L
        s = torch.nn.functional.softplus(self.freq_scale_raw) + 1e-3
        return 2 * math.pi * s * (coords @ xi.t()), xi

    def absolute_spectral_features(self, pos):
        """ABSOLUTE-position FQ lift. Returns [Re(a_S e^{i ph_S(pos)}), Im(...)] over atoms -> [N, 2M].
        A linear readout on this realizes the field y(x)=Re sum_S a_S e^{i<xi_S,x>} (translation-VARIANT, which the
        relative-displacement aggregator alone cannot)."""
        ph, _ = self._phase(pos)                                        # [N, M]
        a = self.secular_amplitudes_complex(); re_a = a.real.to(pos.dtype); im_a = a.imag.to(pos.dtype)
        c, s = torch.cos(ph), torch.sin(ph)
        re_part = re_a * c - im_a * s                                   # Re(a e^{i ph})
        im_part = re_a * s + im_a * c                                   # Im(a e^{i ph})
        return torch.cat([re_part, im_part], dim=-1)                    # [N, 2M]

    def forward(self, messages, edge_index, num_nodes, pos):
        src, dst = edge_index[0], edge_index[1]
        delta = pos[dst] - pos[src]
        ph, _ = self._phase(delta)                                      # [E,M]  2 pi s <xi_S, delta>
        a = self.secular_amplitudes_complex()                           # complex [M]  (variety-fixed, signed)
        re_a = a.real.to(self.g_cos.dtype); im_a = a.imag.to(self.g_cos.dtype)
        cosA = torch.cos(ph); sinA = torch.sin(ph)
        # twisted Eberlein {mu,.}_A: K(delta)=Re sum_S c_S e^{2pi i s<xi_S,delta>}*tap.  On the
        # REVERSE directed edge j<-i, delta -> -delta, so sinA flips sign => Kc_reverse = cosA@Gc + sinA@Gs
        # = the CONJUGATE-REFLECTED tap g-tilde -> translation invariance holds for the directed graph.
        if self.extensive:
            a_norm = torch.ones(dst.shape[0], 1, device=messages.device, dtype=messages.dtype)  # SUM pooling
        else:
            deg = torch.zeros(num_nodes, 1, device=messages.device, dtype=messages.dtype)
            deg.index_add_(0, dst, torch.ones(dst.shape[0], 1, device=messages.device, dtype=messages.dtype))
            a_norm = (1.0 / deg.clamp_min(1.0))[dst]                    # intensive (mean)
        Gc = re_a.unsqueeze(-1) * self.g_cos; Gs = im_a.unsqueeze(-1) * self.g_sin
        Kc = cosA @ Gc - sinA @ Gs                                      # [E,W]  Re sum_S c_S e^{i<xi,delta>} (twisted)
        agg = torch.zeros(num_nodes, self.width, device=messages.device, dtype=messages.dtype)
        agg.index_add_(0, dst, (a_norm * messages) * Kc)
        return agg

    def min_minor(self):
        return self.grassmann_k.min_minor()

"""
fqops.op_spectral_block  --  FQSpectralBlock : the GEOMETRY-ONLY QUASICRYSTAL ablation spectral block
(a baseline -- NOT the faithful FQ; the faithful Op-D is
op_conv_aggregate.SecularCoupledConvAggregate).

What is FAITHFUL here: the GEOMETRY -- the lift L is the (r,theta) Gr_+(2,n) chart (all 2x2 minors
> 0) and the frequencies xi_k = L^T k sit on the genuine aperiodic spectrum module Lambda' =
{L^T k : var(k) < d} (AKKV Thm 2.3; Lambda' is DENSE, only the support is u.d.).

What makes it the ABLATION (NOT faithful): (1) the amplitudes c_k = sum_I L_I w_{I,k} use a FREE
learnable w_{I,k} -- they are NOT the variety-fixed mhat / log-p_U cumulant; (2) the per-head
input projection `proj` is a FREE linear map, so the effective input-space frequencies are arbitrary
(an RFF-style leak).  So this block isolates "does the aperiodic GEOMETRY alone help" from
"does the Lee-Yang COUPLING + variety-fixed amplitude help" (the latter is SecularCoupledConvAggregate).

    gamma(x) = [ c_k cos(2 pi <xi_k, proj x>),  c_k sin(...) ]_{k=1..M}   (free c_k, free proj).
"""

from __future__ import annotations

import itertools
import math

import torch
import torch.nn as nn
import torch.utils.checkpoint

from fqmath.spectrum import enumerate_spectrum_combos

PI = math.pi

# Test override for the M=1-vs-M=8 ablation: if set (int), every FQSpectralBlock uses this many
# aperiodic frequencies per head, overriding the n_freqs/n default. None = normal behavior.
FORCE_NFREQS = None
# HP-search knob: init of the learnable phase scale. Set per Optuna trial. Default 0.15.
PHASE_SCALE_INIT = 0.15


def _softplus(x, floor: float = 1e-4):
    return torch.nn.functional.softplus(x) + floor


def _softplus_inv(y: float) -> float:
    y = max(y - 1e-4, 1e-6)
    return math.log(math.expm1(y)) if y < 20 else y


class FQSpectralBlock(nn.Module):
    SPREAD = 0.95 * PI

    def __init__(self, in_dim: int, out_dim: int, n: int = 8, k_max: int = 3,
                 init_scale: float = 0.4, seed: int = 0, chunk: int | None = None,
                 phase_offset: bool = False, n_freqs: int | None = None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n = n
        self.d = 2
        self.init_scale = init_scale
        self.chunk = chunk

        # M = number of aperiodic frequencies PER HEAD = the quasicrystal spectrum's richness. It is a real
        # hyperparameter (>= n), set DIRECTLY -- NOT derived from the out/in width ratio. Deriving it
        # (num_heads = in_dim/2, M = ceil(out_per_head/2)) would, for the 2W-wide message/update blocks, force
        # num_heads ~= out_dim => out_per_head = 1 => M = 1, i.e. a SINGLE PERIODIC mode per head (NOT a
        # quasicrystal) -- a degeneracy that turns the message passing into a plain cos-feature bank.
        combos_full = enumerate_spectrum_combos(n, self.d, k_max, max_freqs=4096)   # [Mfull, n]
        # FORCE_NFREQS: module-level test override for the M=1-vs-M=8 ablation (None = use n_freqs/n).
        target = int(FORCE_NFREQS) if FORCE_NFREQS is not None else (int(n_freqs) if n_freqs is not None else n)
        M = max(1, min(target, combos_full.shape[0]))
        self.M = M
        self.per_head_out = 2 * M                                                   # cos+sin of M frequencies
        # OUTPUT-driven heads: enough heads (each emitting its full 2M-feature spectrum) to cover out_dim.
        self.num_heads = max(1, math.ceil(out_dim / self.per_head_out))
        self.out_per_head = self.per_head_out                                       # keep ALL 2M (no truncation)
        self.register_buffer("combos", torch.tensor(combos_full[:M]))               # [M, n]
        # Per-head LEARNED d=2 projection of the FULL input: each head picks the 2D plane its quasicrystal
        # lives on, so every input channel reaches every head (gradient to all inputs) and the spectrum stays
        # rich (M>=n) for any in_dim/out_dim. The effective input-space frequency is proj^T L^T k -- still an
        # aperiodic Lee-Yang spectrum; amplitudes still linear in the minors; min_minor>0 via the (r,theta)
        # chart. (Splitting the input into raw 2-channel chunks is the special case proj=I.)
        gP = torch.Generator().manual_seed(seed + 7919)
        self.proj = nn.Parameter(
            torch.randn(self.num_heads, self.d, in_dim, generator=gP) / math.sqrt(in_dim))

        # minor (Plucker) index pairs for the d=2 chart
        pairs = list(itertools.combinations(range(n), 2))
        self.register_buffer("pair_i", torch.tensor([p[0] for p in pairs]))
        self.register_buffer("pair_j", torch.tensor([p[1] for p in pairs]))
        self.C = len(pairs)                                                         # C(n,2)

        H = self.num_heads
        g = torch.Generator().manual_seed(seed)
        # identifiable, well-conditioned (r,theta) Gr_+(2,n) lift per head (no normalize/RMS gauge)
        primes = torch.sqrt(torch.tensor([2., 3., 5., 7., 11., 13., 17., 19., 23., 29., 31., 37.][:n]))
        rho0 = 0.3 * torch.log(primes / primes.mean())
        self.rho = nn.Parameter(rho0.unsqueeze(0).repeat(H, 1) + 0.3 * torch.randn(H, n, generator=g))
        gap0 = _softplus_inv(self.SPREAD / (n + 0.5))
        self.gap_raw = nn.Parameter(gap0 + 0.5 * torch.randn(H, n, generator=g))
        self.head_raw = nn.Parameter(torch.full((H,), _softplus_inv(0.4)) + 0.4 * torch.randn(H, generator=g))
        # amplitude weights w_{I,k}: c_k = sum_I L_I w_{I,k}  (AKKV Thm 2.6(3), linear in minors)
        self.amp_w = nn.Parameter(0.3 * torch.randn(H, self.C, M, generator=g))

        # Phase conditioning: learnable per-head scale so cos/sin starts in its monotone band
        # (an unscaled pre-cos phase RMS of ~1.3-1.9 rad aliases the input and loses linear signal).
        # init ~0.15 -> phase RMS ~0.5 with M=12; learnable so it can grow if higher frequencies earn
        # gradient.
        self.phase_scale_raw = nn.Parameter(torch.full((H,), _softplus_inv(PHASE_SCALE_INIT)))

        self.use_phase_offset = phase_offset
        if phase_offset:
            self.phase_offset = nn.Parameter(0.1 * torch.randn(H, M, generator=g))
        else:
            self.register_parameter("phase_offset", None)

    def L(self) -> torch.Tensor:
        r = torch.exp(self.rho)                                       # [H, n]
        S = torch.cumsum(_softplus(self.gap_raw), dim=-1)             # [H, n] absolute cumulative
        h = _softplus(self.head_raw).unsqueeze(-1)                    # [H, 1]
        theta = self.SPREAD * S / (S[:, -1:] + h)                     # [H, n] in (0, SPREAD), increasing
        L = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)   # [H, n, 2]
        return L * self.init_scale

    def minors(self, L: torch.Tensor) -> torch.Tensor:
        """2x2 minors r_i r_j sin(theta_j - theta_i) for every row pair: [H, C]."""
        Li = L[:, self.pair_i, :]                                     # [H, C, 2]
        Lj = L[:, self.pair_j, :]
        return Li[..., 0] * Lj[..., 1] - Li[..., 1] * Lj[..., 0]      # [H, C]

    def _transform(self, x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """x : [B, in_dim] -> [B, out_dim] aperiodic Fourier features (M frequencies per head)."""
        xh = torch.einsum("bi,hdi->bhd", x, self.proj)               # [B, H, 2]  per-head d=2 projection of FULL input
        xi = torch.einsum("mn,hnd->hmd", self.combos, L)              # [H, M, 2]  xi_k = L^T k
        c = torch.einsum("hc,hcm->hm", self.minors(L), self.amp_w)    # [H, M]     amplitudes linear in minors
        ps = _softplus(self.phase_scale_raw).view(1, self.num_heads, 1)   # learnable phase scale
        a = 2 * PI * ps * torch.einsum("bhd,hmd->bhm", xh, xi)        # [B, H, M]  conditioned phase
        if self.phase_offset is not None:
            a = a + 2 * PI * self.phase_offset.unsqueeze(0)
        cc = c.unsqueeze(0)                                           # [1, H, M]
        feat = torch.cat([cc * torch.cos(a), cc * torch.sin(a)], dim=-1)   # [B, H, 2M]  FULL spectrum per head
        feat = feat.reshape(feat.shape[0], self.num_heads * self.per_head_out)    # [B, H*2M]
        return feat[:, : self.out_dim]                               # trim tail only (every head keeps all M freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        L = self.L()
        chunk = self.chunk if self.chunk is not None else B
        use_ckpt = self.training and torch.is_grad_enabled()
        if B <= chunk and not use_ckpt:
            return self._transform(x, L)
        # gradient-checkpoint each chunk: frees the per-chunk autograd graph (recomputed in backward),
        # bounding memory on large message blocks so width/depth can scale safely.
        outs = []
        for i in range(0, B, chunk):
            xc = x[i:i + chunk]
            if use_ckpt:
                outs.append(torch.utils.checkpoint.checkpoint(self._transform, xc, L, use_reentrant=False))
            else:
                outs.append(self._transform(xc, L))
        return torch.cat(outs, dim=0)

    def min_minor(self) -> torch.Tensor:
        return self.minors(self.L()).min()

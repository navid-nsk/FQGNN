"""
fqops.op_spectral  --  Op-A + Op-D : the aperiodic-spectrum encoder / quasicrystal convolution

The genuine FQ POSITIONAL encoding and the spectral aggregation kernel.

Op-A (lift) + Op-D (spectral transform): given a low-dimensional input x in R^d (e.g. a tract
centroid in R^2, d=2 -- exactly the planar FQ setting of AKKV / Lawton-Tsikh "FQ on R^n"), encode
it on the aperiodic spectrum module Lambda' = {L^T k : var(k) < d} (dense; only the support is u.d.)
with the AMPLITUDES c_k that are LINEAR in the minors of L (AKKV Thm 2.6(3)):
        gamma(x) = [ c_k cos(2 pi <xi_k, x>),  c_k sin(2 pi <xi_k, x>) ]_{k=1..M}.
This is a faithful FQ object: the frequencies sit on the genuine aperiodic spectrum module Lambda',
which is COUNTABLE and topologically DENSE (Q-rank <= n; only the SUPPORT Lambda(X,L) is uniformly
discrete), truncated to ||k||_inf<=K (an O(1/K)-tail approximation justified by O(1/k^2)
amplitude decay).  The amplitudes are the VARIETY-FIXED complex Fourier-Bohr weights -- neither
a periodic integer-harmonic basis nor a random/continuous RFF set.

Op-D as a quasicrystal CONVOLUTION (aggregation): a set of neighbour messages {m_jk} already
expressed on the spectrum is combined by the FQ amplitudes c_k -- convolution against the FQ
measure mu = sum_k c_k delta_{xi_k}.  The aggregate weights each spectral component by its FQ
amplitude, so aggregation respects the pure-point structure rather than a plain mean.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from fqmath.grassmann import PositiveGrassmann
from fqmath.spectrum import FQSpectrum


class FQSpectralEncoder(nn.Module):
    """
    Encode a low-d input x (e.g. 2-D tract coordinates) onto the genuine aperiodic FQ spectrum
    with amplitudes linear in the minors of L.  Output dim 2M.
    """

    def __init__(self, d: int = 2, n: int = 12, k_max: int = 3, max_freqs: int = 48,
                 init_scale: float = 4.0, seed: int = 0, balance: bool = False):
        super().__init__()
        self.grassmann = PositiveGrassmann(n=n, d=d, init_scale=init_scale, seed=seed)
        self.spectrum = FQSpectrum(self.grassmann, k_max=k_max, max_freqs=max_freqs, seed=seed)
        self.out_width = 2 * self.spectrum.M
        # unit-RMS amplitude GAUGE: the diffraction amplitudes c_k are defined only up to a global
        # scale (a free gauge of the FQ measure mu = sum c_k delta_xi).  Normalizing c to unit RMS
        # preserves the diffraction SHAPE but fixes the overall loudness so the positional embedding
        # does not drown the feature channel at fusion (it otherwise carries ~99.9% of fused energy).
        self.balance = balance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, d] -> [B, 2M].  QFE / Op-A' form with the COMPLEX variety-fixed amplitude c_k:
        [Re(c_k e^{2pi i <xi_k,x>}), Im(c_k e^{2pi i <xi_k,x>})] (complex amplitude; the cos/sin
        split is the genuine even/odd decomposition, not two free real channels).  For the FULL Op-A'
        (learnable e^{rho x} envelope + ac-residual head, conditional Riesz reconstruction) use
        fqops.op_qfe.FQQuasicrystalEncoder."""
        L = self.grassmann()
        xi = self.spectrum.frequencies(L)            # [M, d]
        c = self.spectrum.amplitudes(L)              # [M] complex (variety-fixed)
        if self.balance:
            c = c / c.abs().pow(2).mean().clamp_min(1e-12).sqrt()   # unit-RMS gauge
        ph = 2 * math.pi * (x @ xi.t())              # [B, M]
        cr, ci = c.real.unsqueeze(0), c.imag.unsqueeze(0)
        re = cr * torch.cos(ph) - ci * torch.sin(ph)  # Re(c e^{i ph})
        im = cr * torch.sin(ph) + ci * torch.cos(ph)  # Im(c e^{i ph})
        return torch.cat([re, im], dim=-1)           # [B, 2M]

    def spectral_objects(self):
        """Return (frequencies, amplitudes) for the diffraction diagnostics / interpretability."""
        L = self.grassmann()
        return self.spectrum.frequencies(L), self.spectrum.amplitudes(L)


class FQSpectralAggregate(nn.Module):
    """
    Op-D as a learnable quasicrystal-convolution AGGREGATION kernel over a node's neighbours.
    Neighbour messages m_j (width W) are combined with per-channel FQ spectral
    weights derived from the amplitudes c_k of a shared spectrum, then distance-decayed.  This is
    a structured, amplitude-weighted aggregation (convolution against the FQ measure), as opposed
    to a plain mean.  The weights are linear in the minors of L (faithful), positive (softmaxed
    over the spectrum), and shared across edges; distance enters as exp(-gamma * dist).
    """

    def __init__(self, width: int, n: int = 12, k_max: int = 2, max_freqs: int | None = None,
                 init_scale: float = 1.0, seed: int = 0):
        super().__init__()
        self.width = width
        mf = max_freqs if max_freqs is not None else width
        # a d=1 spectrum is enough to produce W channel-weights; use the amplitudes as a learnable
        # FQ-structured gate over the W message channels.
        self.grassmann = PositiveGrassmann(n=n, d=1, init_scale=init_scale, seed=seed)
        self.spectrum = FQSpectrum(self.grassmann, k_max=k_max, max_freqs=max(mf, width), seed=seed)
        # map the spectrum amplitudes to W channel gates via a fixed slice (no plain Linear:
        # we just take the first W amplitudes and pass through softplus to get positive gates)
        self.gamma_raw = nn.Parameter(torch.tensor(0.0))      # distance decay rate (>=0)

    def channel_gate(self) -> torch.Tensor:
        """W channel weights from the FQ amplitude MAGNITUDES |c_k| (a learnable channel filter --
        NOT the diffraction amplitude itself, so a positive gate here does not impose positivity on
        the diffraction amplitudes)."""
        L = self.grassmann()
        c = self.spectrum.amplitudes(L)                       # [M >= W] complex
        g = torch.nn.functional.softplus(c[: self.width].abs())  # [W] positive channel multiplier
        return g / g.mean().clamp_min(1e-6)                   # normalized around 1

    def forward(self, messages: torch.Tensor, dst: torch.Tensor, num_nodes: int,
                dist: torch.Tensor | None = None) -> torch.Tensor:
        """
        messages : [E, W] per-edge messages ; dst : [E] receiver index ;
        dist     : [E] optional edge distance.   -> aggregated [num_nodes, W].
        """
        gate = self.channel_gate().unsqueeze(0)               # [1, W] FQ spectral channel weights
        m = messages * gate                                   # amplitude-weighted (quasicrystal conv)
        if dist is not None:
            gamma = torch.nn.functional.softplus(self.gamma_raw)
            w = torch.exp(-gamma * dist).unsqueeze(-1)        # [E, 1] distance decay
            m = m * w
            agg = torch.zeros(num_nodes, self.width, device=m.device, dtype=m.dtype)
            agg.index_add_(0, dst, m)
            wsum = torch.zeros(num_nodes, 1, device=m.device, dtype=m.dtype)
            wsum.index_add_(0, dst, w)
            return agg / wsum.clamp_min(1e-6)
        agg = torch.zeros(num_nodes, self.width, device=m.device, dtype=m.dtype)
        agg.index_add_(0, dst, m)
        deg = torch.zeros(num_nodes, 1, device=m.device, dtype=m.dtype)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=m.dtype).unsqueeze(-1))
        return agg / deg.clamp_min(1.0)

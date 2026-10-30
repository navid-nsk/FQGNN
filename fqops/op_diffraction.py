"""
fqops.op_diffraction  --  Op-E : the FQ diffraction / structure-factor module (DIAGNOSTIC ONLY).

gamma_hat = sum |c_xi|^2 delta_xi is the alpha=2 Hadamard power that ejects the
Lee-Yang/TP family (Belton-et-al Thm 9.7), so Op-E is NEVER an in-family loss/regularizer -- it is
a reported diagnostic (in particular there is no 'stealthy hyperuniform' or entropy-sparsity loss).
This block reports the structure factor, the Bragg participation (HIGH = dense pure-point,
the desired regime), an unconditional-HU low-frequency proxy (AKKV Thm 2.9), and the gamma^{-1}-l1
low-frequency tail (Favorov-genFQ Cor 2) -- the ONLY admissible optional soft penalty.
The empirical pure-point FRACTION rho_pp (the FQ-vs-RFF discriminator) needs a continuous-component
estimate from data; see fqmath.diagnostics.  Consumes the VARIETY-FIXED amplitudes of FQSpectrum.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fqmath.grassmann import PositiveGrassmann
from fqmath.spectrum import FQSpectrum
from fqmath.diffraction import (structure_factor, low_freq_l1_tail, bragg_participation,
                                hyperuniformity_rate_proxy)


class FQDiffraction(nn.Module):
    def __init__(self, d: int = 2, n: int = 12, k_max: int = 3, max_freqs: int = 64,
                 init_scale: float = 1.0, seed: int = 0, amplitude: str = "secular"):
        super().__init__()
        self.grassmann = PositiveGrassmann(n=n, d=d, init_scale=init_scale, seed=seed)
        self.spectrum = FQSpectrum(self.grassmann, k_max=k_max, max_freqs=max_freqs, seed=seed,
                                   amplitude=amplitude)

    def forward(self):
        L = self.grassmann()
        return self.spectrum.frequencies(L), self.spectrum.amplitudes(L)

    def structure_factor(self):
        freqs, amps = self.forward()
        return freqs, structure_factor(freqs, amps)

    @torch.no_grad()
    def diagnostics(self, radius: float = 0.5) -> dict:
        """Reported Op-E diagnostics (DETACHED -- never added to the training loss)."""
        freqs, amps = self.forward()
        return {
            "bragg_participation": float(bragg_participation(amps)),     # high = dense pure-point
            "lowfreq_l1_tail": float(low_freq_l1_tail(freqs, amps, radius=radius)),
            "hu_lowfreq_proxy": float(hyperuniformity_rate_proxy(freqs, amps, radius=radius)),
        }

    def optional_soft_penalty(self, radius: float = 0.5) -> torch.Tensor:
        """The ONLY admissible soft penalty (Favorov gamma^{-1}-l1 low-frequency tail).
        Diagnostic-FIRST: use a SMALL weight if at all; it does not reward sparsity or positivity."""
        freqs, amps = self.forward()
        return low_freq_l1_tail(freqs, amps, radius=radius)

    def recovered_scales(self, km_per_unit: float = 1.0, top: int = 12):
        """Dominant aperiodic wavelengths (1/|xi|) weighted by structure factor -- interpretability."""
        freqs, S = self.structure_factor()
        mag = freqs.norm(dim=-1).clamp_min(1e-6)
        order = torch.argsort(S, descending=True)[:top]
        return (km_per_unit / mag[order]).detach()

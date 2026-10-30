"""
fqops.op_velocity  --  the velocity / derivative channel on the SHARED FQ spectrum.

Strungaru-Terauds Prop 2.4: the derivative measure D^alpha mu stays translation-bounded on the SAME
spectrum Lambda', with amplitudes scaled by (2 pi i xi).  So the velocity field of the FQ synthesis
f(x) = Re sum_k c_k e^{2 pi i <xi_k, x>} is

    v_c(x) = d/dx_c f(x) = Re sum_k (2 pi i xi_{k,c}) c_k e^{2 pi i <xi_k, x>} ,   c = 1..d,

i.e. the SAME frequencies xi_k and the SAME (variety-fixed, complex) amplitudes c_k as the position
encoder, only re-weighted by (2 pi i xi_{k,c}).  For Task Alpha this gives the SOG/COG velocity channel
a principled SHARED-SPECTRUM identity (not an ad-hoc separate feature).  Pass the position encoder's
FQSpectrum/grassmann so the velocity is the derivative of the SAME FQ.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from fqmath.grassmann import PositiveGrassmann
from fqmath.spectrum import FQSpectrum

PI = math.pi


class FQVelocityChannel(nn.Module):
    """The derivative channel D mu of a SHARED FQ spectrum (amplitudes x 2 pi i xi).  Output [B, 2*M*d]:
    a linear readout reconstructs the d-component velocity v(x)."""

    def __init__(self, d: int = 2, n: int = 8, k_max: int = 3, max_freqs: int = 24,
                 init_scale: float = 4.0, seed: int = 0, grassmann=None, spectrum=None):
        super().__init__()
        self.d = d
        # SHARE the position encoder's geometry/spectrum if given (so velocity = D of the SAME FQ).
        self.grassmann = grassmann if grassmann is not None else PositiveGrassmann(
            n=n, d=d, init_scale=init_scale, seed=seed)
        self.spectrum = spectrum if spectrum is not None else FQSpectrum(
            self.grassmann, k_max=k_max, max_freqs=max_freqs, seed=seed, amplitude="secular")
        self.M = self.spectrum.M
        self.out_dim = 2 * self.M * d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, d] -> [B, 2*M*d]  derivative-channel features on the shared spectrum."""
        L = self.grassmann()
        xi = self.spectrum.frequencies(L)                       # [M, d]
        c = self.spectrum.amplitudes(L)                         # [M] complex (variety-fixed)
        ph = 2 * PI * (x @ xi.t())                              # [B, M]
        cos, sin = torch.cos(ph), torch.sin(ph)                 # [B, M]
        outs = []
        for cdir in range(self.d):
            # derivative amplitude in direction cdir: (2 pi i xi_{k,cdir}) c_k
            cv = (2 * PI) * xi[:, cdir].to(c.dtype) * (1j * c)  # [M] complex
            cr, ci = cv.real.unsqueeze(0), cv.imag.unsqueeze(0)
            re = cr * cos - ci * sin                            # Re(cv e^{i ph})
            im = cr * sin + ci * cos                            # Im(cv e^{i ph})
            outs += [re, im]
        return torch.cat(outs, dim=-1)                          # [B, 2*M*d]

    def min_minor(self) -> torch.Tensor:
        return self.grassmann.min_minor()

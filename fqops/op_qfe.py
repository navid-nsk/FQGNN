"""
fqops.op_qfe  --  Op-A' / QFE : the full absolute-position FQ synthesis.

The base FQ synthesis QFE(x) = [Re(c_k e^{2 pi i <xi_k,x>}), Im(...)] (FQSpectralEncoder, with the
VARIETY-FIXED complex amplitudes c_k) reconstructs a band-limited-to-Lambda' field, subject to three
qualifications that shape this module:

  1. CONDITIONAL reconstruction: only countably many band/window lengths admit a Riesz basis
     (Grepstad-Lev Thm 1.1); a continuous learnable scale lands inadmissible w.p. 1 (frame, not
     basis), and uniform discreteness alone gives no lower-Riesz bound (Cabrelli-Hare-Molter).
     Convergence is BESICOVITCH / average, NOT uniform/pointwise.
  2. MISSING ENVELOPE: the exact representable class carries an e^{lambda_0 x} real-exponential
     envelope a purely-oscillatory basis cannot capture (ACV) -> we add a LEARNABLE envelope
     e^{rho . x}.
  3. AC RESIDUAL: a pure-point QFE basis cannot represent the generic ac / Stepanov residual
     (Spindeler-Strungaru-NAP Cor 7.2) -> we add a small AC-RESIDUAL head.  This head is a plain
     Linear -> it is OUT-of-family BY DESIGN and labeled as such (admissibility rule: task
     nonlinearity lives in non-spectral heads, not in the spectral stage).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .op_spectral import FQSpectralEncoder


class FQQuasicrystalEncoder(nn.Module):
    """Op-A' / QFE (full): enveloped pure-point FQ synthesis + an out-of-family ac-residual head."""

    def __init__(self, d: int = 2, n: int = 12, k_max: int = 3, max_freqs: int = 48,
                 init_scale: float = 4.0, seed: int = 0, ac_dim: int = 8, balance: bool = True):
        super().__init__()
        self.encoder = FQSpectralEncoder(d=d, n=n, k_max=k_max, max_freqs=max_freqs,
                                         init_scale=init_scale, seed=seed, balance=balance)
        # learnable real-exponential envelope e^{rho . x} (ACV; init 0 -> envelope = 1)
        self.rho = nn.Parameter(torch.zeros(d))
        # ac / smooth residual head -- OUT-of-family by design (a plain Linear; labeled, non-spectral)
        self.ac_head = nn.Linear(d, ac_dim) if ac_dim > 0 else None
        self.out_dim = self.encoder.out_width + (ac_dim if ac_dim > 0 else 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, d] -> [B, 2M (+ ac_dim)]."""
        env = torch.exp(x @ self.rho).unsqueeze(-1)            # [B, 1]  e^{rho . x} envelope
        qfe = self.encoder(x) * env                            # [B, 2M] enveloped pp synthesis
        if self.ac_head is None:
            return qfe
        ac = torch.tanh(self.ac_head(x))                       # [B, ac_dim] out-of-family ac residual
        return torch.cat([qfe, ac], dim=-1)

    def spectral_objects(self):
        return self.encoder.spectral_objects()

    def min_minor(self) -> torch.Tensor:
        return self.encoder.grassmann.min_minor()

"""
fqops.op_zero_field  --  Op-F : FQ zero-field synthesis.

Basis (Favorov-genFQ Thm 2): the secular function
f(x) = p_U(Phi_L(x)) = det(I - diag(e^{2 pi i L x}) U) has zero set EXACTLY the FQ point set Lambda.
Its LOGARITHMIC DERIVATIVE (the "zero field")
        f'/f = grad_x log f(x)
is a one-sided Fourier-Bohr feature -- the local field / density-of-states near Lambda, the SYNTHESIS
DUAL of Op-C (membership): Op-C says HOW FAR x is from Lambda (|f|), Op-F gives the local FIELD around
Lambda.  Analytic + differentiable:

    d/dx_k log f = - tr[ (I - M)^{-1} diag( 2 pi i L[:,k] z ) U ] ,   M = diag(z) U, z = e^{2 pi i L x}.

Feature: feat_F(x) = [ Re(grad log f), Im(grad log f) ] in R^{2d}, plus optionally log|f| (the Op-C
distance).  Near a zero f -> 0 the field has a pole (the expected 1/(x-lambda) behaviour); a small
Tikhonov jitter on (I - M) keeps it finite for training stability.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from fqmath.grassmann import PositiveGrassmann
from fqmath.lee_yang import SecularUnitary

PI = math.pi


class FQZeroField(nn.Module):
    """Op-F: the FQ zero-field f'/f of the secular variety, differentiable in (L, U)."""

    def __init__(self, n: int = 6, d: int = 2, init_scale: float = 1.0, seed: int = 0,
                 coupled: bool = True, jitter: float = 1e-3, with_logabs: bool = True):
        super().__init__()
        self.n = n
        self.d = d
        self.jitter = jitter
        self.with_logabs = with_logabs
        self.grassmann = PositiveGrassmann(n=n, d=d, init_scale=init_scale, seed=seed)
        self.secular = SecularUnitary(n=n, coupled=coupled, seed=seed)
        self.out_dim = 2 * d + (1 if with_logabs else 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, d] -> [B, 2d (+1)]   (Re/Im of grad log f, optionally log|f|)."""
        # derive the complex dtype from the input (fp32->complex64, fp64->complex128) so backward
        # never mixes complex64/complex128 (the default working precision is fp32).
        cdtype = torch.complex128 if x.dtype == torch.float64 else torch.complex64
        L = self.grassmann()                                    # [n, d]
        U = self.secular.unitary().to(cdtype)                  # [n, n] complex
        n = self.n
        phase = x @ L.t()                                       # [B, n]  (L x)_j
        z = torch.exp(2j * PI * phase).to(cdtype)              # [B, n]
        M = z.unsqueeze(-1) * U                                 # diag(z) U -> [B, n, n]
        I = torch.eye(n, dtype=cdtype, device=M.device)
        ImM = I - M + self.jitter * I                           # Tikhonov jitter for stability
        Minv = torch.linalg.inv(ImM)                            # [B, n, n]
        grads = []
        for k in range(self.d):
            dz = (2j * PI * L[:, k]).to(cdtype).unsqueeze(0) * z   # [B, n]  dz/dx_k
            dM = dz.unsqueeze(-1) * U                           # diag(dz) U -> [B, n, n]
            g = -torch.diagonal(Minv @ dM, dim1=-2, dim2=-1).sum(-1)        # [B] = d log f / dx_k
            grads.append(g)
        G = torch.stack(grads, dim=-1)                          # [B, d] complex
        feat = [G.real, G.imag]
        if self.with_logabs:
            f = torch.linalg.det(ImM)                           # ~ p_U (with jitter) ; |f| = Op-C distance
            feat.append(f.abs().clamp_min(1e-9).log().unsqueeze(-1))
        return torch.cat(feat, dim=-1)                          # [B, 2d (+1)]

    def min_minor(self) -> torch.Tensor:
        return self.grassmann.min_minor()

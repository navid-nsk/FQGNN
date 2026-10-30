"""
fqmath.membership
=================

The computable almost-periodicity MEMBERSHIP gate (LSS-2020 Cor 3.20/3.21).

Almost-periodicity is a strict hierarchy MAP ⊋ Bap ⊋ Wap, not a single "iff".  The COMPUTABLE
certificate that an amplitude family is the Fourier-Bohr spectrum of a Besicovitch-almost-periodic
function (and therefore that the operator did NOT leak off-spectrum / continuous energy -- the RFF
failure mode) is the PARSEVAL / Riesz-Fischer identity:

        M_A( |f|^2 )  =  sum_k |c_k|^2          (Riesz-Fischer, LSS-2020 Cor 3.20),

where M_A is the Besicovitch (van-Hove) mean and f(x) = Re sum_k c_k e^{2pi i <xi_k, x>} is the FQ
synthesis.  For a real signal over the canonical +representative frequencies (the conjugate partner
c_{-k}=conj(c_k) implied), the per-mode average gives

        M_A( f^2 )  =  (1/2) sum_k |c_k|^2 .

`parseval_residual` measures the relative gap between the EMPIRICAL mean of f^2 over a large sample
and (1/2) sum|c_k|^2.  A SMALL residual certifies Bap-membership (the amplitudes are genuine FB
coefficients of a fixed geometry); a LARGE residual flags off-spectrum leakage (a free tap / RFF
contamination).  Together with the amoeba certificate (genericity.py) it forms the in-family check.
"""

from __future__ import annotations

import torch


def fq_synthesis(frequencies: torch.Tensor, amplitudes: torch.Tensor,
                 x: torch.Tensor) -> torch.Tensor:
    """
    Real FQ field f(x) = Re sum_k c_k e^{2pi i <xi_k, x>} over the canonical +representative
    frequencies (Hermitian partner implied), i.e. f = sum_k [Re(c_k) cos(theta_k) - Im(c_k) sin(theta_k)].
    frequencies [M, d] (real), amplitudes [M] (complex), x [B, d] -> f [B] (real).
    """
    theta = 2 * torch.pi * (x @ frequencies.t())                    # [B, M]
    cr = amplitudes.real.unsqueeze(0)
    ci = amplitudes.imag.unsqueeze(0)
    return (cr * torch.cos(theta) - ci * torch.sin(theta)).sum(-1)   # [B]


def parseval_residual(frequencies: torch.Tensor, amplitudes: torch.Tensor,
                      n_sample: int = 8192, box: float = 64.0, seed: int = 0) -> torch.Tensor:
    """
    Relative Parseval/Riesz-Fischer residual  |M_A(f^2) - (1/2) sum|c_k|^2| / ((1/2) sum|c_k|^2),
    estimated by sampling x ~ Unif([-box, box]^d).  SMALL => Bap-membership certificate passes
    (no off-spectrum leakage); LARGE => the operator leaked continuous/RFF energy.
    """
    d = frequencies.shape[1]
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(n_sample, d, generator=g) * 2 - 1) * box
    f = fq_synthesis(frequencies, amplitudes, x.to(frequencies.dtype))
    lhs = f.pow(2).mean()
    rhs = 0.5 * amplitudes.abs().pow(2).sum()
    return (lhs - rhs).abs() / rhs.clamp_min(1e-12)


def bap_membership_ok(frequencies: torch.Tensor, amplitudes: torch.Tensor,
                      tol: float = 0.05, **kw) -> bool:
    """True iff the Parseval residual is below `tol` (the FB-of-a-fixed-geometry membership gate)."""
    return bool(parseval_residual(frequencies, amplitudes, **kw) < tol)

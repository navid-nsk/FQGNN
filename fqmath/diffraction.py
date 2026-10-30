"""
fqmath.diffraction
==================

The pure-point diffraction / structure factor of the FQ -- DIAGNOSTIC ONLY (Op-E).

Scope:
  * The FQ is NOT "stealthy hyperuniform".  Our spectrum Lambda' is a dense rank-(<=n) Z-module,
    so nonzero Bragg peaks accumulate at the origin -- there is NO exclusion zone and the measure is
    NOT stealthy in Torquato's strict S(k)=0-on-a-punctured-ball sense.  Hyperuniformity itself is
    UNCONDITIONAL (AKKV Thm 2.9), with rate O(R^{d-1}); it is a property of the coupling.
  * gamma_hat = sum |c_xi|^2 delta_xi is the alpha=2 Hadamard power, which Belton-et-al Thm 9.7
    forbids as family-preserving.  So Op-E is ILLEGITIMATE as an in-family loss/regularizer and is
    DIAGNOSTIC ONLY.  In particular, no loss term minimizes spectral ENTROPY: that would reward a
    SPARSE / periodic-comb spectrum (the M=1 degeneracy).
  * If a soft penalty is used at all, it must be the gamma^{-1}-weighted l1 LOW-FREQUENCY tail
    (Favorov-genFQ Cor 2) over a EUCLIDEAN ball, and it must reward a DENSE pure-point spectrum, not
    a sparse one; do NOT impose amplitude positivity (the signed/complex amplitude is the d>=2
    escape from crystallization -- LO-Annals).

This module reports diagnostics on the MODEL's spectrum (frequencies, complex amplitudes); the
DATA-side S(k) pre-flight and the empirical pure-point fraction live in fqmath.preflight /
fqmath.diagnostics.
"""

from __future__ import annotations

import torch


def structure_factor(frequencies: torch.Tensor, amplitudes: torch.Tensor) -> torch.Tensor:
    """S(xi_k) = |c_k|^2 for the discrete spectrum (complex-safe).  Diagnostic readout only.
    frequencies : [M, d]   amplitudes : [M] (complex)   ->   S : [M] (>= 0)."""
    return amplitudes.abs().pow(2)


def low_freq_l1_tail(frequencies: torch.Tensor, amplitudes: torch.Tensor,
                     radius: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """
    The gamma^{-1}-weighted l1 LOW-FREQUENCY amplitude tail  sum_{0 < |xi| < radius} |c_xi| / |xi|
    (Favorov-genFQ Cor 2 -- the quantity that actually controls the exponential-type / FQ class).
    EUCLIDEAN ball (NOT l-inf, which would spuriously read even a lattice as non-HU).

    This is the ONLY admissible soft penalty (diagnostic-FIRST; use a small weight if at all).
    It does NOT reward sparsity and does NOT push amplitudes positive.
    """
    r = frequencies.norm(dim=-1)                                 # Euclidean |xi|
    near = (r > eps) & (r < radius)
    if near.sum() == 0:
        return amplitudes.abs().sum() * 0.0
    return (amplitudes.abs()[near] / r[near].clamp_min(eps)).sum()


def pure_point_fraction(pp_power: torch.Tensor, continuous_power: torch.Tensor) -> torch.Tensor:
    """
    rho_pp = (pure-point mass) / (pure-point mass + continuous mass)  -- the FQ-vs-RFF discriminator
    with a theorem behind it (Strungaru-Terauds: pp-mass > 0 <=> FQ; purely continuous <=> RFF).
    rho_pp -> 1 for a genuine FQ, -> 0 for an RFF / continuous-spectrum object.  The CONTINUOUS mass
    must be estimated EMPIRICALLY from a point configuration (fqmath.diagnostics); this is the pure
    formula.  Both inputs are non-negative scalars/tensors.
    """
    pp = pp_power
    return pp / (pp + continuous_power).clamp_min(1e-12)


def bragg_participation(amplitudes: torch.Tensor) -> torch.Tensor:
    """
    Participation ratio of the Bragg spectrum  (sum S)^2 / (M * sum S^2)  in (0, 1] -- a DIAGNOSTIC
    of how MANY peaks carry the mass (HIGH = dense/distributed pure-point, the desired regime;
    LOW = a few sharp peaks = the periodic-comb degeneracy).  Reported, never
    minimized (we want it HIGH, not low).
    """
    S = amplitudes.abs().pow(2)
    num = S.sum().pow(2)
    den = (amplitudes.numel() * S.pow(2).sum()).clamp_min(1e-12)
    return num / den


def hyperuniformity_rate_proxy(frequencies: torch.Tensor, amplitudes: torch.Tensor,
                               radius: float = 0.25, eps: float = 1e-6) -> torch.Tensor:
    """
    Unconditional-HU diagnostic (AKKV Thm 2.9): the low-frequency diffracted power
    mean_{0<|xi|<radius} |c_xi|^2.  Reported as an interpretability diagnostic of the learned
    spectrum's low-frequency suppression -- NOT a 'stealthy S(k)=0' claim and NOT a loss term.
    """
    r = frequencies.norm(dim=-1)
    near = (r > eps) & (r < radius)
    denom = near.sum().clamp_min(1)
    return (amplitudes.abs().pow(2)[near]).sum() / denom

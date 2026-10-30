"""
fqmath.genericity
=================

The Lee-Yang / FQ in-family CERTIFICATE: the amoeba-disjointness certificate (ACV Cor 3.3 /
Lemma 2.4) + a uniform-genericity guard (LT-II Thm 3).  Its margins are DETERMINISTIC and
closed-form (fqmath.amoeba is only a stochastic visualization helper), so in-family membership is
theorem-backed rather than checked empirically, and the certificate can be evaluated at every
training step as an in-family monitor.

The objects (secular polynomial p_U(z) = det(I - diag(z) U), normalized so that p_U(0)=1):

  * REAL-ROOTEDNESS is structural: on the torus diag(z)U is unitary (U=exp(iH)), so its eigenvalues
    are on the unit circle and p_U=0 <=> an eigenphase hits 0 <=> real -- Lee-Yang by construction.
    We therefore CERTIFY (not enforce) it, and watch the degeneracy that breaks it (M=1).

  * Amoeba-disjointness: p Lee-Yang <=> R_+^n ∪ R_-^n ⊆ A(p)^c, i.e. p_U has NO zero with all
    |z_j| > 1 (or all < 1).  `amoeba_positive_ray_margin` evaluates |p_U| along the positive ray
    z = exp(s * ell) (s>0 => all |z_j|>1) and returns its MINIMUM -- a deterministic margin that is
    > 0 exactly when the ray is in the amoeba complement.

  * M=1 / periodicity degeneracy guards: `coupling_margin` = off-diagonal energy of U (0 <=>
    decoupled product p_U=prod(1-z_j) = the periodic comb), and `eigenphase_min_gap` = the smallest
    eigenphase separation over a torus sample (-> 0 = double root = the M=1 collapse, the
    Alon-Vinzant non-squarefree / binomial-leak failure mode).

All margins are torch tensors (differentiable where it helps) so they can be reported AND used as
soft guards.
"""

from __future__ import annotations

import torch


def secular_poly(z: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """p_U(z) = det(I - diag(z) U), normalized so that p_U(0)=1.  z [..., n] complex; U [n,n]
    complex.  Returns [...] complex, batched and differentiable in U."""
    n = U.shape[-1]
    M = z.unsqueeze(-1) * U                                          # diag(z) U, [..., n, n]
    I = torch.eye(n, dtype=M.dtype, device=M.device)
    return torch.linalg.det(I - M)


def amoeba_positive_ray_margin(U: torch.Tensor, ell: torch.Tensor,
                               s_max: float = 2.0, n_grid: int = 96) -> torch.Tensor:
    """
    Amoeba-disjointness certificate (deterministic).  Along the positive ray z = exp(s * ell),
    s in (0, s_max] (ell > 0 => all |z_j| = e^{s ell_j} > 1, i.e. the positive orthant R_+^n), return
        min_s |p_U(exp(s ell))| .
    p is Lee-Yang on this line iff this is > 0 (the orthant lies in the amoeba complement A(p)^c).
    """
    s = torch.linspace(1e-3, s_max, n_grid, device=ell.device, dtype=ell.real.dtype if ell.is_complex() else ell.dtype)
    z = torch.exp(s.unsqueeze(-1) * ell.to(s.dtype).unsqueeze(0))    # [n_grid, n] real > 1
    val = secular_poly(z.to(torch.complex64), U).abs()              # [n_grid]
    return val.min()


def coupling_margin(U: torch.Tensor) -> torch.Tensor:
    """Off-diagonal energy ||U - diag(diag U)||_F of the unitary U.  0 <=> decoupled product
    p_U = prod_j (1 - z_j) = the PERIODIC COMB degeneracy (the ablation, never the model)."""
    off = U - torch.diag_embed(torch.diagonal(U))
    return off.abs().pow(2).sum().sqrt()


def eigenphase_min_gap(U: torch.Tensor, n_sample: int = 64, seed: int = 0) -> torch.Tensor:
    """
    Smallest eigenphase separation min_z min_{j<k} |theta_j(z) - theta_k(z)| over a torus sample
    z ~ Unif(T^n).  -> 0 signals an eigenphase COLLISION = a double root of p_U = the M=1
    / non-squarefree degeneracy (Alon-Vinzant Thm 1.11).  Diagnostic (no grad through eigvals here).
    """
    n = U.shape[-1]
    g = torch.Generator().manual_seed(seed)
    theta = 2 * torch.pi * torch.rand(n_sample, n, generator=g)
    z = torch.exp(1j * theta.to(torch.complex64))
    with torch.no_grad():
        M = z.unsqueeze(-1) * U                                      # [n_sample, n, n]
        ev = torch.linalg.eigvals(M)                                # [n_sample, n]
        ph = torch.angle(ev)
        ph, _ = torch.sort(ph, dim=-1)
        gaps = torch.diff(ph, dim=-1)                               # adjacent gaps
        # wrap-around gap
        wrap = (ph[:, :1] + 2 * torch.pi) - ph[:, -1:]
        allg = torch.cat([gaps, wrap], dim=-1)
        return allg.min()


def lee_yang_certificate(U: torch.Tensor, ell: torch.Tensor,
                         s_max: float = 2.0, gap_tol: float = 1e-3,
                         coupling_tol: float = 1e-3) -> dict:
    """
    The combined per-step in-family certificate.  Returns a dict of margins + an `ok` flag:
      amoeba_margin   > 0      (positive orthant in A(p)^c)
      coupling_margin > tol    (NOT the decoupled periodic comb)
      eigenphase_gap  > tol    (no M=1 double-root collapse)
    `ok` is True iff all three pass.  (Real-rootedness itself is structural via U=exp(iH).)
    """
    am = amoeba_positive_ray_margin(U, ell, s_max=s_max)
    cm = coupling_margin(U)
    gp = eigenphase_min_gap(U)
    ok = bool((am > 0) and (cm > coupling_tol) and (gp > gap_tol))
    return {"amoeba_margin": float(am), "coupling_margin": float(cm),
            "eigenphase_gap": float(gp), "ok": ok}

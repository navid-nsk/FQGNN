"""
fqmath.lee_yang
===============

The Lee-Yang variety and its secular realization -- the heart of the Fourier-quasicrystal.

Two faithful, differentiable, always-in-family realizations of a strict Lee-Yang variety X
(AKKV Def. 2.2 / Def. 4.1), both consumed by the FQ operators (fqops):

1. SECULAR-UNITARY family (Kurasov-Sarnak -- the d=1 SPECIAL CASE).
   The secular determinant is a d=1 object; completeness is proven ONLY for d=1, N-valued
   masses (ACV Thm 1.1).  It is NOT the universal d>=2 generator (no converse for d>=2; the d>=2
   universal object is the M-stable Laurent map, LT-II Thm 2).  This block follows KS, not AKKV.
   For a UNITARY U in U(n), define the secular polynomial (normalized so that p_U(0)=1)
        p_U(z) = det( I - diag(z) U ) ,   z in (C*)^n.
   On the torus z in T^n, M_U(z) = diag(z) U is unitary, so its eigenvalues lie on the unit
   circle: eig(M_U(z)) = { e^{i theta_j(z)} }.  Then p_U(z) = 0 iff some theta_j(z) = 0 (an
   eigenvalue equals +1).  Because M_U(z) is unitary for ALL z in T^n, every zero of p_U is
   REAL -- this is the Lee-Yang / real-rootedness property, realized concretely.  The variety
   X = { z : p_U(z) = 0 } is a codimension-1 strict Lee-Yang variety; with a positive vector
   ell (d=1, fqmath.grassmann) the pullback Lambda = { x : exp(2 pi i ell x) in X } is a
   genuine 1-D Fourier quasicrystal.

   The learnable object is U = exp(i H) with H Hermitian (H = (A + A^H)/2 from an unconstrained
   complex A).  Unitarity -- hence real-rootedness -- holds for ANY A, so no gradient step can
   leave the Lee-Yang family.  The COUPLING that distinguishes a quasicrystal from decoupled
   Fourier features is literally the off-diagonal of U: if U were diagonal, theta_j(z) = arg z_j
   and the eigenphases decouple per coordinate (the ablation, never the model).

   The PREFERRED Op-B feature is the eigenphases theta_j(z) THEMSELVES (`eigenphases`),
   with the free sanity check prod_j (1 - e^{i theta_j}) = p_U (`secular_from_eigenphases`).  The
   power-sum / trace moments
        t_k(z) = tr( M_U(z)^k ) = sum_j e^{i k theta_j(z)} ,   k = 1..K,
   are a PROJECT-ORIGINAL featurization -- NOT AKKV-faithful (AKKV has no power sums) and, being
   entrywise powers, family-breaking (Belton-et-al).  `trace_moments` serves only as the GPU-fast
   d=1 shadow / ablation; the faithful coupled message routes through the variety (Op-C), not Op-B.

2. MOEBIUS-CURVE family (AKKV Ex. 1.3 / Ex. 4.13; LT Thm 1 -- a genuine d-dim variety).
   A rational curve in (P^1)^n given coordinate-wise by Blaschke/Moebius factors
        m_j(t) = (t - conj(a_j)) / (t - a_j) ,   a_j in the upper half-plane,
   maps real t to the unit circle, so the curve image lies in T^n; distinct a_j make it a strict
   Lee-Yang curve (this is the curve underlying AKKV Example 1.3, whose FQ has density
   1 + sqrt2 + sqrt3).  Eliminating t between adjacent coordinates yields the exact bilinear
   VARIETY RELATIONS
        p_j(z) = A_j z_j + B_j - C_j z_j z_{j+1} - D_j z_{j+1} ,   j = 1..n-1,
   where (A,B,C,D)_j come from the Moebius composition m_{j+1} o m_j^{-1}.  p_j(z) = 0 exactly
   when z lies on the curve, i.e. when x lies on the quasicrystal.  These relations COUPLE
   adjacent torus coordinates multiplicatively (the z_j z_{j+1} term) -- a coupling no decoupled
   cosine basis can express.  The learnable object is { a_j } in the upper half-plane (real part
   free, imaginary part = softplus), Lee-Yang-preserving for any value.

Both families are parameterized so the Lee-Yang property is STRUCTURAL (never penalized).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


PI = math.pi


# ======================================================================================
# 1. SECULAR-UNITARY family
# ======================================================================================
class SecularUnitary(nn.Module):
    """
    A learnable unitary U = exp(i H) and the secular operator M_U(z) = diag(z) U with its
    coupled-eigenphase trace moments.  The decisive FQ-coupling lives in the off-diagonal of U;
    `coupled=False` freezes U to the identity (decoupled ablation, never the model).
    """

    def __init__(self, n: int, n_moments: int | None = None, coupled: bool = True, seed: int = 0):
        super().__init__()
        self.n = int(n)
        self.K = int(n_moments) if n_moments is not None else int(n)  # K >= n (Newton complete)
        if self.K < self.n:
            raise ValueError(f"need K >= n for Newton-complete moments: K={self.K}, n={self.n}")
        self.coupled = bool(coupled)
        g = torch.Generator().manual_seed(seed)
        # unconstrained complex generator A; H = (A + A^H)/2 ; U = exp(iH).  Start near I.
        self.A_real = nn.Parameter(0.05 * torch.randn(n, n, generator=g))
        self.A_imag = nn.Parameter(0.05 * torch.randn(n, n, generator=g))

    def hermitian(self) -> torch.Tensor:
        A = torch.complex(self.A_real, self.A_imag)
        H = 0.5 * (A + A.conj().transpose(-2, -1))
        return H

    def unitary(self) -> torch.Tensor:
        """U = exp(i H), unitary for any parameter values (-> real-rootedness, always-in-family)."""
        if not self.coupled:
            return torch.eye(self.n, dtype=torch.complex64, device=self.A_real.device)
        H = self.hermitian()
        return torch.linalg.matrix_exp(1j * H.to(torch.complex64))

    def operator(self, z: torch.Tensor) -> torch.Tensor:
        """
        Build M_U(z) = diag(z) U for a batch of torus points z (|z|=1).
        z : complex [..., n]   ->   M : complex [..., n, n]
        """
        U = self.unitary()
        return z.unsqueeze(-1) * U  # broadcast diag(z) @ U  ->  [..., n, n]

    def trace_moments(self, z: torch.Tensor) -> torch.Tensor:
        """
        Power sums t_k = tr(M^k) = sum_j e^{i k theta_j(z)}, k=1..K, as a real feature
        [Re t_1, Im t_1, ..., Re t_K, Im t_K].   z complex [B, n] -> [B, 2K].
        Pure matmul (no eig in the forward path), fully differentiable.
        """
        M = self.operator(z)                       # [B, n, n]
        P = M
        moments = []
        for k in range(self.K):
            tk = torch.diagonal(P, dim1=-2, dim2=-1).sum(-1)   # tr(M^{k+1})  [B]
            moments.append(tk)
            if k < self.K - 1:
                P = P @ M
        T = torch.stack(moments, dim=-1)           # [B, K] complex
        return torch.cat([T.real, T.imag], dim=-1) / math.sqrt(self.n)  # [B, 2K], scaled

    def eigenphases(self, z: torch.Tensor) -> torch.Tensor:
        """
        The coupled eigenphases theta_j(z) in (-pi, pi], via eigvals of the unitary M_U(z).
        z complex [B, n] -> theta real [B, n].  (Interpretability head; not used in the hot
        training path which uses trace_moments.)
        """
        M = self.operator(z)
        ev = torch.linalg.eigvals(M)               # [B, n] complex on the unit circle
        return torch.angle(ev)

    def secular_value(self, z: torch.Tensor) -> torch.Tensor:
        """p_U(z) = det(I - M_U(z)) = det(I - diag(z) U), complex [B] (normalized so that
        p_U(0)=1, no monomial factor).  Zero set = the FQ (for z on a curve in T^n)."""
        M = self.operator(z)
        I = torch.eye(self.n, dtype=M.dtype, device=M.device)
        return torch.linalg.det(I - M)

    def secular_from_eigenphases(self, z: torch.Tensor) -> torch.Tensor:
        """Sanity reconstruction prod_j (1 - e^{i theta_j(z)}) which must equal p_U(z)=det(I-M)
        (Op-B sanity check).  Returns complex [B]."""
        theta = self.eigenphases(z)                                  # [B, n]
        return torch.prod(1.0 - torch.exp(1j * theta.to(torch.complex64)), dim=-1)

    @property
    def out_dim(self) -> int:
        return 2 * self.K


# ======================================================================================
# 2. MOEBIUS-CURVE family (genuine d-dim variety relations, AKKV Ex. 1.3)
# ======================================================================================
class MoebiusVariety(nn.Module):
    """
    The strict Lee-Yang rational curve z_j = m_j(t) = (t - conj a_j)/(t - a_j), a_j in the
    upper half-plane, and the exact bilinear variety relations p_j(z) between adjacent
    coordinates.  p_j(z)=0 on the quasicrystal.  Learnable a_j (always Lee-Yang).
    """

    def __init__(self, n: int, seed: int = 0, coupled: bool = True):
        super().__init__()
        self.n = int(n)
        self.coupled = bool(coupled)
        g = torch.Generator().manual_seed(seed)
        # a_j = re + i * softplus(im_raw)  (imaginary part > 0  => upper half-plane => Lee-Yang)
        self.a_re = nn.Parameter(0.3 * torch.randn(n, generator=g))
        self.a_im_raw = nn.Parameter(0.5 + 0.1 * torch.randn(n, generator=g))

    def a(self) -> torch.Tensor:
        return torch.complex(self.a_re, torch.nn.functional.softplus(self.a_im_raw) + 1e-3)

    def _moebius_coeffs(self) -> Tuple[torch.Tensor, ...]:
        """
        Coefficients (A,B,C,D)_j of the bilinear relation between z_j and z_{j+1}, derived from
        the Moebius composition m_{j+1} o m_j^{-1} (represented by 2x2 matrices, det cleared).
        Returns four complex tensors of shape [n-1].
        """
        a = self.a()
        ac = a.conj()
        aj, ak = a[:-1], a[1:]
        acj, ack = ac[:-1], ac[1:]
        # m_j(t) = (t - conj a_j)/(t - a_j) -> matrix [[1, -conj a_j],[1, -a_j]]
        # composition m_{j+1} o m_j^{-1} (the common det factor cancels in p_j = 0):
        A = -aj + ack
        B = acj - ack
        C = -aj + ak
        D = acj - ak
        return A, B, C, D

    def relations(self, z: torch.Tensor) -> torch.Tensor:
        """
        Variety relations p_j(z) = A_j z_j + B_j - C_j z_j z_{j+1} - D_j z_{j+1}, j=1..n-1,
        as a real feature [Re p, Im p].   z complex [B, n] -> [B, 2(n-1)].
        The z_j z_{j+1} term is the genuine multiplicative coupling (no decoupled analog);
        `coupled=False` drops it (C_j=0) -> the decoupled ablation.
        """
        A, B, C, D = self._moebius_coeffs()
        zj, zk = z[:, :-1], z[:, 1:]
        if not self.coupled:
            C = torch.zeros_like(C)
        p = A.unsqueeze(0) * zj + B.unsqueeze(0) - C.unsqueeze(0) * zj * zk - D.unsqueeze(0) * zk
        return torch.cat([p.real, p.imag], dim=-1)  # [B, 2(n-1)]

    @property
    def out_dim(self) -> int:
        return 2 * (self.n - 1)

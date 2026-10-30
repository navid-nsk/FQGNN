"""
fqmath.grassmann
================

The positive-Grassmannian projection matrix L of the Fourier-quasicrystal construction.

Faithful source (the ONLY math source for this project):
  AKKV = Alon, Kummer, Kurasov, Vinzant, "Higher-dimensional Fourier quasicrystals from
         Lee-Yang varieties", Inventiones mathematicae (2024).
  LT   = Lawton, Tsikh, "Fourier Quasicrystals on R^n".

The construction (AKKV Eq. 1, Thm 2.3) maps x in R^d onto the n-torus by
        Phi_L(x) = exp(2 pi i L x) ,   L in R^{n x d},
and the quasicrystal is the real solution set Lambda(X,L) = { x : Phi_L(x) in X } for a strict
Lee-Yang variety X of dimension c = n - d.  AKKV Thm 2.3 requires L to lie in the POSITIVE
GRASSMANNIAN Gr_+(d, n): every d x d minor (Pluecker coordinate) L_I must be strictly positive.
THREE orthogonal facts (do NOT conflate them): (1) positivity of all minors places L in the open
top positroid cell -- good cell geometry (Postnikov; Lam), and a CLOSED BALL of dim d(n-d) so the
in-family chart has no holes/walls (GKL-ball Thm 1.1); (2) RATIONAL INDEPENDENCE of the rows of L
gives aperiodicity (AKKV Thm 2.5) -- positivity does NOT imply this, it must be preserved separately
(see freeze_incommensurate); (3) the Lee-Yang VARIETY X gives Delone + pure-point diffraction (AKKV
Thm 2.3) -- a property of X, not of L.  The off-torus var(v)>=d argument (AKKV Section 3, Prop 4.4)
is what makes the solution set real; it is clause (3)'s mechanism, not a consequence of positivity.

This module provides a DIFFERENTIABLE, ALWAYS-IN-FAMILY parameterization of Gr_+(d, n):

  * d = 1 : L is a single positive vector ell in R^n_+ (softplus).  This is the Kurasov-Sarnak
            secular case and the PERIODIC-TRAP BOUNDARY: a scalar lift is a 1-D Fourier
            lift, and by LO-Annals Thm 1 a u.d. spectrum + u.d. support forces a periodic comb -- so
            d=1 is the periodic ABLATION, not the genuine FQ.  The genuine FQ needs d>=2 with
            Q-independent rows.  (aperiodicity always needs the entries Q-independent.)

  * d = 2 : the (r, theta) chart.  Row i of L is  r_i * (cos theta_i, sin theta_i)  with
            r_i > 0 and 0 < theta_1 < ... < theta_n < pi/2.  Then the 2x2 minor on rows i<j is
                L_{ij} = r_i r_j (cos theta_i sin theta_j - sin theta_i cos theta_j)
                       = r_i r_j sin(theta_j - theta_i)  > 0
            because all the angle gaps are positive.  This is AKKV Example 1.3's L (minors
            1, sqrt2, sqrt3) generalized, and it is provably in Gr_+(2, n) for ANY parameter
            values, so gradient descent can never leave the cone.

  * d >= 3 : the generalized totally-positive (exponential / generalized-Vandermonde) chart.
             L_{i,:} = ( exp(alpha_1 s_i), ..., exp(alpha_d s_i) ) with s_1 < ... < s_n strictly
             increasing and alpha_1 < ... < alpha_d strictly increasing.  The kernel exp(s a) is
             a strictly-totally-positive (STP) kernel (Karlin), so every d x d minor of L is
             strictly positive (a generalized Vandermonde / Gantmacher-Krein determinant).  We
             work in log space and column-normalize to keep magnitudes O(1); positive column /
             row rescaling preserves the sign of every minor, so the cone membership is kept.

All parameterizations are unconstrained real nn.Parameters mapped through softplus / cumulative
sums, so the totally-positive constraint is STRUCTURAL (never penalized, never projected).

AKKV Thm 2.6(3): the Fourier coefficients of the quasicrystal measure are *linear* in the d x d
minors L_I.  We therefore also expose an exact, differentiable computation of the minors, which
the spectral operators (fqmath.spectrum and the fqops spectral blocks) consume.
"""

from __future__ import annotations

import itertools
import math
from typing import List, Tuple

import torch
import torch.nn as nn


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _softplus(x: torch.Tensor, beta: float = 1.0, floor: float = 1e-4) -> torch.Tensor:
    """Strictly-positive map with a small floor so values never collapse to exactly 0."""
    return torch.nn.functional.softplus(x, beta=beta) + floor


def _strictly_increasing(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """
    Map an unconstrained vector to a strictly increasing sequence inside the open
    interval (lo, hi).  Uses cumulative softplus gaps normalized to (0,1) then affinely
    mapped into (lo, hi) with a small interior margin so endpoints are never touched.
    """
    n = raw.shape[-1]
    gaps = _softplus(raw)                      # all > 0  -> strictly increasing partial sums
    cs = torch.cumsum(gaps, dim=-1)            # 0 < cs_1 < cs_2 < ... < cs_n
    # normalize to (0,1): place points at the centres of the cumulative mass
    cs = cs - gaps * 0.5                       # centre offsets, still strictly increasing
    u = cs / (cs[..., -1:] + gaps[..., -1:] * 0.5 + 1e-12)
    margin = 0.02
    return lo + (hi - lo) * (margin + (1.0 - 2.0 * margin) * u)


# --------------------------------------------------------------------------------------
# the positive-Grassmannian projection
# --------------------------------------------------------------------------------------
class PositiveGrassmann(nn.Module):
    """
    A learnable L in R^{n x d} that is *always* in the positive Grassmannian Gr_+(d, n)
    (all d x d minors strictly positive), with d in {1, 2, >=3} handled by dedicated charts.

    Parameters
    ----------
    n : int
        number of torus coordinates (the ambient dimension of the Lee-Yang variety).
    d : int
        input dimension that L projects (d = 1 secular, d = 2 planar, d >= 3 general).
    init_scale : float
        target RMS magnitude of the rows of L (sets the working frequency band).
    seed : int
        RNG seed for the unconstrained parameters.
    incommensurate_init : bool
        if True, initialize so the generating numbers are sqrt of small primes (Q-independent
        -> genuinely aperiodic spectrum per AKKV Thm 2.5).
    """

    _PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]

    def __init__(
        self,
        n: int,
        d: int,
        init_scale: float = 1.0,
        seed: int = 0,
        incommensurate_init: bool = True,
        freeze_incommensurate: bool = False,
    ):
        super().__init__()
        if n <= d:
            raise ValueError(
                f"need n > d for a genuine aperiodic FQ (AKKV Thm 2.5): got n={n}, d={d}"
            )
        self.n = int(n)
        self.d = int(d)
        self.init_scale = float(init_scale)
        # minor-positivity does NOT protect Q-independence of the rows; freezing the
        # incommensurate (sqrt-prime) chart directions and learning only a global scale (+ the
        # variety U elsewhere) keeps the spectrum genuinely aperiodic (no drift to a periodic comb).
        self.freeze_incommensurate = bool(freeze_incommensurate)
        if self.freeze_incommensurate:
            self.log_scale = nn.Parameter(torch.zeros(()))
        g = torch.Generator().manual_seed(seed)

        def _pb(name: str, tensor: torch.Tensor) -> None:
            """register `tensor` as a learnable Parameter, or a frozen buffer if freezing."""
            if self.freeze_incommensurate:
                self.register_buffer(name, tensor)
            else:
                setattr(self, name, nn.Parameter(tensor))

        if d == 1:
            base = self._incommensurate_vector(n) if incommensurate_init else (
                1.0 + 0.1 * torch.randn(n, generator=g)
            )
            # softplus^{-1}(base) so that _softplus(raw) ~ base initially
            _pb("ell_raw", torch.log(torch.expm1(base.clamp_min(1e-3))))

        elif d == 2:
            # radii r_i and strictly-increasing angles in (0, pi/2)
            r0 = self._incommensurate_vector(n) if incommensurate_init else (
                1.0 + 0.1 * torch.randn(n, generator=g)
            )
            _pb("r_raw", torch.log(torch.expm1(r0.clamp_min(1e-3))))
            # angle gaps initialized roughly uniform
            _pb("theta_raw", 0.05 * torch.randn(n, generator=g))

        else:
            # generalized STP exponential chart: L_{i,k} = exp(alpha_k * s_i)
            _pb("s_raw", 0.05 * torch.randn(n, generator=g))       # -> increasing s_i
            _pb("alpha_raw", 0.05 * torch.randn(d, generator=g))   # -> increasing alpha_k
            # learnable positive per-column scale so we can hit init_scale without breaking TP
            _pb("col_log_scale", torch.zeros(d))

    # ---- the incommensurate seed (sqrt of distinct primes) -----------------------------
    def _incommensurate_vector(self, n: int) -> torch.Tensor:
        vals = torch.tensor(
            [math.sqrt(p) for p in self._PRIMES[:n]], dtype=torch.float32
        )
        return vals / vals.mean() * self.init_scale

    # ---- build L -----------------------------------------------------------------------
    def forward(self) -> torch.Tensor:
        """Return L in R^{n x d}, guaranteed in the positive Grassmannian.  A positive global
        scale (a gauge of the FQ measure) is the only learnable factor when freeze_incommensurate."""
        scale = torch.exp(self.log_scale) if self.freeze_incommensurate else 1.0

        if self.d == 1:
            ell = _softplus(self.ell_raw).unsqueeze(-1)            # [n, 1]
            return ell * self.init_scale * scale

        if self.d == 2:
            r = _softplus(self.r_raw)                              # [n]
            theta = _strictly_increasing(self.theta_raw, 0.0, math.pi / 2.0)  # [n], increasing
            L = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)  # [n, 2]
            # rescale to the target RMS magnitude (positive scalar -> preserves minor signs)
            rms = L.pow(2).sum(-1).mean().sqrt().clamp_min(1e-8)
            return L * (self.init_scale / rms) * scale

        # d >= 3 : strictly-totally-positive exponential chart L_{i,k}=exp(alpha_k s_i) (Karlin).
        # this STP chart is in-family for ANY parameters but is a SUBMANIFOLD of Gr_+(d,n)
        # (not surjective onto the open cell; GKL-reg).  Faithful + trainable; full coverage at
        # d>=3 would need the Postnikov/Lam network chart (face weights y_F>0, prod y_F=1).  d>=3 is
        # not used by the d=2 operators.
        s = _strictly_increasing(self.s_raw, -1.0, 1.0)            # [n], increasing
        alpha = _strictly_increasing(self.alpha_raw, 0.2, 1.5)     # [d], increasing
        log_L = s.unsqueeze(-1) * alpha.unsqueeze(0)               # [n, d] = s_i * alpha_k
        L = torch.exp(log_L)
        L = L * torch.exp(self.col_log_scale).unsqueeze(0)
        L = L / L.norm(dim=0, keepdim=True).clamp_min(1e-8)
        return L * self.init_scale * scale

    # ---- minors (Pluecker coordinates) -------------------------------------------------
    def minor_indices(self) -> List[Tuple[int, ...]]:
        """All row-index d-subsets I in lexicographic order."""
        return list(itertools.combinations(range(self.n), self.d))

    def minors(self, L: torch.Tensor | None = None) -> torch.Tensor:
        """
        Exact, differentiable d x d minors L_I for every row-subset I, in the order of
        `minor_indices()`.  AKKV Thm 2.6(3): the FQ Fourier coefficients are linear in these.
        Returns a real tensor of shape [C(n, d)].
        """
        if L is None:
            L = self.forward()
        idx = self.minor_indices()
        mats = torch.stack([L[list(I), :] for I in idx], dim=0)    # [C, d, d]
        return torch.linalg.det(mats)                              # [C]

    def min_minor(self, L: torch.Tensor | None = None) -> torch.Tensor:
        """The smallest minor — should stay > 0 at all times (cone membership witness).  DIAGNOSTIC
        only; the in-family ENFORCEMENT is `log_minor_barrier` over ALL minors."""
        return self.minors(L).min()

    def log_minor_barrier(self, L: torch.Tensor | None = None, scale: float | None = None,
                          eps: float = 1e-12) -> torch.Tensor:
        """
        The all-maximal-minors genericity FLOOR (a guard against the M=1 collapse)
            B(L) = - mean_I log( Delta_I / scale )      over ALL C(n,d) minors Delta_I,
        added (with a small weight lambda) to the training loss to push EVERY Plücker coordinate
        away from 0 -- a minor -> 0 collapses its amplitude term (the periodic / M=1 collapse seen
        through the Plücker lens).  `min_minor` only reports; this is the enforced barrier.
        `scale` defaults to the median minor so the barrier is scale-invariant.
        """
        m = self.minors(L)
        s = scale if scale is not None else m.detach().abs().median().clamp_min(eps)
        return -(m.clamp_min(eps) / s).log().mean()

    def extra_repr(self) -> str:
        return f"n={self.n}, d={self.d}, init_scale={self.init_scale}"

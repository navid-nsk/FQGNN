"""
fqmath.akkv_example
==================

The genuine 2-D Fourier quasicrystal of AKKV Example 1.3 -- built exactly as in the paper and used
to ANCHOR the project's faithfulness (its density must equal 1 + sqrt2 + sqrt3, its spectrum must
sit on {(k1 - sqrt2 k3, k2 + sqrt3 k3)}).  This is not a test harness; it is the canonical strict
Lee-Yang variety + positive-Grassmannian projection that the FQ operators are faithful to, and the
secular/variety relations here are the d=2 reference for the model's variety coupling.

Construction (AKKV Example 1.3, p. 323-324):
  X subset (P^1)^3 is the curve cut out by
      p1(z) = (1 - 2i) - z1 - z2 + (1 + 2i) z1 z2 = 0
      p2(z) = 1 - (1 + i) z1 - (1 - i) z3 + z1 z3 = 0
  parametrized by Moebius factors of a single t.  The linear map
      x = (x1, x2)  ->  L x = (x1, x2, -sqrt2 x1 + sqrt3 x2)
  is a 3 x 2 matrix whose 2 x 2 minors are (1, sqrt3, sqrt2), all positive (positive Grassmannian).
  The FQ is Lambda = { x in R^2 : exp(2 pi i L x) in X }, the real solution set of the two REAL
  trigonometric equations obtained by clearing a phase from p1, p2 (derived by hand; the prefactor
  constants do not change the zero set):
      g1 = sin(pi t1) sin(pi t2) + sin(pi t1) cos(pi t2) + cos(pi t1) sin(pi t2)            = 0
      g2 = sin(pi t1) cos(pi t3) - 2 sin(pi t1) sin(pi t3) - cos(pi t1) sin(pi t3)          = 0
  with (t1, t2, t3) = (x1, x2, -sqrt2 x1 + sqrt3 x2).  Density = 1 + sqrt2 + sqrt3 ~ 4.146.

This module enumerates the FQ points on a box via a coarse-grid + Newton polish on the 2-D system,
reports the density (faithfulness anchor), and provides the spectrum / structure factor.  Pure
NumPy (a reference object, not on the training path).

Notes: the spectrum {(k1 - sqrt2 k3, k2 + sqrt3 k3)} is a COUNTABLE, topologically DENSE rank-3
Z-module -- only the POINT SET Lambda is uniformly discrete; `spectrum()` enumerates a small
truncation.  `structure_factor_at` is the EMPIRICAL |c_xi|^2 from the points (unit weights); the
variety-FIXED Eq.9 amplitude c_{L,k}=sum_I L_I mhat is the faithful object, computed for the
secular case in fqmath.akkv_amplitudes (the Eq.9 mhat of this Moebius curve is not computed here).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

S2 = math.sqrt(2.0)
S3 = math.sqrt(3.0)
PI = math.pi


def thetas(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t1,t2,t3) = (x1, x2, -sqrt2 x1 + sqrt3 x2)  for X = [P,2]."""
    t1 = X[:, 0]
    t2 = X[:, 1]
    t3 = -S2 * X[:, 0] + S3 * X[:, 1]
    return t1, t2, t3


def G(X: np.ndarray) -> np.ndarray:
    """The two real trig equations of Example 1.3, vectorized over X=[P,2] -> [P,2]."""
    t1, t2, t3 = thetas(X)
    s1, c1 = np.sin(PI * t1), np.cos(PI * t1)
    s2, c2 = np.sin(PI * t2), np.cos(PI * t2)
    s3, c3 = np.sin(PI * t3), np.cos(PI * t3)
    g1 = s1 * s2 + s1 * c2 + c1 * s2
    g2 = s1 * c3 - 2 * s1 * s3 - c1 * s3
    return np.stack([g1, g2], axis=1)


def find_points(box: float = 21.0, h: float = 0.03, newton: int = 8) -> np.ndarray:
    """Find the FQ points in [0,box]^2: coarse |G|^2 local minima -> Newton on (g1,g2)=0."""
    g = np.arange(0.0, box, h)
    GX, GY = np.meshgrid(g, g)
    XY = np.stack([GX.ravel(), GY.ravel()], axis=1)
    A = (G(XY) ** 2).sum(1).reshape(GX.shape)
    c = A[1:-1, 1:-1]
    loc = ((c < A[:-2, 1:-1]) & (c < A[2:, 1:-1]) & (c < A[1:-1, :-2]) & (c < A[1:-1, 2:]) &
           (c < A[:-2, :-2]) & (c < A[2:, 2:]) & (c < A[:-2, 2:]) & (c < A[2:, :-2]))
    ii, jj = np.where(loc)
    P = np.stack([g[jj + 1], g[ii + 1]], axis=1)
    P = P[A[ii + 1, jj + 1] < 0.05]
    eps = 1e-6
    for _ in range(newton):
        F = G(P)
        Jx = (G(P + np.array([eps, 0.0])) - F) / eps
        Jy = (G(P + np.array([0.0, eps])) - F) / eps
        det = Jx[:, 0] * Jy[:, 1] - Jy[:, 0] * Jx[:, 1]
        ok = np.abs(det) > 1e-9
        step = np.zeros_like(P)
        step[ok, 0] = (Jy[ok, 1] * F[ok, 0] - Jy[ok, 0] * F[ok, 1]) / det[ok]
        step[ok, 1] = (-Jx[ok, 1] * F[ok, 0] + Jx[ok, 0] * F[ok, 1]) / det[ok]
        P = P - np.clip(step, -h, h)
    res = np.abs(G(P)).max(1)
    keep = (res < 1e-4) & (P[:, 0] > 0.3) & (P[:, 1] > 0.3) & (P[:, 0] < box - 0.3) & (P[:, 1] < box - 0.3)
    P = P[keep]
    # dedupe near-duplicates
    out = []
    for p in P:
        if not out or np.min(np.linalg.norm(np.array(out) - p, axis=1)) > h:
            out.append(p)
    return np.array(out)


def density(P: np.ndarray, box: float = 21.0) -> float:
    area = (box - 0.6) ** 2
    return len(P) / area


def spectrum(kmax: int = 2) -> np.ndarray:
    """The predicted spectrum Lambda' = {(k1 - sqrt2 k3, k2 + sqrt3 k3)} for small k."""
    out = []
    for k1 in range(-kmax, kmax + 1):
        for k2 in range(-kmax, kmax + 1):
            for k3 in range(-kmax, kmax + 1):
                if (k1, k2, k3) != (0, 0, 0):
                    out.append([k1 - S2 * k3, k2 + S3 * k3])
    return np.array(out)


def structure_factor_at(P: np.ndarray, K: np.ndarray) -> np.ndarray:
    amp = np.exp(-2j * np.pi * (K @ P.T)).sum(1) / np.sqrt(len(P))
    return np.abs(amp) ** 2


def reference_density() -> float:
    """The paper's exact density 1 + sqrt2 + sqrt3."""
    return 1.0 + S2 + S3

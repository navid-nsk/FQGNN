"""
fqmath.variety_solver
====================

General Moebius-curve Fourier-quasicrystal solver -- the genuine d-dimensional construction
(AKKV Sect. 4.2 / Ex. 1.3, generalized).  A strict Lee-Yang rational curve in (P^1)^n is given
coordinate-wise by Blaschke/Moebius factors m_j(t) = (t - conj a_j)/(t - a_j), a_j in the upper
half-plane.  With L in R^{n x d} positive-Grassmannian, the FQ is

        Lambda(a, L) = { x in R^d : exists real t with exp(2 pi i (L x)_j) = m_j(t)  for all j }.

For n = d + 1 the solution set is discrete (d real equations from matching coordinates 1..n-1 to a
common t taken from coordinate 0).  We solve it by a coarse grid over the box + Newton on the d
phase-matching residuals, deduplicate, and report the density (which, by AKKV Thm 2.6(1) with unit
multidegree, equals the sum of the d x d minors of L) as a faithfulness check.

This is the reference construction the FQ operators' variety relations are faithful to; it lets us
verify a learned (a, L) is a genuine FQ by checking its density against sum-of-minors.
"""

from __future__ import annotations

import itertools
import math
from typing import Tuple

import numpy as np


def vandermonde_L(n: int, d: int, s: np.ndarray | None = None) -> np.ndarray:
    """Totally-positive n x d matrix (all d x d minors > 0): rows (1, s_i, s_i^2, ..., s_i^{d-1}).
    REFERENCE TP generator only (a Vandermonde is STP for increasing s_i).  NOT the learnable
    d>=3 chart -- that is the strict Diaz-Mainar exponential / Postnikov network chart (the
    Vandermonde submanifold is measure-zero in Gr_+(d,n))."""
    if s is None:
        s = 0.6 + 0.5 * np.arange(n)
    return np.stack([s ** k for k in range(d)], axis=1).astype(np.float64)


def minors_sum(L: np.ndarray) -> float:
    n, d = L.shape
    return float(sum(abs(np.linalg.det(L[list(I)])) for I in itertools.combinations(range(n), d)))


def _phase_residuals(X: np.ndarray, L: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    Residuals e_j(x) = wrap( 2 pi (L x)_j - arg m_j(t0(x)) ), j=1..n-1, where t0 is recovered from
    coordinate 0.  X = [P, d] -> [P, n-1].
    """
    n = len(a)
    p = X @ L.T                                   # [P, n]
    z = np.exp(2j * np.pi * p)                    # [P, n]
    # invert coordinate 0: t0 = (a0 z0 - conj a0)/(z0 - 1)
    t0 = ((a[0] * z[:, 0] - np.conj(a[0])) / (z[:, 0] - 1.0 + 1e-12)).real
    pred = (t0[:, None] - np.conj(a)[None]) / (t0[:, None] - a[None])   # m_j(t0) [P, n]
    e = np.angle(z[:, 1:] * np.conj(pred[:, 1:]))                       # [P, n-1]
    bad = np.abs(z[:, 0] - 1.0) < 1e-3            # t0 -> infinity region
    e[bad] = 10.0
    return e


def solve_fq(a: np.ndarray, L: np.ndarray, box: float = 16.0, h: float = 0.04,
             newton: int = 8) -> np.ndarray:
    """Find the FQ points in [0,box]^d for d = n-1 (discrete).  Returns [M, d]."""
    n, d = L.shape
    assert n == d + 1, "discrete FQ needs n = d + 1"
    axes = [np.arange(0.0, box, h)] * d
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, d)
    e = _phase_residuals(grid, L, a)
    F2 = (e ** 2).sum(1)
    shp = tuple(len(ax) for ax in axes)
    A = F2.reshape(shp)
    keep = np.ones_like(A, dtype=bool)
    for ax in range(d):
        keep &= (A <= np.roll(A, 1, ax)) & (A <= np.roll(A, -1, ax))
    seeds = grid[(keep.ravel()) & (F2 < 0.05)]
    P = seeds.copy(); eps = 1e-6
    for _ in range(newton):
        F = _phase_residuals(P, L, a)
        J = np.zeros((len(P), d, d))
        for k in range(d):
            dP = P.copy(); dP[:, k] += eps
            J[:, :, k] = (_phase_residuals(dP, L, a) - F) / eps
        step = np.zeros_like(P)
        for i in range(len(P)):
            try:
                step[i] = np.linalg.solve(J[i], F[i])
            except np.linalg.LinAlgError:
                pass
        P = P - np.clip(step, -h, h)
    res = np.abs(_phase_residuals(P, L, a)).max(1)
    lo, hi = 0.4, box - 0.4
    inb = np.all((P > lo) & (P < hi), 1)
    P = P[(res < 1e-4) & inb]
    if len(P) == 0:
        return P
    keep_pts = [P[0]]
    for q in P[1:]:
        if np.min(np.linalg.norm(np.array(keep_pts) - q, axis=1)) > h:
            keep_pts.append(q)
    return np.array(keep_pts)


def density_check(a: np.ndarray, L: np.ndarray, box: float = 16.0) -> Tuple[float, float]:
    """Return (measured density, sum-of-minors) -- they should match for a genuine FQ."""
    P = solve_fq(a, L, box=box)
    vol = (box - 0.8) ** (L.shape[1])
    return (len(P) / vol if vol > 0 else float("nan")), minors_sum(L)


def upper_half_plane_a(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, n) + 1j * (0.5 + rng.uniform(0, 1, n))

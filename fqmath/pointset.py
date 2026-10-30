"""
fqmath.pointset
==============

The genuine Fourier-quasicrystal POINT SETS (Lambda) -- the Delone, aperiodic, pure-point sets
the construction produces.  Faithful to AKKV / Kurasov-Sarnak.  Used to anchor the variety
parameters and to expose the FQ point set for interpretability (which aperiodic spatial scales the
learned variety realizes).  Pure NumPy reference (not on the hot training path).

1-D Kurasov-Sarnak FQ (the d=1 SPECIAL CASE; not the universal d>=2 generator): for a
unitary U and an incommensurate positive vector ell, the secular function
f(x) = p_U(e^{2 pi i ell x}) = det( I - diag(e^{2 pi i ell x}) U ) (p_U normalized so that p_U(0)=1) is real-rooted (M(x) is unitary
=> eigenvalues on the unit circle => f(x)=0 exactly when an eigenphase hits 0), and its real zeros
form a 1-D FQ Lambda.  We find the zeros over [0, x_max] as the isolated local minima
of |f|^2 that touch ~0, then parabolically refine.

The richness of the gap spectrum (many distinct gap lengths, vs a cut-and-project model set's two)
is the signature of the genuine Lee-Yang construction.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def haar_unitary(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Z = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / math.sqrt(2)
    Q, R = np.linalg.qr(Z)
    return Q @ np.diag(np.exp(1j * np.angle(np.diag(R))))


def secular(x: np.ndarray, ell: np.ndarray, U: np.ndarray) -> np.ndarray:
    """f(x) = p_U(z) = det(I - diag(z) U) at z = e^{2 pi i ell x}, vectorized over x (p_U(0)=1 at z=0).
    Returns complex [P]. (The zero set is the same as for the det(diag(z)U - I) sign convention.)"""
    x = np.atleast_1d(x).astype(float)
    n = len(ell)
    I = np.eye(n)
    phases = np.exp(2j * np.pi * np.outer(x, ell))     # [P, n]
    out = np.empty(len(x), dtype=complex)
    for i in range(len(x)):
        out[i] = np.linalg.det(I - (phases[i][:, None] * U))
    return out


def lee_yang_1d(ell: np.ndarray, U: np.ndarray, x_max: float = 120.0, dx: float = 0.0015) -> np.ndarray:
    """Real zeros of the secular function over [0, x_max] -- the 1-D FQ point set."""
    xg = np.arange(0.0, x_max, dx)
    f = secular(xg, ell, U)
    a = np.abs(f) ** 2
    loc = (a[1:-1] < a[:-2]) & (a[1:-1] < a[2:])
    cand = np.where(loc)[0] + 1
    thr = 1e-3 * np.median(a)
    cand = cand[a[cand] < max(thr, 1e-8)]
    roots = []
    for c in cand:
        lo, hi = xg[c - 1], xg[c + 1]
        xs = np.linspace(lo, hi, 21)
        fs = np.abs(secular(xs, ell, U)) ** 2
        roots.append(xs[np.argmin(fs)])
    return np.array(sorted(roots))


def gap_spectrum(Lam: np.ndarray) -> Tuple[np.ndarray, int]:
    """Gap lengths and number of distinct gaps (richness = genuine Lee-Yang signature)."""
    g = np.diff(np.sort(Lam))
    g = g[g > 1e-6]
    distinct = len(np.unique(np.round(g, 3)))
    return g, distinct


def peak_to_mean(P: np.ndarray, ng: int = 4000, kmax: float = 6.0, excl: float = 0.3) -> float:
    """Diffraction peak-to-mean (pure-point witness): >> than for a Poisson set."""
    xi = np.linspace(0, kmax, ng)
    amp = np.exp(-2j * np.pi * np.outer(xi, P)).sum(1) / np.sqrt(len(P))
    S = np.abs(amp) ** 2
    m = xi > excl
    return float(S[m].max() / S[m].mean())


def incommensurate_ell(n: int) -> np.ndarray:
    """ell = sqrt of small primes (rationally independent => genuinely aperiodic FQ)."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    v = np.array([math.sqrt(p) for p in primes[:n]])
    return v / v.mean()

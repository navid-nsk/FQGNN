"""
fqmath.diagnostics
=================

Theorem-backed FQ-vs-RFF diagnostics for Task Alpha, computed on DATA:

  * NUMBER RIGIDITY (Bjorklund-Hartnick): interior counts are a.s. determined by the exterior --
    the precise reason an FQ basis should beat RFF on TWO-SIDED imputation.  `number_rigidity` returns
    the conditional variance of an interior count given the exterior count (LOW => rigid => FQ-like).
  * HYPERUNIFORMITY exponent (AKKV Thm 2.9): number variance grows like R^{d-1} (hyperuniform) vs R^d
    (Poisson).  `number_variance` + `hyperuniformity_exponent` fit the growth.
  * GAP distribution (Alon-Vinzant-gaps: Poisson <-> CUE): the richness of distinct nearest-neighbour
    gaps is the genuine-Lee-Yang signature (a model set has ~2 gaps; an FQ has many).

Pure NumPy diagnostics (off the training path).
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def gap_distribution(points_1d: np.ndarray, round_dp: int = 3) -> Dict[str, float]:
    """Nearest-neighbour gap statistics of a 1-D point set: number of DISTINCT gaps (richness =
    genuine Lee-Yang signature), plus mean/std/cv."""
    g = np.diff(np.sort(np.ravel(points_1d)))
    g = g[g > 1e-9]
    if len(g) == 0:
        return {"distinct_gaps": 0.0, "mean_gap": float("nan"), "cv": float("nan")}
    distinct = len(np.unique(np.round(g, round_dp)))
    return {"distinct_gaps": float(distinct), "mean_gap": float(g.mean()),
            "cv": float(g.std() / max(g.mean(), 1e-12))}


def number_variance(points: np.ndarray, radii: np.ndarray, n_windows: int = 400,
                    seed: int = 0) -> np.ndarray:
    """Var of the point count in random balls of each radius in `radii` (the hyperuniformity probe).
    points [N, d] -> variance [len(radii)]."""
    points = np.atleast_2d(points)
    d = points.shape[1]
    lo, hi = points.min(0), points.max(0)
    rng = np.random.default_rng(seed)
    out = []
    for R in radii:
        centers = rng.uniform(lo + R, np.maximum(hi - R, lo + R), size=(n_windows, d))
        # count points within R of each center
        cnt = np.array([(np.linalg.norm(points - c, axis=1) <= R).sum() for c in centers])
        out.append(cnt.var())
    return np.array(out, dtype=float)


def hyperuniformity_exponent(points: np.ndarray, radii: np.ndarray | None = None,
                             **kw) -> float:
    """Fit the number-variance growth Var(R) ~ R^alpha.  alpha ~ d-1 => hyperuniform; ~ d => Poisson."""
    points = np.atleast_2d(points)
    if radii is None:
        span = float(np.linalg.norm(points.max(0) - points.min(0)))
        radii = np.linspace(0.05 * span, 0.25 * span, 8)
    var = number_variance(points, radii, **kw)
    ok = var > 0
    if ok.sum() < 2:
        return float("nan")
    a = np.polyfit(np.log(radii[ok]), np.log(var[ok]), 1)[0]
    return float(a)


def number_rigidity(points: np.ndarray, inner: float, outer: float, n_windows: int = 400,
                    seed: int = 0) -> float:
    """
    Number-rigidity proxy: the variance of the interior count (within `inner`) CONDITIONED on the
    exterior count (in the annulus inner..outer) being held ~fixed.  Estimated as the residual
    variance of interior-count after linearly regressing out the exterior count.  LOW => rigid (FQ);
    ~ unconditional variance => not rigid (Poisson/RFF).  Returns ratio residual_var / total_var.
    """
    points = np.atleast_2d(points)
    d = points.shape[1]
    lo, hi = points.min(0), points.max(0)
    rng = np.random.default_rng(seed)
    centers = rng.uniform(lo + outer, np.maximum(hi - outer, lo + outer), size=(n_windows, d))
    nin, nout = [], []
    for c in centers:
        dist = np.linalg.norm(points - c, axis=1)
        nin.append((dist <= inner).sum())
        nout.append(((dist > inner) & (dist <= outer)).sum())
    nin = np.array(nin, float); nout = np.array(nout, float)
    tot = nin.var()
    if tot < 1e-9 or nout.var() < 1e-9:
        return float("nan")
    beta = np.cov(nin, nout)[0, 1] / nout.var()
    resid = nin - beta * (nout - nout.mean())
    return float(resid.var() / tot)

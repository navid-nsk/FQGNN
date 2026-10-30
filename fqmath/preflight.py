"""
fqmath.preflight
================

The DATA-side S(k) pre-flight diagnostic (Task Alpha): an "is the target pure-point?" check, run
BEFORE relying on the FQ prior.

The FQ prior fits PURE-POINT / hyperuniform targets.  A single broadband trajectory has a CONTINUOUS
spectrum, so the right object to test is the empirical structure factor S(k) of the AIS INCREMENTS /
residuals: if it is continuous (low peak-to-mean), the FQ prior is mis-specified and one should not
expect the FQ to beat RFF; if it has sharp Bragg peaks (high peak-to-mean), the pure-point FQ prior
is appropriate.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def structure_factor(points: np.ndarray, kvecs: np.ndarray) -> np.ndarray:
    """S(k) = |sum_j e^{-2 pi i k . x_j}|^2 / N  for a point/sample set.
    points [N, d], kvecs [K, d] -> S [K] (>= 0)."""
    points = np.atleast_2d(points)
    N = len(points)
    amp = np.exp(-2j * np.pi * (kvecs @ points.T)).sum(axis=1) / np.sqrt(max(N, 1))
    return np.abs(amp) ** 2


def peak_to_mean(points: np.ndarray, kmax: float = 6.0, ngrid: int = 4000,
                 excl: float = 0.3, d: int | None = None, seed: int = 0) -> float:
    """Diffraction peak-to-mean of S(k) away from the origin -- a pure-point witness (>> for a
    quasicrystal / lattice, ~O(1) for a Poisson/continuous spectrum)."""
    points = np.atleast_2d(points)
    d = points.shape[1] if d is None else d
    rng = np.random.default_rng(seed)
    if d == 1:
        kvecs = np.linspace(0, kmax, ngrid)[:, None]
    else:
        kvecs = rng.uniform(-kmax, kmax, size=(ngrid, d))
    S = structure_factor(points, kvecs)
    r = np.linalg.norm(kvecs, axis=1)
    m = r > excl
    if m.sum() == 0:
        return float("nan")
    return float(S[m].max() / max(S[m].mean(), 1e-12))


def preflight_report(increments: np.ndarray, kmax: float = 6.0, ngrid: int = 4000,
                     pure_point_threshold: float = 20.0) -> Dict[str, float]:
    """
    Run the S(k) pre-flight on AIS increments/residuals.  Returns peak_to_mean and a verdict flag
    `pure_point` (1.0 if peak-to-mean exceeds the threshold => FQ prior appropriate; 0.0 => continuous
    spectrum => FQ prior mis-specified, do not expect FQ > RFF).
    """
    ptm = peak_to_mean(increments, kmax=kmax, ngrid=ngrid)
    return {"peak_to_mean": ptm,
            "pure_point": float(np.isfinite(ptm) and ptm > pure_point_threshold),
            "threshold": float(pure_point_threshold)}

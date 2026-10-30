"""Chord-mean Gaussian-process regression: a deterministic, classical, VANILLA two-sided baseline.

Per gap, an independent textbook GP regression per ENU coordinate on time -> position:
  * mean function  = the CHORD (linear trend between the two bracketing anchors aL/aR -- the same
                     two anchors GCD's linear interp uses), so the GP posterior reverts to the
                     GCD-style straight line far from data / when residuals vanish;
  * kernel         = Matern-5/2 (the standard "twice mean-square-differentiable" default);
  * training data  = ALL available anchor POSITIONS of the gap, both sides (the same leak-safe
                     anchors every other baseline sees; held positions are never read);
  * hyperparameters PER GAP by maximising the exact log marginal likelihood over a small fixed grid:
      - lengthscale: 8 geometric values from 0.1x to 10x the gap span (tR - tL),
      - signal variance: set from the data (variance of the chord residuals, per coordinate),
      - noise: fixed small 1e-6 relative to the signal variance (pure jitter; anchors are treated
        as near-noiseless observations, matching the other classical baselines).

Pure numpy/scipy, closed-form posterior mean -- no iterative optimisation, no learned parameters,
no seed dependence (fully deterministic given the anchors).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

N_LENGTHSCALES = 8
ELL_LO_FACTOR = 0.1    # lengthscale grid = geometric, 0.1x .. 10x the gap span
ELL_HI_FACTOR = 10.0
REL_NOISE = 1e-6       # noise variance relative to signal variance
ABS_JITTER = 1e-10     # absolute numerical floor on the diagonal (metres^2)
SQRT5 = np.sqrt(5.0)


def _matern52(ta: np.ndarray, tb: np.ndarray, ell: float) -> np.ndarray:
    """Matern-5/2 correlation matrix between time vectors ta [n] and tb [m] (unit signal variance)."""
    d = np.abs(ta[:, None] - tb[None, :]) / ell
    return (1.0 + SQRT5 * d + (5.0 / 3.0) * d * d) * np.exp(-SQRT5 * d)


def _fit_predict_1d(t_tr: np.ndarray, resid: np.ndarray, t_te: np.ndarray, span: float) -> np.ndarray:
    """GP posterior mean of the chord RESIDUAL at t_te, with per-gap ML-grid lengthscale selection."""
    n = len(t_tr)
    sig2 = float(np.var(resid))
    if sig2 <= ABS_JITTER:                       # residuals are (numerically) zero -> chord is exact
        return np.zeros(len(t_te))
    noise = REL_NOISE * sig2 + ABS_JITTER

    best = None                                   # (ll, ell, cho, alpha)
    for ell in np.geomspace(ELL_LO_FACTOR * span, ELL_HI_FACTOR * span, N_LENGTHSCALES):
        K = sig2 * _matern52(t_tr, t_tr, ell)
        K[np.diag_indices(n)] += noise
        jit = noise
        while True:                               # numerics-only escalation (not a hyperparameter)
            try:
                cho = cho_factor(K, lower=True)
                break
            except np.linalg.LinAlgError:
                jit *= 10.0
                if jit > 1e-2 * sig2:
                    cho = None
                    break
                K[np.diag_indices(n)] += jit
        if cho is None:
            continue
        alpha = cho_solve(cho, resid)
        ll = (-0.5 * float(resid @ alpha) - float(np.sum(np.log(np.diag(cho[0]))))
              - 0.5 * n * np.log(2.0 * np.pi))
        if best is None or ll > best[0]:
            best = (ll, ell, cho, alpha)
    if best is None:                              # all factorizations failed -> fall back to the chord
        return np.zeros(len(t_te))
    _, ell, _, alpha = best
    k_star = sig2 * _matern52(t_te, t_tr, ell)
    return k_star @ alpha


def gp_predict(before: dict, after: dict, held_t: np.ndarray):
    """Chord-mean Matern-5/2 GP posterior mean at the held times, each ENU coordinate independent.

    before, after: dicts from common_gaps.anchor_states_enu (aligned t, x, y arrays in chronological
                   order). ALL anchors of both sides are training points; the chord mean uses the
                   LAST element of `before` and the FIRST element of `after` (identical anchor choice
                   to the GCD/Hermite baselines). Instrument velocities are not needed.
    held_t:        [n_held] absolute-scale seconds of the query (held) times.

    Returns (pred_x_m, pred_y_m): ENU-metre predictions aligned with held_t.
    """
    tL = float(before["t"][-1]); tR = float(after["t"][0])
    span = max(tR - tL, 1e-6)
    t_tr = np.concatenate([before["t"], after["t"]]).astype(np.float64)
    held_t = np.asarray(held_t, np.float64)

    preds = []
    for c in ("x", "y"):
        y_tr = np.concatenate([before[c], after[c]]).astype(np.float64)
        pL = float(before[c][-1]); pR = float(after[c][0])
        slope = (pR - pL) / span
        m_tr = pL + (t_tr - tL) * slope           # chord mean at the training anchors
        m_te = pL + (held_t - tL) * slope         # chord mean at the query times
        preds.append(m_te + _fit_predict_1d(t_tr, y_tr - m_tr, held_t, span))
    return preds[0], preds[1]

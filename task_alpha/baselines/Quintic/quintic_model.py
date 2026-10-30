"""Quintic (minimum-jerk) Hermite: a deterministic, classical, VANILLA two-sided baseline.

Uses the two bracketing anchors (aL = last anchor before the gap, aR = first anchor after it),
blended with the textbook two-point QUINTIC Hermite interpolant that matches position, velocity AND
acceleration at both boundaries. This is exactly the classical "minimum-jerk" trajectory (the unique
minimiser of the integrated squared jerk subject to those six boundary conditions is the quintic).

Boundary velocity = the INSTRUMENT velocity (from sog/cog) at the bracketing anchor. Boundary
acceleration = same-side finite difference of the instrument velocities of the two nearest same-side
anchors ((v_last - v_prev) / (t_last - t_prev) on the left, mirrored on the right); zero acceleration
when a side has fewer than 2 usable anchors.

On a straight, constant-velocity gap (vL == vR == the secant slope, aL == aR == 0) the quintic
reduces EXACTLY to the linear/affine interpolant, so this baseline ties GCD exactly there, by
construction.

No hyperparameters, no training, no seed dependence (fully deterministic given the anchors).
"""
from __future__ import annotations

import numpy as np


def _boundary_accel(side: dict, boundary: str) -> np.ndarray:
    """Same-side finite-difference acceleration [ax, ay] (m/s^2) at the gap boundary.

    side:     dict from common_gaps.anchor_states_enu (chronological t, vx, vy arrays).
    boundary: "left"  -> use the LAST two anchors of the before-side  (v[-1]-v[-2])/(t[-1]-t[-2])
              "right" -> use the FIRST two anchors of the after-side  (v[1]-v[0])/(t[1]-t[0])
    Falls back to zero acceleration when the side has fewer than 2 usable anchors (or a degenerate
    time difference).
    """
    t = side["t"]
    if len(t) < 2:
        return np.zeros(2)
    if boundary == "left":
        i0, i1 = -2, -1
    else:
        i0, i1 = 0, 1
    dt = float(t[i1] - t[i0])
    if dt <= 1e-6:
        return np.zeros(2)
    return np.array([(side["vx"][i1] - side["vx"][i0]) / dt,
                     (side["vy"][i1] - side["vy"][i0]) / dt])


def quintic_predict(before: dict, after: dict, held_t: np.ndarray):
    """Two-point quintic Hermite interpolant matching position + velocity + acceleration at the two
    bracketing anchors.

    before, after: dicts from common_gaps.anchor_states_enu (aligned t, x, y, vx, vy arrays in
                   chronological order). Position/velocity anchors are the LAST element of `before`
                   and the FIRST element of `after`; acceleration additionally reads the
                   second-nearest same-side anchor's velocity.
    held_t:        [n_held] absolute-scale seconds of the query (held) times.

    Returns (pred_x_m, pred_y_m): ENU-metre predictions aligned with held_t.
    """
    tL = before["t"][-1]; pL = np.array([before["x"][-1], before["y"][-1]])
    vL = np.array([before["vx"][-1], before["vy"][-1]])
    tR = after["t"][0]; pR = np.array([after["x"][0], after["y"][0]])
    vR = np.array([after["vx"][0], after["vy"][0]])
    aL = _boundary_accel(before, "left")
    aR = _boundary_accel(after, "right")

    dt = max(tR - tL, 1e-6)
    s = np.clip((held_t - tL) / dt, 0.0, 1.0)
    s2 = s * s; s3 = s2 * s; s4 = s3 * s; s5 = s4 * s
    # Textbook quintic Hermite basis (C^2 two-point form / min-jerk polynomial):
    h0 = 1 - 10 * s3 + 15 * s4 - 6 * s5           # p(0)
    h1 = s - 6 * s3 + 8 * s4 - 3 * s5             # p'(0)
    h2 = 0.5 * s2 - 1.5 * s3 + 1.5 * s4 - 0.5 * s5  # p''(0)
    h3 = 10 * s3 - 15 * s4 + 6 * s5               # p(1)
    h4 = -4 * s3 + 7 * s4 - 3 * s5                # p'(1)
    h5 = 0.5 * s3 - s4 + 0.5 * s5                 # p''(1)

    pred = (h0[:, None] * pL + h1[:, None] * dt * vL + h2[:, None] * dt * dt * aL
            + h3[:, None] * pR + h4[:, None] * dt * vR + h5[:, None] * dt * dt * aR)
    return pred[:, 0], pred[:, 1]

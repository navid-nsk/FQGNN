"""Coordinated-Turn Extended Kalman Filter (Cartesian velocity), mixed via a 2-mode IMM, two-sided.

A classical, VANILLA tracking baseline (Bar-Shalom-style Interacting Multiple Model filter over the
standard "nearly-coordinated-turn" motion model -- see e.g. Blackman & Popoli, "Design and Analysis
of Modern Tracking Systems", or Li & Jilkov, "Survey of Maneuvering Target Tracking"). No learned
parameters, no gradient descent.

State x = [x, y, vx, vy, omega] (ENU position, Cartesian velocity, turn rate). Under a constant turn
rate the velocity vector rotates at rate omega while its magnitude (speed) is conserved -- this is
the STANDARD choice over the textbook (v, theta, omega) polar parametrization precisely because the
polar form is only weakly identifiable from position-only measurements under sparse, irregular
sampling: with (v, theta) the EKF can make wild sign-flip jumps in v between consecutive real AIS
anchors (v oscillating between large negative and positive values step to step) because
(v, theta) and (-v, theta + pi) describe the SAME velocity and a linearized
update cannot always tell them apart. The Cartesian form has no such ambiguity, and because speed is
conserved under a pure rotation it structurally cannot diverge the way the polar form's v/omega
singularity could -- a spurious omega during a long coast produces at worst a bounded circular
excursion, not an exploding trajectory.

Two modes share the SAME 5-dim state (so IMM mixing needs no dimension augmentation):
  mode 0 "CV-like"  -- small turn-rate process noise (velocity direction barely evolves)
  mode 1 "CT"        -- larger turn-rate process noise (adapts to real manoeuvres)
Only sigma_a (accel std, isotropic discretized-white-noise-acceleration model) and sigma_omega_dot
(turn-rate std) are free constants, chosen ONCE via a light validation-set grid search
(run_ctrvkalman_gaps.py --tune) and frozen thereafter -- exactly how any deployed Kalman filter's
process noise is set; never re-tuned per test condition.

TWO-SIDED reconstruction: a forward IMM filters the anchors BEFORE the gap (chronological), a second
IMM filters the anchors AFTER the gap using a REVERSED, NEGATED time axis (so the same forward-
filtering code runs unchanged and stays self-consistent -- no manual velocity-sign flip needed). Each
side is then coasted (predict-only, no more measurements) to every held query time, and the two
Gaussian position estimates are fused by inverse-covariance weighting -- the "two-sided" analogue of
a fixed-interval smoother, giving the classical filter the same two-sided information the GCD
baseline and FQ-GNN both see.
"""
from __future__ import annotations

import numpy as np

PI_MARKOV = np.array([[0.95, 0.05], [0.05, 0.95]])   # mode-transition matrix (both models -> "sticky")
V_MAX = 30.0       # m/s (~58 knots) -- generous physical speed bound, standard safeguard in deployed trackers
Q_DT_CAP = 300.0   # s -- saturation cap on the dt used for DWNA position-noise growth (see ctrv_predict)

# Real AIS anchors carry REPORTED velocity (speed/course over ground), not just position -- every
# real AIS-based tracker uses it directly. Observing [x, y, vx, vy] (not position alone) anchors the
# velocity estimate to the reported value at each anchor instead of re-deriving it purely from noisy
# position finite-differences under irregular, sometimes ~1-h anchor spacing -- a position-only
# design lets a single noisy or kinematically-real large step swing the fitted velocity
# wildly (occasionally hitting V_MAX) between updates.
H_FULL = np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0]])
SOG_COG_VAR = 1.0 ** 2   # m/s^2 -- reported-SOG/COG measurement-noise variance (textbook default, not tuned)


# ------------------------------------------------------------------------------------------------ #
#  Coordinated-turn process model (Cartesian velocity): closed-form mean + numerical Jacobian          #
# ------------------------------------------------------------------------------------------------ #
def _sinc(u):
    """sin(u)/u, well-defined (=1) at u=0 -- numpy's normalized sinc(u/pi) is a robust, tested
    implementation of exactly this removable singularity."""
    return np.sinc(u / np.pi)


def _vers(u):
    """(1 - cos(u))/u, well-defined (=0) at u=0. Computed via 1-cos(u) = 2 sin(u/2)^2 (NOT the naive
    1-cos(u) form) -- the naive form suffers catastrophic cancellation for small u; this identity
    keeps the numerator itself O(u^2) computed from an accurately-evaluated small sine, so the ratio
    is smooth and accurate uniformly down to u=0 with no branch or threshold needed."""
    return np.where(np.abs(u) < 1e-12, 0.0, 2.0 * np.sin(u / 2.0) ** 2 / np.where(u == 0, 1.0, u))


def _ctrv_mean(state, dt):
    """Coordinated-turn mean transition, branch-free. A two-branch form (exact formula
    for |omega|>1e-4, a low-order Taylor expansion below it) does NOT agree at the threshold
    for large dt (the Taylor truncation drops an O(omega^2 dt) term that matters once dt reaches
    thousands of seconds), creating a genuine discontinuity in the mean transition that the
    finite-difference Jacobian would straddle and amplify into wild velocity oscillations on
    real data. The sinc/vers form below is accurate and smooth for every omega,
    dt, with no threshold to get wrong."""
    x, y, vx, vy, om = state
    u = om * dt
    s, c1 = _sinc(u), _vers(u)
    x_n = x + vx * dt * s - vy * dt * c1
    y_n = y + vx * dt * c1 + vy * dt * s
    vx_n = vx * np.cos(u) - vy * np.sin(u)
    vy_n = vx * np.sin(u) + vy * np.cos(u)
    return np.array([x_n, y_n, vx_n, vy_n, om])


def ctrv_predict(state, P, dt, sigma_a, sigma_omega_dot):
    state_n = _ctrv_mean(state, dt)

    # Numerical (central-difference) Jacobian, computed directly from the same _ctrv_mean used for
    # the point prediction (guaranteed consistent with it by construction, at the cost of 10 extra
    # function evaluations) instead of a hand-derived closed form of the sinc/vers transition.
    F = np.zeros((5, 5))
    for i in range(5):
        h = 1e-6 * max(abs(state[i]), 1.0)
        d = np.zeros(5); d[i] = h
        F[:, i] = (_ctrv_mean(state + d, dt) - _ctrv_mean(state - d, dt)) / (2 * h)

    # Process noise: isotropic discretized-white-noise-acceleration (DWNA) on (x,y,vx,vy) -- the
    # standard, well-conditioned CV/CT process-noise model (Bar-Shalom) -- plus an independent
    # random-walk variance on omega.
    #
    # DWNA's dt^3/3 position-noise term is standard for the sub-second-to-few-second sample
    # intervals it was designed for (radar tracking); real AIS anchor spacing is irregular and
    # ordinary "kept" anchors can be up to ~45 min apart even outside the held gap. Applied
    # literally at dt ~ 2500 s, dt^3/3 alone implies a position-uncertainty std of several
    # kilometres from "unmodeled acceleration" over well under an hour -- unphysical for a vessel
    # that in fact holds a roughly steady speed for hours, and it makes the Kalman gain treat any
    # innovation (even one driven by a real, modest heading change) as fully trustworthy, swinging
    # the fitted velocity wildly between updates. The remedy used here is the standard practical
    # saturation approximation for irregular/large sampling intervals: cap the dt used for the
    # qa-driven (position-growing) noise terms at Q_DT_CAP, so process-noise growth flattens out
    # instead of compounding without bound. The mean transition and the omega random walk still use
    # the TRUE dt (only the qa noise-growth is capped).
    dt_q = min(dt, Q_DT_CAP)
    Q = np.zeros((5, 5))
    qa = sigma_a ** 2
    blk = qa * np.array([[dt_q ** 3 / 3, dt_q ** 2 / 2], [dt_q ** 2 / 2, dt_q]])
    Q[np.ix_([0, 2], [0, 2])] = blk
    Q[np.ix_([1, 3], [1, 3])] = blk
    Q[4, 4] = sigma_omega_dot ** 2 * dt

    P_n = F @ P @ F.T + Q
    return state_n, P_n


def kf_update(state, P, z, R):
    """z = [x, y, vx, vy] (position AND reported velocity -- see H_FULL docstring above)."""
    y_innov = z - H_FULL @ state
    S = H_FULL @ P @ H_FULL.T + R
    K = P @ H_FULL.T @ np.linalg.inv(S)
    state_n = state + K @ y_innov
    P_n = (np.eye(5) - K @ H_FULL) @ P
    speed = np.hypot(state_n[2], state_n[3])         # physical speed clamp (standard tracker safeguard)
    if speed > V_MAX:
        state_n[2] *= V_MAX / speed; state_n[3] *= V_MAX / speed
    sign, logdet = np.linalg.slogdet(S)
    loglik = -0.5 * (y_innov @ np.linalg.solve(S, y_innov) + logdet + 2 * np.log(2 * np.pi))
    return state_n, P_n, loglik


# ------------------------------------------------------------------------------------------------ #
#  2-mode IMM (Bar-Shalom mixing / mode-probability update / combination)                            #
# ------------------------------------------------------------------------------------------------ #
class IMMState:
    __slots__ = ("states", "covs", "mu")

    def __init__(self, states, covs, mu):
        self.states, self.covs, self.mu = states, covs, mu

    def combine(self):
        cs = sum(self.mu[j] * self.states[j] for j in range(2))
        cP = sum(self.mu[j] * (self.covs[j] + np.outer(self.states[j] - cs, self.states[j] - cs))
                 for j in range(2))
        return cs, cP


def _imm_mix(imm: IMMState):
    mu = imm.mu
    cbar = PI_MARKOV.T @ mu                                     # predicted mode prob, shape (2,)
    mix_w = (PI_MARKOV * mu[:, None]) / cbar[None, :]           # mix_w[i,j] = P(mode i(k-1) | mode j(k))
    mixed_states, mixed_covs = [], []
    for j in range(2):
        xs = sum(mix_w[i, j] * imm.states[i] for i in range(2))
        Ps = sum(mix_w[i, j] * (imm.covs[i] + np.outer(imm.states[i] - xs, imm.states[i] - xs))
                 for i in range(2))
        mixed_states.append(xs); mixed_covs.append(Ps)
    return mixed_states, mixed_covs, cbar


OMEGA_DECAY_TAU = 900.0     # s -- manoeuvre-persistence relaxation time for coasting (see _coast)
COAST_SUBSTEP = 600.0       # s -- coast step size


def _coast(state, P, dt, sigma_a, sigma_om, tau=OMEGA_DECAY_TAU, substep=COAST_SUBSTEP):
    """Single-mode predict over `dt` in small substeps, exponentially decaying omega -- AND ITS
    UNCERTAINTY -- toward zero each substep.

    A real vessel does not sustain one turn rate for hours, and real AIS anchor spacing is irregular
    (consecutive KEPT anchors can be up to ~1 h apart even outside the held gap), so this protects
    both the per-anchor predict step inside the filter chain and the final gap-crossing extrapolation
    identically. Decaying the MEAN alone is not enough: position is a double integral of a rotating
    velocity, so the coordinated-turn Jacobian's sensitivity to omega (d(position)/d(omega)) grows
    with dt^2. A stale, sizeable omega VARIANCE carried over from the last real measurement -- even
    with an already-decayed, near-zero mean omega -- gets amplified by that dt^2 Jacobian into an
    absurd (multi-thousand-kilometre) position covariance a few minutes into a multi-hour coast. The
    remedy mirrors the mean decay as a linear rescaling of the random variable itself (state[4] <-
    decay*state[4]), so its variance decays as decay^2 and its cross-covariances as decay^1 -- the
    standard covariance-transformation rule for a linear rescaling."""
    remaining = max(dt, 0.0)
    while remaining > 0:
        step = min(substep, remaining)
        state, P = ctrv_predict(state, P, step, sigma_a, sigma_om)
        decay = np.exp(-step / tau)
        state = state.copy(); state[4] *= decay
        P = P.copy(); P[4, :] *= decay; P[:, 4] *= decay
        remaining -= step
    return state, P


def _imm_step(imm: IMMState, dt, z, models, R):
    mixed_states, mixed_covs, cbar = _imm_mix(imm)
    new_states, new_covs, liks = [], [], []
    for j, (sigma_a, sigma_om) in enumerate(models):
        xp, Pp = _coast(mixed_states[j], mixed_covs[j], dt, sigma_a, sigma_om)
        if z is not None:
            xu, Pu, ll = kf_update(xp, Pp, z, R)
        else:
            xu, Pu, ll = xp, Pp, 0.0
        new_states.append(xu); new_covs.append(Pu); liks.append(ll)
    if z is not None:
        liks = np.array(liks); w = np.exp(liks - liks.max()) * cbar
        mu_new = w / w.sum()
    else:
        mu_new = cbar
    return IMMState(new_states, new_covs, mu_new)


def run_chain(t: np.ndarray, x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray,
             models, r_var: float):
    """Filter a chronologically-ordered (t ascending) two-sided anchor sequence with the 2-mode IMM.
    vx, vy are the REPORTED (SOG/COG-derived) velocities at each anchor, observed alongside position
    (see H_FULL). Returns the mode-COMBINED (state, cov, mode_probs) at the LAST processed anchor."""
    n = len(t)
    r_mat = np.diag([r_var, r_var, SOG_COG_VAR, SOG_COG_VAR])
    if n == 1:
        state0 = np.array([x[0], y[0], vx[0], vy[0], 0.0])
        P0 = np.diag([r_var, r_var, SOG_COG_VAR, SOG_COG_VAR, 0.2 ** 2])
        return state0, P0, np.array([0.5, 0.5])
    state0 = np.array([x[0], y[0], vx[0], vy[0], 0.0])       # reported velocity seeds the state directly
    P0 = np.diag([r_var, r_var, SOG_COG_VAR, SOG_COG_VAR, 0.15 ** 2])
    imm = IMMState([state0.copy(), state0.copy()], [P0.copy(), P0.copy()], np.array([0.5, 0.5]))
    for k in range(1, n):
        dt = max(t[k] - t[k - 1], 1e-3)
        z = np.array([x[k], y[k], vx[k], vy[k]])
        imm = _imm_step(imm, dt, z, models, r_mat)
    state, P = imm.combine()
    return state, P, imm.mu


def extrapolate(state, P, dt, models, mu):
    """Predict-only (no measurement) coast forward by dt, using the mode-probability-blended process
    noise -- the standard way an IMM reports a state estimate beyond its last measurement."""
    sigma_a = sum(mu[j] * models[j][0] for j in range(2))
    sigma_om = sum(mu[j] * models[j][1] for j in range(2))
    return _coast(state, P, dt, sigma_a, sigma_om)


def fuse_positions(pos_f, Pf_2x2, pos_b, Pb_2x2):
    """Inverse-covariance fusion of the forward- and backward-filtered position marginals."""
    Sf_inv = np.linalg.inv(Pf_2x2 + 1e-6 * np.eye(2))
    Sb_inv = np.linalg.inv(Pb_2x2 + 1e-6 * np.eye(2))
    P_fused = np.linalg.inv(Sf_inv + Sb_inv)
    return P_fused @ (Sf_inv @ pos_f + Sb_inv @ pos_b)


def _coast_through(state0, P0, t_anchor, held_t, order, models, mu, forward: bool):
    """Cumulative Markov coast through the held times (ascending order for the forward side,
    descending for the backward side), carrying BOTH state and covariance together so each held
    time's substeps are computed exactly once. Returns (pos[n_held,2], P2x2[n_held,2,2])."""
    n = len(held_t)
    pos = np.zeros((n, 2)); P2 = np.zeros((n, 2, 2))
    state, P, t_cur = state0, P0, t_anchor
    for i in order:
        tq = held_t[i]
        dt = (tq - t_cur) if forward else (t_cur - tq)
        state, P = extrapolate(state, P, dt, models, mu)
        pos[i] = state[:2]; P2[i] = P[:2, :2]; t_cur = tq
    return pos, P2


MAX_CHAIN_ANCHORS = 200   # sliding-window cap -- see ctrv_kalman_predict docstring


def _window(d: dict, k: int, closest_last: bool):
    """Truncate an anchor dict to its k entries closest to the gap (the tail if closest_last, else
    the head), preserving chronological order."""
    n = len(d["t"])
    if n <= k:
        return d
    sl = slice(n - k, n) if closest_last else slice(0, k)
    return {key: val[sl] for key, val in d.items()}


# ------------------------------------------------------------------------------------------------ #
#  Top-level: two-sided IMM reconstruction of the held (query) times of ONE gap                      #
# ------------------------------------------------------------------------------------------------ #
def ctrv_kalman_predict(before: dict, after: dict, held_t: np.ndarray, hp: dict):
    """A handful of real runs carry anchor chains up to ~3,000 pings long (a run with unusually dense
    AIS coverage); a Kalman filter's state absorbs older measurements' influence as it goes; and
    process noise causes the filter to steadily 'forget' anchors from far in the past regardless.
    Anchors are therefore windowed to the MAX_CHAIN_ANCHORS closest to the gap on each side --
    standard sliding-window Kalman-filtering practice, not a result-altering tweak (it only discards
    anchors whose influence on the boundary state estimate is already negligible), and it bounds the
    per-graph cost that would otherwise scale with a rare run's total ping count."""
    before = _window(before, MAX_CHAIN_ANCHORS, closest_last=True)
    after = _window(after, MAX_CHAIN_ANCHORS, closest_last=False)
    models = [(hp["sigma_a_cv"], hp["sigma_om_cv"]), (hp["sigma_a_ct"], hp["sigma_om_ct"])]
    r_var = hp["r_var"]

    xf, Pf, muf = run_chain(before["t"], before["x"], before["y"], before["vx"], before["vy"],
                           models, r_var)
    tL = before["t"][-1]

    # Backward chain: negate time (so the SAME forward-filtering code stays self-consistent -- see
    # module docstring) AND negate the reported velocity, since velocity in the negated-time frame is
    # d(position)/d(-t) = -d(position)/dt.
    t_rev = -after["t"][::-1]; x_rev = after["x"][::-1]; y_rev = after["y"][::-1]
    vx_rev = -after["vx"][::-1]; vy_rev = -after["vy"][::-1]
    xb, Pb, mub = run_chain(t_rev, x_rev, y_rev, vx_rev, vy_rev, models, r_var)
    tR = after["t"][0]

    # Coast CUMULATIVELY through the held times (ascending for the forward side, descending for the
    # backward side) rather than re-extrapolating from the anchor independently per point: mathematically
    # equivalent (the coast is Markov) and avoids redoing the same substeps for every held query time.
    order_f = np.argsort(held_t)
    fwd_pos, fwd_P = _coast_through(xf, Pf, tL, held_t, order_f, models, muf, forward=True)
    bwd_pos, bwd_P = _coast_through(xb, Pb, tR, held_t, order_f[::-1], models, mub, forward=False)

    preds = np.zeros((len(held_t), 2))
    for i in range(len(held_t)):
        preds[i] = fuse_positions(fwd_pos[i], fwd_P[i], bwd_pos[i], bwd_P[i])
    return preds[:, 0], preds[:, 1]


# ------------------------------------------------------------------------------------------------ #
#  SELF-TEST: synthetic straight-line and constant-turn-rate gaps                                    #
# ------------------------------------------------------------------------------------------------ #
if __name__ == "__main__":
    hp = dict(sigma_a_cv=0.05, sigma_om_cv=0.005, sigma_a_ct=0.15, sigma_om_ct=0.03, r_var=25.0 ** 2)

    # ---- straight line, constant velocity ----
    t_anchor = np.arange(0, 601, 60.0)                      # 11 anchors, 60s apart, 10 min span
    v = np.array([5.0, -2.0])
    p0 = np.array([1000.0, -500.0])
    pos = p0[None, :] + v[None, :] * t_anchor[:, None]
    vv = np.tile(v, (len(t_anchor), 1))                     # reported velocity = truth (as real AIS SOG/COG would)
    before = dict(t=t_anchor, x=pos[:, 0], y=pos[:, 1], vx=vv[:, 0], vy=vv[:, 1])
    after_t = t_anchor + 3600.0 + 600.0                     # 1h gap after the last "before" anchor
    pos_a = p0[None, :] + v[None, :] * after_t[:, None]
    vv_a = np.tile(v, (len(after_t), 1))
    after = dict(t=after_t, x=pos_a[:, 0], y=pos_a[:, 1], vx=vv_a[:, 0], vy=vv_a[:, 1])
    held_t = np.linspace(t_anchor[-1], after_t[0], 9)
    px, py = ctrv_kalman_predict(before, after, held_t, hp)
    true_x = p0[0] + v[0] * held_t; true_y = p0[1] + v[1] * held_t
    err = np.hypot(px - true_x, py - true_y)
    print(f"[straight-line]      max pos err = {err.max():.2f} m (over a {(after_t[0]-t_anchor[-1])/3600:.1f}h gap)")

    # ---- moderate, realistic turn rate ----
    for om, gap_h in ((0.0003, 1.0), (0.001, 1.0)):
        t_anchor = np.arange(0, 601, 60.0)
        th = om * t_anchor
        cx, cy = 0.0, 5.0 / om
        R = 5.0 / om
        pos = np.stack([cx + R * np.sin(th), cy - R * np.cos(th)], axis=1)
        vel = np.stack([5.0 * np.cos(th), 5.0 * np.sin(th)], axis=1)      # reported tangential velocity
        before = dict(t=t_anchor, x=pos[:, 0], y=pos[:, 1], vx=vel[:, 0], vy=vel[:, 1])
        after_t = t_anchor + gap_h * 3600.0 + 600.0
        th_a = om * after_t
        pos_a = np.stack([cx + R * np.sin(th_a), cy - R * np.cos(th_a)], axis=1)
        vel_a = np.stack([5.0 * np.cos(th_a), 5.0 * np.sin(th_a)], axis=1)
        after = dict(t=after_t, x=pos_a[:, 0], y=pos_a[:, 1], vx=vel_a[:, 0], vy=vel_a[:, 1])
        held_t = np.linspace(t_anchor[-1], after_t[0], 9)
        th_true = om * held_t
        true_x = cx + R * np.sin(th_true); true_y = cy - R * np.cos(th_true)
        px, py = ctrv_kalman_predict(before, after, held_t, hp)
        err = np.hypot(px - true_x, py - true_y)
        cx_ = true_x[0] + (true_x[-1] - true_x[0]) * (held_t - held_t[0]) / (held_t[-1] - held_t[0])
        cy_ = true_y[0] + (true_y[-1] - true_y[0]) * (held_t - held_t[0]) / (held_t[-1] - held_t[0])
        chord_err = np.hypot(cx_ - true_x, cy_ - true_y)
        total_deg = np.degrees(om * gap_h * 3600)
        print(f"[turn om={om:.4f} rad/s, {total_deg:.0f} deg/{gap_h:.0f}h gap]  "
              f"IMM max err={err.max():.1f}m  chord max err={chord_err.max():.1f}m")

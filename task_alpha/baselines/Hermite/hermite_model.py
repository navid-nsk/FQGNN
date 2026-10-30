"""Piecewise linear + piecewise cubic Hermite interpolation with a curvature-based mode selector,
article-faithful implementation of Sections 3.1-3.4 of:

    Nguyen, V.-S., Im, N.-K., & Lee, S.-M. (2015). "The Interpolation Method for the missing AIS
    Data of Ship." Journal of Navigation and Port Research, 39(5), 377-384.
    DOI: 10.5394/KINPR.2015.39.5.377

Transcription source: Sections 3.1-3.4 (Eq. 1-12) and the worked examples of Section 4 of the
article above.

WHAT THIS IS:
  - Every AIS parameter (lat, long, COG, SOG, HDG) is treated as its own scalar time series
    (ti, fi) (Sec.3 preamble). This module only NEEDS lat(t) and long(t), because the shared
    evaluation harness scores position only (model_errors_latlon) -- COG/SOG/HDG channels are not
    implemented (nothing in the article requires angle-unwrapping etc. for them here; out of scope).
  - For a given two-sided gap, a curvature-based identification mechanism (Eq.8-12, Sec.3.3) picks
    ONE mode -- "LINEAR" (Eq.1-2) or "CURVE"/Hermite (Eq.3-7) -- and that SAME mode is used to fill
    BOTH lat(t) and long(t) for that gap (position is one 2-D geometric curve for the purpose of the
    curvature test; see AMBIGUITY (b) below).
  - LINEAR branch (Eq.1-2): the two-point straight line through the last known sample before the
    gap (A_i) and the first known sample after it (A_{i+1}). Strictly local -- no other sample
    participates.
  - CURVE branch (Eq.3-7): ONE cubic Hermite polynomial spans the whole gap, built from
    f_i, f_{i+1} (the same two bracketing anchors) and TWO slopes d_i, d_{i+1} that are BOTH
    backward finite-difference secants of the raw position series itself (Eq.4-5) -- NOT the AIS
    instrument's reported SOG/COG, and NOT a shape-preserving (PCHIP) or centred-secant rule. d_i
    needs one further known sample A_{i-1} immediately before A_i (Eq.4-5).

AMBIGUITY RESOLUTIONS (every implementation choice the article leaves open is recorded here):

  (a) Coordinate system for the curvature/identification test (Eq.8-12): the paper
      states this only generically as a "Cartesian coordinate system" with symbols x, y and never
      says how lat/long map onto them, nor does it re-invoke a degree<->metre projection anywhere
      in Sec.3. Per the vanilla, no-projection reading used throughout the rest of the paper's own
      position handling (Eq.1-7 fit lat/long directly in raw degrees, Tables 2-3/5 report degrees),
      THIS IMPLEMENTATION USES x=long, y=lat DIRECTLY IN RAW DEGREES for the curvature test too --
      no ENU/UTM projection is introduced (that would be importing an idea the article never states,
      which this implementation avoids throughout).

  (b) Whether lat and long get ONE combined identification verdict or two independent ones
      (Sec.3.3-3.4): the generic machinery of Sec.3 (each (ti,fi) series gets its own verdict) reads as
      if every one of the five series is independent. But curvature (Eq.8-12) is a property of a
      SINGLE planar curve (needs both x=long and y=lat jointly -- there is no such thing as "the
      curvature of latitude alone"), and the paper's own worked example in Sec.4.2 confirms this:
      "the linear interpolation is applied for interpolating 4 points of THE SHIP POSITION" (p.384)
      -- one verdict, applied to position as a whole, not separately to lat vs long. This module
      therefore computes ONE curvature-based mode from the joint (long, lat) trajectory and applies
      that same mode to fill both the lat(t) and long(t) series independently via Eq.1-2/3-7 (each
      still fit as its OWN 1-D (ti,fi) series per Eq.6 -- only the mode decision, not the fit, is
      shared). This is the most internally consistent reading of Sec.3.3/3.4 + Sec.4.2 taken
      together and is flagged, not asserted as explicit text.

  (c) Curvature window + discrete-differentiation scheme (Eq.8-12 stated only as continuous
      calculus; the article states neither): this implementation uses a
      fixed lookback/lookahead window of W=8 known samples (or fewer, if the track does not extend
      that far) immediately flanking the gap on each side -- i.e. the LAST 8 known samples before
      the gap for kappa1/sigma1, and the FIRST 8 known samples after the gap for kappa2 -- matching
      the local, near-the-gap character of the paper's own Fig.3 illustration ("Time series 1" /
      "Time series 2" are short segments immediately flanking the missing area) without copying any
      window logic from any other baseline. Within a
      window, xdot/ydot at each INTERIOR sample i are estimated by the standard central secant
      (x[i+1]-x[i-1])/(t[i+1]-t[i-1]), and xddot/yddot by the standard 3-point central second-
      difference for a non-uniform grid, 2*[(x[i+1]-x[i])/dt_next - (x[i]-x[i-1])/dt_prev]/(dt_prev+
      dt_next) -- the plain vanilla finite-difference scheme, not anything specific to our codebase.
      A window with fewer than 3 points cannot produce even one curvature estimate this way; see (d).

  (d) Degenerate/too-short windows (a boundary case the article leaves open, generalised): if
      either the before- or after-window has fewer than 3 known samples, the curvature test (Eq.8-12)
      cannot be evaluated at all (no interior point exists for a central 2nd-difference). Resolution:
      treat the identification mechanism as inconclusive and default to LINEAR for that gap -- the
      paper's own baseline/no-information case ("nearing a straight line", Sec.3 preamble) -- rather
      than inventing a substitute curvature scheme.

  (e) Missing A_{i-1} for Eq.(4) when CURVE mode is otherwise indicated but the before-side has only
      ONE known sample total (i.e. the gap starts at the very first observed sample of the whole
      track, so d_i cannot be formed): the paper does not address this. Resolution:
      fall back to LINEAR for that gap (Eq.4 as literally written is simply not evaluable; using the
      paper's OTHER stated branch is the most literal fallback available, not a new formula).

  (f) Degenerate zero-width segment (t_{i+1} == t_i): not addressed by the article (assumes distinct
      chronological samples). Standard vanilla floor of 1e-9 s applied to any time-difference
      denominator to avoid a divide-by-zero; never triggers on real, distinct-timestamp AIS samples.

FIDELITY CHECK: this module does NOT use AIS-reported SOG/COG anywhere in the position slope
construction (contrast with velocity-clamped Hermite variants, which clamp derivatives to
instrument velocity) -- d_i, d_{i+1} here are pure backward secants of
the position series itself (Eq.4-5, literally transcribed). It does not use a same-side/centred
secant rule, PCHIP/Fritsch-Carlson monotonicity limiting, omega/Sobolev weighting, ENU projection,
or a [0,1]-normalised h00/h10/h01/h11 basis-function shortcut. The knot set for the CURVE branch
is asymmetric
(2 points before the gap, 1 after) purely because Eq.4-5 as printed are BOTH backward differences --
this asymmetry is the article's own structural consequence, not something added here.
"""
from __future__ import annotations

import numpy as np

WINDOW = 8          # AMBIGUITY (c): fixed lookback/lookahead sample count for the curvature test
MIN_WINDOW_PTS = 3  # AMBIGUITY (d): fewer than this -> curvature test undefined -> fallback LINEAR


# ------------------------------------------------------------------------------------------------ #
#  Eq. (8)-(12): curvature of the (long, lat) trajectory, discrete finite-difference estimate        #
# ------------------------------------------------------------------------------------------------ #
def _finite_diff_derivs(t: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Central finite differences on a possibly non-uniform time grid, AMBIGUITY (c). Returns
    (xdot, ydot, xddot, yddot) at the INTERIOR indices [1 .. m-2] of the window (m = len(t) >= 3);
    a central 2nd-derivative estimate needs one neighbour on each side, so edge samples of the
    window are not assigned a curvature value."""
    m = len(t)
    xdot = np.empty(m - 2); ydot = np.empty(m - 2)
    xddot = np.empty(m - 2); yddot = np.empty(m - 2)
    for j, i in enumerate(range(1, m - 1)):
        dt_prev = max(t[i] - t[i - 1], 1e-9)
        dt_next = max(t[i + 1] - t[i], 1e-9)
        xdot[j] = (x[i + 1] - x[i - 1]) / (dt_prev + dt_next)
        ydot[j] = (y[i + 1] - y[i - 1]) / (dt_prev + dt_next)
        xddot[j] = 2.0 * ((x[i + 1] - x[i]) / dt_next - (x[i] - x[i - 1]) / dt_prev) / (dt_prev + dt_next)
        yddot[j] = 2.0 * ((y[i + 1] - y[i]) / dt_next - (y[i] - y[i - 1]) / dt_prev) / (dt_prev + dt_next)
    return xdot, ydot, xddot, yddot


def _curvature_window_stats(t: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Eq.(12): kappa = (xdot*yddot - ydot*xddot) / (xdot^2+ydot^2)^1.5 at every interior sample of
    the window, x=long, y=lat in raw degrees (AMBIGUITY (a)). Returns (mean, std, ok); ok=False if
    the window is too short to evaluate (AMBIGUITY (d))."""
    if len(t) < MIN_WINDOW_PTS:
        return 0.0, 0.0, False
    xdot, ydot, xddot, yddot = _finite_diff_derivs(t, x, y)
    denom = (xdot ** 2 + ydot ** 2) ** 1.5
    denom = np.where(denom > 1e-12, denom, 1e-12)
    kappa = (xdot * yddot - ydot * xddot) / denom
    return float(np.mean(kappa)), float(np.std(kappa)), True


def identify_mode(before_t, before_lon, before_lat, after_t, after_lon, after_lat, window: int = WINDOW):
    """Sec.3.3 selection rule: kappa1/sigma1 from the last `window` known samples before the gap,
    kappa2 from the first `window` known samples after it; CURVE if kappa2 is outside
    kappa1 +/- 2*sigma1, else LINEAR (AMBIGUITY (b): one verdict for the joint position curve)."""
    wb = min(window, len(before_t)); wa = min(window, len(after_t))
    k1, s1, ok1 = _curvature_window_stats(before_t[-wb:], before_lon[-wb:], before_lat[-wb:])
    k2, _s2, ok2 = _curvature_window_stats(after_t[:wa], after_lon[:wa], after_lat[:wa])
    if not (ok1 and ok2):
        return "LINEAR"          # AMBIGUITY (d): identification undefined -> conservative default
    if k2 < (k1 - 2 * s1) or k2 > (k1 + 2 * s1):
        return "CURVE"
    return "LINEAR"


# ------------------------------------------------------------------------------------------------ #
#  Eq. (1)-(2): piecewise linear, and Eq. (3)-(7): piecewise cubic Hermite, one series (ti, fi)       #
# ------------------------------------------------------------------------------------------------ #
def _linear_fill(t_i: float, f_i: float, t_ip1: float, f_ip1: float, t_query: np.ndarray) -> np.ndarray:
    """Eq.(2), literal."""
    denom = t_ip1 - t_i
    denom = denom if abs(denom) > 1e-9 else (1e-9 if denom >= 0 else -1e-9)   # AMBIGUITY (f)
    return f_i + (t_query - t_i) * (f_ip1 - f_i) / denom


def _hermite_fill(t_im1: float, f_im1: float, t_i: float, f_i: float, t_ip1: float, f_ip1: float,
                   t_query: np.ndarray) -> np.ndarray:
    """Eq.(3)-(7), literal: d_i, d_{i+1} are BOTH backward secants (Eq.4-5); P(t) is the standard
    two-point cubic Hermite basis (Eq.6) evaluated on s=t-t_i, h=t_{i+1}-t_i."""
    h = t_ip1 - t_i
    h = h if abs(h) > 1e-9 else (1e-9 if h >= 0 else -1e-9)                    # AMBIGUITY (f)
    dt_left = t_i - t_im1
    dt_left = dt_left if abs(dt_left) > 1e-9 else (1e-9 if dt_left >= 0 else -1e-9)
    d_i = (f_i - f_im1) / dt_left                                              # Eq.(4)
    d_ip1 = (f_ip1 - f_i) / h                                                  # Eq.(5)
    s = t_query - t_i
    return (((3 * h * s ** 2 - 2 * s ** 3) / h ** 3) * f_ip1
            + ((h ** 3 - 3 * h * s ** 2 + 2 * s ** 3) / h ** 3) * f_i
            + ((s ** 3 - h * s ** 2) / h ** 2) * d_ip1
            + ((s ** 3 - 2 * h * s ** 2 + h ** 2 * s) / h ** 2) * d_i)         # Eq.(6)


# ------------------------------------------------------------------------------------------------ #
#  Leak-safe anchor extraction (same boundary-index logic as common_gaps.anchor_states_enu,          #
#  reproduced here so this module is self-contained)                                                #
# ------------------------------------------------------------------------------------------------ #
def anchor_states_lonlat(g):
    """Leak-safe two-sided anchor extraction directly in raw (lat, lon) degrees.

    Only non-held (observed) node fields are read (g.x is all-zeroed at held nodes by construction;
    g.true_lonlat at held nodes is ground truth and is never touched here).

    Returns (before, after, held_t, hidx):
      before, after  dicts {t (sec), lat, lon (deg)} for ALL known samples strictly before / after
                     the gap, in chronological (ascending-time) order -- the full flanking run, so
                     the curvature window (AMBIGUITY c) and the d_i backward-difference (Eq.4) both
                     have as much history as is actually available.
      held_t         [n_held] absolute-scale seconds of every held (query) node.
      hidx           [n_held] the held node indices (original order).
    """
    coord = g.coord.numpy(); held = g.mask.numpy().astype(bool)
    tr = g.true_lonlat.numpy()
    hidx = np.where(held)[0]
    si, last = int(hidx[0]), int(hidx[-1])
    span_norm = max(coord[last, 0] - coord[si, 0], 1e-9)
    dur_sec = (g.gap_minutes * 60.0) / span_norm            # recover absolute time scale
    t_sec = coord[:, 0] * dur_sec

    lat = tr[:, 0]; lon = tr[:, 1]                            # observed-node truth == observation

    def _pack(idx):
        return dict(t=t_sec[idx].astype(np.float64), lat=lat[idx].astype(np.float64),
                    lon=lon[idx].astype(np.float64))

    n = len(held)
    before = _pack(np.arange(0, si))
    after = _pack(np.arange(last + 1, n))
    return before, after, t_sec[hidx].astype(np.float64), hidx


# ------------------------------------------------------------------------------------------------ #
#  Top-level: fill ONE two-sided position gap                                                         #
# ------------------------------------------------------------------------------------------------ #
def art2_predict(before: dict, after: dict, held_t: np.ndarray, window: int = WINDOW):
    """Article-faithful piecewise linear / cubic Hermite fill of ONE two-sided gap for POSITION
    (Sec.3.1-3.4).

    before, after: dicts from anchor_states_lonlat (full flanking runs, chronological order).
    held_t:        [n_held] absolute-scale seconds of the query (held) times.

    Returns (pred_lat, pred_lon) in degrees, aligned with held_t.
    """
    t_b, lat_b, lon_b = before["t"], before["lat"], before["lon"]
    t_a, lat_a, lon_a = after["t"], after["lat"], after["lon"]

    mode = identify_mode(t_b, lon_b, lat_b, t_a, lon_a, lat_a, window=window)

    t_i, lat_i, lon_i = t_b[-1], lat_b[-1], lon_b[-1]
    t_ip1, lat_ip1, lon_ip1 = t_a[0], lat_a[0], lon_a[0]

    if mode == "CURVE" and len(t_b) >= 2:                     # AMBIGUITY (e)
        t_im1, lat_im1, lon_im1 = t_b[-2], lat_b[-2], lon_b[-2]
        pred_lat = _hermite_fill(t_im1, lat_im1, t_i, lat_i, t_ip1, lat_ip1, held_t)
        pred_lon = _hermite_fill(t_im1, lon_im1, t_i, lon_i, t_ip1, lon_ip1, held_t)
    else:
        pred_lat = _linear_fill(t_i, lat_i, t_ip1, lat_ip1, held_t)
        pred_lon = _linear_fill(t_i, lon_i, t_ip1, lon_ip1, held_t)
    return pred_lat, pred_lon

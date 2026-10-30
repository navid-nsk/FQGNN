"""task_alpha.fq_opm -- Op-M: the MEASURED-SPECTRUM MULTI-ANCHOR FQ readout.

Two orthogonal channels are added to the Op-H-W readout (fq_oph.py), each with a continuous dial
and a degenerate setting that IS the Op-H-W operator:

  NORM channel   (dial gamma):  the per-gap MEASURED spectral metric -- detrended
      chord-residual periodogram on the kept comb, floors, Jacobi-band smoothing, then the
      GEOMETRIC-interpolation law  log w_hat = log w_prod + gamma * clamp(0.5*log I - log w_prod,
      +-c_w) * 1[k in K_hi], with w_prod the learned Op-H-W metric.  The clamp sits INSIDE the
      gamma term, and the law is applied MULTIPLICATIVELY (w_hat = w_prod * exp(gamma * corr)), so
      gamma = 0 leaves w_prod BIT-FOR-BIT unchanged (no clamp, no floor, no smoothing touches w_prod).
  CONSTRAINT channel (dial Lambda): exterior (non-bracketing) anchors as SOFT weighted value
      functionals of the detrended chord residuals rho~ -- the (Op-M-sol) closed form below, with
      its (I+G~)^-1 influence structure.  Lambda = 0 (code path lam=None) skips the block
      ENTIRELY: opm_readout DELEGATES to fq_oph.oph_readout, so the degenerate
      case runs the IDENTICAL code path (true bit-identity).

Closed form, in u = W^-1 c coordinates with B~ = B W, V~ = V W,
u0 = B~^+ d, P = I - B~^+ B~ (applied as an OPERATOR, never materialized):

    r0 = rho~_ext - V~ (u0 + g P eta)                          (Lambda-independent)
    G~ = Lam^1/2 V~ P V~^T Lam^1/2      (PSD; I + G~ >= I always -- no rcond, no rank condition)
    y  = P V~^T Lam^1/2 (I + G~)^-1 Lam^1/2 r0
    c  = W (u0 + g P eta + y)                                  (Op-M-sol)

Exactness for EVERY learned/measured component: B c = d holds identically (y and
P eta live in null(B~)), so boundary interpolation + Dmu velocity match survive any (gamma,
Lambda, eta, w).  The exterior channel deforms the reconstruction with ZERO value (and zero
slope at velocity-row endpoints) at the gap boundary -- a bump supported in the open gap.
Gating: Lambda_eff = g_M * Lambda; Tier C (default) g_M = g (the Op-H curvature gate --
preserves locally-affine chord exactness, and inherits the parallel-tangent blind spot);
Tier O g_M = tanh(softplus(rho_M) * e_bar), e_bar = exterior detrended-residual energy ratio
(self-gating on globally-affine data).

The measured metric is estimated in five steps, labelled O1-O5 in the code: (O1) chord
residuals with a per-side affine detrend and an exponential taper; (O2) the increment channel
(same-side secants of the detrended residuals, transfer-corrected per pair); (O3) the value
channel (finite Fourier-Bohr means on the kept comb); (O4) the robust split (correlated
value/increment pair vs. the incoherent per-side variant); (O5) the noise floors and the Jacobi
least-squares band smoothing on K_hi.

Leak-safety: every quantity here is a fixed measurable function of the OBSERVED
anchor rows (at, ay, av) and the gap endpoints only; increments are same-side only; held/query
rows are never read.  fp32 throughout; fp64 only for the tiny N_o x N_o Cholesky.

Implementation notes:
  * `opm_readout` has a strict superset of the oph_readout signature (adds `lam` and the metric
    dials) and DELEGATES to fq_oph.oph_readout when lam is None, so the degenerate case runs the
    same code path, bit for bit.
  * `opm_metric`'s 7th argument is the kept-band Jacobi basis P_kept [K, R+1], not its
    pseudoinverse: O5 defines the smoothing LS over the K_hi rows only, which vary per gap, so a
    globally precomputed pinv cannot exist -- the (P^T P)^-1 P^T is formed inside on the K_hi
    rows.  Extra kwargs w_prod (bit-exact gamma = 0 multiplication;
    exp(log w_prod) alone is not bit-identical) and khi_min_cycles (the K_hi cutoff constant).
  * K_hi (resolvable band): |omega_k| * aperture >= 2*pi*khi_min_cycles with aperture =
    min(anchor span, 2*kappa*T), the realized aperture; khi_min_cycles = 1.0 by default.
  * lambda-head z_j (wired in fqdeep_native): the same-leg/heading-consistency information is
    represented by the single heading-consistency scalar (cog . chord direction);
    no per-anchor leg segmentation exists in the anchor rows.
"""
from __future__ import annotations

import math
import os

import torch
from torch.nn.utils.rnn import pad_sequence

from .fq_oph import oph_readout

PI = math.pi
LOG10 = math.log(10.0)

_PINV_FALLBACKS = 0        # process-wide count of CPU-LAPACK pinv fallbacks (see _pinv_safe)


def _pinv_safe(X, **kw):
    """torch.linalg.pinv with a CPU-LAPACK retry on cuSOLVER SVD non-convergence.

    On some GPU/cuSOLVER builds the batched fp32 pinv of the boundary system B~ [G,4,2M] can
    raise torch.linalg.LinAlgError (cusolver gesvdj "failed to converge", error code 4) on
    batches that other builds handle -- a cuSOLVER convergence difference, NOT a rank failure.
    The retry computes the SAME pinv -- same dtype (fp32 stays fp32), same rcond/rtol
    semantics -- with CPU LAPACK and moves the result back.  Bit-for-bit no-op on every call
    where cuSOLVER converges (only the exception path differs); autograd flows through the CPU
    op unchanged.  The first fallback prints once; the process-wide count is kept in
    fq_opm._PINV_FALLBACKS for diagnostics."""
    global _PINV_FALLBACKS
    try:
        return torch.linalg.pinv(X, **kw)
    except torch.linalg.LinAlgError:
        _PINV_FALLBACKS += 1
        if _PINV_FALLBACKS == 1:
            print("[fq_opm] cuSOLVER pinv non-convergence -> CPU LAPACK fallback (same dtype/"
                  "rcond; prints once, count in fq_opm._PINV_FALLBACKS)", flush=True)
        return torch.linalg.pinv(X.cpu(), **kw).to(X.device)


def _sqrt_nz(x):
    """sqrt whose subgradient at x == 0 is zero.

    Forward is bit-identical to x.sqrt() for x >= 0 (sqrt(0) = 0 either way).  Backward differs
    ONLY at x == 0: plain sqrt has d/dx = inf there, and Lambda_eff = g_M * lam_e contains exact
    zeros BY CONSTRUCTION (non-exterior/padded rows in the batched path; g_M = 0 on straight
    gaps via the tanh curvature gate), so autograd's chain would give inf * 0 = NaN -- and one
    NaN poisons the total clip_grad_norm_, which rescales every gradient to NaN (a NaN-poisoned
    model then yields a non-finite B~ and a failing pinv).  The double-where yields an
    exactly-zero gradient at x == 0 (the appropriate subgradient: sl == 0 identically there)
    and the usual 0.5/sqrt(x) gradient for x > 0."""
    pos = x > 0
    return torch.where(pos, torch.where(pos, x, torch.ones_like(x)).sqrt(),
                       torch.zeros_like(x))


# --------------------------------------------------------------------------- O1: chord residuals
def _detrend_sorted(ts, ys, gapi, kappa: float = 1.0):
    """(O1): chord residuals rho, per-side affine detrend -> rho~, taper varpi.
    Inputs in SORTED time order.  A side with <= 2 anchors detrends to EXACTLY 0 (the O5 absolute
    floor closes the resulting zero-spectrum edge).  The per-side affine detrend is the same
    Lev-Reti {delta_0, delta_0'} closed-class functional as the chord atom.
    Returns (rho_t, rho, tau, varpi, T, slope) -- all sorted order."""
    dev, dt = ts.device, ys.dtype
    a, b = ts[gapi], ts[gapi + 1]
    T = (b - a).clamp_min(1e-9)
    t0 = 0.5 * (a + b)
    tau = ts - t0
    pa = ys[gapi]
    slope = (ys[gapi + 1] - ys[gapi]) / T
    rho = ys - (pa.unsqueeze(0) + (ts - a).unsqueeze(-1) * slope.unsqueeze(0))
    varpi = torch.exp(-tau.abs() / (kappa * T))
    rho_t = torch.zeros_like(rho)
    Ma = ts.shape[0]
    ar = torch.arange(Ma, device=dev)
    for m in (ar <= gapi, ar > gapi):
        n = int(m.sum())
        if n <= 2:
            continue                                   # rho~ = 0 exactly on this side (O1 edge)
        Xd = torch.stack([torch.ones(n, device=dev, dtype=dt), tau[m]], dim=-1)   # [n,2]
        Wd = varpi[m].unsqueeze(-1)                                               # [n,1]
        A2 = Xd.t() @ (Wd * Xd)                                                   # [2,2]
        r2 = Xd.t() @ (Wd * rho[m])                                               # [2,2]
        sol = _pinv_safe(A2) @ r2               # rank-safe closed-form weighted LS
        rho_t[m] = rho[m] - Xd @ sol
    return rho_t, rho, tau, varpi, T, slope


def chord_residual_features(at, ay, kappa: float = 1.0):
    """INPUT-order (rho~ [Ma,2], tau [Ma], T, slope [2]) for the lambda-head.
    Leak-safe: observed anchor rows only."""
    dt = ay.dtype
    at_ = at.to(dt)
    idx = torch.argsort(at_)
    ts, ys = at_[idx], ay[idx]
    gapi = int(torch.argmax(ts[1:] - ts[:-1]))
    rho_t, _, tau, _, T, slope = _detrend_sorted(ts, ys, gapi, kappa)
    inv = torch.empty_like(idx)
    inv[idx] = torch.arange(idx.numel(), device=idx.device)
    return rho_t[inv], tau[inv], T, slope


# --------------------------------------------------------------------------- the measured metric
def opm_metric(at, ay, a, b, om_kept, s_kept, P_kept, log_w_prod, gamma,
               kappa: float = 1.0, nu: float = 1.0, c_w: float = LOG10,
               eps_rel: float = 1e-2, eps_abs: float = 1e-12,
               w_prod=None, khi_min_cycles: float = 1.0):
    """Detrend -> taper -> Fourier-Bohr means (value + increment channels) ->
    I_rob -> floors -> Jacobi LS band smoothing on K_hi -> geometric law with clamp-inside-gamma.
    Pure function of observed rows; ALL measurement inputs are detached inside (the metric
    is a measurement, not a model -- omega gradients still flow through B, V and synthesis in the
    solve).  Returns w_hat [K].  With w_prod given, gamma = 0 returns w_prod BIT-FOR-BIT
    (w_prod * exp(0 * corr) = w_prod * 1.0; corr is guarded finite)."""
    dt = ay.dtype
    om = om_kept.detach().to(dt)                       # omega detached in the estimator
    at_m = at.detach().to(dt)
    ay_m = ay.detach().to(dt)
    idx = torch.argsort(at_m)
    ts, ys = at_m[idx], ay_m[idx]
    gapi = int(torch.argmax(ts[1:] - ts[:-1]))
    rho_t, _, tau, varpi, T, _ = _detrend_sorted(ts, ys, gapi, kappa)
    K, Ma = om.shape[0], ts.shape[0]
    dev = ts.device

    # (O3) value-channel finite Fourier-Bohr means on the comb, all observed anchors
    ang = om.unsqueeze(1) * tau.unsqueeze(0)                                  # [K,Ma]
    ws = varpi.sum().clamp_min(1e-12)
    Re = (torch.cos(ang) * varpi) @ rho_t / ws                                # [K,2]
    Im = (-torch.sin(ang) * varpi) @ rho_t / ws
    I_val = (Re * Re + Im * Im).sum(-1)                                       # [K]

    # (O4) per-side means -> incoherent variant (robust to a mid-gap phase slip)
    ar = torch.arange(Ma, device=dev)
    I_inc = torch.zeros_like(I_val)
    for m in (ar <= gapi, ar > gapi):
        if int(m.sum()) == 0:
            continue
        wsd = varpi[m].sum().clamp_min(1e-12)
        ReS = (torch.cos(ang[:, m]) * varpi[m]) @ rho_t[m] / wsd
        ImS = (-torch.sin(ang[:, m]) * varpi[m]) @ rho_t[m] / wsd
        I_inc = I_inc + (ReS * ReS + ImS * ImS).sum(-1)
    I_inc = 0.5 * I_inc

    # (O2) increment channel: same-side consecutive secants OF THE DETRENDED residuals,
    # transfer-corrected per pair (g_k(D) = 2 sin(om D/2)/D, |g| clipped >= |om|/pi)
    pj = [j for j in range(Ma - 1) if j != gapi]
    if pj:
        j0 = torch.tensor(pj, device=dev, dtype=torch.long)
        dt_p = (tau[j0 + 1] - tau[j0]).clamp_min(1e-9)                        # [P]
        dlt = (rho_t[j0 + 1] - rho_t[j0]) / dt_p.unsqueeze(-1)                # [P,2]
        tmid = 0.5 * (tau[j0 + 1] + tau[j0])
        wmid = torch.exp(-tmid.abs() / (kappa * T))
        gk = 2.0 * torch.sin(om.unsqueeze(1) * dt_p.unsqueeze(0) / 2.0) / dt_p.unsqueeze(0)
        sgn = torch.where(gk >= 0, torch.ones_like(gk), -torch.ones_like(gk))
        gk = sgn * torch.maximum(gk.abs(), om.abs().unsqueeze(1) / PI)        # [K,P] clipped
        contrib = dlt.unsqueeze(0) / gk.unsqueeze(-1)                         # [K,P,2]
        angm = om.unsqueeze(1) * tmid.unsqueeze(0)                            # [K,P]
        wsm = wmid.sum().clamp_min(1e-12)
        ReI = ((torch.cos(angm) * wmid).unsqueeze(-1) * contrib).sum(1) / wsm  # [K,2]
        ImI = ((-torch.sin(angm) * wmid).unsqueeze(-1) * contrib).sum(1) / wsm
        I_der = (ReI * ReI + ImI * ImI).sum(-1)
    else:
        I_der = torch.zeros_like(I_val)

    # (O4) robust split: correlated variance-reduction pair, then the incoherent max
    I_tot = (I_val + nu * I_der) / (1.0 + nu)
    I_rob = torch.maximum(I_tot, I_inc)

    # (O5) floors: chi^2_2 median->mean factor 1/ln2; eps_abs closes the rho~ == 0 edge
    om_med = om.abs().median()
    hi_half = om.abs() > om_med
    if not bool(hi_half.any()):
        hi_half = torch.ones_like(hi_half)
    flr = I_rob[hi_half].median() / math.log(2.0)
    med_all = I_rob.median()
    eps_t = torch.tensor(eps_abs, device=dev, dtype=dt)
    I_raw = torch.maximum(I_rob - flr, torch.maximum(eps_rel * med_all, eps_t))
    logI_raw = torch.log(I_raw)

    # K_hi = resolvable band over the REALIZED aperture (the cutoff is
    # |om|*aperture >= 2 pi * khi_min_cycles)
    span = ts[-1] - ts[0]
    ap = torch.minimum(span, 2.0 * kappa * T).clamp_min(1e-9)
    khi = om.abs() * ap >= 2.0 * PI * khi_min_cycles

    # Jacobi-5 LS smoothing over the K_hi rows (family-aligned compression, O5)
    if P_kept is None:                                 # fallback basis (selftests / standalone)
        s_b = (om.abs() / om.abs().max().clamp_min(1e-12)) if s_kept is None else s_kept.detach().to(dt)
        P_kept = torch.stack([s_b ** r for r in range(5)], dim=-1)
    else:
        P_kept = P_kept.detach().to(dt)
    if bool(khi.any()):
        theta = _pinv_safe(P_kept[khi]) @ logI_raw[khi]                # [J]
        logI_sm = P_kept @ theta                                              # [K]
    else:
        logI_sm = logI_raw                             # correction masked to 0 below anyway

    # diagnostic gamma-hat (Ledoit-Wolf; a switched option, not the default operator):
    # gamma_hat = clamp(1 - v_hat/m_hat, 0, 1) with v_hat = between-channel disagreement
    # variance of the (consistently-detrended) value/increment half-log profiles, m_hat =
    # mean-square deviation of the measured log-profile from the Op-H-W one, both on K_hi.
    # Reached ONLY via gamma == "hat" (env FQ_OPM_GAMMA=hat) -- no other path uses it.
    if isinstance(gamma, str):
        assert gamma == "hat", gamma
        if bool(khi.any()):
            fl2 = torch.maximum(eps_rel * med_all, eps_t)
            Iv = torch.maximum(I_val - flr, fl2)                  # channel-wise floors
            Id = torch.maximum(I_der - flr, fl2)
            Pp = _pinv_safe(P_kept[khi])
            hv = P_kept @ (Pp @ torch.log(Iv[khi]))               # smoothed log-power profiles
            hd = P_kept @ (Pp @ torch.log(Id[khi]))
            vhat = ((0.25 * (hv - hd)[khi]) ** 2).mean()          # var of the 2-channel mean
            mhat = ((0.5 * logI_sm - log_w_prod)[khi] ** 2).mean()
            gamma = torch.clamp(1.0 - vhat / mhat.clamp_min(1e-12), 0.0, 1.0)
        else:
            gamma = torch.zeros((), device=dev, dtype=dt)         # no resolvable band -> off

    # the geometric-interpolation law; clamp INSIDE the gamma term; the finite
    # guard keeps gamma = 0 an exact no-op even on pathological inputs
    corr = torch.clamp(0.5 * logI_sm - log_w_prod, -c_w, c_w) * khi.to(dt)
    corr = torch.where(torch.isfinite(corr), corr, torch.zeros_like(corr))
    if w_prod is not None:
        return w_prod * torch.exp(gamma * corr)        # bit-exact w_prod at gamma == 0
    return torch.exp(log_w_prod + gamma * corr)


def _metric_hat(at, ay, om_all, w, gamma, s_all, P_jac, min_phase,
                kappa, nu, c_w, eps_rel, eps_abs, khi_min_cycles):
    """Route the Op-H-W metric w through the geometric-interpolation law on the kept band.  Returns the
    full-module w_hat [M]; unkept entries and the gamma = 0 case are bit-identical to w."""
    dt = ay.dtype
    at_ = at.to(dt)
    idx = torch.argsort(at_)
    ts = at_[idx]
    gapi = int(torch.argmax(ts[1:] - ts[:-1]))
    a, b = ts[gapi], ts[gapi + 1]
    ba = (b - a).clamp_min(1e-9)
    om = om_all.to(dt)
    keep = om.abs() * ba > min_phase
    if int(keep.sum()) < 2:
        return w                                       # degenerate module: affine fallback anyway
    wk = w.to(dt)[keep]
    log_wp = torch.log(wk.clamp_min(1e-6))
    sk = None if s_all is None else s_all.to(dt)[keep]
    Pk = None if P_jac is None else P_jac.to(dt)[keep]
    w_hat_k = opm_metric(at_, ay, a, b, om[keep], sk, Pk, log_wp, gamma,
                         kappa=kappa, nu=nu, c_w=c_w, eps_rel=eps_rel, eps_abs=eps_abs,
                         w_prod=wk, khi_min_cycles=khi_min_cycles)
    w_new = w.clone()
    w_new[keep] = w_hat_k.to(w.dtype)
    return w_new


# ----------------------------------------------- measured lambda modes + cell-constant gamma law
# Everything in this block is default-off: the model never calls it
# unless an FQ_OPM_* env switch selects it (fqdeep_native).  All quantities are leak-safe
# measurements of the OBSERVED anchor rows (closed class), detached throughout.

# Numerical-validity ceiling for the measured-units lambda modes (c1/c2/c3): 1/sigma^2 is
# unbounded when a gap's residual scale is tiny, and lambda * fp32-roundoff(G) makes I + G~
# numerically indefinite far before the math does.  On real gaps the fp64 Cholesky of I + G~
# fails for an uncapped c3 (lambda ~ 1/eps_abs) and is not reliably solvable at scales of
# 1e5-1e6; 1e4 is the largest decade that stays SPD-solvable -- the default.  Larger ceilings
# are reachable via the FQ_OPM_LAM_MAX dial but are NOT universally solvable.  The clamp is
# numerical validity, not a model choice: it binds only where sigma^2 underflows the operating
# range.
LAM_NUM_MAX = float(os.environ.get("FQ_OPM_LAM_MAX", "1e4"))

def gamma_prime(vbar: float, mbar: float, beta2: float) -> float:
    """The shrinkage law in its CELL-CONSTANT form
        gamma' = clamp( 1 - (vbar + beta2)/mbar , 0, 1 ).
    vbar/mbar are the CELL MEDIANS of the gamma-hat internals v_hat/m_hat (see opm_metric) --
    the per-gap variation of these statistics carries no signal about the oracle gamma, so the
    inputs are flattened; beta2 = the cell-level transfer-defect moment E||b||^2, fit once on
    TRAINING gaps by matching the aggregate grid argmin.  Ledoit-Wolf limit: as v -> 0,
    gamma* -> S/(S + beta2) < 1 for any beta2 > 0 (it does NOT tend to 1)."""
    return float(min(max(1.0 - (vbar + beta2) / max(mbar, 1e-12), 0.0), 1.0))


def _sigma2_eps(rho_t, eps_abs: float = 1e-12):
    """Per-anchor mismatch variance  sigma^2_{eps,j} = ||rho~_j||^2 v sigma^2_flr,
    with sigma^2_flr the anchor-domain transport of the O5 noise-floor convention:
    median of the strictly-positive rows / ln 2 (chi^2_2 median -> mean), eps_abs-floored so
    the rho~ == 0 exact edge (a <= 2-anchor side, O1) stays finite.  At sigma^2 = sigma^2_flr
    the reciprocal 1/sigma^2 is the Bayes plug-in scale.
    Returns (sig2 [Ma], s2flr scalar), same order as rho_t; detached measurement."""
    rn2 = (rho_t.detach() ** 2).sum(-1)
    pos = rn2 > 0
    if bool(pos.any()):
        s2flr = (rn2[pos].median() / math.log(2.0)).clamp_min(eps_abs)
    else:
        s2flr = torch.full((), eps_abs, device=rn2.device, dtype=rn2.dtype)
    return torch.maximum(rn2, s2flr), s2flr


def opm_c2_lambda(at, ay, om_all, min_phase: float = 0.25, kappa: float = 1.0, nu: float = 1.0,
                  eps_abs: float = 1e-12, khi_min_cycles: float = 1.0,
                  return_parts: bool = False):
    """The c2 lambda mode: the CLOSED-FORM lambda-hat (the gamma-hat analogue for the Lambda
    channel), scoreable zero-shot.  lambda_j = h_hat / sigma^2_{eps,j} with the
    Ledoit-Wolf-style trust  h_hat = clamp(1 - v_lam/m_lam, 0, 1):
        m_lam = mean_{K_hi} I_rob / flr     exterior detrended-residual energy over its floor
        v_lam = mean_{K_hi} |I_val - I_der| / flr    between-channel (O3 value vs O2 increment)
                                                     disagreement energy
    both estimated from the exterior anchors' Fourier-Bohr means on the resolvable band K_hi
    (the spectral-mean reading of the disagreement energy at the exterior anchors: the
    channels only differ spectrally -- their anchor-domain values coincide by construction --
    so the Parseval proxy on K_hi is the implementable form; the common floor cancels in the
    ratio).  The g_M gate multiplies INSIDE opm_readout (Lambda_eff = g_M * lambda),
    completing  lambda_j = g_M * h_hat / sigma^2_{eps,j}.  Units check: at h_hat = 1 and
    sigma^2 at its floor this is ~ln2/median||rho~||^2, the Bayes plug-in scale.  Fully
    closed-form, leak-safe (observed anchor rows only), everything detached (a measurement);
    like gamma-hat it guards variance, not coherent bias.  Returns lam [Ma] in INPUT anchor order
    (return_parts: (lam, h_hat, sig2 [Ma, input order], s2flr))."""
    dt = ay.dtype
    at_ = at.detach().to(dt)
    ay_ = ay.detach()
    idx = torch.argsort(at_)
    ts, ys = at_[idx], ay_[idx]
    gapi = int(torch.argmax(ts[1:] - ts[:-1]))
    rho_t, _, tau, varpi, T, _ = _detrend_sorted(ts, ys, gapi, kappa)
    dev = ts.device
    Ma = ts.shape[0]
    sig2, s2flr = _sigma2_eps(rho_t, eps_abs)
    a, b = ts[gapi], ts[gapi + 1]
    ba = (b - a).clamp_min(1e-9)
    om = om_all.detach().to(dt)
    keep = om.abs() * ba > min_phase
    hhat = torch.zeros((), device=dev, dtype=dt)
    if bool((sig2 > s2flr).any()) and int(keep.sum()) >= 2:
        omk = om[keep]
        # (O3) value channel, (O4) incoherent split, (O2) increment channel -- the same
        # Fourier-Bohr estimators as opm_metric, on the kept comb
        ang = omk.unsqueeze(1) * tau.unsqueeze(0)                              # [K,Ma]
        ws = varpi.sum().clamp_min(1e-12)
        Re = (torch.cos(ang) * varpi) @ rho_t / ws
        Im = (-torch.sin(ang) * varpi) @ rho_t / ws
        I_val = (Re * Re + Im * Im).sum(-1)
        ar = torch.arange(Ma, device=dev)
        I_inc = torch.zeros_like(I_val)
        for msk in (ar <= gapi, ar > gapi):
            if int(msk.sum()) == 0:
                continue
            wsd = varpi[msk].sum().clamp_min(1e-12)
            ReS = (torch.cos(ang[:, msk]) * varpi[msk]) @ rho_t[msk] / wsd
            ImS = (-torch.sin(ang[:, msk]) * varpi[msk]) @ rho_t[msk] / wsd
            I_inc = I_inc + (ReS * ReS + ImS * ImS).sum(-1)
        I_inc = 0.5 * I_inc
        pj = [j for j in range(Ma - 1) if j != gapi]
        if pj:
            j0 = torch.tensor(pj, device=dev, dtype=torch.long)
            dt_p = (tau[j0 + 1] - tau[j0]).clamp_min(1e-9)
            dlt = (rho_t[j0 + 1] - rho_t[j0]) / dt_p.unsqueeze(-1)
            tmid = 0.5 * (tau[j0 + 1] + tau[j0])
            wmid = torch.exp(-tmid.abs() / (kappa * T))
            gk = 2.0 * torch.sin(omk.unsqueeze(1) * dt_p.unsqueeze(0) / 2.0) / dt_p.unsqueeze(0)
            sgn = torch.where(gk >= 0, torch.ones_like(gk), -torch.ones_like(gk))
            gk = sgn * torch.maximum(gk.abs(), omk.abs().unsqueeze(1) / PI)
            contrib = dlt.unsqueeze(0) / gk.unsqueeze(-1)
            angm = omk.unsqueeze(1) * tmid.unsqueeze(0)
            wsm = wmid.sum().clamp_min(1e-12)
            ReI = ((torch.cos(angm) * wmid).unsqueeze(-1) * contrib).sum(1) / wsm
            ImI = ((-torch.sin(angm) * wmid).unsqueeze(-1) * contrib).sum(1) / wsm
            I_der = (ReI * ReI + ImI * ImI).sum(-1)
        else:
            I_der = torch.zeros_like(I_val)
        # (O4/O5) robust energy, floor, resolvable band -- opm_metric's conventions verbatim
        I_tot = (I_val + nu * I_der) / (1.0 + nu)
        I_rob = torch.maximum(I_tot, I_inc)
        om_med = omk.abs().median()
        hi = omk.abs() > om_med
        if not bool(hi.any()):
            hi = torch.ones_like(hi)
        flr = (I_rob[hi].median() / math.log(2.0)).clamp_min(eps_abs)
        span = ts[-1] - ts[0]
        ap = torch.minimum(span, 2.0 * kappa * T).clamp_min(1e-9)
        khi = omk.abs() * ap >= 2.0 * PI * khi_min_cycles
        if bool(khi.any()):
            m_lam = I_rob[khi].mean() / flr
            v_lam = (I_val[khi] - I_der[khi]).abs().mean() / flr
            hhat = torch.clamp(1.0 - v_lam / m_lam.clamp_min(1e-12), 0.0, 1.0)
    lam_sorted = (hhat / sig2).clamp_max(LAM_NUM_MAX)
    inv = torch.empty_like(idx)
    inv[idx] = torch.arange(idx.numel(), device=idx.device)
    if return_parts:
        return lam_sorted[inv], hhat, sig2[inv], s2flr
    return lam_sorted[inv]


# --------------------------------------------------------------------------- the Op-M operator
def opm_readout(at, ay, av, qt, om_all, eta,
                gate_rho=None, min_phase: float = 0.25, use_vel: bool = True,
                rcond: float = 1e-6, w=None,
                lam=None, gamma=None, s_all=None, P_jac=None,
                kappa: float = 1.0, nu: float = 1.0, c_w: float = LOG10,
                eps_rel: float = 1e-2, eps_abs: float = 1e-12,
                khi_min_cycles: float = 1.0, tier: str = "C", gate_rho_m=None,
                norm_metric: bool = True):
    """Op-M on one gap (superset of oph_readout).
    First 11 arguments exactly as oph_readout.  Additional arguments:
      lam [Ma] | None   per-anchor soft precisions Lambda (INPUT anchor order).  None -> the
                        soft block is SKIPPED ENTIRELY by delegating to fq_oph.oph_readout
                        (identical code path, bit-identical).
      gamma             metric-law dial (scalar tensor; None -> metric law off).  gamma = 0
                        leaves the metric bit-for-bit unchanged.
      s_all [M], P_jac [M,R+1]   variety coordinate / Jacobi basis of the FULL module for the
                        law's band smoothing (kept rows are selected inside).
      tier "C"|"O"      exterior-channel gate: C -> g_M = g; O -> g_M from the
                        exterior energy ratio via gate_rho_m.
      norm_metric       per-gap max-normalization of the metric (the operator's gauge fix);
                        False only for the gauge selftest."""
    dt = ay.dtype
    # ---- the measured-spectrum channel: replaces w under the gamma dial ----
    if (gamma is not None) and (w is not None):
        w = _metric_hat(at, ay, om_all, w, gamma, s_all, P_jac, min_phase,
                        kappa, nu, c_w, eps_rel, eps_abs, khi_min_cycles)
    # ---- Lambda = 0: the degenerate case routes through the IDENTICAL oph_readout path ----
    if lam is None:
        return oph_readout(at, ay, av, qt, om_all, eta, gate_rho=gate_rho,
                           min_phase=min_phase, use_vel=use_vel, rcond=rcond, w=w)

    # ---- full Op-M path: mirrors oph_readout step-for-step, then adds the soft block ----
    at_ = at.to(dt)
    qt_ = qt.to(dt)
    idx = torch.argsort(at_)
    ts, ys, vs = at_[idx], ay[idx], av[idx]
    lams = lam.to(dt)[idx]
    gapi = int(torch.argmax(ts[1:] - ts[:-1]))
    a, b = ts[gapi], ts[gapi + 1]
    pa, pb = ys[gapi], ys[gapi + 1]
    va, vb = vs[gapi].to(dt), vs[gapi + 1].to(dt)
    ba = (b - a).clamp_min(1e-9)
    slope = (pb - pa) / ba
    A_q = pa.unsqueeze(0) + (qt_ - a).unsqueeze(-1) * slope.unsqueeze(0)      # chord [Q,2]

    om = om_all.to(dt)
    keep = om.abs() * ba > min_phase
    if int(keep.sum()) < 2:
        return A_q
    omk = om[keep]
    etak = eta.to(dt)[keep]
    if w is not None:
        wk = w.to(dt)[keep].clamp_min(1e-6)
        if norm_metric:
            wk = wk / wk.max()                         # the gauge fix, as in oph_readout
        w2 = torch.cat([wk, wk])
    else:
        w2 = None

    t0 = 0.5 * (a + b)
    tau_a, tau_b = a - t0, b - t0
    tau_q = qt_ - t0

    def val_row(tau):
        return torch.cat([torch.cos(omk * tau), torch.sin(omk * tau)])

    def der_row(tau):
        return torch.cat([-omk * torch.sin(omk * tau), omk * torch.cos(omk * tau)])

    zero2 = torch.zeros(2, device=ay.device, dtype=dt)
    rows = [val_row(tau_a), val_row(tau_b)]
    rhs = [zero2, zero2]
    om_ref = omk.abs().mean().clamp_min(1e-9)
    vel_mismatch = []
    if use_vel:
        for v_end, tau_end in ((va, tau_a), (vb, tau_b)):
            if bool((v_end != 0).any()):               # not the zero sentinel
                rows.append(der_row(tau_end) / om_ref)
                rhs.append((v_end - slope) / om_ref)
                vel_mismatch.append((v_end - slope).norm())
    B = torch.stack(rows)                              # [R,2K]
    d = torch.stack(rhs)                               # [R,2]
    if w2 is not None:
        B = B * w2.unsqueeze(0)                        # B~ = B W (u = W^-1 c coordinates)
    Bp = _pinv_safe(B, rcond=rcond)             # [2K,R]
    c_part = Bp @ d                                    # u0 [2K,2]

    if gate_rho is not None and vel_mismatch:
        dbar = torch.stack(vel_mismatch).sum() / (len(vel_mismatch) * slope.norm() + 1e-6)
        g = torch.tanh(torch.nn.functional.softplus(gate_rho).to(dt) * dbar)
    else:
        g = torch.ones((), device=ay.device, dtype=dt)

    z = torch.cat([etak[:, 0, :], etak[:, 1, :]], dim=0)   # [2K,2]
    c_free = z - Bp @ (B @ z)                          # P eta, operator form

    # ---- the soft exterior multi-anchor block (the (Op-M-sol) closed form) ----
    Ma = ts.shape[0]
    ext = torch.ones(Ma, dtype=torch.bool, device=ts.device)
    ext[gapi] = False
    ext[gapi + 1] = False
    y_soft = None
    if int(ext.sum()) > 0:
        rho_t, rho, tau_all, _, _, _ = _detrend_sorted(ts, ys, gapi, kappa)
        if tier == "C":
            g_M = g                                    # Tier C: the curvature gate
        else:                                          # Tier O: exterior-observable gate
            e_num = (rho_t[ext] ** 2).sum()
            e_bar = e_num / ((rho[ext] ** 2).sum() + 1e-12)
            if gate_rho_m is not None:
                g_M = torch.tanh(torch.nn.functional.softplus(gate_rho_m).to(dt) * e_bar)
            else:
                g_M = torch.ones((), device=ts.device, dtype=dt)
        lam_eff = g_M * lams[ext].clamp_min(0.0)       # Lambda_eff = g_M * Lambda
        tau_e = tau_all[ext]                           # [No]
        V = torch.cat([torch.cos(tau_e.unsqueeze(-1) * omk),
                       torch.sin(tau_e.unsqueeze(-1) * omk)], dim=-1)         # [No,2K]
        if w2 is not None:
            V = V * w2.unsqueeze(0)                    # V~ = V W
        u_base = c_part + g * c_free
        r0 = rho_t[ext] - V @ u_base                   # [No,2]  (Lambda-independent)
        sl = _sqrt_nz(lam_eff).unsqueeze(-1)           # Lambda^1/2  [No,1] (zero-safe sqrt grad)
        VT = V.t()                                     # [2K,No]
        PVT = VT - Bp @ (B @ VT)                       # P V~^T, operator form (never 2K x 2K)
        G = V @ PVT                                    # [No,No] PSD
        Gt = sl * G * sl.t()
        Gt = 0.5 * (Gt + Gt.t())                       # symmetrize (fp hygiene; SPD always)
        eyeN = torch.eye(Gt.shape[0], device=Gt.device, dtype=torch.float64)
        Lch = torch.linalg.cholesky(eyeN + Gt.double())          # no rcond (I + G~ >= I); fp64
        X = torch.cholesky_solve((sl * r0).double(), Lch).to(dt)  # (I+G~)^-1 Lam^1/2 r0 [No,2]
        y_soft = PVT @ (sl * X)                        # [2K,2] in null(B~) -> B c = d survives

    coef = c_part + g * c_free + (y_soft if y_soft is not None else 0.0)
    if w2 is not None:
        coef = w2.unsqueeze(-1) * coef                 # c = W u

    Cq = torch.cat([torch.cos(tau_q.unsqueeze(-1) * omk), torch.sin(tau_q.unsqueeze(-1) * omk)], dim=-1)
    return A_q + Cq @ coef                             # [Q,2]


# =========================================================================== batched path
# FQ_OPM_BATCHED=1 (fqdeep_native.forward): the per-gap loop over opm_readout is
# launch-overhead-bound (~200 tiny CUDA kernels + ~10 host syncs PER GAP).  The functions below
# execute the SAME per-gap mathematics for a whole batch of gaps at once: per-gap systems are
# padded (anchors to max Ma, boundary rows to 4, frequencies kept at the full module M with
# masking) and every linear-algebra step is ONE batched torch.linalg call (pinv / cholesky /
# cholesky_solve).  Semantic-identity rules (the batched path reproduces the loop per gap):
#   * padded ANCHOR slots carry the gap's last real time (finite -- no NaN through cos/sin) and
#     weight 0 in every reduction; padded BOUNDARY rows are zero rows with zero rhs (min-norm
#     solution of the augmented system is unchanged); UNKEPT frequencies keep zero columns in
#     B/V (pinv gives them zero rows; the final w2 mask zeroes their coefficients -- exactly the
#     columns the loop never builds).  Padded EXTERIOR anchors get lambda = 0 exactly, so
#     I + G~ is identity on those rows/cols and the fp64 Cholesky factors block-exactly.
#   * masked medians via nanmedian (same lower-middle convention as torch.median).
#   * precision rules as in the loop: fp32 everywhere, fp64 ONLY for the N_o x N_o
#     Cholesky of I + G~, pinv rcond 1e-6 on B~ only, all other pinvs at their default rtol.
# The loop path is the default (FQ_OPM_BATCHED unset/0).


def pad_sort_gaps(at_list, ay_list, aux_list=None, lam_list=None):
    """Pad a list of per-gap anchor systems to [G, Mx] and time-sort each gap.
    Returns (ts, ys, aux, lam, smask, gapi, Ma): all SORTED per gap; padded slots (smask False)
    carry the gap's LAST real anchor time (finite) and zeros elsewhere; gapi = index of the gap
    segment among the real sorted anchors (same argmax-of-diffs rule as the loop)."""
    dt = ay_list[0].dtype
    dev = ay_list[0].device
    Ma = torch.tensor([int(t.numel()) for t in at_list], device=dev)
    ts0 = pad_sequence([t.to(dt) for t in at_list], batch_first=True,
                       padding_value=float("inf"))                        # pads sort last
    idx = ts0.argsort(dim=1)
    Mx = ts0.shape[1]
    smask = torch.arange(Mx, device=dev).unsqueeze(0) < Ma.unsqueeze(1)
    ts = torch.take_along_dim(ts0, idx, 1)
    ys = torch.take_along_dim(pad_sequence(ay_list, batch_first=True),
                              idx.unsqueeze(-1).expand(-1, -1, 2), 1)
    t_last = torch.take_along_dim(ts, (Ma - 1).unsqueeze(1), 1)
    ts = torch.where(smask, ts, t_last)                                   # finite pad times
    aux = None
    if aux_list is not None:
        F_ = aux_list[0].shape[-1]
        aux = torch.take_along_dim(pad_sequence(aux_list, batch_first=True),
                                   idx.unsqueeze(-1).expand(-1, -1, F_), 1)
    lam = None
    if lam_list is not None:
        lam = torch.take_along_dim(pad_sequence([l.to(dt) for l in lam_list],
                                                batch_first=True), idx, 1)
    dts = ts[:, 1:] - ts[:, :-1]
    jj = torch.arange(Mx - 1, device=dev).unsqueeze(0)
    dts = torch.where(jj < (Ma - 1).unsqueeze(1), dts, torch.full_like(dts, float("-inf")))
    gapi = dts.argmax(dim=1)
    return ts, ys, aux, lam, smask, gapi, Ma


def dmu_velocity_batched(ts, ys, smask, gapi, Ma):
    """SORTED-domain batched replica of fq_opk.boundary_velocity: same-side finite differences,
    two-sided average where both sides exist, exact-zero sentinel otherwise -- identical
    elementwise arithmetic per real anchor, detached like fq_opk.boundary_velocity."""
    with torch.no_grad():
        G, Mx = ts.shape
        dev = ts.device
        seg = (ys[:, 1:] - ys[:, :-1]) / (ts[:, 1:] - ts[:, :-1]).clamp_min(1e-9).unsqueeze(-1)
        z2 = torch.zeros(G, 1, 2, device=dev, dtype=ys.dtype)
        segL = torch.cat([z2, seg], dim=1)                                # segL[:, j] = seg[:, j-1]
        segR = torch.cat([seg, z2], dim=1)                                # segR[:, j] = seg[:, j]
        j = torch.arange(Mx, device=dev).unsqueeze(0)
        gi = gapi.unsqueeze(1)
        hasL = smask & (j > 0) & (j - 1 != gi)
        hasR = smask & (j < (Ma - 1).unsqueeze(1)) & (j != gi)
        v = torch.where((hasL & hasR).unsqueeze(-1), 0.5 * (segL + segR),
                        torch.where(hasL.unsqueeze(-1), segL,
                                    torch.where(hasR.unsqueeze(-1), segR,
                                                torch.zeros_like(segL))))
        return v


def _detrend_batched(ts, ys, smask, gapi, kappa: float = 1.0):
    """Batched _detrend_sorted over padded sorted gaps: identical per-gap math.  A side with
    <= 2 real anchors detrends to exactly 0 (the O1 edge); padded rows stay 0."""
    dev, dt = ts.device, ys.dtype
    G, Mx = ts.shape
    gi = gapi.unsqueeze(1)
    a = torch.take_along_dim(ts, gi, 1).squeeze(1)
    b = torch.take_along_dim(ts, gi + 1, 1).squeeze(1)
    T = (b - a).clamp_min(1e-9)
    t0 = 0.5 * (a + b)
    tau = ts - t0.unsqueeze(1)
    gi2 = gi.unsqueeze(-1).expand(-1, -1, 2)
    pa = torch.take_along_dim(ys, gi2, 1).squeeze(1)
    pb = torch.take_along_dim(ys, gi2 + 1, 1).squeeze(1)
    slope = (pb - pa) / T.unsqueeze(-1)
    rho = ys - (pa.unsqueeze(1) + (ts - a.unsqueeze(1)).unsqueeze(-1) * slope.unsqueeze(1))
    varpi = torch.exp(-tau.abs() / (kappa * T).unsqueeze(1))
    ar = torch.arange(Mx, device=dev).unsqueeze(0)
    Xd = torch.stack([torch.ones_like(tau), tau], dim=-1)                 # [G,Mx,2]
    rho_t = torch.zeros_like(rho)
    for m in ((ar <= gi) & smask, (ar > gi) & smask):
        n = m.sum(1)
        Wd = (varpi * m.to(dt)).unsqueeze(-1)                             # [G,Mx,1]
        A2 = Xd.transpose(1, 2) @ (Wd * Xd)                               # [G,2,2]
        r2 = Xd.transpose(1, 2) @ (Wd * rho)                              # [G,2,2]
        sol = _pinv_safe(A2) @ r2                                  # rank-safe weighted LS
        fit = rho - Xd @ sol
        use = (n > 2).view(G, 1, 1) & m.unsqueeze(-1)
        rho_t = torch.where(use, fit, rho_t)
    return rho_t, rho, tau, varpi, T, slope


def _opm_metric_batched(ts, ys, smask, gapi, Ma, om, keep, log_wp, gamma, s_all, P_jac,
                        kappa, nu, c_w, eps_rel, eps_abs, khi_min_cycles):
    """Batched opm_metric core over DETACHED sorted inputs.  Returns (corr [G,M], gamma_eff)
    with corr already khi-masked and finite-guarded, so the caller applies the geometric law
    w_hat = clamp(w) * exp(gamma * corr) on the kept rows (bit-exact w at gamma = 0)."""
    dt, dev = ys.dtype, ys.device
    G, Mx = ts.shape
    M = om.shape[0]
    rho_t, rho, tau, varpi, T, _ = _detrend_batched(ts, ys, smask, gapi, kappa)
    om_abs = om.abs()
    o11 = om.view(1, M, 1)
    ang = o11 * tau.unsqueeze(1)                                          # [G,M,Mx]
    vm = varpi * smask.to(dt)
    ws = vm.sum(1).clamp_min(1e-12).view(G, 1, 1)
    cosang, sinang = torch.cos(ang), torch.sin(ang)
    Re = (cosang * vm.unsqueeze(1)) @ rho_t / ws                          # [G,M,2]
    Im = (-sinang * vm.unsqueeze(1)) @ rho_t / ws
    I_val = (Re * Re + Im * Im).sum(-1)                                   # [G,M]

    gi = gapi.unsqueeze(1)
    ar = torch.arange(Mx, device=dev).unsqueeze(0)
    I_inc = torch.zeros_like(I_val)
    for m in ((ar <= gi) & smask, (ar > gi) & smask):
        vsd = varpi * m.to(dt)
        wsd = vsd.sum(1).clamp_min(1e-12).view(G, 1, 1)
        ReS = (cosang * vsd.unsqueeze(1)) @ rho_t / wsd
        ImS = (-sinang * vsd.unsqueeze(1)) @ rho_t / wsd
        I_inc = I_inc + (ReS * ReS + ImS * ImS).sum(-1)
    I_inc = 0.5 * I_inc

    # (O2) increment channel; invalid pairs (padding / the gap segment) carry weight 0 AND a
    # zeroed secant, so they contribute exact zeros
    jj = torch.arange(Mx - 1, device=dev).unsqueeze(0)
    pv = ((jj < (Ma - 1).unsqueeze(1)) & (jj != gi)).to(dt)               # [G,P]
    dt_p = (tau[:, 1:] - tau[:, :-1]).clamp_min(1e-9)
    dlt = (rho_t[:, 1:] - rho_t[:, :-1]) / dt_p.unsqueeze(-1) * pv.unsqueeze(-1)
    tmid = 0.5 * (tau[:, 1:] + tau[:, :-1])
    wmid = torch.exp(-tmid.abs() / (kappa * T).unsqueeze(1)) * pv
    gk = 2.0 * torch.sin(o11 * dt_p.unsqueeze(1) / 2.0) / dt_p.unsqueeze(1)
    sgn = torch.where(gk >= 0, torch.ones_like(gk), -torch.ones_like(gk))
    # extra clamp_min(1e-30) only bites at |om| < pi*1e-30 -- always UNKEPT rows, so the loop
    # parity on kept rows is unaffected; it keeps the unkept rows finite instead of inf
    gk = sgn * torch.maximum(gk.abs(), o11.abs() / PI).clamp_min(1e-30)
    contrib = dlt.unsqueeze(1) / gk.unsqueeze(-1)                         # [G,M,P,2]
    angm = o11 * tmid.unsqueeze(1)
    wsm = wmid.sum(1).clamp_min(1e-12).view(G, 1, 1)
    ReI = ((torch.cos(angm) * wmid.unsqueeze(1)).unsqueeze(-1) * contrib).sum(2) / wsm
    ImI = ((-torch.sin(angm) * wmid.unsqueeze(1)).unsqueeze(-1) * contrib).sum(2) / wsm
    I_der = (ReI * ReI + ImI * ImI).sum(-1)

    # (O4/O5) robust split + floors; masked medians via nanmedian (lower-middle, = torch.median)
    I_tot = (I_val + nu * I_der) / (1.0 + nu)
    I_rob = torch.maximum(I_tot, I_inc)
    nanv = torch.full_like(I_rob, float("nan"))
    omx = om_abs.unsqueeze(0).expand(G, M)
    om_med = torch.where(keep, omx, nanv).nanmedian(1).values             # [G]
    hi = keep & (omx > om_med.unsqueeze(1))
    hi = torch.where(hi.any(1, keepdim=True), hi, keep)
    flr = torch.where(hi, I_rob, nanv).nanmedian(1).values / math.log(2.0)
    med_all = torch.where(keep, I_rob, nanv).nanmedian(1).values
    eps_t = torch.tensor(eps_abs, device=dev, dtype=dt)
    fl2 = torch.maximum(eps_rel * med_all, eps_t)                         # [G]
    I_raw = torch.maximum(I_rob - flr.unsqueeze(1), fl2.unsqueeze(1))
    logI_raw = torch.log(I_raw)

    # K_hi over the realized aperture
    t_last = torch.take_along_dim(ts, (Ma - 1).unsqueeze(1), 1).squeeze(1)
    span = t_last - ts[:, 0]
    ap = torch.minimum(span, 2.0 * kappa * T).clamp_min(1e-9)
    khi = keep & (omx * ap.unsqueeze(1) >= 2.0 * PI * khi_min_cycles)

    # Jacobi LS smoothing on the khi rows.  The khi-restricted basis can be severely
    # ill-conditioned (measured cond up to ~4e4 on real gaps), so an fp32 pinv of a
    # zero-row-padded batched matrix does NOT reproduce the loop's compact pinv to gate
    # tolerance (eps * cond amplification).  The tiny [K_hi, J] pinv is therefore computed
    # COMPACTLY PER GAP on the bitwise-identical row selection the loop sees (one host sync
    # for the khi masks; ~4 small kernels per gap -- everything else stays batched).
    if P_jac is None:                                  # fallback basis (parity with opm_metric)
        if s_all is None:
            om_kmax = torch.where(keep, omx, torch.zeros_like(omx)).max(1).values.clamp_min(1e-12)
            s_b = omx / om_kmax.unsqueeze(1)                              # [G,M]
        else:
            s_b = s_all.detach().to(dt).unsqueeze(0).expand(G, M)
        Pk = torch.stack([s_b ** r for r in range(5)], dim=-1)            # [G,M,5]
    else:
        Pk = P_jac.detach().to(dt).unsqueeze(0).expand(G, -1, -1)         # [G,M,J]
    J = Pk.shape[-1]
    khf = khi.to(dt)
    hat = isinstance(gamma, str)
    if hat:
        assert gamma == "hat", gamma
        Iv = torch.maximum(I_val - flr.unsqueeze(1), fl2.unsqueeze(1))
        Id = torch.maximum(I_der - flr.unsqueeze(1), fl2.unsqueeze(1))
        lIv, lId = torch.log(Iv), torch.log(Id)
    khi_cpu = khi.cpu()                                # the one host sync of the batched path
    theta = torch.zeros(G, J, device=dev, dtype=dt)
    th_v = torch.zeros(G, J, device=dev, dtype=dt) if hat else None
    th_d = torch.zeros(G, J, device=dev, dtype=dt) if hat else None
    for gdx in range(G):
        rows = torch.nonzero(khi_cpu[gdx]).squeeze(1)
        if rows.numel() == 0:
            continue                                   # corr is khi-masked to 0 anyway
        rows = rows.to(dev)
        Pp_g = _pinv_safe(Pk[gdx][rows])        # SAME compact pinv as the loop
        theta[gdx] = Pp_g @ logI_raw[gdx, rows]
        if hat:
            th_v[gdx] = Pp_g @ lIv[gdx, rows]
            th_d[gdx] = Pp_g @ lId[gdx, rows]
    logI_sm = torch.matmul(Pk, theta.unsqueeze(-1)).squeeze(-1)           # [G,M]

    # diagnostic gamma-hat (per gap), reached only via gamma == "hat"
    if hat:
        zr = torch.zeros_like(logI_raw)
        hv = torch.matmul(Pk, th_v.unsqueeze(-1)).squeeze(-1)
        hd = torch.matmul(Pk, th_d.unsqueeze(-1)).squeeze(-1)
        nhi = khf.sum(1)
        vhat = torch.where(khi, (0.25 * (hv - hd)) ** 2, zr).sum(1) / nhi.clamp_min(1.0)
        mhat = torch.where(khi, (0.5 * logI_sm - log_wp) ** 2, zr).sum(1) / nhi.clamp_min(1.0)
        gamma = torch.where(nhi > 0,
                            torch.clamp(1.0 - vhat / mhat.clamp_min(1e-12), 0.0, 1.0),
                            torch.zeros_like(vhat))                       # [G]

    corr = torch.clamp(0.5 * logI_sm - log_wp, -c_w, c_w) * khf
    corr = torch.where(torch.isfinite(corr), corr, torch.zeros_like(corr))
    return corr, gamma


def opm_readout_batched(ts, ys, vs, smask, gapi, Ma, qt, qgap, om_all, eta_all,
                        gate_rho=None, min_phase: float = 0.25, use_vel: bool = True,
                        rcond: float = 1e-6, w_all=None, lam_s=None, gamma=None,
                        s_all=None, P_jac=None,
                        kappa: float = 1.0, nu: float = 1.0, c_w: float = LOG10,
                        eps_rel: float = 1e-2, eps_abs: float = 1e-12,
                        khi_min_cycles: float = 1.0, tier: str = "C", gate_rho_m=None,
                        norm_metric: bool = True):
    """Batched Op-M over a padded gap batch (see the block comment above).
    ts/ys/vs [G,Mx(,2)] sorted+padded (pad_sort_gaps + dmu_velocity_batched/blend), smask
    [G,Mx], gapi/Ma [G]; qt [sumQ] query times with qgap [sumQ] gap index per query;
    eta_all [G,M,2,2]; w_all [G,M] or None; lam_s [G,Mx] SORTED per-anchor precisions or None
    (None = the oph delegation path, exactly as opm_readout's lam=None).  gamma: None | scalar
    | "hat".  Returns pred [sumQ,2] -- per-gap identical to looping opm_readout."""
    dt, dev = ys.dtype, ys.device
    G, Mx = ts.shape
    om = om_all.to(dt)
    M = om.shape[0]
    gi = gapi.unsqueeze(1)
    gi2 = gi.unsqueeze(-1).expand(-1, -1, 2)
    a = torch.take_along_dim(ts, gi, 1).squeeze(1)
    b = torch.take_along_dim(ts, gi + 1, 1).squeeze(1)
    pa = torch.take_along_dim(ys, gi2, 1).squeeze(1)
    pb = torch.take_along_dim(ys, gi2 + 1, 1).squeeze(1)
    va = torch.take_along_dim(vs, gi2, 1).squeeze(1).to(dt)
    vb = torch.take_along_dim(vs, gi2 + 1, 1).squeeze(1).to(dt)
    ba = (b - a).clamp_min(1e-9)
    slope = (pb - pa) / ba.unsqueeze(-1)
    t0 = 0.5 * (a + b)

    qt_ = qt.to(dt)
    tau_q = qt_ - t0[qgap]
    A_q = pa[qgap] + (qt_ - a[qgap]).unsqueeze(-1) * slope[qgap]          # chord [sumQ,2]

    keep = om.abs().unsqueeze(0) * ba.unsqueeze(1) > min_phase            # [G,M]
    kcnt = keep.sum(1)
    ok = kcnt >= 2                                     # degenerate module -> affine fallback
    kf = keep.to(dt)

    # ---- the measured-spectrum channel ----
    if (gamma is not None) and (w_all is not None):
        w_dt = w_all.to(dt)
        wcl = w_dt.clamp_min(1e-6)
        corr, gam_eff = _opm_metric_batched(ts.detach(), ys.detach(), smask, gapi, Ma,
                                            om.detach(), keep, torch.log(wcl), gamma,
                                            s_all, P_jac, kappa, nu, c_w, eps_rel, eps_abs,
                                            khi_min_cycles)
        gam_b = gam_eff.unsqueeze(1) if (torch.is_tensor(gam_eff) and gam_eff.dim() == 1) \
            else gam_eff
        w_used = torch.where(keep & ok.unsqueeze(1), wcl * torch.exp(gam_b * corr), w_dt)
    else:
        w_used = None if w_all is None else w_all.to(dt)

    # ---- spectral metric w2 (unkept columns exactly 0; kept entries bit-equal to the loop) ----
    if w_used is not None:
        wc = w_used.clamp_min(1e-6)
        if norm_metric or lam_s is None:               # the oph delegation always normalizes
            wmax = torch.where(keep, wc, torch.zeros_like(wc)).max(1).values
            wmax = torch.where(kcnt > 0, wmax, torch.ones_like(wmax))
            wc = wc / wmax.unsqueeze(1)
        w2 = torch.cat([wc, wc], dim=1) * torch.cat([kf, kf], dim=1)      # [G,2M]
    else:
        w2 = torch.cat([kf, kf], dim=1)

    # ---- boundary-Hermite system, 4 padded rows (absent velocity rows = zero row/rhs) ----
    tau_a = (a - t0).unsqueeze(1)
    tau_b = (b - t0).unsqueeze(1)
    o1 = om.unsqueeze(0)                                                  # [1,M]
    ca, sa = torch.cos(o1 * tau_a), torch.sin(o1 * tau_a)
    cb, sb = torch.cos(o1 * tau_b), torch.sin(o1 * tau_b)
    om_ref = ((om.abs().unsqueeze(0) * kf).sum(1) / kcnt.to(dt).clamp_min(1.0)).clamp_min(1e-9)
    orf = om_ref.unsqueeze(1)
    if use_vel:
        mva = (va != 0).any(-1)                        # not the zero sentinel
        mvb = (vb != 0).any(-1)
    else:
        mva = torch.zeros(G, dtype=torch.bool, device=dev)
        mvb = torch.zeros(G, dtype=torch.bool, device=dev)
    fva, fvb = mva.to(dt).unsqueeze(1), mvb.to(dt).unsqueeze(1)
    B = torch.stack([torch.cat([ca, sa], 1),
                     torch.cat([cb, sb], 1),
                     torch.cat([-o1 * sa, o1 * ca], 1) / orf * fva,
                     torch.cat([-o1 * sb, o1 * cb], 1) / orf * fvb], dim=1)   # [G,4,2M]
    B = B * w2.unsqueeze(1)                            # B~ = B W (u = W^-1 c coordinates)
    z2g = torch.zeros(G, 2, device=dev, dtype=dt)
    d = torch.stack([z2g, z2g,
                     (va - slope) / om_ref.unsqueeze(-1) * fva,
                     (vb - slope) / om_ref.unsqueeze(-1) * fvb], dim=1)       # [G,4,2]
    Bp = _pinv_safe(B, rcond=rcond)                                    # [G,2M,4]
    c_part = Bp @ d                                                           # u0 [G,2M,2]

    # ---- curvature gate over the included velocity rows ----
    if gate_rho is not None:
        nv = fva.squeeze(1) + fvb.squeeze(1)
        dbar = ((va - slope).norm(dim=-1) * fva.squeeze(1) +
                (vb - slope).norm(dim=-1) * fvb.squeeze(1)) / (nv * slope.norm(dim=-1) + 1e-6)
        g = torch.where(nv > 0,
                        torch.tanh(torch.nn.functional.softplus(gate_rho).to(dt) * dbar),
                        torch.ones_like(dbar))
    else:
        g = torch.ones(G, device=dev, dtype=dt)

    z = torch.cat([eta_all.to(dt)[:, :, 0, :], eta_all.to(dt)[:, :, 1, :]], dim=1)  # [G,2M,2]
    c_free = z - Bp @ (B @ z)                          # P eta, operator form
    u_base = c_part + g.view(G, 1, 1) * c_free

    # ---- the soft exterior block ((Op-M-sol)); non-exterior/padded rows get lambda = 0 ----
    y_soft = None
    if lam_s is not None:
        rho_t, rho, tau_all, _, _, _ = _detrend_batched(ts, ys, smask, gapi, kappa)
        ar = torch.arange(Mx, device=dev).unsqueeze(0)
        ext = smask & (ar != gi) & (ar != gi + 1)
        if tier == "C":
            g_M = g                                    # Tier C: the curvature gate
        else:                                          # Tier O: exterior-observable gate
            ef = ext.to(dt)
            e_num = ((rho_t ** 2).sum(-1) * ef).sum(1)
            e_bar = e_num / (((rho ** 2).sum(-1) * ef).sum(1) + 1e-12)
            if gate_rho_m is not None:
                g_M = torch.tanh(torch.nn.functional.softplus(gate_rho_m).to(dt) * e_bar)
            else:
                g_M = torch.ones(G, device=dev, dtype=dt)
        lam_e = torch.where(ext, lam_s.to(dt).clamp_min(0.0),
                            torch.zeros_like(lam_s.to(dt)))
        sl = _sqrt_nz(g_M.unsqueeze(1) * lam_e)                               # [G,Mx] (zero-safe)
        sl1 = sl.unsqueeze(-1)
        Vv = torch.cat([torch.cos(tau_all.unsqueeze(-1) * om.view(1, 1, M)),
                        torch.sin(tau_all.unsqueeze(-1) * om.view(1, 1, M))], dim=-1)
        Vv = Vv * w2.unsqueeze(1)                      # V~ = V W  [G,Mx,2M]
        r0 = rho_t - Vv @ u_base                       # [G,Mx,2] (Lambda-independent)
        VT = Vv.transpose(1, 2)
        PVT = VT - Bp @ (B @ VT)                       # P V~^T, operator form
        Gm = Vv @ PVT                                  # [G,Mx,Mx] PSD
        Gt = (sl1 * Gm) * sl.unsqueeze(1)
        Gt = 0.5 * (Gt + Gt.transpose(1, 2))           # symmetrize (fp hygiene; SPD always)
        eyeN = torch.eye(Mx, device=dev, dtype=torch.float64).unsqueeze(0)
        Lch = torch.linalg.cholesky(eyeN + Gt.double())          # fp64 ONLY here
        X = torch.cholesky_solve((sl1 * r0).double(), Lch).to(dt)
        y_soft = PVT @ (sl1 * X)                       # [G,2M,2] in null(B~)

    coef = u_base + y_soft if y_soft is not None else u_base
    coef = coef * w2.unsqueeze(-1)                     # c = W u; unkept coefficients -> 0

    Cq = torch.cat([torch.cos(tau_q.unsqueeze(-1) * o1),
                    torch.sin(tau_q.unsqueeze(-1) * o1)], dim=-1)             # [sumQ,2M]
    fqs = (Cq.unsqueeze(1) @ coef[qgap]).squeeze(1)                           # [sumQ,2]
    return torch.where(ok[qgap].unsqueeze(-1), A_q + fqs, A_q)


# ----------------------------------------------------------------------------- self-test
def _selftest():
    torch.manual_seed(0)
    import torch.nn.functional as F
    from .fq_opk import boundary_velocity

    M = 8
    t = torch.sort(torch.rand(M)).values
    t[M // 2:] += 2.0                                   # open a gap in the middle
    a, b = t[M // 2 - 1], t[M // 2]
    qt = a + (b - a) * torch.rand(40)
    om = 2 * PI * 2.5 * (torch.rand(48) * 2 - 1) * 3.0
    eta_big = 3.0 * torch.randn(48, 2, 2)
    rho = torch.tensor(math.log(math.expm1(4.0)))
    w_rand = F.softplus(torch.randn(48)) + 1e-3
    s_all = om.abs() / om.abs().max()
    P_jac = torch.stack([s_all ** r for r in range(5)], dim=-1)   # 5-dim smooth basis
    lam_r = F.softplus(torch.randn(M))                  # random positive per-anchor precisions
    gam_r = torch.tensor(0.7)
    gam_0 = torch.tensor(0.0)

    p0, v0 = torch.randn(2), torch.randn(2)
    y_line = p0 + t.unsqueeze(-1) * v0
    truth = p0 + qt.unsqueeze(-1) * v0
    y_c = y_line + 0.4 * torch.sin(3 * t).unsqueeze(-1) * torch.tensor([1.0, -1.0])
    av_l = boundary_velocity(t, y_line)
    av_c = boundary_velocity(t, y_c)

    # (M-R1) degeneration: lam=None delegates through the IDENTICAL oph_readout path
    p_oph = oph_readout(t, y_c, av_c, qt, om, eta_big, rho, w=w_rand)
    p_del = opm_readout(t, y_c, av_c, qt, om, eta_big, rho, w=w_rand,
                        lam=None, gamma=gam_0, s_all=s_all, P_jac=P_jac)
    e_r1 = (p_del - p_oph).abs().max().item()
    print(f"[Op-M M-R1a] lam=None, gamma=0: BIT-parity vs oph_readout: max|D| = {e_r1:.3e} ",
          "PASS" if e_r1 == 0.0 else "FAIL")
    p_del2 = opm_readout(t, y_c, av_c, qt, om, eta_big, rho, w=w_rand)   # both dials absent
    e_r1b = (p_del2 - p_oph).abs().max().item()
    print(f"[Op-M M-R1b] lam=None, gamma=None: max|D| = {e_r1b:.3e} ",
          "PASS" if e_r1b == 0.0 else "FAIL")
    p_z = opm_readout(t, y_c, av_c, qt, om, eta_big, rho, w=w_rand,
                      lam=torch.zeros(M), gamma=gam_0, s_all=s_all, P_jac=P_jac)
    e_r1c = (p_z - p_oph).abs().max().item()
    print(f"[Op-M M-R1c] lam=zeros (full path): fp-parity: max|D| = {e_r1c:.3e} ",
          "PASS" if e_r1c < 1e-5 else "FAIL")

    # (M2) curved data, random (gamma, Lambda, eta, w): exact boundary interpolation + velocity
    tq = torch.tensor([float(a), float(b)], requires_grad=True)
    pred_ab = opm_readout(t, y_c, av_c, tq, om, eta_big, rho, w=w_rand,
                          lam=lam_r, gamma=gam_r, s_all=s_all, P_jac=P_jac)
    e2 = (pred_ab - torch.stack([y_c[M // 2 - 1], y_c[M // 2]])).abs().max().item()
    print(f"[Op-M M2a] CURVED, random (gamma,Lambda,eta,w): boundary interp err = {e2:.3e} ",
          "PASS" if e2 < 1e-3 else "FAIL")
    gx = torch.autograd.grad(pred_ab[0, 0], tq, retain_graph=True)[0][0]
    gy = torch.autograd.grad(pred_ab[0, 1], tq, retain_graph=True)[0][0]
    va_true = av_c[torch.argsort(t)][M // 2 - 1]
    e2v = (torch.stack([gx, gy]) - va_true).abs().max().item()
    print(f"[Op-M M2b] CURVED, random dials: boundary VELOCITY match err = {e2v:.3e} ",
          "PASS" if e2v < 1e-2 else "FAIL")

    # (M3i) globally straight data: exact chord for ANY (gamma, Lambda, eta, w), both tiers
    for tier, gm in (("C", None), ("O", rho)):
        pred = opm_readout(t, y_line, av_l, qt, om, eta_big, rho, w=w_rand,
                           lam=lam_r, gamma=gam_r, s_all=s_all, P_jac=P_jac,
                           tier=tier, gate_rho_m=gm)
        e3 = (pred - truth).abs().max().item()
        print(f"[Op-M M3i-{tier}] STRAIGHT, random dials, tier {tier}: chord err = {e3:.3e} ",
              "PASS" if e3 < 5e-4 else "FAIL")

    # (M3i sentinel) straight data, ONE-anchor side (zero-sentinel velocity), random Lambda
    t1 = torch.cat([t[:1], t[M // 2:]])
    y1 = p0 + t1.unsqueeze(-1) * v0
    av1 = boundary_velocity(t1, y1)
    assert bool((av1[0] == 0).all())
    pred1 = opm_readout(t1, y1, av1, qt, om, eta_big, rho, w=w_rand,
                        lam=F.softplus(torch.randn(5)), gamma=gam_r, s_all=s_all, P_jac=P_jac)
    e3b = (pred1 - truth).abs().max().item()
    print(f"[Op-M M3i-sent] STRAIGHT, one-anchor side (sentinel dropped): err = {e3b:.3e} ",
          "PASS" if e3b < 5e-4 else "FAIL")

    # (M3ii) locally-affine jet, CURVED exterior, Tier C: forced to the chord (the blind-spot trade)
    y_loc = y_line.clone()
    y_loc[0] += torch.tensor([0.7, -0.4]); y_loc[1] += torch.tensor([-0.5, 0.3])
    y_loc[6] += torch.tensor([0.6, 0.5]);  y_loc[7] += torch.tensor([-0.3, -0.6])
    av_loc = boundary_velocity(t, y_loc)
    pred_loc = opm_readout(t, y_loc, av_loc, qt, om, eta_big, rho, w=w_rand,
                           lam=100.0 * lam_r, gamma=gam_0, s_all=s_all, P_jac=P_jac, tier="C")
    e3c = (pred_loc - truth).abs().max().item()
    print(f"[Op-M M3ii] locally-affine jet + curved exterior, Tier C: chord err = {e3c:.3e} ",
          "PASS" if e3c < 5e-4 else "FAIL")

    # (M-end) endpoint annihilation of the boundary-conditioned kernel K~
    dtt = y_c.dtype
    idx = torch.argsort(t)
    ts, ys, vs = t[idx], y_c[idx], av_c[idx]
    gapi = int(torch.argmax(ts[1:] - ts[:-1]))
    aa, bb = ts[gapi], ts[gapi + 1]
    ba = (bb - aa).clamp_min(1e-9)
    keepm = om.abs() * ba > 0.25
    omk = om[keepm]
    wk = w_rand[keepm].clamp_min(1e-6); wk = wk / wk.max()
    w2 = torch.cat([wk, wk])
    t0m = 0.5 * (aa + bb)

    def vrow(tau):
        return torch.cat([torch.cos(omk * tau), torch.sin(omk * tau)])

    def drow(tau):
        return torch.cat([-omk * torch.sin(omk * tau), omk * torch.cos(omk * tau)])

    om_ref = omk.abs().mean()
    slope_m = (ys[gapi + 1] - ys[gapi]) / ba
    rowsB = [vrow(aa - t0m), vrow(bb - t0m)]
    for v_end, tau_end in ((vs[gapi], aa - t0m), (vs[gapi + 1], bb - t0m)):
        if bool((v_end != 0).any()):
            rowsB.append(drow(tau_end) / om_ref)
    Bm = torch.stack(rowsB) * w2.unsqueeze(0)
    Bpm = _pinv_safe(Bm, rcond=1e-6)

    def Pop(u):
        return u - Bpm @ (Bm @ u)

    tau_ext = (ts - t0m)[torch.tensor([j for j in range(M) if j not in (gapi, gapi + 1)])]
    Pu_ext = torch.stack([Pop(w2 * vrow(te)) for te in tau_ext], dim=-1)     # [2K,No]
    u_a, u_b = w2 * vrow(aa - t0m), w2 * vrow(bb - t0m)
    da, db = w2 * drow(aa - t0m), w2 * drow(bb - t0m)
    tmid = 0.5 * (aa + bb) - t0m
    ref = (w2 * vrow(tmid)) @ Pu_ext                                          # mid-gap K~ row
    scale = ref.abs().max().clamp_min(1e-12)
    kv = max(float((u_a @ Pu_ext).abs().max() / scale), float((u_b @ Pu_ext).abs().max() / scale))
    kd = max(float((da @ Pu_ext).abs().max() / (scale * om_ref)),
             float((db @ Pu_ext).abs().max() / (scale * om_ref)))
    print(f"[Op-M M-end] K~(a,.)=K~(b,.)=0: rel = {kv:.3e} | slope rows: rel = {kd:.3e} ",
          "PASS" if kv < 1e-4 and kd < 1e-4 else "FAIL")
    # operational form: the exterior-channel deformation vanishes (value + slope) at a and b
    tq2 = torch.tensor([float(a), float(b)], requires_grad=True)
    p_on = opm_readout(t, y_c, av_c, tq2, om, eta_big, rho, w=w_rand,
                       lam=lam_r, gamma=gam_0, s_all=s_all, P_jac=P_jac)
    p_off = opm_readout(t, y_c, av_c, tq2, om, eta_big, rho, w=w_rand,
                        lam=torch.zeros(M), gamma=gam_0, s_all=s_all, P_jac=P_jac)
    dfm = (p_on - p_off)
    e6 = dfm.abs().max().item()
    g6 = torch.autograd.grad(dfm.abs().sum(), tq2)[0].abs().max().item()
    print(f"[Op-M M-end2] deformation at endpoints: value = {e6:.3e} slope = {g6:.3e} ",
          "PASS" if e6 < 1e-4 and g6 < 1e-2 else "FAIL")

    # (M-lim) lam -> 1e6: exterior anchors interpolated (phi(tau_j)^T c ~= rho~_j)
    rho_t_in, tau_in, _, _ = chord_residual_features(t, y_c)
    text = torch.tensor([float(t[j]) for j in range(M) if j not in (M // 2 - 1, M // 2)])
    pred_ext = opm_readout(t, y_c, av_c, text, om, eta_big, w=w_rand,
                           lam=torch.full((M,), 1e6), gamma=gam_0, s_all=s_all, P_jac=P_jac,
                           gate_rho=None)               # gate off -> Lambda_eff = Lambda exactly
    A_ext = torch.stack([y_c[M // 2 - 1] + (te - a) * (y_c[M // 2] - y_c[M // 2 - 1]) / (b - a)
                         for te in text])
    tgt = A_ext + torch.stack([rho_t_in[j] for j in range(M) if j not in (M // 2 - 1, M // 2)])
    rel = float((pred_ext - tgt).norm() / tgt.sub(A_ext).norm().clamp_min(1e-9))
    print(f"[Op-M M-lim] lam=1e6 hard-interpolation limit: rel misfit = {rel:.3e} ",
          "PASS" if rel < 5e-3 else "FAIL")

    # (M-gauge) (alpha w, Lambda/alpha^2, eta/alpha) invariance of the RAW operator,
    # and alpha-invariance of the gauge-fixed (normalized) operator
    al = 3.0
    pg1 = opm_readout(t, y_c, av_c, qt, om, eta_big, rho, w=w_rand,
                      lam=lam_r, norm_metric=False)
    pg2 = opm_readout(t, y_c, av_c, qt, om, eta_big / al, rho, w=al * w_rand,
                      lam=lam_r / al ** 2, norm_metric=False)
    eg = (pg1 - pg2).abs().max().item()
    print(f"[Op-M M-gauge] (aw, L/a^2, eta/a) invariance (raw operator): max|D| = {eg:.3e} ",
          "PASS" if eg < 1e-3 else "FAIL")
    pn1 = opm_readout(t, y_c, av_c, qt, om, eta_big, rho, w=w_rand, lam=lam_r)
    pn7 = opm_readout(t, y_c, av_c, qt, om, eta_big, rho, w=7.0 * w_rand, lam=lam_r)
    egn = (pn1 - pn7).abs().max().item()
    print(f"[Op-M M-gauge2] gauge-fixed: c(7w, L, eta) = c(w, L, eta): max|D| = {egn:.3e} ",
          "PASS" if egn < 1e-3 else "FAIL")

    # (M-floor) two-anchor sides -> rho~ == 0 exactly: finite metric, finite prediction
    t4 = torch.tensor([0.1, 0.35, 2.5, 2.8])
    y4 = torch.randn(4, 2)
    av4 = boundary_velocity(t4, y4)
    p4 = opm_readout(t4, y4, av4, qt, om, eta_big, rho, w=w_rand,
                     lam=F.softplus(torch.randn(4)), gamma=gam_r, s_all=s_all, P_jac=P_jac)
    ok4 = bool(torch.isfinite(p4).all())
    w4 = _metric_hat(t4, y4, om, w_rand, gam_r, s_all, P_jac, 0.25, 1.0, 1.0, LOG10, 1e-2, 1e-12, 1.0)
    ok4 = ok4 and bool(torch.isfinite(w4).all()) and bool((w4 > 0).all())
    print(f"[Op-M M-floor] two-anchor sides (rho~==0): finite metric + prediction: ",
          "PASS" if ok4 else "FAIL")

    # (M-grad) gradients to every dial; omega DETACHED inside the estimator
    om_g = om.clone().requires_grad_(True)
    w_g = w_rand.clone().requires_grad_(True)
    et_g = eta_big.clone().requires_grad_(True)
    rh_g = rho.clone().requires_grad_(True)
    lm_g = lam_r.clone().requires_grad_(True)
    gm_g = gam_r.clone().requires_grad_(True)
    ay_g = y_c.clone().requires_grad_(True)
    opm_readout(t, ay_g, av_c, qt, om_g, et_g, rh_g, w=w_g,
                lam=lm_g, gamma=gm_g, s_all=s_all, P_jac=P_jac).sum().backward()
    leaves = dict(omega=om_g, w=w_g, eta=et_g, rho=rh_g, lam=lm_g, gamma=gm_g, ay=ay_g)
    okg = all(p.grad is not None and torch.isfinite(p.grad).all() for p in leaves.values())
    oknz = all(float(p.grad.abs().sum()) > 0 for p in leaves.values())
    print(f"[Op-M M-grad] finite gradients to (omega, w, eta, rho, lam, gamma, ay): ",
          "PASS" if okg and oknz else "FAIL")
    om_leaf = om[om.abs() * (b - a) > 0.25].detach().requires_grad_(True)
    wk_l = w_rand[om.abs() * (b - a) > 0.25].clamp_min(1e-6)
    gm_leaf = gam_r.clone().requires_grad_(True)       # live graph through gamma only
    wh = opm_metric(t, y_c, a, b, om_leaf, None, None, torch.log(wk_l), gm_leaf, w_prod=wk_l)
    gdet = torch.autograd.grad(wh.sum(), (om_leaf, gm_leaf), allow_unused=True)
    print(f"[Op-M M-grad2] omega-detach inside the estimator (grad None, gamma grad live): ",
          "PASS" if gdet[0] is None and gdet[1] is not None else "FAIL")

    # ---- measured lambda modes and the cell-constant gamma law ----
    # gamma' cell-constant law: exact value + both clamp endpoints +
    # the mbar -> 0 guard
    gp_ok = (abs(gamma_prime(0.1, 1.0, 0.0) - 0.9) < 1e-12
             and gamma_prime(2.0, 1.0, 0.0) == 0.0 and gamma_prime(0.0, 1.0, 5.0) == 0.0
             and gamma_prime(0.0, 1.0, 0.0) == 1.0 and gamma_prime(0.05, 0.0, 0.0) == 0.0)
    print(f"[Op-M gamma'] cell-constant law value/clamp/guard: ", "PASS" if gp_ok else "FAIL")

    # c2 closed-form lambda-hat: finiteness, h_hat in [0,1], lam*sig2 == h_hat
    # exactly, the plug-in-scale cap lam <= h_hat/s2flr
    lam_c2, hh_c2, sig2_c2, s2flr_c2 = opm_c2_lambda(t, y_c, om, return_parts=True)
    ok_c2 = (bool(torch.isfinite(lam_c2).all()) and bool((lam_c2 >= 0).all())
             and 0.0 <= float(hh_c2) <= 1.0
             and float((lam_c2 * sig2_c2 - hh_c2).abs().max()) < 1e-5
             and bool((lam_c2 <= hh_c2 / s2flr_c2 + 1e-6).all()))
    print(f"[Op-M C2a] lambda-hat finite/range/cap (h_hat = {float(hh_c2):.3f}): ",
          "PASS" if ok_c2 else "FAIL")
    # straight data: rho~ == 0 exactly -> zero trust, lambda-hat == 0 identically
    lam_line = opm_c2_lambda(t, y_line, om)
    print(f"[Op-M C2b] straight data -> lambda-hat == 0: max = {float(lam_line.max()):.1e} ",
          "PASS" if float(lam_line.max()) == 0.0 else "FAIL")
    # exactness composition: boundary interpolation (B c = d) survives lam = lambda-hat (any Lambda)
    tqc = torch.tensor([float(a), float(b)])
    pc2 = opm_readout(t, y_c, av_c, tqc, om, eta_big, rho, w=w_rand,
                      lam=lam_c2, gamma=gam_r, s_all=s_all, P_jac=P_jac)
    ec2 = (pc2 - torch.stack([y_c[M // 2 - 1], y_c[M // 2]])).abs().max().item()
    print(f"[Op-M C2c] boundary interp exact under lambda-hat: err = {ec2:.3e} ",
          "PASS" if ec2 < 1e-3 else "FAIL")
    # the rho~ == 0 exact edge (two-anchor sides): finite and exactly zero
    lam4 = opm_c2_lambda(t4, y4, om)
    ok4c2 = bool(torch.isfinite(lam4).all()) and float(lam4.max()) == 0.0
    print(f"[Op-M C2d] two-anchor sides (rho~==0 edge): finite + zero: ",
          "PASS" if ok4c2 else "FAIL")
    # sigma^2_eps: floor positivity + the v (max) semantics
    s2x, s2f = _sigma2_eps(torch.zeros(5, 2))
    ok_s2 = bool((s2x == s2f).all()) and float(s2f) > 0.0
    rt_c, _, _, _ = chord_residual_features(t, y_c)
    s2y, s2fy = _sigma2_eps(rt_c)
    ok_s2 = ok_s2 and bool((s2y >= s2fy).all()) and bool((s2y >= (rt_c ** 2).sum(-1)).all())
    print(f"[Op-M C1a] sigma^2_eps floor/max semantics: ", "PASS" if ok_s2 else "FAIL")
    # numerical-validity ceiling: tiny-residual data (sigma^2 underflow) stays lambda-solvable
    y_tiny = y_line + 1e-6 * torch.randn_like(y_line)
    lam_t = opm_c2_lambda(t, y_tiny, om)
    ok_cap = bool((lam_t <= LAM_NUM_MAX).all()) and bool(torch.isfinite(lam_t).all())
    p_t = opm_readout(t, y_tiny, boundary_velocity(t, y_tiny), qt, om, eta_big, rho, w=w_rand,
                      lam=torch.full((M,), LAM_NUM_MAX), gamma=gam_0, s_all=s_all, P_jac=P_jac)
    ok_cap = ok_cap and bool(torch.isfinite(p_t).all())
    print(f"[Op-M CAPa] LAM_NUM_MAX ceiling: lambda-hat <= {LAM_NUM_MAX:.0e}, SPD-solvable: ",
          "PASS" if ok_cap else "FAIL")


if __name__ == "__main__":
    _selftest()

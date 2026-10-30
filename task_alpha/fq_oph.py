"""task_alpha.fq_oph -- Op-H: the FQ-NATIVE boundary-Hermite readout.

An alternative to the Op-K-Hermite kriging readout (fq_opk.py): a readout whose OUTPUT IS ITSELF an
FQ exponential sum.  The reconstruction lives in

    p_hat(t) = A(t) + r(t),      r(t) = sum_k [ alpha_k cos(omega_k tau) + beta_k sin(omega_k tau) ],

with tau = t - (a+b)/2 and frequencies omega_k = 2 pi s <xi_k, e_t> PRESCRIBED by the learned Lee-Yang
module Lambda' = {L^T k} (its projection onto the time axis) -- i.e. r is a real-valued aperiodic
exponential sum ON the FQ spectrum, not a kriging smoother that merely uses an FQ kernel.

Construction:

  * AFFINE ATOM: the pure-point FQ spectrum spans constants (delta_0) but NOT linear functions; the
    affine functional needs delta'_0, which lies IN THE SAME CLOSED CLASS (Lev-Olevskii).  The
    hard-wired chord A(t) through the two bracketing anchors IS that {delta_0, delta'_0} atom --
    adjoined as a separate atom ("decompose, don't truncate"), never a Vaaler/de Branges sum
    truncated to two nodes.
  * BOUNDARY-HERMITE SYSTEM, applied to the residual: value rows
    r(a) = r(b) = 0 and OPTIONAL Dmu velocity rows r'(a) = v_a - A', r'(b) = v_b - A' -- the confluent
    exponential-Vandermonde in real cos/sin form.  A velocity row is included only when the observed
    boundary velocity exists (fq_opk.boundary_velocity returns the exact-zero sentinel when a side has
    no same-side segment; imposing that sentinel as an exact constraint would bend the path to zero
    velocity, so sentinel rows are DROPPED -- the derivative rows are optional constraints).
  * PRESCRIBED FREQUENCIES: no frequency estimation -- ONE differentiable linear
    solve.  c_part = B^+ d is the min-norm solution of the underdetermined system (pinv with rcond;
    derivative rows equilibrated by 1/mean|omega| -- row scaling leaves the solution SET, hence the
    min-norm solution and the null space, unchanged; it only improves conditioning).
  * NULL-SPACE MANEUVER: c = c_part + g * (I - B^+ B) eta; the network
    predicts ONLY the free coordinates eta, UNSCALED.  (eta is NOT
    rescaled by the Lee-Yang |c_k|: such a rescaling is not part of the construction, and with a
    median |c_k| of about 0.005 it would suppress the learned branch.  The learned variety already
    enters through the frequencies themselves -- they ARE the module -- and through the message
    passing that produces eta.)
    B c = d holds for EVERY eta (the null-space projector identity B(I - B^+B) = 0 is
    unconditional; the PARTICULAR part needs d in range(B) <=> B full row rank -- generic for
    K >= 2 distinct kept frequencies -- and holds modulo the pinv rcond threshold: these are the two
    caveats) => boundary interpolation and the Hermite velocity match are exact under those
    caveats.  task_alpha/_diag_oph_conditioning.py measures these margins on trained checkpoints
    (effective rank sigma_min/sigma_max against rcond = 1e-6 for the weighted and unweighted system,
    the realized boundary residual in fp32, and an fp64-QR re-solve against fp64-pinv).  Run it on
    any newly trained checkpoint: the frequencies and the metric are learned, so the margins depend
    on the checkpoint.
  * min_phase MASK: excludes k = 0 and near-DC modes (the FQ branch is mean/slope-free).
    SCOPE: this mask does NOT enforce pairwise frequency separation (no separation is
    enforced), and it is NOT what protects the affine baseline -- the boundary constraints do that
    (any residual-branch function vanishes at both endpoints, so it cannot be affine except 0).  The
    mask only removes wasted near-affine columns.  It is a hard threshold on learnable omega: modes
    crossing it enter/leave the basis discretely (values far from the threshold dominate in practice).
  * CONDITIONING: we do not solve a square exponential-Vandermonde (at K = 2
    the system happens to be 4x4, handled by the same rcond-pinv); the affine part is factored out
    FIRST.  An exponential-B-spline realization is deliberately NOT used:
    its reproduction filter is proven on the integer grid only.  The Riesz-basis
    condition for the learned spectrum is likewise open -- mitigated numerically,
    not claimed.
  * CURVATURE GATE g = tanh(softplus(rho) * dbar): an implementation-level, leak-safe PROXY (not part
    of the mathematical construction) for the property "the FQ sum only switches on for genuine
    curvature", which defining c_k as projections of the residual would give directly but which is
    not computable leak-safely at inference.
    dbar = relative boundary-velocity mismatch over the INCLUDED velocity rows.  BLIND
    SPOT, stated plainly: a maneuver whose boundary tangents parallel the chord (a parallel-tangent
    S-curve) gives dbar = 0 and is suppressed to the chord -- but such a gap is unidentifiable from
    two-sided boundary observations by ANY operator, so the gate loses nothing that the data supplies.
    With the gate ON and at least one velocity row, straight data gives d = 0 AND g = 0, so the output
    is the EXACT chord for ANY eta (E1).  With the gate off or no velocity rows, chord reproduction
    holds at eta = 0 only.

  * WEIGHTED SPECTRAL METRIC (the GNN channel): the solve minimizes ||W^-1 c||^2 s.t. B c = d
    with W = diag(w_k) a per-frequency metric (cos_k and sin_k share w_k -- a phase-free spectral
    profile).  Substituting u = W^-1 c: c = W (B W)^+ d, free part c_free = W (I - (BW)^+(BW)) eta.
    B c = d holds IDENTICALLY for any w > 0 and any eta, so ALL exactness theorems survive the
    learned metric.  c_part is 0-homogeneous in w and the eta term is 1-HOMOGENEOUS, so the per-gap
    max-normalization (w -> w / max w) is PART OF THE OPERATOR DEFINITION (it fixes the eta-channel
    scale); the normalized operator satisfies c(alpha*w) = c(w) (selftest E2c).  This is the de Branges projection taken in a learned H(E)-type weighted
    metric (the c_k are DEFINED by a weighted inner product); it is also exactly the
    mechanism through which the GNN provably contributes in Op-K (kriging with kernel weights w_k IS
    weighted min-norm interpolation with spectral metric sum |a_k|^2 / w_k).  w = const recovers the
    unweighted operator.

Exactness properties (verified in _selftest):
  (E1)  straight data (any anchor layout with >= 1 usable velocity row) -> exact chord for ANY eta
        and ANY metric w.
  (E1b) straight data with a ONE-ANCHOR side (zero-sentinel velocity)  -> exact chord for ANY eta/w.
  (E2)  any data, any eta, ANY metric w -> exact boundary interpolation AND exact velocity match
        (under the full-row-rank + rcond caveats stated above).
  (E2c) metric scale invariance: c(alpha*w) = c(w).
  (E3)  gradients flow to the FQ spectrum (xi, s via omega), eta, rho, AND the metric w.
"""
from __future__ import annotations

import math

import torch

PI = math.pi


def oph_readout(at: torch.Tensor, ay: torch.Tensor, av: torch.Tensor, qt: torch.Tensor,
                om_all: torch.Tensor, eta: torch.Tensor,
                gate_rho: torch.Tensor | None = None, min_phase: float = 0.25,
                use_vel: bool = True, rcond: float = 1e-6,
                w: torch.Tensor | None = None):
    """Op-H on one gap.
    at [Ma] anchor times, ay [Ma,2] anchor positions, av [Ma,2] observed (Dmu) velocities
    (exact-zero row = boundary_velocity's no-same-side-segment sentinel), qt [Q] held/query times,
    om_all [M] angular frequencies 2 pi s xi_t (PRESCRIBED, learnable),
    eta [M,2,2] learned null-space coordinates (freq x {cos,sin} x {x,y}), UNSCALED free coordinates,
    gate_rho scalar parameter (None -> gate off, g = 1),
    w [M] optional POSITIVE per-frequency spectral metric (None -> uniform, the unweighted operator).
    Returns pred [Q,2]."""
    dt = ay.dtype
    at = at.to(dt); qt = qt.to(dt)

    # ---- the affine atom A(t): chord through the two BRACKETING anchors ----
    idx = torch.argsort(at)
    ts, ys, vs = at[idx], ay[idx], av[idx]
    gapi = int(torch.argmax(ts[1:] - ts[:-1]))                 # the gap = largest anchor time jump
    a, b = ts[gapi], ts[gapi + 1]
    pa, pb = ys[gapi], ys[gapi + 1]
    va, vb = vs[gapi].to(dt), vs[gapi + 1].to(dt)
    ba = (b - a).clamp_min(1e-9)
    slope = (pb - pa) / ba                                     # A'(t)  [2]
    A_q = pa.unsqueeze(0) + (qt - a).unsqueeze(-1) * slope.unsqueeze(0)   # chord at query [Q,2]

    # ---- frequency subset: k = 0 / near-DC excluded (mean/slope-free FQ branch) ----
    om = om_all.to(dt)
    keep = om.abs() * ba > min_phase
    if int(keep.sum()) < 2:                                    # degenerate module: affine atom only
        return A_q
    omk = om[keep]                                             # [K]
    etak = eta.to(dt)[keep]                                    # [K,2,2]
    if w is not None:
        wk = w.to(dt)[keep].clamp_min(1e-6)
        wk = wk / wk.max()                 # part of the operator: fixes the eta-channel scale (c_part
                                           # is 0-homogeneous in w, the eta term 1-homogeneous)
        w2 = torch.cat([wk, wk])                               # [2K] cos/sin share the metric
    else:
        w2 = None

    # ---- centered time (pure phase rotation of the coefficients; conditioning only) ----
    t0 = 0.5 * (a + b)
    tau_a, tau_b = a - t0, b - t0
    tau_q = qt - t0

    # ---- the boundary-Hermite system on the residual: B coef = d ----
    def val_row(tau):
        return torch.cat([torch.cos(omk * tau), torch.sin(omk * tau)])            # [2K]

    def der_row(tau):
        return torch.cat([-omk * torch.sin(omk * tau), omk * torch.cos(omk * tau)])

    zero2 = torch.zeros(2, device=ay.device, dtype=dt)
    rows = [val_row(tau_a), val_row(tau_b)]
    rhs = [zero2, zero2]
    # velocity (Dmu) rows: only where an observed boundary velocity exists (sentinel rows dropped);
    # rows equilibrated by 1/mean|omega| (solution set invariant -- conditioning only).
    om_ref = omk.abs().mean().clamp_min(1e-9)
    vel_mismatch = []
    if use_vel:
        for v_end, tau_end in ((va, tau_a), (vb, tau_b)):
            if bool((v_end != 0).any()):                       # not the zero sentinel
                rows.append(der_row(tau_end) / om_ref)
                rhs.append((v_end - slope) / om_ref)
                vel_mismatch.append((v_end - slope).norm())
    B = torch.stack(rows)                                      # [R, 2K], R in {2,3,4}
    d = torch.stack(rhs)                                       # [R, 2]

    if w2 is not None:
        B = B * w2.unsqueeze(0)                                # solve in u = W^-1 c coordinates
    Bp = torch.linalg.pinv(B, rcond=rcond)                     # [2K, R]
    c_part = Bp @ d                                            # [2K,2] (u-space if weighted)

    # ---- curvature gate: implementation-level proxy for "switches on for genuine curvature";
    #      dbar over INCLUDED velocity rows only.  No velocity rows -> no straight-data evidence -> g=1.
    if gate_rho is not None and vel_mismatch:
        dbar = torch.stack(vel_mismatch).sum() / (len(vel_mismatch) * slope.norm() + 1e-6)
        g = torch.tanh(torch.nn.functional.softplus(gate_rho).to(dt) * dbar)
    else:
        g = torch.ones((), device=ay.device, dtype=dt)

    # ---- learned null-space maneuver:  c = c_part + g (I - B^+ B) eta ----
    z = torch.cat([etak[:, 0, :], etak[:, 1, :]], dim=0)       # [2K,2] free coordinates, unscaled
    c_free = z - Bp @ (B @ z)                                  # (I - B^+ B) z, basis-free projector
    coef = c_part + g * c_free                                 # B coef = d for EVERY eta (u-space)
    if w2 is not None:
        coef = w2.unsqueeze(-1) * coef                         # c = W u  (weighted min-norm solution)

    # ---- evaluate: p_hat = affine atom + FQ exponential sum ----
    Cq = torch.cat([torch.cos(tau_q.unsqueeze(-1) * omk), torch.sin(tau_q.unsqueeze(-1) * omk)], dim=-1)
    return A_q + Cq @ coef                                     # [Q,2]


# ----------------------------------------------------------------------------- self-test
def _selftest():
    torch.manual_seed(0)
    from .fq_opk import boundary_velocity

    M = 8
    t = torch.sort(torch.rand(M)).values
    t[M // 2:] += 2.0                                           # open a gap in the middle
    a, b = t[M // 2 - 1], t[M // 2]
    qt = a + (b - a) * torch.rand(40)
    om = 2 * PI * 2.5 * (torch.rand(48) * 2 - 1) * 3.0
    eta_big = 3.0 * torch.randn(48, 2, 2)                       # LARGE random eta: exactness must not care
    rho = torch.tensor(math.log(math.expm1(4.0)))

    # (E1) straight data -> exact chord for ANY eta
    p0, v0 = torch.randn(2), torch.randn(2)
    y = p0 + t.unsqueeze(-1) * v0
    av = boundary_velocity(t, y)
    pred = oph_readout(t, y, av, qt, om, eta_big, rho)
    truth = p0 + qt.unsqueeze(-1) * v0
    e1 = (pred - truth).abs().max().item()
    # tolerance 5e-4: the g=0 path returns the chord EXACTLY by construction; the residual vs the
    # analytic line is fp32 round-off of the chord arithmetic itself (relative ~4e-5 on O(3) values).
    print(f"[Op-H E1] STRAIGHT gap, LARGE random eta: max abs err = {e1:.3e} ",
          "PASS" if e1 < 5e-4 else "FAIL")

    # (E1b) straight data, ONE-ANCHOR left side -> zero-sentinel velocity row dropped -> exact chord
    t1 = torch.cat([t[:1], t[M // 2:]])                         # 1 anchor left, 4 right
    y1 = p0 + t1.unsqueeze(-1) * v0
    av1 = boundary_velocity(t1, y1)
    assert bool((av1[0] == 0).all()), "left anchor should carry the zero sentinel"
    pred1 = oph_readout(t1, y1, av1, qt, om, eta_big, rho)
    e1b = (pred1 - truth).abs().max().item()
    print(f"[Op-H E1b] STRAIGHT, one-anchor side (sentinel dropped): err = {e1b:.3e} ",
          "PASS" if e1b < 5e-4 else "FAIL")

    # (E2) curved data, random eta -> exact boundary interpolation + velocity match (autograd)
    y_c = y + 0.4 * torch.sin(3 * t).unsqueeze(-1) * torch.tensor([1.0, -1.0])
    av_c = boundary_velocity(t, y_c)
    tq = torch.tensor([float(a), float(b)], requires_grad=True)
    pred_ab = oph_readout(t, y_c, av_c, tq, om, eta_big, rho)
    e2 = (pred_ab - torch.stack([y_c[M // 2 - 1], y_c[M // 2]])).abs().max().item()
    print(f"[Op-H E2a] CURVED, random eta: boundary interpolation err = {e2:.3e} ",
          "PASS" if e2 < 1e-3 else "FAIL")
    gx = torch.autograd.grad(pred_ab[0, 0], tq, retain_graph=True)[0][0]
    gy = torch.autograd.grad(pred_ab[0, 1], tq, retain_graph=True)[0][0]
    va_true = av_c[torch.argsort(t)][M // 2 - 1]
    e2v = (torch.stack([gx, gy]) - va_true).abs().max().item()
    print(f"[Op-H E2b] CURVED, random eta: boundary VELOCITY match err = {e2v:.3e} ",
          "PASS" if e2v < 1e-2 else "FAIL")

    # (E3) gradients flow to omega, amplitudes, eta, rho
    om_g = om.clone().requires_grad_(True)
    et_g = eta_big.clone().requires_grad_(True); rh_g = rho.clone().requires_grad_(True)
    oph_readout(t, y_c, av_c, qt, om_g, et_g, rh_g).sum().backward()
    ok = all(p.grad is not None and torch.isfinite(p.grad).all()
             for p in (om_g, et_g, rh_g))
    print(f"[Op-H E3] gradients to (omega, eta, rho): ", "PASS" if ok else "FAIL")

    # degenerate module (all frequencies masked) -> affine fallback, finite
    pred_deg = oph_readout(t, y_c, av_c, qt, 1e-9 * om, eta_big, rho)
    print(f"[Op-H deg] near-DC module -> affine fallback finite: "
          f"{'PASS' if torch.isfinite(pred_deg).all() else 'FAIL'}")

    # WEIGHTED METRIC: all exactness must hold for ANY positive w
    w_rand = torch.nn.functional.softplus(torch.randn(48)) + 1e-3
    predw = oph_readout(t, y, av, qt, om, eta_big, rho, w=w_rand)
    e1w = (predw - truth).abs().max().item()
    print(f"[Op-H E1w] STRAIGHT, random METRIC w + random eta: err = {e1w:.3e} ",
          "PASS" if e1w < 5e-4 else "FAIL")
    pred_abw = oph_readout(t, y_c, av_c, tq.detach(), om, eta_big, rho, w=w_rand)
    e2w = (pred_abw - torch.stack([y_c[M // 2 - 1], y_c[M // 2]])).abs().max().item()
    print(f"[Op-H E2w] CURVED, random METRIC w: boundary interpolation err = {e2w:.3e} ",
          "PASS" if e2w < 1e-3 else "FAIL")
    p_w1 = oph_readout(t, y_c, av_c, qt, om, eta_big, rho, w=w_rand)
    p_w7 = oph_readout(t, y_c, av_c, qt, om, eta_big, rho, w=7.0 * w_rand)
    e2c = (p_w1 - p_w7).abs().max().item()
    print(f"[Op-H E2c] metric scale invariance c(7w)=c(w): err = {e2c:.3e} ",
          "PASS" if e2c < 1e-3 else "FAIL")
    w_g = w_rand.clone().requires_grad_(True)
    oph_readout(t, y_c, av_c, qt, om, eta_big, rho, w=w_g).sum().backward()
    okw = w_g.grad is not None and torch.isfinite(w_g.grad).all() and float(w_g.grad.abs().sum()) > 0
    print(f"[Op-H E3w] gradient to the metric w: ", "PASS" if okw else "FAIL")

    # duplicate-frequency robustness: coef finite, boundary still exact
    om_dup = om.clone(); om_dup[1] = om_dup[0]; om_dup[7] = om_dup[0]
    pred_dup = oph_readout(t, y_c, av_c, tq.detach(), om_dup, eta_big, rho)
    e_dup = (pred_dup - torch.stack([y_c[M // 2 - 1], y_c[M // 2]])).abs().max().item()
    print(f"[Op-H dup] duplicated frequencies: boundary err = {e_dup:.3e} ",
          "PASS" if e_dup < 1e-3 and torch.isfinite(pred_dup).all() else "FAIL")


if __name__ == "__main__":
    _selftest()

"""task_alpha.fq_opk  --  Op-K: a closed-form, AFFINE-REPRODUCING Fourier-Quasicrystal interpolation
operator (CPD / universal-kriging readout), an add-on primitive from the kriging literature.

Per gap, given two-sided anchor times t_i and positions y_i, solve the bordered saddle-point system

    [ A   P ] [ c ]   [ y ]            A_ij = K_FQ(t_i, t_j)   (FQ reproducing kernel, M x M)
    [ P^T 0 ] [ d ] = [ 0 ]            P_i  = [1, t_i]         (affine drift, M x 2)

and evaluate   s(t) = sum_i c_i K_FQ(t, t_i) + d0 + d1 t   at the held-out times.

K_FQ(s,t) = sum_k |c_k|^2 cos(2 pi s <coord_s - coord_t, xi_k>)  is PD by construction (positive weights),
its spectrum is the shared quasicrystal frequencies xi (so this REUSES Op-A/Op-D, not a foreign basis).

AFFINE REPRODUCTION is guaranteed by the saddle structure + the [1,t] drift, INDEPENDENT of the kernel:
on straight data the kernel coefficients c are exactly 0 and s(t) = the chord, so a straight gap has
no error floor BY CONSTRUCTION.  Differentiable: the solve has a standard implicit-function adjoint,
so gradients flow to xi, c, s (and through them to L).
"""
from __future__ import annotations

import math

import torch

PI = math.pi


def fq_pd_kernel(ca: torch.Tensor, cb: torch.Tensor, xi: torch.Tensor, cw: torch.Tensor,
                 s: torch.Tensor) -> torch.Tensor:
    """PD Fourier-quasicrystal kernel K[a,b] = sum_k |c_k|^2 cos(2 pi s <ca_a - cb_b, xi_k>).
    ca [Na, D], cb [Nb, D], xi [M, D], cw [M] complex amplitudes, s scalar.  -> [Na, Nb]."""
    w = (cw.abs() ** 2).to(ca.dtype)                       # [M] positive weights |c_k|^2
    # phase[a,b,k] = 2 pi s (ca_a - cb_b) . xi_k
    pa = (ca @ xi.t())                                     # [Na, M]
    pb = (cb @ xi.t())                                     # [Nb, M]
    phase = 2 * PI * s * (pa.unsqueeze(1) - pb.unsqueeze(0))   # [Na, Nb, M]
    return (torch.cos(phase) * w).sum(-1)                  # [Na, Nb]


def opk_interp(anchor_coord, anchor_t, anchor_y, query_coord, query_t, xi, cw, s,
               ridge: float = 0.1):
    """Solve Op-K on ONE gap and evaluate at the query (held) times.
    anchor_coord [M, D], anchor_t [M], anchor_y [M, 2], query_coord [Q, D], query_t [Q].
    `ridge` is RELATIVE to K(0)=Σ|c_k|^2 (the FQ Gram is severely ill-conditioned on aperiodic anchors --
    cond ~1e9-1e32 -- so Tikhonov regularization is REQUIRED for a stable solve; ridge>=0.05 keeps the
    straight-gap floor at ~GCD).  Returns pred [Q, 2].  Affine-reproducing."""
    M = anchor_coord.shape[0]
    dt = anchor_y.dtype
    K0 = (cw.abs() ** 2).sum().to(dt)                                  # = K_FQ(0); ridge scale
    A = fq_pd_kernel(anchor_coord, anchor_coord, xi, cw, s).to(dt)      # [M, M]
    A = A + (ridge * K0) * torch.eye(M, device=A.device, dtype=dt)     # relative Tikhonov regularization
    P = torch.stack([torch.ones_like(anchor_t), anchor_t], dim=-1).to(dt)   # [M, 2]
    # bordered saddle matrix  [[A, P], [P^T, 0]]  -> [M+2, M+2]
    top = torch.cat([A, P], dim=1)                                     # [M, M+2]
    bot = torch.cat([P.t(), torch.zeros(2, 2, device=A.device, dtype=dt)], dim=1)  # [2, M+2]
    Sys = torch.cat([top, bot], dim=0)                                 # [M+2, M+2]
    rhs = torch.cat([anchor_y, torch.zeros(2, 2, device=A.device, dtype=dt)], dim=0)  # [M+2, 2]
    sol = torch.linalg.solve(Sys, rhs)                                 # [M+2, 2]
    c, d = sol[:M], sol[M:]                                            # [M,2], [2,2]
    Kqa = fq_pd_kernel(query_coord, anchor_coord, xi, cw, s).to(dt)    # [Q, M]
    Pq = torch.stack([torch.ones_like(query_t), query_t], dim=-1).to(dt)   # [Q, 2]
    return Kqa @ c + Pq @ d                                            # [Q, 2]


def boundary_velocity(at: torch.Tensor, ay: torch.Tensor) -> torch.Tensor:
    """Observed per-anchor velocity dpos/dt from SAME-SIDE finite differences (never crossing the gap).
    at [M] times, ay [M,2] positions -> av [M,2].  Detached (a fixed target, like the positions)."""
    with torch.no_grad():
        idx = torch.argsort(at)
        t, y = at[idx], ay[idx]
        M = t.shape[0]
        seg = (y[1:] - y[:-1]) / (t[1:] - t[:-1]).clamp_min(1e-9).unsqueeze(-1)   # [M-1,2] segment slopes
        gapi = int(torch.argmax(t[1:] - t[:-1]))                                  # the gap segment index
        v = torch.zeros_like(y)
        for j in range(M):
            left = seg[j - 1] if (j > 0 and j - 1 != gapi) else None
            right = seg[j] if (j < M - 1 and j != gapi) else None
            if left is not None and right is not None:
                v[j] = 0.5 * (left + right)
            elif left is not None:
                v[j] = left
            elif right is not None:
                v[j] = right
        out = torch.zeros_like(v); out[idx] = v
        return out


def gap_curvature(at: torch.Tensor, ay: torch.Tensor, av: torch.Tensor) -> torch.Tensor:
    """Scalar gap curvature kappa-hat = turning angle between the entry and exit boundary tangents
    (the velocities at the two anchors bracketing the gap).  Built ONLY from boundary observations
    (leak-safe).  ~0 on a straight constant-heading leg; ~pi/2 on a 90-deg maneuver.  Differentiable."""
    idx = torch.argsort(at)
    gapi = int(torch.argmax((at[idx][1:] - at[idx][:-1])))           # gap = largest time jump
    t0 = av[idx][gapi]; t1 = av[idx][gapi + 1]                       # entry/exit velocity vectors
    n0 = t0 / t0.norm().clamp_min(1e-9); n1 = t1 / t1.norm().clamp_min(1e-9)
    return torch.arccos((n0 * n1).sum().clamp(-1.0 + 1e-6, 1.0 - 1e-6))


def opk_hermite(at, ay, av, qt, xi_t, cw, s, ridge: float = 0.1, ridge_v: float = 0.1,
                kernel_gain=1.0):
    """Dmu-HERMITE Op-K on 1-D time: kriging with BOTH position rows s(t_j)=y_j and VELOCITY rows
    s'(t_j)=v_j, so the interior follows the observed boundary heading (not the chord).
    at [M] anchor times, ay [M,2] positions, av [M,2] observed velocities, qt [Q] query times,
    xi_t [K] 1-D (time-axis) FQ frequencies, cw [K] complex amplitudes, s scale.  Returns [Q,2].
    K(s,t)=Σ w_k cos(ω_k(s-t)), ω=2π s ξ_t, w=|c|^2; ∂_2K=+Σ w ω sin, ∂_1∂_2K=+Σ w ω^2 cos.

    `kernel_gain` G (scalar in [0,1]) is the IN-OPERATOR FQ-AMPLITUDE GATE (PD-safe):
    it scales the FQ kernel K_FQ -> G*K_FQ everywhere (Gram + derivative blocks + prediction) while the
    ridge stays a FIXED floor, so G->0 leaves the operator's OWN affine trend (no external chord) and G=1
    is the full Op-K-H.  Affine reproduction (drift [1,t]) is preserved for ANY G (it multiplies c=e=0)."""
    dt = ay.dtype
    om = (2 * PI * s * xi_t).to(dt)                                   # [K] angular freqs
    w = (cw.abs() ** 2).to(dt)                                        # [K] positive weights
    M = at.shape[0]; at = at.to(dt); qt = qt.to(dt)

    def blocks(a, b):                                                 # a[Na], b[Nb] -> K00,K01,K11
        d = (a.unsqueeze(1) - b.unsqueeze(0)).unsqueeze(-1)           # [Na,Nb,1]
        cos = torch.cos(om * d); sin = torch.sin(om * d)
        return (cos * w).sum(-1), (sin * (w * om)).sum(-1), (cos * (w * om * om)).sum(-1)

    K00, K01, K11 = blocks(at, at)
    K0 = w.sum().to(dt); Kdd0 = (w * om * om).sum().to(dt)
    eyeM = torch.eye(M, device=at.device, dtype=dt)
    A00 = kernel_gain * K00 + ridge * K0 * eyeM                      # FQ kernel scaled by the gate; ridge = fixed floor
    A11 = kernel_gain * K11 + ridge_v * Kdd0 * eyeM
    one = torch.ones_like(at); zero = torch.zeros_like(at)
    P = torch.stack([one, at], -1)                                   # [M,2] drift values  (UNGATED affine trend)
    Pd = torch.stack([zero, one], -1)                               # [M,2] drift derivatives
    z2 = torch.zeros(2, 2, device=at.device, dtype=dt)
    # symmetric saddle system  [[A00, A01, P],[A01^T, A11, Pd],[P^T, Pd^T, 0]]  (A10 = A01^T)
    row0 = torch.cat([A00, kernel_gain * K01, P], dim=1)
    row1 = torch.cat([kernel_gain * K01.t(), A11, Pd], dim=1)
    row2 = torch.cat([P.t(), Pd.t(), z2], dim=1)
    Sys = torch.cat([row0, row1, row2], dim=0)                       # [2M+2, 2M+2]
    rhs = torch.cat([ay.to(dt), av.to(dt), torch.zeros(2, 2, device=at.device, dtype=dt)], dim=0)
    sol = torch.linalg.solve(Sys, rhs)
    c, e, d = sol[:M], sol[M:2 * M], sol[2 * M:]                     # [M,2],[M,2],[2,2]
    qK00, qK01, _ = blocks(qt, at)                                   # [Q,M] each
    Pq = torch.stack([torch.ones_like(qt), qt], -1)                 # [Q,2]
    affine = Pq @ d                                                 # the operator's OWN affine trend (UNGATED)
    kernel = kernel_gain * (qK00 @ c); vel = kernel_gain * (qK01 @ e)   # gated FQ kernel + Hermite velocity
    return affine + kernel + vel                                    # s(t)=G·ΣcK + G·Σe∂_2K + drift


# ----------------------------------------------------------------------------- self-test
def _selftest():
    torch.manual_seed(0)
    D, M, Q, Mfreq = 2, 8, 30, 24
    # STRAIGHT gap: anchors + queries on an affine path  y = a + b t   (2-D), monotone times
    t_anc = torch.sort(torch.rand(M)).values
    t_qry = torch.rand(Q)
    a = torch.randn(2); b = torch.randn(2)
    y_anc = a + t_anc.unsqueeze(-1) * b                                # [M,2] exactly affine
    y_qry_true = a + t_qry.unsqueeze(-1) * b
    # coords carry time in column 0 (+ a dummy 2nd channel); RANDOM (untrained) FQ spectrum
    ca = torch.stack([t_anc, torch.rand(M)], dim=-1)
    cq = torch.stack([t_qry, torch.rand(Q)], dim=-1)
    xi = torch.randn(Mfreq, D); cw = torch.randn(Mfreq, dtype=torch.cfloat); s = torch.tensor(1.7)
    pred = opk_interp(ca, t_anc, y_anc, cq, t_qry, xi, cw, s, ridge=0.0)
    err = (pred - y_qry_true).abs().max().item()
    print(f"[Op-K self-test] STRAIGHT gap, UNTRAINED random FQ spectrum: max abs error = {err:.3e}")
    print("  -> affine reproduction is", "EXACT (PASS)" if err < 1e-4 else f"BROKEN (FAIL: {err})")
    # Hermite Op-K self-test: STRAIGHT + observed velocity -> exact chord (affine+Hermite reproduction)
    xi_t = torch.randn(Mfreq)
    av = boundary_velocity(t_anc, y_anc)                              # observed velocity ~ slope b
    predH = opk_hermite(t_anc, y_anc, av, t_qry, xi_t, cw, torch.tensor(1.7), ridge=0.1, ridge_v=0.1)
    errH = (predH - y_qry_true).abs().max().item()
    print(f"[Op-K Hermite] STRAIGHT+obs-vel, UNTRAINED (ridge 0.1): max abs error = {errH:.3e}")
    print("  -> Hermite affine reproduction is", "OK (PASS)" if errH < 1e-3 else f"BROKEN (FAIL {errH})")
    cwH = cw.clone().requires_grad_(True)
    opk_hermite(t_anc, y_anc, av, t_qry, xi_t, cwH, torch.tensor(1.7)).sum().backward()
    print(f"[Op-K Hermite] grad to FQ amplitudes: {'flows (PASS)' if cwH.grad is not None and torch.isfinite(cwH.grad).all() else 'BROKEN'}")
    # CURVED gap: add a bump -> kernel coefficients must be non-zero (FQ correction active)
    y_curved = y_anc + 0.3 * torch.sin(5 * t_anc).unsqueeze(-1) * torch.tensor([1.0, -1.0])
    pred_c = opk_interp(ca, t_anc, y_curved, cq, t_qry, xi, cw, s, ridge=1e-6)
    print(f"[Op-K self-test] CURVED gap interpolates (finite): {torch.isfinite(pred_c).all().item()}")
    # differentiability: gradient flows to the FQ spectrum
    cw2 = cw.clone().requires_grad_(True)
    opk_interp(ca, t_anc, y_curved, cq, t_qry, xi, cw2, s, ridge=1e-6).sum().backward()
    print(f"[Op-K self-test] grad to FQ amplitudes c_k: {'flows (PASS)' if cw2.grad is not None and torch.isfinite(cw2.grad).all() else 'BROKEN'}")


if __name__ == "__main__":
    _selftest()

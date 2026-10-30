"""
fqops.op_curve  --  FQCurveBlock : the FAITHFUL Example-1.3 Lee-Yang curve unit.

This is the verified-faithful curve unit for the variety primitive (checked against AKKV,
Inventiones 2024).

Each FACTOR is one genuine separating-curve embedding (AKKV Prop 4.12 / Ex 4.13 / Ex 1.3):
  * n real degree-1 functions  g_k(t) = s_k * (a_k t + b_k),  a_k = softplus > 0  (positive leading
    coeff => separating, Ex 4.13),  s_k = (-1)^(k-1)  a FIXED sign-alternation buffer (Prop 4.12,
    mandatory for the strict-Lee-Yang  var(Im f) = n-1  condition).
  * Cayley map  z_k = (g_k + i)/(g_k - i)  (= AKKV's Mobius z -> (z+i)/(z-i)).
  * the curve is cut out by the n-1 STAR relations on z_1 (Ex 1.3 incidence: p_1 couples (z_1,z_2),
    p_2 couples (z_1,z_3), ...): for q = 2..n, eliminating t between g_1 and g_q gives the genuine
    bilinear (1,1) eliminant
        p_q(z) = Al_q z_1 z_q + Be_q z_1 + Ga_q z_q + De_q ,   (det := c_1 m_q - c_q m_1)
        Al = det + i(m_1 - m_q),  Be = -det + i(m_1 + m_q),
        Ga = -det - i(m_1 + m_q), De = det + i(-m_1 + m_q),     m_i = s_i a_i,  c_i = s_i b_i.
    (Verified: with a=(1,1,1), b=(-1,0,1), s=(+,-,+) this reproduces Ex 1.3's printed p_1 and 2*p_2.)
  * the lift L in R^{n x 2} (d = n-1 = 2 for the n=3 anchor) is the IDENTIFIABLE, widened (r,theta)
    positive-Grassmannian chart:  r_i = exp(rho_i);  theta_i = spread * S_i/(S_n + h)  with
    S_i = cumsum(softplus(gap_raw))_{<= i} (ABSOLUTE, no renormalize) and h = softplus(head_raw)
    (separate headroom), spread = 0.95*pi (fixed).  Then 0 < theta_1 < ... < theta_n < spread < pi,
    so every 2x2 minor r_i r_j sin(theta_j - theta_i) > 0 STRUCTURALLY (Gr_+(2,n)).  This chart has a
    full-rank param->L Jacobian (only the projective global-scale null remains) -- it is NOT gauge-
    frozen like a cumsum-normalize + RMS-renorm chart.

WIDTH is obtained by COMPOSING R independent curve factors (AKKV Ex 4.15) at the FEATURE level
(concatenate each factor's [Re p, Im p]) -- NOT by a single block-diagonal L (that lands in the
nonnegative Grassmannian with structural-zero minors) and NOT by inflating one n with a mismatched d.

HONEST SEMANTICS: forward() returns the relation RESIDUAL p(exp(2 pi i L x)) evaluated at the input
activation x -- a learned proxy for AKKV Remark 2.4's real-rooted trig system.  It is FQ-STRUCTURED
(genuine Lee-Yang curve relations on a genuine positive-Grassmannian lift) but does NOT solve the
quasicrystal Lambda(X,L); the point-set FQ guarantees live in spectrum.py / op_spectral.py.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.utils.checkpoint

PI = math.pi


def _softplus(x, floor: float = 1e-4):
    return torch.nn.functional.softplus(x) + floor


def _softplus_inv(y: float) -> float:
    """raw such that softplus(raw)+1e-4 ~ y (for initialization)."""
    y = max(y - 1e-4, 1e-6)
    return math.log(math.expm1(y)) if y < 20 else y


class FQCurveBlock(nn.Module):
    """
    Faithful multi-factor FQ curve transform mapping in_dim -> out_dim, vectorized over R factors.
    Compatible call signature with the other FQ blocks (in_dim -> out_dim) so the GNN steps can use it.

    n      : torus coordinates per factor (curve => d = n-1).  Anchor n=3 (Example 1.3), d=2.
    """

    SPREAD = 0.95 * PI                       # fixed angular spread < pi (Gr_+ positivity guarantee)

    def __init__(self, in_dim: int, out_dim: int, n: int = 3, init_scale: float = 1.0,
                 seed: int = 0, chunk: int | None = None, phase_offset: bool = False):
        super().__init__()
        if n != 3:
            raise ValueError(
                f"FQCurveBlock supports only n=3 (d=2). The well-conditioned (r,theta) lift is a d=2 "
                f"chart, and a separating curve needs d=n-1, so d=2 => n=3 (Example 1.3). Larger n "
                f"would need a d>2 positive-Grassmannian chart (ill-conditioned, the float32-underflow "
                f"pathology). For more capacity use MORE curve factors / width / depth (Ex 4.15 "
                f"products), not larger n. Got n={n}.")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n = n
        self.d = n - 1                                   # curve: dim X = 1 => d = n-1 (AKKV Thm 2.7)
        self.per_factor_out = 2 * (n - 1)                # [Re p, Im p] for the n-1 relations
        # enough factors to (a) cover the output width and (b) consume the input
        self.num_factors = max(math.ceil(out_dim / self.per_factor_out),
                               math.ceil(in_dim / self.d))
        self.padded_in = self.num_factors * self.d
        # take out_per_factor from EVERY factor (not first factors only) so all input regions get
        # gradient -- truncating reshape[:, :out_dim] severs gradient to later factors' inputs.
        self.out_per_factor = min(self.per_factor_out, math.ceil(out_dim / self.num_factors))
        self.init_scale = init_scale
        self.chunk = chunk
        R, nn_ = self.num_factors, n
        g = torch.Generator().manual_seed(seed)

        # fixed sign-alternation buffer s_k = (-1)^(k-1)  (Prop 4.12; non-learned)
        signs = torch.tensor([(-1.0) ** k for k in range(nn_)])
        self.register_buffer("signs", signs)

        # ---- curve params (define the separating curve X): g_k = s_k (a_k t + b_k) ----
        # anchor to Example 1.3: a=(1,..), b ~ (-1,0,1,...) pattern, with per-factor seed jitter
        b_anchor = torch.tensor([float(k - (nn_ - 1) / 2.0) for k in range(nn_)])   # ~(-1,0,1)
        # DIVERSE curve params per factor: each factor is a DIFFERENT separating curve (distinct a_k,b_k)
        # so the R factors span a rich feature space (near-identical clones -> low-rank, cannot fit).
        self.a_raw = nn.Parameter(_softplus_inv(1.0) + 0.3 * torch.randn(R, nn_, generator=g))
        self.b = nn.Parameter(b_anchor.unsqueeze(0).repeat(R, 1) + 0.6 * torch.randn(R, nn_, generator=g))

        # ---- lift params (identifiable widened (r,theta) Gr_+(2,n) chart) ----
        # ANCHOR to Example 1.3: rows (1,0),(0,1),(-sqrt2,sqrt3) => minors {1, sqrt2, sqrt3}, which are
        # Q-linearly independent (genuine aperiodicity, AKKV Thm 2.5).  theta_1 is set slightly > 0
        # (the chart's open interval) so minors are near, not exactly, {1,sqrt2,sqrt3} but stay
        # incommensurate.  For n>3 spread the extra angles by sqrt-prime increments (incommensurate).
        if nn_ == 3:
            theta_t = torch.tensor([0.03, PI / 2.0, math.atan2(math.sqrt(3.0), -math.sqrt(2.0))])
            r_t = torch.tensor([1.0, 1.0, math.sqrt(5.0)])
        else:
            inc = torch.sqrt(torch.tensor([2., 3., 5., 7., 11., 13., 17., 19., 23., 29.][:nn_]))
            theta_t = 0.03 + (self.SPREAD - 0.1) * torch.cumsum(inc, 0) / inc.sum()
            r_t = torch.ones(nn_)
        S_t = (theta_t / self.SPREAD).clamp(1e-3, 0.999)
        h_t = float((1.0 - S_t[-1]).clamp_min(0.05))
        gaps_t = torch.diff(S_t, prepend=torch.zeros(1)).clamp_min(1e-3)
        rho0 = torch.log(r_t)
        gap0 = torch.tensor([_softplus_inv(float(x)) for x in gaps_t])
        head0 = _softplus_inv(h_t)
        # DIVERSE lifts per factor: large jitter on radii / angle-gaps / headroom so every factor is a
        # distinct point of Gr_+(2,n) (distinct minors {spectrum}).  The chart structurally keeps each
        # factor faithful (theta increasing, total spread < 0.95*pi, minors > 0) for ANY raw values.
        self.rho = nn.Parameter(rho0.unsqueeze(0).repeat(R, 1) + 0.4 * torch.randn(R, nn_, generator=g))
        self.gap_raw = nn.Parameter(gap0.unsqueeze(0).repeat(R, 1) + 0.6 * torch.randn(R, nn_, generator=g))
        self.head_raw = nn.Parameter(torch.full((R,), head0) + 0.5 * torch.randn(R, generator=g))

        # input cycling: every factor reads d real features (cycle through in_dim) so no factor is fed
        # all-zeros (zero input -> z=1 -> p(z=1)=0 exactly, a structural dead channel).
        self.register_buffer("gather_idx", torch.arange(self.padded_in) % max(in_dim, 1))

        self.use_phase_offset = phase_offset
        if phase_offset:
            base = (torch.arange(nn_).float() / nn_).unsqueeze(0).repeat(R, 1)
            self.phase_offset = nn.Parameter(base + 0.05 * torch.randn(R, nn_, generator=g))
        else:
            self.register_parameter("phase_offset", None)

    # ---- the positive-Grassmannian lift L : [R, n, 2] ----
    def L(self) -> torch.Tensor:
        r = torch.exp(self.rho)                                  # [R, n]  per-row radius
        S = torch.cumsum(_softplus(self.gap_raw), dim=-1)        # [R, n]  ABSOLUTE cumulative gaps
        h = _softplus(self.head_raw).unsqueeze(-1)               # [R, 1]  separate headroom
        theta = self.SPREAD * S / (S[:, -1:] + h)                # [R, n]  in (0, SPREAD), increasing
        L = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)   # [R, n, 2]
        return L * self.init_scale

    def thetas(self) -> torch.Tensor:
        S = torch.cumsum(_softplus(self.gap_raw), dim=-1)
        h = _softplus(self.head_raw).unsqueeze(-1)
        return self.SPREAD * S / (S[:, -1:] + h)

    # ---- STAR eliminant coefficients : each [R, n-1] complex ----
    def _coeffs(self):
        a = _softplus(self.a_raw)                                # [R, n] > 0 (positive leading coeff)
        m = self.signs.unsqueeze(0) * a                          # [R, n] slopes  m_i = s_i a_i
        c = self.signs.unsqueeze(0) * self.b                     # [R, n] intercepts c_i = s_i b_i
        m1, c1 = m[:, :1], c[:, :1]                              # shared coordinate z_1 (the STAR hub)
        mq, cq = m[:, 1:], c[:, 1:]                              # [R, n-1] the spokes z_2..z_n
        det = c1 * mq - cq * m1                                  # [R, n-1]
        Al = torch.complex(det, m1 - mq)
        Be = torch.complex(-det, m1 + mq)
        Ga = torch.complex(-det, -(m1 + mq))
        De = torch.complex(det, -m1 + mq)
        return Al, Be, Ga, De

    def _relations(self, xh: torch.Tensor, L: torch.Tensor, coeffs) -> torch.Tensor:
        """xh : [B, R, d] -> [B, out_dim] curve-relation residual features (Re/Im of the n-1 stars)."""
        Al, Be, Ga, De = coeffs
        phase = torch.einsum("brd,rnd->brn", xh, L)              # L x  ->  [B, R, n]
        if self.phase_offset is not None:
            phase = phase + self.phase_offset.unsqueeze(0)
        z = torch.exp(2j * PI * phase.to(torch.complex64))       # [B, R, n] on the torus
        z1 = z[:, :, :1]                                         # [B, R, 1]  the hub
        zq = z[:, :, 1:]                                         # [B, R, n-1] the spokes
        p = (Al.unsqueeze(0) * z1 * zq + Be.unsqueeze(0) * z1
             + Ga.unsqueeze(0) * zq + De.unsqueeze(0))           # [B, R, n-1] complex
        feat = torch.cat([p.real, p.imag], dim=-1)               # [B, R, 2(n-1)]
        # take out_per_factor from EVERY factor (all input regions contribute + get gradient), trim
        feat = feat[:, :, : self.out_per_factor].reshape(feat.shape[0], self.num_factors * self.out_per_factor)
        return feat[:, : self.out_dim]

    def _relations_full(self, xh: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """Checkpoint-friendly: recompute coeffs inside (only tensor args xh, L cross the boundary)."""
        return self._relations(xh, L, self._coeffs())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = x[:, self.gather_idx]                                # cycle features -> [B, padded_in]
        xh = x.view(B, self.num_factors, self.d)
        L = self.L()
        chunk = self.chunk if self.chunk is not None else B
        use_ckpt = self.training and torch.is_grad_enabled()
        if B <= chunk and not use_ckpt:
            return self._relations(xh, L, self._coeffs())
        # gradient-checkpoint each chunk to bound memory on large message blocks
        outs = []
        for i in range(0, B, chunk):
            xc = xh[i:i + chunk]
            if use_ckpt:
                outs.append(torch.utils.checkpoint.checkpoint(self._relations_full, xc, L,
                                                              use_reentrant=False))
            else:
                outs.append(self._relations(xc, L, self._coeffs()))
        return torch.cat(outs, dim=0)

    # ---- positivity witness: smallest 2x2 minor of any factor's lift (must stay > 0) ----
    def min_minor(self) -> torch.Tensor:
        L = self.L()                                             # [R, n, 2]
        R, nn_, _ = L.shape
        mins = []
        for i in range(nn_):
            for j in range(i + 1, nn_):
                # minor on rows i<j = det([L_i; L_j]) = Lx_i Ly_j - Ly_i Lx_j
                mij = L[:, i, 0] * L[:, j, 1] - L[:, i, 1] * L[:, j, 0]
                mins.append(mij.min())
        return torch.stack(mins).min()

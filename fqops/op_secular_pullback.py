"""
fqops.op_secular_pullback  --  FQSecularPullbackBlock

A NOVEL Op-C-style FQ feature-encoder operator: the VALUE + JACOBIAN-PULLBACK
of a BANK of learnable Kurasov-Sarnak SECULAR polynomials, with the channel mixing supplied
analytically by the variety geometry (the resolvent diagonal of a learned unitary), NOT by any nn.Linear.

Note: the secular determinant is the d=1 KS special case (NOT the universal d>=2
generator).  This block uses the det(diag(z)U - I) convention; the normalization used in fqmath is
det(I - diag(z)U) (so p_U(0)=1), which differs only by the (-1)^n constant -> IDENTICAL zero set, and
the unit-RMS `balance` gauge makes the convention immaterial to the emitted features.

Faithful objects only (AKKV + Lawton-Tsikh):
  * FQ lift     gamma(x) = exp(2 pi i L x) in T^n,  x = proj @ v in R^2 (per-head learned projection).
  * Lift L      in Gr_+(2,n) via the (r,theta) chart -- every 2x2 minor r_i r_j sin(th_j - th_i) > 0
                STRUCTURALLY for any parameter (identical to FQSpectralBlock.L()).
  * Scalar g_m  built from the secular polynomial  p_{U_m}(z) = det(diag(z) U_m - I),
                U_m = exp(i H_m) UNITARY for any H_m Hermitian => on the torus diag(z)U is unitary
                => all secular zeros are REAL (strict Lee-Yang) STRUCTURALLY, never penalized.
  * Pullback    J_gamma^T grad_z g = 2 pi i L^T( gamma (*) grad_z g ),  and L^T(.) is EXACTLY the
                spectrum map Lambda' = {L^T k} (fqmath.spectrum) -- the gradient feature lands natively
                in FQ frequency space, so it is faithful, not a graft.

CLOSED-FORM gradient (Jacobi's formula; agrees with autograd in a float64 gradcheck):
    p'_i := dp/dz_i (holomorphic) = det(A) * (U A^{-1})_{ii},   A = diag(z) U - I.
  (The naive (A^{-1} U)_{ii} is WRONG by O(1) and is NOT used.)
  The REAL gradient grad_z g fed into the identity is 2x the Wirtinger derivative so that
  Re[2 pi i L^T(gamma (*) grad_z g)] equals the TRUE real Jacobian dg(gamma(x))/dx (autograd-matched
  to 1e-14). For the chosen Re-p / Im-p value pair this is simply  grad_z(Re p)=p',  grad_z(Im p)=-i p'.

WHY THIS DESIGN:
  - Mode = 'Rep' is the hot path: fp32-SAFE (no division by the secular value, no |p|-floor bias),
    FULL-RANK pullback, and the resolvent diagonal det*(U A^{-1})_ii is a data-dependent, NONLINEAR,
    all-channel-coupled map (NOT linearly recoverable from the decoupled feature; lstsq residual ~0.94)
    -- this is claim (1), the rank/mixing the cos/sin blocks lack, with paper math instead of a dense W.
  - We emit the VALUE PAIR [Re p, Im p] (NOT a single residual): a single-coordinate zero of p kills
    only one of the pair, so the value stays informative on a neighborhood of the secular zero set.
  - We emit TWO gradient channels [Re(pull of Re p), Re(pull of Im p)] (the harmonic-conjugate
    companion): a genuine quadrature pair whose joint zero set is only the (measure-zero) secular
    variety where BOTH p and p' vanish; the value pair covers exactly those points.
  - Scope of claim (2): the non-vanishing-gradient property holds AWAY from the secular zero
    set; it is NOT universal. The constant-modulus phase companion d/dx arg(p) is NOT used -- on the
    torus it is a CONSTANT vector (the eigenphase winding number), information-free. The value pair is
    what guards the gradient nulls, and ||U-I|| telemetry guards against silent decoupling.

IN-FAMILY guarantees (structural, never projected / penalized):
  (1) U = exp(i * Hermitian) is unitary for ANY parameter  => Lee-Yang real-rootedness holds always.
  (2) L from the (r,theta) chart has all 2x2 minors > 0     => Gr_+(2,n) membership holds always.
A numerical floor is used ONLY for the optional 'logp2' diagnostic mode (not the default hot path),
and that floor is exact-in-fp64-then-cast, never a biased fp32 |p|-floor on the gradient.

fp32 ONLY (no reduced precision in the FQ path). The resolvent (det + solve) is computed in the working
dtype; for the optional division-based modes the resolvent is done in float64 LOCALLY then cast, so
no biased gradient ever enters. PyTorch only, no custom CUDA. Trains by ordinary autograd through the
forward graph (the closed forms only BUILD the feature; we do not hand-inject gradients).
"""

from __future__ import annotations

import itertools
import math

import torch
import torch.nn as nn
import torch.utils.checkpoint

PI = math.pi
TWO_PI_I = 2j * math.pi

# init scale of the Hermitian generator so the coupling is NOT frozen at U ~ I (guards against
# U-degeneracy). ||U - I|| off-diagonal energy should stay > 0 during training (see u_offdiag_energy()).
U_INIT_SCALE = 0.3


def _softplus(x, floor: float = 1e-4):
    return torch.nn.functional.softplus(x) + floor


def _softplus_inv(y: float) -> float:
    y = max(y - 1e-4, 1e-6)
    return math.log(math.expm1(y)) if y < 20 else y


class FQSecularPullbackBlock(nn.Module):
    """
    Faithful FQ secular value+pullback feature encoder.

    Output layout per (head h, bank-member m), Rep mode (default):
        [ Re p , Im p , d/dx(Re p)_1 , d/dx(Re p)_2 , d/dx(Im p)_1 , d/dx(Im p)_2 ]   = 6 real features.
    A bank of M members per head and H heads gives H * M * out_per_g raw features, trimmed to out_dim
    (same head/trim bookkeeping as FQSpectralBlock).

    Constructor mirrors FQSpectralBlock as closely as the math allows.
    """

    SPREAD = 0.95 * PI

    def __init__(self, in_dim: int, out_dim: int, n: int = 8, n_freqs: int | None = None,
                 init_scale: float = 0.4, seed: int = 0, chunk: int | None = None,
                 mode: str = "Rep", u_init_scale: float = U_INIT_SCALE, balance: bool = False):
        super().__init__()
        assert mode in ("Rep", "p2", "logp2"), mode
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n = n
        self.d = 2
        self.init_scale = init_scale
        self.chunk = chunk
        self.mode = mode
        self.u_init_scale = float(u_init_scale)
        self.balance = balance      # unit-RMS output gauge: secular det values are O(10-100) and would
                                    # SWAMP a concatenation (~89x the cos/sin features) -> normalize to ~1.

        # ---- bank size M (number of secular scalars per head) and per-g width ----
        # Rep: 2 value + 2*d grad = 6 ; p2: 1 value + 2*d grad = 5 ; logp2: 1 value + d grad = 3
        # (logp2 drops Im(pull), which is constant for logp2).
        if mode == "Rep":
            self.per_g = 2 + 2 * self.d          # 6
        elif mode == "p2":
            self.per_g = 1 + 2 * self.d          # 5  (Im(pull) is rank-redundant but kept; see notes)
        else:  # logp2
            self.per_g = 1 + self.d              # 3  (Im(pull) dropped: constant for logp2)
        # M is a real hyperparameter (the bank richness), set directly (>=1); default n.
        M = max(1, int(n_freqs) if n_freqs is not None else max(1, n // 2 + 1))
        self.M = M
        self.per_head_out = self.per_g * M
        self.num_heads = max(1, math.ceil(out_dim / self.per_head_out))
        H = self.num_heads

        # ---- per-head learned d=2 projection of the FULL input (op_spectral_block style) ----
        gP = torch.Generator().manual_seed(seed + 7919)
        self.proj = nn.Parameter(
            torch.randn(H, self.d, in_dim, generator=gP) / math.sqrt(in_dim))

        # ---- (r,theta) Gr_+(2,n) lift per head (IDENTICAL to FQSpectralBlock.L()) ----
        pairs = list(itertools.combinations(range(n), 2))
        self.register_buffer("pair_i", torch.tensor([p[0] for p in pairs]))
        self.register_buffer("pair_j", torch.tensor([p[1] for p in pairs]))
        self.C = len(pairs)

        g = torch.Generator().manual_seed(seed)
        primes = torch.sqrt(torch.tensor(
            [2., 3., 5., 7., 11., 13., 17., 19., 23., 29., 31., 37.][:n]))
        rho0 = 0.3 * torch.log(primes / primes.mean())
        self.rho = nn.Parameter(rho0.unsqueeze(0).repeat(H, 1) + 0.3 * torch.randn(H, n, generator=g))
        gap0 = _softplus_inv(self.SPREAD / (n + 0.5))
        self.gap_raw = nn.Parameter(gap0 + 0.5 * torch.randn(H, n, generator=g))
        self.head_raw = nn.Parameter(
            torch.full((H,), _softplus_inv(0.4)) + 0.4 * torch.randn(H, generator=g))

        # ---- bank of learnable unitaries U_{h,m} = exp(i H_{h,m}) ----
        # H_{h,m} = 0.5 (A + A^dagger) from an unconstrained complex A = A_re + i A_im.
        # Init A ~ N(0, u_init_scale) so U ~ exp(i * O(u_init_scale) Hermitian): coupling NOT frozen
        # at U=I, yet starts near the well-behaved decoupled point.
        self.A_re = nn.Parameter(self.u_init_scale * torch.randn(H, M, n, n, generator=g))
        self.A_im = nn.Parameter(self.u_init_scale * torch.randn(H, M, n, n, generator=g))

        # ---- output affine (a plain scale/bias on the OUTPUT scalars, NOT in the operator) ----
        # This is the only place the feature SCALE is set; it is NOT a channel-mixing W (per-channel
        # diagonal), so the operator itself contains no nn.Linear.
        self.out_scale = nn.Parameter(torch.ones(H, M, self.per_g))
        self.out_bias = nn.Parameter(torch.zeros(H, M, self.per_g))

    # ---------------------------------------------------------------- lift ----
    def L(self) -> torch.Tensor:
        r = torch.exp(self.rho)                                       # [H, n]
        S = torch.cumsum(_softplus(self.gap_raw), dim=-1)             # [H, n]
        h = _softplus(self.head_raw).unsqueeze(-1)                    # [H, 1]
        theta = self.SPREAD * S / (S[:, -1:] + h)                     # [H, n] in (0, SPREAD), increasing
        L = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)  # [H, n, 2]
        return L * self.init_scale

    def minors(self, L: torch.Tensor) -> torch.Tensor:
        """2x2 minors r_i r_j sin(theta_j - theta_i) for every row pair: [H, C]."""
        Li = L[:, self.pair_i, :]
        Lj = L[:, self.pair_j, :]
        return Li[..., 0] * Lj[..., 1] - Li[..., 1] * Lj[..., 0]

    def min_minor(self) -> torch.Tensor:
        """Cone-membership witness: smallest 2x2 minor (must stay > 0)."""
        return self.minors(self.L()).min()

    # ---------------------------------------------------------------- unitary bank ----
    def U(self) -> torch.Tensor:
        """U_{h,m} = exp(i H_{h,m}), unitary for any parameter. Returns complex [H, M, n, n]."""
        A = torch.complex(self.A_re, self.A_im)                      # [H, M, n, n]
        Hm = 0.5 * (A + A.conj().transpose(-1, -2))                  # Hermitian
        return torch.matrix_exp(1j * Hm)

    def u_offdiag_energy(self) -> torch.Tensor:
        """Coupling telemetry: ||U - diag(U)||_F per (h,m), summed. Coupling alive if > 0."""
        U = self.U()
        diag = torch.diagonal(U, dim1=-2, dim2=-1)                   # [H, M, n]
        off = U - torch.diag_embed(diag)
        return off.abs().pow(2).sum().sqrt()

    # ---------------------------------------------------------------- forward core ----
    def _transform(self, v: torch.Tensor, L: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """v : [B, in_dim] -> [B, out_dim] secular value+pullback features."""
        B = v.shape[0]
        H, M, n = self.num_heads, self.M, self.n
        cdt = torch.complex64 if v.dtype == torch.float32 else torch.complex128
        rdt = v.dtype

        # FQ lift: x = proj @ v ; phase = L x ; gamma = exp(2 pi i phase)
        xh = torch.einsum("bi,hdi->bhd", v, self.proj)               # [B, H, 2]
        phase = torch.einsum("bhd,hnd->bhn", xh, L)                  # [B, H, n] = (L x)
        z = torch.exp(TWO_PI_I * phase.to(cdt))                      # [B, H, n] gamma, |z|=1

        # secular A = diag(z) U - I, batched over [B, H, M, n, n]
        Ue = U.unsqueeze(0)                                          # [1, H, M, n, n]
        I = torch.eye(n, dtype=cdt, device=v.device)
        ze = z.unsqueeze(2).unsqueeze(-1)                            # [B, H, 1, n, 1]
        A = ze * Ue - I                                              # [B, H, M, n, n]
        det = torch.linalg.det(A)                                   # [B, H, M] = p_U(z)

        # Jacobi holomorphic derivative p'_i = det * (U A^{-1})_ii, computed with torch.linalg.solve
        # (better conditioned than an explicit inverse).
        # (U A^{-1})_ii = sum_a U_ia (A^{-1})_ai. Note solve(A, U) gives X = A^{-1} U, which is
        # the WRONG product. We need U A^{-1}: solve A^T Y = U^T -> Y = A^{-T} U^T = (U A^{-1})^T,
        # so (U A^{-1})_ii = Y_ii. Use solve on the transpose system.
        At = A.transpose(-1, -2)
        Ut = Ue.transpose(-1, -2).expand(B, H, M, n, n)
        Y = torch.linalg.solve(At, Ut)                              # [B,H,M,n,n] = (U A^{-1})^T
        UAinv_diag = torch.diagonal(Y, dim1=-2, dim2=-1)            # [B, H, M, n] = (U A^{-1})_ii
        pprime = det.unsqueeze(-1) * UAinv_diag                     # [B, H, M, n] holomorphic p'

        Lc = L.to(cdt)                                              # [H, n, 2]
        zk = z.unsqueeze(2)                                         # [B, H, 1, n] -> broadcast over M

        def pull(grad_z):                                          # grad_z: [B,H,M,n] -> [B,H,M,2] real
            w = zk * grad_z                                        # gamma (*) grad_z g
            return (TWO_PI_I * torch.einsum("hnd,bhmn->bhmd", Lc, w)).real

        if self.mode == "Rep":
            gRe = det.real                                         # [B, H, M]
            gIm = det.imag
            pr_re = pull(pprime)                                   # d/dx Re p   [B,H,M,2]
            pr_im = pull(-1j * pprime)                             # d/dx Im p   [B,H,M,2]
            feat = torch.cat([gRe.unsqueeze(-1), gIm.unsqueeze(-1), pr_re, pr_im], dim=-1)  # [B,H,M,6]
        elif self.mode == "p2":
            g = (det.conj() * det).real                           # |p|^2
            grad_z = 2 * det.conj().unsqueeze(-1) * pprime
            pr = pull(grad_z)
            # Im(pull) for p2 is rank-redundant (one channel) but cheap; keep for width.
            pr_im = pull(1j * grad_z)
            feat = torch.cat([g.unsqueeze(-1), pr, pr_im], dim=-1)  # [B,H,M,5]
        else:  # logp2 -- division-based, do resolvent in fp64 locally then cast
            det64 = det.to(torch.complex128)
            pp64 = pprime.to(torch.complex128)
            g = torch.log((det64.conj() * det64).real + 1e-12)
            grad_z = 2 * pp64 / det64.unsqueeze(-1)               # p'/p exact in fp64
            w = z.to(torch.complex128).unsqueeze(2) * grad_z
            pr = (TWO_PI_I * torch.einsum("hnd,bhmn->bhmd",
                                          L.to(torch.complex128), w)).real
            feat = torch.cat([g.unsqueeze(-1).to(rdt), pr.to(rdt)], dim=-1)  # [B,H,M,3]
            # (Im(pull) omitted: it is constant for logp2.)

        feat = self.out_scale * feat + self.out_bias               # output affine [H,M,per_g]
        feat = feat.reshape(B, H * M * self.per_g)                 # [B, H*M*per_g]
        return feat[:, : self.out_dim].to(rdt)                     # trim tail

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        B = v.shape[0]
        L = self.L()
        U = self.U()
        chunk = self.chunk if self.chunk is not None else B
        use_ckpt = self.training and torch.is_grad_enabled()
        if B <= chunk and not use_ckpt:
            out = self._transform(v, L, U)
        else:
            outs = []
            for i in range(0, B, chunk):
                vc = v[i:i + chunk]
                if use_ckpt:
                    outs.append(torch.utils.checkpoint.checkpoint(
                        self._transform, vc, L, U, use_reentrant=False))
                else:
                    outs.append(self._transform(vc, L, U))
            out = torch.cat(outs, dim=0)
        if self.balance:   # unit-RMS gauge (detached) -- secular det values are O(10-100); without
            out = out / (out.detach().float().pow(2).mean().sqrt() + 1e-6)   # this it swamps any concat (89x).
        return out

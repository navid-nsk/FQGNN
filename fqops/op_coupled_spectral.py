"""
fqops.op_coupled_spectral  --  FQCoupledSpectralBlock : an EXPRESSIVE spectral block with a learned
unitary coefficient-mixer.

SCOPE NOTE: the GEOMETRY is faithful (xi_k = L^T k on Gr_+(2,n), the aperiodic Lambda'),
but (1) the amplitudes c_k = sum_I L_I w_{I,k} use a FREE w_{I,k} (NOT the variety-fixed cumulant/Eq.9),
and (2) the unitary U = exp(iH) here mixes the M SPECTRAL COEFFICIENTS Z_k -- this is a
PROJECT-ORIGINAL learned feature-mixer, NOT the Kurasov-Sarnak secular coupling of the n torus
coordinates (which is M(z)=diag(z)U and lives in op_secular / SecularCoupledConvAggregate).  Treat this
block as a stronger geometry-only ablation (free amplitude + coefficient-mixing), NOT as the faithful FQ.

Motivation:
  * The SPECTRAL block (op_spectral_block.FQSpectralBlock) is geometry-faithful + EXPRESSIVE but
    DECOUPLED: per-frequency features  c_k cos/sin(2 pi <xi_k, x>),  with no interaction between the M
    frequencies (no off-diagonal U, no z_j z_{j+1} variety term).  In the papers' terms it is a
    structured aperiodic Fourier map -- it lacks the property that makes an FQ genuinely high-dimensional.
  * The SECULAR block's trace-moment mode (op_secular.FQSecularBlock, feat_mode="trace") IS coupled
    (M(z)=diag(z)U, U=exp(iH)), but the trace moments t_k = tr(M^k) COLLAPSE all n eigenphases into one
    number per k (loses resolution), and near U=I they reduce to decoupled cosines.
  * AKKV / Lawton-Tsikh: the unique FQ property is CHANNEL COUPLING through an irreducible (non-product)
    Lee-Yang variety; the secular realization det(diag(z)U - I) couples the n torus coordinates via the
    OFF-DIAGONAL of a unitary U (U=I => reducible/product => decoupled cosines, the control, never the
    model).  A unitary mixing is Lee-Yang-preserving (always in family).

THIS BLOCK = the spectral block (expressive, faithful spectrum + amplitudes linear in the minors) PLUS a
per-head UNITARY coupling U = exp(i H) (H Hermitian) applied to the M complex spectral coefficients
Z_k = c_k exp(2 pi i <xi_k, proj x>), BEFORE taking real/imag -- so it keeps the individual frequencies
(expressive, no trace collapse) yet adds genuine cross-component coupling (the off-diagonal of U).

    Z'    = U Z              (U = exp(i H),  H Hermitian  =>  U unitary, Lee-Yang preserving)
    feat  = [ Re Z', Im Z' ]

Faithful by construction (inherited from FQSpectralBlock unchanged): lift L is the (r,theta) Gr_+(2,n)
chart (all 2x2 minors r_i r_j sin(theta_j - theta_i) > 0); frequencies xi_k = L^T k are the genuine
aperiodic spectrum Lambda' = {L^T k : var(k) < d} (AKKV Thm 2.3); amplitudes c_k = sum_I L_I w_{I,k}
are LINEAR in the d x d minors (AKKV Thm 2.6(3)).  ADDED: the unitary U = exp(i H) (the Kurasov-Sarnak
secular-style coupling).

Init: the Hermitian generator has random entries of scale U_INIT, so U starts as a genuine unitary
rotation away from the identity (information-preserving, so the spectral features are not degraded)
and the block LEARNS the coupling from there.  coupled=False forces U=I exactly (the decoupled
spectral control / ablation).
"""

from __future__ import annotations

import itertools
import math

import torch
import torch.nn as nn
import torch.utils.checkpoint

from fqmath.spectrum import enumerate_spectrum_combos

PI = math.pi

# Test override for the M=1-vs-M=8 ablation (shared semantics with op_spectral_block.FORCE_NFREQS).
FORCE_NFREQS = None
# HP-search knobs (set per Optuna trial): phase-scale init + unitary-coupling init.
PHASE_SCALE_INIT = 0.15
U_INIT = 0.3


def _softplus(x, floor: float = 1e-4):
    return torch.nn.functional.softplus(x) + floor


def _softplus_inv(y: float) -> float:
    y = max(y - 1e-4, 1e-6)
    return math.log(math.expm1(y)) if y < 20 else y


class FQCoupledSpectralBlock(nn.Module):
    """Expressive + coupled FQ block: FQSpectralBlock features mixed by a per-head unitary U=exp(iH).

    in_dim, out_dim     : as the other primitives (drop-in via make_fq_block).
    n                   : torus coords per head (d=2 lift); M frequencies per head (= n by default).
    coupled             : True -> U=exp(iH) (the coupling); False -> U=I (decoupled spectral control).
    """

    SPREAD = 0.95 * PI

    def __init__(self, in_dim: int, out_dim: int, n: int = 8, k_max: int = 3,
                 init_scale: float = 0.4, coupled: bool = True, seed: int = 0,
                 chunk: int | None = None, phase_offset: bool = False, n_freqs: int | None = None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n = n
        self.d = 2
        self.init_scale = init_scale
        self.coupled = coupled
        self.chunk = chunk

        combos_full = enumerate_spectrum_combos(n, self.d, k_max, max_freqs=4096)     # [Mfull, n]
        target = int(FORCE_NFREQS) if FORCE_NFREQS is not None else (int(n_freqs) if n_freqs is not None else n)
        M = max(1, min(target, combos_full.shape[0]))
        self.M = M
        self.per_head_out = 2 * M                                                     # cos+sin of M (coupled) coeffs
        self.num_heads = max(1, math.ceil(out_dim / self.per_head_out))
        self.out_per_head = self.per_head_out
        self.register_buffer("combos", torch.tensor(combos_full[:M]))                 # [M, n]

        # Per-head learned d=2 projection of the FULL input (identical to FQSpectralBlock).
        gP = torch.Generator().manual_seed(seed + 7919)
        self.proj = nn.Parameter(
            torch.randn(self.num_heads, self.d, in_dim, generator=gP) / math.sqrt(in_dim))

        pairs = list(itertools.combinations(range(n), 2))
        self.register_buffer("pair_i", torch.tensor([p[0] for p in pairs]))
        self.register_buffer("pair_j", torch.tensor([p[1] for p in pairs]))
        self.C = len(pairs)

        H = self.num_heads
        g = torch.Generator().manual_seed(seed)
        # (r,theta) Gr_+(2,n) lift per head (identical to FQSpectralBlock: minors > 0 by construction)
        primes = torch.sqrt(torch.tensor([2., 3., 5., 7., 11., 13., 17., 19., 23., 29., 31., 37.][:n]))
        rho0 = 0.3 * torch.log(primes / primes.mean())
        self.rho = nn.Parameter(rho0.unsqueeze(0).repeat(H, 1) + 0.3 * torch.randn(H, n, generator=g))
        gap0 = _softplus_inv(self.SPREAD / (n + 0.5))
        self.gap_raw = nn.Parameter(gap0 + 0.5 * torch.randn(H, n, generator=g))
        self.head_raw = nn.Parameter(torch.full((H,), _softplus_inv(0.4)) + 0.4 * torch.randn(H, generator=g))
        # amplitude weights w_{I,k}: c_k = sum_I L_I w_{I,k}  (AKKV Thm 2.6(3), linear in minors)
        self.amp_w = nn.Parameter(0.3 * torch.randn(H, self.C, M, generator=g))

        # ---- Phase conditioning: learnable per-head scale on the torus phase so cos/sin starts
        # in its informative MONOTONE band. With an unscaled pre-cos phase RMS of ~1.3-1.9 rad, cos/sin
        # ALIASES the input and much of the linear signal is lost (an underfit floor). Faithful:
        # x's scale is a gauge absorbed by the lift L; learnable so high frequencies return if earned.
        # init ~0.15 -> phase RMS ~0.5 with M=12 freqs; learnable so it can grow.
        self.phase_scale_raw = nn.Parameter(torch.full((H,), _softplus_inv(PHASE_SCALE_INIT)))

        # ---- THE COUPLING: per-head unitary U_h = exp(i H_h), H_h Hermitian [M, M]; the block LEARNS
        #      the coupling.
        if coupled:
            # initial scale U_INIT (0.3) so that the coupling U starts away from the identity with a
            # usable gradient (a unitary rotation is information-preserving, so it does not hurt the
            # spectral start).
            self.U_real = nn.Parameter(U_INIT * torch.randn(H, M, M, generator=g))
            self.U_imag = nn.Parameter(U_INIT * torch.randn(H, M, M, generator=g))
        else:
            self.register_parameter("U_real", None)
            self.register_parameter("U_imag", None)

        self.use_phase_offset = phase_offset
        if phase_offset:
            self.phase_offset = nn.Parameter(0.1 * torch.randn(H, M, generator=g))
        else:
            self.register_parameter("phase_offset", None)

    def L(self) -> torch.Tensor:
        r = torch.exp(self.rho)                                       # [H, n]
        S = torch.cumsum(_softplus(self.gap_raw), dim=-1)            # [H, n]
        h = _softplus(self.head_raw).unsqueeze(-1)                    # [H, 1]
        theta = self.SPREAD * S / (S[:, -1:] + h)                     # [H, n] increasing in (0, SPREAD)
        L = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)   # [H, n, 2]
        return L * self.init_scale

    def minors(self, L: torch.Tensor) -> torch.Tensor:
        Li = L[:, self.pair_i, :]
        Lj = L[:, self.pair_j, :]
        return Li[..., 0] * Lj[..., 1] - Li[..., 1] * Lj[..., 0]      # [H, C]

    def U(self) -> torch.Tensor:
        """Per-head unitary coupling U = exp(i H), H = 0.5(A + A^dagger) Hermitian: [H, M, M] complex."""
        if not self.coupled:
            eye = torch.eye(self.M, dtype=torch.complex64, device=self.rho.device)
            return eye.unsqueeze(0).expand(self.num_heads, -1, -1)
        A = torch.complex(self.U_real, self.U_imag)                  # [H, M, M]
        Herm = 0.5 * (A + A.conj().transpose(-2, -1))                # Hermitian
        return torch.linalg.matrix_exp(1j * Herm.to(torch.complex64))   # unitary [H, M, M]

    def _transform(self, x: torch.Tensor, L: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """x : [B, in_dim] -> [B, out_dim]  coupled aperiodic Fourier features (M coupled coeffs/head)."""
        xh = torch.einsum("bi,hdi->bhd", x, self.proj)               # [B, H, 2]
        xi = torch.einsum("mn,hnd->hmd", self.combos, L)             # [H, M, 2]  xi_k = L^T k
        c = torch.einsum("hc,hcm->hm", self.minors(L), self.amp_w)   # [H, M]     amplitudes linear in minors
        ps = _softplus(self.phase_scale_raw).view(1, self.num_heads, 1)   # learnable phase scale
        a = 2 * PI * ps * torch.einsum("bhd,hmd->bhm", xh, xi)        # [B, H, M]  conditioned phase
        if self.phase_offset is not None:
            a = a + 2 * PI * self.phase_offset.unsqueeze(0)
        # amplitude-weighted torus point (the spectral coefficients), THEN unitary coupling across freqs
        Z = c.unsqueeze(0).to(torch.complex64) * torch.exp(1j * a.to(torch.complex64))   # [B, H, M]
        Zc = torch.einsum("hmk,bhk->bhm", U, Z)                      # [B, H, M]  Z' = U Z  (the coupling)
        feat = torch.cat([Zc.real, Zc.imag], dim=-1)                 # [B, H, 2M]
        feat = feat.reshape(feat.shape[0], self.num_heads * self.per_head_out)
        return feat[:, : self.out_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        L = self.L(); U = self.U()
        chunk = self.chunk if self.chunk is not None else B
        use_ckpt = self.training and torch.is_grad_enabled()
        if B <= chunk and not use_ckpt:
            return self._transform(x, L, U)
        outs = []
        for i in range(0, B, chunk):
            xc = x[i:i + chunk]
            if use_ckpt:
                outs.append(torch.utils.checkpoint.checkpoint(self._transform, xc, L, U,
                                                              use_reentrant=False))
            else:
                outs.append(self._transform(xc, L, U))
        return torch.cat(outs, dim=0)

    def min_minor(self) -> torch.Tensor:
        return self.minors(self.L()).min()

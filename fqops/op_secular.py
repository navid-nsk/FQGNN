"""
fqops.op_secular  --  Op-B : the FQ secular eigenstructure block (a d=1 coupled operator)

The coupled secular operator M(z) = diag(z) U  (z on the n-torus, U unitary).  The DEFAULT
featurization is the EIGENPHASE one, feat_B(x) = [cos theta_j(x), sin theta_j(x)] (feat_mode=
"eigenphase"): the eigenphases are proven continuous, monotone, real-analytic and Lipschitz-bounded
(Alon-Vinzant-gaps), with the free sanity check prod_j (1 - e^{i theta_j}) = p_U.

The power-sum / TRACE moments t_k = tr(M^k) (feat_mode="trace") are a PROJECT-ORIGINAL featurization
-- NOT AKKV-faithful (AKKV has no power sums) and, being entrywise powers, family-breaking
(Belton-et-al) -- available only as the GPU-fast ablation.

Scope: the secular determinant is the d=1 / Kurasov-Sarnak special case (NOT the universal
d>=2 generator).  The FAITHFUL coupled MESSAGE should route through the VARIETY membership Op-C
(op_variety / op_curve), not through this block.

  * M(z) is unitary for any z in T^n  => eigenvalues on the unit circle => p_U=det(I - diag(z)U) is
    real-rooted (Lee-Yang), always-in-family for ANY parameters.
  * the COUPLING is the off-diagonal of U: theta_j(z) is an eigenphase of the WHOLE matrix; U = I
    (coupled=False) collapses to decoupled cosines -- the ablation, never the model.  "No RFF analog",
    stated precisely: no analog that is simultaneously coupled AND real-rooted/Lee-Yang AND
    amplitude-bounded-by-the-variety.
  * L is positive-Grassmannian (STP exponential chart): every d x d minor positive.

A single big secular operator is intractable for wide states (cost O(K n^3)).  We keep n SMALL by
MULTI-HEAD: the width is split into many heads of a fixed small size `in_per_head`, each head owns
its own positive-Grassmannian L_h and unitary U_h, and the whole computation is VECTORIZED over
heads (batched matrix_exp + batched matmul) -- no Python head loop.  Edges can be processed in
chunks to bound memory on large graphs.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint

PI = math.pi


def _softplus(x, floor=1e-4):
    return torch.nn.functional.softplus(x) + floor


def _increasing(raw, lo, hi):
    """Per-row strictly increasing sequence in (lo,hi). raw: [..., m] -> [..., m]."""
    gaps = _softplus(raw)
    cs = torch.cumsum(gaps, dim=-1) - 0.5 * gaps
    u = cs / (cs[..., -1:] + 0.5 * gaps[..., -1:] + 1e-12)
    margin = 0.02
    return lo + (hi - lo) * (margin + (1 - 2 * margin) * u)


class FQSecularBlock(nn.Module):
    """
    Multi-head FQ secular transform mapping in_dim -> out_dim (out_dim = num_heads * 2 * K),
    vectorized over heads.  Per-head secular size n is small and fixed-ish (n > in_per_head).

    in_dim       : input width (padded to num_heads * in_per_head).
    out_dim      : output width; out_dim = num_heads * 2 * K.
    in_per_head  : small per-head input slice (default 4) -> keeps n small.
    n            : secular operator size per head (auto >= in_per_head + 4).
    coupled      : if False, U_h = I (decoupled control).
    chunk        : optional row-chunk size to bound memory (used for edges).
    """

    def __init__(self, in_dim: int, out_dim: int, in_per_head: int = 4, n: int | None = None,
                 init_scale: float = 1.0, coupled: bool = True, seed: int = 0,
                 chunk: int | None = 30000, feat_mode: str = "eigenphase"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.feat_mode = feat_mode  # "eigenphase" (faithful, default) | "trace" (project-original ablation)
        self.in_per_head = in_per_head
        self.num_heads = math.ceil(in_dim / in_per_head)
        self.padded_in = self.num_heads * in_per_head
        # every head must contribute: take out_per_head features from EACH head (all input regions
        # reach the output and get gradient).  per-head features = 2K, so 2K >= out_per_head.
        self.out_per_head = math.ceil(out_dim / self.num_heads)
        self.K = max(1, math.ceil(self.out_per_head / 2))
        self.raw_out = self.num_heads * 2 * self.K
        self.n = n if n is not None else in_per_head + 4
        if self.n <= in_per_head:
            self.n = in_per_head + 4
        self.coupled = coupled
        self.init_scale = init_scale
        self.chunk = chunk

        H, n_, d = self.num_heads, self.n, in_per_head
        g = torch.Generator().manual_seed(seed)
        # positive-Grassmannian L per head via STP exponential chart: L_h = exp(s_h (x) alpha_h)
        self.s_raw = nn.Parameter(0.05 * torch.randn(H, n_, generator=g))       # -> increasing s
        self.alpha_raw = nn.Parameter(0.05 * torch.randn(H, d, generator=g))    # -> increasing alpha
        self.col_scale = nn.Parameter(torch.zeros(H, d))
        # unitary U_h = exp(i H_h), H_h Hermitian from complex A_h
        self.A_real = nn.Parameter(0.05 * torch.randn(H, n_, n_, generator=g))
        self.A_imag = nn.Parameter(0.05 * torch.randn(H, n_, n_, generator=g))

    # ---- positive-Grassmannian L for all heads: [H, n, in_per_head] ----
    def L(self) -> torch.Tensor:
        s = _increasing(self.s_raw, -1.0, 1.0)                  # [H, n]
        alpha = _increasing(self.alpha_raw, 0.2, 1.5)           # [H, d]
        logL = s.unsqueeze(-1) * alpha.unsqueeze(1)             # [H, n, d]
        L = torch.exp(logL) * torch.exp(self.col_scale).unsqueeze(1)
        L = L / L.norm(dim=1, keepdim=True).clamp_min(1e-8)     # column-normalize (positive scaling)
        return L * self.init_scale

    # ---- unitary U for all heads: [H, n, n] ----
    def U(self) -> torch.Tensor:
        if not self.coupled:
            eye = torch.eye(self.n, dtype=torch.complex64, device=self.A_real.device)
            return eye.unsqueeze(0).expand(self.num_heads, -1, -1)
        A = torch.complex(self.A_real, self.A_imag)
        Herm = 0.5 * (A + A.conj().transpose(-2, -1))
        return torch.linalg.matrix_exp(1j * Herm.to(torch.complex64))   # [H, n, n]

    def _moments(self, xh: torch.Tensor, L: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """
        xh : [B, H, in_per_head]  ->  [B, out_dim] trace-moment features (vectorized over heads).
        """
        # phases z = exp(2 pi i L x_h):  [B, H, n]
        phase = torch.einsum("bhd,hnd->bhn", xh, L)            # L x_h
        z = torch.exp(2j * PI * phase.to(torch.complex64))     # [B, H, n]
        M = z.unsqueeze(-1) * U.unsqueeze(0)                   # diag(z) U -> [B, H, n, n]
        P = M
        outs = []
        for k in range(self.K):
            tk = torch.diagonal(P, dim1=-2, dim2=-1).sum(-1)   # tr(M^{k+1}) [B, H]
            outs.append(tk)
            if k < self.K - 1:
                P = P @ M
        T = torch.stack(outs, dim=-1)                          # [B, H, K] complex
        feat = torch.cat([T.real, T.imag], dim=-1) / math.sqrt(self.n)   # [B, H, 2K]
        # take out_per_head from EVERY head (all heads/input regions contribute), then trim
        feat = feat[:, :, : self.out_per_head].reshape(feat.shape[0], self.num_heads * self.out_per_head)
        return feat[:, : self.out_dim]

    def _eigenphases(self, xh: torch.Tensor, L: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """FAITHFUL featurization: [cos theta_j, sin theta_j] of the coupled eigenphases of
        M(z)=diag(z)U (DIFFERENTIABLE via torch.linalg.eigvals).  xh [B,H,d] -> [B, out_dim]."""
        phase = torch.einsum("bhd,hnd->bhn", xh, L)            # L x_h
        z = torch.exp(2j * PI * phase.to(torch.complex64))     # [B, H, n]
        M = z.unsqueeze(-1) * U.unsqueeze(0)                   # diag(z) U -> [B, H, n, n]
        ev = torch.linalg.eigvals(M.reshape(-1, self.n, self.n)).reshape(M.shape[0], self.num_heads, self.n)
        theta, _ = torch.sort(torch.angle(ev), dim=-1)         # [B, H, n] sorted eigenphases
        feat = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)   # [B, H, 2n]
        feat = feat[:, :, : self.out_per_head].reshape(feat.shape[0], self.num_heads * self.out_per_head)
        return feat[:, : self.out_dim]

    def _feat(self, xh: torch.Tensor, L: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """Dispatch to the default eigenphase featurization or the project-original trace ablation."""
        if self.feat_mode == "trace":
            return self._moments(xh, L, U)
        return self._eigenphases(xh, L, U)

    @torch.no_grad()
    def secular_check(self, x: torch.Tensor) -> float:
        """Sanity: max |prod_j(1-e^{i theta_j}) - p_U(z)| over a batch (should be ~0)."""
        B = x.shape[0]
        if self.padded_in != self.in_dim:
            x = torch.cat([x, x.new_zeros(B, self.padded_in - self.in_dim)], dim=-1)
        xh = x.view(B, self.num_heads, self.in_per_head)
        L, U = self.L(), self.U()
        phase = torch.einsum("bhd,hnd->bhn", xh, L)
        z = torch.exp(2j * PI * phase.to(torch.complex64))     # [B, H, n]
        M = z.unsqueeze(-1) * U.unsqueeze(0)
        I = torch.eye(self.n, dtype=M.dtype, device=M.device)
        pU = torch.linalg.det(I - M)                           # p_U = det(I - diag(z)U)
        ev = torch.linalg.eigvals(M.reshape(-1, self.n, self.n)).reshape(B, self.num_heads, self.n)
        prod = torch.prod(1.0 - ev, dim=-1)
        return float((prod - pU).abs().max())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        if self.padded_in != self.in_dim:
            x = torch.cat([x, x.new_zeros(B, self.padded_in - self.in_dim)], dim=-1)
        xh = x.view(B, self.num_heads, self.in_per_head)
        L = self.L(); U = self.U()
        chunk = self.chunk if self.chunk is not None else B
        if B <= chunk and not (self.training and torch.is_grad_enabled()):
            return self._feat(xh, L, U)
        # chunk + gradient-checkpoint each chunk so the full-graph activation memory stays bounded
        outs = []
        use_ckpt = self.training and torch.is_grad_enabled()
        for i in range(0, B, chunk):
            xc = xh[i:i + chunk]
            if use_ckpt:
                outs.append(torch.utils.checkpoint.checkpoint(self._feat, xc, L, U,
                                                              use_reentrant=False))
            else:
                outs.append(self._feat(xc, L, U))
        return torch.cat(outs, dim=0)

    def min_minor(self) -> torch.Tensor:
        """Smallest d x d minor of L across heads (cone witness; STP chart => > 0)."""
        L = self.L()                                           # [H, n, d]
        import itertools
        H, n_, d = L.shape
        idx = list(itertools.combinations(range(n_), d))
        mins = []
        for I in idx[: min(len(idx), 64)]:                     # sample subsets for speed
            sub = L[:, list(I), :]                             # [H, d, d]
            mins.append(torch.linalg.det(sub).min())
        return torch.stack(mins).min()


# pairwise convenience (message step uses FQSecularBlock(2W+edge -> W) directly)
class FQSecularPair(nn.Module):
    def __init__(self, width, edge_dim=0, in_per_head=4, init_scale=1.0, coupled=True,
                 seed=0, chunk=200000):
        super().__init__()
        self.block = FQSecularBlock(2 * width + edge_dim, width, in_per_head=in_per_head,
                                    init_scale=init_scale, coupled=coupled, seed=seed, chunk=chunk)

    def forward(self, h_i, h_j, e=None):
        x = torch.cat([h_i, h_j] + ([e] if e is not None else []), dim=-1)
        return self.block(x)

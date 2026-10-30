"""
fqops.op_variety  --  Op-C : the FQ Lee-Yang VARIETY-COUPLING block (the trainable FQ primitive)

Op-C (this variety block) is the AKKV-FAITHFUL coupled invariant and is where the coupled message
should route -- NOT the op_secular trace moments t_k=tr(M^k), which are PROJECT-ORIGINAL (the Op-B
feature is the eigenphases themselves; see op_secular/op_eigenphase).  The Lee-Yang VARIETY RELATIONS
are also easy to train because they are BILINEAR in z = exp(2 pi i L x):

    p_j(z) = A_j z_j + B_j - C_j z_j z_{j+1} - D_j z_{j+1} ,   j = 1..n-1,

with (A,B,C,D)_j from the Moebius composition of the curve's upper-half-plane parameters a_j; the
relations vanish exactly on the quasicrystal.  z_j and z_j z_{j+1} expose cos/sin(L x) and their
products directly, so gradients to L flow.  The multiplicative z_j z_{j+1} term is the genuine FQ
coupling between adjacent torus coordinates -- no decoupled cosine basis reproduces it
(coupled=False drops it, C_j=0 -> decoupled control).

This is a VECTORIZED, chunked, drop-in replacement for FQSecularBlock with the same
(in_dim -> out_dim) interface: multi-head, small per-head input keeps n small, positive-Grassmannian
L per head (STP exponential chart), Moebius a_j per head.  Output per head = 2(n-1); the block
emits num_heads*2(n-1) features and truncates (param-free) to out_dim.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

PI = math.pi


def _softplus(x, floor=1e-4):
    return torch.nn.functional.softplus(x) + floor


def _increasing(raw, lo, hi):
    gaps = _softplus(raw)
    cs = torch.cumsum(gaps, dim=-1) - 0.5 * gaps
    u = cs / (cs[..., -1:] + 0.5 * gaps[..., -1:] + 1e-12)
    margin = 0.02
    return lo + (hi - lo) * (margin + (1 - 2 * margin) * u)


class FQVarietyBlock(nn.Module):
    """
    Multi-head FQ variety-coupling transform mapping in_dim -> out_dim, vectorized over heads.
    Compatible signature with FQSecularBlock so the GNN steps can swap primitive.
    """

    def __init__(self, in_dim: int, out_dim: int, in_per_head: int = 4, n: int | None = None,
                 init_scale: float = 2.0, coupled: bool = True, seed: int = 0,
                 chunk: int | None = 30000, phase_offset: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_per_head = in_per_head
        self.num_heads = math.ceil(in_dim / in_per_head)
        self.padded_in = self.num_heads * in_per_head
        # every head must contribute: take out_per_head features from EACH head (so all input
        # regions reach the output and receive gradient; otherwise later heads' inputs get zero grad)
        self.out_per_head = math.ceil(out_dim / self.num_heads)
        n_floor = in_per_head + 4
        n_need = math.ceil(self.out_per_head / 2) + 1          # per_head_out = 2(n-1) >= out_per_head
        self.n = max(n if n is not None else 0, n_floor, n_need)
        self.per_head_out = 2 * (self.n - 1)
        self.raw_out = self.num_heads * self.per_head_out
        self.coupled = coupled
        self.init_scale = init_scale
        self.chunk = chunk

        H, n_, d = self.num_heads, self.n, in_per_head
        g = torch.Generator().manual_seed(seed)
        # positive-Grassmannian L per head (STP exponential chart), as in op_secular
        self.s_raw = nn.Parameter(0.05 * torch.randn(H, n_, generator=g))
        self.alpha_raw = nn.Parameter(0.05 * torch.randn(H, d, generator=g))
        self.col_scale = nn.Parameter(torch.zeros(H, d))
        # Moebius curve parameters a_j per head (upper half-plane: im = softplus)
        self.a_re = nn.Parameter(0.3 * torch.randn(H, n_, generator=g))
        self.a_im_raw = nn.Parameter(0.5 + 0.1 * torch.randn(H, n_, generator=g))
        # learnable TORUS TRANSLATION phi_0 (Op-A): z = exp(2 pi i (L x + phi_0)).  Anchors WHERE the
        # quasicrystal Lambda(X,L) sits on R^d.  Without it the variety relation p(z=1)=0 maps any
        # near-zero input to exactly zero (dead channels); a phase offset moves z off 1 so zero/near-
        # zero features still produce signal.  Init to evenly-spaced phases around the circle so the n
        # torus coords sample diverse points even at x=0.  Faithful (a translation of the FQ point set),
        # no bias/Linear added.
        self.use_phase_offset = phase_offset
        if phase_offset:
            base = (torch.arange(n_).float() / n_).unsqueeze(0).repeat(H, 1)   # [H,n] evenly spaced
            self.phase_offset = nn.Parameter(base + 0.05 * torch.randn(H, n_, generator=g))
        else:
            self.register_parameter("phase_offset", None)

    def L(self) -> torch.Tensor:
        s = _increasing(self.s_raw, -1.0, 1.0)              # [H, n]
        alpha = _increasing(self.alpha_raw, 0.2, 1.5)       # [H, d]
        Lm = torch.exp(s.unsqueeze(-1) * alpha.unsqueeze(1)) * torch.exp(self.col_scale).unsqueeze(1)
        Lm = Lm / Lm.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return Lm * self.init_scale                         # [H, n, d]

    def _coeffs(self):
        """Moebius bilinear coefficients (A,B,C,D) per head per adjacent pair: [H, n-1] complex."""
        a = torch.complex(self.a_re, _softplus(self.a_im_raw))   # [H, n]
        ac = a.conj()
        aj, ak = a[:, :-1], a[:, 1:]
        acj, ack = ac[:, :-1], ac[:, 1:]
        A = -aj + ack
        B = acj - ack
        C = -aj + ak
        D = acj - ak
        if not self.coupled:
            C = torch.zeros_like(C)
        return A, B, C, D

    def _relations(self, xh, L, coeffs):
        """xh : [B, H, d] -> [B, out_dim] variety-relation features."""
        A, B, C, D = coeffs
        phase = torch.einsum("bhd,hnd->bhn", xh, L)         # L x_h
        if self.phase_offset is not None:
            phase = phase + self.phase_offset.unsqueeze(0)  # torus translation phi_0
        z = torch.exp(2j * PI * phase.to(torch.complex64))  # [B, H, n]
        zj, zk = z[:, :, :-1], z[:, :, 1:]                  # [B, H, n-1]
        p = (A.unsqueeze(0) * zj + B.unsqueeze(0)
             - C.unsqueeze(0) * zj * zk - D.unsqueeze(0) * zk)   # [B, H, n-1] complex
        feat = torch.cat([p.real, p.imag], dim=-1)          # [B, H, 2(n-1)]
        # take out_per_head from EVERY head (all heads/input regions contribute), then trim
        feat = feat[:, :, : self.out_per_head].reshape(feat.shape[0], self.num_heads * self.out_per_head)
        return feat[:, : self.out_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        if self.padded_in != self.in_dim:
            x = torch.cat([x, x.new_zeros(B, self.padded_in - self.in_dim)], dim=-1)
        xh = x.view(B, self.num_heads, self.in_per_head)
        L = self.L(); coeffs = self._coeffs()
        chunk = self.chunk if self.chunk is not None else B
        if B <= chunk:
            return self._relations(xh, L, coeffs)
        outs = []
        for i in range(0, B, chunk):
            outs.append(self._relations(xh[i:i + chunk], L, coeffs))
        return torch.cat(outs, dim=0)

    def min_minor(self) -> torch.Tensor:
        import itertools
        L = self.L()
        H, n_, d = L.shape
        idx = list(itertools.combinations(range(n_), d))[:64]
        mins = [torch.linalg.det(L[:, list(I), :]).min() for I in idx]
        return torch.stack(mins).min()

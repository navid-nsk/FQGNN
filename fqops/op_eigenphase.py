"""
fqops.op_eigenphase  --  Op-B (the FAITHFUL featurization): explicit coupled eigenphases

The eigenphase featurization theta_j(x) of M(x)=diag(exp(2 pi i L x))U is the PREFERRED Op-B
feature (proven continuous, monotone, real-analytic, Lipschitz-bounded -- Alon-Vinzant-gaps), NOT the
project-original trace moments.  `features` is a DIFFERENTIABLE training feature (grad flows through
eigvals); `phase_sensitivity` is the interpretability readout (which feature combinations co-rotate
the spectrum).  Same positive-Grassmannian L / unitary U as op_secular.FQSecularBlock; sanity check
prod_j(1 - e^{i theta_j}) = p_U via FQSecularBlock.secular_check.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from .op_secular import FQSecularBlock

PI = math.pi


class FQEigenphaseHead(nn.Module):
    """
    Eigenphase features [cos theta_j, sin theta_j, cos 2 theta_j, sin 2 theta_j] of the coupled
    secular operator, per head.  Reuses an FQSecularBlock's L and U so it is the SAME variety as the
    trained model (interpretability of the learned operator).
    """

    def __init__(self, secular_block: FQSecularBlock, harmonics: int = 2):
        super().__init__()
        self.block = secular_block
        self.H = secular_block.num_heads
        self.n = secular_block.n
        self.d = secular_block.in_per_head
        self.harmonics = harmonics

    def eigenphases(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, in_dim] -> eigenphases theta : [B, H, n] in (-pi, pi].  DIFFERENTIABLE:
        grad flows through torch.linalg.eigvals, so `features` is a faithful training feature."""
        B = x.shape[0]
        if self.block.padded_in != self.block.in_dim:
            x = torch.cat([x, x.new_zeros(B, self.block.padded_in - self.block.in_dim)], dim=-1)
        xh = x.view(B, self.H, self.d)
        L = self.block.L()
        U = self.block.U()
        phase = torch.einsum("bhd,hnd->bhn", xh, L)
        z = torch.exp(2j * PI * phase.to(torch.complex64))      # [B, H, n]
        M = z.unsqueeze(-1) * U.unsqueeze(0)                    # [B, H, n, n]
        ev = torch.linalg.eigvals(M.reshape(-1, self.n, self.n))
        ev = ev.reshape(B, self.H, self.n)
        return torch.angle(ev)                                  # [B, H, n]

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Cos/sin harmonics of the eigenphases -> [B, H * n * 2 * harmonics]."""
        theta = self.eigenphases(x)                             # [B, H, n]
        feats = []
        for k in range(1, self.harmonics + 1):
            feats.append(torch.cos(k * theta))
            feats.append(torch.sin(k * theta))
        out = torch.stack(feats, dim=-1)                        # [B, H, n, 2*harm]
        return out.reshape(out.shape[0], -1)

    @torch.no_grad()
    def phase_sensitivity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Finite-difference sensitivity of the mean eigenphase to each input feature -- the
        interpretability readout (which feature combinations rotate the spectrum).  -> [in_dim].
        """
        eps = 1e-3
        base = self.eigenphases(x).mean().item()
        sens = torch.zeros(self.block.in_dim, device=x.device)
        for j in range(self.block.in_dim):
            xp = x.clone(); xp[:, j] += eps
            sens[j] = (self.eigenphases(xp).mean().item() - base) / eps
        return sens

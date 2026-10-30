"""
fqcuda.fq_eig_cuda  --  CUDA Op-B (eigenphases) + Op-C (secular det) as a torch.autograd.Function.

A Blackwell (sm_120) per-thread n=6 unitary eigensolver for fq_imputer.secular_features, which
avoids the CPU round-trip `torch.linalg.eigvals(Mop.cpu())`:
  forward  : Faddeev-LeVerrier char-poly -> Aberth-Ehrlich roots -> sort -> inverse-iteration eigvecs.
  backward : M unitary => normal => orthonormal eigvecs, so dL/dM = sum_j i*cot_j*lam_j*v_jk*conj(v_jl).
Op-C p_U = det(I-M) = prod_j (1 - lam_j) rides along for FREE from the same eigenvalues.

The autograd.Function operates on Mop[N,6,6] complex (the thin chart boundary); grad flows back to
pos/L/U through eager autograd over Mop = diag(z)U. fp32 (production) + fp64 (gradcheck) only.

Build: fqcuda/cuda_eig/setup_fq_eig.py (module fq_eig_ops; see README.md).
"""
from __future__ import annotations

import os
import sys

import torch

# the standalone extension is built in place in fqcuda/cuda_eig and is found by RELATIVE path.
_EXT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda_eig")
if _EXT_DIR not in sys.path:
    sys.path.insert(0, _EXT_DIR)
import fq_eig_ops  # noqa: E402

_MAX_ITER = 60  # Aberth iterations (cubic convergence; 60 is ample headroom for n=6 unitary)


class OpBEig(torch.autograd.Function):
    """Eigenphases theta[N,6] (sorted) + secular det pU[N] of M=diag(z)U, with backward wrt M.

    forward(Mop): Mop complex [N,6,6]  ->  (theta[N,6] real, pU[N] complex)
    The backward returns grad wrt Mop (complex), which eager autograd chains to z, U, pos, L.
    """

    @staticmethod
    def forward(ctx, Mop):
        # Mop: complex [N,6,6]. Split into contiguous real/imag (the kernel takes 2 real tensors).
        real_dtype = torch.float64 if Mop.dtype == torch.complex128 else torch.float32
        N = Mop.shape[0]
        Mre = Mop.real.reshape(N, 36).contiguous().to(real_dtype)
        Mim = Mop.imag.reshape(N, 36).contiguous().to(real_dtype)
        theta, pUre, pUim, vre, vim, rre, rim = fq_eig_ops.opb_forward(Mre, Mim, _MAX_ITER)
        ctx.save_for_backward(vre, vim, rre, rim)
        ctx.cdtype = Mop.dtype
        pU = torch.complex(pUre, pUim)
        return theta, pU

    @staticmethod
    def backward(ctx, grad_theta, grad_pU):
        vre, vim, rre, rim = ctx.saved_tensors
        # grad_theta: real [N,6]; grad_pU: complex [N] (its real/imag are dL/d pU.re, dL/d pU.im)
        gt = grad_theta.contiguous() if grad_theta is not None else torch.zeros_like(rre)
        if grad_pU is None:
            gpr = torch.zeros(rre.shape[0], dtype=rre.dtype, device=rre.device)
            gpi = torch.zeros_like(gpr)
        else:
            gpr = grad_pU.real.contiguous()
            gpi = grad_pU.imag.contiguous()
        gMre, gMim = fq_eig_ops.opb_backward(gt, gpr, gpi, vre, vim, rre, rim)
        N = gMre.shape[0]
        gM = torch.complex(gMre, gMim).reshape(N, 6, 6).to(ctx.cdtype)
        return gM


def opb_theta(Mop: torch.Tensor) -> torch.Tensor:
    """Gate-facing candidate: theta[N,6] only (matches op_b_reference signature)."""
    theta, _ = OpBEig.apply(Mop)
    return theta


def secular_features_cuda(self, pos, L):
    """Drop-in replacement for FQTrajectoryImputer.secular_features using the CUDA eigensolver.

    Identical math to the reference (fq_imputer.py, secular_features): builds Mop=diag(z)U in PyTorch (so grad
    flows to pos, L, U=A_real/A_imag via eager autograd), then the CUDA kernel does Op-B + Op-C.
    NO `s` scale on z; z row-scales U; pU=det(I-M), p_U(0)=1.
    """
    PI = 3.141592653589793
    U = self.secular.unitary()
    z = torch.exp(2j * PI * (pos @ L.t()).to(torch.complex64))   # Phi_L(pos), [N, n]  (NO s)
    Mop = z.unsqueeze(-1) * U                                     # diag(z) U  (row scaling), [N, n, n]
    theta, pU = OpBEig.apply(Mop)                                 # Op-B eigenphases + Op-C det (CUDA)
    return theta.to(pos.dtype), torch.stack([pU.real, pU.imag], dim=-1).to(pos.dtype)

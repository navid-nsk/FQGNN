"""fqcuda.fq_phase_cuda  --  fused QFE (Op-A') + Dmu velocity phase-engine as a torch.autograd.Function.

Drop-in CUDA path for fq_imputer.qfe()/velocity() (references: verify_kernel qfe_spectral_ref/velocity_ref):
a Blackwell (sm_120) fused phase kernel: one shared phase table ph=2*pi*s*(pos@xi^T) feeds both
qfe[N,2M] and vel[N,4M].  Faithful to the Op-A' synthesis and the derivative (velocity) channel:

  - QFE envelope exp(clamp(<rho,x>,-3,3)); backward zeros grad_rho/grad_pos outside [-3,3].
  - Dmu velocity (2*pi*i*xi_d)*c, NO s, NO envelope; i = 90deg rotation -> cv.re=-2pi*xi_d*c.im,
    cv.im=+2pi*xi_d*c.re; layout [re_dx, im_dx, re_dy, im_dy].
  - xi are irrational (L^T k); c is a FIXED complex INPUT, Re AND Im load-bearing.

The autograd.Function operates on REAL tensors (pos, xi, cr, ci, s, rho); the Python wrappers split
the complex amplitude c into (Re c, Im c) and recombine grad_cr/grad_ci into a complex grad_c.
Dual-dtype: fp32 (complex64 c) + fp64 (complex128 c); AT_DISPATCH_FLOATING_TYPES covers both.

ext (fq_phase_ops) found by a path RELATIVE to this wrapper (cuda_phase/)."""
from __future__ import annotations

import os
import sys

import torch

_EXT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda_phase")
if _EXT_DIR not in sys.path:
    sys.path.insert(0, _EXT_DIR)
import fq_phase_ops  # noqa: E402


class PhaseFn(torch.autograd.Function):
    """Fused QFE+Dmu on real tensors.

    forward(pos[N,2], xi[M,2], cr[M], ci[M], rho[2], s scalar) -> (qfe[N,2M], vel[N,4M]).
    """

    @staticmethod
    def forward(ctx, pos, xi, cr, ci, rho, s):
        pos = pos.contiguous(); xi = xi.contiguous()
        cr = cr.contiguous(); ci = ci.contiguous(); rho = rho.contiguous()
        s = s.reshape(()).contiguous()
        qfe, vel = fq_phase_ops.phase_forward(pos, xi, cr, ci, rho, s)
        ctx.save_for_backward(pos, xi, cr, ci, rho, s)
        return qfe, vel

    @staticmethod
    def backward(ctx, gQ, gV):
        pos, xi, cr, ci, rho, s = ctx.saved_tensors
        gQ = gQ.contiguous() if gQ is not None else torch.zeros(pos.shape[0], 2 * xi.shape[0],
                                                                dtype=pos.dtype, device=pos.device)
        gV = gV.contiguous() if gV is not None else torch.zeros(pos.shape[0], 4 * xi.shape[0],
                                                                dtype=pos.dtype, device=pos.device)
        g_pos, g_xi, g_cr, g_ci, g_s, g_rho = fq_phase_ops.phase_backward(
            pos, xi, cr, ci, rho, s, gQ, gV)
        # match the scalar shape of the input s (gate passes s as a 0-dim tensor)
        g_s = g_s.reshape(s.shape)
        return g_pos, g_xi, g_cr, g_ci, g_rho, g_s


def _split_c(c):
    """Complex c[M] -> (cr, ci) real, dtype matching c's real part."""
    return c.real.contiguous(), c.imag.contiguous()


def qfe_velocity_cuda(pos, xi, c, s, rho):
    """Fused: returns (qfe[N,2M], vel[N,4M]). c complex; grad recombined into complex grad_c."""
    cr, ci = _split_c(c)
    qfe, vel = PhaseFn.apply(pos, xi, cr, ci, rho, s)
    return qfe, vel


def qfe_cuda(pos, xi, c, s, rho):
    """Gate-facing QFE candidate (signature == qfe_spectral_ref): returns qfe[N,2M]."""
    cr, ci = _split_c(c)
    qfe, _ = PhaseFn.apply(pos, xi, cr, ci, rho, s)
    return qfe


def velocity_cuda(pos, xi, c, s):
    """Gate-facing Dmu velocity candidate (signature == velocity_ref): returns vel[N,4M].

    velocity_ref has NO rho; supply a zero rho (envelope unused by the velocity output, and the
    velocity backward path never touches rho since gQ=0)."""
    cr, ci = _split_c(c)
    rho = torch.zeros(2, dtype=pos.dtype, device=pos.device)
    _, vel = PhaseFn.apply(pos, xi, cr, ci, rho, s)
    return vel

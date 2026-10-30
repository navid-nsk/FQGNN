"""fqcuda.fq_opd_cuda  --  Op-D twisted-Eberlein convolution as a torch.autograd.Function.

Drop-in CUDA path for the fq_imputer Op-D block (reference: verify_kernel.opd_ref): per-edge phase kernel
K = cos(phk) @ (Re c * g_cos) - sin(phk) @ (Im c * g_sin), phk = 2*pi*s*((coord[dst]-coord[src]) @ xi^T),
scatter-add m[src]*K to dst, intensive /deg. Faithful to the Op-D definition:
two REAL taps Gc=Re(c)*g_cos, Gs=Im(c)*g_sin; intensive /deg from the SAME dst scatter;
xi irrational, c fixed complex. Shared-param grads accumulated in DOUBLE internally.

Dual-dtype: fp32 (complex64 c) + fp64 (complex128 c) via AT_DISPATCH_FLOATING_TYPES.
ext (fq_opd_ops) found by a path RELATIVE to this wrapper (cuda_opd/)."""
from __future__ import annotations

import os
import sys

import torch

_EXT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda_opd")
if _EXT_DIR not in sys.path:
    sys.path.insert(0, _EXT_DIR)
import fq_opd_ops  # noqa: E402


class OpDFn(torch.autograd.Function):
    """forward(m[N,W], coord[N,2], xi[M,2], cr[M], ci[M], gcos[M,W], gsin[M,W], s, src[E], dst[E],
    intensive) -> agg[N,W]."""

    @staticmethod
    def forward(ctx, m, coord, xi, cr, ci, gcos, gsin, s, src, dst, intensive):
        args = [t.contiguous() for t in (m, coord, xi, cr, ci, gcos, gsin)]
        s = s.reshape(()).contiguous()
        src = src.contiguous(); dst = dst.contiguous()
        agg = fq_opd_ops.opd_forward(*args, s, src, dst, int(intensive))
        ctx.save_for_backward(*args, s, src, dst)
        ctx.intensive = int(intensive)
        return agg

    @staticmethod
    def backward(ctx, grad_out):
        m, coord, xi, cr, ci, gcos, gsin, s, src, dst = ctx.saved_tensors
        gm, gco, gxi, gcr, gci, ggc, ggs, gs = fq_opd_ops.opd_backward(
            m, coord, xi, cr, ci, gcos, gsin, s, src, dst, grad_out.contiguous(), ctx.intensive)
        gs = gs.reshape(s.shape)
        return gm, gco, gxi, gcr, gci, ggc, ggs, gs, None, None, None


def opd_cuda(m, coord, xi, c, s, g_cos, g_sin, edge_index, intensive=True):
    """Gate-facing Op-D candidate (signature == opd_ref): c complex; grad recombined into complex grad_c."""
    cr, ci = c.real.contiguous(), c.imag.contiguous()
    src, dst = edge_index[0].contiguous(), edge_index[1].contiguous()
    return OpDFn.apply(m, coord, xi, cr, ci, g_cos, g_sin, s, src, dst, intensive)

// pybind module for the Op-D twisted-Eberlein conv kernel (module: fq_opd_ops).
// .cpp is compiled by cl, so <torch/extension.h> is fine here (the .cu uses common.cuh's
// <torch/types.h> to dodge the torch-2.11/MSVC C2872 under nvcc).
#include <torch/extension.h>
#include <vector>

at::Tensor opd_forward(at::Tensor m, at::Tensor coord, at::Tensor xi, at::Tensor cr, at::Tensor ci,
                       at::Tensor gcos, at::Tensor gsin, at::Tensor s,
                       at::Tensor src, at::Tensor dst, int64_t intensive);
std::vector<at::Tensor> opd_backward(at::Tensor m, at::Tensor coord, at::Tensor xi,
                                     at::Tensor cr, at::Tensor ci, at::Tensor gcos, at::Tensor gsin,
                                     at::Tensor s, at::Tensor src, at::Tensor dst,
                                     at::Tensor grad_out, int64_t intensive);

PYBIND11_MODULE(fq_opd_ops, m) {
    m.def("opd_forward", &opd_forward,
          "Op-D twisted-Eberlein conv forward: agg[N,W] (CUDA)");
    m.def("opd_backward", &opd_backward,
          "Op-D backward: grads wrt m, coord, xi, c(re,im), g_cos, g_sin, s (CUDA)");
}

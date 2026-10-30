// pybind module for the fused QFE+Dmu phase kernel (module: fq_phase_ops).
// .cpp is compiled by cl, so <torch/extension.h> is fine here (the .cu uses common.cuh's
// <torch/types.h> to dodge the torch-2.11/MSVC C2872 under nvcc).
#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> phase_forward(at::Tensor pos, at::Tensor xi, at::Tensor cr, at::Tensor ci,
                                      at::Tensor rho, at::Tensor s);
std::vector<at::Tensor> phase_backward(at::Tensor pos, at::Tensor xi, at::Tensor cr, at::Tensor ci,
                                       at::Tensor rho, at::Tensor s, at::Tensor gQ, at::Tensor gV);

PYBIND11_MODULE(fq_phase_ops, m) {
    m.def("phase_forward", &phase_forward,
          "Fused QFE+Dmu forward: qfe[N,2M], vel[N,4M] from shared phase table (CUDA)");
    m.def("phase_backward", &phase_backward,
          "Fused QFE+Dmu backward: grads wrt pos, xi, c(re,im), s, rho (CUDA)");
}

// pybind module for the Op-B eigenphase kernel (module: fq_eig_ops).
// .cpp is compiled by cl, so <torch/extension.h> is fine here (the .cu uses common.cuh's
// <torch/types.h> to dodge the torch-2.11/MSVC C2872 under nvcc).
#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> opb_forward(at::Tensor Mre, at::Tensor Mim, int64_t max_iter);
std::vector<at::Tensor> opb_backward(at::Tensor grad_theta, at::Tensor grad_pUre, at::Tensor grad_pUim,
                                     at::Tensor vre, at::Tensor vim, at::Tensor rre, at::Tensor rim);

PYBIND11_MODULE(fq_eig_ops, m) {
    m.def("opb_forward", &opb_forward,
          "Op-B eigenphases forward: theta[N,6], pU(re,im), eigvecs+roots stash (CUDA)");
    m.def("opb_backward", &opb_backward,
          "Op-B backward: grad wrt M (re,im) from grad_theta + grad_pU (CUDA)");
}

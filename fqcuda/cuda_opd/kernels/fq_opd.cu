// fq_opd.cu  --  Op-D twisted-Eberlein FQ-spectral convolution.
//
// FORWARD (per edge e=(src,dst)):
//   phk = 2*pi*s*( (coord[dst]-coord[src]) @ xi^T )          [M]   (xi irrational; no FFT/period)
//   Gc  = Re(c) (x) g_cos   [M,W] ;  Gs = Im(c) (x) g_sin   [M,W]   (TWO real taps)
//   K   = cos(phk) @ Gc - sin(phk) @ Gs                      [W]
//   agg[dst] += m[src] * K   (index_add)
//   intensive: agg /= deg.clamp_min(1.0) ; deg from the SAME dst index_add
//
// BACKWARD reproduces torch.autograd of opd_ref EXACTLY: grads wrt m, coord, xi, c(re,im), g_cos, g_sin, s.
//   The /deg is a per-dst constant w.r.t. the summed quantity, so grad_agg_pre[dst] = grad_out[dst]/deg[dst].
//
// fp32 (sincosf) + fp64 (sincos).  NO fast-math, NO TF32.
// large shared-param reductions (xi, c, g_cos, g_sin, s) accumulated in DOUBLE (precision-exact).
// .cu MUST NOT include <torch/extension.h> -- use common.cuh (<torch/types.h>).

#include "common.cuh"

#define TWO_PI 6.283185307179586

__device__ __forceinline__ void sincos_t(float a, float* s, float* c){ sincosf(a,s,c); }
__device__ __forceinline__ void sincos_t(double a, double* s, double* c){ sincos(a,s,c); }
__device__ __forceinline__ void atomadd(float* p, float v){ atomicAdd(p, v); }
__device__ __forceinline__ void atomadd(double* p, double v){ atomicAdd(p, v); }

// =======================================================================================
// degree (per dst), via the SAME dst index_add
// =======================================================================================
template <typename T>
__global__ void deg_kernel(const int64_t* __restrict__ dst, int E, T* __restrict__ deg) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    atomadd(&deg[dst[e]], (T)1);
}

// =======================================================================================
// FORWARD: one thread per edge -> accumulate weighted = m[src]*K into agg[dst] (atomic).
// (division by deg done by a separate elementwise kernel after.)
// =======================================================================================
template <typename T>
__global__ void opd_forward_kernel(
        const T* __restrict__ m,        // [N,W]
        const T* __restrict__ coord,    // [N,2]
        const T* __restrict__ xi,       // [M,2]
        const T* __restrict__ cr,       // [M]
        const T* __restrict__ ci,       // [M]
        const T* __restrict__ gcos,     // [M,W]
        const T* __restrict__ gsin,     // [M,W]
        T s, const int64_t* __restrict__ src, const int64_t* __restrict__ dst,
        int E, int M, int W,
        T* __restrict__ agg)            // [N,W]
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    int64_t is = src[e], id = dst[e];
    T d0 = coord[2*id+0]-coord[2*is+0];
    T d1 = coord[2*id+1]-coord[2*is+1];
    T a = (T)TWO_PI * s;
    // K[w] = sum_k cos(phk)*Re(c_k)*gcos[k,w] - sin(phk)*Im(c_k)*gsin[k,w]
    // accumulate per-edge contributions to agg[dst, w] = m[src,w]*K[w]
    for (int w = 0; w < W; ++w) {
        T Kw = 0;
        for (int k = 0; k < M; ++k) {
            T ph = a * (d0*xi[2*k+0] + d1*xi[2*k+1]);
            T sp, cp; sincos_t(ph, &sp, &cp);
            Kw += cp * cr[k] * gcos[k*W+w] - sp * ci[k] * gsin[k*W+w];
        }
        atomadd(&agg[id*W+w], m[is*W+w] * Kw);
    }
}

// divide agg by deg.clamp_min(1.0)  (intensive)
template <typename T>
__global__ void intensive_kernel(T* __restrict__ agg, const T* __restrict__ deg, int N, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N*W) return;
    int n = idx / W;
    T dg = deg[n]; if (dg < (T)1) dg = (T)1;
    agg[idx] /= dg;
}

// =======================================================================================
// BACKWARD: one thread per edge. grad_out [N,W]; for intensive, grad_pre[dst] = grad_out[dst]/deg[dst].
//   weighted[e,w] = m[src,w]*K[e,w] ; agg_pre[dst] += weighted
//   K[e,w] = sum_k cos(phk)*cr_k*gcos[k,w] - sin(phk)*ci_k*gsin[k,w]
//   grad_m[src,w]    += go * K[e,w]
//   grad_K[e,w]       = go * m[src,w]
//   grad_gcos[k,w]   += grad_K * cos(phk)*cr_k       (atomic)
//   grad_gsin[k,w]   += grad_K * (-sin(phk))*ci_k    (atomic)
//   grad_cr[k]       += sum_w grad_K * cos(phk)*gcos[k,w]   (atomic)
//   grad_ci[k]       += sum_w grad_K * (-sin(phk))*gsin[k,w](atomic)
//   grad_phk[e,k]     = sum_w grad_K * ( -sin(phk)*cr_k*gcos[k,w] - cos(phk)*ci_k*gsin[k,w] )
//   phk = a*(delta . xi_k), a=2pi*s, delta=coord[dst]-coord[src]
//     grad_s          += grad_phk * 2pi*(delta.xi_k)       (atomic)
//     grad_xi[k,:]    += grad_phk * a * delta              (atomic)
//     grad_coord[dst] += grad_phk * a * xi_k ; grad_coord[src] -= same   (atomic)
// =======================================================================================
template <typename T, typename AccT>
__global__ void opd_backward_kernel(
        const T* __restrict__ m, const T* __restrict__ coord,
        const T* __restrict__ xi, const T* __restrict__ cr, const T* __restrict__ ci,
        const T* __restrict__ gcos, const T* __restrict__ gsin,
        T s, const int64_t* __restrict__ src, const int64_t* __restrict__ dst,
        const T* __restrict__ deg, const T* __restrict__ grad_out,  // [N,W]
        int E, int M, int W, int intensive,
        T* __restrict__ grad_m,          // [N,W]   (atomic)
        AccT* __restrict__ grad_coord,   // [N,2]   (atomic, double)
        AccT* __restrict__ grad_xi,      // [M,2]   (atomic, double)
        AccT* __restrict__ grad_cr,      // [M]
        AccT* __restrict__ grad_ci,      // [M]
        AccT* __restrict__ grad_gcos,    // [M,W]   (atomic, double)
        AccT* __restrict__ grad_gsin,    // [M,W]
        AccT* __restrict__ grad_s)       // [1]
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    int64_t is = src[e], id = dst[e];
    T d0 = coord[2*id+0]-coord[2*is+0];
    T d1 = coord[2*id+1]-coord[2*is+1];
    T a = (T)TWO_PI * s;

    // per-dst intensive scale
    T scale = (T)1;
    if (intensive) { T dg = deg[id]; scale = (T)1 / (dg < (T)1 ? (T)1 : dg); }

    AccT gxi0 = 0, gxi1 = 0;     // grad_xi for this edge (sum over k done per-k below, but xi is per-k -> atomic per k)
    AccT gcoord0 = 0, gcoord1 = 0;  // grad to delta -> coord
    AccT gs = 0;

    for (int k = 0; k < M; ++k) {
        T xk0 = xi[2*k+0], xk1 = xi[2*k+1];
        T dotk = d0*xk0 + d1*xk1;
        T ph = a * dotk;
        T sp, cp; sincos_t(ph, &sp, &cp);
        T crk = cr[k], cik = ci[k];

        AccT gcr_k = 0, gci_k = 0;   // grad to cr_k, ci_k from this edge
        AccT gphk = 0;               // grad to phk for this k

        for (int w = 0; w < W; ++w) {
            T go = grad_out[id*W+w] * scale;     // grad_pre[dst,w]
            T grad_K = go * m[is*W+w];           // dweighted/dK = m[src,w]
            // grad_m[src,w] += go * K[e,w] -- but need K[e,w]; recompute contribution incrementally
            // (we accumulate grad_m via K below)
            T gck = gcos[k*W+w], gsk = gsin[k*W+w];
            // grad to taps
            atomadd(&grad_gcos[k*W+w], (AccT)((AccT)grad_K * (AccT)cp * (AccT)crk));
            atomadd(&grad_gsin[k*W+w], (AccT)((AccT)grad_K * (AccT)(-sp) * (AccT)cik));
            // grad to cr,ci
            gcr_k += (AccT)grad_K * (AccT)cp * (AccT)gck;
            gci_k += (AccT)grad_K * (AccT)(-sp) * (AccT)gsk;
            // grad to phk
            gphk += (AccT)grad_K * ( -(AccT)sp*(AccT)crk*(AccT)gck - (AccT)cp*(AccT)cik*(AccT)gsk );
        }
        atomadd(&grad_cr[k], gcr_k);
        atomadd(&grad_ci[k], gci_k);
        // phk -> s, xi, coord
        gs += gphk * (AccT)TWO_PI * (AccT)dotk;
        atomadd(&grad_xi[2*k+0], gphk * (AccT)a * (AccT)d0);
        atomadd(&grad_xi[2*k+1], gphk * (AccT)a * (AccT)d1);
        gcoord0 += gphk * (AccT)a * (AccT)xk0;
        gcoord1 += gphk * (AccT)a * (AccT)xk1;
    }

    // grad_m: grad_m[src,w] += go * K[e,w].  Recompute K[e,w] in a second small loop.
    for (int w = 0; w < W; ++w) {
        T go = grad_out[id*W+w] * scale;
        T Kw = 0;
        for (int k = 0; k < M; ++k) {
            T ph = a * (d0*xi[2*k+0] + d1*xi[2*k+1]);
            T sp, cp; sincos_t(ph, &sp, &cp);
            Kw += cp * cr[k] * gcos[k*W+w] - sp * ci[k] * gsin[k*W+w];
        }
        atomadd(&grad_m[is*W+w], go * Kw);
    }

    // delta = coord[dst]-coord[src] -> grad_coord[dst] += gcoord ; grad_coord[src] -= gcoord
    atomadd(&grad_coord[2*id+0], gcoord0);
    atomadd(&grad_coord[2*id+1], gcoord1);
    atomadd(&grad_coord[2*is+0], -gcoord0);
    atomadd(&grad_coord[2*is+1], -gcoord1);
    atomadd(&grad_s[0], gs);
}

// =======================================================================================
// host launchers (dispatched fp32/fp64)
// =======================================================================================
#include <vector>

at::Tensor opd_forward(at::Tensor m, at::Tensor coord, at::Tensor xi, at::Tensor cr, at::Tensor ci,
                       at::Tensor gcos, at::Tensor gsin, at::Tensor s,
                       at::Tensor src, at::Tensor dst, int64_t intensive) {
    FQ_CHECK_INPUT(m); FQ_CHECK_INPUT(coord); FQ_CHECK_INPUT(xi);
    FQ_CHECK_INPUT(cr); FQ_CHECK_INPUT(ci); FQ_CHECK_INPUT(gcos); FQ_CHECK_INPUT(gsin);
    int N = coord.size(0), M = xi.size(0), W = m.size(1), E = src.size(0);
    auto opt = m.options();
    auto agg = at::zeros({N, W}, opt);
    auto deg = at::zeros({N}, opt);
    auto srcl = src.to(at::kLong), dstl = dst.to(at::kLong);
    int t = 256;
    AT_DISPATCH_FLOATING_TYPES(m.scalar_type(), "opd_deg", ([&]{
        deg_kernel<scalar_t><<<(E+t-1)/t, t>>>(dstl.data_ptr<int64_t>(), E, deg.data_ptr<scalar_t>());
    }));
    AT_DISPATCH_FLOATING_TYPES(m.scalar_type(), "opd_forward", ([&]{
        scalar_t sval = s.item<scalar_t>();
        opd_forward_kernel<scalar_t><<<(E+t-1)/t, t>>>(
            m.data_ptr<scalar_t>(), coord.data_ptr<scalar_t>(), xi.data_ptr<scalar_t>(),
            cr.data_ptr<scalar_t>(), ci.data_ptr<scalar_t>(),
            gcos.data_ptr<scalar_t>(), gsin.data_ptr<scalar_t>(), sval,
            srcl.data_ptr<int64_t>(), dstl.data_ptr<int64_t>(), E, M, W, agg.data_ptr<scalar_t>());
    }));
    if (intensive) {
        AT_DISPATCH_FLOATING_TYPES(m.scalar_type(), "opd_intensive", ([&]{
            intensive_kernel<scalar_t><<<(N*W+t-1)/t, t>>>(
                agg.data_ptr<scalar_t>(), deg.data_ptr<scalar_t>(), N, W);
        }));
    }
    FQ_DEBUG_SYNC();
    return agg;
}

std::vector<at::Tensor> opd_backward(at::Tensor m, at::Tensor coord, at::Tensor xi,
                                     at::Tensor cr, at::Tensor ci, at::Tensor gcos, at::Tensor gsin,
                                     at::Tensor s, at::Tensor src, at::Tensor dst,
                                     at::Tensor grad_out, int64_t intensive) {
    FQ_CHECK_INPUT(m); FQ_CHECK_INPUT(coord); FQ_CHECK_INPUT(grad_out);
    int N = coord.size(0), M = xi.size(0), W = m.size(1), E = src.size(0);
    auto opt = m.options();
    auto optd = opt.dtype(at::kDouble);
    auto deg = at::zeros({N}, opt);
    auto srcl = src.to(at::kLong), dstl = dst.to(at::kLong);
    int t = 256;
    AT_DISPATCH_FLOATING_TYPES(m.scalar_type(), "opd_deg_b", ([&]{
        deg_kernel<scalar_t><<<(E+t-1)/t, t>>>(dstl.data_ptr<int64_t>(), E, deg.data_ptr<scalar_t>());
    }));
    auto grad_m     = at::zeros({N, W}, opt);
    auto grad_coord = at::zeros({N, 2}, optd);
    auto grad_xi    = at::zeros({M, 2}, optd);
    auto grad_cr    = at::zeros({M}, optd);
    auto grad_ci    = at::zeros({M}, optd);
    auto grad_gcos  = at::zeros({M, W}, optd);
    auto grad_gsin  = at::zeros({M, W}, optd);
    auto grad_s     = at::zeros({1}, optd);
    AT_DISPATCH_FLOATING_TYPES(m.scalar_type(), "opd_backward", ([&]{
        scalar_t sval = s.item<scalar_t>();
        opd_backward_kernel<scalar_t, double><<<(E+t-1)/t, t>>>(
            m.data_ptr<scalar_t>(), coord.data_ptr<scalar_t>(), xi.data_ptr<scalar_t>(),
            cr.data_ptr<scalar_t>(), ci.data_ptr<scalar_t>(),
            gcos.data_ptr<scalar_t>(), gsin.data_ptr<scalar_t>(), sval,
            srcl.data_ptr<int64_t>(), dstl.data_ptr<int64_t>(), deg.data_ptr<scalar_t>(),
            grad_out.data_ptr<scalar_t>(), E, M, W, (int)intensive,
            grad_m.data_ptr<scalar_t>(), grad_coord.data_ptr<double>(),
            grad_xi.data_ptr<double>(), grad_cr.data_ptr<double>(), grad_ci.data_ptr<double>(),
            grad_gcos.data_ptr<double>(), grad_gsin.data_ptr<double>(), grad_s.data_ptr<double>());
    }));
    FQ_DEBUG_SYNC();
    auto sd = m.scalar_type();
    return {grad_m, grad_coord.to(sd), grad_xi.to(sd), grad_cr.to(sd), grad_ci.to(sd),
            grad_gcos.to(sd), grad_gsin.to(sd), grad_s.to(sd)};
}

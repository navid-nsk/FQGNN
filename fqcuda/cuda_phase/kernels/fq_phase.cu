// fq_phase.cu  --  fused QFE (Op-A') + Dmu velocity phase-engine kernel.
//
// FORWARD (one thread per node n; loop over M frequencies):
//   shared phase table  ph_{n,k} = 2*pi*s*(pos_n . xi_k)        (xi = L^T k, IRRATIONAL -- no FFT/period)
//   QFE (Op-A'):  e = exp(clamp(<rho,x>,-3,3));
//                 q_re[k] = (cr_k cos - ci_k sin) * e ;  q_im[k] = (cr_k sin + ci_k cos) * e
//                 out qfe[N,2M] = [ q_re(M) , q_im(M) ]
//   Dmu velocity: cv = (2*pi*xi_{k,d}) * (i * c_k)  -> cv.re = -2*pi*xi_d*ci ; cv.im = +2*pi*xi_d*cr
//                 NO s scale, NO envelope.  per d in {0,1}:
//                   vr = cv.re cos - cv.im sin ;  vi = cv.re sin + cv.im cos
//                 out vel[N,4M] = [ re_dx(M), im_dx(M), re_dy(M), im_dy(M) ]  (d outer, [re,im] inner)
//
// BACKWARD reproduces torch.autograd of qfe_spectral_ref + velocity_ref EXACTLY.
//   envelope clamp gradient is masked: grad_rho/grad_pos via env are zero outside [-3,3].
//
// fp32 (sincosf) + fp64 (sincos).  NO fast-math, NO TF32.
// .cu MUST NOT include <torch/extension.h> -- use common.cuh (<torch/types.h>).

#include "common.cuh"

#define TWO_PI 6.283185307179586

// sincos dispatch (full precision, NOT __sinf)
__device__ __forceinline__ void sincos_t(float a, float* s, float* c){ sincosf(a,s,c); }
__device__ __forceinline__ void sincos_t(double a, double* s, double* c){ sincos(a,s,c); }

// atomicAdd dispatch (double is native on sm_60+)
__device__ __forceinline__ void atomadd(float* p, float v){ atomicAdd(p, v); }
__device__ __forceinline__ void atomadd(double* p, double v){ atomicAdd(p, v); }

// =======================================================================================
// FORWARD
// =======================================================================================
template <typename T>
__global__ void phase_forward_kernel(
        const T* __restrict__ pos,   // [N,2]
        const T* __restrict__ xi,    // [M,2]
        const T* __restrict__ cr,    // [M]
        const T* __restrict__ ci,    // [M]
        const T* __restrict__ rho,   // [2]
        T s, int N, int M,
        T* __restrict__ qfe,         // [N,2M]
        T* __restrict__ vel)         // [N,4M]
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    T x0 = pos[2*n+0], x1 = pos[2*n+1];
    T a = (T)TWO_PI * s;
    // envelope (QFE only)
    T u = rho[0]*x0 + rho[1]*x1;
    T uc = u < (T)(-3) ? (T)(-3) : (u > (T)3 ? (T)3 : u);
    T e = exp(uc);
    const T twopi = (T)TWO_PI;
    for (int k = 0; k < M; ++k) {
        T xk0 = xi[2*k+0], xk1 = xi[2*k+1];
        T ph = a * (x0*xk0 + x1*xk1);
        T sp, cp; sincos_t(ph, &sp, &cp);
        T crk = cr[k], cik = ci[k];
        // QFE
        T q_re = (crk*cp - cik*sp) * e;
        T q_im = (crk*sp + cik*cp) * e;
        qfe[n*2*M + k]     = q_re;
        qfe[n*2*M + M + k] = q_im;
        // Dmu velocity, per d
        #pragma unroll
        for (int d = 0; d < 2; ++d) {
            T xid = (d==0) ? xk0 : xk1;
            T cvre = -twopi * xid * cik;   // (i*c).re scaled
            T cvim =  twopi * xid * crk;
            T v_re = cvre*cp - cvim*sp;
            T v_im = cvre*sp + cvim*cp;
            // layout: [re_dx, im_dx, re_dy, im_dy]  (block size M each, d outer)
            vel[n*4*M + (2*d+0)*M + k] = v_re;
            vel[n*4*M + (2*d+1)*M + k] = v_im;
        }
    }
}

// =======================================================================================
// BACKWARD
//   node-local grads (grad_pos, accumulated per-node then written): pos[N,2]
//   shared-parameter grads (atomicAdd): grad_xi[M,2], grad_cr[M], grad_ci[M], grad_s[1], grad_rho[2]
//
//   The shared-parameter grads are LARGE reductions over all N nodes; to keep the fp32 path within
//   the gate's ABSOLUTE 1e-3 tol they are accumulated in DOUBLE (AccT) even when scalar_t=float
//   (precision-exact, NOT a low-precision shortcut -- the FQ values stay scalar_t).
// =======================================================================================
template <typename T, typename AccT>
__global__ void phase_backward_kernel(
        const T* __restrict__ pos,   // [N,2]
        const T* __restrict__ xi,    // [M,2]
        const T* __restrict__ cr,    // [M]
        const T* __restrict__ ci,    // [M]
        const T* __restrict__ rho,   // [2]
        T s, int N, int M,
        const T* __restrict__ gQ,    // [N,2M]
        const T* __restrict__ gV,    // [N,4M]
        T* __restrict__ grad_pos,    // [N,2]
        AccT* __restrict__ grad_xi,  // [M,2]  (double accumulator)
        AccT* __restrict__ grad_cr,  // [M]
        AccT* __restrict__ grad_ci,  // [M]
        AccT* __restrict__ grad_s,   // [1]
        AccT* __restrict__ grad_rho) // [2]
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    T x0 = pos[2*n+0], x1 = pos[2*n+1];
    T a = (T)TWO_PI * s;
    const T twopi = (T)TWO_PI;

    // envelope
    T u = rho[0]*x0 + rho[1]*x1;
    bool inside = (u > (T)(-3)) && (u < (T)3);
    T uc = u < (T)(-3) ? (T)(-3) : (u > (T)3 ? (T)3 : u);
    T e = exp(uc);

    // per-node reductions accumulated in AccT (double) -- precision-exact over the M-frequency sum,
    // which the gate's ABSOLUTE 1e-3 tol requires when grad scale ~1e3+ (NOT a low-precision shortcut).
    AccT gpos0 = 0, gpos1 = 0;   // node-local grad_pos accumulation
    AccT gs    = 0;              // node-local grad_s accumulation (atomic once at end)
    AccT gu    = 0;              // node-local accumulation of dL/d<rho,x> (through env)

    for (int k = 0; k < M; ++k) {
        T xk0 = xi[2*k+0], xk1 = xi[2*k+1];
        T dotk = x0*xk0 + x1*xk1;
        T ph = a * dotk;
        T sp, cp; sincos_t(ph, &sp, &cp);
        T crk = cr[k], cik = ci[k];

        T gqr = gQ[n*2*M + k];
        T gqi = gQ[n*2*M + M + k];

        // QFE oscillatory parts (pre-envelope)
        T osc_re = crk*cp - cik*sp;
        T osc_im = crk*sp + cik*cp;

        // ---- grad_c (QFE) ----  d q_re/dcr = e*cp, d q_im/dcr = e*sp ; d q_re/dci = -e*sp, d q_im/dci = e*cp
        T gcr = e * (gqr*cp + gqi*sp);
        T gci = e * (-gqr*sp + gqi*cp);

        // ---- grad to ph (QFE) ----  dq_re/dph = -osc_im*e ; dq_im/dph = osc_re*e
        AccT gph = (AccT)(e * (gqr*(-osc_im) + gqi*(osc_re)));

        // ---- grad env ----  dL/de = gqr*osc_re + gqi*osc_im
        T gde = gqr*osc_re + gqi*osc_im;
        // de/du = e inside clamp, 0 outside ; accumulate dL/d<rho,x>
        if (inside) gu += gde * e;

        // ---- Dmu velocity ----
        // upstream for the 4 channels at this k
        T gvrx = gV[n*4*M + 0*M + k];   // re_dx
        T gvix = gV[n*4*M + 1*M + k];   // im_dx
        T gvry = gV[n*4*M + 2*M + k];   // re_dy
        T gviy = gV[n*4*M + 3*M + k];   // im_dy

        // accumulators for grad_xi at this k (xi enters via cv AND via ph)
        AccT gxi0 = 0, gxi1 = 0;
        AccT gcr_v = 0, gci_v = 0;   // velocity contribution to grad_c

        // velocity path products kept in AccT (double): the grad_s/grad_xi reductions reach
        // magnitude ~1e4 where fp32 ulp (~2e-3) already exceeds the gate's 1e-3 tol.
        AccT cpd = (AccT)cp, spd = (AccT)sp;
        AccT crd = (AccT)crk, cid = (AccT)cik;
        #pragma unroll
        for (int d = 0; d < 2; ++d) {
            AccT xid  = (AccT)((d==0) ? xk0 : xk1);
            AccT gvr  = (AccT)((d==0) ? gvrx : gvry);
            AccT gvi  = (AccT)((d==0) ? gvix : gviy);
            // cv.re = -2pi*xid*ci ; cv.im = 2pi*xid*cr
            AccT cvre = -(AccT)TWO_PI * xid * cid;
            AccT cvim =  (AccT)TWO_PI * xid * crd;
            // grad wrt cvre, cvim:
            AccT gcvre = gvr*cpd + gvi*spd;
            AccT gcvim = -gvr*spd + gvi*cpd;
            // grad wrt ph:  dv_re/dph = -v_im ; dv_im/dph = v_re
            AccT v_re = cvre*cpd - cvim*spd;
            AccT v_im = cvre*spd + cvim*cpd;
            gph += gvr*(-v_im) + gvi*(v_re);
            // grad to cr,ci (velocity path):  dcvim/dcr = 2pi*xid ; dcvre/dci = -2pi*xid
            gcr_v += gcvim * ((AccT)TWO_PI * xid);
            gci_v += gcvre * (-(AccT)TWO_PI * xid);
            // grad to xid (via cv): dcvre/dxid = -2pi*ci ; dcvim/dxid = 2pi*cr
            AccT gxid = gcvre*(-(AccT)TWO_PI*cid) + gcvim*((AccT)TWO_PI*crd);
            if (d==0) gxi0 += gxid; else gxi1 += gxid;
        }

        // ---- distribute gph to pos, xi, s ----  ph = a*(x0*xk0 + x1*xk1), a = 2pi*s
        // products kept in AccT (double) so the large grad_s/grad_xi reductions stay within the
        // gate's ABSOLUTE 1e-3 tol (these scalars reach magnitude ~1e4; fp32 ulp there is ~3e-3).
        AccT ad = (AccT)TWO_PI * (AccT)s;           // a in double
        AccT dotd = (AccT)x0*(AccT)xk0 + (AccT)x1*(AccT)xk1;  // dotk in double
        gpos0 += gph * ad * (AccT)xk0;
        gpos1 += gph * ad * (AccT)xk1;
        // dph/dxk0 = a*x0 ; dph/dxk1 = a*x1
        gxi0 += gph * ad * (AccT)x0;
        gxi1 += gph * ad * (AccT)x1;
        // dph/ds = 2pi*dotk
        gs += gph * (AccT)TWO_PI * dotd;

        // ---- write grad_c and grad_xi (shared params, atomic, double accumulator) ----
        atomadd(&grad_cr[k], (AccT)(gcr + gcr_v));
        atomadd(&grad_ci[k], (AccT)(gci + gci_v));
        atomadd(&grad_xi[2*k+0], (AccT)gxi0);
        atomadd(&grad_xi[2*k+1], (AccT)gxi1);
    }

    // env -> rho, pos
    // dL/d<rho,x> = gu ; <rho,x> = rho0*x0 + rho1*x1
    gpos0 += gu * rho[0];
    gpos1 += gu * rho[1];
    atomadd(&grad_rho[0], (AccT)(gu * x0));
    atomadd(&grad_rho[1], (AccT)(gu * x1));

    grad_pos[2*n+0] = gpos0;
    grad_pos[2*n+1] = gpos1;
    atomadd(&grad_s[0], (AccT)gs);
}

// =======================================================================================
// host launchers (dispatched fp32/fp64)
// =======================================================================================
#include <vector>

std::vector<at::Tensor> phase_forward(at::Tensor pos, at::Tensor xi, at::Tensor cr, at::Tensor ci,
                                      at::Tensor rho, at::Tensor s) {
    FQ_CHECK_INPUT(pos); FQ_CHECK_INPUT(xi); FQ_CHECK_INPUT(cr);
    FQ_CHECK_INPUT(ci);  FQ_CHECK_INPUT(rho);
    int N = pos.size(0), M = xi.size(0);
    auto opt = pos.options();
    auto qfe = at::empty({N, 2*M}, opt);
    auto vel = at::empty({N, 4*M}, opt);
    int threads = 128, blocks = (N + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(pos.scalar_type(), "phase_forward", ([&]{
        scalar_t sval = s.item<scalar_t>();
        phase_forward_kernel<scalar_t><<<blocks, threads>>>(
            pos.data_ptr<scalar_t>(), xi.data_ptr<scalar_t>(),
            cr.data_ptr<scalar_t>(), ci.data_ptr<scalar_t>(), rho.data_ptr<scalar_t>(),
            sval, N, M, qfe.data_ptr<scalar_t>(), vel.data_ptr<scalar_t>());
    }));
    FQ_DEBUG_SYNC();
    return {qfe, vel};
}

std::vector<at::Tensor> phase_backward(at::Tensor pos, at::Tensor xi, at::Tensor cr, at::Tensor ci,
                                       at::Tensor rho, at::Tensor s, at::Tensor gQ, at::Tensor gV) {
    FQ_CHECK_INPUT(pos); FQ_CHECK_INPUT(xi); FQ_CHECK_INPUT(cr);
    FQ_CHECK_INPUT(ci);  FQ_CHECK_INPUT(rho); FQ_CHECK_INPUT(gQ); FQ_CHECK_INPUT(gV);
    int N = pos.size(0), M = xi.size(0);
    auto opt = pos.options();
    // shared-param grads accumulated in DOUBLE (precision-exact across the N-node reduction),
    // then cast back to the input dtype at the boundary.
    auto optd = opt.dtype(at::kDouble);
    auto grad_pos = at::empty({N, 2}, opt);
    auto grad_xi  = at::zeros({M, 2}, optd);
    auto grad_cr  = at::zeros({M}, optd);
    auto grad_ci  = at::zeros({M}, optd);
    auto grad_s   = at::zeros({1}, optd);
    auto grad_rho = at::zeros({2}, optd);
    int threads = 128, blocks = (N + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(pos.scalar_type(), "phase_backward", ([&]{
        scalar_t sval = s.item<scalar_t>();
        phase_backward_kernel<scalar_t, double><<<blocks, threads>>>(
            pos.data_ptr<scalar_t>(), xi.data_ptr<scalar_t>(),
            cr.data_ptr<scalar_t>(), ci.data_ptr<scalar_t>(), rho.data_ptr<scalar_t>(),
            sval, N, M, gQ.data_ptr<scalar_t>(), gV.data_ptr<scalar_t>(),
            grad_pos.data_ptr<scalar_t>(), grad_xi.data_ptr<double>(),
            grad_cr.data_ptr<double>(), grad_ci.data_ptr<double>(),
            grad_s.data_ptr<double>(), grad_rho.data_ptr<double>());
    }));
    FQ_DEBUG_SYNC();
    // cast the double accumulators back to the input dtype
    auto sd = pos.scalar_type();
    return {grad_pos, grad_xi.to(sd), grad_cr.to(sd), grad_ci.to(sd), grad_s.to(sd), grad_rho.to(sd)};
}

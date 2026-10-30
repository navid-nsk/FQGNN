// fq_eig.cu  --  Op-B eigenphases (+ Op-C secular det) for M = diag(z) U  (6x6 complex, unitary).
//
// FORWARD  (per node, one thread):
//   1. Faddeev-LeVerrier  -> monic char poly coeffs of M (n=6).
//   2. Aberth-Ehrlich      -> 6 eigenvalues (init on unit circle; M unitary).
//   3. sort eigenphases    -> theta_j = atan2(im,re) ascending.
//   4. inverse iteration   -> eigenvector v_j of each eigenvalue (2 steps, robust to clusters).
//   5. outputs theta[N,6], pUri[N,2] = det(I-M) = prod_j (1 - lam_j).
//
// BACKWARD (per node):  M is unitary => normal => orthonormal eigenvectors, so
//   dL/dM[k,l] = sum_j  i * cot_j * lam_j * v_jk * conj(v_jl)        (cot_j = dL/dtheta_j)
//   (verified vs torch autograd to 2.8e-11 fp64; fp32 < 1e-3).  p_U det grad rides along.
//
// fp32 (production) + fp64 (gradcheck) only.  NO fast-math, NO TF32, sincosf/atan2 in full precision.
// .cu MUST NOT include <torch/extension.h> (torch-2.11/MSVC C2872) -- use common.cuh (<torch/types.h>).

#include "common.cuh"

#define NMAT 6          // fixed matrix size n=6 (Op-B is always n=6)

// --------------------------------------------------------------------------------------
// minimal device complex type (templated on real scalar; avoids thrust/cuComplex coupling)
// --------------------------------------------------------------------------------------
template <typename T>
struct cplx {
    T re, im;
    __device__ __forceinline__ cplx() : re(0), im(0) {}
    __device__ __forceinline__ cplx(T r, T i) : re(r), im(i) {}
};
template <typename T> __device__ __forceinline__ cplx<T> cadd(cplx<T> a, cplx<T> b){ return cplx<T>(a.re+b.re, a.im+b.im); }
template <typename T> __device__ __forceinline__ cplx<T> csub(cplx<T> a, cplx<T> b){ return cplx<T>(a.re-b.re, a.im-b.im); }
template <typename T> __device__ __forceinline__ cplx<T> cmul(cplx<T> a, cplx<T> b){ return cplx<T>(a.re*b.re-a.im*b.im, a.re*b.im+a.im*b.re); }
template <typename T> __device__ __forceinline__ cplx<T> cscal(T s, cplx<T> a){ return cplx<T>(s*a.re, s*a.im); }
template <typename T> __device__ __forceinline__ cplx<T> cconj(cplx<T> a){ return cplx<T>(a.re, -a.im); }
template <typename T> __device__ __forceinline__ T cabs2(cplx<T> a){ return a.re*a.re+a.im*a.im; }
template <typename T> __device__ __forceinline__ T cabsv(cplx<T> a){ return sqrt(a.re*a.re+a.im*a.im); }
template <typename T> __device__ __forceinline__ cplx<T> cdiv(cplx<T> a, cplx<T> b){
    T d = b.re*b.re + b.im*b.im;
    return cplx<T>((a.re*b.re+a.im*b.im)/d, (a.im*b.re-a.re*b.im)/d);
}

// sincos dispatch (full precision, NOT __sinf)
__device__ __forceinline__ void sincos_t(float a, float* s, float* c){ sincosf(a,s,c); }
__device__ __forceinline__ void sincos_t(double a, double* s, double* c){ sincos(a,s,c); }

// per-precision Aberth convergence tolerance
template <typename T> __device__ __forceinline__ T aberth_tol();
template <> __device__ __forceinline__ float  aberth_tol<float>(){ return 1e-7f; }
template <> __device__ __forceinline__ double aberth_tol<double>(){ return 1e-14; }

// --------------------------------------------------------------------------------------
// Faddeev-LeVerrier: monic char poly p(x)=x^6 + c1 x^5 + ... + c6.  coeff[0..6], coeff[0]=1.
// --------------------------------------------------------------------------------------
template <typename T>
__device__ void char_poly(const cplx<T> M[NMAT][NMAT], cplx<T> coeff[NMAT+1]) {
    cplx<T> Mk[NMAT][NMAT];   // M_{k}
    cplx<T> tmp[NMAT][NMAT];
    coeff[0] = cplx<T>(T(1), T(0));
    #pragma unroll
    for (int i=0;i<NMAT;i++)
        #pragma unroll
        for (int j=0;j<NMAT;j++) Mk[i][j]=cplx<T>(T(0),T(0));
    for (int k=1;k<=NMAT;k++) {
        // tmp = M @ Mk + coeff[k-1]*I
        #pragma unroll
        for (int i=0;i<NMAT;i++)
            #pragma unroll
            for (int j=0;j<NMAT;j++) {
                cplx<T> s(T(0),T(0));
                #pragma unroll
                for (int p=0;p<NMAT;p++) s = cadd(s, cmul(M[i][p], Mk[p][j]));
                if (i==j) s = cadd(s, coeff[k-1]);
                tmp[i][j]=s;
            }
        // coeff[k] = -trace(M @ tmp)/k
        cplx<T> tr(T(0),T(0));
        #pragma unroll
        for (int i=0;i<NMAT;i++)
            #pragma unroll
            for (int p=0;p<NMAT;p++) tr = cadd(tr, cmul(M[i][p], tmp[p][i]));
        coeff[k] = cscal(T(-1)/T(k), tr);
        // Mk <- tmp
        #pragma unroll
        for (int i=0;i<NMAT;i++)
            #pragma unroll
            for (int j=0;j<NMAT;j++) Mk[i][j]=tmp[i][j];
    }
}

template <typename T>
__device__ __forceinline__ cplx<T> poly_eval(const cplx<T> c[NMAT+1], cplx<T> x){
    cplx<T> r(T(0),T(0));
    #pragma unroll
    for (int i=0;i<=NMAT;i++) r = cadd(cmul(r,x), c[i]);
    return r;
}
template <typename T>
__device__ __forceinline__ cplx<T> poly_eval_d(const cplx<T> c[NMAT+1], cplx<T> x){
    cplx<T> r(T(0),T(0));
    #pragma unroll
    for (int i=0;i<NMAT;i++) r = cadd(cmul(r,x), cscal(T(NMAT-i), c[i]));
    return r;
}

// --------------------------------------------------------------------------------------
// Aberth-Ehrlich simultaneous root finder.  Init on unit circle (M unitary).
// --------------------------------------------------------------------------------------
template <typename T>
__device__ void aberth(const cplx<T> coeff[NMAT+1], cplx<T> roots[NMAT], int max_iter, T tol){
    const T twopi = T(6.283185307179586);
    #pragma unroll
    for (int j=0;j<NMAT;j++){
        T ang = twopi*(T(j)+T(0.5))/T(NMAT);
        T s,c; sincos_t(ang,&s,&c);
        roots[j]=cplx<T>(c,s);
    }
    for (int it=0; it<max_iter; ++it){
        cplx<T> newr[NMAT];
        T maxd = T(0);
        #pragma unroll
        for (int j=0;j<NMAT;j++){
            cplx<T> p  = poly_eval(coeff, roots[j]);
            cplx<T> dp = poly_eval_d(coeff, roots[j]);
            if (cabs2(dp) < T(1e-30)) { newr[j]=roots[j]; continue; }
            cplx<T> ratio = cdiv(p, dp);            // Newton step
            cplx<T> ssum(T(0),T(0));
            #pragma unroll
            for (int k=0;k<NMAT;k++){
                if (k==j) continue;
                cplx<T> d = csub(roots[j], roots[k]);
                if (cabs2(d) < T(1e-30)) continue;
                ssum = cadd(ssum, cdiv(cplx<T>(T(1),T(0)), d));
            }
            cplx<T> denom = csub(cplx<T>(T(1),T(0)), cmul(ratio, ssum));
            if (cabs2(denom) < T(1e-30)) { newr[j]=roots[j]; continue; }
            cplx<T> w = cdiv(ratio, denom);
            newr[j] = csub(roots[j], w);
            T wd = cabsv(w); if (wd>maxd) maxd=wd;
        }
        #pragma unroll
        for (int j=0;j<NMAT;j++) roots[j]=newr[j];
        if (maxd < tol) break;
    }
}

// --------------------------------------------------------------------------------------
// Solve A x = b  (6x6 complex) via Gaussian elimination with partial pivoting.  In-place on a copy.
// --------------------------------------------------------------------------------------
template <typename T>
__device__ void solve6(cplx<T> A[NMAT][NMAT], cplx<T> b[NMAT], cplx<T> x[NMAT]){
    // augmented elimination
    for (int col=0; col<NMAT; ++col){
        int best=col; T bv=cabs2(A[col][col]);
        for (int r=col+1;r<NMAT;r++){ T v=cabs2(A[r][col]); if (v>bv){bv=v;best=r;} }
        if (best!=col){
            #pragma unroll
            for (int cc=0;cc<NMAT;cc++){ cplx<T> t=A[col][cc]; A[col][cc]=A[best][cc]; A[best][cc]=t; }
            cplx<T> tb=b[col]; b[col]=b[best]; b[best]=tb;
        }
        cplx<T> piv=A[col][col];
        if (cabs2(piv) < T(1e-30)) { piv=cplx<T>(T(1e-15),T(0)); A[col][col]=piv; }
        for (int r=col+1;r<NMAT;r++){
            cplx<T> f = cdiv(A[r][col], piv);
            #pragma unroll
            for (int cc=col;cc<NMAT;cc++) A[r][cc]=csub(A[r][cc], cmul(f,A[col][cc]));
            b[r]=csub(b[r], cmul(f,b[col]));
        }
    }
    // back substitution
    for (int r=NMAT-1;r>=0;r--){
        cplx<T> s=b[r];
        for (int cc=r+1;cc<NMAT;cc++) s=csub(s, cmul(A[r][cc], x[cc]));
        x[r]=cdiv(s, A[r][r]);
    }
}

// eigenvector of M for eigenvalue lam via 2 steps inverse iteration (robust, normalized).
template <typename T>
__device__ void eigvec_inviter(const cplx<T> M[NMAT][NMAT], cplx<T> lam, cplx<T> v[NMAT]){
    const T shift = T(1e-5);
    // start vector (deterministic)
    #pragma unroll
    for (int i=0;i<NMAT;i++) v[i]=cplx<T>(T(1)/sqrt(T(NMAT)), T(0));
    for (int step=0; step<2; ++step){
        cplx<T> A[NMAT][NMAT];
        #pragma unroll
        for (int i=0;i<NMAT;i++)
            #pragma unroll
            for (int j=0;j<NMAT;j++){
                A[i][j]=M[i][j];
                if (i==j){ A[i][j]=csub(A[i][j], lam); A[i][j].re += shift; }
            }
        cplx<T> b[NMAT], x[NMAT];
        #pragma unroll
        for (int i=0;i<NMAT;i++) b[i]=v[i];
        solve6(A, b, x);
        T nrm=T(0);
        #pragma unroll
        for (int i=0;i<NMAT;i++) nrm += cabs2(x[i]);
        nrm = sqrt(nrm);
        if (nrm < T(1e-30)) nrm = T(1);
        #pragma unroll
        for (int i=0;i<NMAT;i++) v[i]=cscal(T(1)/nrm, x[i]);
    }
}

// --------------------------------------------------------------------------------------
// FORWARD kernel: input Mop[N,6,6] complex (re,im interleaved as 2*float per entry via
// torch complex tensor viewed as float pairs). We pass Mre[N,36], Mim[N,36].
// outputs theta[N,6], pUre[N], pUim[N], and stash eigvecs (re/im [N,36]) + roots (re/im[N,6]) for bwd.
// --------------------------------------------------------------------------------------
template <typename T>
__global__ void opb_forward_kernel(
        const T* __restrict__ Mre, const T* __restrict__ Mim, int N,
        T* __restrict__ theta, T* __restrict__ pUre, T* __restrict__ pUim,
        T* __restrict__ vre, T* __restrict__ vim,      // [N,36] eigenvectors (column j stored contiguous)
        T* __restrict__ rootre, T* __restrict__ rootim, // [N,6] sorted eigenvalues
        int max_iter)
{
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n>=N) return;
    // -- input matrix in T (for the well-conditioned inverse-iteration eigvecs) --
    cplx<T> M[NMAT][NMAT];
    #pragma unroll
    for (int i=0;i<NMAT;i++)
        #pragma unroll
        for (int j=0;j<NMAT;j++){
            int idx = n*NMAT*NMAT + i*NMAT + j;
            M[i][j]=cplx<T>(Mre[idx], Mim[idx]);
        }
    // -- EIGENVALUES in DOUBLE always: Faddeev-LeVerrier + Aberth suffer catastrophic fp32
    //    cancellation when eigenphases cluster near 0 (U~I, the real-model init regime). The 6x6
    //    double matmuls per thread are cheap; precision here is load-bearing (the FQ path never uses
    //    LOW precision; HIGHER precision for this cancellation-prone step is intended). --
    cplx<double> Md[NMAT][NMAT];
    #pragma unroll
    for (int i=0;i<NMAT;i++)
        #pragma unroll
        for (int j=0;j<NMAT;j++)
            Md[i][j]=cplx<double>((double)M[i][j].re, (double)M[i][j].im);
    cplx<double> coeff[NMAT+1];
    char_poly(Md, coeff);
    cplx<double> rootsd[NMAT];
    aberth(coeff, rootsd, max_iter, 1e-14);

    // sort by angle ascending (insertion sort, n=6) -- in double
    double angd[NMAT];
    #pragma unroll
    for (int j=0;j<NMAT;j++) angd[j]=atan2(rootsd[j].im, rootsd[j].re);
    #pragma unroll
    for (int a=1;a<NMAT;a++){
        cplx<double> rv=rootsd[a]; double av=angd[a];
        int b=a-1;
        while (b>=0 && angd[b]>av){ rootsd[b+1]=rootsd[b]; angd[b+1]=angd[b]; b--; }
        rootsd[b+1]=rv; angd[b+1]=av;
    }
    // outputs theta + pU = prod(1 - lam)  (pU product in double too -- clustered-near-1 cancellation)
    cplx<double> pUd(1.0,0.0);
    #pragma unroll
    for (int j=0;j<NMAT;j++){
        theta[n*NMAT+j]=(T)angd[j];
        rootre[n*NMAT+j]=(T)rootsd[j].re; rootim[n*NMAT+j]=(T)rootsd[j].im;
        pUd = cmul(pUd, csub(cplx<double>(1.0,0.0), rootsd[j]));
    }
    pUre[n]=(T)pUd.re; pUim[n]=(T)pUd.im;
    // eigenvectors (sorted order) via inverse iteration on (M - lam I) in T -- well-conditioned (shifted)
    #pragma unroll
    for (int j=0;j<NMAT;j++){
        cplx<T> lam((T)rootsd[j].re, (T)rootsd[j].im);
        cplx<T> v[NMAT];
        eigvec_inviter(M, lam, v);
        #pragma unroll
        for (int i=0;i<NMAT;i++){
            vre[n*NMAT*NMAT + j*NMAT + i]=v[i].re;
            vim[n*NMAT*NMAT + j*NMAT + i]=v[i].im;
        }
    }
}

// --------------------------------------------------------------------------------------
// BACKWARD kernel: given grad_theta[N,6] (=cot_j) and grad_pUre/pUim[N] (det channel),
// produce grad_Mre/grad_Mim[N,36].
//   theta channel: dM[k,l] = sum_j i*cot_j*lam_j* v_jk * conj(v_jl)
//   det channel  : p_U = prod(1-lam). grad wrt M: PyTorch det backward = gp * cofactor.
//     We compute via the adjugate path on (I - M): grad_M[k,l] for det(I-M) = -det(I-M)*inv(I-M)^T[l,k]
//     in real-loss VJP form. We instead reuse the eigen-decomposition: p_U = prod(1-lam_j),
//     and 1-lam_j has the SAME eigenvectors, so d p_U / dM[k,l] = sum_j (dpU/dlam_j) * (dlam_j/dM[k,l]).
//     dpU/dlam_j = -prod_{i!=j}(1-lam_i).  dlam_j/dM[k,l] = conj(v_jk) v_jl (holo).
//     For the real loss L=Re(gp* * pU), the PyTorch grad is built like the theta channel.
// --------------------------------------------------------------------------------------
template <typename T>
__global__ void opb_backward_kernel(
        const T* __restrict__ grad_theta,                 // [N,6] = cot_j (dL/dtheta_j)
        const T* __restrict__ grad_pUre, const T* __restrict__ grad_pUim,  // [N] dL/d pU.re, dL/d pU.im
        const T* __restrict__ vre, const T* __restrict__ vim,   // [N,36]
        const T* __restrict__ rootre, const T* __restrict__ rootim, // [N,6]
        int N, T* __restrict__ grad_Mre, T* __restrict__ grad_Mim)
{
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n>=N) return;
    cplx<T> roots[NMAT], v[NMAT][NMAT];
    #pragma unroll
    for (int j=0;j<NMAT;j++){
        roots[j]=cplx<T>(rootre[n*NMAT+j], rootim[n*NMAT+j]);
        #pragma unroll
        for (int i=0;i<NMAT;i++)
            v[j][i]=cplx<T>(vre[n*NMAT*NMAT + j*NMAT + i], vim[n*NMAT*NMAT + j*NMAT + i]);
    }
    // grad accumulator
    cplx<T> G[NMAT][NMAT];
    #pragma unroll
    for (int k=0;k<NMAT;k++)
        #pragma unroll
        for (int l=0;l<NMAT;l++) G[k][l]=cplx<T>(T(0),T(0));

    // ---- theta channel ----
    // dL/dM[k,l] += sum_j i*cot_j*lam_j*v_jk*conj(v_jl)
    #pragma unroll
    for (int j=0;j<NMAT;j++){
        T cot = grad_theta[n*NMAT+j];
        cplx<T> coef = cscal(cot, roots[j]);      // cot * lam_j (cot real)
        coef = cmul(cplx<T>(T(0),T(1)), coef);    // i * cot * lam_j
        #pragma unroll
        for (int k=0;k<NMAT;k++)
            #pragma unroll
            for (int l=0;l<NMAT;l++){
                cplx<T> term = cmul(coef, cmul(v[j][k], cconj(v[j][l])));
                G[k][l]=cadd(G[k][l], term);
            }
    }

    // ---- det channel ----  L_det = Re( conj(gp) * pU ), gp = grad_pUre + i grad_pUim
    // Build dL_det/dlam_j first, then the PyTorch-convention grad wrt M with same eigen outer-products.
    // pU = prod(1-lam).  dpU/dlam_j = -prod_{i!=j}(1-lam_i).
    // For dlam_j contribution to a holomorphic scalar f: df = sum_j (df/dlam_j) dlam_j, dlam_j = conj(v_jk)v_jl dM_kl.
    // Real loss L=Re(conj(gp)*pU): treat pU holomorphic in M (det is holomorphic). Then
    //   grad_M[k,l] (PyTorch) = conj( dL/dM_holo )... we derive exactly like theta channel:
    //   dL = Re( conj(gp) * dpU ) = Re( conj(gp) * sum_j A_j * (conj(v_jk) v_jl) dM_kl )  where A_j=dpU/dlam_j
    //   = Re( sum_kl [ sum_j conj(gp) A_j conj(v_jk) v_jl ] dM_kl ).
    //   PyTorch grad convention: dL = Re( sum conj(grad_kl) dM_kl ) => grad_kl = conj( sum_j conj(gp) A_j conj(v_jk) v_jl )
    //                                                                = sum_j gp * conj(A_j) * v_jk * conj(v_jl)
    {
        cplx<T> gp(grad_pUre[n], grad_pUim[n]);
        // precompute prod(1-lam_i) leave-one-out per-j in DOUBLE (clustered-near-1 cancellation)
        cplx<double> oneminus[NMAT];
        #pragma unroll
        for (int j=0;j<NMAT;j++)
            oneminus[j]=csub(cplx<double>(1.0,0.0), cplx<double>((double)roots[j].re,(double)roots[j].im));
        #pragma unroll
        for (int j=0;j<NMAT;j++){
            cplx<double> prodd(1.0,0.0);
            #pragma unroll
            for (int i=0;i<NMAT;i++) if (i!=j) prodd=cmul(prodd, oneminus[i]);
            cplx<T> Aj = cscal(T(-1), cplx<T>((T)prodd.re,(T)prodd.im));  // dpU/dlam_j = -prod_{i!=j}(1-lam_i)
            cplx<T> coef = cmul(gp, cconj(Aj));        // gp * conj(A_j)
            #pragma unroll
            for (int k=0;k<NMAT;k++)
                #pragma unroll
                for (int l=0;l<NMAT;l++){
                    cplx<T> term = cmul(coef, cmul(v[j][k], cconj(v[j][l])));
                    G[k][l]=cadd(G[k][l], term);
                }
        }
    }

    #pragma unroll
    for (int k=0;k<NMAT;k++)
        #pragma unroll
        for (int l=0;l<NMAT;l++){
            int idx=n*NMAT*NMAT + k*NMAT + l;
            grad_Mre[idx]=G[k][l].re; grad_Mim[idx]=G[k][l].im;
        }
}

// =======================================================================================
// host launchers (dispatched fp32/fp64)
// =======================================================================================
#include <vector>

std::vector<at::Tensor> opb_forward(at::Tensor Mre, at::Tensor Mim, int64_t max_iter){
    FQ_CHECK_INPUT(Mre); FQ_CHECK_INPUT(Mim);
    int N = Mre.size(0);
    auto opt = Mre.options();
    auto theta = at::empty({N, NMAT}, opt);
    auto pUre  = at::empty({N}, opt);
    auto pUim  = at::empty({N}, opt);
    auto vre   = at::empty({N, NMAT*NMAT}, opt);
    auto vim   = at::empty({N, NMAT*NMAT}, opt);
    auto rre   = at::empty({N, NMAT}, opt);
    auto rim   = at::empty({N, NMAT}, opt);
    int threads=128, blocks=(N+threads-1)/threads;
    AT_DISPATCH_FLOATING_TYPES(Mre.scalar_type(), "opb_forward", ([&]{
        opb_forward_kernel<scalar_t><<<blocks,threads>>>(
            Mre.data_ptr<scalar_t>(), Mim.data_ptr<scalar_t>(), N,
            theta.data_ptr<scalar_t>(), pUre.data_ptr<scalar_t>(), pUim.data_ptr<scalar_t>(),
            vre.data_ptr<scalar_t>(), vim.data_ptr<scalar_t>(),
            rre.data_ptr<scalar_t>(), rim.data_ptr<scalar_t>(), (int)max_iter);
    }));
    FQ_DEBUG_SYNC();
    return {theta, pUre, pUim, vre, vim, rre, rim};
}

std::vector<at::Tensor> opb_backward(at::Tensor grad_theta, at::Tensor grad_pUre, at::Tensor grad_pUim,
                                     at::Tensor vre, at::Tensor vim, at::Tensor rre, at::Tensor rim){
    FQ_CHECK_INPUT(grad_theta);
    int N = grad_theta.size(0);
    auto opt = grad_theta.options();
    auto gMre = at::empty({N, NMAT*NMAT}, opt);
    auto gMim = at::empty({N, NMAT*NMAT}, opt);
    int threads=128, blocks=(N+threads-1)/threads;
    AT_DISPATCH_FLOATING_TYPES(grad_theta.scalar_type(), "opb_backward", ([&]{
        opb_backward_kernel<scalar_t><<<blocks,threads>>>(
            grad_theta.data_ptr<scalar_t>(), grad_pUre.data_ptr<scalar_t>(), grad_pUim.data_ptr<scalar_t>(),
            vre.data_ptr<scalar_t>(), vim.data_ptr<scalar_t>(),
            rre.data_ptr<scalar_t>(), rim.data_ptr<scalar_t>(), N,
            gMre.data_ptr<scalar_t>(), gMim.data_ptr<scalar_t>());
    }));
    FQ_DEBUG_SYNC();
    return {gMre, gMim};
}

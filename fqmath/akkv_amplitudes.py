"""
fqmath.akkv_amplitudes
======================

Variety-FIXED Fourier-Bohr amplitudes for the FQ measure: the amplitude is determined by the
Lee-Yang variety (through the secular unitary U and the lift L), not learned freely.

WHAT THE AMPLITUDE IS NOT
-------------------------
Two objects look like amplitudes but are not the FQ diffraction mass:
  (a) a FREE learnable weight `amp_w` (the geometry-only ablation: FQSpectrum(amplitude="free"),
      op_spectral_block, op_coupled_spectral) -- it stands in for the fixed mhat, i.e. it decouples
      the amplitudes from the variety; and
  (b) a_S = (-1)^{n-|S|} det(U_S) -- det(U_S) is the COEFFICIENT of p_U in its multilinear
      expansion, NOT the diffraction mass.

THE VARIETY-FIXED AMPLITUDE
---------------------------
  d = 1  (secular / Kurasov-Sarnak):
        c_k = <ell, k> * c_P(k) ,        c_P(k) = [z^k] log p_U(z)
    where <ell,k> = (L^T k) is the (scalar) frequency xi_k for d=1 (a linear weight that is part of
    the amplitude), and c_P(k) is the k-th Taylor coefficient of LOG p_U -- a cumulant / Moebius sum
    over compositions, NOT a single minor det(U_S).  This is EXACT and is what this module computes.

  d >= 2  (AKKV Eq. 9):
        c_{L,k} = sum_{I in C(n,d)} L_I * mhat_{[n] minus I}(k)
    linear in the d x d minors L_I, with mhat_J(k) FIXED by the variety X.  The general Eq.-9 torus
    integral for an arbitrary variety is NOT computed here (it needs the variety's own mhat); see
    `secular_pullback_amplitudes` for the scope we DO cover (the secular-hypersurface var(k)=0
    harmonics, exactly) and `akkv_torus_amplitudes` for the d=2 secular split family.

All amplitudes are COMPLEX with Hermitian symmetry c_{-k} = conj(c_k), and a support mask
1[|c_k| > tol] (the true support S is a strict subset of {L^T k : var(k) < d}).  Extinction
(AKKV Thm 4.32) removes the var(k)>=d modes because the torus INTEGRAL (the Eq.9 mass) vanishes
there -- NOT because the log-p_U cumulant c_P(k) is zero (c_P is generically nonzero); the support
mask is applied on the ENUMERATED var(k)<d modes.

EXACTNESS AND SCOPE:
  * `secular_log_cumulants` computes c_P(k) EXACTLY via the Euler recursion for the log of the
    MULTILINEAR polynomial p_U(z)=det(I - diag(z)U); differentiable in U.  Unit-checked against the
    closed forms (n=1: c_P(k) = -U^k / k) in `_selftest`.
  * `secular_amplitudes_d1` (d=1) is therefore exact and faithful to KS.
  * For d>=2, TWO modes exist.  `secular_pullback_amplitudes` (the default) is EXACT on the
    var(k)=0 (harmonic) modes and masks the var(k)=1 (beat) modes.  `akkv_torus_amplitudes`
    (selectable via FQSpectrum(amplitude="akkv_torus") / FQ_AMP_MODE) evaluates
    the AKKV Thm 2.6(3)/Eq.-9 torus integrals VERBATIM on the canonical secular split family (a
    genuine equidimensional codim-2 Lee-Yang variety built from the same secular data) -- beats
    generically LIVE; its assembly conventions reproduce AKKV Ex. 11.2 to ~4e-6.  This mode
    therefore provides the faithful d=2 FQ amplitude; see the function's block comment for the
    construction and caveats.
"""

from __future__ import annotations

import math
from itertools import combinations, product
from typing import Dict, Tuple

import torch


# ----------------------------------------------------------------------------------------
# secular polynomial coefficients  a_S = (-1)^{|S|} det(U_S)   (p_U(z) = det(I - diag(z) U))
# ----------------------------------------------------------------------------------------
def secular_coeff(U: torch.Tensor, mask: Tuple[int, ...]) -> torch.Tensor:
    """
    a_S = (-1)^{|S|} det(U_S) for S = {j : mask[j]=1}, the coefficient of z^S in
    p_U(z) = det(I - diag(z) U) = sum_{S subset [n]} (-1)^{|S|} det(U_S) z^S.
    Returns a complex scalar tensor, differentiable in U.  (a_emptyset = 1.)
    """
    S = [j for j, b in enumerate(mask) if b]
    if not S:
        return torch.ones((), dtype=U.dtype, device=U.device)
    sub = U.index_select(0, torch.tensor(S, device=U.device)).index_select(
        1, torch.tensor(S, device=U.device))
    sign = 1.0 if (len(S) % 2 == 0) else -1.0
    return sign * torch.linalg.det(sub)


def _downward_closure(combos_nonneg) -> list:
    """All integer points k' with 0 <= k' <= k (componentwise) for some k in combos_nonneg,
    sorted by total degree |k'| ascending (so the recursion sees lower-degree terms first)."""
    closure = set()
    for k in combos_nonneg:
        ranges = [range(int(v) + 1) for v in k]
        for kp in product(*ranges):
            closure.add(kp)
    return sorted(closure, key=lambda t: sum(t))


def secular_log_cumulants(U: torch.Tensor, combos_nonneg) -> torch.Tensor:
    """
    c_P(k) = [z^k] log p_U(z) for every k in `combos_nonneg` (each a non-negative integer vector),
    computed EXACTLY by the Euler recursion for the log of a multilinear polynomial:

        |k| c_P(k) = |k| a_k - sum_{0 < m <= k, m in {0,1}^n} a_m |k-m| c_P(k-m)

    where a_m = (-1)^{|m|} det(U_supp(m)) (nonzero only for m in {0,1}^n, since p_U is multilinear),
    a_k = a_{supp(k)} if k in {0,1}^n else 0.  Returns COMPLEX [M], differentiable in U.

    NOTE: pass the var(k)=0 (all-same-sign) representative as a NON-NEGATIVE vector; the conjugate
    partner c_P(-k) = conj(c_P(k)) is obtained by Hermitian symmetry, not recomputed.
    """
    n = U.shape[0]
    combos_nonneg = [tuple(int(v) for v in k) for k in combos_nonneg]
    order = _downward_closure(combos_nonneg)
    zero = tuple([0] * n)
    g: Dict[Tuple[int, ...], torch.Tensor] = {zero: torch.zeros((), dtype=U.dtype, device=U.device)}
    a_cache: Dict[Tuple[int, ...], torch.Tensor] = {}

    def a_of(mask: Tuple[int, ...]) -> torch.Tensor:
        if mask not in a_cache:
            a_cache[mask] = secular_coeff(U, mask)
        return a_cache[mask]

    for k in order:
        deg = sum(k)
        if deg == 0:
            continue
        supp = [j for j in range(n) if k[j] > 0]
        # a_k : nonzero only when k in {0,1}^n
        if all(v <= 1 for v in k):
            ak = a_of(k)  # k itself is its 0/1 mask
        else:
            ak = torch.zeros((), dtype=U.dtype, device=U.device)
        acc = torch.zeros((), dtype=U.dtype, device=U.device)
        for r in range(1, len(supp) + 1):
            for sub in combinations(supp, r):
                mask = tuple(1 if j in sub else 0 for j in range(n))
                km = tuple(k[j] - mask[j] for j in range(n))
                acc = acc + a_of(mask) * float(sum(km)) * g[km]
        g[k] = ak - acc / float(deg)

    return torch.stack([g[k] for k in combos_nonneg])


# ----------------------------------------------------------------------------------------
# Hermitian symmetry + support mask
# ----------------------------------------------------------------------------------------
def support_mask(c: torch.Tensor, tol: float = 1e-9) -> torch.Tensor:
    """1[|c_k| > tol] -- the variety support pattern: extinct modes (c_P=0) are masked out
    BEFORE Op-D taps, so empty atoms are never populated (Edwin-Lin; AKKV Thm 4.32)."""
    return (c.abs() > tol).to(c.real.dtype)


# ----------------------------------------------------------------------------------------
# the d=1 secular Fourier-Bohr amplitude (EXACT, the verifiable core)
# ----------------------------------------------------------------------------------------
def secular_amplitudes_d1(ell: torch.Tensor, U: torch.Tensor, combos_nonneg) -> torch.Tensor:
    """
    c_k = xi_k * c_P(k) ,  xi_k = <ell, k> ,  for the d=1 Kurasov-Sarnak secular FQ.   EXACT.
    ell  : positive vector [n] (the d=1 generator L = ell)        combos_nonneg : [M, n] int >= 0
    Returns COMPLEX [M].  The linear weight xi_k = <ell,k> multiplies c_P(k), the
    log-p_U cumulant (variety-fixed via U).  Hermitian symmetry c_{-k}=conj(c_k) holds by
    construction (real xi_k, c_P(-k)=conj c_P(k)); apply `support_mask` for the extinct modes.
    SIGN CONVENTION: Kurasov-Sarnak eq. 29 writes the pure-point mass as
    -(xi_k) c_P(k); this code uses the +(xi_k) c_P(k) convention
    throughout (a global sign, invisible to every |c|^2 path and absorbable by the learned Jacobi
    filter).  Trained checkpoints depend on this convention -- do NOT flip the sign
    without a full retrain.
    """
    cP = secular_log_cumulants(U, combos_nonneg)                       # [M] complex
    k = torch.as_tensor([[float(v) for v in kk] for kk in combos_nonneg],
                        dtype=ell.real.dtype if ell.is_complex() else ell.dtype, device=ell.device)
    xi = k @ ell.to(k.dtype)                                           # [M] real scalar frequency
    return xi.to(cP.dtype) * cP


def secular_pullback_amplitudes(L: torch.Tensor, U: torch.Tensor, combos,
                                tol: float = 1e-9) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Variety-fixed amplitude for a d>=2 SECULAR-hypersurface pullback (lift L in R^{n x d}).

    Returns (c, mask) with c COMPLEX [M] and mask [M] in {0,1}.

    SCOPE:
      * var(k)=0 (harmonic) modes: c_k = w * c_P(|k|) with the secular cumulant c_P (exact) times a
        HEURISTIC d>=2 weight w = ||L^T k|| (NOT the AKKV Eq.9 mass sum_I L_I mhat; the d=1 scalar
        weight <ell,k> is the only exact one).  So the MAGNITUDE/phase is the variety cumulant but the
        d>=2 weight is a project-original generalization -- treat this d>=2 amplitude as the
        variety cumulant times a heuristic weight, NOT the faithful Eq.9 mass (see `akkv_torus_amplitudes`).
      * var(k)=1 (beat) modes: their exact value is the AKKV Eq.-9 torus integral
        sum_I L_I mhat_{[n] minus I}(k), which this function does not evaluate.  They are returned as
        ZERO and masked OUT, so an operator using this amplitude is the var(k)=0 secular harmonic FQ,
        not the full d=2 FQ, and its beat tones are not claimed faithful.
    """
    n = U.shape[0]
    combos_list = [tuple(int(v) for v in k) for k in combos]
    var0_idx, nonneg = [], []
    for i, k in enumerate(combos_list):
        signs = [1 if v > 0 else (-1 if v < 0 else 0) for v in k]
        nz = [s for s in signs if s != 0]
        nsign = sum(1 for a, b in zip(nz, nz[1:]) if a != b)
        if nsign == 0:  # var(k)=0
            var0_idx.append(i)
            nonneg.append(tuple(abs(v) for v in k))
    M = len(combos_list)
    c = torch.zeros(M, dtype=U.dtype, device=U.device)
    if nonneg:
        cP = secular_log_cumulants(U, nonneg)                          # [m0] = c_P(|k|)
        kk = torch.as_tensor([[float(v) for v in combos_list[i]] for i in var0_idx],
                             dtype=L.dtype, device=L.device)
        xi = kk @ L                                                    # [m0, d]
        w = xi.norm(dim=-1).to(cP.dtype)                              # frequency-magnitude weight
        amp = w * cP                                                  # = w * c_P(|k|) for the +rep
        # Hermitian symmetry c_{-k} = conj(c_k): a combo stored as the NEGATIVE representative (first
        # nonzero entry < 0) is the partner of |k|, so it carries the CONJUGATE.  (The live call sites
        # pass only non-negative / canonical combos, so this is a no-op there, but it makes the
        # invariant hold for ANY input, including negative representatives.)
        neg = torch.tensor(
            [1.0 if next((v for v in combos_list[i] if v != 0), 0) < 0 else 0.0 for i in var0_idx],
            device=U.device)
        amp = torch.where(neg.bool(), amp.conj(), amp)
        c[torch.tensor(var0_idx, device=U.device)] = amp
    return c, support_mask(c, tol)


# ========================================================================================
# AKKV Thm 2.6(3) torus-integral amplitudes at d = 2 (the "akkv_torus" mode)
# ========================================================================================
#
# MATH ROUTE (read before editing -- every choice below is load-bearing).
#
# AKKV (arXiv:2407.11184) Thm 2.6(3) + Def 7.1 + Eq. (9): for a strict Lee-Yang variety
# X in (P^1)^n of DIMENSION c = n - d, the FQ counting-measure coefficients are
#
#     c_{L,k} = sum_{I in C([n],d)} L_I * mhat_{[n]\I}(k),
#     mhat_J(k) = int_{Y(R) cap [0,1]^n} e^{-2 pi i <x,k>} dm_J(x)
#               = (s(J)/(2 pi i)^c) int_{X(T)} z^{-k} dz_J / z_J,        |J| = c,     (Eq. 9)
#
# with m_J the measure induced by the c-form s(J) dx_J on Y(R) = exp^{-1}(X(T)) in the
# Thm-4.20 orientation, and L_I the d x d minors.
#
# WHY THE SINGLE SECULAR HYPERSURFACE CANNOT CARRY d=2 BEATS.
# The secular hypersurface X_U = {p_U = 0} has dimension n-1, i.e. codimension ONE: it is
# the c = n-1 (d_eff = 1) member of the AKKV family, NOT a d=2 variety.  Its only
# well-defined Eq.-9 moments are the (n-1)-form moments mhat_{[n]\{i}}(k), and AKKV
# Lemma 7.2 / Thm 4.32 (with d_eff = 1) force
#
#     mhat_{[n]\{i}}(k) = 0            for every k with var(k) >= 1  (ALL beat tones),
#     mhat_{[n]\{i}}(k) = -k_i c_P(k)  for var(k) = 0, canonical k >= 0 (k=0 -> +1),
#
# where the second line is the closed form obtained by matching Thm 2.6(3) at d=1 against
# the Kurasov-Sarnak masses (KS eq. 29: mass = -<l,k> c_P(k)) for every positive l --
# both sides are linear in l on an open cone, so the moments are determined exactly.
# (Checked numerically in fqmath/test_akkv_torus.py, test 3.)  Consequently NO faithful
# Eq.-9 pairing on the single hypersurface can make var(k)=1 modes live: Monte Carlo over
# {p_U=0} cannot populate the beats.
# Live beats REQUIRE a genuine dimension-(n-2) Lee-Yang variety.
#
# THE VARIETY THIS MODE USES -- the canonical SECULAR SPLIT FAMILY X_split(U):
#
#     X_split(U) = disjoint union over contiguous cuts j = 1..n-1 of
#                  X_{U_1(j)} x X_{U_2(j)}  in  (P^1)^{B1(j)} x (P^1)^{B2(j)},
#
# where B1(j) = {1..j}, B2(j) = {j+1..n}, U_b(j) = exp(i H[B_b, B_b]) with H the principal
# logarithm of U (H = -i log U, Hermitian since U is unitary), and X_V = {p_V = 0} the KS
# secular hypersurface of the block unitary V.  Each component is a product of two secular
# hypersurfaces: equidimensional of dimension (j-1) + (n-j-1) = n-2 = c  -- a genuine d=2
# Lee-Yang variety built canonically from the SAME secular data (U), with no free choice
# (ALL contiguous cuts are included; a single cut would structurally kill the beats whose
# sign change straddles it).  Products/intersections of Lee-Yang hypersurfaces are exactly
# how AKKV build their d >= 2 examples (Ex. 1.3 = intersection of two bilinear Lee-Yang
# polynomials).  Caveat: the product variety is Lee-Yang but not
# STRICT at d=2 (a point with one factor off-torus and the other on-torus has
# var(log|z|) = 1 < 2), so Thm 2.3's reality statement does not apply verbatim to the
# complexified solution set; the REAL point set and the real-side summation formula
# (Prop 7.5, whose proof is entirely real) are the standard objects.
#
# The measure of a disjoint union is the sum, and product-variety moments factor into
# block moments, so with M1 = B1\{i}, M2 = B2\{i'}:
#
#     mhat_J(k)  =  mhat^{X_1}_{M1}(k|B1) * mhat^{X_2}_{M2}(k|B2)     [J = M1 u M2],
#     c_{L,k}    =  sum_j [var(k|B1(j))=0][var(k|B2(j))=0] *
#                   sum_{i in B1(j), i' in B2(j)} Delta_{i,i'}(L) *
#                   mhat^{(j,1)}_{B1\{i}}(k|B1) * mhat^{(j,2)}_{B2\{i'}}(k|B2),
#
# with Delta_{i,i'}(L) = L[i,0]L[i',1] - L[i,1]L[i',0] the 2x2 minors (i < i' always,
# because the cuts are contiguous -- which also kills all shuffle signs: the increasing
# global order of J IS (B1-part, B2-part)).  var(k|B) >= 1 block moments are EXACT zeros
# by Lemma 7.2/Thm 4.32 (quadrature exactness, not an approximation).  A var(k) >= 2
# combo has, for EVERY cut, at least one block restriction with a sign change, so
# extinction of var >= d modes EMERGES structurally -- it is not imposed.
#
# BLOCK MOMENTS BY MONTE CARLO OVER THE GRAPH PARAMETERIZATION (the numerically sound
# route).  p_V is multilinear, so for torus values of all block coordinates but one, the
# remaining coordinate has EXACTLY ONE root, and the Lee-Yang property puts it ON the
# torus (the Blaschke/eigenphase condition): sample theta_{B\{i}} ~ QMC-uniform on
# T^{b-1} (scrambled Sobol -- deterministic given seed), solve the LINEAR equation
# p_V(z) = alpha + beta z_i = 0 (alpha = p_V at z_i=0, beta = p_V at z_i=1 minus alpha),
# set z_i* = -alpha/beta (renormalized to |z_i*| = 1; deviation is fp-level, monitored),
# and average e^{-2 pi i <k_B, theta_B>}.  This IS the literal Eq.-9 integral: on the
# torus dz_j/z_j = 2 pi i d theta_j, so (1/(2 pi i)^c) int z^{-k} dz_J/z_J =
# int z^{-k} d theta_J, and on the graph chart the form integral is the plain average.
#
# ORIENTATION / SIGNS.  The graph-chart average has mhat(0) = +1 = the multidegree d_i
# (multilinear p_V has degree 1 in every coordinate), which is exactly the positivity
# Thm 2.6(1) demands of the 4.20 orientation (density c_0 = sum_I L_I d_I > 0); the
# per-i consistency of this "positive-at-0" calibration is verified against the exact
# d=1 closed form (-k_i c_P(k)) in the tests, and the FULL Thm-2.6(3) assembly
# convention (minor pairing, z^{-k} direction, normalization) is checked against the
# worked AKKV Example 11.2 number c_xi ~= -0.94053 + 0.132548i on their curve
# (test_akkv_torus.py, test 6).
#
# SIGN CONVENTION vs THE d=1 FAMILY.  This mode returns the AKKV/KS-faithful
# object: restricted to d=1 it satisfies sum_i l_i mhat_{[n]\{i}}(k) = -<l,k> c_P(k)
# = MINUS secular_amplitudes_d1 (which uses the opposite sign to KS eq. 29;
# see the sign-convention note on secular_amplitudes_d1).  A global sign is invisible to every |c|^2
# path and absorbable by the learned per-frequency filter.
#
# PROPERTIES (all tested):  Hermitian symmetry c_{-k} = conj(c_k) EXACTLY (shared QMC
# samples per (block, i) make the estimator conjugation-equivariant to machine
# precision);  homogeneity degree d = 2 under global scale L -> aL (c -> a^2 c: the
# moments are L-independent, the d x d minors are 2-homogeneous -- Thm 2.6(3) is LINEAR
# in the minors);  var(k)=1 beats generically live (every enumerated beat has at least
# one cut separating its sign change);  var(k) >= 2 exact zero.
#
# NON-DIFFERENTIABLE / DETERMINISTIC / PRECISION:  computed under torch.no_grad on CPU
# in fp64/complex128 (no CUDA nondeterminism; the fp32 requirement applies to the model
# graph, and these host-side amplitude constants are computed in HIGHER precision),
# deterministic given `seed` (scrambled-Sobol seeds derived per (block, i)).  The result
# is an offline numerical mhat table; only s and the taps train against it (the same
# learnability scope as the fixed |L^T k| weight of the default secular mode).
# ----------------------------------------------------------------------------------------

# QMC budget default.  Measured on the production geometry (n=6, M=48, seed 0): doubling
# from 2^22 changes every per-mode |c_k| by 0.48% (and 2^23 -> 2^24 by 1.0%), while
# 2^21 -> 2^22 still moves the smallest mode (|c| ~ 1.7e-3 vs max 36) by 2.5% -- the
# <2%-under-doubling criterion holds from 2^22 up.  Runtime ~20 s CPU at 2^22 (chunked,
# cached per (L, U) in FQSpectrum).  Override with FQ_AMP_NMC.
DEFAULT_AKKV_N_MC = 1 << 22


def _var(k) -> int:
    """var(k): sign changes over the nonzero entries in coordinate order (AKKV Def 2.1)."""
    s = [1 if v > 0 else -1 for v in k if v != 0]
    return sum(1 for a, b in zip(s, s[1:]) if a != b)


def _principal_log_hermitian(U: torch.Tensor) -> torch.Tensor:
    """H = -i Log U (principal branch) for unitary U, via the normal eigendecomposition.
    CPU complex128, deterministic.  Hermitianized against fp asymmetry."""
    Uc = U.detach().to("cpu", torch.complex128)
    lam, V = torch.linalg.eig(Uc)
    theta = torch.angle(lam)                                    # principal eigenphases in (-pi, pi]
    H = V @ torch.diag(theta.to(torch.complex128)) @ torch.linalg.inv(V)
    return 0.5 * (H + H.conj().T)


def _block_unitary(H: torch.Tensor, idx) -> torch.Tensor:
    """exp(i H[idx, idx]) -- the principal-block secular unitary of the split family."""
    ii = torch.tensor(idx, dtype=torch.long)
    Hb = H.index_select(0, ii).index_select(1, ii)
    return torch.matrix_exp(1j * Hb)


def _sobol_seed(seed: int, lo: int, hi: int, i: int) -> int:
    """Deterministic per-(block, solve-index) scramble seed (block = global slice [lo, hi))."""
    return (int(seed) * 1000003 + lo * 10007 + hi * 101 + i * 13 + 7) % (2 ** 31 - 1)


def _graph_moments(U_B: torch.Tensor, i_local: int, kB: torch.Tensor,
                   n_mc: int, seed: int) -> torch.Tensor:
    """
    Eq.-9 graph-chart moments of ONE secular hypersurface block:  mhat_{B\\{i}}(k_B) for
    every row k_B of `kB` [K, b] (int), estimated as the QMC average of e^{-2 pi i <k_B,
    theta>} over theta in X_{U_B}(T) parameterized as a graph over theta_{B\\{i}} (the
    remaining coordinate solved from the LINEAR secular equation; positive-at-0
    orientation).  b = 1 blocks are the exact point variety (no sampling).

    QUADRATURE EXACTNESS (no variety input, pure character orthogonality): a row with
    k_i = 0 has an integrand that never involves the solved coordinate theta_i, so the
    graph integral is the PLAIN free-torus character integral -- exactly 1 if k_B = 0 and
    exactly 0 otherwise.  These rows are returned exactly (they would otherwise inject
    pure MC noise through O(1) minors into every sparse combo's assembly).

    IMPLEMENTATION: the secular det is multilinear, so p(z) = sum_S a_S z^S with
    a_S = (-1)^{|S|} det(U_S) precomputed once per block; the affine-in-z_i split
    p = alpha + beta z_i is then two subset-monomial sums over the b-1 free coordinates
    (alpha: S without i; beta: coefficient a_{S u {i}} on z^{S}) -- exact, and much
    faster than batched determinants.  Evaluation is chunked (memory-bounded at large
    n_mc).  Variance note: low-order exact-character control variates are
    INEFFECTIVE here (the graph character e^{-2 pi i k_i theta_i(theta_free)} has a
    broad Fourier spectrum -- the phase-cancellation variance is irreducible), so the
    raw estimator with a large default budget is used.

    Returns complex128 [K].  Shared samples across all rows => conjugation-equivariant
    (the estimator is exactly Hermitian across +-k row pairs).
    """
    b = U_B.shape[0]
    kBf = kB.to(torch.float64)                                          # [K, b]
    if b == 1:
        # X = {z : 1 - u z = 0} -> single point z* = 1/u on T;  moment = z*^{-k}
        u = U_B[0, 0]
        th = torch.angle(torch.tensor(1.0 + 0j, dtype=torch.complex128) / u) / (2 * math.pi)
        ph = kBf[:, 0] * th
        return torch.exp(-2j * math.pi * ph.to(torch.complex128))
    ki_zero = kB[:, i_local] == 0                                       # exact rows (see above)
    if bool(ki_zero.any()):
        out = torch.zeros(kB.shape[0], dtype=torch.complex128)
        out[ki_zero & (kB == 0).all(dim=1)] = 1.0
        need = ~ki_zero
        if bool(need.any()):
            out[need] = _graph_moments(U_B, i_local, kB[need], n_mc, seed)
        return out
    free_idx = [j for j in range(b) if j != i_local]
    kf = kBf[:, free_idx]                                               # [K, b-1] free parts
    ki = kBf[:, i_local]                                                # [K] solved parts (nonzero)
    # multilinear coefficient tables over the free-coordinate subsets (exact, once)
    from itertools import chain, combinations as icomb
    free_subsets = list(chain.from_iterable(icomb(range(b - 1), r) for r in range(b)))
    a_alpha = torch.stack([secular_coeff(U_B, tuple(1 if j in [free_idx[t] for t in S] else 0
                                                    for j in range(b)))
                           for S in free_subsets])                      # [2^(b-1)]
    a_beta = torch.stack([secular_coeff(U_B, tuple(1 if (j == i_local or
                                                         j in [free_idx[t] for t in S]) else 0
                                                   for j in range(b)))
                          for S in free_subsets])                       # [2^(b-1)]
    eng = torch.quasirandom.SobolEngine(dimension=b - 1, scramble=True, seed=seed)
    acc = torch.zeros(kB.shape[0], dtype=torch.complex128)
    n_kept = 0
    CH = 1 << 19
    for start in range(0, n_mc, CH):
        nc = min(CH, n_mc - start)
        th_free = eng.draw(nc, dtype=torch.float64)                     # [Nc, b-1] in [0,1)
        zf = torch.exp(2j * math.pi * th_free.to(torch.complex128))     # [Nc, b-1]
        mono = torch.ones(nc, len(free_subsets), dtype=torch.complex128)
        for s_idx, S in enumerate(free_subsets):
            for t in S:
                mono[:, s_idx] = mono[:, s_idx] * zf[:, t]
        alpha = mono @ a_alpha                                          # [Nc] p(z_i = 0)
        beta = mono @ a_beta                                            # [Nc] z_i-coefficient
        keep = beta.abs() > 1e-12                                       # degenerate fibers (measure 0)
        zi = -alpha[keep] / beta[keep]
        zi = zi / zi.abs().clamp_min(1e-300)                            # Lee-Yang: |z_i*| = 1 up to fp
        thi = torch.angle(zi) / (2 * math.pi)                           # [Nc'] solved coordinate
        ph = thi.unsqueeze(1) * ki.unsqueeze(0) + th_free[keep] @ kf.t()   # [Nc', K]
        acc = acc + torch.exp(-2j * math.pi * ph.to(torch.complex128)).sum(dim=0)
        n_kept += int(keep.sum())
    return acc / max(n_kept, 1)


def akkv_torus_amplitudes(L: torch.Tensor, U: torch.Tensor, combos,
                          n_mc: int = DEFAULT_AKKV_N_MC, seed: int = 0) -> torch.Tensor:
    """
    AKKV Thm 2.6(3) / Eq.-9 variety-fixed amplitudes at d = 2 for the canonical SECULAR
    SPLIT FAMILY X_split(U) (see the block comment above for the full mathematical route,
    including why the single secular hypersurface provably cannot carry d=2 beats).

        c_{L,k} = sum_{I in C([n],2)} Delta_I(L) * mhat_{[n]\\I}(k)

    with the mhat computed by QMC over the torus part of each split component (graph
    parameterization of the block secular hypersurfaces; Eq.-9 literally, positive-at-0 /
    Thm-4.20 orientation).  var(k)=1 beat modes are generically LIVE; var(k) >= 2 modes
    are exact zeros (structural extinction).  Hermitian: c_{-k} = conj(c_k) to machine
    precision.  Homogeneity: c(aL) = a^2 c(L) exactly (same seed).  Deterministic given
    `seed`; fp64/complex128 internally on CPU; NON-differentiable (constants w.r.t. the
    graph -- an offline numerical table); returns complex128 [M] on CPU (cast to your
    model's complex dtype at the call site, e.g. complex64).

    Parameters
    ----------
    L      : [n, 2] real generator (any device/dtype; detached + fp64 internally)
    U      : [n, n] complex secular unitary (principal-block split via H = -i Log U)
    combos : [M, n] integer combos k, evaluated literally (no canonicalization here)
    n_mc   : QMC samples per (block, solve-index) stream
    seed   : master seed for the scrambled-Sobol streams
    """
    with torch.no_grad():
        Ld = L.detach().to("cpu", torch.float64)
        n, d = Ld.shape
        assert d == 2, "akkv_torus_amplitudes is the d=2 operator (got d=%d)" % d
        assert U.shape[0] == n
        combos_list = [tuple(int(v) for v in row) for row in combos]
        M = len(combos_list)
        H = _principal_log_hermitian(U)
        # 2x2 minors Delta[i, i'] = L[i,0]L[i',1] - L[i,1]L[i',0]  (used with i < i' only)
        Dmin = Ld[:, 0].unsqueeze(1) * Ld[:, 1].unsqueeze(0) \
            - Ld[:, 1].unsqueeze(1) * Ld[:, 0].unsqueeze(0)             # [n, n]

        c = torch.zeros(M, dtype=torch.complex128)
        for cut in range(1, n):                                          # contiguous cuts
            B1, B2 = list(range(0, cut)), list(range(cut, n))
            # which combos are live on this component (both block restrictions var = 0)
            live = [m for m, k in enumerate(combos_list)
                    if _var([k[j] for j in B1]) == 0 and _var([k[j] for j in B2]) == 0]
            if not live:
                continue
            k1 = torch.tensor([[combos_list[m][j] for j in B1] for m in live])   # [K, |B1|]
            k2 = torch.tensor([[combos_list[m][j] for j in B2] for m in live])   # [K, |B2|]
            U1, U2 = _block_unitary(H, B1), _block_unitary(H, B2)
            m1 = torch.stack([_graph_moments(U1, i, k1, n_mc,
                                             _sobol_seed(seed, 0, cut, i))
                              for i in range(len(B1))], dim=1)           # [K, |B1|]
            m2 = torch.stack([_graph_moments(U2, i, k2, n_mc,
                                             _sobol_seed(seed, cut, n, i))
                              for i in range(len(B2))], dim=1)           # [K, |B2|]
            Dblk = Dmin[:cut, cut:].to(torch.complex128)                 # [|B1|, |B2|], i < i'
            contrib = torch.einsum("ka,ab,kb->k", m1, Dblk, m2)          # [K]
            c[torch.tensor(live)] += contrib
        return c


# ----------------------------------------------------------------------------------------
# self-test (exactness of the cumulant recursion)
# ----------------------------------------------------------------------------------------
def _selftest() -> None:
    torch.manual_seed(0)
    # n=1: p_U(z)=1 - U z  =>  log p_U = - sum_k (U^k/k) z^k  =>  c_P(k) = -U^k / k
    U1 = torch.tensor([[0.4 + 0.3j]], dtype=torch.complex128)
    combos = [(1,), (2,), (3,), (4,)]
    cP = secular_log_cumulants(U1, combos)
    u = U1[0, 0]
    exact = torch.stack([-(u ** k) / k for k in (1, 2, 3, 4)])
    err = (cP - exact).abs().max().item()
    assert err < 1e-10, f"n=1 cumulant mismatch: {err}"

    # n=2 multilinear cross-check against a brute-force series log
    A = torch.randn(2, 2, dtype=torch.complex128)
    H = 0.5 * (A + A.conj().T)
    U = torch.matrix_exp(1j * H)                                       # unitary
    # brute force: p_U(z) = det(I - diag(z)U); expand log to degree 4 by direct polynomial powers
    a = {(0, 0): 1 + 0j}
    for S in ((1, 0), (0, 1), (1, 1)):
        a[S] = complex(secular_coeff(U, S))
    # log(1+q) = sum_{m>=1} (-1)^{m+1}/m q^m ; accumulate z^k coeffs up to deg 4
    from collections import defaultdict
    q = {k: v for k, v in a.items() if k != (0, 0)}
    g = defaultdict(complex)
    qpow = {(0, 0): 1 + 0j}
    for m in range(1, 6):
        newp = defaultdict(complex)
        for k1, v1 in qpow.items():
            for k2, v2 in q.items():
                kk = (k1[0] + k2[0], k1[1] + k2[1])
                if kk[0] <= 4 and kk[1] <= 4:
                    newp[kk] += v1 * v2
        qpow = newp
        for kk, v in qpow.items():
            g[kk] += ((-1) ** (m + 1)) / m * v
    test = [(1, 0), (0, 1), (1, 1), (2, 0), (2, 1), (2, 2)]
    cP2 = secular_log_cumulants(U, test)
    for i, k in enumerate(test):
        e = abs(complex(cP2[i]) - g[k])
        assert e < 1e-9, f"n=2 cumulant mismatch at {k}: {e}"
    print("akkv_amplitudes._selftest OK (n=1 closed form + n=2 brute-force series)")


if __name__ == "__main__":
    _selftest()

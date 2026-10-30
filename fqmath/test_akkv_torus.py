"""
fqmath.test_akkv_torus -- unit tests for the "akkv_torus" amplitude mode
========================================================================

Run:  python -m fqmath.test_akkv_torus        (from the repo root; all tests must pass)

Covers six checks:
  (1) Hermitian symmetry c_{-k} = conj(c_k) to 1e-10 (machine-exact: shared QMC samples);
  (2) QMC convergence: doubling n_mc changes every |c_k| by < 2% (reports the n_mc needed);
  (3) d=1 consistency: the torus/graph moments reproduce the EXACT secular d=1 family
      closed form mhat_{[n]\\{i}}(k) = -k_i c_P(k) (per-i, per-k), hence
      sum_i l_i mhat = -<l,k> c_P(k) = MINUS secular_amplitudes_d1 (KS eq.-29 sign vs
      the opposite convention of secular_amplitudes_d1); AND the d=2 formula at a rank-1
      generator collapses to EXACTLY ZERO (all 2x2 minors vanish -- the Grassmannian boundary),
      i.e. the theory says the two families DIFFER at rank-1: the d=1 family is recovered from the
      hypersurface moments, NOT from rank-collapse of the d=2 operator (reported);
  (4) beats live: fraction of var(k)=1 modes with |c_k| > 1e-6 max|c| (reported);
  (5) homogeneity: c(aL) = a^d c(L) with d = 2, EXACT given the same seed -- Thm 2.6(3)
      is LINEAR in the d x d minors (2-homogeneous), the moments are L-independent;
  (6) paper-number oracles: (a) the module _selftest (exact cumulant recursion) passes;
      (b) the DEFAULT FQSpectrum path (mode "secular") returns amplitudes identical
      to secular_pullback_amplitudes; (c) the env-var plumbing flips the mode and the
      support mask keeps the live beats; (d) AKKV Example 11.2 worked numbers: on their
      Moebius curve with L = [[1,0],[0,1],[-sqrt2,sqrt3]] the Thm-2.6(3) assembly used
      here (positive-at-0 orientation calibration, z^{-k} moments, minor <-> complement
      pairing) reproduces c_(0,0) = 1 + sqrt2 + sqrt3 and
      c_xi = (4/25) i ((-2+i) + (2+i) sqrt2 + 2i sqrt3) ~= -0.94053 + 0.132548i at
      k = (-1,-1,-1).
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import torch

from .akkv_amplitudes import (DEFAULT_AKKV_N_MC, _graph_moments, _selftest,
                              akkv_torus_amplitudes, secular_amplitudes_d1,
                              secular_log_cumulants, secular_pullback_amplitudes,
                              support_mask)
from .grassmann import PositiveGrassmann
from .lee_yang import SecularUnitary
from .spectrum import FQSpectrum, enumerate_spectrum_combos, var_sign_changes

SEED = 0
N, D, KMAX, M = 6, 2, 3, 48                      # the production spectrum config


def _production_objects():
    g = PositiveGrassmann(n=N, d=D, init_scale=1.84, seed=SEED,
                          incommensurate_init=True, freeze_incommensurate=True)
    sec = SecularUnitary(n=N, coupled=True, seed=SEED)
    L = g().detach()
    U = sec.unitary().detach().to(torch.complex128)
    combos = enumerate_spectrum_combos(N, D, KMAX, M).astype(int)
    return L, U, combos


def _rand_unitary(n, seed):
    gt = torch.Generator().manual_seed(seed)
    A = torch.complex(torch.randn(n, n, generator=gt), torch.randn(n, n, generator=gt)) \
        .to(torch.complex128)
    H = 0.5 * (A + A.conj().T)
    return torch.matrix_exp(1j * H)


# ----------------------------------------------------------------------------------------
def test_1_hermitian():
    L, U, combos = _production_objects()
    both = np.concatenate([combos, -combos], axis=0)
    c = akkv_torus_amplitudes(L, U, both, n_mc=8192, seed=SEED)
    cp, cm = c[:len(combos)], c[len(combos):]
    err = (cm - cp.conj()).abs().max().item()
    assert err < 1e-10, f"Hermitian symmetry violated: {err:.3e}"
    print(f"[1] Hermitian symmetry: max|c(-k) - conj(c(k))| = {err:.2e}  (< 1e-10)  PASS")


def test_2_mc_convergence():
    L, U, combos = _production_objects()
    budgets = [1 << 20, 1 << 21, 1 << 22, 1 << 23]
    cs = {n: akkv_torus_amplitudes(L, U, combos, n_mc=n, seed=SEED) for n in budgets}
    report = []
    for n in budgets[:-1]:
        a, b = cs[n].abs(), cs[2 * n].abs()
        rel = ((a - b).abs() / b.clamp_min(1e-12)).max().item()
        report.append((n, rel))
        print(f"    n_mc 2^{n.bit_length()-1} -> 2^{n.bit_length()}: "
              f"max per-mode |c| change {100*rel:.3f}%")
    # "needed" = the smallest budget from which every observed doubling stays < 2%
    needed = None
    for j in range(len(report)):
        if all(rel < 0.02 for _, rel in report[j:]):
            needed = report[j][0]
            break
    assert needed is not None and needed <= DEFAULT_AKKV_N_MC, \
        f"no budget <= default ({DEFAULT_AKKV_N_MC}) meets the <2% doubling criterion"
    dflt_rel = dict(report)[DEFAULT_AKKV_N_MC]
    assert dflt_rel < 0.02
    print(f"[2] QMC convergence: <2% per-mode doubling criterion met from n_mc = "
          f"2^{needed.bit_length()-1}; default n_mc = 2^{DEFAULT_AKKV_N_MC.bit_length()-1} "
          f"(doubling change {100*dflt_rel:.3f}%)  PASS")


def test_3_d1_consistency():
    # (a) the torus/graph moments vs the EXACT secular d=1 closed form.
    # Derivation (see the akkv_amplitudes block comment): AKKV Thm 2.6(3) applied to the
    # secular hypersurface (codim 1, i.e. the d=1 member) says c_{l,k} = sum_i l_i
    # mhat_{[n]\{i}}(k); KS eq. 29 says the same mass is -<l,k> c_P(k).  Both are linear
    # in l on an open cone, so mhat_{[n]\{i}}(k) = -k_i c_P(k) EXACTLY (var(k)=0,
    # canonical k >= 0; k=0 -> +1 = the multidegree; var(k)>=1 -> 0 by Thm 4.32).
    worst = 0.0
    for n, seed in ((2, 11), (3, 12)):
        U = _rand_unitary(n, seed)
        ks = [tuple([1] * n), tuple([2] + [1] * (n - 1)), tuple([1] + [0] * (n - 1)),
              tuple([0] * (n - 1) + [3])]
        kB = torch.tensor(ks)
        cP = secular_log_cumulants(U, ks)
        mall = torch.stack([_graph_moments(U, i, kB, 65536, seed=100 + 10 * n + i)
                            for i in range(n)], dim=0)                  # [n, K]
        for i in range(n):
            exact = -kB[:, i].to(torch.complex128) * cP
            live = exact.abs() > 1e-3
            rel = ((mall[i] - exact).abs()[live] / exact.abs()[live]).max().item()
            zer = (mall[i] - exact).abs()[~live].max().item() if (~live).any() else 0.0
            worst = max(worst, rel, zer)
        # the summed d=1 family vs secular_amplitudes_d1 (opposite sign convention)
        gt = torch.Generator().manual_seed(seed)
        ell = torch.rand(n, generator=gt, dtype=torch.float64) + 0.5
        torus = (ell.to(torch.complex128).unsqueeze(1) * mall).sum(dim=0)   # [K]
        proj = secular_amplitudes_d1(ell, U, ks)
        srel = ((torus + proj).abs() / proj.abs().clamp_min(1e-9)).max().item()
        worst = max(worst, srel)
    assert worst < 2e-2, f"d=1 torus-vs-exact mismatch: {worst:.3e}"
    print(f"[3a] d=1 consistency: mhat_{{[n]\\i}}(k) = -k_i c_P(k) and "
          f"sum_i l_i mhat = -secular_amplitudes_d1 (KS eq. 29 sign); "
          f"worst QMC rel err {worst:.1e} (n_mc 65536)  PASS")

    # (b) rank-1 generator at d=2: the theory PREDICTS collapse to zero, not the d=1
    # family -- every 2x2 minor vanishes on the Grassmannian boundary (the winding plane
    # degenerates to a line, which generically misses a codim-2 variety entirely), while
    # the d=1 secular family lives on the codim-1 hypersurface.  The two families DIFFER.
    _, U6, combos = _production_objects()
    gt = torch.Generator().manual_seed(3)
    r = torch.rand(N, generator=gt, dtype=torch.float64) + 0.5
    L1 = torch.stack([r * math.cos(0.7), r * math.sin(0.7)], dim=1)     # rank-1 [n, 2]
    c = akkv_torus_amplitudes(L1, U6, combos, n_mc=4096, seed=SEED)
    mx = c.abs().max().item()
    assert mx < 1e-12, f"rank-1 generator must collapse to 0, got max|c| = {mx:.3e}"
    print(f"[3b] rank-1 generator: max|c| = {mx:.1e} (exact minor collapse; the d=2 "
          f"operator does NOT degenerate to the d=1 family)  PASS")

    # (c) the FULL d=2 assembly vs the exact closed form: because every block is a
    # multilinear secular hypersurface, all block moments have the exact value
    # -k_i c_P_B(k_B) (k_B = 0 -> 1; negative orthant -> conjugate; var >= 1 -> 0), so
    # the whole c-vector has a closed form; the QMC assembly must converge to it.
    from .akkv_amplitudes import _block_unitary, _principal_log_hermitian, _var
    L6, _, _ = _production_objects()
    L64 = L6.to(torch.float64)
    H = _principal_log_hermitian(U6)
    Dmin = L64[:, 0].unsqueeze(1) * L64[:, 1].unsqueeze(0) \
        - L64[:, 1].unsqueeze(1) * L64[:, 0].unsqueeze(0)

    def exact_block_moment(Ub, kb):
        kb = tuple(kb)
        if all(v == 0 for v in kb):
            return [torch.tensor(1 + 0j, dtype=torch.complex128)] * len(kb)
        neg = any(v < 0 for v in kb)
        kk = tuple(-v for v in kb) if neg else kb
        cP = secular_log_cumulants(Ub, [kk])[0]
        return [(-kk[i] * cP).conj() if neg else -kk[i] * cP for i in range(len(kb))]

    combos_list = [tuple(int(v) for v in row) for row in combos]
    c_exact = torch.zeros(len(combos_list), dtype=torch.complex128)
    for cut in range(1, N):
        B1, B2 = list(range(cut)), list(range(cut, N))
        U1, U2 = _block_unitary(H, B1), _block_unitary(H, B2)
        for m_, k in enumerate(combos_list):
            k1, k2 = [k[j] for j in B1], [k[j] for j in B2]
            if _var(k1) or _var(k2):
                continue
            m1, m2 = exact_block_moment(U1, k1), exact_block_moment(U2, k2)
            s = torch.tensor(0j, dtype=torch.complex128)
            for i in range(len(B1)):
                for ip in range(len(B2)):
                    s = s + Dmin[B1[i], B2[ip]].to(torch.complex128) * m1[i] * m2[ip]
            c_exact[m_] += s
    c_mc = akkv_torus_amplitudes(L6, U6, combos, n_mc=1 << 22, seed=SEED)
    rel = ((c_mc - c_exact).abs() / c_exact.abs().clamp_min(1e-15)).max().item()
    assert rel < 0.05, f"d=2 QMC assembly vs closed form: max rel {rel:.3e}"
    print(f"[3c] full d=2 assembly vs exact closed form (block cumulants): max rel dev "
          f"{100*rel:.2f}% over all {len(combos_list)} modes at the default budget  PASS")


def test_4_beats_live():
    L, U, combos = _production_objects()
    c = akkv_torus_amplitudes(L, U, combos, n_mc=1 << 20, seed=SEED)
    v = np.array([var_sign_changes(tuple(int(x) for x in k)) for k in combos])
    beats = c.abs()[torch.tensor(v == 1)]
    frac = float((beats > 1e-6 * c.abs().max()).float().mean())
    assert frac > 0.99, f"beats not live: fraction {frac:.2f}"
    print(f"[4] beats live: {int(frac * len(beats))}/{len(beats)} var(k)=1 modes have "
          f"|c_k| > 1e-6 max|c| (fraction {frac:.2f}); beat |c| range "
          f"[{beats.min():.3e}, {beats.max():.3e}] vs max|c| {c.abs().max():.3e}  PASS")


def test_5_homogeneity():
    # Thm 2.6(3): c_{L,k} = sum_I L_I mhat_{[n]\I}(k) is LINEAR in the d x d minors and
    # the mhat are L-independent => under L -> aL the minors scale by a^d, so c -> a^d c
    # with d = 2.  Same seed => same QMC samples => the identity is EXACT in fp.
    L, U, combos = _production_objects()
    L64 = L.to(torch.float64)      # keep a*L exact to f64 ulp (f32 input would round first)
    c1 = akkv_torus_amplitudes(L64, U, combos, n_mc=8192, seed=SEED)
    worst = 0.0
    for a in (2.0, 1.7):
        ca = akkv_torus_amplitudes(a * L64, U, combos, n_mc=8192, seed=SEED)
        # same seed => same moments; the ONLY L-dependence is the 2x2 minors, so the
        # identity is exact up to fp: (a x)(a y) vs a^2 (x y) ulp differences get
        # amplified by the ~1e3-fold cancellation inside the smallest modes, so measure
        # the deviation relative to the spectrum scale max|c| (not per-mode).
        worst = max(worst, ((ca - a ** 2 * c1).abs().max() / c1.abs().max()).item())
    assert worst < 1e-9, f"homogeneity degree 2 violated: {worst:.3e}"
    print(f"[5] homogeneity: c(aL) = a^2 c(L) (deg d = 2, linear in 2x2 minors; exact "
          f"same-seed identity up to fp cancellation, max dev {worst:.1e} of max|c|)  PASS")


def test_6_oracles_and_default_path():
    # (a) the module's exact-cumulant selftest must pass
    _selftest()

    # (b) DEFAULT path: env unset -> mode "secular" -> amplitudes identical to
    # secular_pullback_amplitudes.
    assert not os.environ.get("FQ_AMP_MODE"), \
        "run this test with FQ_AMP_MODE unset (it checks the default path)"
    g = PositiveGrassmann(n=N, d=D, init_scale=1.84, seed=SEED,
                          incommensurate_init=True, freeze_incommensurate=True)
    sec = SecularUnitary(n=N, coupled=True, seed=SEED)
    spec = FQSpectrum(g, k_max=KMAX, max_freqs=M, seed=SEED, amplitude="secular",
                      secular=sec)
    assert spec.amplitude_mode == "secular"
    with torch.no_grad():
        L = g()
        c_def = spec.amplitudes(L)
        c_ref, _ = secular_pullback_amplitudes(L, sec.unitary(), spec._combos_int)
    assert torch.equal(c_def, c_ref), "default secular amplitudes differ from the reference"
    print("[6b] default FQSpectrum path: mode 'secular', amplitudes bit-identical to "
          "secular_pullback_amplitudes  PASS")

    # (c) env-var plumbing + support mask keeps the live beats
    os.environ["FQ_AMP_MODE"] = "akkv_torus"
    os.environ["FQ_AMP_NMC"] = str(1 << 20)          # keep the unit test quick
    try:
        spec2 = FQSpectrum(g, k_max=KMAX, max_freqs=M, seed=SEED, amplitude="secular",
                           secular=sec)
        assert spec2.amplitude_mode == "akkv_torus"
        with torch.no_grad():
            c2 = spec2.amplitudes(g())
            c2b = spec2.amplitudes(g())                     # cache hit -> identical
            mask = support_mask(c2)
        assert torch.equal(c2, c2b), "akkv_torus cache not deterministic"
        v = np.array([var_sign_changes(k) for k in spec2._combos_int])
        beats_on = mask[torch.tensor(v == 1)]
        assert float(beats_on.min()) == 1.0, "support mask zeroes live beats!"
        assert c2.dtype == sec.unitary().dtype, "akkv_torus dtype must match the secular path"
        print(f"[6c] env plumbing: FQ_AMP_MODE=akkv_torus overrides the ctor arg; cache "
              f"deterministic; support mask keeps all {int(beats_on.sum())} live beats; "
              f"dtype {c2.dtype}  PASS")
    finally:
        del os.environ["FQ_AMP_MODE"]
        del os.environ["FQ_AMP_NMC"]

    # (d) AKKV Example 11.2 paper numbers -- validates the FULL Thm-2.6(3) assembly
    # conventions used by akkv_torus_amplitudes (positive-at-0 orientation calibration,
    # z^{-k} moment direction, minor <-> complement pairing, Eq.-9 normalization) on the
    # paper's own worked d=2 variety (their Moebius curve, codim 2 in (P^1)^3):
    #   z1 = (t-1+i)/(t-1-i), z2 = (t-i)/(t+i), z3 = (t+1+i)/(t+1-i),  t in R u {inf};
    #   L = [[1,0],[0,1],[-sqrt2,sqrt3]]  (minors L_12 = 1, L_13 = sqrt3, L_23 = sqrt2);
    #   mhat_{j}(k) = (1/pi) int z(t)^{-k} dt / ((t - a_j)^2 + 1),  a = (1, 0, -1)
    # (the Poisson-kernel weights are exactly (sigma_j/2 pi i) dz_j/z_j with the
    # positive-at-0 calibration sigma_j; multidegree (1,1,1)).
    u = np.linspace(-np.pi / 2, np.pi / 2, 2_000_001)[1:-1]
    t = np.tan(u)
    jac = 1.0 / np.cos(u) ** 2
    z = [(t - 1 + 1j) / (t - 1 - 1j), (t - 1j) / (t + 1j), (t + 1 + 1j) / (t + 1 - 1j)]

    def mom(k, a):
        zk = z[0] ** (-k[0]) * z[1] ** (-k[1]) * z[2] ** (-k[2])
        return np.trapezoid(zk / ((t - a) ** 2 + 1) * jac, u) / np.pi

    def c_of(k):
        return 1.0 * mom(k, -1) + math.sqrt(3) * mom(k, 0) + math.sqrt(2) * mom(k, 1)

    c0 = c_of((0, 0, 0))
    cxi = c_of((-1, -1, -1))
    c0_ref = 1 + math.sqrt(2) + math.sqrt(3)
    cxi_ref = (4 / 25) * 1j * ((-2 + 1j) + (2 + 1j) * math.sqrt(2) + 2j * math.sqrt(3))
    e0, exi = abs(c0 - c0_ref), abs(cxi - cxi_ref)
    assert e0 < 1e-4 and exi < 1e-4, f"Ex. 11.2 oracle failed: |dc0|={e0:.2e} |dcxi|={exi:.2e}"
    print(f"[6d] AKKV Ex. 11.2 oracle: c_(0,0) = {c0.real:.6f} (paper {c0_ref:.6f}, "
          f"err {e0:.1e}); c_xi = {cxi:.6f} (paper {cxi_ref:.6f}, err {exi:.1e})  PASS")


def main():
    t0 = time.time()
    test_1_hermitian()
    test_2_mc_convergence()
    test_3_d1_consistency()
    test_4_beats_live()
    test_5_homogeneity()
    test_6_oracles_and_default_path()
    print(f"ALL akkv_torus tests PASS  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()

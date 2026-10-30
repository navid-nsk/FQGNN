"""
Run the full MATH-INTEGRITY GATE for the Op-B (eigenphase) + Op-C (secular det) CUDA kernel.

Gates (a kernel is ACCEPTED only if ALL pass):
  1. check_op_b : forward max|err| < 1e-3 AND backward max|err| < 1e-3 vs op_b_reference (N=4096 unitaries)
  2. fp64 gradcheck on a tiny case (N=8, n=6) -- certifies backward == Jacobian-transpose (Wirtinger sign)
  3. Op-B<->Op-C tie : prod_j(1-e^{i theta_j}) == p_U to 1e-4 (a free on-device consistency check)
  4. compare_full_model on SMOKE cache (2012 Jul-Aug) -- pred AND every param grad match to 1e-3
  5. perf : ms/batch fwd & fwd+bwd at N~6300 vs the CPU reference (must be faster)

Run:  python fqcuda/cuda_eig/test_gates.py     (from any dir; adds the repository root to the path;
      on Windows set PYTHONUTF8=1 first).  Gate 4 needs the 2012 July-August raw partitions.
"""
import os
import sys
import time
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for fq_eig_ops

import torch

from task_alpha.verify_kernel import (check_op_b, op_b_reference, op_c_reference,
                                       compare_full_model, _random_unitary_batch)
from fqcuda.fq_eig_cuda import OpBEig, opb_theta, secular_features_cuda
from task_alpha.fq_interp import FQTrajectoryInterpolator

PI = 3.141592653589793


def gate1():
    print("\n=== GATE 1: check_op_b (forward+backward vs reference, N=4096) ===")
    fwd, bwd, ok = check_op_b(opb_theta, N=4096, tol=1e-3)
    return ok, dict(fwd=fwd, bwd=bwd)


def gate2():
    print("\n=== GATE 2: fp64 gradcheck (N=8, n=6) ===")

    def mk(N, seed):
        g = torch.Generator().manual_seed(seed)
        H = torch.randn(6, 6, generator=g, dtype=torch.complex128); H = (H + H.conj().T) / 2
        U = torch.linalg.matrix_exp(1j * H)
        z = torch.exp(1j * torch.rand(N, 6, generator=g, dtype=torch.float64) * 2 * PI).to(torch.complex128)
        return (z.unsqueeze(-1) * U).to("cuda").requires_grad_(True)

    ok_t = torch.autograd.gradcheck(lambda M: OpBEig.apply(M)[0], (mk(8, 1),),
                                    eps=1e-6, atol=1e-4, rtol=1e-3, nondet_tol=1e-5)
    ok_p = torch.autograd.gradcheck(lambda M: OpBEig.apply(M)[1], (mk(8, 2),),
                                    eps=1e-6, atol=1e-4, rtol=1e-3, nondet_tol=1e-5)
    ok_b = torch.autograd.gradcheck(lambda M: OpBEig.apply(M), (mk(8, 3),),
                                    eps=1e-6, atol=1e-4, rtol=1e-3, nondet_tol=1e-5)
    print(f"  theta={ok_t}  pU={ok_p}  joint={ok_b}")
    return (ok_t and ok_p and ok_b), dict(theta=ok_t, pU=ok_p, joint=ok_b)


def gate3():
    print("\n=== GATE 3: Op-B<->Op-C tie  prod(1-e^{i theta}) == p_U ===")
    Mop = _random_unitary_batch(N=4096, n=6, seed=7, device="cuda")
    with torch.no_grad():
        theta, pU = OpBEig.apply(Mop)
        recon = torch.prod(1.0 - torch.exp(1j * theta.to(torch.complex64)), dim=-1)
        tie = (recon - pU).abs().max().item()
        pU_ref = op_c_reference(Mop)
        pu_vs_ref = (pU - torch.complex(pU_ref[:, 0], pU_ref[:, 1])).abs().max().item()
    print(f"  tie={tie:.3e}  pU_vs_det_ref={pu_vs_ref:.3e}")
    return (tie < 1e-4 and pu_vs_ref < 1e-4), dict(tie=tie, pu_vs_ref=pu_vs_ref)


def gate4():
    print("\n=== GATE 4: compare_full_model (SMOKE 2012 Jul-Aug) ===")
    def patch(m): m.secular_features = types.MethodType(secular_features_cuda, m)
    wp, wg, ok = compare_full_model(patch, years=(2012,), months=(7, 8), batch_size=128,
                                    tol=1e-3, device="cuda", seed=0, n_batches=4)
    return ok, dict(pred=wp, grad=wg)


def gate5():
    print("\n=== GATE 5: perf at N=6300 vs ~46 ms CPU reference ===")
    dev = "cuda"; N = 6300
    ref = FQTrajectoryInterpolator(n=6, n_freqs=48, width=96, init_scale=0.5, seed=0, in_feat=6).to(dev)
    L = ref.grassmann().detach()
    pos = (torch.rand(N, 2, device=dev) * 0.5)

    def bench(fn, it, wu, grad):
        for _ in range(wu):
            p = pos.clone().requires_grad_(grad); o = fn(p)
            if grad: (o[0].sum() + o[1].sum()).backward()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(it):
            p = pos.clone().requires_grad_(grad); o = fn(p)
            if grad: (o[0].sum() + o[1].sum()).backward()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / it * 1000

    rf = lambda p: ref.secular_features(p, L)
    cf = lambda p: secular_features_cuda(ref, p, L)
    t_ref_f = bench(rf, 10, 3, False); t_cu_f = bench(cf, 30, 5, False)
    t_ref_b = bench(rf, 10, 3, True);  t_cu_b = bench(cf, 30, 5, True)
    print(f"  forward : CPU {t_ref_f:6.2f} ms -> CUDA {t_cu_f:6.2f} ms ({t_ref_f/t_cu_f:.1f}x)")
    print(f"  fwd+bwd : CPU {t_ref_b:6.2f} ms -> CUDA {t_cu_b:6.2f} ms ({t_ref_b/t_cu_b:.1f}x)")
    return (t_cu_f < t_ref_f), dict(ref_fwd=t_ref_f, cu_fwd=t_cu_f, ref_fb=t_ref_b, cu_fb=t_cu_b)


if __name__ == "__main__":
    results = {}
    for g in (gate1, gate2, gate3, gate4, gate5):
        ok, info = g()
        results[g.__name__] = (ok, info)
    print("\n" + "=" * 60)
    print("GATE SUMMARY")
    allok = True
    for name, (ok, info) in results.items():
        allok &= ok
        print(f"  {name}: {'PASS' if ok else 'FAIL'}   {info}")
    print(f"\nALL GATES: {'PASS' if allok else 'FAIL'}")
    sys.exit(0 if allok else 1)

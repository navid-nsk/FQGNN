"""
task_alpha.verify_kernel  --  the MATH-INTEGRITY GATE for any custom CUDA kernel.

A kernel is accepted ONLY if it reproduces the PyTorch reference on BOTH the forward output and
the backward gradient, to fp32 tolerance, on the SAME inputs (random + real-from-cache).  This is how
we guarantee a kernel computes OUR FQ math (the reference operators below, as in fq_imputer.py) and
not some "standard" substitute: it must match the reference function bit-for-bit (within fp),
gradients included.

Tests use a PORTION of the 62 GB cache (a couple of partitions) -- fast, never the whole dataset.
"""

from __future__ import annotations

import torch

PI = 3.141592653589793


# ---------------------------------------------------------------------------------------
# GOLDEN REFERENCES (the faithful ops, verbatim semantics)
# ---------------------------------------------------------------------------------------
def op_b_reference(Mop: torch.Tensor) -> torch.Tensor:
    """Op-B: sorted eigenphases of M=diag(z)U. EXACTLY as fq_imputer.secular_features (eigvals on CPU)."""
    ev = torch.linalg.eigvals(Mop.cpu()).to(Mop.device)
    theta, _ = torch.sort(torch.angle(ev), dim=-1)
    return theta.to(torch.float32)


def op_c_reference(Mop: torch.Tensor) -> torch.Tensor:
    """Op-C: secular value p_U = det(I - M), returned as [...,2] real/imag (as in secular_features)."""
    I = torch.eye(Mop.shape[-1], dtype=Mop.dtype, device=Mop.device)
    pU = torch.linalg.det(I - Mop)
    return torch.stack([pU.real, pU.imag], dim=-1).to(torch.float32)


def _random_unitary_batch(N=4096, n=6, seed=0, device="cuda"):
    """M = diag(z) U with U=exp(iH) unitary and |z|=1 -- the exact structure of Op-B's input."""
    g = torch.Generator().manual_seed(seed)
    H = torch.randn(n, n, generator=g, dtype=torch.cfloat); H = (H + H.conj().T) / 2
    U = torch.linalg.matrix_exp(1j * H)
    z = torch.exp(1j * torch.rand(N, n, generator=g) * (2 * PI)).to(torch.cfloat)
    return (z.unsqueeze(-1) * U).to(device)


# ---------------------------------------------------------------------------------------
# the GATE: forward + backward must match the reference
# ---------------------------------------------------------------------------------------
def check_op(cand_fn, ref_fn, Mop, tol=1e-3, name="op", complex_grad=True):
    """Compare candidate vs reference on forward AND backward (random cotangent). Returns (fwd, bwd, pass)."""
    Mr = Mop.clone().requires_grad_(True)
    Mc = Mop.clone().requires_grad_(True)
    yr = ref_fn(Mr); yc = cand_fn(Mc)
    fwd = (yr - yc).abs().max().item()
    cot = torch.randn_like(yr)
    (yr * cot).real.sum().backward() if yr.is_complex() else (yr * cot).sum().backward()
    (yc * cot).real.sum().backward() if yc.is_complex() else (yc * cot).sum().backward()
    bwd = (Mr.grad - Mc.grad).abs().max().item()
    ok = max(fwd, bwd) < tol
    print(f"{name:18s} forward max|err|={fwd:.2e} | backward max|err|={bwd:.2e} | tol={tol:.0e} | "
          f"{'PASS' if ok else 'FAIL'}")
    return fwd, bwd, ok


def check_op_b(cand_fn, N=4096, n=6, tol=1e-3, seed=0, device="cuda"):
    """Gate Op-B (eigenphases): candidate vs torch.linalg.eigvals reference, fwd+bwd, on random unitaries."""
    Mop = _random_unitary_batch(N, n, seed, device)
    return check_op(cand_fn, op_b_reference, Mop, tol=tol, name="Op-B eigenphases")


def compare_full_model(patch, years=(2016,), months=(7, 8), batch_size=128, tol=1e-3,
                       device="cuda", seed=0, n_batches=4):
    """End-to-end gate: build the reference model and a candidate model with IDENTICAL weights, swap the
    candidate op via `patch(model)` (a function that monkeypatches one method), run forward+backward on a
    CACHE PORTION, and compare the prediction AND every parameter gradient. The strongest faithfulness test."""
    from .data_raw import build_dataset_raw
    from .data import collate_graphs_interp
    from .fq_interp import FQTrajectoryInterpolator
    import copy

    gs = build_dataset_raw(years=years, months=months, n_mmsi=None, max_files_per_part=None,
                           gap_minutes=90.0, thin_sec=180, max_gap_len=40, n_anchor=8, seed=seed)
    gs = [g for g in gs if int(g.mask.sum()) > 0][: batch_size * n_batches]
    ref = FQTrajectoryInterpolator(n=6, n_freqs=48, width=96, init_scale=0.5, seed=seed,
                                   in_feat=6).to(device)
    cand = copy.deepcopy(ref); patch(cand)                       # identical weights, candidate op swapped in
    worst_pred = worst_grad = 0.0
    for bs in range(0, len(gs), batch_size):
        b = gs[bs:bs + batch_size]
        x, coord, yp, mask, ei, ed, aei = [t.to(device) for t in collate_graphs_interp(b)]
        for m in (ref, cand):
            m.zero_grad(set_to_none=True)
        pr = ref(x, coord, ei, aei); pc = cand(x, coord, ei, aei)
        worst_pred = max(worst_pred, (pr - pc).abs().max().item())
        ((pr[mask] - yp[mask]) ** 2).mean().backward()
        ((pc[mask] - yp[mask]) ** 2).mean().backward()
        for (nm, p1), (_, p2) in zip(ref.named_parameters(), cand.named_parameters()):
            if p1.grad is not None and p2.grad is not None:
                worst_grad = max(worst_grad, (p1.grad - p2.grad).abs().max().item())
    ok = max(worst_pred, worst_grad) < tol
    print(f"FULL MODEL  pred max|err|={worst_pred:.2e} | grad max|err|={worst_grad:.2e} | tol={tol:.0e} | "
          f"{'PASS' if ok else 'FAIL'}  (over {len(gs)} graphs)")
    return worst_pred, worst_grad, ok


# =======================================================================================
# FAITHFUL references for the OTHER ops (EXACTLY as fq_imputer.py computes them),
# and a DUAL-DTYPE (fp32 + fp64) forward+backward gate. A kernel is accepted only if it matches
# the reference in BOTH dtypes. The reference IS the mathematical definition -- so a
# kernel that substitutes "standard" math (periodic FFT, free/real amplitudes, s on Op-B/Dmu,
# a different i*xi rotation, etc.) FAILS, no matter how plausible it looks.
# =======================================================================================
def qfe_spectral_ref(pos, xi, c, s, rho):
    """Op-A'/QFE spectral output [N,2M] = e^{clamp(<rho,x>,-3,3)} * [Re(c e^{iph}), Im(c e^{iph})],
    ph = 2*pi*s*(pos@xi^T). EXACTLY fq_imputer.qfe (sans the separate ac head)."""
    ph = 2 * PI * s * (pos @ xi.t())
    cr, ci = c.real.unsqueeze(0), c.imag.unsqueeze(0)
    re = cr * torch.cos(ph) - ci * torch.sin(ph)
    im = cr * torch.sin(ph) + ci * torch.cos(ph)
    env = torch.exp((pos @ rho).clamp(-3.0, 3.0)).unsqueeze(-1)
    return torch.cat([re, im], dim=-1) * env


def velocity_ref(pos, xi, c, s):
    """Dmu velocity [N,4M] = (2*pi*i*xi_d)*c on the SAME spectrum -- NO s scale, NO envelope, layout
    [re_dx, im_dx, re_dy, im_dy]. EXACTLY fq_imputer.velocity."""
    ph = 2 * PI * s * (pos @ xi.t())
    cos, sin = torch.cos(ph), torch.sin(ph)
    outs = []
    for d in range(2):
        cv = (2 * PI) * xi[:, d].to(c.dtype) * (1j * c)
        cr, ci = cv.real.unsqueeze(0), cv.imag.unsqueeze(0)
        outs += [cr * cos - ci * sin, cr * sin + ci * cos]
    return torch.cat(outs, dim=-1)


def opd_ref(m, coord, xi, c, s, g_cos, g_sin, edge_index, intensive=True):
    """Op-D twisted-Eberlein conv [N,W]: K=cos(phk)@(Re c * g_cos) - sin(phk)@(Im c * g_sin), phk on
    edge delta=coord[dst]-coord[src]; scatter to dst; intensive /deg. EXACTLY fq_imputer Op-D block."""
    src, dst = edge_index[0], edge_index[1]
    phk = 2 * PI * s * ((coord[dst] - coord[src]) @ xi.t())
    Gc = c.real.unsqueeze(-1) * g_cos
    Gs = c.imag.unsqueeze(-1) * g_sin
    K = torch.cos(phk) @ Gc - torch.sin(phk) @ Gs
    agg = torch.zeros(coord.shape[0], m.shape[1], device=m.device, dtype=m.dtype)
    agg.index_add_(0, dst, m[src] * K)
    if intensive:
        deg = torch.zeros(coord.shape[0], 1, device=m.device, dtype=m.dtype)
        deg.index_add_(0, dst, torch.ones(dst.shape[0], 1, device=m.device, dtype=m.dtype))
        agg = agg / deg.clamp_min(1.0)
    return agg


def _dual(ref_fn, cand_fn, make_inputs, grad_keys, name, device="cuda"):
    """Run ref vs candidate in fp32 AND fp64; compare forward AND grad of every leaf in grad_keys.
    Uses a RELATIVE error (|ref-cand| / |ref|) -- an ABSOLUTE tol is unsuitable for large-magnitude
    gradients (e.g. the Dmu velocity, whose 2*pi*xi amplitudes put the fp32 floor at ~1e-3 for BOTH
    ref and kernel). rtol: fp32 1e-3, fp64 1e-6. fp64 (exact) is the real faithfulness proof; fp32
    just confirms the kernel works at fp32 precision. Returns all-pass bool."""
    ok_all = True
    rel = lambda d, r: (d.abs().max() / (r.abs().max() + 1e-30)).item()
    for rdt, cdt, rtol in [(torch.float32, torch.complex64, 1e-3), (torch.float64, torch.complex128, 1e-6)]:
        raw = make_inputs(rdt, cdt, device)
        a, b = {}, {}
        for k, v in raw.items():
            if k in grad_keys:
                a[k] = v.detach().clone().requires_grad_(True)
                b[k] = v.detach().clone().requires_grad_(True)
            else:
                a[k] = v; b[k] = v
        yr, yc = ref_fn(**a), cand_fn(**b)
        fwd = rel(yr - yc, yr)
        cot = torch.randn_like(yr)
        (yr * cot).sum().backward(); (yc * cot).sum().backward()
        bwd = max(rel(a[k].grad - b[k].grad, a[k].grad) for k in grad_keys
                  if a[k].grad is not None and b[k].grad is not None)
        ok = max(fwd, bwd) < rtol; ok_all &= ok
        print(f"{name:9s} {str(rdt).split('.')[-1]:>7}: rel-fwd {fwd:.2e} rel-bwd {bwd:.2e} rtol "
              f"{rtol:.0e} {'PASS' if ok else 'FAIL'}")
    return ok_all


def _qfe_inputs(rdt, cdt, dev, N=2048, M=24):
    g = torch.Generator().manual_seed(0)
    return dict(pos=(torch.randn(N, 2, generator=g) * 0.5).to(dev, rdt),
                xi=torch.randn(M, 2, generator=g).to(dev, rdt),
                c=torch.complex(torch.randn(M, generator=g), torch.randn(M, generator=g)).to(dev, cdt),
                s=torch.tensor(0.7).to(dev, rdt), rho=(torch.randn(2, generator=g) * 0.3).to(dev, rdt))


def _vel_inputs(rdt, cdt, dev, N=2048, M=24):
    d = _qfe_inputs(rdt, cdt, dev, N, M); d.pop("rho"); return d


def _opd_inputs(rdt, cdt, dev, N=1024, M=24, W=32, E=6000):
    g = torch.Generator().manual_seed(0)
    return dict(m=torch.randn(N, W, generator=g).to(dev, rdt),
                coord=(torch.randn(N, 2, generator=g) * 0.5).to(dev, rdt),
                xi=torch.randn(M, 2, generator=g).to(dev, rdt),
                c=torch.complex(torch.randn(M, generator=g), torch.randn(M, generator=g)).to(dev, cdt),
                s=torch.tensor(0.7).to(dev, rdt),
                g_cos=(0.1 * torch.randn(M, W, generator=g)).to(dev, rdt),
                g_sin=(0.1 * torch.randn(M, W, generator=g)).to(dev, rdt),
                edge_index=torch.randint(0, N, (2, E), generator=g).to(dev))


def check_qfe(cand):       # cand(pos, xi, c, s, rho) -> [N,2M]
    return _dual(qfe_spectral_ref, cand, _qfe_inputs, ["pos", "xi", "c", "s", "rho"], "QFE")


def check_velocity(cand):  # cand(pos, xi, c, s) -> [N,4M]
    return _dual(velocity_ref, cand, _vel_inputs, ["pos", "xi", "c", "s"], "Dmu-vel")


def check_opd(cand):       # cand(m, coord, xi, c, s, g_cos, g_sin, edge_index) -> [N,W]
    return _dual(opd_ref, cand, _opd_inputs, ["m", "coord", "xi", "c", "s", "g_cos", "g_sin"], "Op-D")


if __name__ == "__main__":
    # self-consistency: every reference vs ITSELF must be ~0 (fp32 AND fp64) -- validates the gate.
    print("self-consistency (reference vs reference, must be ~0, both dtypes):")
    check_op_b(op_b_reference)
    check_qfe(qfe_spectral_ref)
    check_velocity(velocity_ref)
    check_opd(opd_ref)

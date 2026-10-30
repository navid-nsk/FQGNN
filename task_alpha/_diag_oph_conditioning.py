"""OP-H CONDITIONING DIAGNOSTIC.  Diagnostic only -- no model change.  On the TRAINED checkpoints
of the run in FQ_DIAG_RUN_DIR (all 5 seeds), over EVERY gap in train+val+test:

  (a) effective row rank of the WEIGHTED system Bw = B*diag(w2) at the pinv cutoff
      (svals > rcond * sigma_max), vs the actual row count R in {2,3,4} -- and the same for the
      unweighted B, to separate metric-induced from geometric near-degeneracy;
  (b) sigma_min / sigma_max of both;
  (c) the REALIZED production-fp32 boundary residual  B @ coef - d  measured in fp64 -- the direct
      check of the exact boundary interpolation + velocity match that the Op-H solve is built
      to satisfy;
  (d) coefficient norm, metric dynamic range (min w2 after max-normalization), gate value;
  (e) the same solve redone in fp64-pinv and in fp64-QR (reduced QR of Bw^T), residuals
      compared -- to quantify what, if anything, these alternatives would buy.

A per-gap guard asserts the replica prediction matches the production oph_readout() output, so the
diagnostic provably measures the deployed solve, not a reimplementation.  Also runs two synthetic
edge cases (exact +/-omega frequency pairs; stationary trajectory / true-zero velocity, which the
sentinel convention maps to R=2 by design).

Run:  python -m task_alpha._diag_oph_conditioning
Out:  <run_dir>/conditioning_audit.csv (per gap) + printed summary.

Env knobs (the defaults evaluate the fqnative2_5seed run):
  FQ_DIAG_RUN_DIR   checkpoint run dir (default task_alpha/runs/fqnative2_5seed)
  FQ_DIAG_SOBOLEV=1 the sobolev config: ctor +dmu_source="blend", metric_backbone="sobolev2";
                    the replica feeds the BLEND boundary jet and the omega^-2 bending-backbone
                    metric so the guard still proves replica == deployed forward solve.
"""
import csv
import math
import os
import sys
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import torch

from task_alpha.data import collate_graphs_interp
from task_alpha.data_raw import build_dataset_raw
from task_alpha.fq_opk import boundary_velocity
from task_alpha.fq_oph import oph_readout
from task_alpha.fqdeep_native import FQDeepNative, boundary_velocity_blend
from task_alpha.meltin_proof import _bvec
from task_alpha.train_stream import _split

DEV = "cuda"
PI = math.pi
RCOND = 1e-6
MIN_PHASE = 0.25
RUN_DIR = os.environ.get("FQ_DIAG_RUN_DIR", "task_alpha/runs/fqnative2_5seed")
SOBOLEV = os.environ.get("FQ_DIAG_SOBOLEV") == "1"
CK = RUN_DIR + "/seed{}/best.pt"
OUTCSV = RUN_DIR + "/conditioning_audit.csv"


def replica_solve(at, ay, av, qt, om_all, eta, gate_rho, w):
    """Byte-faithful replica of fq_oph.oph_readout (weighted path, fp32 on DEV) that EXPOSES the
    internals (B, d, w2, coef, g) for measurement.  Returns None if the degenerate-module affine
    fallback fires (no system to measure)."""
    dt = ay.dtype
    at = at.to(dt); qt = qt.to(dt)
    idx = torch.argsort(at)
    ts, ys, vs = at[idx], ay[idx], av[idx]
    gapi = int(torch.argmax(ts[1:] - ts[:-1]))
    a, b = ts[gapi], ts[gapi + 1]
    pa, pb = ys[gapi], ys[gapi + 1]
    va, vb = vs[gapi].to(dt), vs[gapi + 1].to(dt)
    ba = (b - a).clamp_min(1e-9)
    slope = (pb - pa) / ba
    A_q = pa.unsqueeze(0) + (qt - a).unsqueeze(-1) * slope.unsqueeze(0)
    om = om_all.to(dt)
    keep = om.abs() * ba > MIN_PHASE
    if int(keep.sum()) < 2:
        return None
    omk = om[keep]
    etak = eta.to(dt)[keep]
    wk = w.to(dt)[keep].clamp_min(1e-6)
    wk = wk / wk.max()
    w2 = torch.cat([wk, wk])
    t0 = 0.5 * (a + b)
    tau_a, tau_b = a - t0, b - t0
    tau_q = qt - t0

    def val_row(tau):
        return torch.cat([torch.cos(omk * tau), torch.sin(omk * tau)])

    def der_row(tau):
        return torch.cat([-omk * torch.sin(omk * tau), omk * torch.cos(omk * tau)])

    zero2 = torch.zeros(2, device=ay.device, dtype=dt)
    rows = [val_row(tau_a), val_row(tau_b)]
    rhs = [zero2, zero2]
    om_ref = omk.abs().mean().clamp_min(1e-9)
    vel_mismatch = []
    for v_end, tau_end in ((va, tau_a), (vb, tau_b)):
        if bool((v_end != 0).any()):
            rows.append(der_row(tau_end) / om_ref)
            rhs.append((v_end - slope) / om_ref)
            vel_mismatch.append((v_end - slope).norm())
    B = torch.stack(rows)
    d = torch.stack(rhs)
    Bw = B * w2.unsqueeze(0)
    Bp = torch.linalg.pinv(Bw, rcond=RCOND)
    c_part = Bp @ d
    if gate_rho is not None and vel_mismatch:
        dbar = torch.stack(vel_mismatch).sum() / (len(vel_mismatch) * slope.norm() + 1e-6)
        g = torch.tanh(torch.nn.functional.softplus(gate_rho).to(dt) * dbar)
    else:
        g = torch.ones((), device=ay.device, dtype=dt)
    z = torch.cat([etak[:, 0, :], etak[:, 1, :]], dim=0)
    c_free = z - Bp @ (Bw @ z)
    coef_u = c_part + g * c_free
    coef = w2.unsqueeze(-1) * coef_u
    Cq = torch.cat([torch.cos(tau_q.unsqueeze(-1) * omk), torch.sin(tau_q.unsqueeze(-1) * omk)], dim=-1)
    pred = A_q + Cq @ coef
    return dict(B=B, d=d, w2=w2, z=z, g=g, coef=coef, pred=pred, K=int(keep.sum()))


def measure(rep):
    """fp64 measurements on the production solve + fp64-pinv and fp64-QR alternatives."""
    B64 = rep["B"].double().cpu(); d64 = rep["d"].double().cpu()
    w264 = rep["w2"].double().cpu(); z64 = rep["z"].double().cpu()
    g64 = float(rep["g"])
    coef64 = rep["coef"].double().cpu()
    R = B64.shape[0]
    Bw64 = B64 * w264.unsqueeze(0)
    sv_w = torch.linalg.svdvals(Bw64)
    sv_u = torch.linalg.svdvals(B64)
    eff_w = int((sv_w > RCOND * sv_w[0]).sum())
    eff_u = int((sv_u > RCOND * sv_u[0]).sum())
    res32 = B64 @ coef64 - d64
    d_norm = float(torch.linalg.norm(d64))
    # fp64 pinv alternative
    Bp64 = torch.linalg.pinv(Bw64, rcond=RCOND)
    u64 = Bp64 @ d64 + g64 * (z64 - Bp64 @ (Bw64 @ z64))
    cp64 = w264.unsqueeze(-1) * u64
    resp64 = Bw64 @ u64 - d64
    # fp64 QR alternative, on the weighted matrix
    try:
        Q, Rm = torch.linalg.qr(Bw64.t(), mode="reduced")
        y = torch.linalg.solve_triangular(Rm.t(), d64, upper=False)
        uq = Q @ y + g64 * (z64 - Q @ (Q.t() @ z64))
        resq = Bw64 @ uq - d64
        res_qr = float(resq.abs().max())
        qr_ok = bool(torch.isfinite(resq).all())
    except Exception:
        res_qr, qr_ok = float("nan"), False
    return dict(
        R=R, K=rep["K"],
        ratio_w=float(sv_w[-1] / sv_w[0]), ratio_u=float(sv_u[-1] / sv_u[0]),
        eff_rank_w=eff_w, eff_rank_u=eff_u,
        rank_fail_w=int(eff_w < R), rank_fail_u=int(eff_u < R),
        res_fp32_max=float(res32.abs().max()),
        res_fp32_scaled=float(torch.linalg.norm(res32) / (d_norm + 1.0)),
        res_fp64pinv_max=float(resp64.abs().max()),
        res_qr_max=res_qr, qr_ok=int(qr_ok),
        d_norm=d_norm, coef_norm=float(torch.linalg.norm(coef64)),
        w2_min=float(w264.min()), gate=g64)


def synthetic_edge_cases():
    """Synthetic edge cases not already covered by real data configurations."""
    print("--- synthetic edge cases ---", flush=True)
    torch.manual_seed(0)
    M = 8
    t = torch.sort(torch.rand(M)).values; t[M // 2:] += 2.0
    qt = t[M // 2 - 1] + (t[M // 2] - t[M // 2 - 1]) * torch.rand(20)
    p0, v0 = torch.randn(2), torch.randn(2)
    y_c = (p0 + t.unsqueeze(-1) * v0
           + 0.4 * torch.sin(3 * t).unsqueeze(-1) * torch.tensor([1.0, -1.0]))
    av_c = boundary_velocity(t, y_c)
    eta = 3.0 * torch.randn(48, 2, 2)
    rho = torch.tensor(math.log(math.expm1(4.0)))
    w = torch.nn.functional.softplus(torch.randn(48)) + 1e-3
    # exact +/-omega pairs (duplicate cos columns, negated sin columns)
    om_half = 2 * PI * 2.5 * (torch.rand(24) + 0.2) * 3.0
    om_sym = torch.cat([om_half, -om_half])
    rep = replica_solve(t, y_c, av_c, qt, om_sym, eta, rho, w)
    mm = measure(rep)
    print(f"  +/-omega pairs: R={mm['R']} eff_rank_w={mm['eff_rank_w']} "
          f"ratio_w={mm['ratio_w']:.2e} res_fp32={mm['res_fp32_max']:.2e} "
          f"{'PASS' if mm['rank_fail_w'] == 0 and mm['res_fp32_max'] < 1e-3 else 'FAIL'}", flush=True)
    # stationary trajectory: all anchors at one point -> velocities are EXACT zeros = the sentinel
    y_s = torch.ones(M, 2) * torch.tensor([1.0, -2.0])
    av_s = boundary_velocity(t, y_s)
    rep_s = replica_solve(t, y_s, av_s, qt, om_sym, eta, rho, w)
    mm_s = measure(rep_s)
    pred_dev = float((rep_s["pred"] - torch.tensor([1.0, -2.0])).abs().max())
    print(f"  stationary (true-zero vel == sentinel by design): R={mm_s['R']} (velocity rows "
          f"dropped) res_fp32={mm_s['res_fp32_max']:.2e} pred==point dev={pred_dev:.2e} "
          f"{'PASS' if mm_s['R'] == 2 and pred_dev < 1e-4 else 'FAIL'}", flush=True)


def main():
    synthetic_edge_cases()
    allg = build_dataset_raw(years=(2016, 2017, 2018), months=(7, 8), gap_minutes=360,
                             max_gap_len=500, min_span_factor=1.5, n_anchor=8, clean_only=True,
                             n_mmsi=None, max_files_per_part=None, seed=0)
    allg = [g for g in allg if int(g.mask.sum()) > 0 and _split(g.mmsi) is not None]
    splits = {nm: [g for g in allg if _split(g.mmsi) == nm] for nm in ("train", "val", "test")}
    print({k: len(v) for k, v in splits.items()}, flush=True)

    rows_out = []
    guard_max = 0.0
    for seed in range(5):
        skw = dict(dmu_source="blend", metric_backbone="sobolev2") if SOBOLEV else {}
        m = FQDeepNative(n=6, n_freqs=48, width=96, init_scale=1.84, seed=0, in_feat=6, layers=3,
                         jacobi_R=4, oph_weighted=True, **skw).to(DEV)
        m.use_cuda = True
        b0 = collate_graphs_interp(splits["val"][:4])
        wk0 = {"gap_minutes": torch.tensor(
            [float(g.gap_minutes) for g in splits["val"][:4]])} \
            if getattr(m, "dmu_source", "fd") == "blend" else {}
        m(b0[0].to(DEV), b0[1].to(DEV), b0[4].to(DEV), b0[6].to(DEV), _bvec(splits["val"][:4]),
          **wk0)
        m.load_state_dict(torch.load(CK.format(seed), map_location=DEV)["state_dict"])
        m.eval()
        for split, graphs in splits.items():
            with torch.no_grad():
                for s0 in range(0, len(graphs), 64):
                    b = graphs[s0:s0 + 64]
                    x, co, yp, mk, ei, _, ae = collate_graphs_interp(b)
                    x, co, ei, ae = x.to(DEV), co.to(DEV), ei.to(DEV), ae.to(DEV)
                    batch = _bvec(b)
                    h, _, _, s_, P, xi0 = m.embed(x, co, ei)
                    om_all = 2 * PI * s_ * xi0
                    asrc = ae[0]
                    held_all = x[:, 5] > 0.5
                    gid_adst = batch[ae[1]]
                    for gi_ in range(len(b)):
                        g = b[gi_]
                        held = torch.where(held_all & (batch == gi_))[0]
                        anc = torch.unique(asrc[gid_adst == gi_])
                        if held.numel() == 0 or anc.numel() < 2:
                            continue
                        at, ay = co[anc, 0], x[anc, :2]
                        av = boundary_velocity(at, ay).to(h.dtype)
                        if getattr(m, "dmu_source", "fd") == "blend":
                            # the deployed forward's blend jet (instrument sog/cog where observed)
                            av = boundary_velocity_blend(at, ay, x, anc, co, held,
                                                         float(g.gap_minutes), av)
                        pool = h[anc].mean(0)
                        eta = m.eta_head(pool).view(m.M, 2, 2)
                        filt = m.w_head(pool) @ P.t()
                        w = torch.nn.functional.softplus(filt) + 1e-3
                        if getattr(m, "metric_backbone", "flat") == "sobolev2":
                            w = w * (1.0 / om_all.abs().clamp_min(1e-6) ** 2)  # bending backbone
                        rep = replica_solve(at, ay, av, co[held, 0], om_all, eta, m.gate_rho, w)
                        if rep is None:
                            continue
                        # guard: replica == deployed operator
                        pref = oph_readout(at, ay, av, co[held, 0], om_all, eta,
                                           gate_rho=m.gate_rho, min_phase=MIN_PHASE,
                                           use_vel=m.use["dmu"], rcond=RCOND, w=w)
                        gdev = float((pref - rep["pred"]).abs().max())
                        guard_max = max(guard_max, gdev)
                        mm = measure(rep)
                        rows_out.append({"seed": seed, "split": split, "mmsi": int(g.mmsi),
                                         "gap_min": round(float(g.gap_minutes), 1),
                                         "guard_dev": f"{gdev:.2e}", **mm})
        print(f"[seed {seed}] done ({len(rows_out)} rows so far), guard max {guard_max:.2e}",
              flush=True)
        del m
        torch.cuda.empty_cache()

    with open(OUTCSV, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        wcsv.writeheader(); wcsv.writerows(rows_out)

    # ---- summary ----
    def col(k):
        return np.array([r[k] for r in rows_out], dtype=float)

    n = len(rows_out)
    print(f"\n================ SUMMARY over {n} (seed, gap) solves ================", flush=True)
    print(f"replica-vs-production guard: max |pred diff| = {guard_max:.3e} "
          f"({'OK -- audit measures the deployed solve' if guard_max < 1e-4 else 'MISMATCH!'})",
          flush=True)
    for name in ("rank_fail_w", "rank_fail_u"):
        print(f"{name}: {int(col(name).sum())} / {n}", flush=True)
    for name in ("ratio_w", "ratio_u"):
        v = col(name)
        print(f"{name}: min {v.min():.3e} | p1 {np.percentile(v, 1):.3e} | "
              f"med {np.median(v):.3e}  (pinv cutoff at {RCOND:.0e})", flush=True)
    for name in ("res_fp32_max", "res_fp32_scaled", "res_fp64pinv_max", "res_qr_max"):
        v = col(name); v = v[np.isfinite(v)]
        print(f"{name}: med {np.median(v):.3e} | p99 {np.percentile(v, 99):.3e} | "
              f"max {v.max():.3e}", flush=True)
    v = col("coef_norm")
    print(f"coef_norm: med {np.median(v):.3e} | max {v.max():.3e}", flush=True)
    v = col("w2_min")
    print(f"w2_min (metric dynamic range): min {v.min():.3e} | med {np.median(v):.3e}", flush=True)
    print(f"qr_ok: {int(col('qr_ok').sum())} / {n}", flush=True)
    rr = col("R")
    for rv in (2, 3, 4):
        print(f"R={rv}: {int((rr == rv).sum())} gaps", flush=True)
    # worst gap by fp32 residual
    wi = int(np.argmax(col("res_fp32_max")))
    print("WORST fp32-residual gap:", rows_out[wi], flush=True)


if __name__ == "__main__":
    main()

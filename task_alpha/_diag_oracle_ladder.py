"""ORACLE LADDER: where is the remaining headroom between the trained model and the per-gap
oracle, and which part is reachable FQ-natively?  All probes are closed-form / per-gap
optimization on the TRAINED fqnative2_5seed checkpoint -- no training, no model changes.
Ground truth used as diagnostic target only.
Reads the raw-AIS 6h gap dataset (data_raw.build_dataset_raw, cached) and the five
runs/fqnative2_5seed checkpoints; prints the ladder (no files written).
"""
import math
import random
import sys
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import torch

from task_alpha.data import collate_graphs_interp
from task_alpha.data_raw import build_dataset_raw
from task_alpha.fq_opk import boundary_velocity
from task_alpha.fqdeep_native import FQDeepNative
from task_alpha.meltin_proof import _bvec
from task_alpha.metrics import gcd_errors, model_errors
from task_alpha.train_stream import _split

DEV = "cuda"
PI = math.pi
CK = "task_alpha/runs/fqnative2_5seed/seed{}/best.pt"

allg = build_dataset_raw(years=(2016, 2017, 2018), months=(7, 8), gap_minutes=360, max_gap_len=500,
                         min_span_factor=1.5, n_anchor=8, clean_only=True, n_mmsi=None,
                         max_files_per_part=None, seed=0)
allg = [g for g in allg if int(g.mask.sum()) > 0 and _split(g.mmsi) is not None]
va = [g for g in allg if _split(g.mmsi) == "val"]

def build_model(seed_ck):
    m = FQDeepNative(n=6, n_freqs=48, width=96, init_scale=1.84, seed=0, in_feat=6, layers=3,
                     jacobi_R=4, oph_weighted=True).to(DEV)
    m.use_cuda = True
    b0 = collate_graphs_interp(va[:4])
    m(b0[0].to(DEV), b0[1].to(DEV), b0[4].to(DEV), b0[6].to(DEV), _bvec(va[:4]))
    m.load_state_dict(torch.load(CK.format(seed_ck), map_location=DEV)["state_dict"])
    m.eval()
    return m

m = build_model(0)

def second_diff(ts, ys, idxs):
    """Acceleration at the gap-edge anchor from the 3 same-side anchors nearest the gap.
    idxs = sorted indices of that side (ascending time).  Returns [2] or None."""
    if len(idxs) < 3:
        return None
    i1, i2, i3 = idxs[-3], idxs[-2], idxs[-1]
    t1, t2, t3 = ts[i1], ts[i2], ts[i3]
    if t3 - t1 < 1e-9:
        return None
    v12 = (ys[i2] - ys[i1]) / max(float(t2 - t1), 1e-9)
    v23 = (ys[i3] - ys[i2]) / max(float(t3 - t2), 1e-9)
    return 2.0 * (v23 - v12) / max(float(t3 - t1), 1e-9)

def gap_pack(graphs):
    packs = []
    with torch.no_grad():
        for s0 in range(0, len(graphs), 64):
            b = graphs[s0:s0 + 64]
            x, co, yp, mk, ei, _, ae = collate_graphs_interp(b)
            x, co, ei, ae = x.to(DEV), co.to(DEV), ei.to(DEV), ae.to(DEV)
            batch = _bvec(b)
            h, _, _, s_, P, xi0 = m.embed(x, co, ei)
            om_all = (2 * PI * s_ * xi0).double().cpu()
            Pj = m._jacobi(m.quasicrystal()[1]).double().cpu()          # [M, R+1] Jacobi at s_k
            asrc = ae[0]
            held_all = x[:, 5] > 0.5
            gid_adst = batch[ae[1]]
            for gi_ in range(len(b)):
                g = b[gi_]
                held = torch.where(held_all & (batch == gi_))[0]
                anc = torch.unique(asrc[gid_adst == gi_])
                if held.numel() == 0 or anc.numel() < 2:
                    continue
                at, ay = co[anc, 0].double().cpu(), x[anc, :2].double().cpu()
                av = boundary_velocity(at, ay)
                qt = co[held, 0].double().cpu()
                ytrue = yp[held.cpu()].double()
                idx = torch.argsort(at)
                ts, ys, vs = at[idx], ay[idx], av[idx]
                gj = int(torch.argmax(ts[1:] - ts[:-1]))
                a_, b_ = ts[gj], ts[gj + 1]
                pa, pb = ys[gj], ys[gj + 1]
                vA, vB = vs[gj], vs[gj + 1]
                ba = float(b_ - a_)
                slope = (pb - pa) / max(ba, 1e-9)
                A_q = pa.unsqueeze(0) + (qt - a_).unsqueeze(-1) * slope.unsqueeze(0)
                keep = om_all.abs() * ba > 0.25
                if int(keep.sum()) < 2:
                    continue
                omk = om_all[keep]
                K = omk.numel()
                t0 = 0.5 * float(a_ + b_)
                ta, tb = float(a_) - t0, float(b_) - t0
                tq = qt - t0
                def vr(tau): return torch.cat([torch.cos(omk * tau), torch.sin(omk * tau)])
                def dr(tau): return torch.cat([-omk * torch.sin(omk * tau), omk * torch.cos(omk * tau)])
                def ar(tau): return torch.cat([-omk**2 * torch.cos(omk * tau), -omk**2 * torch.sin(omk * tau)])
                om1 = omk.abs().mean().clamp_min(1e-9)
                om2 = (omk**2).mean().clamp_min(1e-9)
                rows = [vr(ta), vr(tb)]
                rhs = [torch.zeros(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)]
                for vE, tE in ((vA, ta), (vB, tb)):
                    if bool((vE != 0).any()):
                        rows.append(dr(tE) / om1)
                        rhs.append((vE - slope) / om1)
                # acceleration rows (leak-safe same-side 2nd differences); A'' = 0
                accL = second_diff(ts, ys, list(range(gj + 1)))
                accR = second_diff(ts, ys, list(range(len(ts) - 1, gj, -1)))   # nearest-first ordering
                arows, arhs = [], []
                if accL is not None:
                    arows.append(ar(ta) / om2); arhs.append(accL / om2)
                if accR is not None:
                    arows.append(ar(tb) / om2); arhs.append(accR / om2)
                # learned metric from the trained head
                pool = h[anc].mean(0)
                filt = m.w_head(pool.float()) @ m._jacobi(m.quasicrystal()[1]).t()
                w_tr = (torch.nn.functional.softplus(filt) + 1e-3).double().cpu()[keep]
                Cq = torch.cat([torch.cos(tq.unsqueeze(-1) * omk), torch.sin(tq.unsqueeze(-1) * omk)], -1)
                packs.append(dict(g=g, B=torch.stack(rows), d=torch.stack(rhs),
                                  Ba=(torch.stack(rows + arows), torch.stack(rhs + arhs)) if arows else None,
                                  Cq=Cq, A=A_q, y=ytrue, w_tr=w_tr, Pk=Pj[keep],
                                  hanc_mean=h[anc].mean(0).double().cpu(),
                                  hanc_cat=h[anc].double().cpu(), slope=slope, ba=ba))
    return packs

def solve_w(B, d, w):
    w2 = torch.cat([w, w]); w2 = w2 / w2.max()
    Bw = B * w2.unsqueeze(0)
    return w2.unsqueeze(-1) * (torch.linalg.pinv(Bw, rcond=1e-6) @ d)

def ratio_of(packs, pred_fn):
    fq, gc = [], []
    for p in packs:
        g = p["g"]
        gp = g.x[:, :2].clone().float()
        gp[torch.tensor(g.mask.numpy().astype(bool))] = pred_fn(p).float()
        fe = model_errors(gp, g)
        if len(fe):
            fq.append(float(np.median(fe))); gc.append(float(np.median(gcd_errors(g))))
    return float(np.median(fq) / np.median(gc))

print("packing val...", flush=True)
packs = gap_pack(va)
print(f"{len(packs)} gaps packed", flush=True)
uni = lambda p: torch.ones(p["Cq"].shape[1] // 2, dtype=torch.float64)

# ---- rung 0: references ----
r_cf   = ratio_of(packs, lambda p: p["A"] + p["Cq"] @ solve_w(p["B"], p["d"], uni(p)))
r_tr   = ratio_of(packs, lambda p: p["A"] + p["Cq"] @ solve_w(p["B"], p["d"], p["w_tr"]))
print(f"\n[L0] closed form, uniform metric (eta=0)      : {r_cf:.4f}")
print(f"[L0] trained metric (native model, eta~0)     : {r_tr:.4f}", flush=True)

# ---- rung 1: ACCELERATION Hermite rows (extra leak-safe observable; closed form, no learning) ----
def pred_acc(p, w_fn):
    if p["Ba"] is None:
        return p["A"] + p["Cq"] @ solve_w(p["B"], p["d"], w_fn(p))
    Ba, da = p["Ba"]
    return p["A"] + p["Cq"] @ solve_w(Ba, da, w_fn(p))
n_acc = sum(1 for p in packs if p["Ba"] is not None)
r_acc_u = ratio_of(packs, lambda p: pred_acc(p, uni))
r_acc_t = ratio_of(packs, lambda p: pred_acc(p, lambda q: q["w_tr"]))
print(f"\n[L1] + acceleration rows ({n_acc}/{len(packs)} gaps), uniform metric : {r_acc_u:.4f}")
print(f"[L1] + acceleration rows, TRAINED metric                    : {r_acc_t:.4f}", flush=True)

# ---- rungs 2-3: ORACLE metric (subsample for speed) ----
sub = packs[:250]
r_cf_sub = ratio_of(sub, lambda p: p["A"] + p["Cq"] @ solve_w(p["B"], p["d"], uni(p)))
r_tr_sub = ratio_of(sub, lambda p: p["A"] + p["Cq"] @ solve_w(p["B"], p["d"], p["w_tr"]))

def oracle_metric(p, mode, steps=250, lr=0.08, use_acc=False):
    B, d = (p["Ba"] if (use_acc and p["Ba"] is not None) else (p["B"], p["d"]))
    K = p["Cq"].shape[1] // 2
    if mode == "free":
        u = torch.zeros(K, dtype=torch.float64, requires_grad=True)
        mk = lambda: torch.nn.functional.softplus(u) + 1e-3
    else:  # jacobi-5 through the module's variety coordinate
        th = torch.zeros(p["Pk"].shape[1], dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            th[0] = 1.0
        mk = lambda: torch.nn.functional.softplus(p["Pk"] @ th) + 1e-3
    opt = torch.optim.Adam([u if mode == "free" else th], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        pr = p["A"] + p["Cq"] @ solve_w(B, d, mk())
        loss = ((pr - p["y"]) ** 2).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        return p["A"] + p["Cq"] @ solve_w(B, d, mk())

print(f"\n[subset n={len(sub)}] closed-form {r_cf_sub:.4f} | trained metric {r_tr_sub:.4f}", flush=True)
r_oj  = ratio_of(sub, lambda p: oracle_metric(p, "jacobi"))
print(f"[L2] ORACLE metric, JACOBI-5 (current head ceiling)         : {r_oj:.4f}", flush=True)
r_ofk = ratio_of(sub, lambda p: oracle_metric(p, "free"))
print(f"[L3] ORACLE metric, FREE-K (metric channel ceiling)         : {r_ofk:.4f}", flush=True)
r_ofa = ratio_of(sub, lambda p: oracle_metric(p, "free", use_acc=True))
print(f"[L3+] ORACLE free-K metric + acceleration rows              : {r_ofa:.4f}", flush=True)

# ---- rung 4: pool-bottleneck probe (predictability of the oracle Jacobi metric) ----
half = len(sub) // 2
def jacobi_target(p):
    th = torch.zeros(p["Pk"].shape[1], dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        th[0] = 1.0
    opt = torch.optim.Adam([th], lr=0.08)
    for _ in range(250):
        opt.zero_grad()
        pr = p["A"] + p["Cq"] @ solve_w(p["B"], p["d"],
                                        torch.nn.functional.softplus(p["Pk"] @ th) + 1e-3)
        ((pr - p["y"]) ** 2).mean().backward(); opt.step()
    return th.detach()
print("\ncomputing oracle-Jacobi targets for the probe...", flush=True)
Ys = torch.stack([jacobi_target(p) for p in sub])
Xm = torch.stack([p["hanc_mean"] for p in sub])
def ridge_r2(X, Y):
    mu, sd = X[:half].mean(0), X[:half].std(0).clamp_min(1e-6)
    Xn = (X - mu) / sd
    W = torch.linalg.solve(Xn[:half].t() @ Xn[:half] + 10.0 * torch.eye(X.shape[1], dtype=torch.float64),
                           Xn[:half].t() @ (Y[:half] - Y[:half].mean(0)))
    pr = Xn[half:] @ W + Y[:half].mean(0)
    ssr = ((Y[half:] - pr) ** 2).sum(); sst = ((Y[half:] - Y[half:].mean(0)) ** 2).sum()
    return float(1 - ssr / sst)
r2_mean = ridge_r2(Xm, Ys)
Xc = torch.stack([torch.cat([p["hanc_cat"].mean(0), p["hanc_cat"].max(0).values,
                             p["hanc_cat"][0], p["hanc_cat"][-1]]) for p in sub])
r2_cat = ridge_r2(Xc, Ys)
print(f"[L4] oracle-Jacobi predictability: mean-pool R^2 = {r2_mean:.3f} | mean+max+ends R^2 = {r2_cat:.3f}",
      flush=True)

# ---- rung 5: residual anatomy on the trained model ----
al, cr, thirds = [], [], np.zeros(3)
cnt = np.zeros(3)
for p in packs:
    pr = p["A"] + p["Cq"] @ solve_w(p["B"], p["d"], p["w_tr"])
    res = (p["y"] - pr)
    u = p["slope"] / p["slope"].norm().clamp_min(1e-9)
    along = (res @ u).abs()
    cross = (res - (res @ u).unsqueeze(-1) * u.unsqueeze(0)).norm(dim=-1)
    al.append(float(along.median())); cr.append(float(cross.median()))
    Q = res.shape[0]
    for k in range(Q):
        b_ = min(2, int(3 * k / max(Q, 1)))
        thirds[b_] += float(res[k].norm()); cnt[b_] += 1
print(f"\n[L5] residual anatomy (trained metric): median ALONG-track {np.median(al):.4f} vs "
      f"CROSS-track {np.median(cr):.4f} (coord units)")
print(f"[L5] mean |residual| by gap third (early/mid/late): "
      f"{thirds[0]/max(cnt[0],1):.4f} / {thirds[1]/max(cnt[1],1):.4f} / {thirds[2]/max(cnt[2],1):.4f}",
      flush=True)

# ---- rung 6: 5-seed ensemble ----
preds5 = []
for sd in range(5):
    ms = build_model(sd)
    outs = {}
    with torch.no_grad():
        for s0 in range(0, len(va), 64):
            b = va[s0:s0 + 64]
            x, co, yp, mk, ei, _, ae = collate_graphs_interp(b)
            pr = ms(x.to(DEV), co.to(DEV), ei.to(DEV), ae.to(DEV), _bvec(b)).float().cpu()
            off = 0
            for gi_, g in enumerate(b):
                ng = g.x.shape[0]
                outs[(s0, gi_)] = pr[off:off + ng]; off += ng
    preds5.append(outs)
    del ms; torch.cuda.empty_cache()
fq, gc = [], []
for s0 in range(0, len(va), 64):
    b = va[s0:s0 + 64]
    for gi_, g in enumerate(b):
        gp = torch.stack([preds5[sd][(s0, gi_)] for sd in range(5)]).mean(0)
        fe = model_errors(gp, g)
        if len(fe):
            fq.append(float(np.median(fe))); gc.append(float(np.median(gcd_errors(g))))
print(f"\n[L6] 5-seed ENSEMBLE val ratio: {float(np.median(fq)/np.median(gc)):.4f} "
      f"(single-seed trained ~0.672-0.681)", flush=True)

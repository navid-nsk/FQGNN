"""DISTILLATION PROBE: is the oracle metric PREDICTABLE from leak-safe inputs?
Compute oracle-Jacobi metric targets on TRAIN gaps (leak-free: training already uses that truth),
train a small MLP from [pooled h + boundary kinematics] -> targets, then evaluate the PREDICTED
metric's val ratio on the same 250-gap subset as the ladder.  This turns the L2 oracle ceiling
of the ladder into an estimate of what is reachable.  Also tests the anisotropic (chord-frame)
metric variant at oracle level.
Reads the raw-AIS 6h gap dataset (data_raw.build_dataset_raw, cached) and
runs/fqnative2_5seed/seed0/best.pt; prints the results (no files written).
"""
import math
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

allg = build_dataset_raw(years=(2016, 2017, 2018), months=(7, 8), gap_minutes=360, max_gap_len=500,
                         min_span_factor=1.5, n_anchor=8, clean_only=True, n_mmsi=None,
                         max_files_per_part=None, seed=0)
allg = [g for g in allg if int(g.mask.sum()) > 0 and _split(g.mmsi) is not None]
tr = [g for g in allg if _split(g.mmsi) == "train"]
va = [g for g in allg if _split(g.mmsi) == "val"]

m = FQDeepNative(n=6, n_freqs=48, width=96, init_scale=1.84, seed=0, in_feat=6, layers=3,
                 jacobi_R=4, oph_weighted=True).to(DEV)
m.use_cuda = True
b0 = collate_graphs_interp(tr[:4])
m(b0[0].to(DEV), b0[1].to(DEV), b0[4].to(DEV), b0[6].to(DEV), _bvec(tr[:4]))
m.load_state_dict(torch.load("task_alpha/runs/fqnative2_5seed/seed0/best.pt", map_location=DEV)["state_dict"])
m.eval()


def gap_pack(graphs):
    packs = []
    with torch.no_grad():
        for s0 in range(0, len(graphs), 64):
            b = graphs[s0:s0 + 64]
            x, co, yp, mk, ei, _, ae = collate_graphs_interp(b)
            x, co, ei, ae = x.to(DEV), co.to(DEV), ei.to(DEV), ae.to(DEV)
            batch = _bvec(b)
            h, _, _, s_, _, xi0 = m.embed(x, co, ei)
            om_all = (2 * PI * s_ * xi0).double().cpu()
            Pj = m._jacobi(m.quasicrystal()[1]).double().cpu()
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
                t0 = 0.5 * float(a_ + b_)
                ta, tb = float(a_) - t0, float(b_) - t0
                tq = qt - t0
                def vr(tau): return torch.cat([torch.cos(omk * tau), torch.sin(omk * tau)])
                def dr(tau): return torch.cat([-omk * torch.sin(omk * tau), omk * torch.cos(omk * tau)])
                om1 = omk.abs().mean().clamp_min(1e-9)
                rows = [vr(ta), vr(tb)]
                rhs = [torch.zeros(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)]
                for vE, tE in ((vA, ta), (vB, tb)):
                    if bool((vE != 0).any()):
                        rows.append(dr(tE) / om1)
                        rhs.append((vE - slope) / om1)
                # leak-safe boundary kinematics for the predictor
                u = slope / slope.norm().clamp_min(1e-9)
                uperp = torch.tensor([-u[1], u[0]], dtype=torch.float64)
                kin = torch.tensor([
                    float(slope.norm()), ba, float((pb - pa).norm()),
                    float(vA @ u), float(vA @ uperp), float(vB @ u), float(vB @ uperp),
                    float(vA.norm()), float(vB.norm()),
                    float((vA - slope).norm()), float((vB - slope).norm()),
                    float(torch.dot(vA, vB) / (vA.norm() * vB.norm()).clamp_min(1e-9)),
                ], dtype=torch.float64)
                packs.append(dict(g=g, B=torch.stack(rows), d=torch.stack(rhs),
                                  Cq=torch.cat([torch.cos(tq.unsqueeze(-1) * omk),
                                                torch.sin(tq.unsqueeze(-1) * omk)], -1),
                                  A=A_q, y=ytrue, Pk=Pj[keep],
                                  hmean=h[anc].mean(0).double().cpu(), kin=kin,
                                  u=u, uperp=uperp))
    return packs


def solve_w(B, d, w):
    w2 = torch.cat([w, w]); w2 = w2 / w2.max()
    Bw = B * w2.unsqueeze(0)
    return w2.unsqueeze(-1) * (torch.linalg.pinv(Bw, rcond=1e-6) @ d)


def solve_w_aniso(p, w_al, w_cr):
    """Anisotropic chord-frame metric: rotate the 2-target system into (along, cross), solve each
    component with its own metric, rotate back.  Exactness per component is unchanged."""
    R = torch.stack([p["u"], p["uperp"]])                      # [2,2] rows = along/cross
    d_rot = p["d"] @ R.t()
    c_al = solve_w(p["B"], d_rot[:, :1], w_al)
    c_cr = solve_w(p["B"], d_rot[:, 1:], w_cr)
    c_rot = torch.cat([c_al, c_cr], dim=1)                     # [2K, 2] in rotated frame
    return c_rot @ R                                           # back to x/y


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


def oracle_theta(p, aniso=False, steps=250, lr=0.08):
    nj = p["Pk"].shape[1]
    th = torch.zeros(nj * (2 if aniso else 1), dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        th[0] = 1.0
        if aniso:
            th[nj] = 1.0
    opt = torch.optim.Adam([th], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        if aniso:
            w1 = torch.nn.functional.softplus(p["Pk"] @ th[:nj]) + 1e-3
            w2 = torch.nn.functional.softplus(p["Pk"] @ th[nj:]) + 1e-3
            pr = p["A"] + p["Cq"] @ solve_w_aniso(p, w1, w2)
        else:
            pr = p["A"] + p["Cq"] @ solve_w(p["B"], p["d"],
                                            torch.nn.functional.softplus(p["Pk"] @ th) + 1e-3)
        ((pr - p["y"]) ** 2).mean().backward()
        opt.step()
    return th.detach()


print("packing...", flush=True)
import random
random.Random(0).shuffle(tr)
Ptr = gap_pack(tr[:900])
Pva = gap_pack(va)
sub = Pva[:250]
print(f"{len(Ptr)} train / {len(Pva)} val packed", flush=True)

r_tr_sub = ratio_of(sub, lambda p: p["A"] + p["Cq"] @ solve_w(p["B"], p["d"],
    torch.nn.functional.softplus(p["Pk"] @ torch.tensor([1.0] + [0.0] * (p["Pk"].shape[1] - 1),
                                                        dtype=torch.float64)) + 1e-3))
print(f"[ref] uniform-ish Jacobi metric on subset: {r_tr_sub:.4f}", flush=True)

# ---- anisotropic ORACLE (is the chord-frame split worth anything at ceiling level?) ----
r_oracle_iso = ratio_of(sub, lambda p: p["A"] + p["Cq"] @ solve_w(p["B"], p["d"],
    torch.nn.functional.softplus(p["Pk"] @ oracle_theta(p)) + 1e-3))
print(f"[A1] ORACLE Jacobi-5 isotropic (recheck)   : {r_oracle_iso:.4f}", flush=True)
def pred_aniso_oracle(p):
    th = oracle_theta(p, aniso=True)
    nj = p["Pk"].shape[1]
    w1 = torch.nn.functional.softplus(p["Pk"] @ th[:nj]) + 1e-3
    w2 = torch.nn.functional.softplus(p["Pk"] @ th[nj:]) + 1e-3
    return p["A"] + p["Cq"] @ solve_w_aniso(p, w1, w2)
r_oracle_aniso = ratio_of(sub, pred_aniso_oracle)
print(f"[A2] ORACLE Jacobi-5 ANISOTROPIC (chord frame): {r_oracle_aniso:.4f}", flush=True)

# ---- distillation probe: oracle targets on TRAIN gaps -> MLP -> predicted metric on val ----
print("computing train oracle targets...", flush=True)
TH = torch.stack([oracle_theta(p) for p in Ptr])               # [Ntr, 5]
X = torch.stack([torch.cat([p["hmean"], p["kin"]]) for p in Ptr]).float()
Y = TH.float()
mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
Xn = (X - mu) / sd

torch.manual_seed(0)
net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.GELU(),
                          torch.nn.Linear(64, 64), torch.nn.GELU(),
                          torch.nn.Linear(64, Y.shape[1]))
opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
n_tr = int(0.9 * len(Xn))
for ep in range(400):
    opt.zero_grad()
    loss = ((net(Xn[:n_tr]) - Y[:n_tr]) ** 2).mean()
    loss.backward(); opt.step()
with torch.no_grad():
    vl = ((net(Xn[n_tr:]) - Y[n_tr:]) ** 2).mean()
    base = ((Y[:n_tr].mean(0) - Y[n_tr:]) ** 2).mean()
print(f"[B1] distill MLP: target MSE {float(vl):.4f} vs predict-mean {float(base):.4f} "
      f"(R^2 = {1 - float(vl) / float(base):.3f})", flush=True)

def pred_distilled(p):
    xx = ((torch.cat([p["hmean"], p["kin"]]).float() - mu) / sd)
    th = net(xx).double().detach()
    w = torch.nn.functional.softplus(p["Pk"] @ th) + 1e-3
    return p["A"] + p["Cq"] @ solve_w(p["B"], p["d"], w)
r_distill = ratio_of(sub, pred_distilled)
print(f"[B2] DISTILLED metric on val subset         : {r_distill:.4f}  "
      f"(trained head 0.6144; oracle 0.4116)", flush=True)

# also: distilled metric from KINEMATICS ONLY (no embeddings) -- how much does h even matter?
Xk = torch.stack([p["kin"] for p in Ptr]).float()
muk, sdk = Xk.mean(0), Xk.std(0).clamp_min(1e-6)
torch.manual_seed(0)
netk = torch.nn.Sequential(torch.nn.Linear(Xk.shape[1], 64), torch.nn.GELU(),
                           torch.nn.Linear(64, 64), torch.nn.GELU(),
                           torch.nn.Linear(64, Y.shape[1]))
optk = torch.optim.Adam(netk.parameters(), lr=1e-3, weight_decay=1e-4)
Xkn = (Xk - muk) / sdk
for ep in range(400):
    optk.zero_grad()
    ((netk(Xkn[:n_tr]) - Y[:n_tr]) ** 2).mean().backward(); optk.step()
def pred_distilled_kin(p):
    xx = ((p["kin"].float() - muk) / sdk)
    th = netk(xx).double().detach()
    w = torch.nn.functional.softplus(p["Pk"] @ th) + 1e-3
    return p["A"] + p["Cq"] @ solve_w(p["B"], p["d"], w)
r_dk = ratio_of(sub, pred_distilled_kin)
print(f"[B3] DISTILLED metric, KINEMATICS-ONLY input: {r_dk:.4f}", flush=True)

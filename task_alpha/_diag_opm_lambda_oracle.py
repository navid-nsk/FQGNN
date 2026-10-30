"""LAMBDA DIAGNOSTIC: per-gap ORACLE lambda vs the trained lambda-head, on the FROZEN Op-M
seed-0 checkpoint (runs/fqnative_opm_5seed/seed0/best.pt).

Question: the lambda-head outputs per-anchor soft precisions
lambda_j = softplus(kappa_raw) * softplus(theta_lambda . z_j).  In the fqnative_opm_5seed
checkpoints kappa_raw stays at its init (~ -10) and gamma_raw = 0 exactly in every seed -- so
this script measures
(i) what a per-gap ORACLE scalar exterior weight scale lambda* (fixed per-anchor precision,
Tier-C gate still applied inside opm_readout: Lambda_eff = g * lambda) could buy on the frozen
checkpoint, (ii) what the trained head actually outputs, and (iii) whether the head's
observables carry any signal about lambda* (rank correlations).

Measurement only -- no training, no parameter mutation (attributes opm_lam_mode / opm_gamma_env
are eval-time switches by design).  Logged to runs/conform_baseline/opm_lambda_oracle/.

Run (repo root, env FQ_AMP_MODE=akkv_torus FQ_AMP_NMC=262144 FQ_READOUT_MODE=opm):
    python -m task_alpha._diag_opm_lambda_oracle [--gamma 0.5] [--batched 1]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch

from ._conform_subsample import _gm, build_trained, load_subsample
from .data import collate_graphs_interp
from .experiment_compare import _sinuosity
from .meltin_proof import _bvec
from .metrics import gcd_errors, model_errors

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "task_alpha/runs/conform_baseline/opm_lambda_oracle"
CK = "task_alpha/runs/fqnative_opm_5seed/seed0/best.pt"

# fixed per-anchor precision grid (the exterior weight scale); "off" = lam=None (exact-parity).
# The grid reaches up toward the hard-interpolation limit (lambda -> inf).
LGRID = ["off", 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0,
         300.0, 1e3, 3e3, 1e4]
# (no 1e5: lambda * fp32-roundoff of G makes I + G-tilde numerically indefinite and the fp64
#  Cholesky fails; a measurement-range limit, not a model configuration)


def eval_pergap(m, sub):
    """Per-gap (model_med, gcd_med) in a deterministic order shared by every call."""
    m.eval()
    fq, gc = [], []
    with torch.no_grad():
        for s0 in range(0, len(sub), 64):
            b = sub[s0:s0 + 64]
            x, coord, yp, mask, ei, ed, aei = collate_graphs_interp(b)
            pred = m(x.to(DEV), coord.to(DEV), ei.to(DEV), aei.to(DEV), _bvec(b),
                     gap_minutes=_gm(b)).float().cpu()
            off = 0
            for g in b:
                ng = g.x.shape[0]; gp = pred[off:off + ng]; off += ng
                fe = model_errors(gp, g)
                if len(fe):
                    fq.append(float(np.median(fe))); gc.append(float(np.median(gcd_errors(g))))
    return np.array(fq), np.array(gc)


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", default="auto",
                    help="'auto' (pick 0 or 0.5, whichever gives the lower subsample ratio) | float | 'raw'")
    ap.add_argument("--batched", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    assert os.environ.get("FQ_READOUT_MODE") == "opm", "launch with FQ_READOUT_MODE=opm"
    if a.batched:
        os.environ["FQ_OPM_BATCHED"] = "1"

    sub, ids = load_subsample()
    m = build_trained()
    b0 = collate_graphs_interp(sub[:4])
    m(b0[0].to(DEV), b0[1].to(DEV), b0[4].to(DEV), b0[6].to(DEV), _bvec(sub[:4]),
      gap_minutes=_gm(sub[:4]))
    m.load_state_dict(torch.load(CK, map_location=DEV)["state_dict"])
    kap = float(torch.nn.functional.softplus(m.opm_kappa_raw))
    print(f"[ckpt] kappa_raw={float(m.opm_kappa_raw):.6f} -> kappa_lambda={kap:.3e} | "
          f"gamma_raw={float(m.opm_gamma_raw):.6f} | lam_head |W|={float(m.opm_lam_head.weight.norm()):.4f}",
          flush=True)

    # ---- operating gamma: the checkpoint was trained with gamma env-fixed (gamma_raw == 0 exactly);
    # score the subsample under both candidates and report -- the lambda grid then runs at --gamma.
    cand = {}
    for gname, gval in (("0 (raw)", None), ("0.5", 0.5)):
        m.opm_gamma_env = gval
        m.opm_lam_mode = "head"
        fq, gc = eval_pergap(m, sub)
        cand[gname] = float(np.median(fq) / np.median(gc))
        print(f"[gamma-cand] gamma={gname}: subsample ratio {cand[gname]:.4f}", flush=True)
    if a.gamma == "auto":
        gam_use = 0.5 if cand["0.5"] <= cand["0 (raw)"] else None
    elif a.gamma == "raw":
        gam_use = None
    else:
        gam_use = float(a.gamma)
    m.opm_gamma_env = gam_use
    print(f"[operating point] gamma_env = {gam_use}", flush=True)

    # ---- trained-head lambda per gap (loop-order instrumentation of _opm_lambda) ----
    lam_rec = []
    orig = m._opm_lambda

    def rec(at, ay, h_anc, x_anc):
        lam = orig(at, ay, h_anc, x_anc)
        lam_rec.append(lam.detach().float().cpu().numpy())
        return lam

    m._opm_lambda = rec
    m.opm_lam_mode = "head"
    fq_head, gc_ref = eval_pergap(m, sub)
    m._opm_lambda = orig
    n = len(fq_head)
    assert len(lam_rec) == n, (len(lam_rec), n)
    lam_mean = np.array([v.mean() for v in lam_rec])
    lam_max = np.array([v.max() for v in lam_rec])
    r_head = float(np.median(fq_head) / np.median(gc_ref))
    print(f"[head] ratio {r_head:.4f} | per-gap mean lambda: median {np.median(lam_mean):.3e} "
          f"max {lam_max.max():.3e}", flush=True)

    # ---- the fixed-lambda grid ----
    errs = np.zeros((len(LGRID), n))
    ratios = {}
    for i, lv in enumerate(LGRID):
        m.opm_lam_mode = "off" if lv == "off" else str(lv)
        fq, gc = eval_pergap(m, sub)
        assert len(fq) == n
        errs[i] = fq
        ratios[str(lv)] = float(np.median(fq) / np.median(gc))
        print(f"[grid] lambda={lv}: ratio {ratios[str(lv)]:.4f}", flush=True)
    m.opm_lam_mode = "head"

    # ---- oracle ----
    best_i = errs.argmin(0)
    lam_star = np.array([0.0 if LGRID[i] == "off" else float(LGRID[i]) for i in best_i])
    err_star = errs.min(0)
    r_oracle = float(np.median(err_star) / np.median(gc_ref))
    i_off = LGRID.index("off")
    helped = errs[best_i, np.arange(n)] < errs[i_off, np.arange(n)] - 1e-9
    rel_gain = (errs[i_off, np.arange(n)] - err_star) / np.maximum(errs[i_off, np.arange(n)], 1e-9)
    print(f"\n[ORACLE] per-gap best fixed lambda: ratio {r_oracle:.4f} "
          f"(off {ratios['off']:.4f}, head {r_head:.4f})", flush=True)
    print(f"[ORACLE] gaps preferring lambda>0: {int(helped.sum())}/{n} "
          f"({100 * helped.mean():.1f}%) | median lambda* over those: "
          f"{np.median(lam_star[helped]) if helped.any() else float('nan'):.3g} | "
          f"median rel err gain on helped gaps {100 * np.median(rel_gain[helped]) if helped.any() else 0:.1f}%",
          flush=True)
    if helped.any():
        rg = 100 * rel_gain[helped]
        print(f"[ORACLE] rel err gain on helped gaps: mean {rg.mean():.2f}% | P90 "
              f"{np.percentile(rg, 90):.2f}% | max {rg.max():.2f}% | "
          f"gaps gaining >2%: {int((rg > 2).sum())}", flush=True)
        hurt = errs[-1, np.arange(n)] > errs[i_off, np.arange(n)] * 1.02
        print(f"[ORACLE] gaps HURT >2% by the largest grid lambda: {int(hurt.sum())}/{n}",
              flush=True)

    # ---- observables vs lambda* ----
    sinu = np.array([_sinuosity(g) for g in sub if True])[:n]
    gapm = np.array([float(g.gap_minutes) for g in sub])[:n]
    nanc = np.array([v.size for v in lam_rec])
    print("\n[CORR] Spearman rho vs log10(lambda* + 1e-6) on ALL gaps:", flush=True)
    tgt = np.log10(lam_star + 1e-6)
    for nm, v in (("trained head per-gap mean lambda", lam_mean),
                  ("trained head per-gap max lambda", lam_max),
                  ("sinuosity", sinu), ("gap_minutes", gapm), ("n_anchors", nanc)):
        print(f"    {nm:<34s} rho = {spearman(v, tgt):+.3f}", flush=True)
    print("[CORR] Spearman rho of head-mean-lambda vs lambda* on gaps where oracle helps: "
          f"{spearman(lam_mean[helped], lam_star[helped]):+.3f} (n={int(helped.sum())})", flush=True)

    # ---- head internals: what moves theta_lambda . z? ----
    with torch.no_grad():
        W = m.opm_lam_head.weight[0].float().cpu().numpy()
    print(f"\n[HEAD] kappa_lambda = {kap:.3e} (init 4.541e-5; moved {kap / 4.5412e-5 - 1:+.2%})",
          flush=True)
    print(f"[HEAD] |w|: tau/T {abs(W[0]):.4f} snr {abs(W[1]):.4f} headc {abs(W[2]):.4f} "
          f"h_j-block norm {np.linalg.norm(W[3:]):.4f} (init-expected {0.02 * math.sqrt(len(W)):.3f})",
          flush=True)

    with open(f"{OUT}/pergap_seed0.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "gcd_med", "err_off", "err_head", "lam_head_mean", "lam_head_max",
                    "lam_star", "err_star", "sinuosity", "gap_min", "n_anchors"]
                   + [f"err_lam_{lv}" for lv in LGRID])
        for j in range(n):
            w.writerow([ids[j], round(gc_ref[j], 4), round(errs[i_off, j], 4),
                        round(fq_head[j], 4), f"{lam_mean[j]:.4e}", f"{lam_max[j]:.4e}",
                        lam_star[j], round(err_star[j], 4),
                        round(float(sinu[j]), 4) if np.isfinite(sinu[j]) else "",
                        round(gapm[j], 1), int(nanc[j])] + [round(errs[i, j], 4) for i in range(len(LGRID))])
    json.dump({"gamma_env": gam_use, "kappa_lambda": kap, "ratios": ratios,
               "ratio_head": r_head, "ratio_oracle": r_oracle,
               "n_helped": int(helped.sum()), "n": n},
              open(f"{OUT}/summary_seed0.json", "w"), indent=1)
    print(f"\nsaved {OUT}/pergap_seed0.csv + summary_seed0.json", flush=True)


if __name__ == "__main__":
    main()

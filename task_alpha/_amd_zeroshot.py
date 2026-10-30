"""ZERO-SHOT READOUT HARNESS: eval-time gamma / lambda settings on a frozen Op-M checkpoint.

Evaluates a FROZEN Op-M checkpoint on the fixed 200-gap subsample or the FULL 774-gap 6h cell
under an explicit (gamma, lambda) dial configuration, and logs a tagged row + per-gap CSV +
raw predictions (for bit-identity comparisons, see --diff).

Protocol:
  * FQ_AMP_REFRESH is popped -- the served amplitude table is the exact trained-U table.
  * gamma comparisons are conclusive on --scope full only; sub is for parity checks and screens.

Dial configuration (eval-time attributes, the _diag_opm_lambda_oracle pattern):
  --gamma none|<float>|hat|hatp     hatp = the cell-constant law gamma' =
                                    clamp(1-(vbar+beta2)/mbar, 0, 1), a CELL CONSTANT from
                                    --vbar/--mbar (cell-median v_hat/m_hat, fit on TRAINING
                                    gaps by _amd_fit_gammaprime) and --beta2 (cell
                                    transfer-defect moment).
  --lam off|<float>|head|c1|c2|c3   c2 = the C2 closed-form lambda-hat (fq_opm);
                                    c1/c3 = the reparameterized head candidates (fqdeep_native).
  --force-c1-init                   after ckpt load, re-arm kappa_raw at softplus^-1(0.05)
                                    (the C1 init) -- isolates the effect of that init on a
                                    trained checkpoint.

Run (repo root):  python -m task_alpha._amd_zeroshot --tag sub_g05 --scope sub --gamma 0.5 --lam off
Out: task_alpha/runs/amd/<tag>/{per_gap.csv,preds.pt,summary.json} + runs/amd/amd_log.csv
Measurement only: no training, no parameter mutation beyond the documented eval-time switches.
"""
from __future__ import annotations

import os

os.environ.setdefault("FQ_AMP_MODE", "akkv_torus")
os.environ.setdefault("FQ_AMP_NMC", "262144")
os.environ["FQ_READOUT_MODE"] = "opm"
os.environ.setdefault("FQ_OPM_BATCHED", "1")
os.environ.pop("FQ_AMP_REFRESH", None)    # serve the exact trained-U amplitude table only

import argparse
import csv
import json
import math
import pickle
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch

from .data import collate_graphs_interp
from .experiment_compare import _sinuosity
from .fqdeep_native import FQDeepNative
from .fq_opm import gamma_prime
from .meltin_proof import _bvec
from .metrics import gcd_errors, model_errors

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "task_alpha/runs/amd"
CK_DEFAULT = "task_alpha/runs/fqnative_opm_5seed/seed0/best.pt"
HERMITE = "task_alpha/runs/gap_sweep/our_processed/gap_6h/hermite/seed0/per_gap.csv"


def _gm(b):
    return torch.tensor([float(g.gap_minutes) for g in b], device=DEV)


def build_model(warm, ckpt):
    m = FQDeepNative.production(seed=0).to(DEV)
    assert m.readout_mode == "opm"
    m.use_cuda = True
    with torch.no_grad():
        m.s_raw.fill_(math.log(math.expm1(2.5)))
    b0 = collate_graphs_interp(warm)
    m(b0[0].to(DEV), b0[1].to(DEV), b0[4].to(DEV), b0[6].to(DEV), _bvec(warm),
      gap_minutes=_gm(warm))                       # materialize lazy shapes BEFORE load
    m.load_state_dict(torch.load(ckpt, map_location=DEV)["state_dict"])
    m.eval()
    return m


def eval_pergap(m, graphs, bs=64):
    """Per-gap median NM error, index-aligned with graphs; also returns raw predictions."""
    med = np.full(len(graphs), np.nan)
    preds = []
    with torch.no_grad():
        for s in range(0, len(graphs), bs):
            b = graphs[s:s + bs]
            x, coord, yp, mask, ei, ed, aei = collate_graphs_interp(b)
            pred = m(x.to(DEV), coord.to(DEV), ei.to(DEV), aei.to(DEV), _bvec(b),
                     gap_minutes=_gm(b)).float().cpu()
            preds.append(pred)
            off = 0
            for j, g in enumerate(b):
                ng = g.x.shape[0]; gp = pred[off:off + ng]; off += ng
                fe = model_errors(gp, g)
                if len(fe):
                    med[s + j] = float(np.median(fe))
    return med, torch.cat(preds)


def clustered_ci(err_a, err_b, gcd, mmsi, B=10000, seed=0):
    """Vessel-clustered bootstrap CI95 of ratio(err_a) - ratio(err_b) (as in _diag_d1_gamma_oracle)."""
    ok = np.isfinite(err_a) & np.isfinite(err_b) & np.isfinite(gcd)
    ea, eb, gg, mm = err_a[ok], err_b[ok], gcd[ok], mmsi[ok]
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(mm == v) for v in np.unique(mm)]
    d = []
    for _ in range(B):
        rows = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        gmed = np.median(gg[rows])
        d.append(np.median(ea[rows]) / gmed - np.median(eb[rows]) / gmed)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--scope", choices=("sub", "full"), default="sub")
    ap.add_argument("--gamma", default="none", help="none|<float>|hat|hatp")
    ap.add_argument("--vbar", type=float, default=None)
    ap.add_argument("--mbar", type=float, default=None)
    ap.add_argument("--beta2", type=float, default=0.0)
    ap.add_argument("--lam", default="off", help="off|<float>|head|c1|c2|c3")
    ap.add_argument("--ckpt", default=CK_DEFAULT)
    ap.add_argument("--force-c1-init", action="store_true")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--diff", default="", help="preds.pt to bit-diff against (identity gate)")
    a = ap.parse_args()

    with open("task_alpha/data/our_processed/gap_6h/test.pkl", "rb") as f:
        te_all = pickle.load(f)
    if a.scope == "sub":
        ids = json.load(open("task_alpha/runs/conform_baseline/subsample_ids.json"))
    else:
        ids = list(range(len(te_all)))
    graphs = [te_all[i] for i in ids]
    herm = {int(r["gap_id"]): float(r["model_med"]) for r in csv.DictReader(open(HERMITE))
            if r["model_med"] != ""}
    herm_med = np.array([herm.get(int(i), np.nan) for i in ids])
    gcd_med = np.array([float(np.median(gcd_errors(g))) if int(g.mask.sum()) else np.nan
                        for g in graphs])
    mmsi = np.array([int(g.mmsi) for g in graphs])

    m = build_model(graphs[:4], a.ckpt)
    # ---- dial configuration (documented eval-time switches) ----
    if a.gamma == "none":
        m.opm_gamma_env = None                     # gamma = trained opm_gamma_raw (= 0 exactly)
    elif a.gamma == "hat":
        m.opm_gamma_env = "hat"
    elif a.gamma == "hatp":
        assert a.vbar is not None and a.mbar is not None, "hatp needs --vbar --mbar"
        m.opm_gamma_env = gamma_prime(a.vbar, a.mbar, a.beta2)
        print(f"[{a.tag}] gamma' = clamp(1-({a.vbar:.4g}+{a.beta2:.4g})/{a.mbar:.4g}) "
              f"= {m.opm_gamma_env:.4f}", flush=True)
    else:
        m.opm_gamma_env = float(a.gamma)
    m.opm_lam_mode = a.lam
    if a.force_c1_init:
        with torch.no_grad():
            m.opm_kappa_raw.fill_(math.log(math.expm1(0.05)))
        print(f"[{a.tag}] kappa_raw forced to softplus^-1(0.05) = "
              f"{float(m.opm_kappa_raw):.4f} (C1 init)", flush=True)

    med, preds = eval_pergap(m, graphs)
    fin = np.isfinite(med) & np.isfinite(gcd_med)
    ratio = float(np.nanmedian(med[fin]) / np.nanmedian(gcd_med[fin]))
    hv = fin & np.isfinite(herm_med)
    win = float(np.mean(med[hv] < herm_med[hv])) if hv.sum() else float("nan")
    herm_ratio = float(np.nanmedian(herm_med[hv]) / np.nanmedian(gcd_med[hv])) if hv.sum() else float("nan")
    print(f"[{a.tag}] scope={a.scope} n={int(fin.sum())} gamma={a.gamma} lam={a.lam} | "
          f"ratio {ratio:.4f} | win vs Hermite {win:.3f} (Hermite ratio {herm_ratio:.4f})",
          flush=True)

    summ = {"tag": a.tag, "scope": a.scope, "gamma": a.gamma, "lam": a.lam, "ckpt": a.ckpt,
            "vbar": a.vbar, "mbar": a.mbar, "beta2": a.beta2,
            "force_c1_init": bool(a.force_c1_init),
            "n": int(fin.sum()), "ratio": ratio, "win_vs_hermite": win,
            "hermite_ratio": herm_ratio}
    if a.scope == "full" and hv.sum() > 10 and a.boot > 0:
        ci = clustered_ci(med, herm_med, gcd_med, mmsi, B=a.boot)
        summ["clustered_ratio_diff_CI95_vs_hermite"] = ci
        print(f"[{a.tag}] clustered ratio-diff CI95 vs Hermite: [{ci[0]:+.4f}, {ci[1]:+.4f}] "
              f"{'EXCLUDES zero' if ci[1] < 0 else 'crosses zero'}", flush=True)

    d = f"{OUT}/{a.tag}"
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/per_gap.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["gap_id", "mmsi", "gcd_med", "hermite_med", "model_med"])
        for j in range(len(graphs)):
            w.writerow([ids[j], mmsi[j], round(float(gcd_med[j]), 4),
                        round(float(herm_med[j]), 4) if np.isfinite(herm_med[j]) else "",
                        round(float(med[j]), 4) if np.isfinite(med[j]) else ""])
    torch.save(preds, f"{d}/preds.pt")
    json.dump(summ, open(f"{d}/summary.json", "w"), indent=1)

    if a.diff:
        ref = torch.load(a.diff)
        dmax = float((preds - ref).abs().max()) if preds.shape == ref.shape else float("nan")
        summ["diff_vs"] = a.diff; summ["max_pred_diff"] = dmax
        json.dump(summ, open(f"{d}/summary.json", "w"), indent=1)
        print(f"[{a.tag}] IDENTITY DIFF vs {a.diff}: max|pred diff| = {dmax:.8f} "
              f"{'PASS' if dmax == 0.0 else 'CHECK'}", flush=True)

    logp = f"{OUT}/amd_log.csv"
    new = not os.path.exists(logp)
    with open(logp, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["tag", "scope", "gamma", "lam", "ckpt", "ratio", "win_vs_hermite",
                        "ci_lo", "ci_hi", "max_pred_diff"])
        ci = summ.get("clustered_ratio_diff_CI95_vs_hermite", ["", ""])
        w.writerow([a.tag, a.scope, a.gamma, a.lam, os.path.basename(os.path.dirname(a.ckpt)),
                    round(ratio, 4), round(win, 4) if np.isfinite(win) else "",
                    ci[0], ci[1], summ.get("max_pred_diff", "")])


if __name__ == "__main__":
    main()

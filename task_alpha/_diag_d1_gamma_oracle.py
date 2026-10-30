"""GAMMA DIAGNOSTIC -- calibration of the Op-M gamma-hat law vs the per-gap ORACLE gamma, on the
frozen Op-M seed-0 checkpoint (runs/fqnative_opm_5seed/seed0/best.pt),
batched path, metric channel ISOLATED (FQ_OPM_LAM=off).

Measures, per gap:
  * err(gamma) for the fixed grid gamma in {0.00, 0.05, ..., 1.00}  (21 zero-shot evals)
  * the ORACLE gamma  = argmin_gamma err(gamma)  and its error (the gamma-only ceiling)
  * the Ledoit-Wolf gamma-hat law's output + its internal v_hat / m_hat, K_hi count,
    realized aperture, clamp saturation, smoothing residual, raw channel disagreement, SNR proxy
    (captured by monkeypatching fq_opm._opm_metric_batched during a bs=1 "hat" eval; the captured
    gamma-hat is parity-checked against an in-script recompute of the estimator pipeline).

Scopes: --scope sub  = the fixed 200-gap conformance subsample (subsample_ids.json)
        --scope full = all 774 test gaps of the 6h cell (test.pkl; ratios also on the 770
                       finite-sinuosity paper subset), + vessel-clustered bootstrap of the
                       ratio-diff vs cubic Hermite (Hermite per-gap rows matched by gap_id).

Environment (FQ_AMP_REFRESH is removed so the exact trained amplitude table is served):
  FQ_AMP_MODE=akkv_torus FQ_AMP_NMC=262144 FQ_READOUT_MODE=opm FQ_OPM_BATCHED=1 FQ_OPM_LAM=off

Run:  python -m task_alpha._diag_d1_gamma_oracle --scope sub
Out:  task_alpha/runs/diag_d1/per_gap_{scope}.csv + summary_{scope}.json
Diagnostic only: no training, and none of its outputs is a reported number.
"""
from __future__ import annotations

import os

os.environ.setdefault("FQ_AMP_MODE", "akkv_torus")
os.environ.setdefault("FQ_AMP_NMC", "262144")
os.environ["FQ_READOUT_MODE"] = "opm"
os.environ["FQ_OPM_BATCHED"] = "1"
os.environ["FQ_OPM_LAM"] = "off"          # isolate the metric channel (Lambda = 0 exactly)
os.environ.pop("FQ_AMP_REFRESH", None)    # exact amplitude table at the loaded weights (no refresh)

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
from .meltin_proof import _bvec
from .metrics import gcd_errors, model_errors
from . import fq_opm
from .fq_opm import _detrend_batched

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "task_alpha/runs/diag_d1"
CK = "task_alpha/runs/fqnative_opm_5seed/seed0/best.pt"
HERMITE = "task_alpha/runs/gap_sweep/our_processed/gap_6h/hermite/seed0/per_gap.csv"
GRID = [round(0.05 * i, 2) for i in range(21)]
PI = math.pi


def _gm(b):
    return torch.tensor([float(g.gap_minutes) for g in b], device=DEV)


def build_model(warm):
    m = FQDeepNative.production(seed=0).to(DEV)
    assert m.readout_mode == "opm" and m.opm_lam_mode == "off"
    m.use_cuda = True
    with torch.no_grad():
        m.s_raw.fill_(math.log(math.expm1(2.5)))
    b0 = collate_graphs_interp(warm)
    m(b0[0].to(DEV), b0[1].to(DEV), b0[4].to(DEV), b0[6].to(DEV), _bvec(warm),
      gap_minutes=_gm(warm))                       # materialize lazy shapes BEFORE load
    m.load_state_dict(torch.load(CK, map_location=DEV)["state_dict"])
    m.eval()
    return m


def eval_pergap(m, graphs, bs=64):
    """Per-gap median NM error, index-aligned with `graphs` (nan where no held rows)."""
    med = np.full(len(graphs), np.nan)
    with torch.no_grad():
        for s in range(0, len(graphs), bs):
            b = graphs[s:s + bs]
            x, coord, yp, mask, ei, ed, aei = collate_graphs_interp(b)
            pred = m(x.to(DEV), coord.to(DEV), ei.to(DEV), aei.to(DEV), _bvec(b),
                     gap_minutes=_gm(b)).float().cpu()
            off = 0
            for j, g in enumerate(b):
                ng = g.x.shape[0]; gp = pred[off:off + ng]; off += ng
                fe = model_errors(gp, g)
                if len(fe):
                    med[s + j] = float(np.median(fe))
    return med


# ------------------------------------------------------------------ instrumented estimator copy
def diag_recompute(ts, ys, smask, gapi, Ma, om, keep, log_wp, s_all, P_jac,
                   kappa, nu, c_w, eps_rel, eps_abs, khi_min_cycles):
    """Verbatim replica of fq_opm._opm_metric_batched's estimator pipeline that RETURNS the
    per-gap internals of the gamma-hat law (gamma_hat, v_hat, m_hat, ...).  Parity of its
    gamma_hat against the production function's return is asserted by the caller."""
    dt, dev = ys.dtype, ys.device
    G, Mx = ts.shape
    M = om.shape[0]
    rho_t, rho, tau, varpi, T, _ = _detrend_batched(ts, ys, smask, gapi, kappa)
    om_abs = om.abs()
    o11 = om.view(1, M, 1)
    ang = o11 * tau.unsqueeze(1)
    vm = varpi * smask.to(dt)
    ws = vm.sum(1).clamp_min(1e-12).view(G, 1, 1)
    cosang, sinang = torch.cos(ang), torch.sin(ang)
    Re = (cosang * vm.unsqueeze(1)) @ rho_t / ws
    Im = (-sinang * vm.unsqueeze(1)) @ rho_t / ws
    I_val = (Re * Re + Im * Im).sum(-1)

    gi = gapi.unsqueeze(1)
    ar = torch.arange(Mx, device=dev).unsqueeze(0)
    I_inc = torch.zeros_like(I_val)
    for mm in ((ar <= gi) & smask, (ar > gi) & smask):
        vsd = varpi * mm.to(dt)
        wsd = vsd.sum(1).clamp_min(1e-12).view(G, 1, 1)
        ReS = (cosang * vsd.unsqueeze(1)) @ rho_t / wsd
        ImS = (-sinang * vsd.unsqueeze(1)) @ rho_t / wsd
        I_inc = I_inc + (ReS * ReS + ImS * ImS).sum(-1)
    I_inc = 0.5 * I_inc

    jj = torch.arange(Mx - 1, device=dev).unsqueeze(0)
    pv = ((jj < (Ma - 1).unsqueeze(1)) & (jj != gi)).to(dt)
    dt_p = (tau[:, 1:] - tau[:, :-1]).clamp_min(1e-9)
    dlt = (rho_t[:, 1:] - rho_t[:, :-1]) / dt_p.unsqueeze(-1) * pv.unsqueeze(-1)
    tmid = 0.5 * (tau[:, 1:] + tau[:, :-1])
    wmid = torch.exp(-tmid.abs() / (kappa * T).unsqueeze(1)) * pv
    gk = 2.0 * torch.sin(o11 * dt_p.unsqueeze(1) / 2.0) / dt_p.unsqueeze(1)
    sgn = torch.where(gk >= 0, torch.ones_like(gk), -torch.ones_like(gk))
    gk = sgn * torch.maximum(gk.abs(), o11.abs() / PI).clamp_min(1e-30)
    contrib = dlt.unsqueeze(1) / gk.unsqueeze(-1)
    angm = o11 * tmid.unsqueeze(1)
    wsm = wmid.sum(1).clamp_min(1e-12).view(G, 1, 1)
    ReI = ((torch.cos(angm) * wmid.unsqueeze(1)).unsqueeze(-1) * contrib).sum(2) / wsm
    ImI = ((-torch.sin(angm) * wmid.unsqueeze(1)).unsqueeze(-1) * contrib).sum(2) / wsm
    I_der = (ReI * ReI + ImI * ImI).sum(-1)

    I_tot = (I_val + nu * I_der) / (1.0 + nu)
    I_rob = torch.maximum(I_tot, I_inc)
    nanv = torch.full_like(I_rob, float("nan"))
    omx = om_abs.unsqueeze(0).expand(G, M)
    om_med = torch.where(keep, omx, nanv).nanmedian(1).values
    hi = keep & (omx > om_med.unsqueeze(1))
    hi = torch.where(hi.any(1, keepdim=True), hi, keep)
    flr = torch.where(hi, I_rob, nanv).nanmedian(1).values / math.log(2.0)
    med_all = torch.where(keep, I_rob, nanv).nanmedian(1).values
    eps_t = torch.tensor(eps_abs, device=dev, dtype=dt)
    fl2 = torch.maximum(eps_rel * med_all, eps_t)
    I_raw = torch.maximum(I_rob - flr.unsqueeze(1), fl2.unsqueeze(1))
    logI_raw = torch.log(I_raw)

    t_last = torch.take_along_dim(ts, (Ma - 1).unsqueeze(1), 1).squeeze(1)
    span = t_last - ts[:, 0]
    ap = torch.minimum(span, 2.0 * kappa * T).clamp_min(1e-9)
    khi = keep & (omx * ap.unsqueeze(1) >= 2.0 * PI * khi_min_cycles)

    if P_jac is None:
        if s_all is None:
            om_kmax = torch.where(keep, omx, torch.zeros_like(omx)).max(1).values.clamp_min(1e-12)
            s_b = omx / om_kmax.unsqueeze(1)
        else:
            s_b = s_all.detach().to(dt).unsqueeze(0).expand(G, M)
        Pk = torch.stack([s_b ** r for r in range(5)], dim=-1)
    else:
        Pk = P_jac.detach().to(dt).unsqueeze(0).expand(G, -1, -1)
    J = Pk.shape[-1]
    Iv = torch.maximum(I_val - flr.unsqueeze(1), fl2.unsqueeze(1))
    Id = torch.maximum(I_der - flr.unsqueeze(1), fl2.unsqueeze(1))
    lIv, lId = torch.log(Iv), torch.log(Id)
    khi_cpu = khi.cpu()
    theta = torch.zeros(G, J, device=dev, dtype=dt)
    th_v = torch.zeros(G, J, device=dev, dtype=dt)
    th_d = torch.zeros(G, J, device=dev, dtype=dt)
    for gdx in range(G):
        rows = torch.nonzero(khi_cpu[gdx]).squeeze(1)
        if rows.numel() == 0:
            continue
        rows = rows.to(dev)
        Pp_g = torch.linalg.pinv(Pk[gdx][rows])
        theta[gdx] = Pp_g @ logI_raw[gdx, rows]
        th_v[gdx] = Pp_g @ lIv[gdx, rows]
        th_d[gdx] = Pp_g @ lId[gdx, rows]
    logI_sm = torch.matmul(Pk, theta.unsqueeze(-1)).squeeze(-1)
    hv = torch.matmul(Pk, th_v.unsqueeze(-1)).squeeze(-1)
    hd = torch.matmul(Pk, th_d.unsqueeze(-1)).squeeze(-1)

    khf = khi.to(dt)
    zr = torch.zeros_like(logI_raw)
    nhi = khf.sum(1)
    vhat = torch.where(khi, (0.25 * (hv - hd)) ** 2, zr).sum(1) / nhi.clamp_min(1.0)
    mhat = torch.where(khi, (0.5 * logI_sm - log_wp) ** 2, zr).sum(1) / nhi.clamp_min(1.0)
    gam = torch.where(nhi > 0, torch.clamp(1.0 - vhat / mhat.clamp_min(1e-12), 0.0, 1.0),
                      torch.zeros_like(vhat))

    dev_full = 0.5 * logI_sm - log_wp                                     # unclamped deviation
    clampfrac = torch.where(khi, (dev_full.abs() > c_w).to(dt), zr).sum(1) / nhi.clamp_min(1.0)
    resid_sm = (torch.where(khi, (logI_raw - logI_sm) ** 2, zr).sum(1)
                / nhi.clamp_min(1.0)).sqrt()                              # Jacobi-fit residual rms
    disag_raw = (torch.where(khi, (0.5 * (lIv - lId)) ** 2, zr).sum(1)
                 / nhi.clamp_min(1.0)).sqrt()                             # raw channel disagreement
    snr = torch.where(khi, I_rob, zr).sum(1) / nhi.clamp_min(1.0) / flr.clamp_min(1e-30)
    return {"gamma_hat_re": gam, "vhat": vhat, "mhat": mhat, "nhi": nhi,
            "ap_over_T": ap / T, "clampfrac": clampfrac, "resid_sm": resid_sm,
            "disag_raw": disag_raw, "snr": snr}


CAP = []                                              # per-forward capture (hat run, bs=1)


def install_capture():
    orig = fq_opm._opm_metric_batched

    def patched(ts, ys, smask, gapi, Ma, om, keep, log_wp, gamma, s_all, P_jac,
                kappa, nu, c_w, eps_rel, eps_abs, khi_min_cycles):
        corr, gam = orig(ts, ys, smask, gapi, Ma, om, keep, log_wp, gamma, s_all, P_jac,
                         kappa, nu, c_w, eps_rel, eps_abs, khi_min_cycles)
        if isinstance(gam, torch.Tensor) and gam.dim() == 1:              # hat mode only
            d = diag_recompute(ts, ys, smask, gapi, Ma, om, keep, log_wp, s_all, P_jac,
                               kappa, nu, c_w, eps_rel, eps_abs, khi_min_cycles)
            d = {k: v.detach().cpu().numpy() for k, v in d.items()}
            d["gamma_hat"] = gam.detach().cpu().numpy().copy()
            CAP.append(d)
        return corr, gam

    fq_opm._opm_metric_batched = patched
    return orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=("sub", "full"), default="sub")
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    with open("task_alpha/data/our_processed/gap_6h/test.pkl", "rb") as f:
        te_all = pickle.load(f)
    if a.scope == "sub":
        ids = json.load(open("task_alpha/runs/conform_baseline/subsample_ids.json"))
    else:
        ids = list(range(len(te_all)))
    graphs = [te_all[i] for i in ids]
    herm = {int(r["gap_id"]): float(r["model_med"]) for r in csv.DictReader(open(HERMITE))
            if r["model_med"] != ""}
    gcd_med = np.array([float(np.median(gcd_errors(g))) if int(g.mask.sum()) else np.nan
                        for g in graphs])
    sinu = np.array([float(_sinuosity(g)) for g in graphs])
    mmsi = np.array([int(g.mmsi) for g in graphs])
    print(f"[D1:{a.scope}] {len(graphs)} gaps | ckpt {CK} | amp={os.environ['FQ_AMP_MODE']} "
          f"nmc={os.environ['FQ_AMP_NMC']} lam={os.environ['FQ_OPM_LAM']}", flush=True)

    m = build_model(graphs[:4])

    # ---- fixed-gamma grid (bs=64 batched) ----
    errs = np.full((len(graphs), len(GRID)), np.nan)
    for i, g0 in enumerate(GRID):
        m.opm_gamma_env = float(g0)
        errs[:, i] = eval_pergap(m, graphs)
        r = float(np.nanmedian(errs[:, i]) / np.nanmedian(gcd_med))
        print(f"[D1:{a.scope}] gamma={g0:4.2f}  ratio {r:.4f}", flush=True)

    # ---- gamma-hat law (bs=1 so capture aligns 1:1 with gaps) ----
    CAP.clear()
    install_capture()
    m.opm_gamma_env = "hat"
    err_hat = np.full(len(graphs), np.nan)
    diag = {k: np.full(len(graphs), np.nan) for k in
            ("gamma_hat", "gamma_hat_re", "vhat", "mhat", "nhi", "ap_over_T",
             "clampfrac", "resid_sm", "disag_raw", "snr")}
    with torch.no_grad():
        for j, g in enumerate(graphs):
            n0 = len(CAP)
            e = eval_pergap(m, [g], bs=1)
            err_hat[j] = e[0]
            if len(CAP) == n0 + 1 and CAP[-1]["gamma_hat"].shape[0] == 1:
                for k in diag:
                    diag[k][j] = float(CAP[-1][k][0])
    parity = np.nanmax(np.abs(diag["gamma_hat"] - diag["gamma_hat_re"]))
    r_hat = float(np.nanmedian(err_hat) / np.nanmedian(gcd_med))
    print(f"[D1:{a.scope}] gamma=HAT   ratio {r_hat:.4f} | recompute parity "
          f"max|dgamma| = {parity:.2e}", flush=True)

    # ---- oracle ----
    valid = np.isfinite(errs).all(1) & np.isfinite(gcd_med)
    orc_i = np.full(len(graphs), -1)
    orc_i[valid] = np.argmin(errs[valid], axis=1)
    gam_orc = np.where(valid, np.array(GRID)[np.clip(orc_i, 0, None)], np.nan)
    err_orc = np.where(valid, np.nanmin(errs, axis=1), np.nan)
    r_orc = float(np.nanmedian(err_orc) / np.nanmedian(gcd_med))
    herm_med = np.array([herm.get(int(i), np.nan) for i in ids])

    # ---- per-gap csv ----
    cols = (["gap_id", "mmsi", "gap_min", "sinuosity", "gcd_med", "hermite_med"]
            + [f"err_g{int(round(100 * g0)):03d}" for g0 in GRID]
            + ["err_hat", "gamma_oracle", "err_oracle"] + list(diag.keys()))
    with open(f"{OUT}/per_gap_{a.scope}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for j in range(len(graphs)):
            row = [ids[j], mmsi[j], round(float(graphs[j].gap_minutes), 1),
                   round(float(sinu[j]), 4) if np.isfinite(sinu[j]) else "",
                   round(float(gcd_med[j]), 4), round(float(herm_med[j]), 4)
                   if np.isfinite(herm_med[j]) else ""]
            row += [round(float(errs[j, i]), 4) if np.isfinite(errs[j, i]) else ""
                    for i in range(len(GRID))]
            row += [round(float(err_hat[j]), 4) if np.isfinite(err_hat[j]) else "",
                    gam_orc[j] if np.isfinite(gam_orc[j]) else "",
                    round(float(err_orc[j]), 4) if np.isfinite(err_orc[j]) else ""]
            row += [round(float(diag[k][j]), 6) if np.isfinite(diag[k][j]) else ""
                    for k in diag]
            w.writerow(row)

    # ---- summary ----
    fin = valid & np.isfinite(err_hat)
    summ = {"tag": "D1_gamma_oracle", "scope": a.scope, "n_gaps": int(valid.sum()),
            "ckpt": CK, "parity_max_dgamma": float(parity),
            "ratio_grid": {f"{g0:.2f}": float(np.nanmedian(errs[valid, i])
                                              / np.nanmedian(gcd_med[valid]))
                           for i, g0 in enumerate(GRID)},
            "ratio_hat": float(np.nanmedian(err_hat[fin]) / np.nanmedian(gcd_med[fin])),
            "ratio_oracle": r_orc,
            "gamma_hat_mean": float(np.nanmean(diag["gamma_hat"][fin])),
            "gamma_oracle_mean": float(np.nanmean(gam_orc[fin])),
            "corr_hat_oracle": float(np.corrcoef(diag["gamma_hat"][fin],
                                                 gam_orc[fin])[0, 1]) if fin.sum() > 2 else None}
    best_i = int(np.argmin([summ["ratio_grid"][f"{g0:.2f}"] for g0 in GRID]))
    summ["best_fixed_gamma"] = GRID[best_i]
    summ["ratio_best_fixed"] = summ["ratio_grid"][f"{GRID[best_i]:.2f}"]
    hv = fin & np.isfinite(herm_med)
    if hv.sum() > 10:
        summ["vs_hermite"] = {
            "n": int(hv.sum()),
            "win_g0": float(np.mean(errs[hv, 0] < herm_med[hv])),
            "win_best_fixed": float(np.mean(errs[hv, best_i] < herm_med[hv])),
            "win_hat": float(np.mean(err_hat[hv] < herm_med[hv])),
            "win_oracle": float(np.mean(err_orc[hv] < herm_med[hv])),
            "hermite_ratio": float(np.nanmedian(herm_med[hv]) / np.nanmedian(gcd_med[hv]))}
    if a.scope == "full" and hv.sum() > 10:
        rng = np.random.default_rng(0)
        vecs = np.unique(mmsi[hv])
        groups = [np.flatnonzero(mmsi[hv] == v) for v in vecs]
        eo, eb, eh, gg = err_orc[hv], errs[hv, best_i], herm_med[hv], gcd_med[hv]
        d_or, d_bf = [], []
        for _ in range(a.boot):
            rows = np.concatenate([groups[i] for i in
                                   rng.integers(0, len(groups), len(groups))])
            gmed = np.median(gg[rows])
            d_or.append(np.median(eo[rows]) / gmed - np.median(eh[rows]) / gmed)
            d_bf.append(np.median(eb[rows]) / gmed - np.median(eh[rows]) / gmed)
        summ["clustered_ratio_diff_CI95"] = {
            "oracle_minus_hermite": [float(np.percentile(d_or, 2.5)),
                                     float(np.percentile(d_or, 97.5))],
            "best_fixed_minus_hermite": [float(np.percentile(d_bf, 2.5)),
                                         float(np.percentile(d_bf, 97.5))]}
    json.dump(summ, open(f"{OUT}/summary_{a.scope}.json", "w"), indent=1)
    print(json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    main()

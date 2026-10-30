"""EXTERIOR CROSS-VALIDATION CONTRAST as a candidate per-gap gamma law for the Op-M readout.

Mathematical object being tested: the Ledoit-Wolf ratio v_hat/m_hat of the gamma-hat law is a
MEASUREMENT-NOISE statistic, but the per-gap oracle gamma is governed by a TRANSFER term (does
the exterior-measured spectrum describe the gap interior?) that is common-mode to both channels
and hence invisible to v_hat.  The only leak-safe samples of the field between the exterior and
the gap are the N_o collar anchors themselves -- and at Lambda = 0 the production solve NEVER
uses their values, so they are genuinely out-of-sample.  Candidate statistic, per gap:

    e_j(gamma) = || p_hat_gamma(tau_j) - y_j ||          (collar anchor j, Lambda = 0 solve)
    D_CV       = sum_j varpi_j [ e_j(0)^2 - e_j(1)^2 ] / sum_j varpi_j
    predict "full trust helps" iff D_CV > 0.

This script captures every opm_readout_batched call of a gamma=0 eval over the 200-gap
subsample, replays each captured batch with qt := exterior-anchor times at gamma = 0 and
gamma = 1 (identical inputs otherwise -- the production function itself, no reimplementation),
and correlates sign(D_CV) with the measured per-gap oracle direction from per_gap_sub.csv
(help = err_g100 < err_g000, on the gamma-active gaps rel_spread >= 0.05).

Run:  python -m task_alpha._diag_d1_cv_contrast
Out:  task_alpha/runs/diag_d1/cv_contrast_sub.csv + printed confusion/correlation table.
Diagnostic only; no training; the production readout is wrapped for capture and then restored.
"""
from __future__ import annotations

import os

os.environ.setdefault("FQ_AMP_MODE", "akkv_torus")
os.environ.setdefault("FQ_AMP_NMC", "262144")
os.environ["FQ_READOUT_MODE"] = "opm"
os.environ["FQ_OPM_BATCHED"] = "1"
os.environ["FQ_OPM_LAM"] = "off"
os.environ.pop("FQ_AMP_REFRESH", None)

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
from .meltin_proof import _bvec
from . import fq_opm
from ._diag_d1_gamma_oracle import build_model, _gm, DEV, OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=("sub", "full"), default="sub")
    a = ap.parse_args()
    with open("task_alpha/data/our_processed/gap_6h/test.pkl", "rb") as f:
        te_all = pickle.load(f)
    if a.scope == "sub":
        ids = json.load(open("task_alpha/runs/conform_baseline/subsample_ids.json"))
    else:
        ids = list(range(len(te_all)))
    graphs = [te_all[i] for i in ids]
    m = build_model(graphs[:4])
    m.opm_gamma_env = 0.0

    caps = []                                       # (args, kwargs) per readout call
    orig = fq_opm.opm_readout_batched

    def capture(*args, **kw):
        caps.append((args, kw))
        return orig(*args, **kw)

    fq_opm.opm_readout_batched = capture
    import task_alpha.fqdeep_native as fdn
    fdn.opm_readout_batched = capture               # fqdeep_native imported the name directly

    with torch.no_grad():
        for j, g in enumerate(graphs):              # bs=1: capture aligns 1:1 with gaps
            b = [g]
            x, coord, yp, mask, ei, ed, aei = collate_graphs_interp(b)
            m(x.to(DEV), coord.to(DEV), ei.to(DEV), aei.to(DEV), _bvec(b), gap_minutes=_gm(b))
    fq_opm.opm_readout_batched = orig
    fdn.opm_readout_batched = orig
    assert len(caps) == len(graphs), (len(caps), len(graphs))

    rows_out = []
    with torch.no_grad():
        for j, (args, kw) in enumerate(caps):
            (ts, ys, vs, smask, gapi, Ma, qt, qgap, om_all, eta_all) = args
            G, Mx = ts.shape
            assert G == 1
            gi = int(gapi[0]); Mr = int(Ma[0])
            ext_idx = [k for k in range(Mr) if k not in (gi, gi + 1)]
            if not ext_idx:
                rows_out.append({"gap_id": ids[j]})
                continue
            tau_e = ts[0, ext_idx]
            y_e = ys[0, ext_idx]                    # [No,2] actual positions
            qg = torch.zeros(len(ext_idx), dtype=torch.long, device=ts.device)
            e2 = {}
            for gv in (0.0, 1.0):
                kw2 = dict(kw)
                kw2["gamma"] = torch.tensor(gv, device=ts.device, dtype=ys.dtype)
                pred = fq_opm.opm_readout_batched(ts, ys, vs, smask, gapi, Ma,
                                                  tau_e, qg, om_all, eta_all, **kw2)
                e2[gv] = ((pred - y_e) ** 2).sum(-1)          # [No] squared NM-ish units
            a_t, b_t = ts[0, gi], ts[0, gi + 1]
            T = (b_t - a_t).clamp_min(1e-9)
            t0 = 0.5 * (a_t + b_t)
            dist = (tau_e - t0).abs()
            w_tap = torch.exp(-dist / T)                       # collar-emphasis weight
            d_all = float((e2[0.0] - e2[1.0]).mean())
            d_tap = float(((e2[0.0] - e2[1.0]) * w_tap).sum() / w_tap.sum())
            near = torch.argsort(dist)[:2]                     # the 2 nearest collar anchors
            d_near = float((e2[0.0] - e2[1.0])[near].mean())
            rows_out.append({"gap_id": ids[j], "no_ext": len(ext_idx),
                             "dcv_all": d_all, "dcv_tap": d_tap, "dcv_near": d_near,
                             "e0_tap": float((e2[0.0] * w_tap).sum() / w_tap.sum()),
                             "e1_tap": float((e2[1.0] * w_tap).sum() / w_tap.sum())})

    with open(f"{OUT}/cv_contrast_{a.scope}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)

    # ---- evaluate against the measured oracle direction ----
    import pandas as pd
    d = pd.read_csv(f"{OUT}/per_gap_{a.scope}.csv").merge(pd.DataFrame(rows_out), on="gap_id")
    gcols = [f"err_g{i:03d}" for i in range(0, 101, 5)]
    E = d[gcols].values
    d["rel_spread"] = (E.max(1) - E.min(1)) / d.gcd_med
    d["help"] = (d.err_g100 < d.err_g000).astype(float)
    act = d[(d.rel_spread >= 0.05) & np.isfinite(d.dcv_tap)].copy()
    print(f"[D1-CV] n={len(d)} merged | active={len(act)}")
    for c in ("dcv_all", "dcv_tap", "dcv_near"):
        s = np.sign(act[c])
        pb = np.corrcoef(act[c].rank(), act.help)[0, 1]
        acc = float((s.clip(0, 1) == act.help).mean())
        p_h = float(s[act.help == 1].clip(0, 1).mean())
        p_n = float(s[act.help == 0].clip(0, 1).mean())
        print(f"[D1-CV] {c}: rank-corr with help {pb:+.3f} | sign-accuracy {acc:.3f} | "
              f"P(D>0|helps) {p_h:.3f} vs P(D>0|hurts) {p_n:.3f}")
    # oracle-side check: what ratio would the CV DECISION rule reach (gamma = 1[D>0])?
    for c in ("dcv_all", "dcv_tap", "dcv_near"):
        pick = np.where(d[c].fillna(0) > 0, d.err_g100, d.err_g000)
        print(f"[D1-CV] decision-rule ratio with {c}: "
              f"{np.median(pick) / np.median(d.gcd_med):.4f} "
              f"(g0 {np.median(d.err_g000)/np.median(d.gcd_med):.4f}, "
              f"g1 {np.median(d.err_g100)/np.median(d.gcd_med):.4f}, "
              f"binary-oracle {np.median(np.minimum(d.err_g000, d.err_g100))/np.median(d.gcd_med):.4f})")
    # gap-level win vs Hermite under the decision rule
    hv = np.isfinite(d.hermite_med)
    for c in ("dcv_tap",):
        pick = np.where(d[c].fillna(0) > 0, d.err_g100, d.err_g000)
        print(f"[D1-CV] win vs Hermite (decision {c}): "
              f"{float((pick[hv] < d.hermite_med[hv]).mean()):.3f} "
              f"(binary-oracle {float((np.minimum(d.err_g000, d.err_g100)[hv] < d.hermite_med[hv]).mean()):.3f})")


if __name__ == "__main__":
    main()

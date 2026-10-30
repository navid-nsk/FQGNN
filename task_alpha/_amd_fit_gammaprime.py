"""FIT OF THE gamma' CELL CONSTANTS of the Op-M gamma law (FQ_OPM_GAMMA=hatp), on TRAINING gaps
only.

Measures, on the frozen Op-M seed-0 checkpoint over a 1200-gap TRAINING subsample (the
fasttrain sampling convention -- no test gap is touched):
  * vbar, mbar   = cell medians of the gamma-hat law's internals v_hat / m_hat (captured through
                   the instrumented estimator of _diag_d1_gamma_oracle, parity-checked there
                   against the production estimator)
  * gamma*_agg   = the aggregate fixed-gamma grid argmin on the SAME training gaps
                   (training-gap truth may be used -- it is training data)
  * beta2_hat    = mbar * (1 - gamma*_agg) - vbar   (>= 0 clamp)  -- from the population
                   identity gamma_bar* ~= Sbar/(Sbar + beta2 + vbar)  <=>
                   gamma' = 1 - (vbar + beta2)/mbar  at the aggregate argmin.

Prints the environment settings for FQ_OPM_GAMMA=hatp and saves runs/amd/gammaprime_fit.json.
Measurement only.

Run:  python -m task_alpha._amd_fit_gammaprime
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import torch

# env is set at import by _diag_d1_gamma_oracle (opm / batched / lam off / clean amp; REFRESH popped)
from ._diag_d1_gamma_oracle import (CAP, build_model, eval_pergap, install_capture, _gm)  # noqa
from .data_raw import build_dataset_raw
from .metrics import gcd_errors
from .train_stream import _split

OUT = "task_alpha/runs/amd"
GRID = [0.0, 0.3, 0.6, 0.8, 0.85, 0.9, 1.0]
N_TRAIN = 1200


def main():
    os.makedirs(OUT, exist_ok=True)
    allg = build_dataset_raw(years=(2016, 2017, 2018), months=(7, 8), gap_minutes=360,
                             max_gap_len=500, min_span_factor=1.5, n_anchor=8, clean_only=True,
                             n_mmsi=None, max_files_per_part=None, seed=0)
    allg = [g for g in allg if int(g.mask.sum()) > 0 and _split(g.mmsi) == "train"]
    rng = np.random.default_rng(20260718)
    tr = [allg[i] for i in sorted(rng.choice(len(allg), size=min(N_TRAIN, len(allg)),
                                             replace=False).tolist())]
    gcd_med = np.array([float(np.median(gcd_errors(g))) for g in tr])
    print(f"[AMD_gammaprime_fit] {len(tr)} TRAINING gaps (no test gap touched)", flush=True)

    m = build_model(tr[:4])

    # ---- aggregate grid argmin on training gaps ----
    ratios = {}
    errs = {}
    for g0 in GRID:
        m.opm_gamma_env = float(g0)
        e = eval_pergap(m, tr)
        errs[g0] = e
        ratios[g0] = float(np.nanmedian(e) / np.nanmedian(gcd_med))
        print(f"[AMD_gammaprime_fit] gamma={g0:4.2f}  train-ratio {ratios[g0]:.4f}", flush=True)
    gi = int(np.argmin([ratios[g] for g in GRID]))
    g_star = GRID[gi]
    # quadratic refine on the argmin neighborhood (unequal spacing ok)
    if 0 < gi < len(GRID) - 1:
        xs = np.array(GRID[gi - 1:gi + 2]); ys = np.array([ratios[g] for g in xs])
        c2, c1, _ = np.polyfit(xs, ys, 2)
        if c2 > 0:
            g_ref = float(np.clip(-c1 / (2 * c2), 0.0, 1.0))
        else:
            g_ref = float(g_star)
    else:
        g_ref = float(g_star)

    # ---- capture v_hat / m_hat over the training gaps (hat mode, instrumented) ----
    CAP.clear()
    install_capture()
    m.opm_gamma_env = "hat"
    _ = eval_pergap(m, tr)
    vh = np.concatenate([c["vhat"] for c in CAP])
    mh = np.concatenate([c["mhat"] for c in CAP])
    nh = np.concatenate([c["nhi"] for c in CAP])
    ok = nh > 0
    vbar = float(np.median(vh[ok])); mbar = float(np.median(mh[ok]))
    beta2 = max(mbar * (1.0 - g_ref) - vbar, 0.0)
    gp = min(max(1.0 - (vbar + beta2) / max(mbar, 1e-12), 0.0), 1.0)

    out = {"tag": "AMD_gammaprime_fit", "n_train_gaps": len(tr), "grid_ratios": ratios,
           "gamma_star_grid": g_star, "gamma_star_refined": g_ref,
           "vbar": vbar, "mbar": mbar, "beta2": beta2, "gamma_prime": gp,
           "n_capture": int(ok.sum())}
    json.dump(out, open(f"{OUT}/gammaprime_fit.json", "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)
    print(f"\nENV RECIPE:  FQ_OPM_GAMMA=hatp FQ_OPM_VBAR={vbar:.6g} FQ_OPM_MBAR={mbar:.6g} "
          f"FQ_OPM_BETA2={beta2:.6g}   (-> gamma' = {gp:.4f})", flush=True)


if __name__ == "__main__":
    main()

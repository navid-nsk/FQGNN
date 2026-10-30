"""CLEAN RE-EVAL of the Op-M stack ONE-SIDED + ANCHOR-SWEEP checkpoints.

Same rationale as the other _clean_reeval_* scripts: in-runner TEST evals run with
FQ_AMP_REFRESH=84; the reported numbers use exact torus-integral tables at the loaded weights.
Protocol pieces are IMPORTED from the training runners themselves (one_sided surgery from
run_oneside_native_5seed; dataset loading + node-budget eval from run_anchor_sweep /
run_gap_sweep) -- no protocol is defined here.

The anchor aps4 row = the headline run, re-evaluated by _clean_reeval_opmstack.

Launch (stack env, NO REFRESH):
  FQ_ONESIDE_SOBOLEV=1 FQ_ANCHOR_SOBOLEV=1 FQ_READOUT_MODE=opm <stack env> \
  python -m task_alpha._clean_reeval_oneanchor

Out: oneside_sobolev_5seed_opm/seedN/{per_gap_clean,test_buckets_clean}.csv + summary_clean.csv
     anchor_sweep_sobolev_opm/aps{2,8}/fqgnn/seedN/... + summary_clean.csv
"""
from __future__ import annotations

import csv
import math
import os
import random
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch

from . import run_oneside_native_5seed as osn
from . import run_anchor_sweep as anc
from . import run_gap_sweep as gs
from .baselines.common_gaps import bucket_rows, gap_row, load_gaps, write_buckets, write_per_gap
from .data import collate_graphs_interp
from .fqdeep_native import FQDeepNative
from .meltin_proof import _bvec
from .metrics import model_errors

DEV = "cuda"
SEEDS = (0, 1, 2, 3, 4)


def _write_summary(cdir, summary, label):
    rat = np.array([s["ratio"] for s in summary], float)
    wp = np.array([s["win_pct"] for s in summary], float)
    with open(f"{cdir}/summary_clean.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
        cw = csv.writer(f)
        cw.writerow(["MEAN", "", "", "", round(float(rat.mean()), 4), round(float(wp.mean()), 1)])
        cw.writerow(["STD", "", "", "", round(float(rat.std(ddof=0)), 4),
                     round(float(wp.std(ddof=0)), 1)])
    print(f"=== CLEAN {label}: {rat.mean():.4f} +/- {rat.std(ddof=0):.4f} ===", flush=True)


def _make_model(seed, warm, blend_kw_fn):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    m = FQDeepNative(n=6, n_freqs=48, width=96, init_scale=1.84, seed=seed, in_feat=6, layers=3,
                     jacobi_R=4, oph_weighted=True, dmu_source="blend",
                     metric_backbone="sobolev2").to(DEV)
    m.use_cuda = True
    with torch.no_grad():
        m.s_raw.fill_(math.log(math.expm1(2.5)))
    b0 = collate_graphs_interp(warm)
    m(b0[0].to(DEV), b0[1].to(DEV), b0[4].to(DEV), b0[6].to(DEV), _bvec(warm),
      **blend_kw_fn(m, warm))
    return m


def _test_eval(m, te, blend_kw_fn):
    m.eval(); pg = []
    with torch.no_grad():
        for s0 in range(0, len(te), 64):
            b = te[s0:s0 + 64]
            x, coord, yp, mask, ei, ed, aei = collate_graphs_interp(b)
            pred = m(x.to(DEV), coord.to(DEV), ei.to(DEV), aei.to(DEV), _bvec(b),
                     **blend_kw_fn(m, b)).float().cpu()
            off = 0
            for j, g in enumerate(b):
                ng = g.x.shape[0]; gp = pred[off:off + ng]; off += ng
                fe = model_errors(gp, g)
                if len(fe):
                    pg.append(gap_row(s0 + j, g, fe))
    return pg


def main():
    assert os.environ.get("FQ_READOUT_MODE") == "opm", "launch with the full Op-M stack env"
    assert os.environ.get("FQ_AMP_REFRESH", "0") in ("", "0"), \
        "FQ_AMP_REFRESH must be UNSET for the clean re-eval (exact tables)"
    assert osn.SOBOLEV and osn.OPM, "FQ_ONESIDE_SOBOLEV=1 + opm env required"
    assert anc.ANCHOR_SOBOLEV and anc.OPM and gs.SOBOLEV, "FQ_ANCHOR_SOBOLEV=1 required"
    print(f"[clean-oneanchor] oneside OUT={osn.OUT} | anchor ROOT={anc.SWEEP_ROOT} | "
          f"amp={os.environ.get('FQ_AMP_MODE')}/nmc={os.environ.get('FQ_AMP_NMC')}/refresh=EXACT",
          flush=True)

    # ---- one-sided: the exact surgery on the exact split ----
    tr, va, te = load_gaps()
    tr_os = [osn.one_sided(g) for g in tr[:4]]
    te_os = [osn.one_sided(g) for g in te]
    summary = []
    for seed in SEEDS:
        sdir = f"{osn.OUT}/seed{seed}"
        if not os.path.exists(f"{sdir}/best.pt"):
            print(f"[clean-oneside] MISSING {sdir}/best.pt -- skipped", flush=True)
            continue
        ck = torch.load(f"{sdir}/best.pt", map_location="cpu")["state_dict"]
        m = _make_model(seed, tr_os, osn._bkw)
        m.load_state_dict(ck, strict=True)
        pg = _test_eval(m, te_os, osn._bkw)
        brows = bucket_rows(pg)
        write_per_gap(f"{sdir}/per_gap_clean.csv", pg)
        write_buckets(f"{sdir}/test_buckets_clean.csv", brows)
        ov = brows[0]
        print(f"[clean-oneside seed {seed}] {ov['n']} gaps ratio {ov['ratio']} "
              f"(win {ov['win_pct']}%)", flush=True)
        summary.append({"seed": seed, **{k: ov[k] for k in
                                         ("n", "gcd_med", "model_med", "ratio", "win_pct")}})
        del m
        torch.cuda.empty_cache()
    if summary:
        _write_summary(osn.OUT, summary, "oneside")
    del tr, va, te, te_os

    # ---- anchor aps2 / aps8: the exact aps datasets + node-budget eval ----
    for aps in (2, 8):
        atr, ava, ate = anc.load_dataset(aps, False)
        cdir = f"{anc.SWEEP_ROOT}/aps{aps}/fqgnn"
        summary = []
        for seed in SEEDS:
            sdir = f"{cdir}/seed{seed}"
            if not os.path.exists(f"{sdir}/best.pt"):
                print(f"[clean-anchor aps{aps}] MISSING {sdir}/best.pt -- skipped", flush=True)
                continue
            ck = torch.load(f"{sdir}/best.pt", map_location="cpu")["state_dict"]
            m = _make_model(seed, atr[:4], gs._blend_kw)
            m.load_state_dict(ck, strict=True)
            m.eval()
            _, _, te_errs = gs.eval_fq_budget(m, ate)
            pg = [gap_row(gi, g, fe) for gi, (g, fe) in enumerate(zip(ate, te_errs)) if len(fe)]
            brows = bucket_rows(pg)
            write_per_gap(f"{sdir}/per_gap_clean.csv", pg)
            write_buckets(f"{sdir}/test_buckets_clean.csv", brows)
            ov = brows[0]
            print(f"[clean-anchor aps{aps} seed {seed}] {ov['n']} gaps ratio {ov['ratio']} "
                  f"(win {ov['win_pct']}%)", flush=True)
            summary.append({"seed": seed, **{k: ov[k] for k in
                                             ("n", "gcd_med", "model_med", "ratio", "win_pct")}})
            del m
            torch.cuda.empty_cache()
        if summary:
            _write_summary(cdir, summary, f"anchor aps{aps}")
        del atr, ava, ate

    print("[clean-oneanchor] DONE", flush=True)


if __name__ == "__main__":
    main()

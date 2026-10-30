"""CLEAN RE-EVAL of the Op-M stack GAP-SWEEP checkpoints.

Every in-runner sweep TEST eval runs with FQ_AMP_REFRESH=84 (piecewise-constant amplitude
tables) as set by the sweep launchers.  The reported paper numbers use exact torus-integral
tables at the loaded weights, so each cell's five best.pt are re-evaluated here in a fresh
process with FQ_AMP_REFRESH UNSET.  Model construction, dataset loading and the eval routine
are IMPORTED from run_gap_sweep (same node-budget batching, same gap_row schema) -- this script
adds no protocol of its own.

Launch (per track; the jet source differs by track exactly as in the training runs):
  our:  FQ_SWEEP_SOBOLEV=1 FQ_SWEEP_DMU=blend FQ_READOUT_MODE=opm <stack env, NO REFRESH> \
        python -m task_alpha._clean_reeval_sweep --track our_processed --gaps 3,12,24
  std:  FQ_SWEEP_SOBOLEV=1 FQ_SWEEP_DMU=fd    FQ_READOUT_MODE=opm <stack env, NO REFRESH> \
        python -m task_alpha._clean_reeval_sweep --track standard_process --gaps 2,6,12,24

(our/6h = the headline run, re-evaluated separately by _clean_reeval_opmstack.)
Out per cell: seedN/{per_gap_clean.csv, test_buckets_clean.csv} + summary_clean.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch

from . import run_gap_sweep as gs
from .data import collate_graphs_interp
from .fqdeep_native import FQDeepNative
from .meltin_proof import _bvec

DEV = "cuda"
SEEDS = (0, 1, 2, 3, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=["our_processed", "standard_process"])
    ap.add_argument("--gaps", required=True, help="comma list, e.g. 3,12,24")
    a = ap.parse_args()
    gaps = [int(x) for x in a.gaps.split(",")]

    assert os.environ.get("FQ_READOUT_MODE") == "opm", "launch with the full Op-M stack env"
    assert os.environ.get("FQ_AMP_REFRESH", "0") in ("", "0"), \
        "FQ_AMP_REFRESH must be UNSET for the clean re-eval (exact tables)"
    assert gs.SOBOLEV and gs.NATIVE and gs.OPM, "FQ_SWEEP_SOBOLEV=1 + opm env required"
    exp_dmu = "blend" if a.track == "our_processed" else "fd"
    assert gs.SOB_DMU == exp_dmu, f"FQ_SWEEP_DMU must be {exp_dmu} for {a.track} (got {gs.SOB_DMU})"
    print(f"[clean-sweep] track={a.track} gaps={gaps} root={gs.SWEEP_ROOT} dmu={gs.SOB_DMU} | "
          f"amp={os.environ.get('FQ_AMP_MODE')}/nmc={os.environ.get('FQ_AMP_NMC')}/refresh=EXACT",
          flush=True)

    for gap in gaps:
        cdir = gs.cell_dir(a.track, gap, "fqgnn")
        tr, va, te = gs.load_dataset(a.track, gap)
        summary = []
        for seed in SEEDS:
            sdir = f"{cdir}/seed{seed}"
            ckp = f"{sdir}/best.pt"
            if not os.path.exists(ckp):
                print(f"[clean-sweep] MISSING {ckp} -- skipped", flush=True)
                continue
            ck = torch.load(ckp, map_location="cpu")["state_dict"]
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            m = FQDeepNative(n=6, n_freqs=48, width=96, init_scale=1.84, seed=seed, in_feat=6,
                             layers=3, jacobi_R=4, oph_weighted=True, dmu_source=gs.SOB_DMU,
                             metric_backbone="sobolev2").to(DEV)
            m.use_cuda = True
            with torch.no_grad():
                m.s_raw.fill_(math.log(math.expm1(2.5)))
            b0 = collate_graphs_interp(tr[:4])
            m(b0[0].to(DEV), b0[1].to(DEV), b0[4].to(DEV), b0[6].to(DEV), _bvec(tr[:4]),
              **gs._blend_kw(m, tr[:4]))
            m.load_state_dict(ck, strict=True)
            m.eval()

            _, _, te_errs = gs.eval_fq_budget(m, te)
            pg = [gs.gap_row(gi, g, fe) for gi, (g, fe) in enumerate(zip(te, te_errs)) if len(fe)]
            brows = gs.bucket_rows(pg)
            gs.write_per_gap(f"{sdir}/per_gap_clean.csv", pg)
            gs.write_buckets(f"{sdir}/test_buckets_clean.csv", brows)
            ov = brows[0]
            print(f"[clean-sweep {a.track}/gap_{gap}h seed {seed}] {ov['n']} gaps "
                  f"ratio {ov['ratio']} (win {ov['win_pct']}%)", flush=True)
            summary.append({"seed": seed, **{k: ov[k] for k in
                                             ("n", "gcd_med", "model_med", "ratio", "win_pct")}})
            del m
            torch.cuda.empty_cache()

        if summary:
            rat = np.array([s["ratio"] for s in summary], float)
            wp = np.array([s["win_pct"] for s in summary], float)
            with open(f"{cdir}/summary_clean.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
                w.writeheader(); w.writerows(summary)
                cw = csv.writer(f)
                cw.writerow(["MEAN", "", "", "", round(float(rat.mean()), 4),
                             round(float(wp.mean()), 1)])
                cw.writerow(["STD", "", "", "", round(float(rat.std(ddof=0)), 4),
                             round(float(wp.std(ddof=0)), 1)])
            print(f"=== CLEAN {a.track}/gap_{gap}h: {rat.mean():.4f} +/- {rat.std(ddof=0):.4f} "
                  f"(in-runner reference in summary.csv) ===", flush=True)

    print("[clean-sweep] DONE", flush=True)


if __name__ == "__main__":
    main()

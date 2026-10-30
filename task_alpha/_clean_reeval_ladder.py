"""CLEAN RE-EVAL of the Op-M stack ABLATION-LADDER checkpoints.

Same rationale as _clean_reeval_opmstack: the in-runner TEST evals run with FQ_AMP_REFRESH=84
(amplitude tables refreshed every 84 optimizer steps); the reported rung numbers use exact
torus-integral tables at the loaded weights.  Variant model construction is IMPORTED from
run_ablation_ladder_native.build_model (the exact training classes and post-construction
switches) and the dataset is the same load_gaps() 5350/635/774 split -- this script adds no
protocol of its own.

NOTE the full rung reference is the headline run, re-evaluated by _clean_reeval_opmstack -- this
script covers only the 9 ablated rungs.

Launch (stack env + FQ_LADDER_SOBOLEV=1, NO REFRESH):
  FQ_LADDER_SOBOLEV=1 FQ_READOUT_MODE=opm <stack env> \
  python -m task_alpha._clean_reeval_ladder

Out per rung: seedN/{per_gap_clean.csv, test_buckets_clean.csv} + summary_clean.csv
"""
from __future__ import annotations

import csv
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch

from . import run_ablation_ladder_native as lad
from .data import collate_graphs_interp
from .meltin_proof import _bvec
from .metrics import model_errors

DEV = "cuda"
SEEDS = (0, 1, 2, 3, 4)


def main():
    assert os.environ.get("FQ_READOUT_MODE") == "opm", "launch with the full Op-M stack env"
    assert os.environ.get("FQ_AMP_REFRESH", "0") in ("", "0"), \
        "FQ_AMP_REFRESH must be UNSET for the clean re-eval (exact tables)"
    assert lad.SOBOLEV and lad.OPM, "FQ_LADDER_SOBOLEV=1 + opm env required"
    print(f"[clean-ladder] root={lad.OUT_ROOT} variants={lad.VARIANTS} | "
          f"amp={os.environ.get('FQ_AMP_MODE')}/nmc={os.environ.get('FQ_AMP_NMC')}/refresh=EXACT",
          flush=True)

    tr, va, te = lad.load_gaps()

    for variant in lad.VARIANTS:
        vdir = f"{lad.OUT_ROOT}/{variant}"
        if not os.path.isdir(vdir):
            print(f"[clean-ladder] no dir for {variant} -- skipped", flush=True)
            continue
        summary = []
        for seed in SEEDS:
            sdir = f"{vdir}/seed{seed}"
            ckp = f"{sdir}/best.pt"
            if not os.path.exists(ckp):
                print(f"[clean-ladder] MISSING {ckp} -- skipped", flush=True)
                continue
            ck = torch.load(ckp, map_location="cpu")["state_dict"]
            m = lad.build_model(variant, seed)                 # exact training-run construction + RNG
            b0 = collate_graphs_interp(tr[:4])
            m(b0[0].to(DEV), b0[1].to(DEV), b0[4].to(DEV), b0[6].to(DEV), _bvec(tr[:4]),
              **lad._blend_kw(m, tr[:4]))                      # LazyLinear warmup
            m.load_state_dict(ck, strict=True)
            m.eval()

            pg = []
            gi = 0
            with torch.no_grad():
                for s0 in range(0, len(te), 64):
                    b = te[s0:s0 + 64]
                    x, coord, yp, mask, ei, ed, aei = collate_graphs_interp(b)
                    pred = m(x.to(DEV), coord.to(DEV), ei.to(DEV), aei.to(DEV), _bvec(b),
                             **lad._blend_kw(m, b)).float().cpu()
                    off = 0
                    for g in b:
                        ng = g.x.shape[0]; gp = pred[off:off + ng]; off += ng
                        fe = model_errors(gp, g)
                        if len(fe):
                            pg.append(lad.gap_row(gi, g, fe))
                        gi += 1
            brows = lad.bucket_rows(pg)
            lad.write_per_gap(f"{sdir}/per_gap_clean.csv", pg)
            lad.write_buckets(f"{sdir}/test_buckets_clean.csv", brows)
            ov = brows[0]
            print(f"[clean-ladder {variant} seed {seed}] {ov['n']} gaps ratio {ov['ratio']} "
                  f"(win {ov['win_pct']}%)", flush=True)
            summary.append({"seed": seed, **{k: ov[k] for k in
                                             ("n", "gcd_med", "model_med", "ratio", "win_pct")}})
            del m
            torch.cuda.empty_cache()

        if summary:
            rat = np.array([s["ratio"] for s in summary], float)
            wp = np.array([s["win_pct"] for s in summary], float)
            with open(f"{vdir}/summary_clean.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
                w.writeheader(); w.writerows(summary)
                cw = csv.writer(f)
                cw.writerow(["MEAN", "", "", "", round(float(rat.mean()), 4),
                             round(float(wp.mean()), 1)])
                cw.writerow(["STD", "", "", "", round(float(rat.std(ddof=0)), 4),
                             round(float(wp.std(ddof=0)), 1)])
            print(f"=== CLEAN ladder/{variant}: {rat.mean():.4f} +/- {rat.std(ddof=0):.4f} ===",
                  flush=True)

    print("[clean-ladder] DONE", flush=True)


if __name__ == "__main__":
    main()

"""CLEAN RE-EVAL of the Op-M stack headline checkpoints.

The in-runner TEST numbers run with FQ_AMP_REFRESH=84: the best-checkpoint U is evaluated
against a piecewise-constant amplitude table refreshed every 84 optimizer steps, not the exact
AKKV Thm 2.6(3) torus integrals at the loaded weights.  The reported numbers therefore come from THIS script: a
fresh process launched with the full stack env but FQ_AMP_REFRESH UNSET, so the QMC table is
computed exactly at each loaded checkpoint's (L, U) and cached for the whole eval.

Launch (stack env, NO REFRESH):
  FQ_READOUT_MODE=opm FQ_OPM_BATCHED=1 FQ_OPM_GAMMA=hatp FQ_OPM_VBAR=0.58298 \
  FQ_OPM_MBAR=2.68022 FQ_OPM_BETA2=0.2148 FQ_OPM_LAM=c2 FQ_AMP_MODE=akkv_torus \
  FQ_AMP_NMC=4194304 python -m task_alpha._clean_reeval_opmstack

Out: seedN/per_gap_clean.csv + seedN/test_buckets_clean.csv + summary_clean.csv
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

from .data import collate_graphs_interp
from .data_raw import build_dataset_raw
from .experiment_compare import _sinuosity
from .fqdeep_native import FQDeepNative
from .meltin_proof import _bvec
from .metrics import gcd_errors, model_errors
from .train_stream import _split

DEV = "cuda"
OUT = "task_alpha/runs/fqnative_opmstack_5seed"
SEEDS = (0, 1, 2, 3, 4)


def _gm(b):
    return torch.tensor([float(g.gap_minutes) for g in b], device=DEV)


def main():
    assert os.environ.get("FQ_READOUT_MODE") == "opm", "launch with the full Op-M stack env"
    assert os.environ.get("FQ_AMP_REFRESH", "0") in ("", "0"), \
        "FQ_AMP_REFRESH must be UNSET for the clean re-eval (exact tables)"
    print(f"[clean-reeval] amp={os.environ.get('FQ_AMP_MODE')}/nmc={os.environ.get('FQ_AMP_NMC')}"
          f"/refresh=EXACT | gamma={os.environ.get('FQ_OPM_GAMMA')} lam={os.environ.get('FQ_OPM_LAM')}",
          flush=True)

    allg = build_dataset_raw(years=(2016, 2017, 2018), months=(7, 8), gap_minutes=360,
                             max_gap_len=500, min_span_factor=1.5, n_anchor=8, clean_only=True,
                             n_mmsi=None, max_files_per_part=None, seed=0)
    allg = [g for g in allg if int(g.mask.sum()) > 0 and _split(g.mmsi) is not None]
    tr = [g for g in allg if _split(g.mmsi) == "train"]
    te = [g for g in allg if _split(g.mmsi) == "test"]
    print(f"{len(te)} test gaps", flush=True)

    summary = []
    for seed in SEEDS:
        sdir = f"{OUT}/seed{seed}"
        ck = torch.load(f"{sdir}/best.pt", map_location="cpu")["state_dict"]
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        m = FQDeepNative(n=6, n_freqs=48, width=96, init_scale=1.84, seed=seed, in_feat=6,
                         layers=3, jacobi_R=4, oph_weighted=True, dmu_source="blend",
                         metric_backbone="sobolev2").to(DEV)
        m.use_cuda = True
        with torch.no_grad():
            m.s_raw.fill_(math.log(math.expm1(2.5)))
        b0 = collate_graphs_interp(tr[:4])
        m(b0[0].to(DEV), b0[1].to(DEV), b0[4].to(DEV), b0[6].to(DEV), _bvec(tr[:4]),
          gap_minutes=_gm(tr[:4]))
        m.load_state_dict(ck, strict=True)
        m.eval()

        fq, gc, sn, pg = [], [], [], []
        with torch.no_grad():
            for s0 in range(0, len(te), 64):
                b = te[s0:s0 + 64]
                x, coord, yp, mask, ei, ed, aei = collate_graphs_interp(b)
                pred = m(x.to(DEV), coord.to(DEV), ei.to(DEV), aei.to(DEV), _bvec(b),
                         gap_minutes=_gm(b)).float().cpu()
                off = 0
                for g in b:
                    ng = g.x.shape[0]; gp = pred[off:off + ng]; off += ng
                    fe = model_errors(gp, g)
                    if len(fe):
                        fm = float(np.median(fe)); gm = float(np.median(gcd_errors(g)))
                        si = _sinuosity(g)
                        fq.append(fm); gc.append(gm); sn.append(si)
                        pg.append({"mmsi": int(g.mmsi),
                                   "gap_min": round(float(g.gap_minutes), 1),
                                   "sinuosity": round(float(si), 4) if np.isfinite(si) else "",
                                   "gcd_med": round(gm, 3), "model_med": round(fm, 3),
                                   "win": int(fm < gm)})
        fq, gc, sn = np.array(fq), np.array(gc), np.array(sn)
        with open(f"{sdir}/per_gap_clean.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pg[0].keys())); w.writeheader(); w.writerows(pg)
        keep = np.isfinite(sn); fqk, gck, snk = fq[keep], gc[keep], sn[keep]
        brows = [{"bucket": "OVERALL", "n": len(fqk),
                  "gcd_med": round(float(np.median(gck)), 3),
                  "model_med": round(float(np.median(fqk)), 3),
                  "ratio": round(float(np.median(fqk) / np.median(gck)), 4),
                  "win_pct": round(100 * float(np.mean(fqk < gck)), 1)}]
        for lo, hi in zip((0.0, 0.02, 0.05, 0.10, 0.25), (0.02, 0.05, 0.10, 0.25, 1e9)):
            sel = (snk >= lo) & (snk < hi)
            if sel.sum() < 3:
                continue
            lab = f"{lo:.2f}-{hi:.2f}" if hi < 1e9 else f">{lo:.2f}"
            brows.append({"bucket": lab, "n": int(sel.sum()),
                          "gcd_med": round(float(np.median(gck[sel])), 3),
                          "model_med": round(float(np.median(fqk[sel])), 3),
                          "ratio": round(float(np.median(fqk[sel]) / np.median(gck[sel])), 4),
                          "win_pct": round(100 * float(np.mean(fqk[sel] < gck[sel])), 1)})
        with open(f"{sdir}/test_buckets_clean.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(brows[0].keys())); w.writeheader(); w.writerows(brows)
        ov = brows[0]
        print(f"[clean-reeval seed {seed}] {ov['n']} gaps ratio {ov['ratio']} "
              f"(win {ov['win_pct']}%)", flush=True)
        summary.append({"seed": seed, **ov})
        del m
        torch.cuda.empty_cache()

    rat = np.array([s["ratio"] for s in summary], float)
    wp = np.array([s["win_pct"] for s in summary], float)
    with open(f"{OUT}/summary_clean.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
        cw = csv.writer(f)
        cw.writerow(["MEAN", "", "", "", round(float(rat.mean()), 4), round(float(wp.mean()), 1)])
        cw.writerow(["STD", "", "", "", round(float(rat.std(ddof=0)), 4), round(float(wp.std(ddof=0)), 1)])
    print(f"\n=== CLEAN OP-M STACK 5-SEED (exact tables): ratio {rat.mean():.4f} +/- "
          f"{rat.std(ddof=0):.4f} | win {wp.mean():.1f} +/- {wp.std(ddof=0):.1f}% ===", flush=True)
    print("=== references: in-runner (refreshed-table) 0.5606 +/- 0.0036 | Hermite 0.5724 | "
          "conformant Op-H 0.5717 +/- 0.0055 ===", flush=True)


if __name__ == "__main__":
    main()

"""ZERO-SHOT TEMPORAL HOLDOUT: July-August 2019, raw track, 6-h gaps.

Builds a test-only gap set from a year the deployed model never saw (2019; training used
2016-18), under the IDENTICAL pipeline: same raw partitions (m123/m18/m27), same per-year
clean whitelist (static_clean_2019.parquet), same run segmentation, thinning, graph
construction and gap parameters as the deployed 6-h cell.  Evaluates, on the identical gaps:

  * the FIVE FROZEN Op-M checkpoints (runs/fqnative_opmstack_5seed/seed{0..4}/best.pt),
    exact torus-integral amplitude tables (FQ_AMP_REFRESH unset), no retraining of any kind;
  * the published Nguyen cubic-Hermite baseline (deterministic);
  * the GCD reference (per-gap medians; the shared denominator).

Every gap carries a seen_vessel flag (its MMSI appears in the deployed 6-h train/val vessel
sets), so summaries are reported for ALL 2019 gaps and for the UNSEEN-vessel subset (year AND
vessel held out).

Launch (full Op-M stack env, from the repository root):
  FQ_READOUT_MODE=opm FQ_OPM_BATCHED=1 FQ_OPM_GAMMA=hatp FQ_OPM_VBAR=0.58298 \
  FQ_OPM_MBAR=2.68022 FQ_OPM_BETA2=0.2148 FQ_OPM_LAM=c2 FQ_AMP_MODE=akkv_torus \
  FQ_AMP_NMC=4194304 python -m task_alpha._holdout_eval_2019

Out: task_alpha/runs/holdout_2019/
  fqgnn/seed{k}/per_gap.csv + test_buckets.csv, hermite/per_gap.csv + test_buckets.csv,
  summary.csv (model x scope rows), manifest.json
"""
from __future__ import annotations

import csv
import json
import math
import os
import pickle
import random
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
import numpy as np
import torch

from task_alpha.baselines.common_gaps import model_errors_latlon
from task_alpha.baselines.Hermite.hermite_model import anchor_states_lonlat, art2_predict
from task_alpha.data import RunGraph, collate_graphs_interp  # noqa: F401
from task_alpha.data_raw import build_dataset_raw
from task_alpha.experiment_compare import _sinuosity
from task_alpha.fqdeep_native import FQDeepNative
from task_alpha.meltin_proof import _bvec
from task_alpha.metrics import gcd_errors, model_errors

DEV = "cuda"
HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "runs", "fqnative_opmstack_5seed")
OUT = os.path.join(HERE, "runs", "holdout_2019")
SEEDS = (0, 1, 2, 3, 4)
YEARS, MONTHS, GAP_MIN = (2019,), (7, 8), 360
BUCKETS = list(zip((0.0, 0.02, 0.05, 0.10, 0.25), (0.02, 0.05, 0.10, 0.25, 1e9)))


def _gm(b):
    return torch.tensor([float(g.gap_minutes) for g in b], device=DEV)


def seen_vessels():
    mm = set()
    for name in ("train", "val"):
        with open(os.path.join(HERE, "data", "our_processed", "gap_6h", f"{name}.pkl"), "rb") as f:
            mm |= {int(g.mmsi) for g in pickle.load(f)}
    return mm


def bucket_table(sn, mod, gcd):
    keep = np.isfinite(sn)
    snk, mk, gk = sn[keep], mod[keep], gcd[keep]
    rows = [{"bucket": "OVERALL", "n": len(mk),
             "gcd_med": round(float(np.median(gk)), 3), "model_med": round(float(np.median(mk)), 3),
             "ratio": round(float(np.median(mk) / np.median(gk)), 4),
             "win_pct": round(100 * float(np.mean(mk < gk)), 1)}]
    for lo, hi in BUCKETS:
        sel = (snk >= lo) & (snk < hi)
        if sel.sum() < 3:
            continue
        lab = f"{lo:.2f}-{hi:.2f}" if hi < 1e9 else f">{lo:.2f}"
        rows.append({"bucket": lab, "n": int(sel.sum()),
                     "gcd_med": round(float(np.median(gk[sel])), 3),
                     "model_med": round(float(np.median(mk[sel])), 3),
                     "ratio": round(float(np.median(mk[sel]) / np.median(gk[sel])), 4),
                     "win_pct": round(100 * float(np.mean(mk[sel] < gk[sel])), 1)})
    return rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    assert os.environ.get("FQ_READOUT_MODE") == "opm", "launch with the full Op-M stack env"
    assert os.environ.get("FQ_AMP_REFRESH", "0") in ("", "0"), "FQ_AMP_REFRESH must be UNSET"

    allg = build_dataset_raw(years=YEARS, months=MONTHS, gap_minutes=GAP_MIN,
                             max_gap_len=500, min_span_factor=1.5, n_anchor=8, clean_only=True,
                             n_mmsi=None, max_files_per_part=None, seed=0)
    te = [g for g in allg if int(g.mask.sum()) > 0]
    assert te, "holdout build returned zero gaps -- check the 2019 cache partitions"
    seen = seen_vessels()
    flags = np.array([int(g.mmsi) in seen for g in te], dtype=bool)
    print(f"[holdout-2019] {len(te)} gaps from {len({g.mmsi for g in te})} vessels; "
          f"{int(flags.sum())} gaps on 2016-18 train/val vessels, "
          f"{int((~flags).sum())} on fully unseen vessels", flush=True)

    sn = np.array([_sinuosity(g) for g in te])
    gcd = np.array([float(np.median(gcd_errors(g))) for g in te])
    summary = []

    # ---- FQ-GNN: five frozen checkpoints ------------------------------------------------
    for seed in SEEDS:
        ck = torch.load(os.path.join(CKPT, f"seed{seed}", "best.pt"), map_location="cpu")["state_dict"]
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        m = FQDeepNative(n=6, n_freqs=48, width=96, init_scale=1.84, seed=seed, in_feat=6,
                         layers=3, jacobi_R=4, oph_weighted=True, dmu_source="blend",
                         metric_backbone="sobolev2").to(DEV)
        m.use_cuda = True
        with torch.no_grad():
            m.s_raw.fill_(math.log(math.expm1(2.5)))
        b0 = collate_graphs_interp(te[:4])
        m(b0[0].to(DEV), b0[1].to(DEV), b0[4].to(DEV), b0[6].to(DEV), _bvec(te[:4]),
          gap_minutes=_gm(te[:4]))
        m.load_state_dict(ck, strict=True)
        m.eval()

        med = np.full(len(te), np.nan)
        with torch.no_grad():
            for s0 in range(0, len(te), 64):
                b = te[s0:s0 + 64]
                x, coord, yp, mask, ei, ed, aei = collate_graphs_interp(b)
                pred = m(x.to(DEV), coord.to(DEV), ei.to(DEV), aei.to(DEV), _bvec(b),
                         gap_minutes=_gm(b)).float().cpu()
                off = 0
                for j, g in enumerate(b):
                    ng = g.x.shape[0]; gp = pred[off:off + ng]; off += ng
                    fe = model_errors(gp, g)
                    if len(fe):
                        med[s0 + j] = float(np.median(fe))
        ok = np.isfinite(med)
        pg = [{"gap_id": i, "mmsi": int(te[i].mmsi), "seen_vessel": int(flags[i]),
               "sinuosity": round(float(sn[i]), 4) if np.isfinite(sn[i]) else "",
               "gcd_med": round(gcd[i], 3), "model_med": round(med[i], 3),
               "win": int(med[i] < gcd[i])} for i in range(len(te)) if ok[i]]
        write_csv(os.path.join(OUT, "fqgnn", f"seed{seed}", "per_gap.csv"), pg)
        bt = bucket_table(sn[ok], med[ok], gcd[ok])
        write_csv(os.path.join(OUT, "fqgnn", f"seed{seed}", "test_buckets.csv"), bt)
        for scope, sel in (("all", ok), ("unseen_vessels", ok & ~flags)):
            r = float(np.median(med[sel]) / np.median(gcd[sel]))
            w = float(np.mean(med[sel] < gcd[sel]))
            summary.append({"model": "fqgnn", "seed": seed, "scope": scope, "n": int(sel.sum()),
                            "gcd_med": round(float(np.median(gcd[sel])), 3),
                            "model_med": round(float(np.median(med[sel])), 3),
                            "ratio": round(r, 4), "win_pct": round(100 * w, 1)})
        print(f"[holdout-2019 fqgnn seed {seed}] all ratio "
              f"{summary[-2]['ratio']} (win {summary[-2]['win_pct']}%) | unseen "
              f"{summary[-1]['ratio']} (win {summary[-1]['win_pct']}%)", flush=True)
        del m
        torch.cuda.empty_cache()

    # ---- Nguyen cubic-Hermite (deterministic) -------------------------------------------
    hmed = np.full(len(te), np.nan)
    for i, g in enumerate(te):
        before, after, held_t, hidx = anchor_states_lonlat(g)
        if len(before["t"]) == 0 or len(after["t"]) == 0 or len(held_t) == 0:
            continue
        plat, plon = art2_predict(before, after, held_t)
        err = model_errors_latlon(g, plat, plon)
        if len(err) and np.all(np.isfinite(err)):
            hmed[i] = float(np.median(err))
    hok = np.isfinite(hmed)
    pg = [{"gap_id": i, "mmsi": int(te[i].mmsi), "seen_vessel": int(flags[i]),
           "sinuosity": round(float(sn[i]), 4) if np.isfinite(sn[i]) else "",
           "gcd_med": round(gcd[i], 3), "model_med": round(hmed[i], 3),
           "win": int(hmed[i] < gcd[i])} for i in range(len(te)) if hok[i]]
    write_csv(os.path.join(OUT, "hermite", "per_gap.csv"), pg)
    write_csv(os.path.join(OUT, "hermite", "test_buckets.csv"),
              bucket_table(sn[hok], hmed[hok], gcd[hok]))
    for scope, sel in (("all", hok), ("unseen_vessels", hok & ~flags)):
        summary.append({"model": "hermite", "seed": 0, "scope": scope, "n": int(sel.sum()),
                        "gcd_med": round(float(np.median(gcd[sel])), 3),
                        "model_med": round(float(np.median(hmed[sel])), 3),
                        "ratio": round(float(np.median(hmed[sel]) / np.median(gcd[sel])), 4),
                        "win_pct": round(100 * float(np.mean(hmed[sel] < gcd[sel])), 1)})
    print(f"[holdout-2019 hermite] all ratio {summary[-2]['ratio']} | "
          f"unseen {summary[-1]['ratio']}", flush=True)

    write_csv(os.path.join(OUT, "summary.csv"), summary)
    fq_all = [s["ratio"] for s in summary if s["model"] == "fqgnn" and s["scope"] == "all"]
    fq_un = [s["ratio"] for s in summary if s["model"] == "fqgnn" and s["scope"] == "unseen_vessels"]
    manifest = {
        "purpose": "Zero-shot temporal holdout: frozen 2016-18 checkpoints on Jul-Aug 2019",
        "years": list(YEARS), "months": list(MONTHS), "gap_minutes": GAP_MIN,
        "n_gaps": len(te), "n_vessels": len({g.mmsi for g in te}),
        "n_gaps_seen_vessels": int(flags.sum()), "n_gaps_unseen_vessels": int((~flags).sum()),
        "fq_ratio_all_mean": round(float(np.mean(fq_all)), 4),
        "fq_ratio_all_std": round(float(np.std(fq_all)), 4),
        "fq_ratio_unseen_mean": round(float(np.mean(fq_un)), 4),
        "fq_ratio_unseen_std": round(float(np.std(fq_un)), 4),
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n=== HOLDOUT 2019 (zero-shot): FQ ratio all {manifest['fq_ratio_all_mean']} +/- "
          f"{manifest['fq_ratio_all_std']} | unseen-vessel {manifest['fq_ratio_unseen_mean']} +/- "
          f"{manifest['fq_ratio_unseen_std']} | in-domain reference 0.5646 +/- 0.0025 ===", flush=True)


if __name__ == "__main__":
    main()

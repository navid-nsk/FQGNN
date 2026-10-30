"""CONFORMANCE FAST-TRAIN HARNESS: train on a fixed SUBSAMPLE (1200 train / 200 val graphs, short
schedule) and test on the SAME fixed 200-gap subsample as _conform_subsample.py -- so a code or
configuration change can be compared at the TRAINED level in ~10 minutes instead of hours.
Absolute numbers sit above the full-data results (less data); what matters is the RANKING/DELTA
between configurations.

Config flows through the same env switches as production (FQ_AMP_MODE / FQ_AMP_NMC /
FQ_AMP_REFRESH / FQ_ENUM_RINGS) + the FQDeepNative.production() ctor.

Run: python -m task_alpha._conform_fasttrain --tag <name> [--seed 0] [--epochs 25]
Log: task_alpha/runs/conform_baseline/fasttrain_log.csv
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import pickle
import random
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch

from .data import collate_graphs_interp
from .data_raw import build_dataset_raw
from .fqdeep_native import FQDeepNative
from .meltin_proof import _bvec
from .metrics import gcd_errors, model_errors
from .train_stream import _split

DEV = "cuda"
OUT = "task_alpha/runs/conform_baseline"
N_TRAIN, N_VAL = 1200, 200


def _gm(b):
    return torch.tensor([float(g.gap_minutes) for g in b], device=DEV)


def _eval(m, graphs, bs=64):
    m.eval(); fq, gc = [], []
    with torch.no_grad():
        for s in range(0, len(graphs), bs):
            b = graphs[s:s + bs]
            x, coord, yp, mask, ei, ed, aei = collate_graphs_interp(b)
            pred = m(x.to(DEV), coord.to(DEV), ei.to(DEV), aei.to(DEV), _bvec(b),
                     gap_minutes=_gm(b)).float().cpu()
            off = 0
            for g in b:
                ng = g.x.shape[0]; gp = pred[off:off + ng]; off += ng
                fe = model_errors(gp, g)
                if len(fe):
                    fq.append(float(np.median(fe))); gc.append(float(np.median(gcd_errors(g))))
    return float(np.median(fq) / np.median(gc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=5)
    a = ap.parse_args()
    t0 = time.time()

    allg = build_dataset_raw(years=(2016, 2017, 2018), months=(7, 8), gap_minutes=360,
                             max_gap_len=500, min_span_factor=1.5, n_anchor=8, clean_only=True,
                             n_mmsi=None, max_files_per_part=None, seed=0)
    allg = [g for g in allg if int(g.mask.sum()) > 0 and _split(g.mmsi) is not None]
    tr_full = [g for g in allg if _split(g.mmsi) == "train"]
    va_full = [g for g in allg if _split(g.mmsi) == "val"]
    rng = np.random.default_rng(20260718)
    tr = [tr_full[i] for i in sorted(rng.choice(len(tr_full), size=min(N_TRAIN, len(tr_full)),
                                                replace=False).tolist())]
    va = [va_full[i] for i in sorted(rng.choice(len(va_full), size=min(N_VAL, len(va_full)),
                                                replace=False).tolist())]
    with open("task_alpha/data/our_processed/gap_6h/test.pkl", "rb") as f:
        te_all = pickle.load(f)
    ids = json.load(open(f"{OUT}/subsample_ids.json"))
    te = [te_all[i] for i in ids]
    print(f"[fasttrain:{a.tag}] {len(tr)} train / {len(va)} val / {len(te)} test-subsample | "
          f"amp={os.environ.get('FQ_AMP_MODE', 'secular')} rings={os.environ.get('FQ_ENUM_RINGS', '0')}",
          flush=True)

    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    m = FQDeepNative.production(seed=a.seed).to(DEV)
    m.use_cuda = True
    with torch.no_grad():
        m.s_raw.fill_(math.log(math.expm1(2.5)))
    b0 = collate_graphs_interp(tr[:4])
    m(b0[0].to(DEV), b0[1].to(DEV), b0[4].to(DEV), b0[6].to(DEV), _bvec(tr[:4]),
      gap_minutes=_gm(tr[:4]))
    opt = torch.optim.Adam(m.parameters(), lr=5e-4)

    best, best_state, pctr, bs = 1e9, None, 0, 64
    for ep in range(a.epochs):
        m.train(); order = list(range(len(tr))); random.Random(a.seed * 1000 + ep).shuffle(order)
        ep_loss, ep_nb = 0.0, 0
        for i in range(0, len(order), bs):
            b = [tr[j] for j in order[i:i + bs]]
            x, co, yp, mk, ei, _, ae = collate_graphs_interp(b)
            pred = m(x.to(DEV), co.to(DEV), ei.to(DEV), ae.to(DEV), _bvec(b), gap_minutes=_gm(b))
            L = ((pred[mk.to(DEV)] - yp.to(DEV)[mk.to(DEV)]) ** 2).mean()
            opt.zero_grad()
            if torch.isfinite(L):
                L.backward()
                gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                # finite-grad guard: with a non-finite grad norm, clip_grad_norm_ rescales
                # every grad to NaN and the step would poison the model (U -> NaN -> the
                # eigendecomposition fails).  Only such steps are skipped; steps with a
                # finite grad norm are unaffected (clip_coef unchanged).
                if torch.isfinite(gn):
                    opt.step()
                ep_loss += float(L.detach()); ep_nb += 1
        vr = _eval(m, va)
        print(f"[fasttrain:{a.tag}] ep {ep:2d}: train {ep_loss / max(ep_nb, 1):.5f} | val {vr:.4f} "
              f"(best {best:.4f})", flush=True)
        if vr < best - 1e-4:
            best, best_state, pctr = vr, copy.deepcopy(m.state_dict()), 0
        else:
            pctr += 1
        if pctr >= a.patience:
            print(f"[fasttrain:{a.tag}] early stop", flush=True); break

    m.load_state_dict(best_state)
    tr_ratio = _eval(m, te)
    secs = round(time.time() - t0, 1)
    print(f"[fasttrain:{a.tag}] TEST-SUBSAMPLE ratio {tr_ratio:.4f} | best val {best:.4f} | "
          f"{ep + 1} epochs | {secs}s", flush=True)
    print("references: full-data sobolev 5-seed 0.5720 (770 gaps) | closed form on subsample 0.5642",
          flush=True)
    logp = f"{OUT}/fasttrain_log.csv"
    new = not os.path.exists(logp)
    with open(logp, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["tag", "seed", "amp_mode", "rings", "best_val", "test_subsample_ratio",
                        "epochs", "secs"])
        w.writerow([a.tag, a.seed, os.environ.get("FQ_AMP_MODE", "secular"),
                    os.environ.get("FQ_ENUM_RINGS", "0"), round(best, 4), round(tr_ratio, 4),
                    ep + 1, secs])


if __name__ == "__main__":
    main()

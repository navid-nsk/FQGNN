"""task_alpha.run_gap_sweep -- RESUMABLE gap-length x track x model sweep for the FQGNN paper.

Cells = (track, gap, model):
  tracks   our_processed {3,6,12,24}h and standard_process {2,6,12,24}h
  models   fqgnn (FQDeepFaithful; FQDeepNative in the native/sobolev modes) / rffgnn (matched random-basis control) /
           bigru (textbook RNN control)
  seeds    5 (0..4) per cell.
EXCEPTION: our_processed/gap_6h comes from the main 5-seed runs (fqgnn: runs/full_faithful_5seed,
rffgnn: baselines/RFFGNN/runs_5seed_gaps, bigru: baselines/BiGRU/runs_5seed_gaps) -- the queue
SKIPS it; assemble_gap_sweep.py reads those results from their own run folders.

DATA (built by a separate pipeline -- this runner never builds data):
  task_alpha/data/<track>/gap_<G>h/{train.pkl, val.pkl, test.pkl}  = pickled lists of
  task_alpha.data.RunGraph (+ manifest.json).  A missing folder at runtime prints
  'DATASET MISSING <path>' and the cell is skipped (the queue keeps going).

RECIPES (identical to each model's own 5-seed runner -- the models are used READ-ONLY):
  fqgnn   run_full_faithful_5seed.py: FQDeepFaithful(n=6, n_freqs=48, width=96, init_scale=1.84,
          in_feat=6, layers=3, jacobi_R=4), s_raw init 2.5, LazyLinear warmup on tr[:4],
          Adam 5e-4, clip 1.0, bs 64, <=60 ep, patience 8 on val ratio, full telemetry columns.
  rffgnn  baselines/RFFGNN/run_rffgnn_5seed_gaps.py: RFFGNN class + eval imported from that file;
          identical FQGNN training recipe (Adam 5e-4, clip 1.0, bs 64, <=60 ep, patience 8).
  bigru   baselines/BiGRU/run_bigru_5seed_gaps.py: BiGRUImputer + to_series feature pipeline
          imported from that file; fit_latlon_stats z-scoring on THIS dataset's train split;
          Adam 1e-3, bs 32, <=100 ep, patience 10 on val median-NM.  cuDNN can crash at process
          teardown on Windows AFTER all work is done (0xC0000409), so bigru cells run in a CHILD
          process and success is judged by the per-seed test_buckets.csv files, not the exit code.
  All models: sequential, fp32, per-epoch torch.cuda.empty_cache() (run_ablation_ladder defrag
  pattern), full free between seeds.  Normalization stats are computed per dataset (train split).

RESUME (seed level):  outputs under task_alpha/runs/gap_sweep/<track>/gap_<G>h/<model>/seed<k>/
  {epochs.csv, per_gap.csv, test_buckets.csv, best.pt} -- formats identical to the
  5-seed runners (per_gap via common_gaps.gap_row: gap_id = index in that dataset's te list).
  If seed<k>/test_buckets.csv exists the seed is skipped (RESUME-SKIP).  After every seed the
  cell aggregates (all_seeds_epochs.csv, all_seeds_test_buckets.csv, summary.csv +MEAN/STD) are
  REBUILT FROM DISK and task_alpha/runs/gap_sweep/status.csv is rewritten.  A killed queue
  relaunches and continues exactly where it stopped.

QUEUE ORDER (cheapest first for early signal; fqgnn first within each cell):
  standard 2h -> our 3h -> standard 6h -> our 12h -> standard 12h -> our 24h -> standard 24h,
  models fqgnn -> rffgnn -> bigru.

Run (gpubench env, any cwd -- the script chdirs to the repository root itself):
  full queue :  python task_alpha/run_gap_sweep.py --queue
  one cell   :  ... run_gap_sweep.py --track standard_process --gap 12 --model fqgnn --seeds 5
  smoke      :  ... run_gap_sweep.py --smoke     (all 3 models on data/_smoke/gap_6h, 1 seed, 2 ep)
  native     :  ... run_gap_sweep.py --queue --native   (FQDeepNative Op-H-W, fqgnn cells only;
                root runs/gap_sweep_native)
  sobolev    :  FQ_SWEEP_SOBOLEV=1 [FQ_SWEEP_DMU=fd] [FQ_SWEEP_POOL=raw] ... --queue   (blend/fd jet +
                omega^-2 backbone; root runs/gap_sweep_sobolev[_fd][_mlp], plus the suffix _opm when
                FQ_READOUT_MODE=opm is also set)
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
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))                     # task_alpha
ROOT = os.path.abspath(os.path.join(HERE, ".."))                      # repository root
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np
import torch

from task_alpha.baselines.common_gaps import (bucket_rows, fit_latlon_stats, gap_row,
                                              model_errors_latlon, write_aggregates,
                                              write_buckets, write_epochs, write_per_gap)
from task_alpha.data import collate_graphs_interp                     # also registers RunGraph for unpickling
from task_alpha.diag_training import GROUPS, group_of
from task_alpha.fqdeep_faithful import FQDeepFaithful
from task_alpha.fqdeep_native import FQDeepNative
from task_alpha.meltin_proof import _bvec, _eval
from task_alpha.metrics import gcd_errors, model_errors


# ------------------------------------------------------------------------------------------------ #
#  Baseline modules: import the MODEL code from the baselines' own runner files (single-sourced)     #
# ------------------------------------------------------------------------------------------------ #
def _load_runner_module(pkg_name: str, rel_path: str, alias: str):
    """Import a baseline runner module.  Preferred: normal (namespace-package) import so the class
    keeps its canonical module path.  Fallback: importlib file-location load (same file, same code)."""
    try:
        import importlib
        return importlib.import_module(pkg_name)
    except Exception as e:                                            # pragma: no cover
        print(f"[import] {pkg_name} not importable ({e}); loading {rel_path} by file location", flush=True)
        import importlib.util
        spec = importlib.util.spec_from_file_location(alias, os.path.join(ROOT, rel_path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[alias] = mod
        spec.loader.exec_module(mod)
        return mod


RFFMOD = _load_runner_module("task_alpha.baselines.RFFGNN.run_rffgnn_5seed_gaps",
                             "task_alpha/baselines/RFFGNN/run_rffgnn_5seed_gaps.py", "rffgnn_runner_mod")
BGMOD = _load_runner_module("task_alpha.baselines.BiGRU.run_bigru_5seed_gaps",
                            "task_alpha/baselines/BiGRU/run_bigru_5seed_gaps.py", "bigru_runner_mod")

DEV = "cuda"
SWEEP_ROOT = "task_alpha/runs/gap_sweep"
DATA_ROOT = "task_alpha/data"
MODELS = ("fqgnn", "rffgnn", "bigru")
TRACK_GAPS = {"our_processed": (3, 6, 12, 24), "standard_process": (2, 6, 12, 24)}
# cheapest first (early signal): shortest gaps first, alternating tracks
QUEUE_ORDER = (("standard_process", 2), ("our_processed", 3), ("standard_process", 6),
               ("our_processed", 12), ("standard_process", 12), ("our_processed", 24),
               ("standard_process", 24))
# cells produced by the main 5-seed runs: the queue skips them; the assembler reads these paths
PRE_DONE = {("our_processed", 6): {
    "fqgnn": "task_alpha/runs/full_faithful_5seed",
    "rffgnn": "task_alpha/baselines/RFFGNN/runs_5seed_gaps",
    "bigru": "task_alpha/baselines/BiGRU/runs_5seed_gaps",
}}

# ---- NATIVE mode (--native / FQ_SWEEP_NATIVE=1; the env var reaches worker children) ----
# The fqgnn cells use the Op-H-W native readout (FQDeepNative, oph_weighted=True; 5-seed
# reference runs/fqnative2_5seed).  In native mode ONLY the fqgnn cells run -- the rffgnn/bigru
# cells are FQ-readout-independent and the assembler reads them from the op-k sweep root.
# A separate SWEEP_ROOT keeps the op-k results apart.
NATIVE = os.environ.get("FQ_SWEEP_NATIVE") == "1"


def _apply_native():
    global NATIVE, SWEEP_ROOT, MODELS, PRE_DONE
    NATIVE = True
    os.environ["FQ_SWEEP_NATIVE"] = "1"
    SWEEP_ROOT = "task_alpha/runs/gap_sweep_native"
    MODELS = ("fqgnn",)
    PRE_DONE = {("our_processed", 6): {"fqgnn": "task_alpha/runs/fqnative2_5seed"}}


if NATIVE:
    _apply_native()

# ---- SOBOLEV mode (FQ_SWEEP_SOBOLEV=1; implies NATIVE): blend Dmu + the omega^-2
# bending-metric backbone (run_fqnative_sobolev_5seed.py is the 6h reference).  A separate
# SWEEP_ROOT keeps both the op-k and the native sweep results apart.  FQ_SWEEP_POOL=raw additionally
# swaps the conditioning encoder to the raw-anchor MLP (the no-GNN arm of the encoder control) --
# again its own root, so the GNN and MLP arms are protocol-identical by construction.
SOBOLEV = os.environ.get("FQ_SWEEP_SOBOLEV") == "1"
POOL_RAW = os.environ.get("FQ_SWEEP_POOL") == "raw"
# FQ_SWEEP_DMU: boundary-jet source in sobolev mode.  "blend" (instrument where sog>0.1kn) for the
# raw-AIS our-track (sparse anchors -> poor secants, instrument reliable); "fd" for the standard
# track (10-min grid -> near-exact secants; resampled sog/cog unreliable: 47% of bracketing anchors
# disagree >25% with the dense secant -- input-side statistic, _diag_std2h_decompose.py).
SOB_DMU = os.environ.get("FQ_SWEEP_DMU", "blend")
assert SOB_DMU in ("blend", "fd"), SOB_DMU
if SOBOLEV:
    if not NATIVE:
        _apply_native()
        NATIVE = True
    SWEEP_ROOT = ("task_alpha/runs/gap_sweep_sobolev" + ("_fd" if SOB_DMU == "fd" else "")
                  + ("_mlp" if POOL_RAW else ""))
    PRE_DONE = {} if (POOL_RAW or SOB_DMU == "fd") else \
        {("our_processed", 6): {"fqgnn": "task_alpha/runs/fqnative_sobolev_5seed"}}

# ---- OP-M STACK (FQ_READOUT_MODE=opm).  The readout switch itself lives in
# FQDeepNative (env-read at construction; run_gap_sweep passes no readout kwarg), so here we ONLY
# repoint the output root, keeping the sobolev Op-H results apart.  Suffix logic mirrors the
# FQ_SWEEP_DMU pattern above -- gap_sweep_sobolev -> gap_sweep_sobolev_opm, gap_sweep_sobolev_fd
# (the std-track root) -> gap_sweep_sobolev_fd_opm, _mlp -> _mlp_opm.  our/6h pre-done =
# the Op-M stack five-seed run (run_fqnative_opmstack_5seed) instead of the Op-H sobolev 5-seed.
# Default and sobolev-non-opm invocations use the roots set above.
OPM = os.environ.get("FQ_READOUT_MODE") == "opm"
if OPM and SOBOLEV:
    SWEEP_ROOT += "_opm"
    PRE_DONE = {} if (POOL_RAW or SOB_DMU == "fd") else \
        {("our_processed", 6): {"fqgnn": "task_alpha/runs/fqnative_opmstack_5seed"}}

SMOKE_TRACK, SMOKE_GAP = "_smoke", 6
WORKER_ENV = "GAP_SWEEP_BIGRU_WORKER"
# per-model recipe knobs (EXACT values of each model's own 5-seed runner)
EPOCHS = {"fqgnn": 60, "rffgnn": 60, "bigru": 100}
PATIENCE = {"fqgnn": 8, "rffgnn": 8, "bigru": 10}


# ------------------------------------------------------------------------------------------------ #
#  Paths / dataset IO                                                                                #
# ------------------------------------------------------------------------------------------------ #
def data_dir(track: str, gap: int) -> str:
    return f"{DATA_ROOT}/{track}/gap_{gap}h"


def cell_dir(track: str, gap: int, model: str) -> str:
    return f"{SWEEP_ROOT}/{track}/gap_{gap}h/{model}"


def cell_tag(track: str, gap: int, model: str) -> str:
    return f"{track}/gap_{gap}h/{model}"


def dataset_exists(track: str, gap: int) -> bool:
    d = data_dir(track, gap)
    return all(os.path.exists(os.path.join(d, f"{p}.pkl")) for p in ("train", "val", "test"))


def load_dataset(track: str, gap: int):
    """(train, val, test) RunGraph lists from the pkl folder (contract: built by the data pipeline)."""
    d = data_dir(track, gap)
    parts = []
    for p in ("train", "val", "test"):
        with open(os.path.join(d, f"{p}.pkl"), "rb") as f:
            parts.append(pickle.load(f))
    mp = os.path.join(d, "manifest.json")
    if os.path.exists(mp):
        with open(mp) as f:
            man = json.load(f)
        print(f"[data] {d}: manifest {json.dumps(man)[:300]}", flush=True)
    tr, va, te = parts
    print(f"[data] {d}: {len(tr)} train / {len(va)} val / {len(te)} test RunGraphs", flush=True)
    # MATCHED-TRAINING-BUDGET CAP: the standard_process track is 6-12x larger than the our-track that
    # every recipe was sized on (62k train graphs at 2h).  Cap TRAIN at 8000 and VAL at 1000 with a
    # FIXED rng (deterministic; identical subset for every model and seed -> fair).  TEST is NEVER
    # capped -- reported numbers keep full statistical power.
    TRAIN_CAP, VAL_CAP = 8000, 1000
    if len(tr) > TRAIN_CAP:
        idx = np.random.default_rng(20260702).choice(len(tr), size=TRAIN_CAP, replace=False)
        tr = [tr[i] for i in sorted(idx)]
        print(f"[data] TRAIN capped to {TRAIN_CAP} (matched budget; fixed rng 20260702)", flush=True)
    if len(va) > VAL_CAP:
        idx = np.random.default_rng(20260703).choice(len(va), size=VAL_CAP, replace=False)
        va = [va[i] for i in sorted(idx)]
        print(f"[data] VAL capped to {VAL_CAP} (early-stop metric only; fixed rng 20260703)", flush=True)
    return tr, va, te


def seed_done(cdir: str, k: int) -> bool:
    return os.path.exists(os.path.join(cdir, f"seed{k}", "test_buckets.csv"))


# ------------------------------------------------------------------------------------------------ #
#  NODE-BUDGET BATCHING (VRAM guard).  The long-run datasets (standard track, our 12/24h) have       #
#  heavy-tailed graph sizes (std 2h: median 53 nodes, max 4464; our 12h: median 269, max 4984).      #
#  A fixed 64-GRAPH batch can randomly pack ~286k nodes -> >24 GB VRAM, which can crash the          #
#  driver ("CUDA unknown error" at ~24 GB).  Batches are therefore packed greedily IN THE SAME       #
#  seeded shuffle order up to a total-node budget (always >= 1 graph per batch).                     #
# ------------------------------------------------------------------------------------------------ #
NODE_BUDGET_FQ = 30_000        # ~= the 5-seed recipe's typical-heavy batch (64 x median-ish nodes)
NODE_BUDGET_LIGHT = 60_000     # rffgnn (no eigensolver/kriging -> much lighter per node)


def node_budget_batches(order, graphs, budget, max_graphs=64):
    batch, tot = [], 0
    for j in order:
        n = int(graphs[j].x.shape[0])
        if batch and (tot + n > budget or len(batch) >= max_graphs):
            yield batch
            batch, tot = [], 0
        batch.append(j); tot += n
    if batch:
        yield batch


def _blend_kw(m, b):
    """gap_minutes kwarg iff the model's Dmu source is the blend jet (FQDeepNative-only kwarg)."""
    if getattr(m, "dmu_source", "fd") != "blend":
        return {}
    return {"gap_minutes": torch.tensor([float(g.gap_minutes) for g in b], device=DEV)}


def eval_fq_budget(m, graphs, budget=NODE_BUDGET_FQ):
    """meltin_proof._eval semantics with node-budget batching; also returns per-gap error arrays
    in dataset order (for gap_row)."""
    m.eval(); fq, gc, errs = [], [], []
    with torch.no_grad():
        for bidx in node_budget_batches(range(len(graphs)), graphs, budget):
            b = [graphs[j] for j in bidx]
            x, coord, yp, mask, ei, ed, aei = collate_graphs_interp(b)
            pred = m(x.to(DEV), coord.to(DEV), ei.to(DEV), aei.to(DEV), _bvec(b),
                     **_blend_kw(m, b)).float().cpu()
            off = 0
            for g in b:
                ng = g.x.shape[0]; gp = pred[off:off + ng]; off += ng
                fe = model_errors(gp, g)
                errs.append(fe)
                if len(fe):
                    fq.append(float(np.median(fe))); gc.append(float(np.median(gcd_errors(g))))
    return np.array(fq), np.array(gc), errs


# ------------------------------------------------------------------------------------------------ #
#  Aggregates (rebuilt FROM DISK so resume includes previously-finished seeds) + status               #
# ------------------------------------------------------------------------------------------------ #
def _overall_rows(run_dir: str):
    """[(seed, {OVERALL row})] from every seed<k>/test_buckets.csv under run_dir."""
    out = []
    if not os.path.isdir(run_dir):
        return out
    for d in os.listdir(run_dir):
        if not (d.startswith("seed") and d[4:].isdigit()):
            continue
        p = os.path.join(run_dir, d, "test_buckets.csv")
        if not os.path.exists(p):
            continue
        with open(p, newline="") as f:
            rows = list(csv.DictReader(f))
        if rows and rows[0]["bucket"] == "OVERALL":
            out.append((int(d[4:]), rows[0]))
    return sorted(out)


def rebuild_cell_aggregates(cdir: str, model: str):
    """Re-emit all_seeds_epochs.csv / all_seeds_test_buckets.csv / summary.csv (+MEAN/STD) from the
    per-seed CSVs on disk (identical formats to the 5-seed runners)."""
    seeds = sorted(int(d[4:]) for d in os.listdir(cdir)
                   if d.startswith("seed") and d[4:].isdigit() and seed_done(cdir, int(d[4:])))
    all_ep, all_bk, summary = [], [], []
    for k in seeds:
        sdir = os.path.join(cdir, f"seed{k}")
        with open(os.path.join(sdir, "epochs.csv"), newline="") as f:
            ep = list(csv.DictReader(f))
        with open(os.path.join(sdir, "test_buckets.csv"), newline="") as f:
            bk = list(csv.DictReader(f))
        all_ep += [{"seed": k, **r} for r in ep]
        all_bk += [{"seed": k, **r} for r in bk]
        ov = bk[0]
        # best_val convention of each model's 5-seed runner: bigru = best val median-NM, else best val ratio
        best = min(float(r["val_median_nm" if model == "bigru" else "val_ratio"]) for r in ep)
        summary.append({"seed": k, "n_gaps": int(ov["n"]), "gcd_med": float(ov["gcd_med"]),
                        "model_med": float(ov["model_med"]), "ratio": float(ov["ratio"]),
                        "win_pct": float(ov["win_pct"]), "epochs_trained": len(ep),
                        "best_val": round(best, 4)})
    write_aggregates(cdir, all_ep, all_bk, summary)
    return summary


def _status_pairs(smoke: bool):
    if smoke:
        return [(SMOKE_TRACK, SMOKE_GAP)]
    return [(t, g) for t in ("our_processed", "standard_process") for g in TRACK_GAPS[t]]


def write_status(seeds_total: int, smoke: bool, running=None):
    """task_alpha/runs/gap_sweep/status.csv (status_smoke.csv in smoke mode), rewritten after every
    seed: track, gap_h, model, seeds_done, seeds_total, ratio (mean over done seeds), state."""
    os.makedirs(SWEEP_ROOT, exist_ok=True)
    rows = []
    for track, gap in _status_pairs(smoke):
        for model in MODELS:
            pre = PRE_DONE.get((track, gap), {}).get(model)
            if pre is not None:
                ov = _overall_rows(pre)
                rat = round(float(np.mean([float(r["ratio"]) for _, r in ov])), 4) if ov else ""
                rows.append({"track": track, "gap_h": gap, "model": model, "seeds_done": len(ov),
                             "seeds_total": len(ov), "ratio": rat, "state": "done-preexisting"})
                continue
            cdir = cell_dir(track, gap, model)
            ov = _overall_rows(cdir)
            done = [k for k, _ in ov if k < seeds_total]
            rat = round(float(np.mean([float(r["ratio"]) for k, r in ov if k < seeds_total])), 4) \
                if done else ""
            if running == (track, gap, model):
                state = "running"
            elif len(done) >= seeds_total:
                state = "done"
            elif not dataset_exists(track, gap):
                state = "dataset-missing"
            else:
                state = "pending"
            rows.append({"track": track, "gap_h": gap, "model": model, "seeds_done": len(done),
                         "seeds_total": seeds_total, "ratio": rat, "state": state})
    path = f"{SWEEP_ROOT}/status_smoke.csv" if smoke else f"{SWEEP_ROOT}/status.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["track", "gap_h", "model", "seeds_done", "seeds_total",
                                          "ratio", "state"])
        w.writeheader(); w.writerows(rows)


# ------------------------------------------------------------------------------------------------ #
#  FQGNN seed: EXACT run_full_faithful_5seed.run_seed recipe (+ per-epoch empty_cache defrag)         #
# ------------------------------------------------------------------------------------------------ #
def run_seed_fqgnn(seed, tr, va, te, sdir, n_epochs):
    os.makedirs(sdir, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    if NATIVE:   # Op-H-W: EXACT run_fqnative2_5seed ctor (only the readout differs from op-k)
        skw = {}
        if SOBOLEV:  # sobolev: per-track jet + bending backbone (+ raw-encoder arm if POOL_RAW)
            skw = dict(dmu_source=SOB_DMU, metric_backbone="sobolev2")
            if POOL_RAW:
                skw["pool_source"] = "raw"
        m = FQDeepNative(n=6, n_freqs=48, width=96, init_scale=1.84, seed=seed, in_feat=6,
                         layers=3, jacobi_R=4, oph_weighted=True, **skw).to(DEV)
    else:
        m = FQDeepFaithful(n=6, n_freqs=48, width=96, init_scale=1.84, seed=seed, in_feat=6,
                           layers=3, jacobi_R=4).to(DEV)
    m.use_cuda = True
    with torch.no_grad():
        m.s_raw.fill_(math.log(math.expm1(2.5)))
    b0 = collate_graphs_interp(tr[:4])
    m(b0[0].to(DEV), b0[1].to(DEV), b0[4].to(DEV), b0[6].to(DEV), _bvec(tr[:4]),
      **_blend_kw(m, tr[:4]))  # LazyLinear warmup
    init = {n: p.detach().clone() for n, p in m.named_parameters()}
    opt = torch.optim.Adam(m.parameters(), lr=5e-4)

    db = tr[:64]; dx, dco, dyp, dmk, dei, _, dae = collate_graphs_interp(db)
    DX, DCO, DEI, DAE, DBV = dx.to(DEV), dco.to(DEV), dei.to(DEV), dae.to(DEV), _bvec(db)
    DMK, DYT = dmk.to(DEV), dyp.to(DEV)

    DKW = _blend_kw(m, db)

    def telemetry():
        m.train(); opt.zero_grad()
        (((m(DX, DCO, DEI, DAE, DBV, **DKW)[DMK] - DYT[DMK]) ** 2).mean()).backward()
        gsq = {g: 0.0 for g in GROUPS}; dsq = {g: 0.0 for g in GROUPS}; psq = {g: 0.0 for g in GROUPS}
        for n, p in m.named_parameters():
            g = group_of(n)
            if g not in GROUPS:
                continue
            if p.grad is not None:
                gsq[g] += float(p.grad.norm()) ** 2
            dsq[g] += float((p.detach() - init[n]).norm()) ** 2; psq[g] += float(init[n].norm()) ** 2
        opt.zero_grad()
        with torch.no_grad():
            node, L, xi, c, s = m._feats(DX, DCO); h = m.enc(node); P = m._jacobi(xi)
            fa = float((m.theta[0](h[DEI[0]]) @ P.t()).abs().mean())
            fo = float((m.theta_out(h.mean(0, keepdim=True)) @ P.t() - 1).abs().mean())
        gn = {g: gsq[g] ** 0.5 for g in GROUPS}
        rc = {g: dsq[g] ** 0.5 / max(psq[g] ** 0.5, 1e-9) for g in GROUPS}
        return gn, rc, fa, fo

    fields = ["epoch", "train_loss", "val_ratio", "val_model_med", "val_gcd_med",
              "agg_filter_mag", "readout_cond_dev"] + \
             [f"grad_{g}" for g in GROUPS] + [f"chg_{g}" for g in GROUPS]
    rows = []
    best, best_state, pctr, bs = 1e9, None, 0, 64
    for ep in range(n_epochs):
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        m.train(); order = list(range(len(tr))); random.Random(seed * 1000 + ep).shuffle(order)
        ep_loss, ep_nb = 0.0, 0
        for bidx in node_budget_batches(order, tr, NODE_BUDGET_FQ, max_graphs=bs):
            b = [tr[j] for j in bidx]; x, co, yp, mk, ei, _, ae = collate_graphs_interp(b)
            pred = m(x.to(DEV), co.to(DEV), ei.to(DEV), ae.to(DEV), _bvec(b), **_blend_kw(m, b))
            L = ((pred[mk.to(DEV)] - yp.to(DEV)[mk.to(DEV)]) ** 2).mean()
            opt.zero_grad()
            if torch.isfinite(L):
                L.backward()
                gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                # finite-grad guard: with a non-finite grad norm, clip_grad_norm_
                # rescales EVERY grad to NaN and the step would poison the model (NaN params ->
                # non-finite B~ -> a failing pinv at the next forward).  Skipping the step leaves
                # finite-grad behavior BYTE-IDENTICAL (clip_coef unchanged).
                if torch.isfinite(gn):
                    opt.step()
                ep_loss += float(L.detach()); ep_nb += 1
        fq, gc, _ = eval_fq_budget(m, va)
        vm, vg = float(np.median(fq)), float(np.median(gc)); vr = vm / vg
        gn, rc, fa, fo = telemetry()
        rows.append({"epoch": ep, "train_loss": round(ep_loss / max(ep_nb, 1), 6),
                     "val_ratio": round(vr, 4), "val_model_med": round(vm, 4), "val_gcd_med": round(vg, 4),
                     "agg_filter_mag": round(fa, 5), "readout_cond_dev": round(fo, 5),
                     **{f"grad_{g}": round(gn[g], 6) for g in GROUPS},
                     **{f"chg_{g}": round(rc[g], 6) for g in GROUPS}})
        with open(f"{sdir}/epochs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
        if vr < best - 1e-4:
            best, best_state, pctr = vr, copy.deepcopy(m.state_dict()), 0
            torch.save({"state_dict": best_state}, f"{sdir}/best.pt")
        else:
            pctr += 1
        print(f"[fqgnn seed {seed}] ep {ep:2d}: train {ep_loss / max(ep_nb, 1):.5f} | val {vr:.4f} "
              f"(best {best:.4f}) | aggfilt {fa:.3f} readout_dev {fo:.4f} | {time.time() - t0:.1f}s "
              f"| peakVRAM {torch.cuda.max_memory_allocated() / 2**30:.1f}G",
              flush=True)
        torch.cuda.empty_cache()                              # defrag (run_ablation_ladder pattern)
        if pctr >= PATIENCE["fqgnn"]:
            print(f"[fqgnn seed {seed}] early stop", flush=True); break

    m.load_state_dict(best_state)
    _, _, te_errs = eval_fq_budget(m, te)
    pg = [gap_row(gi, g, fe) for gi, (g, fe) in enumerate(zip(te, te_errs)) if len(fe)]
    brows = bucket_rows(pg)
    write_per_gap(f"{sdir}/per_gap.csv", pg)
    write_buckets(f"{sdir}/test_buckets.csv", brows)
    _print_test("fqgnn", seed, brows)
    del m, opt, init, best_state
    torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------------ #
#  RFF-GNN seed: EXACT run_rffgnn_5seed_gaps recipe (class + eval imported from that file)            #
# ------------------------------------------------------------------------------------------------ #
def run_seed_rffgnn(seed, tr, va, te, sdir, n_epochs):
    os.makedirs(sdir, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    m = RFFMOD.RFFGNN(seed=seed).to(DEV)
    npar = sum(p.numel() for p in m.parameters() if p.requires_grad)
    nfroz = sum(b.numel() for b in m.buffers())
    print(f"[rffgnn seed {seed}] params {npar} (+{nfroz} FROZEN random RFF)", flush=True)
    opt = torch.optim.Adam(m.parameters(), lr=RFFMOD.LR)

    rows, best, best_state, pctr = [], 1e9, None, 0
    for ep in range(n_epochs):
        t0 = time.time()
        m.train(); order = list(range(len(tr))); random.Random(seed * 1000 + ep).shuffle(order)
        ep_loss, nb = 0.0, 0
        for bidx in node_budget_batches(order, tr, NODE_BUDGET_LIGHT, max_graphs=RFFMOD.BS):
            b = [tr[j] for j in bidx]
            x, co, yp, mk, ei, ed, ae = collate_graphs_interp(b)
            pred = m(x.to(DEV), co.to(DEV), ei.to(DEV), ae.to(DEV))
            L = ((pred[mk.to(DEV)] - yp.to(DEV)[mk.to(DEV)]) ** 2).mean()
            opt.zero_grad()
            if torch.isfinite(L):
                L.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
                ep_loss += float(L.detach()); nb += 1
        fq, gc, _ = RFFMOD.eval_gaps(m, va)
        vm, vg = float(np.median(fq)), float(np.median(gc)); vr = vm / vg
        rows.append({"epoch": ep, "train_loss": round(ep_loss / max(nb, 1), 6),
                     "val_ratio": round(vr, 4), "val_model_med": round(vm, 4), "val_gcd_med": round(vg, 4)})
        write_epochs(os.path.join(sdir, "epochs.csv"), rows)
        if vr < best - 1e-4:
            best, best_state, pctr = vr, copy.deepcopy(m.state_dict()), 0
            torch.save({"state_dict": best_state}, os.path.join(sdir, "best.pt"))
        else:
            pctr += 1
        print(f"[rffgnn seed {seed}] ep {ep:2d}: train {ep_loss / max(nb, 1):.5f} | val {vr:.4f} "
              f"(best {best:.4f}) | {time.time() - t0:.1f}s", flush=True)
        torch.cuda.empty_cache()                              # defrag (run_ablation_ladder pattern)
        if pctr >= PATIENCE["rffgnn"]:
            print(f"[rffgnn seed {seed}] early stop", flush=True); break

    m.load_state_dict(best_state)
    _, _, te_errs = RFFMOD.eval_gaps(m, te)
    pg = [gap_row(i, g, e) for i, (g, e) in enumerate(zip(te, te_errs)) if len(e)]
    write_per_gap(os.path.join(sdir, "per_gap.csv"), pg)
    brows = bucket_rows(pg)
    write_buckets(os.path.join(sdir, "test_buckets.csv"), brows)
    _print_test("rffgnn", seed, brows)
    del m, opt, best_state
    torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------------ #
#  Bi-GRU seed: EXACT run_bigru_5seed_gaps recipe (model + feature pipeline imported from that file)  #
# ------------------------------------------------------------------------------------------------ #
def run_seed_bigru(seed, tr_f, va_f, te_f, va_g, te_g, mean, std, val_gcd_med, sdir, n_epochs):
    os.makedirs(sdir, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    m = BGMOD.BiGRUImputer().to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)

    rows, best, best_state, pctr = [], 1e18, None, 0
    for ep in range(n_epochs):
        t0 = time.time()
        m.train()
        order = list(range(len(tr_f))); random.Random(seed * 1000 + ep).shuffle(order)
        ep_loss, nb = 0.0, 0
        for i0 in range(0, len(order), BGMOD.BATCH):
            batch = [tr_f[j] for j in order[i0:i0 + BGMOD.BATCH]]
            X, Y, H, V = BGMOD.collate(batch)
            X, Y, H = X.to(DEV), Y.to(DEV), H.to(DEV)
            pred = m(X, V.sum(1))
            loss = ((pred[H] - Y[H]) ** 2).mean()
            opt.zero_grad()
            if torch.isfinite(loss):
                loss.backward(); opt.step()
                ep_loss += float(loss.detach()); nb += 1
        preds = BGMOD.predict_held(m, va_f, mean, std)
        vmeds = [float(np.median(model_errors_latlon(g, la, lo)))
                 for g, (la, lo) in zip(va_g, preds)]
        vmed = float(np.median(vmeds)); vr = vmed / val_gcd_med
        rows.append({"epoch": ep, "train_loss": round(ep_loss / max(nb, 1), 6),
                     "val_median_nm": round(vmed, 4), "val_ratio": round(vr, 4)})
        write_epochs(os.path.join(sdir, "epochs.csv"), rows)
        if vmed < best - 1e-4:
            best, best_state, pctr = vmed, copy.deepcopy(m.state_dict()), 0
            torch.save({"state_dict": best_state}, os.path.join(sdir, "best.pt"))
        else:
            pctr += 1
        print(f"[bigru seed {seed}] ep {ep:3d}: train {ep_loss / max(nb, 1):.5f} | val {vmed:.3f} NM "
              f"(ratio {vr:.4f}, best {best:.3f}) | {time.time() - t0:.1f}s", flush=True)
        torch.cuda.empty_cache()                              # defrag (run_ablation_ladder pattern)
        if pctr >= PATIENCE["bigru"]:
            print(f"[bigru seed {seed}] early stop", flush=True)
            break

    m.load_state_dict(best_state)
    preds = BGMOD.predict_held(m, te_f, mean, std)
    pg = [gap_row(i, g, model_errors_latlon(g, la, lo))
          for i, (g, (la, lo)) in enumerate(zip(te_g, preds))]
    write_per_gap(os.path.join(sdir, "per_gap.csv"), pg)
    brows = bucket_rows(pg)
    write_buckets(os.path.join(sdir, "test_buckets.csv"), brows)
    _print_test("bigru", seed, brows)
    del m, opt, best_state
    torch.cuda.empty_cache()


def _print_test(model, seed, brows):
    ov = brows[0]
    print(f"[{model} seed {seed}][TEST] {ov['n']} gaps OVERALL ratio_med {ov['ratio']} "
          f"(GCD {ov['gcd_med']} model {ov['model_med']} win {ov['win_pct']}%)", flush=True)
    for r in brows[1:]:
        print(f"   {r['bucket']:<11} n={r['n']:4d}  GCD {r['gcd_med']}  model {r['model_med']}  "
              f"ratio {r['ratio']}", flush=True)


# ------------------------------------------------------------------------------------------------ #
#  One cell = (track, gap, model): seed-level resume + aggregates + status after every seed           #
# ------------------------------------------------------------------------------------------------ #
def run_cell(track, gap, model, seeds_total, smoke, epochs_override, data_cache):
    tag = cell_tag(track, gap, model)
    cdir = cell_dir(track, gap, model)

    pending = [k for k in range(seeds_total) if not seed_done(cdir, k)]
    if not pending:
        for k in range(seeds_total):
            print(f"RESUME-SKIP {tag} seed{k} (test_buckets.csv exists)", flush=True)
        os.makedirs(cdir, exist_ok=True)
        rebuild_cell_aggregates(cdir, model)
        write_status(seeds_total, smoke)
        print(f"[cell {tag}] already complete ({seeds_total}/{seeds_total} seeds)", flush=True)
        return True

    if not dataset_exists(track, gap):
        print(f"DATASET MISSING {data_dir(track, gap)}", flush=True)
        write_status(seeds_total, smoke)
        return False

    # ---- bigru: cuDNN can crash at DLL detach on Windows AFTER all work is done, so
    #      run the cell in a CHILD process; success = every seed's test_buckets.csv exists ----
    if model == "bigru" and os.environ.get(WORKER_ENV) != "1":
        cmd = [sys.executable, os.path.abspath(__file__), "--track", track, "--gap", str(gap),
               "--model", "bigru", "--seeds", str(seeds_total)]
        if smoke:
            cmd.append("--smoke")
        print(f"[cell {tag}] launching bigru worker child (cuDNN teardown-crash isolation)", flush=True)
        rc = subprocess.call(cmd, env=dict(os.environ, **{WORKER_ENV: "1"}), cwd=ROOT)
        ok = all(seed_done(cdir, k) for k in range(seeds_total))
        if ok and rc != 0:
            print(f"[cell {tag}] worker exit code {rc} occurred AFTER completion "
                  f"(known cuDNN teardown crash on this env) -- SUCCESS", flush=True)
        elif not ok:
            print(f"[cell {tag}] worker FAILED before completing all seeds (exit {rc})", flush=True)
        if os.path.isdir(cdir):
            rebuild_cell_aggregates(cdir, model)
        write_status(seeds_total, smoke)
        return ok

    os.makedirs(cdir, exist_ok=True)
    # ---- dataset (cached across the consecutive models of the same (track, gap)) ----
    key = (track, gap)
    if data_cache.get("key") != key:
        data_cache["key"], data_cache["val"] = key, load_dataset(track, gap)
    tr, va, te = data_cache["val"]

    n_epochs = epochs_override if epochs_override else EPOCHS[model]

    # ---- per-dataset preprocessing (train-split statistics of THIS dataset) ----
    if model == "bigru":
        mean, std = fit_latlon_stats(tr)                       # TRAIN-only z-score stats
        print(f"[bigru stats] mean={np.round(mean, 4).tolist()} std={np.round(std, 4).tolist()}", flush=True)
        tr_f = [BGMOD.make_features(g, mean, std) for g in tr]
        va_f = [BGMOD.make_features(g, mean, std) for g in va]
        te_f = [BGMOD.make_features(g, mean, std) for g in te]
        val_gcd_med = float(np.median([np.median(gcd_errors(g)) for g in va]))
        print(f"[bigru val] GCD median-of-medians = {val_gcd_med:.4f} NM", flush=True)

    for k in range(seeds_total):
        if seed_done(cdir, k):
            print(f"RESUME-SKIP {tag} seed{k} (test_buckets.csv exists)", flush=True)
            continue
        write_status(seeds_total, smoke, running=(track, gap, model))
        print(f"\n================ CELL {tag} | SEED {k} ================", flush=True)
        t0 = time.time()
        sdir = os.path.join(cdir, f"seed{k}")
        if model == "fqgnn":
            run_seed_fqgnn(k, tr, va, te, sdir, n_epochs)
        elif model == "rffgnn":
            run_seed_rffgnn(k, tr, va, te, sdir, n_epochs)
        else:
            run_seed_bigru(k, tr_f, va_f, te_f, va, te, mean, std, val_gcd_med, sdir, n_epochs)
        summary = rebuild_cell_aggregates(cdir, model)
        write_status(seeds_total, smoke, running=(track, gap, model))
        rat = [s["ratio"] for s in summary]
        print(f"[status] {tag}: {len(summary)}/{seeds_total} seeds done | ratio_mean "
              f"{np.mean(rat):.4f} | seed wall {time.time() - t0:.1f}s", flush=True)
    write_status(seeds_total, smoke)
    return all(seed_done(cdir, k) for k in range(seeds_total))


# ------------------------------------------------------------------------------------------------ #
#  CLI                                                                                               #
# ------------------------------------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser(description="RESUMABLE gap-length x track x model sweep (FQGNN paper)")
    ap.add_argument("--queue", action="store_true", help="run ALL pending cells sequentially")
    ap.add_argument("--native", action="store_true",
                    help="Op-H-W model (FQDeepNative, oph_weighted=True); fresh root "
                         "runs/gap_sweep_native; fqgnn cells only (baselines reused)")
    ap.add_argument("--track", help="our_processed | standard_process | _smoke")
    ap.add_argument("--gap", help="gap hours, e.g. 12 or 12h")
    ap.add_argument("--model", help="fqgnn | rffgnn | bigru")
    ap.add_argument("--seeds", type=int, default=5, help="seeds 0..N-1 per cell (default 5)")
    ap.add_argument("--smoke", action="store_true",
                    help="1 seed, 2 epochs; default cells = data/_smoke/gap_6h, all 3 models")
    args = ap.parse_args()
    if args.native and not NATIVE:
        _apply_native()
    if NATIVE:
        print(f"[native] Op-H-W mode: model FQDeepNative(oph_weighted=True), "
              f"root {SWEEP_ROOT}, models {MODELS}", flush=True)
    if OPM and SOBOLEV:
        print(f"[opm] FQ_READOUT_MODE=opm stack: root {SWEEP_ROOT} | "
              f"lam={os.environ.get('FQ_OPM_LAM', 'head')} "
              f"gamma={os.environ.get('FQ_OPM_GAMMA', '(raw dial)')} "
              f"vbar={os.environ.get('FQ_OPM_VBAR', '')} mbar={os.environ.get('FQ_OPM_MBAR', '')} "
              f"beta2={os.environ.get('FQ_OPM_BETA2', '')} "
              f"amp={os.environ.get('FQ_AMP_MODE', 'secular')}/"
              f"nmc={os.environ.get('FQ_AMP_NMC', '')}/refresh={os.environ.get('FQ_AMP_REFRESH', '0')} "
              f"| gamma'-law constants are GLOBAL BY DESIGN: same law in every cell, "
              f"no per-cell tuning", flush=True)

    seeds_total = 1 if args.smoke else args.seeds
    epochs_override = 2 if args.smoke else None

    if args.track or args.gap or args.model:
        if not (args.track and args.gap and args.model):
            ap.error("--track/--gap/--model must be given together (or use --queue / --smoke)")
        gap = int(str(args.gap).lower().rstrip("h"))
        track, model = args.track, args.model
        if model not in MODELS:
            ap.error(f"--model must be one of {MODELS}")
        if track != SMOKE_TRACK and (track not in TRACK_GAPS or gap not in TRACK_GAPS[track]):
            ap.error(f"unknown cell {track}/gap_{gap}h; tracks {TRACK_GAPS} (or _smoke)")
        if PRE_DONE.get((track, gap), {}).get(model):
            print(f"[cell {cell_tag(track, gap, model)}] ALREADY DONE (existing run: "
                  f"{PRE_DONE[(track, gap)][model]}) -- nothing to do", flush=True)
            return
        cells = [(track, gap, model)]
    elif args.queue:
        cells = [(t, g, m) for (t, g) in QUEUE_ORDER for m in MODELS]
    elif args.smoke:
        cells = [(SMOKE_TRACK, SMOKE_GAP, m) for m in MODELS]
    else:
        ap.error("choose a mode: --queue, --smoke, or --track/--gap/--model")

    print(f"[gap_sweep] {len(cells)} cell(s), {seeds_total} seed(s) each"
          + (" [SMOKE: 2 epochs]" if args.smoke else ""), flush=True)
    write_status(seeds_total, args.smoke)
    data_cache = {"key": None, "val": None}
    results = []
    for track, gap, model in cells:
        ok = run_cell(track, gap, model, seeds_total, args.smoke, epochs_override, data_cache)
        results.append((cell_tag(track, gap, model), ok))

    print("\n=== GAP-SWEEP QUEUE SUMMARY ===", flush=True)
    for tag, ok in results:
        print(f"  {tag:<45} {'DONE' if ok else 'SKIPPED/INCOMPLETE'}", flush=True)
    print(f"status file: {SWEEP_ROOT}/{'status_smoke.csv' if args.smoke else 'status.csv'}", flush=True)


if __name__ == "__main__":
    main()

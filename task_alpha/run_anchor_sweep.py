"""task_alpha.run_anchor_sweep -- RESUMABLE anchor-sparsity sweep (Fig 6b).

Cells = (aps, model), aps = anchors PER SIDE of the gap (n_anchor = 2*aps):
  aps      2 and 8  (datasets our_processed/gap_6h_aps{2,8}, built by build_anchor_datasets.py --
           SAME 6h gaps as the paper dataset, only anchor_edge_index differs)
  models   fqgnn / rffgnn / bigru        seeds 5 (0..4) per cell.
EXCEPTION: aps4 (n_anchor=8) is the main paper cell (fqgnn: runs/full_faithful_5seed, rffgnn:
baselines/RFFGNN/runs_5seed_gaps, bigru: baselines/BiGRU/runs_5seed_gaps) -- not run here;
assemble_anchor_sweep.py reads those results from their own run folders.

MACHINERY: everything training-related is IMPORTED from task_alpha.run_gap_sweep (single-sourced):
run_seed_fqgnn / run_seed_rffgnn / run_seed_bigru (the exact 5-seed recipes, incl. per-epoch
torch.cuda.empty_cache() + peakVRAM print for fqgnn), node_budget_batches VRAM guard,
rebuild_cell_aggregates / _overall_rows, RFFMOD / BGMOD baseline modules.  Note bigru never reads
anchor_edge_index (to_series sequence view), so its ratios should be ~flat across aps -- a built-in
control that the datasets differ ONLY in anchor wiring.

RESUME (seed level): outputs under task_alpha/runs/anchor_sweep/aps{2,8}/<model>/seed<k>/
  {epochs.csv, per_gap.csv, test_buckets.csv, best.pt} -- formats identical to the gap sweep.
  seed<k>/test_buckets.csv existing => RESUME-SKIP.  After every seed the cell aggregates are
  rebuilt FROM DISK and task_alpha/runs/anchor_sweep/status.csv is rewritten.  A killed queue
  relaunches and continues exactly where it stopped.  bigru cells run in a CHILD process (cuDNN
  can crash at process teardown with 0xC0000409 on Windows AFTER all work is done); success is
  judged by the per-seed test_buckets.csv files, not the exit code.

Run (gpubench env, any cwd -- the script chdirs to the repository root itself):
  full queue :  python task_alpha/run_anchor_sweep.py --queue
  one cell   :  ... run_anchor_sweep.py --aps 2 --model fqgnn --seeds 5
  smoke      :  ... run_anchor_sweep.py --smoke   (aps2 subset 256/64/64, 1 seed, 2 ep, all 3
                models, outputs under runs/anchor_sweep/_smoke/, status_smoke.csv)
  native     :  ... run_anchor_sweep.py --queue --native   (FQDeepNative Op-H-W; fqgnn only; root
                runs/anchor_sweep_native; aps4 = fqnative2_5seed)
  sobolev    :  FQ_ANCHOR_SOBOLEV=1 ... run_anchor_sweep.py --queue   (sobolev
                model: blend Dmu + omega^-2 backbone via run_gap_sweep's FQ_SWEEP_SOBOLEV block;
                fqgnn only; separate root runs/anchor_sweep_sobolev; aps4 = fqnative_sobolev_5seed;
                with FQ_READOUT_MODE=opm as well: root runs/anchor_sweep_sobolev_opm,
                aps4 = fqnative_opmstack_5seed)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))                     # task_alpha
ROOT = os.path.abspath(os.path.join(HERE, ".."))                      # repository root
os.chdir(ROOT)
sys.path.insert(0, ROOT)

# ---- SOBOLEV mode (FQ_ANCHOR_SOBOLEV=1): must be decided BEFORE the run_gap_sweep
# import below -- GS reads FQ_SWEEP_SOBOLEV at import time, and its SOBOLEV block supplies the EXACT
# sobolev machinery this runner then reuses verbatim: FQDeepNative ctor kwargs
# (dmu_source=FQ_SWEEP_DMU default "blend", metric_backbone="sobolev2"), the native model swap, and
# the gap_minutes plumbing (_blend_kw in every train/eval forward).  With the env unset this block
# does nothing. ----
ANCHOR_SOBOLEV = os.environ.get("FQ_ANCHOR_SOBOLEV") == "1"
if ANCHOR_SOBOLEV:
    os.environ["FQ_SWEEP_SOBOLEV"] = "1"

import numpy as np

import task_alpha.run_gap_sweep as GS                                  # the single-sourced machinery
from task_alpha.baselines.common_gaps import fit_latlon_stats
from task_alpha.data import RunGraph  # noqa: F401  (unpickling target)
from task_alpha.metrics import gcd_errors

SWEEP_ROOT = "task_alpha/runs/anchor_sweep"
DATA_ROOT = "task_alpha/data/our_processed"
MODELS = GS.MODELS                                                     # ("fqgnn", "rffgnn", "bigru")
APS_RUN = (2, 8)                                                       # cells this runner trains
APS_ALL = (2, 4, 8)                                                    # status rows (4 = pre-done)
APS_N = {2: 4, 4: 8, 8: 16}                                            # aps -> n_anchor
PRE_DONE_APS4 = {                                                      # the existing paper runs
    "fqgnn": "task_alpha/runs/full_faithful_5seed",
    "rffgnn": "task_alpha/baselines/RFFGNN/runs_5seed_gaps",
    "bigru": "task_alpha/baselines/BiGRU/runs_5seed_gaps",
}
WORKER_ENV = "ANCHOR_SWEEP_BIGRU_WORKER"
SMOKE_APS = 2
SMOKE_SUB = (256, 64, 64)                                              # train/val/test subset sizes

# ---- NATIVE mode (--native / FQ_SWEEP_NATIVE=1): the model swap itself lives in GS
# (_apply_native switches GS.run_seed_fqgnn to FQDeepNative(oph_weighted=True)); here we only
# repoint this runner's root/PRE_DONE and restrict to the fqgnn cells (baselines reused). ----
NATIVE = os.environ.get("FQ_SWEEP_NATIVE") == "1"


def _apply_native():
    global NATIVE, SWEEP_ROOT, MODELS, PRE_DONE_APS4
    GS._apply_native()
    NATIVE = True
    SWEEP_ROOT = "task_alpha/runs/anchor_sweep_native"
    MODELS = ("fqgnn",)
    PRE_DONE_APS4 = {"fqgnn": "task_alpha/runs/fqnative2_5seed"}


if NATIVE:
    _apply_native()

# ---- SOBOLEV mode root/PRE_DONE repoint (implies native; the model swap itself happened inside GS
# at import time via FQ_SWEEP_SOBOLEV -- see the env export above the GS import).  Separate root
# anchor_sweep_sobolev: the aps2/8 op-k (runs/anchor_sweep) and native (runs/anchor_sweep_native)
# results are not overwritten; aps4 = the sobolev paper run (fqnative_sobolev_5seed). ----
if ANCHOR_SOBOLEV:
    assert GS.SOBOLEV and GS.NATIVE, "run_gap_sweep did not enter sobolev mode (env export ordering)"
    NATIVE = True
    SWEEP_ROOT = "task_alpha/runs/anchor_sweep_sobolev"
    MODELS = ("fqgnn",)
    PRE_DONE_APS4 = {"fqgnn": "task_alpha/runs/fqnative_sobolev_5seed"}

# ---- OP-M STACK (FQ_READOUT_MODE=opm): the readout switch lives in
# FQDeepNative (env-read at construction, inside GS.run_seed_fqgnn's ctor); here only a SEPARATE
# root so the sobolev Op-H anchor sweep is never overwritten, and aps4 = the Op-M stack five-seed
# run.  Default and sobolev-non-opm invocations use the roots set above.
OPM = os.environ.get("FQ_READOUT_MODE") == "opm"
if OPM and ANCHOR_SOBOLEV:
    SWEEP_ROOT += "_opm"
    PRE_DONE_APS4 = {"fqgnn": "task_alpha/runs/fqnative_opmstack_5seed"}


# ------------------------------------------------------------------------------------------------ #
#  Paths / dataset IO                                                                                #
# ------------------------------------------------------------------------------------------------ #
def data_dir(aps: int) -> str:
    return f"{DATA_ROOT}/gap_6h_aps{aps}"


def cell_dir(aps: int, model: str, smoke: bool) -> str:
    return f"{SWEEP_ROOT}/{'_smoke/' if smoke else ''}aps{aps}/{model}"


def cell_tag(aps: int, model: str, smoke: bool) -> str:
    return f"{'_smoke/' if smoke else ''}aps{aps}/{model}"


def dataset_exists(aps: int) -> bool:
    d = data_dir(aps)
    return all(os.path.exists(os.path.join(d, f"{p}.pkl")) for p in ("train", "val", "test"))


def load_dataset(aps: int, smoke: bool):
    """(train, val, test) RunGraph lists; same matched-budget caps as the gap sweep (no-ops at the
    our-track 6h sizes 5350/635/774, kept for machinery identity).  Smoke: fixed head subsets."""
    d = data_dir(aps)
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
    TRAIN_CAP, VAL_CAP = 8000, 1000                    # gap-sweep matched budget (identical rngs)
    if len(tr) > TRAIN_CAP:
        idx = np.random.default_rng(20260702).choice(len(tr), size=TRAIN_CAP, replace=False)
        tr = [tr[i] for i in sorted(idx)]
        print(f"[data] TRAIN capped to {TRAIN_CAP} (matched budget; fixed rng 20260702)", flush=True)
    if len(va) > VAL_CAP:
        idx = np.random.default_rng(20260703).choice(len(va), size=VAL_CAP, replace=False)
        va = [va[i] for i in sorted(idx)]
        print(f"[data] VAL capped to {VAL_CAP} (early-stop metric only; fixed rng 20260703)", flush=True)
    if smoke:
        tr, va, te = tr[:SMOKE_SUB[0]], va[:SMOKE_SUB[1]], te[:SMOKE_SUB[2]]
        print(f"[data] SMOKE subset: {len(tr)}/{len(va)}/{len(te)}", flush=True)
    return tr, va, te


# ------------------------------------------------------------------------------------------------ #
#  status.csv (rewritten after every seed): aps 2/4/8 x model; aps4 = the main 5-seed paper runs     #
# ------------------------------------------------------------------------------------------------ #
def write_status(seeds_total: int, smoke: bool, running=None):
    os.makedirs(SWEEP_ROOT, exist_ok=True)
    rows = []
    for aps in ((SMOKE_APS,) if smoke else APS_ALL):
        for model in MODELS:
            if aps == 4 and not smoke:
                ov = GS._overall_rows(PRE_DONE_APS4[model])
                rat = round(float(np.mean([float(r["ratio"]) for _, r in ov])), 4) if ov else ""
                rows.append({"aps": aps, "n_anchor": APS_N[aps], "model": model,
                             "seeds_done": len(ov), "seeds_total": len(ov), "ratio": rat,
                             "state": "done-preexisting"})
                continue
            cdir = cell_dir(aps, model, smoke)
            ov = GS._overall_rows(cdir)
            done = [k for k, _ in ov if k < seeds_total]
            rat = round(float(np.mean([float(r["ratio"]) for k, r in ov if k < seeds_total])), 4) \
                if done else ""
            if running == (aps, model):
                state = "running"
            elif len(done) >= seeds_total:
                state = "done"
            elif not dataset_exists(aps):
                state = "dataset-missing"
            else:
                state = "pending"
            rows.append({"aps": aps, "n_anchor": APS_N[aps], "model": model,
                         "seeds_done": len(done), "seeds_total": seeds_total, "ratio": rat,
                         "state": state})
    path = f"{SWEEP_ROOT}/status_smoke.csv" if smoke else f"{SWEEP_ROOT}/status.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["aps", "n_anchor", "model", "seeds_done", "seeds_total",
                                          "ratio", "state"])
        w.writeheader(); w.writerows(rows)


# ------------------------------------------------------------------------------------------------ #
#  One cell = (aps, model): seed-level resume + aggregates + status after every seed                  #
#  (mirrors run_gap_sweep.run_cell; the seed functions themselves ARE run_gap_sweep's)                #
# ------------------------------------------------------------------------------------------------ #
def run_cell(aps, model, seeds_total, smoke, epochs_override, data_cache):
    tag = cell_tag(aps, model, smoke)
    cdir = cell_dir(aps, model, smoke)

    pending = [k for k in range(seeds_total) if not GS.seed_done(cdir, k)]
    if not pending:
        for k in range(seeds_total):
            print(f"RESUME-SKIP {tag} seed{k} (test_buckets.csv exists)", flush=True)
        os.makedirs(cdir, exist_ok=True)
        GS.rebuild_cell_aggregates(cdir, model)
        write_status(seeds_total, smoke)
        print(f"[cell {tag}] already complete ({seeds_total}/{seeds_total} seeds)", flush=True)
        return True

    if not dataset_exists(aps):
        print(f"DATASET MISSING {data_dir(aps)}", flush=True)
        write_status(seeds_total, smoke)
        return False

    # ---- bigru: cuDNN can crash at DLL detach on Windows AFTER all work is done, so
    #      run the cell in a CHILD process; success = every seed's test_buckets.csv exists ----
    if model == "bigru" and os.environ.get(WORKER_ENV) != "1":
        cmd = [sys.executable, os.path.abspath(__file__), "--aps", str(aps),
               "--model", "bigru", "--seeds", str(seeds_total)]
        if smoke:
            cmd.append("--smoke")
        print(f"[cell {tag}] launching bigru worker child (cuDNN teardown-crash isolation)", flush=True)
        rc = subprocess.call(cmd, env=dict(os.environ, **{WORKER_ENV: "1"}), cwd=ROOT)
        ok = all(GS.seed_done(cdir, k) for k in range(seeds_total))
        if ok and rc != 0:
            print(f"[cell {tag}] worker exit code {rc} occurred AFTER completion "
                  f"(known cuDNN teardown crash on this env) -- SUCCESS", flush=True)
        elif not ok:
            print(f"[cell {tag}] worker FAILED before completing all seeds (exit {rc})", flush=True)
        if os.path.isdir(cdir):
            GS.rebuild_cell_aggregates(cdir, model)
        write_status(seeds_total, smoke)
        return ok

    os.makedirs(cdir, exist_ok=True)
    # ---- dataset (cached across the consecutive models of the same aps) ----
    key = (aps, smoke)
    if data_cache.get("key") != key:
        data_cache["key"], data_cache["val"] = key, load_dataset(aps, smoke)
    tr, va, te = data_cache["val"]

    n_epochs = epochs_override if epochs_override else GS.EPOCHS[model]

    # ---- per-dataset preprocessing (train-split statistics of THIS dataset) ----
    if model == "bigru":
        mean, std = fit_latlon_stats(tr)                       # TRAIN-only z-score stats
        print(f"[bigru stats] mean={np.round(mean, 4).tolist()} std={np.round(std, 4).tolist()}", flush=True)
        tr_f = [GS.BGMOD.make_features(g, mean, std) for g in tr]
        va_f = [GS.BGMOD.make_features(g, mean, std) for g in va]
        te_f = [GS.BGMOD.make_features(g, mean, std) for g in te]
        val_gcd_med = float(np.median([np.median(gcd_errors(g)) for g in va]))
        print(f"[bigru val] GCD median-of-medians = {val_gcd_med:.4f} NM", flush=True)

    for k in range(seeds_total):
        if GS.seed_done(cdir, k):
            print(f"RESUME-SKIP {tag} seed{k} (test_buckets.csv exists)", flush=True)
            continue
        write_status(seeds_total, smoke, running=(aps, model))
        print(f"\n================ CELL {tag} | SEED {k} ================", flush=True)
        t0 = time.time()
        sdir = os.path.join(cdir, f"seed{k}")
        if model == "fqgnn":
            GS.run_seed_fqgnn(k, tr, va, te, sdir, n_epochs)
        elif model == "rffgnn":
            GS.run_seed_rffgnn(k, tr, va, te, sdir, n_epochs)
        else:
            GS.run_seed_bigru(k, tr_f, va_f, te_f, va, te, mean, std, val_gcd_med, sdir, n_epochs)
        summary = GS.rebuild_cell_aggregates(cdir, model)
        write_status(seeds_total, smoke, running=(aps, model))
        rat = [s["ratio"] for s in summary]
        print(f"[status] {tag}: {len(summary)}/{seeds_total} seeds done | ratio_mean "
              f"{np.mean(rat):.4f} | seed wall {time.time() - t0:.1f}s", flush=True)
    write_status(seeds_total, smoke)
    return all(GS.seed_done(cdir, k) for k in range(seeds_total))


# ------------------------------------------------------------------------------------------------ #
#  CLI                                                                                               #
# ------------------------------------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser(description="RESUMABLE anchor-sparsity x model sweep")
    ap.add_argument("--queue", action="store_true", help="run ALL pending cells sequentially")
    ap.add_argument("--native", action="store_true",
                    help="Op-H-W model (FQDeepNative); root runs/anchor_sweep_native; fqgnn only")
    ap.add_argument("--aps", type=int, help="anchors per side: 2 or 8 (4 = the main paper runs)")
    ap.add_argument("--model", help="fqgnn | rffgnn | bigru")
    ap.add_argument("--seeds", type=int, default=5, help="seeds 0..N-1 per cell (default 5)")
    ap.add_argument("--smoke", action="store_true",
                    help=f"1 seed, 2 epochs, aps{SMOKE_APS} subset {SMOKE_SUB}, all 3 models; "
                         f"outputs under {SWEEP_ROOT}/_smoke/")
    args = ap.parse_args()
    if args.native and not NATIVE:
        _apply_native()
    if NATIVE:
        print(f"[native] Op-H-W mode: root {SWEEP_ROOT}, models {MODELS}", flush=True)
    if OPM and ANCHOR_SOBOLEV:
        print(f"[opm] FQ_READOUT_MODE=opm stack: root {SWEEP_ROOT} | aps4 ref "
              f"{PRE_DONE_APS4['fqgnn']} | lam={os.environ.get('FQ_OPM_LAM', 'head')} "
              f"gamma={os.environ.get('FQ_OPM_GAMMA', '(raw dial)')} "
              f"amp={os.environ.get('FQ_AMP_MODE', 'secular')}/"
              f"nmc={os.environ.get('FQ_AMP_NMC', '')}/refresh={os.environ.get('FQ_AMP_REFRESH', '0')} "
              f"| gamma'-law constants (VBAR/MBAR/BETA2) are GLOBAL BY DESIGN: same law in every "
              f"cell, no per-cell tuning", flush=True)

    seeds_total = 1 if args.smoke else args.seeds
    epochs_override = 2 if args.smoke else None

    if args.aps is not None or args.model:
        if args.aps is None or not args.model:
            ap.error("--aps and --model must be given together (or use --queue / --smoke)")
        if args.model not in MODELS:
            ap.error(f"--model must be one of {MODELS}")
        if args.aps == 4 and not args.smoke:
            print(f"[cell aps4/{args.model}] ALREADY DONE (existing paper run: "
                  f"{PRE_DONE_APS4[args.model]}) -- nothing to do", flush=True)
            return
        if args.aps not in APS_RUN:
            ap.error(f"--aps must be one of {APS_RUN} (4 = the main paper runs)")
        cells = [(args.aps, args.model)]
    elif args.queue:
        cells = [(a, m) for a in APS_RUN for m in MODELS]
    elif args.smoke:
        cells = [(SMOKE_APS, m) for m in MODELS]
    else:
        ap.error("choose a mode: --queue, --smoke, or --aps/--model")

    print(f"[anchor_sweep] {len(cells)} cell(s), {seeds_total} seed(s) each"
          + (" [SMOKE: 2 epochs]" if args.smoke else ""), flush=True)
    write_status(seeds_total, args.smoke)
    data_cache = {"key": None, "val": None}
    results = []
    for aps, model in cells:
        ok = run_cell(aps, model, seeds_total, args.smoke, epochs_override, data_cache)
        results.append((cell_tag(aps, model, args.smoke), ok))

    print("\n=== ANCHOR-SWEEP QUEUE SUMMARY ===", flush=True)
    for tag, ok in results:
        print(f"  {tag:<45} {'DONE' if ok else 'SKIPPED/INCOMPLETE'}", flush=True)
    print(f"status file: {SWEEP_ROOT}/{'status_smoke.csv' if args.smoke else 'status.csv'}", flush=True)


if __name__ == "__main__":
    main()

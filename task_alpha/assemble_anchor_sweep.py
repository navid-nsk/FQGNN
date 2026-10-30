"""task_alpha.assemble_anchor_sweep -- collect the anchor-sparsity sweep (Fig 6b) into paper CSVs.

Pulls every finished seed of every (aps, model) cell from
  task_alpha/runs/anchor_sweep/aps{2,8}/<model>/seed<k>/test_buckets.csv   (OVERALL row)
PLUS the aps4 (n_anchor=8, the paper config) cells, read from the 5-seed runs
  fqgnn : task_alpha/runs/full_faithful_5seed
  rffgnn: task_alpha/baselines/RFFGNN/runs_5seed_gaps
  bigru : task_alpha/baselines/BiGRU/runs_5seed_gaps

Writes (plot-ready):
  <FQGNN_FIGDATA>/anchor_sweep.csv        LONG: aps, model, seed, ratio, model_med, gcd_med, win_pct
                                        (aps = anchors per side; n_anchor = 2*aps)
  <FQGNN_FIGDATA>/anchor_sweep_pivot.csv  PIVOT: one row per aps; per-model ratio mean/std/n columns
                                        + the (model-independent) gcd_med
Cells with no finished seeds are simply absent (noted on stdout).  Safe to re-run at any time,
also while sweep runs are still in progress -- it assembles whatever exists.

Run (gpubench env, any cwd):
  python task_alpha/assemble_anchor_sweep.py
  --smoke: assemble the runs/anchor_sweep/_smoke cells + the aps4 pulls into
           <FQGNN_FIGDATA>/anchor_sweep_smoke.csv / anchor_sweep_pivot_smoke.csv (smoke test only)
  --sobolev: fqgnn cells from runs/anchor_sweep_sobolev (aps2/aps8) with aps4 from
           runs/fqnative_sobolev_5seed; rffgnn/bigru sources unchanged
  --opm: fqgnn cells from runs/anchor_sweep_sobolev_opm (aps2/aps8) with aps4 from
           runs/fqnative_opmstack_5seed; per seed the official re-eval test_buckets_clean.csv
           is preferred over test_buckets.csv; rffgnn/bigru sources unchanged
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))                     # task_alpha
ROOT = os.path.abspath(os.path.join(HERE, ".."))                      # repository root
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np

SWEEP_ROOT = "task_alpha/runs/anchor_sweep"
SWEEP_ROOT_NATIVE = "task_alpha/runs/anchor_sweep_native"          # --native: fqgnn cells only
from task_alpha.paths import FIGDATA_DIR as FIGDATA  # noqa: E402  (env FQGNN_FIGDATA)
MODELS = ("fqgnn", "rffgnn", "bigru")
APS_ALL = (2, 4, 8)                                                    # anchors per side
APS_N = {2: 4, 4: 8, 8: 16}                                            # aps -> n_anchor
PRE_DONE_APS4 = {
    "fqgnn": "task_alpha/runs/full_faithful_5seed",
    "rffgnn": "task_alpha/baselines/RFFGNN/runs_5seed_gaps",
    "bigru": "task_alpha/baselines/BiGRU/runs_5seed_gaps",
}
# --native: the fqgnn aps4 cell is the native 5-seed reference run
PRE_DONE_APS4_NATIVE_FQGNN = "task_alpha/runs/fqnative2_5seed"
NATIVE = False
# --sobolev: fqgnn cells from the sobolev sweep; aps4 = the sobolev 5-seed reference run
SWEEP_ROOT_SOBOLEV = "task_alpha/runs/anchor_sweep_sobolev"
PRE_DONE_APS4_SOBOLEV_FQGNN = "task_alpha/runs/fqnative_sobolev_5seed"
SOBOLEV = False
# --opm: fqgnn cells from the Op-M stack sweep; aps4 = the Op-M stack 5-seed reference run
SWEEP_ROOT_OPM = "task_alpha/runs/anchor_sweep_sobolev_opm"
PRE_DONE_APS4_OPM_FQGNN = "task_alpha/runs/fqnative_opmstack_5seed"
OPM = False


def overall_rows(run_dir: str, clean: bool = False):
    """[(seed, OVERALL-row dict)] from seed<k>/test_buckets.csv under run_dir (sorted by seed).

    clean=True (--opm fqgnn cells): prefer the official exact-table re-eval file
    test_buckets_clean.csv per seed; fall back (loudly) to test_buckets.csv if absent."""
    out = []
    if not os.path.isdir(run_dir):
        return out
    for d in os.listdir(run_dir):
        if not (d.startswith("seed") and d[4:].isdigit()):
            continue
        p = os.path.join(run_dir, d, "test_buckets.csv")
        if clean:
            pc = os.path.join(run_dir, d, "test_buckets_clean.csv")
            if os.path.exists(pc):
                p = pc
            else:
                print(f"[assemble] WARNING: {pc} missing -- falling back to test_buckets.csv",
                      flush=True)
        if not os.path.exists(p):
            continue
        with open(p, newline="") as f:
            rows = list(csv.DictReader(f))
        if rows and rows[0]["bucket"] == "OVERALL":
            out.append((int(d[4:]), rows[0]))
    return sorted(out)


def cell_src(aps: int, model: str, smoke: bool) -> str:
    if OPM and model == "fqgnn":
        if aps == 4:
            return PRE_DONE_APS4_OPM_FQGNN
        return f"{SWEEP_ROOT_OPM}/{'_smoke/' if smoke else ''}aps{aps}/{model}"
    if SOBOLEV and model == "fqgnn":
        if aps == 4:
            return PRE_DONE_APS4_SOBOLEV_FQGNN
        return f"{SWEEP_ROOT_SOBOLEV}/{'_smoke/' if smoke else ''}aps{aps}/{model}"
    if NATIVE and model == "fqgnn":
        if aps == 4:
            return PRE_DONE_APS4_NATIVE_FQGNN
        return f"{SWEEP_ROOT_NATIVE}/{'_smoke/' if smoke else ''}aps{aps}/{model}"
    if aps == 4:                                       # the existing 5-seed runs (not re-run)
        return PRE_DONE_APS4[model]
    return f"{SWEEP_ROOT}/{'_smoke/' if smoke else ''}aps{aps}/{model}"


def main():
    global NATIVE, SOBOLEV, OPM
    ap = argparse.ArgumentParser(description="assemble anchor-sweep results into FQGNN_FIGDATA CSVs")
    ap.add_argument("--smoke", action="store_true",
                    help="assemble the _smoke cells (+ real aps4 pulls) into *_smoke.csv")
    ap.add_argument("--native", action="store_true",
                    help="fqgnn rows from the native Op-H-W runs (anchor_sweep_native + "
                         "fqnative2_5seed); rffgnn/bigru unchanged")
    ap.add_argument("--sobolev", action="store_true",
                    help="fqgnn rows from the sobolev runs (anchor_sweep_sobolev + "
                         "fqnative_sobolev_5seed); rffgnn/bigru unchanged")
    ap.add_argument("--opm", action="store_true",
                    help="fqgnn rows from the Op-M stack runs (anchor_sweep_sobolev_opm + "
                         "fqnative_opmstack_5seed), preferring test_buckets_clean.csv; "
                         "rffgnn/bigru unchanged")
    args = ap.parse_args()
    if args.native and args.sobolev:
        ap.error("--native and --sobolev are mutually exclusive")
    if args.opm and (args.native or args.sobolev):
        ap.error("--opm is mutually exclusive with --native/--sobolev")
    if args.native:
        NATIVE = True
        print("[assemble] --native: fqgnn <- anchor_sweep_native / fqnative2_5seed; "
              "other models unchanged", flush=True)
    if args.sobolev:
        SOBOLEV = True
        print("[assemble] --sobolev: fqgnn <- anchor_sweep_sobolev / fqnative_sobolev_5seed; "
              "other models unchanged", flush=True)
    if args.opm:
        OPM = True
        print("[assemble] --opm: fqgnn <- anchor_sweep_sobolev_opm / fqnative_opmstack_5seed "
              "(clean files preferred); other models unchanged", flush=True)

    aps_list = (2, 4) if args.smoke else APS_ALL       # smoke runner only trains aps2 cells
    sfx = "_smoke" if args.smoke else ""
    long_path = f"{FIGDATA}/anchor_sweep{sfx}.csv"
    pivot_path = f"{FIGDATA}/anchor_sweep_pivot{sfx}.csv"

    long_rows = []
    for aps in aps_list:
        for model in MODELS:
            src = cell_src(aps, model, args.smoke)
            ov = overall_rows(src, clean=OPM and model == "fqgnn")
            if not ov:
                print(f"[assemble] aps{aps}/{model}: no finished seeds ({src}) -- skipped", flush=True)
                continue
            for k, r in ov:
                long_rows.append({"aps": aps, "model": model, "seed": k,
                                  "ratio": float(r["ratio"]), "model_med": float(r["model_med"]),
                                  "gcd_med": float(r["gcd_med"]), "win_pct": float(r["win_pct"])})
            print(f"[assemble] aps{aps}/{model}: {len(ov)} seed(s) from {src}", flush=True)

    if not long_rows:
        print("[assemble] nothing to assemble yet", flush=True)
        return
    os.makedirs(FIGDATA, exist_ok=True)
    with open(long_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["aps", "model", "seed", "ratio", "model_med",
                                          "gcd_med", "win_pct"])
        w.writeheader(); w.writerows(long_rows)

    # ---- pivot: one row per aps; per-model ratio mean/std/n + shared gcd_med ----
    pivot = []
    for aps in aps_list:
        cell = [r for r in long_rows if r["aps"] == aps]
        if not cell:
            continue
        row = {"aps": aps, "n_anchor": APS_N[aps],
               "gcd_med": round(float(np.mean([r["gcd_med"] for r in cell])), 3)}
        for model in MODELS:
            rat = np.array([r["ratio"] for r in cell if r["model"] == model], float)
            row[f"{model}_ratio_mean"] = round(float(rat.mean()), 4) if len(rat) else ""
            row[f"{model}_ratio_std"] = round(float(rat.std(ddof=0)), 4) if len(rat) else ""
            row[f"{model}_n_seeds"] = len(rat)
        pivot.append(row)
    pfields = ["aps", "n_anchor", "gcd_med"] + \
              [f"{m}_{s}" for m in MODELS for s in ("ratio_mean", "ratio_std", "n_seeds")]
    with open(pivot_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pfields)
        w.writeheader(); w.writerows(pivot)

    print(f"\n[assemble] {len(long_rows)} long rows -> {long_path}", flush=True)
    print(f"[assemble] {len(pivot)} pivot rows -> {pivot_path}", flush=True)
    for row in pivot:
        cols = "  ".join(f"{m} {row[f'{m}_ratio_mean']}+/-{row[f'{m}_ratio_std']}"
                         f" (n={row[f'{m}_n_seeds']})" for m in MODELS if row[f"{m}_n_seeds"])
        print(f"  aps{row['aps']} (n_anchor={row['n_anchor']})  GCD {row['gcd_med']} NM | {cols}",
              flush=True)


if __name__ == "__main__":
    main()

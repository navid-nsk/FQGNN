"""task_alpha.assemble_gap_sweep -- collect the gap-length x track x model sweep into paper CSVs.

Pulls every finished seed of every cell from
  task_alpha/runs/gap_sweep/<track>/gap_<G>h/<model>/seed<k>/test_buckets.csv  (OVERALL row)
PLUS the three our_processed/gap_6h cells that live outside the sweep root
  fqgnn : task_alpha/runs/full_faithful_5seed
  rffgnn: task_alpha/baselines/RFFGNN/runs_5seed_gaps
  bigru : task_alpha/baselines/BiGRU/runs_5seed_gaps

Writes (plot-ready):
  <FQGNN_FIGDATA>/gap_sweep.csv        LONG: track, gap_h, model, seed, ratio, model_med, gcd_med,
                                     win_pct  (the GCD baseline row is implicit via gcd_med)
  <FQGNN_FIGDATA>/gap_sweep_pivot.csv  PIVOT: one row per track x gap; per-model ratio mean/std/n
                                     columns + the (model-independent) gcd_med
Cells with no finished seeds are simply absent (noted on stdout).  Safe to re-run at any time,
also while sweep runs are still in progress -- it assembles whatever exists.

Run (gpubench env, any cwd):
  python task_alpha/assemble_gap_sweep.py
  DEFAULT (the sobolev model): fqgnn rows come from the SOBOLEV runs --
            our_processed cells from task_alpha/runs/gap_sweep_sobolev (blend D-mu jet),
            standard_process cells from task_alpha/runs/gap_sweep_sobolev_fd (fd jet on that
            track, by design; see run_gap_sweep.py FQ_SWEEP_DMU), and the
            pre-done our/6h cell from task_alpha/runs/fqnative_sobolev_5seed.  ALL OTHER MODELS
            are read from the task_alpha/runs/gap_sweep root + the PRE_DONE paths listed above.
  --smoke: assemble the data/_smoke cells + the our/6h pulls into
           <FQGNN_FIGDATA>/gap_sweep_smoke.csv / gap_sweep_pivot_smoke.csv (smoke test only)
  --native: fqgnn rows come from the NATIVE (FQDeepNative Op-H, fd D-mu) runs instead --
            sweep cells from task_alpha/runs/gap_sweep_native/... and the pre-done our/6h cell
            from task_alpha/runs/fqnative2_5seed.
  --opk:    fqgnn rows come from the Op-K runs (task_alpha/runs/gap_sweep + full_faithful_5seed).
  --opm:    fqgnn rows come from the Op-M stack runs --
            our_processed cells from task_alpha/runs/gap_sweep_sobolev_opm,
            standard_process cells from task_alpha/runs/gap_sweep_sobolev_fd_opm (fd jet on
            that track, as in the sobolev default), and the pre-done
            our/6h cell from task_alpha/runs/fqnative_opmstack_5seed.  FQGNN seed dirs are
            read via the OFFICIAL exact-table re-eval file test_buckets_clean.csv whenever it
            exists (loud warning + in-runner fallback if not).  ALL OTHER MODELS keep the
            sources of the default mode.
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

SWEEP_ROOT = "task_alpha/runs/gap_sweep"
SWEEP_ROOT_NATIVE = "task_alpha/runs/gap_sweep_native"             # --native: fqgnn cells only
# default (sobolev model): per-track fqgnn roots -- the standard track deliberately uses
# the fd boundary jet (run_gap_sweep.py FQ_SWEEP_DMU rationale; _smoke lives in the blend root)
SWEEP_ROOT_SOBOLEV = {"our_processed": "task_alpha/runs/gap_sweep_sobolev",
                      "standard_process": "task_alpha/runs/gap_sweep_sobolev_fd",
                      "_smoke": "task_alpha/runs/gap_sweep_sobolev"}
# --opm (Op-M stack): per-track fqgnn roots -- same fd-on-standard convention as sobolev
SWEEP_ROOT_OPM = {"our_processed": "task_alpha/runs/gap_sweep_sobolev_opm",
                  "standard_process": "task_alpha/runs/gap_sweep_sobolev_fd_opm",
                  "_smoke": "task_alpha/runs/gap_sweep_sobolev_opm"}
from task_alpha.paths import FIGDATA_DIR as FIGDATA  # noqa: E402  (env FQGNN_FIGDATA)
# habit: deterministic route lookup -> seed0 only (same convention as GCD / BF-BiLSTM).
# mhgin: 5 seeds.  multitaskais: 5 seeds, ONE-SIDED reference (forward VRNN, prior sampling --
# reported as a separate reference; captions/labels must carry the one-sided caveat).
# hermite / ctrv_kalman / quintic / constrained_gp: the four CLASSICAL baselines
# -- deterministic, seed0 only, always read from the op-k sweep root
# (task_alpha/runs/gap_sweep/<track>/gap_<G>h/<model>/; they are FQ-readout-independent).
MODELS = ("fqgnn", "rffgnn", "bigru", "habit", "mhgin", "multitaskais",
          "hermite", "ctrv_kalman", "quintic", "constrained_gp")
TRACK_GAPS = {"our_processed": (3, 6, 12, 24), "standard_process": (2, 6, 12, 24)}
PRE_DONE = {("our_processed", 6): {
    "fqgnn": "task_alpha/runs/full_faithful_5seed",
    "rffgnn": "task_alpha/baselines/RFFGNN/runs_5seed_gaps",
    "bigru": "task_alpha/baselines/BiGRU/runs_5seed_gaps",
}}
# --native: the fqgnn pre-done our/6h cell is the native 5-seed reference run
PRE_DONE_NATIVE_FQGNN = {("our_processed", 6): "task_alpha/runs/fqnative2_5seed"}
# default (sobolev): the fqgnn pre-done our/6h cell is the sobolev 5-seed reference run
PRE_DONE_SOBOLEV_FQGNN = {("our_processed", 6): "task_alpha/runs/fqnative_sobolev_5seed"}
# --opm: the fqgnn pre-done our/6h cell is the Op-M stack 5-seed headline run
PRE_DONE_OPM_FQGNN = {("our_processed", 6): "task_alpha/runs/fqnative_opmstack_5seed"}
SMOKE_TRACK, SMOKE_GAP = "_smoke", 6


def overall_rows(run_dir: str, clean: bool = False):
    """[(seed, OVERALL-row dict)] from seed<k>/test_buckets.csv under run_dir (sorted by seed).

    clean=True (--opm fqgnn cells): prefer seed<k>/test_buckets_clean.csv (the OFFICIAL
    exact-table re-eval); fall back to the in-runner test_buckets.csv only if the clean file
    is absent, with a loud warning."""
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
            elif os.path.exists(p):
                print(f"[assemble] WARNING: {pc} missing -- falling back to the in-runner "
                      f"test_buckets.csv (NOT the official clean re-eval)", flush=True)
        if not os.path.exists(p):
            continue
        with open(p, newline="") as f:
            rows = list(csv.DictReader(f))
        if rows and rows[0]["bucket"] == "OVERALL":
            out.append((int(d[4:]), rows[0]))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description="assemble gap-sweep results into FQGNN_FIGDATA CSVs")
    ap.add_argument("--smoke", action="store_true",
                    help="assemble the _smoke cells (+ real our/6h pulls) into *_smoke.csv")
    ap.add_argument("--native", action="store_true",
                    help="fqgnn rows from the native Op-H-W runs (gap_sweep_native + fqnative2_5seed); "
                         "all other models unchanged")
    ap.add_argument("--opk", action="store_true",
                    help="fqgnn rows from the Op-K runs (gap_sweep + full_faithful_5seed); "
                         "all other models unchanged")
    ap.add_argument("--opm", action="store_true",
                    help="fqgnn rows from the Op-M stack runs (gap_sweep_sobolev_opm / "
                         "gap_sweep_sobolev_fd_opm + fqnative_opmstack_5seed, reading the "
                         "official *_clean.csv re-eval files); all other models unchanged")
    args = ap.parse_args()
    if args.native and args.opk:
        ap.error("--native and --opk are mutually exclusive")
    if args.opm and (args.native or args.opk):
        ap.error("--opm is mutually exclusive with --native/--opk")
    if args.native:
        print("[assemble] --native: fqgnn <- gap_sweep_native / fqnative2_5seed; "
              "all other models <- default sources", flush=True)
    elif args.opk:
        print("[assemble] --opk: fqgnn <- gap_sweep / full_faithful_5seed (Op-K); "
              "all other models <- default sources", flush=True)
    elif args.opm:
        print("[assemble] --opm: fqgnn <- gap_sweep_sobolev_opm (our) / "
              "gap_sweep_sobolev_fd_opm (standard) / fqnative_opmstack_5seed (our-6h), "
              "*_clean.csv re-eval files; all other models <- default sources", flush=True)
    else:
        print("[assemble] default (sobolev): fqgnn <- gap_sweep_sobolev (our) / "
              "gap_sweep_sobolev_fd (standard) / fqnative_sobolev_5seed (our-6h); "
              "all other models <- default sources", flush=True)

    if args.smoke:
        pairs = [(SMOKE_TRACK, SMOKE_GAP), ("our_processed", 6)]
        long_path = f"{FIGDATA}/gap_sweep_smoke.csv"
        pivot_path = f"{FIGDATA}/gap_sweep_pivot_smoke.csv"
    else:
        pairs = [(t, g) for t in ("our_processed", "standard_process") for g in TRACK_GAPS[t]]
        long_path = f"{FIGDATA}/gap_sweep.csv"
        pivot_path = f"{FIGDATA}/gap_sweep_pivot.csv"

    long_rows = []
    for track, gap in pairs:
        for model in MODELS:
            if args.native and model == "fqgnn":
                src = PRE_DONE_NATIVE_FQGNN.get((track, gap)) \
                    or f"{SWEEP_ROOT_NATIVE}/{track}/gap_{gap}h/{model}"
            elif args.opm and model == "fqgnn":
                src = PRE_DONE_OPM_FQGNN.get((track, gap)) \
                    or f"{SWEEP_ROOT_OPM[track]}/{track}/gap_{gap}h/{model}"
            elif not args.opk and model == "fqgnn":                    # default: sobolev
                src = PRE_DONE_SOBOLEV_FQGNN.get((track, gap)) \
                    or f"{SWEEP_ROOT_SOBOLEV[track]}/{track}/gap_{gap}h/{model}"
            else:
                src = PRE_DONE.get((track, gap), {}).get(model) \
                    or f"{SWEEP_ROOT}/{track}/gap_{gap}h/{model}"
            ov = overall_rows(src, clean=args.opm and model == "fqgnn")
            if not ov:
                print(f"[assemble] {track}/gap_{gap}h/{model}: no finished seeds ({src}) -- skipped",
                      flush=True)
                continue
            for k, r in ov:
                long_rows.append({"track": track, "gap_h": gap, "model": model, "seed": k,
                                  "ratio": float(r["ratio"]), "model_med": float(r["model_med"]),
                                  "gcd_med": float(r["gcd_med"]), "win_pct": float(r["win_pct"])})
            print(f"[assemble] {track}/gap_{gap}h/{model}: {len(ov)} seed(s) from {src}", flush=True)

    if not long_rows:
        print("[assemble] nothing to assemble yet", flush=True)
        return
    os.makedirs(FIGDATA, exist_ok=True)
    with open(long_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["track", "gap_h", "model", "seed", "ratio", "model_med",
                                          "gcd_med", "win_pct"])
        w.writeheader(); w.writerows(long_rows)

    # ---- pivot: one row per track x gap, per-model ratio mean/std/n + shared gcd_med ----
    pivot = []
    for track, gap in pairs:
        cell = [r for r in long_rows if r["track"] == track and r["gap_h"] == gap]
        if not cell:
            continue
        row = {"track": track, "gap_h": gap,
               "gcd_med": round(float(np.mean([r["gcd_med"] for r in cell])), 3)}
        for model in MODELS:
            rat = np.array([r["ratio"] for r in cell if r["model"] == model], float)
            row[f"{model}_ratio_mean"] = round(float(rat.mean()), 4) if len(rat) else ""
            row[f"{model}_ratio_std"] = round(float(rat.std(ddof=0)), 4) if len(rat) else ""
            row[f"{model}_n_seeds"] = len(rat)
        pivot.append(row)
    pfields = ["track", "gap_h", "gcd_med"] + \
              [f"{m}_{s}" for m in MODELS for s in ("ratio_mean", "ratio_std", "n_seeds")]
    with open(pivot_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pfields)
        w.writeheader(); w.writerows(pivot)

    print(f"\n[assemble] {len(long_rows)} long rows -> {long_path}", flush=True)
    print(f"[assemble] {len(pivot)} pivot rows -> {pivot_path}", flush=True)
    for row in pivot:
        cols = "  ".join(f"{m} {row[f'{m}_ratio_mean']}+/-{row[f'{m}_ratio_std']}"
                         f" (n={row[f'{m}_n_seeds']})" for m in MODELS if row[f"{m}_n_seeds"])
        print(f"  {row['track']}/gap_{row['gap_h']}h  GCD {row['gcd_med']} NM | {cols}", flush=True)


if __name__ == "__main__":
    main()

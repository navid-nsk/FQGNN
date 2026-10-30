"""Runner: MultitaskAIS (Nguyen et al., VRNN Embedding block) across the two-track gap-sweep cells,
5 seeds per cell -- reported as a SEPARATE ONE-SIDED REFERENCE, never as a two-sided peer.

Three-stage pipeline (two python environments -- hard architecture constraint):
  Stage 1  mtais_export.py   gpubench   RunGraph pkls -> vendor-format voyage pickles (+mean/meta/
                                        test_index).  Observed pings only; caps identical to
                                        run_gap_sweep.load_dataset.
  Stage 2  mtais_tf1_run.py  mtais_tf1  (py3.7 / TF1.15 CPU / dm-sonnet 1.36) vendor ELBO training
                                        + vendor traj_reconstruction generalized in 3 documented
                                        ways; writes predictions.npz + loss_curve.csv per seed.
  Stage 3  mtais_score.py    gpubench   predictions -> common_gaps CSVs + cell aggregates
                                        (MHGIN schema) under runs/gap_sweep/<track>/gap_<G>h/
                                        multitaskais/seed<k>/.

All stages run as subprocesses (memory returns to the OS between cells, as in the HABIT runner).
Stage 2 is invoked via `conda run --no-capture-output -n mtais_tf1 python -u ...` (direct
python.exe in that env fails with an SSL error).

RESUMABLE at seed level: seed done <=> seed<k>/test_buckets.csv exists.  Stage-1 exports and
stage-2 predictions are also skipped when their outputs exist, so a killed queue relaunches and
continues where it stopped.

TRAINING BUDGET (documented): min(20 epochs, 30000 steps) at the vendor batch 32 -- with the capped
exports (<= ~16k voyages) 20 epochs is always binding (~2.5k-10k steps).  Vendor default is
max_steps=80000 with no other stopping rule; no early stopping and no validation-based selection
exist in the vendor trainer and none were added.  CPU-only; runs concurrently with GPU jobs.

  python -m task_alpha.baselines.MTAIS.run_mtais_gaps --queue
  python -m task_alpha.baselines.MTAIS.run_mtais_gaps --track standard_process --gap 24 --seeds 0 1
  python -m task_alpha.baselines.MTAIS.run_mtais_gaps --smoke
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

DATA_ROOT = "task_alpha/data"
SWEEP_ROOT = "task_alpha/runs/gap_sweep"
EXPORT_ROOT = "task_alpha/baselines/MTAIS/_export"
MODEL_NAME = "multitaskais"
TRACK_GAPS = {"our_processed": (3, 6, 12, 24), "standard_process": (2, 6, 12, 24)}
# cheapest-first order (early signal), mirroring run_gap_sweep.QUEUE_ORDER + our 6h cell
QUEUE_ORDER = (("standard_process", 2), ("our_processed", 3), ("our_processed", 6),
               ("standard_process", 6), ("our_processed", 12), ("standard_process", 12),
               ("our_processed", 24), ("standard_process", 24))

import shutil

# conda executable: env MTAIS_CONDA, else the active conda (CONDA_EXE), else the one on PATH
CONDA = (os.environ.get("MTAIS_CONDA") or os.environ.get("CONDA_EXE")
         or shutil.which("conda") or "conda")
TF1_ENV = "mtais_tf1"

# stage-2 recipe (see mtais_tf1_run.py docstring for provenance)
EPOCHS, MAX_STEPS, BATCH, LATENT, NSAMP, LR = 20, 30000, 32, 100, 16, 3e-4
SMOKE = dict(track="standard_process", gap=24, seeds=[0], epochs=2, train_segments_cap=300,
             test_cap=120, suffix="_smoke", model_name=MODEL_NAME + "_smoke")


def export_done(export_dir):
    return all(os.path.exists(os.path.join(export_dir, f)) for f in
               ("train.pkl", "valid.pkl", "mean.pkl", "test_index.pkl", "meta.json"))


def seed_done(cdir, k):
    return os.path.exists(os.path.join(cdir, f"seed{k}", "test_buckets.csv"))


def _run(cmd, tag):
    print(f"[{MODEL_NAME}] RUN {tag}: {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.check_call([str(c) for c in cmd])


def run_cell(track, gap, seeds, epochs=EPOCHS, suffix="", train_segments_cap=0, test_cap=0,
             model_name=MODEL_NAME):
    if not all(os.path.exists(f"{DATA_ROOT}/{track}/gap_{gap}h/{p}.pkl")
               for p in ("train", "val", "test")):
        print(f"[{MODEL_NAME}] DATASET MISSING {DATA_ROOT}/{track}/gap_{gap}h -- cell skipped", flush=True)
        return
    export_dir = f"{EXPORT_ROOT}/{track}/gap_{gap}h{suffix}"
    cdir = f"{SWEEP_ROOT}/{track}/gap_{gap}h/{model_name}"

    if export_done(export_dir):
        print(f"[{MODEL_NAME}] export exists: {export_dir} (RESUME-SKIP stage 1)", flush=True)
    else:
        cmd = [sys.executable, "-u", "-m", "task_alpha.baselines.MTAIS.mtais_export",
               "--track", track, "--gap", gap]
        if suffix:
            cmd += ["--suffix", suffix]
        if train_segments_cap:
            cmd += ["--train-segments-cap", train_segments_cap]
        _run(cmd, f"stage1 {track}/gap_{gap}h{suffix}")

    for k in seeds:
        if seed_done(cdir, k):
            print(f"[{MODEL_NAME}] {track}/gap_{gap}h seed{k} done (RESUME-SKIP)", flush=True)
            continue
        t0 = time.time()
        sdir = os.path.abspath(os.path.join(cdir, f"seed{k}"))
        os.makedirs(sdir, exist_ok=True)
        if not os.path.exists(os.path.join(sdir, "predictions.npz")):
            cmd = [CONDA, "run", "--no-capture-output", "-n", TF1_ENV, "python", "-u",
                   os.path.join(HERE, "mtais_tf1_run.py"),
                   "--export-dir", os.path.abspath(export_dir), "--out-dir", sdir,
                   "--seed", k, "--epochs", epochs, "--max-steps", MAX_STEPS,
                   "--batch-size", BATCH, "--latent-size", LATENT,
                   "--num-samples", NSAMP, "--lr", LR]
            if test_cap:
                cmd += ["--test-cap", test_cap]
            _run(cmd, f"stage2 {track}/gap_{gap}h seed{k}")
        else:
            print(f"[{MODEL_NAME}] predictions exist for seed{k} (RESUME-SKIP stage 2)", flush=True)
        cmd = [sys.executable, "-u", "-m", "task_alpha.baselines.MTAIS.mtais_score",
               "--track", track, "--gap", gap, "--seed", k, "--model-name", model_name]
        if suffix:
            cmd += ["--suffix", suffix]
        _run(cmd, f"stage3 {track}/gap_{gap}h seed{k}")
        print(f"[{MODEL_NAME}] {track}/gap_{gap}h seed{k} complete in {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description="MultitaskAIS one-sided VRNN baseline (vanilla)")
    ap.add_argument("--queue", action="store_true", help="all 8 cells, 5 seeds, resumable")
    ap.add_argument("--track")
    ap.add_argument("--gap", type=int)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--smoke", action="store_true",
                    help="standard_process/gap_24h, 300 train segments, 2 epochs, 1 seed, "
                         "120 test gaps, _smoke-suffixed outputs")
    a = ap.parse_args()

    if a.smoke:
        s = SMOKE
        run_cell(s["track"], s["gap"], s["seeds"], epochs=s["epochs"], suffix=s["suffix"],
                 train_segments_cap=s["train_segments_cap"], test_cap=s["test_cap"],
                 model_name=s["model_name"])
        return
    if a.queue:
        cells = list(QUEUE_ORDER)
    elif a.track and a.gap:
        cells = [(a.track, a.gap)]
    else:
        ap.error("choose --queue, --smoke, or --track/--gap")
    for track, gap in cells:
        run_cell(track, gap, a.seeds)


if __name__ == "__main__":
    main()

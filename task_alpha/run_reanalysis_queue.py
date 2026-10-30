"""MASTER RESUMABLE QUEUE for the paper's model-dependent analyses of the Op-H-W native readout.

Runs every model-dependent analysis of the paper against the Op-H-W model (FQDeepNative,
oph_weighted=True; 5-seed reference runs/fqnative2_5seed), in dependency order, one detached
serial pass with per-step .done markers (relaunch = resume; a failed step stops the queue).

    GPU steps (1-5) ................ ~2.5-4 GPU-days total; run them while no other job is
                                     using the GPU.
    analysis steps (6-9) ........... seconds-minutes, seed0 checkpoint only
    assembler steps (10-13) ........ instant; write the FQGNN_FIGDATA CSVs
    step 14 ........................ shared-gaps oracle self-test (checks the GCD column)

Launch (from the repository root; run it detached, e.g. under nohup or tmux, for multi-day queues):
  python -u task_alpha/run_reanalysis_queue.py
Monitor:   task_alpha/runs/_reanalysis_driver/STATUS  (+ per-step logs in the same dir)
Modes:     --list (show steps + state), --only <step>, --from <step>
"""
from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
os.chdir(ROOT)

PY = sys.executable
DRIVER = "task_alpha/runs/_reanalysis_driver"
STATUS = f"{DRIVER}/STATUS"

STEPS = [
    # (name, argv)  -- order = dependency order
    ("oneside_native",      [PY, "-u", "-m", "task_alpha.run_oneside_native_5seed"]),
    ("ablation_native",     [PY, "-u", "-m", "task_alpha.run_ablation_ladder_native", "--queue"]),
    ("gap_sweep_native",    [PY, "-u", "task_alpha/run_gap_sweep.py", "--queue", "--native"]),
    ("anchor_sweep_native", [PY, "-u", "task_alpha/run_anchor_sweep.py", "--queue", "--native"]),
    ("telemetry_native",    [PY, "-u", "-m", "task_alpha.run_telemetry_infamily_native"]),
    ("dump_spectra",        [PY, "-u", "paper_figures_tables/dump_spectra.py"]),
    ("extract_hero_gaps",   [PY, "-u", "paper_figures_tables/extract_hero_gaps.py"]),
    ("make_map_data",       [PY, "-u", "paper_figures_tables/make_map_data.py"]),
    ("params_audit",        [PY, "-u", "paper_figures_tables/params_audit.py"]),
    ("assemble_gap_sweep",  [PY, "-u", "-m", "task_alpha.assemble_gap_sweep", "--native"]),
    ("assemble_anchor",     [PY, "-u", "-m", "task_alpha.assemble_anchor_sweep", "--native"]),
    ("assemble_paper_table", [PY, "-u", "-m", "task_alpha.assemble_paper_table"]),
    ("assemble_ladder",     [PY, "-u", "-m", "task_alpha.assemble_ablation_ladder"]),
    ("gcd_selftest",        [PY, "-u", "-m", "task_alpha.baselines.common_gaps"]),
]


def log(msg):
    line = f"{datetime.datetime.now():%H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(STATUS, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def done_path(name):
    return f"{DRIVER}/{name}.done"


def run_step(i, name, argv):
    if os.path.exists(done_path(name)):
        log(f"[{i + 1}/{len(STEPS)}] {name}: already done, skip")
        return True
    log(f"[{i + 1}/{len(STEPS)}] {name}: start")
    with open(f"{DRIVER}/{name}.log", "w", encoding="utf-8") as lf:
        rc = subprocess.call(argv, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT)
    if rc == 0:
        open(done_path(name), "w").write("done\n")
        log(f"[{i + 1}/{len(STEPS)}] {name}: DONE")
        return True
    log(f"[{i + 1}/{len(STEPS)}] {name}: FAILED rc={rc} -- queue stopped "
        f"(see {DRIVER}/{name}.log; fix then relaunch to resume)")
    return False


def main():
    ap = argparse.ArgumentParser(description="re-analysis master queue (resumable)")
    ap.add_argument("--list", action="store_true", help="show steps + done state, run nothing")
    ap.add_argument("--only", help="run exactly one step (ignores its .done marker)")
    ap.add_argument("--from", dest="from_step", help="start at this step (earlier steps skipped)")
    args = ap.parse_args()
    os.makedirs(DRIVER, exist_ok=True)
    names = [n for n, _ in STEPS]

    if args.list:
        for i, (n, argv) in enumerate(STEPS):
            state = "done" if os.path.exists(done_path(n)) else "pending"
            print(f"  [{i + 1:2d}] {n:<22} {state:<8} {' '.join(argv[1:])}")
        return

    if args.only:
        if args.only not in names:
            ap.error(f"unknown step {args.only!r}; steps: {names}")
        i = names.index(args.only)
        dp = done_path(args.only)
        if os.path.exists(dp):
            os.remove(dp)
        run_step(i, *STEPS[i])
        return

    start = names.index(args.from_step) if args.from_step else 0
    if args.from_step and args.from_step not in names:
        ap.error(f"unknown step {args.from_step!r}; steps: {names}")
    log(f"=== re-analysis queue: {len(STEPS) - start} step(s) from '{names[start]}' ===")
    for i in range(start, len(STEPS)):
        if not run_step(i, *STEPS[i]):
            sys.exit(1)
    log("=== re-analysis queue COMPLETE ===")


if __name__ == "__main__":
    main()

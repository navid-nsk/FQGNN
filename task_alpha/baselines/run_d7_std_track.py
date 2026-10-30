"""Driver: SAITS then CSDI on the standardized-track cells, VANILLA, strictly serial.

Native-regime check: the regular-grid imputers evaluated in their native uniform-cadence regime.
Order (fast model first, headline cell first):
  saits  standard_process 2h, 6h, 12h, 24h     (5 seeds each, fixed Optuna HP, vanilla)
  csdi   standard_process 2h, 6h, 12h, 24h     (5 seeds each, deployed config, vanilla)

RESUMABLE: a step whose <out>/summary.csv exists is skipped -- relaunching IS the resume.
Launch from the repository root (run it detached, e.g. under nohup or tmux, for long queues):
  python -u task_alpha/baselines/run_d7_std_track.py

Progress: task_alpha/baselines/_driver_d7/<step>.log and _driver_d7/STATUS
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.chdir(ROOT)
PY = sys.executable
OUT = "task_alpha/baselines/_driver_d7"
os.makedirs(OUT, exist_ok=True)

GAPS = (2, 6, 12, 24)
STEPS = []
for g in GAPS:
    STEPS.append((f"saits_std_{g}h",
                  ["-u", "task_alpha/baselines/SAITS/run_saits_5seed_gaps.py",
                   "--track", "standard_process", "--gap", str(g)],
                  f"task_alpha/baselines/SAITS/runs_5seed_gaps_standard_process_{g}h/summary.csv"))
for g in GAPS:
    STEPS.append((f"csdi_std_{g}h",
                  ["-u", "task_alpha/baselines/CSDI/run_csdi_5seed_gaps.py",
                   "--track", "standard_process", "--gap", str(g)],
                  f"task_alpha/baselines/CSDI/runs_5seed_gaps_standard_process_{g}h/summary.csv"))


def status(msg):
    with open(os.path.join(OUT, "STATUS"), "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")


def main():
    status(f"driver start (pid {os.getpid()})")
    for name, argv, done_marker in STEPS:
        if os.path.exists(done_marker):
            status(f"SKIP {name} (summary present)")
            continue
        status(f"RUN  {name}")
        log = open(os.path.join(OUT, f"{name}.log"), "a", encoding="utf-8")
        t0 = time.time()
        rc = subprocess.call([PY] + argv, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        log.close()
        status(f"DONE {name} rc={rc} ({(time.time() - t0) / 3600:.2f} h)")
        if rc != 0:
            status(f"ABORT: {name} failed -- fix and relaunch (completed steps are skipped)")
            return
    status("ALL STANDARD-TRACK STEPS COMPLETE")


if __name__ == "__main__":
    main()

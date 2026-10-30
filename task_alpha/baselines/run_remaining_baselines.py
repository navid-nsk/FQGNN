"""Detached driver: the HABIT and MH-GIN AIS baselines on all 8 gap-sweep cells, VANILLA
settings, strictly serial.

Launch from the repository root (run it detached, e.g. under nohup or tmux, for long queues):
    python -u task_alpha/baselines/run_remaining_baselines.py

Progress:  task_alpha/baselines/_driver/<step>.log   and  _driver/STATUS

Neither baseline is tuned.  HABIT runs at its best honest H3 resolution (6) with the vendor's
max_gap=25.  MH-GIN runs its authors' config-DK.yaml schedule (30 epochs) with the
benchmark-metric early stopping of run_mhgin_gaps.py (patience 10) and max_delta 350 (the vendor
default; the cadence-matched max_delta=3600 variant gives essentially the same ratio, so the
vendor default is the reported configuration).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.chdir(ROOT)
PY = sys.executable
OUT = "task_alpha/baselines/_driver"
os.makedirs(OUT, exist_ok=True)

STEPS: list[tuple[str, list[str]]] = []

# .done markers skip every finished step, so relaunching resumes at the first unfinished step.

# 1) HABIT -- all 8 cells, each with the exact-median query (see HabitAdapter.prepare_query, which
#    replaces DuckDB's T-Digest approx_quantile), so every cell in the table is produced by the
#    same aggregation.  The difference from T-Digest is far below one H3 cell.
for track, gaps in (("our_processed", (3, 6, 12, 24)), ("standard_process", (2, 6, 12, 24))):
    for g in gaps:
        STEPS.append((f"habit_{track}_{g}h", [
            "-u", "-m", "task_alpha.baselines.HABIT.run_habit_gaps",
            "--track", track, "--gap", str(g), "--resolution", "6", "--max-gap", "25"]))

# 2) MH-GIN -- 5 seeds x 8 cells, vendor schedule.
for track, gaps in (("our_processed", (3, 6, 12, 24)), ("standard_process", (2, 6, 12, 24))):
    for g in gaps:
        STEPS.append((f"mhgin_{track}_{g}h", [
            "-u", "-m", "task_alpha.baselines.MHGIN.run_mhgin_gaps",
            "--track", track, "--gap", str(g), "--seeds", "0", "1", "2", "3", "4",
            "--epochs", "30", "--patience", "10", "--max-delta", "350", "--vram-frac", "0.50"]))


def status(msg: str) -> None:
    with open(f"{OUT}/STATUS", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')}  {msg}\n")


def main() -> None:
    status(f"driver start: {len(STEPS)} steps, pid {os.getpid()}")
    for i, (name, args) in enumerate(STEPS, 1):
        log = f"{OUT}/{name}.log"
        if os.path.exists(f"{OUT}/{name}.done"):
            status(f"[{i}/{len(STEPS)}] {name}: already done, skip")
            continue
        status(f"[{i}/{len(STEPS)}] {name}: start")
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as f:
            rc = subprocess.call([PY, *args], stdout=f, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        if rc == 0:
            open(f"{OUT}/{name}.done", "w").close()
            status(f"[{i}/{len(STEPS)}] {name}: OK ({dt:.0f}s)")
        else:
            status(f"[{i}/{len(STEPS)}] {name}: FAILED rc={rc} ({dt:.0f}s) -- see {name}.log")
    status("driver finished")


if __name__ == "__main__":
    main()

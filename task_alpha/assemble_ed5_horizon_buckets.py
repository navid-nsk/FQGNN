"""task_alpha.assemble_ed5_horizon_buckets -- per-BUCKET accuracy at every gap horizon (ED5).

The Extended Data 5 companion of assemble_gap_sweep.py: instead of the OVERALL row it pulls
the five SINUOSITY-BUCKET rows of every seed of every (track x horizon x model) cell, so the
figure can show WHERE (which curvature regime) each baseline is tolerable vs losing at every
gap length on both preprocessing tracks.

Models (paper set): fqgnn (SOBOLEV runs by default; --native for the native fd runs; --opm for
the Op-M stack runs), bigru, habit, mhgin, multitaskais + the four classical baselines hermite /
ctrv_kalman / quintic / constrained_gp (deterministic, seed0 only) + saits / csdi on the
standardized track only.
  - fqgnn  : DEFAULT (sobolev model): task_alpha/runs/gap_sweep_sobolev/<track>/... for
             our_processed, task_alpha/runs/gap_sweep_sobolev_fd/<track>/... for
             standard_process (fd boundary jet on that track, by design)
             + the PRE_DONE our_processed/6h cell task_alpha/runs/fqnative_sobolev_5seed.
             --native: task_alpha/runs/gap_sweep_native/<track>/gap_<G>h/fqgnn
             + the PRE_DONE our_processed/6h cell task_alpha/runs/fqnative2_5seed
             --opm: task_alpha/runs/gap_sweep_sobolev_opm/<track>/... for our_processed,
             task_alpha/runs/gap_sweep_sobolev_fd_opm/<track>/... for standard_process
             + the PRE_DONE our_processed/6h cell task_alpha/runs/fqnative_opmstack_5seed;
             fqgnn bucket rows come from seed<k>/test_buckets_clean.csv (the OFFICIAL
             exact-table re-eval) whenever present, falling back to test_buckets.csv
             with a loud warning if a clean file is absent
  - bigru  : task_alpha/runs/gap_sweep/<track>/gap_<G>h/bigru
             + the PRE_DONE our_processed/6h cell task_alpha/baselines/BiGRU/runs_5seed_gaps
  - habit / mhgin / multitaskais / hermite / ctrv_kalman / quintic / constrained_gp :
             task_alpha/runs/gap_sweep/<track>/gap_<G>h/<model>
             (their our_processed/6h cells live in the sweep root already)
  - habit is a deterministic route lookup -> 1 "seed"; multitaskais is a ONE-SIDED reference
    (forward VRNN) -- every label downstream must carry the caveat.
  - saits / csdi : trained natively per standardized cell
             (task_alpha/baselines/{SAITS,CSDI}/runs_5seed_gaps_standard_process_<G>h); their
             raw-track result is the single shared our_processed/6h cell, so they enter the grid
             on the standardized track only.
BF-BiLSTM is absent: it is evaluated zero-shot on the shared our_processed/6h cell only (no
horizon sweep), so it cannot appear in a horizon grid.

Writes (plot-ready, LONG format -- one row per track x gap x model x bucket):
  <FQGNN_FIGDATA>/ed5_horizon_buckets.csv
    track, gap_h, model, bucket, n, gcd_med, model_med_mean, model_med_std,
    ratio_mean, ratio_std, n_seeds
n and gcd_med are seed-independent (fixed test gap set per cell) and verified identical
across seeds.  Buckets missing from a seed file (cells with < 3 gaps in a bucket are dropped
upstream) are skipped with a stdout note.  Safe to re-run at any time.

Run (gpubench env, any cwd):
  python task_alpha/assemble_ed5_horizon_buckets.py
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
SWEEP_ROOT_NATIVE = "task_alpha/runs/gap_sweep_native"                # fqgnn cells only
# default (sobolev model): per-track fqgnn roots (fd boundary jet on the standard track)
SWEEP_ROOT_SOBOLEV = {"our_processed": "task_alpha/runs/gap_sweep_sobolev",
                      "standard_process": "task_alpha/runs/gap_sweep_sobolev_fd"}
# --opm (Op-M stack): per-track fqgnn roots (fd boundary jet on the standard track,
# as in the sobolev sweep); bucket rows from test_buckets_clean.csv
SWEEP_ROOT_OPM = {"our_processed": "task_alpha/runs/gap_sweep_sobolev_opm",
                  "standard_process": "task_alpha/runs/gap_sweep_sobolev_fd_opm"}
from task_alpha.paths import FIGDATA_DIR as FIGDATA  # noqa: E402  (env FQGNN_FIGDATA)
OUT = f"{FIGDATA}/ed5_horizon_buckets.csv"

MODELS = ("fqgnn", "bigru", "habit", "mhgin", "multitaskais",
          "hermite", "ctrv_kalman", "quintic", "constrained_gp",
          "saits", "csdi")
# saits / csdi: NATIVE-cadence training exists on the standardized track only (their raw-track
# result is the single shared 6-h cell, Fig. 2d) -> standardized rows only in the grid
STD_ONLY = {"saits", "csdi"}
STD_ONLY_SRC = {"saits": "task_alpha/baselines/SAITS/runs_5seed_gaps_standard_process_{g}h",
                "csdi": "task_alpha/baselines/CSDI/runs_5seed_gaps_standard_process_{g}h"}
TRACK_GAPS = {"our_processed": (3, 6, 12, 24), "standard_process": (2, 6, 12, 24)}
BUCKETS = ("0.00-0.02", "0.02-0.05", "0.05-0.10", "0.10-0.25", ">0.25")

# our_processed/6h cells that live outside the sweep roots (same convention as
# assemble_gap_sweep.py: PRE_DONE for the op-k sweep models, PRE_DONE_NATIVE for the native fqgnn,
# PRE_DONE_SOBOLEV for the sobolev model, PRE_DONE_OPM for the op-m stack headline run)
PRE_DONE = {("our_processed", 6): {
    "bigru": "task_alpha/baselines/BiGRU/runs_5seed_gaps",
}}
PRE_DONE_NATIVE_FQGNN = {("our_processed", 6): "task_alpha/runs/fqnative2_5seed"}
PRE_DONE_SOBOLEV_FQGNN = {("our_processed", 6): "task_alpha/runs/fqnative_sobolev_5seed"}
PRE_DONE_OPM_FQGNN = {("our_processed", 6): "task_alpha/runs/fqnative_opmstack_5seed"}

FIELDS = ["track", "gap_h", "model", "bucket", "n", "gcd_med", "model_med_mean",
          "model_med_std", "ratio_mean", "ratio_std", "n_seeds"]


def resolve_src(track: str, gap: int, model: str, native: bool = False,
                opm: bool = False) -> str:
    """Run directory of one (track, gap, model) cell -- fqgnn from the SOBOLEV roots by
    default, from the NATIVE root with native=True, from the OP-M STACK roots with
    opm=True."""
    if model in STD_ONLY:
        return STD_ONLY_SRC[model].format(g=gap)
    if model == "fqgnn":
        if native:
            return PRE_DONE_NATIVE_FQGNN.get((track, gap)) \
                or f"{SWEEP_ROOT_NATIVE}/{track}/gap_{gap}h/{model}"
        if opm:
            return PRE_DONE_OPM_FQGNN.get((track, gap)) \
                or f"{SWEEP_ROOT_OPM[track]}/{track}/gap_{gap}h/{model}"
        return PRE_DONE_SOBOLEV_FQGNN.get((track, gap)) \
            or f"{SWEEP_ROOT_SOBOLEV[track]}/{track}/gap_{gap}h/{model}"
    return PRE_DONE.get((track, gap), {}).get(model) \
        or f"{SWEEP_ROOT}/{track}/gap_{gap}h/{model}"


def bucket_rows(run_dir: str, prefer_clean: bool = False):
    """[(seed, {bucket: row-dict})] from seed<k>/test_buckets.csv under run_dir (sorted).
    prefer_clean (fqgnn --opm cells): read seed<k>/test_buckets_clean.csv (the OFFICIAL
    exact-table re-eval) whenever it exists; fall back to the in-runner test_buckets.csv
    only when the clean file is absent, with a loud warning."""
    out = []
    if not os.path.isdir(run_dir):
        return out
    for d in os.listdir(run_dir):
        if not (d.startswith("seed") and d[4:].isdigit()):
            continue
        p = os.path.join(run_dir, d, "test_buckets.csv")
        if prefer_clean:
            p_clean = os.path.join(run_dir, d, "test_buckets_clean.csv")
            if os.path.exists(p_clean):
                p = p_clean
            else:
                print(f"[ed5] WARNING: {p_clean} MISSING -- falling back to the in-runner "
                      f"test_buckets.csv (NOT the official exact-table re-eval)", flush=True)
        if not os.path.exists(p):
            continue
        with open(p, newline="") as f:
            rows = list(csv.DictReader(f))
        by_bucket = {r["bucket"]: r for r in rows if r["bucket"] != "OVERALL"}
        if by_bucket:
            out.append((int(d[4:]), by_bucket))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description="assemble the ED5 horizon x bucket grid CSV")
    ap.add_argument("--native", action="store_true",
                    help="fqgnn cells from the native fd-jet runs (gap_sweep_native + "
                         "fqnative2_5seed) instead of the sobolev default")
    ap.add_argument("--opm", action="store_true",
                    help="fqgnn cells from the Op-M stack runs (gap_sweep_sobolev_opm / "
                         "gap_sweep_sobolev_fd_opm + fqnative_opmstack_5seed; bucket rows "
                         "from the clean re-eval test_buckets_clean.csv) instead of the "
                         "sobolev default")
    args = ap.parse_args()
    if args.native and args.opm:
        ap.error("--native and --opm are mutually exclusive")
    fq_src = ("native (--native)" if args.native
              else "op-m stack (--opm, clean re-eval)" if args.opm
              else "sobolev (default)")
    print(f"[ed5] fqgnn source: {fq_src}", flush=True)
    pairs = [(t, g) for t in ("our_processed", "standard_process") for g in TRACK_GAPS[t]]
    long_rows, missing = [], []
    for track, gap in pairs:
        for model in MODELS:
            if model in STD_ONLY and track == "our_processed":
                continue                       # native-cadence training exists on std only
            src = resolve_src(track, gap, model, native=args.native, opm=args.opm)
            seeds = bucket_rows(src, prefer_clean=args.opm and model == "fqgnn")
            if not seeds:
                print(f"[ed5] {track}/gap_{gap}h/{model}: no finished seeds ({src}) -- skipped",
                      flush=True)
                missing.append(f"{track}/gap_{gap}h/{model} (whole cell)")
                continue
            n_buckets = 0
            for bucket in BUCKETS:
                have = [(k, by[bucket]) for k, by in seeds if bucket in by]
                if not have:
                    missing.append(f"{track}/gap_{gap}h/{model}/{bucket}")
                    print(f"[ed5] {track}/gap_{gap}h/{model}: bucket {bucket} absent in every "
                          "seed (dropped upstream, < 3 gaps) -- skipped", flush=True)
                    continue
                if len(have) != len(seeds):
                    print(f"[ed5] {track}/gap_{gap}h/{model}: bucket {bucket} present in only "
                          f"{len(have)}/{len(seeds)} seeds", flush=True)
                # n and gcd_med are properties of the fixed test gap set -> verify consistency
                ns = {int(r["n"]) for _, r in have}
                gcds = {float(r["gcd_med"]) for _, r in have}
                if len(ns) > 1 or len(gcds) > 1:
                    print(f"[ed5] WARNING {track}/gap_{gap}h/{model}/{bucket}: n/gcd_med differ "
                          f"across seeds (n={sorted(ns)}, gcd={sorted(gcds)})", flush=True)
                rat = np.array([float(r["ratio"]) for _, r in have], float)
                med = np.array([float(r["model_med"]) for _, r in have], float)
                long_rows.append({
                    "track": track, "gap_h": gap, "model": model, "bucket": bucket,
                    "n": max(ns), "gcd_med": round(float(np.mean(sorted(gcds))), 3),
                    "model_med_mean": round(float(med.mean()), 3),
                    "model_med_std": round(float(med.std(ddof=0)), 3),
                    "ratio_mean": round(float(rat.mean()), 4),
                    "ratio_std": round(float(rat.std(ddof=0)), 4),
                    "n_seeds": len(have),
                })
                n_buckets += 1
            print(f"[ed5] {track}/gap_{gap}h/{model}: {len(seeds)} seed(s), "
                  f"{n_buckets}/5 buckets from {src}", flush=True)

    if not long_rows:
        print("[ed5] nothing to assemble", flush=True)
        return
    os.makedirs(FIGDATA, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerows(long_rows)

    expected = len(pairs) * len(MODELS) * len(BUCKETS)
    rats = np.array([r["ratio_mean"] for r in long_rows], float)
    print(f"\n[ed5] {len(long_rows)} data rows (expected {expected}) -> {OUT}", flush=True)
    print(f"[ed5] ratio_mean range {rats.min():.3f} .. {rats.max():.1f}", flush=True)
    if missing:
        print(f"[ed5] {len(missing)} missing bucket(s)/cell(s):", flush=True)
        for m in missing:
            print(f"        {m}", flush=True)
    else:
        print("[ed5] no missing buckets -- every cell has all 5", flush=True)


if __name__ == "__main__":
    main()

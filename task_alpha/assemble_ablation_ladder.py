"""Assemble <FQGNN_FIGDATA>/ablation_ladder.csv (Fig 4b) from the per-variant 5-seed run dirs.

The table is rebuilt reproducibly from the run directories.  Default mode =
the NATIVE ladder (rungs from task_alpha/runs/ablation_ladder_native, full rung from
runs/fqnative2_5seed, one_sided from runs/oneside_native_5seed; appends the two native-only rungs
no_metric / no_eta when their dirs exist).  --check mode = rebuild from the Op-K sources and
assert equality with the on-disk file (a reproducibility check of the assembler).

Schema: variant,label,n_seeds,ratio_mean,ratio_std,
delta_vs_full,win_mean,win_std,ratio_per_seed  with per-seed ratios ';'-joined at fixed 4 decimals,
means/stds/deltas str(round(x, 4)), wins str(round(x, 1)); std is population (ddof=0).

--sobolev mode = the sobolev-metric ladder: full rung from runs/fqnative_sobolev_5seed,
one_sided from runs/oneside_sobolev_5seed, variant rungs from runs/ablation_ladder_native_sobolev;
rffgnn source unchanged.

--opm mode = the Op-M stack (paper) ladder: full rung from runs/fqnative_opmstack_5seed,
one_sided from runs/oneside_sobolev_5seed_opm, variant rungs from
runs/ablation_ladder_native_sobolev_opm; rffgnn source unchanged.  FQGNN seed dirs are read from
the OFFICIAL exact-table re-eval files (test_buckets_clean.csv, falling back to test_buckets.csv
with a loud warning if the clean file is absent); the rffgnn baseline has no clean variant and is
read unchanged.

Run:  python -m task_alpha.assemble_ablation_ladder [--check | --sobolev | --opm]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np

from task_alpha.paths import FIGDATA_DIR  # noqa: E402  (env FQGNN_FIGDATA)
OUT = os.path.join(FIGDATA_DIR, "ablation_ladder.csv")
LABELS = {
    "full_fqgnn": "Full FQ-GNN",
    "free_amps": "Free amplitudes (faithful init)",
    "free_amps_randinit": "Free amplitudes (random init)",
    "frozen_L": "Frozen generator L",
    "d1_lift": "Scalar-time lift (d=1)",
    "decoupled_U": "Decoupled coupling (U=I)",
    "rbf_kernel": "Generic RBF kernel (no FQ comb)",
    "no_dmu": "No D-mu velocity",
    "rffgnn": "RFF-GNN (full operator swap)",
    "one_sided": "One-sided context only",
    "no_metric": "No learned metric (w = const)",
    "no_eta": "No eta head",
}
VARIANT_ORDER = ("full_fqgnn", "free_amps", "free_amps_randinit", "frozen_L", "d1_lift",
                 "decoupled_U", "rbf_kernel", "no_dmu", "rffgnn", "one_sided")
NATIVE_EXTRA = ("no_metric", "no_eta")                  # appended (all modes but --check) when dirs exist


def sources(check: bool, sobolev: bool = False, opm: bool = False):
    if check:                                           # the Op-K ladder
        lad = "task_alpha/runs/ablation_ladder"
        return {**{v: f"{lad}/{v}" for v in VARIANT_ORDER},
                "full_fqgnn": "task_alpha/runs/full_faithful_5seed",
                "rffgnn": "task_alpha/baselines/RFFGNN/runs_5seed_gaps",
                "one_sided": "task_alpha/runs/oneside_5seed"}
    if opm:                                             # the Op-M stack (paper) ladder
        lad = "task_alpha/runs/ablation_ladder_native_sobolev_opm"
        return {**{v: f"{lad}/{v}" for v in VARIANT_ORDER + NATIVE_EXTRA},
                "full_fqgnn": "task_alpha/runs/fqnative_opmstack_5seed",
                "rffgnn": "task_alpha/baselines/RFFGNN/runs_5seed_gaps",
                "one_sided": "task_alpha/runs/oneside_sobolev_5seed_opm"}
    if sobolev:                                         # the sobolev-metric ladder
        lad = "task_alpha/runs/ablation_ladder_native_sobolev"
        return {**{v: f"{lad}/{v}" for v in VARIANT_ORDER + NATIVE_EXTRA},
                "full_fqgnn": "task_alpha/runs/fqnative_sobolev_5seed",
                "rffgnn": "task_alpha/baselines/RFFGNN/runs_5seed_gaps",
                "one_sided": "task_alpha/runs/oneside_sobolev_5seed"}
    lad = "task_alpha/runs/ablation_ladder_native"
    src = {**{v: f"{lad}/{v}" for v in VARIANT_ORDER + NATIVE_EXTRA},
           "full_fqgnn": "task_alpha/runs/fqnative2_5seed",
           "rffgnn": "task_alpha/baselines/RFFGNN/runs_5seed_gaps",
           "one_sided": "task_alpha/runs/oneside_native_5seed"}
    return src


def seed_ratios(run_dir, clean=False):
    """(ratios, wins) sorted by seed from seed<k>/test_buckets.csv OVERALL rows.

    clean=True (FQGNN dirs in --opm mode) prefers the OFFICIAL exact-table re-eval file
    test_buckets_clean.csv, falling back to test_buckets.csv with a LOUD warning.
    """
    out = []
    if not os.path.isdir(run_dir):
        return [], []
    for d in sorted(os.listdir(run_dir)):
        if not (d.startswith("seed") and d[4:].isdigit()):
            continue
        p = os.path.join(run_dir, d, "test_buckets.csv")
        if clean:
            pc = os.path.join(run_dir, d, "test_buckets_clean.csv")
            if os.path.exists(pc):
                p = pc
            elif os.path.exists(p):
                print(f"WARNING: clean file missing, falling back to IN-RUNNER file: {p}",
                      flush=True)
        if not os.path.exists(p):
            continue
        with open(p, newline="") as f:
            r0 = next(csv.DictReader(f))
        assert r0["bucket"] == "OVERALL", p
        out.append((int(d[4:]), float(r0["ratio"]), float(r0["win_pct"])))
    out.sort()
    return [r for _, r, _ in out], [w for _, _, w in out]


def main():
    ap = argparse.ArgumentParser(description="assemble the Fig-4b ablation ladder CSV")
    ap.add_argument("--check", action="store_true",
                    help="rebuild from the Op-K sources and diff against the on-disk file")
    ap.add_argument("--sobolev", action="store_true",
                    help="sobolev sources: full <- fqnative_sobolev_5seed, one_sided <- "
                         "oneside_sobolev_5seed, rungs <- ablation_ladder_native_sobolev "
                         "(rffgnn unchanged)")
    ap.add_argument("--opm", action="store_true",
                    help="Op-M stack sources: full <- fqnative_opmstack_5seed, one_sided <- "
                         "oneside_sobolev_5seed_opm, rungs <- ablation_ladder_native_sobolev_opm "
                         "(rffgnn unchanged); FQGNN seed dirs read test_buckets_clean.csv")
    args = ap.parse_args()
    if sum((args.check, args.sobolev, args.opm)) > 1:
        ap.error("--check, --sobolev and --opm are mutually exclusive")
    src = sources(args.check, args.sobolev, args.opm)

    def rung_ratios(v):                                 # clean re-eval files for FQGNN rungs only
        return seed_ratios(src[v], clean=args.opm and v != "rffgnn")

    variants = list(VARIANT_ORDER)
    if not args.check:
        variants += [v for v in NATIVE_EXTRA if os.path.isdir(src[v])]

    missing = [v for v in variants if not rung_ratios(v)[0]]
    if missing:
        print(f"MISSING RUNGS (no seed results yet): {missing}", flush=True)
        if not args.check:
            print("-> run task_alpha.run_ablation_ladder_native (and the oneside/native runs) "
                  "first; nothing written.", flush=True)
            sys.exit(1)

    full_rat, _ = rung_ratios("full_fqgnn")
    full_mean = round(float(np.mean(full_rat)), 4)      # delta convention: rounded means
    rows = []
    for v in variants:
        rr, ww = rung_ratios(v)
        rmean = round(float(np.mean(rr)), 4)
        rows.append({"variant": v, "label": LABELS[v], "n_seeds": len(rr),
                     "ratio_mean": str(rmean),
                     "ratio_std": str(round(float(np.std(rr)), 4)),
                     "delta_vs_full": str(round(rmean - full_mean, 4)),
                     "win_mean": str(round(float(np.mean(ww)), 1)),
                     "win_std": str(round(float(np.std(ww)), 1)),
                     "ratio_per_seed": ";".join(f"{r:.4f}" for r in rr)})

    fields = ["variant", "label", "n_seeds", "ratio_mean", "ratio_std", "delta_vs_full",
              "win_mean", "win_std", "ratio_per_seed"]
    if args.check:
        import io
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
        w.writeheader(); w.writerows(rows)
        with open(OUT, newline="") as f:
            disk = f.read().replace("\r\n", "\n")
        same = disk == buf.getvalue()
        print(f"[check] {OUT}: {'MATCH' if same else 'DIFF'}", flush=True)
        if not same:
            a, b = disk.splitlines(), buf.getvalue().splitlines()
            for i in range(min(len(a), len(b))):
                if a[i] != b[i]:
                    print(f"  first diff line {i}:\n  disk : {a[i]}\n  built: {b[i]}")
                    break
        sys.exit(0 if same else 1)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    print(f"[assemble] wrote {OUT} ({len(rows)} rungs; full = {src['full_fqgnn']})", flush=True)


if __name__ == "__main__":
    main()

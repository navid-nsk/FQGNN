"""Assemble the same-gaps SOTA paper table: combined_buckets_allmodels.csv + pivot_ratio_mean.csv,
written to BOTH task_alpha/runs/paper_table/ and <FQGNN_FIGDATA>/ (figures read both locations).

This assembler produces both tables reproducibly.  Default mode = the SOBOLEV model (FQGNN rows from
task_alpha/runs/fqnative_sobolev_5seed).  --v2 = the NATIVE fd model
(runs/fqnative2_5seed).  --opm = the Op-M stack paper model
(runs/fqnative_opmstack_5seed); its FQGNN rows come from the OFFICIAL exact-table re-eval files
(test_buckets_clean.csv per seed -- fall back to test_buckets.csv only if the clean file is
absent, with a loud warning).  --check mode = rebuild from the CURRENTLY SELECTED source and assert
equality with the on-disk files (idempotency/reproducibility check).

Row structure:
  combined: model,seed,bucket,n,gcd_med,model_med,ratio,win_pct -- source test_buckets rows
            VERBATIM, prefixed by model/seed.  Model order: FQGNN, CSDI, SAITS, BiGRU, RFFGNN
            (5 seeds each), BF-BiLSTM (seed 0), GCD (seed 0, derived: model_med = gcd_med,
            ratio 1.0, win_pct blank), then HABIT/MH-GIN/MultitaskAIS, then Hermite (the
            classical cubic-Hermite baseline, deterministic seed0; its our-6h per_gap is
            row-aligned with the FQ per_gap -- checked by task_alpha/analysis/clustered_stats.py).
  pivot:    bucket,n,gcd_med,<M>_ratio...,<M>_med_nm... in order GCD,FQGNN,RFFGNN,CSDI,SAITS,
            BiGRU,BF-BiLSTM,HABIT,MH-GIN,MultitaskAIS,Hermite; values = mean over seeds,
            str(round(ratio,3)) / str(round(med,2)).

Run:  python -m task_alpha.assemble_paper_table [--check] [--v2]
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
OUT_DIRS = ("task_alpha/runs/paper_table", FIGDATA_DIR)
FQ_SOBOLEV = "task_alpha/runs/fqnative_sobolev_5seed"       # default: the sobolev model
FQ_OPM = "task_alpha/runs/fqnative_opmstack_5seed"          # --opm: the Op-M stack paper model
FQ_V2 = "task_alpha/runs/fqnative2_5seed"
FQ_OLD = "task_alpha/runs/full_faithful_5seed"
SEEDED = (("CSDI", "task_alpha/baselines/CSDI/runs_5seed_gaps"),
          ("SAITS", "task_alpha/baselines/SAITS/runs_5seed_gaps"),
          ("BiGRU", "task_alpha/baselines/BiGRU/runs_5seed_gaps"),
          ("RFFGNN", "task_alpha/baselines/RFFGNN/runs_5seed_gaps"),
          # the three AIS-native baselines are part of the shared-gaps table
          # (their our-6h cells are on the identical 770 gaps: n=770, gcd_med 2.857).
          ("MH-GIN", "task_alpha/runs/gap_sweep/our_processed/gap_6h/mhgin"),
          # MultitaskAIS is a ONE-SIDED reference (forward VRNN, prior-sampling reconstruction);
          # every figure/caption/table label must carry the one-sided caveat.
          ("MultitaskAIS", "task_alpha/runs/gap_sweep/our_processed/gap_6h/multitaskais"))
BF_SRC = "task_alpha/baselines/BF-BiLSTM/runs_gaps/test_buckets.csv"
HABIT_SRC = "task_alpha/runs/gap_sweep/our_processed/gap_6h/habit/seed0/test_buckets.csv"
# Hermite: classical cubic-Hermite baseline, deterministic -> seed0 only (same 770 shared gaps,
# n=770 / gcd_med 2.857 OVERALL)
HERM_SRC = "task_alpha/runs/gap_sweep/our_processed/gap_6h/hermite/seed0/test_buckets.csv"
PIVOT_ORDER = ("GCD", "FQGNN", "RFFGNN", "CSDI", "SAITS", "BiGRU", "BF-BiLSTM",
               "HABIT", "MH-GIN", "MultitaskAIS", "Hermite")
BUCKETS = ("OVERALL", "0.00-0.02", "0.02-0.05", "0.05-0.10", "0.10-0.25", ">0.25")
EXPECT_N = {"OVERALL": "770", "0.00-0.02": "225", "0.02-0.05": "74", "0.05-0.10": "75",
            "0.10-0.25": "105", ">0.25": "291"}


def read_buckets(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build(fq_src, fq_clean=False):
    combined = []                                       # list of row dicts (strings verbatim)
    per_model = {}                                      # model -> seed -> bucket -> row

    def add(model, seed, rows):
        per_model.setdefault(model, {})[seed] = {r["bucket"]: r for r in rows}
        for r in rows:
            combined.append({"model": model, "seed": seed, **r})

    for k in range(5):
        path = f"{fq_src}/seed{k}/test_buckets.csv"
        if fq_clean:
            # FQGNN clean-file rule: the OFFICIAL exact-table re-eval file wins whenever it
            # exists; the in-runner file is only a (loudly warned) fallback.
            clean = f"{fq_src}/seed{k}/test_buckets_clean.csv"
            if os.path.exists(clean):
                path = clean
            else:
                print(f"[WARNING] {clean} MISSING -- falling back to the in-runner "
                      f"{path}; FQGNN rows will NOT match the official exact-table re-eval!",
                      flush=True)
        add("FQGNN", k, read_buckets(path))
    for model, src in SEEDED:
        for k in range(5):
            add(model, k, read_buckets(f"{src}/seed{k}/test_buckets.csv"))
    add("BF-BiLSTM", 0, read_buckets(BF_SRC))
    add("HABIT", 0, read_buckets(HABIT_SRC))               # deterministic lookup -> seed0 only
    add("Hermite", 0, read_buckets(HERM_SRC))              # deterministic classical -> seed0 only
    gcd_rows = [{"bucket": r["bucket"], "n": r["n"], "gcd_med": r["gcd_med"],
                 "model_med": r["gcd_med"], "ratio": "1.0", "win_pct": ""}
                for r in per_model["FQGNN"][0].values()]
    add("GCD", 0, gcd_rows)

    # combined row order: FQGNN, CSDI, SAITS, BiGRU, RFFGNN, BF-BiLSTM, GCD, then HABIT,
    # MH-GIN, MultitaskAIS, Hermite
    order = {"FQGNN": 0, "CSDI": 1, "SAITS": 2, "BiGRU": 3, "RFFGNN": 4, "BF-BiLSTM": 5, "GCD": 6,
             "HABIT": 7, "MH-GIN": 8, "MultitaskAIS": 9, "Hermite": 10}
    combined.sort(key=lambda r: (order[r["model"]], int(r["seed"]),
                                 BUCKETS.index(r["bucket"])))

    # sanity invariants (shared-gaps oracle)
    ov = per_model["FQGNN"][0]["OVERALL"]
    assert ov["n"] == "770" and ov["gcd_med"] == "2.857", f"oracle drift: {ov}"
    for b, n in EXPECT_N.items():
        assert per_model["FQGNN"][0][b]["n"] == n, (b, per_model["FQGNN"][0][b]["n"])

    pivot = []
    for b in BUCKETS:
        row = {"bucket": b, "n": per_model["FQGNN"][0][b]["n"],
               "gcd_med": per_model["FQGNN"][0][b]["gcd_med"]}
        for m in PIVOT_ORDER:
            rr = [float(per_model[m][s][b]["ratio"]) for s in sorted(per_model[m])]
            row[f"{m}_ratio"] = str(round(float(np.mean(rr)), 3))
        for m in PIVOT_ORDER:
            mm = [float(per_model[m][s][b]["model_med"]) for s in sorted(per_model[m])]
            row[f"{m}_med_nm"] = str(round(float(np.mean(mm)), 2))
        pivot.append(row)
    return combined, pivot


def render(rows, fieldnames):
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\r\n")
    w.writeheader(); w.writerows(rows)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description="assemble the same-gaps SOTA paper table")
    ap.add_argument("--check", action="store_true",
                    help="rebuild from the current sources and diff against the on-disk files "
                         "(idempotency check; writes nothing).")
    ap.add_argument("--v2", action="store_true",
                    help="FQGNN rows from the native fd-jet model (runs/fqnative2_5seed) instead "
                         "of the sobolev default")
    ap.add_argument("--opm", action="store_true",
                    help="FQGNN rows from the Op-M stack model (runs/fqnative_opmstack_5seed), "
                         "read from the OFFICIAL exact-table re-eval test_buckets_clean.csv "
                         "per seed, instead of the sobolev default")
    args = ap.parse_args()

    fq_src = FQ_OPM if args.opm else (FQ_V2 if args.v2 else FQ_SOBOLEV)
    combined, pivot = build(fq_src, fq_clean=args.opm)
    cfields = ["model", "seed", "bucket", "n", "gcd_med", "model_med", "ratio", "win_pct"]
    pfields = (["bucket", "n", "gcd_med"] + [f"{m}_ratio" for m in PIVOT_ORDER]
               + [f"{m}_med_nm" for m in PIVOT_ORDER])
    ctxt, ptxt = render(combined, cfields), render(pivot, pfields)

    if args.check:
        ok = True
        for d in OUT_DIRS:
            for name, txt in (("combined_buckets_allmodels.csv", ctxt),
                              ("pivot_ratio_mean.csv", ptxt)):
                with open(f"{d}/{name}", newline="") as f:
                    disk = f.read()
                same = disk.replace("\r\n", "\n") == txt.replace("\r\n", "\n")
                print(f"[check] {d}/{name}: {'MATCH' if same else 'DIFF'}", flush=True)
                if not same:
                    ok = False
                    a = disk.replace("\r\n", "\n").splitlines()
                    b = txt.replace("\r\n", "\n").splitlines()
                    for i, (x, y) in enumerate(zip(a, b)):
                        if x != y:
                            print(f"   first diff line {i}:\n   disk: {x}\n   built: {y}")
                            break
                    if len(a) != len(b):
                        print(f"   line count: disk {len(a)} vs built {len(b)}")
        print("[check]", "PASS -- assembler reproduces the on-disk originals" if ok
              else "FAIL -- fix before trusting the rebuilt tables", flush=True)
        sys.exit(0 if ok else 1)

    for d in OUT_DIRS:
        os.makedirs(d, exist_ok=True)
        with open(f"{d}/combined_buckets_allmodels.csv", "w", newline="") as f:
            f.write(ctxt)
        with open(f"{d}/pivot_ratio_mean.csv", "w", newline="") as f:
            f.write(ptxt)
        print(f"[assemble] wrote {d}/combined_buckets_allmodels.csv + pivot_ratio_mean.csv "
              f"(FQGNN <- {fq_src})", flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Assemble the standardized 2-h cell's combined per-seed bucket table + seed-mean pivot
(headline-cell analogues of runs/paper_table's 6-h tables) for Figure 3.

Sources: FQGNN from runs/gap_sweep_sobolev_fd_opm/.../gap_2h/fqgnn/seed*/test_buckets_clean.csv
(official exact-table re-evals); every baseline from runs/gap_sweep/standard_process/gap_2h/
<model>/seed*/test_buckets.csv. GCD medians per bucket from the hermite per-gap file (its
gcd_med column is the shared reference on the identical gaps).

Out: <FQGNN_FIGDATA>/combined_buckets_std2h.csv + <FQGNN_FIGDATA>/pivot_ratio_std2h.csv
Run:  python task_alpha/assemble_std2h_table.py
"""
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_FD_ENV = os.environ.get("FQGNN_FIGDATA", "").strip()            # env FQGNN_FIGDATA
FD = os.path.abspath(_FD_ENV) if _FD_ENV else os.path.join(ROOT, "paper_figures_tables", "figdata")
os.makedirs(FD, exist_ok=True)
RUNS = os.path.join(HERE, "runs")

MODELS = {
    "FQGNN": (os.path.join(RUNS, "gap_sweep_sobolev_fd_opm", "standard_process", "gap_2h",
                           "fqgnn"), "test_buckets_clean.csv"),
    "Hermite": (os.path.join(RUNS, "gap_sweep", "standard_process", "gap_2h", "hermite"),
                "test_buckets.csv"),
    "HABIT": (os.path.join(RUNS, "gap_sweep", "standard_process", "gap_2h", "habit"),
              "test_buckets.csv"),
    "RFFGNN": (os.path.join(RUNS, "gap_sweep", "standard_process", "gap_2h", "rffgnn"),
               "test_buckets.csv"),
    "BiGRU": (os.path.join(RUNS, "gap_sweep", "standard_process", "gap_2h", "bigru"),
              "test_buckets.csv"),
    "MH-GIN": (os.path.join(RUNS, "gap_sweep", "standard_process", "gap_2h", "mhgin"),
               "test_buckets.csv"),
    "MultitaskAIS": (os.path.join(RUNS, "gap_sweep", "standard_process", "gap_2h",
                                  "multitaskais"), "test_buckets.csv"),
}

rows = []
for model, (root, fname) in MODELS.items():
    for sd in sorted(glob.glob(os.path.join(root, "seed*"))):
        f = os.path.join(sd, fname)
        if not os.path.exists(f):
            continue
        seed = int(os.path.basename(sd).replace("seed", ""))
        tb = pd.read_csv(f)
        for r in tb.itertuples(index=False):
            rows.append(dict(model=model, seed=seed, bucket=r.bucket, n=r.n,
                             gcd_med=r.gcd_med, model_med=r.model_med, ratio=r.ratio,
                             win_pct=r.win_pct))
comb = pd.DataFrame(rows)

# GCD reference row (ratio = 1 by definition; medians from the shared gcd_med column)
gref = comb[comb.model == "Hermite"][["bucket", "n", "gcd_med"]].drop_duplicates("bucket")
for r in gref.itertuples(index=False):
    rows.append(dict(model="GCD", seed=0, bucket=r.bucket, n=r.n, gcd_med=r.gcd_med,
                     model_med=r.gcd_med, ratio=1.0,
                     win_pct=np.nan))
comb = pd.DataFrame(rows)
comb.to_csv(os.path.join(FD, "combined_buckets_std2h.csv"), index=False)

piv_rows = []
for b in comb.bucket.unique():
    row = {"bucket": b}
    sub = comb[comb.bucket == b]
    row["n"] = int(sub.n.iloc[0])
    row["gcd_med"] = float(sub.gcd_med.iloc[0])
    for m in list(MODELS) + ["GCD"]:
        s = sub[sub.model == m]
        if len(s):
            row[f"{m}_ratio"] = round(float(s.ratio.mean()), 4)
            row[f"{m}_med_nm"] = round(float(s.model_med.mean()), 3)
    piv_rows.append(row)
piv = pd.DataFrame(piv_rows)
piv.to_csv(os.path.join(FD, "pivot_ratio_std2h.csv"), index=False)

print(piv[["bucket", "n", "gcd_med"] + [f"{m}_ratio" for m in MODELS]].to_string(index=False))
print("wrote combined_buckets_std2h.csv +", "pivot_ratio_std2h.csv")

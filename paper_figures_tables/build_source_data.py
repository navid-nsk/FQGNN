"""Build Source_Data.xlsx (per-figure source-data workbook), written to FQGNN_FIG_OUT.

Deterministic, reproducible builder.  Reads ONLY archived CSV/JSON outputs (figdata/ below is
FQGNN_FIGDATA and task_alpha/ is FQGNN_RESULTS_ROOT; see _paths.py):
  figdata/*.csv                                         (all plot-ready figure data)
  task_alpha/runs/fqnative_opmstack_5seed/              (deployed FQGNN 5-seed run: curves)
  task_alpha/runs/fqnative_opmstack_5seed_clean/        (Op-M clean re-eval per-gap, Fig 2c)
  task_alpha/runs/gap_sweep_sobolev_fd_opm/standard_process/gap_2h/   (std-2h per-gap, Fig 3d)
  task_alpha/runs/oneside_sobolev_5seed_opm/            (one-sided ablation, Fig 2e)
  task_alpha/runs/telemetry_infamily_native/            (in-family telemetry + combs)
  task_alpha/baselines/{RFFGNN,BiGRU,SAITS,CSDI}/runs_5seed_gaps/   (training curves)
  task_alpha/baselines/CSDI/hp_optuna/gap_6h/trials.csv (ED Fig 3a)
  task_alpha/baselines/SAITS/hp_saits/trials.csv        (ED Fig 3b)

Layout conventions: one sheet per figure,
bold size-12 sheet/panel titles, dark-blue (1F3864) white-bold header rows, italic
cross-reference notes, red 'Source:' lines on ED Fig 3, freeze panes at the first
data row of the first table.  No formulas -- values only.

The model is labelled FQGNN throughout.  Sheets that enumerate MultitaskAIS
carry a one-sided note: it is a forward-only forecaster reference, not a two-sided peer.

Run:  python paper_figures_tables/build_source_data.py
"""
import os

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _paths import FIGDATA as FD, OUT as OUT_DIR, RESULTS as TA  # noqa: E402  (see _paths.py)
OUT = os.path.join(OUT_DIR, "Source_Data.xlsx")

FQ_RUN = os.path.join(TA, "runs", "fqnative_opmstack_5seed")   # the deployed run: the one the figures plot
# The one-sided ablation is read from the Op-M stack run (oneside_sobolev_5seed_opm); the sheet
# mirrors fig2_build.py, which assembles per-seed test_buckets_clean.csv (the run-root all_seeds
# aggregate holds the in-runner, refreshed-table evaluation -- never read it here).
ONESIDE_RUN = os.path.join(TA, "runs", "oneside_sobolev_5seed_opm")


def oneside_buckets_clean():
    return pd.concat(
        [pd.read_csv(os.path.join(ONESIDE_RUN, f"seed{s}", "test_buckets_clean.csv"))
         .assign(seed=s) for s in range(5)],
        ignore_index=True)
TELE = os.path.join(TA, "runs", "telemetry_infamily_native")

HEADER_FILL = PatternFill("solid", start_color="FF1F3864")
HEADER_FONT = Font(bold=True, color="FFFFFFFF")
TITLE_FONT = Font(bold=True, size=12)
RED_SRC_FONT = Font(bold=True, color="FFCC0000")

ONE_SIDED_NOTE = ("Note: MultitaskAIS is a forward-only (one-sided) forecaster reference -- it "
                  "only sees context BEFORE each gap, unlike the two-sided imputation models -- "
                  "and is included for completeness, not as a like-for-like peer.")

# columns treated as integers (General format); every other numeric cell gets FLOAT_FMT
INT_COLS = {"node_idx", "t_sec", "is_held", "gap_id", "mmsi", "point_idx", "is_gap_segment",
            "seed", "n", "epoch", "k_index", "n_seeds", "number", "trial", "n_params",
            "n_groups", "n_group_inner_layers", "d_model", "d_inner", "n_head", "d_k", "d_v",
            "batch_size", "gap_h", "aps", "n_params_trainable", "n_params_frozen",
            "params_batch_size", "params_channels", "params_diffusion_embedding_dim",
            "params_layers", "params_nheads", "params_num_steps"}
FLOAT_FMT = "0.0000"
WIN_FMT = '0.0"%"'


class SheetWriter:
    def __init__(self, wb, name, widths):
        self.ws = wb.create_sheet(name)
        for i, w in enumerate(widths, 1):
            self.ws.column_dimensions[get_column_letter(i)].width = w
        self.r = 0
        self.first_data_row = None

    def blank(self, n=1):
        self.r += n

    def title(self, text):
        self.r += 1
        self.ws.cell(row=self.r, column=1, value=text).font = TITLE_FONT

    def note(self, text, font=None):
        self.r += 1
        c = self.ws.cell(row=self.r, column=1, value=text)
        c.font = font or Font(italic=True)

    def table(self, df, win_cols=("win_pct",)):
        """Header row + data rows from a DataFrame."""
        self.r += 1
        for c, h in enumerate(df.columns, 1):
            cell = self.ws.cell(row=self.r, column=c, value=str(h))
            cell.fill = HEADER_FILL
            cell.font = HEADER_FONT
        self.r += 1
        if self.first_data_row is None:
            self.first_data_row = self.r
        cols = list(df.columns)
        for row in df.itertuples(index=False):
            for c, (col, v) in enumerate(zip(cols, row), 1):
                if pd.isna(v):
                    continue
                cell = self.ws.cell(row=self.r, column=c)
                if isinstance(v, str):
                    cell.value = v
                    continue
                if col in INT_COLS and float(v) == int(v):
                    cell.value = int(v)
                    continue
                cell.value = float(v)
                cell.number_format = WIN_FMT if col in win_cols else FLOAT_FMT
            self.r += 1
        self.r -= 1  # r points at last written row

    def finish(self):
        if self.first_data_row:
            self.ws.freeze_panes = f"A{self.first_data_row}"


# ----------------------------------------------------------------------------- load shared data
hero_idx = pd.read_csv(os.path.join(FD, "hero_gaps_index.csv"))


def hero_df(bucket):
    row = hero_idx[hero_idx["bucket"] == bucket].iloc[0]
    path = os.path.join(FD, f"hero_gap_{row['bucket']}_{int(row['gap_id'])}.csv")
    return pd.read_csv(path), row


comb = pd.read_csv(os.path.join(FD, "combined_buckets_allmodels.csv"))
pivot = pd.read_csv(os.path.join(FD, "pivot_ratio_mean.csv"))
per_gap0 = pd.read_csv(os.path.join(FQ_RUN, "seed0", "per_gap.csv"))
per_gap0_clean = pd.read_csv(os.path.join(TA, "runs", "fqnative_opmstack_5seed_clean",
                                          "seed0", "per_gap.csv"))
fq_epochs = pd.concat([pd.read_csv(os.path.join(FQ_RUN, f"seed{s}", "epochs.csv")).assign(seed=s)
                       for s in range(5)], ignore_index=True)[["seed", "epoch", "val_ratio"]]
tele = pd.read_csv(os.path.join(TELE, "telemetry_infamily.csv"))
hero_mid, hero_mid_row = hero_df("0.05-0.10")

wb = Workbook()

# =============================================================================== README
ws = wb.active
ws.title = "README"
for i, w in enumerate([60, 24, 60], 1):
    ws.column_dimensions[get_column_letter(i)].width = w
ws.cell(row=1, column=1, value="Source Data -- Table of Contents").font = TITLE_FONT
ws.cell(row=3, column=1, value=(
    "Panels that are pure schematic diagrams with no underlying numeric data are noted as such "
    "and have no data table in this file (only Fig. 1a's pipeline diagram and Extended Data "
    "Fig. 1a's protocol diagram are schematic -- their embedded real-data thumbnails ARE "
    "included below).")).font = Font(italic=True)
ws.cell(row=5, column=1, value=(
    "All error values are in nautical miles (NM) unless stated otherwise. Ratio = model median "
    "error / GCD median error (a ratio below 1 means the model beats the great-circle-distance "
    "baseline). All headline test results are computed on n=770 shared test gaps (out of 774 "
    "vessel-disjoint test gaps; 4 gaps have undefined/infinite sinuosity and are excluded from "
    "the shared-gaps comparison, per Methods).")).font = Font(italic=True)
ws.cell(row=7, column=1, value=(
    "MultitaskAIS (wherever it appears) is a forward-only (one-sided) forecaster reference: it "
    "only sees context BEFORE each gap, unlike the two-sided imputation models, and is included "
    "for completeness, not as a like-for-like peer. HABIT is a deterministic route-library "
    "lookup (single run, no seeds).")).font = Font(italic=True)
toc = [
    ("Fig 1", "Figure 1", "FQ operator family + composition into FQ-GNN: hero-gap thumbnail "
     "data, RFF vs learned-FQ spectrum comparison, in-family training telemetry"),
    ("Fig 2", "Figure 2", "Two-sided reconstruction on Arctic AIS: map tracks/gaps, one hero "
     "gap, GCD error vs sinuosity, per-bucket per-model medians (incl. HABIT, MH-GIN and the "
     "one-sided MultitaskAIS reference), one-sided ablation"),
    ("Fig 3", "Figure 3", "All-model same-gaps comparison on the standardized 2-h cell: "
     "combined per-seed per-bucket table, seed-mean pivot table, per-gap scatter seed 0 "
     "(vessel-clustered classical-family statistics: Extended_Data_Table.xlsx)"),
    ("Fig 4", "Figure 4", "FQ operator family is load-bearing: ablation ladder, FQ-GNN/RFF-GNN "
     "validation curves (5 seeds), spectral rigidity S(k)"),
    ("Fig 5", "Figure 5", "Reconstructions of three real test gaps of increasing curvature: "
     "straight, moderate, sharp"),
    ("Fig 6", "Figure 6", "Robustness: gap-length sweep (incl. HABIT, MH-GIN, one-sided "
     "MultitaskAIS), anchor-budget sweep, seed variability pointer, "
     "parameter/compute audit"),
    ("Extended Data Fig 1", "Extended Data Figure 1", "Protocol and leak audit: split counts "
     "note, example track thumbnail, focused bucket/n/GCD-median table"),
    ("Extended Data Fig 2", "Extended Data Figure 2", "Full 5-seed training curves for FQ-GNN, "
     "RFF-GNN, BiGRU, SAITS, CSDI"),
    ("Extended Data Fig 3", "Extended Data Figure 3", "Baseline hyperparameter-tuning "
     "transparency: CSDI and SAITS optuna trial tables, seed-instability pointer"),
    ("Extended Data Fig 4", "Extended Data Figure 4", "Foundation diagnostics: in-family "
     "telemetry, initial/trained comb, per-stem FQ energy and RFF readout weight"),
    ("Extended Data Fig 5", "Extended Data Figure 5", "Per-bucket ratios at every gap horizon "
     "on both preprocessing tracks for the five AIS-capable models (drawn values of the ED5 "
     "cell grid)"),
]
hr = 9
for c, h in enumerate(["Sheet name", "Manuscript figure", "Description"], 1):
    cell = ws.cell(row=hr, column=c, value=h)
    cell.fill = HEADER_FILL
    cell.font = HEADER_FONT
for i, (a, b, c_) in enumerate(toc, hr + 1):
    ws.cell(row=i, column=1, value=a)
    ws.cell(row=i, column=2, value=b)
    ws.cell(row=i, column=3, value=c_)
ws.freeze_panes = f"A{hr + 1}"

# =============================================================================== Fig 1
S = SheetWriter(wb, "Fig 1", [40, 12, 12, 11, 13, 16, 17, 16, 16, 11, 13, 12, 11, 12])
S.title("Fig 1 -- The Fourier-Quasicrystal operator family and its composition into FQ-GNN")
S.blank(2)
S.title("Panel a -- schematic (pipeline diagram); embedded real-data hero-gap thumbnail "
        "tabulated below")
S.blank(1)
S.table(hero_mid)
S.blank(1)
S.title("Panel b -- spectrum comparison: Random Fourier control")
S.blank(1)
S.table(pd.read_csv(os.path.join(FD, "rff_spectrum.csv")))
S.blank(1)
S.title("Panel b -- spectrum comparison: Learned FQ comb")
S.blank(1)
S.table(pd.read_csv(os.path.join(FD, "fq_spectrum_trained.csv")))
S.blank(1)
S.title("Panel c -- in-family training telemetry")
S.blank(1)
S.table(tele[["epoch", "min_minor", "val_ratio"]])
S.finish()

# =============================================================================== Fig 2
S = SheetWriter(wb, "Fig 2", [40, 13, 11, 11, 13, 16, 13, 11, 13, 11, 13, 12, 11, 12])
S.title("Fig 2 -- Two-sided reconstruction on Arctic AIS and where the great-circle geodesic "
        "fails")
S.blank(2)
S.title("Panel a -- map: vessel tracks")
S.blank(1)
S.table(pd.read_csv(os.path.join(FD, "map_tracks.csv")))
S.blank(1)
S.title("Panel a -- map: gaps index")
S.blank(1)
S.table(pd.read_csv(os.path.join(FD, "map_gaps_index.csv")))
S.blank(1)
S.title("Panel b -- task on one real gap (the gt0.25 hero, gap 594)")
S.blank(1)
S.table(hero_df("gt0.25")[0])
S.blank(1)
S.title("Panel c -- GCD error vs sinuosity")
S.blank(1)
S.table(per_gap0_clean[["mmsi", "gap_min", "sinuosity", "gcd_med"]])
S.blank(1)
S.title("Panel d -- median error per bucket, all models (master table; used by multiple "
        "panels, included in full)")
S.note(ONE_SIDED_NOTE)
S.blank(1)
S.table(comb)
S.blank(1)
S.title("Panel e -- two-sided vs one-sided")
S.blank(1)
S.note("Two-sided FQGNN values: see Panel d table above (same combined_buckets_allmodels.csv).")
S.blank(1)
S.note("One-sided ablation (per-seed clean exact-table re-evals, as drawn)")
S.table(oneside_buckets_clean())
S.finish()

# =============================================================================== Fig 3
# (Fig 3 = the all-model same-gaps comparison on the standardized 2-h cell;
#  the classical-family 8-cell + vessel-clustered tables live in Extended_Data_Table.xlsx)
comb2h = pd.read_csv(os.path.join(FD, "combined_buckets_std2h.csv"))
pivot2h = pd.read_csv(os.path.join(FD, "pivot_ratio_std2h.csv"))
pg2h = pd.read_csv(os.path.join(TA, "runs", "gap_sweep_sobolev_fd_opm", "standard_process",
                                "gap_2h", "fqgnn", "seed0", "per_gap_clean.csv"))
pg2h = pg2h[(pg2h.gcd_med > 0) & pg2h.sinuosity.notna()
            & (pg2h.sinuosity.abs() < float("inf"))]

npiv2h = len(pivot2h.columns)
S = SheetWriter(wb, "Fig 3", [40, 10, 11, 11, 13, 14, 12, 13, 13, 17, 12, 14, 15,
                              13, 14, 14, 18] + [14] * max(0, npiv2h - 17))
S.title("Fig 3 -- FQ-GNN is the only model decisively below the geodesic floor on the "
        "standardized 2-h cell")
S.blank(1)
S.note("Panels a, b, c draw from the same combined per-seed table (included once below); the "
       "heatmap values in panel e also appear in the seed-mean pivot table (included second).")
S.note(ONE_SIDED_NOTE)
S.blank(1)
S.title("Table 1 -- Per-seed, per-bucket results (panels a, b, c)")
S.blank(1)
S.table(comb2h)
S.blank(1)
S.title("Table 2 -- Seed-mean pivot table (panel e)")
S.blank(1)
S.table(pivot2h)
S.blank(1)
S.title("Table 3 -- Per-gap head-to-head, seed 0 (panel d)")
S.blank(1)
S.table(pg2h[["mmsi", "gcd_med", "model_med", "sinuosity"]])
S.blank(1)
S.note("Vessel-clustered FQGNN-vs-Hermite statistics for the standardized track (quoted in "
       "Results): Extended_Data_Table.xlsx, Clustered_Stats_Cells sheet.")
S.finish()

# =============================================================================== Fig 4
S = SheetWriter(wb, "Fig 4", [40, 33, 11, 12, 11, 15, 10, 10, 36])
S.title("Fig 4 -- The FQ operator family is load-bearing")
S.blank(2)
S.title("Panel a")
S.blank(1)
S.note("See Fig 2 Panel d table (combined_buckets_allmodels.csv) for FQGNN and RFFGNN "
       "per-bucket ratios used in the matched-control slope chart.")
S.blank(1)
S.title("Panel b -- ablation ladder")
S.blank(1)
S.table(pd.read_csv(os.path.join(FD, "ablation_ladder.csv")), win_cols=("win_mean", "win_std"))
S.blank(1)
S.title("Panel c -- training dynamics: FQ-GNN validation curves (5 seeds)")
S.blank(1)
S.table(fq_epochs[["seed", "epoch", "val_ratio"]])
S.blank(1)
S.title("Panel c -- training dynamics: RFF-GNN validation curves (5 seeds)")
S.blank(1)
rff_ep = pd.read_csv(os.path.join(TA, "baselines", "RFFGNN", "runs_5seed_gaps",
                                  "all_seeds_epochs.csv"))
S.table(rff_ep[["seed", "epoch", "val_ratio"]])
S.blank(1)
S.title("Panel d")
S.blank(1)
S.note("See Fig 1 Panel b table above for the same rff_spectrum.csv / fq_spectrum_trained.csv "
       "data (reused for this panel).")
S.blank(1)
S.title("Panel e -- spectral rigidity: sk_fq.csv")
S.blank(1)
S.table(pd.read_csv(os.path.join(FD, "sk_fq.csv")))
S.blank(1)
S.title("Panel e -- spectral rigidity: sk_rff.csv")
S.blank(1)
S.table(pd.read_csv(os.path.join(FD, "sk_rff.csv")))
S.finish()

# =============================================================================== Fig 5
S = SheetWriter(wb, "Fig 5", [40, 10, 10, 11, 13, 11, 13, 11, 13, 11, 13, 12, 11, 12])
S.title("Fig 5 -- Reconstructions of three real test gaps of increasing curvature")
S.blank(2)
for label, bucket in [("Straight", "0.00-0.02"), ("Moderate", "0.05-0.10"),
                      ("Sharp", "gt0.25")]:
    df, row = hero_df(bucket)
    S.title(f"{label} gap (sinuosity ~{row['sinuosity']:g})")
    S.blank(1)
    S.table(df)
    S.blank(1)
S.finish()

# =============================================================================== Fig 6
S = SheetWriter(wb, "Fig 6", [40, 30, 17, 20, 25, 19, 40, 10])
S.title("Fig 6 -- Robustness across gap length, anchor budget, seed and preprocessing")
S.blank(2)
S.title("Panel a -- gap-length sweep")
S.note(ONE_SIDED_NOTE)
S.blank(1)
S.table(pd.read_csv(os.path.join(FD, "gap_sweep.csv")))
S.blank(1)
S.title("Panel b -- anchor budget sweep")
S.blank(1)
S.table(pd.read_csv(os.path.join(FD, "anchor_sweep.csv")))
S.blank(1)
S.title("Panel c")
S.blank(1)
S.note("See Fig 2 Panel d table (combined_buckets_allmodels.csv), OVERALL bucket rows only, "
       "for the per-seed box-strip values.")
S.blank(1)
S.title("Panel d -- parameter/compute audit")
S.blank(1)
S.table(pd.read_csv(os.path.join(FD, "params_audit.csv")))
S.finish()

# =============================================================================== Extended Data Fig 1
S = SheetWriter(wb, "Extended Data Fig 1", [40, 13, 11, 11, 13, 16])
S.title("Extended Data Fig 1 -- Protocol and leak audit")
S.blank(2)
S.title("Panel a -- schematic protocol diagram")
S.blank(1)
S.note("Schematic pipeline; the vessel-disjoint split counts are 5,350 train / 635 val / 774 "
       "test (hardcoded in the figure script, not read from a CSV).")
S.blank(1)
S.note("Example track thumbnail data (same file as Fig 2 Panel a)")
S.table(pd.read_csv(os.path.join(FD, "map_tracks.csv")))
S.blank(1)
S.title("Panel b -- bucket / n / GCD median (focused columns)")
S.blank(1)
S.table(pivot[["bucket", "n", "gcd_med"]])
S.finish()

# =============================================================================== Extended Data Fig 2
S = SheetWriter(wb, "Extended Data Fig 2", [40, 10, 15, 11])
S.title("Extended Data Fig 2 -- Full training curves for every model across five seeds")
S.blank(2)
S.title("FQ-GNN")
S.blank(1)
S.table(fq_epochs[["seed", "epoch", "val_ratio"]])
S.blank(1)
curve_specs = [
    ("RFF-GNN", "RFFGNN", ["seed", "epoch", "val_ratio"]),
    ("BiGRU", "BiGRU", ["seed", "epoch", "val_median_nm", "val_ratio"]),
    ("SAITS", "SAITS", ["seed", "epoch", "val_median_nm", "val_ratio"]),
    ("CSDI (validation metric = denoising loss, NOT comparable to the ratio metric used by "
     "other models)", "CSDI", ["seed", "epoch", "val_loss"]),
]
for label, dirname, cols in curve_specs:
    df = pd.read_csv(os.path.join(TA, "baselines", dirname, "runs_5seed_gaps",
                                  "all_seeds_epochs.csv"))
    S.title(label)
    S.blank(1)
    S.table(df[cols])
    S.blank(1)
S.finish()

# =============================================================================== Extended Data Fig 3
S = SheetWriter(wb, "Extended Data Fig 3", [40, 20, 22, 22, 18, 18, 22, 14, 26, 12, 20, 12, 20,
                                            20, 12, 10])
S.title("Extended Data Fig 3 -- Baseline hyperparameter-tuning transparency")
S.blank(2)
S.title("Panel a -- CSDI optuna trials (trial number, validation objective = median NM)")
S.blank(1)
csdi_trials = pd.read_csv(os.path.join(TA, "baselines", "CSDI", "hp_optuna", "gap_6h",
                                       "trials.csv"))
S.note(f"Source: task_alpha/baselines/CSDI/hp_optuna/gap_6h/trials.csv "
       f"({len(csdi_trials)} completed Optuna trials)", font=RED_SRC_FONT)
S.blank(1)
S.table(csdi_trials)
S.blank(1)
S.title("Panel b -- SAITS optuna trials (validation objective = median NM)")
S.blank(1)
saits_trials = pd.read_csv(os.path.join(TA, "baselines", "SAITS", "hp_saits", "trials.csv"))
S.note(f"Source: task_alpha/baselines/SAITS/hp_saits/trials.csv "
       f"({len(saits_trials)} Optuna trials)", font=RED_SRC_FONT)
S.blank(1)
S.table(saits_trials)
S.blank(1)
S.title("Panel c -- tuned-baseline seed stability")
S.blank(1)
S.note("See Fig 2 Panel d table (combined_buckets_allmodels.csv), OVERALL bucket rows, for "
       "the per-seed test ratios behind the seed-stability panel.")
S.finish()

# =============================================================================== Extended Data Fig 4
S = SheetWriter(wb, "Extended Data Fig 4", [40, 20, 24, 15, 13, 12, 12, 12, 12, 13, 12])
S.title("Extended Data Fig 4 -- Foundation diagnostics during training")
S.blank(2)
S.title("Panels a, b -- in-family telemetry (full)")
S.blank(1)
S.table(tele)
S.blank(1)
S.title("Panel c -- Initial comb")
S.blank(1)
S.table(pd.read_csv(os.path.join(TELE, "comb_init.csv")))
S.blank(1)
S.title("Panel c -- Trained comb")
S.blank(1)
S.table(pd.read_csv(os.path.join(TELE, "comb_final.csv")))
S.blank(1)
S.title("Panel d -- FQ comb per-stem energy (abs_c_sq)")
S.blank(1)
S.table(pd.read_csv(os.path.join(FD, "fq_spectrum_trained.csv"))[["k_index", "abs_c_sq"]])
S.blank(1)
S.title("Panel d -- RFF per-stem readout weight (readout_w)")
S.blank(1)
S.table(pd.read_csv(os.path.join(FD, "rff_spectrum.csv"))[["k_index", "readout_w"]])
S.finish()

# =============================================================================== Extended Data Fig 5
S = SheetWriter(wb, "Extended Data Fig 5", [16, 8, 12, 14, 12, 12, 12, 10])
S.title("Extended Data Fig 5 -- Per-bucket ratios at every gap horizon on both preprocessing "
        "tracks (drawn values of the cell grid)")
S.blank(1)
S.note(ONE_SIDED_NOTE)
S.blank(1)
S.table(pd.read_csv(os.path.join(FD, "ed5_horizon_buckets.csv")))
S.finish()

wb.save(OUT)
print("Wrote", OUT)
print("Sheets:", wb.sheetnames)

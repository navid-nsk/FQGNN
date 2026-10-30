"""Build Extended_Data_Table.xlsx from the archived run outputs, written to FQGNN_FIG_OUT.

Deterministic, reproducible builder.  Reads ONLY (paper/figdata/ below is FQGNN_FIGDATA and
task_alpha/ is FQGNN_RESULTS_ROOT; see _paths.py):
  paper/figdata/ed_table_literature.csv          (literature overview: model families, structural
                                                  limitations, code availability -- Extended Data
                                                  Table 1, referenced in the Introduction)
  paper/figdata/combined_buckets_allmodels.csv   (per-model, per-seed, per-bucket test results)
  paper/figdata/pivot_ratio_mean.csv             (seed-mean bucket pivot)
  paper/figdata/params_audit.csv                 (parameter / compute audit; consumes however
                                                  many model rows the CSV has)
  paper/figdata/clustered_stats_cells.csv        (per-cell vessel-clustered FQGNN-vs-Hermite
                                                  bootstrap statistics -- Clustered statistics by cell
                                                  sheet)
  task_alpha/runs/fqnative_sobolev_5seed/        (FQGNN 5-seed reference: summary + per-seed epochs)
  task_alpha/baselines/{RFFGNN,BiGRU,SAITS,CSDI}/runs_5seed_gaps/  (summary + per-seed epochs)
  task_alpha/runs/gap_sweep/our_processed/gap_6h/{habit,mhgin,multitaskais}/  (summary + epochs)
  task_alpha/runs/gap_sweep/{our_processed,standard_process}/gap_*/{hermite,quintic,
      constrained_gp,ctrv_kalman}/summary.csv     (Classical family sheet, deterministic cells)
  task_alpha/runs/controls/{grid,gauss,uniform,shuffle,learned,mlp}_blend/summary.csv
                                                  (Frequency controls sheet, 5 seeds each)
  task_alpha/runs/gap_sweep_sobolev{,_mlp}/our_processed/gap_12h/fqgnn/summary.csv
  task_alpha/runs/gap_sweep_sobolev{,_mlp}/standard_process/gap_2h/fqgnn/summary.csv
                                                  (Encoder control sheet; the 6h MLP arm is
                                                  runs/controls/mlp_blend)

Workbook conventions:
  Calibri font, dark-blue header rows (1F3864, white bold), thin borders on data cells,
  FQGNN rows pink (FCE4EC) + bold, GCD rows gray (F2F2F2) + italic,
  MEAN/STD rows of other models gray (EDEDED) + bold, =AVERAGE()/=STDEVP() formulas.
  STD rows use the POPULATION standard deviation (STDEVP) so they reproduce the
  summary.csv STD rows of each run.

The model is labelled FQGNN throughout.  MultitaskAIS always carries the
"(one-sided)" marker: it is a forward-only forecaster reference, not a two-sided peer.

Run:  python paper_figures_tables/build_extended_data_table.py --opm
  --opm (the Op-M stack paper model): FQGNN rows that are read from run dirs
        (Validation results / Training summary / Encoder control) come from the Op-M stack
        runs -- the 5-seed reference from task_alpha/runs/fqnative_opmstack_5seed
        (training-side per-seed rows from summary_pA/pB/pC.csv + seed<k>/epochs.csv), and
        the encoder-control cells from task_alpha/runs/gap_sweep_sobolev{,_fd}_opm, with
        test values taken from the OFFICIAL exact-table re-eval files (summary_clean.csv).
        The standard-2h encoder arms use the fd boundary jet on that track (fd_opm GNN vs
        fd_mlp MLP), matching the paper's fd-on-standard convention (assemble_gap_sweep).
        ALL OTHER MODELS are read from the sources listed above.  Default (no flag) = the
        sobolev sources.
"""
import argparse
import os
import sys

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _paths import FIGDATA, OUT as OUT_DIR, RESULTS as TA  # noqa: E402  (see _paths.py)
OUT = os.path.join(OUT_DIR, "Extended_Data_Table.xlsx")

ap = argparse.ArgumentParser(description="rebuild Extended_Data_Table.xlsx (written to FQGNN_FIG_OUT)")
ap.add_argument("--opm", action="store_true",
                help="FQGNN run-dir sources from the Op-M stack model "
                     "(runs/fqnative_opmstack_5seed + gap_sweep_sobolev{,_fd}_opm cells, "
                     "official exact-table re-eval test values); all other models unchanged")
ARGS = ap.parse_args()
OPM = ARGS.opm

FQ_RUN_SOBOLEV = os.path.join(TA, "runs", "fqnative_sobolev_5seed")
FQ_RUN_OPM = os.path.join(TA, "runs", "fqnative_opmstack_5seed")
FQ_RUN = FQ_RUN_OPM if OPM else FQ_RUN_SOBOLEV
BASE_RUNS = {
    "RFFGNN": os.path.join(TA, "baselines", "RFFGNN", "runs_5seed_gaps"),
    "BiGRU":  os.path.join(TA, "baselines", "BiGRU",  "runs_5seed_gaps"),
    "SAITS":  os.path.join(TA, "baselines", "SAITS",  "runs_5seed_gaps"),
    "CSDI":   os.path.join(TA, "baselines", "CSDI",   "runs_5seed_gaps"),
}
NEW_RUNS = {
    "HABIT":        os.path.join(TA, "runs", "gap_sweep", "our_processed", "gap_6h", "habit"),
    "MH-GIN":       os.path.join(TA, "runs", "gap_sweep", "our_processed", "gap_6h", "mhgin"),
    "MultitaskAIS": os.path.join(TA, "runs", "gap_sweep", "our_processed", "gap_6h", "multitaskais"),
}

# ----------------------------------------------------------------------------- style constants
HEADER_FILL = PatternFill("solid", start_color="FF1F3864")
PINK_FILL = PatternFill("solid", start_color="FFFCE4EC")     # FQGNN (our model)
GRAY_FILL = PatternFill("solid", start_color="FFF2F2F2")     # GCD reference
MSGRAY_FILL = PatternFill("solid", start_color="FFEDEDED")   # MEAN/STD of baselines
THIN = Side(style="thin")
DATA_BORDER = Border(top=THIN, bottom=THIN)
HEADER_FONT = Font(bold=True, color="FFFFFFFF")
HEADER_AL = Alignment(horizontal="center", wrap_text=True)

ONE_SIDED = "MultitaskAIS (one-sided)"
ONE_SIDED_NOTE = ("MultitaskAIS is a forward-only (one-sided) forecaster reference: it only sees "
                  "context BEFORE each gap, unlike the two-sided imputation models above; it is "
                  "reported for completeness, not as a like-for-like peer.")
STD_NOTE = ("STD rows use the population standard deviation (=STDEVP), matching the STD row of "
            "each run's summary.csv.")

# fixed public row order (README "Models included"), Hermite last, matching the figdata
# column order
MODEL_ORDER = ["GCD", "FQGNN", "RFFGNN", "BiGRU", "SAITS", "CSDI", "BF-BiLSTM",
               "HABIT", "MH-GIN", "MultitaskAIS", "Hermite"]
MULTI_SEED = {"FQGNN", "RFFGNN", "BiGRU", "SAITS", "CSDI", "MH-GIN", "MultitaskAIS"}
SEED_LABEL = {"GCD": "reference", "BF-BiLSTM": "zero-shot", "HABIT": "deterministic",
              "Hermite": "deterministic"}


def disp_name(model):
    """Public label written into the Model column."""
    return ONE_SIDED if model == "MultitaskAIS" else model


def style_for(model, is_meanstd=False):
    """(fill, font) for a data row of this model."""
    if model == "FQGNN":
        return PINK_FILL, Font(bold=True)
    if model == "GCD":
        return GRAY_FILL, Font(italic=True)
    if is_meanstd:
        return MSGRAY_FILL, Font(bold=True)
    return None, Font()


def put(ws, r, c, value, fill=None, font=None, fmt=None, border=True, align=None):
    cell = ws.cell(row=r, column=c)
    cell.value = value
    if fill is not None:
        cell.fill = fill
    if font is not None:
        cell.font = font
    if fmt is not None:
        cell.number_format = fmt
    if border:
        cell.border = DATA_BORDER
    if align is not None:
        cell.alignment = align
    return cell


def write_header(ws, r, headers):
    for c, h in enumerate(headers, 1):
        put(ws, r, c, h, fill=HEADER_FILL, font=HEADER_FONT, align=HEADER_AL)


def sheet_title(ws, text, ncols):
    ws.cell(row=1, column=1, value=text).font = Font(bold=True, size=13)
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncols)


def set_widths(ws, widths):
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def merged_note(ws, r, ncols, text, italic=True, bold=False):
    cell = ws.cell(row=r, column=1, value=text)
    cell.font = Font(italic=italic, bold=bold)
    cell.alignment = Alignment(wrap_text=True)
    ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=ncols)


def normkey(name):
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def na(v, alt="N/A"):
    return alt if pd.isna(v) else v


# ----------------------------------------------------------------------------- load sources
lit = pd.read_csv(os.path.join(FIGDATA, "ed_table_literature.csv"))
comb = pd.read_csv(os.path.join(FIGDATA, "combined_buckets_allmodels.csv"))
pivot = pd.read_csv(os.path.join(FIGDATA, "pivot_ratio_mean.csv"))
params = pd.read_csv(os.path.join(FIGDATA, "params_audit.csv"))
params["_key"] = params["model"].map(normkey)

BUCKET_ORDER = ["OVERALL", "0.00-0.02", "0.02-0.05", "0.05-0.10", "0.10-0.25", ">0.25"]


def run_summary(model):
    """Per-seed summary rows (seed 0..4) for a trained model, from its run dir."""
    if model == "FQGNN" and OPM:
        # Op-M stack reference run: training-side per-seed rows live in the part summaries
        # (pA = seeds 0-1, pB = seeds 2-3, pC = seed 4); test values come from figdata CSVs.
        df = pd.concat([pd.read_csv(os.path.join(FQ_RUN, f"summary_p{p}.csv"))
                        for p in "ABC"], ignore_index=True)
    else:
        d = FQ_RUN if model == "FQGNN" else BASE_RUNS.get(model, NEW_RUNS.get(model))
        df = pd.read_csv(os.path.join(d, "summary.csv"))
    df = df[df["seed"].astype(str).str.isdigit()].copy()
    df["seed"] = df["seed"].astype(int)
    return df.sort_values("seed")


def summary_seed_rows(path):
    """Numeric-seed rows of a run summary.csv (drops MEAN/STD rows), sorted by seed."""
    df = pd.read_csv(path)
    df = df[df["seed"].astype(str).str.isdigit()].copy()
    df["seed"] = df["seed"].astype(int)
    return df.sort_values("seed")


def opm_fq_cell_rows(cell_dir):
    """--opm FQGNN sweep cell: OFFICIAL exact-table re-eval test values (summary_clean.csv)
    merged with the in-run epochs_trained / best_val (summary.csv), by seed."""
    clean = summary_seed_rows(os.path.join(cell_dir, "summary_clean.csv"))
    clean = clean.rename(columns={"n": "n_gaps"})
    inrun = summary_seed_rows(os.path.join(cell_dir, "summary.csv"))
    keep = ["seed", "epochs_trained"] + (["best_val"] if "best_val" in inrun.columns else [])
    return clean.merge(inrun[keep], on="seed").sort_values("seed")


def final_train_losses(model):
    """{seed: last-epoch train_loss} from seedN/epochs.csv."""
    d = FQ_RUN if model == "FQGNN" else BASE_RUNS.get(model, NEW_RUNS.get(model))
    out = {}
    for s in range(5):
        p = os.path.join(d, f"seed{s}", "epochs.csv")
        if not os.path.exists(p):
            continue
        ep = pd.read_csv(p)
        tl = ep["train_loss"].dropna()
        out[s] = float(tl.iloc[-1]) if len(tl) else None
    return out


def train_time(model):
    """Per-model train_s_per_epoch from params_audit.csv (may be absent)."""
    hit = params[params["_key"] == normkey(model)]
    if len(hit) == 0:
        return None
    v = hit.iloc[0]["train_s_per_epoch"]
    return None if pd.isna(v) else float(v)


wb = Workbook()

# =============================================================================== README
ws = wb.active
ws.title = "README"
set_widths(ws, [30, 100])
ws.cell(row=1, column=1, value="Extended Data Tables").font = Font(bold=True, size=16)
ws.cell(row=2, column=1,
        value="FQ-GNN trajectory reconstruction: literature overview (Extended Data Table 1) and "
              "baseline train/validation/test summary (Extended Data Table 2)"
        ).font = Font(italic=True, size=12)

r = 4
ws.cell(row=r, column=1, value="Sheet contents").font = Font(bold=True)
sheet_desc = [
    ("README", "This overview sheet: sheet map, model list, exclusion note, highlighting legend."),
    ("Literature overview", "EXTENDED DATA TABLE 1 (referenced in the Introduction): the "
     "trajectory-reconstruction literature by family (1 classical / 2 deep imputers / 3 generic "
     "spectral / 4 AIS-native), each model's core mechanism, its structural limitation for "
     "two-sided curved-gap reconstruction, public-code availability (verified 2026-07-16), and "
     "the ratio vs GCD on the 770 shared test gaps where tested here."),
    ("Overall test results", "EXTENDED DATA TABLE 2 (this and all following sheets). One row per "
     "(model, seed) test-set result on the OVERALL bucket "
     "(all 770 shared test gaps), with MEAN/STD rows per model."),
    ("Bucket test results", "Full per-model, per-seed, per-curvature-bucket test results, plus a "
     "seed-mean summary pivot (bucket x model)."),
    ("Validation results", "Best validation metric reached and epochs trained, per model per seed "
     "(metric type varies by model; see Metric_type column)."),
    ("Training summary", "Final-epoch training loss per seed, plus per-model training wall-clock "
     "time per epoch."),
    ("Parameters and compute", "Trainable/frozen parameter counts, overall test ratio, and "
     "inference/training compute cost per model."),
    ("Classical family", "Classical interpolation family (cubic Hermite spline, quintic "
     "minimum-jerk Hermite, chord-mean GP regression, CTRV/IMM Kalman) evaluated on every "
     "gap-sweep cell (both preprocessings x gap lengths). Deterministic single runs "
     "(no seeds)."),
    ("Frequency controls", "Matched controls for the FQ frequency set and the GNN conditioning "
     "channel (grid / gauss / uniform / shuffle / learned frequency sources and the no-GNN "
     "raw-anchor conditioning control), 5 seeds each, main 6h cell, training protocol identical "
     "to FQGNN."),
    ("Encoder control", "Conditioning-encoder experiment on the our-12h cell (GNN vs raw-anchor "
     "MLP conditioning), per seed with MEAN/STD, plus the 6h and standard-2h arms as "
     "supplementary summary rows."),
    ("Clustered statistics by cell", "Vessel-clustered bootstrap statistics (B=10,000 vessel "
     "resamples, percentile CIs) for FQGNN vs the cubic-Hermite classical control, per "
     "gap-sweep cell: ratio difference (FQ minus Hermite), both ratios, per-gap win rate, "
     "and a vessel-level Wilcoxon test (exact p, plus the Holm-adjusted p across the 8 "
     "cells)."),
    ("Native-cadence imputers", "The regular-grid deep imputers trained NATIVELY on each "
     "standardized 10-min cell (their own uniform-cadence regime), vanilla protocol, five "
     "seeds per cell; per model x cell x seed with MEAN/STD rows."),
    ("2019 holdout", "Zero-shot temporal holdout: the five FROZEN 2016-18 checkpoints (no "
     "retraining) evaluated on July-August 2019 under the identical pipeline (2,110 gaps, 515 "
     "vessels), with the deterministic Nguyen cubic-Hermite baseline and the GCD reference on "
     "the identical gaps; per model x seed x scope (all gaps / unseen-vessel subset)."),
    ("Reference certificate", "Reference-only substitution certificate on all 8 cells: the "
     "cell median of per-gap median errors for the deployed chart-linear reference vs the "
     "exact fp64 great-circle (slerp) reference between the same bracketing anchors at "
     "matched time fractions, the % shift, and the common multiplicative factor the "
     "substitution applies to every model ratio of that cell (Supplementary Section S4.6; "
     "script task_alpha/_diag_ref_cert_8cells.py)."),
]
for name, desc in sheet_desc:
    r += 1
    ws.cell(row=r, column=1, value=name).font = Font(bold=True)
    ws.cell(row=r, column=2, value=desc).alignment = Alignment(wrap_text=True)

r += 2
ws.cell(row=r, column=1, value="Models included (fixed row order in every table)").font = Font(bold=True)
model_desc = [
    ("GCD", "Great-circle-distance / linear-interpolation reference floor. Ratio vs itself = 1.0 "
     "by definition. Not a trained model (no seeds, no validation)."),
    ("FQGNN", "Our model -- the paper's primary contribution. Highlighted pink/bold in every sheet "
     "(see legend below)."),
    ("RFFGNN", "Random-Fourier-Features GNN -- matched-capacity ablation control using a random "
     "(non-quasicrystalline) spectral basis in place of the FQ operator."),
    ("BiGRU", "Bidirectional GRU sequence-imputation baseline, trained from scratch on this data, "
     "5 seeds."),
    ("SAITS", "Self-Attention-based Imputation for Time Series, trained from scratch on this data, "
     "5 seeds."),
    ("CSDI", "Conditional Score-based Diffusion Imputation, trained from scratch on this data, "
     "5 seeds. Its validation metric is a denoising loss, not a ratio, and is not comparable "
     "across models."),
    ("BF-BiLSTM", "Pretrained bidirectional LSTM (authors' released weights), evaluated zero-shot "
     "-- no training or validation on this dataset, 1 seed only."),
    ("HABIT", "Historical-AIS route-library baseline: deterministic route lookup built from the "
     "training tracks. No gradient training, no seed variance (single deterministic run)."),
    ("MH-GIN", "Multi-scale Heterogeneous Graph Interpolation Network AIS baseline, trained from "
     "scratch on this data, 5 seeds."),
    (ONE_SIDED, "Variational-RNN AIS forecaster (vendor TF1 code), trained from scratch, 5 seeds. "
     "ONE-SIDED: it only sees context BEFORE each gap (forward-only forecaster reference), unlike "
     "every two-sided imputation model above -- reported for completeness, not as a like-for-like "
     "peer. Its vendor protocol has no validation-based model selection."),
    ("Hermite", "Piecewise linear / piecewise cubic Hermite interpolation with a curvature-based "
     "mode selector, article-faithful to Nguyen, Im & Lee (2015); slopes are data-driven backward "
     "secants of the position series itself (no AIS instrument sog/cog) -- deterministic classical "
     "interpolation baseline (no training, no seed variance). The full classical family, per "
     "gap-sweep cell, is in the Classical family sheet."),
]
for name, desc in model_desc:
    r += 1
    ws.cell(row=r, column=1, value=name).font = Font(bold=True)
    ws.cell(row=r, column=2, value=desc).alignment = Alignment(wrap_text=True)

r += 2
ws.cell(row=r, column=1, value="Classical-family note").font = Font(bold=True)
r += 1
merged_note(ws, r, 2, "The classical interpolation family (cubic Hermite spline, quintic "
            "minimum-jerk Hermite, chord-mean GP regression, CTRV/IMM Kalman) is reported per "
            "gap-sweep cell in the Classical family sheet; the cubic Hermite additionally "
            "appears in the main result sheets above.", italic=False)

r += 2
ws.cell(row=r, column=1, value="Highlighting legend").font = Font(bold=True)
r += 1
sw = ws.cell(row=r, column=1)
sw.fill = PINK_FILL
sw.border = DATA_BORDER
ws.cell(row=r, column=2, value="FQGNN -- our model (pink fill, bold font)").font = Font(bold=True)
r += 1
sw = ws.cell(row=r, column=1)
sw.fill = GRAY_FILL
sw.border = DATA_BORDER
ws.cell(row=r, column=2,
        value="GCD -- reference floor, not a trained model (gray fill, italic font)").font = Font(italic=True)
r += 1
ws.cell(row=r, column=2, value="All other baselines -- plain white fill, normal font")

r += 2
ws.cell(row=r, column=1, value="Test-set note").font = Font(bold=True)
r += 1
merged_note(ws, r, 2, "All test results are on n=770 shared test gaps (out of 774 vessel-disjoint "
            "gaps; the 4 excluded gaps were not evaluable by every model -- see Methods) unless "
            "stated otherwise.", italic=False)
r += 2
ws.cell(row=r, column=1, value="Statistics note").font = Font(bold=True)
r += 1
merged_note(ws, r, 2, STD_NOTE, italic=False)

# =============================================================================== Literature overview
# Extended Data Table 1 -- placed FIRST after README.  Read verbatim from
# paper/figdata/ed_table_literature.csv.
ws = wb.create_sheet("Literature overview")
nlit = len(lit.columns)
set_widths(ws, [34, 8, 18, 38, 72, 52, 42])
sheet_title(ws, "Extended Data Table 1 | Trajectory-reconstruction literature by family", nlit)
write_header(ws, 3, list(lit.columns))
ws.freeze_panes = "A4"

r = 4
for _, row in lit.iterrows():
    model_label = str(row[lit.columns[0]])
    if model_label.startswith("FQ-GNN"):
        fill, font = PINK_FILL, Font(bold=True)
    elif model_label.startswith("GCD"):
        fill, font = GRAY_FILL, Font(italic=True)
    else:
        fill, font = None, Font()
    for c, cname in enumerate(lit.columns, 1):
        v = row[cname]
        cell = put(ws, r, c, ("" if pd.isna(v) else v), fill=fill, font=font)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    r += 1

r += 1
merged_note(ws, r, nlit,
            "Family key: 1 = classical geometric interpolation; 2 = deep sequence imputers; "
            "3 = generic spectral encoders; 4 = AIS-native reconstruction.  Public-code "
            "availability was verified by web search on 2026-07-16.  Tested ratios are the "
            "seed-mean OVERALL ratio vs GCD on the 770 shared test gaps (Extended Data Table 2 "
            "sheets carry the per-seed detail).  " + ONE_SIDED_NOTE)

# =============================================================================== Overall test results
ws = wb.create_sheet("Overall test results")
set_widths(ws, [14, 12, 10, 16, 17, 15, 16])
sheet_title(ws, "Overall Test Results (OVERALL bucket, n=770 shared test gaps)", 7)
write_header(ws, 3, ["Model", "Seed", "n_gaps", "GCD_median_NM", "Model_median_NM",
                     "Ratio_vs_GCD", "Win_pct_vs_GCD"])
ws.freeze_panes = "A4"
FMTS_OV = {3: None, 4: "0.0000", 5: "0.0000", 6: "0.0000", 7: "0.0"}

ov = comb[comb["bucket"] == "OVERALL"]
r = 4
for model in MODEL_ORDER:
    rows = ov[ov["model"] == model].sort_values("seed")
    fill, font = style_for(model)
    start = r
    for _, row in rows.iterrows():
        seed = SEED_LABEL.get(model, int(row["seed"]))
        vals = [disp_name(model), seed, int(row["n"]), row["gcd_med"], row["model_med"],
                row["ratio"], (None if pd.isna(row["win_pct"]) else row["win_pct"])]
        for c, v in enumerate(vals, 1):
            put(ws, r, c, v, fill=fill, font=font, fmt=FMTS_OV.get(c))
        r += 1
    if model in MULTI_SEED:
        msfill, msfont = style_for(model, is_meanstd=True)
        for tag, fn in (("MEAN", "AVERAGE"), ("STD", "STDEVP")):
            put(ws, r, 1, disp_name(model), fill=msfill, font=msfont)
            put(ws, r, 2, tag, fill=msfill, font=msfont)
            for c in range(3, 8):
                col = get_column_letter(c)
                put(ws, r, c, f"={fn}({col}{start}:{col}{start + 4})",
                    fill=msfill, font=msfont, fmt=FMTS_OV.get(c))
            r += 1
    r += 1  # blank separator
merged_note(ws, r + 1, 7, ONE_SIDED_NOTE + "  HABIT is a deterministic route-library lookup "
            "(no training, no seed variance).  " + STD_NOTE)

# =============================================================================== Bucket test results
ws = wb.create_sheet("Bucket test results")
npiv = len(pivot.columns)
set_widths(ws, [14, 10, 14, 8, 16, 17, 15, 16] + [14] * (npiv - 8))
sheet_title(ws, "Test Results by Curvature Bucket (all models, all seeds)", 8)
write_header(ws, 3, ["Model", "Seed", "Bucket", "n", "GCD_median_NM", "Model_median_NM",
                     "Ratio_vs_GCD", "Win_pct_vs_GCD"])
ws.freeze_panes = "A4"
FMTS_BK = {5: "0.0000", 6: "0.0000", 7: "0.0000", 8: "0.0"}

r = 4
for model in MODEL_ORDER:
    sub = comb[comb["model"] == model]
    fill, font = style_for(model)
    for bucket in BUCKET_ORDER:
        for _, row in sub[sub["bucket"] == bucket].sort_values("seed").iterrows():
            seed = SEED_LABEL.get(model, int(row["seed"]))
            vals = [disp_name(model), seed, bucket, int(row["n"]), row["gcd_med"],
                    row["model_med"], row["ratio"],
                    (None if pd.isna(row["win_pct"]) else row["win_pct"])]
            for c, v in enumerate(vals, 1):
                put(ws, r, c, v, fill=fill, font=font, fmt=FMTS_BK.get(c))
            r += 1

r += 2
merged_note(ws, r, npiv, "Seed-mean summary by bucket (from pivot_ratio_mean.csv)",
            italic=False, bold=True)
r += 2
write_header(ws, r, list(pivot.columns))
fq_cols = [i + 1 for i, cname in enumerate(pivot.columns) if cname.startswith("FQGNN")]
r += 1
for _, row in pivot.iterrows():
    for c, cname in enumerate(pivot.columns, 1):
        v = row[cname]
        if cname in ("n",):
            v = int(v)
        fmt = None
        if cname.endswith("_ratio"):
            fmt = "0.0000"
        elif cname.endswith("_med_nm"):
            fmt = "0.00"
        fill, font = (PINK_FILL, Font(bold=True)) if c in fq_cols else (None, Font())
        put(ws, r, c, v, fill=fill, font=font, fmt=fmt)
    r += 1
r += 1
merged_note(ws, r, npiv, ONE_SIDED_NOTE)

# =============================================================================== Validation results
ws = wb.create_sheet("Validation results")
set_widths(ws, [14, 12, 15, 26, 40])
sheet_title(ws, "Validation Results (best checkpoint per seed)", 5)
write_header(ws, 3, ["Model", "Seed", "Epochs_trained", "Best_validation_metric_value", "Metric_type"])
ws.freeze_panes = "A4"

VAL_METRIC = {
    "FQGNN": "ratio vs GCD", "RFFGNN": "ratio vs GCD", "BiGRU": "ratio vs GCD",
    "SAITS": "ratio vs GCD", "MH-GIN": "ratio vs GCD",
    "CSDI": "denoising loss (not comparable across models)",
    "MultitaskAIS": "N/A -- vendor protocol has no validation selection",
}
r = 4
for model in MODEL_ORDER:
    if model == "GCD":
        continue
    fill, font = style_for(model)
    metric = VAL_METRIC.get(model, "N/A")
    if model == "BF-BiLSTM":
        for c, v in enumerate(["BF-BiLSTM", "zero-shot", 0, "zero-shot", "N/A -- zero-shot"], 1):
            put(ws, r, c, v, fill=fill, font=font, fmt="0.0000" if c == 4 else None)
        r += 2
        continue
    if model == "HABIT":
        vals = ["HABIT", "deterministic", 0, "deterministic",
                "N/A -- deterministic route lookup (no training)"]
        for c, v in enumerate(vals, 1):
            put(ws, r, c, v, fill=fill, font=font, fmt="0.0000" if c == 4 else None)
        r += 2
        continue
    if model == "Hermite":
        vals = ["Hermite", "deterministic", 0, "deterministic",
                "N/A -- deterministic spline interpolation (no training)"]
        for c, v in enumerate(vals, 1):
            put(ws, r, c, v, fill=fill, font=font, fmt="0.0000" if c == 4 else None)
        r += 2
        continue
    summ = run_summary(model)
    best_col = "best_val_ratio" if "best_val_ratio" in summ.columns else "best_val"
    start = r
    for _, row in summ.iterrows():
        vals = [disp_name(model), int(row["seed"]), int(row["epochs_trained"]),
                na(row[best_col]), metric]
        for c, v in enumerate(vals, 1):
            put(ws, r, c, v, fill=fill, font=font,
                fmt="0.0000" if (c == 4 and not isinstance(v, str)) else None)
        r += 1
    msfill, msfont = style_for(model, is_meanstd=True)
    all_na = summ[best_col].isna().all()
    for tag, fn in (("MEAN", "AVERAGE"), ("STD", "STDEVP")):
        put(ws, r, 1, disp_name(model), fill=msfill, font=msfont)
        put(ws, r, 2, tag, fill=msfill, font=msfont)
        put(ws, r, 3, f"={fn}(C{start}:C{start + 4})", fill=msfill, font=msfont)
        put(ws, r, 4, "N/A" if all_na else f"={fn}(D{start}:D{start + 4})",
            fill=msfill, font=msfont, fmt=None if all_na else "0.0000")
        put(ws, r, 5, metric, fill=msfill, font=msfont)
        r += 1
    r += 1

# =============================================================================== Training summary
ws = wb.create_sheet("Training summary")
set_widths(ws, [14, 12, 15, 18, 22])
sheet_title(ws, "Training Summary (final-epoch training loss per seed)", 5)
write_header(ws, 3, ["Model", "Seed", "Epochs_trained", "Final_train_loss", "Train_time_per_epoch_s"])
ws.freeze_panes = "A4"

r = 4
for model in MODEL_ORDER:
    if model == "GCD":
        continue
    fill, font = style_for(model)
    if model == "BF-BiLSTM":
        for c, v in enumerate(["BF-BiLSTM", "zero-shot", 0, "N/A", "N/A"], 1):
            put(ws, r, c, v, fill=fill, font=font)
        r += 2
        continue
    if model in ("HABIT", "Hermite"):
        tt = train_time(model)
        vals = [model, "deterministic", 0, "N/A", na(tt)]
        for c, v in enumerate(vals, 1):
            put(ws, r, c, v, fill=fill, font=font,
                fmt="0.00" if (c == 5 and not isinstance(v, str)) else None)
        r += 2
        continue
    summ = run_summary(model)
    losses = final_train_losses(model)
    tt = train_time(model)
    start = r
    for _, row in summ.iterrows():
        s = int(row["seed"])
        vals = [disp_name(model), s, int(row["epochs_trained"]),
                na(losses.get(s)), na(tt)]
        for c, v in enumerate(vals, 1):
            fmt = None
            if c == 4 and not isinstance(v, str):
                fmt = "0.000000"
            if c == 5 and not isinstance(v, str):
                fmt = "0.00"
            put(ws, r, c, v, fill=fill, font=font, fmt=fmt)
        r += 1
    msfill, msfont = style_for(model, is_meanstd=True)
    loss_na = all(losses.get(int(row["seed"])) is None for _, row in summ.iterrows())
    tt_na = tt is None
    for tag, fn in (("MEAN", "AVERAGE"), ("STD", "STDEVP")):
        put(ws, r, 1, disp_name(model), fill=msfill, font=msfont)
        put(ws, r, 2, tag, fill=msfill, font=msfont)
        put(ws, r, 3, f"={fn}(C{start}:C{start + 4})", fill=msfill, font=msfont)
        put(ws, r, 4, "N/A" if loss_na else f"={fn}(D{start}:D{start + 4})",
            fill=msfill, font=msfont)
        put(ws, r, 5, "N/A" if tt_na else f"={fn}(E{start}:E{start + 4})",
            fill=msfill, font=msfont)
        r += 1
    r += 1

r += 1
merged_note(ws, r, 5, "Note: Train_time_per_epoch_s is a per-model average, not measured per "
            "seed.  MultitaskAIS Final_train_loss is the vendor negative-ELBO per timestep "
            "(not comparable to other models' losses).")

# =============================================================================== Parameters and compute
ws = wb.create_sheet("Parameters and compute")
set_widths(ws, [14, 17, 15, 17, 22, 18, 60])
sheet_title(ws, "Parameters and Compute Cost", 7)
write_header(ws, 3, ["Model", "Trainable_params", "Frozen_params", "Test_ratio_overall",
                     "Inference_s_per_770_gaps", "Train_s_per_epoch", "Notes"])
ws.freeze_panes = "A4"

# GCD synthetic reference row first, then params_audit.csv rows in fixed model order
r = 4
fill, font = style_for("GCD")
for c in range(1, 8):
    v = "GCD" if c == 1 else ("N/A -- not a trained model" if c < 7 else None)
    put(ws, r, c, v, fill=fill, font=font)
r += 1

CANON = {normkey(m): m for m in MODEL_ORDER}
rows = params.copy()
rows["_canon"] = rows["_key"].map(lambda k: CANON.get(k, None))
rows["_order"] = rows["_canon"].map(
    lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else len(MODEL_ORDER))
rows = rows.sort_values(["_order", "model"])
for _, row in rows.iterrows():
    canon = row["_canon"] or str(row["model"])
    fill, font = style_for(canon)
    vals = [disp_name(canon),
            int(row["n_params_trainable"]), int(row["n_params_frozen"]),
            row["test_ratio_overall"], row["inference_s_per_770gaps"],
            na(row["train_s_per_epoch"]), na(row["notes"], alt=None)]
    fmts = [None, None, None, "0.0000", "0.00",
            "0.00" if not isinstance(vals[5], str) else None, None]
    for c, (v, fmt) in enumerate(zip(vals, fmts), 1):
        cell = put(ws, r, c, v, fill=fill, font=font, fmt=fmt)
        if c == 7:
            cell.alignment = Alignment(wrap_text=True)
    r += 1

# =============================================================================== Classical family
ws = wb.create_sheet("Classical family")
set_widths(ws, [18, 8, 46, 10, 16, 17, 15, 16])
sheet_title(ws, "Classical interpolation family, per gap-sweep cell (deterministic runs)", 8)
write_header(ws, 3, ["Track", "Gap", "Model", "n_gaps", "GCD_median_NM", "Model_median_NM",
                     "Ratio_vs_GCD", "Win_pct_vs_GCD"])
ws.freeze_panes = "A4"

CLASSICAL = [
    ("hermite", "Cubic Hermite spline, curvature-gated (data-driven position secants)"),
    ("quintic", "Quintic minimum-jerk Hermite (+ boundary acceleration)"),
    ("constrained_gp", "Chord-mean GP regression (Matern-5/2, per-gap marginal likelihood)"),
    ("ctrv_kalman", "CTRV EKF + 2-mode IMM (coordinated-turn Kalman)"),
]
GS = os.path.join(TA, "runs", "gap_sweep")
TRACK_CELLS = [("our_processed", ["gap_3h", "gap_6h", "gap_12h", "gap_24h"]),
               ("standard_process", ["gap_2h", "gap_6h", "gap_12h", "gap_24h"])]
FMTS_CF = {5: "0.0000", 6: "0.0000", 7: "0.0000", 8: "0.0"}

r = 4
for track, gaps in TRACK_CELLS:
    for gap in gaps:
        for sub, label in CLASSICAL:
            p = os.path.join(GS, track, gap, sub, "summary.csv")
            for _, row in summary_seed_rows(p).iterrows():
                vals = [track, gap.replace("gap_", ""), label, int(row["n_gaps"]),
                        row["gcd_med"], row["model_med"], row["ratio"], row["win_pct"]]
                for c, v in enumerate(vals, 1):
                    put(ws, r, c, v, fmt=FMTS_CF.get(c))
                r += 1
        r += 1  # blank separator per cell
merged_note(ws, r + 1, 8,
            "Each row is a single deterministic run (no training, no seed variance) on that "
            "cell's test gaps; ratio = model median error / GCD median error on the cell's own "
            "gap set (n_gaps varies by cell).  our_processed = the paper's raw-AIS preprocessing "
            "track; standard_process = the literature-standard interpolated-track preprocessing "
            "(see Methods).  The our_processed 6h cell is the paper's main test set.")

# =============================================================================== Frequency controls
ws = wb.create_sheet("Frequency controls")
set_widths(ws, [12, 56, 8, 10, 16, 17, 15, 16, 15, 15])
sheet_title(ws, "Matched controls for the FQ frequency set and the GNN conditioning channel "
            "(main 6h cell, 5 seeds each)", 10)
write_header(ws, 3, ["Variant", "Control", "Seed", "n_gaps", "GCD_median_NM", "Model_median_NM",
                     "Ratio_vs_GCD", "Win_pct_vs_GCD", "Epochs_trained", "Best_val_ratio"])
ws.freeze_panes = "A4"

CONTROL_DESC = [
    ("grid", "Periodic-comb frequency control (regular lattice, matched M and range)"),
    ("gauss", "Gaussian-matched frequency control (RFF analogue, matched moments)"),
    ("uniform", "Density-matched frequency control (uniform on the FQ bounding box)"),
    ("shuffle", "Shuffled-FQ frequency control (exact marginals, module coupling destroyed)"),
    ("learned", "Free-frequency control (learnable frequencies initialised at the FQ set)"),
    ("mlp", "No-GNN conditioning control (readout conditioning from raw anchor rows)"),
]
FMTS_FQC = {5: "0.0000", 6: "0.0000", 7: "0.0000", 8: "0.0", 10: "0.0000"}

r = 4
for var, desc in CONTROL_DESC:
    p = os.path.join(TA, "runs", "controls", f"{var}_blend", "summary.csv")
    df = summary_seed_rows(p)
    start = r
    for _, row in df.iterrows():
        vals = [var, desc, int(row["seed"]), int(row["n_gaps"]), row["gcd_med"],
                row["model_med"], row["ratio"], row["win_pct"],
                int(row["epochs_trained"]), row["best_val_ratio"]]
        for c, v in enumerate(vals, 1):
            put(ws, r, c, v, fmt=FMTS_FQC.get(c))
        r += 1
    for tag, fn in (("MEAN", "AVERAGE"), ("STD", "STDEVP")):
        put(ws, r, 1, var, fill=MSGRAY_FILL, font=Font(bold=True))
        put(ws, r, 3, tag, fill=MSGRAY_FILL, font=Font(bold=True))
        for c in range(4, 11):
            col = get_column_letter(c)
            put(ws, r, c, f"={fn}({col}{start}:{col}{start + len(df) - 1})",
                fill=MSGRAY_FILL, font=Font(bold=True), fmt=FMTS_FQC.get(c))
        r += 1
    r += 1
merged_note(ws, r + 1, 10,
            "Training/eval protocol, data, splits, optimizer, schedule and early stopping are "
            "identical to the full FQGNN run; the only change per variant is the frequency "
            "source (grid/gauss/uniform/shuffle/learned) or the conditioning source (mlp).  "
            "Full-model FQGNN reference values: see Overall test results.  " + STD_NOTE)

# =============================================================================== Encoder control
ws = wb.create_sheet("Encoder control")
set_widths(ws, [30, 8, 10, 16, 17, 15, 16, 15, 15])
sheet_title(ws, "Conditioning-encoder experiment: GNN vs raw-anchor MLP (our-12h cell)", 9)
write_header(ws, 3, ["Encoder", "Seed", "n_gaps", "GCD_median_NM", "Model_median_NM",
                     "Ratio_vs_GCD", "Win_pct_vs_GCD", "Epochs_trained", "Best_val"])
ws.freeze_panes = "A4"

if OPM:
    ENC_MAIN = [
        ("GNN (paper configuration)",
         os.path.join(TA, "runs", "gap_sweep_sobolev_opm", "our_processed", "gap_12h",
                      "fqgnn")),                       # dir => official clean re-eval merge
        ("MLP (raw-anchor conditioning)",
         os.path.join(TA, "runs", "gap_sweep_sobolev_mlp", "our_processed", "gap_12h", "fqgnn",
                      "summary.csv")),
    ]
else:
    ENC_MAIN = [
        ("GNN (paper configuration)",
         os.path.join(TA, "runs", "gap_sweep_sobolev", "our_processed", "gap_12h", "fqgnn",
                      "summary.csv")),
        ("MLP (raw-anchor conditioning)",
         os.path.join(TA, "runs", "gap_sweep_sobolev_mlp", "our_processed", "gap_12h", "fqgnn",
                      "summary.csv")),
    ]
FMTS_EC = {4: "0.0000", 5: "0.0000", 6: "0.0000", 7: "0.0", 9: "0.0000"}

r = 4
for enc, p in ENC_MAIN:
    df = opm_fq_cell_rows(p) if os.path.isdir(p) else summary_seed_rows(p)
    start = r
    for _, row in df.iterrows():
        vals = [enc, int(row["seed"]), int(row["n_gaps"]), row["gcd_med"], row["model_med"],
                row["ratio"], row["win_pct"], int(row["epochs_trained"]), row["best_val"]]
        for c, v in enumerate(vals, 1):
            put(ws, r, c, v, fmt=FMTS_EC.get(c))
        r += 1
    for tag, fn in (("MEAN", "AVERAGE"), ("STD", "STDEVP")):
        put(ws, r, 1, enc, fill=MSGRAY_FILL, font=Font(bold=True))
        put(ws, r, 2, tag, fill=MSGRAY_FILL, font=Font(bold=True))
        for c in range(3, 10):
            col = get_column_letter(c)
            put(ws, r, c, f"={fn}({col}{start}:{col}{start + len(df) - 1})",
                fill=MSGRAY_FILL, font=Font(bold=True), fmt=FMTS_EC.get(c))
        r += 1
    r += 1

r += 1
merged_note(ws, r, 9, "Supplementary arms (summary values computed from each run's per-seed "
            "rows; ratio std is the population standard deviation)", italic=False, bold=True)
r += 2
write_header(ws, r, ["Arm", "Encoder", "n_seeds", "Ratio_mean", "Ratio_std", "Win_mean_pct",
                     "Source"])
if OPM:
    ENC_ARMS = [
        ("our-6h", "GNN (paper configuration)",
         os.path.join(TA, "runs", "fqnative_opmstack_5seed", "summary_clean.csv"),
         "task_alpha/runs/fqnative_opmstack_5seed (summary_clean.csv, official exact-table "
         "re-eval)"),
        ("our-6h", "MLP (raw-anchor conditioning)",
         os.path.join(TA, "runs", "controls", "mlp_blend", "summary.csv"),
         "task_alpha/runs/controls/mlp_blend"),
        ("standard-2h", "GNN (paper configuration)",
         os.path.join(TA, "runs", "gap_sweep_sobolev_fd_opm", "standard_process", "gap_2h",
                      "fqgnn", "summary_clean.csv"),
         "task_alpha/runs/gap_sweep_sobolev_fd_opm/standard_process/gap_2h/fqgnn "
         "(summary_clean.csv, official exact-table re-eval; fd jet -- the paper convention "
         "on this track)"),
        ("standard-2h", "MLP (raw-anchor conditioning)",
         os.path.join(TA, "runs", "gap_sweep_sobolev_fd_mlp", "standard_process", "gap_2h",
                      "fqgnn", "summary.csv"),
         "task_alpha/runs/gap_sweep_sobolev_fd_mlp/standard_process/gap_2h/fqgnn (fd jet, "
         "matching the GNN arm)"),
    ]
else:
    ENC_ARMS = [
        ("our-6h", "GNN (paper configuration)",
         os.path.join(TA, "runs", "fqnative_sobolev_5seed", "summary.csv"),
         "task_alpha/runs/fqnative_sobolev_5seed"),
        ("our-6h", "MLP (raw-anchor conditioning)",
         os.path.join(TA, "runs", "controls", "mlp_blend", "summary.csv"),
         "task_alpha/runs/controls/mlp_blend"),
        ("standard-2h", "GNN (paper configuration)",
         os.path.join(TA, "runs", "gap_sweep_sobolev", "standard_process", "gap_2h", "fqgnn",
                      "summary.csv"),
         "task_alpha/runs/gap_sweep_sobolev/standard_process/gap_2h/fqgnn"),
        ("standard-2h", "MLP (raw-anchor conditioning)",
         os.path.join(TA, "runs", "gap_sweep_sobolev_mlp", "standard_process", "gap_2h",
                      "fqgnn", "summary.csv"),
         "task_alpha/runs/gap_sweep_sobolev_mlp/standard_process/gap_2h/fqgnn"),
    ]
r += 1
for arm, enc, p, srcnote in ENC_ARMS:
    df = summary_seed_rows(p)
    rat = df["ratio"].astype(float).to_numpy()
    win = df["win_pct"].astype(float).to_numpy()
    vals = [arm, enc, len(df), round(float(rat.mean()), 4),
            round(float(rat.std()), 4), round(float(win.mean()), 1), srcnote]
    for c, v in enumerate(vals, 1):
        put(ws, r, c, v, fmt={4: "0.0000", 5: "0.0000", 6: "0.0"}.get(c))
    r += 1
r += 1
if OPM:
    enc_note = ("The main table is the our-12h cell (n=352 test gaps per seed): GNN = "
                "task_alpha/runs/gap_sweep_sobolev_opm/our_processed/gap_12h/fqgnn (test values "
                "from the official exact-table re-eval summary_clean.csv; epochs/val from the "
                "in-run summary.csv), MLP = task_alpha/runs/gap_sweep_sobolev_mlp/our_processed/"
                "gap_12h/fqgnn.  The standard-2h arms both use the fd boundary jet on that "
                "track (the paper convention; see assemble_gap_sweep).  The our-6h MLP arm is "
                "the mlp control of the Frequency controls sheet (same run); the our-6h GNN arm "
                "is the main 5-seed reference run.  " + STD_NOTE)
else:
    enc_note = ("The main table is the our-12h cell (n=352 test gaps per seed): "
                "task_alpha/runs/gap_sweep_sobolev{,_mlp}/our_processed/gap_12h/fqgnn.  The "
                "our-6h MLP arm is the mlp control of the Frequency controls sheet (same run); "
                "the our-6h GNN arm is the main 5-seed reference run.  " + STD_NOTE)
merged_note(ws, r, 9, enc_note)

# =============================================================================== Clustered statistics by cell
ws = wb.create_sheet("Clustered statistics by cell")
cstats = pd.read_csv(os.path.join(FIGDATA, "clustered_stats_cells.csv"))
ncc = len(cstats.columns)
set_widths(ws, [10, 34, 12, 11, 11, 10, 11, 15])
sheet_title(ws, "Vessel-clustered statistics per gap-sweep cell (FQGNN vs cubic Hermite)", ncc)
write_header(ws, 3, list(cstats.columns))
ws.freeze_panes = "A4"
FMTS_CS = {3: "0.0000", 4: "0.0000", 5: "0.0000"}

r = 4
prev_cell = None
for _, row in cstats.iterrows():
    if prev_cell is not None and row["cell"] != prev_cell:
        r += 1                                   # blank separator between cells
    prev_cell = row["cell"]
    is_p = str(row["metric"]).startswith("wilcoxon")
    for c, cname in enumerate(cstats.columns, 1):
        v = row[cname]
        if pd.isna(v):
            v = None
        elif cname in ("n_gaps", "n_vessels"):
            v = int(v)
        fmt = "0.00E+00" if (is_p and c == 3) else FMTS_CS.get(c)
        put(ws, r, c, v, fmt=fmt)
    r += 1
merged_note(ws, r + 1, ncc,
            "Source: paper/figdata/clustered_stats_cells.csv -- per-cell vessel-clustered "
            "bootstrap (B=10,000 vessel resamples, percentile CIs) of FQGNN (official "
            "exact-table re-evals) vs the deterministic cubic-Hermite control; one row per "
            "cell x metric x (pooled / per-seed) estimate.  Wilcoxon rows carry the exact "
            "vessel-level p (scientific notation) and the Holm-adjusted p across the eight "
            "cells (wilcoxon_vessel_p_holm).")

# =============================================================================== Native-cadence imputers
NATIVE_RUNS = [("SAITS", os.path.join(TA, "baselines", "SAITS")),
               ("CSDI", os.path.join(TA, "baselines", "CSDI"))]
nat_frames = []
for mdl, base in NATIVE_RUNS:
    for g in (2, 6, 12, 24):
        p = os.path.join(base, f"runs_5seed_gaps_standard_process_{g}h", "summary.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        d.insert(0, "cell", f"std_{g}h")
        d.insert(0, "model", mdl)
        nat_frames.append(d)
if nat_frames:
    ws = wb.create_sheet("Native-cadence imputers")
    nat = pd.concat(nat_frames, ignore_index=True)
    ncn = len(nat.columns)
    set_widths(ws, [9, 9, 7, 8, 9, 11, 10, 9, 15, 10])
    sheet_title(ws, "Regular-grid imputers trained NATIVELY on the standardized 10-min cells "
                    "(vanilla, 5 seeds per cell)", ncn)
    write_header(ws, 3, list(nat.columns))
    ws.freeze_panes = "A4"
    r = 4
    prev = None
    for _, row in nat.iterrows():
        if prev is not None and (row["model"], row["cell"]) != prev:
            r += 1
        prev = (row["model"], row["cell"])
        is_ms = str(row["seed"]) in ("MEAN", "STD")
        fill, font = (MSGRAY_FILL, Font(bold=True)) if is_ms else (None, None)
        for c, cname in enumerate(nat.columns, 1):
            v = row[cname]
            if pd.isna(v):
                v = None
            elif cname in ("n_gaps", "epochs_trained") and not is_ms:
                v = int(v)
            fmt = {"gcd_med": "0.000", "model_med": "0.000", "ratio": "0.0000",
                   "win_pct": "0.0", "best_val": "0.000"}.get(cname)
            put(ws, r, c, v, fill=fill, font=font, fmt=fmt)
        r += 1
    merged_note(ws, r + 1, ncn,
                "Source: task_alpha/baselines/<model>/runs_5seed_gaps_standard_process_<G>h/"
                "summary.csv -- each model trained from scratch on each standardized cell under "
                "its own fixed vanilla configuration (SAITS: Optuna-fixed HP, 200-epoch cap, "
                "patience 12; CSDI: deployed config), the uniform 10-min product being these "
                "models' native input regime (native-regime fairness bar).  Ratios are vs the GCD "
                "reference on the identical gaps; FQGNN on the same cells: 0.6243 / 0.7513 / "
                "0.8630 / 0.8061 at 2 / 6 / 12 / 24 h.")

# =============================================================================== 2019 holdout
ws = wb.create_sheet("2019 holdout")
hold = pd.read_csv(os.path.join(TA, "runs", "holdout_2019", "summary.csv"))
nch = len(hold.columns)
set_widths(ws, [10, 6, 16, 8, 10, 11, 9, 9])
sheet_title(ws, "Zero-shot temporal holdout: frozen 2016-18 checkpoints on July-August 2019 "
                "(no retraining)", nch)
write_header(ws, 3, list(hold.columns))
ws.freeze_panes = "A4"
r = 4
for _, row in hold.iterrows():
    fill, font = style_for("FQGNN" if row["model"] == "fqgnn" else "Hermite")
    for c, cname in enumerate(hold.columns, 1):
        v = row[cname]
        if cname in ("seed", "n"):
            v = int(v)
        fmt = {"gcd_med": "0.000", "model_med": "0.000", "ratio": "0.0000",
               "win_pct": "0.0"}.get(cname)
        put(ws, r, c, v, fill=fill, font=font, fmt=fmt)
    r += 1
merged_note(ws, r + 1, nch,
            "Source: task_alpha/runs/holdout_2019/summary.csv (script task_alpha/"
            "_holdout_eval_2019.py; identical raw pipeline incl. the per-year clean whitelist; "
            "2,110 gaps from 515 vessels, of which 910 gaps belong to vessels absent from the "
            "2016-18 train/val sets).  FQGNN rows are the five FROZEN Op-M checkpoints evaluated "
            "zero-shot (exact torus-integral tables); Hermite is the deterministic published "
            "baseline on the identical gaps; ratios are vs the GCD reference on the identical "
            "gaps.  In-domain 5-seed reference: 0.5646 +/- 0.0025.")

# =============================================================================== Reference certificate
ws = wb.create_sheet("Reference certificate")
rcert = pd.read_csv(os.path.join(TA, "runs", "ref_cert_8cells.csv"))
ncr = len(rcert.columns)
set_widths(ws, [18, 8, 8, 7, 22, 22, 11, 13])
sheet_title(ws, "Reference-only substitution certificate: chart-linear vs exact great-circle "
                "reference, all 8 cells", ncr)
write_header(ws, 3, list(rcert.columns))
ws.freeze_panes = "A4"
r = 4
for _, row in rcert.iterrows():
    for c, cname in enumerate(rcert.columns, 1):
        v = row[cname]
        if cname in ("gap_h", "n_gaps", "n_seam"):
            v = int(v)
        fmt = {"ref_med_chart_nm": "0.0000", "ref_med_slerp_nm": "0.0000",
               "pct_shift": "0.000", "ratio_scale": "0.00000"}.get(cname)
        put(ws, r, c, v, fmt=fmt)
    r += 1
merged_note(ws, r + 1, ncr,
            "Source: task_alpha/runs/ref_cert_8cells.csv (script task_alpha/"
            "_diag_ref_cert_8cells.py; fp64; slerp between the bracketing truth anchors at "
            "matched time fractions; per-gap median errors; cell median of per-gap medians; "
            "all gaps retained, incl. each cell's chart-seam gaps).  ratio_scale = chart "
            "median / slerp median is the one common factor the substitution applies to "
            "every model ratio of the cell, so orderings and win counts are exactly "
            "invariant (Supplementary Section S4.6).")

wb.save(OUT)
print("Wrote", OUT)
print("Sheets:", wb.sheetnames)

"""ED5_build.py -- EXTENDED DATA FIGURE 5: "Where baselines are tolerable vs losing".

Full-page 12 x 4 grid (48 cell charts): per-sinuosity-bucket accuracy at EVERY gap horizon
on BOTH preprocessing tracks, for FQ-GNN and the baselines.

  rows    5 models x 2 tracks, model-paired:  FQ-GNN / Bi-GRU / HABIT / MH-GIN /
          MultitaskAIS (one-sided), each as (raw AIS, interp-10m); then CSDI and SAITS on the
          standardized track only (trained natively per standardized cell)
  cols    the 4 gap horizons of that row's track
          raw AIS (our_processed): 3 / 6 / 12 / 24 h; interp-10m (standard_process):
          2 / 6 / 12 / 24 h -- the actual hours are printed on every cell title
  cell    5 bars = the sinuosity buckets (FS.BUCKET_COLOR teal ramp), bar height =
          5-seed MEAN of the bucket's median-NM ratio to GCD (HABIT: its single
          deterministic pass), PER-ROW log y = [0.3, max(3, 1.6 x row max)] so the
          low-ratio FQ-GNN / HABIT rows stay readable (values span 0.38 .. ~2100 across
          models; the strip says "y-axis range differs per row"), thin dark +/- s.d.
          caps (per-seed dots would be ~1000 unreadable points at this scale), dashed
          GCD-parity rule at 1.0 labelled once per row (leftmost cell)

BF-BiLSTM is absent (zero-shot on the shared raw-AIS 6 h cell only -- no horizon sweep); the
bottom condition strip says so.

Reads ONLY figdata/ed5_horizon_buckets.csv (assembled by
task_alpha/assemble_ed5_horizon_buckets.py -- no recomputation).  Style: fig_style.py
(locked).  Output: <FQGNN_FIG_OUT>/ED5.{pdf,png} + ED5_preview.png.
Run:  python paper_figures_tables/ED5_build.py   (any cwd)
"""
from __future__ import annotations

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _paths import FIGDATA, OUT, RESULTS  # noqa: E402,F401  (env FQGNN_FIGDATA / FQGNN_FIG_OUT / FQGNN_RESULTS_ROOT)

import fig_style as FS                                          # noqa: E402
from matplotlib.lines import Line2D                             # noqa: E402
from matplotlib.patches import Patch                            # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator  # noqa: E402

FS.apply()

# ------------------------------------------------------------------------------------------ #
#  DATA                                                                                        #
# ------------------------------------------------------------------------------------------ #
df = pd.read_csv(os.path.join(FIGDATA, "ed5_horizon_buckets.csv"))
# the assembled CSV also carries the classical-family rows (hermite / ctrv_kalman /
# quintic / constrained_gp -- reported in the Extended Data workbook); this locked
# 12 x 4 grid plots the 5 sweep models + the standardized-only CSDI / SAITS rows
df = df[df.model.isin(("fqgnn", "bigru", "habit", "mhgin", "multitaskais",
                       "csdi", "saits"))].reset_index(drop=True)
assert len(df) == 240, f"expected 240 data rows, got {len(df)}"
assert np.isfinite(df["ratio_mean"]).all() and np.isfinite(df["ratio_std"]).all()

TRACK_GAPS = {"our_processed": (3, 6, 12, 24), "standard_process": (2, 6, 12, 24)}
TRACK_LAB = {"our_processed": "raw AIS", "standard_process": "standardized"}
BUCKETS = FS.BUCKETS                                            # low -> high sinuosity
# grid rows: model-paired (raw AIS above interp-10m), FQ-GNN first; then the two
# NATIVE-cadence std-only rows (CSDI / SAITS trained per standardized cell; their raw-track
# result is the single shared 6-h cell, Fig. 2d)
ROWS = [(m, t) for m in ("fqgnn", "bigru", "habit", "mhgin", "multitaskais")
        for t in ("our_processed", "standard_process")] + \
       [("csdi", "standard_process"), ("saits", "standard_process")]
MKEY = {"fqgnn": "FQGNN", "bigru": "BiGRU", "habit": "HABIT",
        "mhgin": "MH-GIN", "multitaskais": "MultitaskAIS", "csdi": "CSDI", "saits": "SAITS"}
# left-margin display name (wrapped where the locked label is long)
MNAME = {"fqgnn": "FQ-GNN (ours)", "bigru": "Bi-GRU", "habit": "HABIT",
         "mhgin": "MH-GIN", "multitaskais": "MultitaskAIS\n(one-sided)",
         "csdi": "CSDI\n(native std)", "saits": "SAITS\n(native std)"}

YLO = 0.3
# PER-ROW y-limits: [0.3, max(3, 1.6 x row max)] keeps the low-ratio rows readable while
# the GCD parity line at 1.0 stays inside every row's range (strip: "y-axis range differs
# per row").  Verified: every error cap (mean + s.d.) also fits inside its row's limit.
ROW_YHI = {}
for model, track in ROWS:
    sub = df[(df.model == model) & (df.track == track)]
    yhi = max(3.0, float(sub.ratio_mean.max()) * 1.6,
              float((sub.ratio_mean + sub.ratio_std).max()) * 1.2)
    assert float((sub.ratio_mean + sub.ratio_std).max()) < yhi, (model, track)
    ROW_YHI[(model, track)] = yhi


def row_ticks(yhi):
    """Sparse clean per-row ticks (>= 2 majors): sub-decade set on tight rows, decades on
    wide rows."""
    if yhi <= 10.0:
        return [t for t in (0.5, 1, 2, 5) if t <= yhi]
    return [t for t in (1, 10, 100, 1000) if t <= yhi]


def darken(hexcolor, f=0.30):
    h = hexcolor.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02X%02X%02X" % (int(r * (1 - f)), int(g * (1 - f)), int(b * (1 - f)))


# ------------------------------------------------------------------------------------------ #
#  LAYOUT -- full page 180 x 250 mm; 12 row bands (5 pairs + 2 singles), 4 columns             #
# ------------------------------------------------------------------------------------------ #
fig = FS.new_figure(width_mm=180, height_mm=250)

L, R = 0.155, 0.993                       # grid outer x bounds (left margin = row labels)
COL_GUT = 0.013
CW = (R - L - 3 * COL_GUT) / 4.0

TOP, BOT = 0.948, 0.092                   # grid outer y bounds
G_PAIR = 0.0225                           # gap INSIDE a model pair (room for cell titles)
G_MODEL = 0.0335                          # gap BETWEEN model pairs / before the single rows
# 12 rows = 5 pairs + 2 std-only singles; gaps after rows 0..10: P,M,P,M,P,M,P,M,P,M,M
GAPS_AFTER = [G_PAIR, G_MODEL] * 5 + [G_MODEL]
CH = (TOP - BOT - sum(GAPS_AFTER)) / 12.0            # cell height

row_tops = []
y = TOP
for i in range(12):
    row_tops.append(y)
    if i < len(GAPS_AFTER):
        y -= CH + GAPS_AFTER[i]

axes = {}
for r, (model, track) in enumerate(ROWS):
    for c in range(4):
        axes[(r, c)] = fig.add_axes([L + c * (CW + COL_GUT), row_tops[r] - CH, CW, CH])

# ------------------------------------------------------------------------------------------ #
#  CELLS                                                                                       #
# ------------------------------------------------------------------------------------------ #
n_cells = n_bars = 0
for r, (model, track) in enumerate(ROWS):
    for c, gap in enumerate(TRACK_GAPS[track]):
        ax = axes[(r, c)]
        cell = df[(df.track == track) & (df.gap_h == gap) & (df.model == model)]
        cell = cell.set_index("bucket").reindex(BUCKETS)
        assert cell["ratio_mean"].notna().all(), f"empty bucket in {model}/{track}/{gap}h"

        ax.set_yscale("log")
        ax.set_ylim(YLO, ROW_YHI[(model, track)])
        xs = np.arange(5)
        for i, b in enumerate(BUCKETS):
            mean = float(cell.loc[b, "ratio_mean"])
            std = float(cell.loc[b, "ratio_std"])
            nsd = int(cell.loc[b, "n_seeds"])
            col = FS.BUCKET_COLOR[b]
            ax.bar(i, mean - YLO, bottom=YLO, width=0.72, color=col,
                   edgecolor="none", zorder=3)
            if nsd > 1 and std > 0:                       # HABIT: 1 deterministic pass
                lo = min(std, mean - YLO * 1.02)          # keep the low cap on the log axis
                ax.errorbar(i, mean, yerr=[[lo], [std]], color="#2B2B2B", lw=0,
                            elinewidth=0.55, capsize=1.4, capthick=0.55, zorder=5)
            n_bars += 1

        FS.ref_hline(ax, 1.0, color=FS.C["gcd"], lw=0.6)
        if c == 0:                                        # "GCD" once per row, leftmost cell;
            ax.annotate("GCD", xy=(0.985, 1.0),           # white chip keeps it legible where
                        xycoords=("axes fraction", "data"),   # bars cross the parity line
                        fontsize=5.3, color=FS.C["gcd"], ha="right", va="bottom",
                        xytext=(0, 0.5), textcoords="offset points", zorder=7,
                        bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.8))

        ax.set_xlim(-0.65, 4.65)
        ax.set_title(f"{gap} h", fontsize=5.8, pad=1.6)
        yticks = row_ticks(ROW_YHI[(model, track)])
        ax.yaxis.set_major_locator(FixedLocator(yticks))
        ax.yaxis.set_minor_locator(NullLocator())
        if c == 0:
            ax.yaxis.set_major_formatter(
                FixedFormatter([("%g" % t) for t in yticks]))
            ax.tick_params(axis="y", labelsize=5.0)
        else:
            ax.tick_params(labelleft=False)
        if r == len(ROWS) - 1:                            # bottom row: bucket tick labels
            ax.set_xticks(xs)
            ax.set_xticklabels(BUCKETS, rotation=45, ha="right", fontsize=4.8)
            ax.tick_params(axis="x", length=1.8, width=0.5, pad=1.2)
        else:
            ax.set_xticks([])
        n_cells += 1

# ------------------------------------------------------------------------------------------ #
#  ROW LABELS (left margin): locked model label + track                                        #
# ------------------------------------------------------------------------------------------ #
# right edge at 0.123: clear of the col-0 y-tick labels (which extend left of L by ~0.028)
for r, (model, track) in enumerate(ROWS):
    yc = row_tops[r] - CH / 2.0
    lines = MNAME[model].count("\n") + 1
    fig.text(0.108, yc + 0.0065 * (1 if lines == 1 else 1.6), MNAME[model],
             fontsize=6.0, color=FS.C["text"], ha="right", va="center", linespacing=1.05)
    fig.text(0.108, yc - 0.0075 * (1 if lines == 1 else 1.55), TRACK_LAB[track],
             fontsize=5.4, color="#6E6E6E", ha="right", va="center", style="italic")

# shared y-axis title, far left, centred on the grid
fig.text(0.017, (TOP + BOT) / 2.0, "Median-NM error ratio vs GCD (log, lower better)",
         fontsize=7.0, color=FS.C["axis"], ha="center", va="center", rotation=90)
# shared x-axis title under the bottom row's bucket labels
fig.text(L + (R - L) / 2.0, BOT - 0.040, "Sinuosity bucket (straight -> curved)",
         fontsize=7.0, color=FS.C["axis"], ha="center", va="top")

# ------------------------------------------------------------------------------------------ #
#  TOP LEGEND: the 5 bucket swatches + GCD-parity rule                                         #
# ------------------------------------------------------------------------------------------ #
handles = [Patch(facecolor=FS.BUCKET_COLOR[b], edgecolor="none",
                 linewidth=0.4, label=b) for b in BUCKETS]
handles.append(Line2D([0], [0], color=FS.C["gcd"], ls=(0, (4, 2.5)), lw=0.7,
                      label="GCD parity (ratio 1)"))
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(L + (R - L) / 2.0, 0.998),
           ncol=6, fontsize=5.8, handlelength=1.5, handletextpad=0.45, columnspacing=1.1,
           title="Sinuosity bucket", title_fontsize=6.2)
fig.legends[-1].get_title().set_color(FS.C["text"])

# ------------------------------------------------------------------------------------------ #
#  BOTTOM CONDITION STRIP (gray)                                                               #
# ------------------------------------------------------------------------------------------ #
fig.text(L + (R - L) / 2.0, 0.0330,
         "bars: 5-seed mean of per-bucket median-NM ratio to GCD  |  caps: $\\pm$ s.d. over"
         " seeds  |  HABIT: single deterministic pass (no spread)  |  y-axis range differs"
         " per row\n"
         "MultitaskAIS: one-sided reference (forward VRNN)  |  raw AIS horizons 3/6/12/24 h;"
         " standardized horizons 2/6/12/24 h  |  bucket n = 10-2871 gaps\n"
         "CSDI / SAITS: trained natively per standardized-product cell (raw AIS: shared 6 h cell only,"
         " Fig. 2d)  |  BF-BiLSTM not shown (zero-shot, shared 6 h raw-AIS cell only)",
         fontsize=5.5, color="#6E6E6E", ha="center", va="top", linespacing=1.4)

# ------------------------------------------------------------------------------------------ #
FS.save(fig, os.path.join(OUT, "ED5"))
print(f"cells rendered: {n_cells} (expected 48); bars drawn: {n_bars} (expected 240)")
print("per-row y-maxima (ylim = [0.3, max(3, 1.6 x row max)]):")
for model, track in ROWS:
    print("  %-13s %-16s yhi = %8.1f  (row max %8.2f)"
          % (model, TRACK_LAB[track], ROW_YHI[(model, track)],
             df[(df.model == model) & (df.track == track)].ratio_mean.max()))
print("ratio_mean range: %.3f .. %.1f" % (df.ratio_mean.min(), df.ratio_mean.max()))
for m in ("fqgnn", "bigru", "habit", "mhgin", "multitaskais", "csdi", "saits"):
    s = df[df.model == m]
    print("  %-13s %8.3f .. %9.3f   (n_seeds max %d)"
          % (m, s.ratio_mean.min(), s.ratio_mean.max(), s.n_seeds.max()))

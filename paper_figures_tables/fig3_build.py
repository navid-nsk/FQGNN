"""Figure 3 -- "FQ-GNN is the only model decisively below the geodesic floor" (standardized 2-h cell).

All-model same-gaps results on the standardized 2-h cell (6,644 finite-sinuosity gaps,
identical GCD column; 6,599 with gcd_med > 0 in the panel-d scatter).
Panels:
  a (12 col, band 1, HERO) : ratio vs GCD per curvature bucket, log y, 7 model series
  b (4 col, band 2)        : overall absolute median NM, 8-bar spine (7 models + GCD), log y
  c (4 col, band 2)        : gap-level win rate vs GCD, 7-bar spine
  d (4 col, band 2)        : per-gap head-to-head scatter (seed 0), log-log, sinuosity-coloured
  e (12 col, band 3)       : annotated ratio heatmap, 8 rows (7 models + GCD) x 6 buckets, log colour

Reads ONLY archived CSVs (figdata/ + task_alpha run outputs); no recomputation.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _paths import FIGDATA, OUT, RESULTS  # noqa: E402,F401  (env FQGNN_FIGDATA / FQGNN_FIG_OUT / FQGNN_RESULTS_ROOT)

import fig_style as FS  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

FS.apply()

# ------------------------------------------------------------------------------------------ #
#  DATA                                                                                        #
# ------------------------------------------------------------------------------------------ #
COMB = pd.read_csv(os.path.join(FIGDATA, "combined_buckets_std2h.csv"))
PIVOT = pd.read_csv(os.path.join(FIGDATA, "pivot_ratio_std2h.csv"))
PERGAP = pd.read_csv(os.path.join(
    RESULTS, "runs", "gap_sweep_sobolev_fd_opm", "standard_process",
    "gap_2h", "fqgnn", "seed0", "per_gap_clean.csv"))
PERGAP = PERGAP[(PERGAP.gcd_med > 0) & np.isfinite(
    pd.to_numeric(PERGAP.sinuosity, errors="coerce"))].copy()
# 6,644 finite-sinuosity gaps minus 45 zero-GCD gaps (log-log scatter needs gcd > 0)
assert len(PERGAP) == 6599, len(PERGAP)

BUCKETS = FS.BUCKETS                       # 5 sinuosity buckets
XCATS = ["OVERALL"] + BUCKETS              # 6 columns everywhere
XLABELS = ["Overall", "0.00–0.02", "0.02–0.05", "0.05–0.10",
           "0.10–0.25", ">0.25"]
SERIES = ["FQGNN", "Hermite", "HABIT", "RFFGNN", "MH-GIN", "BiGRU",     # legend rank order
          "MultitaskAIS"]                                              # (by overall ratio)
HEAT_ROWS = ["FQGNN", "Hermite", "GCD", "HABIT", "RFFGNN", "MH-GIN", "BiGRU",
             "MultitaskAIS"]

PIVOT = PIVOT.set_index("bucket")


def stat(model, col):
    """mean / sd / min / max over seeds for one model, indexed by XCATS."""
    g = COMB[COMB.model == model].groupby("bucket")[col]
    agg = g.agg(["mean", "std", "min", "max"]).reindex(XCATS)
    return agg


def darker(hexcolor, f=0.72):
    r, g, b = mcolors.to_rgb(hexcolor)
    return (r * f, g * f, b * f)


TEAL_CMAP = LinearSegmentedColormap.from_list("fq_teal", FS.SEQ_TEAL[1:])
BLUE_CMAP = LinearSegmentedColormap.from_list("fq_blue", FS.SEQ_BLUE)

# ------------------------------------------------------------------------------------------ #
#  LAYOUT (fig fractions; canvas 180 x 138 mm, 12-col grid, 3 bands)                           #
# ------------------------------------------------------------------------------------------ #
fig = FS.new_figure(180, 138)

AX_A = fig.add_axes([0.088, 0.645, 0.902, 0.285])                    # band 1 hero (12 col)
# band 2: b and c widened slightly for their 8- and 7-bar spines
AX_B = fig.add_axes([0.088, 0.335, 0.245, 0.195])
AX_C = fig.add_axes([0.415, 0.335, 0.245, 0.195])
AX_D = fig.add_axes([0.735, 0.335, 0.175, 0.195])
CAX_D = fig.add_axes([0.918, 0.335, 0.010, 0.195])
# band 3: the 8-row heatmap, shifted right so the long
# "MultitaskAIS (one-sided)" row label clears the canvas edge
AX_E = fig.add_axes([0.155, 0.022, 0.600, 0.180])
CAX_E = fig.add_axes([0.765, 0.022, 0.010, 0.180])

# ------------------------------------------------------------------------------------------ #
#  PANEL a -- HERO: ratio vs curvature bucket                                                  #
# ------------------------------------------------------------------------------------------ #
ax = AX_A
ax.set_yscale("log")
ax.set_xlim(-0.55, 5.45)
ax.set_ylim(0.33, 8000)          # extended: MultitaskAIS reaches ~1665x in the flattest bucket

# pale-pink "beats the straight line" region below y = 1 (pale fill, no border)
ax.axhspan(0.33, 1.0, facecolor=FS.C["pink_light"], alpha=0.15, lw=0, zorder=0.5)
ax.text(4.5, 0.435, "beats the straight line", fontsize=5.5, color="#C9447A",
        ha="center", va="center", zorder=2)

# geodesic floor rule + OVERALL divider
ax.axhline(1.0, color=FS.C["gcd"], ls=(0, (4, 2.5)), lw=0.75, zorder=1.5)
ax.text(1.05, 1.09, "geodesic floor (GCD)", fontsize=5.5, color=FS.C["gcd"],
        ha="center", va="bottom", zorder=2)
ax.axvline(0.5, color=FS.C["sep"], lw=0.75, zorder=1)

xb = np.arange(1, 6)                                   # bucket positions
for m in SERIES[::-1]:                                 # FQGNN drawn last (top z)
    s = FS.MODEL[m]
    agg = stat(m, "ratio")
    mean = agg["mean"].values
    # bucket series: connected line + markers
    ax.plot(xb, mean[1:], color=s["color"], marker=s["marker"], ls=s["ls"],
            lw=s["lw"], markersize=3.2, zorder=s["z"], clip_on=False)
    # OVERALL: marker only, left of the divider
    ax.plot([0], [mean[0]], color=s["color"], marker=s["marker"], ls="none",
            markersize=3.6, zorder=s["z"])
    if m == "FQGNN":
        # min-max over 5 seeds as a 20 %-opacity pink band (+ mini-band at OVERALL)
        ax.fill_between(xb, agg["min"].values[1:], agg["max"].values[1:],
                        color=s["color"], alpha=0.20, lw=0, zorder=s["z"] - 0.5)
        ax.fill_between([-0.14, 0.14], [agg["min"].values[0]] * 2,
                        [agg["max"].values[0]] * 2,
                        color=s["color"], alpha=0.20, lw=0, zorder=s["z"] - 0.5)
    elif agg["std"].notna().all():                     # thin +-sd caps (skip n=1 BF-BiLSTM)
        ax.errorbar(np.r_[0, xb], mean, yerr=agg["std"].values, fmt="none",
                    ecolor=darker(s["color"]), elinewidth=0.6, capsize=1.8,
                    capthick=0.6, zorder=s["z"] - 0.2)

ax.set_xticks(range(6))
ax.set_xticklabels(XLABELS)
ax.set_yticks([0.5, 1, 2, 5, 10, 100, 1000])
ax.set_yticklabels(["0.5", "1", "2", "5", "10", "100", "1000"])
ax.set_ylabel("Median-error ratio vs GCD\n(lower better)")
ax.set_title("Error ratio vs GCD across curvature — standardized 2-h cell, 6,644 gaps", pad=3)

# teal bucket swatches under the tick labels
for x, b in zip(xb, BUCKETS):
    ax.add_patch(mpatches.Rectangle(
        (x - 0.22, -0.092), 0.44, 0.026, transform=mtransforms.blended_transform_factory(
            ax.transData, ax.transAxes),
        facecolor=FS.BUCKET_COLOR[b], lw=0, clip_on=False))

# in-panel legend, inline overall ratios = results table.  7 entries -> 3 columns, in the
# upper-RIGHT dead space (the flattest buckets on the left carry the ~1400x MH-GIN /
# MultitaskAIS points).
metrics = {"FQGNN": "0.62", "Hermite": "0.93", "HABIT": "1.75", "RFFGNN": "2.43",
           "MH-GIN": "4.9", "BiGRU": "16.8", "MultitaskAIS": "315"}
leg = ax.legend(handles=FS.model_legend_handles(SERIES, metrics), loc="upper right",
                bbox_to_anchor=(0.998, 1.005), ncol=3, fontsize=6, handlelength=1.9,
                columnspacing=0.9, labelspacing=0.32, handletextpad=0.5, borderaxespad=0.0)

FS.condition_strip(ax, "x: sinuosity bucket | gaps: 6,644 | gap length: 2 h | standardized "
                       "10-min product | metric: per-gap median NM", dy=-0.165)

# ------------------------------------------------------------------------------------------ #
#  PANEL b -- overall absolute error, 7-bar spine                                              #
# ------------------------------------------------------------------------------------------ #
ax = AX_B
BAR_ORDER = ["GCD", "FQGNN", "Hermite", "HABIT", "RFFGNN", "MH-GIN", "BiGRU",
             "MultitaskAIS"]
BAR_NAMES = ["GCD", "FQ-GNN", "Hermite", "HABIT", "RFF-GNN", "MH-GIN", "Bi-GRU",
             "MultitaskAIS\n(one-sided)"]
YLO = 0.2
ax.set_yscale("log")
ax.set_ylim(YLO, 900)            # MultitaskAIS sits at ~263 NM

vals = [PIVOT.loc["OVERALL", f"{m}_med_nm"] for m in BAR_ORDER]
for i, (m, v) in enumerate(zip(BAR_ORDER, vals)):
    ax.bar(i, v - YLO, bottom=YLO, width=0.72, color=FS.MODEL[m]["color"], lw=0, zorder=3)
    seeds = COMB[(COMB.model == m) & (COMB.bucket == "OVERALL")]["model_med"].values
    if len(seeds) > 1:
        FS.seed_dots(ax, i, seeds, FS.MODEL[m]["color"], width=0.14, size=2.2)
        ax.errorbar(i, seeds.mean(), yerr=seeds.std(ddof=1), fmt="none",
                    ecolor=darker(FS.MODEL[m]["color"]), elinewidth=0.7,
                    capsize=2.0, capthick=0.7, zorder=6)
gcd_val = PIVOT.loc["OVERALL", "GCD_med_nm"]
ax.axhline(gcd_val, color=FS.C["ref_red"], ls=(0, (4, 2.5)), lw=0.7, zorder=4)
ax.annotate("GCD", xy=(0.985, gcd_val), xycoords=("axes fraction", "data"),
            xytext=(0, 1.2), textcoords="offset points",
            fontsize=6, color=FS.C["ref_red"], ha="right", va="bottom")   # inside the frame
ax.set_xticks(range(len(BAR_ORDER)))
ax.set_xticklabels(BAR_NAMES, rotation=45, ha="right", fontsize=5.5)
ax.set_xlim(-0.6, len(BAR_ORDER) - 0.4)
ax.set_yticks([0.5, 1, 10, 100])
ax.set_yticklabels(["0.5", "1", "10", "100"])
ax.set_ylabel("Median error\n(NM, lower better)")
ax.set_title("Overall error — standardized 2-h cell", pad=3)
ax.text(0.03, 0.97, "dots: 5 seeds\ncaps: ±s.d.", transform=ax.transAxes,
        fontsize=5.5, color="#6E6E6E", ha="left", va="top")

# ------------------------------------------------------------------------------------------ #
#  PANEL c -- gap-level win rate vs GCD                                                        #
# ------------------------------------------------------------------------------------------ #
ax = AX_C
WIN_ORDER = ["FQGNN", "Hermite", "HABIT", "RFFGNN", "MH-GIN", "BiGRU",
             "MultitaskAIS"]
WIN_NAMES = ["FQ-GNN", "Hermite", "HABIT", "RFF-GNN", "MH-GIN", "Bi-GRU",
             "MultitaskAIS\n(one-sided)"]
for i, m in enumerate(WIN_ORDER):
    seeds = COMB[(COMB.model == m) & (COMB.bucket == "OVERALL")]["win_pct"].values
    ax.bar(i, seeds.mean(), width=0.72, color=FS.MODEL[m]["color"], lw=0, zorder=3)
    if len(seeds) > 1:
        FS.seed_dots(ax, i, seeds, FS.MODEL[m]["color"], width=0.14, size=2.2)
        ax.errorbar(i, seeds.mean(), yerr=seeds.std(ddof=1), fmt="none",
                    ecolor=darker(FS.MODEL[m]["color"]), elinewidth=0.7,
                    capsize=2.0, capthick=0.7, zorder=6)
ax.axhline(50, color=FS.C["ref_gray"], ls=(0, (4, 2.5)), lw=0.7, zorder=4)
ax.text(len(WIN_ORDER) - 0.55, 51.5, "coin flip", fontsize=5.5, color=FS.C["ref_gray"], ha="right", va="bottom")
ax.set_xticks(range(len(WIN_ORDER)))
ax.set_xticklabels(WIN_NAMES, rotation=45, ha="right", fontsize=5.5)
ax.set_xlim(-0.6, len(WIN_ORDER) - 0.4)
ax.set_ylim(0, 100)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel("Wins vs GCD\n(%, higher better)")
ax.set_title("Gap-level win rate vs GCD", pad=3)
ax.text(0.97, 0.97, "dots: 5 seeds\ncaps: ±s.d.", transform=ax.transAxes,
        fontsize=5.5, color="#6E6E6E", ha="right", va="top")

# ------------------------------------------------------------------------------------------ #
#  PANEL d -- per-gap head-to-head scatter (seed 0)                                            #
# ------------------------------------------------------------------------------------------ #
ax = AX_D
g = np.maximum(PERGAP.gcd_med.values, 4e-3)
f = np.maximum(PERGAP.model_med.values, 4e-3)
sin_norm = LogNorm(vmin=5e-3, vmax=15)
sin_c = np.clip(PERGAP.sinuosity.values, 5e-3, 15)
lo, hi = 3e-3, 4000
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.plot([lo, hi], [lo, hi], color=FS.C["ref_gray"], ls=(0, (4, 2.5)), lw=0.7, zorder=1.5)
sc = ax.scatter(g, f, c=sin_c, cmap=TEAL_CMAP, norm=sin_norm, s=3.0, alpha=0.65,
                linewidths=0, zorder=3)
ax.text(0.865, 0.915, "tie", transform=ax.transAxes, fontsize=5.5,
        color=FS.C["ref_gray"], ha="right", va="bottom", rotation=40,
        rotation_mode="anchor")
win_pct = 100.0 * (PERGAP.model_med < PERGAP.gcd_med).mean()
ax.text(0.96, 0.06, f"{win_pct:.0f}% of gaps\nbelow the diagonal",
        transform=ax.transAxes, fontsize=5.5, color=FS.C["text"], ha="right", va="bottom")
ax.set_xticks([0.01, 1, 100]); ax.set_xticklabels(["0.01", "1", "100"])
ax.set_yticks([0.01, 1, 100]); ax.set_yticklabels(["0.01", "1", "100"])
ax.set_xlabel("GCD error (NM)")
ax.set_ylabel("FQ-GNN error (NM)")
ax.set_title("Per-gap head-to-head (seed 0)", pad=3)
cb = fig.colorbar(sc, cax=CAX_D)
cb.set_ticks([0.01, 0.1, 1, 10])
cb.set_ticklabels(["0.01", "0.1", "1", "10"])
cb.set_label("Sinuosity", fontsize=6.5)
cb.ax.tick_params(labelsize=5.5, length=2.0, width=0.6)
cb.outline.set_linewidth(0.5)
cb.outline.set_edgecolor(FS.C["axis"])

# ------------------------------------------------------------------------------------------ #
#  PANEL e -- annotated per-bucket ratio heatmap (8 rows: 7 models + GCD, x 6 buckets)         #
# ------------------------------------------------------------------------------------------ #
ax = AX_E
HEAT_NAMES = ["FQ-GNN (ours)", "Hermite", "GCD", "HABIT", "RFF-GNN", "MH-GIN", "Bi-GRU",
              "MultitaskAIS (one-sided)"]
NROW = len(HEAT_ROWS)
M = np.array([[1.0] * 6 if m == "GCD" else
              [PIVOT.loc[b, f"{m}_ratio"] for b in XCATS] for m in HEAT_ROWS])
heat_norm = LogNorm(vmin=0.5, vmax=20, clip=True)
im = ax.imshow(M, cmap=BLUE_CMAP, norm=heat_norm, aspect="auto",
               extent=(0, 6, NROW, 0), origin="upper", interpolation="nearest")


def fmt(v):
    return f"{v:.0f}" if v >= 100 else (f"{v:.1f}" if v >= 10 else f"{v:.2f}")


for r in range(NROW):
    for c in range(6):
        face = mcolors.to_hex(BLUE_CMAP(heat_norm(M[r, c])))
        ax.text(c + 0.5, r + 0.5, fmt(M[r, c]), ha="center", va="center",
                fontsize=5.5, color=FS.contrast_text_color(face))
# white separator echoing the OVERALL divider; 2-px pink frame on the FQ-GNN row (optimum)
ax.plot([1, 1], [0, NROW], color="#FFFFFF", lw=1.5, zorder=4)
ax.add_patch(mpatches.Rectangle((0.006, 0.05), 5.988, 0.90, fill=False,
                                edgecolor=FS.C["pink"], lw=2.0, zorder=5, clip_on=False))
ax.set_xticks(np.arange(6) + 0.5)
ax.set_xticklabels(XLABELS)
ax.set_yticks(np.arange(NROW) + 0.5)
ax.set_yticklabels(HEAT_NAMES, fontsize=5.5)
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_title("Ratio to GCD — model × curvature matrix (mean over seeds)", pad=3)
cbe = fig.colorbar(im, cax=CAX_E)
cbe.set_ticks([0.5, 1, 2, 5, 10, 20])
cbe.set_ticklabels(["0.5", "1", "2", "5", "10", "20"])
cbe.set_label("ratio vs GCD (log)", fontsize=6.5)
cbe.ax.tick_params(labelsize=5.5, length=2.0, width=0.6)
cbe.outline.set_linewidth(0.5)
cbe.outline.set_edgecolor(FS.C["axis"])
fig.text(0.840, 0.110,
         "colour clipped at 20;\n"
         "printed values exact\n"
         "GCD row ≡ 1 by definition\n"
         "n = 1 seed: GCD,\n"
         "Hermite, HABIT",
         fontsize=5.5, color="#6E6E6E", style="italic", ha="left", va="center")

# ------------------------------------------------------------------------------------------ #
#  PANEL LETTERS (baseline grid: one y per band, flush-left of each panel block)               #
# ------------------------------------------------------------------------------------------ #
FS.panel_letter(fig, 0.008, 0.985, "a")
FS.panel_letter(fig, 0.008, 0.575, "b")
FS.panel_letter(fig, 0.370, 0.575, "c")
FS.panel_letter(fig, 0.688, 0.575, "d")
FS.panel_letter(fig, 0.008, 0.230, "e")

FS.save(fig, os.path.join(OUT, "fig3"))

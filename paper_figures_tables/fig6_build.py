"""
FIGURE 6 -- "Robustness and generality of the primitive" (stress + transfer).

Canvas 180 x 105 mm, 2 bands, 12-col grid, letters a-d.
  a (6 cols)  gap-duration sweep, TWO tracks (our real-pings / standard interp-10m), share y (log ratio)
  b (3 cols)  anchor-sparsity sweep (Context sparsity; own y-range -- no AIS-native runs here)
  c (3 cols)  seed-stability box strips (7 models + HABIT dot, log ratio)
  d (12 cols) accuracy vs capacity & speed (params-log vs ratio-log scatter, direct-labelled)

AIS-native baselines HABIT / MH-GIN / MultitaskAIS appear in a, c, d.  Their ~66-460x
ratios force the shared a/c log range up to 700; panel b keeps its own (0.5, 26) range.

Reads ONLY archived CSVs -> figures reproducible from disk.  Imports the locked design system.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator, ScalarFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _paths import FIGDATA, OUT, RESULTS  # noqa: E402,F401  (env FQGNN_FIGDATA / FQGNN_FIG_OUT / FQGNN_RESULTS_ROOT)
import fig_style as FS

COMBINED = os.path.join(RESULTS, "runs", "paper_table",
                        "combined_buckets_allmodels.csv")

FS.apply()
C = FS.C
M = FS.MODEL

# ------------------------------------------------------------------------------------------------ #
#  LOAD                                                                                             #
# ------------------------------------------------------------------------------------------------ #
gap = pd.read_csv(os.path.join(FIGDATA, "gap_sweep.csv"))
anc = pd.read_csv(os.path.join(FIGDATA, "anchor_sweep.csv"))
comb = pd.read_csv(COMBINED)
params = pd.read_csv(os.path.join(FIGDATA, "params_audit.csv"))

def agg(df, keys):
    """min / max / mean / std of the per-seed ratio."""
    g = df.groupby(keys)["ratio"]
    out = g.agg(["mean", "min", "max", "std"]).reset_index()
    return out

def darken(hexcolor, f=0.30):
    h = hexcolor.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02X%02X%02X" % (int(r * (1 - f)), int(g * (1 - f)), int(b * (1 - f)))

# panel a : per (track, gap_h, model)
gap_a = agg(gap, ["track", "gap_h", "model"])
# panel b : per (aps, model)
anc_b = agg(anc, ["aps", "model"])
# panel c : OVERALL per-seed ratios
overall = comb[comb["bucket"] == "OVERALL"].copy()

# ------------------------------------------------------------------------------------------------ #
#  FIGURE + GRID                                                                                    #
# ------------------------------------------------------------------------------------------------ #
fig = FS.new_figure(width_mm=180, height_mm=105)

L, R = 0.070, 0.990                       # data-area outer x bounds
GUT = 0.014                               # inter-panel gutter (fig-frac)
def rect(c0, c1, bot, top):
    W = R - L
    x0 = L + (c0 / 12.0) * W
    x1 = L + (c1 / 12.0) * W
    return [x0 + GUT / 2, bot, (x1 - x0) - GUT, top - bot]

# band 1 (top) and band 2 (bottom) data-area y bounds
B1_TOP, B1_BOT = 0.925, 0.585
B2_TOP, B2_BOT = 0.415, 0.165

# ---- band 1 : a (0-6 split into two subs) | b (6-9) | c (9-12) ------------------------------------
# panel a: two sub-panels sharing y, tight together
ax_a1 = fig.add_axes(rect(0.35, 3.2, B1_BOT, B1_TOP))
ax_a2 = fig.add_axes(rect(3.25, 6.0, B1_BOT, B1_TOP), sharey=ax_a1)
ax_b  = fig.add_axes(rect(6.55, 8.70, B1_BOT, B1_TOP))
ax_c  = fig.add_axes(rect(9.45, 12.0, B1_BOT, B1_TOP))    # wider gap: keeps c's y-label off b's frame

# ---- band 2 : d (0-12), full width ---------------------------------------------------------------
ax_d = fig.add_axes(rect(0.35, 12.0, B2_BOT, B2_TOP))

# shared y-limits / ticks for the ratio panels a,c -- up to 700 because MH-GIN /
# MultitaskAIS sit at 66-460x GCD.  Floor 0.45 (FQGNN dips to 0.4913 at the 3 h point;
# 5-seed band min 0.4879).
# Panel b keeps its OWN range: the AIS-native models were not run on the anchor sweep,
# so stretching its axis 27x would only flatten the three curves it actually shows.
RATIO_YLIM = (0.45, 700)
RATIO_TICKS = [1, 5, 20, 100, 500]
RATIO_YLIM_B = (0.5, 26)
RATIO_TICKS_B = [1, 2, 5, 10, 20]

def style_ratio_axis(ax, ylabel=None, show_yticklabels=True,
                     ylim=RATIO_YLIM, ticks=RATIO_TICKS):
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    ax.yaxis.set_minor_locator(NullLocator())
    if not show_yticklabels:
        # NOTE: set_yticklabels([]) would wipe the SHARED major formatter of a sharey pair
        # (killing the left sub-plot's labels too); hide this axes' labels only.
        ax.tick_params(labelleft=False)
    if ylabel:
        ax.set_ylabel(ylabel)

# ================================================================================================ #
#  PANEL a : gap-duration sweep, two preprocessing tracks                                           #
# ================================================================================================ #
TRACK_META = {
    "our_processed":    dict(ax=ax_a1, xt=[3, 6, 12, 24],  title="Raw AIS (native cadence)"),
    "standard_process": dict(ax=ax_a2, xt=[2, 6, 12, 24],  title="Standardized 10-min product"),
}
SER_A = ["fqgnn", "rffgnn", "bigru", "habit", "mhgin", "multitaskais"]
KEY = {"fqgnn": "FQGNN", "rffgnn": "RFFGNN", "bigru": "BiGRU",
       "habit": "HABIT", "mhgin": "MH-GIN", "multitaskais": "MultitaskAIS"}

for track, meta in TRACK_META.items():
    ax = meta["ax"]
    sub = gap_a[gap_a["track"] == track]
    for s in SER_A:
        mk = M[KEY[s]]
        d = sub[sub["model"] == s].sort_values("gap_h")
        if d.empty:
            continue
        x = d["gap_h"].to_numpy(float)
        y = d["mean"].to_numpy(float)
        if s == "fqgnn":
            ax.fill_between(x, d["min"], d["max"], color=mk["color"], alpha=0.18, lw=0, zorder=3)
            ax.plot(x, y, color=mk["color"], marker=mk["marker"], ls=mk["ls"], lw=mk["lw"],
                    markersize=3.6, zorder=mk["z"], clip_on=True)
        elif d["std"].isna().all():
            # HABIT: deterministic single run per horizon -> plain line, no caps
            ax.plot(x, y, color=mk["color"], marker=mk["marker"], ls=mk["ls"], lw=mk["lw"],
                    markersize=3.2, zorder=mk["z"], clip_on=True)
        else:
            ax.errorbar(x, y, yerr=d["std"], color=mk["color"], marker=mk["marker"], ls=mk["ls"],
                        lw=mk["lw"], markersize=3.2, elinewidth=0.6, capsize=2.0, capthick=0.6,
                        zorder=mk["z"], clip_on=True)
    FS.ref_hline(ax, 1.0, color=C["gcd"], label="GCD")
    ax.set_xscale("log")
    ax.set_xlim(meta["xt"][0] * 0.78, meta["xt"][-1] * 1.28)
    ax.xaxis.set_major_locator(FixedLocator(meta["xt"]))
    ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in meta["xt"]]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_title(meta["title"], fontsize=6.6, pad=3)

style_ratio_axis(ax_a1, ylabel="Median-error ratio vs GCD\n(log, lower better)", show_yticklabels=True)
style_ratio_axis(ax_a2, show_yticklabels=False)
# centred x-title under panel a; the 6-series legend lives BELOW the band (the MH-GIN /
# MultitaskAIS lines occupy the in-panel dead space), with the
# named-uncertainty micro-note to its right
r_a = rect(0.35, 6.0, B1_BOT, B1_TOP)
fig.text(r_a[0] + r_a[2] / 2, B1_BOT - 0.062, "Gap length (h)", ha="center", va="top", fontsize=7)
fig.legend(handles=FS.model_legend_handles(
               ["FQGNN", "RFFGNN", "BiGRU", "HABIT", "MH-GIN", "MultitaskAIS"]),
           loc="upper center", bbox_to_anchor=(0.313, 0.505), ncol=3, fontsize=5.3,
           handlelength=1.7, handletextpad=0.4, columnspacing=0.9, labelspacing=0.30)
fig.text(0.53, 0.478,
         "shaded = FQ-GNN 5-seed min–max  ·  caps = ±s.d.",
         ha="left", va="top", fontsize=5.3, color="#6E6E6E")

# ================================================================================================ #
#  PANEL b : anchor-sparsity sweep                                                                  #
# ================================================================================================ #
APS = [2, 4, 8]
for s in SER_A:
    mk = M[KEY[s]]
    d = anc_b[anc_b["model"] == s].sort_values("aps")
    if d.empty:          # AIS-native baselines were not run on the anchor sweep
        continue
    x = d["aps"].to_numpy(float)
    y = d["mean"].to_numpy(float)
    if s == "fqgnn":
        ax_b.fill_between(x, d["min"], d["max"], color=mk["color"], alpha=0.18, lw=0, zorder=3)
        ax_b.plot(x, y, color=mk["color"], marker=mk["marker"], ls=mk["ls"], lw=mk["lw"],
                  markersize=3.6, zorder=mk["z"])
    elif s == "bigru":
        # anchor-blind by construction: identical across aps -> flat rule + direct label
        ax_b.plot(x, y, color=mk["color"], marker=mk["marker"], ls=mk["ls"], lw=mk["lw"],
                  markersize=3.2, zorder=mk["z"])
        ax_b.annotate("Bi-GRU\n(anchor-blind)", xy=(x[-1], y[-1]), xytext=(-1, 6),
                      textcoords="offset points", fontsize=5.3, color=mk["color"],
                      ha="right", va="bottom", linespacing=0.95)
    else:
        ax_b.errorbar(x, y, yerr=d["std"], color=mk["color"], marker=mk["marker"], ls=mk["ls"],
                      lw=mk["lw"], markersize=3.2, elinewidth=0.6, capsize=2.0, capthick=0.6,
                      zorder=mk["z"])
FS.ref_hline(ax_b, 1.0, color=C["gcd"], label="GCD")
ax_b.set_xscale("log")
ax_b.set_xlim(1.6, 10.5)
ax_b.xaxis.set_major_locator(FixedLocator(APS))
ax_b.xaxis.set_major_formatter(FixedFormatter([str(a) for a in APS]))
ax_b.xaxis.set_minor_locator(NullLocator())
style_ratio_axis(ax_b, ylabel="Ratio vs GCD (log)", show_yticklabels=True,
                 ylim=RATIO_YLIM_B, ticks=RATIO_TICKS_B)
ax_b.set_title("Context sparsity", fontsize=6.8, pad=3)
ax_b.set_xlabel("Anchors per side")

# ================================================================================================ #
#  PANEL c : seed-stability box strips                                                              #
# ================================================================================================ #
C_MODELS = ["FQGNN", "RFFGNN", "CSDI", "SAITS", "BiGRU", "HABIT", "MH-GIN", "MultitaskAIS"]
C_LABEL = {"FQGNN": "FQ-GNN", "RFFGNN": "RFF-GNN", "CSDI": "CSDI", "SAITS": "SAITS",
           "BiGRU": "Bi-GRU", "HABIT": "HABIT", "MH-GIN": "MH-GIN",
           "MultitaskAIS": "MultitaskAIS (one-sided)"}   # single line: a wrapped rotated label
                                                         # collides with the MH-GIN tick label
# median annotation side: MH-GIN prints LEFT of its box (its right neighbour MultitaskAIS
# occupies the same ~370x altitude, a right-side label would cross that box)
C_ANN_SIDE = {"MH-GIN": -1}
positions = np.arange(1, len(C_MODELS) + 1)
box_data = []
for m in C_MODELS:
    vals = overall[overall["model"] == m]["ratio"].to_numpy(float)
    box_data.append(vals)

for i, m in enumerate(C_MODELS):
    col = M[m]["color"]
    vals = box_data[i]
    pos = positions[i]
    if len(vals) == 1:
        # HABIT: deterministic, a single run -> one dot instead of a degenerate box;
        # its value prints ABOVE the dot (a side label would sit on the GCD rule at 1.0)
        ax_c.plot([pos], vals, ls="none", marker="o", markersize=3.2, markerfacecolor=col,
                  markeredgewidth=0, alpha=0.9, zorder=6)
        ax_c.annotate(f"{float(vals[0]):.2f}", xy=(pos, float(vals[0])), xytext=(0, 4),
                      textcoords="offset points", fontsize=5.5, color=darken(col, 0.30),
                      ha="center", va="bottom")
        continue
    else:
        bp = ax_c.boxplot([vals], positions=[pos], widths=0.5, showfliers=False,
                          patch_artist=True, zorder=4,
                          medianprops=dict(color=col, lw=1.1),
                          boxprops=dict(facecolor="none", edgecolor=col, lw=0.7),
                          whiskerprops=dict(color=col, lw=0.7),
                          capprops=dict(color=col, lw=0.7))
        # per-seed dots (5), jittered
        xs = pos + np.linspace(-0.14, 0.14, len(vals))
        ax_c.plot(xs, vals, ls="none", marker="o", markersize=2.5, markerfacecolor=col,
                  markeredgewidth=0, alpha=0.75, zorder=6)
    # median printed beside the box: fixed dx past the box edge (0.25) + seed-dot jitter,
    # top-aligned to the median line so no number crosses the next box's lower whisker
    med = float(np.median(vals))
    side = C_ANN_SIDE.get(m, 1)
    ax_c.annotate(f"{med:.0f}" if med >= 100 else f"{med:.2f}",
                  xy=(pos + side * 0.38, med), xytext=(0, -1),
                  textcoords="offset points", fontsize=5.5, color=darken(col, 0.30),
                  ha="left" if side > 0 else "right", va="top")
FS.ref_hline(ax_c, 1.0, color=C["gcd"], label="GCD")
style_ratio_axis(ax_c, ylabel="Ratio vs GCD (log)", show_yticklabels=True)
ax_c.set_xlim(0.4, len(C_MODELS) + 1.35)
ax_c.set_xticks(positions)
ax_c.set_xticklabels([C_LABEL[m] for m in C_MODELS], rotation=45, ha="right", fontsize=5.0)
ax_c.set_title("Seed stability", fontsize=6.8, pad=3)

# ================================================================================================ #
#  PANEL d : accuracy vs capacity & speed                                                           #
# ================================================================================================ #
PKEY = {"FQGNN": "FQGNN", "RFF-GNN": "RFFGNN", "BiGRU": "BiGRU",
        "CSDI": "CSDI", "SAITS": "SAITS", "BF-BiLSTM": "BF-BiLSTM",
        "HABIT": "HABIT", "MH-GIN": "MH-GIN", "MultitaskAIS": "MultitaskAIS"}
def fmt_time(s):
    if s >= 60:
        return f"{s/60:.0f} min"
    return f"{s:.1f} s" if s >= 1 else f"{s:.2f} s"

# HABIT is parameter-free (0 trainable AND 0 frozen): a log axis has no zero, so it is drawn
# at a nominal position LEFT of a marked axis break, with an explicit "0 params" annotation
# (honest placement instead of silently dropping the point).
HABIT_X = 1.35e4
XLIM_E = (1.05e4, 1.9e6)
BREAK_X = (1.85e4, 2.20e4)                 # the two slanted break ticks on the x-spine

# explicit (dx pt, dy pt, ha, va) for the model LABEL and the gray TIME sub-annotation, tuned to
# avoid every collision (title / neighbouring points / axis edges)
LBL = {
    "FQGNN":        ( 9,   0, "left",  "center"),   # centred on the marker: at ratio 0.56 the
                                                    # point rides the y-floor between the GCD
                                                    # rule and the x-spine, and top/bottom
                                                    # anchoring strikes one or the other
    "RFFGNN":       ( 0,  10, "center","bottom"),
    "BiGRU":        ( 0,  10, "center","bottom"),
    "CSDI":         ( 0,  -2, "center","top"),
    "SAITS":        (-8,   7, "right", "bottom"),
    "BF-BiLSTM":    (-9,   0, "right", "center"),
    "HABIT":        (-4,  10, "left",  "bottom"),
    "MH-GIN":       (-9,   0, "right", "center"),
    "MultitaskAIS": ( 9,   0, "left",  "center"),
}
TLBL = {
    "FQGNN":        ( 9,  -8, "left",  "top"),
    "RFFGNN":       ( 0,  -9, "center","top"),
    "BiGRU":        ( 0,  -9, "center","top"),
    "CSDI":         ( 0,  -9, "center","top"),
    "SAITS":        ( 2,  -9, "left",  "top"),   # below-RIGHT: a left-side time label would
    "BF-BiLSTM":    (-9,  -8, "right", "top"),   # cross the CSDI label at the compressed scale
    "HABIT":        (-4,  21, "left",  "bottom"),  # stacked ABOVE the model label -- the point
    "MH-GIN":       (-9,  -8, "right", "top"),     # sits ~4 pt over the axis, below-text would
    "MultitaskAIS": ( 9,  -8, "left",  "top"),     # land on the spine + break ticks
}
for _, row in params.iterrows():
    name = row["model"]
    if name not in PKEY:
        continue
    key = PKEY[name]
    mk = M[key]
    param_free = (row["n_params_trainable"] == 0) and (row["n_params_frozen"] == 0)
    frozen = (row["n_params_trainable"] == 0) and not param_free
    xval = HABIT_X if param_free else (
        row["n_params_frozen"] if frozen else row["n_params_trainable"])
    yval = row["test_ratio_overall"]
    ax_d.plot([xval], [yval], marker=mk["marker"], ls="none", markersize=5.2,
              markerfacecolor=mk["color"], markeredgecolor=mk["color"], markeredgewidth=0.5,
              zorder=6, clip_on=True)
    dx, dy, ha, va = LBL[key]
    disp = mk["label"] + (" (frozen)" if frozen else "")
    ax_d.annotate(disp, xy=(xval, yval), xytext=(dx, dy), textcoords="offset points",
                  fontsize=5.8, color=mk["color"], ha=ha, va=va, zorder=7)
    # inference-time small gray sub-annotation, always stacked directly below the label
    # (no inference timing for MH-GIN / MultitaskAIS in params_audit.csv, so no time label)
    t_raw = row["inference_s_per_770gaps"]
    if pd.notna(t_raw):
        tt = fmt_time(float(t_raw))
        if param_free:
            tt = "0 params · " + tt
        tdx, tdy, tha, tva = TLBL[key]
        ax_d.annotate(tt, xy=(xval, yval), xytext=(tdx, tdy), textcoords="offset points",
                      fontsize=5.3, color="#8A8A8A", ha=tha, va=tva, zorder=7)
FS.ref_hline(ax_d, 1.0, color=C["gcd"], label="GCD")
ax_d.set_xscale("log")
ax_d.set_yscale("log")
ax_d.set_xlim(*XLIM_E)
ax_d.set_ylim(0.42, 700)                    # MH-GIN 372x / MultitaskAIS 379x
ax_d.yaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 50, 100, 500]))
ax_d.yaxis.set_major_formatter(FixedFormatter(["1", "2", "5", "10", "50", "100", "500"]))
ax_d.yaxis.set_minor_locator(NullLocator())
# axis-break ticks separating the nominal 0-param slot from the true log scale
tr_e = mtransforms.blended_transform_factory(ax_d.transData, ax_d.transAxes)
for bx in BREAK_X:
    ax_d.plot([bx], [0], transform=tr_e, marker=(2, 0, 60), markersize=4.0,
              color=C["axis"], markeredgewidth=0.75, ls="none", clip_on=False, zorder=8)
ax_d.set_xlabel("Trainable parameters")
ax_d.set_ylabel("Ratio vs GCD (log)")
ax_d.set_title("Accuracy vs capacity & speed", fontsize=6.8, pad=3)
FS.condition_strip(ax_d, "inference: 770-gap test pass  |  gray = wall-clock time "
                         "(MH-GIN, MultitaskAIS not timed)  |  axis break: HABIT has 0 params",
                   dy=-0.44)

# ================================================================================================ #
#  PANEL LETTERS  (top-left outside corner, on band baselines)                                      #
# ================================================================================================ #
LY1 = 0.965
LY2 = 0.447
FS.panel_letter(fig, rect(0.35, 3.2, B1_BOT, B1_TOP)[0] - 0.052, LY1, "a")
FS.panel_letter(fig, rect(6.55, 9.0, B1_BOT, B1_TOP)[0] - 0.040, LY1, "b")
FS.panel_letter(fig, rect(9.30, 12.0, B1_BOT, B1_TOP)[0] - 0.040, LY1, "c")
FS.panel_letter(fig, rect(0.35, 12.0, B2_BOT, B2_TOP)[0] - 0.052, LY2, "d")

# ------------------------------------------------------------------------------------------------ #
FS.save(fig, os.path.join(OUT, "fig6"))
print("done")

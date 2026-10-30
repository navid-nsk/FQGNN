"""Extended Data Figure 2 -- full training curves, every trained model x 5 seeds.

ONE lettered panel 'a' spanning a sub-panel grid (the extended small-multiples
idiom): one sub-plot per model, each = that model's 5 per-seed
curves (native objective) in the model's LOCKED colour (fig_style MODEL) at
~0.8 pt, plus the across-seed mean at 1.25 pt.  With the AIS-native baselines
the grid is 4 + 3 over two rows:

  row 1 (shared log-ratio y): FQGNN / RFFGNN / BiGRU / SAITS -- native metric is
        val_ratio (vs GCD); shared limits + FS.ref_hline at 1.0.
        Early-stop marker at each seed's last recorded epoch.
  row 2 (own axes): MH-GIN  -- val_ratio too, but it lives at 320-1390x GCD, so it
        gets its OWN log axis (same metric, own scale; GCD floor off-scale, noted).
        CSDI -- early-stops on val *denoising loss*; own linear axis, no GCD rule.
        MultitaskAIS -- the vendor TF pipeline exposes NO validation metric at all;
        the only honest curve is the TRAINING loss (-ELBO/t), labelled as exactly
        that.  No early-stop markers on MH-GIN / MultitaskAIS (fixed-length runs).

HABIT is a deterministic route-frequency lookup: it has no training and therefore
no panel -- noted in the caption strip.

Reads ONLY the archived epochs.csv per model per seed; no recomputation.
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
from matplotlib.lines import Line2D  # noqa: E402

FS.apply()

TA = RESULTS
GS6 = os.path.join(TA, "runs", "gap_sweep", "our_processed", "gap_6h")

# ------------------------------------------------------------------------------------------ #
#  DATA -- one epochs.csv per model per seed (columns differ per model)                        #
# ------------------------------------------------------------------------------------------ #
# (model, run_dir, native val column, y-axis kind, draw last-epoch early-stop marker)
SPECS = [
    ("FQGNN",  os.path.join(TA, "runs", "fqnative_opmstack_5seed"),               "val_ratio",  "ratio", True),
    ("RFFGNN", os.path.join(TA, "baselines", "RFFGNN", "runs_5seed_gaps"), "val_ratio",  "ratio", True),
    ("BiGRU",  os.path.join(TA, "baselines", "BiGRU",  "runs_5seed_gaps"), "val_ratio",  "ratio", True),
    ("SAITS",  os.path.join(TA, "baselines", "SAITS",  "runs_5seed_gaps"), "val_ratio",  "ratio", True),
    ("MH-GIN", os.path.join(GS6, "mhgin"),                                 "val_ratio",  "ratio_own", False),
    ("CSDI",   os.path.join(TA, "baselines", "CSDI",   "runs_5seed_gaps"), "val_loss",   "loss", True),
    ("MultitaskAIS", os.path.join(GS6, "multitaskais"),                    "train_loss", "elbo", False),
]
ROW1 = [s for s in SPECS if s[3] == "ratio"]          # 4 shared-scale panels
ROW2 = [s for s in SPECS if s[3] != "ratio"]          # 3 own-axis panels
N_SEED = 5


def load_seeds(run_dir, col):
    """Return list of (epoch_array, value_array) per seed (skips missing/empty)."""
    out = []
    for s in range(N_SEED):
        p = os.path.join(run_dir, f"seed{s}", "epochs.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        if col not in df.columns or len(df) == 0:
            continue
        m = df[col].notna().values
        out.append((df["epoch"].values[m].astype(float), df[col].values[m].astype(float)))
    return out


def seed_mean(seeds):
    """Across-seed mean on the union epoch grid (each seed held at its last value past
    its own early stop, so the mean is defined over the full x-range)."""
    if not seeds:
        return np.array([]), np.array([])
    grid = np.unique(np.concatenate([e for e, _ in seeds]))
    stack = []
    for e, v in seeds:
        # step-hold: value at grid point = value at the largest recorded epoch <= grid point
        idx = np.searchsorted(e, grid, side="right") - 1
        idx = np.clip(idx, 0, len(v) - 1)
        held = v[idx]
        held[grid < e[0]] = np.nan            # before this seed's first epoch -> undefined
        stack.append(held)
    return grid, np.nanmean(np.vstack(stack), axis=0)


DATA = {m: load_seeds(rd, col) for m, rd, col, _, _ in SPECS}


def darker(hexcolor, f=0.70):
    r, g, b = mcolors.to_rgb(hexcolor)
    return (r * f, g * f, b * f)


# shared ratio y-limits (row 1): FQGNN dips to ~0.67, BiGRU/SAITS peak ~74 -> 0.6..90 log
RATIO_YLIM = (0.6, 90.0)
RATIO_YTICKS = [1, 2, 5, 10, 20, 50]
# MH-GIN own log ratio axis: observed 317..1387
MHGIN_YLIM = (250.0, 1600.0)
MHGIN_YTICKS = [300, 500, 1000, 1500]
# CSDI own linear axis: observed 0.003..0.093
LOSS_YLIM = (0.0, 0.10)
LOSS_YTICKS = [0.0, 0.025, 0.05, 0.075, 0.10]
# MultitaskAIS own linear axis (training loss = -ELBO/t): observed 7.1..20.4
ELBO_YLIM = (6.0, 21.5)
ELBO_YTICKS = [8, 12, 16, 20]

# per-model native x-range (epoch counts differ hugely: FQGNN ~20, CSDI ~100)
XMAX = {}
for m, rd, col, _, _ in SPECS:
    seeds = DATA[m]
    XMAX[m] = max((e.max() for e, _ in seeds), default=1.0)

# ------------------------------------------------------------------------------------------ #
#  LAYOUT -- 4 + 3 grid over two rows, under ONE letter 'a' (canvas 180 x 118 mm)              #
# ------------------------------------------------------------------------------------------ #
fig = FS.new_figure(180, 118)

LEFT, RIGHT = 0.062, 0.988
# row 1: four cells sharing the ratio scale -> only the leftmost keeps y-tick labels
R1_TOP, R1_H, GUT1 = 0.905, 0.315, 0.022
cellw1 = (RIGHT - LEFT - GUT1 * 3) / 4
AXES1 = [fig.add_axes([LEFT + i * (cellw1 + GUT1), R1_TOP - R1_H, cellw1, R1_H])
         for i in range(4)]
# row 2: three own-axis cells, wider gutters for their individual y-axes + labels;
# extra left margin -- MH-GIN's 4-digit tick labels + 2-line y-label need more room than
# row 1's shared axis
LEFT2 = 0.082
R2_TOP, R2_H, GUT2 = 0.470, 0.305, 0.078
cellw2 = (RIGHT - LEFT2 - GUT2 * 2) / 3
AXES2 = [fig.add_axes([LEFT2 + i * (cellw2 + GUT2), R2_TOP - R2_H, cellw2, R2_H])
         for i in range(3)]

# ------------------------------------------------------------------------------------------ #
#  DRAW each sub-panel                                                                         #
# ------------------------------------------------------------------------------------------ #
for i, ((m, rd, col, kind, stop_marker), ax) in enumerate(
        list(zip(ROW1, AXES1)) + list(zip(ROW2, AXES2))):
    s = FS.MODEL[m]
    seeds = DATA[m]
    row1_pos = i if kind == "ratio" else None          # position within the shared row

    if kind == "ratio":
        ax.set_yscale("log")
        ax.set_ylim(*RATIO_YLIM)
        # GCD floor rule at 1.0 (dashed grey), labelled once on the line (leftmost only)
        ax.axhline(1.0, color=FS.C["gcd"], ls=(0, (4, 2.5)), lw=0.7, zorder=1.5)
        if row1_pos == 0:
            ax.annotate("GCD floor", xy=(0.5, 1.0), xycoords=("axes fraction", "data"),
                        xytext=(0, 1.5), textcoords="offset points",
                        fontsize=5.2, color=FS.C["gcd"], ha="center", va="bottom")
    elif kind == "ratio_own":
        ax.set_yscale("log")
        ax.set_ylim(*MHGIN_YLIM)
    elif kind == "loss":
        ax.set_ylim(*LOSS_YLIM)
    else:                                              # elbo
        ax.set_ylim(*ELBO_YLIM)

    ax.set_xlim(-0.02 * XMAX[m], XMAX[m] * 1.05)

    # 5 thin per-seed curves (~0.8 pt) in the model's locked colour
    for e, v in seeds:
        ax.plot(e, v, color=s["color"], lw=0.8, alpha=0.55, zorder=s["z"] - 1,
                solid_capstyle="round")
        if stop_marker:
            # early-stop marker at the seed's last recorded epoch (down-triangle)
            ax.plot(e[-1], v[-1], marker="v", markersize=3.6, markerfacecolor=s["color"],
                    markeredgecolor=darker(s["color"]), markeredgewidth=0.5,
                    ls="none", zorder=s["z"] + 1, clip_on=False)

    # across-seed mean at 1.25 pt (drawn on top)
    ge, gm = seed_mean(seeds)
    ok = np.isfinite(gm)
    ax.plot(ge[ok], gm[ok], color=darker(s["color"], 0.82), lw=1.25, zorder=s["z"] + 2,
            solid_capstyle="round")

    # axes cosmetics
    ax.set_xlabel("Epoch")
    ax.set_title(s["label"], pad=3)
    if kind == "ratio":
        ax.set_yticks(RATIO_YTICKS)
        ax.set_yticklabels([str(t) for t in RATIO_YTICKS])
        ax.minorticks_off()
        if row1_pos == 0:
            ax.set_ylabel("Val. ratio vs GCD (log)\n(lower better)", fontsize=6.5)   # 7.5 pt clipped at the canvas top
        else:
            ax.set_yticklabels([])            # shared scale -> suppress repeat labels
            ax.set_ylabel("")
    elif kind == "ratio_own":
        ax.set_yticks(MHGIN_YTICKS)
        ax.set_yticklabels([str(t) for t in MHGIN_YTICKS])
        ax.minorticks_off()
        ax.set_ylabel("Val ratio vs GCD (log)\n(own scale, lower better)", labelpad=2)
        ax.annotate("same metric as top row; GCD floor = 1 off-scale\n"
                    "fixed 30 epochs (no early stop)",
                    xy=(0.97, 0.03), xycoords="axes fraction", fontsize=5.2,
                    color="#6E6E6E", ha="right", va="bottom", linespacing=1.25)
    elif kind == "loss":
        ax.set_yticks(LOSS_YTICKS)
        ax.set_yticklabels([f"{t:.3f}".rstrip("0").rstrip(".") if t else "0"
                            for t in LOSS_YTICKS])
        ax.set_ylabel("Val denoising loss\n(lower better)", labelpad=2)
    else:                                              # elbo
        ax.set_yticks(ELBO_YTICKS)
        ax.set_yticklabels([str(t) for t in ELBO_YTICKS])
        ax.set_ylabel("Training loss = −ELBO / t\n(lower better)", labelpad=2)
        ax.annotate("TRAINING objective, not a val series\n"
                    "(vendor pipeline has no val selection);\n"
                    "fixed 20 epochs (no early stop)",
                    xy=(0.97, 0.97), xycoords="axes fraction", fontsize=5.2,
                    color="#6E6E6E", ha="right", va="top", linespacing=1.25)

    # sparse x ticks (4-6 major)
    ax.locator_params(axis="x", nbins=5)

# ------------------------------------------------------------------------------------------ #
#  in-panel legend (mean vs seed vs early-stop marker) -- placed once, in the FQGNN panel      #
# ------------------------------------------------------------------------------------------ #
axL = AXES1[0]
handles = [
    Line2D([0], [0], color=darker(FS.C["pink"], 0.82), lw=1.25, label="across-seed mean"),
    Line2D([0], [0], color=FS.C["pink"], lw=0.8, alpha=0.55, label="single seed (n = 5)"),
    Line2D([0], [0], color=FS.C["pink"], marker="v", ls="none", markersize=3.6,
           markeredgecolor=darker(FS.C["pink"]), markeredgewidth=0.5, label="early stop"),
]
axL.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.0),
           fontsize=5.2, handlelength=1.6, labelspacing=0.3, handletextpad=0.5,
           borderaxespad=0.2)

# ------------------------------------------------------------------------------------------ #
#  ONE condition strip under the grid (centered)                                               #
# ------------------------------------------------------------------------------------------ #
fig.text(0.5, 0.060,
         "5 seeds per model  |  early stop on val (marker); MH-GIN + MultitaskAIS train "
         "fixed-length (no marker)  |  top row shares log y; MH-GIN same metric, own log y; "
         "CSDI + MultitaskAIS own linear y",
         fontsize=5.6, color="#6E6E6E", ha="center", va="center")
fig.text(0.5, 0.032,
         "MultitaskAIS curve = TF training loss (−ELBO/t) -- the vendor exposes no "
         "validation metric  |  HABIT: deterministic route-frequency lookup -- no training, "
         "hence no panel",
         fontsize=5.6, color="#6E6E6E", ha="center", va="center")

# ------------------------------------------------------------------------------------------ #
#  PANEL LETTER -- one 'a' for the whole 7-sub-panel block (small-multiples share one)         #
# ------------------------------------------------------------------------------------------ #
FS.panel_letter(fig, 0.008, 0.982, "a")
fig.text(0.032, 0.974, "Training curves — every trained model × 5 seeds",
         fontsize=7.5, color=FS.C["text"], ha="left", va="top")

FS.save(fig, os.path.join(OUT, "ED2"))

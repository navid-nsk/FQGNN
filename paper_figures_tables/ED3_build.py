"""Extended Data Figure 3 -- baseline hyperparameter-tuning transparency.

Documents that the two tuned neural baselines (CSDI, SAITS) were given an honest,
real Optuna search, and that -- even at their selected optima -- their 5-seed spread
dwarfs FQ-GNN's.  Nothing here is our method winning by under-tuning a rival.

Panels (canvas 180 x 115 mm):
  a  CSDI unstable val signal:  the REAL 16-trial TPE scatter (navy hollow, log y), val
                          median-NM spanning 487 -> 5903 on the tuning subsample -- there is
                          NO stable optimum, so no trial is starred.  The config actually
                          DEPLOYED for the paper (lr 9.2e-4, T100) came from a prior capped
                          sweep and was confirmed by a 5-seed run (test 405 +/- 178 NM);
                          it is annotated, not drawn as a rule (different sweep).
  b  SAITS Optuna search: the genuine 20-trial TPE scatter, val median-NM (log),
                          best trial gold-starred + annotated, 'selected' rule.
  c  Baseline seed instability (KEY EXHIBIT): per-seed OVERALL ratio-vs-GCD boxes for
                          MH-GIN / MultitaskAIS / CSDI / SAITS / RFF-GNN / FQ-GNN, log y,
                          5 dots each, GCD reference at 1.0, median printed on each box.
                          The AIS-native pair ran on VENDOR-DEFAULT
                          configs (not Optuna-tuned); the panel title says "baseline",
                          the tuning story of panels a-b applies to CSDI / SAITS only.

Reads ONLY archived run outputs (the two Optuna DBs, the two candidate final_summary.json,
and figdata/combined_buckets_allmodels.csv).  No recomputation, no fabricated trials.
"""
from __future__ import annotations

import json
import os
import sqlite3
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

BASE = os.path.join(RESULTS, "baselines")
CSDI_DB = os.path.join(BASE, "CSDI", "hp_optuna", "gap_6h", "csdi_hp.db")
CSDI_CAP = os.path.join(BASE, "CSDI", "hp_optuna", "gap_6h",
                        "final_capped_shorttrials_406nm", "final_summary.json")
SAITS_TRIALS = os.path.join(BASE, "SAITS", "hp_saits", "trials.csv")


def darker(hexcolor, f=0.72):
    r, g, b = mcolors.to_rgb(hexcolor)
    return (r * f, g * f, b * f)


# ------------------------------------------------------------------------------------------ #
#  DATA -- CSDI                                                                                #
# ------------------------------------------------------------------------------------------ #
# CSDI full-trial Optuna study: 16 COMPLETE trials.  Read the raw DB
# (optuna load also works, but sqlite is dependency-free) and surface every finished trial's
# validation median-NM honestly -- the point of panel a is that this val signal is UNSTABLE.
con = sqlite3.connect(CSDI_DB)
cur = con.cursor()
# map optuna trial_id -> displayed trial number, state, and completed value
csdi_number = dict(cur.execute("SELECT trial_id, number FROM trials").fetchall())
csdi_states = dict(cur.execute("SELECT trial_id, state FROM trials").fetchall())
csdi_vals = dict(cur.execute("SELECT trial_id, value FROM trial_values").fetchall())
con.close()
# finished (trial number, val median-NM) pairs, sorted by trial number
csdi_trials = sorted(
    (csdi_number[tid], v) for tid, v in csdi_vals.items()
    if v is not None and str(csdi_states.get(tid, "")).upper() == "COMPLETE")
csdi_tnum = np.array([t for t, _ in csdi_trials], dtype=float)
csdi_tval = np.array([v for _, v in csdi_trials], dtype=float)
csdi_n_finish = len(csdi_trials)
csdi_val_lo, csdi_val_hi = csdi_tval.min(), csdi_tval.max()

# The config DEPLOYED for the paper (Fig 3 CSDI) was picked from a PRIOR capped sweep and
# confirmed by a full 5-seed run -> test 405 +/- 178 NM.  Read those seed values for the
# annotation only (it is a different sweep, so it is NOT drawn as a rule on this scatter).
cap = json.load(open(CSDI_CAP))
dep_lr = cap["hp"]["lr"]                       # 9.2e-4 (deployed)
dep_T = cap["hp"]["num_steps"]                 # 100
dep_test_mean = cap["test_median_nm_mean"]     # 405.7
dep_test_std = cap["test_median_nm_std"]       # 178.2

# ------------------------------------------------------------------------------------------ #
#  DATA -- SAITS  (genuine 20-trial Optuna)                                                    #
# ------------------------------------------------------------------------------------------ #
S = pd.read_csv(SAITS_TRIALS)
S = S[S.state == "COMPLETE"].copy()
s_best_i = int(S.val_median_nm.idxmin())
s_best = S.loc[s_best_i]
s_best_trial = int(s_best.trial)
s_best_val = float(s_best.val_median_nm)

# ------------------------------------------------------------------------------------------ #
#  DATA -- 5-seed OVERALL ratio-vs-GCD (panel c)                                               #
# ------------------------------------------------------------------------------------------ #
COMB = pd.read_csv(os.path.join(FIGDATA, "combined_buckets_allmodels.csv"))
OV = COMB[COMB.bucket == "OVERALL"]
ratios = {m: np.sort(OV[OV.model == m].ratio.values)
          for m in ["MH-GIN", "MultitaskAIS", "CSDI", "SAITS", "RFFGNN", "FQGNN"]}

# ------------------------------------------------------------------------------------------ #
#  LAYOUT (fig fractions; canvas 180 x 115 mm, 3 panels a|b|c across two-ish bands)            #
# ------------------------------------------------------------------------------------------ #
fig = FS.new_figure(180, 115)

AX_A = fig.add_axes([0.090, 0.195, 0.232, 0.705])
AX_B = fig.add_axes([0.425, 0.195, 0.232, 0.705])
AX_C = fig.add_axes([0.760, 0.195, 0.222, 0.705])

# ------------------------------------------------------------------------------------------ #
#  PANEL a -- CSDI tuning story                                                                #
# ------------------------------------------------------------------------------------------ #
ax = AX_A
ax.set_yscale("log")
navy = FS.C["navy"]

YLO = 300.0
YHI = 8500.0
ax.set_xlim(-1.2, 16.0)
ax.set_ylim(YLO, YHI)

# gray band spanning the full trial spread (487 -> 5903): emphasises how noisy the val signal
# is on the tuning subsample -- there is no stable optimum for TPE to lock onto.
ax.axhspan(csdi_val_lo, csdi_val_hi, color=FS.C["sep"], alpha=0.22, lw=0, zorder=1)
ax.axhline(csdi_val_lo, color=FS.C["ref_gray"], ls=(0, (2, 2)), lw=0.55, zorder=2)
ax.axhline(csdi_val_hi, color=FS.C["ref_gray"], ls=(0, (2, 2)), lw=0.55, zorder=2)

# all 16 finished trials: navy hollow circles (no single trial starred -- TPE did NOT cleanly
# choose one; the deployed config came from a different, prior sweep).
ax.plot(csdi_tnum, csdi_tval, ls="none", marker="o", markersize=4.0,
        markerfacecolor="none", markeredgecolor=navy, markeredgewidth=0.9, zorder=6)

# annotation 1: the raw fact -- 16 trials, huge val spread on the tuning subsample.
ax.annotate(f"16 TPE trials; val NM spans\n{csdi_val_lo:.0f}–{csdi_val_hi:.0f} "
            "on the tuning subsample",
            xy=(0.045, 0.955), xycoords="axes fraction", fontsize=5.5,
            color=FS.C["text"], ha="left", va="top", linespacing=1.25)
# annotation 2: the deployed config was picked from a PRIOR sweep and confirmed by 5 seeds.
ax.annotate(f"deployed config (lr {dep_lr * 1e4:.1f}e-4, T{dep_T:.0f})\n"
            f"confirmed by 5-seed run → {int(dep_test_mean)} ± {int(dep_test_std)} NM test",
            xy=(0.045, 0.135), xycoords="axes fraction", fontsize=5.5,
            color=navy, ha="left", va="top", linespacing=1.25)
# cross-reference to the seed-instability panel c.
ax.annotate("cf. seed instability, panel c", xy=(0.955, 0.045), xycoords="axes fraction",
            fontsize=5.5, color="#6E6E6E", ha="right", va="bottom")

ax.set_xticks([0, 4, 8, 12])
ax.set_yticks([300, 500, 1000, 2000, 5000])
ax.set_yticklabels(["300", "500", "1000", "2000", "5000"])
ax.yaxis.set_minor_formatter(plt.NullFormatter())
ax.set_xlabel("Trial")
ax.set_ylabel("Validation median NM\n(subsample, log, lower better)")
ax.set_title("CSDI search: unstable val signal", pad=3)
FS.condition_strip(ax, "20-trial TPE on tuning subsample\n"
                       "lr capped 1e-3 (3e-3 destabilised)",
                   dy=-0.135, x=0.0, ha="left")

# ------------------------------------------------------------------------------------------ #
#  PANEL b -- SAITS Optuna (genuine 20-trial scatter)                                          #
# ------------------------------------------------------------------------------------------ #
ax = AX_B
blue = FS.C["blue"]
ax.set_yscale("log")
xt = S.trial.values.astype(float)
yt = S.val_median_nm.values.astype(float)
# non-best trials: hollow blue diamonds
mask = S.trial.values != s_best_trial
ax.plot(xt[mask], yt[mask], ls="none", marker="D", markersize=3.4,
        markerfacecolor=blue, markeredgecolor=darker(blue), markeredgewidth=0.4,
        alpha=0.85, zorder=4)
# best trial: gold star
ax.plot([s_best_trial], [s_best_val], ls="none", marker="*", markersize=10,
        markerfacecolor=FS.C["gold"], markeredgecolor=darker(FS.C["gold"]),
        markeredgewidth=0.6, zorder=7)
# selected rule
FS.ref_hline(ax, s_best_val, color=FS.C["ref_gray"], label="selected", lw=0.7)
ax.annotate(f"best (trial {s_best_trial}):\nn_groups 3, d256, lr 1.9e-3\n"
            f"(val NM {s_best_val:.1f})",
            xy=(s_best_trial, s_best_val), xytext=(6, -2), textcoords="offset points",
            fontsize=5.0, color=FS.C["text"], ha="left", va="top",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=FS.C["sep"]))
ax.set_xlim(-1.2, 20.5)
ax.set_ylim(70, 500)
ax.set_xticks([0, 5, 10, 15, 20])
ax.set_yticks([80, 100, 200, 300, 400])
ax.set_yticklabels(["80", "100", "200", "300", "400"])
ax.yaxis.set_minor_formatter(plt.NullFormatter())   # kill stray 6x10^1 minor label
ax.set_xlabel("Trial")
ax.set_ylabel("Validation median NM\n(subsample, lower better)")
ax.set_title("SAITS hyperparameter search (Optuna)", pad=3)
ax.plot([18.6], [430], marker="*", markersize=6.5, markerfacecolor=FS.C["gold"],
        markeredgecolor=darker(FS.C["gold"]), markeredgewidth=0.5, zorder=8)
ax.annotate("20-trial TPE\nselected config", xy=(18.6, 430), xytext=(-6, 0),
            textcoords="offset points", fontsize=5.0, color="#6E6E6E",
            ha="right", va="center")
FS.condition_strip(ax, "TPE search, 20 trials all complete · gap length 6 h\n"
                       "metric: subsample val median NM", dy=-0.135, x=0.0, ha="left")

# ------------------------------------------------------------------------------------------ #
#  PANEL c -- tuned-baseline seed instability (KEY EXHIBIT)                                    #
# ------------------------------------------------------------------------------------------ #
ax = AX_C
ax.set_yscale("log")
# worst -> best, left to right; the AIS-native pair (vendor-default configs) tops the strip
C_ORDER = ["MH-GIN", "MultitaskAIS", "CSDI", "SAITS", "RFFGNN", "FQGNN"]
C_NAMES = ["MH-GIN", "MultitaskAIS\n(one-sided)", "CSDI", "SAITS", "RFF-GNN", "FQ-GNN\n(ours)"]
C_COL = {m: FS.MODEL[m]["color"] for m in C_ORDER}
# annotation placement: MH-GIN prints BELOW its box (leftmost strip -- a left-side number
# would clip at the y-spine, a right-side one would cross the equal-altitude MultitaskAIS
# box); FQ-GNN prints LEFT (rightmost strip -- a right-side number clips at the panel edge);
# CSDI prints LEFT (a right-side number crosses the SAITS lower whisker)
C_ANN_SIDE = {"MH-GIN": 0, "FQGNN": -1, "CSDI": -1}

positions = np.arange(len(C_ORDER))
box_data = [ratios[m] for m in C_ORDER]
bp = ax.boxplot(box_data, positions=positions, widths=0.52, patch_artist=True,
                showfliers=False, zorder=3,
                medianprops=dict(color=FS.C["axis"], lw=1.0),
                whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7),
                boxprops=dict(lw=0.7))
for i, m in enumerate(C_ORDER):
    col = C_COL[m]
    bp["boxes"][i].set(facecolor=col, alpha=0.32, edgecolor=darker(col))
    bp["whiskers"][2 * i].set_color(darker(col))
    bp["whiskers"][2 * i + 1].set_color(darker(col))
    bp["caps"][2 * i].set_color(darker(col))
    bp["caps"][2 * i + 1].set_color(darker(col))
    # 5 per-seed dots
    vals = ratios[m]
    jx = positions[i] + np.linspace(-0.15, 0.15, len(vals))
    ax.plot(jx, vals, ls="none", marker="o", markersize=2.6,
            markerfacecolor="#2B2B2B", markeredgewidth=0, alpha=0.6, zorder=5)
    # median printed on the box
    med = np.median(vals)
    txt = f"{med:.0f}" if med >= 100 else f"{med:.2f}"
    side = C_ANN_SIDE.get(m, 1)
    if side == 0:                              # print under the lower whisker
        ax.annotate(txt, xy=(positions[i], vals.min()), xytext=(0, -4),
                    textcoords="offset points", fontsize=5.5, color=FS.C["text"],
                    ha="center", va="top", zorder=6)
    else:
        ax.annotate(txt, xy=(positions[i], med), xytext=(9 * side, 0),
                    textcoords="offset points", fontsize=5.5, color=FS.C["text"],
                    ha="left" if side > 0 else "right", va="center", zorder=6)

# GCD reference at 1.0
FS.ref_hline(ax, 1.0, color=FS.C["gcd"], label="GCD floor", lw=0.75)

ax.set_xlim(-0.6, 5.6)
ax.set_ylim(0.55, 700)           # extended: MH-GIN / MultitaskAIS at ~365-405x
ax.set_xticks(positions)
ax.set_xticklabels(C_NAMES, rotation=45, ha="right", fontsize=5.5)
ax.set_yticks([0.7, 1, 2, 5, 10, 100, 500])
ax.set_yticklabels(["0.7", "1", "2", "5", "10", "100", "500"])
ax.yaxis.set_minor_formatter(plt.NullFormatter())
ax.set_ylabel("Overall error ratio vs GCD\n(5 seeds, lower better)")
ax.set_title("Baseline seed stability", pad=3)
# dead band between the ~10x cluster and the ~370x pair -> the key numbers live there
ax.text(0.97, 0.60,
        "CSDI ratio spans 2.3–10.5 across seeds;\n"
        "FQ-GNN 0.56–0.57 (only model < 1)\n"
        "MH-GIN, MultitaskAIS: vendor defaults",
        transform=ax.transAxes, fontsize=5.5, color=FS.C["text"], ha="right", va="center")
FS.condition_strip(ax, "box: IQR · line: median · dots: 5 seeds\n"
                       "metric: overall median-NM ratio to GCD",
                   dy=-0.195, x=0.0, ha="left")

# ------------------------------------------------------------------------------------------ #
#  PANEL LETTERS                                                                               #
# ------------------------------------------------------------------------------------------ #
FS.panel_letter(fig, 0.010, 0.985, "a")
FS.panel_letter(fig, 0.345, 0.985, "b")
FS.panel_letter(fig, 0.685, 0.985, "c")

FS.save(fig, os.path.join(OUT, "ED3"))
print("CSDI finished trials:", csdi_n_finish,
      "val NM spread %.0f -> %.0f" % (csdi_val_lo, csdi_val_hi))
print("CSDI deployed (prior sweep): lr %.2e T%.0f -> test %.1f +/- %.1f NM"
      % (dep_lr, dep_T, dep_test_mean, dep_test_std))
print("SAITS best trial", s_best_trial, "val", s_best_val)
print("CSDI ratios:", ratios["CSDI"].tolist())
print("FQGNN ratios:", ratios["FQGNN"].tolist())
print("MH-GIN ratios:", ratios["MH-GIN"].tolist())
print("MultitaskAIS ratios:", ratios["MultitaskAIS"].tolist())

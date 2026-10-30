"""ED4_build.py -- EXTENDED DATA FIGURE 4: "FQ foundation diagnostics".

The in-family / genericity / non-degeneracy evidence -- the trained operator stays a
faithful FQ throughout, and its learned comb is a genuine M >> 1 quasicrystal, not the
degenerate M = 1 collapse.

Panels
  a  in-family & genericity over training   dual-axis: min_I Δ_I (pink, LEFT) never reaching
                                            the #D1352B family boundary at 0; eig_gap (azure,
                                            RIGHT, color-bound) -- the M=1-collision monitor.
  b  amplitude & spectrum bounded           amp_l2 / amp_max (pink family, LEFT) + spec_spread
                                            (azure, RIGHT, color-bound) -- no blow-up ⇔ real-
                                            rootedness preserved.
  c  learned comb is non-degenerate         stem plot |c| vs |ξ|: comb_init pale-gray ghost vs
                                            comb_final pink -- 28 live stems, M >> 1.
  d  spectral concentration FQ vs random    2 bars top-10-stem energy fraction: FQ 89.16%
                                            (pink) vs RFF 22.64% (azure) -- pure-point rigidity.

Reads ONLY CSVs (no recomputation).  Style: fig_style.py (locked).  Output: <FQGNN_FIG_OUT>/ED4.{pdf,png}.
Run:  python paper_figures_tables/ED4_build.py     (any cwd)
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
from matplotlib.ticker import NullFormatter                     # noqa: E402

FS.apply()

TELE = os.path.join(RESULTS, "runs", "telemetry_infamily_native")

PINK = FS.C["pink"]
PINK_D = FS.SEQ_PINK[4]                                          # #C9447A darker pink for 2nd L-axis line
AZURE = FS.C["azure"]
AZURE_D = "#26456E"                                             # SEQ_BLUE[4] deep azure for legibility
GRAY6E = "#6E6E6E"
GHOST = FS.C["sep"]                                             # #BFBFBF pale ghost stems


def darken(hexcolor, f=0.30):
    h = hexcolor.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02X%02X%02X" % (int(r * (1 - f)), int(g * (1 - f)), int(b * (1 - f)))


def null():
    return NullFormatter()


# ================================================================================================ #
#  DATA                                                                                             #
# ================================================================================================ #
tele = pd.read_csv(os.path.join(TELE, "telemetry_infamily.csv"))
# epoch = -1 is the pre-training init snapshot; keep it as the x=-1 point so the reader sees the
# starting operator, then plot epochs 0..15 (the native telemetry run trains 16 epochs).
ci = pd.read_csv(os.path.join(TELE, "comb_init.csv"))           # BEFORE training (seeded comb)
cf = pd.read_csv(os.path.join(TELE, "comb_final.csv"))          # AFTER training (learned comb)
stats_df = pd.read_csv(os.path.join(FIGDATA, "spectrum_stats.csv")).set_index("model")

fq_top10 = 100.0 * float(stats_df.loc["FQGNN_trained", "top10_energy_frac"])   # 89.16
rff_top10 = 100.0 * float(stats_df.loc["RFF_seed0", "top10_energy_frac"])      # 22.64

# panel d: cumulative energy-concentration curves.  FQ comb energy = |c|^2 (48 stems, all live);
# RFF energy = readout_w^2 (96 stems).  Sort each descending by amplitude, cumulate as a % of the
# total -> a proper multi-mark line panel showing concentration (FQ ~100% by rank ~10 vs slow RFF).
fq_spec = pd.read_csv(os.path.join(FIGDATA, "fq_spectrum_trained.csv"))
rff_spec = pd.read_csv(os.path.join(FIGDATA, "rff_spectrum.csv"))
fq_e = np.sort(fq_spec["abs_c_sq"].to_numpy())[::-1]                       # |c|^2, desc
rff_e = np.sort((rff_spec["readout_w"].to_numpy() ** 2))[::-1]            # readout_w^2, desc
fq_cum = 100.0 * np.cumsum(fq_e) / fq_e.sum()                             # -> 89.16% by rank 10
rff_cum = 100.0 * np.cumsum(rff_e) / rff_e.sum()                         # -> 22.64% by rank 10
fq_rank = np.arange(1, len(fq_cum) + 1)
rff_rank = np.arange(1, len(rff_cum) + 1)

n_live = int((cf.c_abs > 0).sum())                              # 28 nonzero stems (M >> 1)
n_total = len(cf)                                               # 48
mm_lo, mm_hi = tele.min_minor.min(), tele.min_minor.max()       # 0.3045, 0.3175

# ================================================================================================ #
#  LAYOUT  (180 x 115 mm, 2 x 2 grid on the 12-col grid; rows share top edge + height)              #
# ================================================================================================ #
fig = FS.new_figure(180, 115)

R1_Y0, R1_Y1 = 0.580, 0.930                                     # row 1 axes rect (fig fraction)
R2_Y0, R2_Y1 = 0.115, 0.465                                     # row 2
COL_L, COL_R = 0.085, 0.585                                     # left / right column x0
PW_A, PW_B = 0.345, 0.335                                       # data-area widths (leave room for R axes)
PW_C, PW_D = 0.360, 0.300

ax_a = fig.add_axes([COL_L, R1_Y0, PW_A, R1_Y1 - R1_Y0])
ax_b = fig.add_axes([COL_R, R1_Y0, PW_B, R1_Y1 - R1_Y0])
ax_c = fig.add_axes([COL_L, R2_Y0, PW_C, R2_Y1 - R2_Y0])
ax_d = fig.add_axes([COL_R, R2_Y0, PW_D, R2_Y1 - R2_Y0])

LET_Y1, LET_Y2 = 0.980, 0.510
FS.panel_letter(fig, 0.012, LET_Y1, "a")
FS.panel_letter(fig, 0.512, LET_Y1, "b")
FS.panel_letter(fig, 0.012, LET_Y2, "c")
FS.panel_letter(fig, 0.512, LET_Y2, "d")

ep = tele.epoch.values                                          # -1..12

# ================================================================================================ #
#  PANEL a -- in-family & genericity over training (dual-axis, color-bound)                          #
# ================================================================================================ #
# LEFT (pink): min_I Δ_I = min_minor -- the leading-principal-minor margin.  It must NEVER reach 0
#  (would = a periodic / M=1 degenerate operator on the family boundary).  Values ~0.30-0.32.
ax_a.plot(ep, tele.min_minor, color=PINK, marker="o", markersize=3.0, markeredgewidth=0,
          lw=1.25, zorder=6)
# family boundary at 0 (red dashed threshold) -- but the axis lives up near 0.30, so we mark the
# boundary with a broken-axis cue: a red dashed rule near the bottom + an explicit "= 0" tag.
ax_a.set_ylim(0.0, 0.345)
ax_a.axhline(0.0, color=FS.C["ref_red"], ls=(0, (4, 2.5)), lw=0.8, zorder=3)
ax_a.annotate("family boundary  min$_I\\,\\Delta_I=0$  (periodic / M$\\,$=$\\,$1)",
              xy=(0.985, 0.0), xycoords=("axes fraction", "data"), fontsize=5.5,
              color=FS.C["ref_red"], ha="right", va="bottom",
              xytext=(0, 1.5), textcoords="offset points")
# shaded "in-family safe zone" band from the observed floor upward (pale, encodes the margin)
ax_a.axhspan(mm_lo, mm_hi, color=FS.SEQ_PINK[0], alpha=0.55, lw=0, zorder=1)

ax_a.set_ylabel("min$_I\\,\\Delta_I$  (family margin)", color=PINK_D)
ax_a.tick_params(axis="y", colors=PINK_D)
ax_a.spines["left"].set_color(PINK_D)
ax_a.set_yticks([0.0, 0.1, 0.2, 0.3])

# RIGHT (azure, color-bound): eig_gap -- the eigenphase separation (M=1-collision monitor).
axa2 = ax_a.twinx()
axa2.plot(ep, tele.eig_gap, color=AZURE_D, marker="^", markersize=3.0, markeredgewidth=0,
          lw=1.0, ls=(0, (4, 2.0)), zorder=5)
axa2.set_ylim(0.0, 0.085)
axa2.set_yticks([0.0, 0.02, 0.04, 0.06, 0.08])
axa2.set_ylabel("eig_gap  (eigenphase separation)", color=AZURE_D, rotation=270, labelpad=11)
axa2.tick_params(axis="y", colors=AZURE_D)
axa2.spines["right"].set_color(AZURE_D)
axa2.spines["right"].set_visible(True)
axa2.spines["top"].set_visible(False)

# direct color-matched series tags (2 series across 2 axes -> no legend box)
ax_a.annotate("min$_I\\,\\Delta_I$", xy=(8.5, tele.min_minor.iloc[-1]), fontsize=6, color=PINK_D,
              ha="left", va="bottom", xytext=(2, 3), textcoords="offset points")
axa2.annotate("eig_gap", xy=(4.0, tele.eig_gap.iloc[6]), fontsize=6, color=AZURE_D,
              ha="left", va="bottom", xytext=(2, 2), textcoords="offset points")

ax_a.set_xlim(-1.8, 15.8)
ax_a.set_xticks([0, 5, 10, 15])
ax_a.set_xlabel("Epoch")
ax_a.set_title("In-family & generic throughout training")
ax_a.annotate(f"min$_I\\,\\Delta_I \\in$ [{mm_lo:.2f}, {mm_hi:.2f}]: never collapses to 0",
              xy=(0.03, 0.60), xycoords="axes fraction", fontsize=5.5, color=darken(PINK, 0.25),
              ha="left", va="bottom")
ax_a.set_zorder(axa2.get_zorder() + 1)                          # pink line/labels above azure axis
ax_a.patch.set_visible(False)

# ================================================================================================ #
#  PANEL b -- amplitude & spectrum bounded (dual-axis, color-bound)                                  #
# ================================================================================================ #
# LEFT (pink family): amp_l2 (|c|_2) + amp_max (max|c|) -- the amplitude blow-up monitors.
ax_b.plot(ep, tele.amp_l2, color=PINK, marker="o", markersize=3.0, markeredgewidth=0,
          lw=1.25, zorder=6)
ax_b.plot(ep, tele.amp_max, color=PINK_D, marker="D", markersize=2.6, markeredgewidth=0,
          lw=1.0, ls=(0, (1, 1)), zorder=5)
ax_b.set_ylim(0.0, 5.4)
ax_b.set_yticks([0, 1, 2, 3, 4, 5])
ax_b.set_ylabel("amplitude  $\\|c\\|_2$ , max$|c|$", color=PINK_D)
ax_b.tick_params(axis="y", colors=PINK_D)
ax_b.spines["left"].set_color(PINK_D)

ax_b.annotate("$\\|c\\|_2$", xy=(9.0, tele.amp_l2.iloc[-1]), fontsize=6, color=PINK_D,
              ha="left", va="bottom", xytext=(2, 2), textcoords="offset points")
ax_b.annotate("max$|c|$", xy=(9.0, tele.amp_max.iloc[-1]), fontsize=6, color=darken(PINK_D, 0.15),
              ha="left", va="top", xytext=(2, -2), textcoords="offset points")

# RIGHT (azure, color-bound): spec_spread = max|ξ| / min|ξ| (spectrum span, essentially fixed).
axb2 = ax_b.twinx()
axb2.plot(ep, tele.spec_spread, color=AZURE_D, marker="^", markersize=3.0, markeredgewidth=0,
          lw=1.0, ls=(0, (4, 2.0)), zorder=4)
sp0 = tele.spec_spread.iloc[0]
axb2.set_ylim(sp0 - 0.9, sp0 + 0.9)                             # zoom so the flat line is visible
axb2.set_yticks([sp0 - 0.5, sp0, sp0 + 0.5],
                [f"{sp0 - 0.5:.1f}", f"{sp0:.1f}", f"{sp0 + 0.5:.1f}"])
axb2.set_ylabel("spec_spread  max$|\\xi|/$min$|\\xi|$", color=AZURE_D, rotation=270, labelpad=11)
axb2.tick_params(axis="y", colors=AZURE_D)
axb2.spines["right"].set_color(AZURE_D)
axb2.spines["right"].set_visible(True)
axb2.spines["top"].set_visible(False)
axb2.annotate("spec_spread (fixed)", xy=(0.5, sp0), xycoords=("axes fraction", "data"),
              fontsize=5.5, color=AZURE_D, ha="center", va="bottom",
              xytext=(0, 2), textcoords="offset points")

ax_b.set_xlim(-1.8, 15.8)
ax_b.set_xticks([0, 5, 10, 15])
ax_b.set_xlabel("Epoch")
ax_b.set_title("Amplitude & spectrum bounded (no blow-up)")
ax_b.annotate("bounded amplitudes  $\\Leftrightarrow$  real-rootedness preserved",
              xy=(0.03, 0.045), xycoords="axes fraction", fontsize=5.5,
              color=darken(PINK, 0.25), ha="left", va="bottom")
ax_b.set_zorder(axb2.get_zorder() + 1)
ax_b.patch.set_visible(False)

# ================================================================================================ #
#  PANEL c -- learned comb is non-degenerate (the key spectral exhibit)                              #
# ================================================================================================ #
# ghost = seeded init comb (pale gray); pink = learned final comb.  Sort by |xi| so the comb reads
# left-to-right as a frequency axis.  Both have 28 nonzero stems, but the learned comb redistributes
# energy across the tail -- decisively M >> 1, not the M=1 collapse.
ci_s = ci.sort_values("xi_abs")
cf_s = cf.sort_values("xi_abs")
# a faint offset so overlapping big stems don't perfectly hide the ghost behind the pink
DX = 0.028
ax_c.vlines(ci_s.xi_abs - DX, 0, ci_s.c_abs, color=GHOST, lw=1.3, zorder=2)
ax_c.vlines(cf_s.xi_abs + DX, 0, cf_s.c_abs, color=PINK, lw=1.3, zorder=4)
# mark the live-stem tips (final) with small dots so the low-amplitude tail stems are visible
live = cf_s[cf_s.c_abs > 0]
ax_c.plot(live.xi_abs + DX, live.c_abs, ls="none", marker="o", markersize=1.9,
          markerfacecolor=PINK, markeredgewidth=0, zorder=5)
# tiny leader ring on the largest tail stem cluster so the reader sees the comb continues past 6
tail = live[live.c_abs < 0.15]
ax_c.plot(tail.xi_abs + DX, tail.c_abs, ls="none", marker="o", markersize=1.9,
          markerfacecolor=darken(PINK, 0.10), markeredgewidth=0, zorder=6)

ax_c.set_xlim(0.15, 5.05)
ax_c.set_ylim(0.0, 2.78)
ax_c.set_xticks([1, 2, 3, 4, 5])
ax_c.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
ax_c.set_xlabel("Frequency  $|\\xi|$", labelpad=2)
ax_c.set_ylabel("|amplitude|  $|c|$")
ax_c.set_title("Learned comb is non-degenerate (M $\\gg$ 1)")

# direct color-matched labels (2 combs -> no legend box), parked in the empty upper-right
ax_c.annotate("learned (trained)", xy=(0.635, 0.93), xycoords="axes fraction", fontsize=6,
              color=darken(PINK, 0.10), ha="left", va="center", fontweight="bold")
ax_c.annotate("init (seeded comb)", xy=(0.635, 0.83), xycoords="axes fraction", fontsize=6,
              color=darken(GHOST, 0.28), ha="left", va="center")
# short color swatches to the left of each label
ax_c.plot([0.585, 0.615], [0.93, 0.93], transform=ax_c.transAxes, color=PINK, lw=1.6,
          clip_on=False)
ax_c.plot([0.585, 0.615], [0.83, 0.83], transform=ax_c.transAxes, color=GHOST, lw=1.6,
          clip_on=False)
ax_c.annotate(f"{n_live}/{n_total} live stems  (M $\\gg$ 1, not periodic)\n"
              f"top-10 = {fq_top10:.2f}% of energy",
              xy=(0.585, 0.71), xycoords="axes fraction", fontsize=5.5,
              color=darken(PINK, 0.25), ha="left", va="top", linespacing=1.3)
FS.condition_strip(ax_c, "seed-0 | comb before vs after training", dy=-0.225)

# ================================================================================================ #
#  PANEL d -- spectral energy concentration, cumulative (line plot, >=20 marks)                      #
# ================================================================================================ #
# Top-10 spectral energy (FQ 89.16% vs RFF 22.64%) shown as a
# multi-mark concentration curve: cumulative fraction of total spectral energy vs stem rank.  FQ
# (pink solid) saturates to ~100% within ~10 stems; RFF (azure dashed) climbs slowly toward 100%.
# 100% dashed guide = the pure-point ceiling.
ax_d.axhline(100.0, color=FS.C["ref_gray"], ls=(0, (4, 2.5)), lw=0.7, zorder=2)
ax_d.annotate("100% (pure-point limit)", xy=(0.985, 100.0),
              xycoords=("axes fraction", "data"), fontsize=5.5, color=GRAY6E,
              ha="right", va="bottom", xytext=(0, 1.5), textcoords="offset points")

# RFF first (drawn below FQ): azure dashed with open triangles
ax_d.plot(rff_rank, rff_cum, color=AZURE, lw=1.0, ls=(0, (4, 2.0)), zorder=4)
ax_d.plot(rff_rank, rff_cum, ls="none", marker="^", markersize=2.4, markerfacecolor="none",
          markeredgecolor=AZURE, markeredgewidth=0.7, zorder=5)
# FQ on top: pink solid with filled dots
ax_d.plot(fq_rank, fq_cum, color=PINK, lw=1.25, zorder=6)
ax_d.plot(fq_rank, fq_cum, ls="none", marker="o", markersize=2.4, markerfacecolor=PINK,
          markeredgewidth=0, zorder=7)

# vertical rule at rank 10 -- the "top-10" locus the two end-labels quantify
ax_d.axvline(10, color=FS.C["sep"], lw=0.6, ls=(0, (1, 1.6)), zorder=1)
ax_d.annotate("rank 10", xy=(10, 2), xycoords=("data", "data"), fontsize=5.0, color=GRAY6E,
              ha="center", va="bottom", xytext=(0, 0), textcoords="offset points",
              rotation=90)

# direct labels (no legend box): each series' TOP-10 energy fraction, quantified AT rank 10.
# FQ saturates at rank ~6 -> label parked in the open plateau interior, below the 100% guide,
# with a short leader down to the pink curve at rank 10; RFF is only 22.64% at rank 10 -> label
# lifted up-left into the empty space above the slow azure curve, leader down to its rank-10 point.
ax_d.annotate(f"FQ comb: {fq_top10:.2f}%", xy=(10, fq_cum[9]), xytext=(24, 88),
              textcoords="data", fontsize=6, color=darken(PINK, 0.15), ha="left", va="center",
              fontweight="bold",
              arrowprops=dict(arrowstyle="-", color=darken(PINK, 0.15), lw=0.6,
                              shrinkA=1.5, shrinkB=1.5), zorder=8)
ax_d.annotate(f"RFF basis: {rff_top10:.2f}%", xy=(10, rff_cum[9]), xytext=(17, 46),
              textcoords="data", fontsize=6, color=darken(AZURE, 0.28), ha="left", va="center",
              arrowprops=dict(arrowstyle="-", color=darken(AZURE, 0.28), lw=0.6,
                              shrinkA=1.5, shrinkB=1.5), zorder=8)

ax_d.set_xlim(0, len(rff_cum) + 1)
ax_d.set_ylim(0, 112)
ax_d.set_xticks([1, 10, 25, 50, 75, 96])
ax_d.set_yticks([0, 25, 50, 75, 100])
ax_d.set_xlabel("Stem rank (sorted by amplitude)")
ax_d.set_ylabel("Cumulative energy (%)")
ax_d.set_title("Spectral energy concentration (cumulative)")
FS.condition_strip(ax_d, "seed-0 trained checkpoints | 48 FQ vs 96 RFF stems", dy=-0.275)

# ================================================================================================ #
FS.save(fig, os.path.join(OUT, "ED4"))

# ---- console diagnostics ----
print("\n[a] min_minor (min_I dI): min={:.5f} max={:.5f}  (family boundary = 0; never reached)"
      .format(mm_lo, mm_hi))
print("[a] eig_gap: min={:.5f} max={:.5f}".format(tele.eig_gap.min(), tele.eig_gap.max()))
print("[b] amp_l2 [{:.3f},{:.3f}]  amp_max [{:.3f},{:.3f}]  spec_spread [{:.4f},{:.4f}]".format(
    tele.amp_l2.min(), tele.amp_l2.max(), tele.amp_max.min(), tele.amp_max.max(),
    tele.spec_spread.min(), tele.spec_spread.max()))
print("[c] live stems (final): {}/{}  (init live: {})".format(
    n_live, n_total, int((ci.c_abs > 0).sum())))
print("[d] top-10 energy fraction: FQ {:.2f}%  vs  RFF {:.2f}%".format(fq_top10, rff_top10))

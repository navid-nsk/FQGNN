"""EXTENDED DATA FIGURE 1 -- 'Protocol & leak audit' (schematic in the Fig-1 grammar).

The reproducibility figure: raw Arctic AIS -> per-vessel runs -> vessel-disjoint md5 split
(5350 train / 635 val / 774 test) -> contiguous-gap construction -> leak-safe zeroed inputs
-> shared-gaps oracle reproducing the FQ-GNN GCD column exactly.

Panel a = the 5-stage L->R typed-box pipeline (each container has a REAL data thumbnail).
Panel b = the oracle bucket-GCD bar (teal bars), so ED1 carries a quantitative panel too.

Style: fig_style.py ONLY (locked design system). All thumbnails read REAL CSVs from
FQGNN_FIGDATA.  Reuses Fig-1's container / block_arrow / thumbnail idioms verbatim.
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
from matplotlib.patches import FancyBboxPatch, Polygon, Circle  # noqa: E402

FS.apply()
C = FS.C

W_MM, H_MM = 180.0, 88.0

# ------------------------------------------------------------------ data (ALL real)
FD = FIGDATA
MT = pd.read_csv(os.path.join(FD, "map_tracks.csv"))
HG = pd.read_csv(os.path.join(FD, "hero_gap_0.05-0.10_245.csv"))
# hero CSVs carry the FULL track; crop to held window +- 12 anchors
_hw = np.where(HG.is_held.to_numpy() == 1)[0]
HG = HG.iloc[max(0, _hw[0] - 12):_hw[-1] + 13].reset_index(drop=True)
PV = pd.read_csv(os.path.join(FD, "pivot_ratio_mean.csv"))

# split counts (verified): vessel-disjoint md5(mmsi) 80/10/10
N_TRAIN, N_VAL, N_TEST = 5350, 635, 774

# oracle bucket-GCD table (median GCD NM per bucket) -- pivot gcd_med column
PVB = PV[PV.bucket != "OVERALL"].copy()
OVERALL_GCD = float(PV[PV.bucket == "OVERALL"].gcd_med.iloc[0])   # 2.857

fig = FS.new_figure(W_MM, H_MM)

# full-canvas overlay in mm coordinates: containers, arrows, free text
axB = fig.add_axes([0, 0, 1, 1], zorder=0)
axB.set_xlim(0, W_MM)
axB.set_ylim(0, H_MM)
axB.axis("off")


def ax_mm(x, y, w, h, **kw):
    return fig.add_axes([x / W_MM, y / H_MM, w / W_MM, h / H_MM], **kw)


def txt(x, y, s, size=6, color=None, ha="center", va="center", style="normal",
        weight="normal", rotation=0, zorder=5):
    axB.text(x, y, s, fontsize=size, color=color or C["text"], ha=ha, va=va,
             fontstyle=style, fontweight=weight, rotation=rotation, zorder=zorder)


def container(x0, y0, x1, y1, token, tab):
    """Group container in the house idiom: panel tint + the same hue saturated
    on the border, generous rounding, italic tab inside the top-left."""
    FS.panel(axB, x0, y0, x1, y1, token, lw=0.9, r=2.4, z=1)
    axB.text(x0 + 2.2, y1 - 2.6, tab, fontsize=6, fontstyle="italic",
             color=FS.SCHEM_PANEL[token][1], ha="left", va="center", zorder=4)


def block_arrow(x0, x1, y, shaft=1.6, head_l=2.2, head_h=3.4):
    """Thick light-gray block arrow = forward data (typed connector 1)."""
    s, hh = shaft / 2, head_h / 2
    axB.add_patch(Polygon(
        [(x0, y - s), (x1 - head_l, y - s), (x1 - head_l, y - hh), (x1, y),
         (x1 - head_l, y + hh), (x1 - head_l, y + s), (x0, y + s)],
        closed=True, facecolor="#9DAEC2", edgecolor="none", zorder=3))


def thin_arrow(pts, color=None, lw=0.7, ls="-", head=1.1, zorder=4):
    """Black rounded-corner Manhattan connector via the locked style system."""
    FS.elbow(axB, pts, color=color or C["axis"], lw=lw, ls=ls, head=head, z=zorder, r=1.0)


def hairline_frame(ax, lw=0.8):
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(lw)
        sp.set_color("#5B6B7C")
    ax.set_xticks([])
    ax.set_yticks([])


# ================================================================== PANEL a
FS.panel_letter(fig, 1.5 / W_MM, 87.6 / H_MM, "a")

# five stage containers across the width; reading axis strictly L->R
CB, CT = 30.0, 83.5                       # container band bottom / top
# tints must encode a consistent grouping. These 5 are ONE pipeline (no sub-grouping),
# so all containers use the SAME 'peri' panel tint -- uniform, not per-stage color-coded.
GAPS = [
    (2.0, 34.0, "peri", "Raw AIS $\\to$ runs"),
    (37.0, 69.0, "peri", "Vessel-disjoint split"),
    (72.0, 106.0, "peri", "Contiguous-gap construction"),
    (109.0, 143.0, "peri", "Leak-safe inputs"),
    (146.0, 178.0, "peri", "Shared-gaps oracle"),
]
for x0, x1, fill, tab in GAPS:
    container(x0, CB, x1, CT, fill, tab)

# forward-data block arrows between stages (y at container mid-height)
YMID = (CB + CT) / 2
for (x0a, x1a, *_), (x0b, *_rest) in zip(GAPS[:-1], GAPS[1:]):
    block_arrow(x1a + 0.4, x0b - 0.4, YMID)

# ---------------------------------------------------------------- STAGE 1: Raw AIS -> runs
g1c = (GAPS[0][0] + GAPS[0][1]) / 2       # 18
# one REAL track polyline (gap 25: sinuous, informative), with the held run-segment overlaid
trk = MT[MT.gap_id == 25].sort_values("point_idx")
tlat = trk.lat.to_numpy()
tlon = trk.lon.to_numpy()
tgap = trk.is_gap_segment.to_numpy().astype(bool)
axT1 = ax_mm(g1c - 12.0, 44.0, 24.0, 24.0)
lat0 = tlat.mean()
kx = 60.0 * np.cos(np.deg2rad(lat0))
xx = (tlon - tlon.mean()) * kx
yy = (tlat - lat0) * 60.0
axT1.plot(xx, yy, color=C["navy"], lw=0.7, zorder=2)
# the held run-segment (a >1 h gap window) drawn as open gray circles on this run
axT1.plot(xx[tgap], yy[tgap], ls="none", marker="o", ms=1.9, mfc="none",
          mec=C["ref_gray"], mew=0.6, zorder=3)
# anchor endpoints of the run
axT1.plot([xx[0], xx[-1]], [yy[0], yy[-1]], ls="none", marker="o", ms=2.2,
          mfc=C["blue"], mec="none", zorder=4)
axT1.set_aspect("equal")
axT1.axis("off")
axT1.margins(0.08)
txt(g1c, 42.6, "one vessel track (real)", size=5.5, color="#6E6E6E")
txt(g1c, 39.4, "per-MMSI runs, split at", size=6)
txt(g1c, 36.7, "gaps $> 1$ h", size=6)
txt(g1c, 33.6, "lat $\\geq 60\\degree$, Jul-Aug 2016-18", size=5.5, color="#6E6E6E")

# ---------------------------------------------------------------- STAGE 2: vessel-disjoint split
g2c = (GAPS[1][0] + GAPS[1][1]) / 2       # 53
axS2 = ax_mm(g2c - 12.0, 48.5, 24.0, 17.5)
splits = [("train", N_TRAIN, C["navy"]), ("val", N_VAL, C["azure"]),
          ("test", N_TEST, C["paleblue"])]
bx = np.arange(3)
bvals = np.array([s[1] for s in splits])
bcols = [s[2] for s in splits]
axS2.bar(bx, bvals, width=0.68, color=bcols, edgecolor="none", zorder=3)
for i, (lab, v, col) in enumerate(splits):
    axS2.text(i, v + 130, f"{v}", ha="center", va="bottom", fontsize=5.5,
              color=C["text"])
    axS2.text(i, -230, lab, ha="center", va="top", fontsize=5.5, color=C["text"])
axS2.set_ylim(0, 6300)
axS2.set_xlim(-0.62, 2.62)
axS2.axis("off")
txt(g2c, 44.4, "md5(mmsi) $\\to$ 80/10/10", size=6)
txt(g2c, 40.9, "0 vessel overlap", size=6, color=C["good"])
# green check node next to the 'verified' claim
axB.add_patch(Circle((g2c + 10.6, 40.9), 1.35, facecolor="none",
                     edgecolor=C["good"], lw=0.9, zorder=5))
axB.plot([g2c + 9.9, g2c + 10.4, g2c + 11.3], [40.9, 40.2, 41.6],
         color=C["good"], lw=0.9, zorder=6, solid_capstyle="round")
txt(g2c, 37.8, "(verified)", size=5.5, color="#6E6E6E")

# ---------------------------------------------------------------- STAGE 3: contiguous-gap
g3c = (GAPS[2][0] + GAPS[2][1]) / 2       # 89
# REAL gap timeline: anchors (blue filled) both sides, held window (open gray) between
held = HG.is_held.to_numpy().astype(bool)
tsec = HG.t_sec.to_numpy()
tn = (tsec - tsec.min()) / (tsec.max() - tsec.min())      # normalized time in [0,1]
axT3 = ax_mm(g3c - 15.0, 51.0, 30.0, 12.0)
y0 = 0.5
axT3.plot([0, 1], [y0, y0], color=C["sep"], lw=0.5, zorder=1)
axT3.plot(tn[~held], np.full((~held).sum(), y0), ls="none", marker="o", ms=2.6,
          mfc=C["blue"], mec="none", zorder=3)
axT3.plot(tn[held], np.full(held.sum(), y0), ls="none", marker="o", ms=2.6,
          mfc="none", mec=C["ref_gray"], mew=0.6, zorder=2)
# two-sided anchor edges: brackets marking where the held window starts / ends
hi = np.where(held)[0]
ha, hb = tn[hi[0] - 1], tn[hi[-1] + 1]
for xe in (ha, hb):
    axT3.plot([xe, xe], [y0 - 0.28, y0 + 0.28], color=C["axis"], lw=0.6, zorder=4)
axT3.annotate("", xy=(hb, y0 + 0.5), xytext=(ha, y0 + 0.5),
              arrowprops=dict(arrowstyle="<->", color=C["axis"], lw=0.6))
axT3.text((ha + hb) / 2, y0 + 0.62, "held window", ha="center", va="bottom",
          fontsize=5.5, color=C["text"])
axT3.set_xlim(-0.04, 1.04)
axT3.set_ylim(-0.15, 1.35)
axT3.axis("off")
txt(g3c, 49.0, "held = contiguous time window", size=6)
txt(g3c, 46.2, "anchors on both sides", size=6)
# mini-key for this stage's glyphs (in the container, clear of the timeline)
axB.plot([g3c - 14.0], [43.6], ls="none", marker="o", ms=2.4, mfc=C["blue"],
         mec="none", zorder=5)
txt(g3c - 12.6, 43.6, "observed anchor", size=5.5, ha="left")
axB.plot([g3c - 14.0], [40.8], ls="none", marker="o", ms=2.4, mfc="none",
         mec=C["ref_gray"], mew=0.6, zorder=5)
txt(g3c - 12.6, 40.8, "held node", size=5.5, ha="left")

# ---------------------------------------------------------------- STAGE 4: leak-safe inputs
g4c = (GAPS[3][0] + GAPS[3][1]) / 2       # 126
# two feature-row glyphs (input channels): observed [pos,sog,cog,0] vs held [0,0,0,1]
CHAN = ["pos", "sog", "cog", "held"]      # 4th channel = is_held flag
FX0 = g4c - 8.0                           # cells start (leaves room for row title at left)
CW = 4.6
# column header naming the four input channels
for i, ch in enumerate(CHAN):
    txt(FX0 + i * CW + (CW - 0.7) / 2, 66.4, ch, size=5.0, color="#6E6E6E")


def feat_row(cy, cells, colors, title):
    txt(FX0 - 1.4, cy, title, size=5.5, ha="right", color=C["text"])
    for i, (val, col) in enumerate(zip(cells, colors)):
        cx = FX0 + i * CW
        axB.add_patch(FancyBboxPatch(
            (cx, cy - 1.8), CW - 0.7, 3.6, boxstyle="round,pad=0,rounding_size=0.5",
            facecolor=col, edgecolor=FS.SCHEM["slate"][1], lw=0.7, zorder=3))
        txt(cx + (CW - 0.7) / 2, cy, val, size=5, color=FS.contrast_text_color(col),
            zorder=4)


OBS = C["azure"]
ZER = "#F2F2F2"
feat_row(63.0, ["pos", "sog", "cog", "0"], [OBS, OBS, OBS, ZER], "observed")
# is_held flag cell = GOLD (distinct-category token), NOT pink: pink is the paper-wide
# 'our method' (FQ-GNN) lock, reserved exclusively for the model.
feat_row(56.4, ["0", "0", "0", "1"], [ZER, ZER, ZER, C["gold"]], "held")
# down-arrow between the two rows: the zeroing transform
thin_arrow([(FX0 - 6.2, 63.0), (FX0 - 6.2, 56.4)], "#6E6E6E", lw=0.55, head=0.9)
txt(g4c, 51.4, "held channels zeroed", size=6)
txt(g4c, 48.6, "$\\to$ leak audit $= 0.0$", size=6, color=C["good"])
# green check for the leak audit
axB.add_patch(Circle((g4c + 12.6, 48.6), 1.35, facecolor="none",
                     edgecolor=C["good"], lw=0.9, zorder=5))
axB.plot([g4c + 11.9, g4c + 12.4, g4c + 13.3], [48.6, 47.9, 49.3],
         color=C["good"], lw=0.9, zorder=6, solid_capstyle="round")
txt(g4c, 45.8, "(verified)", size=5.5, color="#6E6E6E")

# ---------------------------------------------------------------- STAGE 5: shared-gaps oracle
g5c = (GAPS[4][0] + GAPS[4][1]) / 2       # 162
# labeled strip = the reproduced GCD column: OVERALL + 5 sinuosity buckets, value per cell.
# bucket labels kept short (all, b1-b5) so they stay >=5 pt; OVERALL cell tinted.
strip_y = 65.0                             # value-cell centre
strip_x0 = g5c - 14.4
cellw = 4.7
BLAB = ["all", "b1", "b2", "b3", "b4", "b5"]
oracle_vals = [OVERALL_GCD] + [float(v) for v in PVB.gcd_med]
txt(g5c, strip_y + 4.2, "reproduced GCD column (median NM)", size=5.3, color="#6E6E6E")
for i, (lab, v) in enumerate(zip(BLAB, oracle_vals)):
    cx = strip_x0 + i * cellw
    hl = (i == 0)                          # highlight the OVERALL cell
    axB.add_patch(FancyBboxPatch(
        (cx, strip_y - 2.0), cellw - 0.5, 4.0, boxstyle="round,pad=0,rounding_size=0.4",
        facecolor=FS.SCHEM["sky"][0] if hl else "white",
        edgecolor=FS.SCHEM["slate"][1], lw=0.7, zorder=3))
    txt(cx + (cellw - 0.5) / 2, strip_y, f"{v:.2f}", size=5.0, zorder=4)
    txt(cx + (cellw - 0.5) / 2, strip_y - 3.6, lab, size=5.0, color="#6E6E6E",
        zorder=4)
txt(g5c, strip_y - 6.2, "b1-b5 = sinuosity buckets (panel b)", size=5.0,
    color="#6E6E6E")
# green check node
axB.add_patch(Circle((g5c, 52.6), 1.7, facecolor="none", edgecolor=C["good"],
                     lw=1.0, zorder=5))
axB.plot([g5c - 0.85, g5c - 0.2, g5c + 1.05], [52.6, 51.75, 53.5],
         color=C["good"], lw=1.1, zorder=6, solid_capstyle="round")
txt(g5c, 49.0, "common_gaps reproduces the", size=5.7)
txt(g5c, 46.4, "FQ-GNN GCD column exactly", size=5.7)
txt(g5c, 43.4, "every model scored on", size=5.5, color="#6E6E6E")
txt(g5c, 41.2, "identical gaps", size=5.5, color="#6E6E6E")

# ---------------------------------------------------------------- in-figure connector key (no box)
KX, KTX = 128.0, 136.0
KY = 20.0
block_arrow(KX, KX + 6.5, KY, shaft=1.2, head_l=1.7, head_h=2.4)
txt(KTX, KY, "forward data (pipeline stage $\\to$ stage)", size=5.5, ha="left")

# ================================================================== PANEL b (oracle bar)
FS.panel_letter(fig, 1.5 / W_MM, 27.2 / H_MM, "b")

axO = ax_mm(14.0, 11.0, 100.0, 14.0)
labels = list(PVB.bucket)
vals = PVB.gcd_med.to_numpy()
ncts = PVB.n.to_numpy().astype(int)
xpos = np.arange(len(labels))
bar_cols = FS.SEQ_TEAL
axO.bar(xpos, vals, width=0.66, color=bar_cols, edgecolor="none", zorder=3)
# OVERALL as a dashed reference rule (it is the aggregate, not a 6th category)
axO.axhline(OVERALL_GCD, color=C["gcd"], ls=(0, (4, 2.5)), lw=0.8, zorder=2)
axO.annotate(f"OVERALL {OVERALL_GCD:.2f} NM", xy=(0.015, OVERALL_GCD),
             xycoords=("axes fraction", "data"), fontsize=5.3, color=C["gcd"],
             ha="left", va="bottom", xytext=(0, 1.6), textcoords="offset points")
axO.set_xticks(xpos)
axO.set_xticklabels(labels, fontsize=5.3)
axO.set_ylim(0, 7.9)
axO.set_yticks([0, 2, 4, 6])
axO.set_ylabel("GCD median NM", fontsize=6)
axO.set_xlabel("sinuosity bucket", fontsize=6.5, labelpad=1.5)
axO.set_title("Shared-gaps oracle: GCD error per bucket", fontsize=7, loc="left", pad=3)
for sp in ("top", "right"):
    axO.spines[sp].set_visible(False)
# per-bucket n printed above bars (in-data annotation; cannot be read off an axis)
for x, v, nc in zip(xpos, vals, ncts):
    axO.annotate(f"n={nc}", xy=(x, v), xytext=(0, 2), textcoords="offset points",
                 ha="center", va="bottom", fontsize=5.0, color="#6E6E6E")
FS.condition_strip(
    axO, "n = 770 test gaps | source: common_gaps oracle (median GCD NM)", dy=-0.62)

# ================================================================== save
FS.save(fig, os.path.join(OUT, "ED1"))

"""fig5_build.py -- FIGURE 5: qualitative reconstructions (triptych layout).

a  3x3 track grid: rows = 3 scripted hero gaps (straight / moderate / sharp), columns =
   'Truth + anchors' | 'FQ-GNN (ours)' | 'RFF-GNN'.  Prediction cells carry the co-registered
   error as a segment-coloured track (shared sequential pink ramp, ONE labelled colorbar,
   limits 0 -> robust 95th-pct max over all held errors of the 3 gaps x 2 models).
b  error-vs-time-in-gap profiles, 3-across, shared log y; GCD dashed gray, FQ pink solid o,
   RFF azure dashed ^; one shared legend strip.

Reads ONLY FQGNN_FIGDATA/hero_gap_*.csv + hero_gaps_index.csv (no recomputation).
Run:  python paper_figures_tables/fig5_build.py
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
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

FS.apply()
C = FS.C

# ------------------------------------------------------------------------------------------------ #
#  Canvas + mm helpers                                                                              #
# ------------------------------------------------------------------------------------------------ #
W_MM, H_MM = 180.0, 120.0
fig = FS.new_figure(W_MM, H_MM)


def rect(x, y, w, h):
    return [x / W_MM, y / H_MM, w / W_MM, h / H_MM]


def fx(x):
    return x / W_MM


def fy(y):
    return y / H_MM


# ------------------------------------------------------------------------------------------------ #
#  Data                                                                                             #
# ------------------------------------------------------------------------------------------------ #
idx = pd.read_csv(os.path.join(FIGDATA, "hero_gaps_index.csv"))
BUCKET_NAME = {"0.00-0.02": "Straight", "0.05-0.10": "Moderate", "gt0.25": "Sharp"}
ROWS = []
for _, r in idx.iterrows():
    lab = BUCKET_NAME[r["bucket"]]
    s = float(r["sinuosity"])
    s_txt = f"{s:.4f}" if s < 0.001 else f"{s:.3f}"
    _df = pd.read_csv(os.path.join(FIGDATA, f"hero_gap_{r['bucket']}_{int(r['gap_id'])}.csv"))
    # crop to the held window +- 20 anchors by INDEX (same convention as fig2 panel b): the
    # hero CSVs carry the FULL track, and drawing it re-plots every other traversal of
    # the same waters inside the axes bbox, burying the manoeuvre in unrelated track segments
    _hw = np.where(_df.is_held.to_numpy() == 1)[0]
    _df = _df.iloc[max(0, _hw[0] - 20):_hw[-1] + 21].reset_index(drop=True)
    ROWS.append(dict(
        name=lab, s=s, label=f"{lab} (s = {s_txt})",
        gap_h=float(r["gap_min"]) / 60.0,
        fq_med=float(r["fq_med"]), rff_med=float(r["rff_med"]),
        df=_df,
    ))


def fmt_med(v):
    """2 significant digits (values >= 10 keep one decimal: 13.161 -> 13.2)."""
    return f"{v:.1f}" if v >= 10 else f"{v:.2g}"

# shared robust colour max: 95th pct across all held errors of the 3 gaps x 2 models (FQ, RFF)
allerr = np.concatenate([np.r_[d["df"].loc[d["df"].is_held == 1, "err_nm_fq"].values,
                               d["df"].loc[d["df"].is_held == 1, "err_nm_rff"].values] for d in ROWS])
V95 = float(np.percentile(allerr, 95))
VMAX = float(round(V95))                    # 9.25 -> 9 (robust shared max)
NORM = Normalize(0.0, VMAX)
# sequential pink error ramp; drop the near-white step so low-error segments stay visible on white
CMAP = LinearSegmentedColormap.from_list("err_pink", FS.SEQ_PINK[1:])

# ------------------------------------------------------------------------------------------------ #
#  Geometry helpers                                                                                 #
# ------------------------------------------------------------------------------------------------ #
CELL_W, CELL_H = 40.0, 22.0                 # mm; equal-aspect cells, limits sized to this box
CTX = 12                                    # context nodes shown either side of the held window


def local_xy(lat, lon, lat0, lon0):
    """Row-local coords in NM: x = (lon-lon0) cos(lat0) * 60, y = (lat-lat0) * 60."""
    x = (np.asarray(lon) - lon0) * np.cos(np.radians(lat0)) * 60.0
    y = (np.asarray(lat) - lat0) * 60.0
    return x, y


def prep_row(d):
    """Per-row plotting bundle: local coords, crop limits matched to the cell aspect."""
    df = d["df"]
    held = df.index[df.is_held == 1].to_numpy()
    h0, h1 = held.min(), held.max()
    lat0 = df.lat_true.iloc[h0:h1 + 1].mean()
    lon0 = df.lon_true.iloc[h0:h1 + 1].mean()

    P = dict(held=held, h0=h0, h1=h1)
    P["xt"], P["yt"] = local_xy(df.lat_true, df.lon_true, lat0, lon0)          # truth, all nodes
    P["xg"], P["yg"] = local_xy(df.lat_gcd, df.lon_gcd, lat0, lon0)            # GCD chord
    P["xf"], P["yf"] = local_xy(df.lat_fq, df.lon_fq, lat0, lon0)              # FQ pred
    P["xr"], P["yr"] = local_xy(df.lat_rff, df.lon_rff, lat0, lon0)            # RFF pred
    P["ef"] = df.err_nm_fq.to_numpy()
    P["er"] = df.err_nm_rff.to_numpy()
    P["eg"] = df.err_nm_gcd.to_numpy()
    P["obs"] = df.index[df.is_held == 0].to_numpy()
    P["t"] = df.t_sec.to_numpy()

    # data limits: cropped truth + chord + both predictions over the held window
    c0, c1 = max(0, h0 - CTX), min(len(df) - 1, h1 + CTX)
    xs = np.r_[P["xt"][c0:c1 + 1], P["xg"][h0:h1 + 1], P["xf"][held], P["xr"][held]]
    ys = np.r_[P["yt"][c0:c1 + 1], P["yg"][h0:h1 + 1], P["yf"][held], P["yr"][held]]
    xm, xM, ym, yM = xs.min(), xs.max(), ys.min(), ys.max()
    pad = 0.06
    dx, dy = (xM - xm) * (1 + 2 * pad), (yM - ym) * (1 + 2 * pad)
    cxm, cym = (xm + xM) / 2, (ym + yM) / 2
    A = CELL_W / CELL_H                     # box aspect; expand the deficient axis (equal aspect)
    if dx / dy < A:
        dx = dy * A
    else:
        dy = dx / A
    P["xlim"] = (cxm - dx / 2, cxm + dx / 2)
    P["ylim"] = (cym - dy / 2, cym + dy / 2)
    return P


def nice_scalebar(xr):
    """Largest 1-2-5 x 10^k value <= 28% of the visible x-range."""
    target = 0.28 * xr
    best = 1.0
    for k in (-1, 0, 1, 2):
        for m in (1, 2, 5):
            v = m * 10 ** k
            if v <= target:
                best = v
    return best


def image_cell(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(C["sep"]); sp.set_linewidth(0.5)


def draw_base(ax, P, truth_color, truth_lw, chord_alpha, anchors=True, held_circles=False):
    """Common underlay: truth track, GCD chord, anchor dots (+ held circles in the truth cell)."""
    ax.plot(P["xt"], P["yt"], color=truth_color, lw=truth_lw, solid_capstyle="round", zorder=2)
    a, b = max(0, P["h0"] - 1), min(len(P["xt"]) - 1, P["h1"] + 1)
    ax.plot(P["xg"][a:b + 1], P["yg"][a:b + 1], color=C["gcd"], lw=0.7, ls=(0, (4, 2.5)),
            alpha=chord_alpha, zorder=3)
    if anchors:
        ax.plot(P["xt"][P["obs"]], P["yt"][P["obs"]], ls="none", marker="o", ms=2.1,
                mfc=C["paleblue"], mec=C["azure"], mew=0.35, zorder=6)
    if held_circles:
        ax.plot(P["xt"][P["held"]], P["yt"][P["held"]], ls="none", marker="o", ms=2.1,
                mfc="none", mec=C["ref_gray"], mew=0.45, zorder=5)


def draw_pred_track(ax, P, xp, yp, err):
    """Held-window prediction as an error-coloured LineCollection, bridged to the two anchors."""
    hd = P["held"]
    xs = np.r_[P["xt"][P["h0"] - 1], xp[hd], P["xt"][P["h1"] + 1]]
    ys = np.r_[P["yt"][P["h0"] - 1], yp[hd], P["yt"][P["h1"] + 1]]
    ev = np.r_[np.nan, err[hd], np.nan]
    pts = np.column_stack([xs, ys])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    cval = np.array([np.nanmean([ev[i], ev[i + 1]]) for i in range(len(ev) - 1)])
    # pale-pink 1.6pt underlay stroke: keeps the track visible where point error ~ 0
    # (lowest ramp colour is near-white); the 1.1pt error-coloured line draws on top.
    ax.plot(xs, ys, color=C["pink_light"], lw=1.6, solid_capstyle="round", zorder=3.8)
    lc = LineCollection(segs, cmap=CMAP, norm=NORM, linewidths=1.1, capstyle="round", zorder=4)
    lc.set_array(cval)
    ax.add_collection(lc)


# ------------------------------------------------------------------------------------------------ #
#  Panel a -- 3x3 grid                                                                              #
# ------------------------------------------------------------------------------------------------ #
FS.panel_letter(fig, fx(2.0), fy(119.0), "a")
COL_TITLES = ["Truth + anchors", "FQ-GNN (ours)", "RFF-GNN"]
CELL_X = [11.0, 11.0 + CELL_W + 2.5, 11.0 + 2 * (CELL_W + 2.5)]
ROW_Y = [92.0, 65.5, 39.0]

for ci, t in enumerate(COL_TITLES):
    fig.text(fx(CELL_X[ci] + CELL_W / 2), fy(ROW_Y[0] + CELL_H + 1.3), t,
             fontsize=7, ha="center", va="bottom", color=C["axis"])

for ri, d in enumerate(ROWS):
    P = prep_row(d)
    y0 = ROW_Y[ri]
    # row label (rotated), row condition strip
    fig.text(fx(8.0), fy(y0 + CELL_H / 2), d["label"], fontsize=6.5, rotation=90,
             ha="center", va="center", color=C["axis"])
    n_held = len(P["held"])

    axes_row = []
    for ci in range(3):
        ax = fig.add_axes(rect(CELL_X[ci], y0, CELL_W, CELL_H))
        image_cell(ax)
        axes_row.append(ax)

    # col 1: truth + anchors + held circles + chord
    draw_base(axes_row[0], P, C["axis"], 0.8, 1.0, anchors=True, held_circles=True)
    # cols 2/3: gray truth underlay + faint chord + error-coloured prediction + anchors
    for ax, (xp, yp, err) in zip(axes_row[1:], [(P["xf"], P["yf"], P["ef"]),
                                                (P["xr"], P["yr"], P["er"])]):
        draw_base(ax, P, C["ref_gray"], 0.55, 0.45, anchors=True, held_circles=False)
        draw_pred_track(ax, P, xp, yp, err)

    # median point error over the held window, bottom-left of each prediction cell
    # (white glyph halo: the context track exits bottom-left in rows 2-3 and would cross the text)
    for ax, mv in ((axes_row[1], d["fq_med"]), (axes_row[2], d["rff_med"])):
        ax.annotate(f"med {fmt_med(mv)} NM", xy=(0.03, 0.05), xycoords="axes fraction",
                    fontsize=5.5, color=C["text"], ha="left", va="bottom", zorder=9,
                    path_effects=[pe.withStroke(linewidth=1.4, foreground="#FFFFFF")])

    for ax in axes_row:
        ax.set_xlim(P["xlim"]); ax.set_ylim(P["ylim"])
        ax.set_aspect("equal", adjustable="box")

    # condition strip directly under the row's last cell, right-aligned, in the gutter
    fig.text(fx(CELL_X[2] + CELL_W), fy(y0 - 0.7),
             f"gap: {d['gap_h']:.1f} h | held: {n_held} pts", fontsize=5.5,
             ha="right", va="top", color="#6E6E6E")

    # per-row scale bar, bottom-left of first cell only
    xr = P["xlim"][1] - P["xlim"][0]
    yr = P["ylim"][1] - P["ylim"][0]
    L = nice_scalebar(xr)
    bx = P["xlim"][0] + 0.06 * xr
    by = P["ylim"][0] + 0.09 * yr
    axes_row[0].plot([bx, bx + L], [by, by], color="#000000", lw=1.1, solid_capstyle="butt",
                     zorder=8)
    axes_row[0].text(bx + L / 2, by + 0.035 * yr, f"{L:g} NM", fontsize=5.5, ha="center",
                     va="bottom", color=C["axis"], zorder=8)
    if ri == 0:                             # single north arrow, ~60% size, flush top-right corner
        ax = axes_row[0]
        nx = P["xlim"][0] + 0.965 * xr
        ax.annotate("N", xy=(nx, P["ylim"][0] + 0.965 * yr),
                    xytext=(nx, P["ylim"][0] + 0.885 * yr),
                    fontsize=4.5, ha="center", va="bottom", color="#6E6E6E",
                    arrowprops=dict(arrowstyle="-|>", color="#6E6E6E", lw=0.45,
                                    shrinkA=0, shrinkB=0, mutation_scale=6))

# shared colorbar, height-matched to the grid, flush right
cax = fig.add_axes(rect(CELL_X[2] + CELL_W + 4.0, ROW_Y[2], 3.2, ROW_Y[0] + CELL_H - ROW_Y[2]))
from matplotlib.colorbar import ColorbarBase  # noqa: E402  (explicit bar: a data-less
# ScalarMappable leaves the 7-9 NM band of the ramp unpainted before the overflow triangle)
cb = ColorbarBase(cax, cmap=CMAP, norm=NORM, extend="max",
                  ticks=[t for t in (0, 2, 4, 6, 8, 10, 12) if t <= VMAX],   # never beyond VMAX
                  orientation="vertical")
cb.solids.set_edgecolor("face")
cax.tick_params(labelsize=5.5, length=2.0, width=0.6)
cb.outline.set_linewidth(0.5); cb.outline.set_edgecolor(C["sep"])
cax.set_ylabel("Point error (NM, lower better)", fontsize=6, labelpad=2.0)
cax.yaxis.set_label_position("right")

# map key (the ONE shared legend of panel a)
key_handles = [
    Line2D([], [], color=C["axis"], lw=0.8, label="True track"),
    Line2D([], [], ls="none", marker="o", ms=2.4, mfc=C["paleblue"], mec=C["azure"], mew=0.35,
           label="Observed anchor"),
    Line2D([], [], ls="none", marker="o", ms=2.4, mfc="none", mec=C["ref_gray"], mew=0.45,
           label="Held point (truth)"),
    Line2D([], [], color=C["gcd"], lw=0.7, ls=(0, (4, 2.5)), label="GCD chord"),
    Line2D([], [], color=FS.SEQ_PINK[3], lw=1.4, label="Model track\n(colour = error)"),
]
fig.legend(handles=key_handles, loc="center left",
           bbox_to_anchor=(fx(CELL_X[2] + CELL_W + 19.0), fy((ROW_Y[2] + ROW_Y[0] + CELL_H) / 2)),
           fontsize=6, handlelength=1.7, labelspacing=0.9, borderaxespad=0)

# ------------------------------------------------------------------------------------------------ #
#  Panel b -- error-vs-time profiles (3-across, shared log y)                                       #
# ------------------------------------------------------------------------------------------------ #
FS.panel_letter(fig, fx(2.0), fy(34.8), "b")
B_X = [11.0, 68.5, 126.0]
B_W, B_Y, B_H = 49.0, 13.0, 16.0
YLIM = (0.01, 40.0)                         # the hero errors dip to 0.014 NM

for bi, d in enumerate(ROWS):
    P = prep_row(d)
    hd = P["held"]
    t = P["t"][hd]
    tau = (t - t[0]) / (t[-1] - t[0])
    ax = fig.add_axes(rect(B_X[bi], B_Y, B_W, B_H))
    ax.set_yscale("log")
    ax.plot(tau, P["eg"][hd], color=C["gcd"], lw=0.9, ls=(0, (4, 2.5)), zorder=3)
    ax.plot(tau, P["er"][hd], color=C["azure"], lw=0.9, ls="--", marker="^", ms=2.2,
            mew=0, zorder=4)
    ax.plot(tau, P["ef"][hd], color=C["pink"], lw=1.1, ls="-", marker="o", ms=2.0,
            mew=0, zorder=5)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(YLIM)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_yticks([0.1, 1, 10])
    ax.set_yticklabels(["0.1", "1", "10"])
    ax.set_title(d["label"], fontsize=7, pad=2.5)
    ax.tick_params(labelsize=6)
    ax.tick_params(which="minor", left=False)               # no minor ticks
    if bi == 0:
        ax.set_ylabel("Point error\n(NM, log; lower better)", fontsize=6.0, labelpad=1.5)
    else:
        ax.set_yticklabels([])

fig.text(fx(B_X[1] + B_W / 2), fy(7.3), "Position in gap (0 = entry, 1 = exit)",
         fontsize=7, ha="center", va="center", color=C["axis"])

leg_handles = [
    Line2D([], [], color=C["pink"], lw=1.1, ls="-", marker="o", ms=2.0, mew=0,
           label="FQ-GNN (ours)"),
    Line2D([], [], color=C["azure"], lw=0.9, ls="--", marker="^", ms=2.2, mew=0,
           label="RFF-GNN"),
    Line2D([], [], color=C["gcd"], lw=0.9, ls=(0, (4, 2.5)), label="GCD chord"),
]
fig.legend(handles=leg_handles, loc="center", bbox_to_anchor=(0.5, fy(4.4)), ncol=3,
           fontsize=6, handlelength=2.2, columnspacing=2.2, borderaxespad=0)


fig.text(0.5, fy(3.4), "scripted selection per bucket (Methods)  |  seed-0 checkpoints; 5-seed spread in Fig. 6c",
         fontsize=6, color="#6E6E6E", ha="center", va="top")

FS.save(fig, os.path.join(OUT, "fig5"))
print(f"[fig5] shared colour max: 95th pct = {V95:.2f} NM -> VMAX = {VMAX:g} NM")

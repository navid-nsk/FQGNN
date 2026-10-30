"""FIGURE 2 -- "Two-sided reconstruction on Arctic AIS and where the straight line fails".

Panels (180 x 130 mm, 12-col grid, 2 panel bands):
  a (7 cols, band 1)  Arctic map: all test tracks, evaluated gaps by sinuosity bucket, hero gaps.
  b (5 cols, band 1)  The task drawn literally on hero gap 594 (ENU coords).
  c (4 cols, band 2)  GCD median error vs sinuosity (log-log) + running median.
  d (4 cols, band 2)  Absolute median error per bucket for all 10 models (curvature-blindness;
                      incl. AIS-native HABIT / MH-GIN / MultitaskAIS).
  e (4 cols, band 2)  Two-sided vs one-sided FQGNN ratio per bucket (2x5 grouped bars).

Reads ONLY archived CSVs; no recomputation.  Run (any cwd):
    python paper_figures_tables/fig2_build.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import FIGDATA, OUT, RESULTS  # noqa: E402,F401  (env FQGNN_FIGDATA / FQGNN_FIG_OUT / FQGNN_RESULTS_ROOT)
import fig_style as FS  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.path as mpath  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402

import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402

TA = RESULTS

FS.apply()

# ------------------------------------------------------------------------------------------ #
#  DATA                                                                                        #
# ------------------------------------------------------------------------------------------ #
tracks = pd.read_csv(os.path.join(FIGDATA, "map_tracks.csv"))
gindex = pd.read_csv(os.path.join(FIGDATA, "map_gaps_index.csv"))
hero = pd.read_csv(os.path.join(FIGDATA, "hero_gap_gt0.25_594.csv"))
# hero CSVs carry the FULL track; crop to held window +- 20 anchors
_hw = np.where(hero.is_held.to_numpy() == 1)[0]
hero = hero.iloc[max(0, _hw[0] - 20):_hw[-1] + 21].reset_index(drop=True)
# Op-M paper model: per-gap + headline from the CLEAN exact-table re-evals; one-sided buckets
# assembled from the per-seed test_buckets_clean.csv files (the run-root all_seeds aggregate
# holds the in-runner, refreshed-table evaluation -- never read it here).
per_gap = pd.read_csv(os.path.join(TA, "runs", "fqnative_opmstack_5seed_clean", "seed0", "per_gap.csv"))
combined = pd.read_csv(os.path.join(TA, "runs", "paper_table", "combined_buckets_allmodels.csv"))
oneside_b = pd.concat(
    [pd.read_csv(os.path.join(TA, "runs", "oneside_sobolev_5seed_opm", f"seed{s}",
                              "test_buckets_clean.csv")).assign(seed=s) for s in range(5)],
    ignore_index=True)
full_sum = pd.read_csv(os.path.join(TA, "runs", "fqnative_opmstack_5seed", "summary_clean.csv"))
oneside_sum = pd.read_csv(os.path.join(TA, "runs", "oneside_sobolev_5seed_opm", "summary_clean.csv"))

BUCKETS = FS.BUCKETS
BUCKET_N = gindex.bucket.value_counts().reindex(BUCKETS).astype(int)

# headline overall numbers (per-seed rows only; MEAN/STD rows ignored)
fs_seeds = full_sum[full_sum.seed.astype(str).str.isdigit()]
os_seeds = oneside_sum[oneside_sum.seed.astype(str).str.isdigit()]
RATIO_2S = fs_seeds.ratio.astype(float).mean()          # 0.5646 (Op-M clean)
RATIO_1S = os_seeds.ratio.astype(float).mean()          # 3.6946 (Op-M clean)

# ------------------------------------------------------------------------------------------ #
#  CANVAS + GRID                                                                               #
# ------------------------------------------------------------------------------------------ #
fig = FS.new_figure(width_mm=180.0, height_mm=130.0)

B1_TOP, B1_BOT = 0.925, 0.580          # band 1 (a, b): shared plot-area top edge
B2_TOP, B2_BOT = 0.470, 0.190          # band 2 (c, d, e): shared top edge

AX_A = [0.030, B1_BOT, 0.360, B1_TOP - B1_BOT]        # disc is height-limited; key sits in the dead space right of the disc
AX_B = [0.575, B1_BOT, 0.410, B1_TOP - B1_BOT]         # equal-aspect box at the top of the band
AX_C = [0.075, B2_BOT, 0.235, B2_TOP - B2_BOT]        # 4 cols
AX_D = [0.415, B2_BOT, 0.235, B2_TOP - B2_BOT]
AX_E = [0.755, B2_BOT, 0.235, B2_TOP - B2_BOT]

# ------------------------------------------------------------------------------------------ #
#  PANEL a -- Arctic map                                                                       #
# ------------------------------------------------------------------------------------------ #
proj = ccrs.NorthPolarStereo(central_longitude=-80)
pc = ccrs.PlateCarree()
ax_a = fig.add_axes(AX_A, projection=proj, anchor="W")
ax_a.set_extent([-180, 180, 58.5, 90], pc)

theta = np.linspace(0, 2 * np.pi, 200)
circle = mpath.Path(np.column_stack([0.5 + 0.5 * np.cos(theta), 0.5 + 0.5 * np.sin(theta)]))
ax_a.set_boundary(circle, transform=ax_a.transAxes)

ax_a.add_feature(cfeature.NaturalEarthFeature(
    "physical", "land", "50m", facecolor=FS.C["tint_gray"],
    edgecolor=FS.C["sep"], linewidth=0.4, zorder=0))
ax_a.spines["geo"].set_edgecolor(FS.C["sep"])
ax_a.spines["geo"].set_linewidth(0.6)

ax_a.gridlines(crs=pc, draw_labels=False, linewidth=0.35, color=FS.C["grid"],
               linestyle=(0, (2, 2)), xlocs=range(-180, 180, 60), ylocs=[60, 70, 80],
               zorder=1)

# pre-project all track points once (fast path)
xyz = proj.transform_points(pc, tracks.lon.values, tracks.lat.values)
tracks = tracks.assign(px=xyz[:, 0], py=xyz[:, 1])

# full tracks: 0.3 pt navy, alpha 0.3
segs_track = [g[["px", "py"]].values for _, g in tracks.groupby("gap_id", sort=False)]
ax_a.add_collection(LineCollection(segs_track, colors=FS.C["navy"], linewidths=0.3,
                                   alpha=0.30, zorder=2))

# gap segments: 0.8 pt, colored by sinuosity bucket (light first, darker on top)
bucket_of = dict(zip(gindex.gap_id, gindex.bucket))
gap_pts = tracks[tracks.is_gap_segment == 1]
for b in BUCKETS:
    ids = [gid for gid, bb in bucket_of.items() if bb == b]
    sub = gap_pts[gap_pts.gap_id.isin(ids)]
    segs = [g[["px", "py"]].values for _, g in sub.groupby("gap_id", sort=False)]
    if segs:
        ax_a.add_collection(LineCollection(segs, colors=FS.BUCKET_COLOR[b],
                                           linewidths=0.8, zorder=3))

# hero gaps: gold 1.2 pt + small gold circles
heroes = gindex[gindex.hero == 1]
hero_pts = gap_pts[gap_pts.gap_id.isin(heroes.gap_id)]
segs = [g[["px", "py"]].values for _, g in hero_pts.groupby("gap_id", sort=False)]
ax_a.add_collection(LineCollection(segs, colors=FS.C["gold"], linewidths=1.2, zorder=4))
hxyz = proj.transform_points(pc, heroes.center_lon.values, heroes.center_lat.values)
ax_a.plot(hxyz[:, 0], hxyz[:, 1], ls="none", marker="o", markersize=3.2,
          markerfacecolor="none", markeredgecolor=FS.C["gold"], markeredgewidth=0.9, zorder=5)

# graticule labels (manual, 5.5 pt gray)
for lat in (70, 80):
    ax_a.text(-80, lat + 0.6, f"{lat}° N", transform=pc, fontsize=5.5,
              color="#6E6E6E", ha="center", va="bottom", zorder=6)
for lon in (0, 180, 90, -90):
    lab = {0: "0°", 180: "180°", 90: "90° E", -90: "90° W"}[lon]
    ax_a.text(lon, 59.4, lab, transform=pc, fontsize=5.5, color="#6E6E6E",
              ha="center", va="center", zorder=6)

ax_a.set_title("Arctic AIS test tracks and evaluated gaps", fontsize=7.0, pad=4)


# in-panel key in the dead space right of the disc (no box)
def panel_a_key(x0, y0):
    """Panel-a key in the dead space right of the disc (the buckets recur in c, d, e)."""
    dy = 0.031
    fig.text(x0, y0, "gap sinuosity", fontsize=6.0, color=FS.C["text"], ha="left",
             va="center")
    for i, b in enumerate(BUCKETS):
        y = y0 - (i + 1) * dy
        fig.patches.append(Rectangle((x0, y - 0.007), 0.022, 0.014,
                                     facecolor=FS.SEQ_TEAL[i], edgecolor="none",
                                     transform=fig.transFigure))
        fig.text(x0 + 0.030, y, f"{b}  (n = {BUCKET_N[b]})", fontsize=6.0,
                 color=FS.C["text"], ha="left", va="center")
    y = y0 - 6.6 * dy
    fig.lines.append(Line2D([x0, x0 + 0.022], [y, y], color=FS.C["gold"], lw=1.2,
                            transform=fig.transFigure))
    fig.lines.append(Line2D([x0 + 0.011], [y], color=FS.C["gold"], marker="o", ls="none",
                            markersize=3.2, markerfacecolor="none", markeredgewidth=0.9,
                            transform=fig.transFigure))
    fig.text(x0 + 0.030, y, "example gaps (Fig. 5)", fontsize=6.0, color=FS.C["text"], ha="left",
             va="center")
    y = y0 - 7.6 * dy
    fig.lines.append(Line2D([x0, x0 + 0.022], [y, y], color=FS.C["navy"], lw=0.7, alpha=0.45,
                            transform=fig.transFigure))
    fig.text(x0 + 0.030, y, "vessel track", fontsize=6.0, color=FS.C["text"], ha="left",
             va="center")


panel_a_key(0.340, 0.870)

fig.text(0.195, 0.552,
         "region: lat ≥ 60° N | months: Jul–Aug 2016–18 | vessels: test split (disjoint)",
         fontsize=5.5, color="#6E6E6E", ha="center", va="top")

# ------------------------------------------------------------------------------------------ #
#  PANEL b -- the task on hero gap 594 (ENU coords)                                            #
# ------------------------------------------------------------------------------------------ #
ax_b = fig.add_axes(AX_B)
lat0, lon0 = hero.lat_true.mean(), hero.lon_true.mean()
ex = ((hero.lon_true - lon0) * 60.0 * np.cos(np.radians(lat0))).values
ny = ((hero.lat_true - lat0) * 60.0).values
held = (hero.is_held == 1).values
idx = np.arange(len(hero))
i_pre = idx[~held][idx[~held] < idx[held][0]]
i_post = idx[~held][idx[~held] > idx[held][-1]]
i_a, i_z = i_pre[-1], i_post[0]                       # bracketing anchors
i_held = idx[held]

ax_b.plot(ex, ny, color=FS.C["axis"], lw=0.75, zorder=2)                 # truth
ax_b.plot(ex[~held], ny[~held], ls="none", marker="o", markersize=3.0,
          markerfacecolor=FS.C["blue"], markeredgewidth=0, zorder=4)     # anchors
ax_b.plot(ex[held], ny[held], ls="none", marker="o", markersize=2.6,
          markerfacecolor="white", markeredgecolor=FS.C["ref_gray"],
          markeredgewidth=0.6, zorder=3)                                 # held
ax_b.plot([ex[i_a], ex[i_z]], [ny[i_a], ny[i_z]], color=FS.C["gcd"],
          ls=(0, (4, 2.5)), lw=1.0, zorder=2.5)                          # GCD chord

# two-sided cross-reference arrows (curved dotted) converging on one mid-curve held node
tgt = i_held[len(i_held) // 2]                        # held node on the manoeuvre
# hero 594: one arrow from the pre-gap side, two from the post-gap side (the two-sided
# message)
for src, rad in ((i_pre[-3], -0.25), (i_post[2], 0.30), (i_post[6], 0.22)):
    ax_b.add_patch(FancyArrowPatch(
        (ex[src], ny[src]), (ex[tgt], ny[tgt]),
        connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=4,
        linestyle=(0, (1.5, 2)), linewidth=0.6, color=FS.C["ref_gray"],
        shrinkA=2.5, shrinkB=3.0, zorder=5))

# "6 h gap" bracket: hero 594's chord runs ESE (down-right); the held arc sags BELOW the
# chord, so the bracket is placed parallel to the chord on the sag side, offset just beyond
# the arc's farthest excursion (data-driven projection), ticks pointing back at the chord
gm = float(gindex.loc[gindex.gap_id == 594, "gap_min"].iloc[0])
p_a = np.array([ex[i_a], ny[i_a]])
p_z = np.array([ex[i_z], ny[i_z]])
u = (p_z - p_a) / np.linalg.norm(p_z - p_a)
perp = np.array([u[1], -u[0]])                        # normal pointing to the sag side
proj = ((np.c_[ex[i_held], ny[i_held]] - p_a) @ perp)
koff = max(proj.max(), 0.0) + 1.0                     # clear the held arc by 1 NM
b0, b1 = p_a + perp * koff, p_z + perp * koff
tick = -perp * 0.6
for pts in ([b0, b1], [b0, b0 + tick], [b1, b1 + tick]):
    ax_b.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]],
              color="#6E6E6E", lw=0.6, zorder=2)
bm = (b0 + b1) / 2 + perp * 0.45
ang = np.degrees(np.arctan2(u[1], u[0]))
if ang > 90:                                          # keep the bracket label upright
    ang -= 180
elif ang < -90:
    ang += 180
ax_b.text(bm[0], bm[1], f"{gm/60:.0f} h gap · {held.sum()} held points",
          fontsize=6.0, color="#6E6E6E", ha="center", va="top",
          rotation=ang, rotation_mode="anchor", zorder=2)

# direct labels (hero 594: pre-gap anchors descend from the top-left; post-gap anchors run
# far to the east; the held arc sags below the chord)
i_ptop = i_pre[int(np.argmax(ny[i_pre]))]
ax_b.annotate("anchors" + chr(10) + "(before gap)", xy=(ex[i_ptop], ny[i_ptop]),
              xytext=(-9, 0), textcoords="offset points", fontsize=6.0,
              color=FS.C["blue"], ha="right", va="center", linespacing=1.1)
i_pmid = i_post[min(12, len(i_post) - 1)]
ax_b.annotate("anchors" + chr(10) + "(after gap)", xy=(ex[i_pmid], ny[i_pmid]),
              xytext=(0, 9), textcoords="offset points", fontsize=6.0,
              color=FS.C["blue"], ha="center", va="bottom", linespacing=1.1)
# held arc: label anchored on its right shoulder, pushed into the pocket under the chord
i_h60 = i_held[int(0.60 * (len(i_held) - 1))]
ax_b.annotate("held (reconstruct)", xy=(ex[i_h60], ny[i_h60]), xytext=(10, -5),
              textcoords="offset points", fontsize=6.0, color="#6E6E6E", ha="left",
              va="top")
# truth label on the arc's left descent, pulled into the clear left margin
i_h10 = i_held[int(0.10 * (len(i_held) - 1))]
ax_b.annotate("truth", xy=(ex[i_h10], ny[i_h10]), xytext=(-10, -3),
              textcoords="offset points", fontsize=6.0, color=FS.C["axis"], ha="right",
              va="center")
# chord-parallel label in the empty band ABOVE the chord midpoint, white halo
gpos = (p_a + p_z) / 2 + 0.10 * np.linalg.norm(p_z - p_a) * u - perp * 0.9
ax_b.text(gpos[0], gpos[1], "GCD chord", fontsize=6.0, color=FS.C["gcd"],
          ha="center", va="center", rotation=ang, rotation_mode="anchor", zorder=2.6,
          bbox=dict(facecolor="white", edgecolor="none", pad=0.4, alpha=0.85))

ax_b.set_aspect("equal", anchor="N")
ax_b.set_title("Two-sided reconstruction", fontsize=7.0, pad=7)
ax_b.set_xlabel("East (NM)", fontsize=7.0)
ax_b.set_ylabel("North (NM)", fontsize=7.0)

# ------------------------------------------------------------------------------------------ #
#  PANEL c -- GCD error vs curvature (log-log scatter + running median)                        #
# ------------------------------------------------------------------------------------------ #
ax_c = fig.add_axes(AX_C)
pg = per_gap[per_gap.sinuosity.notna()].copy()
sx = pg.sinuosity.clip(lower=1e-4).values
sy = pg.gcd_med.clip(lower=1e-3).values
ax_c.scatter(sx, sy, s=2.0, color=FS.C["navy"], alpha=0.4, linewidths=0, zorder=3)

lb = np.linspace(np.log10(1e-4), np.log10(sx.max()), 15)
mx, my = [], []
for lo, hi in zip(lb[:-1], lb[1:]):
    m = (np.log10(sx) >= lo) & (np.log10(sx) < hi)
    if m.sum() >= 8:
        mx.append(10 ** ((lo + hi) / 2))
        my.append(np.median(sy[m]))
ax_c.plot(mx, my, color=FS.C["navy"], lw=1.25, zorder=4)
ax_c.annotate("running median", xy=(mx[3], my[3]), xytext=(2, -16),
              textcoords="offset points", fontsize=5.5, color=FS.C["navy"], ha="left")

ax_c.set_xscale("log")
ax_c.set_yscale("log")
ax_c.set_xlim(7e-5, 1.2e3)
ax_c.set_ylim(7e-4, 5e3)

edges = [0.02, 0.05, 0.10, 0.25]
for e in edges:
    ax_c.axvline(e, color=FS.C["grid"], lw=0.6, zorder=1)
# teal bucket-span bars along the top edge
spans = [7e-5] + edges + [1.2e3]
tr = mtransforms.blended_transform_factory(ax_c.transData, ax_c.transAxes)
for i in range(5):
    ax_c.add_patch(Rectangle((spans[i], 1.012), spans[i + 1] - spans[i], 0.030,
                             transform=tr, clip_on=False, facecolor=FS.SEQ_TEAL[i],
                             edgecolor="white", linewidth=0.4))

ax_c.set_title("GCD error grows with curvature", fontsize=7.0, pad=8)
ax_c.set_xlabel("Sinuosity (path/chord − 1)", fontsize=7.0)
ax_c.set_ylabel("GCD median error (NM)", fontsize=7.0)
ax_c.set_xticks([1e-4, 1e-2, 1e0, 1e2])
ax_c.set_yticks([1e-3, 1e-1, 1e1, 1e3])

# ------------------------------------------------------------------------------------------ #
#  PANEL d -- absolute error per bucket, all models                                            #
# ------------------------------------------------------------------------------------------ #
ax_d = fig.add_axes(AX_D)
xs = np.arange(5)
overall = combined[combined.bucket == "OVERALL"].groupby("model").model_med.mean()
order = ["FQGNN", "RFFGNN", "CSDI", "SAITS", "BiGRU", "BF-BiLSTM",
         "HABIT", "MH-GIN", "MultitaskAIS", "GCD"]
metrics = {m: f"{overall[m]:.1f}" if overall[m] < 100 else f"{overall[m]:.0f}" for m in order}
for m in order:
    st = FS.MODEL[m]
    sub = combined[(combined.model == m) & (combined.bucket != "OVERALL")]
    mu = sub.groupby("bucket").model_med.mean().reindex(BUCKETS)
    sd = sub.groupby("bucket").model_med.std().reindex(BUCKETS)
    ax_d.errorbar(xs, mu.values, yerr=None if sd.isna().all() else sd.values,
                  color=st["color"], marker=st["marker"], ls=st["ls"], lw=st["lw"],
                  markersize=2.8 if st["marker"] != "*" else 4.0,
                  elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=st["z"])
ax_d.set_yscale("log")
# headroom band above MH-GIN / MultitaskAIS (~1.5e3 NM): the 5-row legend lives there, data-free
ax_d.set_ylim(0.45, 1.5e6)
ax_d.set_yticks([1, 10, 100, 1000])
ax_d.set_xticks(xs)
ax_d.set_xticklabels(BUCKETS, rotation=45, ha="right", fontsize=5.5)
ax_d.set_xlim(-0.5, 4.5)

trd = mtransforms.blended_transform_factory(ax_d.transData, ax_d.transAxes)
for i in range(5):
    ax_d.plot([i], [-0.045], transform=trd, clip_on=False, marker="s", markersize=3.0,
              markerfacecolor=FS.SEQ_TEAL[i], markeredgecolor=FS.C["sep"],
              markeredgewidth=0.3, ls="none")

hs = FS.model_legend_handles(order, metrics)
# 10 entries (5 rows x 2 cols): fontsize drops 5.5 -> 5.0; the long "MultitaskAIS (one-sided)"
# column spills into the d-e gutter (clear of panel e's y-label), the data-free headroom band
# above ~1.5e3 NM absorbs the 5 rows
ax_d.legend(handles=hs, loc="upper left", bbox_to_anchor=(0.01, 1.02), ncol=2,
            fontsize=5.0, handlelength=1.3, handletextpad=0.3, columnspacing=0.6,
            labelspacing=0.18, borderaxespad=0.0)
# below-axis strip (in-axes corners all carry data at the compressed log scale);
# two lines so it does not run into panel e's right-anchored strip
FS.condition_strip(ax_d, "caps = ±s.d. over 5 seeds (n = 1: BF-BiLSTM, HABIT)\n"
                         "medians (NM) in key",
                   dy=-0.50)
ax_d.set_title("Baselines are curvature-blind", fontsize=7.0, pad=4)
ax_d.set_xlabel("Sinuosity bucket", fontsize=7.0)
ax_d.set_ylabel("Median error (NM, lower better)", fontsize=7.0)

# ------------------------------------------------------------------------------------------ #
#  PANEL e -- two-sided vs one-sided per bucket (2x5 grouped bars)                             #
# ------------------------------------------------------------------------------------------ #
ax_e = fig.add_axes(AX_E)
two = combined[(combined.model == "FQGNN") & (combined.bucket != "OVERALL")]
one = oneside_b[oneside_b.bucket != "OVERALL"]
BOT = 0.3
w = 0.38
for i, b in enumerate(BUCKETS):
    v2 = two[two.bucket == b].ratio.values
    v1 = one[one.bucket == b].ratio.values
    ax_e.bar(i - w / 2, v2.mean() - BOT, width=w, bottom=BOT, color=FS.C["pink"], zorder=3)
    ax_e.bar(i + w / 2, v1.mean() - BOT, width=w, bottom=BOT, color=FS.C["pink_light"],
             zorder=3)
    ax_e.errorbar([i - w / 2, i + w / 2], [v2.mean(), v1.mean()],
                  yerr=[v2.std(), v1.std()], ls="none", ecolor="#9C3A63",
                  elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=4)
    FS.seed_dots(ax_e, i - w / 2, v2, FS.C["pink"], width=0.10, size=1.8)
    FS.seed_dots(ax_e, i + w / 2, v1, FS.C["pink_light"], width=0.10, size=1.8)

ax_e.set_yscale("log")
ax_e.set_ylim(BOT, 12)
FS.ref_hline(ax_e, 1.0)
ax_e.set_xticks(np.arange(5))
ax_e.set_xticklabels(BUCKETS, rotation=45, ha="right", fontsize=5.5)
ax_e.set_xlim(-0.6, 4.6)
tre = mtransforms.blended_transform_factory(ax_e.transData, ax_e.transAxes)
for i in range(5):
    ax_e.plot([i], [-0.045], transform=tre, clip_on=False, marker="s", markersize=3.0,
              markerfacecolor=FS.SEQ_TEAL[i], markeredgecolor=FS.C["sep"],
              markeredgewidth=0.3, ls="none")

ax_e.set_yticks([0.5, 1, 2, 5, 10])
ax_e.set_yticklabels(["0.5", "1", "2", "5", "10"])
hs_e = [Rectangle((0, 0), 1, 1, facecolor=FS.C["pink"],
                  label=f"two-sided ({RATIO_2S:.2f} overall)"),
        Rectangle((0, 0), 1, 1, facecolor=FS.C["pink_light"],
                  label=f"one-sided ({RATIO_1S:.2f} overall)"),
        Line2D([0], [0], color=FS.C["gcd"], ls=(0, (4, 2.5)), lw=0.7,
               label="GCD floor (1.0)")]
ax_e.legend(handles=hs_e, loc="upper right", fontsize=5.0, handlelength=1.4,
            handleheight=0.9, handletextpad=0.4, labelspacing=0.3, borderaxespad=0.2)
# seed/cap note lives in the condition strip below the axes (an in-axes note at the upper
# right collides with the >0.25 one-sided bar's seed dots and error cap)
ax_e.set_title("Two-sided context is the lever", fontsize=7.0, pad=4)
ax_e.set_xlabel("Sinuosity bucket", fontsize=7.0)
ax_e.set_ylabel("Median-error ratio vs GCD\n(lower better)", fontsize=7.0)
# right-anchored at the panel edge: a centered strip would clip at the canvas edge
FS.condition_strip(ax_e, "ablation: post-gap anchors removed | GCD stays two-sided\n"
                         "dots = 5 seeds; caps = ±s.d.",
                   dy=-0.53, x=1.0, ha="right")

# ------------------------------------------------------------------------------------------ #
#  PANEL LETTERS (baseline grid: same y per band, flush left of each panel)                    #
# ------------------------------------------------------------------------------------------ #
Y1, Y2 = 0.985, 0.525
FS.panel_letter(fig, 0.012, Y1, "a")
FS.panel_letter(fig, 0.525, Y1, "b")
FS.panel_letter(fig, 0.012, Y2, "c")
FS.panel_letter(fig, 0.352, Y2, "d")
FS.panel_letter(fig, 0.692, Y2, "e")

FS.save(fig, os.path.join(OUT, "fig2"))

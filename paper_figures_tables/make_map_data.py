"""paper_figures_tables/make_map_data.py -- Arctic study-area map data + draft render (Fig 2a).

ANALYSIS ONLY (no training, no model surgery).  Data/models READ-ONLY.

1. Loads the paper test split via task_alpha.baselines.common_gaps.load_gaps() (774 test graphs,
   identical vessel-disjoint split to run_full_faithful_5seed).
2. Per test graph: the full TRUE track polyline (g.true_lonlat, thinned to <=100 points), the GAP
   segment (held rows' lat/lon, thinned to <=100 points), sinuosity (experiment_compare._sinuosity),
   curvature bucket, gap_id = index into te.  Hero gap_ids read from hero_gaps_index.csv (the
   extract_hero_gaps.py output) and flagged -- rerun extract_hero_gaps.py first if the model changed.
3. Writes plot-ready CSVs:
     map_tracks.csv      gap_id, mmsi, point_idx, lat, lon, is_gap_segment
                         (point_idx = ORIGINAL node index in the graph; rows ordered gap_id,
                          then track rows [is_gap_segment=0], then gap rows [is_gap_segment=1])
     map_gaps_index.csv  gap_id, mmsi, sinuosity, bucket, gap_min, center_lat, center_lon, hero
                         (center = the middle HELD node's true lat/lon -- a real on-path point,
                          immune to antimeridian averaging artifacts)
4. DRAFT render fig2a_map_draft.png (300 dpi) -- layout validation only, NOT the final figure:
   cartopy (installed, Natural Earth 50m land/coastline cached offline) LambertAzimuthalEqualArea
   centered on the data; all tracks thin low-alpha navy; gap segments colored by sinuosity bucket
   (5-step teal ramp); hero gaps gold + annotated.  Fallback (no cartopy): simple north-polar
   azimuthal equidistant transform, no land, noted on the figure.

Run:  python paper_figures_tables/make_map_data.py   (any cwd)
"""
from __future__ import annotations

import csv
import math
import os
import sys
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))                  # repository root
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from _paths import FIGDATA  # noqa: E402  (env FQGNN_FIGDATA)
os.makedirs(FIGDATA, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from task_alpha.baselines import common_gaps as CG
from task_alpha.experiment_compare import _sinuosity



def _load_heroes() -> tuple:
    """Scripted hero picks from hero_gaps_index.csv (the extract_hero_gaps.py output)."""
    p = os.path.join(FIGDATA, "hero_gaps_index.csv")
    with open(p, newline="") as f:
        ids = tuple(int(r["gap_id"]) for r in csv.DictReader(f))
    assert len(ids) == 3, f"expected 3 hero gaps in {p}, got {ids}"
    return ids


HEROES = _load_heroes()
MAX_PTS = 100                                                     # per-track / per-gap-segment thinning cap
BUCKET_EDGES = CG.BUCKET_EDGES                                    # (0, .02, .05, .10, .25, 1e9)
BUCKET_LABELS = ("0.00-0.02", "0.02-0.05", "0.05-0.10", "0.10-0.25", ">0.25")
TEAL_RAMP = ("#E8F2EC", "#BFE0D2", "#7FC2AC", "#4F9E84", "#2E7D6B")   # 5-step, one per bucket
HERO_GOLD = "#F2C230"
TRACK_NAVY = "#1F3A5F"


def bucket_of(sn: float) -> str:
    if not np.isfinite(sn):
        return ""
    for lo, hi, lab in zip(BUCKET_EDGES[:-1], BUCKET_EDGES[1:], BUCKET_LABELS):
        if lo <= sn < hi:
            return lab
    return ""


def thin_idx(idx: np.ndarray, cap: int = MAX_PTS) -> np.ndarray:
    """Uniform index thinning to <= cap points, always keeping the first and last."""
    if len(idx) <= cap:
        return idx
    keep = np.unique(np.round(np.linspace(0, len(idx) - 1, cap)).astype(int))
    return idx[keep]


def split_antimeridian(lat: np.ndarray, lon: np.ndarray):
    """Split a polyline wherever consecutive longitudes jump > 180 deg (plot-only cosmetics)."""
    if len(lon) < 2:
        return [(lat, lon)]
    cuts = np.where(np.abs(np.diff(lon)) > 180.0)[0] + 1
    if len(cuts) == 0:
        return [(lat, lon)]
    segs, s = [], 0
    for c in list(cuts) + [len(lon)]:
        if c - s >= 2:
            segs.append((lat[s:c], lon[s:c]))
        s = c
    return segs if segs else [(lat, lon)]


# ------------------------------------------------------------------------------------------------ #
#  1+2+3: extract + write CSVs                                                                       #
# ------------------------------------------------------------------------------------------------ #
def extract(te):
    track_rows, index_rows = [], []
    plot = []                                                     # (gap_id, tr_lat, tr_lon, gp_lat, gp_lon, bucket, hero)
    n_bucket = {lab: 0 for lab in BUCKET_LABELS}; n_nosin = 0
    for gid, g in enumerate(te):
        ll = g.true_lonlat.numpy()                                # [N, 2] = (lat, lon)
        held = g.mask.numpy().astype(bool)
        hidx = np.where(held)[0]
        assert len(hidx) > 0, f"gap {gid}: no held rows"
        sn = _sinuosity(g)
        lab = bucket_of(sn)
        if lab:
            n_bucket[lab] += 1
        else:
            n_nosin += 1
        hero = int(gid in HEROES)
        ti = thin_idx(np.arange(len(held)))                       # full track, <=100 pts
        gi = thin_idx(hidx)                                       # gap segment (held rows), <=100 pts
        for i in ti:
            track_rows.append([gid, int(g.mmsi), int(i), f"{ll[i, 0]:.6f}", f"{ll[i, 1]:.6f}", 0])
        for i in gi:
            track_rows.append([gid, int(g.mmsi), int(i), f"{ll[i, 0]:.6f}", f"{ll[i, 1]:.6f}", 1])
        mid = hidx[len(hidx) // 2]                                # center = middle held node (on-path)
        index_rows.append([gid, int(g.mmsi), f"{sn:.4f}" if np.isfinite(sn) else "", lab,
                           round(float(g.gap_minutes), 1), f"{ll[mid, 0]:.6f}", f"{ll[mid, 1]:.6f}", hero])
        plot.append((gid, ll[ti, 0], ll[ti, 1], ll[gi, 0], ll[gi, 1], lab, hero))

    with open(os.path.join(FIGDATA, "map_tracks.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gap_id", "mmsi", "point_idx", "lat", "lon", "is_gap_segment"]); w.writerows(track_rows)
    with open(os.path.join(FIGDATA, "map_gaps_index.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gap_id", "mmsi", "sinuosity", "bucket", "gap_min", "center_lat", "center_lon", "hero"])
        w.writerows(index_rows)

    print(f"[csv] map_tracks.csv: {len(track_rows)} rows "
          f"({sum(1 for r in track_rows if r[5] == 0)} track pts + "
          f"{sum(1 for r in track_rows if r[5] == 1)} gap-segment pts) over {len(te)} gaps", flush=True)
    print(f"[csv] map_gaps_index.csv: {len(index_rows)} rows | buckets " +
          " ".join(f"{k}:{v}" for k, v in n_bucket.items()) + f" | no-sinuosity:{n_nosin}", flush=True)
    print(f"[csv] hero flags set for gap_ids {[r[0] for r in index_rows if r[7] == 1]}", flush=True)
    return plot


# ------------------------------------------------------------------------------------------------ #
#  4: draft render                                                                                   #
# ------------------------------------------------------------------------------------------------ #
def render(plot):
    all_lat = np.concatenate([p[1] for p in plot]); all_lon = np.concatenate([p[2] for p in plot])
    lat0, lat1 = float(all_lat.min()), float(all_lat.max())
    lon0, lon1 = float(all_lon.min()), float(all_lon.max())
    clat, clon = float(np.median(all_lat)), float(np.median(all_lon))
    print(f"[extent] lat [{lat0:.2f}, {lat1:.2f}]  lon [{lon0:.2f}, {lon1:.2f}]  "
          f"center ({clat:.2f}, {clon:.2f})", flush=True)

    use_cartopy, land_note = False, ""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        use_cartopy = True
    except Exception as e:                                        # pragma: no cover
        print(f"[render] cartopy unavailable ({e}) -> plain azimuthal fallback, no land", flush=True)

    fig = plt.figure(figsize=(10, 9))
    if use_cartopy:
        proj = ccrs.LambertAzimuthalEqualArea(central_longitude=clon, central_latitude=clat)
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.set_extent([lon0 - 3, lon1 + 3, max(lat0 - 1.5, 40.0), min(lat1 + 1.5, 89.5)],
                      crs=ccrs.PlateCarree())
        try:                                                      # Natural Earth 50m is cached offline
            ax.add_feature(cfeature.NaturalEarthFeature("physical", "land", "50m"),
                           facecolor="#E5E2DA", edgecolor="none", zorder=0)
            ax.add_feature(cfeature.NaturalEarthFeature("physical", "coastline", "50m"),
                           facecolor="none", edgecolor="#B8B2A6", linewidth=0.4, zorder=1)
            land_note = "land: Natural Earth 50m (offline cache)"
        except Exception as e:
            land_note = f"land skipped (no offline coastline source: {e})"
            print(f"[render] {land_note}", flush=True)
        ax.set_facecolor("#F5F9FA")
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#9AA5AD", alpha=0.5, linestyle=":")
        gl.top_labels = gl.right_labels = False
        tf = dict(transform=ccrs.PlateCarree())

        def draw(lat, lon, **kw):
            for sl, so in split_antimeridian(np.asarray(lat), np.asarray(lon)):
                ax.plot(so, sl, **tf, **kw)
    else:
        ax = fig.add_subplot(1, 1, 1)
        land_note = "no cartopy -> north-polar azimuthal-equidistant transform, land skipped"

        def _xy(lat, lon):
            r = 90.0 - np.asarray(lat)
            th = np.radians(np.asarray(lon))
            return r * np.sin(th), -r * np.cos(th)

        def draw(lat, lon, **kw):
            x, y = _xy(lat, lon)
            ax.plot(x, y, **kw)
        ax.set_aspect("equal")
        ax.set_xlabel("azimuthal x (deg from pole)"); ax.set_ylabel("azimuthal y (deg from pole)")

    # ---- layers: tracks (bottom) -> non-hero gap segments (by bucket) -> hero gaps (top) ----
    for gid, tla, tlo, gla, glo, lab, hero in plot:
        draw(tla, tlo, color=TRACK_NAVY, lw=0.35, alpha=0.15, zorder=2, solid_capstyle="round")
    for gid, tla, tlo, gla, glo, lab, hero in plot:
        if hero or not lab:
            continue
        c = TEAL_RAMP[BUCKET_LABELS.index(lab)]
        draw(gla, glo, color=c, lw=1.3, alpha=0.9, zorder=3 + BUCKET_LABELS.index(lab),
             solid_capstyle="round")
    for gid, tla, tlo, gla, glo, lab, hero in plot:
        if not hero:
            continue
        draw(gla, glo, color=HERO_GOLD, lw=2.8, alpha=1.0, zorder=10, solid_capstyle="round")
        mid = len(gla) // 2
        if use_cartopy:
            import cartopy.crs as ccrs
            ax.annotate(f"gap {gid}", xy=(glo[mid], gla[mid]),
                        xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                        xytext=(14, 10), textcoords="offset points", fontsize=9, fontweight="bold",
                        color="#7A5C00", zorder=11,
                        arrowprops=dict(arrowstyle="-", color="#7A5C00", lw=0.8))

    handles = [Line2D([0], [0], color=TRACK_NAVY, lw=1.2, alpha=0.5, label="vessel track (test)")]
    handles += [Line2D([0], [0], color=c, lw=3, label=f"gap, sinuosity {lab}")
                for c, lab in zip(TEAL_RAMP, BUCKET_LABELS)]
    handles += [Line2D([0], [0], color=HERO_GOLD, lw=3, label=f"hero gaps {HEROES}")]
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title(f"Fig 2a DRAFT -- Arctic study area: {len(plot)} test gaps (2016-2018 Jul-Aug AIS)\n"
                 f"[layout validation only; {land_note}]", fontsize=10)

    out = os.path.join(FIGDATA, "fig2a_map_draft.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[render] wrote {os.path.relpath(out, ROOT)} (300 dpi, cartopy={use_cartopy})", flush=True)


def main():
    tr, va, te = CG.load_gaps()
    assert len(te) == 774, f"expected 774 test graphs, got {len(te)}"
    plot = extract(te)

    # validation: bucket counts must match the paper table (770 finite-sinuosity gaps)
    exp = {"0.00-0.02": 225, "0.02-0.05": 74, "0.05-0.10": 75, "0.10-0.25": 105, ">0.25": 291}
    got = {lab: sum(1 for p in plot if p[5] == lab) for lab in BUCKET_LABELS}
    ok = all(got[k] == v for k, v in exp.items())
    print(f"[check] bucket counts vs paper table: {'PASS' if ok else 'FAIL ' + str(got)}", flush=True)
    for gid in HEROES:
        assert any(p[0] == gid and p[6] for p in plot), f"hero {gid} not flagged"
    print(f"[check] hero gaps {HEROES} present + flagged: PASS", flush=True)

    render(plot)
    print("[N8] DONE", flush=True)


if __name__ == "__main__":
    main()

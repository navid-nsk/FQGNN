"""_diag_ref_cert_8cells.py -- reference-substitution certificate on all 8 cells.

Extends the S4.6 comparator certificate (_diag_atom_vs_gcd.py, shared 6-h cell) to every
preprocessing-by-horizon cell of the paper: for each cell's test gaps it scores the stored
chart-linear reference (RunGraph.linterp_lonlat, the deployed GCD column) and an exact fp64
great-circle slerp between the same bracketing truth anchors at matched time fractions, both
against the truth, and reports the cell medians of the per-gap median errors, the % shift, the
implied scale on every model-to-reference ratio (ratio-of-medians: models share the reference
denominator, so the substitution multiplies every cell ratio by chart_med / slerp_med), and the
chart-seam gaps (bracketing anchors straddling +-180 deg), which are kept in both columns.

Run from the repository root:  python -m task_alpha._diag_ref_cert_8cells
Output: task_alpha/runs/ref_cert_8cells.csv + console table.
"""
import csv
import math
import os
import pickle

import numpy as np

R_EARTH_NM = 6371000.0 / 1852.0

CELLS = [("our_processed", g) for g in (3, 6, 12, 24)] + \
        [("standard_process", g) for g in (2, 6, 12, 24)]

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
OUT_CSV = os.path.join(os.path.dirname(__file__), "runs", "ref_cert_8cells.csv")


def haversine_nm(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(np.radians, (np.asarray(lat1, np.float64), np.asarray(lon1, np.float64),
                                          np.asarray(lat2, np.float64), np.asarray(lon2, np.float64)))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R_EARTH_NM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _unitvec(lat_deg, lon_deg):
    la, lo = math.radians(float(lat_deg)), math.radians(float(lon_deg))
    return np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])


def _to_latlon(u):
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    return (np.degrees(np.arcsin(np.clip(u[..., 2], -1, 1))),
            np.degrees(np.arctan2(u[..., 1], u[..., 0])))


def _slerp(ua, ub, w):
    th = math.atan2(float(np.linalg.norm(np.cross(ua, ub))),
                    float(np.clip(np.dot(ua, ub), -1, 1)))
    w = np.asarray(w, dtype=np.float64)
    if th < 1e-12:
        return np.outer(1 - w, ua) + np.outer(w, ub)
    return (np.sin((1 - w) * th)[:, None] * ua[None, :]
            + np.sin(w * th)[:, None] * ub[None, :]) / math.sin(th)


def cell_cert(track, gap_h):
    with open(os.path.join(DATA_ROOT, track, f"gap_{gap_h}h", "test.pkl"), "rb") as f:
        te = pickle.load(f)
    med_chart, med_slerp, n_seam, n_used = [], [], 0, 0
    for g in te:
        m = g.mask.numpy().astype(bool)
        if m.sum() == 0:
            continue
        held = np.where(m)[0]
        si, last = int(held[0]), int(held[-1])
        if si == 0 or last + 1 >= len(m):
            continue                      # no bracketing truth anchor on one side
        trll = g.true_lonlat.numpy().astype(np.float64)
        li = g.linterp_lonlat.numpy().astype(np.float64)[m]
        err_chart = haversine_nm(trll[m, 0], trll[m, 1], li[:, 0], li[:, 1])

        Pa, Pb = trll[si - 1], trll[last + 1]
        dlon_naive = Pb[1] - Pa[1]
        dlon_wrap = ((dlon_naive + 180) % 360) - 180
        if abs(dlon_wrap - dlon_naive) > 1e-9:
            n_seam += 1
        coord = g.coord.numpy().astype(np.float64)
        ta, tb = coord[si - 1, 0], coord[last + 1, 0]
        w = (coord[held, 0] - ta) / max(tb - ta, 1e-12)
        sl_lat, sl_lon = _to_latlon(_slerp(_unitvec(*Pa), _unitvec(*Pb), w))
        err_slerp = haversine_nm(trll[m, 0], trll[m, 1], sl_lat, sl_lon)

        med_chart.append(float(np.median(err_chart)))
        med_slerp.append(float(np.median(err_slerp)))
        n_used += 1
    mc, ms = float(np.median(med_chart)), float(np.median(med_slerp))
    return dict(track=track, gap_h=gap_h, n_gaps=n_used, n_seam=n_seam,
                ref_med_chart_nm=mc, ref_med_slerp_nm=ms,
                pct_shift=100.0 * (ms - mc) / mc,
                ratio_scale=mc / ms)


def main():
    rows = []
    for track, gap_h in CELLS:
        r = cell_cert(track, gap_h)
        rows.append(r)
        print(f"{track:>17s} {gap_h:>2d}h : n={r['n_gaps']:5d} seam={r['n_seam']} "
              f"chart {r['ref_med_chart_nm']:.4f} NM  slerp {r['ref_med_slerp_nm']:.4f} NM  "
              f"shift {r['pct_shift']:+.3f}%  ratio x{r['ratio_scale']:.5f}", flush=True)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)
    print(f"\nwrote {OUT_CSV}")
    worst = max(rows, key=lambda r: abs(r["pct_shift"]))
    print(f"largest |shift| across the 8 cells: {worst['pct_shift']:+.3f}% "
          f"({worst['track']} {worst['gap_h']}h)")


if __name__ == "__main__":
    main()

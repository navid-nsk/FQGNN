"""_diag_atom_vs_gcd.py -- SI S4.6 certificate: the readout's affine atom vs the evaluated GCD
reference, and the chart-chord vs exact-great-circle deviation.

Certifies, over the shared paper-table test cell (task_alpha.baselines.common_gaps.load_gaps —
774 test gaps, 33,234 held vertices, 770 finite-sinuosity forming the shared table):

  (1) IDENTITY [Prop S4.3]: a bit-faithful fp32 replica of the deployed atom construction
      (fq_oph.oph_readout / fq_opm.opm_readout: sort anchors, argmax time-jump, chord), pushed
      through the scoring map of metrics.model_errors (x POS_OUT_SCALE, enu_to_lonlat at the
      run's (lat0, lon0)), against the STORED reference RunGraph.linterp_lonlat that
      metrics.gcd_errors scores.  In exact arithmetic the two are the same map (both are
      per-coordinate linear interpolation of (lat, lon) between the bracketing anchors at
      matched time fractions); the certificate measures the fp residual and checks the (H1)
      bracketing-identification hypothesis per gap (argmax pair == (si-1, last+1); margin =
      max same-side anchor spacing / held span, structurally < 63 min / held span on the raw
      track: runs split at >1 h, thinned to >=180 s).
  (2) GEODESIC DEVIATION [Prop S4.4]: the implemented chart chord vs the exact great-circle
      slerp between the same anchors at the same fractions (fp64, 3-D unit vectors), against
      the leading-order meridian-convergence bound
          d_max ~ (tan(phi_m) Lx / (8 R)) sqrt(Lx^2 + 4 Ly^2) + L^3/(62 R^2),
      Lx = R cos(phi_m)|dlambda|, Ly = R|dphi|.  The two gaps whose bracketing anchors straddle
      the +-180 deg meridian (chart-seam gaps: the run chart is not local there, reference
      errors ~2,000-2,800 NM, model frame identically wrapped) are excluded from (2) and
      reported separately; the identity (1) holds on them regardless.

Run from the repository root:  python -m task_alpha._diag_atom_vs_gcd
Expected output (dataset cache dd428e5ef99f):  (H1) 0/774 failures, margin max 0.161; identity
residual max 5.9e-3 NM (median 1.1e-4); deviation median 0.040 NM, p95(per-gap max) 0.74 NM,
max 2.10 NM at L=138 NM / 69.6 N vs formula 2.12 NM; seam gaps 196 & 381.
"""
import math

import numpy as np
import torch

from task_alpha.baselines.common_gaps import load_gaps
from task_alpha.data import POS_OUT_SCALE, enu_to_lonlat, haversine_nm
from task_alpha.experiment_compare import _sinuosity
from task_alpha.metrics import gcd_errors

R_NM = 6371000.0 / 1852.0


def _unitvec(lat_deg, lon_deg):
    la, lo = np.radians(float(lat_deg)), np.radians(float(lon_deg))
    return np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])


def _to_latlon(u):
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    return (np.degrees(np.arcsin(np.clip(u[..., 2], -1, 1))),
            np.degrees(np.arctan2(u[..., 1], u[..., 0])))


def _slerp(ua, ub, w):
    """Exact constant-angular-speed great-circle interpolation at fractions w (fp64)."""
    th = math.atan2(float(np.linalg.norm(np.cross(ua, ub))),
                    float(np.clip(np.dot(ua, ub), -1, 1)))
    w = np.asarray(w, dtype=np.float64)
    if th < 1e-12:
        return np.outer(1 - w, ua) + np.outer(w, ub)
    return (np.sin((1 - w) * th)[:, None] * ua[None, :]
            + np.sin(w * th)[:, None] * ub[None, :]) / math.sin(th)


def main():
    _, _, te = load_gaps()
    pg, d_atom_all, err_diff_all, d_slerp_ns = [], [], [], []
    for gi, g in enumerate(te):
        m = g.mask.numpy().astype(bool)
        held = np.where(m)[0]
        si, last = int(held[0]), int(held[-1])
        coord, x = g.coord, g.x

        # ---- (1) bit-faithful fp32 replica of the deployed atom (fq_oph.oph_readout) ----
        anc = torch.unique(g.anchor_edge_index[0])
        at, ay = coord[anc, 0], x[anc, :2]
        idx = torch.argsort(at)
        ts, ys = at[idx], ay[idx]
        gapi = int(torch.argmax(ts[1:] - ts[:-1]))
        a, b = ts[gapi], ts[gapi + 1]
        pa, pb = ys[gapi], ys[gapi + 1]
        slope = (pb - pa) / (b - a).clamp_min(1e-9)
        qt = coord[torch.from_numpy(held), 0]
        A_q = pa.unsqueeze(0) + (qt - a).unsqueeze(-1) * slope.unsqueeze(0)

        anc_sorted = anc[idx]
        ok = (int(anc_sorted[gapi]) == si - 1) and (int(anc_sorted[gapi + 1]) == last + 1)
        sp = np.diff(ts.double().numpy())
        margin = np.delete(sp, gapi).max() / sp[gapi] if len(sp) > 1 else 0.0

        lat0, lon0 = g.ref
        enu = A_q.numpy().astype(np.float64) * POS_OUT_SCALE
        plat, plon = enu_to_lonlat(enu[:, 0], enu[:, 1], lat0, lon0)
        li = g.linterp_lonlat.numpy()[m].astype(np.float64)
        d_atom = haversine_nm(plat, plon, li[:, 0], li[:, 1])
        d_atom_all.append(d_atom)

        trll = g.true_lonlat.numpy().astype(np.float64)
        err_atom = haversine_nm(trll[m, 0], trll[m, 1], plat, plon)
        err_gcd = gcd_errors(g)
        err_diff_all.append(np.abs(err_atom - err_gcd))
        med_diff = abs(float(np.median(err_atom)) - float(np.median(err_gcd)))

        # ---- (2) exact geodesic slerp at matched fractions + leading-order bound ----
        Pa, Pb = trll[si - 1], trll[last + 1]
        dlon_naive = Pb[1] - Pa[1]
        dlon_wrap = ((dlon_naive + 180) % 360) - 180
        seam = abs(dlon_wrap - dlon_naive) > 1e-9      # anchors straddle +-180 in the chart
        ta, tb = float(coord[si - 1, 0]), float(coord[last + 1, 0])
        w = (qt.double().numpy() - ta) / max(tb - ta, 1e-12)
        sl_lat, sl_lon = _to_latlon(_slerp(_unitvec(*Pa), _unitvec(*Pb), w))
        d_slerp = haversine_nm(sl_lat, sl_lon, li[:, 0], li[:, 1])
        if not seam:
            d_slerp_ns.append(d_slerp)

        phi_m = math.radians(0.5 * (Pa[0] + Pb[0]))
        Lx = R_NM * math.cos(phi_m) * abs(math.radians(dlon_wrap))
        Ly = R_NM * abs(math.radians(Pb[0] - Pa[0]))
        L = float(haversine_nm(np.array([Pa[0]]), np.array([Pa[1]]),
                               np.array([Pb[0]]), np.array([Pb[1]]))[0])
        bound = (math.tan(phi_m) * Lx / (8 * R_NM)) * math.sqrt(Lx ** 2 + 4 * Ly ** 2) \
            + L ** 3 / (62.0 * R_NM ** 2)
        pg.append(dict(gap=gi, n=len(held), ok=ok, margin=margin, seam=seam,
                       d_atom_max=float(d_atom.max()), d_slerp_max=float(d_slerp.max()),
                       bound=bound, L=L, lat=0.5 * (Pa[0] + Pb[0]), med_diff=med_diff,
                       gcd_med=float(np.median(err_gcd)), sin=_sinuosity(g)))

    d_atom_all = np.concatenate(d_atom_all)
    err_diff_all = np.concatenate(err_diff_all)
    d_slerp_ns = np.concatenate(d_slerp_ns)
    ns = [r for r in pg if not r["seam"]]
    seams = [r for r in pg if r["seam"]]

    print(f"\n===== IDENTITY (Prop S4.3): {len(pg)} gaps, {sum(r['n'] for r in pg)} vertices =====")
    print(f"(H1) bracketing failures: {sum(not r['ok'] for r in pg)} / {len(pg)} ; "
          f"margin max {max(r['margin'] for r in pg):.4f} (< 1 required, < 0.175 structural)")
    print(f"atom image vs stored reference : max {d_atom_all.max():.3e} NM  "
          f"median {np.median(d_atom_all):.3e} NM")
    print(f"per-vertex |err diff| max {err_diff_all.max():.3e} NM ; "
          f"per-gap |median-err diff| max {max(r['med_diff'] for r in pg):.3e} NM")

    print(f"\n===== GEODESIC DEVIATION (Prop S4.4): {len(ns)} non-seam gaps =====")
    pgm = [r["d_slerp_max"] for r in ns]
    print(f"per-vertex: max {d_slerp_ns.max():.4f}  median {np.median(d_slerp_ns):.5f} NM ; "
          f"per-gap max: p95 {np.percentile(pgm, 95):.4f}  median {np.median(pgm):.5f} NM")
    wr = max(ns, key=lambda r: r["d_slerp_max"])
    print(f"worst gap: dev {wr['d_slerp_max']:.4f} NM vs formula {wr['bound']:.4f} NM "
          f"(L {wr['L']:.1f} NM, lat {wr['lat']:.1f})")
    viol = [r for r in ns if r["d_slerp_max"] > 1.5 * r["bound"] + 1e-9]
    print(f"excursions >1.5x formula: {len(viol)} gaps, largest ABSOLUTE deviation among them "
          f"{max((r['d_slerp_max'] for r in viol), default=0):.2e} NM (formula-tail at short L)")
    print(f"chords: L max {max(r['L'] for r in ns):.1f} / median "
          f"{np.median([r['L'] for r in ns]):.1f} NM ; lat "
          f"[{min(r['lat'] for r in ns):.1f}, {max(r['lat'] for r in ns):.1f}]")

    print(f"\n===== chart-seam gaps (excluded from deviation cert; identity certified) =====")
    for r in seams:
        print(f"gap {r['gap']}: reference med err {r['gcd_med']:.1f} NM, "
              f"d_atom_max {r['d_atom_max']:.2e} NM, sinuosity {r['sin']:.4f}")


if __name__ == "__main__":
    main()

"""paper_figures_tables/extract_hero_gaps.py -- hero-gap reconstruction extraction.

ANALYSIS ONLY (no training; GPU inference only; fp32).  Never modifies any model source.

SCRIPTED hero-gap selection (no cherry-picking): for each sinuosity bucket [0.00-0.02), [0.05-0.10),
(>0.25), among the test gaps on which FQGNN seed0 beats GCD, the published cubic-Hermite baseline
(Nguyen 2015) AND RFF-GNN simultaneously (the triple-win rule in main()), pick the gap whose ratio
(model_med / gcd_med) is the MEDIAN of that winning set (lower median: sorted ratios, index
(n_win-1)//2 -- a deterministic actual member of the set).

For each chosen gap the full tracks are reconstructed:
  truth lat/lon (g.true_lonlat), held mask, times (common_gaps.to_series t_sec, full graph),
  GCD chord (g.linterp_lonlat),
  FQGNN seed0 prediction  (the SOBOLEV-backbone Op-H model, exact run_fqnative_sobolev_5seed test
                           path: forward(+gap_minutes) + metrics conversion
                           enu = pred*POS_OUT_SCALE -> enu_to_lonlat(...g.ref)),
  RFF-GNN seed0 prediction (the runner's own RFFGNN class + eval path),
  per-held-point great-circle NM error for each model.

Writes (FQGNN_FIGDATA):
  hero_gap_<bucket>_<gapid>.csv   node_idx, t_sec, is_held, lat_true, lon_true, lat_gcd, lon_gcd,
                                  lat_fq, lon_fq, lat_rff, lon_rff, err_nm_gcd, err_nm_fq, err_nm_rff
                                  (errors only at held rows, blank elsewhere)
  hero_gaps_index.csv             bucket, gap_id, mmsi, gap_min, sinuosity, gcd_med, fq_med, rff_med

Run:  python paper_figures_tables/extract_hero_gaps.py  (any cwd)
"""
from __future__ import annotations

import csv
import importlib.util
import math
import os
import random
import sys
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))                  # repository root
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from _paths import FIGDATA  # noqa: E402  (env FQGNN_FIGDATA)
os.makedirs(FIGDATA, exist_ok=True)

import numpy as np
import torch

from task_alpha.baselines import common_gaps as CG
from task_alpha.data import POS_OUT_SCALE, collate_graphs_interp, enu_to_lonlat
from task_alpha.fqdeep_native import FQDeepNative
from task_alpha.metrics import gcd_errors, model_errors

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FQ_CKPT = os.path.join(ROOT, "task_alpha", "runs", "fqnative_sobolev_5seed", "seed0", "best.pt")
FQ_PERGAP = os.path.join(ROOT, "task_alpha", "runs", "fqnative_sobolev_5seed", "seed0", "per_gap.csv")
# ---- OP-M STACK (FQ_READOUT_MODE=opm): the paper model uses the Op-M readout; heroes are
# derived from its seed-0 checkpoint via the SAME scripted selection, ranked on the OFFICIAL
# exact-table re-eval per-gaps (per_gap_clean.csv).  Launch under the full stack env with
# FQ_AMP_REFRESH unset so the fresh model constructs the Op-M params and predicts with exact
# amplitude tables.  With the env unset, the sobolev sources above are used.
if os.environ.get("FQ_READOUT_MODE") == "opm":
    assert os.environ.get("FQ_AMP_REFRESH", "0") in ("", "0"), \
        "FQ_AMP_REFRESH must be UNSET for official hero predictions (exact tables)"
    FQ_CKPT = os.path.join(ROOT, "task_alpha", "runs", "fqnative_opmstack_5seed", "seed0", "best.pt")
    FQ_PERGAP = os.path.join(ROOT, "task_alpha", "runs", "fqnative_opmstack_5seed", "seed0",
                             "per_gap_clean.csv")
RFF_CKPT = os.path.join(ROOT, "task_alpha", "baselines", "RFFGNN", "runs_5seed_gaps", "seed0", "best.pt")
RFF_PERGAP = os.path.join(ROOT, "task_alpha", "baselines", "RFFGNN", "runs_5seed_gaps", "seed0", "per_gap.csv")
RFF_RUNNER = os.path.join(ROOT, "task_alpha", "baselines", "RFFGNN", "run_rffgnn_5seed_gaps.py")
CTOR = dict(n=6, n_freqs=48, width=96, init_scale=1.84, seed=0, in_feat=6, layers=3, jacobi_R=4,
            oph_weighted=True, dmu_source="blend", metric_backbone="sobolev2")
BUCKETS = ((0.00, 0.02, "0.00-0.02"), (0.05, 0.10, "0.05-0.10"), (0.25, float("inf"), "gt0.25"))


def _gm(graphs):
    """gap_minutes kwarg tensor -- the blend Dmu jet needs per-graph absolute gap durations."""
    return torch.tensor([float(g.gap_minutes) for g in graphs], device=DEV)


def build_fq(tr4):
    """Fresh FQDeepNative exactly as run_fqnative_sobolev_5seed.run_seed(seed=0) builds it."""
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    m = FQDeepNative(**CTOR).to(DEV)
    m.use_cuda = True
    with torch.no_grad():
        m.s_raw.fill_(math.log(math.expm1(2.5)))
    b0 = collate_graphs_interp(tr4)
    bv = torch.cat([torch.full((g.x.shape[0],), i, dtype=torch.long) for i, g in enumerate(tr4)]).to(DEV)
    with torch.no_grad():
        m(b0[0].to(DEV), b0[1].to(DEV), b0[4].to(DEV), b0[6].to(DEV), bv, gap_minutes=_gm(tr4))
    sd = torch.load(FQ_CKPT, map_location=DEV)
    m.load_state_dict(sd["state_dict"]); m.eval()
    return m


def load_rff():
    spec = importlib.util.spec_from_file_location("rff_runner_mod", RFF_RUNNER)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    m = mod.RFFGNN(seed=0).to(DEV)
    sd = torch.load(RFF_CKPT, map_location=DEV)
    m.load_state_dict(sd["state_dict"]); m.eval()
    return m


def read_pergap(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def pred_lonlat(pred, g):
    """metrics.model_errors conversion, applied to ALL nodes: pred [N,2] POS_OUT_SCALE units ->
    ENU metres -> (lat, lon) about g.ref."""
    enu = pred.detach().cpu().numpy() * POS_OUT_SCALE
    lat0, lon0 = g.ref
    return enu_to_lonlat(enu[:, 0], enu[:, 1], lat0, lon0)


def main():
    tr, va, te = CG.load_gaps()
    fq_rows = read_pergap(FQ_PERGAP)               # row i = te[i] (no gap_id column)
    rff_rows = read_pergap(RFF_PERGAP)             # HAS gap_id column = te index
    assert len(fq_rows) == len(te) and len(rff_rows) == len(te), \
        f"per_gap rows ({len(fq_rows)}/{len(rff_rows)}) != test graphs ({len(te)})"
    rff_by_id = {int(r["gap_id"]): r for r in rff_rows}
    # row-alignment check: per_gap row i must describe te[i]
    for i, (r, g) in enumerate(zip(fq_rows, te)):
        assert int(r["mmsi"]) == int(g.mmsi) and abs(float(r["gap_min"]) - float(g.gap_minutes)) < 0.06, \
            f"per_gap row {i} misaligned with te[{i}]"
        assert int(rff_by_id[i]["mmsi"]) == int(g.mmsi), f"RFF per_gap gap_id {i} misaligned"
    print(f"[align] FQ per_gap rows == te order verified for all {len(te)} gaps", flush=True)

    # ---- SCRIPTED hero selection ----
    # Rule (disclosed; no manual curation): per bucket, take the MEDIAN-ratio gap among those
    # on which FQ-GNN beats the geodesic, the published cubic-Hermite AIS interpolation
    # baseline (Nguyen, Im & Lee 2015) AND RFF-GNN simultaneously (a triple-win), so every
    # showcased reconstruction is one on which the model outperforms all three references at once.
    herm_rows = read_pergap(os.path.join(ROOT, "task_alpha", "runs", "gap_sweep",
                                         "our_processed", "gap_6h", "hermite", "seed0",
                                         "per_gap.csv"))
    rff_by_row = {int(float(r["gap_id"])): float(r["model_med"]) for r in rff_rows}
    chosen = []
    print("\n=== HERO SELECTION (median-ratio FQGNN triple-win per bucket: beats GCD, "
          "Hermite and RFF-GNN) ===", flush=True)
    for lo, hi, lab in BUCKETS:
        wins = []
        for i, r in enumerate(fq_rows):
            if r["sinuosity"] == "" or int(r["win"]) != 1:
                continue
            sn = float(r["sinuosity"])
            if not (lo <= sn < hi):
                continue
            gm, fm = float(r["gcd_med"]), float(r["model_med"])
            hv = float(herm_rows[i]["model_med"])
            rv = rff_by_row.get(i)
            if rv is None or fm >= hv or fm >= rv:
                continue
            wins.append((fm / gm, i, sn, gm, fm))
        assert wins, f"no FQGNN triple-wins in bucket {lab}"
        wins.sort(key=lambda w: w[0])
        ratio, gid, sn, gm, fm = wins[(len(wins) - 1) // 2]          # lower median (actual member)
        chosen.append((lab, gid, sn, gm, fm, ratio))
        print(f"  bucket {lab:<9}: {len(wins):3d} winning gaps | ratio range "
              f"[{wins[0][0]:.3f}, {wins[-1][0]:.3f}] | MEDIAN pick gap_id={gid} "
              f"(mmsi {fq_rows[gid]['mmsi']}, sinuosity {sn:.4f}, gap {fq_rows[gid]['gap_min']} min, "
              f"GCD {gm:.3f} NM, FQ {fm:.3f} NM, ratio {ratio:.3f})", flush=True)

    # ---- models (loaded once) ----
    m_fq = build_fq(tr[:4])
    m_rff = load_rff()

    index_rows = []
    for lab, gid, sn, gm_csv, fm_csv, ratio in chosen:
        g = te[gid]
        x, co, yp, mk, ei, ed, aei = collate_graphs_interp([g])
        bv = torch.zeros(g.x.shape[0], dtype=torch.long, device=DEV)
        with torch.no_grad():
            pred_fq = m_fq(x.to(DEV), co.to(DEV), ei.to(DEV), aei.to(DEV), bv,
                           gap_minutes=_gm([g])).float().cpu()
            pred_rf = m_rff(x.to(DEV), co.to(DEV), ei.to(DEV), aei.to(DEV)).float().cpu()

        lat_fq, lon_fq = pred_lonlat(pred_fq, g)
        lat_rf, lon_rf = pred_lonlat(pred_rf, g)
        err_fq = model_errors(pred_fq, g)          # NM at held (held-node order)
        err_rf = model_errors(pred_rf, g)
        err_gc = gcd_errors(g)
        held = g.mask.numpy().astype(bool)
        tr_ll = g.true_lonlat.numpy(); li_ll = g.linterp_lonlat.numpy()
        s = CG.to_series(g, crop_ctx=10 ** 9)      # full graph (no crop): exact to_series t_sec
        assert len(s["t_sec"]) == len(held) and (s["node_idx"] == np.arange(len(held))).all()
        t_sec = s["t_sec"]

        fm, rm_, gm = float(np.median(err_fq)), float(np.median(err_rf)), float(np.median(err_gc))
        rff_csv = float(rff_by_id[gid]["model_med"])
        print(f"\n[gap {gid} | bucket {lab}] mmsi {g.mmsi} | {float(g.gap_minutes):.1f} min gap | "
              f"{int(held.sum())} held / {len(held)} nodes | sinuosity {sn:.4f}", flush=True)
        print(f"  median NM err: GCD {gm:.3f} (csv {gm_csv:.3f}) | FQ {fm:.3f} (csv {fm_csv:.3f}) | "
              f"RFF {rm_:.3f} (csv {rff_csv:.3f})", flush=True)
        assert abs(gm - gm_csv) < 5e-3, "GCD median does not reproduce per_gap.csv"
        assert abs(fm - fm_csv) < 0.05, "FQ median does not reproduce per_gap.csv"
        assert abs(rm_ - rff_csv) < 0.05, "RFF median does not reproduce per_gap.csv"

        # per-node errors (held order -> node order)
        efq = np.full(len(held), np.nan); erf = np.full(len(held), np.nan); egc = np.full(len(held), np.nan)
        efq[held], erf[held], egc[held] = err_fq, err_rf, err_gc
        for name, arr in (("t_sec", t_sec), ("truth", tr_ll), ("gcd", li_ll),
                          ("fq", np.c_[lat_fq, lon_fq]), ("rff", np.c_[lat_rf, lon_rf]),
                          ("err_held", np.c_[err_fq, err_rf, err_gc])):
            assert np.isfinite(np.asarray(arr, float)).all(), f"NaN/Inf in {name} of gap {gid}"

        path = os.path.join(FIGDATA, f"hero_gap_{lab}_{gid}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["node_idx", "t_sec", "is_held", "lat_true", "lon_true", "lat_gcd", "lon_gcd",
                        "lat_fq", "lon_fq", "lat_rff", "lon_rff", "err_nm_gcd", "err_nm_fq", "err_nm_rff"])
            for i in range(len(held)):
                w.writerow([i, f"{t_sec[i]:.1f}", int(held[i]),
                            f"{tr_ll[i,0]:.6f}", f"{tr_ll[i,1]:.6f}",
                            f"{li_ll[i,0]:.6f}", f"{li_ll[i,1]:.6f}",
                            f"{lat_fq[i]:.6f}", f"{lon_fq[i]:.6f}",
                            f"{lat_rf[i]:.6f}", f"{lon_rf[i]:.6f}",
                            f"{egc[i]:.4f}" if held[i] else "",
                            f"{efq[i]:.4f}" if held[i] else "",
                            f"{erf[i]:.4f}" if held[i] else ""])
        print(f"  wrote {os.path.relpath(path, ROOT)}  ({len(held)} rows)", flush=True)
        index_rows.append(dict(bucket=lab, gap_id=gid, mmsi=int(g.mmsi),
                               gap_min=round(float(g.gap_minutes), 1), sinuosity=round(sn, 4),
                               gcd_med=round(gm, 3), fq_med=round(fm, 3), rff_med=round(rm_, 3),
                               ratio_fq_gcd=round(fm / gm, 4)))

    ipath = os.path.join(FIGDATA, "hero_gaps_index.csv")
    with open(ipath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index_rows[0].keys())); w.writeheader(); w.writerows(index_rows)
    print(f"\nwrote {os.path.relpath(ipath, ROOT)}", flush=True)

    print("\n=== N6 SUMMARY ===", flush=True)
    for r in index_rows:
        print(f"  {r['bucket']:<9} gap {r['gap_id']:4d} | mmsi {r['mmsi']} | {r['gap_min']} min | "
              f"sin {r['sinuosity']} | GCD {r['gcd_med']} NM  FQ {r['fq_med']} NM  RFF {r['rff_med']} NM "
              f"| FQ/GCD {r['ratio_fq_gcd']}", flush=True)
    print("all outputs finite (asserted); medians reproduce per_gap.csv (asserted).", flush=True)


if __name__ == "__main__":
    main()

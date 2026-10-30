"""
task_alpha.experiment_compare  --  fast A/B: FQ model vs a generic MLP vs GCD, in fp32 + fp64.

Trains on a small cached SUBSAMPLE (fast), 3-way vessel-disjoint split, early-stops on val, then scores
the held-out TEST set. The MLP is a fair generic baseline: it sees the SAME anchor information (anchor
positions, time gaps, kinematics) via a DeepSets MLP + mean-pool -- NO FQ structure -- so the gap to the
FQ model isolates what the FQ primitives buy. GCD = linear interpolation between bracketing anchors.

Run:  python -c "from task_alpha.experiment_compare import run_compare; run_compare()"
"""

from __future__ import annotations

import copy
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from .data import collate_graphs_interp, haversine_nm
from .data_raw import build_dataset_raw
from .fq_interp import FQTrajectoryInterpolator
from .metrics import gcd_errors, model_errors
from .train_stream import _split


class MLPImputer(nn.Module):
    """Generic DeepSets-MLP baseline (no FQ): per-anchor MLP on [anchor_pos(2), dt(2), kinematics(3)],
    mean-pool over each held node's anchors, then a head MLP on [held_coord(2), pooled] -> position(2)."""

    def __init__(self, in_feat: int = 6, width: int = 128):
        super().__init__()
        self.use_cuda = False                                # (ignored; for a uniform train loop)
        self.anc = nn.Sequential(nn.Linear(7, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.head = nn.Sequential(nn.Linear(2 + width, width), nn.GELU(), nn.Linear(width, width),
                                  nn.GELU(), nn.Linear(width, 2))

    def forward(self, x, coord, edge_index, anchor_edge_index, edge_dist=None, extensive=False):
        asrc, adst = anchor_edge_index[0], anchor_edge_index[1]
        N = coord.shape[0]
        dt = coord[adst] - coord[asrc]                        # [A,2] time gap anchor->held
        af = torch.cat([x[asrc, :2], dt, x[asrc, 2:5]], dim=-1)   # anchor pos + dt + (sog,sin,cos) = 7
        e = self.anc(af)                                      # [A,width]
        agg = torch.zeros(N, e.shape[1], device=e.device, dtype=e.dtype).index_add(0, adst, e)
        cnt = torch.zeros(N, 1, device=e.device, dtype=e.dtype).index_add(
            0, adst, torch.ones(adst.shape[0], 1, device=e.device, dtype=e.dtype))
        agg = agg / cnt.clamp_min(1.0)
        pred = self.head(torch.cat([coord, agg], dim=-1))     # [N,2]
        return torch.where((cnt > 0), pred, x[:, :2])         # held -> MLP; else identity


def _evaluate(model, graphs, cast, device, batch_size):
    model.eval(); errs, gcd = [], []
    with torch.no_grad():
        for bs in range(0, len(graphs), batch_size):
            b = graphs[bs:bs + batch_size]
            x, coord, yp, mask, ei, ed, aei = collate_graphs_interp(b)
            x, coord = cast(x.to(device)), cast(coord.to(device)); ei, aei = ei.to(device), aei.to(device)
            pred = model(x, coord, ei, aei).float(); off = 0
            for g in b:
                ng = g.x.shape[0]
                errs.append(model_errors(pred[off:off + ng], g)); gcd.append(gcd_errors(g)); off += ng
    errs = np.concatenate(errs); gcd = np.concatenate(gcd)
    return errs, gcd


def _train_eval(build, is_fq, dtype, train, val, test, epochs, patience, batch_size, lr, device, seed):
    torch.manual_seed(seed)
    model = build().to(device)
    model.use_cuda = is_fq
    if dtype == "fp64":
        model = model.double()
    if is_fq:
        keys = ("grassmann.log_scale", "s_raw", "rho")
        sp = [p for nm, p in model.named_parameters() if any(k in nm for k in keys)]
        spid = {id(p) for p in sp}
        op = [p for p in model.parameters() if id(p) not in spid]
        opt = torch.optim.Adam([{"params": op, "lr": lr}, {"params": sp, "lr": 8e-5}])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    cast = (lambda t: t.double()) if dtype == "fp64" else (lambda t: t)
    best, best_state, pctr = float("inf"), None, 0
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train(); order = list(range(len(train))); random.Random(seed + epoch).shuffle(order)
        for bs in range(0, len(order), batch_size):
            b = [train[i] for i in order[bs:bs + batch_size]]
            x, coord, yp, mask, ei, ed, aei = collate_graphs_interp(b)
            x, coord, yp = cast(x.to(device)), cast(coord.to(device)), cast(yp.to(device))
            ei, aei, mask = ei.to(device), aei.to(device), mask.to(device)
            pred = model(x, coord, ei, aei)
            loss = ((pred[mask] - yp[mask]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        ve, _ = _evaluate(model, val, cast, device, batch_size)
        vade = float(np.nanmean(ve))
        if vade < best - 1e-4:
            best, best_state, pctr = vade, copy.deepcopy(model.state_dict()), 0
        else:
            pctr += 1
            if pctr >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    sec = time.perf_counter() - t0
    errs, gcd = _evaluate(model, test, cast, device, batch_size)
    return dict(ade=float(np.nanmean(errs)), med=float(np.nanmedian(errs)),
                gcd=float(np.nanmean(gcd)), gcd_med=float(np.nanmedian(gcd)),
                ratio=float(np.nanmean(errs)) / max(float(np.nanmean(gcd)), 1e-9),
                ratio_med=float(np.nanmedian(errs)) / max(float(np.nanmedian(gcd)), 1e-9),
                epochs=epoch + 1, sec=sec)


def run_compare(years=(2016, 2017), months=(7, 8), epochs=40, patience=6, batch_size=128,
                n_freqs=48, width=96, mlp_width=128, lr=1e-3, device="cuda", seed=0,
                gap_minutes=240.0, max_gap_len=400, min_span_factor=1.5, dtypes=("fp32", "fp64")):
    print(f"loading CLEAN subsample years={years} months={months} gap={gap_minutes}min ...", flush=True)
    gs = build_dataset_raw(years=years, months=months, n_mmsi=None, max_files_per_part=None,
                           gap_minutes=gap_minutes, thin_sec=180, max_gap_len=max_gap_len, n_anchor=8,
                           min_span_factor=min_span_factor, clean_only=True, seed=seed)
    gs = [g for g in gs if int(g.mask.sum()) > 0 and _split(g.mmsi) is not None]
    train = [g for g in gs if _split(g.mmsi) == "train"]
    val = [g for g in gs if _split(g.mmsi) == "val"]
    test = [g for g in gs if _split(g.mmsi) == "test"]
    print(f"subsample: {len(train)} train / {len(val)} val / {len(test)} test graphs (vessel-disjoint)\n",
          flush=True)
    fq = lambda: FQTrajectoryInterpolator(n=6, n_freqs=n_freqs, width=width, init_scale=0.5, seed=seed,
                                          in_feat=6)
    mlp = lambda: MLPImputer(width=mlp_width)
    res = {}
    for name, build, is_fq in [("FQ", fq, True), ("MLP", mlp, False)]:
        for dtype in dtypes:
            r = _train_eval(build, is_fq, dtype, train, val, test, epochs, patience, batch_size, lr,
                            device, seed)
            res[f"{name}-{dtype}"] = r
            print(f"  {name}-{dtype}: test ADE {r['ade']:.3f} (med {r['med']:.3f}) | ratio_mean "
                  f"{r['ratio']:.3f} ratio_median {r['ratio_med']:.3f} | {r['epochs']} ep, {r['sec']:.0f}s",
                  flush=True)
    g = next(iter(res.values()))
    print("\n================ FQ vs MLP vs GCD (held-out TEST) ================")
    print(f"GCD baseline: ADE {g['gcd']:.3f} NM (median {g['gcd_med']:.3f})   [ratio 1.000]")
    print(f"{'model':10s} {'test ADE':>9} {'median':>8} {'ratio(mean)':>12} {'ratio(med)':>11} {'time':>7}")
    for k, r in res.items():
        print(f"{k:10s} {r['ade']:9.3f} {r['med']:8.3f} {r['ratio']:12.3f} {r['ratio_med']:11.3f} {r['sec']:6.0f}s")
    return res


def _sinuosity(g):
    """Geometric curvature of the TRUE path across the gap span: (true path length / anchor chord) - 1.
    Model-INDEPENDENT and GCD-prediction-independent (uses only the true lat/lon). 0 = straight; >0 = detour."""
    m = g.mask.numpy(); tr = g.true_lonlat.numpy()
    held = np.where(m)[0]
    if len(held) == 0:
        return np.nan
    aL = max(int(held[0]) - 1, 0); aR = min(int(held[-1]) + 1, len(m) - 1)
    lat = tr[aL:aR + 1, 0]; lon = tr[aL:aR + 1, 1]
    if len(lat) < 2:
        return np.nan
    seg = haversine_nm(lat[:-1], lon[:-1], lat[1:], lon[1:])
    path = float(np.nansum(seg))
    chord = float(haversine_nm(np.array([lat[0]]), np.array([lon[0]]),
                               np.array([lat[-1]]), np.array([lon[-1]]))[0])
    if chord < 1e-6:
        return np.nan
    return path / chord - 1.0


def curvature_buckets(model, graphs, device, batch_size=128, cast=lambda t: t,
                      edges=(0.0, 0.02, 0.05, 0.10, 0.25, 1e9), out_csv=None):
    """THE headline analysis: per-gap FQ vs GCD median error, stratified by TRUE-path curvature (sinuosity).
    Thesis = GCD error grows with curvature (it predicts the chord); FQ's grows slower, so FQ wins the
    high-curvature bucket. Non-circular: curvature is a property of the true path, not of GCD's error."""
    model.eval()
    rows = []  # (sinuosity, fq_med, gcd_med, gap_min)
    with torch.no_grad():
        for s in range(0, len(graphs), batch_size):
            b = graphs[s:s + batch_size]
            x, coord, yp, mask, ei, ed, aei = collate_graphs_interp(b)
            x, coord = cast(x.to(device)), cast(coord.to(device)); ei, aei = ei.to(device), aei.to(device)
            pred = model(x, coord, ei, aei).float(); off = 0
            for g in b:
                ng = g.x.shape[0]
                fe = model_errors(pred[off:off + ng], g); ge = gcd_errors(g); off += ng
                if len(fe):
                    rows.append((_sinuosity(g), float(np.median(fe)), float(np.median(ge)),
                                 float(getattr(g, "gap_minutes", np.nan))))
    rows = [r for r in rows if np.isfinite(r[0])]
    sin = np.array([r[0] for r in rows]); fq = np.array([r[1] for r in rows]); gc = np.array([r[2] for r in rows])
    print(f"\nCURVATURE BUCKETS (sinuosity = true-path/chord - 1), {len(rows)} gaps:")
    print(f"  {'bucket':<13}{'n':>6}{'GCDmed':>9}{'FQmed':>9}{'ratio':>7}{'FQ<GCD':>8}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (sin >= lo) & (sin < hi)
        if sel.sum() == 0:
            continue
        gm = float(np.median(gc[sel])); fm = float(np.median(fq[sel])); win = 100 * float(np.mean(fq[sel] < gc[sel]))
        lab = f"{lo:.2f}-{hi:.2f}" if hi < 1e9 else f">{lo:.2f}"
        print(f"  {lab:<13}{int(sel.sum()):>6}{gm:>9.3f}{fm:>9.3f}{fm/max(gm,1e-9):>7.3f}{win:>7.0f}%")
    if out_csv:
        import csv
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["sinuosity", "fq_med_nm", "gcd_med_nm", "gap_min"])
            w.writerows(rows)
        print(f"  per-gap rows -> {out_csv}")
    return rows


def _train_fq_val(hp, train, val, epochs, patience, device, seed, return_model=False):
    """Train a FQ model (CUDA kernels, fp32) with HP `hp`, early-stop on val MEDIAN ratio, return best
    val metrics (and optionally the model). Median ratio = model_median/gcd_median -- the honest signal."""
    torch.manual_seed(seed)
    m = FQTrajectoryInterpolator(n=hp["n"], n_freqs=hp["n_freqs"], width=hp["width"],
                                 init_scale=hp["init_scale"], seed=seed, in_feat=6).to(device)
    m.use_cuda = True
    keys = ("grassmann.log_scale", "s_raw", "rho")
    sp = [p for nm, p in m.named_parameters() if any(k in nm for k in keys)]
    spid = {id(p) for p in sp}; op = [p for p in m.parameters() if id(p) not in spid]
    opt = torch.optim.Adam([{"params": op, "lr": hp["lr"]}, {"params": sp, "lr": hp["fq_scalar_lr"]}])
    bs, mgn = hp["batch_size"], hp["max_grad_norm"]
    best, best_state, pctr = float("inf"), None, 0
    for epoch in range(epochs):
        m.train(); order = list(range(len(train))); random.Random(seed + epoch).shuffle(order)
        for i in range(0, len(order), bs):
            b = [train[j] for j in order[i:i + bs]]
            x, coord, yp, mask, ei, ed, aei = [t.to(device) for t in collate_graphs_interp(b)]
            pred = m(x, coord, ei, aei); loss = ((pred[mask] - yp[mask]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(op, mgn); torch.nn.utils.clip_grad_norm_(sp, mgn)
                opt.step()
        errs, gcd = _evaluate(m, val, lambda t: t, device, bs)
        rm = float(np.nanmedian(errs)) / max(float(np.nanmedian(gcd)), 1e-9)
        if rm < best - 1e-4:
            best, best_state, pctr = rm, copy.deepcopy(m.state_dict()), 0
        else:
            pctr += 1
            if pctr >= patience:
                break
    if best_state is not None:
        m.load_state_dict(best_state)
    errs, gcd = _evaluate(m, val, lambda t: t, device, bs)
    r = dict(ade=float(np.nanmean(errs)), med=float(np.nanmedian(errs)), gcd=float(np.nanmean(gcd)),
             gcd_med=float(np.nanmedian(gcd)),
             ratio_mean=float(np.nanmean(errs)) / max(float(np.nanmean(gcd)), 1e-9),
             ratio_med=float(np.nanmedian(errs)) / max(float(np.nanmedian(gcd)), 1e-9), epochs=epoch + 1)
    return (r, m) if return_model else r


def hp_search(n_trials=25, hp_years=(2016, 2017), hp_months=(7, 8), gap_minutes=360, max_gap_len=500,
              min_span_factor=1.5, epochs=35, patience=6, device="cuda", seed=0,
              out_dir="task_alpha/runs/hp_6h", final_all_data=True):
    """Optuna HP search for FQ on CLEAN long-horizon data. Objective = val MEDIAN ratio (model/GCD,
    minimize). Saves trials.csv (params + per-trial performances) + best_config.json. Best config then
    trains on the FULL all-data horizon and is scored ONCE on the held-out TEST set."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    os.makedirs(out_dir, exist_ok=True)
    print(f"HP search: clean {gap_minutes//60}h | search-subsample years={hp_years} months={hp_months}",
          flush=True)
    gs = build_dataset_raw(years=hp_years, months=hp_months, n_mmsi=None, max_files_per_part=None,
                           gap_minutes=gap_minutes, max_gap_len=max_gap_len, n_anchor=8,
                           min_span_factor=min_span_factor, clean_only=True, seed=seed)
    gs = [g for g in gs if int(g.mask.sum()) > 0 and _split(g.mmsi) is not None]
    train = [g for g in gs if _split(g.mmsi) == "train"]
    val = [g for g in gs if _split(g.mmsi) == "val"]
    print(f"HP data: {len(train)} train / {len(val)} val graphs", flush=True)

    def objective(trial):
        hp = dict(lr=trial.suggest_float("lr", 3e-4, 5e-3, log=True),
                  fq_scalar_lr=trial.suggest_float("fq_scalar_lr", 1e-5, 1e-3, log=True),
                  width=trial.suggest_categorical("width", [64, 96, 128]),
                  n_freqs=trial.suggest_categorical("n_freqs", [24, 48, 96]),
                  n=6,                                        # not searched: the Op-B CUDA kernel is a per-thread n=6 solver
                  init_scale=trial.suggest_float("init_scale", 0.3, 3.0),
                  max_grad_norm=trial.suggest_float("max_grad_norm", 0.3, 3.0),
                  batch_size=trial.suggest_categorical("batch_size", [64, 128]))
        try:
            r = _train_fq_val(hp, train, val, epochs, patience, device, seed)
        except Exception as e:
            print(f"  trial {trial.number} FAILED: {str(e)[:100]}", flush=True); return float("inf")
        for k, v in r.items():
            trial.set_user_attr(k, v)
        print(f"  trial {trial.number:2d}: ratio_med {r['ratio_med']:.3f} (val med {r['med']:.3f} / gcd "
              f"{r['gcd_med']:.3f}) ratio_mean {r['ratio_mean']:.3f} | {r['epochs']}ep lr{hp['lr']:.1e} "
              f"W{hp['width']} M{hp['n_freqs']}", flush=True)
        return r["ratio_med"]

    storage = f"sqlite:///{os.path.join(out_dir, 'study.db').replace(chr(92), '/')}"
    study = optuna.create_study(direction="minimize", study_name="hp_6h", storage=storage,
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=seed))
    def _save_cb(st, tr):                                     # incremental CSV after every trial (crash-safe)
        st.trials_dataframe().to_csv(os.path.join(out_dir, "trials.csv"), index=False)
    done = len([t for t in study.trials if t.state.is_finished()])
    if done:
        print(f"RESUMED study: {done} trials already done", flush=True)
    study.optimize(objective, n_trials=max(0, n_trials - done), show_progress_bar=False, callbacks=[_save_cb])
    study.trials_dataframe().to_csv(os.path.join(out_dir, "trials.csv"), index=False)
    best = study.best_params
    json.dump({"best_params": best, "best_val_ratio_median": study.best_value,
               "best_val_attrs": study.best_trial.user_attrs},
              open(os.path.join(out_dir, "best_config.json"), "w"), indent=2)
    print(f"\nBEST val ratio_median {study.best_value:.3f} | params {best}", flush=True)

    if final_all_data:
        print("\nFINAL: best config trained on FULL all-data + scored on held-out TEST ...", flush=True)
        allg = build_dataset_raw(years=tuple(range(2012, 2022)), months=tuple(range(1, 13)),
                                 n_mmsi=None, max_files_per_part=None, gap_minutes=gap_minutes,
                                 max_gap_len=max_gap_len, min_span_factor=min_span_factor, n_anchor=8,
                                 clean_only=True, seed=seed)
        allg = [g for g in allg if int(g.mask.sum()) > 0 and _split(g.mmsi) is not None]
        tr = [g for g in allg if _split(g.mmsi) == "train"]
        va = [g for g in allg if _split(g.mmsi) == "val"]
        te = [g for g in allg if _split(g.mmsi) == "test"]
        hp = dict(best); hp.setdefault("batch_size", 128); hp["n"] = 6
        r, model = _train_fq_val(hp, tr, va, epochs=60, patience=10, device=device, seed=seed,
                                 return_model=True)
        errs, gcd = _evaluate(model, te, lambda t: t, device, 128)
        test = dict(model_ade=float(np.nanmean(errs)), model_med=float(np.nanmedian(errs)),
                    gcd_ade=float(np.nanmean(gcd)), gcd_med=float(np.nanmedian(gcd)),
                    ratio_mean=float(np.nanmean(errs)) / max(float(np.nanmean(gcd)), 1e-9),
                    ratio_med=float(np.nanmedian(errs)) / max(float(np.nanmedian(gcd)), 1e-9),
                    n_test=int(len(errs)), n_train=len(tr))
        json.dump(test, open(os.path.join(out_dir, "final_test.json"), "w"), indent=2)
        torch.save({"state_dict": model.state_dict(), "hp": hp, "test": test},
                   os.path.join(out_dir, "final_model.pt"))
        print(f"FINAL TEST ({len(te)} graphs, all-data {gap_minutes//60}h): FQ med {test['model_med']:.3f} "
              f"vs GCD med {test['gcd_med']:.3f} -> ratio_med {test['ratio_med']:.3f} | "
              f"ratio_mean {test['ratio_mean']:.3f}", flush=True)
        curvature_buckets(model, te, device, batch_size=128,
                          out_csv=os.path.join(out_dir, "curvature_test.csv"))
    return study


if __name__ == "__main__":
    run_compare()

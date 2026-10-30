"""INIT-TABLE vs TRAINED-TABLE comparison for the AKKV amplitude table.

Zero-shot evaluation of frozen checkpoints with the INIT-state AKKV amplitude table (the table
of the untrained (L,U)) served in place of the exact trained-(L,U) table, to measure how much
the trained table contributes to the test error:
  * env set in-process (nothing inherited from the shell): FQ_AMP_MODE=akkv_torus,
    FQ_AMP_NMC=262144, FQ_AMP_REFRESH removed (exact per-content cache);
  * the INIT table is captured from the freshly constructed model BEFORE load_state_dict
    (exact same QMC seed/nodes as the trained table -- only (L,U) differ);
  * the init condition is forced by writing the cache as (L_trained, U_trained, c_init), so the
    cache HITS on the trained keys and serves the init amplitudes -- bit-exact, no refresh
    counter, asserted unchanged after every eval;
  * frozen checkpoints: runs/fqnative_conformant_5seed/seed*/best.pt (oph production) and
    runs/fqnative_opm_5seed/seed*/best.pt (opm; gamma via env as in that run);
  * evaluated on the fixed 200-gap subsample AND the full 774-gap 6h test cell, with
    sinuosity-bucket splits and per-gap CSVs.

Run: python -m task_alpha._diag_init_table --probe           (match the eval env of the opm run, seed0)
     python -m task_alpha._diag_init_table --family opm      (all seeds, opm)
     python -m task_alpha._diag_init_table --family conformant
Out: task_alpha/runs/conform_baseline/init_table_anomaly/
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import warnings

warnings.filterwarnings("ignore")

# ---- in-process env (read at model construction / amplitudes() time) ----
os.environ["FQ_AMP_MODE"] = "akkv_torus"
os.environ["FQ_AMP_NMC"] = "262144"
os.environ.pop("FQ_AMP_REFRESH", None)          # exact per-content cache (no periodic refresh)
os.environ.pop("FQ_ENUM_RINGS", None)           # default combo cut, as in the checkpoints (asserted below)

import numpy as np
import torch

from .data import collate_graphs_interp
from .experiment_compare import _sinuosity
from .meltin_proof import _bvec
from .metrics import gcd_errors, model_errors

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "task_alpha/runs/conform_baseline/init_table_anomaly"
CKS = {
    "conformant": "task_alpha/runs/fqnative_conformant_5seed/seed{seed}/best.pt",
    "opm": "task_alpha/runs/fqnative_opm_5seed/seed{seed}/best.pt",
    # the _conform_subsample harness checkpoint (zero-shot opm readout on top)
    "sobolev": "task_alpha/runs/fqnative_sobolev_5seed/seed{seed}/best.pt",
}
BUCKETS = list(zip((0.0, 0.02, 0.05, 0.10, 0.25), (0.02, 0.05, 0.10, 0.25, 1e9)))


def _gm(b):
    return torch.tensor([float(g.gap_minutes) for g in b], device=DEV)


def _build(family: str, seed: int):
    """Construct the production model for the family (env already set), warmup-forward to
    materialize lazies, and capture the INIT-state AKKV table BEFORE loading any weights."""
    if family in ("opm", "sobolev"):
        os.environ["FQ_READOUT_MODE"] = "opm"
        os.environ["FQ_OPM_LAM"] = "head" if family == "opm" else "off"
        os.environ.setdefault("FQ_OPM_TIER", "C")
        os.environ["FQ_OPM_BATCHED"] = "1"
    else:
        os.environ["FQ_READOUT_MODE"] = "oph"
        for k in ("FQ_OPM_LAM", "FQ_OPM_TIER", "FQ_OPM_BATCHED", "FQ_OPM_GAMMA"):
            os.environ.pop(k, None)
    from .fqdeep_native import FQDeepNative     # after env: readout mode read at construction
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = FQDeepNative.production(seed=seed).to(DEV)
    m.use_cuda = True
    with torch.no_grad():
        m.s_raw.fill_(math.log(math.expm1(2.5)))
    return m


def _tables(m, ck_path):
    """(c_init, c_trained, cache_key) -- init table captured pre-load; combos verified."""
    with torch.no_grad():
        m.quasicrystal()                        # fills the AKKV cache at INIT (L0, U0)
    Lk0, Uk0, c_init = m.spectrum._akkv_cache
    c_init = c_init.clone()
    sd = torch.load(ck_path, map_location=DEV)["state_dict"]
    assert torch.equal(m.spectrum.combos.cpu(), sd["spectrum.combos"].cpu()), \
        "combo enumeration mismatch (FQ_ENUM_RINGS?) -- table would be on the wrong modes"
    m.load_state_dict(sd)
    m.eval()
    with torch.no_grad():
        m.quasicrystal()                        # cache miss -> recompute at TRAINED (L, U)
    Lkt, Ukt, c_tr = m.spectrum._akkv_cache
    assert not torch.equal(c_init, c_tr), "init and trained AKKV tables identical?!"
    return c_init, c_tr.clone(), (Lkt.clone(), Ukt.clone())


def _force_table(m, key, c128):
    """Serve `c128` under the trained (L,U) cache key (forces the chosen table, bit-exact)."""
    m.spectrum._akkv_cache = (key[0], key[1], c128.clone())


def _eval(m, graphs, bs=64):
    """Per-gap medians (model, gcd, sinuosity) -- verbatim runner evaluation."""
    m.eval()
    fq, gc, sn = [], [], []
    with torch.no_grad():
        for s0 in range(0, len(graphs), bs):
            b = graphs[s0:s0 + bs]
            x, coord, yp, mask, ei, ed, aei = collate_graphs_interp(b)
            pred = m(x.to(DEV), coord.to(DEV), ei.to(DEV), aei.to(DEV), _bvec(b),
                     gap_minutes=_gm(b)).float().cpu()
            off = 0
            for g in b:
                ng = g.x.shape[0]
                gp = pred[off:off + ng]
                off += ng
                fe = model_errors(gp, g)
                if len(fe):
                    fq.append(float(np.median(fe)))
                    gc.append(float(np.median(gcd_errors(g))))
                    sn.append(float(_sinuosity(g)))
    return np.array(fq), np.array(gc), np.array(sn)


def _ratio(fq, gc):
    return float(np.median(fq) / np.median(gc))


def _bucket_rows(fq, gc, sn):
    keep = np.isfinite(sn)
    fqk, gck, snk = fq[keep], gc[keep], sn[keep]
    rows = [("OVERALL", len(fqk), _ratio(fqk, gck), 100 * float(np.mean(fqk < gck)))]
    for lo, hi in BUCKETS:
        sel = (snk >= lo) & (snk < hi)
        if sel.sum() < 3:
            continue
        lab = f"{lo:.2f}-{hi:.2f}" if hi < 1e9 else f">{lo:.2f}"
        rows.append((lab, int(sel.sum()), _ratio(fqk[sel], gck[sel]),
                     100 * float(np.mean(fqk[sel] < gck[sel]))))
    return rows


def _log(tag, row):
    os.makedirs(OUT, exist_ok=True)
    p = f"{OUT}/log.csv"
    new = not os.path.exists(p)
    with open(p, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["tag", "family", "seed", "gamma", "table", "cell", "n", "ratio",
                        "win_pct"])
        w.writerow([tag] + row)


def load_cells():
    with open("task_alpha/data/our_processed/gap_6h/test.pkl", "rb") as f:
        te = pickle.load(f)
    ids = json.load(open("task_alpha/runs/conform_baseline/subsample_ids.json"))
    return te, [te[i] for i in ids]


def probe(seed=0):
    """Match the eval env of the opm run: compare trained-table full-cell per-gap medians against
    runs/fqnative_opm_5seed/seed0/per_gap.csv under candidate gammas."""
    te, _ = load_cells()
    ref = list(csv.DictReader(open(
        f"task_alpha/runs/fqnative_opm_5seed/seed{seed}/per_gap.csv")))
    ref_med = np.array([float(r["model_med"]) for r in ref])
    m = _build("opm", seed)
    c_init, c_tr, key = _tables(m, CKS["opm"].format(seed=seed))
    for gam in ("0.5", None, "hat", "0.0"):
        if gam is None:
            m.opm_gamma_env = None
        else:
            m.opm_gamma_env = "hat" if gam == "hat" else float(gam)
        _force_table(m, key, c_tr)
        fq, gc, sn = _eval(m, te)
        d = np.abs(fq - ref_med)
        print(f"[init-table probe] gamma={gam}: ratio {_ratio(fq, gc):.4f} | vs recorded per_gap: "
              f"max|dmed| {d.max():.4f} med|dmed| {np.median(d):.5f} "
              f"(match if ~rounding 5e-4)", flush=True)


def run_family(family: str, seeds, gammas):
    te, sub = load_cells()
    os.makedirs(OUT, exist_ok=True)
    for seed in seeds:
        m = _build(family, seed)
        ck = CKS[family].format(seed=seed)
        c_init, c_tr, key = _tables(m, ck)
        # table drift stats
        d_abs = (c_tr.abs() - c_init.abs())
        print(f"[init-table {family} s{seed}] table drift: max||c_tr|-|c_init|| "
              f"{float(d_abs.abs().max()):.3e} | rel L2 "
              f"{float((c_tr - c_init).abs().pow(2).sum().sqrt() / c_init.abs().pow(2).sum().sqrt()):.4f}",
              flush=True)
        torch.save({"c_init": c_init, "c_trained": c_tr},
                   f"{OUT}/tables_{family}_seed{seed}.pt")
        for gam in gammas:
            if family in ("opm", "sobolev"):
                m.opm_gamma_env = None if gam is None else \
                    ("hat" if gam == "hat" else float(gam))
                gtag = "none" if gam is None else str(gam)
            else:
                gtag = "na"
            for tab_name, tab in (("trained", c_tr), ("init", c_init)):
                _force_table(m, key, tab)
                res = {}
                for cell_name, cell in (("sub200", sub), ("full", te)):
                    fq, gc, sn = _eval(m, cell)
                    # cache must not have been recomputed mid-eval
                    assert torch.equal(m.spectrum._akkv_cache[2], tab), "cache clobbered"
                    r = _ratio(fq, gc)
                    win = 100 * float(np.mean(fq < gc))
                    res[cell_name] = (fq, gc, sn, r, win)
                    _log("init_table", [family, seed, gtag, tab_name, cell_name,
                                           len(fq), round(r, 4), round(win, 1)])
                # per-gap csv + buckets on the full cell
                fq, gc, sn, r, win = res["full"]
                with open(f"{OUT}/pergap_{family}_s{seed}_g{gtag}_{tab_name}.csv", "w",
                          newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["idx", "model_med", "gcd_med", "sinuosity"])
                    for i, (a_, b_, c_) in enumerate(zip(fq, gc, sn)):
                        w.writerow([i, round(a_, 4), round(b_, 4), round(c_, 4)])
                bk = _bucket_rows(fq, gc, sn)
                print(f"[init-table {family} s{seed} gamma={gtag} {tab_name}] "
                      f"sub200 {res['sub200'][3]:.4f} | full {r:.4f} win {win:.1f}% | " +
                      " ".join(f"{lab}:{rr:.3f}" for lab, _n, rr, _w in bk[1:]), flush=True)
        del m
        torch.cuda.empty_cache()


def decompose(family: str, seed: int, gam):
    """WHICH CHANNEL carries the table effect: eta head vs learned metric w_prod.
    Four ablation states x two tables, full cell.  w_const=True freezes the metric to the
    c-independent |omega|^-2 backbone; eta0 zeroes the eta head.  Both -> the closed form:
    the table can then act on nothing (control row, must show ~0 effect)."""
    te, _ = load_cells()
    m = _build(family, seed)
    c_init, c_tr, key = _tables(m, CKS[family].format(seed=seed))
    if family in ("opm", "sobolev"):
        m.opm_gamma_env = None if gam is None else ("hat" if gam == "hat" else float(gam))
    gtag = "na" if family == "conformant" else ("none" if gam is None else str(gam))
    wsave = m.eta_head[-1].weight.detach().clone()
    bsave = m.eta_head[-1].bias.detach().clone()
    for state in ("full", "w_const", "eta0", "both"):
        m.oph_w_const = state in ("w_const", "both")
        with torch.no_grad():
            if state in ("eta0", "both"):
                m.eta_head[-1].weight.zero_(); m.eta_head[-1].bias.zero_()
            else:
                m.eta_head[-1].weight.copy_(wsave); m.eta_head[-1].bias.copy_(bsave)
        rr = {}
        for tab_name, tab in (("trained", c_tr), ("init", c_init)):
            _force_table(m, key, tab)
            fq, gc, sn = _eval(m, te)
            assert torch.equal(m.spectrum._akkv_cache[2], tab)
            rr[tab_name] = _ratio(fq, gc)
            _log("init_table_decompose", [family, seed, f"{gtag}/{state}", tab_name, "full",
                                  len(fq), round(rr[tab_name], 4),
                                  round(100 * float(np.mean(fq < gc)), 1)])
        print(f"[init-table decompose {family} s{seed} gamma={gtag} {state}] trained "
              f"{rr['trained']:.4f} | init {rr['init']:.4f} | d {rr['init'] - rr['trained']:+.4f}",
              flush=True)
    m.oph_w_const = False
    with torch.no_grad():
        m.eta_head[-1].weight.copy_(wsave); m.eta_head[-1].bias.copy_(bsave)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--decompose", action="store_true")
    ap.add_argument("--family", choices=("opm", "conformant", "sobolev"))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--gammas", default=None,
                    help="opm only; comma list from {0.5,none,hat,0.0}")
    a = ap.parse_args()
    if a.probe:
        probe()
        return
    seeds = [int(s) for s in a.seeds.split(",")]
    if a.decompose:
        for s in seeds:
            g = (a.gammas or ("hat" if a.family != "conformant" else "none")).split(",")[0]
            decompose(a.family, s, None if g == "none" else g)
        return
    if a.family in ("opm", "sobolev"):
        gam_raw = (a.gammas or "0.5,none").split(",")
        gammas = [None if g == "none" else g for g in gam_raw]
    else:
        gammas = [None]
    run_family(a.family, seeds, gammas)


if __name__ == "__main__":
    main()

"""TRUST-GATE PROBE: MMSI-clustered OOF linear probe of the trust bit
T = 1[err(gamma_bar) < err(0)] on the per-gap pooled embedding h_bar, stake weight
|Delta err|/GCD, on the ACTIVE gaps of the full 6h cell (rel_spread >= 0.05 * GCD).

Pass threshold: stake-weighted OOF AUC > 0.60.  Below it, whether gamma_bar helps a given gap
is not predictable from h_bar, so the embedding cannot drive a per-gap trust gate on gamma
(the printed verdict states which case holds).

Inputs: task_alpha/runs/diag_d1/per_gap_full.csv (the fixed-gamma grid written by
_diag_d1_gamma_oracle --scope full; err_g085 = the cell scale gamma_bar = 0.85, err_g000 =
gamma 0) + h_bar extracted here from the frozen Op-M seed-0 checkpoint (pool = h[anc].mean(0),
the exact forward-pass pooling).
Out: runs/amd/d1a4_probe.json.  Measurement only.

Run:  python -m task_alpha._amd_d1a4_probe
"""
from __future__ import annotations

import csv
import json
import os
import pickle

import numpy as np
import torch

from ._diag_d1_gamma_oracle import DEV, build_model, _gm  # env set at import (opm, clean amp)
from .data import collate_graphs_interp

OUT = "task_alpha/runs/amd"
PG = "task_alpha/runs/diag_d1/per_gap_full.csv"
GAMMA_BAR_COL = "err_g085"          # the cell scale (aggregate argmin on this checkpoint)


def extract_pools(m, graphs, bs=64):
    """Per-gap pooled anchor embedding h_bar, exactly the forward pass's pool."""
    pools = np.full((len(graphs), None or 1), np.nan)  # placeholder replaced on first batch
    out = [None] * len(graphs)
    with torch.no_grad():
        for s in range(0, len(graphs), bs):
            b = graphs[s:s + bs]
            x, coord, yp, mask, ei, ed, aei = collate_graphs_interp(b)
            x, coord, ei, aei = x.to(DEV), coord.to(DEV), ei.to(DEV), aei.to(DEV)
            batch = torch.cat([torch.full((g.x.shape[0],), i, dtype=torch.long)
                               for i, g in enumerate(b)]).to(DEV)
            h, xi, c, sc, P, xi0 = m.embed(x, coord, ei)
            asrc, adst = aei[0], aei[1]
            gid_adst = batch[adst]
            held_all = x[:, 5] > 0.5
            for i, g in enumerate(b):
                held = torch.where(held_all & (batch == i))[0]
                anc = torch.unique(asrc[gid_adst == i])
                if held.numel() == 0 or anc.numel() < 2:
                    continue
                out[s + i] = h[anc].mean(0).float().cpu().numpy()
    return out


def wauc(y, s, w):
    """Stake-weighted AUC: P_w(score_pos > score_neg) with weight w_i * w_j (ties 0.5)."""
    p, n = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    sp, sn = s[p][:, None], s[n][None, :]
    wp, wn = w[p][:, None], w[n][None, :]
    W = wp * wn
    return float((W * ((sp > sn) + 0.5 * (sp == sn))).sum() / W.sum())


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = list(csv.DictReader(open(PG)))
    with open("task_alpha/data/our_processed/gap_6h/test.pkl", "rb") as f:
        te_all = pickle.load(f)
    gids = [int(r["gap_id"]) for r in rows]
    graphs = [te_all[i] for i in gids]

    grid_cols = [c for c in rows[0] if c.startswith("err_g")]
    E = np.array([[float(r[c]) if r[c] != "" else np.nan for c in grid_cols] for r in rows])
    gcd = np.array([float(r["gcd_med"]) for r in rows])
    mmsi = np.array([int(r["mmsi"]) for r in rows])
    e0 = np.array([float(r["err_g000"]) if r["err_g000"] != "" else np.nan for r in rows])
    eg = np.array([float(r[GAMMA_BAR_COL]) if r[GAMMA_BAR_COL] != "" else np.nan for r in rows])

    valid = np.isfinite(E).all(1) & np.isfinite(gcd) & (gcd > 0)
    rel_spread = (np.nanmax(E, 1) - np.nanmin(E, 1)) / gcd
    active = valid & (rel_spread >= 0.05)
    T = (eg < e0).astype(int)
    stake = np.abs(eg - e0) / gcd
    print(f"[AMD_gammaprime_probe] {int(valid.sum())} valid gaps, {int(active.sum())} active | "
          f"P(T=1 | active) = {T[active].mean():.3f}", flush=True)

    m = build_model(graphs[:4])
    pools = extract_pools(m, graphs)
    have = np.array([p is not None for p in pools])
    use = active & have
    X = np.stack([pools[i] for i in np.flatnonzero(use)])
    y = T[use]; w = stake[use]; mm = mmsi[use]
    print(f"[AMD_gammaprime_probe] probe set n={len(y)} dim={X.shape[1]}", flush=True)

    # MMSI-clustered 5-fold OOF logistic probe (stake-weighted fit)
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    vecs = rng.permutation(np.unique(mm))
    fold_of = {v: i % 5 for i, v in enumerate(vecs)}
    folds = np.array([fold_of[v] for v in mm])
    oof = np.full(len(y), np.nan)
    for k in range(5):
        tr, tst = folds != k, folds == k
        if tst.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit((X[tr] - mu) / sd, y[tr], sample_weight=w[tr])
        oof[tst] = clf.predict_proba((X[tst] - mu) / sd)[:, 1]
    ok = np.isfinite(oof)
    auc_w = wauc(y[ok], oof[ok], w[ok])
    auc_u = wauc(y[ok], oof[ok], np.ones(ok.sum()))
    verdict = "PASS (gate licensed)" if auc_w > 0.60 else \
        "FAIL -- trust bit not predictable from h_bar; no per-gap trust gate on gamma"
    print(f"[AMD_gammaprime_probe] OOF AUC: stake-weighted {auc_w:.3f} | unweighted {auc_u:.3f} "
          f"| bar 0.60 -> {verdict}", flush=True)

    json.dump({"tag": "AMD_gammaprime_probe", "n_active": int(active.sum()),
               "n_probe": int(ok.sum()), "gamma_bar_col": GAMMA_BAR_COL,
               "auc_stake_weighted": auc_w, "auc_unweighted": auc_u,
               "bar": 0.60, "pass": bool(auc_w > 0.60)},
              open(f"{OUT}/d1a4_probe.json", "w"), indent=1)


if __name__ == "__main__":
    main()

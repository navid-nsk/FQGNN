"""Vanilla RFF-GNN baseline, built from scratch -- the generic-random-spectral control for the FQGNN.

WHAT IT IS (all textbook, deliberately NO FQ ideas):
  * RFF (Rahimi-Recht 2007): z(c) = sqrt(2/D) * cos(W c + b), with W ~ N(0, sigma^2 I) FROZEN random
    (never trained) and b ~ U[0, 2pi].  Applied to the SAME time coordinate `coord` the FQGNN lifts.
    D = 96 random frequencies, sigma = 5.0 (lengthscale 0.2 on the unit-normalized time coord).
  * Basic GNN: 3 GraphSAGE-mean message-passing layers (h_i' = ReLU(W_s h_i + W_n mean_j h_j) + h_i)
    over the SAME edges the FQGNN sees (temporal kNN edges + two-sided anchor edges).
  * Same node features x [N,6], same ABSOLUTE ENU target y_pos, same MSE-at-held loss, same training
    recipe as run_full_faithful_5seed.py (Adam 5e-4, clip 1.0, bs 64, <=60 epochs, patience 8), same
    val metric (median-NM ratio vs GCD).  The ONLY difference from the FQGNN is the operator family:
    random fixed Fourier basis + mean aggregation instead of the Lee-Yang FQ operators.

5 seeds (0..4): each seed re-draws the frozen RFF basis + re-inits the GNN + reshuffles batches.
Outputs (task_alpha/baselines/RFFGNN/runs_5seed_gaps/) -- IDENTICAL formats to the FQGNN run:
  seed<k>/epochs.csv        epoch, train_loss, val_ratio, val_model_med, val_gcd_med
  seed<k>/per_gap.csv       gap_id, mmsi, gap_min, sinuosity, gcd_med, model_med, win  (te order)
  seed<k>/test_buckets.csv  OVERALL + sinuosity buckets (identical GCD column)
  seed<k>/best.pt
  all_seeds_epochs.csv / all_seeds_test_buckets.csv / summary.csv (MEAN/STD)
Run (cwd = the repository root):
  python task_alpha/baselines/RFFGNN/run_rffgnn_5seed_gaps.py
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import random
import sys
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))          # repository root
os.chdir(ROOT); sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn as nn

from task_alpha.baselines import common_gaps as CG
from task_alpha.data import collate_graphs_interp
from task_alpha.metrics import gcd_errors, model_errors

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(HERE, "runs_5seed_gaps")
SEEDS = (0, 1, 2, 3, 4)
WIDTH, D_RFF, SIGMA, LAYERS = 96, 96, 5.0, 3        # width/depth match the FQGNN (96, 3 layers)
LR, BS, EPOCHS, PATIENCE = 5e-4, 64, 60, 8          # EXACT FQGNN training recipe


# ------------------------------------------------------------------------------------------------ #
#  MODEL: frozen random Fourier features + textbook SAGE-mean message passing                        #
# ------------------------------------------------------------------------------------------------ #
class RFFGNN(nn.Module):
    def __init__(self, in_feat=6, width=WIDTH, d_rff=D_RFF, sigma=SIGMA, layers=LAYERS, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        # vanilla Rahimi-Recht RFF: FROZEN random draw (buffers, never trained)
        self.register_buffer("W_rff", torch.randn(2, d_rff, generator=g) * sigma)
        self.register_buffer("b_rff", torch.rand(d_rff, generator=g) * 2 * math.pi)
        self.d_rff = d_rff
        self.enc_x = nn.Linear(in_feat, width)
        self.enc_r = nn.Linear(d_rff, width)
        self.self_lin = nn.ModuleList([nn.Linear(width, width) for _ in range(layers)])
        self.neigh_lin = nn.ModuleList([nn.Linear(width, width) for _ in range(layers)])
        self.head = nn.Linear(width, 2)

    def rff(self, coord):                                              # [N,2] -> [N,D]
        return math.sqrt(2.0 / self.d_rff) * torch.cos(coord @ self.W_rff + self.b_rff)

    def forward(self, x, coord, edge_index, anchor_edge_index):
        h = torch.relu(self.enc_x(x) + self.enc_r(self.rff(coord)))
        # one combined directed edge set: temporal kNN + two-sided anchor edges (same info as FQGNN)
        ei = torch.cat([edge_index, anchor_edge_index], dim=1) if anchor_edge_index.numel() else edge_index
        src, dst = ei[0], ei[1]
        ones = torch.ones(src.shape[0], device=h.device)
        deg = torch.zeros(h.shape[0], device=h.device).index_add_(0, dst, ones).clamp(min=1.0)
        for ls, ln in zip(self.self_lin, self.neigh_lin):
            agg = torch.zeros_like(h).index_add_(0, dst, h[src]) / deg.unsqueeze(1)   # mean aggregation
            h = torch.relu(ls(h) + ln(agg)) + h                                       # SAGE-mean + residual
        return self.head(h)                                            # ABSOLUTE ENU pos (y_pos units)


# ------------------------------------------------------------------------------------------------ #
#  EVAL: per-gap model/GCD medians (identical semantics to the FQGNN runner)                         #
# ------------------------------------------------------------------------------------------------ #
@torch.no_grad()
def eval_gaps(m, graphs, bs=64):
    m.eval(); fq, gc, errs = [], [], []
    for s0 in range(0, len(graphs), bs):
        b = graphs[s0:s0 + bs]
        x, co, yp, mk, ei, ed, ae = collate_graphs_interp(b)
        pred = m(x.to(DEV), co.to(DEV), ei.to(DEV), ae.to(DEV)).float().cpu(); off = 0
        for g in b:
            ng = g.x.shape[0]; gp = pred[off:off + ng]; off += ng
            fe = model_errors(gp, g)
            errs.append(fe)
            if len(fe):
                fq.append(float(np.median(fe))); gc.append(float(np.median(gcd_errors(g))))
    return np.array(fq), np.array(gc), errs


def run_seed(seed, tr, va, te, epochs, out_root):
    sdir = os.path.join(out_root, f"seed{seed}"); os.makedirs(sdir, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    m = RFFGNN(seed=seed).to(DEV)
    npar = sum(p.numel() for p in m.parameters() if p.requires_grad)
    nfroz = sum(b.numel() for b in m.buffers())
    print(f"[seed {seed}] RFF-GNN params {npar} (+{nfroz} FROZEN random RFF) | device {DEV}", flush=True)
    opt = torch.optim.Adam(m.parameters(), lr=LR)

    rows, best, best_state, pctr = [], 1e9, None, 0
    for ep in range(epochs):
        m.train(); order = list(range(len(tr))); random.Random(seed * 1000 + ep).shuffle(order)
        ep_loss, nb = 0.0, 0
        for i in range(0, len(order), BS):
            b = [tr[j] for j in order[i:i + BS]]
            x, co, yp, mk, ei, ed, ae = collate_graphs_interp(b)
            pred = m(x.to(DEV), co.to(DEV), ei.to(DEV), ae.to(DEV))
            L = ((pred[mk.to(DEV)] - yp.to(DEV)[mk.to(DEV)]) ** 2).mean()
            opt.zero_grad()
            if torch.isfinite(L):
                L.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
                ep_loss += float(L.detach()); nb += 1
        fq, gc, _ = eval_gaps(m, va)
        vm, vg = float(np.median(fq)), float(np.median(gc)); vr = vm / vg
        rows.append({"epoch": ep, "train_loss": round(ep_loss / max(nb, 1), 6),
                     "val_ratio": round(vr, 4), "val_model_med": round(vm, 4), "val_gcd_med": round(vg, 4)})
        CG.write_epochs(os.path.join(sdir, "epochs.csv"), rows)
        if vr < best - 1e-4:
            best, best_state, pctr = vr, copy.deepcopy(m.state_dict()), 0
            torch.save({"state_dict": best_state}, os.path.join(sdir, "best.pt"))
        else:
            pctr += 1
        print(f"[seed {seed}] ep {ep:2d}: train {ep_loss / max(nb, 1):.5f} | val {vr:.4f} "
              f"(best {best:.4f})", flush=True)
        if pctr >= PATIENCE:
            print(f"[seed {seed}] early stop", flush=True); break

    m.load_state_dict(best_state)
    _, _, te_errs = eval_gaps(m, te)
    pg = [CG.gap_row(i, g, e) for i, (g, e) in enumerate(zip(te, te_errs)) if len(e)]
    CG.write_per_gap(os.path.join(sdir, "per_gap.csv"), pg)
    brows = CG.bucket_rows(pg)
    CG.write_buckets(os.path.join(sdir, "test_buckets.csv"), brows)
    ov = brows[0]
    print(f"[seed {seed}][TEST] {ov['n']} gaps OVERALL ratio_med {ov['ratio']} "
          f"(GCD {ov['gcd_med']} model {ov['model_med']} win {ov['win_pct']}%)", flush=True)
    for r in brows[1:]:
        print(f"   {r['bucket']:<11} n={r['n']:4d}  GCD {r['gcd_med']}  model {r['model_med']}  "
              f"ratio {r['ratio']}", flush=True)
    del m, opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, brows, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="200/60/60 graphs, 2 epochs, 1 seed")
    args = ap.parse_args()

    tr, va, te = CG.load_gaps()
    seeds, epochs, out_root = SEEDS, EPOCHS, OUT
    if args.smoke:
        tr, va, te = tr[:200], va[:60], te[:60]
        seeds, epochs, out_root = (0,), 2, os.path.join(HERE, "runs_5seed_gaps_smoke")
        print(f"[smoke] {len(tr)} train / {len(va)} val / {len(te)} test | 2 epochs | 1 seed", flush=True)
    os.makedirs(out_root, exist_ok=True)

    all_ep, all_bk, summary = [], [], []
    for seed in seeds:
        print(f"\n================ SEED {seed} ================", flush=True)
        rows, brows, best_val = run_seed(seed, tr, va, te, epochs, out_root)
        all_ep += [{"seed": seed, **r} for r in rows]
        all_bk += [{"seed": seed, **r} for r in brows]
        summary.append(CG.summary_row(seed, brows, n_epochs=len(rows), best_val=round(best_val, 4)))
        CG.write_aggregates(out_root, all_ep, all_bk, summary)         # crash-safe: after every seed

    print("\n=== RFF-GNN 5-SEED SUMMARY (TEST, OVERALL) ===", flush=True)
    for s in summary:
        print(f"  seed {s['seed']}: ratio {s['ratio']} | model_med {s['model_med']} "
              f"| GCD {s['gcd_med']} | win {s['win_pct']}%", flush=True)
    rat = np.array([s["ratio"] for s in summary], float)
    print(f"  ratio = {rat.mean():.4f} +/- {rat.std(ddof=0):.4f}", flush=True)
    print(f"saved under {out_root}", flush=True)


if __name__ == "__main__":
    main()

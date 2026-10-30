"""Vanilla bidirectional-GRU imputer -- OUR standard-RNN control baseline, 5 seeds, evaluated on the
FQGNN's EXACT test gaps via task_alpha.baselines.common_gaps (same split, same metrics, same CSVs).

This model is deliberately TEXTBOOK (no FQ ideas, no attention, no residual tricks):
  input  [n, 8]  = [lat_n, lon_n, sog_n, cog_sin, cog_cos, is_held, dt_prev_h, dt_next_h]
                   lat/lon/sog z-scored with fit_latlon_stats(TRAIN) and ZEROED at held timesteps
                   (leak-safe: held truth never enters the input); dt_* = time delta to the
                   previous/next timestep in hours (0 at sequence ends, clipped at 6 h)
  model          = nn.GRU(8 -> 128, 2 layers, bidirectional, dropout 0.1)  ->  Linear(256 -> 2)
  output [n, 2]  = z-scored (lat, lon) per timestep; MSE loss at HELD timesteps only
  training       = Adam lr 1e-3, batch 32 (packed sequences -- padding excluded from everything),
                   100 epochs, early stop patience 10 on VAL median-NM, restore best-val

Outputs (task_alpha/baselines/BiGRU/runs_5seed_gaps/, FQGNN 5-seed format):
  seed<k>/epochs.csv   epoch, train_loss, val_median_nm, val_ratio
  seed<k>/per_gap.csv  gap_id, mmsi, gap_min, sinuosity, gcd_med, model_med, win  (774 test gaps)
  seed<k>/test_buckets.csv, seed<k>/best.pt
  all_seeds_epochs.csv, all_seeds_test_buckets.csv, summary.csv (+MEAN/STD)  -- after every seed

Run (gpubench env, any cwd -- the script chdirs to the repository root itself):
  python task_alpha/baselines/BiGRU/run_bigru_5seed_gaps.py
  add --smoke for a tiny end-to-end check (200/60/60 graphs, 2 epochs, 1 seed, *_smoke out dir)
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))          # repository root
sys.path.insert(0, ROOT)
os.chdir(ROOT)                                                        # data loaders assume this cwd

from task_alpha.baselines.common_gaps import (                        # noqa: E402
    bucket_rows, fit_latlon_stats, gap_row, load_gaps, model_errors_latlon, summary_row,
    to_series, write_aggregates, write_buckets, write_epochs, write_per_gap)
from task_alpha.metrics import gcd_errors                             # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(HERE, "runs_5seed_gaps")
SEEDS = (0, 1, 2, 3, 4)
DT_CLIP_H = 6.0
BATCH = 32


# ------------------------------------------------------------------------------------------------ #
#  Features                                                                                          #
# ------------------------------------------------------------------------------------------------ #
def make_features(g, mean, std):
    """RunGraph -> (x [n,8] float32, y [n,2] float32, held [n] bool).  Inputs are z-scored then
    ZEROED at held timesteps (0 in normalized space + is_held flag -- standard masked imputation);
    truth y is used ONLY as the loss target at held / metrics, never as input."""
    s = to_series(g)
    h = s["held"]
    lat_n = np.where(h, 0.0, (s["lat_true"] - mean[0]) / std[0])
    lon_n = np.where(h, 0.0, (s["lon_true"] - mean[1]) / std[1])
    sog_n = np.where(h, 0.0, (s["sog_in"] - mean[2]) / std[2])
    dt = np.diff(s["t_sec"]) / 3600.0
    dt_prev = np.clip(np.concatenate([[0.0], dt]), 0.0, DT_CLIP_H)
    dt_next = np.clip(np.concatenate([dt, [0.0]]), 0.0, DT_CLIP_H)
    x = np.stack([lat_n, lon_n, sog_n, s["cog_sin"], s["cog_cos"], h.astype(np.float64),
                  dt_prev, dt_next], axis=1).astype(np.float32)
    y = np.stack([(s["lat_true"] - mean[0]) / std[0],
                  (s["lon_true"] - mean[1]) / std[1]], axis=1).astype(np.float32)
    return x, y, h.copy()


def collate(feats):
    """Pad a list of (x, y, held) to batch-max length.  Returns X, Y, HELD, VALID (bool pad mask);
    padding rows are False in both HELD and VALID, so they are excluded from loss and metrics."""
    n = max(len(x) for x, _, _ in feats)
    B = len(feats)
    X = torch.zeros(B, n, 8)
    Y = torch.zeros(B, n, 2)
    H = torch.zeros(B, n, dtype=torch.bool)
    V = torch.zeros(B, n, dtype=torch.bool)
    for i, (x, y, h) in enumerate(feats):
        L = len(x)
        X[i, :L] = torch.from_numpy(x)
        Y[i, :L] = torch.from_numpy(y)
        H[i, :L] = torch.from_numpy(h)
        V[i, :L] = True
    return X, Y, H, V


# ------------------------------------------------------------------------------------------------ #
#  Model: textbook bidirectional GRU imputer                                                         #
# ------------------------------------------------------------------------------------------------ #
class BiGRUImputer(nn.Module):
    def __init__(self, d_in=8, hidden=128, layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, num_layers=layers, bidirectional=True,
                          batch_first=True, dropout=dropout)
        self.head = nn.Linear(2 * hidden, 2)

    def forward(self, x, lengths):
        """Packed sequences so padding never enters the recurrence (either direction)."""
        p = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                              enforce_sorted=False)
        o, _ = self.gru(p)
        o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=x.shape[1])
        return self.head(o)


@torch.no_grad()
def predict_held(model, feats, mean, std, bs=64):
    """Forward a feature list -> per-graph (pred_lat, pred_lon) at HELD timesteps (node order),
    denormalized to degrees."""
    model.eval()
    out = []
    for i0 in range(0, len(feats), bs):
        chunk = feats[i0:i0 + bs]
        X, Y, H, V = collate(chunk)
        pred = model(X.to(DEV), V.sum(1)).float().cpu().numpy()
        for j, (x, _, h) in enumerate(chunk):
            pz = pred[j, :len(x)][h]                                  # (n_held, 2) z-scored
            out.append((pz[:, 0] * std[0] + mean[0], pz[:, 1] * std[1] + mean[1]))
    return out


# ------------------------------------------------------------------------------------------------ #
#  One seed: train -> early stop on val median-NM -> restore best -> test                            #
# ------------------------------------------------------------------------------------------------ #
def run_seed(seed, tr_f, va_f, te_f, va_g, te_g, mean, std, val_gcd_med, epochs, patience, out_root):
    sdir = os.path.join(out_root, f"seed{seed}")
    os.makedirs(sdir, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    m = BiGRUImputer().to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)

    rows, best, best_state, pctr = [], 1e18, None, 0
    for ep in range(epochs):
        t0 = time.time()
        m.train()
        order = list(range(len(tr_f))); random.Random(seed * 1000 + ep).shuffle(order)
        ep_loss, nb = 0.0, 0
        for i0 in range(0, len(order), BATCH):
            batch = [tr_f[j] for j in order[i0:i0 + BATCH]]
            X, Y, H, V = collate(batch)
            X, Y, H = X.to(DEV), Y.to(DEV), H.to(DEV)
            pred = m(X, V.sum(1))
            loss = ((pred[H] - Y[H]) ** 2).mean()                     # held timesteps only (pad is False in H)
            opt.zero_grad()
            if torch.isfinite(loss):
                loss.backward(); opt.step()
                ep_loss += float(loss.detach()); nb += 1
        # ---- VAL: per-gap median NM (identical metric to the FQGNN run) ----
        preds = predict_held(m, va_f, mean, std)
        vmeds = [float(np.median(model_errors_latlon(g, la, lo)))
                 for g, (la, lo) in zip(va_g, preds)]
        vmed = float(np.median(vmeds)); vr = vmed / val_gcd_med
        rows.append({"epoch": ep, "train_loss": round(ep_loss / max(nb, 1), 6),
                     "val_median_nm": round(vmed, 4), "val_ratio": round(vr, 4)})
        write_epochs(os.path.join(sdir, "epochs.csv"), rows)
        if vmed < best - 1e-4:
            best, best_state, pctr = vmed, copy.deepcopy(m.state_dict()), 0
            torch.save({"state_dict": best_state}, os.path.join(sdir, "best.pt"))
        else:
            pctr += 1
        print(f"[seed {seed}] ep {ep:3d}: train {ep_loss / max(nb, 1):.5f} | val {vmed:.3f} NM "
              f"(ratio {vr:.4f}, best {best:.3f}) | {time.time() - t0:.1f}s", flush=True)
        if pctr >= patience:
            print(f"[seed {seed}] early stop (patience {patience})", flush=True)
            break

    m.load_state_dict(best_state)
    # ---- TEST: per-gap rows + curvature buckets (common_gaps = FQGNN-identical) ----
    preds = predict_held(m, te_f, mean, std)
    prows = [gap_row(i, g, model_errors_latlon(g, la, lo))
             for i, (g, (la, lo)) in enumerate(zip(te_g, preds))]
    write_per_gap(os.path.join(sdir, "per_gap.csv"), prows)
    brows = bucket_rows(prows)
    write_buckets(os.path.join(sdir, "test_buckets.csv"), brows)
    ov = brows[0]
    print(f"[seed {seed}][TEST] {ov['n']} gaps OVERALL ratio {ov['ratio']} "
          f"(GCD {ov['gcd_med']} model {ov['model_med']} win {ov['win_pct']}%)", flush=True)
    for r in brows[1:]:
        print(f"   {r['bucket']:<11} n={r['n']:4d}  GCD {r['gcd_med']}  model {r['model_med']}  "
              f"ratio {r['ratio']}", flush=True)

    del m, opt, best_state
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return rows, brows, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end check: 200/60/60 graphs, 2 epochs, seed 0 only, *_smoke dir")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    cli = ap.parse_args()

    out_root = OUT + ("_smoke" if cli.smoke else "")
    os.makedirs(out_root, exist_ok=True)
    seeds = (0,) if cli.smoke else SEEDS
    epochs = 2 if cli.smoke else cli.epochs

    tr, va, te = load_gaps()
    if cli.smoke:
        tr, va, te = tr[:200], va[:60], te[:60]
        print(f"[smoke] subset: {len(tr)} train / {len(va)} val / {len(te)} test", flush=True)

    mean, std = fit_latlon_stats(tr)                                  # TRAIN-only z-score stats
    print(f"[stats] mean={np.round(mean, 4).tolist()} std={np.round(std, 4).tolist()}", flush=True)
    tr_f = [make_features(g, mean, std) for g in tr]
    va_f = [make_features(g, mean, std) for g in va]
    te_f = [make_features(g, mean, std) for g in te]
    val_gcd_med = float(np.median([np.median(gcd_errors(g)) for g in va]))
    print(f"[val] GCD median-of-medians = {val_gcd_med:.4f} NM (val_ratio reference)", flush=True)
    print(f"[cfg] device={DEV} batch={BATCH} epochs={epochs} patience={cli.patience} "
          f"seeds={list(seeds)} out={out_root}", flush=True)

    all_ep, all_bk, summary = [], [], []
    for seed in seeds:
        print(f"\n================ SEED {seed} ================", flush=True)
        rows, brows, best = run_seed(seed, tr_f, va_f, te_f, va, te, mean, std, val_gcd_med,
                                     epochs, cli.patience, out_root)
        all_ep += [{"seed": seed, **r} for r in rows]
        all_bk += [{"seed": seed, **r} for r in brows]
        summary.append(summary_row(seed, brows, len(rows), round(best, 4)))
        write_aggregates(out_root, all_ep, all_bk, summary)           # crash-safe: after every seed

    rat = np.array([s["ratio"] for s in summary], float)
    mm = np.array([s["model_med"] for s in summary], float)
    wp = np.array([s["win_pct"] for s in summary], float)
    print("\n=== Bi-GRU 5-SEED SUMMARY (TEST, OVERALL) ===", flush=True)
    for s in summary:
        print(f"  seed {s['seed']}: ratio {s['ratio']} | model_med {s['model_med']} "
              f"| GCD {s['gcd_med']} | win {s['win_pct']}%", flush=True)
    print(f"  ratio     = {rat.mean():.4f} +/- {rat.std(ddof=0):.4f}", flush=True)
    print(f"  model_med = {mm.mean():.3f} +/- {mm.std(ddof=0):.3f} NM", flush=True)
    print(f"  win_pct   = {wp.mean():.1f} +/- {wp.std(ddof=0):.1f} %", flush=True)
    print(f"saved under {out_root}", flush=True)

    # completion sentinel for the parent wrapper (see __main__): everything above is written+flushed
    with open(_sentinel_path(), "w") as f:
        f.write("ratio %.4f +/- %.4f\n" % (rat.mean(), rat.std(ddof=0)))
    sys.stdout.flush(); sys.stderr.flush()


def _sentinel_path():
    """Mirrors main()'s out_root choice (only --smoke changes it)."""
    return os.path.join(OUT + ("_smoke" if "--smoke" in sys.argv[1:] else ""), "_RUN_COMPLETE")


if __name__ == "__main__":
    if os.environ.get("BIGRU_WORKER") == "1":
        main()
    else:
        # Using any cuDNN RNN crashes this Windows env's interpreter at DLL detach (0xC0000409)
        # AFTER all work is done -- verified with a bare nn.GRU; even os._exit(0)/TerminateProcess
        # cannot force a clean code, and disabling cuDNN is 4x slower.  So: run the real work in a
        # child process and judge success by the completion sentinel, not the child's exit code.
        import subprocess
        sp = _sentinel_path()
        if os.path.exists(sp):
            os.remove(sp)
        rc = subprocess.call([sys.executable, os.path.abspath(__file__)] + sys.argv[1:],
                             env=dict(os.environ, BIGRU_WORKER="1"), cwd=ROOT)
        if os.path.exists(sp):
            if rc != 0:
                print(f"[wrapper] worker exit code {rc} occurred AFTER run completion "
                      f"(known cuDNN teardown crash on this env) -- SUCCESS", flush=True)
            sys.exit(0)
        print(f"[wrapper] worker FAILED before completion (exit {rc})", flush=True)
        sys.exit(rc if rc != 0 else 1)

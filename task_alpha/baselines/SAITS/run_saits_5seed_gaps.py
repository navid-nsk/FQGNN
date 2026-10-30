"""5-seed VANILLA SAITS baseline on the FQGNN's EXACT test gaps (head-to-head, FQGNN CSV formats).

SAITS model code (modeling/saits.py + friends) is imported UNCHANGED -- this file is an external
runner only.  Data / metrics / CSV writers all come from task_alpha.baselines.common_gaps, so the
per-gap rows and curvature buckets are row-aligned with task_alpha/runs/full_faithful_5seed.

Protocol (vanilla SAITS, fixed HP from the Optuna sweep -- hp_saits/best_params.json):
  * Every RunGraph -> a sequence (common_gaps.to_series) of 5 features [lat, lon, sog, cog_sin,
    cog_cos]; lat/lon/sog z-scored with TRAIN-observed stats; padded with NaN to one global
    SEQ_LEN = max series length over ALL splits.
  * LEAK RULE: held-gap cells are NaN (missing_mask=0) in EVERY model input -- their truth never
    enters the network.  Training is SAITS's own MIT protocol (mask 30% of the OBSERVED cells,
    ORT+MIT loss, weights=1) -- held cells are simply missing during training.
  * Val each epoch = impute the val graphs, median over per-gap median great-circle NM at held
    nodes (common_gaps.model_errors_latlon); plateau early-stop patience 12, best-val restore.
  * Test = impute, per-gap rows via common_gaps.gap_row / bucket_rows (identical GCD column).

Outputs (task_alpha/baselines/SAITS/runs_5seed_gaps/):
  seed<k>/epochs.csv        epoch, train_loss, val_median_nm, val_ratio
  seed<k>/per_gap.csv       gap_id, mmsi, gap_min, sinuosity, gcd_med, model_med, win
  seed<k>/test_buckets.csv  bucket, n, gcd_med, model_med, ratio, win_pct
  seed<k>/best.pt           best-val checkpoint (+cfg, seq_len, norm stats)
  all_seeds_epochs.csv / all_seeds_test_buckets.csv / summary.csv  (rewritten after every seed)

Run from the repository root:
  python task_alpha/baselines/SAITS/run_saits_5seed_gaps.py
  add --smoke for a tiny end-to-end check (200/60/60 graphs, 2 epochs, 1 seed, separate out dir).
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

HERE = os.path.dirname(os.path.abspath(__file__))                     # .../baselines/SAITS
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))        # repository root
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_saits_h2h as R                       # the VANILLA head-to-head runner: build_model + mit_mask_batch
from task_alpha.baselines import common_gaps as CG
from task_alpha.metrics import gcd_errors

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = (0, 1, 2, 3, 4)
EPOCHS, PATIENCE = 200, 12
N_FEAT = 5                                      # [lat, lon, sog, cog_sin, cog_cos]

# Fixed HP from the Optuna sweep (hp_saits/best_params.json) -- do NOT re-tune.
CFG = dict(
    n_groups=3, n_group_inner_layers=1, d_model=256, d_inner=256, n_head=2, d_k=16, d_v=16,
    dropout=0.03476071785753891,
    # vanilla fixed knobs (never tuned):
    param_sharing_strategy="inner_group", input_with_mask=True, diagonal_attention_mask=True,
)
LR = 0.0018832519048593583
BATCH_SIZE = 16
MIT_RATE = 0.3


# ------------------------------------------------------------------------------------------------ #
#  DATA: RunGraphs -> fixed-length NaN-padded arrays                                                 #
# ------------------------------------------------------------------------------------------------ #
def build_split_arrays(graphs, seq_len, mean, std):
    """[N, seq_len, 5] float32 with NaN at held cells AND padding (SAITS treats NaN as missing;
    missing_mask=0 there -> held truth NEVER enters the model input).  Also returns the [N, seq_len]
    held-cell mask (sequence positions of the gap, ascending == original node order)."""
    N = len(graphs)
    X = np.full((N, seq_len, N_FEAT), np.nan, dtype=np.float32)
    held = np.zeros((N, seq_len), dtype=bool)
    for i, g in enumerate(graphs):
        s = CG.to_series(g)
        n = len(s["t_sec"])
        feats = np.stack([
            (s["lat_in"] - mean[0]) / std[0],       # lat_in/lon_in are already zeroed at held --
            (s["lon_in"] - mean[1]) / std[1],       # we additionally overwrite held cells with NaN
            (s["sog_in"] - mean[2]) / std[2],
            s["cog_sin"], s["cog_cos"],
        ], axis=1).astype(np.float32)
        feats[s["held"], :] = np.nan                # ALL features missing at the held gap
        X[i, :n] = feats
        held[i, :n] = s["held"]
    return X, held


# ------------------------------------------------------------------------------------------------ #
#  IMPUTE + METRICS                                                                                  #
# ------------------------------------------------------------------------------------------------ #
@torch.no_grad()
def impute_batched(model, X, bs=64):
    """Vanilla SAITS.impute on NaN->0 inputs with missing_mask marking the observed cells.
    Returns the full imputed array [N, L, F] (normalized), original order."""
    model.eval()
    outs = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs]
        mm = (~np.isnan(xb)).astype(np.float32)
        xh = np.nan_to_num(xb).astype(np.float32)
        imputed, _ = model.impute({"X": torch.from_numpy(xh).to(DEV),
                                   "missing_mask": torch.from_numpy(mm).to(DEV)})
        outs.append(imputed.float().cpu().numpy())
    return np.concatenate(outs, axis=0)


def gap_errors(model, X, held, graphs, mean, std, bs=64):
    """Per-gap arrays of great-circle NM error at the held nodes (denormalized lat/lon vs graph truth)."""
    imp = impute_batched(model, X, bs)
    errs = []
    for i, g in enumerate(graphs):
        m = held[i]
        lat = imp[i, m, 0].astype(np.float64) * std[0] + mean[0]
        lon = imp[i, m, 1].astype(np.float64) * std[1] + mean[1]
        errs.append(CG.model_errors_latlon(g, lat, lon))
    return errs


def median_of_gap_medians(errs):
    return float(np.median([np.median(e) for e in errs]))


# ------------------------------------------------------------------------------------------------ #
#  ONE SEED                                                                                          #
# ------------------------------------------------------------------------------------------------ #
def run_seed(seed, Xtr, Xva, Hva, va, Xte, Hte, te, mean, std, val_gcd_med, out_root,
             epochs=EPOCHS, patience=PATIENCE):
    sdir = os.path.join(out_root, f"seed{seed}")
    os.makedirs(sdir, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    rng = np.random.RandomState(seed)                       # MIT artificial-masking rng

    seq_len = Xtr.shape[1]
    model = R.build_model(seq_len, N_FEAT, DEV, CFG).to(DEV)
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    if seed == SEEDS[0]:
        print(f"[run] SAITS (vanilla) params: {nparam} | device {DEV} | seq_len {seq_len} "
              f"| cfg {CFG} | lr {LR} bs {BATCH_SIZE} mit_rate {MIT_RATE}", flush=True)

    rows, best, best_state, pctr = [], 1e18, None, 0
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        order = list(range(len(Xtr))); random.Random(seed * 1000 + ep).shuffle(order)
        ep_loss, ep_nb = 0.0, 0
        for i in range(0, len(order), BATCH_SIZE):
            Xb = Xtr[order[i:i + BATCH_SIZE]]               # [B,L,F] float32, NaN = missing (held+pad)
            X_hat, mm, im, X_gt = R.mit_mask_batch(Xb, rate=MIT_RATE, rng=rng)  # vanilla MIT on OBSERVED cells
            inputs = {
                "indices": torch.arange(Xb.shape[0]),
                "X": torch.from_numpy(X_hat).to(DEV),
                "missing_mask": torch.from_numpy(mm).to(DEV),
                "X_holdout": torch.from_numpy(X_gt).to(DEV),
                "indicating_mask": torch.from_numpy(im).to(DEV),
            }
            out = model(inputs, "train")
            loss = out["reconstruction_loss"] + out["imputation_loss"]   # ORT+MIT, weights=1 (vanilla)
            opt.zero_grad()
            if torch.isfinite(loss):
                loss.backward(); opt.step()
                ep_loss += float(loss.detach()); ep_nb += 1

        # ---- VAL: impute the val graphs, median over per-gap median NM (the real task metric) ----
        vm = median_of_gap_medians(gap_errors(model, Xva, Hva, va, mean, std))
        vr = vm / val_gcd_med
        rows.append({"epoch": ep, "train_loss": round(ep_loss / max(ep_nb, 1), 6),
                     "val_median_nm": round(vm, 4), "val_ratio": round(vr, 4)})
        CG.write_epochs(os.path.join(sdir, "epochs.csv"), rows)          # crash-safe rewrite
        improved = vm < best - 1e-4
        if improved:
            best, best_state, pctr = vm, copy.deepcopy(model.state_dict()), 0
            torch.save({"state_dict": best_state, "cfg": CFG, "seq_len": seq_len,
                        "norm_mean": mean, "norm_std": std, "val_median_nm": best},
                       os.path.join(sdir, "best.pt"))
        else:
            pctr += 1
        print(f"[seed {seed}] ep {ep:3d}: train {ep_loss / max(ep_nb, 1):.5f} | "
              f"val {vm:.3f} NM ratio {vr:.4f} (best {best:.3f})"
              + ("" if improved else f" (no-improve {pctr}/{patience})")
              + f" | {time.time() - t0:.1f}s", flush=True)
        if patience and pctr >= patience:
            print(f"[seed {seed}] early stop @ ep {ep} (best val {best:.3f} NM)", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- TEST: per-gap rows + curvature buckets (identical semantics to the FQGNN runner) ----
    te_errs = gap_errors(model, Xte, Hte, te, mean, std)
    pg = [CG.gap_row(i, g, e) for i, (g, e) in enumerate(zip(te, te_errs))]
    CG.write_per_gap(os.path.join(sdir, "per_gap.csv"), pg)
    brows = CG.bucket_rows(pg)
    CG.write_buckets(os.path.join(sdir, "test_buckets.csv"), brows)
    ov = brows[0]
    print(f"\n[seed {seed}][TEST] {ov['n']} gaps OVERALL ratio_med {ov['ratio']} "
          f"(GCD {ov['gcd_med']} model {ov['model_med']} win {ov['win_pct']}%)", flush=True)
    for r in brows[1:]:
        print(f"   {r['bucket']:<11} n={r['n']:4d}  GCD {r['gcd_med']:.2f}  "
              f"model {r['model_med']:.2f}  ratio {r['ratio']:.3f}", flush=True)

    del model, opt, best_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, brows, round(best, 4)


# ------------------------------------------------------------------------------------------------ #
#  MAIN                                                                                              #
# ------------------------------------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end check: 200/60/60 graphs, 2 epochs, 1 seed, separate out dir")
    ap.add_argument("--track", default=None, choices=["our_processed", "standard_process"],
                    help="run on a gap-sweep cell (task_alpha/data/<track>/gap_<G>h) instead of "
                         "the default raw 6-h headline cell; protocol otherwise identical")
    ap.add_argument("--gap", type=int, default=None, help="gap hours for --track (2/3/6/12/24)")
    args = ap.parse_args()

    cell_tag = f"_{args.track}_{args.gap}h" if args.track else ""
    out_root = os.path.join(HERE, "runs_5seed_gaps" + cell_tag + ("_smoke" if args.smoke else ""))
    os.makedirs(out_root, exist_ok=True)

    if args.track:                              # gap-sweep cell: same pickles every model shares
        import pickle
        ddir = os.path.join(ROOT, "task_alpha", "data", args.track, f"gap_{args.gap}h")
        tr, va, te = (pickle.load(open(os.path.join(ddir, f"{n}.pkl"), "rb"))
                      for n in ("train", "val", "test"))
        print(f"[run] cell {args.track}/gap_{args.gap}h: "
              f"{len(tr)}/{len(va)}/{len(te)} graphs", flush=True)
    else:
        tr, va, te = CG.load_gaps()             # the FQGNN gaps, identical vessel-disjoint split

    # Fixed SEQ_LEN = global max series length over ALL splits (identical for smoke and real runs).
    seq_len = max(len(CG.to_series(g)["t_sec"]) for g in tr + va + te)
    print(f"[run] SEQ_LEN (global max series length) = {seq_len}", flush=True)

    seeds, epochs = SEEDS, EPOCHS
    if args.smoke:
        tr, va, te = tr[:200], va[:60], te[:60]
        seeds, epochs = (0,), 2
        print(f"[run] SMOKE: {len(tr)} train / {len(va)} val / {len(te)} test | "
              f"{epochs} epochs | 1 seed", flush=True)

    mean, std = CG.fit_latlon_stats(tr)         # z-score stats from TRAIN observed cells only
    print(f"[run] norm stats (lat,lon,sog): mean {np.round(mean, 4).tolist()} "
          f"std {np.round(std, 4).tolist()}", flush=True)

    Xtr, _ = build_split_arrays(tr, seq_len, mean, std)
    Xva, Hva = build_split_arrays(va, seq_len, mean, std)
    Xte, Hte = build_split_arrays(te, seq_len, mean, std)
    print(f"[run] arrays: train {Xtr.shape} | val {Xva.shape} | test {Xte.shape}", flush=True)

    # leak check: every held cell must be NaN in the model-input arrays
    for name, X, H in (("val", Xva, Hva), ("test", Xte, Hte)):
        assert np.isnan(X[H]).all(), f"LEAK: non-NaN held cell in {name} input array"
    print("[run] leak audit: all held cells NaN in model inputs (train/val/test) -- OK", flush=True)

    # val GCD constant (computed once): median over val gaps of median gcd_errors
    val_gcd_med = float(np.median([np.median(gcd_errors(g)) for g in va]))
    print(f"[run] val GCD median-NM = {val_gcd_med:.4f}", flush=True)

    all_ep, all_bk, summary = [], [], []
    for seed in seeds:
        print(f"\n================ SEED {seed} ================", flush=True)
        rows, brows, best_val = run_seed(seed, Xtr, Xva, Hva, va, Xte, Hte, te,
                                         mean, std, val_gcd_med, out_root, epochs=epochs)
        for r in rows:
            all_ep.append({"seed": seed, **r})
        for r in brows:
            all_bk.append({"seed": seed, **{k: r[k] for k in CG.BUCKET_FIELDS}})
        summary.append(CG.summary_row(seed, brows, n_epochs=len(rows), best_val=best_val))
        CG.write_aggregates(out_root, all_ep, all_bk, summary)   # rewritten after EVERY seed

    print("\n=== SAITS (vanilla) 5-SEED SUMMARY (TEST, OVERALL) ===", flush=True)
    rat = np.array([s["ratio"] for s in summary], float)
    mm_ = np.array([s["model_med"] for s in summary], float)
    wp = np.array([s["win_pct"] for s in summary], float)
    for s in summary:
        print(f"  seed {s['seed']}: ratio {s['ratio']} | model_med {s['model_med']} "
              f"| GCD {s['gcd_med']} | win {s['win_pct']}%", flush=True)
    print(f"  ratio     = {rat.mean():.4f} +/- {rat.std(ddof=0):.4f}", flush=True)
    print(f"  model_med = {mm_.mean():.3f} +/- {mm_.std(ddof=0):.3f} NM", flush=True)
    print(f"  win_pct   = {wp.mean():.1f} +/- {wp.std(ddof=0):.1f} %", flush=True)
    print(f"\nsaved per-seed epochs/per_gap/test_buckets/best.pt under {out_root}/seed*/ "
          f"+ all_seeds_epochs.csv, all_seeds_test_buckets.csv, summary.csv", flush=True)
    if args.smoke:
        print("\n[SMOKE OK] end-to-end ran error-free. Do NOT read these numbers as results "
              "(2 epochs / 200 train graphs / 1 seed).", flush=True)


if __name__ == "__main__":
    main()

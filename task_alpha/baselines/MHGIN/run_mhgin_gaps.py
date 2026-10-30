"""Runner: MH-GIN (MTS-HGNN) across the two-track gap-sweep cells, 5 seeds per cell.

Vanilla-faithfulness contract
-----------------------------
The model (`src/models/*`) is imported UNCHANGED and its authors' `config-DK.yaml` hyperparameters are
used verbatim: d_data_model = d_mining_model = 32, Adam(lr 1e-3, weight_decay), MultiStepLR at 75%/90%
of training, epochs 30, patience 3, batch 64.

Deviations, all necessary, none of which are tuning:
  * We bypass `main.py`/`preprocess_data.py`.  Their corruption scheme masks lat/lon at SCATTERED
    single points and masks sog/cog/time at *different* random points; it cannot express a single
    contiguous two-sided gap.  We instead feed the benchmark's own held blocks as the `masks`, which
    trains MH-GIN on exactly the corruption it is evaluated on -- strictly in its favour.
  * Stock `train()` selects its "best" checkpoint on the TRAINING loss and passes the TEST loader in as
    `valid_loader` (main.py:205).  We select on a real validation split instead.  Rigour, not capability.
  * `graph_mask_values` is extended to the 9 channels our dataset does not carry, switching off their
    reconstruction losses (their default is [4, 8]).

`--max-delta` is the one data-appropriateness knob.  `MtsHGnn.max_delta=350.0` is a hardcoded default
matched to Danish coastal AIS cadence; `get_time_interval` CLAMPS every inter-ping delta to [0, max_delta].
Our median inter-ping spacing is ~665 s, so at the vanilla 350 s every delta saturates and the model can no
longer tell a 3-minute step from an 11-minute one.  We report the vanilla setting and a
cadence-matched one, and take whichever is better for MH-GIN.

  python -m task_alpha.baselines.MHGIN.run_mhgin_gaps --all
  python -m task_alpha.baselines.MHGIN.run_mhgin_gaps --track our_processed --gap 6 --seeds 0
  python -m task_alpha.baselines.MHGIN.run_mhgin_gaps --smoke
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np

# Must be set BEFORE torch initialises CUDA: cuts allocator fragmentation on the long eval loops.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from task_alpha.baselines.common_gaps import (bucket_rows, gap_row, model_errors_latlon,  # noqa: E402
                                              summary_row, write_aggregates, write_buckets,
                                              write_epochs, write_per_gap)
from task_alpha.baselines.MHGIN.mhgin_data import build_arrays, fit_status, max_seq_len  # noqa: E402
from task_alpha.baselines.MHGIN.mhgin_model import build_model, predict_latlon, seed_everything  # noqa: E402
from task_alpha.data import collate_graphs_interp  # noqa: F401  (registers RunGraph)

DATA_ROOT = "task_alpha/data"
SWEEP_ROOT = "task_alpha/runs/gap_sweep"
MODEL_NAME = "mhgin"
TRACK_GAPS = {"our_processed": (3, 6, 12, 24), "standard_process": (2, 6, 12, 24)}

# config-DK.yaml
EPOCHS, LR, WD, BATCH, PATIENCE, D_MODEL = 30, 1e-3, 0.0, 64, 10, 32
# Inference needs no autograd graph, so it can run at a much smaller batch without touching the
# vanilla training batch size (64).  Keeps the GPU footprint down; results are batch-invariant here.
EVAL_BATCH = 8
# MH-GIN at the vanilla batch of 64 with our sequence length (~236) peaks near 18 GiB.  We keep the
# EFFECTIVE batch at 64 via gradient accumulation over micro-batches, so the optimizer sees the vendor
# batch size while peak memory drops ~4x.  (The per-component losses are masked MEANS, so a mean over
# micro-batch means differs from the exact 64-sample mean only when the micro-batches contain unequal
# numbers of valid cells -- a negligible reweighting, not a change of objective.)
MICRO_BATCH = 16


def _load(track, gap, split):
    with open(f"{DATA_ROOT}/{track}/gap_{gap}h/{split}.pkl", "rb") as f:
        return pickle.load(f)


def _batches(n, bs, shuffle, rng=None):
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for lo in range(0, n, bs):
        yield idx[lo:lo + bs]


def _to_batch(arrs, sel, device):
    d, l, m, p = arrs
    return {"observed_data": torch.as_tensor(d[sel], device=device),
            "observed_labels": torch.as_tensor(l[sel], device=device),
            "masks": torch.as_tensor(m[sel], device=device),
            "padding_masks": torch.as_tensor(p[sel], device=device)}


def _eval_gaps(model, data, meta, graphs, device):
    preds = predict_latlon(model, data, meta, device, batch_size=EVAL_BATCH)
    rows = []
    for i, (g, (plat, plon)) in enumerate(zip(graphs, preds)):
        err = model_errors_latlon(g, plat, plon)
        if len(err):
            rows.append(gap_row(i, g, err))
    return rows


def run_seed(track, gap, seed, tr, va, te, max_len, status, device, max_delta, epochs, out_root,
             patience=PATIENCE, micro=MICRO_BATCH):
    seed_everything(seed)
    # max_delta must go through the CONSTRUCTOR: TimeEncoder (M1Encoder.py:73) and the decoder
    # (MTSHGNN.py:76) capture it at __init__; assigning model.max_delta afterwards would only rebind
    # the loss clamp and leave the Delta-t feature clamped at the vendor default.
    model = build_model(status, device, d_model=D_MODEL, max_delta=max_delta)
    assert float(model.data_encoder.time_encoder.max_delta) == float(max_delta)

    a_tr = build_arrays(tr, max_len)[:4]
    _va = build_arrays(va, max_len)
    d_va, m_va = _va[0], _va[4]
    d_te, _, _, _, m_te = build_arrays(te, max_len)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(.75 * epochs), int(.9 * epochs)],
                                                 gamma=0.1)
    rng = np.random.default_rng(seed)
    best, best_ep, bad, ep_rows = float("inf"), -1, 0, []
    best_state = None

    for ep in range(epochs):
        model.train()
        tot, nb, spa = 0.0, 0, 0.0
        for sel in _batches(len(tr), BATCH, True, rng):
            opt.zero_grad(set_to_none=True)
            chunks = [sel[k:k + micro] for k in range(0, len(sel), micro)]
            bl = bs_ = 0.0
            for ch in chunks:
                (loss, ll), _ = model(_to_batch(a_tr, ch, device), evaluate=False)
                (loss / len(chunks)).backward()
                bl += float(loss.detach()) / len(chunks)
                bs_ += float(ll[0].detach()) / len(chunks)
            opt.step()
            tot += bl; spa += bs_; nb += 1
        sched.step()
        tr_loss, tr_spa = tot / max(nb, 1), spa / max(nb, 1)

        # ONE criterion for BOTH decisions: the benchmark metric (val median gap error), with a GENEROUS
        # patience (default 10, vs the vendor's 3).  Neither alternative works here:
        #   * patience 3 on this metric stops after about four epochs -- it is non-monotone early (worsens
        #     while gamma inflates, then recovers), so a tight patience selects an essentially untrained model;
        #   * patience 3 on the vendor's val LOSS stops around epoch 11 while positional error is still
        #     falling monotonically -- that loss is ~10^4:1 dominated by the temporal Huber term (seconds)
        #     over the spatial term (radians) and plateaus by epoch 1, so it carries no signal.
        # Patience 10 rides out the early dip (the best epochs lie at the start or around epochs 13-14)
        # while cutting the wasted epochs after convergence (standard_process/gap_2h is flat from epoch
        # ~5).  Still strictly more training than the vendor's own patience-3 rule would allow.  The
        # 30-epoch cap is theirs.
        vrows = _eval_gaps(model, d_va, m_va, va, device)
        if device == "cuda":
            torch.cuda.empty_cache()
        vov = bucket_rows(vrows)[0]
        vr = float(vov["ratio"])
        sd = model.data_decoder.spatio_decoder
        gl, gp = float(sd.gamma_lambda), float(sd.gamma_phi)
        ep_rows.append({"epoch": ep, "train_loss": round(tr_loss, 5), "val_ratio": vr,
                        "val_model_med": vov["model_med"], "val_gcd_med": vov["gcd_med"],
                        "note": f"train_spatial_rad={tr_spa:.5f};gamma_lon={gl:.4f};gamma_lat={gp:.4f}"})
        print(f"[{MODEL_NAME}] {track}/gap_{gap}h s{seed} ep{ep:02d} train {tr_loss:.2f} "
              f"(spatial {tr_spa:.4f} rad)  val_med {vov['model_med']} NM  "
              f"ratio {vr:.3f}  gamma=({gl:.3f},{gp:.3f})", flush=True)

        if vr < best - 1e-6:
            best, best_ep, bad = vr, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"[{MODEL_NAME}] early stop at ep{ep} (no val-metric gain in {patience} "
                      f"epochs; best ep{best_ep})", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    rows = _eval_gaps(model, d_te, m_te, te, device)
    brows = bucket_rows(rows)
    sdir = f"{out_root}/seed{seed}"
    os.makedirs(sdir, exist_ok=True)
    write_per_gap(f"{sdir}/per_gap.csv", rows)
    write_buckets(f"{sdir}/test_buckets.csv", brows)
    write_epochs(f"{sdir}/epochs.csv", ep_rows)
    ov = brows[0]
    print(f"[{MODEL_NAME}][TEST] {track}/gap_{gap}h s{seed}  n={ov['n']}  GCD {ov['gcd_med']}  "
          f"model {ov['model_med']}  ratio {ov['ratio']}  win {ov['win_pct']}%", flush=True)
    return ep_rows, brows, summary_row(seed, brows, len(ep_rows), best)


def run_cell(track, gap, seeds, device, max_delta, epochs, limit=0, patience=PATIENCE,
             micro=MICRO_BATCH):
    tr, va, te = _load(track, gap, "train"), _load(track, gap, "val"), _load(track, gap, "test")
    if limit:
        tr, va, te = tr[:limit], va[:max(2, limit // 8)], te[:max(8, limit // 8)]
    max_len = max_seq_len(tr, va, te)
    status = fit_status(tr)
    print(f"[{MODEL_NAME}] {track}/gap_{gap}h  train {len(tr)} val {len(va)} test {len(te)}  "
          f"max_len {max_len}  max_delta {max_delta}", flush=True)

    cdir = f"{SWEEP_ROOT}/{track}/gap_{gap}h/{MODEL_NAME}"
    all_ep, all_bk, summ = [], [], []
    for s in seeds:
        # SEED-LEVEL RESUME: a completed seed's CSVs are reloaded instead of retraining --
        # orchestration only; the training procedure is unchanged.
        sd = f"{cdir}/seed{s}"
        if os.path.exists(f"{sd}/test_buckets.csv") and os.path.exists(f"{sd}/epochs.csv"):
            import csv as _csv
            with open(f"{sd}/epochs.csv", newline="") as f:
                ep = list(_csv.DictReader(f))
            with open(f"{sd}/test_buckets.csv", newline="") as f:
                bk = list(_csv.DictReader(f))
            best = min(float(r["val_ratio"]) for r in ep)
            print(f"[{MODEL_NAME}] seed {s} already done ({len(ep)} epochs on disk) -- skip", flush=True)
            all_ep += [{"seed": s, **r} for r in ep]
            all_bk += [{"seed": s, **r} for r in bk]
            summ.append(summary_row(s, bk, len(ep), best))
            continue
        t0 = time.time()
        ep, bk, sm = run_seed(track, gap, s, tr, va, te, max_len, status, device, max_delta, epochs, cdir,
                              patience=patience, micro=micro)
        peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
        print(f"[{MODEL_NAME}] seed {s} done in {time.time()-t0:.0f}s  peak VRAM {peak:.2f} GiB", flush=True)
        all_ep += [{"seed": s, **r} for r in ep]
        all_bk += [{"seed": s, **r} for r in bk]
        summ.append(sm)
    write_aggregates(cdir, all_ep, all_bk, summ)
    return summ


def main():
    ap = argparse.ArgumentParser(description="MH-GIN (MTS-HGNN) two-sided gap baseline")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--track"); ap.add_argument("--gap", type=int)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-delta", type=float, default=350.0, help="350 = vendor default")
    ap.add_argument("--limit", type=int, default=0, help="subsample graphs (smoke)")
    ap.add_argument("--patience", type=int, default=PATIENCE,
                    help="early-stop patience on the val gap metric (10 = generous; vendor 3)")
    ap.add_argument("--micro-batch", type=int, default=MICRO_BATCH,
                    help="grad-accum micro-batch; effective batch stays 64")
    ap.add_argument("--vram-frac", type=float, default=0.50,
                    help="hard cap on this process's share of GPU memory")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and 0 < a.vram_frac < 1:
        torch.cuda.set_per_process_memory_fraction(a.vram_frac)
        tot = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"[{MODEL_NAME}] VRAM capped at {a.vram_frac:.0%} of {tot:.1f} GiB "
              f"= {a.vram_frac*tot:.1f} GiB", flush=True)
    if a.smoke:
        cells, a.seeds, a.epochs, a.limit = [("our_processed", 6)], a.seeds[:1], min(a.epochs, 4), a.limit or 400
    elif a.all:
        cells = [(t, g) for t in ("our_processed", "standard_process") for g in TRACK_GAPS[t]]
    elif a.track and a.gap:
        cells = [(a.track, a.gap)]
    else:
        ap.error("choose --all, --smoke, or --track/--gap")

    for track, gap in cells:
        run_cell(track, gap, a.seeds, device, a.max_delta, a.epochs, a.limit, a.patience, a.micro_batch)


if __name__ == "__main__":
    main()

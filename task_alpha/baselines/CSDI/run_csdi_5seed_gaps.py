"""5-seed VANILLA CSDI baseline on the FQGNN's EXACT test gaps -> FQGNN-format CSVs (head-to-head table).

External runner ONLY: no CSDI model source is touched (main_model.py / diff_models.py / utils.py / config
all vanilla).  We reuse run_csdi_h2h.CSDI_AIS (a pure CSDI_base subclass: physio-style process_data +
forward that conditions on OUR contiguous gt_mask) and run_csdi_h2h.train_earlystop (CSDI's vanilla recipe:
Adam wd=1e-6 + MultiStepLR 0.75/0.9, plus plateau early-stop on the VAL denoising loss, is_train=0).

DATA = common_gaps.load_gaps(): the VERIFIED shared eval base = the FQGNN 5-seed run's 5350/635/774 split.
Each RunGraph -> common_gaps.to_series(g) -> a [L, K=5] CSDI sample with channels
    [lat, lon, sog, cog_sin, cog_cos]
lat/lon/sog z-scored with common_gaps.fit_latlon_stats(train); sin/cos raw.  Masks:
    observed_mask = 1 on real timesteps (all K), 0 on padding
    gt_mask(cond) = 1 on observed,  0 on the HELD gap,  0 on padding
    timepoints    = t_sec / 3600.0  (hours; padded tail repeats the last real value)
LEAK RULE: train/val observed_data carries TRUTH at held ONLY as the diffusion denoising TARGET (standard
supervised training; cond_obs = cond_mask * observed_data never reads it).  TEST observed_data is ZEROED at
held (belt-and-braces; impute() mathematically never reads it since cond_mask=0 there).

HP are FIXED from our Optuna sweep (do NOT re-tune): channels=128 nheads=8 lr=9.2188e-4 batch=32 layers=2
diffusion_embedding_dim=128 num_steps=100 beta_end=0.1; everything else vanilla base.yaml defaults.

TEST per seed (best-val weights): CSDI.evaluate sampling, nsample=50, median over samples, denormalize,
common_gaps.model_errors_latlon -> gap_row -> bucket_rows.  VAL median-NM (nsample=25) -> summary best_val.

Outputs under runs_5seed_gaps/ (runs_5seed_gaps_smoke/ with --smoke), all via common_gaps writers:
    seed{k}/epochs.csv  seed{k}/per_gap.csv  seed{k}/test_buckets.csv  seed{k}/best.pt
    all_seeds_epochs.csv  all_seeds_test_buckets.csv  summary.csv   (rewritten after EVERY seed; crash-safe)

Smoke (pipeline proof only -- result meaningless):
    python task_alpha/baselines/CSDI/run_csdi_5seed_gaps.py --smoke
Full run (cwd = repository root):
    python task_alpha/baselines/CSDI/run_csdi_5seed_gaps.py
"""
import argparse
import gc
import os
import random
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)                       # so `main_model` / `diff_models` resolve to CSDI's
FQGNN = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # repository root
if FQGNN not in sys.path:
    sys.path.insert(0, FQGNN)

from run_csdi_h2h import CSDI_AIS, train_earlystop          # pure CSDI_base subclass + vanilla recipe
from task_alpha.baselines import common_gaps

LAT, LON, SOG, CSIN, CCOS = 0, 1, 2, 3, 4
K = 5
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# FIXED tuned HP (Optuna sweep result -- do NOT re-tune here) --------------------------------------------------
HP = dict(channels=128, nheads=8, lr=9.2188e-4, batch_size=32, layers=2,
          diffusion_embedding_dim=128, num_steps=100, beta_end=0.1)


def build_config(epochs):
    """CSDI config dict: tuned HP values + vanilla base.yaml defaults for everything else.  Values only --
    no CSDI code is changed.  target_strategy is irrelevant (CSDI_AIS.forward conditions on our gt_mask)."""
    return {
        "train": {"epochs": epochs, "batch_size": HP["batch_size"], "lr": HP["lr"],
                  "itr_per_epoch": int(1e8)},
        "diffusion": {"layers": HP["layers"], "channels": HP["channels"], "nheads": HP["nheads"],
                      "diffusion_embedding_dim": HP["diffusion_embedding_dim"],
                      "beta_start": 0.0001, "beta_end": HP["beta_end"], "num_steps": HP["num_steps"],
                      "schedule": "quad", "is_linear": False},
        "model": {"is_unconditional": 0, "timeemb": 128, "featureemb": 16, "target_strategy": "random"},
    }


# ------------------------------------------------------------------------------------------------------------ #
#  SERIES -> CSDI samples (variable length; padded per batch)                                                    #
# ------------------------------------------------------------------------------------------------------------ #
def make_samples(graphs, mean, std, zero_held: bool, check_node_order: bool = False):
    """Per graph: normalized values [L,5] + masks + timepoints(h) + held bool.  zero_held=True (TEST):
    observed_data is zeroed at held so truth cannot reach the model even in principle."""
    out = []
    for g in graphs:
        s = common_gaps.to_series(g)
        held = s["held"]
        if check_node_order:                        # series held order must equal ORIGINAL held-node order
            assert np.array_equal(s["node_idx"][held], np.where(g.mask.numpy().astype(bool))[0]), \
                "held-node order mismatch between to_series and g.mask"
        vals = np.stack([(s["lat_true"] - mean[0]) / std[0],
                         (s["lon_true"] - mean[1]) / std[1],
                         (s["sog_in"] - mean[2]) / std[2],
                         s["cog_sin"], s["cog_cos"]], axis=1).astype(np.float32)
        if zero_held:
            vals[held] = 0.0
        n = len(held)
        gt = np.ones((n, K), np.float32); gt[held] = 0.0
        out.append(dict(vals=torch.from_numpy(vals),
                        obs=torch.ones(n, K),
                        gt=torch.from_numpy(gt),
                        tp=torch.from_numpy((s["t_sec"] / 3600.0).astype(np.float32)),
                        held=held))
    return out


def collate(items):
    """Pad a list of samples to the batch max length.  Padded tail: values/masks 0 (never conditioned,
    never scored), timepoints repeat the last real value."""
    B = len(items); L = max(len(it["tp"]) for it in items)
    vals = torch.zeros(B, L, K); obs = torch.zeros(B, L, K); gt = torch.zeros(B, L, K); tp = torch.zeros(B, L)
    for b, it in enumerate(items):
        n = len(it["tp"])
        vals[b, :n] = it["vals"]; obs[b, :n] = it["obs"]; gt[b, :n] = it["gt"]
        tp[b, :n] = it["tp"]; tp[b, n:] = it["tp"][-1]
    return {"observed_data": vals, "observed_mask": obs, "gt_mask": gt, "timepoints": tp}


def length_chunks(samples, bs):
    """Sort-by-length batching (pure engineering: minimizes padding)."""
    order = sorted(range(len(samples)), key=lambda i: len(samples[i]["tp"]))
    return [order[i:i + bs] for i in range(0, len(order), bs)]


class PaddedBatches:
    """Iterable of padded batch dicts for train_earlystop.  Batch COMPOSITION is fixed (length-sorted);
    with shuffle=True the batch ORDER is reshuffled every epoch (seeded per seed+epoch)."""

    def __init__(self, samples, bs, shuffle=False, base_seed=0):
        self.samples = samples; self.chunks = length_chunks(samples, bs)
        self.shuffle = shuffle; self.base_seed = base_seed; self._ep = 0

    def __iter__(self):
        idx = list(range(len(self.chunks)))
        if self.shuffle:
            random.Random(self.base_seed * 100000 + self._ep).shuffle(idx); self._ep += 1
        for ci in idx:
            yield collate([self.samples[j] for j in self.chunks[ci]])


# ------------------------------------------------------------------------------------------------------------ #
#  SAMPLING EVAL: CSDI.evaluate (vanilla impute) -> median over nsample -> denormalized (lat,lon) at held        #
# ------------------------------------------------------------------------------------------------------------ #
@torch.no_grad()
def predict_held(model, samples, chunks, nsample, mean, std):
    """dict: sample index -> (pred_lat_deg, pred_lon_deg) at the HELD positions, in held-node order."""
    model.eval()
    preds = {}
    for chunk in chunks:
        batch = collate([samples[j] for j in chunk])
        smp, _, _, _, _ = model.evaluate(batch, nsample)     # [B, nsample, K, Lmax]
        med = smp.median(dim=1).values.cpu().numpy()         # median over posterior samples
        for pos, j in enumerate(chunk):
            hp = np.where(samples[j]["held"])[0]             # indices < L_j, below any padding
            preds[j] = (med[pos, LAT, hp] * std[0] + mean[0],
                        med[pos, LON, hp] * std[1] + mean[1])
    return preds


def run_seed(seed, tr_s, va_s, te_s, te, va, mean, std, args, sdir):
    """One full train+val+test at `seed`.  Returns (epoch_rows, bucket_rows_list, val_nm, timings)."""
    os.makedirs(sdir, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    cfg = build_config(args.epochs)
    model = CSDI_AIS(cfg, DEV, target_dim=K).to(DEV)
    print(f"[seed {seed}] CSDI params {sum(p.numel() for p in model.parameters())} | device {DEV} | "
          f"HP {HP}", flush=True)

    train_loader = PaddedBatches(tr_s, HP["batch_size"], shuffle=True, base_seed=seed)
    val_loader = PaddedBatches(va_s, HP["batch_size"], shuffle=False)

    history = []
    t0 = time.time()
    train_earlystop(model, cfg["train"], train_loader, val_loader, sdir, args.patience, DEV,
                    history=history)                          # restores best-val weights into `model`
    t_train = time.time() - t0
    s_per_epoch = t_train / max(len(history), 1)

    ep_rows = [{"epoch": ep, "train_loss": round(tl, 6), "val_loss": round(vl, 6)} for ep, tl, vl in history]
    common_gaps.write_epochs(f"{sdir}/epochs.csv", ep_rows)
    torch.save({"state_dict": model.state_dict()}, f"{sdir}/best.pt")
    mp = os.path.join(sdir, "model.pth")                      # train_earlystop's duplicate checkpoint
    if os.path.exists(mp):
        os.remove(mp)

    # ---- VAL median-NM at best weights (summary best_val) ----
    t1 = time.time()
    vp = predict_held(model, va_s, length_chunks(va_s, HP["batch_size"]), args.nsample_val, mean, std)
    v_meds = [float(np.median(common_gaps.model_errors_latlon(g, *vp[j]))) for j, g in enumerate(va)]
    val_nm = round(float(np.median(v_meds)), 4)               # median over val gaps of per-gap median NM
    t_val = time.time() - t1

    # ---- TEST at best weights: per-gap rows + buckets (FQGNN-format) ----
    t2 = time.time()
    tp = predict_held(model, te_s, length_chunks(te_s, HP["batch_size"]), args.nsample, mean, std)
    rows = [common_gaps.gap_row(j, g, common_gaps.model_errors_latlon(g, *tp[j])) for j, g in enumerate(te)]
    t_test = time.time() - t2
    common_gaps.write_per_gap(f"{sdir}/per_gap.csv", rows)
    brows = common_gaps.bucket_rows(rows)
    common_gaps.write_buckets(f"{sdir}/test_buckets.csv", brows)

    ov = brows[0]
    print(f"\n[seed {seed}][TEST] {ov['n']} gaps OVERALL ratio_med {ov['ratio']} "
          f"(GCD {ov['gcd_med']} model {ov['model_med']} win {ov['win_pct']}%)", flush=True)
    for r in brows[1:]:
        print(f"   {r['bucket']:<11} n={r['n']:4d}  GCD {r['gcd_med']:.2f}  "
              f"model {r['model_med']:.2f}  ratio {r['ratio']:.3f}", flush=True)
    print(f"[seed {seed}] per_gap rows {len(rows)} (test graphs {len(te)}) | val median-NM {val_nm} | "
          f"timing: {s_per_epoch:.1f} s/epoch x {len(history)} ep = {t_train:.0f}s train, "
          f"{t_val:.0f}s val-eval (ns={args.nsample_val}), {t_test:.0f}s test-eval (ns={args.nsample})",
          flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"[seed {seed}] VRAM after free: {torch.cuda.memory_allocated() / 1e9:.3f} GB allocated "
              f"(peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB)", flush=True)
        torch.cuda.reset_peak_memory_stats()
    return ep_rows, brows, val_nm


def main():
    ap = argparse.ArgumentParser(description="5-seed VANILLA CSDI on the FQGNN's exact test gaps.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10, help="plateau early-stop on VAL denoising loss")
    ap.add_argument("--nsample", type=int, default=50, help="posterior samples at TEST (median over these)")
    ap.add_argument("--nsample-val", type=int, default=25, help="posterior samples for the VAL median-NM")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end proof: 200/60/60 graphs, 2 epochs, 1 seed, nsample 4")
    ap.add_argument("--track", default=None, choices=["our_processed", "standard_process"],
                    help="run on a gap-sweep cell (task_alpha/data/<track>/gap_<G>h) instead of "
                         "the default raw 6-h headline cell; protocol otherwise identical")
    ap.add_argument("--gap", type=int, default=None, help="gap hours for --track (2/3/6/12/24)")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    cell_tag = f"_{args.track}_{args.gap}h" if args.track else ""
    out = os.path.join(HERE, "runs_5seed_gaps" + cell_tag)
    if args.smoke:
        seeds = seeds[:1]; args.epochs = 2; args.nsample = 4; args.nsample_val = 4
        out = os.path.join(HERE, "runs_5seed_gaps" + cell_tag + "_smoke")
    os.makedirs(out, exist_ok=True)

    # ---- data: the FQGNN gaps (verified shared base), built once, shared by all seeds ----
    if args.track:                              # gap-sweep cell: same pickles every model shares
        import pickle
        droot = os.path.dirname(os.path.dirname(HERE))          # .../task_alpha
        ddir = os.path.join(droot, "data", args.track, f"gap_{args.gap}h")
        tr, va, te = (pickle.load(open(os.path.join(ddir, f"{n}.pkl"), "rb"))
                      for n in ("train", "val", "test"))
        print(f"[run] cell {args.track}/gap_{args.gap}h: "
              f"{len(tr)}/{len(va)}/{len(te)} graphs", flush=True)
    else:
        tr, va, te = common_gaps.load_gaps()
    if args.smoke:
        tr, va, te = tr[:200], va[:60], te[:60]
        print(f"[smoke] subset: {len(tr)} train / {len(va)} val / {len(te)} test", flush=True)
    mean, std = common_gaps.fit_latlon_stats(tr)
    print(f"z-score (lat,lon,sog) mean {np.round(mean, 4)} std {np.round(std, 4)}", flush=True)

    t0 = time.time()
    tr_s = make_samples(tr, mean, std, zero_held=False)               # truth at held = training TARGET only
    va_s = make_samples(va, mean, std, zero_held=False)               # (cond_mask=0 there; never conditioned)
    te_s = make_samples(te, mean, std, zero_held=True, check_node_order=True)   # test: held values ZEROED
    lens = np.array([len(s["tp"]) for s in tr_s + va_s + te_s])
    print(f"series built in {time.time() - t0:.0f}s | K={K} | len min {lens.min()} / med "
          f"{int(np.median(lens))} / max {lens.max()} | timepoints in hours", flush=True)

    all_ep, all_bk, summary = [], [], []
    for seed in seeds:
        print(f"\n================ SEED {seed} ================", flush=True)
        ep_rows, brows, val_nm = run_seed(seed, tr_s, va_s, te_s, te, va, mean, std, args,
                                          os.path.join(out, f"seed{seed}"))
        for r in ep_rows:
            all_ep.append({"seed": seed, **r})
        for r in brows:
            all_bk.append({"seed": seed, **r})
        summary.append(common_gaps.summary_row(seed, brows, len(ep_rows), val_nm))
        common_gaps.write_aggregates(out, all_ep, all_bk, summary)    # after EVERY seed (crash-safe)

    print("\n=== CSDI 5-SEED SUMMARY (TEST, OVERALL) ===", flush=True)
    for s in summary:
        print(f"  seed {s['seed']}: ratio {s['ratio']} | model_med {s['model_med']} "
              f"| GCD {s['gcd_med']} | win {s['win_pct']}%", flush=True)
    rat = np.array([s["ratio"] for s in summary], float)
    mm = np.array([s["model_med"] for s in summary], float)
    wp = np.array([s["win_pct"] for s in summary], float)
    print(f"  ratio     = {rat.mean():.4f} +/- {rat.std(ddof=0):.4f}", flush=True)
    print(f"  model_med = {mm.mean():.3f} +/- {mm.std(ddof=0):.3f} NM", flush=True)
    print(f"  win_pct   = {wp.mean():.1f} +/- {wp.std(ddof=0):.1f} %", flush=True)
    if args.smoke:
        print("  [SMOKE] undertrained -- numbers are MEANINGLESS; this only proves the pipeline runs.")
    print(f"saved per-seed epochs/per_gap/test_buckets/best.pt under {out}/seed*/ "
          f"+ all_seeds_epochs.csv, all_seeds_test_buckets.csv, summary.csv", flush=True)


if __name__ == "__main__":
    main()

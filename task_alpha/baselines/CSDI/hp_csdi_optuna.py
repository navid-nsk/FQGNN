"""Optuna hyper-parameter sweep harness for the VANILLA CSDI baseline on the Arctic-AIS two-sided
contiguous-gap trajectory-imputation task.

This is an EXTERNAL harness.  It does NOT touch any CSDI model source (main_model.py / diff_models.py /
utils.py / config).  It only:
  (1) imports the task pipeline from run_csdi_h2h.py  (build_runs, place_gap, AISCSDIDataset, fit_stats,
      split_runs_vessel_disjoint, CSDI_AIS, train_earlystop, eval_nm) -- unchanged behaviour;
  (2) tunes HYPERPARAMETERS that CSDI already exposes through its config dict (see SEARCH SPACE below);
  (3) minimises the REAL task metric -- validation median great-circle error in NAUTICAL MILES on the gap
      points (NOT the diffusion val-loss).

The CSDI architecture / equations stay exactly vanilla.  We only change config VALUES, never model code.

Two modes:
  --mode sweep : 20-trial Optuna study (TPE sampler + MedianPruner), sequential (n_jobs=1), on a SUBSAMPLE
                 (fewer vessels/files) with a reduced epoch budget + early stop.  Objective = val median-NM.
                 Study persisted to a sqlite db (resumable).  Writes best_params.json + trials CSV.
  --mode final : load best_params.json, train 5 seeds at the best HP on the FULL train split, report TEST
                 median-NM as mean +/- std.  Saves a per-seed CSV + each seed's best model .pth.

SEARCH SPACE (ACTUAL CSDI config keys):
    train.lr                          : loguniform 1e-4 .. 1e-3
    train.batch_size                  : {16, 32, 64}
    diffusion.layers                  : {2, 3, 4}      (# residual layers)
    diffusion.channels                : {32, 64, 128}
    diffusion.nheads                  : {4, 8}         (constrained to divide channels)
    diffusion.diffusion_embedding_dim : {64, 128}
    diffusion.num_steps               : {50, 100, 200} (diffusion T)
    diffusion.beta_end                : {0.1, 0.2, 0.5}
Everything else (beta_start, schedule="quad", is_unconditional, timeemb, featureemb, target_strategy,
weight decay, LR schedule) stays at the vanilla base.yaml default.

Verify (TINY smoke, not the real sweep):
    python hp_csdi_optuna.py --mode sweep  --n-trials 1 --n-mmsi 60 --max-files-per-part 1 \
        --trial-epochs 2 --trial-patience 2 --nsample 3 --window 96
    python hp_csdi_optuna.py --mode final  --n-mmsi 60 --max-files-per-part 1 --seeds 1 \
        --final-epochs 2 --final-patience 2 --nsample 3 --window 96
The full runs:
    python hp_csdi_optuna.py --mode sweep   (defaults: 20 trials, subsample, 60ep/patience8)
    python hp_csdi_optuna.py --mode final   (defaults: 5 seeds, full data, 300ep/patience20)
"""
import argparse
import csv
import gc
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)                      # so `main_model` / `utils` resolve to CSDI's
FQGNN = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # repository root
if FQGNN not in sys.path:
    sys.path.insert(0, FQGNN)

import optuna

# reuse the task pipeline UNCHANGED (pure imports; no CSDI model code is touched) -----------------------------
from run_csdi_h2h import (
    build_runs,
    fit_stats,
    split_runs_vessel_disjoint,
    AISCSDIDataset,
    CSDI_AIS,
    train_earlystop,
    eval_nm,
)

SENTINEL = 1.0e6                                   # returned for a degenerate trial (no valid gaps / NaN)


# ------------------------------------------------------------------------------------------------------------ #
#  CONFIG BUILDER: vanilla base.yaml values, with the SWEPT hyper-parameters overridden from `hp`.               #
#  Only config VALUES change; no model code.  Keys match main_model.py / diff_models.py exactly.                 #
# ------------------------------------------------------------------------------------------------------------ #
def build_config(hp, epochs, itr_per_epoch):
    """hp = dict of swept hyper-parameters.  All other fields are the vanilla base.yaml defaults."""
    return {
        "train": {
            "epochs": epochs,
            "batch_size": hp["batch_size"],
            "lr": hp["lr"],
            "itr_per_epoch": itr_per_epoch if itr_per_epoch else int(1e8),
        },
        "diffusion": {
            "layers": hp["layers"],
            "channels": hp["channels"],
            "nheads": hp["nheads"],
            "diffusion_embedding_dim": hp["diffusion_embedding_dim"],
            "beta_start": 0.0001,                 # vanilla default (not swept)
            "beta_end": hp["beta_end"],
            "num_steps": hp["num_steps"],
            "schedule": "quad",                   # vanilla default (not swept)
            "is_linear": False,                   # vanilla default (not swept)
        },
        "model": {                                # vanilla base.yaml model block (not swept)
            "is_unconditional": 0, "timeemb": 128, "featureemb": 16, "target_strategy": "random",
        },
    }


def suggest_hp(trial):
    """Sample one hyper-parameter set from the search space (ACTUAL CSDI config keys).  nheads is constrained
    to divide channels (TransformerEncoderLayer requires d_model % nhead == 0)."""
    channels = trial.suggest_categorical("channels", [32, 64, 128])
    valid_heads = [h for h in (4, 8) if channels % h == 0]     # 32/64/128 all divisible by 4 and 8
    nheads = trial.suggest_categorical("nheads", valid_heads)
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),   # capped at 1e-3: lr 3e-3 destabilises diffusion training
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "layers": trial.suggest_categorical("layers", [2, 3, 4]),
        "channels": channels,
        "nheads": nheads,
        "diffusion_embedding_dim": trial.suggest_categorical("diffusion_embedding_dim", [64, 128]),
        "num_steps": trial.suggest_categorical("num_steps", [50, 100, 200]),
        "beta_end": trial.suggest_categorical("beta_end", [0.1, 0.2, 0.5]),
    }


def free(model, opt=None):
    """Free the model + optimizer between trials/seeds (GPU-careful)."""
    try:
        del model
    except Exception:
        pass
    if opt is not None:
        try:
            del opt
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------------------------ #
#  DATA: build runs + vessel-disjoint split + datasets ONCE, then reuse across trials/seeds.                     #
# ------------------------------------------------------------------------------------------------------------ #
def build_datasets(args):
    """Build runs (subsample controlled by --n-mmsi / --max-files-per-part), vessel-disjoint split, z-score on
    train, and the three CSDI datasets.  Returns (train_ds, val_ds, test_ds, mean, std) or None if too few."""
    runs = build_runs(years=(2013,), months=(6, 7), n_mmsi=args.n_mmsi,
                      max_files_per_part=args.max_files_per_part, thin_sec=180, split_gap_h=1.0,
                      min_len=40, min_span_h=args.gap_hours + 4.0, min_sog=1.0, seed=args.seed)
    (train_runs, val_runs, test_runs,
     train_mmsi, val_mmsi, test_mmsi) = split_runs_vessel_disjoint(runs, args.seed)
    print(f"runs: {len(runs)} vessels split -> train {len(train_runs)}/{len(train_mmsi)}v | "
          f"val {len(val_runs)}/{len(val_mmsi)}v | test {len(test_runs)}/{len(test_mmsi)}v", flush=True)

    mean, std = fit_stats(train_runs, args.window, args.gap_hours, seed=1)
    print(f"z-score mean {np.round(mean, 3)} std {np.round(std, 3)}", flush=True)

    train_ds = AISCSDIDataset(train_runs, args.window, args.gap_hours, mean, std, seed=1)
    val_ds = AISCSDIDataset(val_runs, args.window, args.gap_hours, mean, std, seed=3)
    test_ds = AISCSDIDataset(test_runs, args.window, args.gap_hours, mean, std, seed=2)
    print(f"usable samples (contiguous gap): train {len(train_ds)} / val {len(val_ds)} / test {len(test_ds)}",
          flush=True)
    if len(train_ds) < 2 or len(val_ds) < 1 or len(test_ds) < 1:
        print("!! too few runs host a valid two-sided gap -- increase --n-mmsi / months, or lower --gap-hours.",
              flush=True)
        return None
    return train_ds, val_ds, test_ds, mean, std


def make_loaders(train_ds, eval_ds, batch_size):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=min(batch_size, 16), shuffle=False)
    return train_loader, eval_loader


# ------------------------------------------------------------------------------------------------------------ #
#  SWEEP MODE                                                                                                    #
# ------------------------------------------------------------------------------------------------------------ #
def run_sweep(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    ds = build_datasets(args)
    if ds is None:
        print("ABORT: no usable data for the sweep.", flush=True)
        return
    train_ds, val_ds, test_ds, mean, std = ds

    out_dir = os.path.join(HERE, "hp_optuna", f"gap_{int(args.gap_hours)}h")
    os.makedirs(out_dir, exist_ok=True)
    db_path = os.path.join(out_dir, "csdi_hp.db")
    storage = f"sqlite:///{db_path}"

    def objective(trial):
        hp = suggest_hp(trial)
        print(f"\n--- Trial {trial.number} HP: {hp} ---", flush=True)
        cfg = build_config(hp, epochs=args.trial_epochs, itr_per_epoch=args.max_train_steps)
        train_loader, val_loader = make_loaders(train_ds, val_ds, hp["batch_size"])

        # per-trial checkpoint dir so concurrent trials (--n-jobs>1) never clobber each other's best-model file
        trial_out = os.path.join(out_dir, "_ckpt", f"trial_{trial.number}")
        os.makedirs(trial_out, exist_ok=True)

        model = None; opt = None
        try:
            model = CSDI_AIS(cfg, device, target_dim=4).to(device)
            # CSDI's vanilla training recipe + plateau early-stop on the val denoising loss (behaviour of
            # run_csdi_h2h.train_earlystop -- unchanged).  We ignore its returned val-LOSS and instead score
            # the trial on the REAL task metric below.
            train_earlystop(model, cfg["train"], train_loader, val_loader, trial_out,
                            args.trial_patience, device)
            pooled, per_traj = eval_nm(model, val_loader, mean, std, nsample=args.nsample, device=device)
            if len(pooled) == 0 or not np.isfinite(np.median(pooled)):
                print(f"  trial {trial.number}: ZERO valid gaps / non-finite -> sentinel {SENTINEL}",
                      flush=True)
                return SENTINEL
            val_nm = float(np.median(pooled))     # REAL task metric: val median great-circle NM on gap points
            print(f"  trial {trial.number}: VAL median-NM = {val_nm:.4f} "
                  f"(over {len(pooled)} gap points, {len(per_traj)} traj)", flush=True)
            return val_nm
        except RuntimeError as e:
            # e.g. CUDA OOM on a large channels/batch combo -> deprioritise, don't crash the study
            print(f"  trial {trial.number}: RuntimeError ({e}) -> sentinel {SENTINEL}", flush=True)
            return SENTINEL
        finally:
            free(model, opt)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    study = optuna.create_study(
        study_name="csdi_ais_median_nm",
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,                      # RESUMABLE
    )
    already = len([t for t in study.trials if t.state.is_finished()])
    remaining = max(0, args.n_trials - already)
    print(f"Optuna study @ {db_path} | already finished {already} trials | running {remaining} more "
          f"(target total {args.n_trials})", flush=True)

    if remaining > 0:
        study.optimize(objective, n_trials=remaining, n_jobs=args.n_jobs, gc_after_trial=True)

    # persist best params + trials CSV -----------------------------------------------------------------------
    best_params_path = os.path.join(out_dir, "best_params.json")
    best = {
        "best_value_val_median_nm": (float(study.best_value)
                                     if study.best_trial is not None else None),
        "best_trial_number": (study.best_trial.number if study.best_trial is not None else None),
        "best_params": (dict(study.best_params) if study.best_trial is not None else None),
        "gap_hours": args.gap_hours, "window": args.window, "nsample": args.nsample,
        "n_trials_finished": len([t for t in study.trials if t.state.is_finished()]),
    }
    with open(best_params_path, "w") as f:
        json.dump(best, f, indent=2)

    trials_csv = os.path.join(out_dir, "trials.csv")
    df = study.trials_dataframe()
    df.to_csv(trials_csv, index=False)

    print("\n=== SWEEP DONE ===", flush=True)
    print(f"  best VAL median-NM: {best['best_value_val_median_nm']}  (trial {best['best_trial_number']})",
          flush=True)
    print(f"  best params: {best['best_params']}", flush=True)
    print(f"  wrote {best_params_path}", flush=True)
    print(f"  wrote {trials_csv}", flush=True)
    print(f"  study db (resumable): {db_path}", flush=True)


# ------------------------------------------------------------------------------------------------------------ #
#  FINAL MODE: 5 seeds at best HP on FULL train split -> TEST median-NM mean +/- std                            #
# ------------------------------------------------------------------------------------------------------------ #
def run_final(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = os.path.join(HERE, "hp_optuna", f"gap_{int(args.gap_hours)}h")
    best_params_path = os.path.join(out_dir, "best_params.json")
    if not os.path.exists(best_params_path):
        print(f"ABORT: {best_params_path} not found -- run `--mode sweep` first.", flush=True)
        return
    with open(best_params_path) as f:
        best = json.load(f)
    hp = best.get("best_params")
    if hp is None:
        print("ABORT: best_params.json has no best_params (sweep produced no valid trial).", flush=True)
        return
    print(f"Loaded best HP: {hp}  (best VAL median-NM {best.get('best_value_val_median_nm')})", flush=True)

    ds = build_datasets(args)
    if ds is None:
        print("ABORT: no usable data for the final run.", flush=True)
        return
    train_ds, val_ds, test_ds, mean, std = ds

    final_dir = os.path.join(out_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    per_seed = []
    for si, seed in enumerate(range(args.seeds)):
        torch.manual_seed(seed); np.random.seed(seed)
        print(f"\n=== FINAL seed {si + 1}/{args.seeds} (seed={seed}) ===", flush=True)
        cfg = build_config(hp, epochs=args.final_epochs, itr_per_epoch=args.max_train_steps)
        train_loader, val_loader = make_loaders(train_ds, val_ds, hp["batch_size"])
        _, test_loader = make_loaders(train_ds, test_ds, hp["batch_size"])

        model = None
        try:
            model = CSDI_AIS(cfg, device, target_dim=4).to(device)
            print(f"  params: {sum(p.numel() for p in model.parameters())} | device {device} | "
                  f"L={args.window}", flush=True)
            history = []
            best_val_loss = train_earlystop(model, cfg["train"], train_loader, val_loader, final_dir,
                                            args.final_patience, device, history=history)
            # per-seed TRAINING PERFORMANCE curve (epoch, train denoising loss, val denoising loss)
            curve_path = os.path.join(final_dir, f"train_curve_seed{seed}.csv")
            with open(curve_path, "w", newline="") as f:
                w = csv.writer(f); w.writerow(["epoch", "train_loss", "val_loss"])
                for ep, tl, vl in history:
                    w.writerow([ep, round(tl, 6), (round(vl, 6) if np.isfinite(vl) else "")])
            best_epoch = int(min(range(len(history)), key=lambda i: history[i][2]) + 1) if history else -1
            # save this seed's best-val checkpoint under a seed-specific name (train_earlystop wrote model.pth)
            seed_pth = os.path.join(final_dir, f"model_seed{seed}.pth")
            torch.save(model.state_dict(), seed_pth)
            # REAL task metric (median great-circle NM on the gap) on BOTH val and test at the best-val weights
            vpool, vtraj = eval_nm(model, val_loader, mean, std, nsample=args.nsample, device=device)
            val_nm = (float(np.median(vpool)) if len(vpool) and np.isfinite(np.median(vpool)) else float("nan"))
            pooled, per_traj = eval_nm(model, test_loader, mean, std, nsample=args.nsample, device=device)
            test_nm = (float(np.median(pooled)) if len(pooled) and np.isfinite(np.median(pooled))
                       else float("nan"))
            print(f"  seed {seed}: VAL median-NM = {val_nm:.4f} | TEST median-NM = {test_nm:.4f} "
                  f"(best_ep {best_epoch}, best_val_loss {best_val_loss:.4f}) -> saved {seed_pth}", flush=True)
            per_seed.append((seed, best_epoch, best_val_loss, val_nm, test_nm, len(pooled), len(per_traj)))
        finally:
            free(model)

    # per-seed metrics CSV (training summary + val + test) ----------------------------------------------------
    csv_path = os.path.join(final_dir, "final_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "best_epoch", "best_val_loss", "val_median_nm", "test_median_nm",
                    "n_gap_points", "n_traj"])
        for seed, bep, bvl, vnm, tnm, ngp, ntr in per_seed:
            w.writerow([seed, bep, round(bvl, 6), round(vnm, 4), round(tnm, 4), ngp, ntr])

    tvals = np.array([tnm for _, _, _, _, tnm, _, _ in per_seed if np.isfinite(tnm)], dtype=np.float64)
    vvals = np.array([vnm for _, _, _, vnm, _, _, _ in per_seed if np.isfinite(vnm)], dtype=np.float64)
    mean_nm = float(tvals.mean()) if len(tvals) else float("nan")
    std_nm = float(tvals.std(ddof=0)) if len(tvals) else float("nan")

    summary = {
        "hp": hp, "seeds": args.seeds, "gap_hours": args.gap_hours, "window": args.window,
        "nsample": args.nsample,
        "val_median_nm_per_seed": [round(vnm, 4) for _, _, _, vnm, _, _, _ in per_seed],
        "test_median_nm_per_seed": [round(tnm, 4) for _, _, _, _, tnm, _, _ in per_seed],
        "val_median_nm_mean": round(float(vvals.mean()) if len(vvals) else float("nan"), 4),
        "test_median_nm_mean": round(mean_nm, 4), "test_median_nm_std": round(std_nm, 4),
    }
    with open(os.path.join(final_dir, "final_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== FINAL DONE ===", flush=True)
    print(f"  per-seed CSV: {csv_path}", flush=True)
    print(f"  VAL  median-NM per seed: {summary['val_median_nm_per_seed']}", flush=True)
    print(f"  TEST median-NM per seed: {summary['test_median_nm_per_seed']}", flush=True)
    print(f"  TEST median-NM: {mean_nm:.4f} +/- {std_nm:.4f}  (n={len(tvals)} seeds)", flush=True)
    print(f"  wrote {csv_path}", flush=True)
    print(f"  wrote {os.path.join(final_dir, 'final_summary.json')}", flush=True)


# ------------------------------------------------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser(description="Optuna HP sweep for the VANILLA CSDI AIS baseline.")
    ap.add_argument("--mode", choices=["sweep", "final"], required=True)

    # task geometry (shared) ---------------------------------------------------------------------------------
    ap.add_argument("--gap-hours", type=float, default=6.0)
    ap.add_argument("--window", type=int, default=200, help="fixed L (pings/sample after ~180s thinning)")
    ap.add_argument("--nsample", type=int, default=50, help="posterior samples at eval (median over these)")
    ap.add_argument("--seed", type=int, default=42, help="data build + split + TPE sampler seed")
    ap.add_argument("--max-train-steps", type=int, default=0, help="cap optimizer steps/epoch (0 = no cap)")

    # subsample (sweep uses a SMALLER slice for fast trials; final uses the full data by default) -------------
    ap.add_argument("--n-mmsi", type=int, default=400,
                    help="# vessels to sample (None-like via -1 = ALL). Sweep default is a fast subsample.")
    ap.add_argument("--max-files-per-part", type=int, default=2,
                    help="# part files per source partition (-1 = ALL).")

    # sweep budget -------------------------------------------------------------------------------------------
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--trial-epochs", type=int, default=60)
    ap.add_argument("--trial-patience", type=int, default=8)
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="concurrent trials (threads) run on the SAME GPU to fill VRAM. Each trial gets its own "
                         "model/optimizer/loaders and its own checkpoint dir, so trials do not interfere.")

    # final budget -------------------------------------------------------------------------------------------
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--final-epochs", type=int, default=300)
    ap.add_argument("--final-patience", type=int, default=20)
    ap.add_argument("--final-full-data", action="store_true",
                    help="shortcut: use ALL vessels + ALL files for the final run (sets n-mmsi/-1 files/-1).")
    args = ap.parse_args()

    # allow -1 to mean "ALL" (build_runs uses None for that) -------------------------------------------------
    if args.final_full_data:
        args.n_mmsi = -1; args.max_files_per_part = -1
    args.n_mmsi = None if args.n_mmsi is not None and args.n_mmsi < 0 else args.n_mmsi
    args.max_files_per_part = (None if args.max_files_per_part is not None and args.max_files_per_part < 0
                               else args.max_files_per_part)

    if args.mode == "sweep":
        run_sweep(args)
    else:
        run_final(args)


if __name__ == "__main__":
    main()

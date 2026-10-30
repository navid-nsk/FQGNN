"""Optuna hyperparameter-sweep harness for the VANILLA SAITS baseline on the
Arctic-AIS two-sided (contiguous-gap) trajectory-imputation task.

This is an EXTERNAL harness for a research paper in which SAITS is a COMPETITOR
baseline.  It changes NOTHING in the SAITS model (modeling/saits.py, modeling/
layers.py, ...).  It only:
  (1) builds a SAITS-format HDF5 subsample via build_ais_h5.build (unchanged), and
  (2) drives the VANILLA train/impute loop from run_saits_h2h.py (train_saits,
      impute_split, evaluate_nm) while Optuna tunes only the HYPERPARAMETERS the
      SAITS constructor / training loop already expose.

We tune ONLY:
  n_groups, n_group_inner_layers, d_model, d_inner, n_head, d_k, d_v, dropout, lr,
  batch_size, and the MIT artificial-missing rate (mit_rate, default 0.2 -> the
  vanilla behaviour).  The diagonal-attention mask, ORT+MIT loss (weights=1),
  param_sharing_strategy, and input_with_mask stay VANILLA (fixed).

Objective (minimised): VAL median great-circle error in NAUTICAL MILES on the
val split's held-out gap points (the real task metric, via evaluate_nm on the VAL
split -- NOT val MAE in normalized space).

Two modes:
  --mode sweep : 20-trial TPE study (sequential, n_jobs=1) on a SUBSAMPLE with
                 reduced epochs + early stop.  Persists to sqlite (resumable),
                 writes best_params.json + trials CSV.
  --mode final : load best_params.json, train 5 seeds at best HP on the FULL train
                 split, report TEST median-NM mean+/-std, save per-seed CSV + .pt.

Interpreter: the main (gpubench) environment.

Real commands:
  Sweep (20 trials, subsample):
    python hp_saits_optuna.py --mode sweep --n-trials 20 --gap-hours 6 \
        --seq-len 128 --thin-sec 300 --n-mmsi 400 --max-files-per-part 2 \
        --trial-epochs 60 --trial-patience 8
  Final (5 seeds, full data):
    python hp_saits_optuna.py --mode final --gap-hours 6 --seq-len 128 \
        --thin-sec 300 --final-epochs 200 --final-patience 12 --n-seeds 5
"""
import argparse
import copy
import csv
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import optuna

import build_ais_h5 as B
import run_saits_h2h as R  # VANILLA runner: train_saits / impute_split / evaluate_nm / load_split / build_model


# ----------------------------------------------------------------------------- data
def build_dataset(gap_hours, seq_len, thin_sec, n_mmsi, max_files_per_part,
                  years, months, seed, out_dir, rebuild=False):
    """Build (or reuse) a SAITS-format h5 subsample via the UNCHANGED build_ais_h5.build.
    Returns (meta, scaler, splits) where splits = dict(train/val/test) from R.load_split."""
    h5_path = os.path.join(out_dir, "datasets.h5")
    if rebuild or not os.path.exists(h5_path):
        print(f"[data] building dataset -> {out_dir}", flush=True)
        bargs = argparse.Namespace(
            gap_hours=gap_hours, seq_len=seq_len, thin_sec=thin_sec,
            n_mmsi=n_mmsi, max_files_per_part=max_files_per_part,
            years=years, months=months, val_frac=0.15, test_frac=0.15,
            seed=seed, out_dir=out_dir, smoke=False)
        meta = B.build(bargs)
        if meta is None:
            raise RuntimeError("build_ais_h5.build produced no usable dataset "
                               "(increase --n-mmsi / --max-files-per-part or lower seq-len/gap).")
    else:
        print(f"[data] reusing existing dataset at {h5_path}", flush=True)

    with open(os.path.join(out_dir, "meta.json")) as f:
        meta = json.load(f)
    import pickle
    with open(os.path.join(out_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    splits = {
        "train": R.load_split(h5_path, "train"),
        "val": R.load_split(h5_path, "val"),
        "test": R.load_split(h5_path, "test"),
    }
    return meta, scaler, splits


def median_nm(imputed_n, split, scaler):
    """Pooled median great-circle NM over the held-out gap (the task metric).
    Returns np.inf if there are no valid (finite) gap points."""
    pooled, _ = R.evaluate_nm(imputed_n, split, scaler)
    if pooled.size == 0:
        return float("inf")
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        return float("inf")
    return float(np.median(pooled))


# ----------------------------------------------------------------------------- HP space
# Vanilla knobs SAITS already exposes (constructor args / train loop). Names match
# build_model()'s cfg dict and train_saits()'s signature EXACTLY.
def suggest_params(trial):
    return {
        "n_groups": trial.suggest_categorical("n_groups", [1, 2, 3, 4]),
        "n_group_inner_layers": trial.suggest_categorical("n_group_inner_layers", [1, 2]),
        "d_model": trial.suggest_categorical("d_model", [128, 256, 512]),
        "d_inner": trial.suggest_categorical("d_inner", [64, 128, 256]),
        "n_head": trial.suggest_categorical("n_head", [2, 4, 8]),
        "d_k": trial.suggest_categorical("d_k", [16, 32, 64]),
        "d_v": trial.suggest_categorical("d_v", [16, 32, 64]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "mit_rate": trial.suggest_categorical("mit_rate", [0.1, 0.2, 0.3]),
    }


# Fixed VANILLA (non-tuned) model knobs -- kept identical to run_saits_h2h.DEFAULT_CFG.
VANILLA_FIXED = dict(
    param_sharing_strategy="inner_group",
    input_with_mask=True,
    diagonal_attention_mask=True,
)


def make_cfg(params):
    """Merge tuned params + fixed vanilla knobs into the cfg dict build_model expects."""
    cfg = dict(VANILLA_FIXED)
    cfg.update(dict(
        n_groups=params["n_groups"],
        n_group_inner_layers=params["n_group_inner_layers"],
        d_model=params["d_model"],
        d_inner=params["d_inner"],
        n_head=params["n_head"],
        d_k=params["d_k"],
        d_v=params["d_v"],
        dropout=params["dropout"],
    ))
    return cfg


def train_eval_one(params, meta, scaler, splits, device, epochs, patience, seed,
                   eval_split="val", batch_cap=64):
    """Build VANILLA SAITS at `params`, train via R.train_saits (ORT+MIT, val early-stop),
    impute the eval split, return (median_nm, n_params). Frees model+optimizer on return."""
    seq_len = meta["seq_len"]
    n_features = meta["n_features"]
    batch_size = min(int(params["batch_size"]), batch_cap)  # GPU-careful cap

    cfg = make_cfg(params)
    model = R.build_model(seq_len, n_features, device, cfg).to(device)
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)

    val_split = splits["val"] if eval_split != "train_only" else None
    try:
        model = R.train_saits(
            model, splits["train"]["X"], val_split, device,
            epochs=epochs, batch_size=batch_size, lr=float(params["lr"]),
            patience=patience, max_steps=None, seed=seed,
            mit_rate=float(params["mit_rate"]),
        )
        imputed_n = R.impute_split(model, splits[eval_split], device, batch_size)
        mnm = median_nm(imputed_n, splits[eval_split], scaler)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return mnm, int(nparam)


# ----------------------------------------------------------------------------- sweep
def run_sweep(args):
    device = args.device
    out_dir = args.data_dir or os.path.join(
        HERE, "generated_datasets", f"ais_gap_{int(args.gap_hours)}h_hpsub")
    study_dir = args.study_dir or os.path.join(HERE, "hp_saits")
    os.makedirs(study_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Build the SUBSAMPLE once; every trial reuses it (we tune model/training, not data).
    meta, scaler, splits = build_dataset(
        args.gap_hours, args.seq_len, args.thin_sec, args.n_mmsi,
        args.max_files_per_part, args.years, args.months, args.seed, out_dir,
        rebuild=args.rebuild)
    print(f"[sweep] train {splits['train']['X'].shape} | val {splits['val']['X'].shape} "
          f"| test {splits['test']['X'].shape}", flush=True)

    db_path = os.path.join(study_dir, "saits_hp.db")
    storage = f"sqlite:///{db_path}"
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=f"saits_gap{int(args.gap_hours)}h",
        direction="minimize", sampler=sampler,
        storage=storage, load_if_exists=True)

    SENTINEL = 1e6

    def objective(trial):
        params = suggest_params(trial)
        t0 = time.time()
        try:
            mnm, nparam = train_eval_one(
                params, meta, scaler, splits, device,
                epochs=args.trial_epochs, patience=args.trial_patience,
                seed=args.seed, eval_split="val", batch_cap=64)
        except RuntimeError as e:
            # e.g. CUDA OOM for a huge d_model*n_head*d_k combo -> free + sentinel
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[trial {trial.number}] RuntimeError -> sentinel {SENTINEL}: {e}", flush=True)
            trial.set_user_attr("error", str(e)[:300])
            return SENTINEL
        dt = time.time() - t0
        trial.set_user_attr("n_params", nparam)
        trial.set_user_attr("wall_sec", round(dt, 1))
        if not np.isfinite(mnm):
            print(f"[trial {trial.number}] non-finite val median-NM (no valid gap) "
                  f"-> sentinel {SENTINEL}", flush=True)
            trial.set_user_attr("note", "non-finite val median-NM")
            return SENTINEL
        print(f"[trial {trial.number}] val median-NM = {mnm:.4f} | params {nparam} "
              f"| {dt:.1f}s | {params}", flush=True)
        return mnm

    study.optimize(objective, n_trials=args.n_trials, n_jobs=1,
                   gc_after_trial=True)

    # ---- persist best_params.json + trials CSV ----
    best = study.best_trial
    best_params = dict(best.params)
    best_out = {
        "best_value_val_median_nm": float(best.value),
        "best_trial_number": int(best.number),
        "best_params": best_params,
        "fixed_vanilla": VANILLA_FIXED,
        "gap_hours": args.gap_hours,
        "seq_len": args.seq_len,
        "thin_sec": args.thin_sec,
        "sweep_subsample": {
            "n_mmsi": args.n_mmsi, "max_files_per_part": args.max_files_per_part,
            "years": args.years, "months": args.months,
            "trial_epochs": args.trial_epochs, "trial_patience": args.trial_patience,
        },
        "n_trials_completed": len(study.trials),
    }
    bp_path = os.path.join(study_dir, "best_params.json")
    with open(bp_path, "w") as f:
        json.dump(best_out, f, indent=2)

    csv_path = os.path.join(study_dir, "trials.csv")
    keys = ["n_groups", "n_group_inner_layers", "d_model", "d_inner", "n_head",
            "d_k", "d_v", "dropout", "lr", "batch_size", "mit_rate"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "state", "val_median_nm", "n_params", "wall_sec"] + keys)
        for t in study.trials:
            val = "" if t.value is None else round(float(t.value), 5)
            w.writerow(
                [t.number, t.state.name, val,
                 t.user_attrs.get("n_params", ""), t.user_attrs.get("wall_sec", "")]
                + [t.params.get(k, "") for k in keys])

    print("\n=== SAITS Optuna sweep done ===", flush=True)
    print(f"  best val median-NM: {best.value:.4f} (trial {best.number})", flush=True)
    print(f"  best params: {best_params}", flush=True)
    print(f"  wrote {bp_path}", flush=True)
    print(f"  wrote {csv_path}", flush=True)
    print(f"  study db: {db_path}", flush=True)


# ----------------------------------------------------------------------------- final
def run_final(args):
    device = args.device
    study_dir = args.study_dir or os.path.join(HERE, "hp_saits")
    bp_path = os.path.join(study_dir, "best_params.json")
    if not os.path.exists(bp_path):
        raise FileNotFoundError(f"best_params.json not found at {bp_path} -- run --mode sweep first.")
    with open(bp_path) as f:
        best_out = json.load(f)
    params = best_out["best_params"]
    print(f"[final] loaded best params from {bp_path}: {params}", flush=True)

    # FULL data (n_mmsi=None, max_files_per_part=None => all vessels / all files).
    out_dir = args.data_dir or os.path.join(
        HERE, "generated_datasets", f"ais_gap_{int(args.gap_hours)}h_full")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    meta, scaler, splits = build_dataset(
        args.gap_hours, args.seq_len, args.thin_sec,
        n_mmsi=None, max_files_per_part=None,
        years=args.years, months=args.months, seed=args.seed, out_dir=out_dir,
        rebuild=args.rebuild)
    print(f"[final] train {splits['train']['X'].shape} | val {splits['val']['X'].shape} "
          f"| test {splits['test']['X'].shape}", flush=True)

    os.makedirs(study_dir, exist_ok=True)
    rows, test_nms, val_nms = [], [], []
    for si in range(args.n_seeds):
        seed = args.seed + si
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"\n[final] === seed {si + 1}/{args.n_seeds} (seed={seed}) ===", flush=True)

        seq_len = meta["seq_len"]
        n_features = meta["n_features"]
        batch_size = min(int(params["batch_size"]), 64)
        cfg = make_cfg(params)
        model = R.build_model(seq_len, n_features, device, cfg).to(device)
        nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
        t0 = time.time()
        history = []
        try:
            model = R.train_saits(
                model, splits["train"]["X"], splits["val"], device,
                epochs=args.final_epochs, batch_size=batch_size,
                lr=float(params["lr"]), patience=args.final_patience,
                max_steps=None, seed=seed, mit_rate=float(params["mit_rate"]), history=history)
            # save this seed's (best-val) weights BEFORE the impute
            pt_path = os.path.join(study_dir, f"final_seed{seed}.pt")
            torch.save(model.state_dict(), pt_path)
            # REAL task metric (median great-circle NM on the gap) on BOTH val and test
            imputed_val = R.impute_split(model, splits["val"], device, batch_size)
            vnm = median_nm(imputed_val, splits["val"], scaler)
            imputed_n = R.impute_split(model, splits["test"], device, batch_size)
            tnm = median_nm(imputed_n, splits["test"], scaler)
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        dt = time.time() - t0
        # per-seed TRAINING PERFORMANCE curve (epoch, train ORT+MIT loss, val gap-MAE)
        curve_path = os.path.join(study_dir, f"train_curve_seed{seed}.csv")
        with open(curve_path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["epoch", "train_loss", "val_gap_mae"])
            for ep, tl, vm in history:
                w.writerow([ep, round(tl, 6), (round(vm, 6) if np.isfinite(vm) else "")])
        best_epoch = int(min(range(len(history)), key=lambda i: history[i][2]) + 1) if history else -1
        best_val_mae = float(min((h[2] for h in history if np.isfinite(h[2])), default=float("nan")))
        print(f"[final] seed {seed}: VAL median-NM = {vnm:.4f} | TEST median-NM = {tnm:.4f} | {dt:.1f}s "
              f"| best_ep {best_epoch} | params {nparam} | saved {pt_path}", flush=True)
        test_nms.append(tnm); val_nms.append(vnm)
        rows.append({"seed": seed, "best_epoch": best_epoch,
                     "best_val_gap_mae": round(best_val_mae, 6),
                     "val_median_nm": round(float(vnm), 4), "test_median_nm": round(float(tnm), 4),
                     "n_params": int(nparam), "wall_sec": round(dt, 1), "pt_path": pt_path})

    test_nms = np.array(test_nms, float)
    val_nms = np.array(val_nms, float)
    finite = test_nms[np.isfinite(test_nms)]
    mean = float(finite.mean()) if finite.size else float("nan")
    std = float(finite.std(ddof=0)) if finite.size else float("nan")
    vfinite = val_nms[np.isfinite(val_nms)]

    csv_path = os.path.join(study_dir, "final_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "best_epoch", "best_val_gap_mae", "val_median_nm",
                                          "test_median_nm", "n_params", "wall_sec", "pt_path"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary = {
        "model": "SAITS (vanilla, HP-tuned)",
        "best_params": params, "fixed_vanilla": VANILLA_FIXED,
        "gap_hours": args.gap_hours, "seq_len": args.seq_len, "thin_sec": args.thin_sec,
        "n_seeds": args.n_seeds, "final_epochs": args.final_epochs,
        "final_patience": args.final_patience,
        "val_median_nm_per_seed": [round(float(x), 4) for x in val_nms.tolist()],
        "test_median_nm_per_seed": [round(float(x), 4) for x in test_nms.tolist()],
        "val_median_nm_mean": round(float(vfinite.mean()) if vfinite.size else float("nan"), 4),
        "test_median_nm_mean": round(mean, 4), "test_median_nm_std": round(std, 4),
    }
    with open(os.path.join(study_dir, "final_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SAITS (vanilla, HP-tuned) FINAL: 5-seed ===", flush=True)
    print(f"  per-seed VAL  median-NM: {[round(float(x), 4) for x in val_nms.tolist()]}", flush=True)
    print(f"  per-seed TEST median-NM: {[round(float(x), 4) for x in test_nms.tolist()]}", flush=True)
    print(f"  TEST median-NM = {mean:.4f} +/- {std:.4f}  (mean +/- std, n={args.n_seeds})", flush=True)
    print(f"  wrote {csv_path}", flush=True)
    print(f"  wrote {os.path.join(study_dir, 'final_summary.json')}", flush=True)


# ----------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["sweep", "final"], required=True)

    # shared task/data knobs
    ap.add_argument("--gap-hours", type=float, default=6.0)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--thin-sec", type=int, default=300)
    ap.add_argument("--years", type=int, nargs="+", default=[2013])
    ap.add_argument("--months", type=int, nargs="+", default=[6, 7])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data-dir", type=str, default=None,
                    help="dir for datasets.h5/scaler.pkl (default derived from mode+gap)")
    ap.add_argument("--study-dir", type=str, default=None,
                    help="dir for saits_hp.db / best_params.json / CSVs (default: ./hp_saits)")
    ap.add_argument("--rebuild", action="store_true", help="force rebuild of the h5")

    # sweep-only
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--n-mmsi", type=int, default=400,
                    help="sweep SUBSAMPLE: vessels to sample (None-like via -1 = all)")
    ap.add_argument("--max-files-per-part", type=int, default=2,
                    help="sweep SUBSAMPLE: part files per (msg,year,month)")
    ap.add_argument("--trial-epochs", type=int, default=60)
    ap.add_argument("--trial-patience", type=int, default=8)

    # final-only
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--final-epochs", type=int, default=200)
    ap.add_argument("--final-patience", type=int, default=12)

    args = ap.parse_args()
    if args.n_mmsi is not None and args.n_mmsi < 0:
        args.n_mmsi = None  # allow -1 => all vessels in the sweep subsample

    print(f"[hp_saits] mode={args.mode} device={args.device} gap={args.gap_hours}h "
          f"seq_len={args.seq_len} thin_sec={args.thin_sec}", flush=True)

    if args.mode == "sweep":
        run_sweep(args)
    else:
        run_final(args)


if __name__ == "__main__":
    main()

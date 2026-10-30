"""EXTERNAL head-to-head runner (NOT part of SAITS): train the VANILLA SAITS model
(imported unchanged from modeling.saits.SAITS) on OUR Arctic AIS contiguous-gap dataset,
impute the held-out test gap, denormalize lat/lon with the saved StandardScaler, and
report the MEDIAN great-circle error in NAUTICAL MILES on the gap points.

We change NOTHING in the SAITS repo. We only:
  (1) build/read a SAITS-format HDF5 (via build_ais_h5.py), and
  (2) drive a minimal train/impute loop around the vanilla SAITS class + its own
      forward()/impute() (which implement MIT + ORT exactly as the paper).

Vanilla-fidelity notes:
  - Loss = ORT reconstruction + MIT imputation, taken straight from SAITS.forward().
  - Training uses SAITS's intended MIT random masking (we re-mask 20% of observed points
    per step, exactly like modeling.unified_dataloader.LoadTrainDataset). Test/val gaps are
    the CONTIGUOUS task gap (built into the h5). => fair train-random / test-contiguous,
    the setup the SAITS authors used.
  - Model hyperparameters mirror configs/PhysioNet2012_SAITS_base.ini.

CLI:
  python run_saits_h2h.py --gap-hours 6 --batch-size 8 --epochs 100 [--smoke]

Outputs (runs_h2h/gap_<H>h/): per_traj_nm.csv, summary.json
"""
import argparse
import csv
import json
import os
import sys

import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from modeling.saits import SAITS  # VANILLA SAITS model, imported unchanged

import build_ais_h5 as B  # our external builder (haversine_nm, FEATURES, indices)

LAT, LON = B.LAT, B.LON
FEATURES = B.FEATURES
haversine_nm = B.haversine_nm


# ---------- data ----------
def load_split(h5_path, name):
    with h5py.File(h5_path, "r") as hf:
        g = hf[name]
        d = {"X": g["X"][:]}
        if name in ("val", "test"):
            d["X_hat"] = g["X_hat"][:]
            d["missing_mask"] = g["missing_mask"][:]
            d["indicating_mask"] = g["indicating_mask"][:]
    return d


def mit_mask_batch(X, rate=0.2, rng=None):
    """SAITS-style Masked Imputation Task: randomly hold out `rate` of the observed values.
    X: [B,L,F] complete (train) tensor. Returns (X_hat, missing_mask, indicating_mask) as in
    modeling.unified_dataloader.LoadTrainDataset.__getitem__ (vanilla behavior, re-implemented
    only because we drive our own loop -- we do NOT modify SAITS files)."""
    rng = np.random.RandomState() if rng is None else rng
    B_, L, F = X.shape
    Xf = X.reshape(B_, -1)
    X_hat = Xf.copy()
    for b in range(B_):
        obs = np.where(~np.isnan(Xf[b]))[0]
        if len(obs) == 0:
            continue
        sel = rng.choice(obs, round(len(obs) * rate), replace=False) if len(obs) > 1 else obs
        X_hat[b, sel] = np.nan
    missing_mask = (~np.isnan(X_hat)).astype(np.float32)
    indicating_mask = ((~np.isnan(Xf)) ^ (~np.isnan(X_hat))).astype(np.float32)
    X_out = np.nan_to_num(Xf).reshape(B_, L, F)
    X_hat = np.nan_to_num(X_hat).reshape(B_, L, F)
    return (X_hat, missing_mask.reshape(B_, L, F), indicating_mask.reshape(B_, L, F), X_out)


# ---------- model ----------
def build_model(seq_len, n_features, device, cfg):
    """Construct vanilla SAITS with model_args mirroring run_models.py / SAITS base config."""
    model_args = {
        "device": device,
        "MIT": True,
        "n_groups": cfg["n_groups"],
        "n_group_inner_layers": cfg["n_group_inner_layers"],
        "d_time": seq_len,
        "d_feature": n_features,
        "dropout": cfg["dropout"],
        "d_model": cfg["d_model"],
        "d_inner": cfg["d_inner"],
        "n_head": cfg["n_head"],
        "d_k": cfg["d_k"],
        "d_v": cfg["d_v"],
        "input_with_mask": cfg["input_with_mask"],
        "diagonal_attention_mask": cfg["diagonal_attention_mask"],
        "param_sharing_strategy": cfg["param_sharing_strategy"],
    }
    return SAITS(**model_args)


DEFAULT_CFG = dict(  # from configs/PhysioNet2012_SAITS_base.ini
    n_groups=2, n_group_inner_layers=1, param_sharing_strategy="inner_group",
    d_model=256, d_inner=128, n_head=4, d_k=64, d_v=64, dropout=0.1,
    input_with_mask=True, diagonal_attention_mask=True,
)
SMOKE_CFG = dict(  # tiny model for the smoke test
    n_groups=1, n_group_inner_layers=1, param_sharing_strategy="inner_group",
    d_model=32, d_inner=32, n_head=2, d_k=16, d_v=16, dropout=0.1,
    input_with_mask=True, diagonal_attention_mask=True,
)


@torch.no_grad()
def val_gap_mae(model, split, device, batch_size):
    """Val imputation quality = mean-abs-error of imputed lat/lon at the CONTIGUOUS gap (indicating_mask),
    in normalized space (fine for early-stop tracking).  SAITS isn't diffusion, so this directly reflects the
    imputation quality -- no diffusion sample-quality caveat."""
    imputed = impute_split(model, split, device, batch_size)          # [N,L,F] normalized
    X = split["X"].astype(np.float32)                                 # truth (normalized), NaN where naturally missing
    gap = split["indicating_mask"].astype(bool)[..., LAT]             # [N,L] held-out gap positions
    errs = []
    for b in range(imputed.shape[0]):
        m = gap[b]
        if m.sum() == 0:
            continue
        e = np.abs(imputed[b][m][:, [LAT, LON]] - X[b][m][:, [LAT, LON]])
        e = e[~np.isnan(e)]
        if e.size:
            errs.append(float(e.mean()))
    return float(np.mean(errs)) if errs else 1e18


def train_saits(model, Xtr, val_split, device, epochs, batch_size, lr, patience,
                max_steps=None, seed=42, log_every=50, mit_rate=0.2, history=None):
    """Vanilla SAITS train loop (ORT+MIT loss from SAITS.forward) PLUS plateau early-stop on the val gap-MAE.
    Restores the best-val weights.  If no val split, trains fixed epochs and keeps the final weights."""
    import copy
    rng = np.random.RandomState(seed)
    ds = TensorDataset(torch.from_numpy(Xtr.astype(np.float32)))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, no_imp, step = 1e18, None, 0, 0
    for ep in range(epochs):
        model.train(); tot = 0.0; nb = 0
        for (xb,) in dl:
            X = xb.numpy()
            X_hat, mm, im, X_gt = mit_mask_batch(X, rate=mit_rate, rng=rng)
            inputs = {
                "indices": torch.arange(X.shape[0]),
                "X": torch.from_numpy(X_hat).to(device),
                "missing_mask": torch.from_numpy(mm).to(device),
                "X_holdout": torch.from_numpy(X_gt).to(device),
                "indicating_mask": torch.from_numpy(im).to(device),
            }
            out = model(inputs, "train")
            loss = out["reconstruction_loss"] + out["imputation_loss"]    # ORT+MIT, weights=1 (vanilla)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1; step += 1
            if max_steps is not None and step >= max_steps:
                print(f"  [train] hit max_steps={max_steps} (smoke).", flush=True)
                best_state = copy.deepcopy(model.state_dict())
                model.load_state_dict(best_state); return model
        if val_split is None:
            best_state = copy.deepcopy(model.state_dict())
            if history is not None:
                history.append((ep + 1, tot / max(nb, 1), float("nan")))
            print(f"  ep {ep+1}/{epochs} train_loss {tot/max(nb,1):.4f} (no val)", flush=True)
            continue
        vmae = val_gap_mae(model, val_split, device, batch_size)
        if history is not None:
            history.append((ep + 1, tot / max(nb, 1), vmae))
        improved = vmae < best - 1e-6
        print(f"  ep {ep+1}/{epochs} train_loss {tot/max(nb,1):.4f} | val_gap_MAE {vmae:.5f}"
              + ("" if improved else f" (no-improve {no_imp+1}/{patience})"), flush=True)
        if improved:
            best, best_state, no_imp = vmae, copy.deepcopy(model.state_dict()), 0
        else:
            no_imp += 1
        if patience and no_imp >= patience:
            print(f"  EARLY STOP @ ep {ep+1} (val gap-MAE plateaued {patience} ep; best {best:.5f})", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def impute_split(model, split, device, batch_size):
    """Run vanilla SAITS.impute on X_hat (gap=NaN->0, missing_mask marks observed). Returns the
    full imputed array [N,L,F] in ORIGINAL order (no shuffle)."""
    model.eval()
    X_hat = np.nan_to_num(split["X_hat"]).astype(np.float32)
    mm = split["missing_mask"].astype(np.float32)
    N = X_hat.shape[0]
    outs = []
    for i in range(0, N, batch_size):
        xb = torch.from_numpy(X_hat[i:i + batch_size]).to(device)
        mb = torch.from_numpy(mm[i:i + batch_size]).to(device)
        imputed, _ = model.impute({"X": xb, "missing_mask": mb})
        outs.append(imputed.cpu().numpy())
    return np.concatenate(outs, axis=0)


def evaluate_nm(imputed_n, split, scaler):
    """Denormalize imputations + ground truth, compute per-position and per-traj median NM on the
    held-out gap (indicating_mask==1)."""
    N, L, F = imputed_n.shape
    inv = lambda A: scaler.inverse_transform(A.reshape(-1, F)).reshape(N, L, F)
    imp = inv(imputed_n)
    gt = inv(split["X"])  # ground truth (already denormalized back to degrees)
    ind = split["indicating_mask"].astype(bool)
    gap = ind[..., LAT]  # gap positions (same across features)

    pooled, per_traj = [], []
    for b in range(N):
        m = gap[b]
        if m.sum() == 0:
            continue
        nm = haversine_nm(imp[b][m, LAT], imp[b][m, LON], gt[b][m, LAT], gt[b][m, LON])
        pooled.extend(nm.tolist())
        per_traj.append(float(np.median(nm)))
    return np.array(pooled), np.array(per_traj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-hours", type=float, default=6.0)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--thin-sec", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=12, help="plateau early-stop on val gap-MAE (0=off)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-mmsi", type=int, default=None)
    ap.add_argument("--max-files-per-part", type=int, default=None)
    ap.add_argument("--years", type=int, nargs="+", default=[2013])
    ap.add_argument("--months", type=int, nargs="+", default=[6, 7])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", type=str, default=None,
                    help="dir with datasets.h5+scaler.pkl; if absent it is built here")
    ap.add_argument("--rebuild", action="store_true", help="force rebuild of the h5")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end check: ~60 vessels, small model, few steps")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # smoke uses shorter windows / denser thinning so the tiny 250-vessel sample yields
    # enough contiguous-gap windows for a vessel-disjoint split (see build_ais_h5 diagnostics).
    if args.smoke:
        if "--seq-len" not in sys.argv:
            args.seq_len = 48
        if "--thin-sec" not in sys.argv:
            args.thin_sec = 120

    tag = "smoke" if args.smoke else "full"
    data_dir = args.data_dir or os.path.join(
        HERE, "generated_datasets", f"ais_gap_{int(args.gap_hours)}h_{tag}")
    h5_path = os.path.join(data_dir, "datasets.h5")

    # ---- build dataset if needed ----
    if args.rebuild or not os.path.exists(h5_path):
        print(f"[run] building dataset -> {data_dir}", flush=True)
        bargs = argparse.Namespace(
            gap_hours=args.gap_hours, seq_len=args.seq_len, thin_sec=args.thin_sec,
            n_mmsi=args.n_mmsi, max_files_per_part=args.max_files_per_part,
            years=args.years, months=args.months, val_frac=0.15, test_frac=0.15,
            seed=args.seed, out_dir=data_dir, smoke=args.smoke)
        meta = B.build(bargs)
        if meta is None:
            print("[run] build produced no usable dataset -- aborting.", flush=True)
            return
    else:
        print(f"[run] reusing existing dataset at {h5_path}", flush=True)

    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    seq_len = meta["seq_len"]
    n_features = meta["n_features"]
    import pickle
    with open(os.path.join(data_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    train_split = load_split(h5_path, "train")
    val_split = load_split(h5_path, "val")
    test_split = load_split(h5_path, "test")
    Xtr = train_split["X"]
    print(f"[run] train {Xtr.shape} | val {val_split['X'].shape} | test {test_split['X'].shape}", flush=True)

    device = args.device
    cfg = SMOKE_CFG if args.smoke else DEFAULT_CFG
    cfg = dict(cfg)
    cfg["d_k"] = cfg["d_model"] // cfg["n_head"]
    cfg["d_v"] = cfg["d_model"] // cfg["n_head"]
    model = build_model(seq_len, n_features, device, cfg).to(device)
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[run] SAITS (vanilla) params: {nparam} | device {device} | cfg {cfg}", flush=True)

    epochs = 1 if args.smoke else args.epochs
    max_steps = 3 if args.smoke else None
    batch_size = min(args.batch_size, 8) if args.smoke else args.batch_size
    patience = 0 if args.smoke else args.patience
    vs = None if args.smoke else val_split

    print(f"[run] training: epochs={epochs} batch={batch_size} patience={patience} lr={args.lr}", flush=True)
    model = train_saits(model, Xtr, vs, device, epochs, batch_size, args.lr, patience,
                        max_steps=max_steps, seed=args.seed)

    print("[run] imputing test gap ...", flush=True)
    imputed_n = impute_split(model, test_split, device, batch_size)
    pooled, per_traj = evaluate_nm(imputed_n, test_split, scaler)

    out_dir = os.path.join(HERE, "runs_h2h", f"gap_{int(args.gap_hours)}h" + ("_smoke" if args.smoke else ""))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "per_traj_nm.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["traj", "median_nm"])
        for i, v in enumerate(per_traj):
            w.writerow([i, round(float(v), 4)])
    summary = {
        "model": "SAITS (vanilla)", "gap_hours": args.gap_hours, "smoke": args.smoke,
        "seq_len": seq_len, "thin_sec": meta["thin_sec"], "n_features": n_features,
        "n_test_traj": int(len(per_traj)), "n_gap_points": int(len(pooled)),
        "median_nm_pooled": round(float(np.median(pooled)), 4) if len(pooled) else None,
        "mean_nm_pooled": round(float(np.mean(pooled)), 4) if len(pooled) else None,
        "median_of_traj_medians_nm": round(float(np.median(per_traj)), 4) if len(per_traj) else None,
        "n_params": int(nparam), "batch_size": batch_size, "epochs": epochs,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SAITS (vanilla) on our Arctic AIS, contiguous gap ===", flush=True)
    print(f"  gap {args.gap_hours}h | test traj {len(per_traj)} | gap points {len(pooled)}", flush=True)
    print(f"  MEDIAN NM (pooled): {summary['median_nm_pooled']}   mean {summary['mean_nm_pooled']}", flush=True)
    print(f"  median of per-traj medians: {summary['median_of_traj_medians_nm']}", flush=True)
    print(f"saved -> {out_dir}/per_traj_nm.csv + summary.json", flush=True)
    if args.smoke:
        print("\n[SMOKE OK] end-to-end ran error-free. Do NOT read these numbers as results "
              "(1 epoch / 3 steps / tiny model / 60 vessels).", flush=True)


if __name__ == "__main__":
    main()

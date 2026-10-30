"""EXTERNAL head-to-head runner (NOT part of CSDI's model): train the VANILLA CSDI (CSDI_base + its own
diff_CSDI backbone, and its OWN train()/impute()) on OUR Arctic AIS, with a CONTIGUOUS two-sided lat/lon/sog/cog
gap (the task definition), and read out MEDIAN great-circle position error in NAUTICAL MILES on the gap points.

We change NOTHING in CSDI's model/training/eval code (main_model.py / diff_models.py / utils.py / config).
Everything task-specific lives in THIS file:
  (1) the data source  -- our task_alpha.data_raw -> fixed-L windows, K=4 = [lat, lon, sog, cog];
  (2) the MASK pattern -- a CONTIGUOUS time-window gap (not CSDI's random masks) for BOTH train and test;
  (3) a per-feature z-score (fit on OBSERVED non-gap TRAIN values) so we can invert lat/lon to degrees.

CSDI subclass:  CSDI_AIS mirrors CSDI_Physio.process_data (permute [B,L,K]->[B,K,L]) with target_dim=4, and
overrides forward() ONLY to condition on our fixed contiguous gt_mask during training (cond_mask = gt_mask),
so the model is trained on-task instead of on CSDI's default random masks. No CSDI method is edited.

Batch dict (float32): observed_data [B,L,K], observed_mask [B,L,K] (1=real ping), gt_mask [B,L,K]
(1=condition-on, 0=impute-target = the gap), timepoints [L] = arange(L).  We follow dataset_physio.py:
observed_mask=1 where a real ping exists; gt_mask=1 everywhere EXCEPT the gap (gap gt_mask=0 => imputed);
target = observed_mask - cond_mask (CSDI convention).  Padded tail positions: observed_mask=0, gt_mask=0.

Usage (SMOKE only -- proves the pipeline runs; result is meaningless/undertrained):
    python run_csdi_h2h.py --smoke
Full run (one gap length per invocation):
    python run_csdi_h2h.py --gap-hours 6 --batch-size 8 --epochs 200
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                           # so `main_model` / `utils` resolve to CSDI's
FQGNN = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # repository root
if FQGNN not in sys.path:
    sys.path.insert(0, FQGNN)

from main_model import CSDI_base                                   # CSDI (vanilla) -- we only SUBCLASS it
from utils import train                                           # CSDI's OWN training loop (unchanged)
from task_alpha.data_raw import load_raw, _ts_seconds             # our raw AIS loader (read-only)

# feature order in K=4 -----------------------------------------------------------------------------------------
LAT, LON, SOG, COG = 0, 1, 2, 3
FEATS = ["latitude", "longitude", "sog", "cog"]
R_KM, KM_PER_NM = 6371.0, 1.852


def haversine_nm(lat1, lon1, lat2, lon2):
    r = math.pi / 180.0
    dlat = (lat2 - lat1) * r; dlon = (lon2 - lon1) * r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1 * r) * np.cos(lat2 * r) * np.sin(dlon / 2) ** 2
    return 2 * R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1))) / KM_PER_NM


# ------------------------------------------------------------------------------------------------------------- #
#  DATA: build fixed-L, K=4 windows with a contiguous two-sided gap (pattern from MH-GIN's adapter/preprocess)   #
# ------------------------------------------------------------------------------------------------------------- #
def build_runs(years, months, n_mmsi, max_files_per_part, thin_sec=180, split_gap_h=1.0,
               min_len=40, min_span_h=10.0, min_sog=1.0, seed=0):
    """Per-vessel: split at >split_gap_h coverage gaps, thin to ~thin_sec spacing, keep MOVING runs long enough
    to hold the gap + two-sided anchors.  Returns list of runs: dict(mmsi, t[s], lat, lon, sog, cog)."""
    df = load_raw(years=years, months=months, n_mmsi=n_mmsi, max_files_per_part=max_files_per_part,
                  seed=seed, clean_only=True)
    df["t_s"] = _ts_seconds(df["time"])
    runs = []
    for mmsi, g in df.groupby("mmsi"):
        mmsi = int(mmsi)
        g = g.sort_values("t_s").reset_index(drop=True)
        t = g["t_s"].values.astype(np.float64)
        brk = np.where(np.diff(t) > split_gap_h * 3600)[0]         # split at coverage gaps
        for s in np.split(np.arange(len(g)), brk + 1):
            if len(s) < min_len:
                continue
            keep, last = [], -1e18                                 # thin to >= thin_sec spacing
            for i in s:
                if t[i] - last >= thin_sec:
                    keep.append(i); last = t[i]
            if len(keep) < 12:
                continue
            keep = np.array(keep)
            tt = t[keep]
            sog = g["sog"].values[keep].astype(np.float32)
            if (tt[-1] - tt[0]) < min_span_h * 3600 or np.nanmedian(sog) < min_sog:
                continue                                            # need span + genuinely MOVING
            cog = g["cog"].values[keep].astype(np.float32)
            cog = np.where(np.isnan(cog), 0.0, cog)                # invalid course -> 0 (kept as observed)
            sog = np.where(np.isnan(sog), 0.0, sog)
            runs.append(dict(
                mmsi=mmsi,
                t=tt.astype(np.float64),
                lat=g["latitude"].values[keep].astype(np.float32),
                lon=g["longitude"].values[keep].astype(np.float32),
                sog=sog, cog=cog,
            ))
    return runs


def place_gap(run, L, gap_hours, rng):
    """Take the first L pings of a run; place a contiguous TIME-window gap of gap_hours in the middle (with a
    15% margin on each side so anchors exist BEFORE and AFTER).  Returns (values[L,4], obs_mask[L,4],
    gt_mask[L,4], n_real, gap_bool[L]) or None if the run can't host a valid two-sided gap.  values/masks are
    padded with zeros beyond n_real (padded tail: obs=0, gt=0 => never scored, never conditioned)."""
    n = min(len(run["t"]), L)
    ts = run["t"][:n]
    values = np.zeros((L, 4), np.float32)
    obs_mask = np.zeros((L, 4), np.float32)
    gt_mask = np.zeros((L, 4), np.float32)
    values[:n, LAT] = run["lat"][:n]; values[:n, LON] = run["lon"][:n]
    values[:n, SOG] = run["sog"][:n]; values[:n, COG] = run["cog"][:n]
    obs_mask[:n, :] = 1.0                                          # every real ping is observed
    gt_mask[:n, :] = 1.0                                          # condition on everything...

    gap_sec = gap_hours * 3600.0
    span = ts[-1] - ts[0]
    if n < 8 or span <= gap_sec * 1.3:
        return None                                               # too short to host gap + anchors
    margin = 0.15 * span
    lo = ts[0] + margin
    hi = ts[0] + span - margin - gap_sec
    if hi <= lo:
        return None
    gstart = rng.uniform(lo, hi)
    gap = (ts >= gstart) & (ts < gstart + gap_sec)
    if gap.sum() < 2 or (n - gap.sum()) < 4:                       # need >=2 gap points and two-sided anchors
        return None
    gap_full = np.zeros(L, bool); gap_full[:n] = gap
    gt_mask[gap_full, :] = 0.0                                     # ...except the gap => impute target
    return values, obs_mask, gt_mask, n, gap_full


class AISCSDIDataset(Dataset):
    """CSDI-format samples with a contiguous gap.  Yields observed_data[L,K], observed_mask[L,K],
    gt_mask[L,K], timepoints[L].  Values are z-scored per feature with the SUPPLIED (train) stats."""

    def __init__(self, runs, L, gap_hours, mean, std, seed):
        self.L = L; self.mean = mean.astype(np.float32); self.std = std.astype(np.float32)
        rng = np.random.RandomState(seed)
        self.samples = []
        self.meta = []                                            # (mmsi, gap_bool, n_real) for eval
        for run in runs:
            out = place_gap(run, L, gap_hours, rng)
            if out is None:
                continue
            values, obs_mask, gt_mask, n_real, gap = out
            norm = (values - self.mean) / self.std * obs_mask     # z-score; zero on padded/unobserved
            self.samples.append((norm.astype(np.float32), obs_mask, gt_mask))
            self.meta.append((run["mmsi"], gap, n_real))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        norm, obs_mask, gt_mask = self.samples[i]
        return {
            "observed_data": norm,
            "observed_mask": obs_mask,
            "gt_mask": gt_mask,
            "timepoints": np.arange(self.L).astype(np.float32),
        }


def split_runs_vessel_disjoint(runs, seed, test_frac=0.15, val_frac=0.15):
    """VESSEL-DISJOINT split (test/val vessels unseen in train).  Pure extraction of main()'s split logic --
    identical behaviour: sort mmsis, shuffle with RandomState(seed), take the first 15% vessels as test, the
    next 15% as val, the rest as train.  Returns (train_runs, val_runs, test_runs, train_mmsi, val_mmsi,
    test_mmsi)."""
    mmsis = sorted({r["mmsi"] for r in runs})
    rng = np.random.RandomState(seed); rng.shuffle(mmsis)
    n_test = max(1, int(round(test_frac * len(mmsis))))
    n_val = max(1, int(round(val_frac * len(mmsis))))
    test_mmsi = set(mmsis[:n_test]); val_mmsi = set(mmsis[n_test:n_test + n_val])
    train_mmsi = set(mmsis[n_test + n_val:])
    train_runs = [r for r in runs if r["mmsi"] in train_mmsi]
    val_runs = [r for r in runs if r["mmsi"] in val_mmsi]
    test_runs = [r for r in runs if r["mmsi"] in test_mmsi]
    return train_runs, val_runs, test_runs, train_mmsi, val_mmsi, test_mmsi


def fit_stats(runs, L, gap_hours, seed):
    """Per-feature z-score mean/std over OBSERVED, NON-GAP values of the TRAIN runs (never the test set)."""
    rng = np.random.RandomState(seed)
    acc = [[] for _ in range(4)]
    for run in runs:
        out = place_gap(run, L, gap_hours, rng)
        if out is None:
            continue
        values, obs_mask, gt_mask, n_real, gap = out
        cond = (gt_mask[:, 0] > 0.5)                              # condition points (observed, not gap, not pad)
        for k in range(4):
            acc[k].append(values[cond, k])
    mean = np.zeros(4, np.float64); std = np.ones(4, np.float64)
    for k in range(4):
        if acc[k]:
            v = np.concatenate(acc[k])
            mean[k] = float(v.mean())
            s = float(v.std())
            std[k] = s if s > 1e-6 else 1.0
    return mean, std


# ------------------------------------------------------------------------------------------------------------- #
#  MODEL: subclass CSDI_base (mirror CSDI_Physio.process_data) -- train on our fixed CONTIGUOUS gt_mask          #
# ------------------------------------------------------------------------------------------------------------- #
class CSDI_AIS(CSDI_base):
    """target_dim=4.  process_data mirrors CSDI_Physio verbatim (permute [B,L,K]->[B,K,L]).  forward() is
    overridden ONLY so BOTH train and eval condition on our contiguous gt_mask (on-task), instead of CSDI's
    default random masks.  No CSDI method is modified -- this is a pure subclass.  process_data and forward
    follow CSDI_Physio.process_data / CSDI_base.forward of CSDI (MIT License; see
    task_alpha/baselines/THIRD_PARTY.md)."""

    def __init__(self, config, device, target_dim=4):
        super(CSDI_AIS, self).__init__(target_dim, config, device)

    def process_data(self, batch):                                # identical to CSDI_Physio.process_data
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask
        return observed_data, observed_mask, observed_tp, gt_mask, for_pattern_mask, cut_length

    def forward(self, batch, is_train=1):
        (observed_data, observed_mask, observed_tp, gt_mask, _, _) = self.process_data(batch)
        cond_mask = gt_mask                                       # ON-TASK: condition on the contiguous mask
        side_info = self.get_side_info(observed_tp, cond_mask)
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)


# ------------------------------------------------------------------------------------------------------------- #
#  EVAL: median-over-nsample -> denormalize lat/lon -> haversine NM on gap positions                             #
# ------------------------------------------------------------------------------------------------------------- #
@torch.no_grad()
def eval_nm(model, loader, mean, std, nsample, device):
    model.eval()
    pooled, per_traj = [], []
    mlat, mlon = float(mean[LAT]), float(mean[LON])
    slat, slon = float(std[LAT]), float(std[LON])
    for batch in loader:
        samples, observed_data, target_mask, observed_mask, observed_tp = model.evaluate(batch, nsample)
        # samples [B,nsample,K,L] ; target_mask [B,K,L] = observed_mask - cond_mask (the gap points)
        med = samples.median(dim=1).values                       # [B,K,L]  (normalized)
        med = med.cpu().numpy(); tgt = observed_data.cpu().numpy(); tm = target_mask.cpu().numpy()
        B, K, L = med.shape
        for b in range(B):
            gp = tm[b, LON] > 0.5                                 # gap positions (same across K)
            if gp.sum() == 0:
                continue
            pred_lat = med[b, LAT, gp] * slat + mlat              # denormalize back to degrees
            pred_lon = med[b, LON, gp] * slon + mlon
            true_lat = tgt[b, LAT, gp] * slat + mlat
            true_lon = tgt[b, LON, gp] * slon + mlon
            nm = haversine_nm(pred_lat, pred_lon, true_lat, true_lon)
            pooled.extend(nm.tolist())
            per_traj.append(float(np.median(nm)))
    return np.array(pooled), np.array(per_traj)


# ------------------------------------------------------------------------------------------------------------- #
def build_config(args):
    """CSDI base config (mirrors config/base.yaml) with our train knobs.  We do NOT read the yaml so nothing in
    CSDI's config dir is touched; values match base.yaml except epochs/batch/itr from CLI."""
    return {
        "train": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": 1.0e-3,
            "itr_per_epoch": args.max_train_steps if args.max_train_steps else int(1e8),
        },
        "diffusion": {
            "layers": 4, "channels": 64, "nheads": 8, "diffusion_embedding_dim": 128,
            "beta_start": 0.0001, "beta_end": 0.5,
            "num_steps": 10 if args.smoke else 50,               # fewer diffusion steps in smoke (faster impute)
            "schedule": "quad", "is_linear": False,
        },
        "model": {
            "is_unconditional": 0, "timeemb": 128, "featureemb": 16, "target_strategy": "random",
        },
    }


def train_earlystop(model, cfg, train_loader, val_loader, out_dir, patience, device, history=None):
    """CSDI's VANILLA model trained with its own recipe (Adam wd=1e-6 + MultiStepLR at 0.75/0.9 epochs) PLUS
    plateau early-stopping on the val denoising loss (calc_loss_valid, is_train=0).  If no val split, trains the
    full epochs and keeps the final weights.  Restores + saves the best-val checkpoint to out_dir/model.pth."""
    import copy
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-6)
    p1, p2 = int(0.75 * cfg["epochs"]), int(0.9 * cfg["epochs"])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[p1, p2], gamma=0.1)
    itr_cap = cfg.get("itr_per_epoch", int(1e8))
    best, best_state, no_imp = 1e18, None, 0
    for ep in range(cfg["epochs"]):
        model.train(); tot = 0.0; nb = 0
        for bi, batch in enumerate(train_loader, 1):
            opt.zero_grad(); loss = model(batch); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
            if bi >= itr_cap:
                break
        sched.step()
        if val_loader is None:                                    # no val split -> train fixed epochs, keep final
            best_state = copy.deepcopy(model.state_dict())
            if history is not None:
                history.append((ep + 1, tot / max(nb, 1), float("nan")))
            print(f"  ep {ep+1}/{cfg['epochs']} train {tot/max(nb,1):.4f} (no val)", flush=True)
            continue
        model.eval(); vt = 0.0; vn = 0
        with torch.no_grad():
            for vb in val_loader:
                vt += model(vb, is_train=0).item(); vn += 1
        val = vt / max(vn, 1)
        if history is not None:
            history.append((ep + 1, tot / max(nb, 1), val))
        improved = val < best - 1e-4
        print(f"  ep {ep+1}/{cfg['epochs']} train {tot/max(nb,1):.4f} | val {val:.4f}"
              + ("" if improved else f" (no-improve {no_imp+1}/{patience})"), flush=True)
        if improved:
            best, best_state, no_imp = val, copy.deepcopy(model.state_dict()), 0
        else:
            no_imp += 1
        if patience and no_imp >= patience:
            print(f"  EARLY STOP @ ep {ep+1} (val plateaued {patience} ep; best val {best:.4f})", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pth"))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-hours", type=float, default=6.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=20,
                    help="plateau early-stop: val-loss epochs w/o improvement (0=off; generous for diffusion)")
    ap.add_argument("--nsample", type=int, default=50)           # posterior samples at eval (median over these)
    ap.add_argument("--window", type=int, default=200)           # fixed L (pings per sample after ~180s thinning)
    ap.add_argument("--n-mmsi", type=int, default=None)
    ap.add_argument("--max-files-per-part", type=int, default=None)
    ap.add_argument("--max-train-steps", type=int, default=0, help="cap optimizer steps/epoch (0 = no cap)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end sanity run (undertrained)")
    args = ap.parse_args()

    if args.smoke:                                               # tiny + brief; proves the pipeline runs
        args.n_mmsi = args.n_mmsi or 60
        args.max_files_per_part = args.max_files_per_part or 1
        args.batch_size = min(args.batch_size, 4)
        args.epochs = 1
        args.nsample = 3
        args.max_train_steps = args.max_train_steps or 3
        args.window = min(args.window, 96)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir = os.path.join(HERE, "runs_h2h", f"gap_{int(args.gap_hours)}h")
    os.makedirs(out_dir, exist_ok=True)

    # --- build runs, VESSEL-DISJOINT split (test vessels unseen) ---
    runs = build_runs(years=(2013,), months=(6, 7), n_mmsi=args.n_mmsi,
                      max_files_per_part=args.max_files_per_part, thin_sec=180, split_gap_h=1.0,
                      min_len=40, min_span_h=args.gap_hours + 4.0, min_sog=1.0, seed=args.seed)
    mmsis = sorted({r["mmsi"] for r in runs})
    (train_runs, val_runs, test_runs,
     train_mmsi, val_mmsi, test_mmsi) = split_runs_vessel_disjoint(runs, args.seed)
    print(f"runs: {len(runs)} from {len(mmsis)} vessels -> train {len(train_runs)}/{len(train_mmsi)}v | "
          f"val {len(val_runs)}/{len(val_mmsi)}v | test {len(test_runs)}/{len(test_mmsi)}v", flush=True)

    # --- z-score fit on TRAIN observed non-gap values ---
    mean, std = fit_stats(train_runs, args.window, args.gap_hours, seed=1)
    print(f"z-score mean {np.round(mean,3)} std {np.round(std,3)}", flush=True)

    train_ds = AISCSDIDataset(train_runs, args.window, args.gap_hours, mean, std, seed=1)
    val_ds = AISCSDIDataset(val_runs, args.window, args.gap_hours, mean, std, seed=3)
    test_ds = AISCSDIDataset(test_runs, args.window, args.gap_hours, mean, std, seed=2)
    print(f"usable samples (contiguous gap): train {len(train_ds)} / val {len(val_ds)} / test {len(test_ds)}", flush=True)
    if len(train_ds) < 2 or len(test_ds) < 1:
        print("!! too few runs host a valid two-sided gap -- increase --n-mmsi / months, or lower --gap-hours.",
              flush=True)
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=min(args.batch_size, 16), shuffle=False)
    val_loader = (DataLoader(val_ds, batch_size=min(args.batch_size, 16), shuffle=False)
                  if len(val_ds) > 0 else None)

    config = build_config(args)
    model = CSDI_AIS(config, device, target_dim=4).to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters())} | device {device} | L={args.window}",
          flush=True)

    # --- CSDI's VANILLA model, trained by our harness loop (mirrors CSDI's Adam wd=1e-6 + MultiStepLR) with
    #     PLATEAU EARLY-STOP on the val denoising loss.  Diffusion caveat: val-loss plateau can
    #     precede sample-quality convergence -> patience is deliberately generous.  Best-val checkpoint restored. ---
    train_earlystop(model, config["train"], train_loader, val_loader, out_dir, args.patience, device)

    # --- impute + median NM ---
    pooled, per_traj = eval_nm(model, test_loader, mean, std, nsample=args.nsample, device=device)
    with open(os.path.join(out_dir, "per_traj_nm.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["traj", "median_nm"])
        for i, v in enumerate(per_traj):
            w.writerow([i, round(float(v), 4)])
    summary = {
        "gap_hours": args.gap_hours, "window_L": args.window, "nsample": args.nsample,
        "n_test_traj": int(len(per_traj)), "n_gap_points": int(len(pooled)),
        "median_nm_pooled": round(float(np.median(pooled)), 4) if len(pooled) else None,
        "mean_nm_pooled": round(float(np.mean(pooled)), 4) if len(pooled) else None,
        "median_of_traj_medians_nm": round(float(np.median(per_traj)), 4) if len(per_traj) else None,
        "smoke": bool(args.smoke),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== CSDI (vanilla) on our Arctic AIS, contiguous two-sided gap ===")
    print(f"  gap {args.gap_hours}h | test traj {len(per_traj)} | gap points {len(pooled)}")
    print(f"  MEDIAN NM (pooled): {summary['median_nm_pooled']}   mean {summary['mean_nm_pooled']}")
    print(f"  median-of-traj-medians NM: {summary['median_of_traj_medians_nm']}")
    if args.smoke:
        print("  [SMOKE] undertrained -- this number is MEANINGLESS; it only proves the pipeline runs.")
    print(f"saved {out_dir}/per_traj_nm.csv + summary.json", flush=True)


if __name__ == "__main__":
    main()

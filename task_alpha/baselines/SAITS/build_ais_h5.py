"""EXTERNAL dataset builder (NOT part of SAITS): turn OUR Arctic AIS positions
(task_alpha.data_raw) into a SAITS-format HDF5 dataset with a CONTIGUOUS ~gap_hours
position gap in the middle of each fixed-length window, plus a VESSEL-DISJOINT
train/val/test split and a saved StandardScaler for NM denormalization.

We touch NOTHING in the SAITS repo -- this only WRITES a `datasets.h5` (+ scaler.pkl,
+ meta.json) that SAITS's own UnifiedDataLoader / SAITS class read verbatim.

SAITS h5 layout (verified against modeling/unified_dataloader.py + dataset_generating_scripts):
  group train : X                                             [N, seq_len, 4]  (NaN=missing)
  group val   : X, X_hat, missing_mask, indicating_mask       [N, seq_len, 4]
  group test  : X, X_hat, missing_mask, indicating_mask       [N, seq_len, 4]
  top-level   : empirical_mean_for_GRUD  (harmless; some loaders expect it)
  where (val/test): X = ground truth (NaN allowed), X_hat = X with the gap set to NaN,
  missing_mask = 1 where X_hat is observed (0 in the gap), indicating_mask = 1 on the
  held-out gap points ONLY (== where we evaluate).

Features order: [latitude, longitude, sog, cog]  (n_features = 4).
NaN cog is filled with 0 (an observed sentinel) so it never leaks into the gap metric.

CLI:
  python build_ais_h5.py --gap-hours 6 --seq-len 128 --thin-sec 300 [--smoke]
"""
import argparse
import json
import os
import pickle
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FQGNN = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # repository root
if FQGNN not in sys.path:
    sys.path.insert(0, FQGNN)

from task_alpha.data_raw import load_raw, build_runs_raw, _ts_seconds  # our raw AIS (read-only)

FEATURES = ["latitude", "longitude", "sog", "cog"]
LAT, LON, SOG, COG = 0, 1, 2, 3
R_KM, KM_PER_NM = 6371.0, 1.852


def haversine_nm(lat1, lon1, lat2, lon2):
    """Great-circle distance in nautical miles (vectorized). Matches the MH-GIN h2h helper."""
    r = np.pi / 180.0
    dlat = (lat2 - lat1) * r
    dlon = (lon2 - lon1) * r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1 * r) * np.cos(lat2 * r) * np.sin(dlon / 2) ** 2
    return 2 * R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1))) / KM_PER_NM


def carve_windows(runs, seq_len, gap_hours, rng, min_anchor_each_side=3, min_real=24):
    """From each thinned vessel-run, build ONE fixed-length [seq_len,4] window whose REAL pings (the first
    min(len,seq_len)) hold a CONTIGUOUS time-gap of ~gap_hours with anchors on both sides.  Runs shorter than
    seq_len are PADDED with NaN -- SAITS treats NaN as unobserved (missing_mask=0), so padding is never
    conditioned on nor scored (indicating_mask=0 there).  Returns {mmsi, feats[seq_len,4], gap_mask[seq_len]}."""
    gap_sec = gap_hours * 3600.0
    out = []
    for seg in runs:
        R = len(seg)
        if R < min_real:
            continue
        use = min(R, seq_len)                                          # real pings in this window (rest = NaN pad)
        w = seg.iloc[:use].reset_index(drop=True)
        t = _ts_seconds(w["time"]).astype(np.float64)
        span = t[-1] - t[0]
        if span <= gap_sec * 1.3:
            continue
        margin = 0.15 * span
        lo, hi = t[0] + margin, t[0] + span - margin - gap_sec
        if hi <= lo:
            continue
        gs = rng.uniform(lo, hi)
        gap_use = (t >= gs) & (t < gs + gap_sec)
        if int(gap_use.sum()) < 2 or int((t < gs).sum()) < min_anchor_each_side \
                or int((t >= gs + gap_sec).sum()) < min_anchor_each_side:
            continue
        feats = np.full((seq_len, 4), np.nan, np.float32)              # NaN beyond `use` = padding (unobserved)
        feats[:use, LAT] = w["latitude"].values.astype(np.float32)
        feats[:use, LON] = w["longitude"].values.astype(np.float32)
        feats[:use, SOG] = w["sog"].values.astype(np.float32)
        feats[:use, COG] = np.where(np.isnan(w["cog"].values), 0.0, w["cog"].values).astype(np.float32)
        gap_mask = np.zeros(seq_len, bool); gap_mask[:use] = gap_use
        out.append({"mmsi": int(seg["mmsi"].iloc[0]), "feats": feats, "gap_mask": gap_mask})
    return out


def vessel_disjoint_split(samples, rng, val_frac=0.15, test_frac=0.15):
    """Split windows by MMSI so no vessel appears in more than one split (no leakage)."""
    mmsis = sorted({s["mmsi"] for s in samples})
    rng.shuffle(mmsis)
    n = len(mmsis)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    test_v = set(mmsis[:n_test])
    val_v = set(mmsis[n_test:n_test + n_val])
    train_v = set(mmsis[n_test + n_val:])
    buckets = {"train": [], "val": [], "test": []}
    for s in samples:
        if s["mmsi"] in test_v:
            buckets["test"].append(s)
        elif s["mmsi"] in val_v:
            buckets["val"].append(s)
        elif s["mmsi"] in train_v:
            buckets["train"].append(s)
    return buckets, {"n_vessels": n, "train_v": len(train_v), "val_v": len(val_v), "test_v": len(test_v)}


def stack(samples):
    X = np.stack([s["feats"] for s in samples], axis=0).astype(np.float32)      # [N,L,4]
    gap = np.stack([s["gap_mask"] for s in samples], axis=0).astype(bool)       # [N,L]
    mmsi = np.array([s["mmsi"] for s in samples], dtype=np.int64)
    return X, gap


def build(args):
    rng = np.random.RandomState(args.seed)
    n_mmsi = 250 if args.smoke else args.n_mmsi
    mfp = 1 if args.smoke else args.max_files_per_part

    print(f"[build] load_raw years={args.years} months={args.months} n_mmsi={n_mmsi} "
          f"max_files_per_part={mfp} ...", flush=True)
    df = load_raw(years=tuple(args.years), months=tuple(args.months), n_mmsi=n_mmsi,
                  max_files_per_part=mfp, seed=args.seed, clean_only=True)
    print(f"[build] raw pings: {len(df)} / vessels: {df['mmsi'].nunique()}", flush=True)

    # runs need to span gap + two-sided anchors -> require span >= gap_hours + 4h
    runs = build_runs_raw(df, gap_hours=1.0, thin_sec=args.thin_sec, min_len=30,
                          min_span_h=args.gap_hours + 4.0, min_sog=1.0)
    # attach mmsi onto each run (build_runs_raw preserves the mmsi column)
    print(f"[build] vessel-runs (thinned, moving): {len(runs)}", flush=True)

    samples = carve_windows(runs, args.seq_len, args.gap_hours, rng)
    print(f"[build] fixed-length windows with contiguous {args.gap_hours}h gap: {len(samples)}", flush=True)
    if len(samples) < 8:
        print("!! too few windows -- increase data (months/vessels/files) or lower seq_len/gap.", flush=True)
        return None

    buckets, split_info = vessel_disjoint_split(samples, rng,
                                                val_frac=args.val_frac, test_frac=args.test_frac)
    print(f"[build] vessel-disjoint split: train {len(buckets['train'])} / "
          f"val {len(buckets['val'])} / test {len(buckets['test'])} windows "
          f"(vessels train/val/test = {split_info['train_v']}/{split_info['val_v']}/{split_info['test_v']})",
          flush=True)
    for k in ("train", "val", "test"):
        if len(buckets[k]) == 0:
            print(f"!! split '{k}' is empty -- need more distinct vessels.", flush=True)
            return None

    Xtr, gtr = stack(buckets["train"])
    Xva, gva = stack(buckets["val"])
    Xte, gte = stack(buckets["test"])

    # ---- StandardScaler fit on TRAIN only (per feature), inverted later for NM ----
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    flat_tr = Xtr.reshape(-1, len(FEATURES))
    real = ~np.isnan(flat_tr[:, LAT])                                  # fit on real (non-padded) points only
    scaler.fit(flat_tr[real])

    def norm(X):
        s = X.shape
        return scaler.transform(X.reshape(-1, len(FEATURES))).reshape(s).astype(np.float32)

    Xtr_n, Xva_n, Xte_n = norm(Xtr), norm(Xva), norm(Xte)

    # ---- build SAITS arrays -------------------------------------------------------
    # TRAIN: X = full normalized sequence (all observed). SAITS's MIT randomly masks
    #        20% during training for imputation supervision (its intended usage).
    #        No NaNs here -> the model learns from complete windows.
    # VAL/TEST: X = ground truth; X_hat = X with the contiguous gap set to NaN;
    #        missing_mask = 1 where observed (0 in gap); indicating_mask = 1 on the gap.
    def make_eval_group(Xn, gap):
        N, L, F = Xn.shape
        gap3 = np.broadcast_to(gap[:, :, None], (N, L, F))          # [N,L,4] gap over all 4 feats
        X = Xn.astype(np.float32).copy()                            # ground truth (no NaN)
        X_hat = X.copy()
        X_hat[gap3] = np.nan                                        # blank the gap
        missing_mask = (~np.isnan(X_hat)).astype(np.float32)       # 1 = observed in X_hat
        indicating_mask = (np.isnan(X_hat) & ~np.isnan(X)).astype(np.float32)  # 1 = held-out gap
        return X, X_hat, missing_mask, indicating_mask

    Xva_gt, Xva_hat, mm_va, im_va = make_eval_group(Xva_n, gva)
    Xte_gt, Xte_hat, mm_te, im_te = make_eval_group(Xte_n, gte)

    # empirical mean (GRU-D) over train observed values -- harmless, some tooling expects it
    tr_mask = (~np.isnan(Xtr_n)).astype(np.float32)
    tr_fill = np.nan_to_num(Xtr_n)
    emp_mean = (tr_mask * tr_fill).sum(axis=(0, 1)) / np.maximum(tr_mask.sum(axis=(0, 1)), 1)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    h5_path = os.path.join(out_dir, "datasets.h5")
    with h5py.File(h5_path, "w") as hf:
        hf.create_dataset("empirical_mean_for_GRUD", data=emp_mean.astype(np.float32))
        g = hf.create_group("train")
        g.create_dataset("X", data=Xtr_n.astype(np.float32))
        for name, (X, Xh, mm, im) in [("val", (Xva_gt, Xva_hat, mm_va, im_va)),
                                      ("test", (Xte_gt, Xte_hat, mm_te, im_te))]:
            grp = hf.create_group(name)
            grp.create_dataset("X", data=X.astype(np.float32))
            grp.create_dataset("X_hat", data=Xh.astype(np.float32))
            grp.create_dataset("missing_mask", data=mm.astype(np.float32))
            grp.create_dataset("indicating_mask", data=im.astype(np.float32))

    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "features": FEATURES, "n_features": len(FEATURES), "seq_len": args.seq_len,
        "gap_hours": args.gap_hours, "thin_sec": args.thin_sec, "years": args.years,
        "months": args.months, "smoke": args.smoke,
        "n_train": int(Xtr_n.shape[0]), "n_val": int(Xva_gt.shape[0]), "n_test": int(Xte_gt.shape[0]),
        "split_vessels": split_info,
        "gap_points_val_total": int(gva.sum()), "gap_points_test_total": int(gte.sum()),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[build] wrote {h5_path}", flush=True)
    print(f"[build] wrote {os.path.join(out_dir, 'scaler.pkl')} + meta.json", flush=True)
    print(f"[build] shapes: train X {Xtr_n.shape} | val X {Xva_gt.shape} | test X {Xte_gt.shape}", flush=True)
    return meta


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-hours", type=float, default=6.0)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--thin-sec", type=int, default=300, help="thinning spacing in seconds (5min default)")
    ap.add_argument("--n-mmsi", type=int, default=None, help="None = all vessels (full run)")
    ap.add_argument("--max-files-per-part", type=int, default=None, help="None = all part files")
    ap.add_argument("--years", type=int, nargs="+", default=[2013])
    ap.add_argument("--months", type=int, nargs="+", default=[6, 7])
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default=None,
                    help="dir to write datasets.h5 (default: generated_datasets/ais_gap_<H>h)")
    ap.add_argument("--smoke", action="store_true", help="tiny build (~60 vessels, 1 file/part)")
    args = ap.parse_args()
    if args.out_dir is None:
        tag = "smoke" if args.smoke else "full"
        args.out_dir = os.path.join(HERE, "generated_datasets",
                                    f"ais_gap_{int(args.gap_hours)}h_{tag}")
    return args


if __name__ == "__main__":
    build(parse_args())

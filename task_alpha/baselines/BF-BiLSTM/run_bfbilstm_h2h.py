"""EXTERNAL head-to-head runner (NOT part of the BF-BiLSTM repo): evaluate the AUTHORS' PRETRAINED
BF-BiLSTM (`pretrained_models_ais/model_dblstm.h5`, trained on Mediterranean FISHING vessels) on OUR
Arctic AIS gap-filling task, INFERENCE ONLY. We change NOTHING in the repo: we LOAD their pretrained
model unchanged and run it as-is. This is a DOMAIN-TRANSFER test (Mediterranean fishing -> Arctic).

Why this is the only "vanilla" option: the repo ships ONLY a pretrained model + an eval-only notebook
(no training code). We do NOT re-implement the architecture: we load their .h5 and
predict on our data. We fit a FRESH MinMaxScaler on OUR data (their Mediterranean scaler would be wrong).

BF-BiLSTM's RIGID structure (verified from jupyter_notebook.ipynb DatasetManagerDI + the h5 config):
  - window = 48 consecutive grid points: 12 "before" + 24 "gap" (held out) + 12 "after"
  - model input  = concat([before_12, reversed(after_12)]) -> (24,2) -> flattened to (1, 48)
  - model output = (1, 48) -> reshaped to (24, 2) = the imputed gap lat/lon
  - uses ONLY lat/lon (drops sog/cog); MinMaxScaler on [latitude, longitude]

The 24 gap points span the requested gap_hours, so the grid step = gap_hours/24 (default 6h -> 15 min);
the 12+12 anchors then extend gap_hours/2 (3h) on each side, total window = 2*gap_hours (12h).

Metric: MEDIAN great-circle (haversine) error in NAUTICAL MILES on the gap points.
Split: VESSEL-DISJOINT (test MMSIs never appear in the non-test pool).

ENVIRONMENT: run with the dedicated Python 3.9 env `bfbilstm` (TF-CPU 2.11) -- the pretrained .h5 stores
its Lambda layers as Python-3.7/3.8-era marshalled bytecode that Python 3.12+ CANNOT unmarshal.
  python run_bfbilstm_h2h.py --gap-hours 6 [--smoke]

Inference only. CPU only (CUDA_VISIBLE_DEVICES=-1). No training.
"""
import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import types

import numpy as np

# --- CPU-only, quiet TF (inference is tiny; never train) ---
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

HERE = os.path.dirname(os.path.abspath(__file__))
FQGNN = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # repository root
TASK_ALPHA = os.path.join(FQGNN, "task_alpha")

MODEL_PATH = os.path.join(HERE, "pretrained_models_ais", "model_dblstm.h5")
R_KM, KM_PER_NM = 6371.0, 1.852

# BF-BiLSTM rigid geometry (do NOT change: it is baked into the pretrained weights)
N_ANCHOR = 12        # points before, and points after
N_GAP = 24           # gap points the model predicts
WIN = N_ANCHOR + N_GAP + N_ANCHOR   # = 48


# ----------------------------------------------------------------------------------------------------
# Load OUR raw AIS loader WITHOUT importing the torch-heavy `task_alpha` package (this env has no torch).
# We stub `torch` (load_raw never calls it -- only the caching path does) and load data.py + data_raw.py
# as a tiny synthetic package `ta` so data_raw's `from .data import ...` resolves. Repo files unchanged.
# ----------------------------------------------------------------------------------------------------
def _load_data_raw():
    def _stub(name):
        m = types.ModuleType(name); sys.modules[name] = m; return m
    if "torch" not in sys.modules:
        t = _stub("torch")
        for n in ("Tensor", "float32", "long", "bool", "device", "dtype"):
            setattr(t, n, type(n, (), {}))
        for fn in ("tensor", "save", "load", "cat", "zeros", "stack", "from_numpy", "arange", "no_grad"):
            setattr(t, fn, lambda *a, **k: None)
        nn = _stub("torch.nn"); nn.Module = type("Module", (), {})
        nnf = _stub("torch.nn.functional"); nn.functional = nnf; t.nn = nn
        tu = _stub("torch.utils"); tud = _stub("torch.utils.data")
        tud.Dataset = type("Dataset", (), {}); tud.DataLoader = type("DataLoader", (), {})
        tu.data = tud; t.utils = tu
    pkg = types.ModuleType("ta"); pkg.__path__ = [TASK_ALPHA]; sys.modules["ta"] = pkg

    def _load(mod, fname):
        spec = importlib.util.spec_from_file_location("ta." + mod, os.path.join(TASK_ALPHA, fname))
        m = importlib.util.module_from_spec(spec); sys.modules["ta." + mod] = m
        spec.loader.exec_module(m); return m
    _load("data", "data.py")            # provides ta.data (needed by data_raw's relative import)
    dr = _load("data_raw", "data_raw.py")
    return dr.load_raw, dr._ts_seconds   # use the project's OWN robust time->seconds (handles datetime64[ms])


def haversine_nm(lat1, lon1, lat2, lon2):
    r = math.pi / 180.0
    dlat = (lat2 - lat1) * r; dlon = (lon2 - lon1) * r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1 * r) * np.cos(lat2 * r) * np.sin(dlon / 2) ** 2
    return 2 * R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1))) / KM_PER_NM


# ----------------------------------------------------------------------------------------------------
# Build runs -> resample to a fixed grid -> extract rigid 48-point 12/24/12 windows.
# ----------------------------------------------------------------------------------------------------
def build_runs(df, ts_seconds, break_gap_h=1.0):
    """Split each vessel's pings into runs at time gaps > break_gap_h. Yields (mmsi, t_sec, lat, lon).
    `ts_seconds` is the project's own data_raw._ts_seconds (correctly handles the datetime64[ms] column)."""
    runs = []
    for mmsi, g in df.groupby("mmsi"):
        g = g.sort_values("time")
        t = np.asarray(ts_seconds(g["time"]), dtype=np.float64)  # seconds (via project helper)
        lat = g["latitude"].to_numpy(np.float64); lon = g["longitude"].to_numpy(np.float64)
        if len(t) < 2:
            continue
        brk = np.where(np.diff(t) > break_gap_h * 3600.0)[0]
        for s in np.split(np.arange(len(t)), brk + 1):
            if len(s) >= 3:
                runs.append((int(mmsi), t[s], lat[s], lon[s]))
    return runs


def resample_run(t, lat, lon, step_sec):
    """Uniform-grid resample of a run by linear interpolation in time. Returns (tg, lat_g, lon_g) or None."""
    t0, t1 = t[0], t[-1]
    if t1 - t0 < step_sec * (WIN - 1):
        return None                                             # too short to hold one 48-pt window
    n = int((t1 - t0) // step_sec) + 1
    tg = t0 + np.arange(n) * step_sec
    return tg, np.interp(tg, t, lat), np.interp(tg, t, lon)


def extract_windows(runs, step_sec, max_gap_interp_sec, stride):
    """From resampled runs, extract rigid 48-pt windows (12 before / 24 gap / 12 after). We only keep a
    window if the 24 gap grid-points are all within `max_gap_interp_sec` of a REAL observation (so the
    'truth' we score against is a faithful resample, not an extrapolation across a huge hole).
    Returns list of dicts with the vessel id (for a vessel-disjoint split) and the 48x2 lat/lon block."""
    wins = []
    for mmsi, t, lat, lon in runs:
        rs = resample_run(t, lat, lon, step_sec)
        if rs is None:
            continue
        tg, lat_g, lon_g = rs
        # nearest real-observation gap per grid point (for the 'faithful truth' guard)
        idx = np.searchsorted(t, tg)
        idx = np.clip(idx, 1, len(t) - 1)
        near = np.minimum(np.abs(tg - t[idx]), np.abs(tg - t[idx - 1]))
        n = len(tg)
        for st in range(0, n - WIN + 1, stride):
            sl = slice(st, st + WIN)
            if near[sl].max() > max_gap_interp_sec:
                continue                                        # this window straddles a real data hole
            block = np.stack([lat_g[sl], lon_g[sl]], axis=1)    # (48, 2) lat/lon
            wins.append({"mmsi": mmsi, "block": block.astype(np.float64)})
    return wins


def window_to_io(block):
    """block: (48,2) [lat,lon]. Return (before_reversed_after (24,2), gap_truth (24,2))."""
    before = block[:N_ANCHOR]                                   # first 12
    after = block[-N_ANCHOR:][::-1]                             # last 12, REVERSED (BF-BiLSTM convention)
    gap = block[N_ANCHOR:N_ANCHOR + N_GAP]                      # middle 24 (held out)
    model_in = np.concatenate([before, after], axis=0)          # (24, 2)
    return model_in, gap


# ----------------------------------------------------------------------------------------------------
def load_pretrained():
    """Load the AUTHORS' pretrained BF-BiLSTM UNCHANGED. We only supply the runtime namespace the repo's
    notebook itself had in scope: `ade` (their custom metric) and `split` (=tf.split, referenced by the
    model's Lambda layers via `from tensorflow import split`). No architecture is reconstructed."""
    import tensorflow as tf
    import tensorflow.keras.backend as K
    from tensorflow.keras.models import load_model

    def ade(y_true, y_pred):
        return K.mean(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True)))

    custom = {"ade": ade, "split": tf.split, "tf": tf}
    model = load_model(MODEL_PATH, custom_objects=custom, compile=False)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-hours", type=float, default=6.0, help="gap the 24 middle points span; grid step = gap_hours/24")
    ap.add_argument("--years", type=int, nargs="+", default=[2013])
    ap.add_argument("--months", type=int, nargs="+", default=[6, 7])
    ap.add_argument("--n-mmsi", type=int, default=None, help="vessels to sample (None=all); overridden to 40 in --smoke")
    ap.add_argument("--max-files-per-part", type=int, default=None, help="raw part files per partition (None=all); 1 in --smoke")
    ap.add_argument("--test-ratio", type=float, default=0.2, help="fraction of VESSELS held out for the test set")
    ap.add_argument("--stride", type=int, default=24, help="grid-step stride between consecutive windows (24 = non-overlapping gaps)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="tiny run: few vessels/files, cap to a handful of test windows, then stop")
    ap.add_argument("--out", default=None)
    cli = ap.parse_args()

    step_sec = (cli.gap_hours * 3600.0) / N_GAP                 # 24 gap points span the gap
    max_gap_interp_sec = 2.0 * step_sec                         # 'faithful truth' guard: no window across a >2-step real hole
    n_mmsi = 60 if cli.smoke else cli.n_mmsi
    mfp = 4 if cli.smoke else cli.max_files_per_part          # a few files so smoke test-vessels have >=12h runs
    out_dir = cli.out or os.path.join(HERE, "runs_h2h", f"gap_{int(cli.gap_hours)}h" + ("_smoke" if cli.smoke else ""))
    os.makedirs(out_dir, exist_ok=True)

    print(f"[cfg] gap={cli.gap_hours}h  grid_step={step_sec/60:.1f}min  window={WIN}pts(12/24/12)  "
          f"total_span={WIN*step_sec/3600:.1f}h  smoke={cli.smoke}", flush=True)

    # ---- OUR data ----
    load_raw, ts_seconds = _load_data_raw()
    df = load_raw(years=tuple(cli.years), months=tuple(cli.months), n_mmsi=n_mmsi,
                  max_files_per_part=mfp, seed=cli.seed, clean_only=True)
    print(f"[data] {len(df)} pings / {df['mmsi'].nunique()} vessels", flush=True)

    # ---- VESSEL-DISJOINT split (on MMSI) ----
    rng = np.random.default_rng(cli.seed)
    vessels = np.sort(df["mmsi"].unique())
    rng.shuffle(vessels)
    n_test = max(1, int(round(len(vessels) * cli.test_ratio)))
    test_vessels = set(vessels[:n_test].tolist())
    df_test = df[df["mmsi"].isin(test_vessels)]
    print(f"[split] vessel-disjoint: {len(test_vessels)} TEST vessels / {len(vessels)-len(test_vessels)} non-test "
          f"(scaler fit on TEST-vessel positions only; no train data available to leak)", flush=True)

    # ---- windows (test vessels only) ----
    runs = build_runs(df_test, ts_seconds, break_gap_h=1.0)
    wins = extract_windows(runs, step_sec, max_gap_interp_sec, stride=cli.stride)
    print(f"[win] {len(runs)} test-vessel runs -> {len(wins)} rigid 48-pt windows", flush=True)
    if not wins:
        print("!! no windows -- runs too short for this gap/grid. Try more data (months/vessels/files) or smaller --gap-hours.", flush=True)
        return
    if cli.smoke:
        wins = wins[:16]
        print(f"[smoke] capped to {len(wins)} windows", flush=True)

    # ---- FRESH MinMaxScaler on OUR data (lat/lon of the test windows; NOT the Mediterranean scaler) ----
    from sklearn.preprocessing import MinMaxScaler
    all_ll = np.concatenate([w["block"] for w in wins], axis=0)     # (Nwin*48, 2)
    scaler = MinMaxScaler().fit(all_ll)

    # ---- load pretrained model (unchanged) ----
    model = load_pretrained()
    print(f"[model] loaded pretrained model_dblstm.h5  input={model.input_shape} output={model.output_shape}", flush=True)

    # ---- predict + score ----
    ins = np.zeros((len(wins), WIN), np.float32)                    # (N, 48)
    gaps_true = np.zeros((len(wins), N_GAP, 2), np.float64)         # (N, 24, 2) lat/lon truth (unscaled)
    for i, w in enumerate(wins):
        blk_s = scaler.transform(w["block"])                        # scale the whole 48-pt block
        model_in, _ = window_to_io(blk_s)                           # (24,2) scaled -> flatten
        ins[i] = model_in.reshape(-1)                               # (48,)
        _, gap_truth = window_to_io(w["block"])                     # (24,2) UNSCALED truth
        gaps_true[i] = gap_truth

    preds = model.predict(ins, verbose=0)                           # (N, 48)
    preds = preds.reshape(len(wins), N_GAP, 2)                      # (N, 24, 2) scaled lat/lon
    preds_ll = np.stack([scaler.inverse_transform(p) for p in preds], axis=0)  # unscale each

    per_win_median, per_win_mean, rows = [], [], []
    pooled = []
    for i in range(len(wins)):
        nm = haversine_nm(preds_ll[i, :, 0], preds_ll[i, :, 1], gaps_true[i, :, 0], gaps_true[i, :, 1])
        per_win_median.append(float(np.median(nm)))
        per_win_mean.append(float(np.mean(nm)))
        pooled.extend(nm.tolist())
        rows.append([i, wins[i]["mmsi"], round(float(np.median(nm)), 4), round(float(np.mean(nm)), 4),
                     round(float(np.max(nm)), 4)])
    pooled = np.array(pooled)

    # ---- save ----
    with open(os.path.join(out_dir, "per_win_nm.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["win", "mmsi", "median_nm", "mean_nm", "max_nm"])
        for r in rows:
            w.writerow(r)
    summary = {
        "baseline": "BF-BiLSTM (pretrained, Mediterranean fishing) -- domain transfer, inference only",
        "model_file": "pretrained_models_ais/model_dblstm.h5",
        "gap_hours": cli.gap_hours, "grid_step_min": round(step_sec / 60, 2),
        "window_pts": WIN, "structure": "12 before / 24 gap / 24 predict / 12 after",
        "years": cli.years, "months": cli.months, "n_test_vessels": len(test_vessels),
        "n_windows": len(wins), "n_gap_points": int(len(pooled)),
        "smoke": bool(cli.smoke),
        "median_nm_pooled": round(float(np.median(pooled)), 4),
        "mean_nm_pooled": round(float(np.mean(pooled)), 4),
        "median_of_window_medians_nm": round(float(np.median(per_win_median)), 4),
        "caveats": [
            "PRETRAINED on Mediterranean FISHING vessels (5-min AIS); zero-shot domain transfer to Arctic.",
            "RIGID 24-point gap + 12/12 anchors baked into the weights (cannot vary gap geometry).",
            "Uses ONLY lat/lon (drops sog/cog). Fresh MinMaxScaler fit on OUR data.",
            "No training code shipped -> this is the only vanilla option (load-and-predict).",
        ],
    }
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)

    print("\n=== BF-BiLSTM (pretrained, Mediterranean fishing) on OUR Arctic AIS -- DOMAIN TRANSFER, inference only ===")
    print(f"  gap {cli.gap_hours}h | test vessels {len(test_vessels)} | windows {len(wins)} | gap points {len(pooled)}")
    print(f"  MEDIAN NM (pooled)          : {summary['median_nm_pooled']}")
    print(f"  MEDIAN of window-medians NM : {summary['median_of_window_medians_nm']}")
    print(f"  mean NM (pooled)            : {summary['mean_nm_pooled']}")
    print(f"saved -> {out_dir}\\per_win_nm.csv + summary.json")
    print("  CAVEAT: pretrained on Mediterranean fishing, rigid 24-pt gap, lat/lon only -- weak transfer expected.")


if __name__ == "__main__":
    main()

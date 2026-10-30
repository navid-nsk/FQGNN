"""BF-BiLSTM on the FQGNN gaps -- STAGE B (bfbilstm env, Python 3.9 / TF-CPU 2.11): ZERO-SHOT predict.

Loads the AUTHORS' PRETRAINED model_dblstm.h5 EXACTLY as run_bfbilstm_h2h.py does (same custom
objects `ade`/`split`/`tf`, compile=False, CPU only) and applies its RIGID 12/24/12 protocol to each
exported gap from test_gaps.npz (Stage A):

  step   = (t_post_first - t_pre_last) / 25        # 48-pt regular grid: before ends at index 11,
  gap    = t_pre_last + step * (1..24)             # gap occupies 12..35, after starts at 36
  before = observed pre-gap track linearly resampled at t_pre_last - step*(11..0)
  after  = observed post-gap track linearly resampled at t_post_first + step*(0..11)
  input  = concat([before, reversed(after)]) -> (24,2) -> flatten (48,)   [BF-BiLSTM convention]
  output = (48,) -> (24,2) = the imputed gap lat/lon on the regular grid

Normalization: FRESH MinMaxScaler fit on OUR observed pre/post positions pooled over all gaps
(same "fresh scaler on our data" choice as run_bfbilstm_h2h.py; held truth is NOT available here,
so it cannot leak into the scaler).  NO training, NO seeds -- frozen deterministic model, one pass.

Degenerate sides (<2 observed points): still resampled from what exists (np.interp with 1 point =
constant); an EMPTY side (should not occur: n_anchor=8) borrows the other side's boundary point.
Both are counted and printed, never crash.

Run (any cwd):
  python task_alpha/baselines/BF-BiLSTM/predict_gaps_bf.py
Input : runs_gaps/test_gaps.npz     Output: runs_gaps/preds_bf.npz
"""
import os

import numpy as np

# --- CPU-only, quiet TF (inference only; never train) ---
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(HERE, "runs_gaps")
NPZ_IN = os.path.join(RUN_DIR, "test_gaps.npz")
NPZ_OUT = os.path.join(RUN_DIR, "preds_bf.npz")
MODEL_PATH = os.path.join(HERE, "pretrained_models_ais", "model_dblstm.h5")

# BF-BiLSTM rigid geometry (baked into the pretrained weights -- do NOT change)
N_ANCHOR = 12
N_GAP = 24
WIN = N_ANCHOR + N_GAP + N_ANCHOR                                    # = 48


def load_pretrained():
    """Load the AUTHORS' pretrained BF-BiLSTM UNCHANGED -- identical to run_bfbilstm_h2h.py."""
    import tensorflow as tf
    import tensorflow.keras.backend as K
    from tensorflow.keras.models import load_model

    def ade(y_true, y_pred):
        return K.mean(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True)))

    custom = {"ade": ade, "split": tf.split, "tf": tf}
    return load_model(MODEL_PATH, custom_objects=custom, compile=False)


def side_arrays(d, name, k):
    off = d[name + "_off"]
    sl = slice(int(off[k]), int(off[k + 1]))
    return d[name + "_t"][sl], d[name + "_lat"][sl], d[name + "_lon"][sl]


def main():
    d = np.load(NPZ_IN)
    gap_id = d["gap_id"]
    n = len(gap_id)
    print(f"[stage B] {n} gaps from {NPZ_IN}", flush=True)

    # ---- FRESH MinMaxScaler on OUR observed positions (pooled pre+post; no held truth exists here) ----
    from sklearn.preprocessing import MinMaxScaler
    all_ll = np.stack([np.concatenate([d["pre_lat"], d["post_lat"]]),
                       np.concatenate([d["pre_lon"], d["post_lon"]])], axis=1)
    scaler = MinMaxScaler().fit(all_ll)
    print(f"[scaler] fit on {len(all_ll)} observed points  "
          f"lat=({scaler.data_min_[0]:.3f},{scaler.data_max_[0]:.3f}) "
          f"lon=({scaler.data_min_[1]:.3f},{scaler.data_max_[1]:.3f})", flush=True)

    model = load_pretrained()
    print(f"[model] loaded pretrained model_dblstm.h5 (as published)  "
          f"input={model.input_shape} output={model.output_shape}", flush=True)

    ins = np.zeros((n, WIN), np.float32)
    grid_t = np.zeros((n, N_GAP), np.float64)
    n_deg_pre = n_deg_post = n_empty = 0
    for k in range(n):
        tp, lap, lop = side_arrays(d, "pre", k)
        tq, laq, loq = side_arrays(d, "post", k)
        if len(tp) < 2:
            n_deg_pre += 1
        if len(tq) < 2:
            n_deg_post += 1
        if len(tp) == 0:                                              # borrow other side's boundary pt
            n_empty += 1
            tp, lap, lop = tq[:1] - 1.0, laq[:1], loq[:1]
        if len(tq) == 0:
            n_empty += 1
            tq, laq, loq = tp[-1:] + 1.0, lap[-1:], lop[-1:]
        t0, t1 = float(tp[-1]), float(tq[0])
        step = max(t1 - t0, 1e-6) / (N_GAP + 1)                       # 25 steps: grid idx 11 -> 36
        before_t = t0 - step * np.arange(N_ANCHOR - 1, -1, -1)        # ascending, ends at t0
        after_t = t1 + step * np.arange(N_ANCHOR)                     # ascending, starts at t1
        grid_t[k] = t0 + step * np.arange(1, N_GAP + 1)               # the 24 gap queries
        before = np.stack([np.interp(before_t, tp, lap), np.interp(before_t, tp, lop)], axis=1)
        after = np.stack([np.interp(after_t, tq, laq), np.interp(after_t, tq, loq)], axis=1)
        before_s = scaler.transform(before)
        after_s = scaler.transform(after)
        ins[k] = np.concatenate([before_s, after_s[::-1]], axis=0).reshape(-1)   # (48,)

    preds = model.predict(ins, verbose=0).reshape(n, N_GAP, 2)        # scaled (lat, lon)
    preds_ll = np.stack([scaler.inverse_transform(p) for p in preds], axis=0)

    np.savez(NPZ_OUT, gap_id=gap_id, grid_t=grid_t,
             pred_lat=preds_ll[:, :, 0], pred_lon=preds_ll[:, :, 1],
             n_deg_pre=n_deg_pre, n_deg_post=n_deg_post, n_empty=n_empty)
    print(f"[degenerate] pre sides <2 pts: {n_deg_pre} | post sides <2 pts: {n_deg_post} "
          f"| empty sides (borrowed boundary): {n_empty}", flush=True)
    print(f"[stage B] wrote {n} x {N_GAP} predictions -> {NPZ_OUT}", flush=True)


if __name__ == "__main__":
    main()

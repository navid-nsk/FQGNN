"""paper_figures_tables/params_audit.py -- params/compute audit for Fig 6d.

ANALYSIS ONLY: builds each model EXACTLY as its runner does, loads the existing seed0 best
checkpoint (READ-ONLY), counts parameters, and times (a) one full test-eval pass over the 774
paper test gaps and (b) ~20 real training batches extrapolated to one epoch.  NO model source is
modified, NO checkpoint is (re)written.

Output: FQGNN_FIGDATA/params_audit.csv, one row per model:
  model, n_params_trainable, n_params_frozen, test_ratio_overall, inference_s_per_770gaps,
  train_s_per_epoch, notes
    n_params_frozen        = non-trainable nn.Parameters + registered buffers (for BF-BiLSTM: the
                             whole pretrained Keras model -- zero-shot, nothing trained by us)
    test_ratio_overall     = the KNOWN paper-table ratios (5-seed mean; BF-BiLSTM single pass)
    inference_s_per_770gaps= wall seconds for the full test-eval pass (774 graphs; CSDI measured on
                             20 representative graphs at nsample=50 and extrapolated x38.7 -- noted)
    train_s_per_epoch      = 20 timed training batches (after 3 warmup) x batches/epoch, since no
                             runner logs a per-epoch train-only time (BiGRU/SAITS logs include the
                             val eval).  BF-BiLSTM: zero-shot pretrained, no training -> blank.

BiGRU note: cuDNN RNNs crash THIS env's interpreter at DLL detach AFTER all work completes
(documented in run_bigru_5seed_gaps.py).  BiGRU is therefore measured LAST and the CSV is written
after every model ("crash-after-write" pattern); the final line 'AUDIT COMPLETE' certifies success.

Run:  python paper_figures_tables/params_audit.py   (any cwd)

EXTENSION ROWS: three AIS-baseline rows -- HABIT, MH-GIN, MultitaskAIS (see the "EXTENSION ROWS"
section at the bottom).  These are CPU-only audits (no GPU memory is touched, so they can run
while the GPU is busy).  With
    --extend-only
the six base rows are loaded VERBATIM from the existing CSV (never recomputed, byte-identical
round-trip) and only the three extension rows are (re)generated:
  python paper_figures_tables/params_audit.py --extend-only
"""
from __future__ import annotations

import copy
import csv
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))                  # repository root
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from _paths import FIGDATA  # noqa: E402  (env FQGNN_FIGDATA)
os.makedirs(FIGDATA, exist_ok=True)

import numpy as np
import torch

from task_alpha.baselines import common_gaps as CG
from task_alpha.data import collate_graphs_interp
from task_alpha.fqdeep_native import FQDeepNative
from task_alpha.meltin_proof import _bvec
from task_alpha.metrics import gcd_errors, model_errors

DEV = "cuda" if torch.cuda.is_available() else "cpu"
assert DEV == "cuda", "audit expects CUDA (all training runs were CUDA)"
OUT_CSV = os.path.join(FIGDATA, "params_audit.csv")
# python of the separate BF-BiLSTM environment (Python 3.9, TensorFlow-CPU 2.11)
BF_PY = os.environ.get("FQGNN_BFBILSTM_PYTHON", "python")
BL = os.path.join(ROOT, "task_alpha", "baselines")

# KNOWN paper-table overall test ratios (median NM / GCD median NM; 5-seed mean, BF single pass)
# FQGNN = the native Op-H model (run_fqnative2_5seed)
RATIOS = {"FQGNN": 0.6331, "RFF-GNN": 1.845, "BiGRU": 7.234, "CSDI": 6.026,
          "SAITS": 10.038, "BF-BiLSTM": 137.1}
MODEL_ORDER = ["FQGNN", "RFF-GNN", "BiGRU", "CSDI", "SAITS", "BF-BiLSTM"]

# ---- EXTENSION ROWS: AIS baselines, our-6h cell of the gap sweep -------------------------------- #
# HABIT        = task_alpha/runs/gap_sweep/our_processed/gap_6h/habit seed0 test_buckets OVERALL
#                (deterministic single-pass reference, same convention as GCD/Hermite/BF-BiLSTM).
# MH-GIN       = 5-seed MEAN of .../gap_6h/mhgin/summary.csv.
# MultitaskAIS = 5-seed MEAN of .../gap_6h/multitaskais/summary.csv -- the
#                same convention as every other seeded model row (and as the pivot table).
RATIOS.update({"HABIT": 1.2572, "MH-GIN": 371.6115, "MultitaskAIS": 378.5368})
EXT_MODEL_ORDER = ["HABIT", "MH-GIN", "MultitaskAIS"]
MODEL_ORDER = MODEL_ORDER + EXT_MODEL_ORDER
FIELDS = ["model", "n_params_trainable", "n_params_frozen", "test_ratio_overall",
          "inference_s_per_770gaps", "train_s_per_epoch", "notes"]

WARMUP_B, TIMED_B = 3, 20                                          # training-batch timing protocol


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def count_params(m: torch.nn.Module):
    tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    fr = sum(p.numel() for p in m.parameters() if not p.requires_grad)
    bu = sum(b.numel() for b in m.buffers())
    big = sorted(((n, tuple(b.shape)) for n, b in m.named_buffers()),
                 key=lambda x: -int(np.prod(x[1]) if x[1] else 1))[:4]
    print(f"    params: trainable {tr} | frozen-params {fr} | buffers {bu} "
          f"(largest: {big})", flush=True)
    return tr, fr + bu


def time_train_batches(step_fn, batch_iter, n_batches_per_epoch, what):
    """3 warmup + 20 timed real training steps -> extrapolated s/epoch (train loop only, no val)."""
    it = iter(batch_iter)
    for _ in range(WARMUP_B):
        step_fn(next(it))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(TIMED_B):
        step_fn(next(it))
    torch.cuda.synchronize()
    per_b = (time.perf_counter() - t0) / TIMED_B
    per_ep = per_b * n_batches_per_epoch
    print(f"    train timing [{what}]: {per_b * 1e3:.1f} ms/batch x {n_batches_per_epoch} "
          f"batches/epoch = {per_ep:.1f} s/epoch (timed {TIMED_B} batches after {WARMUP_B} warmup)",
          flush=True)
    return per_ep


ROWS = {}


def write_csv():
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for k in MODEL_ORDER:
            if k in ROWS:
                w.writerow(ROWS[k])


def add_row(model, n_tr, n_fr, inf_s, tr_s, notes):
    ROWS[model] = {"model": model, "n_params_trainable": n_tr, "n_params_frozen": n_fr,
                   "test_ratio_overall": RATIOS[model],
                   "inference_s_per_770gaps": round(inf_s, 2) if inf_s is not None else "",
                   "train_s_per_epoch": round(tr_s, 1) if tr_s is not None else "",
                   "notes": notes}
    write_csv()
    print(f"  -> row written: {ROWS[model]}", flush=True)


# ================================================================================================ #
#  FQGNN                                                                                            #
# ================================================================================================ #
def audit_fqgnn(tr, te):
    print("\n=== FQGNN (FQDeepNative, seed0 best.pt) ===", flush=True)
    ckpt = os.path.join(ROOT, "task_alpha", "runs", "fqnative2_5seed", "seed0", "best.pt")
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    m = FQDeepNative(n=6, n_freqs=48, width=96, init_scale=1.84, seed=0, in_feat=6, layers=3,
                     jacobi_R=4, oph_weighted=True).to(DEV)
    m.use_cuda = True
    with torch.no_grad():
        m.s_raw.fill_(math.log(math.expm1(2.5)))
    b0 = collate_graphs_interp(tr[:4])                            # materialize LazyLinear
    with torch.no_grad():
        m(b0[0].to(DEV), b0[1].to(DEV), b0[4].to(DEV), b0[6].to(DEV), _bvec(tr[:4]))
    m.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=False)["state_dict"])
    n_tr, n_fr = count_params(m)

    # ---- inference: ONE full test-eval pass, exact run_full_faithful_5seed test-loop pattern ----
    def eval_pass(graphs):
        meds_m, meds_g = [], []
        with torch.no_grad():
            for s0 in range(0, len(graphs), 64):
                b = graphs[s0:s0 + 64]
                x, coord, yp, mask, ei, ed, aei = collate_graphs_interp(b)
                pred = m(x.to(DEV), coord.to(DEV), ei.to(DEV), aei.to(DEV), _bvec(b)).float().cpu()
                off = 0
                for g in b:
                    ng = g.x.shape[0]
                    fe = model_errors(pred[off:off + ng], g); off += ng
                    if len(fe):
                        meds_m.append(float(np.median(fe)))
                        meds_g.append(float(np.median(gcd_errors(g))))
        return meds_m, meds_g

    m.eval()
    eval_pass(te[:64])                                            # warmup (CUDA autotune)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    meds_m, meds_g = eval_pass(te)
    torch.cuda.synchronize(); inf_s = time.perf_counter() - t0
    ratio = float(np.median(meds_m) / np.median(meds_g))
    print(f"    inference: {inf_s:.2f} s for {len(meds_m)} gaps (bs 64, no_grad, CUDA) | "
          f"computed overall ratio {ratio:.4f} (seed0 test_buckets.csv OVERALL = 0.6205)", flush=True)
    assert abs(ratio - 0.6205) < 0.02, "FQGNN eval does not reproduce seed0 checkpoint result"

    # ---- train timing: seed0 epochs.csv logs NO time column -> time 20 real batches, extrapolate ----
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=5e-4)
    order = list(range(len(tr))); random.Random(0).shuffle(order)
    batches = ([tr[j] for j in order[i:i + 64]] for i in range(0, len(order), 64))

    def step(b):
        x, co, yp, mk, ei, _, ae = collate_graphs_interp(b)
        pred = m(x.to(DEV), co.to(DEV), ei.to(DEV), ae.to(DEV), _bvec(b))
        L = ((pred[mk.to(DEV)] - yp.to(DEV)[mk.to(DEV)]) ** 2).mean()
        opt.zero_grad(); L.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()

    nb = math.ceil(len(tr) / 64)
    tr_s = time_train_batches(step, batches, nb, "FQGNN bs64")
    add_row("FQGNN", n_tr, n_fr, inf_s, tr_s,
            f"eval=full 774-gap pass bs64 (computed ratio {ratio:.4f} vs seed0 0.6205); "
            f"train=20 timed batches x{nb} (no epoch time logged); frozen=FQ geometry buffers")
    del m, opt; torch.cuda.empty_cache()


# ================================================================================================ #
#  RFF-GNN                                                                                          #
# ================================================================================================ #
def audit_rff(tr, te):
    print("\n=== RFF-GNN (runner class, seed0 best.pt) ===", flush=True)
    mod = load_module("rff_runner_mod", os.path.join(BL, "RFFGNN", "run_rffgnn_5seed_gaps.py"))
    ckpt = os.path.join(BL, "RFFGNN", "runs_5seed_gaps", "seed0", "best.pt")
    m = mod.RFFGNN(seed=0).to(DEV)
    m.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=False)["state_dict"])
    m.eval()
    n_tr, n_fr = count_params(m)

    mod.eval_gaps(m, te[:64])                                     # warmup
    torch.cuda.synchronize(); t0 = time.perf_counter()
    fq, gc, _ = mod.eval_gaps(m, te)                              # its own eval_gaps (bs 64)
    torch.cuda.synchronize(); inf_s = time.perf_counter() - t0
    ratio = float(np.median(fq) / np.median(gc))
    print(f"    inference: {inf_s:.2f} s for {len(fq)} gaps (runner eval_gaps) | "
          f"computed overall ratio {ratio:.4f}", flush=True)

    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=mod.LR)
    order = list(range(len(tr))); random.Random(0).shuffle(order)
    batches = ([tr[j] for j in order[i:i + mod.BS]] for i in range(0, len(order), mod.BS))

    def step(b):
        x, co, yp, mk, ei, ed, ae = collate_graphs_interp(b)
        pred = m(x.to(DEV), co.to(DEV), ei.to(DEV), ae.to(DEV))
        L = ((pred[mk.to(DEV)] - yp.to(DEV)[mk.to(DEV)]) ** 2).mean()
        opt.zero_grad(); L.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()

    nb = math.ceil(len(tr) / mod.BS)
    tr_s = time_train_batches(step, batches, nb, "RFF-GNN bs64")
    add_row("RFF-GNN", n_tr, n_fr, inf_s, tr_s,
            f"eval=runner eval_gaps full 774 (computed seed0 ratio {ratio:.4f}); "
            f"train=20 timed batches x{nb}; frozen=Rahimi-Recht random Fourier basis W,b")
    del m, opt; torch.cuda.empty_cache()


# ================================================================================================ #
#  CSDI                                                                                             #
# ================================================================================================ #
def audit_csdi(tr, te, mean, std):
    print("\n=== CSDI (runner construction, seed0 best.pt) ===", flush=True)
    mod = load_module("csdi_runner_mod", os.path.join(BL, "CSDI", "run_csdi_5seed_gaps.py"))
    ckpt = os.path.join(BL, "CSDI", "runs_5seed_gaps", "seed0", "best.pt")
    cfg = mod.build_config(epochs=100)
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    model = mod.CSDI_AIS(cfg, DEV, target_dim=mod.K).to(DEV)
    model.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=False)["state_dict"])
    model.eval()
    n_tr, n_fr = count_params(model)

    te_s = mod.make_samples(te, mean, std, zero_held=True)
    # 20 length-representative test graphs (every 774/20-th of the length-sorted order), ONE batch
    order = sorted(range(len(te_s)), key=lambda i: len(te_s[i]["tp"]))
    sel = [order[i] for i in np.round(np.linspace(0, len(order) - 1, 20)).astype(int)]
    lens = [len(te_s[j]["tp"]) for j in sel]
    print(f"    sampling-eval subset: 20 graphs, series lengths {min(lens)}..{max(lens)} "
          f"(test range {len(te_s[order[0]]['tp'])}..{len(te_s[order[-1]]['tp'])})", flush=True)
    mod.predict_held(model, te_s, [sel[:2]], 2, mean, std)        # warmup (2 graphs, nsample 2)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    mod.predict_held(model, te_s, [sel], 50, mean, std)           # nsample=50, as the real test eval
    torch.cuda.synchronize(); t20 = time.perf_counter() - t0
    inf_s = t20 * len(te_s) / len(sel)
    print(f"    inference: {t20:.2f} s for 20 gaps at nsample=50 (100 diffusion steps) "
          f"-> extrapolated x{len(te_s) / len(sel):.2f} = {inf_s:.1f} s / 774 gaps "
          f"(diffusion sampling is the honest cost)", flush=True)

    tr_s_samples = mod.make_samples(tr, mean, std, zero_held=False)
    loader = mod.PaddedBatches(tr_s_samples, mod.HP["batch_size"], shuffle=False)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=mod.HP["lr"], weight_decay=1e-6)

    def step(batch):
        loss = model(batch)
        opt.zero_grad(); loss.backward(); opt.step()

    nb = math.ceil(len(tr_s_samples) / mod.HP["batch_size"])
    tr_s = time_train_batches(step, loader, nb, "CSDI bs32")
    add_row("CSDI", n_tr, n_fr, inf_s, tr_s,
            f"inference EXTRAPOLATED: 20 length-representative gaps, nsample=50, one batch, "
            f"{t20:.1f}s x38.7; train=20 timed batches x{nb}; frozen=diffusion beta/alpha schedule buffers",
            )
    del model, opt; torch.cuda.empty_cache()


# ================================================================================================ #
#  SAITS                                                                                            #
# ================================================================================================ #
def audit_saits(tr, te):
    print("\n=== SAITS (build_model, seed0 best.pt) ===", flush=True)
    mod = load_module("saits_runner_mod", os.path.join(BL, "SAITS", "run_saits_5seed_gaps.py"))
    ck = torch.load(os.path.join(BL, "SAITS", "runs_5seed_gaps", "seed0", "best.pt"),
                    map_location=DEV, weights_only=False)
    seq_len, mean_s, std_s = ck["seq_len"], ck["norm_mean"], ck["norm_std"]
    model = mod.R.build_model(seq_len, mod.N_FEAT, DEV, ck["cfg"]).to(DEV)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    n_tr, n_fr = count_params(model)

    Xte, Hte = mod.build_split_arrays(te, seq_len, mean_s, std_s)
    print(f"    test arrays {Xte.shape} (the runner's own eval input)", flush=True)
    mod.impute_batched(model, Xte[:64])                           # warmup
    torch.cuda.synchronize(); t0 = time.perf_counter()
    errs = mod.gap_errors(model, Xte, Hte, te, mean_s, std_s)     # full 774-graph impute + NM errors
    torch.cuda.synchronize(); inf_s = time.perf_counter() - t0
    meds = [float(np.median(e)) for e in errs]
    gcds = [float(np.median(gcd_errors(g))) for g in te]
    ratio = float(np.median(meds) / np.median(gcds))
    print(f"    inference: {inf_s:.2f} s for 774 gaps (impute bs64 on [774,{seq_len},5] + errors) | "
          f"computed overall ratio {ratio:.4f}", flush=True)

    Xtr, _ = mod.build_split_arrays(tr, seq_len, mean_s, std_s)
    rng = np.random.RandomState(0)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=mod.LR)
    order = list(range(len(Xtr))); random.Random(0).shuffle(order)
    batches = (Xtr[order[i:i + mod.BATCH_SIZE]] for i in range(0, len(order), mod.BATCH_SIZE))

    def step(Xb):
        X_hat, mm, im, X_gt = mod.R.mit_mask_batch(Xb, rate=mod.MIT_RATE, rng=rng)
        inputs = {"indices": torch.arange(Xb.shape[0]),
                  "X": torch.from_numpy(X_hat).to(DEV),
                  "missing_mask": torch.from_numpy(mm).to(DEV),
                  "X_holdout": torch.from_numpy(X_gt).to(DEV),
                  "indicating_mask": torch.from_numpy(im).to(DEV)}
        out = model(inputs, "train")
        loss = out["reconstruction_loss"] + out["imputation_loss"]
        opt.zero_grad(); loss.backward(); opt.step()

    nb = math.ceil(len(Xtr) / mod.BATCH_SIZE)
    tr_s = time_train_batches(step, batches, nb, "SAITS bs16")
    add_row("SAITS", n_tr, n_fr, inf_s, tr_s,
            f"eval=runner impute path full 774 (computed seed0 ratio {ratio:.4f}); "
            f"train=20 timed MIT batches x{nb}")
    del model, opt; torch.cuda.empty_cache()


# ================================================================================================ #
#  BF-BiLSTM (child process, bfbilstm env, TF-CPU -- exactly as its runner runs it)                  #
# ================================================================================================ #
BF_WORKER = r"""
import json, os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import importlib.util
HERE = sys.argv[1]                                     # .../baselines/BF-BiLSTM
spec = importlib.util.spec_from_file_location("bf_predict", os.path.join(HERE, "predict_gaps_bf.py"))
bf = importlib.util.module_from_spec(spec); spec.loader.exec_module(bf)
d = np.load(os.path.join(HERE, "runs_gaps", "test_gaps.npz"))
gap_id = d["gap_id"]; n = len(gap_id)
from sklearn.preprocessing import MinMaxScaler
all_ll = np.stack([np.concatenate([d["pre_lat"], d["post_lat"]]),
                   np.concatenate([d["pre_lon"], d["post_lon"]])], axis=1)
scaler = MinMaxScaler().fit(all_ll)
model = bf.load_pretrained()
n_total = int(model.count_params())
n_tr_w = int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))
# rebuild the (n, 48) stage-B inputs exactly as predict_gaps_bf.main() does (no file is rewritten)
ins = np.zeros((n, bf.WIN), np.float32)
for k in range(n):
    tp, lap, lop = bf.side_arrays(d, "pre", k); tq, laq, loq = bf.side_arrays(d, "post", k)
    if len(tp) == 0: tp, lap, lop = tq[:1] - 1.0, laq[:1], loq[:1]
    if len(tq) == 0: tq, laq, loq = tp[-1:] + 1.0, lap[-1:], lop[-1:]
    t0, t1 = float(tp[-1]), float(tq[0])
    step = max(t1 - t0, 1e-6) / (bf.N_GAP + 1)
    bt = t0 - step * np.arange(bf.N_ANCHOR - 1, -1, -1); at = t1 + step * np.arange(bf.N_ANCHOR)
    before = np.stack([np.interp(bt, tp, lap), np.interp(bt, tp, lop)], axis=1)
    after = np.stack([np.interp(at, tq, laq), np.interp(at, tq, loq)], axis=1)
    ins[k] = np.concatenate([scaler.transform(before), scaler.transform(after)[::-1]], 0).reshape(-1)
model.predict(ins[:32], verbose=0)                     # warmup
t0 = time.perf_counter()
model.predict(ins, verbose=0)
infer_s = time.perf_counter() - t0
print("BF_AUDIT_JSON: " + json.dumps(dict(n_gaps=int(n), params_total=n_total,
      keras_trainable_weights=n_tr_w, infer_s=infer_s)))
"""


def audit_bf():
    print("\n=== BF-BiLSTM (pretrained model_dblstm.h5, bfbilstm env, CPU zero-shot) ===", flush=True)
    bf_dir = os.path.join(BL, "BF-BiLSTM")
    r = subprocess.run([BF_PY, "-c", BF_WORKER, bf_dir], capture_output=True, text=True,
                       cwd=ROOT, timeout=900)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("BF_AUDIT_JSON: ")]
    if not line:
        print("    WORKER STDOUT:\n" + r.stdout[-3000:] + "\n    WORKER STDERR:\n" + r.stderr[-3000:],
              flush=True)
        raise RuntimeError("BF-BiLSTM worker produced no result")
    res = json.loads(line[0][len("BF_AUDIT_JSON: "):])
    print(f"    Keras count_params {res['params_total']} (all in trainable_weights={res['keras_trainable_weights']} "
          f"as loaded, but ZERO-SHOT here: nothing trained by us -> all frozen) | "
          f"predict {res['n_gaps']} x (48,) inputs in {res['infer_s']:.2f} s (TF-CPU, the stage-B path)",
          flush=True)
    add_row("BF-BiLSTM", 0, res["params_total"], res["infer_s"], None,
            "authors' pretrained model_dblstm.h5, zero-shot (0 trainable in our pipeline); "
            f"inference=stage-B model.predict on the {res['n_gaps']} exported gaps, TF-CPU; no training")


# ================================================================================================ #
#  BiGRU (LAST -- cuDNN teardown crash happens only at interpreter exit, after the CSV is written)   #
# ================================================================================================ #
def audit_bigru(tr, te, mean, std):
    print("\n=== BiGRU (runner class, seed0 best.pt) ===", flush=True)
    mod = load_module("bigru_runner_mod", os.path.join(BL, "BiGRU", "run_bigru_5seed_gaps.py"))
    ckpt = os.path.join(BL, "BiGRU", "runs_5seed_gaps", "seed0", "best.pt")
    m = mod.BiGRUImputer().to(DEV)
    m.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=False)["state_dict"])
    m.eval()
    n_tr, n_fr = count_params(m)

    te_f = [mod.make_features(g, mean, std) for g in te]          # runner precomputes these once
    mod.predict_held(m, te_f[:64], mean, std)                     # warmup
    torch.cuda.synchronize(); t0 = time.perf_counter()
    preds = mod.predict_held(m, te_f, mean, std)
    meds = [float(np.median(CG.model_errors_latlon(g, la, lo))) for g, (la, lo) in zip(te, preds)]
    torch.cuda.synchronize(); inf_s = time.perf_counter() - t0
    gcds = [float(np.median(gcd_errors(g))) for g in te]
    ratio = float(np.median(meds) / np.median(gcds))
    print(f"    inference: {inf_s:.2f} s for 774 gaps (predict_held bs64 + NM errors) | "
          f"computed overall ratio {ratio:.4f}", flush=True)

    tr_f = [mod.make_features(g, mean, std) for g in tr]
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    order = list(range(len(tr_f))); random.Random(0).shuffle(order)
    batches = ([tr_f[j] for j in order[i:i + mod.BATCH]] for i in range(0, len(order), mod.BATCH))

    def step(batch):
        X, Y, H, V = mod.collate(batch)
        X, Y, H = X.to(DEV), Y.to(DEV), H.to(DEV)
        pred = m(X, V.sum(1))
        loss = ((pred[H] - Y[H]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    nb = math.ceil(len(tr_f) / mod.BATCH)
    tr_s = time_train_batches(step, batches, nb, "BiGRU bs32")
    add_row("BiGRU", n_tr, n_fr, inf_s, tr_s,
            f"eval=runner predict_held full 774 (computed seed0 ratio {ratio:.4f}); train=20 timed "
            f"batches x{nb} (log epoch ~11s incl. val eval); cuDNN teardown crash after write is benign")
    del m, opt; torch.cuda.empty_cache()


# ================================================================================================ #
# ================================================================================================ #
#  EXTENSION ROWS: HABIT / MH-GIN / MultitaskAIS -- CPU-only.                                        #
#  The six rows above are NEVER touched by this section; with --extend-only they are loaded          #
#  verbatim from the existing CSV.  All three models are run on the OUR-6H cell of the gap sweep     #
#  (task_alpha/data/our_processed/gap_6h == the paper's 774-gap test set; same graphs, same GCD).    #
# ================================================================================================ #
# ================================================================================================ #
GAP6_TRACK, GAP6_H = "our_processed", 6
SWEEP6 = os.path.join(ROOT, "task_alpha", "runs", "gap_sweep", GAP6_TRACK, f"gap_{GAP6_H}h")
DRIVER_LOG_MHGIN = os.path.join(BL, "_driver", f"mhgin_{GAP6_TRACK}_{GAP6_H}h.log")
# conda executable used to reach the mtais_tf1 environment (env MTAIS_CONDA, else the active one)
CONDA = os.environ.get("MTAIS_CONDA") or os.environ.get("CONDA_EXE") or "conda"


def _read_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------------------------------------ #
#  HABIT (EDBT 2026) -- deterministic H3 route-frequency lookup: ZERO learned parameters.            #
#  Its our-6h run (runs/gap_sweep/.../habit, seed0) has OVERALL ratio RATIOS["HABIT"].  The run      #
#  is not regenerated; the SAME cell-transition graph is rebuilt from the run's own corpus CSV       #
#  and one fresh full-test A* prediction pass is timed, asserting it reproduces the run's ratio      #
#  (HABIT is deterministic with the canonical sorted insertion order, so the match is exact).        #
# ------------------------------------------------------------------------------------------------ #
def audit_habit():
    print("\n=== HABIT (deterministic H3 route lookup, our-6h corpus) ===", flush=True)
    mod = load_module("habit_runner_mod", os.path.join(BL, "HABIT", "run_habit_gaps.py"))
    from task_alpha.baselines.HABIT.habit_model import HabitAdapter, predict_graph

    csv_path = mod.corpus_path(GAP6_TRACK, GAP6_H)
    if not os.path.exists(csv_path):                 # rebuild exactly as the runner does (phase A)
        subprocess.check_call([sys.executable, "-u", "-m",
                               "task_alpha.baselines.HABIT.run_habit_gaps", "--build-corpus",
                               "--track", GAP6_TRACK, "--gap", str(GAP6_H)], cwd=ROOT)
    te = mod._load(GAP6_TRACK, GAP6_H, "test")
    print(f"    corpus {os.path.basename(csv_path)} | test graphs {len(te)}", flush=True)

    t0 = time.perf_counter()
    imp = HabitAdapter(resolution=6, max_gap=mod.MAX_GAP, schema=mod.SCHEMA)   # run settings: res 6
    nn_, ne_ = imp.data_load(csv_path)
    build_s = time.perf_counter() - t0
    print(f"    graph build {build_s:.1f} s ({nn_} nodes / {ne_} edges) -- the only 'fit' HABIT has "
          f"(route-frequency aggregation; nothing is learned)", flush=True)

    # ONE fresh full-test prediction pass, the runner's own eval loop verbatim (run_habit_gaps.run_cell)
    t0 = time.perf_counter()
    rows, nfb = [], 0
    for i, g in enumerate(te):
        held = g.mask.numpy().astype(bool)
        if held.sum() == 0 or not held[1:-1].any():
            continue
        plat, plon, fb, _, _ = predict_graph(imp, g)
        nfb += int(fb)
        err = CG.model_errors_latlon(g, plat, plon)
        if len(err):
            rows.append(CG.gap_row(i, g, err))
    inf_s = time.perf_counter() - t0
    ov = CG.bucket_rows(rows)[0]
    ratio = float(ov["ratio"])
    print(f"    inference: {inf_s:.2f} s for {ov['n']} scored gaps (A* + arclength resample + errors) "
          f"| computed OVERALL ratio {ratio:.4f} (run seed0 test_buckets = 1.2572) | "
          f"A* fallbacks {nfb}", flush=True)
    assert abs(ratio - RATIOS["HABIT"]) < 5e-4, "HABIT re-eval does not reproduce the run's ratio"

    add_row("HABIT", 0, 0, inf_s, None,
            "deterministic route-frequency lookup (H3 res 6); no learned parameters; eval=fresh "
            f"full-test A* pass on the run's corpus (computed ratio {ratio:.4f} vs run 1.2572); "
            f"graph build {build_s:.1f}s from route-frequency aggregation, no training")


# ------------------------------------------------------------------------------------------------ #
#  MH-GIN (MTS-HGNN) -- params counted on the EXACT runner construction for the our-6h cell:         #
#  build_model(fit_status(train), d_model=D_MODEL(=32), max_delta=350.0 vendor default; both read    #
#  from run_mhgin_gaps + the driver log's cell line "max_delta 350.0").  Counted on CPU: parameter   #
#  counts are device-independent, and the GPU stays free for other jobs.  train_s_per_epoch          #
#  is recovered from the driver log's per-seed wall times ("seed N done in Ss") / epochs_trained     #
#  (summary.csv) -- it therefore INCLUDES the per-epoch val eval; inference is left blank rather     #
#  than timed on CPU (all paper timings are CUDA; a CPU number would not be comparable).             #
# ------------------------------------------------------------------------------------------------ #
def audit_mhgin():
    print("\n=== MH-GIN (vendor MTS-HGNN via runner build_model, our-6h cell) ===", flush=True)
    mod = load_module("mhgin_runner_mod", os.path.join(BL, "MHGIN", "run_mhgin_gaps.py"))
    tr = mod._load(GAP6_TRACK, GAP6_H, "train")
    print(f"    train split {len(tr)} graphs -> fit_status (sog stats + dummy class lists)", flush=True)
    status = mod.fit_status(tr)
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    m = mod.build_model(status, "cpu", d_model=mod.D_MODEL, max_delta=350.0)
    n_tr, n_fr = count_params(m)

    # train s/epoch from the completed run's own artifacts (log wall times / epochs trained)
    secs = [float(ln.split("done in", 1)[1].split("s", 1)[0].strip())
            for ln in open(DRIVER_LOG_MHGIN) if "done in" in ln]
    eps = [int(float(r["epochs_trained"])) for r in _read_csv_rows(os.path.join(SWEEP6, "mhgin", "summary.csv"))
           if r["seed"] not in ("MEAN", "STD") and r.get("epochs_trained")]
    assert len(secs) == len(eps) == 5, f"expected 5 seeds in log+summary, got {len(secs)}/{len(eps)}"
    tr_s = float(np.mean([s / e for s, e in zip(secs, eps)]))
    print(f"    train timing [driver log]: seeds {[int(s) for s in secs]} s / {eps} epochs "
          f"-> mean {tr_s:.1f} s/epoch (incl. per-epoch val eval; CUDA run)", flush=True)

    mean_ratio = next(r["ratio"] for r in _read_csv_rows(os.path.join(SWEEP6, "mhgin", "summary.csv"))
                      if r["seed"] == "MEAN")
    assert abs(float(mean_ratio) - RATIOS["MH-GIN"]) < 5e-4, "MH-GIN summary MEAN ratio drifted"
    add_row("MH-GIN", n_tr, n_fr, None, tr_s,
            "vendor MTS-HGNN; 5 seeds; params=runner build_model (d_model 32, max_delta 350 vendor "
            "default, status from our-6h train split); ratio=5-seed mean (runs-tree summary.csv); "
            "train=driver-log seed wall/epochs incl. per-epoch val eval; inference not re-timed "
            "(a CPU timing would not be comparable)")
    del m


# ------------------------------------------------------------------------------------------------ #
#  MultitaskAIS -- vendor TF1.15 VRNN, trained per cell in the LOCAL mtais_tf1 conda env (CPU).      #
#  Param count = tf.trainable_variables() after building the vendor graph EXACTLY as                 #
#  mtais_tf1_run.run_train does (_build_model(data_dim, latent 100, mean.pkl) connected through      #
#  bounds.elbo), with data_dim/mean from the our-6h export (regenerated via mtais_export if the      #
#  export dir is absent).  Cross-checked against the trained seed0 checkpoint (non-Adam,             #
#  non-global-step variables must sum to the same count).  train_s_per_epoch from the run's own      #
#  train_meta.json (seed0: train_seconds / epochs_run, CPU TF1.15).                                  #
# ------------------------------------------------------------------------------------------------ #
MTAIS_WORKER = r"""
import json, os, sys
try:
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    pass
import numpy as np
import tensorflow as tf
try:
    delattr(tf.flags.FLAGS, "log_dir")      # avoid DuplicateFlagError with vendor flags_config
except Exception:
    pass
import importlib.util
HERE, EXPORT, CHKPT = sys.argv[1], sys.argv[2], sys.argv[3]
spec = importlib.util.spec_from_file_location("mtais_tf1_run", os.path.join(HERE, "mtais_tf1_run.py"))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
with open(os.path.join(EXPORT, "meta.json")) as f:
    meta = json.load(f)
data_dim = int(meta["roi"]["data_dim"])
mean = np.asarray(M._np_compat_load(os.path.join(EXPORT, "mean.pkl")), np.float64)
assert mean.shape[0] == data_dim
with tf.Graph().as_default():
    tf.set_random_seed(0)
    inputs = tf.placeholder(tf.float32, [None, None, data_dim])    # time-major, as run_train's dataset
    targets = tf.placeholder(tf.float32, [None, None, data_dim])
    lengths = tf.placeholder(tf.int32, [None])
    model = M._build_model(data_dim, 100, mean)                    # latent_size=100 (vendor README)
    M.bounds.elbo(model, (inputs, targets), lengths, num_samples=1)  # connects every submodule
    tvars = tf.trainable_variables()
    n_tr = sum(int(np.prod(v.shape.as_list())) for v in tvars)
n_ck = -1
# NOT tf.train.latest_checkpoint: the run executed on a Linux server, so the `checkpoint` pointer
# file stores absolute /home/... paths.  Resolve the newest step from the local *.index files.
ck = None
if CHKPT and os.path.isdir(CHKPT):
    import glob, re
    idx = glob.glob(os.path.join(CHKPT, "model.ckpt-*.index"))
    if idx:
        ck = max((f[:-len(".index")] for f in idx),
                 key=lambda p: int(re.search(r"-(\d+)$", p).group(1)))
if ck:
    n_ck = sum(int(np.prod(s)) for name, s in tf.train.list_variables(ck)
               if "Adam" not in name and "beta1_power" not in name
               and "beta2_power" not in name and name != "global_step")
print("MTAIS_AUDIT_JSON: " + json.dumps(dict(n_trainable=n_tr, n_variables=len(tvars),
      data_dim=data_dim, ckpt_model_params=n_ck, tf=tf.__version__)))
"""


def audit_mtais():
    print("\n=== MultitaskAIS (vendor VRNN, TF1.15 CPU, mtais_tf1 env, our-6h export) ===", flush=True)
    export_dir = os.path.join(BL, "MTAIS", "_export", GAP6_TRACK, f"gap_{GAP6_H}h")
    if not os.path.exists(os.path.join(export_dir, "meta.json")):
        print("    export dir absent -- regenerating via mtais_export (stage 1, this env)", flush=True)
        subprocess.check_call([sys.executable, "-u", "-m", "task_alpha.baselines.MTAIS.mtais_export",
                               "--track", GAP6_TRACK, "--gap", str(GAP6_H)], cwd=ROOT)
    chkpt = os.path.join(SWEEP6, "multitaskais", "seed0", "chkpt")
    # conda run mangles a multi-line `python -c` argument on Windows (conda crashes before python
    # starts), so the worker goes through a temp script file instead.
    import tempfile
    fd, worker_py = tempfile.mkstemp(suffix="_mtais_count.py", text=True)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(MTAIS_WORKER)
        r = subprocess.run([CONDA, "run", "-n", "mtais_tf1", "python", "-u", worker_py,
                            os.path.join(BL, "MTAIS"), export_dir, chkpt],
                           capture_output=True, text=True, cwd=ROOT, timeout=900)
    finally:
        os.unlink(worker_py)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("MTAIS_AUDIT_JSON: ")]
    if not line:
        print("    WORKER STDOUT:\n" + r.stdout[-3000:] + "\n    WORKER STDERR:\n" + r.stderr[-3000:],
              flush=True)
        raise RuntimeError("MultitaskAIS worker produced no result")
    res = json.loads(line[0][len("MTAIS_AUDIT_JSON: "):])
    print(f"    params: trainable {res['n_trainable']} in {res['n_variables']} variables "
          f"(data_dim {res['data_dim']}, latent 100, TF {res['tf']}) | seed0 checkpoint "
          f"cross-check {res['ckpt_model_params']}", flush=True)
    assert res["ckpt_model_params"] in (-1, res["n_trainable"]), \
        "graph param count does not match the trained seed0 checkpoint"

    with open(os.path.join(SWEEP6, "multitaskais", "seed0", "train_meta.json")) as f:
        tm = json.load(f)
    tr_s = float(tm["train_seconds"]) / max(int(tm["epochs_run"]), 1)
    print(f"    train timing [train_meta.json seed0]: {tm['train_seconds']} s / {tm['epochs_run']} "
          f"epochs = {tr_s:.1f} s/epoch (CPU TF1.15, batch {tm['batch_size']})", flush=True)

    add_row("MultitaskAIS", res["n_trainable"], 0, None, tr_s,
            "ONE-SIDED reference (forward VRNN, prior-sampling reconstruction); CPU-only TF1.15; "
            "trained per cell (param count for the our-6h ROI); ratio=5-seed mean "
            "(runs-tree summary.csv, +/-15.28); train=seed0 train_meta 20 ep; params=vendor graph "
            "via mtais_tf1_run._build_model+bounds.elbo, ckpt cross-checked")


def run_extension():
    audit_habit()
    audit_mhgin()
    audit_mtais()


def print_table():
    print("\n=== PARAMS/COMPUTE AUDIT (paper/figdata/params_audit.csv) ===", flush=True)
    hdr = (f"{'model':<13} {'trainable':>10} {'frozen':>8} {'ratio':>9} "
           f"{'infer_s/770':>12} {'train_s/ep':>11}")
    print(hdr); print("-" * len(hdr), flush=True)
    for k in MODEL_ORDER:
        if k not in ROWS:
            continue
        r = ROWS[k]
        print(f"{r['model']:<13} {r['n_params_trainable']:>10} {r['n_params_frozen']:>8} "
              f"{r['test_ratio_overall']:>9} {str(r['inference_s_per_770gaps']):>12} "
              f"{str(r['train_s_per_epoch']):>11}", flush=True)


def main_extend_only():
    """Load the six existing rows VERBATIM (byte-identical round-trip) and (re)generate only the
    HABIT / MH-GIN / MultitaskAIS rows.  No GPU memory is allocated on this path."""
    print(f"[audit] --extend-only: existing rows loaded verbatim from {OUT_CSV}", flush=True)
    for r in _read_csv_rows(OUT_CSV):
        if r["model"] not in EXT_MODEL_ORDER:          # never re-derive the six base rows
            ROWS[r["model"]] = r
    missing = [k for k in MODEL_ORDER[:6] if k not in ROWS]
    assert not missing, f"existing CSV is missing rows {missing}; run the full audit first"
    run_extension()
    print_table()
    print("\nAUDIT COMPLETE (extension rows only; the six base rows are not recomputed)", flush=True)


def main():
    print(f"[audit] device {DEV} | torch {torch.__version__} | "
          f"{torch.cuda.get_device_name(0)}", flush=True)
    tr, va, te = CG.load_gaps()
    assert len(te) == 774
    mean, std = CG.fit_latlon_stats(tr)                           # shared by BiGRU/CSDI (train-only stats)
    print(f"[audit] z-stats mean {np.round(mean, 4).tolist()} std {np.round(std, 4).tolist()}", flush=True)

    audit_fqgnn(tr, te)
    audit_rff(tr, te)
    audit_csdi(tr, te, mean, std)
    audit_saits(tr, te)
    audit_bf()
    run_extension()                                               # HABIT / MH-GIN / MultitaskAIS (CPU)
    audit_bigru(tr, te, mean, std)                                # LAST: cuDNN teardown crash-after-write

    print_table()
    print("\nAUDIT COMPLETE (CSV written; any exit-code crash below this line is the documented "
          "cuDNN DLL-detach issue, not a failure)", flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    if "--extend-only" in sys.argv[1:]:
        main_extend_only()
    else:
        main()

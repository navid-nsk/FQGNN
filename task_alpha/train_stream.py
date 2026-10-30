"""
task_alpha.train_stream  --  STREAMING all-data trainer with REAL train + validation, using the CUDA
Op-B eigensolver.  Streams the cached partitions (62 GB) one at a time so RAM stays bounded; the
vessel-disjoint split is by an md5 hash of the MMSI (see _split), so no vessel is in more than one
of train / val / test.

Reports the real 1-epoch TRAIN time (per-batch forward/backward) with the CUDA kernels, printed next
to CPU reference timings, AND the validation result (ADE vs GCD, bucketed) on held-out vessels.

Outputs: run_stream -> task_alpha/runs/stream_cuda/ (model_1ep.pt, result.txt); train_multi ->
task_alpha/runs/train_fp64/ by default (epochs.csv, steps.csv, best.pt, last.pt).
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import random
import time
import types

import numpy as np
import torch


def _split(mmsi):
    """Stateless VESSEL-DISJOINT 3-way split for streaming: 'train'(80%) / 'val'(10%) / 'test'(10%) /
    None(=drop invalid placeholder MMSI). Uses an md5 hash because Arctic MMSIs cluster at last-digit 0
    (so mmsi%k / low-bit hashes put ~96% on one side). A vessel always lands in the SAME set, so train,
    val and test share no vessel. test is RESERVED -- never trained on, evaluated once at the very end."""
    m = int(mmsi)
    if not (100000000 <= m <= 999999999):
        return None
    h = int(hashlib.md5(str(m).encode()).hexdigest(), 16) % 10
    return "test" if h == 0 else "val" if h == 1 else "train"

from .data import collate_graphs_interp, POS_OUT_SCALE
from .data_raw import build_dataset_raw
from .fq_interp import FQTrajectoryInterpolator
from .metrics import bucket_by_tsa, gcd_errors, model_errors


def _load_part(y, mo, gap_minutes, thin_sec, seed):
    return build_dataset_raw(years=[y], months=[mo], n_mmsi=None, max_files_per_part=None,
                             gap_minutes=gap_minutes, thin_sec=thin_sec, max_gap_len=40,
                             n_anchor=8, seed=seed)  # cached -> fast load, never rebuilds


def run_stream(years=tuple(range(2012, 2022)), months=tuple(range(1, 13)), n=6, n_freqs=48, width=96,
               init_scale=0.5, lr=1e-3, fq_scalar_lr=8e-5, max_grad_norm=1.0, batch_size=128,
               epochs=1, val_mod=5, cuda_eig=True, device="cuda", seed=0, gap_minutes=90.0,
               thin_sec=180):
    torch.manual_seed(seed)
    model = FQTrajectoryInterpolator(n=n, n_freqs=n_freqs, width=width, init_scale=init_scale,
                                     seed=seed, in_feat=6).to(device)
    model.use_cuda = bool(cuda_eig)                       # route ALL primitives (Op-B/C, QFE, Dmu, Op-D) through CUDA
    SCALE_KEYS = ("grassmann.log_scale", "s_raw", "rho")
    sp = [p for nm, p in model.named_parameters() if any(k in nm for k in SCALE_KEYS)]
    op = [p for nm, p in model.named_parameters() if not any(k in nm for k in SCALE_KEYS)]
    opt = torch.optim.Adam([{"params": op, "lr": lr}, {"params": sp, "lr": fq_scalar_lr}])
    parts = [(y, mo) for y in years for mo in months]
    cuda = device == "cuda"
    def sync():
        if cuda:
            torch.cuda.synchronize()
    print(f"STREAMING all-data train+val | cuda_eig={cuda_eig} | val=mmsi%%{val_mod}==0 (vessel-disjoint)",
          flush=True)

    # ---------------- TRAIN (streamed, timed) ----------------
    model.train()
    t_fwd = t_bwd = 0.0; n_batch = n_train = 0; warmed = False
    val_graphs = []                                          # SINGLE PASS: stash val graphs (no 2nd 62 GB load)
    t0 = time.perf_counter()
    for (y, mo) in parts:
        gs = _load_part(y, mo, gap_minutes, thin_sec, seed)
        tr = []
        for g in gs:
            if int(g.mask.sum()) == 0:
                continue
            s = _split(g.mmsi)                                       # 'test' is dropped here (reserved)
            (tr if s == "train" else val_graphs if s == "val" else []).append(g)
        random.Random(seed + y * 13 + mo).shuffle(tr)
        for bs in range(0, len(tr), batch_size):
            b = tr[bs:bs + batch_size]
            x, coord, yp, mask, ei, ed, aei = [t.to(device) for t in collate_graphs_interp(b)]
            sync(); a = time.perf_counter()
            pred = model(x, coord, ei, aei)                          # FORWARD (CUDA Op-B inside)
            sync(); bb = time.perf_counter()
            loss = ((pred[mask] - yp[mask]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            if torch.isfinite(loss):
                loss.backward()
                if max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()
            sync(); c = time.perf_counter()
            if warmed:                                               # skip first batch (CUDA warm-up)
                t_fwd += bb - a; t_bwd += c - bb; n_batch += 1
            else:
                warmed = True
        n_train += len(tr); del gs, tr
        if mo == 12:
            print(f"  train through {y}: cum train {n_train}, val stashed {len(val_graphs)}", flush=True)
    train_wall = time.perf_counter() - t0
    # checkpoint + report TRAIN results IMMEDIATELY (robust to interruption during the val pass)
    import os
    ckpt = "task_alpha/runs/stream_cuda/model_1ep.pt"
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": dict(n=n, n_freqs=n_freqs, width=width,
                init_scale=init_scale, seed=seed, cuda_eig=cuda_eig)}, ckpt)
    print(f"\n[TRAIN DONE] graphs {n_train} | batches {n_batch} | "
          f"FWD {1000*t_fwd/max(n_batch,1):.1f} ms/batch (CPU 190) | "
          f"BWD {1000*t_bwd/max(n_batch,1):.1f} ms/batch (CPU 42) | "
          f"epoch {train_wall/60:.1f} min (CPU 35.4) | ckpt -> {ckpt}", flush=True)

    # ---------------- VALIDATION (streamed, held-out vessels) ----------------
    model.eval(); errs, gcd, tsa = [], [], []
    t0v = time.perf_counter()
    with torch.no_grad():
        for bs in range(0, len(val_graphs), batch_size):             # BATCHED val on the STASHED graphs
            b = val_graphs[bs:bs + batch_size]
            x, coord, yp, mask, ei, ed, aei = [t.to(device) for t in collate_graphs_interp(b)]
            pred = model(x, coord, ei, aei)
            off = 0
            for g in b:
                ng = g.x.shape[0]
                errs.append(model_errors(pred[off:off + ng], g)); gcd.append(gcd_errors(g))
                tsa.append(g.t_since_anchor[g.mask].numpy()); off += ng
    val_wall = time.perf_counter() - t0v
    errs = np.concatenate(errs); gcd = np.concatenate(gcd); tsa = np.concatenate(tsa)
    assert np.isfinite(errs).mean() > 0.99, "non-finite val errors (NaN poison?)"
    m_ade = float(np.nanmean(errs)); g_ade = float(np.nanmean(gcd)); ratio = m_ade / max(g_ade, 1e-9)

    import os
    lines = [f"train graphs {n_train} | batches {n_batch} | val held points {len(errs)}",
             f"FORWARD {1000*t_fwd/max(n_batch,1):.1f} ms/batch (CPU 190) | BACKWARD "
             f"{1000*t_bwd/max(n_batch,1):.1f} ms/batch (CPU 42)",
             f"TRAIN epoch {train_wall/60:.1f} min (CPU 35.4) | VAL pass {val_wall:.1f} s",
             f"VAL ADE {m_ade:.3f} NM (median {np.nanmedian(errs):.3f}) | GCD {g_ade:.3f} "
             f"(median {np.nanmedian(gcd):.3f}) | ratio {ratio:.3f}"]
    for k, v in bucket_by_tsa(errs, tsa).items():
        lines.append(f"  bucket {k}: model {v:.3f} NM | gcd {bucket_by_tsa(gcd, tsa).get(k, float('nan')):.3f}")
    os.makedirs("task_alpha/runs/stream_cuda", exist_ok=True)
    with open("task_alpha/runs/stream_cuda/result.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n================ ALL-DATA 1-EPOCH (CUDA Op-B) ================")
    print(f"train graphs {n_train} | batches {n_batch} | val held-out points {len(errs)}")
    print(f"FORWARD  {t_fwd:7.1f} s | {1000*t_fwd/max(n_batch,1):6.1f} ms/batch   (CPU baseline 190 ms)")
    print(f"BACKWARD {t_bwd:7.1f} s | {1000*t_bwd/max(n_batch,1):6.1f} ms/batch   (CPU baseline 42 ms)")
    print(f"TRAIN epoch (fwd+bwd+load+step): {train_wall:7.1f} s = {train_wall/60:.1f} min "
          f"(CPU baseline ~35.4 min)")
    print(f"VAL pass: {val_wall:.1f} s")
    print(f"VAL ADE {m_ade:.3f} NM (median {np.nanmedian(errs):.3f}) | GCD {g_ade:.3f} "
          f"(median {np.nanmedian(gcd):.3f}) | ratio {ratio:.3f}")
    for k, v in bucket_by_tsa(errs, tsa).items():
        print(f"   model {k}: {v:.3f} NM | gcd {bucket_by_tsa(gcd, tsa).get(k, float('nan')):.3f}")
    return dict(train_min=train_wall / 60, fwd_ms=1000 * t_fwd / max(n_batch, 1),
                bwd_ms=1000 * t_bwd / max(n_batch, 1), model_ade=m_ade, gcd_ade=g_ade, ratio=ratio,
                n_train=n_train, n_val=int(len(errs)))


def train_multi(years=tuple(range(2012, 2022)), months=tuple(range(1, 13)), epochs_max=40, patience=6,
                batch_size=128, dtype="fp64", n=6, n_freqs=48, width=96, init_scale=0.5, lr=1e-3,
                fq_scalar_lr=8e-5, max_grad_norm=1.0, weight_decay=0.0, device="cuda", seed=0,
                gap_minutes=90.0, thin_sec=180, log_every=200, out_dir="task_alpha/runs/train_fp64"):
    """RESUMABLE multi-epoch all-data trainer (all CUDA kernels), early-stopping on val_ratio. Writes
    epochs.csv (the training curve + FQ-health ledger), steps.csv (per-step loss/grad), best.pt/last.pt.
    Survives process restarts: re-run the same call and it resumes from last.pt. dtype 'fp32' or 'fp64'."""
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    cfg = dict(n=n, n_freqs=n_freqs, width=width, init_scale=init_scale, seed=seed, cuda_eig=True,
               dtype=dtype)
    model = FQTrajectoryInterpolator(n=n, n_freqs=n_freqs, width=width, init_scale=init_scale,
                                     seed=seed, in_feat=6).to(device)
    model.use_cuda = True
    if dtype == "fp64":
        model = model.double()
    SCALE_KEYS = ("grassmann.log_scale", "s_raw", "rho")
    sp = [p for nm, p in model.named_parameters() if any(k in nm for k in SCALE_KEYS)]
    op = [p for nm, p in model.named_parameters() if not any(k in nm for k in SCALE_KEYS)]
    opt = torch.optim.Adam([{"params": op, "lr": lr, "weight_decay": weight_decay},
                            {"params": sp, "lr": fq_scalar_lr}])
    epoch0, best, pctr, step = 0, float("inf"), 0, 0
    last_ckpt = os.path.join(out_dir, "last.pt")
    if os.path.exists(last_ckpt):                                # RESUME
        ck = torch.load(last_ckpt, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        epoch0, best, pctr, step = ck["epoch"] + 1, ck["best"], ck["patience"], ck.get("step", 0)
        print(f"RESUMED from epoch {epoch0} | best val_ratio {best:.4f} | patience {pctr}/{patience}",
              flush=True)
    cuda = device == "cuda"
    def sync():
        if cuda:
            torch.cuda.synchronize()
    cast = (lambda t: t.double()) if dtype == "fp64" else (lambda t: t)
    parts = [(y, mo) for y in years for mo in months]
    ep_f = open(os.path.join(out_dir, "epochs.csv"), "a", newline="", encoding="utf-8"); ep_w = csv.writer(ep_f)
    st_f = open(os.path.join(out_dir, "steps.csv"), "a", newline="", encoding="utf-8"); st_w = csv.writer(st_f)
    if epoch0 == 0:
        ep_w.writerow(["epoch", "train_loss", "train_ade_nm", "val_ade_nm", "val_median_nm",
                       "val_gcd_nm", "val_ratio", "grad_norm", "min_minor", "lee_yang_ok",
                       "val_finite_frac", "minutes"]); ep_f.flush()
        st_w.writerow(["step", "epoch", "loss", "grad_norm", "ade_nm"]); st_f.flush()
    print(f"TRAIN-MULTI {dtype} | epochs_max {epochs_max} patience {patience} | all CUDA kernels | "
          f"out {out_dir}", flush=True)

    for epoch in range(epoch0, epochs_max):
        t0 = time.perf_counter(); model.train()
        ep_loss = ep_ade = gnorm_sum = 0.0; ep_n = 0; val_graphs = []
        for (y, mo) in parts:
            gs = _load_part(y, mo, gap_minutes, thin_sec, seed)
            tr = []
            for g in gs:
                if int(g.mask.sum()) == 0:
                    continue
                sset = _split(g.mmsi)
                (tr if sset == "train" else val_graphs if sset == "val" else []).append(g)
            random.Random(seed + epoch + y * 13 + mo).shuffle(tr)
            for bs in range(0, len(tr), batch_size):
                b = tr[bs:bs + batch_size]
                x, coord, yp, mask, ei, ed, aei = collate_graphs_interp(b)
                x, coord, yp = cast(x.to(device)), cast(coord.to(device)), cast(yp.to(device))
                ei, aei, mask = ei.to(device), aei.to(device), mask.to(device)
                pred = model(x, coord, ei, aei)
                loss = ((pred[mask] - yp[mask]) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                if torch.isfinite(loss):
                    loss.backward()
                    mgn = max_grad_norm if max_grad_norm else 1e9
                    gn = float(torch.nn.utils.clip_grad_norm_(op, mgn))     # main params (clip + report)
                    torch.nn.utils.clip_grad_norm_(sp, mgn)                 # FQ scalars clipped SEPARATELY
                    opt.step()
                    ade = float((pred[mask] - yp[mask]).pow(2).sum(-1).sqrt().mean() * (POS_OUT_SCALE / 1852.0))
                    ep_loss += float(loss); ep_ade += ade; gnorm_sum += gn if math.isfinite(gn) else 0.0
                else:
                    gn, ade = float("nan"), float("nan")            # health: NaN-guarded step
                ep_n += 1
                if step % log_every == 0:
                    st_w.writerow([step, epoch, float(loss), gn, ade]); st_f.flush()
                step += 1
            del gs, tr

        # ---- validate on held-out vessels ----
        model.eval(); errs, gcd, tsa = [], [], []
        with torch.no_grad():
            for bs in range(0, len(val_graphs), batch_size):
                b = val_graphs[bs:bs + batch_size]
                x, coord, yp, mask, ei, ed, aei = collate_graphs_interp(b)
                x, coord = cast(x.to(device)), cast(coord.to(device)); ei, aei = ei.to(device), aei.to(device)
                pred = model(x, coord, ei, aei).float(); off = 0
                for g in b:
                    ng = g.x.shape[0]
                    errs.append(model_errors(pred[off:off + ng], g)); gcd.append(gcd_errors(g))
                    tsa.append(g.t_since_anchor[g.mask].numpy()); off += ng
        errs = np.concatenate(errs); gcd = np.concatenate(gcd)
        fin = float(np.isfinite(errs).mean())
        val_ade = float(np.nanmean(errs)); val_med = float(np.nanmedian(errs))
        val_gcd = float(np.nanmean(gcd)); val_ratio = val_ade / max(val_gcd, 1e-9)
        try:                                                    # FQ in-family health: all minors > 0 (Lee-Yang)
            mm = float(model.min_minor()); ly = 1 if mm > 0 else 0
        except Exception:
            mm, ly = float("nan"), -1
        sec = time.perf_counter() - t0
        ep_w.writerow([epoch, ep_loss / max(ep_n, 1), ep_ade / max(ep_n, 1), val_ade, val_med, val_gcd,
                       val_ratio, gnorm_sum / max(ep_n, 1), mm, ly, fin, sec / 60]); ep_f.flush()
        improved = val_ratio < best - 1e-4
        if improved:
            best = val_ratio; pctr = 0
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "val_ratio": val_ratio},
                       os.path.join(out_dir, "best.pt"))
        else:
            pctr += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "best": best,
                    "patience": pctr, "step": step, "cfg": cfg}, last_ckpt)
        flag = "*BEST*" if improved else f"(patience {pctr}/{patience})"
        health = "OK" if (fin > 0.99 and ly == 1 and math.isfinite(mm) and mm > 0) else "CHECK"
        print(f"epoch {epoch}: loss {ep_loss/max(ep_n,1):.4f} | val ADE {val_ade:.3f} (med {val_med:.3f}) "
              f"GCD {val_gcd:.3f} ratio {val_ratio:.4f} {flag} | gnorm {gnorm_sum/max(ep_n,1):.2f} | "
              f"min_minor {mm:.3f} LY {ly} finite {fin:.3f} [{health}] | {sec/60:.1f} min", flush=True)
        del val_graphs
        if pctr >= patience:
            print(f"EARLY STOP at epoch {epoch} (best val_ratio {best:.4f})", flush=True); break
    ep_f.close(); st_f.close()
    print(f"DONE. best val_ratio {best:.4f} -> {out_dir}/best.pt", flush=True)
    return dict(best_val_ratio=best, out_dir=out_dir)


def speed_test(years=(2016, 2017, 2018), months=(7, 8), batch_size=128, n=6, n_freqs=48, width=96,
               init_scale=0.5, device="cuda", seed=0, max_batches=300, gap_minutes=90.0, thin_sec=180):
    """Per-batch forward/backward TRAINING speed with ALL CUDA kernels, in fp32 AND fp64, on the SAME
    cached graphs. Reports ms/batch + the extrapolated all-data train epoch (~922k graphs)."""
    torch.manual_seed(seed)
    graphs = []
    for y in years:
        for mo in months:
            graphs += [g for g in _load_part(y, mo, gap_minutes, thin_sec, seed)
                       if int(g.mask.sum()) > 0 and _split(g.mmsi) == "train"]
    print(f"speed test: {len(graphs)} graphs (warm cache), batch={batch_size}, max_batches={max_batches}")
    cuda = device == "cuda"
    def sync():
        if cuda:
            torch.cuda.synchronize()
    out = {}
    for dtype in ["fp32", "fp64"]:
        model = FQTrajectoryInterpolator(n=n, n_freqs=n_freqs, width=width, init_scale=init_scale,
                                         seed=seed, in_feat=6).to(device)
        model.use_cuda = True
        if dtype == "fp64":
            model = model.double()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        t_fwd = t_bwd = 0.0; nb = 0; warmed = False
        for bs in range(0, len(graphs), batch_size):
            b = graphs[bs:bs + batch_size]
            x, coord, yp, mask, ei, ed, aei = collate_graphs_interp(b)
            x, coord, yp = x.to(device), coord.to(device), yp.to(device)
            ei, aei, mask = ei.to(device), aei.to(device), mask.to(device)
            if dtype == "fp64":
                x, coord, yp = x.double(), coord.double(), yp.double()
            sync(); a = time.perf_counter()
            pred = model(x, coord, ei, aei)
            sync(); bb = time.perf_counter()
            loss = ((pred[mask] - yp[mask]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            if torch.isfinite(loss):
                loss.backward(); opt.step()
            sync(); cc = time.perf_counter()
            if warmed:
                t_fwd += bb - a; t_bwd += cc - bb; nb += 1
            else:
                warmed = True
            if nb >= max_batches:
                break
        fwd_ms = 1000 * t_fwd / max(nb, 1); bwd_ms = 1000 * t_bwd / max(nb, 1)
        epoch_min = (fwd_ms + bwd_ms) * (922496 / batch_size) / 1000 / 60
        out[dtype] = dict(fwd_ms=fwd_ms, bwd_ms=bwd_ms, epoch_min=epoch_min)
        print(f"  {dtype}: FWD {fwd_ms:6.1f} ms/batch | BWD {bwd_ms:6.1f} | fwd+bwd {fwd_ms+bwd_ms:6.1f} "
              f"| extrapolated all-data train epoch ~{epoch_min:.1f} min  ({nb} batches)")
    return out


def evaluate(ckpt_path, which="test", years=tuple(range(2012, 2022)), months=tuple(range(1, 13)),
             batch_size=128, device="cuda", gap_minutes=90.0, thin_sec=180, seed=0):
    """Load a trained checkpoint and score it ONCE on the held-out `which` split ('test' or 'val').
    Vessel-disjoint (same _split), streamed. Use 'test' only for the FINAL reported number."""
    ck = torch.load(ckpt_path, weights_only=False); cfg = ck["cfg"]
    model = FQTrajectoryInterpolator(n=cfg["n"], n_freqs=cfg["n_freqs"], width=cfg["width"],
                                     init_scale=cfg["init_scale"], seed=cfg["seed"], in_feat=6).to(device)
    model.use_cuda = bool(cfg.get("cuda_eig"))
    model.load_state_dict(ck["state_dict"]); model.eval()
    errs, gcd, tsa = [], [], []
    with torch.no_grad():
        for (y, mo) in [(y, mo) for y in years for mo in months]:
            gs = _load_part(y, mo, gap_minutes, thin_sec, seed)
            ev = [g for g in gs if int(g.mask.sum()) > 0 and _split(g.mmsi) == which]
            for bs in range(0, len(ev), batch_size):
                b = ev[bs:bs + batch_size]
                x, coord, yp, mask, ei, ed, aei = [t.to(device) for t in collate_graphs_interp(b)]
                pred = model(x, coord, ei, aei); off = 0
                for g in b:
                    ng = g.x.shape[0]
                    errs.append(model_errors(pred[off:off + ng], g)); gcd.append(gcd_errors(g))
                    tsa.append(g.t_since_anchor[g.mask].numpy()); off += ng
            del gs, ev
    errs = np.concatenate(errs); gcd = np.concatenate(gcd); tsa = np.concatenate(tsa)
    m_ade = float(np.nanmean(errs)); g_ade = float(np.nanmean(gcd))
    print(f"[{which.upper()}] held points {len(errs)} | model ADE {m_ade:.3f} (median "
          f"{np.nanmedian(errs):.3f}) | GCD {g_ade:.3f} (median {np.nanmedian(gcd):.3f}) | "
          f"ratio {m_ade/max(g_ade,1e-9):.3f}", flush=True)
    for k, v in bucket_by_tsa(errs, tsa).items():
        print(f"   {k}: model {v:.3f} | gcd {bucket_by_tsa(gcd, tsa).get(k, float('nan')):.3f}")
    return dict(which=which, model_ade=m_ade, gcd_ade=g_ade, ratio=m_ade / max(g_ade, 1e-9),
                n=int(len(errs)))


if __name__ == "__main__":
    run_stream()

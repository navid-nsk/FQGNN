"""paper_figures_tables/dump_spectra.py -- learned-quasicrystal spectrum + diffraction dump.

ANALYSIS ONLY (no training; GPU inference only; fp32).  Never modifies any model source.

Loads
  * the trained FQGNN  seed0  (task_alpha/runs/fqnative_sobolev_5seed/seed0/best.pt -- the
    SOBOLEV-backbone Op-H model) into a fresh FQDeepNative with the EXACT
    run_fqnative_sobolev_5seed ctor (n=6, n_freqs=48, width=96, init_scale=1.84, seed=0,
    in_feat=6, layers=3, jacobi_R=4, oph_weighted=True, dmu_source="blend",
    metric_backbone="sobolev2", s_raw init 2.5; blend Dmu -> the dummy forward needs gap_minutes),
  * a byte-identical FRESH INIT copy (untrained, seed 0, same construction sequence),
  * the trained RFF-GNN seed0 (task_alpha/baselines/RFFGNN/runs_5seed_gaps/seed0/best.pt) via the
    runner's own RFFGNN class (frozen W_rff/b_rff buffers restored from the checkpoint).

Writes (FQGNN_FIGDATA):
  fq_spectrum_trained.csv / fq_spectrum_init.csv
      k_index, xi_0, xi_1, xi_norm, eff_freq (=2*pi*s*|xi|, the angular frequency actually applied
      to coord -- directly comparable to the RFF |W column|), Re_c, Im_c, abs_c, abs_c_sq
  fq_L_trained.csv / fq_L_init.csv    the Grassmann lift L [n,2]
  fq_U_trained.csv                    |U_ij| grid of the secular unitary U=exp(iH) (Fig-1 badge)
  rff_spectrum.csv                    j, w_0, w_1, freq_norm, b_rff, readout_w (=||enc_r.weight[:,j]||,
                                      the learned per-frequency height analog), readout_w_sq
  spectrum_stats.csv                  comb sparsity/decay diagnostics per model (+ scalar s, L scale)
  sk_fq.csv / sk_rff.csv              1-D structure factor of the frequency SET itself: Gaussian-
                                      smoothed histogram of pairwise |freq_i - freq_j| differences

Run:  python paper_figures_tables/dump_spectra.py   (any cwd)
"""
from __future__ import annotations

import csv
import importlib.util
import math
import os
import random
import sys
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

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FQ_CKPT = os.path.join(ROOT, "task_alpha", "runs", "fqnative_sobolev_5seed", "seed0", "best.pt")
# ---- OP-M STACK (FQ_READOUT_MODE=opm): spectra are derived from the Op-M paper model's seed-0
# checkpoint (the spectrum lives in the shared backbone; the readout switch only adds params, so
# the same builder + env constructs the matching module).  Launch under the full stack env with
# FQ_AMP_REFRESH unset.  With the env unset, the sobolev checkpoint above is used.
if os.environ.get("FQ_READOUT_MODE") == "opm":
    assert os.environ.get("FQ_AMP_REFRESH", "0") in ("", "0"), \
        "FQ_AMP_REFRESH must be UNSET for official spectra (exact tables)"
    FQ_CKPT = os.path.join(ROOT, "task_alpha", "runs", "fqnative_opmstack_5seed", "seed0", "best.pt")
RFF_CKPT = os.path.join(ROOT, "task_alpha", "baselines", "RFFGNN", "runs_5seed_gaps", "seed0", "best.pt")
RFF_RUNNER = os.path.join(ROOT, "task_alpha", "baselines", "RFFGNN", "run_rffgnn_5seed_gaps.py")
CTOR = dict(n=6, n_freqs=48, width=96, init_scale=1.84, seed=0, in_feat=6, layers=3, jacobi_R=4,
            oph_weighted=True, dmu_source="blend", metric_backbone="sobolev2")
PI = math.pi


# ------------------------------------------------------------------------------------------------ #
#  model builders (EXACT run_fqnative_sobolev_5seed construction sequence, seed 0)                          #
# ------------------------------------------------------------------------------------------------ #
def build_fq(tr4):
    """Fresh FQDeepNative exactly as run_fqnative_sobolev_5seed.run_seed(seed=0) builds it
    (same seeding, same s_raw init, same dummy forward on tr[:4] materializing the LazyLinear;
    gap_minutes passed because the blend Dmu jet requires it in forward())."""
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    m = FQDeepNative(**CTOR).to(DEV)
    m.use_cuda = True
    with torch.no_grad():
        m.s_raw.fill_(math.log(math.expm1(2.5)))
    b0 = collate_graphs_interp(tr4)
    bv = torch.cat([torch.full((g.x.shape[0],), i, dtype=torch.long) for i, g in enumerate(tr4)]).to(DEV)
    gm = torch.tensor([float(g.gap_minutes) for g in tr4], device=DEV)
    with torch.no_grad():
        m(b0[0].to(DEV), b0[1].to(DEV), b0[4].to(DEV), b0[6].to(DEV), bv, gap_minutes=gm)
    return m


def load_rff():
    spec = importlib.util.spec_from_file_location("rff_runner_mod", RFF_RUNNER)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    m = mod.RFFGNN(seed=0).to(DEV)
    sd = torch.load(RFF_CKPT, map_location=DEV)
    m.load_state_dict(sd["state_dict"]); m.eval()
    return m


# ------------------------------------------------------------------------------------------------ #
#  extraction + diagnostics                                                                          #
# ------------------------------------------------------------------------------------------------ #
def extract_quasicrystal(m, batch):
    """Run m._feats on a small REAL batch (exact telemetry call of run_full_faithful.py) and return
    the learned quasicrystal (L, xi, c, s) as numpy."""
    x, co = batch
    with torch.no_grad():
        node, L, xi, c, s = m._feats(x.to(DEV), co.to(DEV))
    assert torch.isfinite(node).all(), "non-finite node features"
    return (L.detach().cpu().numpy(), xi.detach().cpu().numpy(),
            c.detach().cpu().numpy(), float(s.detach().cpu()))


def write_fq_spectrum(path, xi, c, s):
    nrm = np.linalg.norm(xi, axis=1)
    eff = 2.0 * PI * s * nrm
    rows = [dict(k_index=k, xi_0=f"{xi[k,0]:.6f}", xi_1=f"{xi[k,1]:.6f}", xi_norm=f"{nrm[k]:.6f}",
                 eff_freq=f"{eff[k]:.6f}", Re_c=f"{c[k].real:.8e}", Im_c=f"{c[k].imag:.8e}",
                 abs_c=f"{abs(c[k]):.8e}", abs_c_sq=f"{abs(c[k])**2:.8e}")
            for k in range(len(c))]
    _check_finite(np.c_[xi, nrm, eff, c.real, c.imag, np.abs(c)], path)
    _write(path, rows)
    return nrm, eff


def comb_stats(freq, amp):
    """Comb sparsity + decay of a discrete spectrum: energy of top-10 stems / total, and the fitted
    power-law exponent alpha of |amp| ~ |freq|^alpha (log-log least squares over nonzero stems)."""
    freq = np.asarray(freq, float); amp = np.abs(np.asarray(amp, float))
    e = amp ** 2; tot = e.sum()
    top10 = float(np.sort(e)[::-1][:10].sum() / tot)
    good = (amp > 1e-14) & (freq > 1e-14)
    alpha = float(np.polyfit(np.log(freq[good]), np.log(amp[good]), 1)[0])
    return top10, alpha


def sk_curve(freqs, nbins=240, sigma_bins=2.0):
    """1-D structure factor of the frequency SET: Gaussian-smoothed density histogram of pairwise
    |freq_i - freq_j| (i<j) -- a simple, honest spectral-rigidity visual."""
    r = np.sort(np.asarray(freqs, float))
    iu = np.triu_indices(len(r), k=1)
    d = np.abs(r[:, None] - r[None, :])[iu]
    hi = float(d.max()) * 1.02
    hist, edges = np.histogram(d, bins=nbins, range=(0.0, hi), density=True)
    rad = int(max(1, round(4 * sigma_bins)))
    ker = np.exp(-0.5 * (np.arange(-rad, rad + 1) / sigma_bins) ** 2); ker /= ker.sum()
    smooth = np.convolve(hist, ker, mode="same")
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, smooth


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"  wrote {os.path.relpath(path, ROOT)}  ({len(rows)} rows)", flush=True)


def _check_finite(arr, name):
    a = np.asarray(arr, float)
    assert np.isfinite(a).all(), f"NaN/Inf in {name}"


def main():
    os.makedirs(FIGDATA, exist_ok=True)
    tr, va, te = CG.load_gaps()
    tr4 = tr[:4]                                   # the small REAL batch (same one the runner uses)
    b = collate_graphs_interp(tr4); batch = (b[0], b[1])

    # ---- FQGNN: trained + fresh init ----
    m_tr = build_fq(tr4)
    sd = torch.load(FQ_CKPT, map_location=DEV)
    m_tr.load_state_dict(sd["state_dict"]); m_tr.eval()
    m_in = build_fq(tr4); m_in.eval()              # untrained, byte-identical init (seed 0)

    L_tr, xi_tr, c_tr, s_tr = extract_quasicrystal(m_tr, batch)
    L_in, xi_in, c_in, s_in = extract_quasicrystal(m_in, batch)
    M = xi_tr.shape[0]
    print(f"\n[FQ] spectrum M={M} atoms, xi shape {xi_tr.shape} (2-D), c complex, "
          f"s_trained={s_tr:.4f} s_init={s_in:.4f}", flush=True)

    nrm_tr, eff_tr = write_fq_spectrum(os.path.join(FIGDATA, "fq_spectrum_trained.csv"), xi_tr, c_tr, s_tr)
    nrm_in, eff_in = write_fq_spectrum(os.path.join(FIGDATA, "fq_spectrum_init.csv"), xi_in, c_in, s_in)

    for tag, L in (("trained", L_tr), ("init", L_in)):
        _check_finite(L, f"L_{tag}")
        _write(os.path.join(FIGDATA, f"fq_L_{tag}.csv"),
               [dict(row=i, L_0=f"{L[i,0]:.8f}", L_1=f"{L[i,1]:.8f}") for i in range(L.shape[0])])

    with torch.no_grad():
        U_tr = m_tr.secular.unitary().detach().cpu().numpy()
        U_in = m_in.secular.unitary().detach().cpu().numpy()
    _check_finite(np.abs(U_tr), "U_trained")
    _write(os.path.join(FIGDATA, "fq_U_trained.csv"),
           [dict(row=i, **{f"absU_{j}": f"{abs(U_tr[i,j]):.6f}" for j in range(U_tr.shape[1])})
            for i in range(U_tr.shape[0])])

    # ---- RFF-GNN: frozen random spectrum + learned readout weight ----
    rm = load_rff()
    W = rm.W_rff.detach().cpu().numpy()            # [2, 96] frozen random frequency vectors
    brf = rm.b_rff.detach().cpu().numpy()          # [96]
    enc_w = rm.enc_r.weight.detach().cpu().numpy() # [width, 96]
    fr_rff = np.linalg.norm(W, axis=0)             # per-frequency norm |W[:,j]|
    rw = np.linalg.norm(enc_w, axis=0)             # learned per-frequency readout weight (height analog)
    _check_finite(np.c_[W.T, fr_rff, brf, rw], "rff_spectrum")
    _write(os.path.join(FIGDATA, "rff_spectrum.csv"),
           [dict(k_index=j, w_0=f"{W[0,j]:.6f}", w_1=f"{W[1,j]:.6f}", freq_norm=f"{fr_rff[j]:.6f}",
                 b_rff=f"{brf[j]:.6f}", readout_w=f"{rw[j]:.6f}", readout_w_sq=f"{rw[j]**2:.6f}")
            for j in range(W.shape[1])])

    # ---- diagnostics: comb sparsity / decay (rho_pp proxy on an all-discrete comb) ----
    stats = []
    for model, freq, amp, s_scale, L_scale in (
            ("FQGNN_trained", eff_tr, np.abs(c_tr), s_tr, float(np.exp(m_tr.grassmann.log_scale.detach().cpu()))),
            ("FQGNN_init",    eff_in, np.abs(c_in), s_in, float(np.exp(m_in.grassmann.log_scale.detach().cpu()))),
            ("RFF_seed0",     fr_rff, rw,           "",   "")):
        top10, alpha = comb_stats(freq, amp)
        stats.append(dict(model=model, n_freqs=len(freq),
                          s_scale=f"{s_scale:.4f}" if s_scale != "" else "",
                          L_scale=f"{L_scale:.4f}" if L_scale != "" else "",
                          top10_energy_frac=f"{top10:.4f}", energy_decay_exponent=f"{alpha:.4f}",
                          freq_min=f"{np.min(freq):.4f}", freq_med=f"{np.median(freq):.4f}",
                          freq_max=f"{np.max(freq):.4f}"))
    _write(os.path.join(FIGDATA, "spectrum_stats.csv"), stats)

    # ---- S(k) of the frequency sets themselves (spectral rigidity visual) ----
    for tag, freqs in (("fq", eff_tr), ("rff", fr_rff)):
        cbin, val = sk_curve(freqs)
        _check_finite(np.c_[cbin, val], f"sk_{tag}")
        _write(os.path.join(FIGDATA, f"sk_{tag}.csv"),
               [dict(bin_center=f"{c0:.6f}", value=f"{v0:.6f}") for c0, v0 in zip(cbin, val)])

    # ---- compact stdout summary ----
    dL = float(np.linalg.norm(L_tr - L_in) / np.linalg.norm(L_in))
    dU = float(np.linalg.norm(U_tr - U_in) / np.linalg.norm(U_in))
    dc = float(np.linalg.norm(np.abs(c_tr) - np.abs(c_in)) / np.linalg.norm(np.abs(c_in)))
    print("\n=== N5 SUMMARY ===", flush=True)
    print(f"{'model':<15} {'M':>4} {'s':>8} {'top10_E%':>9} {'decay_a':>8} "
          f"{'f_min':>8} {'f_med':>8} {'f_max':>8}", flush=True)
    for r in stats:
        print(f"{r['model']:<15} {r['n_freqs']:>4} {r['s_scale'] or '-':>8} "
              f"{100*float(r['top10_energy_frac']):>8.1f}% {r['energy_decay_exponent']:>8} "
              f"{r['freq_min']:>8} {r['freq_med']:>8} {r['freq_max']:>8}", flush=True)
    print(f"\nmovement from init:  ||L_tr-L_in||/||L_in|| = {dL:.4f}  "
          f"(L scale exp(log_scale): init {float(np.exp(m_in.grassmann.log_scale.detach().cpu())):.4f} "
          f"-> trained {float(np.exp(m_tr.grassmann.log_scale.detach().cpu())):.4f})", flush=True)
    print(f"                     ||U_tr-U_in||/||U_in|| = {dU:.4f}   "
          f"|||c_tr|-|c_in|||/|||c_in||| = {dc:.4f}", flush=True)
    print(f"                     band scale s: init {s_in:.4f} -> trained {s_tr:.4f}", flush=True)
    print(f"effective angular freq = 2*pi*s*|xi| (FQ) vs |W_rff column| (RFF); both applied to the "
          f"same unit-normalized coord.", flush=True)
    print("all outputs finite (asserted).", flush=True)


if __name__ == "__main__":
    main()

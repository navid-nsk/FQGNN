"""diag_training -- READ-ONLY diagnostic: understand what happens DURING training of
FQDeepFaithful.  Does NOT change the model or any code; it trains a fresh instance and logs telemetry.

Per epoch it records:
  - val_ratio                : the learning curve vs GCD
  - grad_<group>             : gradient norm per parameter group  (what FLOWS)
  - chg_<group>              : relative param movement ||p-p0||/||p0|| from init  (what actually LEARNS)
  - agg_filter_mag           : |theta_agg(h)*P|  -- magnitude of the per-layer Op-D data-conditioned kernel
                               filter (if ~0 the aggregation kernel is inert -> message passing not shaping it)
  - readout_cond_dev         : |filt_out - 1|    -- how far the readout kernel amplitudes deviate from the
                               FIXED Op-K-Hermite (if ~0 the readout ~ the closed form -> conditioning inactive)
Writes task_alpha/runs/diag_training.csv.   Run: python -m task_alpha.diag_training
"""
from __future__ import annotations

import csv
import math
import random
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch

from .data import collate_graphs_interp
from .data_raw import build_dataset_raw
from .fqdeep_faithful import FQDeepFaithful
from .meltin_proof import _bvec, _eval
from .train_stream import _split

DEV = "cuda"

GROUPS = {
    "enc": ["enc."], "jacobi_filter(agg)": ["theta."], "phase_lift(score)": ["phase_lift."],
    "score_scale": ["score_scale"], "wv(value)": ["wv."], "update(lin/upd/norm)": ["lin.", "upd.", "norm."],
    "readout_filter": ["theta_out."], "fq_spectrum(L/s/env)": ["grassmann", "s_raw", "rho"],
    "secular_U(variety)": ["secular."], "w_fq(resid)": ["w_fq"],
}


def group_of(name):
    # native-readout heads (FQDeepNative: eta_head / w_head / gate_rho) stay OUT of the op-k
    # telemetry groups -- without this guard "gate_rho" substring-matches the QFE envelope "rho"
    # pattern and contaminates grad_/chg_fq_spectrum.  No-op for FQDeepFaithful (no such names).
    # "opm_" (Op-M readout dials, FQ_READOUT_MODE=opm: opm_gamma_raw/opm_kappa_raw/opm_lam_head/
    # opm_rho_m) likewise -- "opm_rho_m" would otherwise substring-match the same "rho" pattern.
    if any(k in name for k in ("gate_rho", "eta_head", "w_head", "opm_")):
        return "oph_heads"
    for g, ks in GROUPS.items():
        if any(k in name for k in ks):
            return g
    return "other"


def main():
    allg = build_dataset_raw(years=(2016,), months=(8,), gap_minutes=360, max_gap_len=500, min_span_factor=1.5,
                             n_anchor=8, clean_only=True, n_mmsi=None, max_files_per_part=None, seed=0)
    allg = [g for g in allg if int(g.mask.sum()) > 0 and _split(g.mmsi) is not None]
    tr = [g for g in allg if _split(g.mmsi) == "train"]
    va = [g for g in allg if _split(g.mmsi) == "val"]
    print(f"{len(tr)} train / {len(va)} val", flush=True)

    m = FQDeepFaithful(n=6, n_freqs=48, width=96, init_scale=1.84, seed=0, in_feat=6, layers=3, jacobi_R=4).to(DEV)
    m.use_cuda = True
    with torch.no_grad():
        m.s_raw.fill_(math.log(math.expm1(2.5)))
    b0 = collate_graphs_interp(tr[:4])
    m(b0[0].to(DEV), b0[1].to(DEV), b0[4].to(DEV), b0[6].to(DEV), _bvec(tr[:4]))    # init lazy enc
    init = {n: p.detach().clone() for n, p in m.named_parameters()}
    opt = torch.optim.Adam(m.parameters(), lr=5e-4)

    db = tr[:64]; dx, dco, dyp, dmk, dei, _, dae = collate_graphs_interp(db)         # fixed diagnostic batch
    DX, DCO, DEI, DAE, DBV = dx.to(DEV), dco.to(DEV), dei.to(DEV), dae.to(DEV), _bvec(db)
    DMK, DYT = dmk.to(DEV), dyp.to(DEV)

    def measure():
        fq, gc, _ = _eval(m, va); vr = float(np.median(fq) / np.median(gc))
        m.train(); opt.zero_grad()
        loss = ((m(DX, DCO, DEI, DAE, DBV)[DMK] - DYT[DMK]) ** 2).mean()
        loss.backward()
        gsq = {g: 0.0 for g in list(GROUPS) + ["other"]}
        dsq = {g: 0.0 for g in list(GROUPS) + ["other"]}; psq = {g: 0.0 for g in list(GROUPS) + ["other"]}
        for n, p in m.named_parameters():
            g = group_of(n)
            if p.grad is not None:
                gsq[g] += float(p.grad.norm()) ** 2
            dsq[g] += float((p.detach() - init[n]).norm()) ** 2; psq[g] += float(init[n].norm()) ** 2
        opt.zero_grad()
        gn = {g: gsq[g] ** 0.5 for g in gsq}
        rc = {g: dsq[g] ** 0.5 / max(psq[g] ** 0.5, 1e-9) for g in dsq}
        with torch.no_grad():
            node, L, xi, c, s = m._feats(DX, DCO); h = m.enc(node); P = m._jacobi(xi)
            fa = float((m.theta[0](h[DEI[0]]) @ P.t()).abs().mean())          # agg kernel filter magnitude
            fo = float((m.theta_out(h.mean(0, keepdim=True)) @ P.t() - 1).abs().mean())  # readout cond deviation
        return vr, gn, rc, fa, fo

    rows = []
    for ep in range(16):
        if ep > 0:
            m.train(); o = list(range(len(tr))); random.Random(ep).shuffle(o)
            for i in range(0, len(o), 64):
                b = [tr[j] for j in o[i:i + 64]]; x, co, yp, mk, ei, _, ae = collate_graphs_interp(b)
                pred = m(x.to(DEV), co.to(DEV), ei.to(DEV), ae.to(DEV), _bvec(b))
                L = ((pred[mk.to(DEV)] - yp.to(DEV)[mk.to(DEV)]) ** 2).mean()
                opt.zero_grad()
                if torch.isfinite(L):
                    L.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        vr, gn, rc, fa, fo = measure()
        rows.append(dict(epoch=ep, val_ratio=round(vr, 4), agg_filter_mag=round(fa, 5), readout_cond_dev=round(fo, 5),
                         **{f"grad_{g}": gn[g] for g in GROUPS}, **{f"chg_{g}": rc[g] for g in GROUPS}))
        print(f"ep {ep:2d}: val {vr:.3f} | agg_filter {fa:.4f} | readout_dev {fo:.4f}", flush=True)

    with open("task_alpha/runs/diag_training.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print("\n=== GRADIENT (last epoch) + PARAM MOVEMENT init->last, per group ===")
    last = rows[-1]
    for g in GROUPS:
        print(f"  {g:24s} grad {last[f'grad_{g}']:.2e}   moved {100*last[f'chg_{g}']:.2f}%")
    print(f"\nconditioning activity: agg_filter_mag {rows[0]['agg_filter_mag']:.4f}->{last['agg_filter_mag']:.4f} | "
          f"readout_cond_dev {rows[0]['readout_cond_dev']:.4f}->{last['readout_cond_dev']:.4f}")
    print("saved task_alpha/runs/diag_training.csv")


if __name__ == "__main__":
    main()

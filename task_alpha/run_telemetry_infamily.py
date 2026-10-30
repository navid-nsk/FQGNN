"""IN-FAMILY telemetry run on FQDeepFaithful (Fig 1c + ED4 use run_telemetry_infamily_native.py).

ONE seed (0), the EXACT main recipe of run_full_faithful_5seed.run_seed (model n=6, n_freqs=48,
width=96, init_scale=1.84, layers=3, jacobi_R=4, s_raw init 2.5; Adam lr 5e-4, clip 1.0, bs 64,
<=60 epochs, patience 8) on the load_gaps() data -- the model / data builders are NOT touched.
The ONLY addition: per-epoch IN-FAMILY telemetry of the learned quasicrystal itself.

What is inspected (from fqdeep_faithful.FQDeepFaithful -> fq_imputer.FQTrajectoryImputer):
  * L = m.grassmann() in Gr_+(2, n=6): the (r, theta) chart, d=2, freeze_incommensurate=True (frozen
    sqrt-prime chart directions + ONE learnable global log_scale).  min_minor = min over ALL
    C(6,2)=15 exact 2x2 minors via m.grassmann.min_minor(L) (itertools.combinations inside
    grassmann.minors -- exactly the "all dxd minors" definition).  The chart is structurally
    in-family, so min_minor > 0 for every parameter value; this run MEASURES that it stays > 0.
  * THE MODEL HAS AN EXPLICIT COUPLING UNITARY: m.secular (fqmath.lee_yang.SecularUnitary),
    U = exp(iH), the shared Lee-Yang variety used by BOTH the amplitudes c_k (log-p_U cumulant)
    and the per-layer Op-B/Op-C attention scores.  So eig_gap = the minimum CIRCULAR gap between
    the sorted eigenphases of U (wrap-around gap included).  The min_xi_gap fallback is therefore
    NOT needed; we still log min_xi_gap (min pairwise ||xi_j - xi_k||) as a free extra column for
    ED4 (aperiodicity witness of the truncated spectrum) -- documented bonus, not a substitute.
  * amplitudes c = m.quasicrystal()[2] (variety-fixed cumulant, complex [M=48]):
    amp_l2 = ||c||_2 and amp_max = max|c_k| (amplitude blow-up monitor).
  * spec_spread = max||xi_k|| / min||xi_k|| over the nonzero frequency vectors xi = L^T k [M,2].

Outputs (task_alpha/runs/telemetry_infamily/):
  telemetry_infamily.csv  epoch (-1 = init), train_loss, val_ratio, val_model_med, val_gcd_med,
                          min_minor, eig_gap, amp_l2, amp_max, spec_spread, min_xi_gap
  L_init.csv / L_final.csv        the [n=6, 2] generator matrix L at init / after the last epoch
  comb_init.csv / comb_final.csv  k_index, xi_abs (=||xi_k||_2), c_abs (=|c_k|) at init / final
  best.pt                         best-val checkpoint (recipe-standard)

Run (from the repository root):  python -m task_alpha.run_telemetry_infamily [--smoke]
  --smoke: 200 train / 60 val graphs, 2 epochs, outputs to runs/telemetry_infamily_smoke/.
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import random
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch

from .baselines.common_gaps import load_gaps
from .data import collate_graphs_interp
from .fqdeep_faithful import FQDeepFaithful
from .meltin_proof import _bvec, _eval

DEV = "cuda"
SEED = 0
PI = math.pi


# ------------------------------------------------------------------------------------------------ #
#  IN-FAMILY telemetry (read-only inspection of the CURRENT learned quasicrystal)                    #
# ------------------------------------------------------------------------------------------------ #
@torch.no_grad()
def infamily_stats(m) -> dict:
    """min_minor / eig_gap / amp_l2 / amp_max / spec_spread (+ bonus min_xi_gap) of the model NOW."""
    L, xi, c, s = m.quasicrystal()                       # L [n,2], xi [M,2], c complex [M]
    # 1. min over ALL C(n,2) 2x2 minors of the CURRENT L (grassmann.minors = itertools.combinations)
    min_minor = float(m.grassmann.min_minor(L))
    # 2. the model HAS an explicit coupling unitary U = exp(iH) (SecularUnitary) -> eigenphase gap.
    #    Circular: gaps between consecutive sorted phases + the wrap-around gap.
    U = m.secular.unitary()                              # [n,n] complex, unitary by construction
    ph = torch.sort(torch.angle(torch.linalg.eigvals(U.cpu()))).values   # tiny n=6 -> CPU eig
    gaps = torch.cat([ph[1:] - ph[:-1], (ph[0] + 2 * PI - ph[-1]).unsqueeze(0)])
    eig_gap = float(gaps.min())
    # 3. amplitude blow-up monitor
    ca = c.abs()
    amp_l2, amp_max = float(torch.linalg.vector_norm(ca)), float(ca.max())
    # 4. spectral spread over the NONZERO frequency vectors
    nrm = xi.norm(dim=-1)
    nz = nrm[nrm > 1e-12]
    spec_spread = float(nz.max() / nz.min())
    # bonus (ED4 aperiodicity witness; NOT the eig_gap substitute -- U exists): min pairwise xi gap
    D = torch.cdist(xi.unsqueeze(0), xi.unsqueeze(0)).squeeze(0)
    D.fill_diagonal_(float("inf"))
    min_xi_gap = float(D.min())
    return dict(min_minor=min_minor, eig_gap=eig_gap, amp_l2=amp_l2, amp_max=amp_max,
                spec_spread=spec_spread, min_xi_gap=min_xi_gap)


@torch.no_grad()
def dump_state(m, out: str, tag: str):
    """Write L_<tag>.csv ([n,2] generator) and comb_<tag>.csv (k_index, |xi|, |c|)."""
    L, xi, c, s = m.quasicrystal()
    Lc = L.detach().cpu().numpy()
    with open(f"{out}/L_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["row", "L0", "L1"])
        for i in range(Lc.shape[0]):
            w.writerow([i, repr(float(Lc[i, 0])), repr(float(Lc[i, 1]))])
    xn = xi.norm(dim=-1).detach().cpu().numpy(); cn = c.abs().detach().cpu().numpy()
    with open(f"{out}/comb_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["k_index", "xi_abs", "c_abs"])
        for k in range(len(xn)):
            w.writerow([k, repr(float(xn[k])), repr(float(cn[k]))])


# ------------------------------------------------------------------------------------------------ #
#  the run: EXACT main recipe + the per-epoch in-family row                                          #
# ------------------------------------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="200/60 graphs, 2 epochs, _smoke out dir")
    args = ap.parse_args()

    out = "task_alpha/runs/telemetry_infamily" + ("_smoke" if args.smoke else "")
    os.makedirs(out, exist_ok=True)
    tr, va, te = load_gaps()                             # identical data + vessel-disjoint split
    max_ep = 60
    if args.smoke:
        tr, va, max_ep = tr[:200], va[:60], 2

    # ---- model construction IDENTICAL to run_full_faithful_5seed.run_seed(seed=0) ----
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    m = FQDeepFaithful(n=6, n_freqs=48, width=96, init_scale=1.84, seed=SEED, in_feat=6, layers=3,
                       jacobi_R=4).to(DEV)
    m.use_cuda = True
    with torch.no_grad():
        m.s_raw.fill_(math.log(math.expm1(2.5)))
    b0 = collate_graphs_interp(tr[:4])
    m(b0[0].to(DEV), b0[1].to(DEV), b0[4].to(DEV), b0[6].to(DEV), _bvec(tr[:4]))   # lazy-init fwd
    opt = torch.optim.Adam(m.parameters(), lr=5e-4)

    # ---- epoch-0 (init) dumps + init telemetry row ----
    dump_state(m, out, "init")
    fields = ["epoch", "train_loss", "val_ratio", "val_model_med", "val_gcd_med",
              "min_minor", "eig_gap", "amp_l2", "amp_max", "spec_spread", "min_xi_gap"]
    st0 = infamily_stats(m)
    rows = [{"epoch": -1, "train_loss": "", "val_ratio": "", "val_model_med": "", "val_gcd_med": "",
             **{k: round(v, 8) for k, v in st0.items()}}]
    print(f"[init] min_minor {st0['min_minor']:.6f} | eig_gap {st0['eig_gap']:.6f} | "
          f"amp_l2 {st0['amp_l2']:.4f} amp_max {st0['amp_max']:.4f} | "
          f"spec_spread {st0['spec_spread']:.3f} | min_xi_gap {st0['min_xi_gap']:.6f}", flush=True)

    def write_rows():
        with open(f"{out}/telemetry_infamily.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    write_rows()

    # ---- training loop: byte-identical recipe (Adam 5e-4, clip 1.0, bs 64, patience 8) ----
    best, best_state, pctr, bs = 1e9, None, 0, 64
    for ep in range(max_ep):
        m.train(); order = list(range(len(tr))); random.Random(SEED * 1000 + ep).shuffle(order)
        ep_loss, ep_nb = 0.0, 0
        for i in range(0, len(order), bs):
            b = [tr[j] for j in order[i:i + bs]]; x, co, yp, mk, ei, _, ae = collate_graphs_interp(b)
            L = ((m(x.to(DEV), co.to(DEV), ei.to(DEV), ae.to(DEV), _bvec(b))[mk.to(DEV)]
                  - yp.to(DEV)[mk.to(DEV)]) ** 2).mean()
            opt.zero_grad()
            if torch.isfinite(L):
                L.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
                ep_loss += float(L.detach()); ep_nb += 1
        fq, gc, _ = _eval(m, va)
        vm, vg = float(np.median(fq)), float(np.median(gc)); vr = vm / vg
        st = infamily_stats(m)
        rows.append({"epoch": ep, "train_loss": round(ep_loss / max(ep_nb, 1), 6),
                     "val_ratio": round(vr, 4), "val_model_med": round(vm, 4),
                     "val_gcd_med": round(vg, 4), **{k: round(v, 8) for k, v in st.items()}})
        write_rows()
        if vr < best - 1e-4:
            best, best_state, pctr = vr, copy.deepcopy(m.state_dict()), 0
            torch.save({"state_dict": best_state}, f"{out}/best.pt")
        else:
            pctr += 1
        print(f"ep {ep:2d}: train {ep_loss / max(ep_nb, 1):.5f} | val {vr:.4f} (best {best:.4f}) | "
              f"min_minor {st['min_minor']:.6f} eig_gap {st['eig_gap']:.6f} | "
              f"amp_l2 {st['amp_l2']:.4f} amp_max {st['amp_max']:.4f} | "
              f"spread {st['spec_spread']:.3f}", flush=True)
        if pctr >= 8:
            print("early stop", flush=True); break

    # ---- final dumps (state after the LAST trained epoch, matching the last CSV row) ----
    dump_state(m, out, "final")
    mmins = [r["min_minor"] for r in rows]
    print(f"\n[telemetry] {len(rows) - 1} epochs trained | best val_ratio {best:.4f}", flush=True)
    print(f"[telemetry] min_minor over run: min {min(mmins):.6f} max {max(mmins):.6f} -> "
          f"stayed > 0 the WHOLE run: {min(mmins) > 0}", flush=True)
    print(f"[telemetry] eig_gap over run: min {min(r['eig_gap'] for r in rows):.6f} | "
          f"amp_max final {rows[-1]['amp_max']:.4f} | saved {out}/telemetry_infamily.csv "
          f"+ L_init/L_final + comb_init/comb_final + best.pt", flush=True)


if __name__ == "__main__":
    main()

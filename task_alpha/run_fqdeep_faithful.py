"""Train the FAITHFUL FQ-DEEP and run its load-bearing ablation protocol (tests 1, 3 and 6 below).
Arms (all trained identically; the prediction comes from the message passing, NO closed-form kriging):
  - FQ-DEEP full            (Op-D per-layer data-conditioned aggregation + FQ readout)
  - kernel-null (test 1)    use['opd']=False: K_Th(delta)->const in the AGGREGATION -> local; must DROP if MP load-bearing
  - global-theta (test 3)   use['cond']=False: theta_r const (fixed Lee-Yang amps); conditioned>global => message drives kernel
  - depth T=1 (test 6)      one layer vs T=3
Results are printed to stdout.
Run: python -m task_alpha.run_fqdeep_faithful
"""
from __future__ import annotations

import copy
import math
import random
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch

from .data import collate_graphs_interp
from .data_raw import build_dataset_raw
from .fqdeep_faithful import FQDeepFaithful, jacobi_basis
from .meltin_proof import _bvec, _eval, _report
from .train_stream import _split

DEV = "cuda"


def train(tr, va, epochs, seed, factory, tag, fqp_lr=5e-4):
    torch.manual_seed(seed)
    m = factory(seed)
    m(*[t.to(DEV) for t in collate_graphs_interp(tr[:4])[:2]], collate_graphs_interp(tr[:4])[4].to(DEV),
      collate_graphs_interp(tr[:4])[6].to(DEV), _bvec(tr[:4]))   # init lazy enc
    fqk = ("grassmann.log_scale", "s_raw", "rho")
    fqp = [p for n, p in m.named_parameters() if any(k in n for k in fqk)]
    fqid = {id(p) for p in fqp}; gp = [p for p in m.parameters() if id(p) not in fqid]
    opt = torch.optim.Adam([{"params": gp, "lr": 5e-4}, {"params": fqp, "lr": fqp_lr}])
    best, bs_state, pc, bs = 1e9, None, 0, 64
    print(f"\n=== {tag} (seed {seed}) ===", flush=True)
    for ep in range(epochs):
        m.train(); order = list(range(len(tr))); random.Random(ep).shuffle(order)
        for i in range(0, len(order), bs):
            b = [tr[j] for j in order[i:i + bs]]
            x, coord, yp, mask, ei, ed, aei = collate_graphs_interp(b)
            pred = m(x.to(DEV), coord.to(DEV), ei.to(DEV), aei.to(DEV), _bvec(b))
            msk = mask.to(DEV); loss = ((pred[msk] - yp.to(DEV)[msk]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            if torch.isfinite(loss):
                loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        fq, gc, _ = _eval(m, va); rm = float(np.median(fq) / np.median(gc))
        if rm < best - 1e-4:
            best, bs_state, pc = rm, copy.deepcopy(m.state_dict()), 0
        else:
            pc += 1
        print(f"  ep {ep:2d}: val {rm:.4f} (best {best:.4f})", flush=True)
        if pc >= 7:
            print("  early stop", flush=True); break
    if bs_state is not None:
        m.load_state_dict(bs_state)
    return m


def fq_factory(seed, layers=3):
    m = FQDeepFaithful(n=6, n_freqs=48, width=96, init_scale=1.84, seed=seed, in_feat=6, layers=layers, jacobi_R=4).to(DEV)
    m.use_cuda = True
    with torch.no_grad():
        m.s_raw.fill_(math.log(math.expm1(2.5)))
    return m


def main():
    allg = build_dataset_raw(years=(2016, 2017, 2018), months=(7, 8), gap_minutes=360, max_gap_len=500,
                             min_span_factor=1.5, n_anchor=8, clean_only=True, n_mmsi=None,
                             max_files_per_part=None, seed=0)
    allg = [g for g in allg if int(g.mask.sum()) > 0 and _split(g.mmsi) is not None]
    tr = [g for g in allg if _split(g.mmsi) == "train"]
    va = [g for g in allg if _split(g.mmsi) == "val"]
    te = [g for g in allg if _split(g.mmsi) == "test"]
    print(f"{len(tr)} train / {len(va)} val / {len(te)} test", flush=True)

    m = train(tr, va, 40, 0, fq_factory, "FQ-DEEP full (T=3)")
    rFull = _report(m, te, "FQ-DEEP full (TEST)")["overall"]
    # POSITIVE load-bearing proof: knock out each of OUR FQ ops, see if error rises
    res = {"full": rFull}
    for key, lab in [("opd", "kernel-null (test1: MP local)"), ("cond", "global-theta (test3)"), ("vel", "no-Dmu")]:
        m.use[key] = False; res[key] = _report(m, te, f"  ablate {lab}")["overall"]; m.use[key] = True
    mT1 = train(tr, va, 40, 0, lambda s: fq_factory(s, layers=1), "depth T=1 (test6)")
    res["T1"] = _report(mT1, te, "depth T=1 (TEST)")["overall"]

    # POSITIVE load-bearing proof of OUR OWN FQ ops (no RFF, no competitor):
    print("\n========== FAITHFUL FQ-DEEP -- LOAD-BEARING OF OUR FQ OPS ==========")
    print(f"  FQ-DEEP full            {res['full']:.4f}  (vs GCD 1.0)")
    print(f"  kernel-null (Op-D off)  {res['opd']:.4f}   gain {res['opd']-res['full']:+.4f}  (>0 => Op-D message passing carries weight)")
    print(f"  global-theta (no cond)  {res['cond']:.4f}   gain {res['cond']-res['full']:+.4f}  (>0 => Op-B/C data-conditioning carries weight)")
    print(f"  no-Dmu velocity         {res['vel']:.4f}   gain {res['vel']-res['full']:+.4f}  (>0 => Dmu carries weight)")
    print(f"  depth T=1 vs T=3        {res['T1']:.4f}   gain {res['T1']-res['full']:+.4f}  (>0 => stacking Op-D layers helps)")


if __name__ == "__main__":
    main()

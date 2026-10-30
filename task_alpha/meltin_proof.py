"""meltin_proof -- the rigorous load-bearing experiment for the melt-in FQGNN, on ONE fixed data split.

Three arms, identical data / spectrum init / epochs, so the ONLY difference is what is allowed to learn:
  A) GCD                         (floor)
  B) closed-form Op-K-Hermite    (hcond=False: ONLY the FQ spectrum trains; the kriging IS the prediction)
  C) melt-in FQGNN               (hcond=True : spectrum + message passing shaping the kernel)
The message passing is load-bearing IFF C beats B (same spectrum freedom; C adds the conditioning).

Diagnostics tracked so the plateau is UNDERSTOOD not guessed: the FQ scale s over epochs (is the spectrum
training?), and ||g-1|| the conditioning magnitude (is the message passing actually moving the kernel?).
Per-curvature buckets so we see WHERE any gain comes from.  fqp has its own LR (fqp_lr) so the spectrum trains.
Writes task_alpha/runs/meltin_proof.pt (state dicts of arms B and C) and prints the comparison.
Run: python -m task_alpha.meltin_proof
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
from .experiment_compare import _sinuosity
from .fqgnn import FQGNN
from .metrics import gcd_errors, model_errors
from .train_stream import _split

DEV = "cuda"


def _bvec(b):
    return torch.cat([torch.full((g.x.shape[0],), i, dtype=torch.long) for i, g in enumerate(b)]).to(DEV)


def _eval(m, graphs, bs=64):
    m.eval(); fq, gc, sn = [], [], []
    with torch.no_grad():
        for s in range(0, len(graphs), bs):
            b = graphs[s:s + bs]
            x, coord, yp, mask, ei, ed, aei = collate_graphs_interp(b)
            pred = m(x.to(DEV), coord.to(DEV), ei.to(DEV), aei.to(DEV), _bvec(b)).float().cpu()
            off = 0
            for g in b:
                ng = g.x.shape[0]; gp = pred[off:off + ng]; off += ng
                fe = model_errors(gp, g)
                if len(fe):
                    fq.append(np.median(fe)); gc.append(np.median(gcd_errors(g))); sn.append(_sinuosity(g))
    return np.array(fq), np.array(gc), np.array(sn)


def _report(m, te, tag):
    fq, gc, sn = _eval(m, te); keep = np.isfinite(sn); fq, gc, sn = fq[keep], gc[keep], sn[keep]
    r = float(np.median(fq) / np.median(gc))
    print(f"[{tag}] {len(fq)} gaps OVERALL ratio_med {r:.4f} (GCD {np.median(gc):.2f} model {np.median(fq):.2f})")
    out = {"overall": r}
    for lo, hi in zip((0.0, 0.02, 0.05, 0.10, 0.25), (0.02, 0.05, 0.10, 0.25, 1e9)):
        sel = (sn >= lo) & (sn < hi)
        if sel.sum() < 3:
            continue
        rr = float(np.median(fq[sel]) / np.median(gc[sel]))
        lab = f"{lo:.2f}-{hi:.2f}" if hi < 1e9 else f">{lo:.2f}"
        print(f"     {lab:<11} n={int(sel.sum()):4d} GCD {np.median(gc[sel]):.2f} model {np.median(fq[sel]):.2f} ratio {rr:.3f}")
        out[lab] = rr
    return out


def _cur_s(m):
    with torch.no_grad():
        _, _, _, s = m.quasicrystal()
        return float(s)


def _gdev(m, te, n=32):
    """mean ||g-1|| over n test gaps -- how far the learned conditioning moves the kernel from closed-form."""
    m.eval(); devs = []
    with torch.no_grad():
        b = te[:n]; x, coord, yp, mask, ei, ed, aei = collate_graphs_interp(b)
        h, xi, c, s = m.embed(x.to(DEV), coord.to(DEV), ei.to(DEV))
        bv = _bvec(b); held_all = x[:, 5].to(DEV) > 0.5
        asrc, adst = aei[0].to(DEV), aei[1].to(DEV); gid_adst = bv[adst]
        for gid in torch.unique(bv):
            anc = torch.unique(asrc[gid_adst == gid])
            if anc.numel() < 2:
                continue
            pool = h[anc].mean(0)
            g = 1.0 + 0.5 * torch.tanh(m.amp_head(pool))
            devs.append(float((g - 1).abs().mean()))
    return float(np.mean(devs)) if devs else 0.0


def train_arm(tr, va, te, hcond, tag, epochs=40, seed=0, fqp_lr=5e-4, model_factory=None):
    torch.manual_seed(seed)
    if model_factory is not None:
        m = model_factory(seed)                            # custom model factory (already on DEV, s set)
    else:
        m = FQGNN(n=6, n_freqs=48, width=128, init_scale=1.84, seed=seed, in_feat=6, opk_ridge=0.03).to(DEV)
        m.use_cuda = True
        with torch.no_grad():                              # init the FQ scale at its SWEEP OPTIMUM (s=2.5)
            m.s_raw.fill_(math.log(math.expm1(2.5)))       # so the spectrum starts near-optimal, not 3.6x too small
    m.use["hcond"] = hcond
    fqkeys = ("grassmann.log_scale", "s_raw", "rho")
    fqp = [p for nm, p in m.named_parameters() if any(k in nm for k in fqkeys)]
    fqid = {id(p) for p in fqp}; gnnp = [p for p in m.parameters() if id(p) not in fqid]
    opt = torch.optim.Adam([{"params": gnnp, "lr": 1e-3}, {"params": fqp, "lr": fqp_lr}])
    best, best_state, pctr, bs = 1e9, None, 0, 64
    print(f"\n=== ARM {tag} (hcond={hcond}, seed={seed}, fqp_lr={fqp_lr}) ===  init s={_cur_s(m):.3f}")
    for ep in range(epochs):
        m.train(); order = list(range(len(tr))); random.Random(ep).shuffle(order)
        for i in range(0, len(order), bs):
            b = [tr[j] for j in order[i:i + bs]]
            x, coord, yp, mask, ei, ed, aei = collate_graphs_interp(b)
            pred = m(x.to(DEV), coord.to(DEV), ei.to(DEV), aei.to(DEV), _bvec(b))
            msk = mask.to(DEV); yt = yp.to(DEV); loss = ((pred[msk] - yt[msk]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            if torch.isfinite(loss):
                loss.backward(); torch.nn.utils.clip_grad_norm_(gnnp, 1.0); torch.nn.utils.clip_grad_norm_(fqp, 1.0); opt.step()
        fq, gc, _ = _eval(m, va); rm = float(np.median(fq) / np.median(gc))
        diag = f"s={_cur_s(m):.3f}" + (f" |g-1|={_gdev(m, va):.3f}" if hcond else "")
        if rm < best - 1e-4:
            best, best_state, pctr = rm, copy.deepcopy(m.state_dict()), 0
        else:
            pctr += 1
        print(f"  ep {ep:2d}: val {rm:.4f} (best {best:.4f})  {diag}", flush=True)
        if pctr >= 8:
            print("  early stop", flush=True); break
    if best_state is not None:
        m.load_state_dict(best_state)
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

    mB = train_arm(tr, va, te, hcond=False, tag="B closed-form", seed=0)
    rB = _report(mB, te, "B closed-form Op-K-Hermite (TEST)")
    mC = train_arm(tr, va, te, hcond=True, tag="C melt-in", seed=0)
    rC = _report(mC, te, "C melt-in FQGNN (TEST)")
    # in-model toggle on the trained melt-in (same weights, kernel conditioning off):
    mC.use["hcond"] = False; rCoff = _report(mC, te, "C melt-in with conditioning OFF (TEST)")
    mC.use["hcond"] = True
    print("\n========== LOAD-BEARING VERDICT ==========")
    print(f"  GCD               ratio 1.0000")
    print(f"  B closed-form     ratio {rB['overall']:.4f}")
    print(f"  C melt-in         ratio {rC['overall']:.4f}   (vs B: {rB['overall']-rC['overall']:+.4f})")
    print(f"  C cond OFF        ratio {rCoff['overall']:.4f}   (in-model toggle)")
    print("  => message passing IS load-bearing" if rC['overall'] < rB['overall'] - 0.005
          else "  => message passing NOT load-bearing on this task (conditioning <= closed-form)")
    torch.save({"B": mB.state_dict(), "C": mC.state_dict()}, "task_alpha/runs/meltin_proof.pt")


if __name__ == "__main__":
    main()

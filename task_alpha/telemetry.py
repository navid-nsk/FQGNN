"""
task_alpha.telemetry  --  full CSV telemetry of the training process.

Writes FOUR CSVs (one per category) into an output dir, with FIXED columns so they are
easy to load/plot:

  forward.csv    : per logged step, ONE ROW PER FQ OPERATOR -- the forward-pass output statistics of
                   each operator (captured via forward hooks): mean/std/L2-norm/min/max/shape.
  backward.csv   : per logged step -- the loss (+ components), the TOTAL gradient norm, lr, and the
                   batch ADE/FDE metric (the backward-pass summary).
  gradient.csv   : per logged step, ONE ROW PER PARAMETER TENSOR -- grad norm/mean/absmax + weight
                   norm + grad/weight ratio (the per-parameter gradient detail).
  operators.csv  : per logged step, ONE ROW PER FQ OPERATOR -- the in-family DIAGNOSTICS
                   (min_minor, log-minor barrier, amoeba/coupling/eigenphase-gap certificate,
                   bragg-participation, Parseval residual, U off-diagonal energy).

Everything is flushed immediately (so a crash still leaves the rows on disk).
"""

from __future__ import annotations

import csv
import math
import os
from typing import Dict, List

import torch

# fixed column schemas
COLS = {
    "forward": ["step", "epoch", "split", "module", "kind", "shape", "out_mean", "out_std",
                "out_l2", "out_min", "out_max", "frac_finite"],
    "backward": ["step", "epoch", "split", "loss", "loss_resid", "loss_extra", "grad_norm_total",
                 "lr", "ade_nm", "fde_nm", "n_held"],
    "gradient": ["step", "epoch", "param", "numel", "grad_l2", "grad_mean", "grad_absmax",
                 "weight_l2", "grad_over_weight"],
    "operators": ["step", "epoch", "module", "optype", "min_minor", "log_minor_barrier",
                  "amoeba_margin", "coupling_margin", "eigenphase_gap", "bragg_participation",
                  "parseval_residual", "u_offdiag", "lee_yang_ok"],
}


class CSVTelemetry:
    def __init__(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self.dir = out_dir
        self._fh, self._wr = {}, {}
        for name, cols in COLS.items():
            fh = open(os.path.join(out_dir, name + ".csv"), "w", newline="", encoding="utf-8")
            wr = csv.DictWriter(fh, fieldnames=cols)
            wr.writeheader(); fh.flush()
            self._fh[name], self._wr[name] = fh, wr
        self._fwd_buf: List[dict] = []
        self._hooks = []

    def _row(self, name: str, row: dict):
        self._wr[name].writerow({k: row.get(k, "") for k in COLS[name]})
        self._fh[name].flush()

    # ---- forward hooks ----------------------------------------------------------------
    def attach_forward_hooks(self, model, type_names=("FQ", "Secular", "Spectral", "Variety",
                                                      "Curve", "Eigenphase", "Quasicrystal",
                                                      "Velocity", "ZeroField", "Conv")):
        """Register forward hooks on every submodule whose class name contains an FQ token, to
        capture its output statistics each forward pass."""
        def mk(name, cls):
            def hook(_m, _inp, out):
                t = out[0] if isinstance(out, (tuple, list)) and len(out) else out
                if not torch.is_tensor(t):
                    return
                tf = t.detach()
                fin = torch.isfinite(tf)
                vals = tf[fin].float() if fin.any() else tf.float().nan_to_num()
                self._fwd_buf.append({
                    "module": name, "kind": cls, "shape": "x".join(map(str, tuple(tf.shape))),
                    "out_mean": float(vals.mean()) if vals.numel() else 0.0,
                    "out_std": float(vals.std()) if vals.numel() > 1 else 0.0,
                    "out_l2": float(tf.float().norm()),
                    "out_min": float(vals.min()) if vals.numel() else 0.0,
                    "out_max": float(vals.max()) if vals.numel() else 0.0,
                    "frac_finite": float(fin.float().mean()),
                })
            return hook
        for name, m in model.named_modules():
            cls = type(m).__name__
            if any(tok in cls for tok in type_names) and name:
                self._hooks.append(m.register_forward_hook(mk(name, cls)))

    def flush_forward(self, step: int, epoch: int, split: str):
        for r in self._fwd_buf:
            r.update(step=step, epoch=epoch, split=split)
            self._row("forward", r)
        self._fwd_buf.clear()

    def log_forward_primitives(self, step: int, epoch: int, split: str, model):
        """forward.csv from a model exposing primitive_stats() (the faithful imputer): one row per
        FQ primitive (Op-A'/QFE, Op-B, Op-C, Op-D kernel/aggregate, D-mu velocity, ...)."""
        if not hasattr(model, "primitive_stats"):
            return
        for r in model.primitive_stats():
            r.update(step=step, epoch=epoch, split=split)
            self._row("forward", r)

    # ---- backward / metric ------------------------------------------------------------
    def log_backward(self, step, epoch, split, loss, loss_resid, loss_extra, grad_norm_total,
                     lr, ade_nm, fde_nm, n_held):
        self._row("backward", dict(step=step, epoch=epoch, split=split, loss=float(loss),
                                   loss_resid=float(loss_resid), loss_extra=float(loss_extra),
                                   grad_norm_total=float(grad_norm_total), lr=float(lr),
                                   ade_nm=float(ade_nm), fde_nm=float(fde_nm), n_held=int(n_held)))

    # ---- per-parameter gradients ------------------------------------------------------
    def log_gradients(self, step, epoch, model):
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach().float(); w = p.detach().float()
            gl2 = float(g.norm()); wl2 = float(w.norm())
            self._row("gradient", dict(
                step=step, epoch=epoch, param=name, numel=p.numel(), grad_l2=gl2,
                grad_mean=float(g.mean()), grad_absmax=float(g.abs().max()),
                weight_l2=wl2, grad_over_weight=gl2 / max(wl2, 1e-12)))

    # ---- FQ operator diagnostics ------------------------------------------------------
    @torch.no_grad()
    def log_operators(self, step, epoch, model):
        from fqmath.genericity import (amoeba_positive_ray_margin, coupling_margin,
                                        eigenphase_min_gap)
        from fqmath.diffraction import bragg_participation
        # model-level shared-quasicrystal ledger (the faithful imputer's diagnostics)
        if hasattr(model, "diagnostics"):
            try:
                d = model.diagnostics()
                self._row("operators", dict(step=step, epoch=epoch, module="__shared_quasicrystal__",
                                            optype=type(model).__name__, **d))
            except Exception:
                pass
        for name, m in model.named_modules():
            if not name:
                continue
            row = dict(step=step, epoch=epoch, module=name, optype=type(m).__name__)
            hit = False
            # cone witness + barrier
            if hasattr(m, "min_minor"):
                try:
                    row["min_minor"] = float(m.min_minor()); hit = True
                except Exception:
                    pass
            for attr, gname in (("grassmann", "log_minor_barrier"), ("grassmann_k", "log_minor_barrier")):
                gm = getattr(m, attr, None)
                if gm is not None and hasattr(gm, "log_minor_barrier"):
                    try:
                        row["log_minor_barrier"] = float(gm.log_minor_barrier()); hit = True
                    except Exception:
                        pass
            # secular unitary -> certificate
            U = None; ell = None
            if hasattr(m, "U") and callable(getattr(m, "U")):
                try:
                    U = m.U()
                except Exception:
                    U = None
            if U is None and hasattr(m, "secular") and hasattr(m.secular, "unitary"):
                try:
                    U = m.secular.unitary()
                except Exception:
                    U = None
            gobj = getattr(m, "grassmann_k", None) or getattr(m, "grassmann", None)
            if gobj is not None:
                try:
                    ell = gobj.forward()[:, 0]
                except Exception:
                    ell = None
            if U is not None and U.dim() == 2:
                try:
                    row["coupling_margin"] = float(coupling_margin(U)); hit = True
                    row["eigenphase_gap"] = float(eigenphase_min_gap(U))
                    if ell is not None:
                        row["amoeba_margin"] = float(amoeba_positive_ray_margin(U, ell.detach()))
                        row["lee_yang_ok"] = int((row.get("amoeba_margin", 0) > 0)
                                                 and (row["coupling_margin"] > 1e-3)
                                                 and (row["eigenphase_gap"] > 1e-3))
                except Exception:
                    pass
            if hasattr(m, "u_offdiag_energy"):
                try:
                    row["u_offdiag"] = float(m.u_offdiag_energy()); hit = True
                except Exception:
                    pass
            # spectrum diagnostics
            spec = getattr(m, "spectrum", None)
            if spec is not None and hasattr(spec, "amplitudes"):
                try:
                    row["bragg_participation"] = float(bragg_participation(spec.amplitudes())); hit = True
                except Exception:
                    pass
            if hit:
                self._row("operators", row)

    def close(self):
        for h in self._hooks:
            h.remove()
        for fh in self._fh.values():
            fh.close()

"""
task_alpha.metrics  --  ADE / FDE in nautical miles (great-circle).

The model predicts the ABSOLUTE held-out position (in POS_OUT_SCALE units), NOT a residual over any
baseline.  We map the predicted ENU -> lat/lon and measure the great-circle (Haversine) error in NM
on the HELD-OUT vertices.  The GCD baseline (the linear-interp guess) is computed SEPARATELY for
comparison -- it is never given to the model.
"""

from __future__ import annotations

import numpy as np
import torch

from .data import enu_to_lonlat, haversine_nm, POS_OUT_SCALE


def model_errors(pred: torch.Tensor, g) -> np.ndarray:
    """Per-held-out great-circle error (NM) of the model's ABSOLUTE position prediction.
    pred [N,2] in POS_OUT_SCALE units -> ENU metres -> lat/lon -> Haversine vs truth."""
    m = g.mask.numpy()
    if m.sum() == 0:
        return np.array([])
    enu = pred.detach().cpu().numpy() * POS_OUT_SCALE                  # ENU metres
    lat0, lon0 = g.ref
    plat, plon = enu_to_lonlat(enu[m, 0], enu[m, 1], lat0, lon0)
    tr = g.true_lonlat.numpy()
    return haversine_nm(tr[m, 0], tr[m, 1], plat, plon)


def gcd_errors(g) -> np.ndarray:
    """Per-held-out great-circle error (NM) of the GCD / linear-interp BASELINE (never fed to model)."""
    m = g.mask.numpy()
    if m.sum() == 0:
        return np.array([])
    li = g.linterp_lonlat.numpy(); tr = g.true_lonlat.numpy()
    return haversine_nm(tr[m, 0], tr[m, 1], li[m, 0], li[m, 1])


def ade_fde(pred: torch.Tensor, g):
    e = model_errors(pred, g)
    if len(e) == 0:
        return float("nan"), float("nan"), e
    return float(e.mean()), float(e[-1]), e


def bucket_by_tsa(errors_nm: np.ndarray, tsa_min: np.ndarray,
                  edges=(0, 30, 60, 120, 240, 480, 1e9)) -> dict:
    """Mean ADE per time-since-anchor bucket (minutes)."""
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (tsa_min >= lo) & (tsa_min < hi)
        if sel.any():
            out[f"{int(lo)}-{int(hi) if hi < 1e9 else 'inf'}min"] = float(errors_nm[sel].mean())
    return out

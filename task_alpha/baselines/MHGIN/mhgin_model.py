"""MH-GIN (MTS-HGNN) model wrapper.  The vendor model is imported and used UNCHANGED.

Nothing under MH-GIN-main/src/models is edited.  We bypass `main.py` /
`denmark_data_process_module.py` / `preprocess_data.py` (they are Denmark-CSV specific, they
auto-DOWNLOAD AIS from the internet, and their corruption scheme can only express scattered
single-point masking -- never our contiguous two-sided gap).  Everything below is a data adapter and
an inference path.
"""
from __future__ import annotations

import logging
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR = os.path.abspath(os.path.join(_HERE, "..", "MH-GIN-main"))
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

from src.models.MTSHGNN import MtsHGnn  # noqa: E402  (vendor, unmodified)
from src.utils.utils import seed_everything  # noqa: E402

from task_alpha.baselines.MHGIN.mhgin_data import FAKE, LAT, LON  # noqa: E402

__all__ = ["build_model", "predict_latlon", "seed_everything", "MtsHGnn"]


def _logger():
    lg = logging.getLogger("mhgin")
    if not lg.handlers:
        lg.addHandler(logging.NullHandler())
    lg.setLevel(logging.ERROR)
    return lg


def build_model(data_status, device, d_model: int = 32, max_delta: float = 350.0) -> MtsHGnn:
    """Vanilla hyperparameters from the authors' `config-DK.yaml` (d_data_model = d_mining_model = 32).

    `graph_mask_values=FAKE` switches off the reconstruction losses of the nine channels our dataset
    does not carry.  It deliberately does NOT contain 7 (sog), 9 (cog), 11/12 (lat/lon) or 13 (time),
    so every loss MH-GIN can legitimately compute on our data stays on.

    `max_delta` MUST be passed at construction: `TimeEncoder` captures it at __init__ (M1Encoder.py:73)
    and so does `HeterogeneousAttributeDecoder` (MTSHGNN.py:76).  Assigning `model.max_delta` afterwards
    only rebinds the LOSS clamp (MTSHGNN.py:216-217, 408), leaving the Delta-t FEATURE still clamped at
    the vendor default -- a silent half-change.
    """
    return MtsHGnn(
        logger=_logger(), device=device, dataset_name="task_alpha",
        d_data_model=d_model, data_status=data_status, d_mining_model=d_model,   # must be equal
        hidden_size=d_model, output_size=d_model, horizon=1,                     # ESN-only (dead path)
        graph_mask_values=list(FAKE), max_delta=float(max_delta),
    ).to(device)


@torch.no_grad()
def predict_latlon(model: MtsHGnn, data: np.ndarray, meta, device, batch_size: int = 32):
    """Run the imputation head and return per-graph (lat, lon) at the HELD nodes, in node order.

    `impute_missing_values` returns `(p_pred, delta_coord), ...` where p_pred carries
    (lon_rad, lat_rad) -- longitude FIRST -- flattened over (batch, time).  There is no
    inverse-transform anywhere in the vendor repo, so we do the rad->deg conversion here.
    """
    model.eval()
    out = []
    for lo in range(0, len(data), batch_size):
        xb = torch.as_tensor(data[lo:lo + batch_size], dtype=torch.float32, device=device).contiguous()
        (p_pred, _), *_ = model.impute_missing_values(xb)
        p = p_pred.reshape(xb.shape[0], xb.shape[1], 2).cpu().numpy()
        lon = np.rad2deg(p[..., 0]); lat = np.rad2deg(p[..., 1])
        for j in range(xb.shape[0]):
            m = meta[lo + j]
            h = m["held"]
            out.append((lat[j, :m["L"]][h], lon[j, :m["L"]][h]))
    return out

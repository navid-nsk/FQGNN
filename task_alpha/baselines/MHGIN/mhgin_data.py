"""RunGraph -> MH-GIN (MTS-HGNN) tensor adapter.  The model is imported UNCHANGED.

MH-GIN consumes a fixed 14-channel AIS schema:
    0 vessel_type   1 vessel_width  2 vessel_length  3 vessel_draught  4 destination
    5 cargo_type    6 navi_status   7 sog            8 rot             9 cog
   10 heading_angle 11 latitude    12 longitude     13 timestamp
Our benchmark carries only sog(7), cog(9), lat(11), lon(12), t(13).  The other nine channels are
architecturally required (`n=14` and the group split [3,3,1,4,3] are hardcoded), so they are filled
with the model's own MISSING sentinels at every timestep and added to `graph_mask_values` so their
reconstruction losses are switched off.  A constant, always-missing channel leaks nothing.

MISSING is signalled by exact sentinel VALUE, not by the mask tensors (M1Encoder tests
`lon==181.0`, `lat==91.0`, `x==-1`).  Our held nodes arrive ZEROED, which MH-GIN would read as a
perfectly valid observation at 0N/0E, 0 kn, 0deg course.  Translating zeros -> sentinels is therefore
a correctness requirement, not a favour.

Mask semantics (verified in code):
  masks         True = observed, False = synthetically hidden.  Used only by the eval metrics.
  padding_masks True = a real timestep inside the trajectory.  This is what GATES EVERY LOSS TERM.

Timestamps at held steps are KNOWN (they are the query times) and are supplied unhidden, exactly as
every other baseline receives them.  sog/cog have no ground truth at held steps, so their
padding_masks are cleared there and their loss simply skips those cells.  lat/lon/t labels are known
at every timestep, so MH-GIN's native whole-trajectory reconstruction loss runs as designed.
"""
from __future__ import annotations

import numpy as np

from task_alpha.baselines.common_gaps import to_series

N_ATTR = 14
VESSEL_TYPE, V_WIDTH, V_LENGTH, V_DRAUGHT = 0, 1, 2, 3
DESTINATION, CARGO_TYPE, NAVI_STATUS = 4, 5, 6
SOG, ROT, COG, HEADING, LAT, LON, TIME = 7, 8, 9, 10, 11, 12, 13

# Channels we do not possess -> always missing, losses disabled.  (Discrete channels 0/4/5/6 use the
# embedding's padding index rather than -1; with one dummy class that index is 1.)
FAKE = [VESSEL_TYPE, V_WIDTH, V_LENGTH, V_DRAUGHT, DESTINATION, CARGO_TYPE, NAVI_STATUS, ROT, HEADING]
DISCRETE = [VESSEL_TYPE, DESTINATION, CARGO_TYPE, NAVI_STATUS]
PAD_CLASS = 1.0            # == num_classes when data_status["*_unique"] has a single dummy entry

LAT_MISSING, LON_MISSING, SCALAR_MISSING = 91.0, 181.0, -1.0
# Held steps carry no sog/cog truth; a value the encoder will never see, and whose loss is gated off.
NO_LABEL = -1.0


def cog_degrees(sin_c, cos_c):
    return np.degrees(np.arctan2(sin_c, cos_c)) % 360.0


def build_arrays(graphs, max_len: int, crop_ctx: int = 64):
    """-> data, labels, masks, padding_masks  [N, max_len, 14]  (+ per-graph held bookkeeping)."""
    n = len(graphs)
    data = np.zeros((n, max_len, N_ATTR), dtype=np.float32)
    labels = np.zeros((n, max_len, N_ATTR), dtype=np.float32)
    masks = np.zeros((n, max_len, N_ATTR), dtype=bool)
    pad = np.zeros((n, max_len, N_ATTR), dtype=bool)
    meta = []

    for i, g in enumerate(graphs):
        s = to_series(g, crop_ctx=crop_ctx)
        L = min(len(s["held"]), max_len)
        h = s["held"][:L]
        obs = ~h

        # ---- inputs: sentinels wherever the model must not see a value ----------------------
        d = data[i]
        d[:, :] = SCALAR_MISSING                    # default every cell to "missing"
        d[:, DISCRETE] = PAD_CLASS
        # lat/lon have their OWN sentinels; padding cells must carry them too, or the decoder's
        # sliding-window base averages the padding fill into the position prior.
        d[:, LAT] = LAT_MISSING
        d[:, LON] = LON_MISSING
        d[:L, LAT] = np.where(h, LAT_MISSING, s["lat_true"][:L])
        d[:L, LON] = np.where(h, LON_MISSING, s["lon_true"][:L])
        d[:L, SOG] = np.where(h, SCALAR_MISSING, s["sog_in"][:L])
        d[:L, COG] = np.where(h, SCALAR_MISSING, cog_degrees(s["cog_sin"][:L], s["cog_cos"][:L]))
        d[:L, TIME] = s["t_sec"][:L]                # query times are known at held steps

        # ---- labels: truth where we have it -------------------------------------------------
        lb = labels[i]
        lb[:, DISCRETE] = PAD_CLASS
        lb[:L, LAT] = s["lat_true"][:L]
        lb[:L, LON] = s["lon_true"][:L]
        lb[:L, SOG] = np.where(h, NO_LABEL, s["sog_in"][:L])
        lb[:L, COG] = np.where(h, NO_LABEL, cog_degrees(s["cog_sin"][:L], s["cog_cos"][:L]))
        lb[:L, TIME] = s["t_sec"][:L]

        # ---- masks: True = observed (eval only) ---------------------------------------------
        masks[i, :L, :] = True
        masks[i, :L, LAT] = obs
        masks[i, :L, LON] = obs
        masks[i, :L, SOG] = obs
        masks[i, :L, COG] = obs

        # ---- padding_masks: True = real timestep; gates EVERY loss term ----------------------
        pad[i, :L, :] = True
        pad[i, :L, SOG] = obs        # no sog truth inside the gap -> skip its loss there
        pad[i, :L, COG] = obs        # ditto cog
        pad[i, L:, :] = False

        meta.append({"L": L, "held": h, "node_idx": s["node_idx"][:L]})

    return data, labels, masks, pad, meta


def max_seq_len(*graph_lists, crop_ctx: int = 64) -> int:
    m = 0
    for gs in graph_lists:
        for g in gs:
            m = max(m, len(to_series(g, crop_ctx=crop_ctx)["held"]))
    return m


def fit_status(train_graphs, crop_ctx: int = 64) -> dict:
    """MH-GIN's `data_status`: z-score stats for the ContinuousEncoder + class lists for the
    DiscreteEncoder.  Only sog is real; the unused continuous channels get inert (0,1) stats and the
    discrete channels a single dummy class (so `num_classes=1` and the padding index is 1)."""
    sg = np.concatenate([to_series(g, crop_ctx)["sog_in"][~to_series(g, crop_ctx)["held"]]
                         for g in train_graphs])
    st = {"sog_mean": float(sg.mean()), "sog_std": float(max(sg.std(), 1e-6)),
          "sog_min": float(sg.min()), "sog_max": float(sg.max())}
    for k in ("rot", "vessel_width", "vessel_length", "vessel_draught"):
        st |= {f"{k}_mean": 0.0, f"{k}_std": 1.0, f"{k}_min": 0.0, f"{k}_max": 1.0}
    for k in ("navi_status", "cargo_type", "destination", "vessel_type"):
        st[f"{k}_unique"] = ["__missing__"]          # len 1 -> num_classes=1, padding_idx=1
    return st

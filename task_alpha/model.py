"""
task_alpha.model  --  THE Task Alpha model.

The model is the faithful full-FQ-primitive imputer (task_alpha.fq_imputer.FQTrajectoryImputer): ONE
shared Lee-Yang quasicrystal (genuine d=2, Q-independent L + secular variety U + the verified
variety-fixed cumulant amplitude) drives EVERY primitive -- Op-A'/QFE, Op-B eigenphases, Op-C secular
value, Op-D twisted Eberlein, the D-mu velocity channel -- plus the amoeba/Parseval certificate ledger.
This is the showcase of the invented primitive family on Task Alpha (the precondition for
generalization), NOT an applied generic GNN.
"""

from __future__ import annotations

from .fq_imputer import FQTrajectoryImputer

N_FEATURES = 6   # [anchor_pos_x, anchor_pos_y (0 for held-out), sog, sin cog, cos cog, is_held]
N_TASKS = 2      # ABSOLUTE ENU position (x, y) -- NOT a residual over any baseline


def make_model(n: int = 6, n_freqs: int = 24, k_max: int = 3, width: int = 64,
               init_scale: float = 2.0, seed: int = 0) -> FQTrajectoryImputer:
    # in_feat=6: the model receives anchor POSITIONS as features (zeroed for held-out) and the TIME
    # coordinate as the FQ-lift input; it predicts the ABSOLUTE held-out position (no GCD leakage).
    return FQTrajectoryImputer(n=n, n_freqs=n_freqs, k_max=k_max, width=width,
                               init_scale=init_scale, seed=seed, in_feat=N_FEATURES)

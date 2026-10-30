"""
fqmath.amoeba  (visualization helper)
=====================================

Not used for certification.  The Lee-Yang / FQ in-family CERTIFICATE is the DETERMINISTIC
amoeba-disjointness + uniform-genericity in `fqmath.genericity`.  This module holds only a
small VISUALIZATION helper (sampling Log|V(p_U)| near the torus) and is not the faithfulness
witness -- use `fqmath.genericity.lee_yang_certificate`.

Conventions:
  * normalization: p_U(z) = det(I - diag(z) U)  (p_U(0)=1), not det(diag(z)U - I);
  * naming: the float-log|z| sign-variation count is called `var_log_sign_changes`
    (the canonical integer-k `var_sign_changes` lives in fqmath.spectrum and is re-exported by
    fqmath.__init__; do NOT shadow it here).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def var_log_sign_changes(v: np.ndarray) -> int:
    """Sign-variation count of a real (log-magnitude) vector -- distinct from the integer-k
    `fqmath.spectrum.var_sign_changes` (deliberately a different name)."""
    s = [int(np.sign(x)) for x in v if abs(x) > 1e-12]
    return sum(1 for i in range(1, len(s)) if s[i] != s[i - 1])


def secular_poly(z: np.ndarray, U: np.ndarray) -> complex:
    """p_U(z) = det(I - diag(z) U)  (normalized so that p_U(0) = 1)."""
    n = U.shape[0]
    return np.linalg.det(np.eye(n) - (z[:, None] * U))


def amoeba_shell(U: np.ndarray, n_dirs: int = 200, r: float = 0.15, seed: int = 0
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    VISUALIZATION ONLY: sample Log|z| of near-zeros of p_U on a small off-torus shell.  This is NOT
    a certificate (it is stochastic and gate-dependent); for the in-family check use
    fqmath.genericity.lee_yang_certificate (deterministic amoeba margin + genericity guards).
    """
    n = U.shape[0]
    rng = np.random.default_rng(seed)
    logs, vars_ = [], []
    for _ in range(n_dirs):
        theta = rng.uniform(0, 2 * np.pi, n)
        rho = r * rng.standard_normal(n)
        z = np.exp(rho + 1j * theta)
        if abs(secular_poly(z, U)) < 0.5:
            logs.append(rho)
            vars_.append(var_log_sign_changes(rho))
    if not logs:
        return np.zeros((0, n)), np.zeros(0)
    return np.array(logs), np.array(vars_)

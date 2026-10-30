"""FQ mathematics core — Fourier-quasicrystal constructions faithful to AKKV, KS, ACV and LT, plus the
Eberlein/diffraction and Postnikov/Lam/Karp TP layers."""
from .grassmann import PositiveGrassmann
from .lee_yang import SecularUnitary, MoebiusVariety
from .spectrum import FQSpectrum, var_sign_changes, enumerate_spectrum_combos
from .akkv_amplitudes import (secular_log_cumulants, secular_amplitudes_d1,
                              secular_pullback_amplitudes, support_mask)
from .diffraction import (structure_factor, pure_point_fraction, low_freq_l1_tail,
                          bragg_participation, hyperuniformity_rate_proxy)
from .genericity import (lee_yang_certificate, amoeba_positive_ray_margin,
                         coupling_margin, eigenphase_min_gap)
from .membership import parseval_residual, bap_membership_ok

__all__ = [
    "PositiveGrassmann",
    "SecularUnitary",
    "MoebiusVariety",
    "FQSpectrum",
    "var_sign_changes",
    "enumerate_spectrum_combos",
    # variety-fixed amplitudes
    "secular_log_cumulants",
    "secular_amplitudes_d1",
    "secular_pullback_amplitudes",
    "support_mask",
    # diffraction diagnostics (Op-E, diagnostic-only)
    "structure_factor",
    "pure_point_fraction",
    "low_freq_l1_tail",
    "bragg_participation",
    "hyperuniformity_rate_proxy",
    # in-family certificate (amoeba margin + genericity guards) + membership gate
    "lee_yang_certificate",
    "amoeba_positive_ray_margin",
    "coupling_margin",
    "eigenphase_min_gap",
    "parseval_residual",
    "bap_membership_ok",
]

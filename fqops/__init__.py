"""FQ operator library — every GNN transform is one of these (no plain Linear in the core)."""
from .op_secular import FQSecularBlock, FQSecularPair
from .op_variety import FQVarietyBlock
from .op_curve import FQCurveBlock
from .op_spectral_block import FQSpectralBlock
from .op_coupled_spectral import FQCoupledSpectralBlock
from .op_spectral import FQSpectralEncoder, FQSpectralAggregate
from .op_eigenphase import FQEigenphaseHead
from .op_diffraction import FQDiffraction


def make_fq_block(primitive, in_dim, out_dim, in_per_head=4, n=None, init_scale=2.0,
                  coupled=True, seed=0, chunk=30000, phase_offset=False):
    """
    Factory for the core FQ transform.  All primitives map in_dim -> out_dim with the identical
    signature so the GNN steps can swap them.
      Faithfulness labels:
      primitive="variety" : Op-C Lee-Yang VARIETY-COUPLING relations (bilinear in z) -- the
                            AKKV-FAITHFUL coupled invariant; route the coupled message here. DEFAULT.
      primitive="curve"   : Op-C Example-1.3 Lee-Yang curve residual (n=3,d=2) -- faithful, low ceiling.
      primitive="secular" : Op-B coupled secular operator; uses the EIGENPHASE featurization
                            (faithful default), with tr(M^k) trace moments only as the project-original ablation.
      primitive="spectral": GEOMETRY-ONLY ABLATION (free-amplitude aperiodic Fourier features on the
                            Gr_+ spectrum Lambda'; NOT the faithful FQ -- isolates geometry vs coupling).
    phase_offset: add a learnable torus translation phi_0 to the lift.
    """
    if primitive == "spectral":
        sn = n if n is not None else 8               # torus coords per head (d=2)
        return FQSpectralBlock(in_dim, out_dim, n=sn, n_freqs=12, init_scale=init_scale, seed=seed,
                               chunk=(chunk if chunk else None), phase_offset=phase_offset)
    if primitive == "coupled_spectral":
        # spectral (expressive) + a learned unitary U=exp(iH) coupling of the spectral coefficients.
        # coupled=False -> U=I -> decoupled spectral control (the ablation).
        # n_freqs=12: a rich aperiodic spectrum per head (RFF-style universality needs enough freqs).
        sn = n if n is not None else 8
        return FQCoupledSpectralBlock(in_dim, out_dim, n=sn, n_freqs=12, init_scale=init_scale, coupled=coupled,
                                      seed=seed, chunk=(chunk if chunk else None), phase_offset=phase_offset)
    if primitive == "curve":
        cn = n if n is not None else 3                # n=3, d=2 anchor (Example 1.3)
        return FQCurveBlock(in_dim, out_dim, n=cn, init_scale=init_scale, seed=seed,
                            chunk=(chunk if chunk else None), phase_offset=phase_offset)
    if primitive == "variety":
        return FQVarietyBlock(in_dim, out_dim, in_per_head=in_per_head, n=n, init_scale=init_scale,
                              coupled=coupled, seed=seed, chunk=chunk, phase_offset=phase_offset)
    if primitive == "secular":
        return FQSecularBlock(in_dim, out_dim, in_per_head=in_per_head, n=n, init_scale=init_scale,
                              coupled=coupled, seed=seed, chunk=chunk)
    raise ValueError(f"unknown primitive {primitive}")


from .op_qfe import FQQuasicrystalEncoder        # Op-A' / QFE (full)
from .op_zero_field import FQZeroField           # Op-F (FQ zero-field synthesis)
from .op_velocity import FQVelocityChannel       # velocity channel = D mu on the shared spectrum

__all__ = [
    "FQSecularBlock", "FQSecularPair", "FQVarietyBlock", "FQCurveBlock", "FQSpectralBlock",
    "FQSpectralEncoder", "FQSpectralAggregate", "FQEigenphaseHead", "FQDiffraction",
    "FQQuasicrystalEncoder", "FQZeroField", "FQVelocityChannel",
    "make_fq_block",
]
from .op_secular_pullback import FQSecularPullbackBlock

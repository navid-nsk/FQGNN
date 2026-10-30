# Scope of the Lean formalization

This document is for readers of the FQ-GNN paper who want to know exactly which statements are
machine-checked, in which form, and under which hypotheses. "Manuscript" refers to the main text,
"SI" to the Supplementary Information. All public statements are in `MainTheorems.lean` (namespace
`FQGNN.Main`); each carries a docstring with its statement in words and its location in the paper.
`README.md` maps every public theorem to the module theorem that proves it.

## (a) What is formalized, by paper location

### Manuscript

| Paper statement | Public theorems |
|---|---|
| Proposition 1 (affine reproduction), Op-H readout of eq. (6) | `prop1_affine_reproduction_opH`, `prop1_affine_reproduction_opH_mp`, `propS45_zero_data_zero_gate` |
| Proposition 1, full readout of eqs. (7)–(8) (exterior self-gating) | `prop1_affine_reproduction_opM` (= SI Thm S7.5(i)) |
| Proposition 2 (always-in-family training, deployed `d = 2` chart) | `prop2_chart_mem_GrPlus2`, `prop2_GrPlus2_every_update`, `prop2_GrPlus2_gradient_descent`, `prop2_coupling_unitary`, `lemmaS27_hermPart_isHermitian`, `corS210_coupling_unitary_every_step`, `thmS28c_real_rootedness` |
| Proposition 2, aperiodicity sentence | `aperiodicity_criterion`, `aperiodicity_criterion_columnSpace`, `counterexample_rowIndependent_but_periodic`, `counterexampleTP_rowIndependent_but_periodic`, `chart_attains_rowIndependent_but_periodic` |
| Proposition 3 (exact degeneration) | `prop3_zero_precision`, `prop3_exact_degeneration`, `prop3_exact_degeneration_geom`, `prop3_zero_precision_mp` |
| Eq. (1), the torus lift, and Op-A | `eq1_torusLift_mem_torus`, `opA_features_eq_torusLift` |
| Eq. (2), the secular polynomial, and Op-C | `eq2_pU_zero`, `opC_residual_re_im`, `opC_mem_fqPointSet_iff_eigenvalue_one` |
| Eqs. (3)–(4), the Dμ kernel | `eq4_Dmu_is_derivative`, `eq4_Dmu_is_directional_derivative` |
| Eq. (10), the chart minor | `chart_minor_closed_form` |
| Results, the chord identity | `chord_identity_S46` |

### Supplementary Information

| SI location | Public theorems |
|---|---|
| §S1.1, period and aperiodicity criteria | `period_criterion`, `fqPointSet_period_invariance`, `aperiodicity_criterion`, `aperiodicity_criterion_columnSpace`, `aperiodicity_d1_iff`, `counterexample_rowIndependent_but_periodic`, `counterexampleTP_rowIndependent_but_periodic`, `chart_attains_rowIndependent_but_periodic` |
| §S1.1, ℚ-independence and Kronecker density | `rowIndependent_iff_no_integer_relation`, `kronecker_density_iff`, `kronecker_density` |
| §S1.4, eq. (S1.5), principal-minor expansion | `secular_principal_minor_expansion` |
| §S1.5, the spectrum module and the thinned spectrum `Λ'` | `spectrumModule_span_eq_rows`, `qRank_le`, `qRank_eq_iff`, `d1_spectrumModule_dense`, `d1_no_exclusion_zone`, `thinnedSpectrum_subset_and_symm`, `thinnedSpectrum_exclusion_zone_deployed`, `thinnedSpectrum_locally_finite`, `thinned_vs_ambient_sqrt_two`, `thinnedSpectrum_span_eq_spectrumModule`, `qRank_thinned_le`, `qRank_thinned_eq_iff`, `spectrumModule_dense_iff_aperiodic`, `spectrumModule_not_dense_counterexample` |
| §S2, Lemma S2.1, Lemma S2.2 | `prop2_chart_mem_GrPlus2`, `chart_minor_closed_form`, `lemmaS22_chart_image_eq_GrPlus2` |
| §S2, Prop. S2.5(i)–(iii) | `prop2_GrPlus2_every_update`, `prop2_GrPlus2_gradient_descent`, `propS25iii_interiority`, `propS25iii_interiorityConst_pos`, `propS25iii_interiority_oneSided`, `propS25iii_norm_tendsto_atTop`, `propS25iii_coordinate_case_split` |
| §S2, Lemma S2.7, Cor. S2.10 | `lemmaS27_hermPart_isHermitian`, `prop2_coupling_unitary`, `corS210_coupling_unitary_every_step` |
| §S2, Thm S2.8(a)–(d) | `thmS28a_stability`, `thmS28b_functional_equation`, `thmS28c_exterior_exclusion`, `thmS28c_zero_confinement`, `thmS28c_real_rootedness`, `thmS28d_torus_zeros_eigenvalue_one` |
| §S2.6, scope of Proposition 2 | `counterexample_rowIndependent_but_periodic`, `counterexampleTP_rowIndependent_but_periodic`, `chart_attains_rowIndependent_but_periodic` |
| §S4.1, the reconstruction class | `S41_affine_comb_with_two_zeros_is_zero`, `S41_direct_sum`, `S41_reconBasis_linearIndependent` |
| §S4.2, row equilibration | `row_equilibration_invariance` |
| §S4.3, the weighted minimum-norm solve | `opHW_weighted_min_norm`, `mpinv_min_norm` |
| §S4.4, Prop. S4.1 and the pseudoinverse | `propS41_exact_interpolation`, `propS41_hermite_interpolation`, `propS41_exact_interpolation_mp`, `propS41_exact_interpolation_mp_iff`, `propS41_hermite_interpolation_mp`, `S44_null_space_annihilation`, `moorePenrose_unique`, `moorePenrose_exists`, `moorePenrose_iff_eq_mpinv`, `mpinv_eq_pinvR_of_fullRowRank`, `mpinv_feasible_iff_mem_range`, `opHW_mp_eq_opHW_of_fullRowRank` |
| §S4.4, genericity of the rank hypothesis at `R = 2` | `S44_value_block_rank_deficient_iff`, `S44_value_block_fullRowRank` |
| §S4.5, Proposition 1 restated | `prop1_affine_reproduction_opH`, `prop1_affine_reproduction_opH_mp`, `propS45_zero_data_zero_gate` |
| §S4.6, the chord identity | `chord_identity_S46` |
| §S5, Prop. S5.1, finite spectrum | `propS51_kernel_posSemidef`, `propS51_kernel_complex_nonneg`, `propS51_sum_of_squares`, `propS51_profile` |
| §S5, Prop. S5.1, infinite spectrum under (H1) and (H3) | `propS51_infinite_posSemidef`, `propS51_infinite_complex_nonneg`, `propS51_infinite_profile`, `propS51_truncation_tendstoUniformly` |
| §S5.4, decay criterion | `propS54_decay_posSemidef` |
| §S5.5(a), the idealized infinite limit | `propS55a_posDef_of_aperiodic`, `propS55a_not_posDef_of_periodic` |
| §S6.1, Prop. S6.1(i)–(iii) | `propS41_exact_interpolation`, `propS45_zero_data_zero_gate`, `propS61i_zero_data_mp`, `propS61ii_metric_homogeneity`, `propS61iii_normalization_invariance` |
| §S6.2, Prop. S6.2 (kriging unification) | `propS62_reproducing`, `propS62_reproducing_deriv`, `propS62_kriging_eq_opHW`, `opHW_weighted_min_norm` |
| §S7.1–S7.2, Prop. S7.1, Thm S7.2 | `propS71_opM_unique_minimizer`, `thmS72_soft_block_invertible`, `opM_mp_eq_opM_of_fullRowRank` |
| §S7.2, Thm S7.3 | `thmS73_exactness_every_dial`, `thmS73_hermite_interpolation`, `thmS73_exactness_every_dial_mp`, `thmS73_exactness_mp_iff`, `thmS73_hermite_interpolation_mp` |
| §S7.2, Cor. S7.4 | `corS74_endpoint_annihilation`, `corS74_exterior_in_null_space_mp`, `corS74_endpoint_annihilation_mp`, `opM_bcKernel_value_row`, `opM_bcKernel_deriv_row` |
| §S7.2, Thm S7.5 | (i) `prop1_affine_reproduction_opM`; (ii) `thmS75ii_if_gated`, `thmS75ii_converse_iff`, `thmS75ii_converse_of_injective`, `thmS75ii_only_if_counterexample` |
| §S7.2, Prop. S7.6 and the remark after it | `propS76_chordExact_imp_exteriorBlind`, `propS76_responds_imp_not_chordExact`, `propS76_misses_sCurve`, `propS76_exterior_oracle_trade` |
| §S7.3, proof of Proposition 3; §S7.4, metric law (S7.3) | `prop3_zero_precision`, `prop3_exact_degeneration`, `prop3_exact_degeneration_geom` |
| Closed form (S7.2): further exact identities of Op-M | `opM_soft_deformation`, `opM_influence_law`, `opM_gauge_invariance`, `opM_single_anchor_response` |

The theorem names with the prefixes `M3exp`, `M6`, `M7`, `M10b` and `T27` are labels only; the
docstrings and the tables above give their content and their place in the paper.

## (b) Modelling conventions

- **Full row rank.** Full row rank of a constraint matrix `B : Matrix ι κ ℝ` is formalized as
  invertibility of the Gram matrix, `FullRowRank B := IsUnit (B * Bᵀ).det`, and is proved
  equivalent to `rank B = |ι|` (`rank_eq_card_iff_fullRowRank`, `Common/Pseudoinverse.lean`).
- **Two models of the pseudoinverse.**
  - The right inverse `pinvR B = Bᵀ(BBᵀ)⁻¹`, with the projector `proj B = 1 − pinvR B * B`. Under
    full row rank it is the Moore–Penrose inverse. The readouts `opHW` (manuscript eq. (6)) and
    `opM` (eq. (8)) are defined with it.
  - The genuine Moore–Penrose inverse `mpinv B` of every real matrix (`Common/MoorePenrose.lean`):
    the four Penrose equations, uniqueness (`moorePenrose_unique`), existence
    (`moorePenrose_exists`), and agreement with `pinvR` under full row rank
    (`mpinv_eq_pinvR_of_fullRowRank`). The readouts `opHW_mp` and `opM_mp` are eqs. (6) and (8)
    with `mpinv`; they equal `opHW` and `opM` under full row rank and a positive metric
    (`opHW_mp_eq_opHW_of_fullRowRank`, `opM_mp_eq_opM_of_fullRowRank`).
- **Which statements are rank-free.**
  - No rank hypothesis at all: Prop. 1 for both readouts, for every metric `w` (positivity of `w`
    is not needed); Prop. 3, including the deployed geometric metric law; Thm S7.2; the Op-M
    influence law, gauge invariance and single-anchor response; the "if" direction of
    Thm S7.5(ii); Prop. S6.1(ii)–(iii); row equilibration by any invertible `D`; and, with
    `mpinv`, the null-space annihilation of §S4.4 (`S44_null_space_annihilation`, for every
    square `W`), Prop. S6.1(i) at `d = 0` (`propS61i_zero_data_mp`, for every `w`), and Cor. S7.4
    (`corS74_exterior_in_null_space_mp`, `corS74_endpoint_annihilation_mp`, with neither full row
    rank nor `ŵ > 0`).
  - Under the consistency condition `d ∈ range B` only, with `mpinv`: Prop. S4.1
    (`propS41_exact_interpolation_mp`, `propS41_hermite_interpolation_mp`) and Thm S7.3
    (`thmS73_exactness_every_dial_mp`, `thmS73_hermite_interpolation_mp`). The condition is also
    necessary (`propS41_exact_interpolation_mp_iff`, `thmS73_exactness_mp_iff`).
  - Under full row rank, with `pinvR`: Prop. S4.1 and Thm S7.3 in their `pinvR` forms, the
    weighted minimum-norm characterization (`opHW_weighted_min_norm`), Prop. S6.2, Prop. S7.1 /
    Thm S7.2 as a unique-minimizer statement, Cor. S7.4 in its `pinvR` form, the kernel-row lemmas
    `opM_bcKernel_value_row` and `opM_bcKernel_deriv_row`, and the converse of Thm S7.5(ii). Under
    full row rank and a positive metric these transfer to `mpinv` through the agreement theorems.
- **Exact arithmetic.** All statements are over `ℝ` (and `ℂ`). The `rcond` cutoff of the
  implemented singular-value solve and floating point are not modelled; "bit-for-bit" is checked
  as exact equality of real vectors.
- **Spectra.** Kernel statements (SI §S5, §S6) are for a finite spectrum, the truncation `Λ'_K`
  the model evaluates (`Readout/Kernel.lean`, any finite index type). For Prop. S5.1 they also hold
  for an arbitrary, possibly infinite, spectrum index under (H1) nonnegative masses and (H3)
  summable total mass (`Readout/KernelInfinite.lean`); (H2) and a positive scale `s` are not needed
  for the cosine form. "Positive-definite" in Prop. S5.1 is formalized as positive semidefinite,
  which is the sense of Bochner used there.
- **`Gr₊(2,n)` on matrix representatives.** `InGrPlus2 L` means that every `2 × 2` minor `Δ_ij`
  of the `n × 2` matrix `L` with `i < j` is strictly positive. Changes of representative
  `L ↦ L g` are explicit.
- **Comb coefficients.** The coefficients of the residual comb are indexed by `Fin K ⊕ Fin K`: the
  cosine part is `Sum.inl k` and the sine part `Sum.inr k`. Each spatial coordinate is solved
  separately; the curvature gate is computed from vector velocities in an arbitrary normed space.
- **Weaker hypotheses than the text.** Several statements use weaker hypotheses than the
  informal text: `a ≠ b` instead of `a < b` for the gap endpoints, `s_e ≠ 0` instead of
  `s_e > 0` for the equilibration factors, any `α ≠ 0` in Prop. S6.1(ii), and no restriction on
  the scale `s` in Prop. S5.1.

## (c) Precise hypotheses where the formal statement is sharper than the informal text

- **Lemma S2.2 needs `n ≥ 2`.** `lemmaS22_chart_image_eq_GrPlus2` assumes `2 ≤ n`. For `n ≤ 1` the
  minor condition is vacuous on representatives, so a zero row would satisfy it; `Gr(2,n)` is only
  meaningful for `n ≥ 2`, and the deployed model has `n = 6`. The proof uses an explicit
  compressing frame rather than a continuous angle lift.

- **The exact aperiodicity criterion is a condition on the column space.** The torus lift `Φ_L`
  has a nonzero period if and only if `Lp ∈ ℤⁿ` for some `p ≠ 0` (`aperiodicity_criterion`). For
  injective `L`, which includes every `L ∈ Gr₊(d,n)`, this holds if and only if the column space
  `Lℝᵈ` contains a nonzero integer vector (`aperiodicity_criterion_columnSpace`). Every period of
  `Φ_L` is a period of every point set `Λ(X,L)` (`fqPointSet_period_invariance`), so
  `Lℝᵈ ∩ ℤⁿ = {0}` is necessary for an aperiodic point set.
  - Rational independence of the rows does not imply aperiodicity for `d ≥ 2`. Three generators
    with `ℚ`-linearly independent rows and a nonzero period are machine-checked:
    1. `L = [[1,0],[√2,0],[0,1]]`, with `L(0,1)ᵀ = (0,0,1)ᵀ ∈ ℤ³`
       (`counterexample_rowIndependent_but_periodic`);
    2. the totally positive `L = [[1,1],[√2,2],[√3,3]] ∈ Gr₊(2,3)`, with minors `2 − √2`,
       `3 − √3`, `3√2 − 2√3` and `L(0,1)ᵀ = (1,2,3)ᵀ ∈ ℤ³`
       (`counterexampleTP_rowIndependent_but_periodic`);
    3. a generator `L(φ)` in the image of the deployed chart of manuscript eq. (10), obtained from
       the second one through Lemma S2.2 and the invariance of both properties under `L ↦ L g`,
       `det g ≠ 0` (`chart_attains_rowIndependent_but_periodic`).
  - For `d = 1` the row condition is a sufficient certificate: `Φ_L` is periodic if and only if
    every ratio `L_j / L_i` with `L_i ≠ 0` is rational (`aperiodicity_d1_iff`).
  - Rational independence of the rows is equivalent to `ℚ`-rank `n` of the spectrum module
    (`qRank_eq_iff`, and `qRank_thinned_eq_iff` for the thinned spectrum `Λ'`) and to the
    absence of a nonzero integer relation `Lᵀk = 0` (`rowIndependent_iff_no_integer_relation`).

- **Density of the ambient module is exactly aperiodicity.** The ambient module `M_L = Lᵀℤⁿ` is
  dense in `ℝᵈ` if and only if no nonzero `p ∈ ℝᵈ` has `Lp ∈ ℤⁿ`
  (`spectrumModule_dense_iff_aperiodic`). Neither `n > d`, nor positivity of the minors, nor
  `ℚ`-independence of the rows suffices: the generator `L = [[3,1],[2,1+√2],[1,3]]` lies in
  `Gr₊(2,3)` (minors `1 + 3√2`, `8`, `5 − √2`), has `ℚ`-linearly independent rows, and has
  `L(1,0)ᵀ = (3,2,1)ᵀ ∈ ℤ³`, so the first coordinate of every element of `M_L` is an integer and
  `M_L` is not dense (`spectrumModule_not_dense_counterexample`). For `d = 1` the criterion reduces
  to an irrational ratio of two entries (`d1_spectrumModule_dense`, `d1_no_exclusion_zone`).
  Consequently the idealized strict positive-definiteness of SI §S5.5(a) holds when `Φ_L` has no
  nonzero period, every atom of `ℤⁿ` carries positive summable mass and `s ≠ 0`
  (`propS55a_posDef_of_aperiodic`); if `Φ_L` has a nonzero period `p`, the Gram matrix at the
  distinct anchors `0` and `p` is singular for every choice of masses
  (`propS55a_not_posDef_of_periodic`).

- **The thinned spectrum is discrete; the ambient module is the dense object.** For
  `L ∈ Gr₊(2,n)`, `n ≥ 2`, and in particular for every chart generator, the thinned spectrum
  `Λ' = {Lᵀk : var(k) < d}` has an exclusion zone around the origin and is locally finite
  (`thinnedSpectrum_exclusion_zone_deployed`, `thinnedSpectrum_locally_finite`); for
  `L = (1, √2)` the thinned spectrum has an exclusion zone while `M_L = ℤ + √2ℤ` does not
  (`thinned_vs_ambient_sqrt_two`). The thinning does not change the rational span:
  `span_ℚ Λ' = span_ℚ M_L` for every `d` (`thinnedSpectrum_span_eq_spectrumModule`).

- **Kronecker density is an iff with ℚ-independence of the rows.** The image of
  `Φ_L : ℝᵈ → 𝕋ⁿ` is dense in the torus if and only if the rows of `L` are `ℚ`-linearly
  independent (`kronecker_density_iff`; the integer-relation form is `kronecker_density`). Mathlib
  has no Kronecker theorem; the proof goes through a duality criterion for density of subgroups of
  `ℝⁿ` proved in `Backbone/Kronecker.lean`. Only density is formalized, not equidistribution.

- **The converse of Thm S7.5(ii) holds exactly under `PṼᵀΛρ̃ = 0`.** The "if" direction holds
  with no hypothesis: on a locally affine gap (`d = 0`, `g = 0`) the gated precisions `0 · Λ` give
  `c = 0` for every exterior, residual, metric and `η` (`thmS75ii_if_gated`). For the converse,
  assume `B` has full row rank, `ŵ > 0` and `Λ ⪰ 0`. Then `c = 0` if and only if `PṼᵀΛρ̃ = 0`,
  where `P` projects onto the null space of the whitened hard block (`thmS75ii_converse_iff`). If
  in addition `PṼᵀ` is injective, `c = 0` if and only if `λ_j ρ̃_j = 0` for every exterior anchor
  `j` (`thmS75ii_converse_of_injective`). Without that condition the unqualified converse
  ("`c = 0` only if the gated precisions vanish") fails: with `K = 1`, the identity hard block
  (two rows spanning the coefficient space, so `P = 0`), unit metric, one exterior anchor with
  `λ = 1` and `ρ̃ = 1`, and `d = 0`, `g = 0`, Op-M returns `c = 0`
  (`thmS75ii_only_if_counterexample`). An exterior row in the whitened hard-row space contributes
  nothing for any `λ_j`, and an anchor with `ρ̃_j = 0` leaves `λ_j` unconstrained. The equivalence
  of `c = 0` with `p̂ ≡ A` uses the linear independence of the reconstruction basis for distinct
  positive kept frequencies (`S41_reconBasis_linearIndependent`).

- **Prop. S2.5(iii): the coordinate case split holds along a subsequence.** The bound (S2.3) holds
  with the exact constant `m(B) = e^{−2B} softplus(−B)² sin δ_B`,
  `δ_B = (π/2) softplus(−B) / ((n+1) softplus B)`, which is positive for `B ≥ 0`
  (`propS25iii_interiority`, `propS25iii_interiorityConst_pos`); one-sided bounds `ρ_i ≥ −B`,
  `log s ≥ −B`, `|α_l| ≤ B` already suffice (`propS25iii_interiority_oneSided`). If one fixed minor
  tends to `0` along a parameter sequence, then `‖φ_k‖_∞ → ∞` (`propS25iii_norm_tendsto_atTop`),
  and some radius `ρ_i` or the scale `log s` is unbounded below, or some angle parameter `α_l` is
  unbounded below or above (`propS25iii_coordinate_case_split`). Convergence of a single
  coordinate to `∓∞` along the whole sequence does not hold in general: the sequence with
  `ρ_0 = −k` for even `k` and `log s = −k` for odd `k` (all other coordinates fixed) has
  `Δ_01 → 0`, yet no coordinate tends to `−∞`. The formal hypothesis is a fixed vanishing minor.

- **Prop. S6.2 is formalized at the level of coefficient vectors.** The `v`-weighted pairing
  `vInner`, the reproducing identities for point evaluation and for derivative functionals
  (`propS62_reproducing`, `propS62_reproducing_deriv`, for nonzero prior variances `v_k` and with no
  hypothesis on the frequencies) and the
  kriging statement (`propS62_kriging_eq_opHW`) are proved on coefficient vectors. The kriging
  statement assumes that the constraint matrix `B` has full row rank and `v > 0`: the Op-H-W solve
  at `η = 0` with `w_k = √v_k`, for any gate, is feasible, is the unique minimizer of the RKHS norm
  on the feasible set, and equals `V Bᵀ (B V Bᵀ)⁻¹ d`.

- **Further precise forms.**
  - Prop. 1 assumes at least one included boundary velocity (`R_v ≥ 1`) and the gate enabled; it
    holds for every `η`, every metric and every gate parameter.
  - The direct sum of SI §S4.1 (`S41_direct_sum`, `S41_reconBasis_linearIndependent`) assumes
    distinct positive frequencies; `S41_affine_comb_with_two_zeros_is_zero` needs no hypothesis
    on the frequencies.
  - Manuscript eq. (4) with `ξ_k ∈ ℝᵈ` is formalized as a directional derivative along an
    arbitrary direction `e` (`eq4_Dmu_is_directional_derivative`).
  - The Op-C residual statement (`opC_residual_re_im`) is definitional once `Λ(X_U, L)` is defined
    as the pull-back `{x | Φ_L x ∈ X_U}`; the eigenvalue-one characterization
    (`opC_mem_fqPointSet_iff_eigenvalue_one`) carries the content.
  - In the geometric metric law (S7.3) the clamp acts on the `γ`-term,
    `log ŵ_k = log w_k + γ · clamp(½ log Ĩ_k − log w_k)` on the band. The floors, the smoothing
    of the periodogram and the Ledoit–Wolf trust law enter only as universally quantified inputs.
  - Thm S2.8(a)–(c), which the SI cites from Kurasov–Sarnak, are proved from unitarity alone, for
    every unitary `U`.
  - The concrete witness of Prop. S7.6 is proved on the unit gap `[0,1]` with a scalar track and
    the S-curve `sin 2πu − ½ sin 4πu`.

## (d) What is not formalized, and why

- **Results cited from the literature.** The Delone, Meyer and pure-point (Fourier
  quasicrystal) theorems of Alon–Kummer–Kurasov–Vinzant [AKKV2024] for the point sets `Λ(X,L)`,
  including the identification of the diffraction masses; aperiodicity of `Λ(X,L)` itself, which
  also depends on `X`; the closed-ball topology of [GKL2022] and the smoothness of the chart; and
  Bochner's converse theorem. These need measure-theoretic and algebro-geometric machinery that
  is not part of the statements the paper proves.
- **Weyl equidistribution** of the flow `Φ_L(ℝᵈ)` on `𝕋ⁿ` (the equidistribution half of SI
  §S1.1). It needs Haar measure on the torus and ergodic averages; no formalized statement uses
  it. Density is formalized.
- **The `rcond` cutoff and floating point** of the implemented solve. The formalization works in
  exact real arithmetic; the conditioning of the deployed solve is an empirical certification
  (SI §S6.4).
- **The `R ∈ {3,4}` genericity sketch** of SI §S4.4. Only the explicit `R = 2` computation is
  formalized (`S44_value_block_rank_deficient_iff`, `S44_value_block_fullRowRank`); the
  real-analytic argument for `R ∈ {3,4}` is not.
- **The min-minor form of Prop. S2.5(iii).** The formal hypothesis is that one fixed minor tends
  to `0`. The SI's hypothesis `min_I Δ_I → 0` allows the index of the smallest minor to change
  along the sequence; the same proof applies because one constant `m(B)` bounds every minor, but
  this variant is not stated.
- **The necessity direction of the §S5.4 decay criterion.** Only `p > n ⇒ (H3)` is formalized
  (`propS54_decay_posSemidef`); that (H3) fails for decay exponents `p ≤ n` is not. The
  conditioning question of §S5.5(b), which the SI states as open, is not formalized either.
- **The function-level RKHS identification** in Prop. S6.2 ("the RKHS of `K_v` is `𝓕`"). It needs
  the `2K` functions `cos ω_k τ`, `sin ω_k τ` to be linearly independent, which holds for distinct
  positive `ω_k` (`S41_reconBasis_linearIndependent`), but the identification of the function
  space is not stated.
- **Measure-zero genericity of the aperiodicity condition.** For `n > d`, the condition
  `Lℝᵈ ∩ ℤⁿ = {0}` holds for all generators outside a null set; this measure-theoretic statement
  is not formalized.
- **Further items.** The evaluation chart of SI §S4.6 (the equirectangular chart, `Φ⁻¹ ∘ Φ = id`,
  the bracketing hypothesis and haversine scoring; only the algebraic chord identity and its
  affine invariances are formalized); the `O(‖Λ‖²)` bound on the remainder of the first-order
  expansion of the Op-M deformation (the exact split is proved); the `λ → ∞` limit of the
  single-anchor response, which is given as a uniform bound; monotonicity of the posterior
  variance under Schur complements; the equivalence of injectivity of `PṼᵀ` with full row rank of
  `[B; V]`; uniform discreteness of `Λ'` and the coercivity of the thinning for `d ≥ 3`; the
  extension of the Prop. S7.6 witness to general gaps and vector tracks by affine
  reparameterization; the canonical secular split family of SI §S1.7; the operator-level
  statements of SI §S3; and the differentiability and empirical certification statements of
  SI §S6.3–§S6.5.

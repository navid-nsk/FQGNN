# FQ-GNN: machine-checked mathematical statements (Lean 4 + Mathlib)

This directory accompanies the FQ-GNN paper. It formalizes in Lean 4, on top of the Mathlib
library, the three propositions of the manuscript and the principal theorems of Supplementary
Sections S1, S2 and S4–S7. The development has 19 modules, 609 theorems and lemmas, and 123
public statements collected in `MainTheorems.lean`. It compiles with no `sorry`, no `admit`, no
added axiom and no compiler warning, and every public statement depends only on the three standard
axioms of Lean's core logic (`propext`, `Classical.choice`, `Quot.sound`).

`FORMALIZATION_SCOPE.md` states precisely what is formalized, under which modelling conventions,
where a formal statement carries hypotheses that are sharper than the informal text, and what is
not formalized.

## Build

The toolchain is pinned in `lean-toolchain` (`leanprover/lean4:v4.35.0-rc3`) and Mathlib is pinned in
`lakefile.toml` and `lake-manifest.json` (tag `v4.35.0-rc3`). With [elan](https://github.com/leanprover/elan)
installed:

```
cd lean
lake exe cache get   # downloads the prebuilt Mathlib build files
lake build           # builds the FQGNN library and MainTheorems
```

`lakefile.toml` passes `-DwarningAsError=true` to this package (Mathlib keeps its own options), so
any warning in the development is reported as an error and fails the build. A successful build
ends with `Build completed successfully` and prints no warning or error lines.

## Axiom audit

`AxiomAudit.lean` runs `#print axioms` on each of the 123 public theorems. After `lake build`, run

```
cd lean
lake env lean AxiomAudit.lean
```

Each theorem depends only on `propext`, `Classical.choice` and `Quot.sound`: 121 of them use
exactly these three, and two (`propS76_chordExact_imp_exteriorBlind`,
`propS76_responds_imp_not_chordExact`) depend on no axioms at all.

## Layout

| Path | Lines | Theorems and lemmas | Content |
|---|---:|---:|---|
| `MainTheorems.lean` | 1227 | 123 | The public statements (namespace `FQGNN.Main`). Each is restated in full, with a docstring giving the statement in words and its location in the paper, and is proved by the corresponding module theorem. |
| `FQGNN.lean` | 26 | | Library root; imports every module. |
| `AxiomAudit.lean` | 129 | | `#print axioms` for every public theorem (not part of the library). |
| `FQGNN/Common/Pseudoinverse.lean` | 498 | 54 | `FullRowRank`, the right-inverse pseudoinverse `pinvR = Bᵀ(BBᵀ)⁻¹`, the projector `proj`, the Penrose identities, the feasible-set parameterization, the minimum-norm characterization, scaling and row-equilibration lemmas, the push-through identity, and the equivalence of `FullRowRank` with `rank B = \|ι\|`. |
| `FQGNN/Common/MoorePenrose.lean` | 411 | 36 | The Moore–Penrose inverse of every real matrix: the four Penrose equations, uniqueness, existence, `mpinv`, agreement with `pinvR` under full row rank, and the rank-free projector, feasibility and minimum-norm facts. |
| `FQGNN/Backbone/Chart.lean` | 572 | 36 | Softplus, the cumulative angles (S2.1), the minor formula (S2.2), Lemma S2.1, Prop. S2.5(ii)/(iii) with the exact constant of (S2.3), and Lemma S2.2. |
| `FQGNN/Backbone/ChartLimit.lean` | 222 | 4 | Prop. S2.5(iii): `‖φ‖_∞ → ∞` along a vanishing minor, the one-sided form of (S2.3), and the coordinate case split along a subsequence. |
| `FQGNN/Backbone/Unitary.lean` | 116 | 10 | `H = ½(A + Aᴴ)` is Hermitian, `exp(iH)` is unitary (Lemma S2.7), and Cor. S2.10. |
| `FQGNN/Backbone/Secular.lean` | 356 | 22 | `p_U(z) = det(I − diag(z)U)`, Thm S2.8(a)–(d) from unitarity alone, the functional equation (S2.4), the principal-minor expansion (S1.5), and the Op-C residual. |
| `FQGNN/Backbone/TorusEmbedding.lean` | 416 | 21 | The torus lift `Φ_L`, the period criterion, the Op-A features, the aperiodicity criterion, and a generator with `ℚ`-independent rows and a nonzero period. |
| `FQGNN/Backbone/AperiodicityTP.lean` | 298 | 17 | A totally positive generator in `Gr₊(2,3)` with `ℚ`-independent rows and a nonzero period, and a chart generator with the same properties. |
| `FQGNN/Backbone/SpectrumModule.lean` | 893 | 49 | The ambient module `M_L = Lᵀℤⁿ`, its `ℚ`-rank, density and the absence of an exclusion zone for `d = 1`, and the thinned spectrum `Λ'` with its exclusion zone and local finiteness on `Gr₊(2,n)`. |
| `FQGNN/Backbone/SpectrumRankThinned.lean` | 132 | 9 | `span_ℚ Λ' = span_ℚ M_L` and the `ℚ`-rank theorems for `Λ'`. |
| `FQGNN/Backbone/Kronecker.lean` | 660 | 27 | Kronecker density of `Φ_L(ℝᵈ)` in `𝕋ⁿ` (iff the rows of `L` are `ℚ`-independent) through a duality criterion for subgroups of `ℝⁿ`, density of `M_L` iff `Φ_L` has no nonzero period, and a non-dense example in `Gr₊(2,3)`. |
| `FQGNN/Readout/ExponentialSum.lean` | 637 | 42 | The comb `φ, φ'`, the chord and the reconstruction, the §S4.6 chord identity, `Dμ` as the derivative of the Op-D kernel (eqs. (3)–(4)), and the §S4.1 direct sum and linear independence. |
| `FQGNN/Readout/OpH.lean` | 559 | 49 | Op-H-W (eq. (6)), Prop. S4.1, Prop. 1 for Op-H, Prop. S6.1(ii)/(iii), row-equilibration invariance, and the weighted minimum-norm characterization. |
| `FQGNN/Readout/OpHGeneral.lean` | 412 | 30 | `opHW_mp` and `opM_mp` (eqs. (6), (8) with `mpinv`); rank-free Prop. S4.1, Prop. S6.1(i), Thm S7.3 and Cor. S7.4; Moore–Penrose forms of Props. 1 and 3. |
| `FQGNN/Readout/Kernel.lean` | 505 | 40 | Prop. S5.1 for a finite spectrum (real and complex forms), the kernel profile, and Prop. S6.2 (kriging as a weighted minimum-norm solve). |
| `FQGNN/Readout/KernelInfinite.lean` | 703 | 40 | Prop. S5.1 for an infinite spectrum under (H1) and (H3), the uniform limit of the finite truncations, the §S5.4 decay criterion for `p > n`, and §S5.5(a): strict positive-definiteness under aperiodicity and its failure without it. |
| `FQGNN/Readout/OpM.lean` | 1294 | 82 | Op-M (eqs. (7)–(8)), Prop. S7.1, Thm S7.2–S7.4, Prop. 3, Thm S7.5(i) (Prop. 1 for Op-M), Thm S7.5(ii) with its exact converse, the influence law, gauge invariance and the single-anchor response. |
| `FQGNN/Readout/BlindSpot.lean` | 303 | 25 | Prop. S7.6, the exactness/blind-spot trade, abstractly and with a concrete S-curve witness. |
| `FQGNN/Readout/Genericity.lean` | 155 | 5 | §S4.4: the exact rank condition for the value block at `R = 2`. |
| `FORMALIZATION_SCOPE.md` | | | Scope, conventions, precise hypotheses, and what is not formalized. |

The 19 modules have 9142 lines and 609 theorems and lemmas, counted as every `theorem` or `lemma` declaration, including attribute-prefixed and private ones (607 public, 2 private). Module theorems live in the namespaces `FQGNN.Common`,
`FQGNN.Backbone` and `FQGNN.Readout`.

## Conventions

The main modelling conventions are listed here; `FORMALIZATION_SCOPE.md` gives them in full.

- Full row rank is `FullRowRank B := IsUnit (B * Bᵀ).det`, proved equivalent to `rank B = |ι|`.
- Two pseudoinverses are formalized: the right inverse `pinvR B = Bᵀ(BBᵀ)⁻¹` (the Moore–Penrose
  inverse under full row rank), used by `opHW` and `opM`; and the Moore–Penrose inverse `mpinv B`
  of every real matrix, with existence and uniqueness, used by `opHW_mp` and `opM_mp` and by the
  rank-free statements. The two agree under full row rank.
- Arithmetic is exact: the `rcond` cutoff of the implemented solve and floating point are not
  modelled, and "bit-for-bit" is checked as exact equality of real vectors.
- Kernel statements are for a finite spectrum and, for Prop. S5.1, also for an infinite spectrum
  under (H1) and (H3).
- `Gr₊(2,n)` is handled on matrix representatives: `InGrPlus2 L` means that every `2 × 2` minor
  `Δ_ij` with `i < j` is strictly positive.
- Comb coefficients are indexed by `Fin K ⊕ Fin K` (cosine part `inl k`, sine part `inr k`), and
  each spatial coordinate is solved separately.

## Theorem → source map

The table lists every public theorem of `MainTheorems.lean` (namespace `FQGNN.Main`), in file
order and grouped by the file's sections, with its statement, its location in the paper ("Manuscript"
is the main text, "SI" the Supplementary Information) and the module theorem that proves it (module
in parentheses). It is generated from the docstrings of `MainTheorems.lean`.

#### Section `Backbone`

| Main theorem | Statement | Paper location | Proved by |
|---|---|---|---|
| `chart_minor_closed_form` | Chart minor closed form | Manuscript Methods eq. (10) paragraph; SI eq. (S2.2) | `minor2_chartL` (Chart) |
| `prop2_chart_mem_GrPlus2` | Proposition 2(i), membership in the family | Manuscript Prop. 2; SI Lemma S2.1 | `chartL_inGrPlus2` (Chart) |
| `prop2_GrPlus2_every_update` | Proposition 2, always-in-family under any update rule | Manuscript Prop. 2; SI Prop. S2.5(ii) | `chartL_updateSeq_inGrPlus2` (Chart) |
| `prop2_GrPlus2_gradient_descent` | Proposition 2, gradient-descent instance | Manuscript Prop. 2 ("every gradient step"); SI Prop. S2.5(ii) | `chartL_gradientDescent_inGrPlus2` (Chart) |
| `propS25iii_interiority` | Quantitative interiority | SI Prop. S2.5(iii), eq. (S2.3) | `minor2_chartL_ge` (Chart) |
| `propS25iii_interiorityConst_pos` | Positivity of the interiority constant | SI Prop. S2.5(iii) | `interiorityConst_pos` (Chart) |
| `lemmaS22_chart_image_eq_GrPlus2` | The chart covers `Gr₊(2,n)` | SI Lemma S2.2; SI Prop. S2.5(i) | `chart_image_eq_GrPlus2` (Chart) |
| `lemmaS27_hermPart_isHermitian` | Hermitian part | SI Lemma S2.7; manuscript Methods | `hermPart_isHermitian` (Unitary) |
| `prop2_coupling_unitary` | Proposition 2, unitary coupling | Manuscript Prop. 2 and Methods; SI Lemma S2.7 | `couplingU_mem_unitaryGroup` (Unitary) |
| `corS210_coupling_unitary_every_step` | Proposition 2, unitary at every step | SI Cor. S2.10 (the `U` side) | `couplingU_update_mem_unitaryGroup` (Unitary) |
| `eq2_pU_zero` | Normalization of the secular polynomial | Manuscript eq. (2) | `pU_zero` (Secular) |
| `thmS28a_stability` | Lee-Yang stability (Kurasov-Sarnak) | SI Thm S2.8(a) | `pU_ne_zero_of_norm_lt_one` (Secular) |
| `thmS28b_functional_equation` | Self-inversive functional equation | SI Thm S2.8(b), eq. (S2.4) | `pU_functional_equation` (Secular) |
| `thmS28c_exterior_exclusion` | Exterior exclusion | SI Thm S2.8(c), first clause | `pU_ne_zero_of_one_lt_norm` (Secular) |
| `thmS28c_zero_confinement` | Zero confinement | SI Thm S2.8(c) | `pU_eq_zero_confinement` (Secular) |
| `thmS28c_real_rootedness` | Real-rootedness along positive directions | SI Thm S2.8(c); Remark S2.9 | `pU_expLine_ne_zero` (Secular) |
| `thmS28d_torus_zeros_eigenvalue_one` | Torus zeros are eigenphase crossings | SI Thm S2.8(d) | `torus_zero_iff_eigenvalue_one` (Secular) |
| `secular_principal_minor_expansion` | Principal-minor expansion | SI eq. (S1.5) | `pU_eq_sum_principal_minors` (Secular) |
| `opC_residual_re_im` | Op-C residual | Manuscript Methods (Op-C) | `opC_residual_re_im_eq_zero_iff` (Secular) |
| `opC_mem_fqPointSet_iff_eigenvalue_one` | Op-B / Op-C link | SI Thm S2.8(d); manuscript Methods | `mem_fqPointSet_iff_eigenvalue_one` (Secular) |

#### Section `Spectrum`

| Main theorem | Statement | Paper location | Proved by |
|---|---|---|---|
| `eq1_torusLift_mem_torus` | Torus lift | Manuscript eq. (1); SI eq. (S1.1) | `torusLift_mem_torus` (TorusEmbedding) |
| `opA_features_eq_torusLift` | Op-A features | Manuscript Methods (Op-A) | `torusLift_eq_opAFeat` (TorusEmbedding) |
| `period_criterion` | Period criterion | SI §S1.1 | `isPeriod_iff` (TorusEmbedding) |
| `fqPointSet_period_invariance` | Periods pass to every FQ point set | SI §S1.1, eq. (S1.3) | `fqPointSet_add_period` (TorusEmbedding) |
| `aperiodicity_criterion` | Aperiodicity criterion, exact form | Manuscript Prop. 2; SI §S1.1 | `hasNonzeroPeriod_iff` (TorusEmbedding) |
| `aperiodicity_criterion_columnSpace` | Aperiodicity criterion, column-space form | Manuscript Prop. 2; SI §S1.1 | `hasNonzeroPeriod_iff_range_of_injective` (TorusEmbedding) |
| `aperiodicity_d1_iff` | Aperiodicity for `d = 1` | SI §S1.1 | `hasNonzeroPeriod_one_iff` (TorusEmbedding) |
| `counterexample_rowIndependent_but_periodic` | Counterexample: ℚ-independent rows do not give aperiodicity when `d ≥ 2` | Manuscript Prop. 2; SI §S1.1, §S2.6 | `counterexample_rows_linearIndependent` (TorusEmbedding), `counterexample_int_relation_trivial` (TorusEmbedding), `counterexample_mulVec` (TorusEmbedding), `counterexample_hasNonzeroPeriod` (TorusEmbedding) |
| `spectrumModule_span_eq_rows` | Spectrum module generated by the rows | SI §S1.1, §S1.5 | `span_spectrumModule` (SpectrumModule) |
| `qRank_le` | `ℚ`-rank bound | SI §S1.5 | `finrank_span_spectrumModule_le` (SpectrumModule) |
| `qRank_eq_iff` | `ℚ`-rank theorem | SI §S1.5 | `finrank_span_spectrumModule_eq_iff` (SpectrumModule) |
| `rowIndependent_iff_no_integer_relation` | ℚ-independence in the SI's integer-relation form | SI §S1.1 | `linearIndependent_rat_iff` (SpectrumModule) |
| `d1_spectrumModule_dense` | Density of the spectrum module for `d = 1` | SI §S1.5 | `dense_spectrumModule1` (SpectrumModule) |
| `d1_no_exclusion_zone` | No exclusion zone for the ambient module | SI §S1.5 | `no_exclusion_zone` (SpectrumModule) |
| `thinnedSpectrum_subset_and_symm` | Basic properties of the thinned spectrum | Manuscript eq. (1); SI §S1.5 | `extinctionSpectrum_subset` (SpectrumModule), `neg_mem_extinctionSpectrum_iff` (SpectrumModule) |
| `thinnedSpectrum_exclusion_zone_deployed` | The thinned spectrum has an exclusion zone for every deployed chart generator | SI §S1.5 ("`Λ'` … is a discrete, nonlattice set") | `chartL_extinctionSpectrum_exclusion_zone` (SpectrumModule) |
| `thinnedSpectrum_locally_finite` | The thinned spectrum is locally finite on `Gr₊(2,n)` | SI §S1.5 and the §S1 summary table | `extinctionSpectrum_finite_inter_closedBall_of_inGrPlus2` (SpectrumModule) |
| `thinned_vs_ambient_sqrt_two` | Thinned spectrum versus ambient module: an explicit contrast | SI §S1.5 | `extinctionSpectrum_vs_spectrumModule_sqrt_two` (SpectrumModule) |

#### Section `BackboneDensity`

| Main theorem | Statement | Paper location | Proved by |
|---|---|---|---|
| `counterexampleTP_rowIndependent_but_periodic` | Totally positive counterexample: ℚ-independent rows do not give aperiodicity, even in `Gr₊(2,3)` | Manuscript Prop. 2; SI §S1.1, §S2.6 | `counterexampleTP_rowIndependent_but_periodic` (AperiodicityTP) |
| `chart_attains_rowIndependent_but_periodic` | The deployed chart attains a periodic generator with ℚ-independent rows | Manuscript Prop. 2 and eq. (10); SI Lemma S2.2, §S1.1, §S2.6 | `exists_chartL_rowIndependent_but_periodic` (AperiodicityTP) |
| `thinnedSpectrum_span_eq_spectrumModule` | The thinning does not change the rational span | SI §S1.5 | `span_extinctionSpectrum_eq` (SpectrumRankThinned) |
| `qRank_thinned_le` | `ℚ`-rank bound for the thinned spectrum | SI §S1.5 ("rational span may have dimension up to `n`") | `qRank_thinned_le` (SpectrumRankThinned) |
| `qRank_thinned_eq_iff` | `ℚ`-rank theorem for the thinned spectrum | SI §S1.5 ("`dim_ℚ = n` exactly under `ℚ`-independence of the rows of `L`") | `qRank_thinned_eq_iff` (SpectrumRankThinned) |
| `propS25iii_norm_tendsto_atTop` | Prop. S2.5(iii), limit form | SI Prop. S2.5(iii), "consequently … forces `‖φ‖_∞ → ∞`" | `tendsto_norm_atTop_of_minor2_tendsto_zero` (ChartLimit) |
| `propS25iii_interiority_oneSided` | Sharpened (S2.3): one-sided bounds suffice | SI Prop. S2.5(iii), eq. (S2.3) | `minor2_chartL_ge_of_oneSided` (ChartLimit) |
| `propS25iii_coordinate_case_split` | Prop. S2.5(iii), coordinate case split, subsequence form | SI Prop. S2.5(iii), "a radius `ρ_i → −∞`, the scale `log s → −∞`, or … some `α_l → −∞` or some `α_m → +∞`" | `chart_coordinate_case_split` (ChartLimit) |
| `kronecker_density_iff` | Kronecker's theorem for the torus lift, exact form | SI §S1.1; manuscript eq. (1) | `closure_range_torusLift_eq_torus_iff` (Kronecker) |
| `kronecker_density` | Kronecker density in the SI's integer-relation form | SI §S1.1 | `closure_range_torusLift` (Kronecker) |
| `spectrumModule_dense_iff_aperiodic` | Density of the ambient module is exactly aperiodicity | SI §S1.5, the §S1 summary table, §S5.2 caveat (ii) and §S5.5(a) | `dense_spectrumModule_iff` (Kronecker) |
| `spectrumModule_not_dense_counterexample` | Counterexample: the ambient module need not be dense for `n > d` | SI §S1.5 ("neither `n > d`, nor positivity of the minors, nor ℚ-independence of the rows suffices") | `grPlusExample_inGrPlus2` (Kronecker), `grPlusExample_rows_linearIndependent` (Kronecker), `grPlusExample_hasNonzeroPeriod` (Kronecker), `grPlusExample_spectrumModule_not_dense` (Kronecker) |

#### Section `ReadoutComb`

| Main theorem | Statement | Paper location | Proved by |
|---|---|---|---|
| `chord_identity_S46` | The S4.6 chord identity | Manuscript Results ("an exact identity we prove"); SI §S4.6, eq. (S4.10) | `timeLinearInterp_eq_chord` (ExponentialSum) |
| `eq4_Dmu_is_derivative` | `Dμ` is the derivative of the Op-D kernel | Manuscript eqs. (3)-(4), one-variable form | `hasDerivAt_kernelOpD` (ExponentialSum) |
| `eq4_Dmu_is_directional_derivative` | `Dμ` in `ℝᵈ`, as a directional derivative | Manuscript eq. (4) with `ξ_k ∈ ℝᵈ` | `hasDerivAt_kernelOpDVec` (ExponentialSum) |
| `S41_affine_comb_with_two_zeros_is_zero` | Constraints protect the affine atom | SI §S4.1 | `rcomb_eq_zero_of_affine` (ExponentialSum) |
| `S41_direct_sum` | Direct sum of the affine atom and the comb | SI §S4.1 | `affine_add_rcomb_eqOn_zero` (ExponentialSum) |
| `S41_reconBasis_linearIndependent` | Linear independence of the reconstruction basis | SI §S4.1 | `linearIndependent_reconBasis` (ExponentialSum) |

#### Section `ReadoutOpH`

| Main theorem | Statement | Paper location | Proved by |
|---|---|---|---|
| `propS41_exact_interpolation` | Proposition S4.1 (exact interpolation for every `η` and every metric) | SI Prop. S4.1 | `mulVec_opHW` (OpH) |
| `propS41_hermite_interpolation` | Proposition S4.1 on the boundary-Hermite system | SI Prop. S4.1, with SI eqs. (S4.5)-(S4.6) | `opHW_hermite_interpolation` (OpH) |
| `prop1_affine_reproduction_opH` | Proposition 1 (affine reproduction), Op-H readout of eq. (6) | Manuscript Prop. 1; SI §S4.5 | `prop1_affine_reproduction` (OpH) |
| `prop1_affine_reproduction_opM` | Proposition 1 and Theorem S7.5(i), full readout of eqs. (7)-(8) | Manuscript Prop. 1 (exterior self-gating clause); SI Thm S7.5(i) | `thmS75i_chord_exactness` (OpM) |
| `propS45_zero_data_zero_gate` | Zero data and a closed gate give zero coefficients | SI §S4.5 | `opHW_zero_zero` (OpH) |
| `propS61ii_metric_homogeneity` | Metric homogeneity | SI Prop. S6.1(ii) | `opHW_smul_metric` (OpH) |
| `propS61iii_normalization_invariance` | Max-normalization makes the solve scale-invariant | SI Prop. S6.1(iii); manuscript Methods ("max-normalized per gap") | `opHW_normalizeW_smul` (OpH) |
| `row_equilibration_invariance` | Row equilibration leaves the solve unchanged | SI §S4.2 | `opHW_mul_left` (OpH) |
| `opHW_weighted_min_norm` | Weighted minimum-norm characterization | SI §S4.3; SI Prop. S6.2 (first half) | `partOpHW_isWeightedMinNorm` (OpH) |

#### Section `ReadoutKernel`

| Main theorem | Statement | Paper location | Proved by |
|---|---|---|---|
| `propS51_kernel_posSemidef` | Proposition S5.1 (positive semidefiniteness of the FQ kernel, finite spectrum) | SI Prop. S5.1, eqs. (S5.2)-(S5.3) | `fqGram_posSemidef` (Kernel) |
| `propS51_kernel_complex_nonneg` | Proposition S5.1, complex-coefficient form | SI eq. (S5.3) | `fqGram_complex_nonneg` (Kernel) |
| `propS51_sum_of_squares` | Proposition S5.1, sum-of-squares identity | SI eq. (S5.4), finite spectrum | `dotProduct_fqGram_mulVec` (Kernel) |
| `propS51_profile` | Profile of the FQ kernel | SI Prop. S5.1 and §S5.3 | `fqProfile_zero` (Kernel), `fqProfile_neg` (Kernel), `continuous_fqProfile` (Kernel) |
| `propS62_reproducing` | Reproducing property of the kriging kernel | SI Prop. S6.2 (proof) | `vInner_krigSection` (Kernel) |
| `propS62_reproducing_deriv` | Reproducing property for derivative functionals | SI Prop. S6.2 (proof) | `vInner_krigSectionDeriv` (Kernel) |
| `propS62_kriging_eq_opHW` | Proposition S6.2 (kriging unification) | SI Prop. S6.2; manuscript Methods ("kernel weights and spectral metric are one object") | `kriging_eq_opHW` (Kernel) |

#### Section `ReadoutOpM`

| Main theorem | Statement | Paper location | Proved by |
|---|---|---|---|
| `thmS72_soft_block_invertible` | The soft block is always invertible | SI Thm S7.2 | `isUnit_det_one_add_opMGram` (OpM) |
| `propS71_opM_unique_minimizer` | Proposition S7.1 / Theorem S7.2 (Op-M is the unique constrained minimizer) | Manuscript eqs. (7)-(8); SI Prop. S7.1, Thm S7.2 | `opM_isUniqueMinimizer` (OpM) |
| `thmS73_exactness_every_dial` | Theorem S7.3 (exactness survives every dial) | SI Thm S7.3 | `mulVec_opM` (OpM) |
| `thmS73_hermite_interpolation` | Theorem S7.3 on the boundary-Hermite system | SI Thm S7.3; manuscript Methods | `opM_hermite_interpolation` (OpM) |
| `corS74_endpoint_annihilation` | Corollary S7.4 (endpoint annihilation) | SI Cor. S7.4; manuscript Methods | `exterior_endpoint_annihilation` (OpM) |
| `prop3_zero_precision` | Proposition 3 (exact degeneration), `Λ = 0` | Manuscript Prop. 3; SI §S7.3 | `opM_zero_prec` (OpM) |
| `prop3_exact_degeneration` | Proposition 3 (exact degeneration), both dials | Manuscript Prop. 3; SI §S7.3 | `prop3_exact_degeneration` (OpM) |
| `prop3_exact_degeneration_geom` | Proposition 3 with the deployed geometric metric law | Manuscript Prop. 3; SI eq. (S7.3) | `prop3_exact_degeneration_geom` (OpM) |
| `opM_soft_deformation` | Influence law, coefficient form | Op-M closed form, manuscript eq. (8), SI eq. (S7.2); SI §S7.3 | `opM_sub_opM_zero_prec` (OpM) |
| `opM_influence_law` | Influence law, prediction form | Op-M closed form, SI eq. (S7.2) | `influence_law` (OpM) |
| `opM_bcKernel_value_row` | The boundary-conditioned kernel vanishes at value rows | SI Cor. S7.4, kernel form | `bcKernel_eq_zero_of_valueRow` (OpM) |
| `opM_bcKernel_deriv_row` | The boundary-conditioned kernel has zero slope at derivative rows | SI Cor. S7.4, kernel form | `hasDerivAt_bcKernel_of_derivRow` (OpM) |
| `opM_gauge_invariance` | Gauge invariance | Op-M closed form, SI eq. (S7.2) | `opM_gauge` (OpM) |
| `opM_single_anchor_response` | Single soft anchor response (Sherman-Morrison) | Op-M closed form, SI eq. (S7.2) | `singleAnchor_response` (OpM) |
| `thmS75ii_if_gated` | Theorem S7.5(ii), "if" direction (conservative gating) | SI Thm S7.5(ii) | `thmS75ii_gated` (OpM) |
| `thmS75ii_converse_iff` | Theorem S7.5(ii), converse | SI Thm S7.5(ii) ("Conversely, with `B` of full row rank …") | `thmS75ii_converse_iff` (OpM) |
| `thmS75ii_converse_of_injective` | Theorem S7.5(ii), converse under the genericity condition | SI Thm S7.5(ii) | `thmS75ii_converse_of_injective` (OpM) |
| `thmS75ii_only_if_counterexample` | Counterexample to the unqualified converse of SI Thm S7.5(ii) | SI Thm S7.5(ii) ("the unqualified converse fails") | `thmS75ii_only_if_fails` (OpM) |

#### Section `ReadoutBlindSpot`

| Main theorem | Statement | Paper location | Proved by |
|---|---|---|---|
| `propS76_chordExact_imp_exteriorBlind` | Proposition S7.6 (blind-spot trade), first half | SI Prop. S7.6 | `exteriorBlind_of_chordExact` (BlindSpot) |
| `propS76_responds_imp_not_chordExact` | Proposition S7.6, converse | SI Prop. S7.6 ("conversely") | `not_chordExact_of_respondsAt` (BlindSpot) |
| `propS76_misses_sCurve` | Proposition S7.6, concrete parallel-tangent S-curve | SI Prop. S7.6 | `chordExact_misses_sCurve` (BlindSpot) |
| `propS76_exterior_oracle_trade` | The blind spot is a trade, not an information-theoretic limit | SI §S7.2, remark after Prop. S7.6 | `exterior_oracle_trade` (BlindSpot) |

#### Section `MoorePenrose`

| Main theorem | Statement | Paper location | Proved by |
|---|---|---|---|
| `moorePenrose_unique` | Uniqueness of the Moore–Penrose inverse | SI §S4.4 (proof of Prop. S4.1, "the Penrose identity") | `IsMoorePenrose.unique` (MoorePenrose) |
| `moorePenrose_exists` | Existence of the Moore–Penrose inverse | SI §S4.4, SI eq. (S4.7) and manuscript eqs. (6), (8), which use `B⁺` with no rank hypothesis | `exists_isMoorePenrose` (MoorePenrose) |
| `moorePenrose_iff_eq_mpinv` | Characterization of `mpinv` | SI §S4.4 | `isMoorePenrose_iff_eq_mpinv` (MoorePenrose) |
| `mpinv_eq_pinvR_of_fullRowRank` | Agreement with the right-inverse model | SI §S4.4; manuscript eqs. (6), (8) | `mpinv_eq_pinvR` (MoorePenrose) |
| `mpinv_feasible_iff_mem_range` | Rank-free feasibility criterion | SI Prop. S4.1 ("`d ∈ range(B)`") | `mulVec_mpinv_mulVec_eq_iff` (MoorePenrose) |
| `mpinv_min_norm` | Rank-free minimum-norm characterization | SI §S4.3, with no rank hypothesis | `mpinv_mulVec_isMinNorm` (MoorePenrose) |
| `S44_null_space_annihilation` | Null-space annihilation with no rank hypothesis | SI §S4.4 ("`X(I − X⁺X) = 0` unconditionally (no rank hypothesis)") | `mulVec_mul_one_sub_mpinv_mulVec` (OpHGeneral) |

#### Section `ReadoutMP`

| Main theorem | Statement | Paper location | Proved by |
|---|---|---|---|
| `opHW_mp_eq_opHW_of_fullRowRank` | Op-H-W with the Moore–Penrose inverse agrees with the right-inverse model | Manuscript eq. (6); SI eq. (S4.7) | `opHW_mp_eq_opHW` (OpHGeneral) |
| `propS41_exact_interpolation_mp` | Proposition S4.1, rank-free | SI Prop. S4.1 and its proof; manuscript eq. (6) | `mulVec_opHW_mp` (OpHGeneral) |
| `propS41_exact_interpolation_mp_iff` | Proposition S4.1, sharp form | SI Prop. S4.1 | `mulVec_opHW_mp_eq_iff` (OpHGeneral) |
| `propS41_hermite_interpolation_mp` | Proposition S4.1 on the boundary-Hermite system, rank-free | SI Prop. S4.1 with SI eqs. (S4.5)-(S4.6) | `opHW_mp_hermite_interpolation` (OpHGeneral) |
| `propS61i_zero_data_mp` | Proposition S6.1(i), "unconditionally when `d = 0`" | SI Prop. S6.1(i) | `mulVec_opHW_mp_zero` (OpHGeneral) |
| `prop1_affine_reproduction_opH_mp` | Proposition 1 (affine reproduction), Op-H readout with the Moore–Penrose inverse | Manuscript Prop. 1; SI §S4.5 | `prop1_affine_reproduction_mp` (OpHGeneral) |
| `opM_mp_eq_opM_of_fullRowRank` | Op-M with the Moore–Penrose inverse agrees with the right-inverse model | Manuscript eq. (8); SI eq. (S7.2) | `opM_mp_eq_opM` (OpHGeneral) |
| `thmS73_exactness_every_dial_mp` | Theorem S7.3 (exactness survives every dial), rank-free | SI Thm S7.3 | `mulVec_opM_mp` (OpHGeneral) |
| `thmS73_exactness_mp_iff` | Theorem S7.3, sharp form | SI Thm S7.3 | `mulVec_opM_mp_eq_iff` (OpHGeneral) |
| `thmS73_hermite_interpolation_mp` | Theorem S7.3 on the boundary-Hermite system, rank-free | SI Thm S7.3; manuscript Methods | `opM_mp_hermite_interpolation` (OpHGeneral) |
| `corS74_exterior_in_null_space_mp` | Corollary S7.4, algebraic form, no hypothesis | SI Cor. S7.4 | `mulVec_metricW_opMSoft_mp` (OpHGeneral) |
| `corS74_endpoint_annihilation_mp` | Corollary S7.4 (endpoint annihilation), no rank and no metric hypothesis | SI Cor. S7.4 | `exterior_endpoint_annihilation_mp` (OpHGeneral) |
| `prop3_zero_precision_mp` | Proposition 3, `Λ = 0`, with the Moore–Penrose inverse | Manuscript Prop. 3; SI §S7.3 | `opM_mp_zero_prec` (OpHGeneral) |

#### Section `ReadoutGenericity`

| Main theorem | Statement | Paper location | Proved by |
|---|---|---|---|
| `S44_value_block_rank_deficient_iff` | Genericity of the rank hypothesis at `R = 2` | SI §S4.4, "Genericity of the rank hypothesis" | `not_fullRowRank_valueBlock_iff` (Genericity) |
| `S44_value_block_fullRowRank` | Generic full row rank at `R = 2` | SI §S4.4 | `fullRowRank_valueBlock_of_phase` (Genericity) |

#### Section `ReadoutKernelInfinite`

| Main theorem | Statement | Paper location | Proved by |
|---|---|---|---|
| `propS51_infinite_posSemidef` | Proposition S5.1, infinite spectrum | SI Prop. S5.1 under (H1) and (H3) (§S5.1, §S5.4) | `fqGramInf_posSemidef` (KernelInfinite) |
| `propS51_infinite_complex_nonneg` | Proposition S5.1, infinite spectrum, complex-coefficient form | SI eq. (S5.3) under (H1) and (H3) | `fqGramInf_complex_nonneg` (KernelInfinite) |
| `propS51_infinite_profile` | Profile of the infinite-spectrum kernel | SI Prop. S5.1 and §S5.3 ("Reality and evenness"; "Continuity of `κ` follows from (H3)") | `fqProfileInf_zero` (KernelInfinite), `fqProfileInf_neg` (KernelInfinite), `continuous_fqProfileInf` (KernelInfinite) |
| `propS51_truncation_tendstoUniformly` | Finite truncations converge uniformly | SI §S5.4 ("Proposition S5.1 extends in the limit") | `tendstoUniformly_fqProfile_trunc` (KernelInfinite) |
| `propS54_decay_posSemidef` | Proposition S5.1 for the full module under polynomial decay | SI §S5.4 (the convergent direction of the decay criterion) | `fqGramInf_posSemidef_of_decay` (KernelInfinite) |
| `propS55a_posDef_of_aperiodic` | SI §S5.5(a), idealized infinite limit, under the aperiodicity hypothesis | SI §S5.5(a) | `fqGramInf_posDef_ambient` (KernelInfinite) |
| `propS55a_not_posDef_of_periodic` | SI §S5.5(a) fails without aperiodicity | SI §S5.5(a) | `hasNonzeroPeriod_iff` (TorusEmbedding), `fqGramInf_ambient_not_posDef` (KernelInfinite) |

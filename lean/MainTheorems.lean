import FQGNN

/-!
# FQ-GNN: main machine-checked theorems

This file contains only the final public statements of the formalization. Each one is restated in full
and proved by the corresponding module theorem of the `FQGNN` library. Nothing is proved here from
scratch, and nothing is stated that the modules do not prove.

Each docstring gives the statement in words and its location in the paper: "manuscript" refers to
the main text (Props. 1-3, eqs. (1)-(10)) and "SI" to the Supplementary Information (§§S1-S7).
`FORMALIZATION_SCOPE.md` lists the hypotheses under which the formal statements hold where they are
sharper than the informal text, and what is not formalized.

Modelling conventions used throughout:
* "full row rank" is `FQGNN.Common.FullRowRank B := IsUnit (B * Bᵀ).det`;
* two pseudoinverses are used. `pinvR B = Bᵀ (B Bᵀ)⁻¹` is the Moore-Penrose inverse under full row
  rank; the statements of the sections up to `ReadoutBlindSpot` use it (`opHW`, `opM`). `mpinv B` is
  the genuine Moore-Penrose inverse of every real matrix (constructed in `FQGNN.Common.MoorePenrose`);
  the rank-free statements of section `ReadoutMP` use it (`opHW_mp`, `opM_mp`), and the two agree under
  full row rank. The `rcond` cutoff and floating point are not modelled;
* comb coefficients live on `Fin K ⊕ Fin K` (cosine part `inl k`, sine part `inr k`);
* kernel statements are for a finite spectrum, except in section `ReadoutKernelInfinite`, which allows
  any spectrum index with summable nonnegative masses.
-/

namespace FQGNN.Main

/-! ## Backbone: the generator chart, the unitary coupling and the secular polynomial (Prop. 2) -/

section Backbone

open Matrix FQGNN.Backbone

variable {n : ℕ}

/-- **Chart minor closed form.** Manuscript Methods eq. (10) paragraph; SI eq. (S2.2).
For every chart parameter `φ = (ρ, α, log s)`, the `2×2` minor of rows `i, j` of the chart generator
`L(φ)` is `Δ_ij = s² r_i r_j sin(θ_j − θ_i)`. Here `r_i = softplus ρ_i`, `s = exp(log s)`, and the
`θ` are the normalized cumulative-softplus angles (S2.1). No hypotheses. -/
theorem chart_minor_closed_form (φ : ChartParams n) (i j : Fin n) :
    minor2 (chartL φ) i j = chartScale φ ^ 2 * chartRadius φ i * chartRadius φ j *
      Real.sin (chartAngle φ.2.1 j - chartAngle φ.2.1 i) :=
  FQGNN.Backbone.minor2_chartL φ i j

/-- **Proposition 2(i), membership in the family.** Manuscript Prop. 2; SI Lemma S2.1.
Every finite chart parameter `φ` gives a generator `L(φ)` all of whose `2×2` minors `Δ_ij`, `i < j`,
are strictly positive. In other words, `L(φ) ∈ Gr₊(2,n)` on representatives. -/
theorem prop2_chart_mem_GrPlus2 (φ : ChartParams n) : InGrPlus2 (chartL φ) :=
  FQGNN.Backbone.chartL_inGrPlus2 φ

/-- **Proposition 2, always-in-family under any update rule.** Manuscript Prop. 2;
SI Prop. S2.5(ii). Take any time-dependent update rule `T` and any initial parameter `φ₀`. Every
iterate `φ_t` of the rule gives a generator in `Gr₊(2,n)`. No projection or penalty is involved. -/
theorem prop2_GrPlus2_every_update (T : ℕ → ChartParams n → ChartParams n)
    (φ₀ : ChartParams n) (t : ℕ) : InGrPlus2 (chartL (updateSeq T φ₀ t)) :=
  FQGNN.Backbone.chartL_updateSeq_inGrPlus2 T φ₀ t

/-- **Proposition 2, gradient-descent instance.** Manuscript Prop. 2 ("every gradient step");
SI Prop. S2.5(ii). Consider the iteration `φ_{t+1} = φ_t − η_t G_t(φ_t)` with arbitrary step sizes
and an arbitrary field `G`. It keeps the generator in `Gr₊(2,n)` at every step. -/
theorem prop2_GrPlus2_gradient_descent (G : ℕ → ChartParams n → ChartParams n)
    (η : ℕ → ℝ) (φ₀ : ChartParams n) (t : ℕ) :
    InGrPlus2 (chartL (updateSeq (fun k φ => φ - η k • G k φ) φ₀ t)) :=
  FQGNN.Backbone.chartL_gradientDescent_inGrPlus2 G η φ₀ t

/-- **Quantitative interiority.** SI Prop. S2.5(iii), eq. (S2.3).
If `‖φ‖_∞ ≤ B`, then every minor `Δ_ij(L(φ))` with `i < j` is at least the explicit constant
`m(B) = e^{−2B} softplus(−B)² sin δ_B`, with `δ_B = (π/2) softplus(−B) / ((n+1) softplus B)`.
The boundary of `Gr₊(2,n)` is therefore reached only at parameter infinity. -/
theorem propS25iii_interiority (φ : ChartParams n) {B : ℝ} (hφ : ‖φ‖ ≤ B) {i j : Fin n}
    (hij : i < j) : interiorityConst n B ≤ minor2 (chartL φ) i j :=
  FQGNN.Backbone.minor2_chartL_ge φ hφ hij

/-- **Positivity of the interiority constant.** SI Prop. S2.5(iii). `m(B) > 0` for every `B ≥ 0`. -/
theorem propS25iii_interiorityConst_pos (n : ℕ) {B : ℝ} (hB : 0 ≤ B) :
    0 < interiorityConst n B :=
  FQGNN.Backbone.interiorityConst_pos n hB

/-- **The chart covers `Gr₊(2,n)`.** SI Lemma S2.2; SI Prop. S2.5(i).
Assume `n ≥ 2`. A real `n × 2` matrix has all `2×2` minors positive iff it is a chart generator up to
a change of basis `L' ↦ L' g` with `det g > 0`. The hypothesis `n ≥ 2` is needed: for `n ≤ 1` the
minor condition is vacuous on representatives, so a zero row would satisfy it. -/
theorem lemmaS22_chart_image_eq_GrPlus2 (hn : 2 ≤ n) (L' : Matrix (Fin n) (Fin 2) ℝ) :
    InGrPlus2 L' ↔
      ∃ φ : ChartParams n, ∃ g : Matrix (Fin 2) (Fin 2) ℝ, 0 < g.det ∧ chartL φ = L' * g :=
  FQGNN.Backbone.chart_image_eq_GrPlus2 hn L'

/-- **Hermitian part.** SI Lemma S2.7; manuscript Methods. For every complex square matrix `A`,
`H = ½(A + Aᴴ)` is Hermitian. -/
theorem lemmaS27_hermPart_isHermitian (A : Matrix (Fin n) (Fin n) ℂ) :
    (hermPart A).IsHermitian :=
  FQGNN.Backbone.hermPart_isHermitian A

/-- **Proposition 2, unitary coupling.** Manuscript Prop. 2 and Methods; SI Lemma S2.7.
For every complex parameter matrix `A`, the coupling `U = exp(i · ½(A + Aᴴ))` lies in the unitary
group `U(n)`. -/
theorem prop2_coupling_unitary (A : Matrix (Fin n) (Fin n) ℂ) :
    couplingU A ∈ Matrix.unitaryGroup (Fin n) ℂ :=
  FQGNN.Backbone.couplingU_mem_unitaryGroup A

/-- **Proposition 2, unitary at every step.** SI Cor. S2.10 (the `U` side). Under any update rule
`T` for the parameter `A`, the coupling `U_t = exp(i · ½(A_t + A_tᴴ))` is unitary at every step `t`. -/
theorem corS210_coupling_unitary_every_step
    (T : ℕ → Matrix (Fin n) (Fin n) ℂ → Matrix (Fin n) (Fin n) ℂ)
    (A₀ : Matrix (Fin n) (Fin n) ℂ) (t : ℕ) :
    couplingU (updateSeqA T A₀ t) ∈ Matrix.unitaryGroup (Fin n) ℂ :=
  FQGNN.Backbone.couplingU_update_mem_unitaryGroup T A₀ t

/-- **Normalization of the secular polynomial.** Manuscript eq. (2). `p_U(0) = 1`, where
`p_U(z) = det(I − diag(z) U)`. -/
theorem eq2_pU_zero (U : Matrix (Fin n) (Fin n) ℂ) : pU U 0 = 1 :=
  FQGNN.Backbone.pU_zero U

/-- **Lee-Yang stability (Kurasov-Sarnak).** SI Thm S2.8(a). If `U` is unitary and `|z_j| < 1` for
all `j`, then `p_U(z) ≠ 0`. The SI cites this result; here it is proved. -/
theorem thmS28a_stability {U : Matrix (Fin n) (Fin n) ℂ} (hU : U ∈ unitaryGroup (Fin n) ℂ)
    {z : Fin n → ℂ} (hz : ∀ j, ‖z j‖ < 1) : pU U z ≠ 0 :=
  FQGNN.Backbone.pU_ne_zero_of_norm_lt_one hU hz

/-- **Self-inversive functional equation.** SI Thm S2.8(b), eq. (S2.4). For unitary `U` and
`z ∈ (ℂ^×)ⁿ`, `(∏ z̄_j) p_U(1/z̄) = (−1)ⁿ det U · conj(p_U(z))`. -/
theorem thmS28b_functional_equation {U : Matrix (Fin n) (Fin n) ℂ}
    (hU : U ∈ unitaryGroup (Fin n) ℂ) {z : Fin n → ℂ} (hz : ∀ j, z j ≠ 0) :
    (∏ j, star (z j)) * pU U (fun j => (star (z j))⁻¹) =
      (-1) ^ n * U.det * star (pU U z) :=
  FQGNN.Backbone.pU_functional_equation hU hz

/-- **Exterior exclusion.** SI Thm S2.8(c), first clause. If `U` is unitary and `|z_j| > 1` for all
`j`, then `p_U(z) ≠ 0`. -/
theorem thmS28c_exterior_exclusion {U : Matrix (Fin n) (Fin n) ℂ}
    (hU : U ∈ unitaryGroup (Fin n) ℂ) {z : Fin n → ℂ} (hz : ∀ j, 1 < ‖z j‖) : pU U z ≠ 0 :=
  FQGNN.Backbone.pU_ne_zero_of_one_lt_norm hU hz

/-- **Zero confinement.** SI Thm S2.8(c). Every zero `z` of `p_U`, for unitary `U`, has some
coordinate with `|z_j| ≥ 1` and some coordinate with `|z_j| ≤ 1`, i.e.
`min_j |z_j| ≤ 1 ≤ max_j |z_j|`. -/
theorem thmS28c_zero_confinement {U : Matrix (Fin n) (Fin n) ℂ}
    (hU : U ∈ unitaryGroup (Fin n) ℂ) {z : Fin n → ℂ} (h : pU U z = 0) :
    (∃ j, 1 ≤ ‖z j‖) ∧ (∃ j, ‖z j‖ ≤ 1) :=
  FQGNN.Backbone.pU_eq_zero_confinement hU h

/-- **Real-rootedness along positive directions.** SI Thm S2.8(c); Remark S2.9.
Let `U` be unitary and `ℓ_j > 0`. Restrict `p_U` to the line `z_j = e^{i ℓ_j t}`. This restriction
has no zero with `Im t ≠ 0`. -/
theorem thmS28c_real_rootedness {U : Matrix (Fin n) (Fin n) ℂ}
    (hU : U ∈ unitaryGroup (Fin n) ℂ) {ℓ : Fin n → ℝ} (hℓ : ∀ j, 0 < ℓ j) {t : ℂ}
    (ht : t.im ≠ 0) : pU U (expLine ℓ t) ≠ 0 :=
  FQGNN.Backbone.pU_expLine_ne_zero hU hℓ ht

/-- **Torus zeros are eigenphase crossings.** SI Thm S2.8(d). Let `U` be unitary and `z` lie on the
torus. Then `diag(z) U` is unitary, and `p_U(z) = 0` iff `diag(z) U` has a nonzero fixed vector,
that is, eigenvalue `1`. -/
theorem thmS28d_torus_zeros_eigenvalue_one {U : Matrix (Fin n) (Fin n) ℂ}
    (hU : U ∈ unitaryGroup (Fin n) ℂ) {z : Fin n → ℂ} (hz : ∀ j, ‖z j‖ = 1) :
    diagonal z * U ∈ unitaryGroup (Fin n) ℂ ∧
      (pU U z = 0 ↔ ∃ v : Fin n → ℂ, v ≠ 0 ∧ (diagonal z * U) *ᵥ v = v) :=
  FQGNN.Backbone.torus_zero_iff_eigenvalue_one hU hz

/-- **Principal-minor expansion.** SI eq. (S1.5).
`p_U(z) = Σ_S (−1)^{|S|} (∏_{i∈S} z_i) det(U_S)`, where `U_S` is the principal submatrix on `S`.
Holds for every matrix `U`. -/
theorem secular_principal_minor_expansion (U : Matrix (Fin n) (Fin n) ℂ) (z : Fin n → ℂ) :
    pU U z = ∑ s : Finset (Fin n),
      (-1) ^ s.card * (∏ i ∈ s, z i) *
        (U.submatrix (Subtype.val : s → Fin n) Subtype.val).det :=
  FQGNN.Backbone.pU_eq_sum_principal_minors U z

/-- **Op-C residual.** Manuscript Methods (Op-C). The real and imaginary parts of `p_U(Φ_L x)` both
vanish exactly when `x` lies in the FQ point set `Λ(X_U, L) = {x | Φ_L x ∈ X_U}`, where
`X_U = {p_U = 0}`. This is definitional once `Λ(X_U, L)` is defined this way. The theorem does not
claim that this set is a Delone or Fourier-quasicrystal set. -/
theorem opC_residual_re_im {d : ℕ} (U : Matrix (Fin n) (Fin n) ℂ)
    (L : Matrix (Fin n) (Fin d) ℝ) (x : Fin d → ℝ) :
    ((pU U (torusLift L x)).re = 0 ∧ (pU U (torusLift L x)).im = 0) ↔
      x ∈ fqPointSet (secularVariety U) L :=
  FQGNN.Backbone.opC_residual_re_im_eq_zero_iff U L x

/-- **Op-B / Op-C link.** SI Thm S2.8(d); manuscript Methods. Let `U` be unitary. Then `x` lies in
the FQ point set `Λ(X_U, L)` iff the unitary matrix `diag(Φ_L x) U` has eigenvalue `1`. -/
theorem opC_mem_fqPointSet_iff_eigenvalue_one {d : ℕ} {U : Matrix (Fin n) (Fin n) ℂ}
    (hU : U ∈ unitaryGroup (Fin n) ℂ) (L : Matrix (Fin n) (Fin d) ℝ) (x : Fin d → ℝ) :
    diagonal (torusLift L x) * U ∈ unitaryGroup (Fin n) ℂ ∧
      (x ∈ fqPointSet (secularVariety U) L ↔
        ∃ v : Fin n → ℂ, v ≠ 0 ∧ (diagonal (torusLift L x) * U) *ᵥ v = v) :=
  FQGNN.Backbone.mem_fqPointSet_iff_eigenvalue_one hU L x

end Backbone

/-! ## Spectrum: the torus lift, periods, the spectrum module and its thinned subset -/

section Spectrum

open Matrix FQGNN.Backbone

variable {n d : ℕ}

/-- **Torus lift.** Manuscript eq. (1); SI eq. (S1.1). The lift `Φ_L x = exp(2πi Lx)` takes values
in the torus `𝕋ⁿ`. -/
theorem eq1_torusLift_mem_torus (L : Matrix (Fin n) (Fin d) ℝ) (x : Fin d → ℝ) :
    torusLift L x ∈ torus n :=
  FQGNN.Backbone.torusLift_mem_torus L x

/-- **Op-A features.** Manuscript Methods (Op-A). The Op-A lift `[cos 2π(Lx)_i, sin 2π(Lx)_i]` gives
the real and imaginary parts of `Φ_L x`. -/
theorem opA_features_eq_torusLift (L : Matrix (Fin n) (Fin d) ℝ) (x : Fin d → ℝ) (i : Fin n) :
    torusLift L x i =
      (opAFeat L x (Sum.inl i) : ℂ) + (opAFeat L x (Sum.inr i) : ℂ) * Complex.I :=
  FQGNN.Backbone.torusLift_eq_opAFeat L x i

/-- **Period criterion.** SI §S1.1. A vector `p` is a period of
`Φ_L` (`Φ_L(x + p) = Φ_L x` for all `x`) iff `L p ∈ ℤⁿ`. -/
theorem period_criterion (L : Matrix (Fin n) (Fin d) ℝ) (p : Fin d → ℝ) :
    IsPeriod L p ↔ ∀ j, ∃ m : ℤ, (L *ᵥ p) j = m :=
  FQGNN.Backbone.isPeriod_iff L p

/-- **Periods pass to every FQ point set.** SI §S1.1, eq. (S1.3). If `L p ∈ ℤⁿ`, then
`Λ(X, L) + p = Λ(X, L)` for every variety `X`. -/
theorem fqPointSet_period_invariance {L : Matrix (Fin n) (Fin d) ℝ} {p : Fin d → ℝ}
    (hp : ∀ j, ∃ m : ℤ, (L *ᵥ p) j = m) (X : Set (Fin n → ℂ)) :
    (fun x => x + p) '' fqPointSet X L = fqPointSet X L :=
  FQGNN.Backbone.fqPointSet_add_period hp X

/-- **Aperiodicity criterion, exact form.** Manuscript Prop. 2; SI §S1.1. `Φ_L` has a nonzero period iff
`L p ∈ ℤⁿ` for some `p ≠ 0`. -/
theorem aperiodicity_criterion (L : Matrix (Fin n) (Fin d) ℝ) :
    HasNonzeroPeriod L ↔ ∃ p : Fin d → ℝ, p ≠ 0 ∧ ∀ j, ∃ m : ℤ, (L *ᵥ p) j = m :=
  FQGNN.Backbone.hasNonzeroPeriod_iff L

/-- **Aperiodicity criterion, column-space form.** Manuscript Prop. 2; SI §S1.1. Suppose `L` is
injective (rank `d`;
every `L ∈ Gr₊(d,n)` qualifies). Then `Φ_L` has a nonzero period iff the column space of `L`
contains a nonzero integer vector. The condition is on the column space of `L`, not on its rows. -/
theorem aperiodicity_criterion_columnSpace {L : Matrix (Fin n) (Fin d) ℝ}
    (hL : Function.Injective L.mulVec) :
    HasNonzeroPeriod L ↔
      ∃ m : Fin n → ℤ, m ≠ 0 ∧ ∃ p : Fin d → ℝ, L *ᵥ p = fun j => (m j : ℝ) :=
  FQGNN.Backbone.hasNonzeroPeriod_iff_range_of_injective hL

/-- **Aperiodicity for `d = 1`.** SI §S1.1. For a one-column generator `L ∈ ℝⁿ`, `Φ_L` is periodic
iff no ratio `L_j / L_i` (with `L_i ≠ 0`) is irrational. For `d = 1`, rational independence of the
entries is therefore a sufficient certificate of aperiodicity. -/
theorem aperiodicity_d1_iff (L : Fin n → ℝ) :
    HasNonzeroPeriod (colGen L) ↔ ∀ i j, L i ≠ 0 → ¬ Irrational (L j / L i) :=
  FQGNN.Backbone.hasNonzeroPeriod_one_iff L

/-- **Counterexample: ℚ-independent rows do not give aperiodicity when `d ≥ 2`.** Manuscript
Prop. 2; SI §S1.1, §S2.6. Take `L = [[1,0],[√2,0],[0,1]]`. Its rows are `ℚ`-linearly independent; equivalently, in the SI's own
form, no nonzero `k ∈ ℤ³` has `Lᵀk = 0`. Yet `L (0,1)ᵀ = (0,0,1)ᵀ ∈ ℤ³`. So `Φ_L` has the nonzero
period `(0,1)`, and every FQ point set `Λ(X, L)` is invariant under translation by `(0,1)`. -/
theorem counterexample_rowIndependent_but_periodic :
    LinearIndependent ℚ (fun i : Fin 3 => counterexampleGen i) ∧
      (∀ k : Fin 3 → ℤ, counterexampleGenᵀ *ᵥ (fun j => (k j : ℝ)) = 0 → k = 0) ∧
      counterexampleGen *ᵥ ![0, 1] = ![0, 0, 1] ∧
      HasNonzeroPeriod counterexampleGen ∧
      ∀ X : Set (Fin 3 → ℂ), (fun x => x + ![0, 1]) '' fqPointSet X counterexampleGen =
        fqPointSet X counterexampleGen :=
  ⟨FQGNN.Backbone.counterexample_rows_linearIndependent,
    FQGNN.Backbone.counterexample_int_relation_trivial,
    FQGNN.Backbone.counterexample_mulVec,
    FQGNN.Backbone.counterexample_hasNonzeroPeriod.1,
    FQGNN.Backbone.counterexample_hasNonzeroPeriod.2⟩

/-- **Spectrum module generated by the rows.** SI §S1.1, §S1.5. The ambient module
`M_L = Lᵀ ℤⁿ` and the rows of `L` span the same `ℚ`-subspace of `ℝᵈ`. -/
theorem spectrumModule_span_eq_rows (L : Matrix (Fin n) (Fin d) ℝ) :
    Submodule.span ℚ (spectrumModule L : Set (Fin d → ℝ)) =
      Submodule.span ℚ (Set.range L.row) :=
  FQGNN.Backbone.span_spectrumModule L

/-- **`ℚ`-rank bound.** SI §S1.5. `dim_ℚ span(M_L) ≤ n`. -/
theorem qRank_le (L : Matrix (Fin n) (Fin d) ℝ) :
    Module.finrank ℚ (Submodule.span ℚ (spectrumModule L : Set (Fin d → ℝ))) ≤ n :=
  FQGNN.Backbone.finrank_span_spectrumModule_le L

/-- **`ℚ`-rank theorem.** SI §S1.5. `dim_ℚ span(M_L) = n` exactly when the rows of `L` are
`ℚ`-linearly independent. -/
theorem qRank_eq_iff (L : Matrix (Fin n) (Fin d) ℝ) :
    Module.finrank ℚ (Submodule.span ℚ (spectrumModule L : Set (Fin d → ℝ))) = n ↔
      LinearIndependent ℚ L.row :=
  FQGNN.Backbone.finrank_span_spectrumModule_eq_iff L

/-- **ℚ-independence in the SI's integer-relation form.** SI §S1.1. The rows of `L` are
`ℚ`-independent iff no nonzero `k ∈ ℤⁿ` has `Lᵀk = 0`. -/
theorem rowIndependent_iff_no_integer_relation (L : Matrix (Fin n) (Fin d) ℝ) :
    LinearIndependent ℚ L.row ↔ ∀ k : Fin n → ℤ, Lᵀ *ᵥ (fun j => (k j : ℝ)) = 0 → k = 0 :=
  FQGNN.Backbone.linearIndependent_rat_iff L

/-- **Density of the spectrum module for `d = 1`.** SI §S1.5. If some ratio
`L_j / L_i` is irrational, then `Σ_j ℤ L_j` is dense in `ℝ`. -/
theorem d1_spectrumModule_dense {L : Fin n → ℝ} (h : ∃ i j, L i ≠ 0 ∧ Irrational (L j / L i)) :
    Dense (spectrumModule1 L : Set ℝ) :=
  FQGNN.Backbone.dense_spectrumModule1 h

/-- **No exclusion zone for the ambient module.** SI §S1.5. Under the same
hypothesis, there is no `ε > 0` with `|ξ| ≥ ε` for every nonzero `ξ ∈ M_L`. -/
theorem d1_no_exclusion_zone {L : Fin n → ℝ} (h : ∃ i j, L i ≠ 0 ∧ Irrational (L j / L i)) :
    ¬ ∃ ε > 0, ∀ ξ ∈ spectrumModule1 L, ξ ≠ 0 → ε ≤ |ξ| :=
  FQGNN.Backbone.no_exclusion_zone h

/-- **Basic properties of the thinned spectrum.** Manuscript eq. (1); SI §S1.5.
`Λ' = {Lᵀk : var(k) < d} ⊆ M_L`, and `Λ'` is symmetric: `−ξ ∈ Λ' ↔ ξ ∈ Λ'`. -/
theorem thinnedSpectrum_subset_and_symm (L : Matrix (Fin n) (Fin d) ℝ) :
    extinctionSpectrum L ⊆ spectrumModule L ∧
      ∀ ξ, -ξ ∈ extinctionSpectrum L ↔ ξ ∈ extinctionSpectrum L :=
  ⟨FQGNN.Backbone.extinctionSpectrum_subset L,
    FQGNN.Backbone.neg_mem_extinctionSpectrum_iff L⟩

/-- **The thinned spectrum has an exclusion zone for every deployed chart generator.** SI §S1.5
("`Λ'` … is a discrete, nonlattice set"). Assume `n ≥ 2`. For every chart parameter `φ`, the thinned spectrum `Λ'` of `L(φ) ∈ Gr₊(2,n)`
contains no nonzero point inside some ball of radius `ε > 0` around the origin. -/
theorem thinnedSpectrum_exclusion_zone_deployed (hn : 2 ≤ n) (φ : ChartParams n) :
    ∃ ε > 0, ∀ ξ ∈ extinctionSpectrum (chartL φ), ξ ≠ 0 → ε ≤ ‖ξ‖ :=
  FQGNN.Backbone.chartL_extinctionSpectrum_exclusion_zone hn φ

/-- **The thinned spectrum is locally finite on `Gr₊(2,n)`.** SI §S1.5 and the §S1 summary
table. Assume `n ≥ 2` and `L ∈ Gr₊(2,n)`. Then every closed ball contains
only finitely many points of `Λ'`. -/
theorem thinnedSpectrum_locally_finite {L : Matrix (Fin n) (Fin 2) ℝ} (hn : 2 ≤ n)
    (hL : InGrPlus2 L) (R : ℝ) : {ξ ∈ extinctionSpectrum L | ‖ξ‖ ≤ R}.Finite :=
  FQGNN.Backbone.extinctionSpectrum_finite_inter_closedBall_of_inGrPlus2 hn hL R

/-- **Thinned spectrum versus ambient module: an explicit contrast.** SI §S1.5. For `L = (1, √2)`, the thinned spectrum `Λ'` has an exclusion zone, while the ambient
module `M_L = ℤ + √2 ℤ` has nonzero points arbitrarily close to `0`. -/
theorem thinned_vs_ambient_sqrt_two :
    (∃ ε > 0, ∀ ξ ∈ extinctionSpectrum (colGen ![1, √2]), ξ ≠ 0 → ε ≤ ‖ξ‖) ∧
      ∀ ε > 0, ∃ ξ ∈ spectrumModule (colGen ![1, √2]), ξ ≠ 0 ∧ ‖ξ‖ < ε :=
  FQGNN.Backbone.extinctionSpectrum_vs_spectrumModule_sqrt_two

end Spectrum

/-! ## Backbone: totally positive period instance, thinned-spectrum rank, chart limit,
Kronecker density and density of the ambient module -/

section BackboneDensity

open Matrix Filter Topology FQGNN.Backbone

/-- **Totally positive counterexample: ℚ-independent rows do not give aperiodicity, even in
`Gr₊(2,3)`.** Manuscript Prop. 2; SI §S1.1, §S2.6. Take `L = [[1,1],[√2,2],[√3,3]]`. All three `2×2` minors (`2−√2`, `3−√3`,
`3√2−2√3`) are positive, so `L ∈ Gr₊(2,3)`. Its rows are `ℚ`-linearly independent; equivalently no
nonzero `k ∈ ℤ³` has `Lᵀk = 0`. Yet `L (0,1)ᵀ = (1,2,3)ᵀ ∈ ℤ³`, so `Φ_L` has the nonzero period
`(0,1)` and every FQ point set `Λ(X, L)` is invariant under translation by `(0,1)`. -/
theorem counterexampleTP_rowIndependent_but_periodic :
    InGrPlus2 counterexampleTP ∧
      LinearIndependent ℚ (fun i : Fin 3 => counterexampleTP i) ∧
      (∀ k : Fin 3 → ℤ, counterexampleTPᵀ *ᵥ (fun j => (k j : ℝ)) = 0 → k = 0) ∧
      counterexampleTP *ᵥ ![0, 1] = ![1, 2, 3] ∧
      HasNonzeroPeriod counterexampleTP ∧
      ∀ X : Set (Fin 3 → ℂ), (fun x => x + ![0, 1]) '' fqPointSet X counterexampleTP =
        fqPointSet X counterexampleTP :=
  FQGNN.Backbone.counterexampleTP_rowIndependent_but_periodic

/-- **The deployed chart attains a periodic generator with ℚ-independent rows.** Manuscript
Prop. 2 and eq. (10); SI Lemma S2.2, §S1.1, §S2.6. There is a
chart parameter `φ` such that `L(φ) ∈ Gr₊(2,3)` has `ℚ`-linearly independent rows and `Φ_{L(φ)}` has
a nonzero period. -/
theorem chart_attains_rowIndependent_but_periodic :
    ∃ φ : ChartParams 3, InGrPlus2 (chartL φ) ∧ LinearIndependent ℚ (chartL φ).row ∧
      HasNonzeroPeriod (chartL φ) :=
  FQGNN.Backbone.exists_chartL_rowIndependent_but_periodic

/-- **The thinning does not change the rational span.** SI §S1.5.
For every generator `L` (every `d`, including `d = 0`), `span_ℚ Λ' = span_ℚ M_L`, where
`Λ' = {Lᵀk : var(k) < d}` and `M_L = Lᵀℤⁿ`. -/
theorem thinnedSpectrum_span_eq_spectrumModule {n d : ℕ} (L : Matrix (Fin n) (Fin d) ℝ) :
    Submodule.span ℚ (extinctionSpectrum L) =
      Submodule.span ℚ (spectrumModule L : Set (Fin d → ℝ)) :=
  FQGNN.Backbone.span_extinctionSpectrum_eq L

/-- **`ℚ`-rank bound for the thinned spectrum.** SI §S1.5 ("rational span may have dimension up
to `n`"). `dim_ℚ span_ℚ Λ' ≤ n`. -/
theorem qRank_thinned_le {n d : ℕ} (L : Matrix (Fin n) (Fin d) ℝ) :
    Module.finrank ℚ (Submodule.span ℚ (extinctionSpectrum L)) ≤ n :=
  FQGNN.Backbone.qRank_thinned_le L

/-- **`ℚ`-rank theorem for the thinned spectrum.** SI §S1.5 ("`dim_ℚ = n` exactly under
`ℚ`-independence of the rows of `L`"). `dim_ℚ span_ℚ Λ' = n` iff the rows of `L`
are `ℚ`-linearly independent. -/
theorem qRank_thinned_eq_iff {n d : ℕ} (L : Matrix (Fin n) (Fin d) ℝ) :
    Module.finrank ℚ (Submodule.span ℚ (extinctionSpectrum L)) = n ↔
      LinearIndependent ℚ L.row :=
  FQGNN.Backbone.qRank_thinned_eq_iff L

/-- **Prop. S2.5(iii), limit form.** SI Prop. S2.5(iii), "consequently … forces `‖φ‖_∞ → ∞`".
If along a parameter sequence `φ_k` one fixed minor `Δ_ij(L(φ_k))` (`i < j`) tends to `0`, then
`‖φ_k‖_∞ → ∞`. The hypothesis is stated for a fixed vanishing minor; the SI states it as
`min_I Δ_I → 0`. -/
theorem propS25iii_norm_tendsto_atTop {n : ℕ} (φ : ℕ → ChartParams n) {i j : Fin n}
    (hij : i < j) (h : Tendsto (fun k => minor2 (chartL (φ k)) i j) atTop (𝓝 0)) :
    Tendsto (fun k => ‖φ k‖) atTop atTop :=
  FQGNN.Backbone.tendsto_norm_atTop_of_minor2_tendsto_zero φ hij h

/-- **Sharpened (S2.3): one-sided bounds suffice.** SI Prop. S2.5(iii), eq. (S2.3). If every radius
parameter `ρ_i ≥ −B`, the scale parameter `log s ≥ −B` and every angle parameter `|α_l| ≤ B`, then
every minor `Δ_ij(L(φ))` (`i < j`) is at least `m(B)`. No upper bound on `ρ` or `log s` is needed. -/
theorem propS25iii_interiority_oneSided {n : ℕ} (φ : ChartParams n) {B : ℝ}
    (hρ : ∀ i, -B ≤ φ.1 i) (hs : -B ≤ φ.2.2) (hα : ∀ l, |φ.2.1 l| ≤ B) {i j : Fin n}
    (hij : i < j) : interiorityConst n B ≤ minor2 (chartL φ) i j :=
  FQGNN.Backbone.minor2_chartL_ge_of_oneSided φ hρ hs hα hij

/-- **Prop. S2.5(iii), coordinate case split, subsequence form.** SI Prop. S2.5(iii), "a radius
`ρ_i → −∞`, the scale `log s → −∞`, or … some `α_l → −∞` or some `α_m → +∞`". If one fixed minor `Δ_ij(L(φ_k))` (`i < j`) tends to `0`, then one
fixed coordinate is unbounded in the stated direction along the sequence (it tends to `∓∞` along a
subsequence): some `ρ_i` is unbounded below, or `log s` is unbounded below, or some `α_l` is
unbounded below, or some `α_m` is unbounded above. Convergence of a single coordinate along the
whole sequence does not hold in general: a sequence may alternate between two degeneration
mechanisms. -/
theorem propS25iii_coordinate_case_split {n : ℕ} (φ : ℕ → ChartParams n) {i j : Fin n}
    (hij : i < j) (h : Tendsto (fun k => minor2 (chartL (φ k)) i j) atTop (𝓝 0)) :
    (∃ i, ∀ B : ℝ, ∃ᶠ k in atTop, (φ k).1 i < -B) ∨
      (∀ B : ℝ, ∃ᶠ k in atTop, (φ k).2.2 < -B) ∨
      (∃ l, ∀ B : ℝ, ∃ᶠ k in atTop, (φ k).2.1 l < -B) ∨
      (∃ l, ∀ B : ℝ, ∃ᶠ k in atTop, B < (φ k).2.1 l) :=
  FQGNN.Backbone.chart_coordinate_case_split φ hij h

/-- **Kronecker's theorem for the torus lift, exact form.** SI §S1.1; manuscript eq. (1). The image
of `Φ_L : ℝᵈ → 𝕋ⁿ` is dense in the torus iff the rows of `L` are `ℚ`-linearly independent.
Equidistribution (the Weyl half of the SI sentence) is not formalized. -/
theorem kronecker_density_iff {n d : ℕ} (L : Matrix (Fin n) (Fin d) ℝ) :
    closure (Set.range (torusLift L)) = torus n ↔ LinearIndependent ℚ L.row :=
  FQGNN.Backbone.closure_range_torusLift_eq_torus_iff L

/-- **Kronecker density in the SI's integer-relation form.** SI §S1.1. If no nonzero
`k ∈ ℤⁿ` has `Lᵀk = 0`, then `closure (range Φ_L) = 𝕋ⁿ`. -/
theorem kronecker_density {n d : ℕ} {L : Matrix (Fin n) (Fin d) ℝ}
    (hL : ∀ k : Fin n → ℤ, Lᵀ *ᵥ (fun j => (k j : ℝ)) = 0 → k = 0) :
    closure (Set.range (torusLift L)) = torus n :=
  FQGNN.Backbone.closure_range_torusLift hL

/-- **Density of the ambient module is exactly aperiodicity.** SI §S1.5, the §S1 summary table,
§S5.2 caveat (ii) and §S5.5(a). `M_L = Lᵀℤⁿ` is dense in `ℝᵈ` iff `Φ_L` has no nonzero period, i.e. iff no nonzero
`p ∈ ℝᵈ` has `Lp ∈ ℤⁿ`. -/
theorem spectrumModule_dense_iff_aperiodic {n d : ℕ} (L : Matrix (Fin n) (Fin d) ℝ) :
    Dense (spectrumModule L : Set (Fin d → ℝ)) ↔ ¬ HasNonzeroPeriod L :=
  FQGNN.Backbone.dense_spectrumModule_iff L

/-- **Counterexample: the ambient module need not be dense for `n > d`.** SI §S1.5 ("neither
`n > d`, nor positivity of the minors, nor ℚ-independence of the rows suffices"). The generator `L = [[3,1],[2,1+√2],[1,3]]` lies in `Gr₊(2,3)` (minors `1+3√2`, `8`,
`5−√2`), has `ℚ`-linearly independent rows (so `M_L` has rank `3 > 2`), has the nonzero period
`(1,0)` (`L(1,0)ᵀ = (3,2,1)ᵀ ∈ ℤ³`), and `M_L = Lᵀℤ³` is not dense in `ℝ²`. -/
theorem spectrumModule_not_dense_counterexample :
    InGrPlus2 grPlusExample ∧ LinearIndependent ℚ grPlusExample.row ∧
      HasNonzeroPeriod grPlusExample ∧
      ¬ Dense (spectrumModule grPlusExample : Set (Fin 2 → ℝ)) :=
  ⟨FQGNN.Backbone.grPlusExample_inGrPlus2, FQGNN.Backbone.grPlusExample_rows_linearIndependent,
    FQGNN.Backbone.grPlusExample_hasNonzeroPeriod,
    FQGNN.Backbone.grPlusExample_spectrumModule_not_dense.2.2⟩

end BackboneDensity

/-! ## Readout: the comb, Op-H (eq. (6)), the kernel, Op-M (eqs. (7)-(8)) and the blind spot -/

section ReadoutComb

open Matrix FQGNN.Common FQGNN.Readout

variable {K : ℕ}

/-- **The S4.6 chord identity.** Manuscript Results ("an exact identity we prove"); SI §S4.6,
eq. (S4.10). Assume `a ≠ b`. The time-linear interpolation `((b − t) p_a + (t − a) p_b)/(b − a)` is
exactly the affine chord atom `A(t)` of the readout. -/
theorem chord_identity_S46 {a b : ℝ} (hab : a ≠ b) (pa pb t : ℝ) :
    ((b - t) * pa + (t - a) * pb) / (b - a) = chord a b pa pb t :=
  FQGNN.Readout.timeLinearInterp_eq_chord hab pa pb t

/-- **`Dμ` is the derivative of the Op-D kernel.** Manuscript eqs. (3)-(4), one-variable form. The
kernel `K(δ) = Re Σ_k c_k ĝ_k e^{2πi s ξ_k δ}` has derivative `Dμ(δ)`, the same sum with each term
multiplied by `2πi s ξ_k`. -/
theorem eq4_Dmu_is_derivative (c g : Fin K → ℂ) (ξ : Fin K → ℝ) (s δ : ℝ) :
    HasDerivAt (kernelOpD c g ξ s) (kernelDmu c g ξ s δ) δ :=
  FQGNN.Readout.hasDerivAt_kernelOpD c g ξ s δ

/-- **`Dμ` in `ℝᵈ`, as a directional derivative.** Manuscript eq. (4) with `ξ_k ∈ ℝᵈ`. Along every
direction `e`, the Op-D kernel has directional derivative given by the multiplier
`2πi s ⟨ξ_k, e⟩`. -/
theorem eq4_Dmu_is_directional_derivative {d : ℕ} (c g : Fin K → ℂ) (ξ : Fin K → Fin d → ℝ)
    (s : ℝ) (e δ : Fin d → ℝ) (h : ℝ) :
    HasDerivAt (fun x : ℝ => kernelOpDVec c g ξ s (δ + x • e))
      (kernelDmuVec c g ξ s e (δ + h • e)) h :=
  FQGNN.Readout.hasDerivAt_kernelOpDVec c g ξ s e δ h

/-- **Constraints protect the affine atom.** SI §S4.1. Suppose a comb `r_c` vanishes at two distinct
times and is affine. Then it is identically zero. No hypothesis on the frequencies is needed. -/
theorem S41_affine_comb_with_two_zeros_is_zero (ω : Fin K → ℝ) (c : Fin K ⊕ Fin K → ℝ)
    {τa τb u₀ u₁ : ℝ} (hab : τa ≠ τb) (ha : rcomb ω c τa = 0) (hb : rcomb ω c τb = 0)
    (haff : ∀ τ, rcomb ω c τ = u₀ + u₁ * τ) : ∀ τ, rcomb ω c τ = 0 :=
  FQGNN.Readout.rcomb_eq_zero_of_affine ω c hab ha hb haff

/-- **Direct sum of the affine atom and the comb.** SI §S4.1. Take distinct positive frequencies and a
nondegenerate interval `(l, r)`. If `u₀ + u₁ τ + r_c(τ) = 0` on `(l, r)`, then `u₀ = u₁ = 0` and
`c = 0`. -/
theorem S41_direct_sum {ω : Fin K → ℝ} (hinj : Function.Injective ω) (hpos : ∀ k, 0 < ω k)
    {l r : ℝ} (hlr : l < r) {u₀ u₁ : ℝ} {c : Fin K ⊕ Fin K → ℝ}
    (h : ∀ τ ∈ Set.Ioo l r, u₀ + u₁ * τ + rcomb ω c τ = 0) : u₀ = 0 ∧ u₁ = 0 ∧ c = 0 :=
  FQGNN.Readout.affine_add_rcomb_eqOn_zero hinj hpos hlr h

/-- **Linear independence of the reconstruction basis.** SI §S4.1. Take distinct positive
frequencies and a nondegenerate interval. The functions `{1, τ} ∪ {cos ω_k τ, sin ω_k τ}`, restricted
to that interval, are linearly independent. -/
theorem S41_reconBasis_linearIndependent {ω : Fin K → ℝ} (hinj : Function.Injective ω)
    (hpos : ∀ k, 0 < ω k) {l r : ℝ} (hlr : l < r) :
    LinearIndependent ℝ (reconBasis ω l r) :=
  FQGNN.Readout.linearIndependent_reconBasis hinj hpos hlr

end ReadoutComb

section ReadoutOpH

open Matrix FQGNN.Common FQGNN.Readout

variable {K : ℕ}

/-- **Proposition S4.1 (exact interpolation for every `η` and every metric).** SI Prop. S4.1.
Assume `B` has full row rank and `w > 0`. Then the Op-H-W solution of manuscript
eq. (6) satisfies `B c = d` for every gate value `g` and every null-space input `η`. -/
theorem propS41_exact_interpolation {ι : Type*} [Fintype ι] [DecidableEq ι]
    {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B) {w : Fin K → ℝ} (hw : ∀ k, 0 < w k)
    (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    B *ᵥ opHW B w d g η = d :=
  FQGNN.Readout.mulVec_opHW hB hw d g η

/-- **Proposition S4.1 on the boundary-Hermite system.** SI Prop. S4.1, with SI eqs. (S4.5)-(S4.6).
Assume the boundary-Hermite matrix has full row rank, `a ≠ b`, `w > 0`, and the equilibration
factors are nonzero. Then for every `η` and every `g`, the reconstruction `p̂` interpolates both
anchors (`p̂(a) = p_a`, `p̂(b) = p_b`) and matches every included boundary velocity
(`p̂'(t_e) = v_e`). -/
theorem propS41_hermite_interpolation (ω : Fin K → ℝ) {a b : ℝ} (hab : a ≠ b) (pa pb : ℝ)
    {Rv : ℕ} (te s v : Fin Rv → ℝ) (hs : ∀ e, s e ≠ 0)
    (hB : FullRowRank (hermiteB ω a b te s)) {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) :
    let c := opHW (hermiteB ω a b te s) w (hermiteD a b pa pb s v) g η
    recon ω c a b pa pb a = pa ∧ recon ω c a b pa pb b = pb ∧
      ∀ e, HasDerivAt (recon ω c a b pa pb) (v e) (te e) :=
  FQGNN.Readout.opHW_hermite_interpolation ω hab pa pb te s v hs hB hw g η

/-- **Proposition 1 (affine reproduction), Op-H readout of eq. (6).** Manuscript Prop. 1; SI §S4.5.
The setting:
* the track is affine and constant-velocity, `P t = U₀ + t U₁`, observed in any linear coordinate;
* the `R_v ≥ 1` included boundary velocities are exact (`V_e = U₁`);
* the curvature gate is enabled.

Then the coefficients vanish, `c = 0`, and the reconstruction equals the true track at every `t`.
This holds for every `η`, every metric `w` (positivity not even needed) and every gate parameter,
with no rank hypothesis. -/
theorem prop1_affine_reproduction_opH {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (π : E →ₗ[ℝ] ℝ) {P : ℝ → E} {U₀ U₁ : E} (hP : ∀ t, P t = U₀ + t • U₁)
    {a b : ℝ} (hab : a ≠ b) (ω : Fin K → ℝ) {Rv : ℕ} (hRv : 0 < Rv) (te s : Fin Rv → ℝ)
    {V : Fin Rv → E} (hV : ∀ e, V e = U₁) (w : Fin K → ℝ) (ρ ε : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) :
    let g := gateValue true ρ ε V (chordSlopeVec a b (P a) (P b))
    let c := opHW (hermiteB ω a b te s) w
      (hermiteD a b (π (P a)) (π (P b)) s (fun e => π (V e))) g η
    c = 0 ∧ ∀ t, recon ω c a b (π (P a)) (π (P b)) t = π (P t) :=
  FQGNN.Readout.prop1_affine_reproduction π hP hab ω hRv te s hV w ρ ε η

/-- **Proposition 1 and Theorem S7.5(i), full readout of eqs. (7)-(8).** Manuscript Prop. 1
(exterior self-gating clause); SI Thm S7.5(i). The setting:
* the track is affine and constant-velocity;
* the boundary velocities are exact and the gate is enabled with `R_v ≥ 1`;
* the soft-anchor values are the per-side detrended chord residuals (any linear detrend).

Then the Op-M coefficients vanish, `c = 0`, and the reconstruction equals the true track. This holds
for every precision vector `Λ`, every `η` and every metric `w`, with no rank hypothesis. -/
theorem prop1_affine_reproduction_opM {ο : Type*} [Fintype ο] [DecidableEq ο] {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E]
    (π : E →ₗ[ℝ] ℝ) {X : ℝ → E} {U₀ U₁ : E} (hX : ∀ t, X t = U₀ + t • U₁) {a b : ℝ}
    (hab : a ≠ b) (ω : Fin K → ℝ) {Rv : ℕ} (hRv : 0 < Rv) (te s : Fin Rv → ℝ)
    {Vel : Fin Rv → E} (hVel : ∀ e, Vel e = U₁) (ρg ε : ℝ) (tj : ο → ℝ)
    (detrend : (ο → ℝ) →ₗ[ℝ] (ο → ℝ)) (w : Fin K → ℝ) (lam : ο → ℝ)
    (η : Fin K ⊕ Fin K → ℝ) :
    let g := gateValue true ρg ε Vel (chordSlopeVec a b (X a) (X b))
    let ρt := detrend (fun j => π (X (tj j)) - chord a b (π (X a)) (π (X b)) (tj j))
    let c := opM (hermiteB ω a b te s) w (extV ω (fun j => ctime a b (tj j))) lam ρt
      (hermiteD a b (π (X a)) (π (X b)) s (fun e => π (Vel e))) g η
    c = 0 ∧ ∀ t, recon ω c a b (π (X a)) (π (X b)) t = π (X t) :=
  FQGNN.Readout.thmS75i_chord_exactness π hX hab ω hRv te s hVel ρg ε tj detrend w lam η

/-- **Zero data and a closed gate give zero coefficients.** SI §S4.5. If `d = 0` and
`g = 0`, then `opHW B w 0 0 η = 0` for every `B`, `w` and `η`, with no rank hypothesis. -/
theorem propS45_zero_data_zero_gate {ι : Type*} [Fintype ι] [DecidableEq ι]
    (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    opHW B w 0 0 η = 0 :=
  FQGNN.Readout.opHW_zero_zero B w η

/-- **Metric homogeneity.** SI Prop. S6.1(ii). Scaling the metric `w ↦ α w` (with `α ≠ 0`) is the
same as scaling the gate `g ↦ α g`. The particular part is 0-homogeneous in `w`; the null-space map
is 1-homogeneous. -/
theorem propS61ii_metric_homogeneity {ι : Type*} [Fintype ι] [DecidableEq ι] {α : ℝ}
    (hα : α ≠ 0) (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) :
    opHW B (α • w) d g η = opHW B w d (α * g) η :=
  FQGNN.Readout.opHW_smul_metric hα B w d g η

/-- **Max-normalization makes the solve scale-invariant.** SI Prop. S6.1(iii); manuscript Methods
("max-normalized per gap"). For `α > 0`, the max-normalized metrics of `α w` and of `w` give the same
Op-H-W solution. -/
theorem propS61iii_normalization_invariance {ι : Type*} [Fintype ι] [DecidableEq ι] {α : ℝ}
    (hα : 0 < α) (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) :
    opHW B (normalizeW (α • w)) d g η = opHW B (normalizeW w) d g η :=
  FQGNN.Readout.opHW_normalizeW_smul hα B w d g η

/-- **Row equilibration leaves the solve unchanged.** SI §S4.2. For every
invertible `D`, `opHW (D B) w (D d) g η = opHW B w d g η`. This covers positive diagonal row
equilibration in particular. -/
theorem row_equilibration_invariance {ι : Type*} [Fintype ι] [DecidableEq ι]
    {D : Matrix ι ι ℝ} (hD : IsUnit D.det) (B : Matrix ι (Fin K ⊕ Fin K) ℝ)
    (w : Fin K → ℝ) (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    opHW (D * B) w (D *ᵥ d) g η = opHW B w d g η :=
  FQGNN.Readout.opHW_mul_left hD B w d g η

/-- **Weighted minimum-norm characterization.** SI §S4.3; SI Prop. S6.2 (first half). Assume
`B` has full row rank and `w > 0`. Then `c⋆ = W (BW)⁺ d` is feasible (`B c⋆ = d`). It minimizes the
weighted norm `‖W⁻¹ c‖²` over the feasible set, and it is the only feasible vector that does.
It also equals `W² Bᵀ (B W² Bᵀ)⁻¹ d`. -/
theorem opHW_weighted_min_norm {ι : Type*} [Fintype ι] [DecidableEq ι]
    {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B) {w : Fin K → ℝ} (hw : ∀ k, 0 < w k)
    (d : ι → ℝ) :
    B *ᵥ partOpHW B w d = d ∧
      (∀ c, B *ᵥ c = d → weightedNormSq w (partOpHW B w d) ≤ weightedNormSq w c) ∧
      (∀ c, B *ᵥ c = d → weightedNormSq w c = weightedNormSq w (partOpHW B w d) →
        c = partOpHW B w d) ∧
      partOpHW B w d =
        ((metricW w * metricW w) * Bᵀ * (B * (metricW w * metricW w) * Bᵀ)⁻¹) *ᵥ d :=
  FQGNN.Readout.partOpHW_isWeightedMinNorm hB hw d

end ReadoutOpH

section ReadoutKernel

open Matrix Real FQGNN.Common FQGNN.Readout
open scoped ComplexOrder

/-- **Proposition S5.1 (positive semidefiniteness of the FQ kernel, finite spectrum).** SI
Prop. S5.1, eqs. (S5.2)-(S5.3). Take nonnegative masses `m_k`, frequencies `ξ_k ∈ ℝᵈ` and any scale
`s`. For any points `u_p`, the Gram matrix `G_pq = Σ_k m_k cos(2π s ⟨ξ_k, u_p − u_q⟩)` is positive
semidefinite. -/
theorem propS51_kernel_posSemidef {σ : Type*} [Fintype σ] {d : ℕ} {ι : Type*} [Fintype ι]
    {m : σ → ℝ} (hm : ∀ k, 0 ≤ m k) (ξ : σ → Fin d → ℝ) (s : ℝ) (u : ι → Fin d → ℝ) :
    (fqGram m ξ s u).PosSemidef :=
  FQGNN.Readout.fqGram_posSemidef hm ξ s u

/-- **Proposition S5.1, complex-coefficient form.** SI eq. (S5.3). For nonnegative masses and complex
coefficients `a`, `Σ_{p,q} a_p conj(a_q) K_FQ(u_p, u_q) ≥ 0`. -/
theorem propS51_kernel_complex_nonneg {σ : Type*} [Fintype σ] {d : ℕ} {ι : Type*} [Fintype ι]
    {m : σ → ℝ} (hm : ∀ k, 0 ≤ m k) (ξ : σ → Fin d → ℝ) (s : ℝ) (u : ι → Fin d → ℝ)
    (a : ι → ℂ) :
    0 ≤ ∑ p, ∑ q, a p * (starRingEnd ℂ) (a q) * (fqKernel m ξ s (u p) (u q) : ℂ) :=
  FQGNN.Readout.fqGram_complex_nonneg hm ξ s u a

/-- **Proposition S5.1, sum-of-squares identity.** SI eq. (S5.4), finite spectrum.
`aᵀ G a = Σ_k m_k [(Σ_p a_p cos θ_kp)² + (Σ_p a_p sin θ_kp)²]`, where `θ_kp = 2π s ⟨ξ_k, u_p⟩`. -/
theorem propS51_sum_of_squares {σ : Type*} [Fintype σ] {d : ℕ} {ι : Type*} [Fintype ι]
    (m : σ → ℝ) (ξ : σ → Fin d → ℝ) (s : ℝ) (u : ι → Fin d → ℝ) (a : ι → ℝ) :
    a ⬝ᵥ (fqGram m ξ s u *ᵥ a) =
      ∑ k, m k * ((∑ p, a p * Real.cos (2 * π * s * (ξ k ⬝ᵥ u p))) ^ 2 +
        (∑ p, a p * Real.sin (2 * π * s * (ξ k ⬝ᵥ u p))) ^ 2) :=
  FQGNN.Readout.dotProduct_fqGram_mulVec m ξ s u a

/-- **Profile of the FQ kernel.** SI Prop. S5.1 and §S5.3. The profile
`κ(δ) = Σ_k m_k cos(2π s ⟨ξ_k, δ⟩)` satisfies `κ(0) = Σ_k m_k`, is even, and is continuous. -/
theorem propS51_profile {σ : Type*} [Fintype σ] {d : ℕ} (m : σ → ℝ) (ξ : σ → Fin d → ℝ)
    (s : ℝ) :
    fqProfile m ξ s 0 = ∑ k, m k ∧ (∀ δ, fqProfile m ξ s (-δ) = fqProfile m ξ s δ) ∧
      Continuous (fqProfile m ξ s) :=
  ⟨FQGNN.Readout.fqProfile_zero m ξ s, FQGNN.Readout.fqProfile_neg m ξ s,
    FQGNN.Readout.continuous_fqProfile m ξ s⟩

/-- **Reproducing property of the kriging kernel.** SI Prop. S6.2 (proof). For a nonzero prior
variance vector `v`, consider the section `K_v(t,·)`, with coefficients `(v_k cos ω_k t, v_k sin ω_k t)`.
It reproduces point evaluation in the `v`-weighted coefficient pairing:
`⟪c, K_v(t,·)⟫_v = r_c(t)`. -/
theorem propS62_reproducing {K : ℕ} {v : Fin K → ℝ} (hv : ∀ k, v k ≠ 0) (ω : Fin K → ℝ)
    (c : Fin K ⊕ Fin K → ℝ) (t : ℝ) : vInner v c (krigSection ω v t) = rcomb ω c t :=
  FQGNN.Readout.vInner_krigSection hv ω c t

/-- **Reproducing property for derivative functionals.** SI Prop. S6.2 (proof).
`⟪c, ∂_t K_v(t,·)⟫_v = φ'(t) · c`, which is the derivative `r_c'(t)`. -/
theorem propS62_reproducing_deriv {K : ℕ} {v : Fin K → ℝ} (hv : ∀ k, v k ≠ 0)
    (ω : Fin K → ℝ) (c : Fin K ⊕ Fin K → ℝ) (t : ℝ) :
    vInner v c (krigSectionDeriv ω v t) = dphi ω t ⬝ᵥ c :=
  FQGNN.Readout.vInner_krigSectionDeriv hv ω c t

/-- **Proposition S6.2 (kriging unification).** SI Prop. S6.2; manuscript Methods ("kernel weights
and spectral metric are one object"). Assume `B` has full row rank and `v > 0`. Take the Op-H-W solve
at `η = 0` with metric `w_k = √v_k`, for any gate `g`. It is feasible, it minimizes the RKHS norm
`‖c‖²_v` over the feasible set, it is the only feasible vector that does, and it equals
`V Bᵀ (B V Bᵀ)⁻¹ d`. The statement carries the hypothesis that `B` has full row rank. -/
theorem propS62_kriging_eq_opHW {K : ℕ} {ι : Type*} [Fintype ι] [DecidableEq ι]
    {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B) {v : Fin K → ℝ}
    (hv : ∀ k, 0 < v k) (d : ι → ℝ) (g : ℝ) :
    B *ᵥ opHW B (fun k => √(v k)) d g 0 = d ∧
      (∀ c, B *ᵥ c = d →
        vInner v (opHW B (fun k => √(v k)) d g 0) (opHW B (fun k => √(v k)) d g 0) ≤
          vInner v c c) ∧
      (∀ c, B *ᵥ c = d →
        vInner v c c ≤
          vInner v (opHW B (fun k => √(v k)) d g 0) (opHW B (fun k => √(v k)) d g 0) →
        c = opHW B (fun k => √(v k)) d g 0) ∧
      opHW B (fun k => √(v k)) d g 0 = (metricW v * Bᵀ * (B * metricW v * Bᵀ)⁻¹) *ᵥ d :=
  FQGNN.Readout.kriging_eq_opHW hB hv d g

end ReadoutKernel

section ReadoutOpM

open Matrix FQGNN.Common FQGNN.Readout

variable {K : ℕ} {ι ο : Type*} [Fintype ι] [DecidableEq ι] [Fintype ο] [DecidableEq ο]

/-- **The soft block is always invertible.** SI Thm S7.2. For every `B`, `w`,
exterior matrix `V` and precision vector `λ`, the matrix `I + G̃` of manuscript eq. (8) is invertible.
No hypotheses: the soft block adds no rank condition. -/
theorem thmS72_soft_block_invertible (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam : ο → ℝ) : IsUnit (1 + opMGram B w V lam).det :=
  FQGNN.Readout.isUnit_det_one_add_opMGram B w V lam

/-- **Proposition S7.1 / Theorem S7.2 (Op-M is the unique constrained minimizer).** Manuscript
eqs. (7)-(8); SI Prop. S7.1, Thm S7.2. Assume `B` has full row rank, `w > 0`
and `Λ ⪰ 0`. Then the closed form (8) is feasible (`B c = d`). It minimizes the objective (7),
`J(c) = ‖Ŵ⁻¹c − gη‖² + Σ_j λ_j (φ(τ_j)·c − ρ̃_j)²`, over `{B c = d}`, and it is the only feasible
vector that does. -/
theorem propS71_opM_unique_minimizer {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B)
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (V : Matrix ο (Fin K ⊕ Fin K) ℝ) {lam : ο → ℝ}
    (hlam : ∀ j, 0 ≤ lam j) (ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    B *ᵥ opM B w V lam ρ d g η = d ∧
      (∀ c, B *ᵥ c = d →
        opMObj w V lam ρ g η (opM B w V lam ρ d g η) ≤ opMObj w V lam ρ g η c) ∧
      (∀ c, B *ᵥ c = d →
        opMObj w V lam ρ g η c = opMObj w V lam ρ g η (opM B w V lam ρ d g η) →
          c = opM B w V lam ρ d g η) :=
  FQGNN.Readout.opM_isUniqueMinimizer hB hw V hlam ρ d g η

/-- **Theorem S7.3 (exactness survives every dial).** SI Thm S7.3. Assume `B`
has full row rank and `w > 0`. Then `B c = d` for every precision vector `Λ`, every soft-anchor
value `ρ̃`, every gate `g` and every `η`. -/
theorem thmS73_exactness_every_dial {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B)
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ)
    (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) : B *ᵥ opM B w V lam ρ d g η = d :=
  FQGNN.Readout.mulVec_opM hB hw V lam ρ d g η

/-- **Theorem S7.3 on the boundary-Hermite system.** SI Thm S7.3; manuscript Methods. Under the
hypotheses of Prop. S4.1, the Op-M reconstruction interpolates both anchors and matches every
included boundary velocity. This holds for every exterior matrix, every `Λ`, `ρ̃`, `g` and `η`. -/
theorem thmS73_hermite_interpolation (ω : Fin K → ℝ) {a b : ℝ} (hab : a ≠ b) (pa pb : ℝ)
    {Rv : ℕ} (te s v : Fin Rv → ℝ) (hs : ∀ e, s e ≠ 0)
    (hB : FullRowRank (hermiteB ω a b te s)) {w : Fin K → ℝ} (hw : ∀ k, 0 < w k)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    let c := opM (hermiteB ω a b te s) w V lam ρ (hermiteD a b pa pb s v) g η
    recon ω c a b pa pb a = pa ∧ recon ω c a b pa pb b = pb ∧
      ∀ e, HasDerivAt (recon ω c a b pa pb) (v e) (te e) :=
  FQGNN.Readout.opM_hermite_interpolation ω hab pa pb te s v hs hB hw V lam ρ g η

/-- **Corollary S7.4 (endpoint annihilation).** SI Cor. S7.4; manuscript Methods.
On the boundary-Hermite system, the exterior contribution `δ = r_{Ŵy}` has zero value at
both anchors and zero slope at every boundary whose velocity row is included. -/
theorem corS74_endpoint_annihilation (ω : Fin K → ℝ) (a b pa pb : ℝ) {Rv : ℕ}
    (te s v : Fin Rv → ℝ) (hs : ∀ e, s e ≠ 0) (hB : FullRowRank (hermiteB ω a b te s))
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ)
    (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    let δ := rcomb ω (metricW w *ᵥ
      opMSoft (hermiteB ω a b te s) w V lam ρ (hermiteD a b pa pb s v) g η)
    δ (ctime a b a) = 0 ∧ δ (ctime a b b) = 0 ∧ ∀ e, HasDerivAt δ 0 (ctime a b (te e)) :=
  FQGNN.Readout.exterior_endpoint_annihilation ω a b pa pb te s v hs hB hw V lam ρ g η

/-- **Proposition 3 (exact degeneration), `Λ = 0`.** Manuscript Prop. 3; SI §S7.3. With `Λ = 0`, the
Op-M readout equals the Op-H-W readout of eq. (6) exactly (`y = 0`). This holds for every `B`, `w`,
`V`, `ρ̃`, `d`, `g` and `η`, with no hypotheses. -/
theorem prop3_zero_precision (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) : opM B w V 0 ρ d g η = opHW B w d g η :=
  FQGNN.Readout.opM_zero_prec B w V ρ d g η

/-- **Proposition 3 (exact degeneration), both dials.** Manuscript Prop. 3; SI §S7.3.
Let a metric law satisfy `metricLaw 0 = w`. Then at `(γ, Λ) = (0, 0)` the Op-M readout coincides with
the Op-H-W readout of eq. (6) at metric `w`. -/
theorem prop3_exact_degeneration (metricLaw : ℝ → Fin K → ℝ) {w : Fin K → ℝ}
    (hlaw : metricLaw 0 = w) (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (V : Matrix ο (Fin K ⊕ Fin K) ℝ)
    (ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    opM B (metricLaw 0) V 0 ρ d g η = opHW B w d g η :=
  FQGNN.Readout.prop3_exact_degeneration metricLaw hlaw B V ρ d g η

/-- **Proposition 3 with the deployed geometric metric law.** Manuscript Prop. 3; SI eq. (S7.3).
Use the geometric interpolation law, with the clamp acting on the `γ`-term, any band, any clamp
and any measured profile. At `γ = 0` it returns the base metric `w > 0`, and the Op-M readout at
`Λ = 0` equals the Op-H-W readout. -/
theorem prop3_exact_degeneration_geom (band : Finset (Fin K)) (clampC : ℝ → ℝ) {w : Fin K → ℝ}
    (hw : ∀ k, 0 < w k) (Ihat : Fin K → ℝ) (B : Matrix ι (Fin K ⊕ Fin K) ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    opM B (metricLawGeom band clampC w Ihat 0) V 0 ρ d g η = opHW B w d g η :=
  FQGNN.Readout.prop3_exact_degeneration_geom band clampC hw Ihat B V ρ d g η

/-- **Influence law, coefficient form.** Op-M closed form, manuscript eq. (8), SI eq. (S7.2);
SI §S7.3. The deformation caused by the soft
block is exactly `c(Λ) − c(0) = Ŵ y`. No hypotheses. -/
theorem opM_soft_deformation (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) :
    opM B w V lam ρ d g η - opM B w V 0 ρ d g η = metricW w *ᵥ opMSoft B w V lam ρ d g η :=
  FQGNN.Readout.opM_sub_opM_zero_prec B w V lam ρ d g η

/-- **Influence law, prediction form.** Op-M closed form, SI eq. (S7.2). At every query time `τ_q`, the change
in the prediction is `δp̂(τ_q) = Σ_j K̃(τ_q, τ_j) β_j`. Here `K̃(τ,σ) = φ(τ)ᵀ Ŵ P Ŵ φ(σ)` is the
boundary-conditioned kernel and `β = Λ^{1/2}(I + G̃)⁻¹Λ^{1/2} r₀` are the shrunk dual coefficients.
This is exact, with no hypotheses. -/
theorem opM_influence_law (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ) (ω : Fin K → ℝ)
    (τo : ο → ℝ) (lam ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) (τq : ℝ) :
    rcomb ω (opM B w (extV ω τo) lam ρ d g η) τq - rcomb ω (opM B w (extV ω τo) 0 ρ d g η) τq =
      ∑ j, bcKernel B w ω τq (τo j) * opMDual B w (extV ω τo) lam ρ d g η j :=
  FQGNN.Readout.influence_law B w ω τo lam ρ d g η τq

/-- **The boundary-conditioned kernel vanishes at value rows.** SI Cor. S7.4, kernel form. Assume `B`
has full row rank and `w > 0`. If `φ(τ_e)` is a row of `B`, then `K̃(τ_e, σ) = 0` for every `σ`. -/
theorem opM_bcKernel_value_row {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B)
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (ω : Fin K → ℝ) {i : ι} {τe : ℝ}
    (hrow : B i = phi ω τe) (σ : ℝ) : bcKernel B w ω τe σ = 0 :=
  FQGNN.Readout.bcKernel_eq_zero_of_valueRow hB hw ω hrow σ

/-- **The boundary-conditioned kernel has zero slope at derivative rows.** SI Cor. S7.4, kernel form.
Assume `B` has full row rank and `w > 0`. If `s φ'(τ_e)` (with `s ≠ 0`) is a row of `B`, then
`∂_τ K̃(τ, σ)` vanishes at `τ = τ_e` for every `σ`. -/
theorem opM_bcKernel_deriv_row {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B)
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (ω : Fin K → ℝ) {i : ι} {τe s : ℝ} (hs : s ≠ 0)
    (hrow : B i = s • dphi ω τe) (σ : ℝ) :
    HasDerivAt (fun τ => bcKernel B w ω τ σ) 0 τe :=
  FQGNN.Readout.hasDerivAt_bcKernel_of_derivRow hB hw ω hs hrow σ

/-- **Gauge invariance.** Op-M closed form, SI eq. (S7.2). For `α > 0`, the substitution
`(w, Λ, η) ↦ (α w, Λ/α², η/α)` leaves the Op-M solution unchanged. No other hypotheses. -/
theorem opM_gauge_invariance {α : ℝ} (hα : 0 < α) (B : Matrix ι (Fin K ⊕ Fin K) ℝ)
    (w : Fin K → ℝ) (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) :
    opM B (α • w) V (fun j => lam j / α ^ 2) ρ d g (α⁻¹ • η) = opM B w V lam ρ d g η :=
  FQGNN.Readout.opM_gauge hα B w V lam ρ d g η

/-- **Single soft anchor response (Sherman-Morrison).** Op-M closed form, SI eq. (S7.2). With one soft
anchor at `τ₁` and precision `λ ≥ 0`, shifting its value by `δ` changes the prediction at `τ_q` by
exactly `λ K̃(τ_q, τ₁) / (1 + λ K̃(τ₁, τ₁)) · δ`. -/
theorem opM_single_anchor_response (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w ω : Fin K → ℝ)
    (τ1 : ℝ) {lam : ℝ} (hlam : 0 ≤ lam) (ρ : Fin 1 → ℝ) (δ : ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) (τq : ℝ) :
    rcomb ω (opM B w (extV ω fun _ => τ1) (fun _ => lam) (ρ + fun _ => δ) d g η) τq -
        rcomb ω (opM B w (extV ω fun _ => τ1) (fun _ => lam) ρ d g η) τq =
      lam * bcKernel B w ω τq τ1 / (1 + lam * bcKernel B w ω τ1 τ1) * δ :=
  FQGNN.Readout.singleAnchor_response B w ω τ1 hlam ρ δ d g η τq

/-- **Theorem S7.5(ii), "if" direction (conservative gating).** SI Thm S7.5(ii).
On a locally affine gap (`d = 0`, `g = 0`), gated precisions `0 · Λ` give `c = 0`, for every exterior
and every `ρ̃`. -/
theorem thmS75ii_if_gated (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    opM B w V ((0 : ℝ) • lam) ρ 0 0 η = 0 :=
  FQGNN.Readout.thmS75ii_gated B w V lam ρ η

/-- **Theorem S7.5(ii), converse.** SI Thm S7.5(ii) ("Conversely, with `B` of full row rank …").
Assume `B` has full row rank, `w > 0` and `Λ ⪰ 0`. On a locally
affine gap (`d = 0`, `g = 0`), `c = 0` holds exactly when `P Ṽᵀ Λ ρ̃ = 0`. Here `P` projects onto the
null space of the whitened hard block. -/
theorem thmS75ii_converse_iff {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B)
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (V : Matrix ο (Fin K ⊕ Fin K) ℝ) {lam : ο → ℝ}
    (hlam : ∀ j, 0 ≤ lam j) (ρ : ο → ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    opM B w V lam ρ 0 0 η = 0 ↔
      (proj (B * metricW w) * (V * metricW w)ᵀ * diagonal lam) *ᵥ ρ = 0 :=
  FQGNN.Readout.thmS75ii_converse_iff hB hw V hlam ρ η

/-- **Theorem S7.5(ii), converse under the genericity condition.** SI Thm S7.5(ii). Assume also
that `P Ṽᵀ` is injective. Then `c = 0` iff `λ_j ρ̃_j = 0` for every anchor `j`:
the gated precisions vanish at every anchor whose residual is nonzero. -/
theorem thmS75ii_converse_of_injective {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B)
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (V : Matrix ο (Fin K ⊕ Fin K) ℝ) {lam : ο → ℝ}
    (hlam : ∀ j, 0 ≤ lam j) (ρ : ο → ℝ) (η : Fin K ⊕ Fin K → ℝ)
    (hinj : ∀ x : ο → ℝ, (proj (B * metricW w) * (V * metricW w)ᵀ) *ᵥ x = 0 → x = 0) :
    opM B w V lam ρ 0 0 η = 0 ↔ ∀ j, lam j * ρ j = 0 :=
  FQGNN.Readout.thmS75ii_converse_of_injective hB hw V hlam ρ η hinj

/-- **Counterexample to the unqualified converse of SI Thm S7.5(ii).** SI Thm S7.5(ii) ("the
unqualified converse fails"). There is a full-row-rank hard block `B` whose rows span the
coefficient space (so `P = 0`), an exterior row, a precision `λ ≠ 0` and a residual `ρ̃ ≠ 0` for
which the locally affine Op-M solution is still `c = 0`. -/
theorem thmS75ii_only_if_counterexample :
    ∃ (B : Matrix (Fin 1 ⊕ Fin 1) (Fin 1 ⊕ Fin 1) ℝ) (V : Matrix (Fin 1) (Fin 1 ⊕ Fin 1) ℝ)
      (lam ρ : Fin 1 → ℝ), FullRowRank B ∧ (∀ j, 0 ≤ lam j) ∧ (∀ j, lam j ≠ 0) ∧
        (∀ j, ρ j ≠ 0) ∧ opM B (fun _ => 1) V lam ρ 0 0 0 = 0 :=
  FQGNN.Readout.thmS75ii_only_if_fails

end ReadoutOpM

section ReadoutBlindSpot

open FQGNN.Readout

/-- **Proposition S7.6 (blind-spot trade), first half.** SI Prop. S7.6. Let a
readout `O` return the chord on every jet-straight gap, whatever the exterior data. Then `O` is
constant in the exterior observations on every jet-straight gap. -/
theorem propS76_chordExact_imp_exteriorBlind {Jet Ext Out : Type*} {O : Jet → Ext → Out}
    {JetStraight : Jet → Prop} {chordOf : Jet → Out} (h : ChordExact O JetStraight chordOf) :
    ExteriorBlind O JetStraight :=
  FQGNN.Readout.exteriorBlind_of_chordExact h

/-- **Proposition S7.6, converse.** SI Prop. S7.6 ("conversely"). Suppose a readout responds to the
exterior (two exteriors give different outputs) at some jet-straight jet. Then it is not chord-exact. -/
theorem propS76_responds_imp_not_chordExact {Jet Ext Out : Type*} {O : Jet → Ext → Out}
    {JetStraight : Jet → Prop} {chordOf : Jet → Out} {j : Jet} (hj : JetStraight j)
    (hr : RespondsAt O j) : ¬ ChordExact O JetStraight chordOf :=
  FQGNN.Readout.not_chordExact_of_respondsAt hj hr

/-- **Proposition S7.6, concrete parallel-tangent S-curve.** SI Prop. S7.6. On the
unit gap, a chord-exact readout sees only the boundary 1-jet. Given the jet of the S-curve track
`α + β t + ε (sin 2πt − ½ sin 4πt)` with `ε ≠ 0`, it returns the straight line `α + β t` for every
exterior, which differs from the true track. -/
theorem propS76_misses_sCurve {Ext : Type*} {O : Jet1 → Ext → (ℝ → ℝ)}
    (hO : ChordExact O Jet1.Straight Jet1.chord) (α β : ℝ) {ε : ℝ} (hε : ε ≠ 0) (e : Ext) :
    O (jetOf (sTrack α β ε)) e = lineTrack α β ∧ O (jetOf (sTrack α β ε)) e ≠ sTrack α β ε :=
  FQGNN.Readout.chordExact_misses_sCurve hO α β hε e

/-- **The blind spot is a trade, not an information-theoretic limit.** SI §S7.2, remark after Prop. S7.6. An
exterior-aware readout that returns its exterior input is exact on both the S-curve track and the
line, and it is not chord-exact. -/
theorem propS76_exterior_oracle_trade (α β : ℝ) {ε : ℝ} (hε : ε ≠ 0) :
    (fun (_ : Jet1) (e : ℝ → ℝ) => e) (jetOf (sTrack α β ε)) (sTrack α β ε) = sTrack α β ε ∧
      (fun (_ : Jet1) (e : ℝ → ℝ) => e) (jetOf (lineTrack α β)) (lineTrack α β) =
        lineTrack α β ∧
      ¬ ChordExact (fun (_ : Jet1) (e : ℝ → ℝ) => e) Jet1.Straight Jet1.chord :=
  FQGNN.Readout.exterior_oracle_trade α β hε

end ReadoutBlindSpot


/-! ## The genuine Moore–Penrose inverse and the rank-free readout statements -/

section MoorePenrose

open Matrix FQGNN.Common

variable {ι κ : Type*} [Fintype ι] [Fintype κ]

/-- **Uniqueness of the Moore–Penrose inverse.** SI §S4.4 (proof of Prop. S4.1, "the Penrose
identity"). Two matrices satisfying the four Penrose equations `BXB = B`, `XBX = X`,
`(BX)ᵀ = BX`, `(XB)ᵀ = XB` for the same real `B` are equal. No rank hypothesis. -/
theorem moorePenrose_unique {B : Matrix ι κ ℝ} {X Y : Matrix κ ι ℝ} (hX : IsMoorePenrose B X)
    (hY : IsMoorePenrose B Y) : X = Y :=
  hX.unique hY

/-- **Existence of the Moore–Penrose inverse.** SI §S4.4, SI eq. (S4.7) and manuscript eqs. (6), (8), which
use `B⁺` with no rank hypothesis. Every real matrix `B` has a matrix `X` satisfying the four Penrose
equations. -/
theorem moorePenrose_exists (B : Matrix ι κ ℝ) : ∃ X : Matrix κ ι ℝ, IsMoorePenrose B X :=
  FQGNN.Common.exists_isMoorePenrose B

/-- **Characterization of `mpinv`.** SI §S4.4. `X` satisfies the four Penrose equations for `B` iff
`X = mpinv B`, the Moore–Penrose inverse of `B` (existence and uniqueness together). -/
theorem moorePenrose_iff_eq_mpinv {B : Matrix ι κ ℝ} {X : Matrix κ ι ℝ} :
    IsMoorePenrose B X ↔ X = mpinv B :=
  FQGNN.Common.isMoorePenrose_iff_eq_mpinv

/-- **Agreement with the right-inverse model.** SI §S4.4; manuscript eqs. (6), (8). Under full row
rank, `mpinv B = Bᵀ(BBᵀ)⁻¹ = pinvR B`.
Every statement made with `pinvR` under `FullRowRank B` is therefore a statement about the genuine
Moore–Penrose inverse. -/
theorem mpinv_eq_pinvR_of_fullRowRank [DecidableEq ι] {B : Matrix ι κ ℝ} (h : FullRowRank B) :
    mpinv B = pinvR B :=
  FQGNN.Common.mpinv_eq_pinvR h

/-- **Rank-free feasibility criterion.** SI Prop. S4.1 ("`d ∈ range(B)`"). For every real `B` and
every `d`, `B (B⁺ d) = d` iff `d ∈ range B`. -/
theorem mpinv_feasible_iff_mem_range (B : Matrix ι κ ℝ) (d : ι → ℝ) :
    B *ᵥ (mpinv B *ᵥ d) = d ↔ d ∈ LinearMap.range (Matrix.mulVecLin B) :=
  FQGNN.Common.mulVec_mpinv_mulVec_eq_iff B d

/-- **Rank-free minimum-norm characterization.** SI §S4.3, with no rank hypothesis.
For every consistent `d ∈ range B`, `B⁺ d` is feasible, has least squared norm on `{u | B u = d}`,
and is the only feasible vector attaining it. -/
theorem mpinv_min_norm {B : Matrix ι κ ℝ} {d : ι → ℝ}
    (hd : d ∈ LinearMap.range (Matrix.mulVecLin B)) :
    B *ᵥ (mpinv B *ᵥ d) = d ∧
      (∀ u, B *ᵥ u = d → (mpinv B *ᵥ d) ⬝ᵥ (mpinv B *ᵥ d) ≤ u ⬝ᵥ u) ∧
      (∀ u, B *ᵥ u = d → u ⬝ᵥ u = (mpinv B *ᵥ d) ⬝ᵥ (mpinv B *ᵥ d) → u = mpinv B *ᵥ d) :=
  FQGNN.Common.mpinv_mulVec_isMinNorm hd

/-- **Null-space annihilation with no rank hypothesis.** SI §S4.4 ("`X(I − X⁺X) = 0`
unconditionally (no rank hypothesis)"). For every real `B`, every square `W` (invertibility is not needed) and every `η`, with `X = BW` and the genuine Moore–Penrose
inverse: `B (W ((I − X⁺X) η)) = 0`. -/
theorem S44_null_space_annihilation [DecidableEq κ] (B : Matrix ι κ ℝ) (W : Matrix κ κ ℝ)
    (η : κ → ℝ) : B *ᵥ (W *ᵥ ((1 - mpinv (B * W) * (B * W)) *ᵥ η)) = 0 :=
  FQGNN.Readout.mulVec_mul_one_sub_mpinv_mulVec B W η

end MoorePenrose

section ReadoutMP

open Matrix FQGNN.Common FQGNN.Readout

variable {K : ℕ} {ι ο : Type*} [Fintype ι] [Fintype ο] [DecidableEq ο]

/-- **Op-H-W with the Moore–Penrose inverse agrees with the right-inverse model.** Manuscript
eq. (6); SI eq. (S4.7).
For full row rank `B` and `w > 0`, `opHW_mp B w d g η = opHW B w d g η`. -/
theorem opHW_mp_eq_opHW_of_fullRowRank [DecidableEq ι] {B : Matrix ι (Fin K ⊕ Fin K) ℝ}
    (hB : FullRowRank B) {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) : opHW_mp B w d g η = opHW B w d g η :=
  FQGNN.Readout.opHW_mp_eq_opHW hB hw d g η

/-- **Proposition S4.1, rank-free.** SI Prop. S4.1 and its proof; manuscript eq. (6).
With the genuine Moore–Penrose inverse, for every consistent right-hand side `d ∈ range B` (which
full row rank guarantees), every metric `w > 0`, every gate `g` and every `η`, the Op-H-W
coefficients satisfy `B c = d`. No rank hypothesis. -/
theorem propS41_exact_interpolation_mp {B : Matrix ι (Fin K ⊕ Fin K) ℝ} {w : Fin K → ℝ}
    (hw : ∀ k, 0 < w k) {d : ι → ℝ} (hd : d ∈ LinearMap.range (Matrix.mulVecLin B)) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) : B *ᵥ opHW_mp B w d g η = d :=
  FQGNN.Readout.mulVec_opHW_mp hw hd g η

/-- **Proposition S4.1, sharp form.** SI Prop. S4.1. For `w > 0`, exact interpolation holds iff the
boundary system is consistent: `B (opHW_mp B w d g η) = d ↔ d ∈ range B`, for every `g` and `η`. -/
theorem propS41_exact_interpolation_mp_iff {B : Matrix ι (Fin K ⊕ Fin K) ℝ} {w : Fin K → ℝ}
    (hw : ∀ k, 0 < w k) (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    B *ᵥ opHW_mp B w d g η = d ↔ d ∈ LinearMap.range (Matrix.mulVecLin B) :=
  FQGNN.Readout.mulVec_opHW_mp_eq_iff hw d g η

/-- **Proposition S4.1 on the boundary-Hermite system, rank-free.** SI Prop. S4.1 with SI
eqs. (S4.5)-(S4.6). Assume the boundary-Hermite system is consistent (automatic under full row rank),
`a ≠ b`, `w > 0` and the equilibration factors are nonzero. Then for every `η` and `g` the
reconstruction interpolates both anchors and matches every included boundary velocity. -/
theorem propS41_hermite_interpolation_mp (ω : Fin K → ℝ) {a b : ℝ} (hab : a ≠ b) (pa pb : ℝ)
    {Rv : ℕ} (te s v : Fin Rv → ℝ) (hs : ∀ e, s e ≠ 0)
    (hd : hermiteD a b pa pb s v ∈ LinearMap.range (Matrix.mulVecLin (hermiteB ω a b te s)))
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    let c := opHW_mp (hermiteB ω a b te s) w (hermiteD a b pa pb s v) g η
    recon ω c a b pa pb a = pa ∧ recon ω c a b pa pb b = pb ∧
      ∀ e, HasDerivAt (recon ω c a b pa pb) (v e) (te e) :=
  FQGNN.Readout.opHW_mp_hermite_interpolation ω hab pa pb te s v hs hd hw g η

/-- **Proposition S6.1(i), "unconditionally when `d = 0`".** SI Prop. S6.1(i). With the
genuine Moore–Penrose inverse, `B c(w) = 0` for `d = 0`, for every `B`, every `w` (positivity is not
needed), every gate `g` and every `η`. -/
theorem propS61i_zero_data_mp (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) : B *ᵥ opHW_mp B w 0 g η = 0 :=
  FQGNN.Readout.mulVec_opHW_mp_zero B w g η

/-- **Proposition 1 (affine reproduction), Op-H readout with the Moore–Penrose inverse.** Manuscript Prop. 1;
SI §S4.5. Same setting as `prop1_affine_reproduction_opH`: an affine constant-velocity track, exact
included boundary velocities (`R_v ≥ 1`) and the gate enabled. Then `c = 0` and the reconstruction
equals the true track, for every `η`, every `w` and every `B`. -/
theorem prop1_affine_reproduction_opH_mp {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (π : E →ₗ[ℝ] ℝ) {P : ℝ → E} {U₀ U₁ : E} (hP : ∀ t, P t = U₀ + t • U₁)
    {a b : ℝ} (hab : a ≠ b) (ω : Fin K → ℝ) {Rv : ℕ} (hRv : 0 < Rv) (te s : Fin Rv → ℝ)
    {V : Fin Rv → E} (hV : ∀ e, V e = U₁) (w : Fin K → ℝ) (ρ ε : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) :
    let g := gateValue true ρ ε V (chordSlopeVec a b (P a) (P b))
    let c := opHW_mp (hermiteB ω a b te s) w
      (hermiteD a b (π (P a)) (π (P b)) s (fun e => π (V e))) g η
    c = 0 ∧ ∀ t, recon ω c a b (π (P a)) (π (P b)) t = π (P t) :=
  FQGNN.Readout.prop1_affine_reproduction_mp π hP hab ω hRv te s hV w ρ ε η

/-- **Op-M with the Moore–Penrose inverse agrees with the right-inverse model.** Manuscript eq. (8);
SI eq. (S7.2).
For full row rank `B` and `ŵ > 0`, `opM_mp = opM`. -/
theorem opM_mp_eq_opM_of_fullRowRank [DecidableEq ι] {B : Matrix ι (Fin K ⊕ Fin K) ℝ}
    (hB : FullRowRank B) {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (V : Matrix ο (Fin K ⊕ Fin K) ℝ)
    (lam ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    opM_mp B w V lam ρ d g η = opM B w V lam ρ d g η :=
  FQGNN.Readout.opM_mp_eq_opM hB hw V lam ρ d g η

/-- **Theorem S7.3 (exactness survives every dial), rank-free.** SI Thm S7.3. With
the genuine Moore–Penrose inverse, for every consistent `d ∈ range B` and every metric `ŵ > 0`,
`B c = d` for every exterior matrix `V`, precision vector `Λ`, soft-anchor value `ρ̃`, gate `g` and
`η`. No rank hypothesis. -/
theorem thmS73_exactness_every_dial_mp {B : Matrix ι (Fin K ⊕ Fin K) ℝ} {w : Fin K → ℝ}
    (hw : ∀ k, 0 < w k) {d : ι → ℝ} (hd : d ∈ LinearMap.range (Matrix.mulVecLin B))
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    B *ᵥ opM_mp B w V lam ρ d g η = d :=
  FQGNN.Readout.mulVec_opM_mp hw hd V lam ρ g η

/-- **Theorem S7.3, sharp form.** SI Thm S7.3. For `ŵ > 0`,
`B (opM_mp …) = d ↔ d ∈ range B`, for every `V`, `Λ`, `ρ̃`, `g` and `η`. -/
theorem thmS73_exactness_mp_iff {B : Matrix ι (Fin K ⊕ Fin K) ℝ} {w : Fin K → ℝ}
    (hw : ∀ k, 0 < w k) (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) :
    B *ᵥ opM_mp B w V lam ρ d g η = d ↔ d ∈ LinearMap.range (Matrix.mulVecLin B) :=
  FQGNN.Readout.mulVec_opM_mp_eq_iff hw V lam ρ d g η

/-- **Theorem S7.3 on the boundary-Hermite system, rank-free.** SI Thm S7.3; manuscript Methods. If the
boundary-Hermite system is consistent, `a ≠ b`, `ŵ > 0` and the equilibration factors are nonzero,
the Op-M reconstruction interpolates both anchors and matches every included boundary velocity, for
every `V`, `Λ`, `ρ̃`, `g` and `η`. -/
theorem thmS73_hermite_interpolation_mp (ω : Fin K → ℝ) {a b : ℝ} (hab : a ≠ b) (pa pb : ℝ)
    {Rv : ℕ} (te s v : Fin Rv → ℝ) (hs : ∀ e, s e ≠ 0)
    (hd : hermiteD a b pa pb s v ∈ LinearMap.range (Matrix.mulVecLin (hermiteB ω a b te s)))
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ)
    (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    let c := opM_mp (hermiteB ω a b te s) w V lam ρ (hermiteD a b pa pb s v) g η
    recon ω c a b pa pb a = pa ∧ recon ω c a b pa pb b = pb ∧
      ∀ e, HasDerivAt (recon ω c a b pa pb) (v e) (te e) :=
  FQGNN.Readout.opM_mp_hermite_interpolation ω hab pa pb te s v hs hd hw V lam ρ g η

/-- **Corollary S7.4, algebraic form, no hypothesis.** SI Cor. S7.4. With the
genuine Moore–Penrose inverse, the exterior contribution `Ŵ y` lies in the null space of the hard
constraints, `B (Ŵ y) = 0`, for every `B`, `ŵ`, `V`, `Λ`, `ρ̃`, `d`, `g` and `η`. -/
theorem corS74_exterior_in_null_space_mp (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) : B *ᵥ (metricW w *ᵥ opMSoft_mp B w V lam ρ d g η) = 0 :=
  FQGNN.Readout.mulVec_metricW_opMSoft_mp B w V lam ρ d g η

/-- **Corollary S7.4 (endpoint annihilation), no rank and no metric hypothesis.** SI Cor. S7.4.
On the boundary-Hermite system with nonzero equilibration factors, the exterior
deformation `δ = r_{Ŵy}` (genuine Moore–Penrose inverse) vanishes at both anchors and has zero slope
at every included velocity-row time, for every `ŵ`, `V`, `Λ`, `ρ̃`, `g` and `η`. -/
theorem corS74_endpoint_annihilation_mp (ω : Fin K → ℝ) (a b pa pb : ℝ) {Rv : ℕ}
    (te s v : Fin Rv → ℝ) (hs : ∀ e, s e ≠ 0) (w : Fin K → ℝ) (V : Matrix ο (Fin K ⊕ Fin K) ℝ)
    (lam ρ : ο → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    let δ := rcomb ω (metricW w *ᵥ
      opMSoft_mp (hermiteB ω a b te s) w V lam ρ (hermiteD a b pa pb s v) g η)
    δ (ctime a b a) = 0 ∧ δ (ctime a b b) = 0 ∧ ∀ e, HasDerivAt δ 0 (ctime a b (te e)) :=
  FQGNN.Readout.exterior_endpoint_annihilation_mp ω a b pa pb te s v hs w V lam ρ g η

/-- **Proposition 3, `Λ = 0`, with the Moore–Penrose inverse.** Manuscript Prop. 3; SI §S7.3. At `Λ = 0` the
Op-M readout equals the Op-H-W readout, `opM_mp B w V 0 ρ d g η = opHW_mp B w d g η`, with no
hypotheses. -/
theorem prop3_zero_precision_mp (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) : opM_mp B w V 0 ρ d g η = opHW_mp B w d g η :=
  FQGNN.Readout.opM_mp_zero_prec B w V ρ d g η

end ReadoutMP

section ReadoutGenericity

open Matrix Real FQGNN.Common FQGNN.Readout

variable {K : ℕ}

/-- **Genericity of the rank hypothesis at `R = 2`.** SI §S4.4, "Genericity of the rank hypothesis".
The value block with rows `φ(−h/2)`, `φ(h/2)` fails full row rank iff every kept phase
`ω_k h/2` lies in `πℤ`, or every kept phase lies in `π/2 + πℤ`. -/
theorem S44_value_block_rank_deficient_iff (ω : Fin K → ℝ) (h : ℝ) :
    ¬ FullRowRank (valueBlock ω h) ↔
      (∀ k, ∃ m : ℤ, ω k * (h / 2) = m * π) ∨
        (∀ k, ∃ m : ℤ, ω k * (h / 2) = π / 2 + m * π) :=
  FQGNN.Readout.not_fullRowRank_valueBlock_iff ω h

/-- **Generic full row rank at `R = 2`.** SI §S4.4. If one kept phase `ω_k h/2` is not an
integer multiple of `π/2`, the value block has full row rank. -/
theorem S44_value_block_fullRowRank (ω : Fin K → ℝ) (h : ℝ)
    (hk : ∃ k, ∀ m : ℤ, ω k * (h / 2) ≠ m * (π / 2)) : FullRowRank (valueBlock ω h) :=
  FQGNN.Readout.fullRowRank_valueBlock_of_phase ω h hk

end ReadoutGenericity

section ReadoutKernelInfinite

open Matrix Real Filter FQGNN.Readout
open scoped ComplexOrder

variable {σ : Type*} {d : ℕ}

/-- **Proposition S5.1, infinite spectrum.** SI Prop. S5.1 under (H1) and (H3) (§S5.1, §S5.4). Take
any (possibly infinite) spectrum index `σ`, masses `m_k ≥ 0` with `Σ_k m_k < ∞`, frequencies
`ξ_k ∈ ℝᵈ` and any scale `s`. For any points `u_p`, the Gram matrix
`A_pq = Σ'_k m_k cos(2π s ⟨ξ_k, u_p − u_q⟩)` is positive semidefinite. (H2) is not needed. -/
theorem propS51_infinite_posSemidef {ι : Type*} [Fintype ι] {m : σ → ℝ} (hm : ∀ k, 0 ≤ m k)
    (hsum : Summable m) (ξ : σ → Fin d → ℝ) (s : ℝ) (u : ι → Fin d → ℝ) :
    (fqGramInf m ξ s u).PosSemidef :=
  FQGNN.Readout.fqGramInf_posSemidef hm hsum ξ s u

/-- **Proposition S5.1, infinite spectrum, complex-coefficient form.** SI eq. (S5.3) under (H1) and
(H3). For complex coefficients `a`, `Σ_{p,q} a_p conj(a_q) K_FQ(u_p, u_q) ≥ 0`. -/
theorem propS51_infinite_complex_nonneg {ι : Type*} [Fintype ι] {m : σ → ℝ} (hm : ∀ k, 0 ≤ m k)
    (hsum : Summable m) (ξ : σ → Fin d → ℝ) (s : ℝ) (u : ι → Fin d → ℝ) (a : ι → ℂ) :
    0 ≤ ∑ p, ∑ q, a p * (starRingEnd ℂ) (a q) * (fqKernelInf m ξ s (u p) (u q) : ℂ) :=
  FQGNN.Readout.fqGramInf_complex_nonneg hm hsum ξ s u a

/-- **Profile of the infinite-spectrum kernel.** SI Prop. S5.1 and §S5.3 ("Reality and evenness";
"Continuity of `κ` follows from (H3)"). Under (H1) and (H3), `κ(0) = Σ'_k m_k`, `κ` is even, and
`κ` is continuous. -/
theorem propS51_infinite_profile {m : σ → ℝ} (hm : ∀ k, 0 ≤ m k) (hsum : Summable m)
    (ξ : σ → Fin d → ℝ) (s : ℝ) :
    fqProfileInf m ξ s 0 = ∑' k, m k ∧
      (∀ δ, fqProfileInf m ξ s (-δ) = fqProfileInf m ξ s δ) ∧
      Continuous (fqProfileInf m ξ s) :=
  ⟨FQGNN.Readout.fqProfileInf_zero m ξ s, FQGNN.Readout.fqProfileInf_neg m ξ s,
    FQGNN.Readout.continuous_fqProfileInf hm hsum ξ s⟩

/-- **Finite truncations converge uniformly.** SI §S5.4 ("Proposition S5.1 extends in the limit").
Under (H1) and (H3), the finite-spectrum profiles over finite sets `t ⊆ σ` converge uniformly to the
infinite-spectrum profile `κ`. -/
theorem propS51_truncation_tendstoUniformly {m : σ → ℝ} (hm : ∀ k, 0 ≤ m k) (hsum : Summable m)
    (ξ : σ → Fin d → ℝ) (s : ℝ) :
    TendstoUniformly (fun t : Finset σ => fqProfile (fun k : t => m k) (fun k : t => ξ k) s)
      (fqProfileInf m ξ s) atTop :=
  FQGNN.Readout.tendstoUniformly_fqProfile_trunc hm hsum ξ s

/-- **Proposition S5.1 for the full module under polynomial decay.** SI §S5.4 (the convergent
direction of the decay criterion). Index the spectrum by the full module `k ∈ ℤⁿ`, `ξ_k = Lᵀk`. If
`0 ≤ m_k ≤ C ‖k‖^{−p}` for `k ≠ 0` with `p > n`, then (H3) holds and every Gram matrix is positive
semidefinite. -/
theorem propS54_decay_posSemidef {n : ℕ} (L : Matrix (Fin n) (Fin d) ℝ)
    {m : (Fin n → ℤ) → ℝ} {C p : ℝ} (hp : (n : ℝ) < p) (hm0 : ∀ k, 0 ≤ m k)
    (hm : ∀ k, k ≠ 0 → m k ≤ C * ‖(fun j => (k j : ℝ))‖ ^ (-p)) (s : ℝ) {ι : Type*} [Fintype ι]
    (u : ι → Fin d → ℝ) :
    (fqGramInf m (fun k => Lᵀ *ᵥ fun j => (k j : ℝ)) s u).PosSemidef :=
  FQGNN.Readout.fqGramInf_posSemidef_of_decay L hp hm0 hm s u

/-- **SI §S5.5(a), idealized infinite limit, under the aperiodicity hypothesis.** SI §S5.5(a).
Suppose `Φ_L` has no nonzero period, the
masses on the full module `ℤⁿ` are strictly positive and summable, and `s ≠ 0`. Then the Gram matrix
of any family of distinct anchors is positive definite. -/
theorem propS55a_posDef_of_aperiodic {n : ℕ} {L : Matrix (Fin n) (Fin d) ℝ}
    (hL : ¬ FQGNN.Backbone.HasNonzeroPeriod L) {m : (Fin n → ℤ) → ℝ} (hm : ∀ k, 0 < m k)
    (hsum : Summable m) {s : ℝ} (hs : s ≠ 0) {ι : Type*} [Fintype ι] {u : ι → Fin d → ℝ}
    (hu : Function.Injective u) :
    (fqGramInf m (fun k => Lᵀ *ᵥ fun j => (k j : ℝ)) s u).PosDef :=
  FQGNN.Readout.fqGramInf_posDef_ambient hL hm hsum hs hu

/-- **SI §S5.5(a) fails without aperiodicity.** SI §S5.5(a). If
`Φ_L` has a nonzero period, then there is `p ≠ 0` such that, for every choice of masses on the full
module `ℤⁿ`, the Gram matrix of the two distinct anchors `0` and `p` (scale `s = 1`) is not positive
definite. -/
theorem propS55a_not_posDef_of_periodic {n : ℕ} {L : Matrix (Fin n) (Fin d) ℝ}
    (hL : FQGNN.Backbone.HasNonzeroPeriod L) :
    ∃ p : Fin d → ℝ, p ≠ 0 ∧ ∀ m : (Fin n → ℤ) → ℝ,
      ¬ (fqGramInf m (fun k => Lᵀ *ᵥ fun j => (k j : ℝ)) 1 ![0, p]).PosDef :=
  let ⟨p, hp0, hp⟩ := (FQGNN.Backbone.hasNonzeroPeriod_iff L).1 hL
  ⟨p, hp0, fun m => FQGNN.Readout.fqGramInf_ambient_not_posDef hp m⟩

end ReadoutKernelInfinite

end FQGNN.Main

import FQGNN.Backbone.SpectrumModule

/-!
# A totally positive generator with ℚ-independent rows and a nonzero period

ℚ-independence of the generator rows does not imply aperiodicity of the torus lift `Φ_L` when
`d ≥ 2`. The counterexample of `FQGNN.Backbone.TorusEmbedding`
(`counterexampleGen = [[1,0],[√2,0],[0,1]]`) is not totally positive (its minor `Δ₀₁` vanishes),
so it lies outside `Gr₊(2,3)`. This module gives a totally positive instance,
`L = [[1,1],[√2,2],[√3,3]]`, and shows that the deployed chart attains such a generator.

Sources:
* Manuscript Prop. 2 ("Aperiodicity is a separate hypothesis outside this proposition"; "for
  `d ≥ 2`, rational independence of the generator rows does not imply it") and Methods.
* SI §S1.1 (aperiodicity-criterion paragraph) and §S2.6, third bullet.

Main results:
* `rat_eq_zero_of_sqrt_two_sqrt_three` — `1, √2, √3` are ℚ-linearly independent (from
  `Irrational √2`, `Irrational √3`, `Irrational √6`).
* `counterexampleTP_inGrPlus2` — all three minors `2 − √2`, `3 − √3`, `3√2 − 2√3` are `> 0`.
* `counterexampleTP_rows_linearIndependent`, `counterexampleTP_int_relation_trivial` — the rows
  are ℚ-independent, equivalently no nonzero `k ∈ ℤ³` has `Lᵀk = 0`.
* `counterexampleTP_mulVec` — `L (0,1)ᵀ = (1,2,3)ᵀ`; `counterexampleTP_hasNonzeroPeriod` and the
  point-set invariance.
* `counterexampleTP_rowIndependent_but_periodic` — the bundle.
* `exists_chartL_rowIndependent_but_periodic` — the deployed chart
  `L(φ)` of manuscript eq. (10) itself attains a generator with ℚ-independent rows and a nonzero
  period (via Lemma S2.2 and invariance of both properties under `L ↦ L g`, `det g ≠ 0`).
-/

noncomputable section

open Matrix

namespace FQGNN.Backbone

/-! ## ℚ-independence of `1, √2, √3` -/

/-- `√6` is irrational (`6` is not a perfect square). -/
theorem irrational_sqrt_six : Irrational (√6) := by
  have h : ¬ IsSquare (6 : ℕ) := by
    rintro ⟨r, hr⟩
    have hr3 : r ≤ 3 := by nlinarith
    interval_cases r <;> omega
  exact_mod_cast irrational_sqrt_natCast_iff.2 h

/-- `√3` is irrational (`3` is not a perfect square). -/
theorem irrational_sqrt_three : Irrational (√3) := by
  have h : ¬ IsSquare (3 : ℕ) := by
    rintro ⟨r, hr⟩
    have hr3 : r ≤ 2 := by nlinarith
    interval_cases r <;> omega
  exact_mod_cast irrational_sqrt_natCast_iff.2 h

/-- If `a + b√2 = 0` with `a, b ∈ ℚ`, then `a = b = 0`. -/
lemma rat_eq_zero_of_sqrt_two {a b : ℚ} (h : (a : ℝ) + b * √2 = 0) : a = 0 ∧ b = 0 := by
  have hb : b = 0 := by
    by_contra hb
    apply irrational_sqrt_two
    refine ⟨-a / b, ?_⟩
    have hb' : (b : ℝ) ≠ 0 := by exact_mod_cast hb
    push_cast
    field_simp
    linarith
  refine ⟨?_, hb⟩
  rw [hb] at h
  exact_mod_cast (by simpa using h : (a : ℝ) = 0)

/-- If `a + c√3 = 0` with `a, c ∈ ℚ`, then `a = c = 0`. -/
lemma rat_eq_zero_of_sqrt_three {a c : ℚ} (h : (a : ℝ) + c * √3 = 0) : a = 0 ∧ c = 0 := by
  have hc : c = 0 := by
    by_contra hc
    apply irrational_sqrt_three
    refine ⟨-a / c, ?_⟩
    have hc' : (c : ℝ) ≠ 0 := by exact_mod_cast hc
    push_cast
    field_simp
    linarith
  refine ⟨?_, hc⟩
  rw [hc] at h
  exact_mod_cast (by simpa using h : (a : ℝ) = 0)

/-- **`1, √2, √3` are ℚ-linearly independent**: `a + b√2 + c√3 = 0` with `a, b, c ∈ ℚ` forces
`a = b = c = 0`. Proof: `b√2 + c√3 = −a`; squaring gives `2b² + 3c² + 2bc√6 = a²`, so
`bc = 0` by irrationality of `√6`; the two remaining cases use irrationality of `√2`, `√3`. -/
theorem rat_eq_zero_of_sqrt_two_sqrt_three {a b c : ℚ}
    (h : (a : ℝ) + b * √2 + c * √3 = 0) : a = 0 ∧ b = 0 ∧ c = 0 := by
  have h2 : √2 * √2 = 2 := Real.mul_self_sqrt (by norm_num)
  have h3 : √3 * √3 = 3 := Real.mul_self_sqrt (by norm_num)
  have h6 : √2 * √3 = √6 := by
    rw [← Real.sqrt_mul (by norm_num)]
    norm_num
  have hbc : b * c = 0 := by
    by_contra hbc
    apply irrational_sqrt_six
    refine ⟨(a ^ 2 - 2 * b ^ 2 - 3 * c ^ 2) / (2 * (b * c)), ?_⟩
    have hbc' : ((b : ℝ) * c) ≠ 0 := by exact_mod_cast hbc
    have hsq : ((a : ℝ)) ^ 2 = (b * √2 + c * √3) ^ 2 := by
      have : (a : ℝ) = -(b * √2 + c * √3) := by linarith
      rw [this]
      ring
    have hexp : ((b : ℝ) * √2 + c * √3) ^ 2 = 2 * b ^ 2 + 3 * c ^ 2 + 2 * (b * c) * √6 := by
      rw [← h6]
      linear_combination (b : ℝ) ^ 2 * h2 + (c : ℝ) ^ 2 * h3
    push_cast
    rw [div_eq_iff (mul_ne_zero two_ne_zero hbc')]
    linear_combination hsq.trans hexp
  rcases mul_eq_zero.1 hbc with hb | hc
  · rw [hb] at h
    have h' : (a : ℝ) + c * √3 = 0 := by simpa using h
    obtain ⟨ha, hc⟩ := rat_eq_zero_of_sqrt_three h'
    exact ⟨ha, hb, hc⟩
  · rw [hc] at h
    have h' : (a : ℝ) + b * √2 = 0 := by simpa using h
    obtain ⟨ha, hb⟩ := rat_eq_zero_of_sqrt_two h'
    exact ⟨ha, hb, hc⟩

/-! ## The totally positive generator `L = [[1,1],[√2,2],[√3,3]]` -/

/-- The explicit totally positive `3 × 2` generator `L = [[1, 1], [√2, 2], [√3, 3]]`. -/
def counterexampleTP : Matrix (Fin 3) (Fin 2) ℝ :=
  !![1, 1; √2, 2; √3, 3]

/-- The three minors of `L`: `Δ₀₁ = 2 − √2`, `Δ₀₂ = 3 − √3`, `Δ₁₂ = 3√2 − 2√3`.

Source: SI (S2.2) notation `Δ_{ij}`. -/
theorem counterexampleTP_minors :
    minor2 counterexampleTP 0 1 = 2 - √2 ∧ minor2 counterexampleTP 0 2 = 3 - √3 ∧
      minor2 counterexampleTP 1 2 = 3 * √2 - 2 * √3 := by
  refine ⟨?_, ?_, ?_⟩
  · simp [minor2, counterexampleTP]
  · simp [minor2, counterexampleTP]
  · simp [minor2, counterexampleTP]
    ring

/-- **`L ∈ Gr₊(2,3)`**: every `2 × 2` minor `Δ_{ij}` (`i < j`) of `[[1,1],[√2,2],[√3,3]]` is
strictly positive (`2 − √2 > 0`, `3 − √3 > 0`, `3√2 − 2√3 > 0` since `18 > 12`).

Source: membership in `Gr₊(2,n)` as in SI §S2 / manuscript Prop. 2. -/
theorem counterexampleTP_inGrPlus2 : InGrPlus2 counterexampleTP := by
  obtain ⟨h01, h02, h12⟩ := counterexampleTP_minors
  have s2 : √2 < 2 := by
    rw [Real.sqrt_lt' (by norm_num)]
    norm_num
  have s3 : √3 < 3 := by
    rw [Real.sqrt_lt' (by norm_num)]
    norm_num
  have s23 : 2 * √3 < 3 * √2 := by
    have l : 2 * √3 = √12 := by
      rw [show (12 : ℝ) = 2 ^ 2 * 3 by norm_num, Real.sqrt_mul (by norm_num),
        Real.sqrt_sq (by norm_num)]
    have r : 3 * √2 = √18 := by
      rw [show (18 : ℝ) = 3 ^ 2 * 2 by norm_num, Real.sqrt_mul (by norm_num),
        Real.sqrt_sq (by norm_num)]
    rw [l, r]
    exact Real.sqrt_lt_sqrt (by norm_num) (by norm_num)
  intro i j hij
  fin_cases i <;> fin_cases j <;> simp at hij
  · change 0 < minor2 counterexampleTP 0 1
    rw [h01]
    linarith
  · change 0 < minor2 counterexampleTP 0 2
    rw [h02]
    linarith
  · change 0 < minor2 counterexampleTP 1 2
    rw [h12]
    linarith

/-- **Rows ℚ-independent.** The rows `(1,1)`, `(√2,2)`, `(√3,3)` are `ℚ`-linearly independent
as vectors of the ℚ-vector space `ℝ²` (`Fin 2 → ℝ`).

Source: manuscript Prop. 2, SI §S1.1, §S2.6. -/
theorem counterexampleTP_rows_linearIndependent :
    LinearIndependent ℚ (fun i : Fin 3 => counterexampleTP i) := by
  rw [Fintype.linearIndependent_iff]
  intro g hg
  have h0 := congrFun hg 0
  simp [Fin.sum_univ_three, counterexampleTP, Rat.smul_def] at h0
  obtain ⟨ha, hb, hc⟩ := rat_eq_zero_of_sqrt_two_sqrt_three h0
  intro i
  fin_cases i
  · exact ha
  · exact hb
  · exact hc

/-- **Integer-relation form.** No nonzero `k ∈ ℤ³` has `Lᵀ k = 0` (the form of rational
independence used in SI §S1.1: "no nonzero `k ∈ ℤⁿ` has `Lᵀk = 0`").

Source: SI §S1.1. -/
theorem counterexampleTP_int_relation_trivial (k : Fin 3 → ℤ)
    (hk : counterexampleTPᵀ *ᵥ (fun j => (k j : ℝ)) = 0) : k = 0 :=
  (linearIndependent_rat_iff counterexampleTP).1 counterexampleTP_rows_linearIndependent k hk

/-- **`L (0, 1)ᵀ = (1, 2, 3)ᵀ ∈ ℤ³`.** -/
theorem counterexampleTP_mulVec : counterexampleTP *ᵥ ![0, 1] = ![1, 2, 3] := by
  funext j
  fin_cases j <;> simp [counterexampleTP, mulVec, dotProduct, Fin.sum_univ_two]

/-- `L (0,1)ᵀ ∈ ℤ³`, entrywise. -/
lemma counterexampleTP_mulVec_int :
    ∀ j, ∃ m : ℤ, (counterexampleTP *ᵥ ![0, 1]) j = m := by
  intro j
  rw [counterexampleTP_mulVec]
  fin_cases j
  · exact ⟨1, by simp⟩
  · exact ⟨2, by simp⟩
  · exact ⟨3, by simp⟩

/-- **`p = (0,1)` is a nonzero period of `Φ_L`** (criterion `isPeriod_iff`). -/
theorem counterexampleTP_hasNonzeroPeriod : HasNonzeroPeriod counterexampleTP := by
  refine ⟨![0, 1], fun h => ?_, (isPeriod_iff _ _).2 counterexampleTP_mulVec_int⟩
  have := congrFun h 1
  simp at this

/-- **Point-set invariance.** Every FQ point set `Λ(X, L)` is invariant under translation by
`(0, 1)`.

Source: SI §S1.1 eq. (S1.3); `fqPointSet_add_period`. -/
theorem counterexampleTP_fqPointSet_periodic (X : Set (Fin 3 → ℂ)) :
    (fun x => x + ![0, 1]) '' fqPointSet X counterexampleTP = fqPointSet X counterexampleTP :=
  fqPointSet_add_period counterexampleTP_mulVec_int X

/-- **Totally positive generator with ℚ-independent rows and a period (bundle).** The generator `L = [[1,1],[√2,2],[√3,3]]`
* lies in `Gr₊(2,3)` (all minors `> 0`),
* has ℚ-linearly independent rows, equivalently no nonzero integer relation `Lᵀk = 0`,
* satisfies `L (0,1)ᵀ = (1,2,3)ᵀ`, hence `Φ_L` has the nonzero period `(0,1)`,
* and every FQ point set `Λ(X,L)` is `(0,1)`-periodic.

So even inside the totally positive cell, ℚ-independence of the rows does not give
aperiodicity when `d ≥ 2`.

Source: manuscript Prop. 2, Methods; SI §S1.1, §S2.6. -/
theorem counterexampleTP_rowIndependent_but_periodic :
    InGrPlus2 counterexampleTP ∧
      LinearIndependent ℚ (fun i : Fin 3 => counterexampleTP i) ∧
      (∀ k : Fin 3 → ℤ, counterexampleTPᵀ *ᵥ (fun j => (k j : ℝ)) = 0 → k = 0) ∧
      counterexampleTP *ᵥ ![0, 1] = ![1, 2, 3] ∧
      HasNonzeroPeriod counterexampleTP ∧
      ∀ X : Set (Fin 3 → ℂ), (fun x => x + ![0, 1]) '' fqPointSet X counterexampleTP =
        fqPointSet X counterexampleTP :=
  ⟨counterexampleTP_inGrPlus2, counterexampleTP_rows_linearIndependent,
    counterexampleTP_int_relation_trivial, counterexampleTP_mulVec,
    counterexampleTP_hasNonzeroPeriod, counterexampleTP_fqPointSet_periodic⟩

/-! ## The deployed chart attains such a generator -/

/-- A nonzero period survives a change of representative `L ↦ L g` with `det g ≠ 0`:
if `Lp ∈ ℤⁿ` then `(L g)(g⁻¹ p) ∈ ℤⁿ`.

Source: SI §S2 preamble (change of representative). -/
theorem HasNonzeroPeriod.mul_right {n : ℕ} {L : Matrix (Fin n) (Fin 2) ℝ}
    (hL : HasNonzeroPeriod L) {g : Matrix (Fin 2) (Fin 2) ℝ} (hg : g.det ≠ 0) :
    HasNonzeroPeriod (L * g) := by
  rw [hasNonzeroPeriod_iff] at hL ⊢
  obtain ⟨p, hp0, hp⟩ := hL
  have hgu : IsUnit g.det := isUnit_iff_ne_zero.2 hg
  refine ⟨g⁻¹ *ᵥ p, fun h => hp0 ?_, ?_⟩
  · have hgp : g *ᵥ (g⁻¹ *ᵥ p) = p := by
      rw [mulVec_mulVec, mul_nonsing_inv _ hgu, one_mulVec]
    rw [h, mulVec_zero] at hgp
    exact hgp.symm
  · have hLp : (L * g) *ᵥ (g⁻¹ *ᵥ p) = L *ᵥ p := by
      rw [mulVec_mulVec, Matrix.mul_assoc, mul_nonsing_inv _ hgu, Matrix.mul_one]
    rw [hLp]
    exact hp

/-- ℚ-independence of the rows survives `L ↦ L g` with `det g ≠ 0`:
`(L g)ᵀ k = gᵀ (Lᵀ k)` and `gᵀ` is injective.

Source: SI §S2 preamble (change of representative); SI §S1.1 (integer-relation form). -/
theorem rows_linearIndependent_mul_right {n : ℕ} {L : Matrix (Fin n) (Fin 2) ℝ}
    (hL : LinearIndependent ℚ L.row) {g : Matrix (Fin 2) (Fin 2) ℝ} (hg : g.det ≠ 0) :
    LinearIndependent ℚ (L * g).row := by
  rw [linearIndependent_rat_iff] at hL ⊢
  intro k hk
  apply hL k
  rw [transpose_mul, ← mulVec_mulVec] at hk
  exact eq_zero_of_mulVec_eq_zero (by rwa [det_transpose]) hk

/-- **The deployed chart attains a periodic generator with ℚ-independent rows.** There is a
chart parameter `φ` such that `L(φ)` (manuscript eq. (10)) has ℚ-linearly independent rows,
lies in `Gr₊(2,3)` (as every chart matrix does), and yet `Φ_{L(φ)}` has a nonzero period.

Proof: Lemma S2.2 (`chartL_surjective`) gives `L(φ) = L g` with `det g > 0` for the totally
positive `L = [[1,1],[√2,2],[√3,3]]`; both properties are invariant under such `g`.

Source: manuscript Prop. 2 and eq. (10); SI Lemma S2.2, §S1.1, §S2.6. -/
theorem exists_chartL_rowIndependent_but_periodic :
    ∃ φ : ChartParams 3, InGrPlus2 (chartL φ) ∧ LinearIndependent ℚ (chartL φ).row ∧
      HasNonzeroPeriod (chartL φ) := by
  obtain ⟨φ, g, hg, hφ⟩ := chartL_surjective (by norm_num) _ counterexampleTP_inGrPlus2
  refine ⟨φ, chartL_inGrPlus2 φ, ?_, ?_⟩
  · rw [hφ]
    exact rows_linearIndependent_mul_right counterexampleTP_rows_linearIndependent hg.ne'
  · rw [hφ]
    exact counterexampleTP_hasNonzeroPeriod.mul_right hg.ne'

end FQGNN.Backbone

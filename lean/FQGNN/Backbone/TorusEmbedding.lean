import FQGNN.Backbone.Secular

/-!
# The torus embedding `Φ_L` and its periods

This module develops the Fourier–Quasicrystal torus lift
`Φ_L(x) = exp(2πi·Lx) ∈ 𝕋ⁿ` (`x ∈ ℝᵈ`, `L ∈ ℝ^{n×d}`):

* Manuscript (Results, eq. (1)) and Methods
  (Op-A: "Op-A lifts a coordinate to `[cos 2π(Lx)_i, sin 2π(Lx)_i]` on the torus").
* Supplementary Information (SI) §S1.1, eq. (S1.1) and the aperiodicity-criterion paragraph;
  §S2.6, third bullet.

The torus lift itself, `torusLift`, is the definition already introduced in
`FQGNN.Backbone.Secular`; it is reused here unchanged.

Main results:

* `torusLift_mem_torus` — `Φ_L(x) ∈ 𝕋ⁿ`.
* `isPeriod_iff` — **exact period criterion**: `p` is a period of `Φ_L` iff `Lp ∈ ℤⁿ`,
  and `fqPointSet_add_period` — every FQ point set `Λ(X,L)` is then `p`-periodic.
* `opAFeat_inl`, `opAFeat_inr` — the Op-A features are `(Re, Im)` of `Φ_L(x)`.
* `hasNonzeroPeriod_iff`, `hasNonzeroPeriod_iff_ker_or_range` — **aperiodicity criterion**,
  and for `d = 1` `not_hasNonzeroPeriod_of_irrational` (an irrational ratio of two
  generator entries excludes every nonzero period), with the converse
  `hasNonzeroPeriod_one_iff`.
* `counterexample_rows_linearIndependent`, `counterexample_hasNonzeroPeriod` — **for `d ≥ 2`,
  ℚ-independence of the generator rows does not exclude a period**. Aperiodicity is the
  column-space condition `Lℝᵈ ∩ ℤⁿ = {0}` (manuscript Prop. 2; SI §S1.1, §S2.6).
-/

noncomputable section

open Matrix Complex

namespace FQGNN.Backbone

variable {n d : ℕ}

/-! ## The lift lands on the torus -/

/-- The `n`-torus `𝕋ⁿ = {z ∈ ℂⁿ | |z_j| = 1 ∀ j}`.

Source: manuscript eq. (1) ("`Φ_L(x) ∈ 𝕋ⁿ`"); SI eq. (S1.1). -/
def torus (n : ℕ) : Set (Fin n → ℂ) :=
  {z | ∀ j, ‖z j‖ = 1}

/-- The torus lift lands on the torus: `Φ_L(x) ∈ 𝕋ⁿ` for every `x ∈ ℝᵈ`.

Source: manuscript eq. (1) ("`Φ_L` maps each coordinate `x ∈ ℝᵈ` onto the
`n`-torus"); SI eq. (S1.1). -/
theorem torusLift_mem_torus (L : Matrix (Fin n) (Fin d) ℝ) (x : Fin d → ℝ) :
    torusLift L x ∈ torus n :=
  fun j => norm_torusLift L x j

/-- `Φ_L(0) = (1, …, 1)`.

Source: manuscript eq. (1) (evaluation at the origin). -/
theorem torusLift_zero (L : Matrix (Fin n) (Fin d) ℝ) : torusLift L 0 = 1 := by
  funext j
  simp [torusLift]

/-- Additivity: `Φ_L(x + y) = Φ_L(x) · Φ_L(y)` coordinatewise (the lift is a group
homomorphism `ℝᵈ → 𝕋ⁿ`).

Source: SI §S1.1 ("the map `x ↦ Φ_L(x)` traces a linear flow … on `𝕋ⁿ`"). -/
theorem torusLift_add (L : Matrix (Fin n) (Fin d) ℝ) (x y : Fin d → ℝ) :
    torusLift L (x + y) = torusLift L x * torusLift L y := by
  funext j
  simp only [torusLift, mulVec_add, Pi.add_apply, Pi.mul_apply, ofReal_add, ← Complex.exp_add]
  ring_nf

/-! ## The exact period criterion -/

/-- `p ∈ ℝᵈ` is a period of the torus lift: `Φ_L(x + p) = Φ_L(x)` for every `x`.

Source: SI §S1.1 (aperiodicity discussion). -/
def IsPeriod (L : Matrix (Fin n) (Fin d) ℝ) (p : Fin d → ℝ) : Prop :=
  ∀ x, torusLift L (x + p) = torusLift L x

/-- `exp(2πi r) = 1` iff `r ∈ ℤ` (`Complex.exp_eq_one_iff`); the key step of the period
criterion. -/
lemma cexp_two_pi_I_mul_eq_one_iff (r : ℝ) :
    cexp (2 * Real.pi * I * (r : ℂ)) = 1 ↔ ∃ m : ℤ, r = m := by
  rw [Complex.exp_eq_one_iff]
  have hne : (2 * (Real.pi : ℂ) * I) ≠ 0 := by
    simp [Real.pi_ne_zero, Complex.I_ne_zero]
  constructor
  · rintro ⟨m, hm⟩
    refine ⟨m, ?_⟩
    have h : (r : ℂ) = (m : ℂ) := by
      apply mul_left_cancel₀ hne
      rw [hm]
      ring
    exact_mod_cast h
  · rintro ⟨m, rfl⟩
    exact ⟨m, by push_cast; ring⟩

/-- **Exact period criterion.** `p` is a period of `Φ_L` iff `Lp ∈ ℤⁿ`.

Source: manuscript eq. (1); SI §S1.1 eq. (S1.1).
(The criterion is the precise form of the aperiodicity discussion in SI §S1.1/§S2.6.) -/
theorem isPeriod_iff (L : Matrix (Fin n) (Fin d) ℝ) (p : Fin d → ℝ) :
    IsPeriod L p ↔ ∀ j, ∃ m : ℤ, (L *ᵥ p) j = m := by
  constructor
  · intro h j
    have h0 := congrFun (h 0) j
    rw [zero_add, torusLift_zero] at h0
    exact (cexp_two_pi_I_mul_eq_one_iff _).1 h0
  · intro h x
    rw [torusLift_add]
    funext j
    have hj : torusLift L p j = 1 := (cexp_two_pi_I_mul_eq_one_iff _).2 (h j)
    rw [Pi.mul_apply, hj, mul_one]

/-- **Corollary.** If `Lp ∈ ℤⁿ`, then every FQ point set `Λ(X,L) = {x | Φ_L(x) ∈ X}`
is invariant under translation by `p`: `x + p ∈ Λ(X,L) ↔ x ∈ Λ(X,L)`.

Source: SI §S1.1 eq. (S1.3) (the point set as a pull-back) and the aperiodicity-criterion
paragraph. -/
theorem add_period_mem_fqPointSet_iff {L : Matrix (Fin n) (Fin d) ℝ} {p : Fin d → ℝ}
    (hp : ∀ j, ∃ m : ℤ, (L *ᵥ p) j = m) (X : Set (Fin n → ℂ)) (x : Fin d → ℝ) :
    x + p ∈ fqPointSet X L ↔ x ∈ fqPointSet X L := by
  change torusLift L (x + p) ∈ X ↔ torusLift L x ∈ X
  rw [(isPeriod_iff L p).2 hp x]

/-- **Corollary (set form).** If `Lp ∈ ℤⁿ` then `Λ(X,L) + p = Λ(X,L)` for every torus
set `X`.

Source: SI §S1.1. -/
theorem fqPointSet_add_period {L : Matrix (Fin n) (Fin d) ℝ} {p : Fin d → ℝ}
    (hp : ∀ j, ∃ m : ℤ, (L *ᵥ p) j = m) (X : Set (Fin n → ℂ)) :
    (fun x => x + p) '' fqPointSet X L = fqPointSet X L := by
  ext y
  constructor
  · rintro ⟨x, hx, rfl⟩
    exact (add_period_mem_fqPointSet_iff hp X x).2 hx
  · intro hy
    refine ⟨y - p, ?_, sub_add_cancel y p⟩
    rw [← add_period_mem_fqPointSet_iff hp X (y - p), sub_add_cancel]
    exact hy

/-! ## The Op-A features are `(Re, Im)` of the lift -/

/-- The Op-A lift `feat_A(x) = [cos 2π(Lx)_i, sin 2π(Lx)_i]`, indexed by `Fin n ⊕ Fin n`
(cos part `Sum.inl i`, sin part `Sum.inr i`).

Source: manuscript Methods, Op-A. -/
def opAFeat (L : Matrix (Fin n) (Fin d) ℝ) (x : Fin d → ℝ) : Fin n ⊕ Fin n → ℝ :=
  Sum.elim (fun i => Real.cos (2 * Real.pi * (L *ᵥ x) i))
    (fun i => Real.sin (2 * Real.pi * (L *ᵥ x) i))

/-- Rewriting the exponent of the lift as `↑(2π (Lx)_j) · I`. -/
lemma torusLift_eq_exp_ofReal_mul_I (L : Matrix (Fin n) (Fin d) ℝ) (x : Fin d → ℝ)
    (j : Fin n) :
    torusLift L x j = cexp (((2 * Real.pi * (L *ᵥ x) j : ℝ) : ℂ) * I) := by
  rw [torusLift]
  push_cast
  ring_nf

/-- **Op-A, cosine part.** `cos 2π(Lx)_i = Re Φ_L(x)_i`.

Source: manuscript Methods, Op-A ("lifts a coordinate to
`[cos 2π(Lx)_i, sin 2π(Lx)_i]` on the torus") together with eq. (1). -/
theorem opAFeat_inl (L : Matrix (Fin n) (Fin d) ℝ) (x : Fin d → ℝ) (i : Fin n) :
    opAFeat L x (Sum.inl i) = (torusLift L x i).re := by
  rw [torusLift_eq_exp_ofReal_mul_I, Complex.exp_ofReal_mul_I_re]
  rfl

/-- **Op-A, sine part.** `sin 2π(Lx)_i = Im Φ_L(x)_i`.

Source: manuscript Methods, Op-A, together with eq. (1). -/
theorem opAFeat_inr (L : Matrix (Fin n) (Fin d) ℝ) (x : Fin d → ℝ) (i : Fin n) :
    opAFeat L x (Sum.inr i) = (torusLift L x i).im := by
  rw [torusLift_eq_exp_ofReal_mul_I, Complex.exp_ofReal_mul_I_im]
  rfl

/-- **Op-A, complex form.** `Φ_L(x)_i = cos 2π(Lx)_i + i sin 2π(Lx)_i`; in particular the
Op-A feature vector determines `Φ_L(x)` and conversely.

Source: manuscript Methods, Op-A; eq. (1). -/
theorem torusLift_eq_opAFeat (L : Matrix (Fin n) (Fin d) ℝ) (x : Fin d → ℝ) (i : Fin n) :
    torusLift L x i = (opAFeat L x (Sum.inl i) : ℂ) + (opAFeat L x (Sum.inr i) : ℂ) * I := by
  rw [opAFeat_inl, opAFeat_inr]
  exact (Complex.re_add_im _).symm

/-! ## The aperiodicity criterion -/

/-- `Φ_L` has a nonzero period.

Source: manuscript Prop. 2 ("Aperiodicity"); SI §S1.1. -/
def HasNonzeroPeriod (L : Matrix (Fin n) (Fin d) ℝ) : Prop :=
  ∃ p : Fin d → ℝ, p ≠ 0 ∧ IsPeriod L p

/-- **Exact aperiodicity criterion.** `Φ_L` has a nonzero period iff some nonzero
`p ∈ ℝᵈ` has `Lp ∈ ℤⁿ`.

Source: manuscript Prop. 2 ("Aperiodicity is a separate hypothesis"); SI §S1.1 and §S2.6,
third bullet. -/
theorem hasNonzeroPeriod_iff (L : Matrix (Fin n) (Fin d) ℝ) :
    HasNonzeroPeriod L ↔ ∃ p : Fin d → ℝ, p ≠ 0 ∧ ∀ j, ∃ m : ℤ, (L *ᵥ p) j = m := by
  simp only [HasNonzeroPeriod, isPeriod_iff]

/-- **Kernel/column-space form.** `Φ_L` has a nonzero period iff either `L` has a
nontrivial kernel, or the column space of `L` contains a nonzero integer vector.

(The pure column-space form "the column space of `L` meets `ℤⁿ ∖ {0}`" needs `L` injective,
i.e. `rank L = d`, as it is for every generator in `Gr₊(d,n)`; see
`hasNonzeroPeriod_iff_range_of_injective`.)

Source: SI §S1.1. -/
theorem hasNonzeroPeriod_iff_ker_or_range (L : Matrix (Fin n) (Fin d) ℝ) :
    HasNonzeroPeriod L ↔
      (∃ p : Fin d → ℝ, p ≠ 0 ∧ L *ᵥ p = 0) ∨
        (∃ m : Fin n → ℤ, m ≠ 0 ∧ ∃ p : Fin d → ℝ, L *ᵥ p = fun j => (m j : ℝ)) := by
  rw [hasNonzeroPeriod_iff]
  constructor
  · rintro ⟨p, hp0, hp⟩
    choose m hm using hp
    by_cases hm0 : m = 0
    · left
      refine ⟨p, hp0, funext fun j => ?_⟩
      rw [hm j, hm0]
      simp
    · exact Or.inr ⟨m, hm0, p, funext hm⟩
  · rintro (⟨p, hp0, hp⟩ | ⟨m, hm0, p, hp⟩)
    · exact ⟨p, hp0, fun j => ⟨0, by rw [hp]; simp⟩⟩
    · refine ⟨p, ?_, fun j => ⟨m j, by rw [hp]⟩⟩
      rintro rfl
      apply hm0
      funext j
      have := congrFun hp j
      rw [mulVec_zero, Pi.zero_apply] at this
      exact_mod_cast this.symm

/-- **Column-space form, injective generator.** If `L` is injective (`rank L = d`, true
for every `L ∈ Gr₊(d,n)`), `Φ_L` has a nonzero period iff the column space of `L` contains a
nonzero integer vector.

Source: SI eq. (S1.2) (rank `d`); SI §S1.1. -/
theorem hasNonzeroPeriod_iff_range_of_injective {L : Matrix (Fin n) (Fin d) ℝ}
    (hL : Function.Injective L.mulVec) :
    HasNonzeroPeriod L ↔
      ∃ m : Fin n → ℤ, m ≠ 0 ∧ ∃ p : Fin d → ℝ, L *ᵥ p = fun j => (m j : ℝ) := by
  rw [hasNonzeroPeriod_iff_ker_or_range]
  refine or_iff_right ?_
  rintro ⟨p, hp0, hp⟩
  exact hp0 (hL (by rw [hp, mulVec_zero]))

/-- The `d = 1` generator: a column `L ∈ ℝⁿ` viewed as an `n × 1` matrix.

Source: SI §S1.1 (the `d = 1` case). -/
abbrev colGen (L : Fin n → ℝ) : Matrix (Fin n) (Fin 1) ℝ :=
  replicateCol (Fin 1) L

/-- `(colGen L · p)_j = L_j p_0`. -/
lemma colGen_mulVec (L : Fin n → ℝ) (p : Fin 1 → ℝ) (j : Fin n) :
    (colGen L *ᵥ p) j = L j * p 0 := by
  simp [colGen, mulVec, dotProduct]

/-- **`d = 1`, sufficient condition.** For a scalar generator `L ∈ ℝⁿ`, if some entry
`L i ≠ 0` has an irrational ratio `L j / L i` with another entry, then `Φ_L` has no nonzero
period.

Source: SI §S1.1. -/
theorem not_hasNonzeroPeriod_of_irrational {L : Fin n → ℝ}
    (h : ∃ i j, L i ≠ 0 ∧ Irrational (L j / L i)) : ¬ HasNonzeroPeriod (colGen L) := by
  obtain ⟨i, j, hi, hirr⟩ := h
  rw [hasNonzeroPeriod_iff]
  rintro ⟨p, hp0, hp⟩
  have hp00 : p 0 ≠ 0 := by
    intro h0
    apply hp0
    funext k
    rw [Fin.fin_one_eq_zero k, h0, Pi.zero_apply]
  obtain ⟨mi, hmi⟩ := hp i
  obtain ⟨mj, hmj⟩ := hp j
  rw [colGen_mulVec] at hmi hmj
  have hmi0 : (mi : ℝ) ≠ 0 := by
    rw [← hmi]
    exact mul_ne_zero hi hp00
  apply hirr
  refine ⟨(mj : ℚ) / mi, ?_⟩
  push_cast
  rw [← hmi, ← hmj]
  field_simp

/-- **`d = 1`, exact form.** For a scalar generator `L ∈ ℝⁿ`, `Φ_L` has a nonzero period
iff all ratios `L j / L i` (with `L i ≠ 0`) are rational. (In particular, the irrational-ratio
condition of `not_hasNonzeroPeriod_of_irrational` is also necessary for aperiodicity.)

Source: SI §S1.1. -/
theorem hasNonzeroPeriod_one_iff (L : Fin n → ℝ) :
    HasNonzeroPeriod (colGen L) ↔ ∀ i j, L i ≠ 0 → ¬ Irrational (L j / L i) := by
  constructor
  · intro hP i j hi hirr
    exact not_hasNonzeroPeriod_of_irrational ⟨i, j, hi, hirr⟩ hP
  · intro h
    by_cases hL : ∃ i, L i ≠ 0
    · obtain ⟨i, hi⟩ := hL
      have hq : ∀ j, ∃ q : ℚ, (q : ℝ) = L j / L i := fun j => by
        have := h i j hi
        simpa [Irrational] using this
      choose q hq using hq
      set N : ℕ := ∏ j, (q j).den with hN
      have hNpos : (0 : ℝ) < N := by
        rw [hN]
        exact_mod_cast Finset.prod_pos fun j _ => (q j).den_pos
      rw [hasNonzeroPeriod_iff]
      refine ⟨fun _ => (N : ℝ) / L i, ?_, fun j => ?_⟩
      · intro h0
        have := congrFun h0 0
        simp only [Pi.zero_apply, div_eq_zero_iff] at this
        rcases this with h1 | h1
        · exact hNpos.ne' h1
        · exact hi h1
      · rw [colGen_mulVec]
        have hdvd : (q j).den ∣ N := Finset.dvd_prod_of_mem _ (Finset.mem_univ j)
        obtain ⟨c, hc⟩ := hdvd
        refine ⟨(q j).num * c, ?_⟩
        have hLj : L j = (q j : ℝ) * L i := by
          rw [hq j]
          field_simp
        rw [hLj, hc]
        have hden : ((q j).den : ℝ) ≠ 0 := by exact_mod_cast (q j).den_ne_zero
        rw [show ((q j : ℚ) : ℝ) = ((q j).num : ℝ) / ((q j).den : ℝ) by
          exact_mod_cast (Rat.num_div_den (q j)).symm]
        push_cast
        field_simp
    · push Not at hL
      rw [hasNonzeroPeriod_iff]
      refine ⟨fun _ => 1, fun h0 => one_ne_zero (congrFun h0 0), fun j => ⟨0, ?_⟩⟩
      rw [colGen_mulVec, hL j]
      simp

/-! ## ℚ-independent rows do not exclude a period when `d ≥ 2` -/

/-- The explicit `3 × 2` generator `L = [[1, 0], [√2, 0], [0, 1]]`. -/
def counterexampleGen : Matrix (Fin 3) (Fin 2) ℝ :=
  !![1, 0; √2, 0; 0, 1]

/-- **(a)** The rows of `L = [[1,0],[√2,0],[0,1]]` are `ℚ`-linearly independent
(as vectors of the `ℚ`-vector space `ℝ²`).

Source: manuscript Prop. 2; SI §S1.1 and §S2.6, third bullet. -/
theorem counterexample_rows_linearIndependent :
    LinearIndependent ℚ (fun i : Fin 3 => counterexampleGen i) := by
  rw [Fintype.linearIndependent_iff]
  intro g hg
  have h0 := congrFun hg 0
  have h1 := congrFun hg 1
  simp [Fin.sum_univ_three, counterexampleGen, Rat.smul_def] at h0 h1
  have hg1 : g 1 = 0 := by
    by_contra hne
    apply irrational_sqrt_two
    refine ⟨-(g 0) / g 1, ?_⟩
    have hne' : ((g 1 : ℚ) : ℝ) ≠ 0 := by exact_mod_cast hne
    push_cast
    field_simp
    linarith
  have hg0 : g 0 = 0 := by
    rw [hg1] at h0
    exact_mod_cast (by simpa using h0 : ((g 0 : ℚ) : ℝ) = 0)
  intro i
  fin_cases i
  · exact hg0
  · exact hg1
  · exact_mod_cast h1

/-- **(a′)** Equivalently, no nonzero `k ∈ ℤ³` has `Lᵀ k = 0` (the form of rational
independence used in SI §S1.1: "no nonzero `k ∈ ℤⁿ` has `Lᵀk = 0`").

Source: SI §S1.1. -/
theorem counterexample_int_relation_trivial (k : Fin 3 → ℤ)
    (hk : counterexampleGenᵀ *ᵥ (fun j => (k j : ℝ)) = 0) : k = 0 := by
  have h := Fintype.linearIndependent_iff.1 counterexample_rows_linearIndependent
    (fun j => (k j : ℚ)) (by
      funext c
      have := congrFun hk c
      rw [mulVec, dotProduct] at this
      simp only [transpose_apply, Pi.zero_apply] at this
      simp only [Finset.sum_apply, Pi.smul_apply, Rat.smul_def, Rat.cast_intCast,
        Pi.zero_apply]
      rw [← this]
      exact Finset.sum_congr rfl fun j _ => mul_comm _ _)
  funext j
  exact_mod_cast h j

/-- **(b)** Nevertheless `L · (0, 1)ᵀ = (0, 0, 1)ᵀ ∈ ℤ³`. -/
theorem counterexample_mulVec : counterexampleGen *ᵥ ![0, 1] = ![0, 0, 1] := by
  funext j
  fin_cases j <;> simp [counterexampleGen, mulVec, dotProduct, Fin.sum_univ_two]

/-- **(b′)** Hence `p = (0, 1)` is a nonzero period of `Φ_L`, and every FQ point set
`Λ(X, L)` is invariant under translation by `(0, 1)`: rational independence of the generator
rows does **not** imply aperiodicity of the torus lift (nor of the point set) when `d ≥ 2`.

Aperiodicity is the column-space condition of `hasNonzeroPeriod_iff`, not ℚ-independence of
the rows. Source: manuscript Prop. 2; SI §S1.1 and §S2.6, third bullet. -/
theorem counterexample_hasNonzeroPeriod :
    HasNonzeroPeriod counterexampleGen ∧
      ∀ X : Set (Fin 3 → ℂ), (fun x => x + ![0, 1]) '' fqPointSet X counterexampleGen =
        fqPointSet X counterexampleGen := by
  have hint : ∀ j, ∃ m : ℤ, (counterexampleGen *ᵥ ![0, 1]) j = m := by
    intro j
    rw [counterexample_mulVec]
    fin_cases j
    · exact ⟨0, by simp⟩
    · exact ⟨0, by simp⟩
    · exact ⟨1, by simp⟩
  refine ⟨⟨![0, 1], fun h => ?_, (isPeriod_iff _ _).2 hint⟩,
    fun X => fqPointSet_add_period hint X⟩
  have := congrFun h 1
  simp at this

end FQGNN.Backbone

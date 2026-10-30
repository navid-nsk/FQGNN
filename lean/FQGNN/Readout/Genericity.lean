import FQGNN.Readout.ExponentialSum
import FQGNN.Common.Pseudoinverse

/-!
# Genericity of the rank hypothesis: the explicit `R = 2` computation of SI §S4.4

SI §S4.4, "Genericity of the rank hypothesis (sketch, plus the honest boundary)":
"At `R = 2` (value rows only) this is explicit: with `h = b − a` and midpoint centering, the
value rows are `[cos(ω_k h/2), ∓ sin(ω_k h/2)]_k`, and proportionality of the two rows forces
`(1 − λ) cos(ω_k h/2) = 0` and `(1 + λ) sin(ω_k h/2) = 0` for every kept `k` — i.e. every kept
phase `ω_k h/2` in `πℤ` (if `λ = 1`) or every one in `π/2 + πℤ` (if `λ = −1`) …".

Here the value block is the `2 × 2K` matrix with rows `φ(−h/2)` and `φ(h/2)` (centred time
`τ = t − (a+b)/2`, so the endpoints are `τ = ∓h/2`), and full row rank is the project's
`FullRowRank B := IsUnit (B Bᵀ).det`.

Main results:
* `transpose_mulVec_valueBlock_inl/inr` — `Bᵀx` on the cos and sin coordinates:
  `(x₀ + x₁) cos(ω_k h/2)` and `(x₁ − x₀) sin(ω_k h/2)`.
* `not_fullRowRank_valueBlock_iff` — **the SI's `R = 2` statement**: the value block fails full
  row rank iff every kept phase `ω_k h/2` lies in `πℤ` or every one lies in `π/2 + πℤ`.
* `fullRowRank_valueBlock_of_phase` — hence one kept phase outside `(π/2)ℤ` already gives full
  row rank.

No hypothesis `K ≥ 1` is needed: for `K = 0` the block is `2 × 0`, never of full row rank, and
both alternatives hold vacuously.
-/

noncomputable section

open Matrix Real

namespace FQGNN.Readout

open FQGNN.Common

variable {K : ℕ}

/-- The `R = 2` value block of the boundary-Hermite system: rows `φ(−h/2)` (left endpoint
`τ_a = −h/2`) and `φ(h/2)` (right endpoint `τ_b = h/2`).

Source: SI §S4.2, (S4.6) (value rows `B_i = φ(τ_e)`); SI §S4.4 genericity sketch. -/
def valueBlock (ω : Fin K → ℝ) (h : ℝ) : Matrix (Fin 2) (Fin K ⊕ Fin K) ℝ :=
  Matrix.of ![phi ω (-(h / 2)), phi ω (h / 2)]

/-- Cos coordinates of `Bᵀx`: `(Bᵀx)(inl k) = (x₀ + x₁) cos(ω_k h/2)`.

Source: SI §S4.4, "`(1 − λ) cos(ω_k h/2) = 0`". -/
theorem transpose_mulVec_valueBlock_inl (ω : Fin K → ℝ) (h : ℝ) (x : Fin 2 → ℝ) (k : Fin K) :
    ((valueBlock ω h)ᵀ *ᵥ x) (Sum.inl k) = (x 0 + x 1) * Real.cos (ω k * (h / 2)) := by
  simp only [valueBlock, mulVec, dotProduct, Fin.sum_univ_two, transpose_apply, of_apply,
    Matrix.cons_val_zero, Matrix.cons_val_one, phi, Sum.elim_inl, mul_neg, Real.cos_neg]
  ring

/-- Sin coordinates of `Bᵀx`: `(Bᵀx)(inr k) = (x₁ − x₀) sin(ω_k h/2)`.

Source: SI §S4.4, "`(1 + λ) sin(ω_k h/2) = 0`". -/
theorem transpose_mulVec_valueBlock_inr (ω : Fin K → ℝ) (h : ℝ) (x : Fin 2 → ℝ) (k : Fin K) :
    ((valueBlock ω h)ᵀ *ᵥ x) (Sum.inr k) = (x 1 - x 0) * Real.sin (ω k * (h / 2)) := by
  simp only [valueBlock, mulVec, dotProduct, Fin.sum_univ_two, transpose_apply, of_apply,
    Matrix.cons_val_zero, Matrix.cons_val_one, phi, Sum.elim_inr, mul_neg, Real.sin_neg]
  ring

/-- `Bᵀx = 0` iff, for every kept `k`, `(x₀ + x₁) cos(ω_k h/2) = 0` and
`(x₁ − x₀) sin(ω_k h/2) = 0`.

Source: SI §S4.4. -/
theorem transpose_mulVec_valueBlock_eq_zero_iff (ω : Fin K → ℝ) (h : ℝ) (x : Fin 2 → ℝ) :
    (valueBlock ω h)ᵀ *ᵥ x = 0 ↔
      ∀ k, (x 0 + x 1) * Real.cos (ω k * (h / 2)) = 0 ∧
        (x 1 - x 0) * Real.sin (ω k * (h / 2)) = 0 := by
  constructor
  · intro hx k
    refine ⟨?_, ?_⟩
    · rw [← transpose_mulVec_valueBlock_inl, hx, Pi.zero_apply]
    · rw [← transpose_mulVec_valueBlock_inr, hx, Pi.zero_apply]
  · intro hx
    funext i
    rcases i with k | k
    · rw [transpose_mulVec_valueBlock_inl, (hx k).1, Pi.zero_apply]
    · rw [transpose_mulVec_valueBlock_inr, (hx k).2, Pi.zero_apply]

/-- **SI §S4.4, the explicit `R = 2` computation.** The value block with rows `φ(−h/2)`,
`φ(h/2)` fails full row rank iff **every** kept phase `ω_k h/2` lies in `πℤ`, or **every** kept
phase lies in `π/2 + πℤ`.

Source: SI §S4.4, "Genericity of the rank hypothesis"; SI §S4.2 (full row rank of the
short, fat `B`). -/
theorem not_fullRowRank_valueBlock_iff (ω : Fin K → ℝ) (h : ℝ) :
    ¬ FullRowRank (valueBlock ω h) ↔
      (∀ k, ∃ m : ℤ, ω k * (h / 2) = m * π) ∨
        (∀ k, ∃ m : ℤ, ω k * (h / 2) = π / 2 + m * π) := by
  rw [fullRowRank_iff_transpose_mulVec_eq_zero]
  push Not
  constructor
  · rintro ⟨x, hx, hx0⟩
    rw [transpose_mulVec_valueBlock_eq_zero_iff] at hx
    by_cases hs : x 0 + x 1 = 0
    · -- `λ = 1` branch: every `sin(ω_k h/2) = 0`
      have hd : x 1 - x 0 ≠ 0 := by
        intro hd
        apply hx0
        funext i
        fin_cases i
        · simp only [Fin.zero_eta, Fin.isValue, Pi.zero_apply]
          linarith
        · simp only [Fin.mk_one, Fin.isValue, Pi.zero_apply]
          linarith
      left
      intro k
      have hsin : Real.sin (ω k * (h / 2)) = 0 :=
        (mul_eq_zero.1 (hx k).2).resolve_left hd
      obtain ⟨m, hm⟩ := Real.sin_eq_zero_iff.1 hsin
      exact ⟨m, hm.symm⟩
    · -- `λ = −1` branch: every `cos(ω_k h/2) = 0`
      right
      intro k
      have hcos : Real.cos (ω k * (h / 2)) = 0 :=
        (mul_eq_zero.1 (hx k).1).resolve_left hs
      obtain ⟨m, hm⟩ := Real.cos_eq_zero_iff.1 hcos
      exact ⟨m, by rw [hm]; ring⟩
  · rintro (hsin | hcos)
    · refine ⟨![1, -1], ?_, fun h0 => by simpa using congrFun h0 0⟩
      rw [transpose_mulVec_valueBlock_eq_zero_iff]
      intro k
      obtain ⟨m, hm⟩ := hsin k
      have : Real.sin (ω k * (h / 2)) = 0 := by
        rw [hm]
        exact Real.sin_int_mul_pi m
      simp [this]
    · refine ⟨![1, 1], ?_, fun h0 => by simpa using congrFun h0 0⟩
      rw [transpose_mulVec_valueBlock_eq_zero_iff]
      intro k
      obtain ⟨m, hm⟩ := hcos k
      have : Real.cos (ω k * (h / 2)) = 0 := by
        rw [Real.cos_eq_zero_iff]
        exact ⟨m, by rw [hm]; ring⟩
      simp [this]

/-- **Generic full row rank at `R = 2`.** If at least one kept phase `ω_k h/2` is not an integer
multiple of `π/2`, the value block has full row rank.

Source: SI §S4.4, "Rank deficiency thus requires all kept phases to sit in a fixed
discrete exceptional set — excluded for generic frequencies". -/
theorem fullRowRank_valueBlock_of_phase (ω : Fin K → ℝ) (h : ℝ)
    (hk : ∃ k, ∀ m : ℤ, ω k * (h / 2) ≠ m * (π / 2)) : FullRowRank (valueBlock ω h) := by
  by_contra hnot
  obtain ⟨k, hk⟩ := hk
  rcases (not_fullRowRank_valueBlock_iff ω h).1 hnot with hs | hc
  · obtain ⟨m, hm⟩ := hs k
    exact hk (2 * m) (by rw [hm]; push_cast; ring)
  · obtain ⟨m, hm⟩ := hc k
    exact hk (2 * m + 1) (by rw [hm]; push_cast; ring)

end FQGNN.Readout

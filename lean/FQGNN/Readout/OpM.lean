import Mathlib
import FQGNN.Common.Pseudoinverse
import FQGNN.Readout.ExponentialSum
import FQGNN.Readout.OpH

/-!
# The measured-spectrum multi-anchor readout Op-M

Op-M extends the weighted minimum-norm boundary-Hermite solve of eq. (6) by a soft exterior block:
per output coordinate it solves (manuscript eq. (7), SI eq. (S7.1))
`c = argmin_{Bc = d} ‖Ŵ⁻¹c − gη‖² + Σ_j λ_j (φ(τ_j)ᵀc − ρ̃_j)²`
and has the closed form (manuscript eq. (8), SI eq. (S7.2))
`B̃ = BŴ`, `u₀ = B̃⁺d`, `P = I − B̃⁺B̃`, `Ṽ = VŴ`, `r₀ = ρ̃ − Ṽ(u₀ + gPη)`,
`G̃ = Λ^{1/2}ṼPṼᵀΛ^{1/2}`, `y = PṼᵀΛ^{1/2}(I + G̃)⁻¹Λ^{1/2}r₀`, `c = Ŵ(u₀ + gPη + y)`.

Sources:
* Manuscript, Results, eqs. (7)–(8) and Proposition 3 (exact degeneration).
* SI §S7.1–S7.3 (Prop. S7.1, Thm S7.2, Thm S7.3, Cor. S7.4, Thm S7.5, proof of Prop. 3) and
  §S7.4 (metric law (S7.3)).

In addition to the SI statements, the module proves further exact identities of the closed
form (8): the influence law of the exterior channel through the boundary-conditioned kernel
`K̃(τ, σ) = φ(τ)ᵀ Ŵ P Ŵ φ(σ)`, the annihilation of `K̃` at the hard rows, the gauge invariance
`(ŵ, Λ, η) ↦ (αŵ, Λ/α², η/α)`, and the single-anchor (Sherman–Morrison) response.

Modelling conventions:
* `FullRowRank B := IsUnit (B * Bᵀ).det`, `pinvR B := Bᵀ (B Bᵀ)⁻¹`, `proj B := 1 − B⁺B`
  (`FQGNN.Common.Pseudoinverse`); `metricW w = diag(w) ⊕ diag(w)` and `opHW` (`FQGNN.Readout.OpH`).
* The exterior anchors are indexed by an arbitrary finite type `ο` (`N_o = |ο|`); the exterior
  evaluation matrix `V : Matrix ο (Fin K ⊕ Fin K) ℝ` has rows `φ(τ_j)` in the paper (`extV ω τo`), but
  the algebraic theorems hold for any `V`.
* `Λ^{1/2} = diagonal (fun j => √(λ j))` (`sqrtPrec`). Real `√` clamps negative arguments to
  `0`; every theorem that needs `Λ ⪰ 0` says so.
-/

namespace FQGNN.Readout

open Matrix FQGNN.Common

variable {ι ο : Type*} [Fintype ι] [DecidableEq ι] [Fintype ο] [DecidableEq ο] {K : ℕ}

/-! ## Definitions -/

/-- `Λ^{1/2} = diag(√λ_1, …, √λ_{N_o})` (manuscript eq. (8), SI eq. (S7.2)). -/
noncomputable def sqrtPrec (lam : ο → ℝ) : Matrix ο ο ℝ :=
  diagonal fun j => Real.sqrt (lam j)

/-- Exterior evaluation matrix with rows `φ(τ_j)` (manuscript eq. (7): "`φ(τ_j)` evaluates the comb
at that anchor's centred time"; SI §S7.1). -/
noncomputable def extV (ω : Fin K → ℝ) (τo : ο → ℝ) : Matrix ο (Fin K ⊕ Fin K) ℝ :=
  Matrix.of fun j => phi ω (τo j)

/-- Exterior innovation `r₀ = ρ̃ − Ṽ(u₀ + gPη)` with `Ṽ = VŴ`, `u₀ = (BŴ)⁺d`, `P = I − (BŴ)⁺(BŴ)`
(manuscript eq. (8); SI eq. (S7.2)). It does not depend on `Λ`. -/
noncomputable def opMInnov (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) : ο → ℝ :=
  ρ - (V * metricW w) *ᵥ
    (pinvR (B * metricW w) *ᵥ d + g • (proj (B * metricW w) *ᵥ η))

/-- Soft-block Gram matrix `G̃ = Λ^{1/2} Ṽ P Ṽᵀ Λ^{1/2}` (manuscript eq. (8); SI eq. (S7.2)). -/
noncomputable def opMGram (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam : ο → ℝ) : Matrix ο ο ℝ :=
  sqrtPrec lam * (V * metricW w) * proj (B * metricW w) * (V * metricW w)ᵀ * sqrtPrec lam

/-- Exterior-channel term `y = P Ṽᵀ Λ^{1/2} (I + G̃)⁻¹ Λ^{1/2} r₀` (manuscript eq. (8);
SI eq. (S7.2)). -/
noncomputable def opMSoft (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) : Fin K ⊕ Fin K → ℝ :=
  (proj (B * metricW w) * (V * metricW w)ᵀ * sqrtPrec lam * (1 + opMGram B w V lam)⁻¹ *
    sqrtPrec lam) *ᵥ opMInnov B w V ρ d g η

/-- **Op-M closed form, manuscript eq. (8) / SI eq. (S7.2):**
`c = Ŵ (u₀ + g P η + y)`. -/
noncomputable def opM (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) : Fin K ⊕ Fin K → ℝ :=
  metricW w *ᵥ (pinvR (B * metricW w) *ᵥ d + g • (proj (B * metricW w) *ᵥ η) +
    opMSoft B w V lam ρ d g η)

/-- **Op-M objective, manuscript eq. (7) / SI eq. (S7.1):**
`J(c) = ‖Ŵ⁻¹c − gη‖² + Σ_j λ_j (φ(τ_j)ᵀc − ρ̃_j)²` (with `φ(τ_j)ᵀc = (V c)_j`). -/
noncomputable def opMObj (w : Fin K → ℝ) (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ)
    (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) (c : Fin K ⊕ Fin K → ℝ) : ℝ :=
  ((metricW w)⁻¹ *ᵥ c - g • η) ⬝ᵥ ((metricW w)⁻¹ *ᵥ c - g • η) +
    ∑ j, lam j * ((V *ᵥ c) j - ρ j) ^ 2

omit [Fintype ο] [DecidableEq ο] in
/-- With `V = extV ω τo`, the soft term of eq. (7) is exactly `Σ_j λ_j (φ(τ_j)ᵀc − ρ̃_j)²`. -/
theorem extV_mulVec (ω : Fin K → ℝ) (τo : ο → ℝ) (c : Fin K ⊕ Fin K → ℝ) (j : ο) :
    (extV ω τo *ᵥ c) j = rcomb ω c (τo j) := rfl

/-! ### Basic facts about `Λ^{1/2}` -/

omit [Fintype ο] in
/-- `Λ^{1/2}` is symmetric. -/
theorem sqrtPrec_transpose (lam : ο → ℝ) : (sqrtPrec lam)ᵀ = sqrtPrec lam :=
  diagonal_transpose _

/-- `Λ^{1/2} Λ^{1/2} = Λ` for `Λ ⪰ 0`. -/
theorem sqrtPrec_mul_self {lam : ο → ℝ} (hlam : ∀ j, 0 ≤ lam j) :
    sqrtPrec lam * sqrtPrec lam = diagonal lam := by
  unfold sqrtPrec
  rw [diagonal_mul_diagonal]
  congr 1
  funext j
  exact Real.mul_self_sqrt (hlam j)

omit [Fintype ο] in
/-- `Λ = 0 ⇒ Λ^{1/2} = 0` (used for Prop. 3). -/
theorem sqrtPrec_zero : sqrtPrec (0 : ο → ℝ) = 0 := by
  unfold sqrtPrec
  simp

/-- `Σ_j λ_j x_j² = ‖Λ^{1/2} x‖²` for `Λ ⪰ 0` (the soft term of eq. (7) as a squared norm,
SI §S7.1 proof "`‖Λ^{1/2}(Ṽy − r₀)‖²`"). -/
theorem sum_prec_sq_eq {lam : ο → ℝ} (hlam : ∀ j, 0 ≤ lam j) (x : ο → ℝ) :
    ∑ j, lam j * x j ^ 2 = (sqrtPrec lam *ᵥ x) ⬝ᵥ (sqrtPrec lam *ᵥ x) := by
  unfold sqrtPrec
  simp only [dotProduct, mulVec_diagonal]
  refine Finset.sum_congr rfl fun j _ => ?_
  have h := Real.mul_self_sqrt (hlam j)
  calc lam j * x j ^ 2 = (Real.sqrt (lam j) * Real.sqrt (lam j)) * x j ^ 2 := by rw [h]
    _ = Real.sqrt (lam j) * x j * (Real.sqrt (lam j) * x j) := by ring

/-! ## The quadratic core (generic, used for SI Prop. S7.1)

The algebra of the SI §S7.1 proof in whitened coordinates `u = Ŵ⁻¹c`, stated for an arbitrary
symmetric idempotent `P`, symmetric `S` (later `Λ^{1/2}`), whitened exterior matrix `Ṽ` and
particular solution `u₀` with `P u₀ = 0`. -/

section Core

variable {κ : Type*} [Fintype κ] [DecidableEq κ]

omit [DecidableEq κ] in
/-- Self-adjointness of a symmetric matrix: `(P a) ⬝ b = a ⬝ (P b)`. -/
theorem mulVec_dotProduct_of_transpose_eq {P : Matrix κ κ ℝ} (hPt : Pᵀ = P) (a b : κ → ℝ) :
    (P *ᵥ a) ⬝ᵥ b = a ⬝ᵥ (P *ᵥ b) := by
  rw [dotProduct_mulVec, ← mulVec_transpose, hPt]

/-- Stationarity core (SI §S7.1 proof, push-through step): `(I + AᵀA) Aᵀ (I + AAᵀ)⁻¹ = Aᵀ`. -/
theorem one_add_transpose_mul_self_mul_gain (A : Matrix ο κ ℝ) :
    (1 + Aᵀ * A) * (Aᵀ * (1 + A * Aᵀ)⁻¹) = Aᵀ := by
  have hQ := isUnit_det_one_add_mul_transpose_self A
  have hcomm : (1 + Aᵀ * A) * Aᵀ = Aᵀ * (1 + A * Aᵀ) := by
    rw [Matrix.add_mul, Matrix.mul_add, Matrix.one_mul, Matrix.mul_one, Matrix.mul_assoc]
  rw [← Matrix.mul_assoc, hcomm, Matrix.mul_assoc, mul_nonsing_inv _ hQ, Matrix.mul_one]

/-- **Pythagorean identity of the soft block (core of SI Prop. S7.1).**
Let `P` be symmetric idempotent, `S` symmetric, `P u₀ = 0`, and let
`r₀ = ρ − Ṽ(u₀ + gPη)`, `y = P Ṽᵀ S (I + S Ṽ P Ṽᵀ S)⁻¹ S r₀`, `u⋆ = u₀ + gPη + y`.
For the whitened objective `F(v) = ‖v − gη‖² + ‖S(Ṽv − ρ)‖²` and every `h ∈ range P`:
`F(u⋆ + h) = F(u⋆) + ‖h‖² + ‖SṼh‖²`. -/
theorem soft_pythagoras {P : Matrix κ κ ℝ} (hPt : Pᵀ = P) (hPP : P * P = P)
    {S : Matrix ο ο ℝ} (hSt : Sᵀ = S) (Vt : Matrix ο κ ℝ) {u0 : κ → ℝ} (hPu0 : P *ᵥ u0 = 0)
    (η : κ → ℝ) (g : ℝ) (ρ : ο → ℝ) {h : κ → ℝ} (hh : P *ᵥ h = h) :
    let r0 := ρ - Vt *ᵥ (u0 + g • (P *ᵥ η))
    let y := (P * Vtᵀ * S * (1 + S * Vt * P * Vtᵀ * S)⁻¹ * S) *ᵥ r0
    let u := u0 + g • (P *ᵥ η) + y
    (u + h - g • η) ⬝ᵥ (u + h - g • η) +
        (S *ᵥ (Vt *ᵥ (u + h) - ρ)) ⬝ᵥ (S *ᵥ (Vt *ᵥ (u + h) - ρ)) =
      ((u - g • η) ⬝ᵥ (u - g • η) + (S *ᵥ (Vt *ᵥ u - ρ)) ⬝ᵥ (S *ᵥ (Vt *ᵥ u - ρ))) +
        h ⬝ᵥ h + (S *ᵥ (Vt *ᵥ h)) ⬝ᵥ (S *ᵥ (Vt *ᵥ h)) := by
  intro r0 y u
  set A : Matrix ο κ ℝ := S * Vt * P with hA
  have hAT : Aᵀ = P * Vtᵀ * S := by
    rw [hA, transpose_mul, transpose_mul, hPt, hSt, Matrix.mul_assoc]
  have hG : S * Vt * P * Vtᵀ * S = A * Aᵀ := by
    rw [hAT, hA]
    simp only [Matrix.mul_assoc]
    rw [← Matrix.mul_assoc P P, hPP]
  have hy : y = (Aᵀ * (1 + A * Aᵀ)⁻¹ * S) *ᵥ r0 := by
    change (P * Vtᵀ * S * (1 + S * Vt * P * Vtᵀ * S)⁻¹ * S) *ᵥ r0 = _
    rw [hG, hAT]
  -- `y ∈ range P`
  have hPy : P *ᵥ y = y := by
    rw [hy, mulVec_mulVec, hAT]
    congr 1
    simp only [← Matrix.mul_assoc]
    rw [hPP]
  -- stationarity `(I + AᵀA) y = Aᵀ S r₀`
  have hstat : y + Aᵀ *ᵥ (A *ᵥ y) = Aᵀ *ᵥ (S *ᵥ r0) := by
    have h1 : y + Aᵀ *ᵥ (A *ᵥ y) = (1 + Aᵀ * A) *ᵥ y := by
      rw [add_mulVec, one_mulVec, mulVec_mulVec]
    rw [h1, hy, mulVec_mulVec, mulVec_mulVec, ← Matrix.mul_assoc, ← Matrix.mul_assoc,
      Matrix.mul_assoc (1 + Aᵀ * A), one_add_transpose_mul_self_mul_gain]
  -- `P (u⋆ − gη) = y`
  have hPu : P *ᵥ (u - g • η) = y := by
    change P *ᵥ (u0 + g • (P *ᵥ η) + y - g • η) = y
    rw [mulVec_sub, mulVec_add, mulVec_add, hPu0, hPy, mulVec_smul, mulVec_smul, mulVec_mulVec,
      hPP]
    abel
  -- `S(Ṽu⋆ − ρ) = A y − S r₀`
  have hx : S *ᵥ (Vt *ᵥ u - ρ) = A *ᵥ y - S *ᵥ r0 := by
    have e1 : Vt *ᵥ u - ρ = Vt *ᵥ y - r0 := by
      change Vt *ᵥ (u0 + g • (P *ᵥ η) + y) - ρ = Vt *ᵥ y - (ρ - Vt *ᵥ (u0 + g • (P *ᵥ η)))
      rw [mulVec_add]
      abel
    rw [e1, mulVec_sub, ← hPy, mulVec_mulVec, mulVec_mulVec, hPy, hA, Matrix.mul_assoc S Vt P]
  -- the cross term vanishes
  have hk : S *ᵥ (Vt *ᵥ h) = A *ᵥ h := by
    rw [← hh, mulVec_mulVec, mulVec_mulVec, hh, hA, Matrix.mul_assoc S Vt P]
  have hcross : (u - g • η) ⬝ᵥ h + (S *ᵥ (Vt *ᵥ u - ρ)) ⬝ᵥ (S *ᵥ (Vt *ᵥ h)) = 0 := by
    have t1 : (u - g • η) ⬝ᵥ h = y ⬝ᵥ h := by
      conv_lhs => rw [← hh]
      rw [← mulVec_dotProduct_of_transpose_eq hPt, hPu]
    have t2 : (S *ᵥ (Vt *ᵥ u - ρ)) ⬝ᵥ (S *ᵥ (Vt *ᵥ h)) =
        (Aᵀ *ᵥ (S *ᵥ (Vt *ᵥ u - ρ))) ⬝ᵥ h := by
      rw [hk, dotProduct_mulVec, ← mulVec_transpose]
    rw [t1, t2, ← add_dotProduct, hx, mulVec_sub, add_sub_assoc', hstat, sub_self,
      zero_dotProduct]
  have e2 : u + h - g • η = (u - g • η) + h := by abel
  have e3 : S *ᵥ (Vt *ᵥ (u + h) - ρ) = S *ᵥ (Vt *ᵥ u - ρ) + S *ᵥ (Vt *ᵥ h) := by
    rw [← mulVec_add, mulVec_add]
    congr 1
    abel
  rw [e2, e3]
  simp only [add_dotProduct, dotProduct_add]
  rw [dotProduct_comm h (u - g • η), dotProduct_comm (S *ᵥ (Vt *ᵥ h)) (S *ᵥ (Vt *ᵥ u - ρ))]
  linarith [hcross]

end Core

/-! ## `I + G̃` is positive definite, hence invertible — with no hypothesis at all -/

section Positivity

variable {κ : Type*} [Fintype κ] [DecidableEq κ]

/-- The null-space projector is a Gram matrix, `P = Pᵀ P`, for **every** real matrix `M`: under full
row rank because `P` is a symmetric idempotent (`FQGNN.Common.Pseudoinverse`); otherwise because
the formal model then has `(MMᵀ)⁻¹ = 0` (Mathlib's convention), so `P = I`. -/
theorem proj_eq_transpose_mul_self (M : Matrix ι κ ℝ) : proj M = (proj M)ᵀ * proj M := by
  by_cases h : FullRowRank M
  · rw [transpose_proj, proj_mul_proj h]
  · have hp : pinvR M = 0 := by
      unfold pinvR
      rw [nonsing_inv_apply_not_isUnit _ h, Matrix.mul_zero]
    unfold proj
    rw [hp, Matrix.zero_mul, sub_zero, transpose_one, Matrix.one_mul]

end Positivity

/-- `G̃ = AᵀA` with `A = P Ṽᵀ Λ^{1/2}` (SI §S7.1 proof: `A = Λ^{1/2} Ṽ P`, up to transposition).
Holds for every `B`, `ŵ`, `V`, `λ`. -/
theorem opMGram_eq_transpose_mul_self (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam : ο → ℝ) :
    opMGram B w V lam =
      (proj (B * metricW w) * (V * metricW w)ᵀ * sqrtPrec lam)ᵀ *
        (proj (B * metricW w) * (V * metricW w)ᵀ * sqrtPrec lam) := by
  unfold opMGram
  rw [transpose_mul (proj (B * metricW w) * (V * metricW w)ᵀ),
    transpose_mul (proj (B * metricW w)), transpose_transpose, sqrtPrec_transpose]
  conv_lhs => rw [proj_eq_transpose_mul_self (B * metricW w)]
  simp only [Matrix.mul_assoc]

/-- **Manuscript eq. (8) "`G̃ ⪰ 0`" (SI eq. (S7.2), Thm S7.2).** The soft-block Gram matrix is
positive semidefinite for every anchor layout, every metric and every `λ`. -/
theorem opMGram_posSemidef (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam : ο → ℝ) : (opMGram B w V lam).PosSemidef := by
  rw [opMGram_eq_transpose_mul_self]
  have h := posSemidef_conjTranspose_mul_self
    (proj (B * metricW w) * (V * metricW w)ᵀ * sqrtPrec lam)
  rwa [conjTranspose_eq_transpose_of_trivial] at h

/-- **Manuscript eq. (8) "`I + G̃ ⪰ I`" (SI Thm S7.2).** `I + G̃` is positive
definite for every anchor layout, every metric and every `λ` — the soft block adds no rank
hypothesis. -/
theorem one_add_opMGram_posDef (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam : ο → ℝ) : (1 + opMGram B w V lam).PosDef := by
  rw [opMGram_eq_transpose_mul_self]
  exact posDef_one_add_transpose_mul_self _

/-- **Manuscript eq. (8), "`I + G̃` is invertible for every anchor layout and every
`Λ ⪰ 0`, so the soft block adds no rank hypothesis anywhere" (SI Thm S7.2).**
`IsUnit (det (I + G̃))`, with no hypothesis on `B`, `ŵ`, `V` or `λ`. -/
theorem isUnit_det_one_add_opMGram (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam : ο → ℝ) : IsUnit (1 + opMGram B w V lam).det := by
  rw [opMGram_eq_transpose_mul_self]
  exact isUnit_det_one_add_transpose_mul_self _

/-! ## Structural identities -/

/-- Op-M is Op-H-W (eq. (6), metric `ŵ`) plus the exterior contribution `Ŵ y`
(`c(Λ) − c(0) = Ŵy`; SI §S7.3). Holds for every argument. -/
theorem opM_eq_opHW_add (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) :
    opM B w V lam ρ d g η = opHW B w d g η + metricW w *ᵥ opMSoft B w V lam ρ d g η := by
  unfold opM opHW
  rw [mulVec_add, mulVec_add, mulVec_smul]

/-- `y = P (Ṽᵀ Λ^{1/2} (I + G̃)⁻¹ Λ^{1/2} r₀)`: the exterior term lies in `range P`
(SI Thm S7.3 proof, "`y ∈ range(P)`"). -/
theorem opMSoft_eq_proj_mulVec (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) :
    opMSoft B w V lam ρ d g η = proj (B * metricW w) *ᵥ
      (((V * metricW w)ᵀ * sqrtPrec lam * (1 + opMGram B w V lam)⁻¹ * sqrtPrec lam) *ᵥ
        opMInnov B w V ρ d g η) := by
  unfold opMSoft
  rw [mulVec_mulVec]
  simp only [Matrix.mul_assoc]

/-- `B (Ŵ (P z)) = 0` for full row rank `B` and `ŵ > 0` (SI Thm S7.3 proof, "`B̃P = 0`"). -/
theorem mulVec_metricW_proj_mulVec {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B)
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (z : Fin K ⊕ Fin K → ℝ) :
    B *ᵥ (metricW w *ᵥ (proj (B * metricW w) *ᵥ z)) = 0 := by
  rw [mulVec_mulVec, mulVec_proj_mulVec (fullRowRank_mul_metricW hB hw)]

/-! ## Endpoint annihilation (SI Cor. S7.4) -/

/-- **SI Cor. S7.4 (endpoint annihilation), algebraic form.** The exterior channel's
contribution `Ŵ y` lies in the null space of the hard constraints: `B (Ŵ y) = 0`, for every `λ`, `V`,
`ρ̃`, `d`, `g`, `η` (full row rank `B`, `ŵ > 0`). -/
theorem mulVec_metricW_opMSoft {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B)
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ)
    (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    B *ᵥ (metricW w *ᵥ opMSoft B w V lam ρ d g η) = 0 := by
  rw [opMSoft_eq_proj_mulVec]
  exact mulVec_metricW_proj_mulVec hB hw _

/-- **SI Cor. S7.4, value rows.** If row `i` of `B` is the value row `φ(τ_e)`, the exterior
contribution has zero value there: `r_{Ŵy}(τ_e) = 0`. -/
theorem rcomb_exterior_eq_zero_of_valueRow {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B)
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ)
    (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) (ω : Fin K → ℝ) {i : ι} {τe : ℝ}
    (hrow : B i = phi ω τe) :
    rcomb ω (metricW w *ᵥ opMSoft B w V lam ρ d g η) τe = 0 := by
  have h := congrFun (mulVec_metricW_opMSoft hB hw V lam ρ d g η) i
  unfold rcomb
  rw [← hrow]
  exact h

/-- **SI Cor. S7.4, included derivative rows.** If row `i` of `B` is the
equilibrated derivative row `s • φ'(τ_e)` (`s ≠ 0`), the exterior contribution has zero slope there:
`HasDerivAt r_{Ŵy} 0 τ_e`. -/
theorem hasDerivAt_rcomb_exterior_of_derivRow {B : Matrix ι (Fin K ⊕ Fin K) ℝ}
    (hB : FullRowRank B) {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (V : Matrix ο (Fin K ⊕ Fin K) ℝ)
    (lam ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) (ω : Fin K → ℝ) {i : ι}
    {τe s : ℝ} (hs : s ≠ 0) (hrow : B i = s • dphi ω τe) :
    HasDerivAt (rcomb ω (metricW w *ᵥ opMSoft B w V lam ρ d g η)) 0 τe := by
  have h : B i ⬝ᵥ (metricW w *ᵥ opMSoft B w V lam ρ d g η) = 0 :=
    congrFun (mulVec_metricW_opMSoft hB hw V lam ρ d g η) i
  rw [hrow, smul_dotProduct, smul_eq_mul] at h
  have h' : dphi ω τe ⬝ᵥ (metricW w *ᵥ opMSoft B w V lam ρ d g η) = 0 :=
    (mul_eq_zero.mp h).resolve_left hs
  have hd := hasDerivAt_rcomb ω (metricW w *ᵥ opMSoft B w V lam ρ d g η) τe
  rwa [h'] at hd

/-- The Op-M reconstruction is the Op-H-W reconstruction plus the exterior deformation
`r_{Ŵy}(τ)` (SI Cor. S7.4: "it deforms the reconstruction"). -/
theorem recon_opM (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) (ω : Fin K → ℝ) (a b pa pb t : ℝ) :
    recon ω (opM B w V lam ρ d g η) a b pa pb t =
      recon ω (opHW B w d g η) a b pa pb t +
        rcomb ω (metricW w *ᵥ opMSoft B w V lam ρ d g η) (ctime a b t) := by
  rw [recon_eq, recon_eq, opM_eq_opHW_add, rcomb_add]
  ring

/-- **SI Cor. S7.4, packaged on the boundary-Hermite system.** "The soft anchors can reshape
the manoeuvre; they can never move the endpoints": for every `λ`, `V`, `ρ̃`, gate `g` and `η`, the
exterior deformation vanishes at both bracketing anchors and has zero derivative at every included
velocity-row time. -/
theorem exterior_endpoint_annihilation (ω : Fin K → ℝ) (a b pa pb : ℝ) {Rv : ℕ}
    (te s v : Fin Rv → ℝ) (hs : ∀ e, s e ≠ 0) (hB : FullRowRank (hermiteB ω a b te s))
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ)
    (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    let δ := rcomb ω (metricW w *ᵥ
      opMSoft (hermiteB ω a b te s) w V lam ρ (hermiteD a b pa pb s v) g η)
    δ (ctime a b a) = 0 ∧ δ (ctime a b b) = 0 ∧ ∀ e, HasDerivAt δ 0 (ctime a b (te e)) := by
  refine ⟨?_, ?_, fun e => ?_⟩
  · exact rcomb_exterior_eq_zero_of_valueRow hB hw V lam ρ _ g η ω (i := Sum.inl 0) rfl
  · exact rcomb_exterior_eq_zero_of_valueRow hB hw V lam ρ _ g η ω (i := Sum.inl 1) rfl
  · exact hasDerivAt_rcomb_exterior_of_derivRow hB hw V lam ρ _ g η ω (hs e) (i := Sum.inr e) rfl

/-! ## Exactness survives every dial (SI Thm S7.3) -/

/-- **SI Thm S7.3 (manuscript: "exactness survives every dial").** For full
row rank `B` and every metric `ŵ > 0`, the Op-M coefficients satisfy `B c = d` for **every** exterior
layout `V`, precisions `λ`, residuals `ρ̃`, gate `g` and null-space coordinates `η`. -/
theorem mulVec_opM {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B) {w : Fin K → ℝ}
    (hw : ∀ k, 0 < w k) (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) : B *ᵥ opM B w V lam ρ d g η = d := by
  rw [opM_eq_opHW_add, mulVec_add, mulVec_opHW hB hw, mulVec_metricW_opMSoft hB hw, add_zero]

/-- **SI Thm S7.3, geometric form on the boundary-Hermite system (manuscript: "boundary
interpolation and the velocity match are exact for all values of every learned and measured
quantity").** For every `λ`, `V`, `ρ̃`, `g`, `η` and every metric `ŵ > 0`, the Op-M reconstruction
interpolates both bracketing anchors and matches every included boundary velocity. -/
theorem opM_hermite_interpolation (ω : Fin K → ℝ) {a b : ℝ} (hab : a ≠ b) (pa pb : ℝ)
    {Rv : ℕ} (te s v : Fin Rv → ℝ) (hs : ∀ e, s e ≠ 0)
    (hB : FullRowRank (hermiteB ω a b te s)) {w : Fin K → ℝ} (hw : ∀ k, 0 < w k)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    let c := opM (hermiteB ω a b te s) w V lam ρ (hermiteD a b pa pb s v) g η
    recon ω c a b pa pb a = pa ∧ recon ω c a b pa pb b = pb ∧
      ∀ e, HasDerivAt (recon ω c a b pa pb) (v e) (te e) := by
  intro c
  have hc : hermiteB ω a b te s *ᵥ c = hermiteD a b pa pb s v := mulVec_opM hB hw V lam ρ _ g η
  refine ⟨?_, ?_, fun e => ?_⟩
  · exact recon_left_of_valueRow hc ω a b pa pb (i := Sum.inl 0) rfl rfl
  · exact recon_right_of_valueRow hc ω hab pa pb (i := Sum.inl 1) rfl rfl
  · exact hasDerivAt_recon_of_derivRow hc ω a b pa pb (te e) (v e) (s e) (hs e)
      (i := Sum.inr e) rfl rfl

/-! ## Exact degeneration (manuscript Prop. 3; SI §S7.3) -/

/-- At `Λ = 0` the soft Gram matrix vanishes, `G̃ = 0` (SI §S7.3: "the soft block is absent:
`G̃ = 0`"). -/
theorem opMGram_zero_prec (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) : opMGram B w V 0 = 0 := by
  unfold opMGram
  rw [sqrtPrec_zero, Matrix.mul_zero]

/-- At `Λ = 0` the exterior term vanishes exactly, `y = 0` (SI §S7.3). -/
theorem opMSoft_zero_prec (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) : opMSoft B w V 0 ρ d g η = 0 := by
  unfold opMSoft
  rw [sqrtPrec_zero, Matrix.mul_zero, zero_mulVec]

/-- **Manuscript Prop. 3, `Λ`-dial (SI §S7.3).** At `Λ = 0`, Op-M
*is* Op-H-W (eq. (6)) with metric `ŵ`: `c = Ŵ(BŴ)⁺d + g Ŵ(I − (BŴ)⁺(BŴ))η`, for every `B`, `ŵ`,
`V`, `ρ̃`, `d`, `g`, `η` — with no rank or positivity hypothesis. -/
theorem opM_zero_prec (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) : opM B w V 0 ρ d g η = opHW B w d g η := by
  rw [opM_eq_opHW_add, opMSoft_zero_prec, mulVec_zero, add_zero]

/-- **Manuscript Prop. 3 (exact degeneration), abstract metric law.** If the metric law
returns the base metric at `γ = 0` (`metricLaw 0 = w`), then Op-M at `(γ, Λ) = (0, 0)` coincides with
the weighted minimum-norm readout of eq. (6): `c = opHW B w d g η` (SI §S7.3). -/
theorem prop3_exact_degeneration (metricLaw : ℝ → Fin K → ℝ) {w : Fin K → ℝ}
    (hlaw : metricLaw 0 = w) (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (V : Matrix ο (Fin K ⊕ Fin K) ℝ)
    (ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    opM B (metricLaw 0) V 0 ρ d g η = opHW B w d g η := by
  rw [hlaw, opM_zero_prec]

/-- The geometric metric law of SI eq. (S7.3) / manuscript Results ("`log ŵ_k = (1−γ) log w_k +
γ ½ log Ĩ_k` on the resolvable band (clamped)"): on the band,
`log ŵ_k = log w_k + γ · clampC(½ log Ĩ_k − log w_k)` (the clamp acts on the deviation of the
measured term from the base profile, inside the `γ`-term); off the band `ŵ_k = w_k`.
`clampC` is an arbitrary function (the deployed clamp); `band` is the resolvable band. -/
noncomputable def metricLawGeom (band : Finset (Fin K)) (clampC : ℝ → ℝ) (w Ihat : Fin K → ℝ)
    (γ : ℝ) : Fin K → ℝ :=
  fun k => if k ∈ band then
    Real.exp (Real.log (w k) + γ * clampC (Real.log (Ihat k) / 2 - Real.log (w k))) else w k

/-- With the identity clamp the law is exactly SI eq. (S7.3) on the band:
`log ŵ_k = (1 − γ) log w_k + γ · ½ log Ĩ_k`. -/
theorem log_metricLawGeom_id (band : Finset (Fin K)) (w Ihat : Fin K → ℝ) (γ : ℝ) {k : Fin K}
    (hk : k ∈ band) :
    Real.log (metricLawGeom band id w Ihat γ k) =
      (1 - γ) * Real.log (w k) + γ * (Real.log (Ihat k) / 2) := by
  simp only [metricLawGeom, hk, ↓reduceIte, id, Real.log_exp]
  ring

/-- **SI §S7.4: "`γ = 0` is the base metric bit-for-bit".** For `w > 0`
the geometric law returns `w` at `γ = 0`, whatever the clamp, band and periodogram. -/
theorem metricLawGeom_zero (band : Finset (Fin K)) (clampC : ℝ → ℝ) {w : Fin K → ℝ}
    (hw : ∀ k, 0 < w k) (Ihat : Fin K → ℝ) : metricLawGeom band clampC w Ihat 0 = w := by
  funext k
  unfold metricLawGeom
  split_ifs
  · rw [zero_mul, add_zero, Real.exp_log (hw k)]
  · rfl

/-- The measured metric is a metric: `ŵ > 0` for every `γ` (SI Thm S7.3: "the metric enters only
through `Ŵ > 0`"). -/
theorem metricLawGeom_pos (band : Finset (Fin K)) (clampC : ℝ → ℝ) {w : Fin K → ℝ}
    (hw : ∀ k, 0 < w k) (Ihat : Fin K → ℝ) (γ : ℝ) (k : Fin K) :
    0 < metricLawGeom band clampC w Ihat γ k := by
  unfold metricLawGeom
  split_ifs
  · exact Real.exp_pos _
  · exact hw k

/-- **Manuscript Prop. 3 (exact degeneration) with the deployed geometric metric law
(SI eq. (S7.3), §S7.3).** At `(γ, Λ) = (0, 0)` Op-M coincides with Op-H-W of eq. (6) with the base
metric `w`, for every band, clamp and periodogram. -/
theorem prop3_exact_degeneration_geom (band : Finset (Fin K)) (clampC : ℝ → ℝ) {w : Fin K → ℝ}
    (hw : ∀ k, 0 < w k) (Ihat : Fin K → ℝ) (B : Matrix ι (Fin K ⊕ Fin K) ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    opM B (metricLawGeom band clampC w Ihat 0) V 0 ρ d g η = opHW B w d g η :=
  prop3_exact_degeneration _ (metricLawGeom_zero band clampC hw Ihat) B V ρ d g η

/-- **With the per-gap max-normalization** ("followed by the operator's per-gap
max-normalization", SI §S7.4): at `(γ, Λ) = (0, 0)` the normalized Op-M is the normalized Op-H-W. -/
theorem prop3_exact_degeneration_geom_normalized (band : Finset (Fin K)) (clampC : ℝ → ℝ)
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (Ihat : Fin K → ℝ) (B : Matrix ι (Fin K ⊕ Fin K) ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    opM B (normalizeW (metricLawGeom band clampC w Ihat 0)) V 0 ρ d g η =
      opHW B (normalizeW w) d g η := by
  rw [metricLawGeom_zero band clampC hw Ihat, opM_zero_prec]

/-- **Exactness with the deployed metric law** (manuscript: "`Bc = d` holds for every `(γ, Λ, η, g)`
and every data realization"). For full row rank `B` and a base metric `w > 0`, exactness holds at
every trust `γ`, every band/clamp/periodogram, every `λ`, `V`, `ρ̃`, `g`, `η`. -/
theorem mulVec_opM_metricLawGeom {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B)
    (band : Finset (Fin K)) (clampC : ℝ → ℝ) {w : Fin K → ℝ} (hw : ∀ k, 0 < w k)
    (Ihat : Fin K → ℝ) (γ : ℝ) (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ) (d : ι → ℝ)
    (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    B *ᵥ opM B (metricLawGeom band clampC w Ihat γ) V lam ρ d g η = d :=
  mulVec_opM hB (metricLawGeom_pos band clampC hw Ihat γ) V lam ρ d g η

/-! ## Chord exactness on globally affine gaps (SI Thm S7.5(i)) -/

omit [Fintype ο] [DecidableEq ο] in
/-- `d = 0`, `ρ̃ = 0`, `g = 0` ⇒ `r₀ = 0` (SI Thm S7.5(i) proof). -/
theorem opMInnov_zero_data (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (η : Fin K ⊕ Fin K → ℝ) : opMInnov B w V 0 0 0 η = 0 := by
  unfold opMInnov
  simp

/-- **SI Thm S7.5(i), coefficient form.** `d = 0`, `ρ̃ = 0`, `g = 0` give
`c = 0` for **every** `Λ`, `η`, metric `ŵ` and exterior layout `V`, with no rank or positivity
hypothesis ("`ρ̃ ≡ 0`, hence `r₀ = 0` and `y = 0` for any `Λ`"). -/
theorem opM_chord_exact (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam : ο → ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    opM B w V lam 0 0 0 η = 0 := by
  unfold opM opMSoft
  rw [opMInnov_zero_data]
  simp

/-- **Reconstruction form.** Under the hypotheses of `opM_chord_exact` the Op-M reconstruction
is the chord: `p̂ ≡ A` (SI Thm S7.5(i)). -/
theorem recon_opM_chord_exact (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam : ο → ℝ) (η : Fin K ⊕ Fin K → ℝ) (ω : Fin K → ℝ)
    (a b pa pb t : ℝ) : recon ω (opM B w V lam 0 0 0 η) a b pa pb t = chord a b pa pb t := by
  rw [opM_chord_exact, recon_zero]

/-- **SI Thm S7.5(i), packaged (also the Op-M half of manuscript
Prop. 1).** Let the track be globally affine, `X t = U₀ + t • U₁` in a real normed space `E`, observed
in a linear coordinate `π`; let the `R_v ≥ 1` included boundary velocities be exact (`Vel e = U₁`) and
the gate be enabled. Let the exterior residuals be the chord residuals at the exterior anchor times
`t_j` passed through any linear detrend (`ρ̃ = detrend(ρ)`, SI §S7.4 (O1)), and let the soft rows be
`φ(τ_j)`. Then for **every** metric `ŵ` (any `γ`), **every** precision vector `λ` and **every** `η` —
with no rank hypothesis — the Op-M coefficients vanish and the reconstruction is the truth:
`p̂ ≡ A ≡ π ∘ X` ("the exterior channel is self-gating on globally affine data"). -/
theorem thmS75i_chord_exactness {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (π : E →ₗ[ℝ] ℝ) {X : ℝ → E} {U₀ U₁ : E} (hX : ∀ t, X t = U₀ + t • U₁) {a b : ℝ}
    (hab : a ≠ b) (ω : Fin K → ℝ) {Rv : ℕ} (hRv : 0 < Rv) (te s : Fin Rv → ℝ)
    {Vel : Fin Rv → E} (hVel : ∀ e, Vel e = U₁) (ρg ε : ℝ) (tj : ο → ℝ)
    (detrend : (ο → ℝ) →ₗ[ℝ] (ο → ℝ)) (w : Fin K → ℝ) (lam : ο → ℝ)
    (η : Fin K ⊕ Fin K → ℝ) :
    let g := gateValue true ρg ε Vel (chordSlopeVec a b (X a) (X b))
    let ρt := detrend (fun j => π (X (tj j)) - chord a b (π (X a)) (π (X b)) (tj j))
    let c := opM (hermiteB ω a b te s) w (extV ω (fun j => ctime a b (tj j))) lam ρt
      (hermiteD a b (π (X a)) (π (X b)) s (fun e => π (Vel e))) g η
    c = 0 ∧ ∀ t, recon ω c a b (π (X a)) (π (X b)) t = π (X t) := by
  intro g ρt c
  have hA : chordSlopeVec a b (X a) (X b) = U₁ := chordSlopeVec_of_affine hX (Ne.symm hab)
  have hg : g = 0 := gateValue_eq_zero hRv ρg ε (fun e => (hVel e).trans hA.symm)
  have hp : ∀ t, π (X t) = π U₀ + π U₁ * t := fun t => by
    rw [hX t, map_add, map_smul, smul_eq_mul, mul_comm]
  have hchord := (chord_eq_of_affine (p := fun t => π (X t)) hp hab).1
  have hρ : ρt = 0 := by
    have h0 : (fun j => π (X (tj j)) - chord a b (π (X a)) (π (X b)) (tj j)) = 0 := by
      funext j
      rw [hchord (tj j), sub_self, Pi.zero_apply]
    change detrend _ = 0
    rw [h0, map_zero]
  have hd : hermiteD a b (π (X a)) (π (X b)) s (fun e => π (Vel e)) = 0 :=
    hermiteD_eq_zero_of_affine (p := fun t => π (X t)) hp hab s (fun e => by simp [hVel e])
  have hc : c = 0 := by
    change opM _ w _ lam ρt _ g η = 0
    rw [hd, hg, hρ, opM_chord_exact]
  refine ⟨hc, fun t => ?_⟩
  rw [hc, recon_zero]
  exact hchord t

/-- **SI Thm S7.5(ii), the "if" direction / tier C.** With the deployed
conservative gating `Λ_eff = g · Λ`, a jet-straight gap (`d = 0`) with a closed gate (`g = 0`) gives
`c = 0` — chord exactness — for **every** exterior residual `ρ̃` (curved exterior allowed), every `λ`,
metric and `η`. (The converse holds exactly under the condition of `thmS75ii_converse_iff`.) -/
theorem thmS75ii_gated (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    opM B w V ((0 : ℝ) • lam) ρ 0 0 η = 0 := by
  rw [zero_smul, opM_zero_prec, opHW_zero_zero]

/-! ## The influence law and the boundary-conditioned kernel -/

/-- The shrunk dual coefficients `β = Λ^{1/2}(I + G̃)⁻¹Λ^{1/2} r₀` of the exterior anchors. -/
noncomputable def opMDual (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) : ο → ℝ :=
  (sqrtPrec lam * (1 + opMGram B w V lam)⁻¹ * sqrtPrec lam) *ᵥ opMInnov B w V ρ d g η

/-- Boundary-conditioned FQ kernel `K̃(τ, σ) = φ(τ)ᵀ Ŵ P Ŵ φ(σ)`. -/
noncomputable def bcKernel (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ) (ω : Fin K → ℝ)
    (τ σ : ℝ) : ℝ :=
  phi ω τ ⬝ᵥ ((metricW w * proj (B * metricW w) * metricW w) *ᵥ phi ω σ)

/-- **First clause.** `c(Λ) − c(0) = Ŵ y` (for every argument). -/
theorem opM_sub_opM_zero_prec (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) :
    opM B w V lam ρ d g η - opM B w V 0 ρ d g η = metricW w *ᵥ opMSoft B w V lam ρ d g η := by
  rw [opM_zero_prec, opM_eq_opHW_add, add_sub_cancel_left]

/-- `Ŵ y = (Ŵ P Ŵ Vᵀ) β` (used for the influence law; uses `Ŵᵀ = Ŵ`). -/
theorem metricW_mulVec_opMSoft (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) :
    metricW w *ᵥ opMSoft B w V lam ρ d g η =
      ((metricW w * proj (B * metricW w) * metricW w) * Vᵀ) *ᵥ opMDual B w V lam ρ d g η := by
  unfold opMSoft opMDual
  rw [mulVec_mulVec, mulVec_mulVec, transpose_mul, metricW_transpose]
  simp only [Matrix.mul_assoc]

/-- **Influence law (general query row).** For every query row `q`,
`q ⬝ (Ŵ y) = Σ_j (q ⬝ Ŵ P Ŵ V_j) β_j` with `β = Λ^{1/2}(I + G̃)⁻¹Λ^{1/2} r₀`. -/
theorem dotProduct_metricW_opMSoft (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) (q : Fin K ⊕ Fin K → ℝ) :
    q ⬝ᵥ (metricW w *ᵥ opMSoft B w V lam ρ d g η) =
      ∑ j, (q ⬝ᵥ ((metricW w * proj (B * metricW w) * metricW w) *ᵥ V j)) *
        opMDual B w V lam ρ d g η j := by
  rw [metricW_mulVec_opMSoft, dotProduct_mulVec, ← vecMul_vecMul, vecMul_transpose]
  refine Finset.sum_congr rfl fun j _ => ?_
  congr 1
  rw [dotProduct_mulVec]
  exact dotProduct_comm _ _

/-- **Influence law (prediction form).** With soft rows `φ(τ_j)`, the
exterior-channel deformation of the prediction at the query time `τ_q` is exactly
`δp̂(τ_q) = Σ_j K̃(τ_q, τ_j) (Λ^{1/2}(I + G̃)⁻¹Λ^{1/2} r₀)_j`. -/
theorem influence_law (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ) (ω : Fin K → ℝ)
    (τo : ο → ℝ) (lam ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) (τq : ℝ) :
    rcomb ω (opM B w (extV ω τo) lam ρ d g η) τq - rcomb ω (opM B w (extV ω τo) 0 ρ d g η) τq =
      ∑ j, bcKernel B w ω τq (τo j) * opMDual B w (extV ω τo) lam ρ d g η j := by
  have h := opM_sub_opM_zero_prec B w (extV ω τo) lam ρ d g η
  unfold rcomb
  rw [← dotProduct_sub, h]
  exact dotProduct_metricW_opMSoft B w (extV ω τo) lam ρ d g η (phi ω τq)

/-- **Reconstruction form.** `p̂_Λ(t) − p̂_0(t) = Σ_j K̃(τ, τ_j) β_j`, `τ = t − (a+b)/2`
(the exterior-channel deformation of the prediction, exactly). -/
theorem recon_influence_law (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ) (ω : Fin K → ℝ)
    (τo : ο → ℝ) (lam ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ)
    (a b pa pb t : ℝ) :
    recon ω (opM B w (extV ω τo) lam ρ d g η) a b pa pb t -
        recon ω (opM B w (extV ω τo) 0 ρ d g η) a b pa pb t =
      ∑ j, bcKernel B w ω (ctime a b t) (τo j) * opMDual B w (extV ω τo) lam ρ d g η j := by
  rw [recon_eq, recon_eq, ← influence_law]
  ring

/-- `B_i ⬝ (Ŵ P Ŵ v) = 0` for every row `i` (used for kernel annihilation). -/
theorem row_dotProduct_metricW_proj_metricW {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B)
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (i : ι) (v : Fin K ⊕ Fin K → ℝ) :
    B i ⬝ᵥ ((metricW w * proj (B * metricW w) * metricW w) *ᵥ v) = 0 := by
  rw [← mulVec_mulVec, ← mulVec_mulVec]
  exact congrFun (mulVec_metricW_proj_mulVec hB hw (metricW w *ᵥ v)) i

/-- **Kernel annihilation at value rows.** If `φ(τ_e)` is a row of `B`
(a value row), then `K̃(τ_e, ·) ≡ 0`. -/
theorem bcKernel_eq_zero_of_valueRow {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B)
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (ω : Fin K → ℝ) {i : ι} {τe : ℝ}
    (hrow : B i = phi ω τe) (σ : ℝ) : bcKernel B w ω τe σ = 0 := by
  unfold bcKernel
  rw [← hrow]
  exact row_dotProduct_metricW_proj_metricW hB hw i (phi ω σ)

/-- **Kernel annihilation at included velocity rows.** If `s • φ'(τ_e)`
(`s ≠ 0`) is a row of `B`, then `∂_τ K̃(τ, σ)|_{τ = τ_e} = 0` for every `σ`. -/
theorem hasDerivAt_bcKernel_of_derivRow {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B)
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (ω : Fin K → ℝ) {i : ι} {τe s : ℝ} (hs : s ≠ 0)
    (hrow : B i = s • dphi ω τe) (σ : ℝ) :
    HasDerivAt (fun τ => bcKernel B w ω τ σ) 0 τe := by
  have h := row_dotProduct_metricW_proj_metricW hB hw i (phi ω σ)
  rw [hrow, smul_dotProduct, smul_eq_mul] at h
  have h' := (mul_eq_zero.mp h).resolve_left hs
  have hd := hasDerivAt_rcomb ω ((metricW w * proj (B * metricW w) * metricW w) *ᵥ phi ω σ) τe
  rw [h'] at hd
  exact hd

/-! ## Gauge invariance -/

omit [Fintype ο] in
/-- `√(λ/α²) = α⁻¹ √λ` entrywise for `α > 0` (used for gauge invariance). -/
theorem sqrtPrec_div_sq {α : ℝ} (hα : 0 < α) (lam : ο → ℝ) :
    sqrtPrec (fun j => lam j / α ^ 2) = α⁻¹ • sqrtPrec lam := by
  unfold sqrtPrec
  rw [← diagonal_smul]
  congr 1
  funext j
  rw [Pi.smul_apply, smul_eq_mul, Real.sqrt_div' _ (sq_nonneg α), Real.sqrt_sq hα.le,
    div_eq_inv_mul]

/-- **gauge structure.** Under `(ŵ, Λ, η) ↦ (αŵ, Λ/α², η/α)`, `α > 0`, the
Op-M output is invariant, for every `B`, `V`, `ρ̃`, `d`, `g` (no rank hypothesis). -/
theorem opM_gauge {α : ℝ} (hα : 0 < α) (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) :
    opM B (α • w) V (fun j => lam j / α ^ 2) ρ d g (α⁻¹ • η) = opM B w V lam ρ d g η := by
  have hα0 : α ≠ 0 := hα.ne'
  have hW : metricW (α • w) = α • metricW w := metricW_smul α w
  have hBt : B * metricW (α • w) = α • (B * metricW w) := by rw [hW, Matrix.mul_smul]
  have hVt : V * metricW (α • w) = α • (V * metricW w) := by rw [hW, Matrix.mul_smul]
  have hP : proj (B * metricW (α • w)) = proj (B * metricW w) := by rw [hBt, proj_smul hα0]
  have hpinv : pinvR (B * metricW (α • w)) = α⁻¹ • pinvR (B * metricW w) := by
    rw [hBt, pinvR_smul hα0]
  have hm : pinvR (B * metricW (α • w)) *ᵥ d + g • (proj (B * metricW (α • w)) *ᵥ (α⁻¹ • η)) =
      α⁻¹ • (pinvR (B * metricW w) *ᵥ d + g • (proj (B * metricW w) *ᵥ η)) := by
    rw [hpinv, hP, smul_mulVec, mulVec_smul, smul_add, smul_comm g α⁻¹]
  have hr : opMInnov B (α • w) V ρ d g (α⁻¹ • η) = opMInnov B w V ρ d g η := by
    unfold opMInnov
    rw [hm, hVt, smul_mulVec, mulVec_smul, smul_smul, mul_inv_cancel₀ hα0, one_smul]
  have hG : opMGram B (α • w) V (fun j => lam j / α ^ 2) = opMGram B w V lam := by
    unfold opMGram
    rw [sqrtPrec_div_sq hα, hVt, hP, transpose_smul]
    simp only [Matrix.smul_mul, Matrix.mul_smul, smul_smul]
    convert one_smul ℝ _ using 2
    field_simp
  have hy : opMSoft B (α • w) V (fun j => lam j / α ^ 2) ρ d g (α⁻¹ • η) =
      α⁻¹ • opMSoft B w V lam ρ d g η := by
    unfold opMSoft
    rw [hr, hG, sqrtPrec_div_sq hα, hVt, hP, transpose_smul]
    simp only [Matrix.smul_mul, Matrix.mul_smul, smul_smul, smul_mulVec]
    congr 1
    field_simp
  unfold opM
  rw [hm, hy, hW, ← smul_add, smul_mulVec, mulVec_smul, smul_smul, mul_inv_cancel₀ hα0, one_smul]

/-! ## Op-M is the unique minimizer of eq. (7) (SI Prop. S7.1, Thm S7.2)

Route of the SI proof: substitute `u = Ŵ⁻¹c`; the feasible set is `u₀ + range P`; the
objective splits orthogonally; stationarity plus the push-through identity give the closed form.
Here the minimality is proved through the exact Pythagorean identity
`J(c) = J(c⋆) + ‖Ŵ⁻¹(c − c⋆)‖² + Σ_j λ_j (φ(τ_j)ᵀ(c − c⋆))²` on the feasible set, an exact form
of the SI's uniqueness argument (a strictly convex quadratic on the feasible affine set, with
Hessian `⪰ 2I` there). -/

section CoreMin

variable {κ : Type*} [Fintype κ] [DecidableEq κ]

/-- `soft_pythagoras` with the minimizer given by an equation (generic auxiliary lemma). -/
theorem soft_pythagoras_of_eq {P : Matrix κ κ ℝ} (hPt : Pᵀ = P) (hPP : P * P = P)
    {S : Matrix ο ο ℝ} (hSt : Sᵀ = S) (Vt : Matrix ο κ ℝ) {u0 : κ → ℝ} (hPu0 : P *ᵥ u0 = 0)
    (η : κ → ℝ) (g : ℝ) (ρ : ο → ℝ) {h : κ → ℝ} (hh : P *ᵥ h = h) {u : κ → ℝ}
    (hu : u = u0 + g • (P *ᵥ η) + (P * Vtᵀ * S * (1 + S * Vt * P * Vtᵀ * S)⁻¹ * S) *ᵥ
      (ρ - Vt *ᵥ (u0 + g • (P *ᵥ η)))) :
    (u + h - g • η) ⬝ᵥ (u + h - g • η) +
        (S *ᵥ (Vt *ᵥ (u + h) - ρ)) ⬝ᵥ (S *ᵥ (Vt *ᵥ (u + h) - ρ)) =
      ((u - g • η) ⬝ᵥ (u - g • η) + (S *ᵥ (Vt *ᵥ u - ρ)) ⬝ᵥ (S *ᵥ (Vt *ᵥ u - ρ))) +
        h ⬝ᵥ h + (S *ᵥ (Vt *ᵥ h)) ⬝ᵥ (S *ᵥ (Vt *ᵥ h)) := by
  subst hu
  exact soft_pythagoras hPt hPP hSt Vt hPu0 η g ρ hh

omit [DecidableEq κ] in
/-- **SI §S7.1 proof, orthogonal split (generic).** For symmetric idempotent `P`, `P u₀ = 0` and
`y ∈ range P`: `‖u₀ + gPη + y − gη‖² = ‖u₀ − g(I − P)η‖² + ‖y‖²`. -/
theorem orth_split {P : Matrix κ κ ℝ} (hPt : Pᵀ = P) (hPP : P * P = P) {u0 : κ → ℝ}
    (hPu0 : P *ᵥ u0 = 0) (η : κ → ℝ) (g : ℝ) {y : κ → ℝ} (hy : P *ᵥ y = y) :
    (u0 + g • (P *ᵥ η) + y - g • η) ⬝ᵥ (u0 + g • (P *ᵥ η) + y - g • η) =
      (u0 - g • (η - P *ᵥ η)) ⬝ᵥ (u0 - g • (η - P *ᵥ η)) + y ⬝ᵥ y := by
  have hPP' : P *ᵥ (P *ᵥ η) = P *ᵥ η := by rw [mulVec_mulVec, hPP]
  have hcross : (u0 - g • (η - P *ᵥ η)) ⬝ᵥ y = 0 := by
    rw [← hy, ← mulVec_dotProduct_of_transpose_eq hPt, mulVec_sub, mulVec_smul, mulVec_sub, hPP',
      sub_self, smul_zero, hPu0, sub_zero, zero_dotProduct]
  have e : u0 + g • (P *ᵥ η) + y - g • η = (u0 - g • (η - P *ᵥ η)) + y := by
    rw [smul_sub]
    abel
  rw [e, add_dotProduct, dotProduct_add, dotProduct_add, dotProduct_comm y (u0 - g • (η - P *ᵥ η)),
    hcross]
  ring

omit [DecidableEq κ] in
/-- The soft gain in `A`-form: with `A = S Ṽ P`, `P Ṽᵀ S (I + S Ṽ P Ṽᵀ S)⁻¹ S = Aᵀ (I + AAᵀ)⁻¹ S`,
`P Ṽᵀ S S Ṽ P = AᵀA` and `P Ṽᵀ S S = Aᵀ S` (generic auxiliary lemma). -/
theorem soft_gain_A {P : Matrix κ κ ℝ} (hPt : Pᵀ = P) (hPP : P * P = P) {S : Matrix ο ο ℝ}
    (hSt : Sᵀ = S) (Vt : Matrix ο κ ℝ) :
    P * Vtᵀ * S * (1 + S * Vt * P * Vtᵀ * S)⁻¹ * S =
        (S * Vt * P)ᵀ * (1 + (S * Vt * P) * (S * Vt * P)ᵀ)⁻¹ * S ∧
      P * Vtᵀ * (S * S) * Vt * P = (S * Vt * P)ᵀ * (S * Vt * P) ∧
      P * Vtᵀ * (S * S) = (S * Vt * P)ᵀ * S := by
  have hAT : (S * Vt * P)ᵀ = P * Vtᵀ * S := by
    rw [transpose_mul, transpose_mul, hPt, hSt, Matrix.mul_assoc]
  have hG : S * Vt * P * Vtᵀ * S = (S * Vt * P) * (S * Vt * P)ᵀ := by
    rw [hAT]
    simp only [Matrix.mul_assoc]
    rw [← Matrix.mul_assoc P P, hPP]
  refine ⟨?_, ?_, ?_⟩
  · rw [hG, hAT]
  · rw [hAT]
    simp only [Matrix.mul_assoc]
  · rw [hAT]
    simp only [Matrix.mul_assoc]

/-- **SI §S7.1 proof, stationarity (generic).** With `y = P Ṽᵀ S (I + G̃)⁻¹ S r₀`:
`y + P Ṽᵀ S S Ṽ P y = P Ṽᵀ S S r₀`, i.e. `(I_P + PṼᵀΛṼP) y = PṼᵀΛ r₀` when `S S = Λ`. -/
theorem soft_stationarity {P : Matrix κ κ ℝ} (hPt : Pᵀ = P) (hPP : P * P = P)
    {S : Matrix ο ο ℝ} (hSt : Sᵀ = S) (Vt : Matrix ο κ ℝ) (r0 : ο → ℝ) :
    (P * Vtᵀ * S * (1 + S * Vt * P * Vtᵀ * S)⁻¹ * S) *ᵥ r0 +
        (P * Vtᵀ * (S * S) * Vt * P) *ᵥ ((P * Vtᵀ * S * (1 + S * Vt * P * Vtᵀ * S)⁻¹ * S) *ᵥ r0) =
      (P * Vtᵀ * (S * S)) *ᵥ r0 := by
  obtain ⟨h1, h2, h3⟩ := soft_gain_A hPt hPP hSt Vt
  rw [h1, h2, h3]
  set A : Matrix ο κ ℝ := S * Vt * P
  have key : (1 + Aᵀ * A) * (Aᵀ * (1 + A * Aᵀ)⁻¹ * S) = Aᵀ * S := by
    rw [← Matrix.mul_assoc, one_add_transpose_mul_self_mul_gain]
  calc (Aᵀ * (1 + A * Aᵀ)⁻¹ * S) *ᵥ r0 + (Aᵀ * A) *ᵥ ((Aᵀ * (1 + A * Aᵀ)⁻¹ * S) *ᵥ r0)
      = ((1 + Aᵀ * A) * (Aᵀ * (1 + A * Aᵀ)⁻¹ * S)) *ᵥ r0 := by
        rw [Matrix.add_mul, Matrix.one_mul, add_mulVec, mulVec_mulVec]
    _ = (Aᵀ * S) *ᵥ r0 := by rw [key]

/-- **SI §S7.1 proof, push-through step (generic).** The `N_o × N_o` form of `y` equals the
`2K × 2K` form: `P Ṽᵀ S (I + S Ṽ P Ṽᵀ S)⁻¹ S = (I + P Ṽᵀ S S Ṽ P)⁻¹ P Ṽᵀ S S`. -/
theorem soft_pushThrough {P : Matrix κ κ ℝ} (hPt : Pᵀ = P) (hPP : P * P = P)
    {S : Matrix ο ο ℝ} (hSt : Sᵀ = S) (Vt : Matrix ο κ ℝ) :
    P * Vtᵀ * S * (1 + S * Vt * P * Vtᵀ * S)⁻¹ * S =
      (1 + P * Vtᵀ * (S * S) * Vt * P)⁻¹ * (P * Vtᵀ * (S * S)) := by
  obtain ⟨h1, h2, h3⟩ := soft_gain_A hPt hPP hSt Vt
  rw [h1, h2, h3, ← Matrix.mul_assoc, one_add_transpose_mul_self_inv_mul]

end CoreMin

/-- Whitened coordinates of Op-M: `Ŵ⁻¹ c = u₀ + g P η + y` for `ŵ > 0` (SI §S7.1 proof, substitution
`u = Ŵ⁻¹c`). -/
theorem metricW_inv_mulVec_opM {w : Fin K → ℝ} (hw : ∀ k, 0 < w k)
    (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ)
    (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    (metricW w)⁻¹ *ᵥ opM B w V lam ρ d g η =
      pinvR (B * metricW w) *ᵥ d + g • (proj (B * metricW w) *ᵥ η) +
        opMSoft B w V lam ρ d g η := by
  unfold opM
  rw [mulVec_mulVec, nonsing_inv_mul _ (isUnit_det_metricW hw), one_mulVec]

/-- The objective (7) in whitened coordinates `u = Ŵ⁻¹c` (SI §S7.1 proof):
`J(c) = ‖u − gη‖² + ‖Λ^{1/2}(Ṽu − ρ̃)‖²` for `ŵ > 0`, `Λ ⪰ 0`. -/
theorem opMObj_eq_whitened {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (V : Matrix ο (Fin K ⊕ Fin K) ℝ)
    {lam : ο → ℝ} (hlam : ∀ j, 0 ≤ lam j) (ρ : ο → ℝ) (g : ℝ) (η c : Fin K ⊕ Fin K → ℝ) :
    opMObj w V lam ρ g η c =
      ((metricW w)⁻¹ *ᵥ c - g • η) ⬝ᵥ ((metricW w)⁻¹ *ᵥ c - g • η) +
        (sqrtPrec lam *ᵥ ((V * metricW w) *ᵥ ((metricW w)⁻¹ *ᵥ c) - ρ)) ⬝ᵥ
          (sqrtPrec lam *ᵥ ((V * metricW w) *ᵥ ((metricW w)⁻¹ *ᵥ c) - ρ)) := by
  have hVc : (V * metricW w) *ᵥ ((metricW w)⁻¹ *ᵥ c) = V *ᵥ c := by
    rw [mulVec_mulVec, Matrix.mul_assoc, mul_nonsing_inv _ (isUnit_det_metricW hw),
      Matrix.mul_one]
  unfold opMObj
  rw [hVc, ← sum_prec_sq_eq hlam]
  rfl

/-- **SI §S7.1 proof, orthogonal split.** For every `y ∈ null(B̃)` (= `range P`), the objective
at `c = Ŵ(u₀ + gPη + y)` splits as `‖u₀ − g(I − P)η‖² + ‖y‖² + ‖Λ^{1/2}(Ṽy − r₀)‖²`. -/
theorem opMObj_split {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B) {w : Fin K → ℝ}
    (hw : ∀ k, 0 < w k) (V : Matrix ο (Fin K ⊕ Fin K) ℝ) {lam : ο → ℝ} (hlam : ∀ j, 0 ≤ lam j)
    (ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) {y : Fin K ⊕ Fin K → ℝ}
    (hy : (B * metricW w) *ᵥ y = 0) :
    opMObj w V lam ρ g η (metricW w *ᵥ
        (pinvR (B * metricW w) *ᵥ d + g • (proj (B * metricW w) *ᵥ η) + y)) =
      (pinvR (B * metricW w) *ᵥ d - g • (η - proj (B * metricW w) *ᵥ η)) ⬝ᵥ
          (pinvR (B * metricW w) *ᵥ d - g • (η - proj (B * metricW w) *ᵥ η)) + y ⬝ᵥ y +
        (sqrtPrec lam *ᵥ ((V * metricW w) *ᵥ y - opMInnov B w V ρ d g η)) ⬝ᵥ
          (sqrtPrec lam *ᵥ ((V * metricW w) *ᵥ y - opMInnov B w V ρ d g η)) := by
  have hBt := fullRowRank_mul_metricW hB hw
  have hWW : (metricW w)⁻¹ *ᵥ (metricW w *ᵥ
      (pinvR (B * metricW w) *ᵥ d + g • (proj (B * metricW w) *ᵥ η) + y)) =
      pinvR (B * metricW w) *ᵥ d + g • (proj (B * metricW w) *ᵥ η) + y := by
    rw [mulVec_mulVec, nonsing_inv_mul _ (isUnit_det_metricW hw), one_mulVec]
  have hres : (V * metricW w) *ᵥ
      (pinvR (B * metricW w) *ᵥ d + g • (proj (B * metricW w) *ᵥ η) + y) - ρ =
      (V * metricW w) *ᵥ y - opMInnov B w V ρ d g η := by
    unfold opMInnov
    rw [mulVec_add]
    abel
  rw [opMObj_eq_whitened hw V hlam, hWW, hres,
    orth_split transpose_proj (proj_mul_proj hBt) (proj_mulVec_pinvR_mulVec hBt d) η g
      ((proj_mulVec_eq_self_iff hBt y).mpr hy)]

/-- **SI §S7.1 proof, stationarity.** The exterior term solves the normal equation
`(I + P Ṽᵀ Λ Ṽ P) y = P Ṽᵀ Λ r₀` (for full row rank `B`, `ŵ > 0`, `Λ ⪰ 0`). -/
theorem opMSoft_stationarity {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B)
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (V : Matrix ο (Fin K ⊕ Fin K) ℝ) {lam : ο → ℝ}
    (hlam : ∀ j, 0 ≤ lam j) (ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    opMSoft B w V lam ρ d g η +
        (proj (B * metricW w) * (V * metricW w)ᵀ * diagonal lam * (V * metricW w) *
          proj (B * metricW w)) *ᵥ opMSoft B w V lam ρ d g η =
      (proj (B * metricW w) * (V * metricW w)ᵀ * diagonal lam) *ᵥ opMInnov B w V ρ d g η := by
  rw [← sqrtPrec_mul_self hlam]
  exact soft_stationarity transpose_proj (proj_mul_proj (fullRowRank_mul_metricW hB hw))
    (sqrtPrec_transpose lam) (V * metricW w) (opMInnov B w V ρ d g η)

/-- **SI §S7.1 proof, push-through form.** The `N_o × N_o` closed form (8) equals the
`2K × 2K` form `y = (I + P Ṽᵀ Λ Ṽ P)⁻¹ P Ṽᵀ Λ r₀` (full row rank `B`, `ŵ > 0`, `Λ ⪰ 0`). -/
theorem opMSoft_pushThrough {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B)
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (V : Matrix ο (Fin K ⊕ Fin K) ℝ) {lam : ο → ℝ}
    (hlam : ∀ j, 0 ≤ lam j) (ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    opMSoft B w V lam ρ d g η =
      ((1 + proj (B * metricW w) * (V * metricW w)ᵀ * diagonal lam * (V * metricW w) *
          proj (B * metricW w))⁻¹ * (proj (B * metricW w) * (V * metricW w)ᵀ * diagonal lam)) *ᵥ
        opMInnov B w V ρ d g η := by
  rw [← sqrtPrec_mul_self hlam, ← soft_pushThrough transpose_proj
    (proj_mul_proj (fullRowRank_mul_metricW hB hw)) (sqrtPrec_transpose lam) (V * metricW w)]
  rfl

/-- **The Pythagorean identity of Op-M (SI Prop. S7.1 / Thm S7.2).** For full
row rank `B`, `ŵ > 0` and `Λ ⪰ 0`, every feasible `c` (`B c = d`) satisfies
`J(c) = J(c⋆) + ‖Ŵ⁻¹(c − c⋆)‖² + Σ_j λ_j (φ(τ_j)ᵀ(c − c⋆))²` with `c⋆ = opM …` (eq. (8)). -/
theorem opMObj_pythagoras {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B)
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (V : Matrix ο (Fin K ⊕ Fin K) ℝ) {lam : ο → ℝ}
    (hlam : ∀ j, 0 ≤ lam j) (ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ)
    {c : Fin K ⊕ Fin K → ℝ} (hc : B *ᵥ c = d) :
    opMObj w V lam ρ g η c =
      opMObj w V lam ρ g η (opM B w V lam ρ d g η) +
        ((metricW w)⁻¹ *ᵥ (c - opM B w V lam ρ d g η)) ⬝ᵥ
          ((metricW w)⁻¹ *ᵥ (c - opM B w V lam ρ d g η)) +
        ∑ j, lam j * (V *ᵥ (c - opM B w V lam ρ d g η)) j ^ 2 := by
  have hBt := fullRowRank_mul_metricW hB hw
  set cs := opM B w V lam ρ d g η with hcs
  set h := (metricW w)⁻¹ *ᵥ (c - cs) with hh_def
  have hBh : (B * metricW w) *ᵥ h = 0 :=
    mul_metricW_mulVec_inv hw (by rw [mulVec_sub, hc, hcs, mulVec_opM hB hw, sub_self])
  have hPh : proj (B * metricW w) *ᵥ h = h := proj_mulVec_of_mulVec_eq_zero hBh
  have hu : (metricW w)⁻¹ *ᵥ cs =
      pinvR (B * metricW w) *ᵥ d + g • (proj (B * metricW w) *ᵥ η) +
        (proj (B * metricW w) * (V * metricW w)ᵀ * sqrtPrec lam *
          (1 + sqrtPrec lam * (V * metricW w) * proj (B * metricW w) * (V * metricW w)ᵀ *
            sqrtPrec lam)⁻¹ * sqrtPrec lam) *ᵥ
          (ρ - (V * metricW w) *ᵥ
            (pinvR (B * metricW w) *ᵥ d + g • (proj (B * metricW w) *ᵥ η))) := by
    rw [hcs, metricW_inv_mulVec_opM hw]
    rfl
  have hsplit : (metricW w)⁻¹ *ᵥ c = (metricW w)⁻¹ *ᵥ cs + h := by
    rw [hh_def, mulVec_sub]
    abel
  have key := soft_pythagoras_of_eq transpose_proj (proj_mul_proj hBt) (sqrtPrec_transpose lam)
    (V * metricW w) (proj_mulVec_pinvR_mulVec hBt d) η g ρ hPh hu
  have hVh : (V * metricW w) *ᵥ h = V *ᵥ (c - cs) := by
    rw [hh_def, mulVec_mulVec, Matrix.mul_assoc, mul_nonsing_inv _ (isUnit_det_metricW hw),
      Matrix.mul_one]
  rw [opMObj_eq_whitened hw V hlam, opMObj_eq_whitened hw V hlam, hsplit, key, hVh,
    sum_prec_sq_eq hlam]

/-- **Minimality (SI Prop. S7.1).** `J(c⋆) ≤ J(c)` for every feasible `c`
(full row rank `B`, `ŵ > 0`, `Λ ⪰ 0`; every `V`, `ρ̃`, `d`, `g`, `η`). -/
theorem opMObj_opM_le {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B) {w : Fin K → ℝ}
    (hw : ∀ k, 0 < w k) (V : Matrix ο (Fin K ⊕ Fin K) ℝ) {lam : ο → ℝ}
    (hlam : ∀ j, 0 ≤ lam j) (ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ)
    {c : Fin K ⊕ Fin K → ℝ} (hc : B *ᵥ c = d) :
    opMObj w V lam ρ g η (opM B w V lam ρ d g η) ≤ opMObj w V lam ρ g η c := by
  rw [opMObj_pythagoras hB hw V hlam ρ d g η hc]
  have h1 := dotProduct_self_nonneg' ((metricW w)⁻¹ *ᵥ (c - opM B w V lam ρ d g η))
  have h2 : 0 ≤ ∑ j, lam j * (V *ᵥ (c - opM B w V lam ρ d g η)) j ^ 2 :=
    Finset.sum_nonneg fun j _ => mul_nonneg (hlam j) (sq_nonneg _)
  linarith

/-- **Uniqueness (SI Prop. S7.1 "the unique minimizer"; Thm S7.2).** A
feasible `c` with `J(c) ≤ J(c⋆)` equals `c⋆`. -/
theorem eq_opM_of_opMObj_le {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B)
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (V : Matrix ο (Fin K ⊕ Fin K) ℝ) {lam : ο → ℝ}
    (hlam : ∀ j, 0 ≤ lam j) (ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ)
    {c : Fin K ⊕ Fin K → ℝ} (hc : B *ᵥ c = d)
    (hle : opMObj w V lam ρ g η c ≤ opMObj w V lam ρ g η (opM B w V lam ρ d g η)) :
    c = opM B w V lam ρ d g η := by
  rw [opMObj_pythagoras hB hw V hlam ρ d g η hc] at hle
  have h1 := dotProduct_self_nonneg' ((metricW w)⁻¹ *ᵥ (c - opM B w V lam ρ d g η))
  have h2 : 0 ≤ ∑ j, lam j * (V *ᵥ (c - opM B w V lam ρ d g η)) j ^ 2 :=
    Finset.sum_nonneg fun j _ => mul_nonneg (hlam j) (sq_nonneg _)
  have h0 : ((metricW w)⁻¹ *ᵥ (c - opM B w V lam ρ d g η)) ⬝ᵥ
      ((metricW w)⁻¹ *ᵥ (c - opM B w V lam ρ d g η)) = 0 := by linarith
  have hz := dotProduct_self_eq_zero.mp h0
  have hz' : metricW w *ᵥ ((metricW w)⁻¹ *ᵥ (c - opM B w V lam ρ d g η)) = 0 := by
    rw [hz, mulVec_zero]
  rw [mulVec_mulVec, mul_nonsing_inv _ (isUnit_det_metricW hw), one_mulVec] at hz'
  exact sub_eq_zero.mp hz'

/-- **SI Prop. S7.1 / Thm S7.2, bundled (well-posedness and closed form).**
For full row rank `B`, every metric `ŵ > 0` and every `Λ ⪰ 0`, and for every exterior layout `V`,
residuals `ρ̃`, data `d`, gate `g` and prior mean `η`: the closed form (8) is feasible, minimizes the
objective (7) over `{c | B c = d}`, and is the unique feasible point attaining the minimum. -/
theorem opM_isUniqueMinimizer {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B)
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (V : Matrix ο (Fin K ⊕ Fin K) ℝ) {lam : ο → ℝ}
    (hlam : ∀ j, 0 ≤ lam j) (ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    B *ᵥ opM B w V lam ρ d g η = d ∧
      (∀ c, B *ᵥ c = d →
        opMObj w V lam ρ g η (opM B w V lam ρ d g η) ≤ opMObj w V lam ρ g η c) ∧
      (∀ c, B *ᵥ c = d →
        opMObj w V lam ρ g η c = opMObj w V lam ρ g η (opM B w V lam ρ d g η) →
          c = opM B w V lam ρ d g η) :=
  ⟨mulVec_opM hB hw V lam ρ d g η, fun _ hc => opMObj_opM_le hB hw V hlam ρ d g η hc,
    fun _ hc heq => eq_opM_of_opMObj_le hB hw V hlam ρ d g η hc heq.le⟩

/-! ## Further structure: the kernel decomposition and the exact first-order split -/

/-- Comb autocorrelation in the metric `ŵ`: `κ_Ŵ(δ) = Σ_k ŵ_k² cos(ω_k δ)`. -/
noncomputable def combAutocorr (w ω : Fin K → ℝ) (δ : ℝ) : ℝ :=
  ∑ k, w k ^ 2 * Real.cos (ω k * δ)

/-- **Definition of `K̃`.** `K̃(τ, σ) = κ_Ŵ(τ − σ) − q(τ)ᵀ M⁻¹ q(σ)` with
`q(τ) = B̃ Ŵ φ(τ)` and `M = B̃ B̃ᵀ` (with `M⁻¹`, which equals `M⁺` under full row rank).
Holds for every `B` and `ŵ` (no hypothesis). -/
theorem bcKernel_eq_autocorr_sub (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w ω : Fin K → ℝ)
    (τ σ : ℝ) :
    bcKernel B w ω τ σ = combAutocorr w ω (τ - σ) -
      ((B * metricW w) *ᵥ (metricW w *ᵥ phi ω τ)) ⬝ᵥ
        (((B * metricW w) * (B * metricW w)ᵀ)⁻¹ *ᵥ
          ((B * metricW w) *ᵥ (metricW w *ᵥ phi ω σ))) := by
  have hsplit : metricW w * proj (B * metricW w) * metricW w =
      metricW w * metricW w - ((B * metricW w) * metricW w)ᵀ *
        ((B * metricW w) * (B * metricW w)ᵀ)⁻¹ * ((B * metricW w) * metricW w) := by
    unfold proj pinvR
    rw [transpose_mul (B * metricW w) (metricW w), metricW_transpose]
    simp only [Matrix.mul_sub, Matrix.sub_mul, Matrix.mul_one, Matrix.mul_assoc]
  have hauto : phi ω τ ⬝ᵥ ((metricW w * metricW w) *ᵥ phi ω σ) = combAutocorr w ω (τ - σ) := by
    rw [metricW_mul_metricW]
    unfold metricW combAutocorr
    simp only [mulVec_diagonal, dotProduct, Fintype.sum_sum_type, phi_inl, phi_inr,
      Sum.elim_inl, Sum.elim_inr, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [mul_sub, Real.cos_sub]
    ring
  unfold bcKernel
  rw [hsplit, sub_mulVec, dotProduct_sub, hauto]
  congr 1
  rw [← mulVec_mulVec, ← mulVec_mulVec, dotProduct_mulVec, vecMul_transpose]
  simp only [← mulVec_mulVec]

/-- **The first-order term made exact.** For `Λ ⪰ 0`,
`Λ^{1/2}(I + G̃)⁻¹Λ^{1/2} r₀ = Λ r₀ − Λ^{1/2} G̃ (I + G̃)⁻¹ Λ^{1/2} r₀`; hence
`δp̂(τ_q) = Σ_j λ_j K̃(τ_q, τ_j) r₀_j − Σ_j K̃(τ_q, τ_j) (Λ^{1/2}G̃(I + G̃)⁻¹Λ^{1/2}r₀)_j`
exactly (the `O(‖Λ‖²)` remainder of the first-order expansion is the second sum; its norm bound is
not formalized). -/
theorem opMDual_eq_first_order (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) {lam : ο → ℝ} (hlam : ∀ j, 0 ≤ lam j) (ρ : ο → ℝ)
    (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    opMDual B w V lam ρ d g η = diagonal lam *ᵥ opMInnov B w V ρ d g η -
      (sqrtPrec lam * opMGram B w V lam * (1 + opMGram B w V lam)⁻¹ * sqrtPrec lam) *ᵥ
        opMInnov B w V ρ d g η := by
  have hu := isUnit_det_one_add_opMGram B w V lam
  have h1 : (1 + opMGram B w V lam)⁻¹ + opMGram B w V lam * (1 + opMGram B w V lam)⁻¹ = 1 := by
    calc (1 + opMGram B w V lam)⁻¹ + opMGram B w V lam * (1 + opMGram B w V lam)⁻¹
        = (1 + opMGram B w V lam) * (1 + opMGram B w V lam)⁻¹ := by
          rw [Matrix.add_mul, Matrix.one_mul]
      _ = 1 := mul_nonsing_inv _ hu
  have hinv : (1 + opMGram B w V lam)⁻¹ =
      1 - opMGram B w V lam * (1 + opMGram B w V lam)⁻¹ := eq_sub_of_add_eq h1
  unfold opMDual
  rw [← sub_mulVec]
  congr 1
  conv_lhs => rw [hinv]
  rw [Matrix.mul_sub, Matrix.sub_mul, Matrix.mul_one, sqrtPrec_mul_self hlam]
  simp only [Matrix.mul_assoc]

/-! ## SI Thm S7.5(ii): the exact converse -/

omit [Fintype ο] [DecidableEq ο] in
/-- With `d = 0` and `g = 0` the innovation is the residual itself: `r₀ = ρ̃`. -/
theorem opMInnov_zero_rhs_gate (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (ρ : ο → ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    opMInnov B w V ρ 0 0 η = ρ := by
  unfold opMInnov
  simp

/-- **SI Thm S7.5(ii), converse.** On a jet-straight gap (`d = 0`) with the gate closed
(`g = 0`), for full row rank `B`, `ŵ > 0`, `Λ ⪰ 0` and **any** exterior residuals `ρ̃`, the Op-M
coefficients vanish (chord exactness `p̂ ≡ A`) **iff** `P Ṽᵀ Λ ρ̃ = 0`. When `P Ṽᵀ` is injective
this reduces to `λ_j ρ̃_j = 0` for every anchor (`thmS75ii_converse_of_injective`); without that
condition the unqualified converse "`c = 0` only if the gated precisions vanish" fails
(`thmS75ii_only_if_fails`). -/
theorem thmS75ii_converse_iff {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B)
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (V : Matrix ο (Fin K ⊕ Fin K) ℝ) {lam : ο → ℝ}
    (hlam : ∀ j, 0 ≤ lam j) (ρ : ο → ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    opM B w V lam ρ 0 0 η = 0 ↔
      (proj (B * metricW w) * (V * metricW w)ᵀ * diagonal lam) *ᵥ ρ = 0 := by
  have hstat := opMSoft_stationarity hB hw V hlam ρ 0 0 η
  have hpt := opMSoft_pushThrough hB hw V hlam ρ 0 0 η
  rw [opMInnov_zero_rhs_gate] at hstat hpt
  have hc : opM B w V lam ρ 0 0 η = metricW w *ᵥ opMSoft B w V lam ρ 0 0 η := by
    rw [opM_eq_opHW_add, opHW_zero_zero, zero_add]
  rw [hc]
  constructor
  · intro h
    have hy : opMSoft B w V lam ρ 0 0 η = 0 := by
      have h2 : (metricW w)⁻¹ *ᵥ (metricW w *ᵥ opMSoft B w V lam ρ 0 0 η) = 0 := by
        rw [h, mulVec_zero]
      rwa [mulVec_mulVec, nonsing_inv_mul _ (isUnit_det_metricW hw), one_mulVec] at h2
    rw [hy, mulVec_zero, add_zero] at hstat
    exact hstat.symm
  · intro h
    rw [hpt, ← mulVec_mulVec, h, mulVec_zero, mulVec_zero]

/-- **SI Thm S7.5(ii) under a non-degeneracy hypothesis.** If the whitened exterior rows are linearly
independent modulo the hard rows (`P Ṽᵀ x = 0 → x = 0`), then on a jet-straight gap with closed gate:
`c = 0 ⟺ λ_j ρ̃_j = 0` for every anchor `j` (so `c = 0` forces `λ_j = 0` at every anchor with
`ρ̃_j ≠ 0`). -/
theorem thmS75ii_converse_of_injective {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B)
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (V : Matrix ο (Fin K ⊕ Fin K) ℝ) {lam : ο → ℝ}
    (hlam : ∀ j, 0 ≤ lam j) (ρ : ο → ℝ) (η : Fin K ⊕ Fin K → ℝ)
    (hinj : ∀ x : ο → ℝ, (proj (B * metricW w) * (V * metricW w)ᵀ) *ᵥ x = 0 → x = 0) :
    opM B w V lam ρ 0 0 η = 0 ↔ ∀ j, lam j * ρ j = 0 := by
  rw [thmS75ii_converse_iff hB hw V hlam ρ η, ← mulVec_mulVec]
  constructor
  · intro h j
    have h2 := congrFun (hinj _ h) j
    rwa [mulVec_diagonal] at h2
  · intro h
    have h2 : diagonal lam *ᵥ ρ = 0 := by
      funext j
      rw [mulVec_diagonal, h j, Pi.zero_apply]
    rw [h2, mulVec_zero]

/-- If the hard rows span the whole coefficient space (`P = 0`, e.g. `R = 2K` independent rows), the
exterior channel is inert: `y = 0` for every `Λ`, `ρ̃` (no hypothesis). -/
theorem opMSoft_eq_zero_of_proj_eq_zero {B : Matrix ι (Fin K ⊕ Fin K) ℝ} {w : Fin K → ℝ}
    (hP : proj (B * metricW w) = 0) (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ)
    (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) : opMSoft B w V lam ρ d g η = 0 := by
  unfold opMSoft
  rw [hP]
  simp

/-- **Counterexample to the unqualified converse of SI Thm S7.5(ii).** With `K = 1`, the identity
hard block (full row rank, two rows spanning the 2-dimensional coefficient space), unit metric, one
exterior anchor with precision `λ = 1 ≠ 0` and residual `ρ̃ = 1 ≠ 0`, and `d = 0`, `g = 0`: Op-M
returns `c = 0` (chord exactness) although the precisions do not vanish. -/
theorem thmS75ii_only_if_fails :
    ∃ (B : Matrix (Fin 1 ⊕ Fin 1) (Fin 1 ⊕ Fin 1) ℝ) (V : Matrix (Fin 1) (Fin 1 ⊕ Fin 1) ℝ)
      (lam ρ : Fin 1 → ℝ), FullRowRank B ∧ (∀ j, 0 ≤ lam j) ∧ (∀ j, lam j ≠ 0) ∧
        (∀ j, ρ j ≠ 0) ∧ opM B (fun _ => 1) V lam ρ 0 0 0 = 0 := by
  refine ⟨1, Matrix.of fun _ _ => 1, fun _ => 1, fun _ => 1, ?_, fun _ => zero_le_one,
    fun _ => one_ne_zero, fun _ => one_ne_zero, ?_⟩
  · unfold FullRowRank
    simp
  · have hP : proj ((1 : Matrix (Fin 1 ⊕ Fin 1) (Fin 1 ⊕ Fin 1) ℝ) *
        metricW (fun _ : Fin 1 => (1 : ℝ))) = 0 := by
      rw [metricW_one, Matrix.mul_one]
      unfold proj pinvR
      simp
    rw [opM_eq_opHW_add, opHW_zero_zero, zero_add, opMSoft_eq_zero_of_proj_eq_zero hP,
      mulVec_zero]

/-! ## Single soft anchor: the Sherman–Morrison response -/

omit [Fintype ο] [DecidableEq ο] in
/-- Rows of the whitened exterior matrix: `(V Ŵ)_j = Ŵ V_j`. -/
theorem row_mul_metricW (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ) (j : ο) :
    (V * metricW w) j = metricW w *ᵥ V j := by
  funext k
  unfold metricW
  rw [mul_diagonal, mulVec_diagonal, mul_comm]

omit [Fintype ο] [DecidableEq ο] in
/-- Entries of a congruence `X M Xᵀ`. -/
theorem mul_mul_transpose_apply_eq {κ : Type*} [Fintype κ] (X : Matrix ο κ ℝ)
    (M : Matrix κ κ ℝ) (i j : ο) : (X * M * Xᵀ) i j = X i ⬝ᵥ (M *ᵥ X j) := by
  rw [Matrix.mul_assoc, mul_apply']
  rfl

/-- Entries of the soft Gram matrix: `G̃_{ij} = √λ_i (V_i ⬝ Ŵ P Ŵ V_j) √λ_j`; with `V = extV ω τo`,
`G̃_{ij} = √λ_i K̃(τ_i, τ_j) √λ_j`. -/
theorem opMGram_apply (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam : ο → ℝ) (i j : ο) :
    opMGram B w V lam i j = Real.sqrt (lam i) *
      (V i ⬝ᵥ ((metricW w * proj (B * metricW w) * metricW w) *ᵥ V j)) * Real.sqrt (lam j) := by
  unfold opMGram sqrtPrec
  rw [show diagonal (fun j => Real.sqrt (lam j)) * (V * metricW w) * proj (B * metricW w) *
      (V * metricW w)ᵀ * diagonal (fun j => Real.sqrt (lam j)) =
      diagonal (fun j => Real.sqrt (lam j)) * ((V * metricW w) * proj (B * metricW w) *
      (V * metricW w)ᵀ) * diagonal (fun j => Real.sqrt (lam j)) by simp only [Matrix.mul_assoc],
    mul_diagonal, diagonal_mul, mul_mul_transpose_apply_eq, row_mul_metricW, row_mul_metricW]
  congr 2
  rw [mulVec_dotProduct_of_transpose_eq (metricW_transpose w), mulVec_mulVec, mulVec_mulVec]

/-- The diagonal of the boundary-conditioned kernel is nonnegative, `K̃(τ, τ) ≥ 0` (no hypothesis). -/
theorem bcKernel_self_nonneg (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w ω : Fin K → ℝ) (τ : ℝ) :
    0 ≤ bcKernel B w ω τ τ := by
  unfold bcKernel
  rw [← mulVec_mulVec, ← mulVec_mulVec,
    ← mulVec_dotProduct_of_transpose_eq (metricW_transpose w),
    proj_eq_transpose_mul_self (B * metricW w), ← mulVec_mulVec, dotProduct_mulVec,
    vecMul_transpose]
  exact dotProduct_self_nonneg' _

/-- The `1 × 1` inverse: `(I + G)⁻¹₀₀ = (1 + G₀₀)⁻¹`. -/
theorem fin_one_one_add_inv_apply (G : Matrix (Fin 1) (Fin 1) ℝ) :
    (1 + G)⁻¹ 0 0 = (1 + G 0 0)⁻¹ := by
  rw [inv_subsingleton, diagonal_apply_eq, Ring.inverse_eq_inv]
  simp

/-- The soft gain matrix `P Ṽᵀ Λ^{1/2}(I + G̃)⁻¹Λ^{1/2}`: `y = (gain) r₀`. -/
noncomputable def opMGain (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam : ο → ℝ) : Matrix (Fin K ⊕ Fin K) ο ℝ :=
  proj (B * metricW w) * (V * metricW w)ᵀ * sqrtPrec lam * (1 + opMGram B w V lam)⁻¹ *
    sqrtPrec lam

/-- Influence law for an arbitrary soft input `r` (generalizes `dotProduct_metricW_opMSoft`). -/
theorem dotProduct_metricW_opMGain (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam : ο → ℝ) (r : ο → ℝ) (q : Fin K ⊕ Fin K → ℝ) :
    q ⬝ᵥ (metricW w *ᵥ (opMGain B w V lam *ᵥ r)) =
      ∑ j, (q ⬝ᵥ ((metricW w * proj (B * metricW w) * metricW w) *ᵥ V j)) *
        ((sqrtPrec lam * (1 + opMGram B w V lam)⁻¹ * sqrtPrec lam) *ᵥ r) j := by
  have h : metricW w *ᵥ (opMGain B w V lam *ᵥ r) =
      ((metricW w * proj (B * metricW w) * metricW w) * Vᵀ) *ᵥ
        ((sqrtPrec lam * (1 + opMGram B w V lam)⁻¹ * sqrtPrec lam) *ᵥ r) := by
    unfold opMGain
    rw [mulVec_mulVec, mulVec_mulVec, transpose_mul, metricW_transpose]
    simp only [Matrix.mul_assoc]
  rw [h, dotProduct_mulVec, ← vecMul_vecMul, vecMul_transpose]
  refine Finset.sum_congr rfl fun j _ => ?_
  congr 1
  rw [dotProduct_mulVec]
  exact dotProduct_comm _ _

omit [Fintype ο] [DecidableEq ο] in
/-- The innovation is affine in the residuals with unit slope: `r₀(ρ̃ + ρ') = r₀(ρ̃) + ρ'`. -/
theorem opMInnov_add_rho (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (ρ ρ' : ο → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) :
    opMInnov B w V (ρ + ρ') d g η = opMInnov B w V ρ d g η + ρ' := by
  unfold opMInnov
  abel

/-- Response of Op-M to a residual perturbation `ρ̃ ↦ ρ̃ + ρ'`: `Δc = Ŵ (gain) ρ'`. -/
theorem opM_add_rho_sub (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ ρ' : ο → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) :
    opM B w V lam (ρ + ρ') d g η - opM B w V lam ρ d g η =
      metricW w *ᵥ (opMGain B w V lam *ᵥ ρ') := by
  rw [opM_eq_opHW_add, opM_eq_opHW_add]
  unfold opMSoft
  rw [opMInnov_add_rho, mulVec_add, mulVec_add]
  unfold opMGain
  abel

/-- **Single soft anchor (Sherman–Morrison).** With one exterior anchor
at `τ₁` with precision `λ ≥ 0`, a perturbation `ρ̃ ↦ ρ̃ + δ` moves the prediction at `τ_q` by exactly
`δp̂(τ_q) = λ K̃(τ_q, τ₁) / (1 + λ K̃(τ₁, τ₁)) · δ` (no rank or metric hypothesis). -/
theorem singleAnchor_response (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w ω : Fin K → ℝ) (τ1 : ℝ)
    {lam : ℝ} (hlam : 0 ≤ lam) (ρ : Fin 1 → ℝ) (δ : ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) (τq : ℝ) :
    rcomb ω (opM B w (extV ω fun _ => τ1) (fun _ => lam) (ρ + fun _ => δ) d g η) τq -
        rcomb ω (opM B w (extV ω fun _ => τ1) (fun _ => lam) ρ d g η) τq =
      lam * bcKernel B w ω τq τ1 / (1 + lam * bcKernel B w ω τ1 τ1) * δ := by
  set V : Matrix (Fin 1) (Fin K ⊕ Fin K) ℝ := extV ω fun _ => τ1 with hV
  have hG : opMGram B w V (fun _ => lam) 0 0 = lam * bcKernel B w ω τ1 τ1 := by
    rw [opMGram_apply]
    unfold bcKernel
    rw [show Real.sqrt lam * (V 0 ⬝ᵥ ((metricW w * proj (B * metricW w) * metricW w) *ᵥ V 0)) *
      Real.sqrt lam = (Real.sqrt lam * Real.sqrt lam) *
        (V 0 ⬝ᵥ ((metricW w * proj (B * metricW w) * metricW w) *ᵥ V 0)) by ring,
      Real.mul_self_sqrt hlam]
    rfl
  have hβ : ((sqrtPrec (fun _ : Fin 1 => lam) * (1 + opMGram B w V fun _ => lam)⁻¹ *
      sqrtPrec fun _ => lam) *ᵥ (fun _ => δ)) 0 =
      lam / (1 + lam * bcKernel B w ω τ1 τ1) * δ := by
    unfold sqrtPrec
    simp only [mulVec, dotProduct, Fin.sum_univ_one, Matrix.mul_apply, diagonal_apply_eq,
      fin_one_one_add_inv_apply, hG]
    rw [show Real.sqrt lam * (1 + lam * bcKernel B w ω τ1 τ1)⁻¹ * Real.sqrt lam * δ =
      (Real.sqrt lam * Real.sqrt lam) * (1 + lam * bcKernel B w ω τ1 τ1)⁻¹ * δ by ring,
      Real.mul_self_sqrt hlam, div_eq_mul_inv]
  unfold rcomb
  rw [← dotProduct_sub, opM_add_rho_sub, dotProduct_metricW_opMGain, Fin.sum_univ_one, hβ]
  show bcKernel B w ω τq τ1 * (lam / (1 + lam * bcKernel B w ω τ1 τ1) * δ) = _
  ring

/-- The single-anchor denominator is positive: `1 + λ K̃(τ₁, τ₁) ≥ 1 > 0` for `λ ≥ 0`. -/
theorem singleAnchor_denom_pos (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w ω : Fin K → ℝ) (τ1 : ℝ)
    {lam : ℝ} (hlam : 0 ≤ lam) : 0 < 1 + lam * bcKernel B w ω τ1 τ1 := by
  have := mul_nonneg hlam (bcKernel_self_nonneg B w ω τ1)
  linarith

/-- **The single-anchor gain `λ/(1 + λk)` (`k = K̃(τ₁,τ₁) ≥ 0`) is monotone in
the precision.** -/
theorem singleAnchor_gain_mono {k lam₁ lam₂ : ℝ} (hk : 0 ≤ k) (h₁ : 0 ≤ lam₁) (h₁₂ : lam₁ ≤ lam₂) :
    lam₁ / (1 + lam₁ * k) ≤ lam₂ / (1 + lam₂ * k) := by
  have hp₁ : 0 < 1 + lam₁ * k := by nlinarith
  have hp₂ : 0 < 1 + lam₂ * k := by nlinarith
  rw [div_le_div_iff₀ hp₁ hp₂]
  nlinarith

/-- **The single-anchor gain is `O(λ)` at small precision**:
`0 ≤ λ/(1 + λk) ≤ λ` for `λ, k ≥ 0`. -/
theorem singleAnchor_gain_le_prec {k lam : ℝ} (hk : 0 ≤ k) (hlam : 0 ≤ lam) :
    0 ≤ lam / (1 + lam * k) ∧ lam / (1 + lam * k) ≤ lam := by
  have hp : 0 < 1 + lam * k := by nlinarith
  refine ⟨div_nonneg hlam hp.le, ?_⟩
  rw [div_le_iff₀ hp]
  nlinarith [mul_nonneg (mul_nonneg hlam hlam) hk]

/-- **Bounded by the hard-interpolation response.** For `k = K̃(τ₁,τ₁) > 0` and
`λ ≥ 0`: `|λ K̃(τ_q,τ₁)/(1 + λk)| ≤ |K̃(τ_q,τ₁)|/k` (the `λ → ∞` limit value). -/
theorem singleAnchor_response_le_hard {k kq lam : ℝ} (hk : 0 < k) (hlam : 0 ≤ lam) :
    |lam * kq / (1 + lam * k)| ≤ |kq| / k := by
  have hp : 0 < 1 + lam * k := by nlinarith
  rw [abs_div, abs_mul, abs_of_nonneg hlam, abs_of_pos hp, div_le_div_iff₀ hp hk]
  have := abs_nonneg kq
  nlinarith

end FQGNN.Readout

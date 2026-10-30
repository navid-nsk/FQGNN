import Mathlib

/-!
# Right-inverse pseudoinverse toolkit

Linear-algebra layer used by every readout module (Op-H⁰, Op-H-W, Op-M).

For a real "short, fat" constraint matrix `B : Matrix ι κ ℝ` of full row rank (formalized as
`FullRowRank B := IsUnit (B * Bᵀ).det`) we use the right-inverse formula of the
Moore–Penrose pseudoinverse
`pinvR B = Bᵀ * (B * Bᵀ)⁻¹`  and the null-space projector  `proj B = 1 - pinvR B * B`.

Sources:
* SI §S4.2–S4.4: full row rank of `B`, `X X⁺ = I`, `X X⁺ X = X`,
  null-space projector `I − (BW)⁺(BW)`, and "row scaling of a consistent system leaves the solution
  set — hence the min-norm solution and the null space — unchanged" (§S4.2, bullet 1).
* SI §S7.1 (proof of Prop. S7.1): feasible-set parameterization `u = u₀ + P z`, `P u₀ = 0`,
  push-through identity `(I + AᵀA)⁻¹Aᵀ = Aᵀ(I + AAᵀ)⁻¹`.
* SI §S4.2–S4.3: the equilibrated boundary-Hermite system and its weighted min-norm solve.

Index types stay general (`ι` an abstract finite row type, `κ` e.g. `Fin K ⊕ Fin K`).
-/

namespace FQGNN.Common

open Matrix

variable {ι κ : Type*} [Fintype ι] [Fintype κ] [DecidableEq ι]

/-- Full row rank of a real matrix, formalized as invertibility of the Gram matrix `B Bᵀ`
(SI §S4.4, hypothesis of Prop. S4.1 "suppose `B` has full row rank `R`"). -/
def FullRowRank (B : Matrix ι κ ℝ) : Prop := IsUnit (B * Bᵀ).det

/-- Right-inverse pseudoinverse `B⁺ = Bᵀ (B Bᵀ)⁻¹` (SI §S4.3). -/
noncomputable def pinvR (B : Matrix ι κ ℝ) : Matrix κ ι ℝ := Bᵀ * (B * Bᵀ)⁻¹

/-- Null-space projector `P_B = 1 − B⁺ B` (SI §S4.3 "the projector `I − (BW)⁺(BW)`
maps them into `null(BW)`"). -/
noncomputable def proj [DecidableEq κ] (B : Matrix ι κ ℝ) : Matrix κ κ ℝ := 1 - pinvR B * B

section FullRowRankLemmas

variable {B : Matrix ι κ ℝ}

/-- Convenience: `FullRowRank B` means `det (B Bᵀ) ≠ 0`. -/
theorem fullRowRank_iff_det_ne_zero : FullRowRank B ↔ (B * Bᵀ).det ≠ 0 :=
  isUnit_iff_ne_zero

/-- Convenience: `FullRowRank B` gives `det (B Bᵀ) ≠ 0`. -/
theorem FullRowRank.det_ne_zero (h : FullRowRank B) : (B * Bᵀ).det ≠ 0 :=
  h.ne_zero

/-- Convenience: `FullRowRank B` gives invertibility of the Gram matrix `B Bᵀ`. -/
theorem FullRowRank.isUnit (h : FullRowRank B) : IsUnit (B * Bᵀ) :=
  (isUnit_iff_isUnit_det _).mpr h

/-- Full row rank ⇔ the rows of `B` are linearly independent, i.e. `Bᵀ x = 0 → x = 0`
(SI §S4.4 "full row rank `R`"). -/
theorem fullRowRank_iff_transpose_mulVec_eq_zero :
    FullRowRank B ↔ ∀ x : ι → ℝ, Bᵀ *ᵥ x = 0 → x = 0 := by
  constructor
  · intro h x hx
    by_contra hx0
    have hdet : (B * Bᵀ).det = 0 :=
      exists_mulVec_eq_zero_iff.mp ⟨x, hx0, by rw [← mulVec_mulVec, hx, mulVec_zero]⟩
    exact h.det_ne_zero hdet
  · intro hker
    rw [fullRowRank_iff_det_ne_zero]
    intro hdet
    obtain ⟨v, hv0, hv⟩ := exists_mulVec_eq_zero_iff.mpr hdet
    apply hv0
    apply hker
    have hdot : v ⬝ᵥ ((B * Bᵀ) *ᵥ v) = 0 := by rw [hv, dotProduct_zero]
    rw [← mulVec_mulVec, dotProduct_mulVec, ← mulVec_transpose] at hdot
    exact dotProduct_self_eq_zero.mp hdot

/-- Full row rank is preserved by right multiplication with an invertible square matrix
(used by Op-H-W: `B̃ = B W`; SI §S4.3 / §S6.1 "the learned metric deforms
the norm of the solve, never its constraints"). -/
theorem FullRowRank.mul_right [DecidableEq κ] (h : FullRowRank B) {W : Matrix κ κ ℝ}
    (hW : IsUnit W.det) :
    FullRowRank (B * W) := by
  rw [fullRowRank_iff_transpose_mulVec_eq_zero] at h ⊢
  intro x hx
  apply h
  have hWt : IsUnit Wᵀ.det := by rwa [det_transpose]
  rw [transpose_mul, ← mulVec_mulVec] at hx
  have h2 : Wᵀ⁻¹ *ᵥ (Wᵀ *ᵥ (Bᵀ *ᵥ x)) = 0 := by rw [hx, mulVec_zero]
  rwa [mulVec_mulVec, nonsing_inv_mul _ hWt, one_mulVec] at h2

/-- The form used by Op-H: `IsUnit ((B W)(B W)ᵀ).det` from `FullRowRank B` and an invertible
metric `W` (SI Prop. S4.1 "every metric `w > 0`"). -/
theorem isUnit_det_mul_mul_transpose [DecidableEq κ] (h : FullRowRank B) {W : Matrix κ κ ℝ}
    (hW : IsUnit W.det) : IsUnit ((B * W) * (B * W)ᵀ).det :=
  h.mul_right hW

/-- Full row rank is invariant under left multiplication by an invertible square matrix
(row operations; SI §S4.2 bullet 1, derivative-row equilibration). -/
theorem fullRowRank_mul_left_iff {D : Matrix ι ι ℝ} (hD : IsUnit D.det) :
    FullRowRank (D * B) ↔ FullRowRank B := by
  unfold FullRowRank
  have hDt : IsUnit Dᵀ.det := by rwa [det_transpose]
  rw [transpose_mul, show D * B * (Bᵀ * Dᵀ) = D * (B * Bᵀ) * Dᵀ by
    simp only [Matrix.mul_assoc], det_mul, det_mul, IsUnit.mul_iff, IsUnit.mul_iff]
  exact ⟨fun h => h.1.2, fun h => ⟨⟨hD, h⟩, hDt⟩⟩

end FullRowRankLemmas

section Penrose

variable {B : Matrix ι κ ℝ}

/-- `B B⁺ = 1` for full row rank `B` (SI §S4.4 proof of Prop. S4.1,
"`XX⁺ = I`"). -/
theorem mul_pinvR (h : FullRowRank B) : B * pinvR B = 1 := by
  unfold pinvR
  rw [← Matrix.mul_assoc, mul_nonsing_inv _ h]

/-- Vector form: `B (B⁺ d) = d`: the particular solution is feasible
(SI §S4.4 proof of Prop. S4.1). -/
theorem mulVec_pinvR_mulVec (h : FullRowRank B) (d : ι → ℝ) : B *ᵥ (pinvR B *ᵥ d) = d := by
  rw [mulVec_mulVec, mul_pinvR h, one_mulVec]

/-- Penrose identity (i): `B B⁺ B = B` (SI §S4.4 "`XX⁺X = X`"). -/
theorem mul_pinvR_mul (h : FullRowRank B) : B * pinvR B * B = B := by
  rw [mul_pinvR h, Matrix.one_mul]

/-- Penrose identity (ii): `B⁺ B B⁺ = B⁺`. -/
theorem pinvR_mul_pinvR (h : FullRowRank B) : pinvR B * B * pinvR B = pinvR B := by
  rw [Matrix.mul_assoc, mul_pinvR h, Matrix.mul_one]

/-- Penrose identity (iii): `(B B⁺)ᵀ = B B⁺`. -/
theorem transpose_mul_pinvR (h : FullRowRank B) : (B * pinvR B)ᵀ = B * pinvR B := by
  rw [mul_pinvR h, transpose_one]

/-- Transpose of the pseudoinverse: `(B⁺)ᵀ = (B Bᵀ)⁻¹ B`. -/
theorem transpose_pinvR : (pinvR B)ᵀ = (B * Bᵀ)⁻¹ * B := by
  unfold pinvR
  rw [transpose_mul, transpose_nonsing_inv, transpose_mul, transpose_transpose]

/-- Penrose identity (iv): `(B⁺ B)ᵀ = B⁺ B`. Holds with no rank hypothesis,
because the Gram matrix `B Bᵀ` is symmetric. -/
theorem transpose_pinvR_mul : (pinvR B * B)ᵀ = pinvR B * B := by
  rw [transpose_mul, transpose_pinvR]
  unfold pinvR
  rw [Matrix.mul_assoc]

end Penrose

section Projector

variable [DecidableEq κ] {B : Matrix ι κ ℝ}

/-- `B P_B = 0`: the projector maps into `null B` (SI §S4.3). -/
theorem mul_proj (h : FullRowRank B) : B * proj B = 0 := by
  unfold proj
  rw [Matrix.mul_sub, Matrix.mul_one, ← Matrix.mul_assoc, mul_pinvR h, Matrix.one_mul, sub_self]

/-- `P_B B⁺ = 0` (SI §S7.1 proof "`P u₀ = 0` since `u₀ ∈ range(B̃ᵀ)`"). -/
theorem proj_mul_pinvR (h : FullRowRank B) : proj B * pinvR B = 0 := by
  unfold proj
  rw [Matrix.sub_mul, Matrix.one_mul, Matrix.mul_assoc, mul_pinvR h, Matrix.mul_one, sub_self]

/-- `P_B Bᵀ = 0`: the projector kills the row space (SI §S7.1 proof). -/
theorem proj_mul_transpose (h : FullRowRank B) : proj B * Bᵀ = 0 := by
  unfold proj pinvR
  rw [Matrix.sub_mul, Matrix.one_mul, Matrix.mul_assoc, Matrix.mul_assoc,
    nonsing_inv_mul _ h, Matrix.mul_one, sub_self]

/-- `P_B` is idempotent: `P_B P_B = P_B`. -/
theorem proj_mul_proj (h : FullRowRank B) : proj B * proj B = proj B := by
  calc proj B * proj B = proj B * (1 - pinvR B * B) := rfl
    _ = proj B := by
      rw [Matrix.mul_sub, Matrix.mul_one, ← Matrix.mul_assoc, proj_mul_pinvR h, Matrix.zero_mul,
        sub_zero]

/-- `P_B` is symmetric: `P_Bᵀ = P_B` (no rank hypothesis needed). -/
theorem transpose_proj : (proj B)ᵀ = proj B := by
  unfold proj
  rw [transpose_sub, transpose_one, transpose_pinvR_mul]

/-- `P_B (B⁺ d) = 0` (SI §S7.1 proof "`P u₀ = 0`"). -/
theorem proj_mulVec_pinvR_mulVec (h : FullRowRank B) (d : ι → ℝ) :
    proj B *ᵥ (pinvR B *ᵥ d) = 0 := by
  rw [mulVec_mulVec, proj_mul_pinvR h, zero_mulVec]

/-- Vector form: `B (P_B z) = 0` for every `z`: null-space coordinates never violate the
hard constraints (SI §S4.3 "so they can never violate (S4.6)"). -/
theorem mulVec_proj_mulVec (h : FullRowRank B) (z : κ → ℝ) : B *ᵥ (proj B *ᵥ z) = 0 := by
  rw [mulVec_mulVec, mul_proj h, zero_mulVec]

/-- `P_B` fixes every vector of `null B` (no rank hypothesis needed). -/
theorem proj_mulVec_of_mulVec_eq_zero {e : κ → ℝ} (he : B *ᵥ e = 0) : proj B *ᵥ e = e := by
  unfold proj
  rw [sub_mulVec, one_mulVec, ← mulVec_mulVec, he, mulVec_zero, sub_zero]

/-- Idempotence, vector form: `P_B (P_B z) = P_B z`. -/
theorem proj_mulVec_proj_mulVec (h : FullRowRank B) (z : κ → ℝ) :
    proj B *ᵥ (proj B *ᵥ z) = proj B *ᵥ z := by
  rw [mulVec_mulVec, proj_mul_proj h]

/-- Self-adjointness, vector form: `(P_B x) ⬝ᵥ y = x ⬝ᵥ (P_B y)` (no rank
hypothesis needed; used for the orthogonal split in SI §S7.1). -/
theorem proj_mulVec_dotProduct (x y : κ → ℝ) : (proj B *ᵥ x) ⬝ᵥ y = x ⬝ᵥ (proj B *ᵥ y) := by
  rw [dotProduct_comm, dotProduct_mulVec, ← mulVec_transpose, transpose_proj, dotProduct_comm]

/-- For full row rank `B`, `P_B z = z ↔ B z = 0`: `range P_B = null B` (SI §S7.1 proof,
"`y ∈ range(P)`"). -/
theorem proj_mulVec_eq_self_iff (h : FullRowRank B) (z : κ → ℝ) :
    proj B *ᵥ z = z ↔ B *ᵥ z = 0 := by
  constructor
  · intro hz
    rw [← hz]
    exact mulVec_proj_mulVec h z
  · exact proj_mulVec_of_mulVec_eq_zero

/-- Feasible-set parameterization, forward direction: every solution of `B u = d` is
`u = B⁺ d + P_B (u − B⁺ d)`, i.e. of the form `u₀ + P z` (SI §S7.1 proof
"parameterize the feasible affine set as `u = u₀ + P z`"). -/
theorem eq_pinvR_add_proj_of_mulVec_eq (h : FullRowRank B) {u : κ → ℝ} {d : ι → ℝ}
    (hu : B *ᵥ u = d) : u = pinvR B *ᵥ d + proj B *ᵥ (u - pinvR B *ᵥ d) := by
  have he : B *ᵥ (u - pinvR B *ᵥ d) = 0 := by
    rw [mulVec_sub, hu, mulVec_pinvR_mulVec h, sub_self]
  rw [proj_mulVec_of_mulVec_eq_zero he, add_sub_cancel]

/-- Feasible-set parameterization, converse: `B (B⁺ d + P_B z) = d` for every `z`
(SI §S4.4 proof of Prop. S4.1, "for every `η` … `Bc = d`"). -/
theorem mulVec_pinvR_add_proj (h : FullRowRank B) (d : ι → ℝ) (z : κ → ℝ) :
    B *ᵥ (pinvR B *ᵥ d + proj B *ᵥ z) = d := by
  rw [mulVec_add, mulVec_pinvR_mulVec h, mulVec_proj_mulVec h, add_zero]

/-- Feasible set = `u₀ + range P_B` (both directions bundled) (SI §S7.1). -/
theorem mulVec_eq_iff_exists_proj (h : FullRowRank B) (u : κ → ℝ) (d : ι → ℝ) :
    B *ᵥ u = d ↔ ∃ z : κ → ℝ, u = pinvR B *ᵥ d + proj B *ᵥ z := by
  constructor
  · intro hu
    exact ⟨_, eq_pinvR_add_proj_of_mulVec_eq h hu⟩
  · rintro ⟨z, rfl⟩
    exact mulVec_pinvR_add_proj h d z

end Projector

section MinNorm

variable {B : Matrix ι κ ℝ}

/-- The squared Euclidean norm `u ⬝ᵥ u` of a real vector is nonnegative. -/
theorem dotProduct_self_nonneg' (u : κ → ℝ) : 0 ≤ u ⬝ᵥ u :=
  Finset.sum_nonneg fun i _ => mul_self_nonneg (u i)

/-- Orthogonality: `B⁺ d ⊥ null B`, i.e. `(B⁺ d) ⬝ᵥ e = 0` whenever `B e = 0`
(SI §S7.1 proof, "`u₀ ∈ range(B̃ᵀ)`"). No rank hypothesis is needed. -/
theorem pinvR_mulVec_dotProduct_of_mulVec_eq_zero (d : ι → ℝ) {e : κ → ℝ} (he : B *ᵥ e = 0) :
    (pinvR B *ᵥ d) ⬝ᵥ e = 0 := by
  unfold pinvR
  rw [← mulVec_mulVec, mulVec_transpose, ← dotProduct_mulVec, he, dotProduct_zero]

/-- Orthogonality in projector form: `(B⁺ d) ⬝ᵥ (P_B z) = 0` (SI §S7.1 proof,
the orthogonal split of the objective). -/
theorem pinvR_mulVec_dotProduct_proj_mulVec [DecidableEq κ] (h : FullRowRank B) (d : ι → ℝ)
    (z : κ → ℝ) :
    (pinvR B *ᵥ d) ⬝ᵥ (proj B *ᵥ z) = 0 :=
  pinvR_mulVec_dotProduct_of_mulVec_eq_zero d (mulVec_proj_mulVec h z)

/-- Pythagoras on the feasible set: if `B u = d` then
`‖u‖² = ‖B⁺ d‖² + ‖u − B⁺ d‖²`. -/
theorem dotProduct_self_eq_pinvR_add (h : FullRowRank B) {u : κ → ℝ} {d : ι → ℝ}
    (hu : B *ᵥ u = d) :
    u ⬝ᵥ u = (pinvR B *ᵥ d) ⬝ᵥ (pinvR B *ᵥ d) +
      (u - pinvR B *ᵥ d) ⬝ᵥ (u - pinvR B *ᵥ d) := by
  set u₀ := pinvR B *ᵥ d with hu₀
  set e := u - u₀ with he_def
  have he : B *ᵥ e = 0 := by
    rw [he_def, mulVec_sub, hu, hu₀, mulVec_pinvR_mulVec h, sub_self]
  have horth : u₀ ⬝ᵥ e = 0 := pinvR_mulVec_dotProduct_of_mulVec_eq_zero d he
  have hsplit : u = u₀ + e := by rw [he_def, add_sub_cancel]
  rw [hsplit, add_dotProduct, dotProduct_add, dotProduct_add, dotProduct_comm e u₀, horth]
  ring

/-- Minimum-norm property: for full row rank `B`, `u₀ = B⁺ d` has the least squared norm
among all solutions of `B u = d` (SI §S4.3 "weighted min-norm solve"). -/
theorem pinvR_mulVec_dotProduct_le (h : FullRowRank B) {u : κ → ℝ} {d : ι → ℝ}
    (hu : B *ᵥ u = d) : (pinvR B *ᵥ d) ⬝ᵥ (pinvR B *ᵥ d) ≤ u ⬝ᵥ u := by
  rw [dotProduct_self_eq_pinvR_add h hu]
  exact le_add_of_nonneg_right (dotProduct_self_nonneg' _)

/-- Uniqueness of the minimum-norm solution: a feasible `u` whose squared norm does not
exceed that of `B⁺ d` equals `B⁺ d`. -/
theorem eq_pinvR_mulVec_of_dotProduct_le (h : FullRowRank B) {u : κ → ℝ} {d : ι → ℝ}
    (hu : B *ᵥ u = d) (hle : u ⬝ᵥ u ≤ (pinvR B *ᵥ d) ⬝ᵥ (pinvR B *ᵥ d)) :
    u = pinvR B *ᵥ d := by
  rw [dotProduct_self_eq_pinvR_add h hu] at hle
  have h0 : (u - pinvR B *ᵥ d) ⬝ᵥ (u - pinvR B *ᵥ d) = 0 :=
    le_antisymm (by linarith) (dotProduct_self_nonneg' _)
  exact sub_eq_zero.mp (dotProduct_self_eq_zero.mp h0)

/-- Minimum-norm characterization, bundled: for full row rank `B`, `B⁺ d` is feasible,
has least squared norm on `{u | B u = d}`, and is the unique feasible vector attaining it
(SI §S4.3). -/
theorem pinvR_mulVec_isMinNorm (h : FullRowRank B) (d : ι → ℝ) :
    B *ᵥ (pinvR B *ᵥ d) = d ∧
      (∀ u, B *ᵥ u = d → (pinvR B *ᵥ d) ⬝ᵥ (pinvR B *ᵥ d) ≤ u ⬝ᵥ u) ∧
      (∀ u, B *ᵥ u = d → u ⬝ᵥ u = (pinvR B *ᵥ d) ⬝ᵥ (pinvR B *ᵥ d) → u = pinvR B *ᵥ d) :=
  ⟨mulVec_pinvR_mulVec h d, fun _ hu => pinvR_mulVec_dotProduct_le h hu,
    fun _ hu heq => eq_pinvR_mulVec_of_dotProduct_le h hu heq.le⟩

end MinNorm

section Scaling

/-- Scalar homogeneity of the pseudoinverse: `(α B)⁺ = α⁻¹ B⁺` for `α ≠ 0`
(SI Prop. S6.1, 0-homogeneity of the particular part). No rank hypothesis needed. -/
theorem pinvR_smul {α : ℝ} (hα : α ≠ 0) (B : Matrix ι κ ℝ) :
    pinvR (α • B) = α⁻¹ • pinvR B := by
  unfold pinvR
  have hG : (α • B) * (α • B)ᵀ = (α * α) • (B * Bᵀ) := by
    rw [transpose_smul, Matrix.smul_mul, Matrix.mul_smul, smul_smul]
  have hαα : α * α ≠ 0 := mul_ne_zero hα hα
  have hinv : ((α * α) • (B * Bᵀ))⁻¹ = (α * α)⁻¹ • (B * Bᵀ)⁻¹ := by
    by_cases hB : IsUnit (B * Bᵀ).det
    · exact inv_eq_left_inv (by
        rw [Matrix.smul_mul, Matrix.mul_smul, smul_smul, nonsing_inv_mul _ hB,
          inv_mul_cancel₀ hαα, one_smul])
    · have hB' : ¬ IsUnit ((α * α) • (B * Bᵀ)).det := by
        rw [det_smul, IsUnit.mul_iff, not_and_or]
        exact Or.inr hB
      rw [nonsing_inv_apply_not_isUnit _ hB, nonsing_inv_apply_not_isUnit _ hB', smul_zero]
  rw [hG, hinv, transpose_smul, Matrix.smul_mul, Matrix.mul_smul, smul_smul]
  congr 1
  field_simp

/-- The null-space projector is invariant under nonzero scalar scaling:
`P_{αB} = P_B` (SI Prop. S6.1). -/
theorem proj_smul [DecidableEq κ] {α : ℝ} (hα : α ≠ 0) (B : Matrix ι κ ℝ) :
    proj (α • B) = proj B := by
  unfold proj
  rw [pinvR_smul hα, Matrix.smul_mul, Matrix.mul_smul, smul_smul, inv_mul_cancel₀ hα, one_smul]

/-- Left multiplication by an invertible square matrix (row operations):
`(D B)⁺ = B⁺ D⁻¹` (SI §S4.2 bullet 1). No rank hypothesis on `B` needed. -/
theorem pinvR_mul_left {D : Matrix ι ι ℝ} (hD : IsUnit D.det) (B : Matrix ι κ ℝ) :
    pinvR (D * B) = pinvR B * D⁻¹ := by
  unfold pinvR
  have hDt : IsUnit Dᵀ.det := by rwa [det_transpose]
  rw [transpose_mul, show D * B * (Bᵀ * Dᵀ) = D * (B * Bᵀ) * Dᵀ by simp only [Matrix.mul_assoc],
    Matrix.mul_inv_rev, Matrix.mul_inv_rev]
  calc Bᵀ * Dᵀ * (Dᵀ⁻¹ * ((B * Bᵀ)⁻¹ * D⁻¹))
      = Bᵀ * (Dᵀ * Dᵀ⁻¹) * ((B * Bᵀ)⁻¹ * D⁻¹) := by simp only [Matrix.mul_assoc]
    _ = Bᵀ * (B * Bᵀ)⁻¹ * D⁻¹ := by
      rw [mul_nonsing_inv _ hDt, Matrix.mul_one, Matrix.mul_assoc]

/-- Row equilibration leaves the minimum-norm solution unchanged:
`(D B)⁺ (D d) = B⁺ d` for invertible `D` (SI §S4.2 bullet 1 "row scaling of a consistent
system leaves … the min-norm solution … unchanged"). -/
theorem pinvR_mul_left_mulVec {D : Matrix ι ι ℝ} (hD : IsUnit D.det) (B : Matrix ι κ ℝ)
    (d : ι → ℝ) : pinvR (D * B) *ᵥ (D *ᵥ d) = pinvR B *ᵥ d := by
  rw [pinvR_mul_left hD, mulVec_mulVec, Matrix.mul_assoc, nonsing_inv_mul _ hD, Matrix.mul_one]

/-- Row equilibration leaves the null-space projector unchanged: `P_{DB} = P_B`
(SI §S4.2 bullet 1 "… and the null space — unchanged"). -/
theorem proj_mul_left [DecidableEq κ] {D : Matrix ι ι ℝ} (hD : IsUnit D.det) (B : Matrix ι κ ℝ) :
    proj (D * B) = proj B := by
  unfold proj
  rw [pinvR_mul_left hD, show pinvR B * D⁻¹ * (D * B) = pinvR B * (D⁻¹ * D) * B by
    simp only [Matrix.mul_assoc], nonsing_inv_mul _ hD, Matrix.mul_one]

/-- Row equilibration leaves the solution set unchanged:
`(D B) u = D d ↔ B u = d` for invertible `D` (SI §S4.2 bullet 1). -/
theorem mul_left_mulVec_eq_iff {D : Matrix ι ι ℝ} (hD : IsUnit D.det) (B : Matrix ι κ ℝ)
    (u : κ → ℝ) (d : ι → ℝ) : (D * B) *ᵥ u = D *ᵥ d ↔ B *ᵥ u = d := by
  constructor
  · intro hu
    rw [← mulVec_mulVec] at hu
    have h2 : D⁻¹ *ᵥ (D *ᵥ (B *ᵥ u)) = D⁻¹ *ᵥ (D *ᵥ d) := by rw [hu]
    simpa only [mulVec_mulVec, ← Matrix.mul_assoc, nonsing_inv_mul _ hD, Matrix.one_mul,
      one_mulVec] using h2
  · intro hu
    rw [← mulVec_mulVec, hu]

/-- A diagonal matrix with positive entries is invertible (the equilibration
factors `s_e > 0`). -/
theorem isUnit_det_diagonal_of_pos {s : ι → ℝ} (hs : ∀ i, 0 < s i) :
    IsUnit (diagonal s).det := by
  rw [det_diagonal, isUnit_iff_ne_zero]
  exact Finset.prod_ne_zero_iff.mpr fun i _ => (hs i).ne'

/-- Positive-diagonal row equilibration, min-norm solution: `(D B)⁺ (D d) = B⁺ d` for
`D = diagonal s`, `s > 0` (SI §S4.2 bullet 1: derivative rows equilibrated by
`1/mean|ω|`). -/
theorem pinvR_diagonal_mul_mulVec {s : ι → ℝ} (hs : ∀ i, 0 < s i) (B : Matrix ι κ ℝ)
    (d : ι → ℝ) : pinvR (diagonal s * B) *ᵥ (diagonal s *ᵥ d) = pinvR B *ᵥ d :=
  pinvR_mul_left_mulVec (isUnit_det_diagonal_of_pos hs) B d

/-- Positive-diagonal row equilibration, null space: `P_{diag(s) B} = P_B`
(SI §S4.2 bullet 1). -/
theorem proj_diagonal_mul [DecidableEq κ] {s : ι → ℝ} (hs : ∀ i, 0 < s i) (B : Matrix ι κ ℝ) :
    proj (diagonal s * B) = proj B :=
  proj_mul_left (isUnit_det_diagonal_of_pos hs) B

/-- Positive-diagonal row equilibration preserves full row rank. -/
theorem fullRowRank_diagonal_mul_iff {s : ι → ℝ} (hs : ∀ i, 0 < s i) (B : Matrix ι κ ℝ) :
    FullRowRank (diagonal s * B) ↔ FullRowRank B :=
  fullRowRank_mul_left_iff (isUnit_det_diagonal_of_pos hs)

/-- Metric-scale behaviour of the particular part (SI Prop. S6.1 (ii),
0-homogeneity): `(α W)(B (α W))⁺ = W (B W)⁺` for `α ≠ 0`. -/
theorem smul_mul_pinvR_mul_smul {α : ℝ} (hα : α ≠ 0) (B : Matrix ι κ ℝ) (W : Matrix κ κ ℝ) :
    (α • W) * pinvR (B * (α • W)) = W * pinvR (B * W) := by
  rw [Matrix.mul_smul, pinvR_smul hα, Matrix.smul_mul, Matrix.mul_smul, smul_smul,
    mul_inv_cancel₀ hα, one_smul]

/-- Metric-scale behaviour of the null-space projector (SI Prop. S6.1 (ii)):
`P_{B (α W)} = P_{B W}` for `α ≠ 0`, so the `η`-channel `(αW) P_{B(αW)} η = α • W P_{BW} η` is
1-homogeneous. -/
theorem proj_mul_smul [DecidableEq κ] {α : ℝ} (hα : α ≠ 0) (B : Matrix ι κ ℝ) (W : Matrix κ κ ℝ) :
    proj (B * (α • W)) = proj (B * W) := by
  rw [Matrix.mul_smul, proj_smul hα]

end Scaling

section PushThrough

variable {m n : Type*} [Fintype m] [Fintype n]

/-- `1 + Aᵀ A` is positive definite for every real `A` (SI Thm S7.2
"`I + G̃ ⪰ I` is always invertible"). -/
theorem posDef_one_add_transpose_mul_self [DecidableEq n] (A : Matrix m n ℝ) :
    (1 + Aᵀ * A).PosDef := by
  have h := posSemidef_conjTranspose_mul_self A
  rw [conjTranspose_eq_transpose_of_trivial] at h
  exact PosDef.one.add_posSemidef h

/-- `1 + A Aᵀ` is positive definite for every real `A` (SI Thm S7.2). -/
theorem posDef_one_add_mul_transpose_self [DecidableEq m] (A : Matrix m n ℝ) :
    (1 + A * Aᵀ).PosDef := by
  simpa only [transpose_transpose] using posDef_one_add_transpose_mul_self Aᵀ

/-- `det (1 + Aᵀ A)` is a unit. -/
theorem isUnit_det_one_add_transpose_mul_self [DecidableEq n] (A : Matrix m n ℝ) :
    IsUnit (1 + Aᵀ * A).det :=
  (isUnit_iff_isUnit_det _).mp (posDef_one_add_transpose_mul_self A).isUnit

/-- `det (1 + A Aᵀ)` is a unit. -/
theorem isUnit_det_one_add_mul_transpose_self [DecidableEq m] (A : Matrix m n ℝ) :
    IsUnit (1 + A * Aᵀ).det :=
  (isUnit_iff_isUnit_det _).mp (posDef_one_add_mul_transpose_self A).isUnit

/-- Push-through identity `(1 + AᵀA)⁻¹ Aᵀ = Aᵀ (1 + AAᵀ)⁻¹` for every real matrix `A`
(SI §S7.1, proof of Prop. S7.1, used with `A = Λ^{1/2} Ṽ P`). -/
theorem one_add_transpose_mul_self_inv_mul [DecidableEq m] [DecidableEq n] (A : Matrix m n ℝ) :
    (1 + Aᵀ * A)⁻¹ * Aᵀ = Aᵀ * (1 + A * Aᵀ)⁻¹ := by
  have hP := isUnit_det_one_add_transpose_mul_self A
  have hQ := isUnit_det_one_add_mul_transpose_self A
  have hcomm : Aᵀ * (1 + A * Aᵀ) = (1 + Aᵀ * A) * Aᵀ := by
    rw [Matrix.mul_add, Matrix.add_mul, Matrix.mul_one, Matrix.one_mul, Matrix.mul_assoc]
  calc (1 + Aᵀ * A)⁻¹ * Aᵀ
      = (1 + Aᵀ * A)⁻¹ * (Aᵀ * (1 + A * Aᵀ)) * (1 + A * Aᵀ)⁻¹ := by
        rw [Matrix.mul_assoc, Matrix.mul_assoc, mul_nonsing_inv _ hQ, Matrix.mul_one]
    _ = (1 + Aᵀ * A)⁻¹ * (1 + Aᵀ * A) * Aᵀ * (1 + A * Aᵀ)⁻¹ := by
        rw [hcomm]
        simp only [Matrix.mul_assoc]
    _ = Aᵀ * (1 + A * Aᵀ)⁻¹ := by
        rw [nonsing_inv_mul _ hP, Matrix.one_mul]

end PushThrough

section RankBridge

/-- A real square matrix has full rank iff its determinant is a unit. -/
theorem rank_eq_card_iff_isUnit_det (M : Matrix ι ι ℝ) :
    M.rank = Fintype.card ι ↔ IsUnit M.det := by
  constructor
  · intro hr
    rw [isUnit_iff_ne_zero]
    intro hdet
    obtain ⟨v, hv0, hv⟩ := exists_mulVec_eq_zero_iff.mpr hdet
    have hrn := LinearMap.finrank_range_add_finrank_ker M.mulVecLin
    rw [Module.finrank_fintype_fun_eq_card] at hrn
    have hk : Module.finrank ℝ (LinearMap.ker M.mulVecLin) = 0 := by
      unfold Matrix.rank at hr
      omega
    have hmem : v ∈ LinearMap.ker M.mulVecLin := by
      rw [LinearMap.mem_ker, mulVecLin_apply, hv]
    rw [Submodule.finrank_eq_zero] at hk
    rw [hk, Submodule.mem_bot] at hmem
    exact hv0 hmem
  · intro h
    exact rank_of_isUnit M ((isUnit_iff_isUnit_det M).mpr h)

/-- Bridge to the usual notion: `rank B = |ι| ↔ FullRowRank B` over `ℝ`
(SI §S4.4 "full row rank `R`"). -/
theorem rank_eq_card_iff_fullRowRank (B : Matrix ι κ ℝ) :
    B.rank = Fintype.card ι ↔ FullRowRank B := by
  rw [FullRowRank, ← rank_eq_card_iff_isUnit_det, rank_self_mul_transpose]

end RankBridge

end FQGNN.Common

import Mathlib
import FQGNN.Common.Pseudoinverse

/-!
# The genuine Moore–Penrose inverse of a real matrix

`FQGNN.Common.Pseudoinverse` models the pseudoinverse by the right-inverse formula
`pinvR B = Bᵀ (B Bᵀ)⁻¹`, which is the Moore–Penrose inverse only under full row rank. Several
sentences of the SI are stated **without** a rank hypothesis and are true only for the genuine
Moore–Penrose inverse:

* SI §S4.4, proof of Prop. S4.1: "the Penrose identity `XX⁺X = X` gives `X(I − X⁺X) = 0`
  **unconditionally** (no rank hypothesis)";
* SI Prop. S6.1(i): "`B c(w) = d` … unconditionally when `d = 0`";
* SI Thm S7.3 and Cor. S7.4 (Op-M exactness and endpoint annihilation).

This module supplies that inverse for every real matrix `B : Matrix ι κ ℝ` (finite index types):

1. `IsMoorePenrose B X`: the four Penrose equations `BXB = B`, `XBX = X`, `(BX)ᵀ = BX`,
   `(XB)ᵀ = XB`; **uniqueness** (`IsMoorePenrose.unique`, the classical algebraic argument).
2. **Existence** for every real matrix (`exists_isMoorePenrose`). Route: a `{1}`-inverse
   (`BXB = B`) from the factorization `B = ι ∘ s` of `toLin' B` through its range (a right inverse
   of the surjection `s`, a left inverse of the injection `ι`); a least-squares `{1,3}`-inverse
   `Z = (BᵀB)⁻ Bᵀ` from a `{1}`-inverse of the Gram matrix; then `X = (Bᵀ Y) Z` with `Y` a
   `{1,3}`-inverse of `Bᵀ` satisfies all four Penrose equations.
3. `mpinv B` (by choice), its Penrose equations, `mpinv B = pinvR B` under `FullRowRank B`,
   the null-space projector `projMP B = 1 − B⁺B` and the **unconditional** facts:
   `B (1 − XB) = 0` for every `{1}`-inverse `X`, `B (B⁺ d) = d ↔ d ∈ range B`, the feasible-set
   parameterization and the minimum-norm characterization on every consistent system.

Sources: SI §S4.3–S4.4 (Prop. S4.1 and its proof), SI §S6.1 (Prop. S6.1(i)), SI §S7.2
(Thm S7.3, Cor. S7.4). The Penrose identities, the projector and the min-norm facts of
`FQGNN.Common.Pseudoinverse` are proved here without the rank hypothesis.
-/

namespace FQGNN.Common

open Matrix

variable {ι κ : Type*} [Fintype ι] [Fintype κ]

/-! ## The four Penrose equations and uniqueness -/

/-- `X` is a Moore–Penrose inverse of `B`: the four Penrose equations over `ℝ`
(`BXB = B`, `XBX = X`, `(BX)ᵀ = BX`, `(XB)ᵀ = XB`). SI §S4.4 proof of Prop. S4.1 uses the first
("`XX⁺X = X`"). -/
structure IsMoorePenrose (B : Matrix ι κ ℝ) (X : Matrix κ ι ℝ) : Prop where
  /-- Penrose equation (1): `B X B = B`. -/
  mul_mul_self : B * X * B = B
  /-- Penrose equation (2): `X B X = X`. -/
  mul_mul_self' : X * B * X = X
  /-- Penrose equation (3): `(B X)ᵀ = B X`. -/
  transpose_mul : (B * X)ᵀ = B * X
  /-- Penrose equation (4): `(X B)ᵀ = X B`. -/
  transpose_mul' : (X * B)ᵀ = X * B

namespace IsMoorePenrose

variable {B : Matrix ι κ ℝ} {X Y : Matrix κ ι ℝ}

/-- Two Moore–Penrose inverses give the same range projector: `B X = B Y` (uniqueness argument,
step 1; uses only equations (1) and (3)). -/
theorem mul_left_eq (hX : IsMoorePenrose B X) (hY : IsMoorePenrose B Y) : B * X = B * Y := by
  calc B * X = (B * X)ᵀ := hX.transpose_mul.symm
    _ = Xᵀ * Bᵀ := Matrix.transpose_mul _ _
    _ = Xᵀ * (B * Y * B)ᵀ := by rw [hY.mul_mul_self]
    _ = (B * X)ᵀ * (B * Y)ᵀ := by
        rw [Matrix.transpose_mul (B * Y) B, Matrix.transpose_mul B X, Matrix.mul_assoc]
    _ = B * X * (B * Y) := by rw [hX.transpose_mul, hY.transpose_mul]
    _ = B * Y := by rw [← Matrix.mul_assoc, hX.mul_mul_self]

/-- Two Moore–Penrose inverses give the same co-range projector: `X B = Y B` (uniqueness argument,
step 2; uses only equations (1) and (4)). -/
theorem mul_right_eq (hX : IsMoorePenrose B X) (hY : IsMoorePenrose B Y) : X * B = Y * B := by
  calc X * B = (X * B)ᵀ := hX.transpose_mul'.symm
    _ = Bᵀ * Xᵀ := Matrix.transpose_mul _ _
    _ = (B * Y * B)ᵀ * Xᵀ := by rw [hY.mul_mul_self]
    _ = (Y * B)ᵀ * (X * B)ᵀ := by
        simp only [Matrix.transpose_mul, Matrix.mul_assoc]
    _ = Y * B * (X * B) := by rw [hX.transpose_mul', hY.transpose_mul']
    _ = Y * B := by rw [Matrix.mul_assoc, ← Matrix.mul_assoc B X B, hX.mul_mul_self]

/-- **Uniqueness of the Moore–Penrose inverse** (classical algebraic argument):
`X = XBX = X(BY) = (XB)Y = (YB)Y = Y`. -/
theorem unique (hX : IsMoorePenrose B X) (hY : IsMoorePenrose B Y) : X = Y := by
  calc X = X * B * X := hX.mul_mul_self'.symm
    _ = X * (B * Y) := by rw [Matrix.mul_assoc, hX.mul_left_eq hY]
    _ = Y * B * Y := by rw [← Matrix.mul_assoc, hX.mul_right_eq hY]
    _ = Y := hY.mul_mul_self'

end IsMoorePenrose

/-! ## Existence -/

/-- Every real matrix has a `{1}`-inverse: `∃ X, B X B = B`. Built from the factorization of
`toLin' B` through its range: a right inverse of the surjection onto the range composed with a left
inverse of the inclusion of the range (helper for the existence of the Moore–Penrose inverse). -/
theorem exists_mul_mul_eq_self (B : Matrix ι κ ℝ) : ∃ X : Matrix κ ι ℝ, B * X * B = B := by
  classical
  set f := Matrix.toLin' B
  obtain ⟨r, hr⟩ := LinearMap.exists_rightInverse_of_surjective f.rangeRestrict
    (LinearMap.range_rangeRestrict f)
  obtain ⟨l, hl⟩ := LinearMap.exists_leftInverse_of_injective (LinearMap.range f).subtype
    (Submodule.ker_subtype _)
  refine ⟨LinearMap.toMatrix' (r ∘ₗ l), ?_⟩
  apply Matrix.toLin'.injective
  rw [Matrix.toLin'_mul, Matrix.toLin'_mul, Matrix.toLin'_toMatrix']
  change f ∘ₗ (r ∘ₗ l) ∘ₗ f = f
  refine LinearMap.ext fun x => ?_
  have h1 : l (f x) = f.rangeRestrict x := by
    have := LinearMap.congr_fun hl (f.rangeRestrict x)
    simpa using this
  have h2 : ∀ y, f (r y) = (y : ι → ℝ) := fun y => by
    have h3 := congrArg Subtype.val (LinearMap.congr_fun hr y)
    simpa using h3
  simp only [LinearMap.comp_apply, h1, h2]
  rfl

/-- A real matrix with `Eᵀ E = 0` is zero (helper for `exists_lsq_inverse`). -/
theorem eq_zero_of_transpose_mul_self_eq_zero {m n : Type*} [Fintype m] {E : Matrix m n ℝ}
    (h : Eᵀ * E = 0) : E = 0 := by
  rw [← conjTranspose_eq_transpose_of_trivial] at h
  exact conjTranspose_mul_self_eq_zero.mp h

/-- Every real matrix has a least-squares `{1,3}`-inverse: `∃ Z, A Z A = A ∧ (A Z)ᵀ = A Z`.
Take `Z = H Aᵀ` with `H` a `{1}`-inverse of the Gram matrix `AᵀA` (helper for the existence of the
Moore–Penrose inverse). -/
theorem exists_lsq_inverse (A : Matrix ι κ ℝ) :
    ∃ Z : Matrix κ ι ℝ, A * Z * A = A ∧ (A * Z)ᵀ = A * Z := by
  obtain ⟨H, hH⟩ := exists_mul_mul_eq_self (Aᵀ * A)
  -- Step 1: `A H (AᵀA) = A`.
  have h1 : A * H * (Aᵀ * A) = A := by
    set E := A * H * (Aᵀ * A) - A with hE_def
    have hE : Aᵀ * E = 0 := by
      rw [hE_def, Matrix.mul_sub, sub_eq_zero]
      calc Aᵀ * (A * H * (Aᵀ * A)) = Aᵀ * A * H * (Aᵀ * A) := by simp only [Matrix.mul_assoc]
        _ = Aᵀ * A := hH
    have hEE : Eᵀ * E = 0 := by
      have hEt : Eᵀ = (Aᵀ * A)ᵀ * Hᵀ * Aᵀ - Aᵀ := by
        rw [hE_def, transpose_sub]
        simp only [Matrix.transpose_mul, transpose_transpose, Matrix.mul_assoc]
      rw [hEt, Matrix.sub_mul, Matrix.mul_assoc, hE, Matrix.mul_zero, sub_zero]
    exact sub_eq_zero.mp (eq_zero_of_transpose_mul_self_eq_zero hEE)
  -- Step 2: transposing step 1, `Aᵀ = (AᵀA) Hᵀ Aᵀ`.
  have h2 : Aᵀ = Aᵀ * A * Hᵀ * Aᵀ := by
    conv_lhs => rw [← h1]
    simp only [Matrix.transpose_mul, transpose_transpose, Matrix.mul_assoc]
  refine ⟨H * Aᵀ, ?_, ?_⟩
  · calc A * (H * Aᵀ) * A = A * H * (Aᵀ * A) := by simp only [Matrix.mul_assoc]
      _ = A := h1
  · calc (A * (H * Aᵀ))ᵀ = A * Hᵀ * Aᵀ := by
          rw [Matrix.transpose_mul, Matrix.transpose_mul, transpose_transpose, Matrix.mul_assoc]
      _ = A * H * (Aᵀ * A) * Hᵀ * Aᵀ := by rw [h1]
      _ = A * H * (Aᵀ * A * Hᵀ * Aᵀ) := by simp only [Matrix.mul_assoc]
      _ = A * (H * Aᵀ) := by rw [← h2, Matrix.mul_assoc]

/-- **Existence of the Moore–Penrose inverse** of every real matrix `B : Matrix ι κ ℝ`
(finite index types). With `Z` a `{1,3}`-inverse of `B` and `Y` a `{1,3}`-inverse of `Bᵀ`,
`Q = Bᵀ Y` is the orthogonal projector onto the row space and `X = Q Z` satisfies the four Penrose
equations (`BX = BZ`, `XB = Q`). No full-row-rank hypothesis is needed, in contrast to the
`pinvR` model. -/
theorem exists_isMoorePenrose (B : Matrix ι κ ℝ) : ∃ X : Matrix κ ι ℝ, IsMoorePenrose B X := by
  obtain ⟨Z, hZ1, hZ3⟩ := exists_lsq_inverse B
  obtain ⟨Y, hY1, hY3⟩ := exists_lsq_inverse Bᵀ
  set Q := Bᵀ * Y with hQ_def
  -- `Q` is symmetric, `Q = Yᵀ B`, `B Q = B`, `Q Q = Q`.
  have hQt : Qᵀ = Q := hY3
  have hQ' : Q = Yᵀ * B := by
    conv_lhs => rw [← hQt]
    rw [hQ_def, Matrix.transpose_mul, transpose_transpose]
  have hBQ : B * Q = B := by
    have h : (B * Q)ᵀ = Bᵀ := by
      rw [Matrix.transpose_mul, hQt, hQ_def, hY1]
    rw [← transpose_transpose (B * Q), h, transpose_transpose]
  have hQQ : Q * Q = Q := by
    rw [hQ_def, ← Matrix.mul_assoc, hY1]
  have hBX : B * (Q * Z) = B * Z := by rw [← Matrix.mul_assoc, hBQ]
  have hXB : Q * Z * B = Q := by
    conv_lhs => rw [hQ']
    rw [Matrix.mul_assoc, Matrix.mul_assoc, ← Matrix.mul_assoc B Z B, hZ1, ← hQ']
  refine ⟨Q * Z, ⟨?_, ?_, ?_, ?_⟩⟩
  · rw [hBX, hZ1]
  · rw [hXB, ← Matrix.mul_assoc, hQQ]
  · rw [hBX, hZ3]
  · rw [hXB, hQt]

/-! ## The Moore–Penrose inverse `mpinv` -/

/-- The Moore–Penrose inverse `B⁺` of a real matrix, chosen from `exists_isMoorePenrose`; it is
unique (`isMoorePenrose_iff_eq_mpinv`). This is the pseudoinverse of SI eq. (S4.7) and of
manuscript eqs. (6), (8) **without** any rank hypothesis. -/
noncomputable def mpinv (B : Matrix ι κ ℝ) : Matrix κ ι ℝ :=
  (exists_isMoorePenrose B).choose

/-- `mpinv B` satisfies the four Penrose equations. -/
theorem isMoorePenrose_mpinv (B : Matrix ι κ ℝ) : IsMoorePenrose B (mpinv B) :=
  (exists_isMoorePenrose B).choose_spec

/-- Penrose equation (1) for `mpinv`, unconditionally: `B B⁺ B = B` (SI §S4.4, "`XX⁺X = X`"). -/
theorem mul_mpinv_mul (B : Matrix ι κ ℝ) : B * mpinv B * B = B :=
  (isMoorePenrose_mpinv B).mul_mul_self

/-- Penrose equation (2) for `mpinv`: `B⁺ B B⁺ = B⁺`. -/
theorem mpinv_mul_mpinv (B : Matrix ι κ ℝ) : mpinv B * B * mpinv B = mpinv B :=
  (isMoorePenrose_mpinv B).mul_mul_self'

/-- Penrose equation (3) for `mpinv`: `(B B⁺)ᵀ = B B⁺`. -/
theorem transpose_mul_mpinv (B : Matrix ι κ ℝ) : (B * mpinv B)ᵀ = B * mpinv B :=
  (isMoorePenrose_mpinv B).transpose_mul

/-- Penrose equation (4) for `mpinv`: `(B⁺ B)ᵀ = B⁺ B`. -/
theorem transpose_mpinv_mul (B : Matrix ι κ ℝ) : (mpinv B * B)ᵀ = mpinv B * B :=
  (isMoorePenrose_mpinv B).transpose_mul'

/-- `X` satisfies the four Penrose equations iff it is `mpinv B` (existence + uniqueness). -/
theorem isMoorePenrose_iff_eq_mpinv {B : Matrix ι κ ℝ} {X : Matrix κ ι ℝ} :
    IsMoorePenrose B X ↔ X = mpinv B :=
  ⟨fun h => h.unique (isMoorePenrose_mpinv B), fun h => h ▸ isMoorePenrose_mpinv B⟩

/-- Under full row rank, `pinvR B = Bᵀ(BBᵀ)⁻¹` satisfies the four Penrose equations
(the Penrose identities of `FQGNN.Common.Pseudoinverse`). -/
theorem isMoorePenrose_pinvR [DecidableEq ι] {B : Matrix ι κ ℝ} (h : FullRowRank B) :
    IsMoorePenrose B (pinvR B) :=
  ⟨mul_pinvR_mul h, pinvR_mul_pinvR h, transpose_mul_pinvR h, transpose_pinvR_mul⟩

/-- **Agreement with the `pinvR` model:** under full row rank, `mpinv B = pinvR B = Bᵀ(BBᵀ)⁻¹`
(uniqueness and the Penrose identities of `pinvR`). So every statement made with `pinvR` under
`FullRowRank B` is a statement about the genuine Moore–Penrose inverse. -/
theorem mpinv_eq_pinvR [DecidableEq ι] {B : Matrix ι κ ℝ} (h : FullRowRank B) :
    mpinv B = pinvR B :=
  ((isMoorePenrose_pinvR h).unique (isMoorePenrose_mpinv B)).symm

/-- `B⁺ 0 = 0`: the pseudoinverse annihilates the zero vector (SI §S4.5 proof). -/
theorem mpinv_mulVec_zero (B : Matrix ι κ ℝ) : mpinv B *ᵥ (0 : ι → ℝ) = 0 :=
  mulVec_zero _

/-! ## Unconditional consequences of `B X B = B` -/

section GInverse

variable {B : Matrix ι κ ℝ} {X : Matrix κ ι ℝ}

/-- **SI §S4.4, "`X(I − X⁺X) = 0` unconditionally (no rank hypothesis)".** For every `X` with
`B X B = B` (in particular `X = mpinv B`), `B (1 − X B) = 0`. -/
theorem mul_one_sub_mul_eq_zero [DecidableEq κ] (hX : B * X * B = B) : B * (1 - X * B) = 0 := by
  rw [Matrix.mul_sub, Matrix.mul_one, ← Matrix.mul_assoc, hX, sub_self]

/-- `B (B⁺ d) = d` holds exactly on the range of `B`: for every `X` with `B X B = B`,
`B (X d) = d ↔ d ∈ range B` (SI Prop. S4.1, "`d ∈ range(B)`"). -/
theorem mulVec_mulVec_eq_iff_mem_range (hX : B * X * B = B) (d : ι → ℝ) :
    B *ᵥ (X *ᵥ d) = d ↔ d ∈ LinearMap.range (Matrix.mulVecLin B) := by
  constructor
  · intro h
    exact ⟨X *ᵥ d, by rw [mulVecLin_apply, h]⟩
  · rintro ⟨u, rfl⟩
    rw [mulVecLin_apply, mulVec_mulVec, mulVec_mulVec, hX]

end GInverse

/-- Full row rank makes every right-hand side consistent: `d ∈ range B` for all `d`
(SI Prop. S4.1: "full row rank `R` — equivalently `range(B) = ℝ^R`, so `d ∈ range(B)` whatever
the boundary data"). -/
theorem mem_range_of_fullRowRank [DecidableEq ι] {B : Matrix ι κ ℝ} (h : FullRowRank B)
    (d : ι → ℝ) : d ∈ LinearMap.range (Matrix.mulVecLin B) :=
  ⟨pinvR B *ᵥ d, by rw [mulVecLin_apply, mulVec_pinvR_mulVec h]⟩

/-- **The unconditional feasibility criterion for `mpinv`:**
`B (B⁺ d) = d ↔ d ∈ range B` (SI Prop. S4.1 / S6.1(i) without the rank hypothesis). -/
theorem mulVec_mpinv_mulVec_eq_iff (B : Matrix ι κ ℝ) (d : ι → ℝ) :
    B *ᵥ (mpinv B *ᵥ d) = d ↔ d ∈ LinearMap.range (Matrix.mulVecLin B) :=
  mulVec_mulVec_eq_iff_mem_range (mul_mpinv_mul B) d

/-- `B (B⁺ d) = d` for every consistent right-hand side `d ∈ range B` (no rank hypothesis). -/
theorem mulVec_mpinv_mulVec_of_mem_range {B : Matrix ι κ ℝ} {d : ι → ℝ}
    (hd : d ∈ LinearMap.range (Matrix.mulVecLin B)) : B *ᵥ (mpinv B *ᵥ d) = d :=
  (mulVec_mpinv_mulVec_eq_iff B d).mpr hd

/-- `B (B⁺ (B u)) = B u` for every `u` (no hypothesis). -/
theorem mulVec_mpinv_mulVec_mulVec (B : Matrix ι κ ℝ) (u : κ → ℝ) :
    B *ᵥ (mpinv B *ᵥ (B *ᵥ u)) = B *ᵥ u := by
  rw [mulVec_mulVec, mulVec_mulVec, mul_mpinv_mul]

/-! ## The null-space projector `P = 1 − B⁺B` without rank hypotheses -/

section ProjectorMP

variable [DecidableEq κ]

/-- Null-space projector built from the genuine Moore–Penrose inverse, `P_B = 1 − B⁺B`
(SI §S4.3, "the projector `I − (BW)⁺(BW)` maps them into `null(BW)`"). -/
noncomputable def projMP (B : Matrix ι κ ℝ) : Matrix κ κ ℝ := 1 - mpinv B * B

/-- Under full row rank `projMP B = proj B` (`mpinv_eq_pinvR`). -/
theorem projMP_eq_proj [DecidableEq ι] {B : Matrix ι κ ℝ} (h : FullRowRank B) :
    projMP B = proj B := by
  unfold projMP proj
  rw [mpinv_eq_pinvR h]

/-- **SI §S4.4, `X(I − X⁺X) = 0` unconditionally:** `B P_B = 0` for every real `B`. -/
theorem mul_projMP (B : Matrix ι κ ℝ) : B * projMP B = 0 :=
  mul_one_sub_mul_eq_zero (mul_mpinv_mul B)

/-- `B (P_B z) = 0` for every `z` and every real `B` (SI §S4.3, "so they can never violate
(S4.6)"), with no rank hypothesis. -/
theorem mulVec_projMP_mulVec (B : Matrix ι κ ℝ) (z : κ → ℝ) : B *ᵥ (projMP B *ᵥ z) = 0 := by
  rw [mulVec_mulVec, mul_projMP, zero_mulVec]

/-- `P_B` is symmetric (Penrose equation (4)). -/
theorem transpose_projMP (B : Matrix ι κ ℝ) : (projMP B)ᵀ = projMP B := by
  unfold projMP
  rw [transpose_sub, transpose_one, transpose_mpinv_mul]

/-- `P_B B⁺ = 0` (Penrose equation (2)). -/
theorem projMP_mul_mpinv (B : Matrix ι κ ℝ) : projMP B * mpinv B = 0 := by
  unfold projMP
  rw [Matrix.sub_mul, Matrix.one_mul, mpinv_mul_mpinv, sub_self]

/-- `P_B` is idempotent. -/
theorem projMP_mul_projMP (B : Matrix ι κ ℝ) : projMP B * projMP B = projMP B := by
  calc projMP B * projMP B = projMP B * (1 - mpinv B * B) := rfl
    _ = projMP B := by
      rw [Matrix.mul_sub, Matrix.mul_one, ← Matrix.mul_assoc, projMP_mul_mpinv, Matrix.zero_mul,
        sub_zero]

/-- `P_B` fixes `null B`. -/
theorem projMP_mulVec_of_mulVec_eq_zero {B : Matrix ι κ ℝ} {e : κ → ℝ} (he : B *ᵥ e = 0) :
    projMP B *ᵥ e = e := by
  unfold projMP
  rw [sub_mulVec, one_mulVec, ← mulVec_mulVec, he, mulVec_zero, sub_zero]

/-- `range P_B = null B`, unconditionally: `P_B z = z ↔ B z = 0`. -/
theorem projMP_mulVec_eq_self_iff (B : Matrix ι κ ℝ) (z : κ → ℝ) :
    projMP B *ᵥ z = z ↔ B *ᵥ z = 0 := by
  constructor
  · intro hz
    rw [← hz]
    exact mulVec_projMP_mulVec B z
  · exact projMP_mulVec_of_mulVec_eq_zero

/-- Feasible-set parameterization without a rank hypothesis: for a consistent right-hand side
`d ∈ range B`, `B u = d ↔ ∃ z, u = B⁺ d + P_B z` (SI §S7.1 proof). -/
theorem mulVec_eq_iff_exists_projMP {B : Matrix ι κ ℝ} {d : ι → ℝ}
    (hd : d ∈ LinearMap.range (Matrix.mulVecLin B)) (u : κ → ℝ) :
    B *ᵥ u = d ↔ ∃ z : κ → ℝ, u = mpinv B *ᵥ d + projMP B *ᵥ z := by
  constructor
  · intro hu
    refine ⟨u - mpinv B *ᵥ d, ?_⟩
    have he : B *ᵥ (u - mpinv B *ᵥ d) = 0 := by
      rw [mulVec_sub, hu, mulVec_mpinv_mulVec_of_mem_range hd, sub_self]
    rw [projMP_mulVec_of_mulVec_eq_zero he, add_sub_cancel]
  · rintro ⟨z, rfl⟩
    rw [mulVec_add, mulVec_mpinv_mulVec_of_mem_range hd, mulVec_projMP_mulVec, add_zero]

end ProjectorMP

/-! ## Minimum-norm characterization without a rank hypothesis -/

section MinNormMP

variable {B : Matrix ι κ ℝ}

/-- `B⁺ d ⊥ null B` for every `d` (Penrose equations (2) and (4)); no rank hypothesis. -/
theorem mpinv_mulVec_dotProduct_of_mulVec_eq_zero (d : ι → ℝ) {e : κ → ℝ} (he : B *ᵥ e = 0) :
    (mpinv B *ᵥ d) ⬝ᵥ e = 0 := by
  have h1 : mpinv B *ᵥ d = (mpinv B * B)ᵀ *ᵥ (mpinv B *ᵥ d) := by
    rw [transpose_mpinv_mul, mulVec_mulVec, mpinv_mul_mpinv]
  rw [h1, mulVec_transpose, ← dotProduct_mulVec, ← mulVec_mulVec, he, mulVec_zero,
    dotProduct_zero]

/-- Pythagoras on the feasible set, unconditionally: if `B u = d` then
`‖u‖² = ‖B⁺ d‖² + ‖u − B⁺ d‖²`. -/
theorem dotProduct_self_eq_mpinv_add {u : κ → ℝ} {d : ι → ℝ} (hu : B *ᵥ u = d) :
    u ⬝ᵥ u = (mpinv B *ᵥ d) ⬝ᵥ (mpinv B *ᵥ d) +
      (u - mpinv B *ᵥ d) ⬝ᵥ (u - mpinv B *ᵥ d) := by
  set u₀ := mpinv B *ᵥ d with hu₀
  set e := u - u₀ with he_def
  have he : B *ᵥ e = 0 := by
    rw [he_def, mulVec_sub, hu₀, ← hu, mulVec_mpinv_mulVec_mulVec, sub_self]
  have horth : u₀ ⬝ᵥ e = 0 := mpinv_mulVec_dotProduct_of_mulVec_eq_zero d he
  have hsplit : u = u₀ + e := by rw [he_def, add_sub_cancel]
  rw [hsplit, add_dotProduct, dotProduct_add, dotProduct_add, dotProduct_comm e u₀, horth]
  ring

/-- **Minimum norm, unconditionally:** `B⁺ d` has the least squared norm among all solutions of
`B u = d` (SI §S4.3 "min-norm solve"), with no rank hypothesis. -/
theorem mpinv_mulVec_dotProduct_le {u : κ → ℝ} {d : ι → ℝ} (hu : B *ᵥ u = d) :
    (mpinv B *ᵥ d) ⬝ᵥ (mpinv B *ᵥ d) ≤ u ⬝ᵥ u := by
  rw [dotProduct_self_eq_mpinv_add hu]
  exact le_add_of_nonneg_right (dotProduct_self_nonneg' _)

/-- Uniqueness of the minimum-norm solution, unconditionally. -/
theorem eq_mpinv_mulVec_of_dotProduct_le {u : κ → ℝ} {d : ι → ℝ} (hu : B *ᵥ u = d)
    (hle : u ⬝ᵥ u ≤ (mpinv B *ᵥ d) ⬝ᵥ (mpinv B *ᵥ d)) : u = mpinv B *ᵥ d := by
  rw [dotProduct_self_eq_mpinv_add hu] at hle
  have h0 : (u - mpinv B *ᵥ d) ⬝ᵥ (u - mpinv B *ᵥ d) = 0 :=
    le_antisymm (by linarith) (dotProduct_self_nonneg' _)
  exact sub_eq_zero.mp (dotProduct_self_eq_zero.mp h0)

/-- **Minimum-norm characterization, bundled, no rank hypothesis:** for every consistent
`d ∈ range B`, `B⁺ d` is feasible, has least squared norm on `{u | B u = d}`, and is the unique
feasible vector attaining it (SI §S4.3; no rank hypothesis). -/
theorem mpinv_mulVec_isMinNorm {d : ι → ℝ} (hd : d ∈ LinearMap.range (Matrix.mulVecLin B)) :
    B *ᵥ (mpinv B *ᵥ d) = d ∧
      (∀ u, B *ᵥ u = d → (mpinv B *ᵥ d) ⬝ᵥ (mpinv B *ᵥ d) ≤ u ⬝ᵥ u) ∧
      (∀ u, B *ᵥ u = d → u ⬝ᵥ u = (mpinv B *ᵥ d) ⬝ᵥ (mpinv B *ᵥ d) → u = mpinv B *ᵥ d) :=
  ⟨mulVec_mpinv_mulVec_of_mem_range hd, fun _ hu => mpinv_mulVec_dotProduct_le hu,
    fun _ hu heq => eq_mpinv_mulVec_of_dotProduct_le hu heq.le⟩

end MinNormMP

end FQGNN.Common

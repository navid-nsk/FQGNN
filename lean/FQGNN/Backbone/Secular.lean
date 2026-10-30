import Mathlib

/-!
# The Lee–Yang / Kurasov–Sarnak secular determinant

This module formalizes the secular-determinant layer of the FQ-GNN backbone:

* Manuscript (Results, eq. (2)):
  `p_U(z) = det(I − diag(z)·U)`, `p_U(0) = 1`.
* Supplementary Information (SI §S2.5),
  Theorem S2.8 (a)–(d) (Kurasov–Sarnak stability, self-inversive functional equation,
  Lee–Yang zero confinement / real-rootedness, torus zeros = eigenphase crossings).
* The normalization, real-rootedness, and the decoupled case `U = I`.
* Methods (Op-C residual): `p_U(Φ_L x)` vanishes exactly on the FQ point set.

The SI *cites* parts (a)–(c) of Theorem S2.8 from Kurasov–Sarnak rather than re-deriving them;
here they are proved from scratch for an arbitrary unitary `U` (elementary norm argument on a
fixed vector of `diag(z)·U`).
-/

noncomputable section

open Matrix Complex

namespace FQGNN.Backbone

variable {n : ℕ}

/-! ## The secular determinant -/

/-- The Kurasov–Sarnak secular determinant `p_U(z) = det(I − diag(z)·U)`.

Source: manuscript eq. (2); SI §S2.5 (before Thm S2.8). -/
def pU (U : Matrix (Fin n) (Fin n) ℂ) (z : Fin n → ℂ) : ℂ :=
  (1 - diagonal z * U).det

/-- Normalization `p_U(0) = 1`.

Source: manuscript eq. (2) ("the normalization `p_U(0)=1` fixes the constant term"). -/
theorem pU_zero (U : Matrix (Fin n) (Fin n) ℂ) : pU U 0 = 1 := by
  simp [pU]

/-- Decoupled case: `p_I(z) = ∏_j (1 − z_j)` (periodic comb degeneracy).

With `U = I` the coordinates decouple and the zero set is a union of coordinate hyperplanes
`z_j = 1`. -/
theorem pU_one (z : Fin n → ℂ) : pU 1 z = ∏ j, (1 - z j) := by
  rw [pU, Matrix.mul_one, ← diagonal_one, diagonal_sub, det_diagonal]

/-- **Principal-minor expansion.** `p_U(z) = Σ_{S ⊆ [n]} (−1)^{|S|} det(U_S) z^S`, where `U_S` is
the principal submatrix on the index set `S` and `z^S = ∏_{j ∈ S} z_j`. In particular `p_U` is
multi-affine (degree `≤ 1` in each `z_j`). Valid for every matrix `U`.

Source: SI §S1 eq. (S1.5) and the remark after it. -/
theorem pU_eq_sum_principal_minors (U : Matrix (Fin n) (Fin n) ℂ) (z : Fin n → ℂ) :
    pU U z = ∑ s : Finset (Fin n),
      (-1) ^ s.card * (∏ i ∈ s, z i) *
        (U.submatrix (Subtype.val : s → Fin n) Subtype.val).det := by
  let D := (detRowAlternating : (Fin n → ℂ) [⋀^Fin n]→ₗ[ℂ] ℂ)
  have hrows : (1 - diagonal z * U) =
      Matrix.of ((fun i => (-z i) • U.row i) + (1 : Matrix (Fin n) (Fin n) ℂ).row) := by
    ext i j
    simp [diagonal_mul, sub_eq_add_neg, add_comm]
  rw [pU, hrows]
  change D ((fun i => (-z i) • U.row i) + (1 : Matrix (Fin n) (Fin n) ℂ).row) = _
  rw [D.map_add_univ]
  refine Finset.sum_congr rfl fun s _ => ?_
  have hpw : s.piecewise (fun i => (-z i) • U.row i) (1 : Matrix (Fin n) (Fin n) ℂ).row =
      s.piecewise (fun i => (-z i) • (s.piecewise U.row (1 : Matrix (Fin n) (Fin n) ℂ).row) i)
        (s.piecewise U.row (1 : Matrix (Fin n) (Fin n) ℂ).row) := by
    ext i j
    by_cases hi : i ∈ s <;> simp [hi]
  rw [hpw]
  have hsm := D.toMultilinearMap.map_piecewise_smul (fun i => -z i)
    (s.piecewise U.row (1 : Matrix (Fin n) (Fin n) ℂ).row) s
  rw [AlternatingMap.coe_multilinearMap] at hsm
  rw [hsm]
  have hminor := det_piecewise_one_eq_submatrix_det U s
  change (∏ i ∈ s, -z i) *
    det (Matrix.of (s.piecewise U.row (1 : Matrix (Fin n) (Fin n) ℂ).row)) = _
  rw [hminor, Finset.prod_neg]

/-- `p_U(z) = 0` iff `diag(z)·U` has a nonzero fixed vector (eigenvalue `1`). Valid for every
matrix `U` and every `z ∈ ℂⁿ`.

Source: SI Thm S2.8(d). -/
theorem pU_eq_zero_iff_exists_fixed (U : Matrix (Fin n) (Fin n) ℂ) (z : Fin n → ℂ) :
    pU U z = 0 ↔ ∃ v : Fin n → ℂ, v ≠ 0 ∧ (diagonal z * U) *ᵥ v = v := by
  rw [pU, ← Matrix.exists_mulVec_eq_zero_iff]
  refine exists_congr fun v => and_congr_right fun _ => ?_
  rw [sub_mulVec, one_mulVec, sub_eq_zero, eq_comm]

/-- Spectral form: `p_U(z) = 0 ↔ 1 ∈ spec(diag(z)·U)`. Valid for every `U` and `z`.

Source: SI Thm S2.8(d) ("`p_U(z)=0` iff `1 ∈ spec(diag(z)U)`"). -/
theorem pU_eq_zero_iff_one_mem_spectrum (U : Matrix (Fin n) (Fin n) ℂ) (z : Fin n → ℂ) :
    pU U z = 0 ↔ (1 : ℂ) ∈ spectrum ℂ (diagonal z * U) := by
  rw [spectrum.mem_iff, map_one, Matrix.isUnit_iff_isUnit_det, isUnit_iff_ne_zero, not_not, pU]

/-! ## Norm bookkeeping for unitary matrices -/

/-- `Σ_i ‖v_i‖²`, cast to `ℂ`, is `star v ⬝ᵥ v`. -/
lemma ofReal_sum_norm_sq (v : Fin n → ℂ) :
    ((∑ i, ‖v i‖ ^ 2 : ℝ) : ℂ) = star v ⬝ᵥ v := by
  simp [dotProduct, Complex.conj_mul']

/-- A unitary matrix preserves `Σ_i ‖v_i‖²`. -/
lemma sum_norm_sq_mulVec_of_mem_unitaryGroup {U : Matrix (Fin n) (Fin n) ℂ}
    (hU : U ∈ unitaryGroup (Fin n) ℂ) (v : Fin n → ℂ) :
    ∑ i, ‖(U *ᵥ v) i‖ ^ 2 = ∑ i, ‖v i‖ ^ 2 := by
  have h : star (U *ᵥ v) ⬝ᵥ (U *ᵥ v) = star v ⬝ᵥ v := by
    rw [star_mulVec, ← dotProduct_mulVec, mulVec_mulVec, ← star_eq_conjTranspose,
      mem_unitaryGroup_iff'.mp hU, one_mulVec]
  apply Complex.ofReal_injective
  rw [ofReal_sum_norm_sq, ofReal_sum_norm_sq, h]

/-- Core of the stability argument: a zero of `p_U` (with `U` unitary) produces a nonzero vector
`w = U v` with `Σ_i ‖z_i‖² ‖w_i‖² = Σ_i ‖w_i‖²` (because `v = diag(z) w` and `‖Uv‖ = ‖v‖`). -/
lemma exists_weighted_balance {U : Matrix (Fin n) (Fin n) ℂ} (hU : U ∈ unitaryGroup (Fin n) ℂ)
    {z : Fin n → ℂ} (h : pU U z = 0) :
    ∃ w : Fin n → ℂ, (∃ i, w i ≠ 0) ∧
      ∑ i, ‖z i‖ ^ 2 * ‖w i‖ ^ 2 = ∑ i, ‖w i‖ ^ 2 := by
  obtain ⟨v, hv0, hv⟩ := (pU_eq_zero_iff_exists_fixed U z).1 h
  have hcoord : ∀ i, v i = z i * (U *ᵥ v) i := fun i => by
    have := congrFun hv i
    rw [← mulVec_mulVec, mulVec_diagonal] at this
    exact this.symm
  refine ⟨U *ᵥ v, ?_, ?_⟩
  · by_contra hc
    push Not at hc
    exact hv0 (funext fun i => by rw [hcoord i, hc i, mul_zero, Pi.zero_apply])
  · rw [sum_norm_sq_mulVec_of_mem_unitaryGroup hU v]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [hcoord i, norm_mul, mul_pow]

/-! ## Stability and exterior exclusion -/

/-- **Stability (Kurasov–Sarnak).** For `U` unitary, `p_U` has no zero in the open unit polydisc.

Source: SI Thm S2.8(a) ([KS2020], cited in the SI, proved here). -/
theorem pU_ne_zero_of_norm_lt_one {U : Matrix (Fin n) (Fin n) ℂ}
    (hU : U ∈ unitaryGroup (Fin n) ℂ) {z : Fin n → ℂ} (hz : ∀ j, ‖z j‖ < 1) :
    pU U z ≠ 0 := by
  intro h
  obtain ⟨w, ⟨i0, hi0⟩, hbal⟩ := exists_weighted_balance hU h
  have hlt : ∑ i, ‖z i‖ ^ 2 * ‖w i‖ ^ 2 < ∑ i, ‖w i‖ ^ 2 := by
    apply Finset.sum_lt_sum
    · intro i _
      exact mul_le_of_le_one_left (by positivity) (pow_le_one₀ (norm_nonneg _) (hz i).le)
    · exact ⟨i0, Finset.mem_univ _, mul_lt_of_lt_one_left (pow_pos (norm_pos_iff.2 hi0) 2)
        (pow_lt_one₀ (norm_nonneg _) (hz i0) two_ne_zero)⟩
  exact hlt.ne hbal

/-- **Exterior exclusion.** For `U` unitary, `p_U` has no zero with all `|z_j| > 1`.

Source: SI Thm S2.8(c), first clause ("no zero with `min_j |z_j| > 1`";
[KS2020] stable pair `(R_S, R_{S⁻¹})`, cited in the SI, proved here). -/
theorem pU_ne_zero_of_one_lt_norm {U : Matrix (Fin n) (Fin n) ℂ}
    (hU : U ∈ unitaryGroup (Fin n) ℂ) {z : Fin n → ℂ} (hz : ∀ j, 1 < ‖z j‖) :
    pU U z ≠ 0 := by
  intro h
  obtain ⟨w, ⟨i0, hi0⟩, hbal⟩ := exists_weighted_balance hU h
  have hlt : ∑ i, ‖w i‖ ^ 2 < ∑ i, ‖z i‖ ^ 2 * ‖w i‖ ^ 2 := by
    apply Finset.sum_lt_sum
    · intro i _
      exact le_mul_of_one_le_left (by positivity) (one_le_pow₀ (hz i).le)
    · exact ⟨i0, Finset.mem_univ _, lt_mul_of_one_lt_left (pow_pos (norm_pos_iff.2 hi0) 2)
        (one_lt_pow₀ (hz i0) two_ne_zero)⟩
  exact hlt.ne hbal.symm

/-- **Lee–Yang zero confinement.** For `U` unitary, every zero `z` of `p_U` satisfies
`min_j |z_j| ≤ 1 ≤ max_j |z_j|` (stated as: some coordinate has modulus `≥ 1` and some
coordinate has modulus `≤ 1`).

Source: SI Thm S2.8(c) ("every finite zero satisfies `min_j|z_j| ≤ 1 ≤ max_j|z_j|`"). -/
theorem pU_eq_zero_confinement {U : Matrix (Fin n) (Fin n) ℂ}
    (hU : U ∈ unitaryGroup (Fin n) ℂ) {z : Fin n → ℂ} (h : pU U z = 0) :
    (∃ j, 1 ≤ ‖z j‖) ∧ (∃ j, ‖z j‖ ≤ 1) := by
  constructor
  · by_contra hc
    push Not at hc
    exact pU_ne_zero_of_norm_lt_one hU hc h
  · by_contra hc
    push Not at hc
    exact pU_ne_zero_of_one_lt_norm hU hc h

/-! ## Real-rootedness along positive directions -/

/-- The point `(e^{iℓ_1 t}, …, e^{iℓ_n t})` on the torus flow in direction `ℓ`, at complex
time `t`.

Source: SI Thm S2.8(c) (the one-variable restriction `t ↦ p_U(e^{iℓ_1 t}, …, e^{iℓ_n t})`). -/
def expLine (ℓ : Fin n → ℝ) (t : ℂ) : Fin n → ℂ :=
  fun j => cexp (I * ℓ j * t)

/-- `|e^{iℓ_j t}| = e^{−ℓ_j Im t}`. -/
lemma norm_expLine (ℓ : Fin n → ℝ) (t : ℂ) (j : Fin n) :
    ‖expLine ℓ t j‖ = Real.exp (-(ℓ j * t.im)) := by
  simp [expLine, Complex.norm_exp, Complex.mul_re]

/-- **Real-rootedness along positive directions.** For `U` unitary and `ℓ ∈ (0,∞)ⁿ`, the
restriction `t ↦ p_U(e^{iℓ_1 t}, …, e^{iℓ_n t})` does not vanish off the real axis
(if `Im t > 0` all moduli are `< 1`, if `Im t < 0` all are `> 1`).

Source: SI Thm S2.8(c) and its proof ("if `Im t > 0` every coordinate has modulus
`< 1` … if `Im t < 0` every coordinate has modulus `> 1`"). -/
theorem pU_expLine_ne_zero {U : Matrix (Fin n) (Fin n) ℂ} (hU : U ∈ unitaryGroup (Fin n) ℂ)
    {ℓ : Fin n → ℝ} (hℓ : ∀ j, 0 < ℓ j) {t : ℂ} (ht : t.im ≠ 0) :
    pU U (expLine ℓ t) ≠ 0 := by
  rcases lt_or_gt_of_ne ht with h | h
  · refine pU_ne_zero_of_one_lt_norm hU fun j => ?_
    rw [norm_expLine, Real.one_lt_exp_iff]
    exact neg_pos.2 (mul_neg_of_pos_of_neg (hℓ j) h)
  · refine pU_ne_zero_of_norm_lt_one hU fun j => ?_
    rw [norm_expLine, Real.exp_lt_one_iff]
    exact neg_neg_of_pos (mul_pos (hℓ j) h)

/-- **Every zero of the one-variable restriction is real.** For `U` unitary and `ℓ ∈ (0,∞)ⁿ`,
if `p_U(e^{iℓ_1 t}, …, e^{iℓ_n t}) = 0` then `t` is real.

Source: SI Thm S2.8(c) ("the one-variable restriction … has only **real** zeros");
SI Remark S2.9. -/
theorem pU_expLine_eq_zero_imp_real {U : Matrix (Fin n) (Fin n) ℂ}
    (hU : U ∈ unitaryGroup (Fin n) ℂ) {ℓ : Fin n → ℝ} (hℓ : ∀ j, 0 < ℓ j) {t : ℂ}
    (h : pU U (expLine ℓ t) = 0) : ∃ r : ℝ, (r : ℂ) = t := by
  have him : t.im = 0 := by
    by_contra hne
    exact pU_expLine_ne_zero hU hℓ hne h
  exact ⟨t.re, Complex.ext (by simp) (by simp [him])⟩

/-! ## Torus zeros are eigenphase crossings -/

/-- For `z` on the torus `𝕋ⁿ` (`|z_j| = 1`), `diag(z)` is unitary.

Source: SI Thm S2.8(d) ("for `z ∈ 𝕋ⁿ`, `diag(z) ∈ U(n)`"). -/
theorem diagonal_mem_unitaryGroup {z : Fin n → ℂ} (hz : ∀ j, ‖z j‖ = 1) :
    diagonal z ∈ unitaryGroup (Fin n) ℂ := by
  rw [mem_unitaryGroup_iff', star_eq_conjTranspose, diagonal_conjTranspose,
    diagonal_mul_diagonal, ← diagonal_one]
  congr 1
  funext j
  simp [Complex.conj_mul', hz j]

/-- For `z ∈ 𝕋ⁿ` and `U` unitary, `diag(z)·U` is unitary.

Source: SI Thm S2.8(d) ("`diag(z)U ∈ U(n)`"). -/
theorem diagonal_mul_mem_unitaryGroup {U : Matrix (Fin n) (Fin n) ℂ}
    (hU : U ∈ unitaryGroup (Fin n) ℂ) {z : Fin n → ℂ} (hz : ∀ j, ‖z j‖ = 1) :
    diagonal z * U ∈ unitaryGroup (Fin n) ℂ :=
  mul_mem (diagonal_mem_unitaryGroup hz) hU

/-- **Torus zeros = eigenphase crossings.** For `U` unitary and `z ∈ 𝕋ⁿ`: `diag(z)·U` is
unitary, and `p_U(z) = 0` iff `diag(z)·U` has eigenvalue `1` (a nonzero fixed vector).

Source: SI Thm S2.8(d). -/
theorem torus_zero_iff_eigenvalue_one {U : Matrix (Fin n) (Fin n) ℂ}
    (hU : U ∈ unitaryGroup (Fin n) ℂ) {z : Fin n → ℂ} (hz : ∀ j, ‖z j‖ = 1) :
    diagonal z * U ∈ unitaryGroup (Fin n) ℂ ∧
      (pU U z = 0 ↔ ∃ v : Fin n → ℂ, v ≠ 0 ∧ (diagonal z * U) *ᵥ v = v) :=
  ⟨diagonal_mul_mem_unitaryGroup hU hz, pU_eq_zero_iff_exists_fixed U z⟩

/-! ## The Op-C residual and the FQ point set -/

/-- The torus lift `Φ_L(x)_j = exp(2πi (L x)_j)` for a generator `L : ℝ^{n×d}`.

Minimal definition for the secular module; `FQGNN.Backbone.TorusEmbedding`
develops it. Source: manuscript eq. (1). -/
def torusLift {d : ℕ} (L : Matrix (Fin n) (Fin d) ℝ) (x : Fin d → ℝ) : Fin n → ℂ :=
  fun j => cexp (2 * Real.pi * I * ((L *ᵥ x) j : ℂ))

/-- The torus lift lands on `𝕋ⁿ`: `|Φ_L(x)_j| = 1`.

Source: manuscript eq. (1) ("`Φ_L` maps each coordinate onto the `n`-torus"). -/
theorem norm_torusLift {d : ℕ} (L : Matrix (Fin n) (Fin d) ℝ) (x : Fin d → ℝ) (j : Fin n) :
    ‖torusLift L x j‖ = 1 := by
  rw [torusLift, Complex.norm_exp]
  simp [Complex.mul_re]

/-- The FQ point set `Λ(X, L) = {x ∈ ℝᵈ | Φ_L(x) ∈ X}` cut out by a torus set `X`.

Source: manuscript eq. (1) (pull-back of a torus variety). -/
def fqPointSet {d : ℕ} (X : Set (Fin n → ℂ)) (L : Matrix (Fin n) (Fin d) ℝ) :
    Set (Fin d → ℝ) :=
  {x | torusLift L x ∈ X}

/-- The secular variety `X_U = {z | p_U(z) = 0}`.

Source: manuscript eq. (2) ("whose torus zeros are the vertices of the point
set"). -/
def secularVariety (U : Matrix (Fin n) (Fin n) ℂ) : Set (Fin n → ℂ) :=
  {z | pU U z = 0}

/-- **Op-C residual.** `p_U(Φ_L x) = 0 ↔ x ∈ Λ(X_U, L)`.

Source: manuscript Methods, Op-C ("whose real and imaginary parts vanish exactly
when the coordinate lies on the quasicrystal"). -/
theorem pU_torusLift_eq_zero_iff {d : ℕ} (U : Matrix (Fin n) (Fin n) ℂ)
    (L : Matrix (Fin n) (Fin d) ℝ) (x : Fin d → ℝ) :
    pU U (torusLift L x) = 0 ↔ x ∈ fqPointSet (secularVariety U) L :=
  Iff.rfl

/-- **Op-C residual, real/imaginary form.** Both the real and the imaginary part of
`p_U(Φ_L x)` vanish iff `x ∈ Λ(X_U, L)`.

Source: manuscript Methods, Op-C ("whose real and imaginary parts vanish exactly
when the coordinate lies on the quasicrystal"). -/
theorem opC_residual_re_im_eq_zero_iff {d : ℕ} (U : Matrix (Fin n) (Fin n) ℂ)
    (L : Matrix (Fin n) (Fin d) ℝ) (x : Fin d → ℝ) :
    ((pU U (torusLift L x)).re = 0 ∧ (pU U (torusLift L x)).im = 0) ↔
      x ∈ fqPointSet (secularVariety U) L := by
  rw [← pU_torusLift_eq_zero_iff, Complex.ext_iff]
  simp

/-- **Points of the FQ set are eigenphase crossings.** For `U` unitary,
`x ∈ Λ(X_U, L)` iff the unitary `diag(Φ_L x)·U` has eigenvalue `1`.

Source: SI Thm S2.8(d) combined with the Op-C residual. -/
theorem mem_fqPointSet_iff_eigenvalue_one {d : ℕ} {U : Matrix (Fin n) (Fin n) ℂ}
    (hU : U ∈ unitaryGroup (Fin n) ℂ) (L : Matrix (Fin n) (Fin d) ℝ) (x : Fin d → ℝ) :
    diagonal (torusLift L x) * U ∈ unitaryGroup (Fin n) ℂ ∧
      (x ∈ fqPointSet (secularVariety U) L ↔
        ∃ v : Fin n → ℂ, v ≠ 0 ∧ (diagonal (torusLift L x) * U) *ᵥ v = v) :=
  torus_zero_iff_eigenvalue_one hU (norm_torusLift L x)

/-! ## Self-inversive functional equation -/

/-- **Self-inversive functional equation.** For `U` unitary and `z ∈ (ℂ^×)ⁿ`,
`(∏_j z̄_j) · p_U(1/z̄) = (−1)ⁿ det(U) · conj(p_U(z))`.

Source: SI Thm S2.8(b), eq. (S2.4) ([KS2020], cited in the SI, proved here). -/
theorem pU_functional_equation {U : Matrix (Fin n) (Fin n) ℂ}
    (hU : U ∈ unitaryGroup (Fin n) ℂ) {z : Fin n → ℂ} (hz : ∀ j, z j ≠ 0) :
    (∏ j, star (z j)) * pU U (fun j => (star (z j))⁻¹) =
      (-1) ^ n * U.det * star (pU U z) := by
  have hD : diagonal (star z) * diagonal (fun j => (star (z j))⁻¹) = 1 := by
    rw [diagonal_mul_diagonal, ← diagonal_one]
    congr 1
    funext j
    exact mul_inv_cancel₀ (star_ne_zero.2 (hz j))
  have hUU : U * Uᴴ = 1 := by
    rw [← star_eq_conjTranspose]
    exact mem_unitaryGroup_iff.mp hU
  calc (∏ j, star (z j)) * pU U (fun j => (star (z j))⁻¹)
      = (diagonal (star z) * (1 - diagonal (fun j => (star (z j))⁻¹) * U)).det := by
        rw [det_mul, det_diagonal, pU]
        rfl
    _ = (diagonal (star z) - U).det := by
        rw [Matrix.mul_sub, Matrix.mul_one, ← Matrix.mul_assoc, hD, Matrix.one_mul]
    _ = ((-U) * (1 - Uᴴ * diagonal (star z))).det := by
        rw [Matrix.mul_sub, Matrix.mul_one, ← Matrix.mul_assoc, Matrix.neg_mul, hUU,
          Matrix.neg_mul, Matrix.one_mul, sub_neg_eq_add, neg_add_eq_sub]
    _ = (-1) ^ n * U.det * star (pU U z) := by
        rw [det_mul, det_neg, Fintype.card_fin, pU, ← det_conjTranspose, conjTranspose_sub,
          conjTranspose_one, conjTranspose_mul, diagonal_conjTranspose]

end FQGNN.Backbone

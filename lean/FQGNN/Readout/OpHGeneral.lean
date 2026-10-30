import Mathlib
import FQGNN.Common.MoorePenrose
import FQGNN.Readout.OpH
import FQGNN.Readout.OpM

/-!
# Op-H-W and Op-M with the genuine Moore–Penrose inverse

`FQGNN.Readout.OpH` and `FQGNN.Readout.OpM` define the readouts of manuscript eqs. (6) and (8) with
the right-inverse model `pinvR B = Bᵀ(BBᵀ)⁻¹`, so their exactness theorems carry `FullRowRank B`.
Here the same operators are defined with the genuine Moore–Penrose inverse `mpinv` of
`FQGNN.Common.MoorePenrose`:

* `opHW_mp B w d g η = W (BW)⁺ d + g · W (I − (BW)⁺(BW)) η`  (manuscript eq. (6), SI eq. (S4.7));
* `opM_mp` = manuscript eq. (8) / SI eq. (S7.2) with `(BŴ)⁺ = mpinv (BŴ)`.

Results:
* both coincide with `opHW` / `opM` under `FullRowRank B` and `w > 0` (`opHW_mp_eq_opHW`,
  `opM_mp_eq_opM`), so every statement proved for `opHW` / `opM` under full row rank is a
  statement about the genuine Moore–Penrose inverse;
* **with no rank hypothesis**:
  - (d) the null-space annihilation `B (W (I − (BW)⁺(BW)) η) = 0` for every square `W`
    (SI §S4.4, "`X(I − X⁺X) = 0` unconditionally");
  - (a) `B c = d` for Op-H-W whenever `d ∈ range B` (every `w > 0`, `g`, `η`): SI Prop. S4.1 and
    Prop. S6.1(i) as the SI states them, and `B c = 0` for `d = 0` with no hypothesis at all
    (S6.1(i), "unconditionally when `d = 0`");
  - (b) SI Thm S7.3 for Op-M under `d ∈ range B`;
  - (c) SI Cor. S7.4, `B (Ŵ y) = 0`, with no hypothesis at all.

Sources: manuscript eqs. (6)–(8); SI §S4.3–S4.5 (Prop. S4.1 and its proof), §S6.1 (Prop. S6.1(i)), §S7.2 (Thm S7.3, Cor. S7.4), §S7.3 (Prop. 3).
-/

namespace FQGNN.Readout

open Matrix FQGNN.Common

variable {ι ο : Type*} [Fintype ι] [Fintype ο] {K : ℕ}

/-! ## (d) The null-space annihilation, unconditionally -/

section NullSpace

variable {κ : Type*} [Fintype κ] [DecidableEq κ]

/-- **(d) SI §S4.4, proof of Prop. S4.1: "`X(I − X⁺X) = 0` unconditionally (no rank hypothesis)",
with `X = BW`.** For every real `B`, every square `W` (no invertibility needed) and every `η`:
`B (W ((1 − (BW)⁺(BW)) η)) = 0`. -/
theorem mulVec_mul_one_sub_mpinv_mulVec (B : Matrix ι κ ℝ) (W : Matrix κ κ ℝ) (η : κ → ℝ) :
    B *ᵥ (W *ᵥ ((1 - mpinv (B * W) * (B * W)) *ᵥ η)) = 0 := by
  rw [mulVec_mulVec, mulVec_mulVec,
    mul_one_sub_mul_eq_zero (mul_mpinv_mul (B * W)), zero_mulVec]

/-- Same statement with the projector `projMP (BW) = 1 − (BW)⁺(BW)`. -/
theorem mulVec_mul_projMP_mulVec (B : Matrix ι κ ℝ) (W : Matrix κ κ ℝ) (η : κ → ℝ) :
    B *ᵥ (W *ᵥ (projMP (B * W) *ᵥ η)) = 0 :=
  mulVec_mul_one_sub_mpinv_mulVec B W η

omit [Fintype ι] in
/-- An invertible right factor does not change the range: `range (B W) = range B`
(SI §S4.4 proof of Prop. S4.1, "`W` is invertible, so `range(BW) = range(B)`"). -/
theorem mem_range_mul_iff {B : Matrix ι κ ℝ} {W : Matrix κ κ ℝ} (hW : IsUnit W.det)
    (d : ι → ℝ) :
    d ∈ LinearMap.range (Matrix.mulVecLin (B * W)) ↔ d ∈ LinearMap.range (Matrix.mulVecLin B) := by
  constructor
  · rintro ⟨u, rfl⟩
    exact ⟨W *ᵥ u, by rw [mulVecLin_apply, mulVecLin_apply, mulVec_mulVec]⟩
  · rintro ⟨u, rfl⟩
    refine ⟨W⁻¹ *ᵥ u, ?_⟩
    rw [mulVecLin_apply, mulVecLin_apply, mulVec_mulVec, Matrix.mul_assoc,
      mul_nonsing_inv _ hW, Matrix.mul_one]

end NullSpace

/-! ## Op-H-W with the Moore–Penrose inverse -/

/-- Op-H-W with the genuine Moore–Penrose inverse (manuscript eq. (6), SI eq. (S4.7)):
`c = W (BW)⁺ d + g · W (I − (BW)⁺(BW)) η`, `(BW)⁺ = mpinv (BW)`. Mirrors `opHW` with `mpinv` in
place of `pinvR`. -/
noncomputable def opHW_mp (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) : Fin K ⊕ Fin K → ℝ :=
  metricW w *ᵥ (mpinv (B * metricW w) *ᵥ d) + g • (metricW w *ᵥ (projMP (B * metricW w) *ᵥ η))

/-- Under full row rank of `BW` the two definitions agree: `opHW_mp = opHW`. -/
theorem opHW_mp_eq_opHW_of_fullRowRank [DecidableEq ι] {B : Matrix ι (Fin K ⊕ Fin K) ℝ}
    {w : Fin K → ℝ} (h : FullRowRank (B * metricW w)) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) : opHW_mp B w d g η = opHW B w d g η := by
  unfold opHW_mp opHW
  rw [mpinv_eq_pinvR h, projMP_eq_proj h]

/-- **Agreement with the right-inverse model.** For full row rank `B` and `w > 0`,
`opHW_mp B w d g η = opHW B w d g η`; every theorem about `opHW` is a theorem about eq. (6) with the
genuine Moore–Penrose inverse. -/
theorem opHW_mp_eq_opHW [DecidableEq ι] {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B)
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    opHW_mp B w d g η = opHW B w d g η :=
  opHW_mp_eq_opHW_of_fullRowRank (fullRowRank_mul_metricW hB hw) d g η

/-- **(d) for the metric of eq. (6), no hypothesis:** the null-space channel never touches the
constraints, `B (W (I − (BW)⁺(BW)) η) = 0`, for every `B`, every `w` (not even `w > 0` is needed)
and every `η` (SI §S4.4; manuscript: "`η` ranges over the null space, where it can never violate
the constraints"). -/
theorem mulVec_metricW_projMP_mulVec (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (η : Fin K ⊕ Fin K → ℝ) : B *ᵥ (metricW w *ᵥ (projMP (B * metricW w) *ᵥ η)) = 0 :=
  mulVec_mul_projMP_mulVec B (metricW w) η

/-- The particular part is feasible whenever `d ∈ range (BW)`, with no other hypothesis:
`B (W (BW)⁺ d) = d`. -/
theorem mulVec_metricW_mpinv_mulVec_of_mem_range' {B : Matrix ι (Fin K ⊕ Fin K) ℝ}
    {w : Fin K → ℝ} {d : ι → ℝ} (hd : d ∈ LinearMap.range (Matrix.mulVecLin (B * metricW w))) :
    B *ᵥ (metricW w *ᵥ (mpinv (B * metricW w) *ᵥ d)) = d := by
  rw [mulVec_mulVec]
  exact mulVec_mpinv_mulVec_of_mem_range hd

/-- The particular part is feasible for every consistent `d ∈ range B` and every metric `w > 0`
(SI §S4.4 proof: "`B[W(BW)⁺d] = (BW)(BW)⁺d = d`"), with no rank hypothesis. -/
theorem mulVec_metricW_mpinv_mulVec_of_mem_range {B : Matrix ι (Fin K ⊕ Fin K) ℝ}
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) {d : ι → ℝ}
    (hd : d ∈ LinearMap.range (Matrix.mulVecLin B)) :
    B *ᵥ (metricW w *ᵥ (mpinv (B * metricW w) *ᵥ d)) = d :=
  mulVec_metricW_mpinv_mulVec_of_mem_range'
    ((mem_range_mul_iff (isUnit_det_metricW hw) d).mpr hd)

/-- Exactness of Op-H-W for every `d ∈ range (BW)` and every `w`, `g`, `η` (no rank hypothesis). -/
theorem mulVec_opHW_mp_of_mem_range' {B : Matrix ι (Fin K ⊕ Fin K) ℝ} {w : Fin K → ℝ}
    {d : ι → ℝ} (hd : d ∈ LinearMap.range (Matrix.mulVecLin (B * metricW w))) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) : B *ᵥ opHW_mp B w d g η = d := by
  unfold opHW_mp
  rw [mulVec_add, mulVec_smul, mulVec_metricW_mpinv_mulVec_of_mem_range' hd,
    mulVec_metricW_projMP_mulVec, smul_zero, add_zero]

/-- **(a) SI Prop. S4.1 and Prop. S6.1(i), as the SI states them, with no rank hypothesis.** For
every consistent right-hand side `d ∈ range B`, every metric `w > 0`, every gate `g` and every `η`,
the Op-H-W coefficients of eq. (6) (with the genuine Moore–Penrose inverse) satisfy `B c = d`. -/
theorem mulVec_opHW_mp {B : Matrix ι (Fin K ⊕ Fin K) ℝ} {w : Fin K → ℝ} (hw : ∀ k, 0 < w k)
    {d : ι → ℝ} (hd : d ∈ LinearMap.range (Matrix.mulVecLin B)) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) : B *ᵥ opHW_mp B w d g η = d :=
  mulVec_opHW_mp_of_mem_range' ((mem_range_mul_iff (isUnit_det_metricW hw) d).mpr hd) g η

/-- **(a), sharp form.** For `w > 0`, exact interpolation holds **iff** the boundary system is
consistent: `B (opHW_mp B w d g η) = d ↔ d ∈ range B`, for every `g`, `η` (no rank hypothesis). -/
theorem mulVec_opHW_mp_eq_iff {B : Matrix ι (Fin K ⊕ Fin K) ℝ} {w : Fin K → ℝ}
    (hw : ∀ k, 0 < w k) (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    B *ᵥ opHW_mp B w d g η = d ↔ d ∈ LinearMap.range (Matrix.mulVecLin B) := by
  constructor
  · intro h
    exact ⟨opHW_mp B w d g η, by rw [mulVecLin_apply, h]⟩
  · intro hd
    exact mulVec_opHW_mp hw hd g η

/-- **SI Prop. S6.1(i), "unconditionally when `d = 0`".** `B c(w) = 0` for `d = 0`, for every `B`,
every `w` (no positivity needed), every `g` and every `η`. -/
theorem mulVec_opHW_mp_zero (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) : B *ᵥ opHW_mp B w 0 g η = 0 :=
  mulVec_opHW_mp_of_mem_range' (Submodule.zero_mem _) g η

/-- SI Prop. S4.1 under its own hypothesis (full row rank, so every `d` is consistent), for the
Moore–Penrose form of eq. (6). -/
theorem mulVec_opHW_mp_of_fullRowRank [DecidableEq ι] {B : Matrix ι (Fin K ⊕ Fin K) ℝ}
    (hB : FullRowRank B) {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) : B *ᵥ opHW_mp B w d g η = d :=
  mulVec_opHW_mp hw (mem_range_of_fullRowRank hB d) g η

/-- **SI Prop. S4.1, packaged on the boundary-Hermite system, with no rank hypothesis.** If the
boundary-Hermite system (S4.6) is consistent (`d ∈ range B`; automatic under full row rank),
`a ≠ b`, `w > 0` and the equilibration factors are nonzero, then for every `η` and every gate `g`
the reconstruction interpolates both anchors and matches every included boundary velocity. -/
theorem opHW_mp_hermite_interpolation (ω : Fin K → ℝ) {a b : ℝ} (hab : a ≠ b) (pa pb : ℝ)
    {Rv : ℕ} (te s v : Fin Rv → ℝ) (hs : ∀ e, s e ≠ 0)
    (hd : hermiteD a b pa pb s v ∈ LinearMap.range (Matrix.mulVecLin (hermiteB ω a b te s)))
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    let c := opHW_mp (hermiteB ω a b te s) w (hermiteD a b pa pb s v) g η
    recon ω c a b pa pb a = pa ∧ recon ω c a b pa pb b = pb ∧
      ∀ e, HasDerivAt (recon ω c a b pa pb) (v e) (te e) := by
  intro c
  have hc : hermiteB ω a b te s *ᵥ c = hermiteD a b pa pb s v := mulVec_opHW_mp hw hd g η
  refine ⟨?_, ?_, fun e => ?_⟩
  · exact recon_left_of_valueRow hc ω a b pa pb (i := Sum.inl 0) rfl rfl
  · exact recon_right_of_valueRow hc ω hab pa pb (i := Sum.inl 1) rfl rfl
  · exact hasDerivAt_recon_of_derivRow hc ω a b pa pb (te e) (v e) (s e) (hs e)
      (i := Sum.inr e) rfl rfl

/-- Manuscript Prop. 1 (coefficient form) for the Moore–Penrose form of eq. (6): `d = 0` and
`g = 0` give `c = 0`, for every `B`, `w`, `η` (SI §S4.5). -/
theorem opHW_mp_zero_zero (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (η : Fin K ⊕ Fin K → ℝ) : opHW_mp B w 0 0 η = 0 := by
  unfold opHW_mp
  rw [mulVec_zero, mulVec_zero, zero_smul, add_zero]

/-- **Manuscript Prop. 1 (affine reproduction) for the Moore–Penrose form of eq. (6)**, with the
same hypotheses as `prop1_affine_reproduction` (affine constant-velocity track, exact included
boundary velocities, gate enabled, `R_v ≥ 1`): `c = 0` and `p̂(t) = π(P t)` for every `t`, every
`η`, every `w`, every `B` (SI §S4.5). -/
theorem prop1_affine_reproduction_mp {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (π : E →ₗ[ℝ] ℝ) {P : ℝ → E} {U₀ U₁ : E} (hP : ∀ t, P t = U₀ + t • U₁)
    {a b : ℝ} (hab : a ≠ b) (ω : Fin K → ℝ) {Rv : ℕ} (hRv : 0 < Rv) (te s : Fin Rv → ℝ)
    {V : Fin Rv → E} (hV : ∀ e, V e = U₁) (w : Fin K → ℝ) (ρ ε : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) :
    let g := gateValue true ρ ε V (chordSlopeVec a b (P a) (P b))
    let c := opHW_mp (hermiteB ω a b te s) w
      (hermiteD a b (π (P a)) (π (P b)) s (fun e => π (V e))) g η
    c = 0 ∧ ∀ t, recon ω c a b (π (P a)) (π (P b)) t = π (P t) := by
  intro g c
  have hA : chordSlopeVec a b (P a) (P b) = U₁ := chordSlopeVec_of_affine hP (Ne.symm hab)
  have hg : g = 0 := gateValue_eq_zero hRv ρ ε (fun e => (hV e).trans hA.symm)
  have hp : ∀ t, π (P t) = π U₀ + π U₁ * t := fun t => by
    rw [hP t, map_add, map_smul, smul_eq_mul, mul_comm]
  have hd : hermiteD a b (π (P a)) (π (P b)) s (fun e => π (V e)) = 0 :=
    hermiteD_eq_zero_of_affine (p := fun t => π (P t)) hp hab s (fun e => by simp [hV e])
  have hc : c = 0 := by
    change opHW_mp (hermiteB ω a b te s) w _ g η = 0
    rw [hd, hg, opHW_mp_zero_zero]
  refine ⟨hc, fun t => ?_⟩
  rw [hc, recon_zero]
  exact (chord_eq_of_affine (p := fun t => π (P t)) hp hab).1 t

/-! ## Op-M with the Moore–Penrose inverse -/

section OpM

variable [DecidableEq ο]

/-- Exterior innovation `r₀ = ρ̃ − Ṽ(u₀ + gPη)` with `u₀ = (BŴ)⁺d`, `P = I − (BŴ)⁺(BŴ)` and the
genuine Moore–Penrose inverse (manuscript eq. (8); SI eq. (S7.2)). Mirrors `opMInnov`. -/
noncomputable def opMInnov_mp (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) : ο → ℝ :=
  ρ - (V * metricW w) *ᵥ
    (mpinv (B * metricW w) *ᵥ d + g • (projMP (B * metricW w) *ᵥ η))

/-- Soft-block Gram matrix `G̃ = Λ^{1/2} Ṽ P Ṽᵀ Λ^{1/2}` with `P = I − (BŴ)⁺(BŴ)` (manuscript
eq. (8); SI eq. (S7.2)). Mirrors `opMGram`. -/
noncomputable def opMGram_mp (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam : ο → ℝ) : Matrix ο ο ℝ :=
  sqrtPrec lam * (V * metricW w) * projMP (B * metricW w) * (V * metricW w)ᵀ * sqrtPrec lam

/-- Exterior-channel term `y = P Ṽᵀ Λ^{1/2} (I + G̃)⁻¹ Λ^{1/2} r₀` (manuscript eq. (8);
SI eq. (S7.2)). Mirrors `opMSoft`. -/
noncomputable def opMSoft_mp (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) : Fin K ⊕ Fin K → ℝ :=
  (projMP (B * metricW w) * (V * metricW w)ᵀ * sqrtPrec lam * (1 + opMGram_mp B w V lam)⁻¹ *
    sqrtPrec lam) *ᵥ opMInnov_mp B w V ρ d g η

/-- Op-M closed form with the genuine Moore–Penrose inverse, manuscript eq. (8) / SI eq. (S7.2):
`c = Ŵ (u₀ + g P η + y)`. Mirrors `opM`. -/
noncomputable def opM_mp (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) : Fin K ⊕ Fin K → ℝ :=
  metricW w *ᵥ (mpinv (B * metricW w) *ᵥ d + g • (projMP (B * metricW w) *ᵥ η) +
    opMSoft_mp B w V lam ρ d g η)

/-- Under full row rank of `BŴ` the two definitions of Op-M agree: `opM_mp = opM`. -/
theorem opM_mp_eq_opM_of_fullRowRank [DecidableEq ι] {B : Matrix ι (Fin K ⊕ Fin K) ℝ}
    {w : Fin K → ℝ} (h : FullRowRank (B * metricW w)) (V : Matrix ο (Fin K ⊕ Fin K) ℝ)
    (lam ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    opM_mp B w V lam ρ d g η = opM B w V lam ρ d g η := by
  unfold opM_mp opM opMSoft_mp opMSoft opMInnov_mp opMInnov opMGram_mp opMGram
  rw [mpinv_eq_pinvR h, projMP_eq_proj h]

/-- **Agreement with the right-inverse model.** For full row rank `B` and `ŵ > 0`,
`opM_mp = opM`; every theorem about `opM` is a theorem about eq. (8) with the genuine
Moore–Penrose inverse. -/
theorem opM_mp_eq_opM [DecidableEq ι] {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B)
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ)
    (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    opM_mp B w V lam ρ d g η = opM B w V lam ρ d g η :=
  opM_mp_eq_opM_of_fullRowRank (fullRowRank_mul_metricW hB hw) V lam ρ d g η

/-- Op-M is Op-H-W plus the exterior contribution `Ŵ y` (SI §S7.3), for the
Moore–Penrose forms; holds for every argument. -/
theorem opM_mp_eq_opHW_mp_add (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) :
    opM_mp B w V lam ρ d g η = opHW_mp B w d g η + metricW w *ᵥ opMSoft_mp B w V lam ρ d g η := by
  unfold opM_mp opHW_mp
  rw [mulVec_add, mulVec_add, mulVec_smul]

/-- `y = P (Ṽᵀ Λ^{1/2} (I + G̃)⁻¹ Λ^{1/2} r₀)`: the exterior term lies in `range P`
(SI Thm S7.3 proof, "`y ∈ range(P)`"). -/
theorem opMSoft_mp_eq_projMP_mulVec (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) :
    opMSoft_mp B w V lam ρ d g η = projMP (B * metricW w) *ᵥ
      (((V * metricW w)ᵀ * sqrtPrec lam * (1 + opMGram_mp B w V lam)⁻¹ * sqrtPrec lam) *ᵥ
        opMInnov_mp B w V ρ d g η) := by
  unfold opMSoft_mp
  rw [mulVec_mulVec]
  simp only [Matrix.mul_assoc]

/-- **(c) SI Cor. S7.4 (endpoint annihilation), algebraic form, with no hypothesis at all.** The
exterior channel's contribution `Ŵ y` lies in the null space of the hard constraints:
`B (Ŵ y) = 0` for every `B`, `ŵ`, `V`, `λ`, `ρ̃`, `d`, `g`, `η` (no rank, no positivity). -/
theorem mulVec_metricW_opMSoft_mp (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) : B *ᵥ (metricW w *ᵥ opMSoft_mp B w V lam ρ d g η) = 0 := by
  rw [opMSoft_mp_eq_projMP_mulVec]
  exact mulVec_metricW_projMP_mulVec B w _

/-- **(c) SI Cor. S7.4, value rows, no hypothesis:** if row `i` of `B` is the value row `φ(τ_e)`,
the exterior contribution vanishes there, `r_{Ŵy}(τ_e) = 0`. -/
theorem rcomb_exterior_mp_eq_zero_of_valueRow (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) (ω : Fin K → ℝ) {i : ι} {τe : ℝ} (hrow : B i = phi ω τe) :
    rcomb ω (metricW w *ᵥ opMSoft_mp B w V lam ρ d g η) τe = 0 := by
  have h := congrFun (mulVec_metricW_opMSoft_mp B w V lam ρ d g η) i
  unfold rcomb
  rw [← hrow]
  exact h

/-- **(c) SI Cor. S7.4, included derivative rows, no rank hypothesis:** if row `i` of `B` is the
equilibrated derivative row `s • φ'(τ_e)` with `s ≠ 0`, the exterior contribution has zero slope
there. -/
theorem hasDerivAt_rcomb_exterior_mp_of_derivRow (B : Matrix ι (Fin K ⊕ Fin K) ℝ)
    (w : Fin K → ℝ) (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) (ω : Fin K → ℝ) {i : ι} {τe s : ℝ} (hs : s ≠ 0)
    (hrow : B i = s • dphi ω τe) :
    HasDerivAt (rcomb ω (metricW w *ᵥ opMSoft_mp B w V lam ρ d g η)) 0 τe := by
  have h : B i ⬝ᵥ (metricW w *ᵥ opMSoft_mp B w V lam ρ d g η) = 0 :=
    congrFun (mulVec_metricW_opMSoft_mp B w V lam ρ d g η) i
  rw [hrow, smul_dotProduct, smul_eq_mul] at h
  have h' : dphi ω τe ⬝ᵥ (metricW w *ᵥ opMSoft_mp B w V lam ρ d g η) = 0 :=
    (mul_eq_zero.mp h).resolve_left hs
  have hd := hasDerivAt_rcomb ω (metricW w *ᵥ opMSoft_mp B w V lam ρ d g η) τe
  rwa [h'] at hd

/-- **(c) SI Cor. S7.4, packaged on the boundary-Hermite system, with no rank hypothesis and no
metric hypothesis.** "The soft anchors can reshape the manoeuvre; they can never move the
endpoints": the exterior deformation vanishes at both bracketing anchors and has zero derivative at
every included velocity-row time, for every `λ`, `V`, `ρ̃`, `g`, `η`, `ŵ`. -/
theorem exterior_endpoint_annihilation_mp (ω : Fin K → ℝ) (a b pa pb : ℝ) {Rv : ℕ}
    (te s v : Fin Rv → ℝ) (hs : ∀ e, s e ≠ 0) (w : Fin K → ℝ) (V : Matrix ο (Fin K ⊕ Fin K) ℝ)
    (lam ρ : ο → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    let δ := rcomb ω (metricW w *ᵥ
      opMSoft_mp (hermiteB ω a b te s) w V lam ρ (hermiteD a b pa pb s v) g η)
    δ (ctime a b a) = 0 ∧ δ (ctime a b b) = 0 ∧ ∀ e, HasDerivAt δ 0 (ctime a b (te e)) := by
  refine ⟨?_, ?_, fun e => ?_⟩
  · exact rcomb_exterior_mp_eq_zero_of_valueRow _ w V lam ρ _ g η ω (i := Sum.inl 0) rfl
  · exact rcomb_exterior_mp_eq_zero_of_valueRow _ w V lam ρ _ g η ω (i := Sum.inl 1) rfl
  · exact hasDerivAt_rcomb_exterior_mp_of_derivRow _ w V lam ρ _ g η ω (hs e)
      (i := Sum.inr e) rfl

/-- Exactness of Op-M for every `d ∈ range (BŴ)` and every `ŵ`, `V`, `λ`, `ρ̃`, `g`, `η` (no rank
hypothesis). -/
theorem mulVec_opM_mp_of_mem_range' {B : Matrix ι (Fin K ⊕ Fin K) ℝ} {w : Fin K → ℝ}
    {d : ι → ℝ} (hd : d ∈ LinearMap.range (Matrix.mulVecLin (B * metricW w)))
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    B *ᵥ opM_mp B w V lam ρ d g η = d := by
  rw [opM_mp_eq_opHW_mp_add, mulVec_add, mulVec_opHW_mp_of_mem_range' hd,
    mulVec_metricW_opMSoft_mp, add_zero]

/-- **(b) SI Thm S7.3 (exactness survives every dial) with no rank hypothesis.** For every
consistent `d ∈ range B` and every metric `ŵ > 0`, the Op-M coefficients of eq. (8) satisfy
`B c = d` for every exterior layout `V`, precisions `λ`, residuals `ρ̃`, gate `g` and `η`. -/
theorem mulVec_opM_mp {B : Matrix ι (Fin K ⊕ Fin K) ℝ} {w : Fin K → ℝ} (hw : ∀ k, 0 < w k)
    {d : ι → ℝ} (hd : d ∈ LinearMap.range (Matrix.mulVecLin B))
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    B *ᵥ opM_mp B w V lam ρ d g η = d :=
  mulVec_opM_mp_of_mem_range' ((mem_range_mul_iff (isUnit_det_metricW hw) d).mpr hd) V lam ρ g η

/-- **(b), sharp form.** For `ŵ > 0`, `B (opM_mp …) = d ↔ d ∈ range B`, for every `V`, `λ`, `ρ̃`,
`g`, `η`. -/
theorem mulVec_opM_mp_eq_iff {B : Matrix ι (Fin K ⊕ Fin K) ℝ} {w : Fin K → ℝ}
    (hw : ∀ k, 0 < w k) (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) :
    B *ᵥ opM_mp B w V lam ρ d g η = d ↔ d ∈ LinearMap.range (Matrix.mulVecLin B) := by
  constructor
  · intro h
    exact ⟨opM_mp B w V lam ρ d g η, by rw [mulVecLin_apply, h]⟩
  · intro hd
    exact mulVec_opM_mp hw hd V lam ρ g η

/-- SI Thm S7.3 under full row rank (every `d` consistent), for the Moore–Penrose form of
eq. (8). -/
theorem mulVec_opM_mp_of_fullRowRank [DecidableEq ι] {B : Matrix ι (Fin K ⊕ Fin K) ℝ}
    (hB : FullRowRank B) {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (V : Matrix ο (Fin K ⊕ Fin K) ℝ)
    (lam ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    B *ᵥ opM_mp B w V lam ρ d g η = d :=
  mulVec_opM_mp hw (mem_range_of_fullRowRank hB d) V lam ρ g η

/-- **(b) SI Thm S7.3, geometric form on the boundary-Hermite system, with no rank hypothesis.**
If the boundary-Hermite system is consistent, `a ≠ b`, `ŵ > 0` and the equilibration factors are
nonzero, the Op-M reconstruction interpolates both anchors and matches every included boundary
velocity, for every `λ`, `V`, `ρ̃`, `g`, `η`. -/
theorem opM_mp_hermite_interpolation (ω : Fin K → ℝ) {a b : ℝ} (hab : a ≠ b) (pa pb : ℝ)
    {Rv : ℕ} (te s v : Fin Rv → ℝ) (hs : ∀ e, s e ≠ 0)
    (hd : hermiteD a b pa pb s v ∈ LinearMap.range (Matrix.mulVecLin (hermiteB ω a b te s)))
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (lam ρ : ο → ℝ)
    (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    let c := opM_mp (hermiteB ω a b te s) w V lam ρ (hermiteD a b pa pb s v) g η
    recon ω c a b pa pb a = pa ∧ recon ω c a b pa pb b = pb ∧
      ∀ e, HasDerivAt (recon ω c a b pa pb) (v e) (te e) := by
  intro c
  have hc : hermiteB ω a b te s *ᵥ c = hermiteD a b pa pb s v :=
    mulVec_opM_mp hw hd V lam ρ g η
  refine ⟨?_, ?_, fun e => ?_⟩
  · exact recon_left_of_valueRow hc ω a b pa pb (i := Sum.inl 0) rfl rfl
  · exact recon_right_of_valueRow hc ω hab pa pb (i := Sum.inl 1) rfl rfl
  · exact hasDerivAt_recon_of_derivRow hc ω a b pa pb (te e) (v e) (s e) (hs e)
      (i := Sum.inr e) rfl rfl

/-- **Manuscript Prop. 3, `Λ`-dial, for the Moore–Penrose forms (SI §S7.3).** At `Λ = 0`,
`opM_mp B w V 0 ρ d g η = opHW_mp B w d g η` for every argument, with no hypothesis. -/
theorem opM_mp_zero_prec (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (V : Matrix ο (Fin K ⊕ Fin K) ℝ) (ρ : ο → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) : opM_mp B w V 0 ρ d g η = opHW_mp B w d g η := by
  rw [opM_mp_eq_opHW_mp_add]
  unfold opMSoft_mp
  rw [sqrtPrec_zero, Matrix.mul_zero, zero_mulVec, mulVec_zero, add_zero]

end OpM

end FQGNN.Readout

import Mathlib
import FQGNN.Readout.ExponentialSum
import FQGNN.Readout.OpH

/-!
# The FQ intensity kernel and the kriging unification

Sources:
* Manuscript, Methods, "Fourier–Quasicrystal operators" paragraph: the intensity
  kernel `K_int(δ) = Σ_k |c_k|² cos(2π s ⟨ξ_k, δ⟩)` "is real, even and positive-semidefinite in the
  sense of Bochner, its nonnegative pure-point diffraction masses making every Gram matrix
  positive-semidefinite (Supplementary Section S5)".
* SI §S5: eqs. (S5.1)–(S5.4), Proposition S5.1
  (positive-definiteness of `K_FQ`, hypotheses (H1)–(H3)), the finite-truncation rank remark of §S5.5 (a),
  and the summary §S5.6.
* SI §S6.2: Proposition S6.2 (kriging = weighted
  min-norm interpolation) and its proof.

Design. The finite-spectrum statement of Prop. S5.1 is proved through one *core* lemma about the
phase-difference Gram matrix `G_{pq} = Σ_k m_k cos(θ_{kp} − θ_{kq})` (`cosGram`), which factors as
`G = Φ diag(m ⊕ m) Φᵀ` with `Φ_{p,(inl k)} = cos θ_{kp}`, `Φ_{p,(inr k)} = sin θ_{kp}` (`phaseMat`) —
exactly the identity `cos(x − y) = cos x cos y + sin x sin y` used in SI §S5.5 (a). The FQ kernel
(`fqKernel`, phases `θ_{kp} = 2π s ⟨ξ_k, u_p⟩`) and the one-dimensional kriging kernel `K_v` of
Prop. S6.2 (`krigKernel`, phases `ω_k t_p`) are both instances.

The spectrum index type `σ` is an arbitrary finite type (the enumerated truncation `Λ'_K` of §S5.4);
points live in `Fin d → ℝ`.
-/

namespace FQGNN.Readout

open Matrix Real FQGNN.Common
open scoped ComplexOrder

/-! ## Core: the phase-difference cosine Gram matrix -/

section Core

variable {ι σ : Type*} [Fintype σ]

/-- Phase-difference cosine Gram matrix `G_{pq} = Σ_k m_k cos(θ_{kp} − θ_{kq})` (SI §S5.3, the Gram
matrix `A_{pq} = K_FQ(u_p, u_q)` written in terms of the phases `θ_{kp}`; SI §S5.5 (a)). -/
noncomputable def cosGram (m : σ → ℝ) (θ : σ → ι → ℝ) : Matrix ι ι ℝ :=
  Matrix.of fun p q => ∑ k, m k * Real.cos (θ k p - θ k q)

/-- The real feature matrix `Φ_{p,(inl k)} = cos θ_{kp}`, `Φ_{p,(inr k)} = sin θ_{kp}` of the
decomposition in SI §S5.5 (a). -/
noncomputable def phaseMat (θ : σ → ι → ℝ) : Matrix ι (σ ⊕ σ) ℝ :=
  Matrix.of fun p i => Sum.elim (fun k => Real.cos (θ k p)) (fun k => Real.sin (θ k p)) i

@[simp] lemma cosGram_apply (m : σ → ℝ) (θ : σ → ι → ℝ) (p q : ι) :
    cosGram m θ p q = ∑ k, m k * Real.cos (θ k p - θ k q) := rfl

/-- The Gram matrix is symmetric (SI Prop. S5.1, "real … even"; `cos` is even). -/
theorem cosGram_transpose (m : σ → ℝ) (θ : σ → ι → ℝ) : (cosGram m θ)ᵀ = cosGram m θ := by
  ext p q
  simp only [transpose_apply, cosGram_apply]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [← Real.cos_neg, neg_sub]

/-- **Factorization** `G = Φ diag(m ⊕ m) Φᵀ`: each frequency contributes the two real rank-one terms
`m_k (cos θ_k)(cos θ_k)ᵀ + m_k (sin θ_k)(sin θ_k)ᵀ` (SI §S5.5 (a), "writing
`cos(θ_p − θ_q) = cos θ_p cos θ_q + sin θ_p sin θ_q` … each frequency contributes two real rank-one
positive-semidefinite terms"). -/
theorem cosGram_eq_factor [DecidableEq σ] (m : σ → ℝ) (θ : σ → ι → ℝ) :
    cosGram m θ = phaseMat θ * diagonal (Sum.elim m m) * (phaseMat θ)ᵀ := by
  ext p q
  simp [cosGram, phaseMat, Matrix.mul_apply, diagonal_apply, Fintype.sum_sum_type, Real.cos_sub,
    ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun k _ => ?_
  ring

variable [Fintype ι]

/-- **Quadratic-form identity.** For real coefficients `a`,
`aᵀ G a = Σ_k m_k [(Σ_p a_p cos θ_{kp})² + (Σ_p a_p sin θ_{kp})²]` (SI §S5.3, eq. (S5.4) in its real
form; SI §S5.5 (a)). -/
theorem dotProduct_cosGram_mulVec (m : σ → ℝ) (θ : σ → ι → ℝ) (a : ι → ℝ) :
    a ⬝ᵥ (cosGram m θ *ᵥ a) =
      ∑ k, m k * ((∑ p, a p * Real.cos (θ k p)) ^ 2 + (∑ p, a p * Real.sin (θ k p)) ^ 2) := by
  classical
  rw [cosGram_eq_factor, ← mulVec_mulVec, ← mulVec_mulVec, dotProduct_mulVec, ← mulVec_transpose]
  simp [dotProduct, diagonal_apply, phaseMat, Fintype.sum_sum_type, mulVec,
    ← Finset.sum_add_distrib, sq]
  refine Finset.sum_congr rfl fun k _ => ?_
  simp only [mul_comm (a _)]
  ring

/-- **Core PSD lemma.** With nonnegative masses `m_k ≥ 0` the phase-difference cosine
Gram matrix is positive semidefinite (SI Prop. S5.1 under (H1); proof of SI §S5.3 / §S5.5 (a)). -/
theorem cosGram_posSemidef {m : σ → ℝ} (hm : ∀ k, 0 ≤ m k) (θ : σ → ι → ℝ) :
    (cosGram m θ).PosSemidef := by
  classical
  rw [cosGram_eq_factor, ← conjTranspose_eq_transpose_of_trivial]
  refine PosSemidef.mul_mul_conjTranspose_same ?_ _
  refine PosSemidef.diagonal fun i => ?_
  cases i <;> simp [hm]

/-- **Finite-truncation rank bound (SI §S5.5 (a)).** `rank G ≤ 2 |σ|`: the Gram matrix is a sum of
at most `2K` rank-one terms (`K = |σ|` kept taps). -/
theorem rank_cosGram_le (m : σ → ℝ) (θ : σ → ι → ℝ) :
    (cosGram m θ).rank ≤ 2 * Fintype.card σ := by
  classical
  rw [cosGram_eq_factor]
  calc (phaseMat θ * diagonal (Sum.elim m m) * (phaseMat θ)ᵀ).rank
      ≤ (phaseMat θ * diagonal (Sum.elim m m)).rank := rank_mul_le_left _ _
    _ ≤ (phaseMat θ).rank := rank_mul_le_left _ _
    _ ≤ Fintype.card (σ ⊕ σ) := rank_le_card_width _
    _ = 2 * Fintype.card σ := by rw [Fintype.card_sum, two_mul]

/-- **Exact singularity with more anchors than taps (SI §S5.5 (a)).** If the number of points exceeds
`2K`, the Gram matrix is singular: "Whenever the number of anchors exceeds the number of taps,
`m > 2K`, the matrix `A` is exactly singular." This is why Prop. S5.1 claims only
positive-*semi*definiteness. -/
theorem det_cosGram_eq_zero [DecidableEq ι] (m : σ → ℝ) (θ : σ → ι → ℝ)
    (h : 2 * Fintype.card σ < Fintype.card ι) : (cosGram m θ).det = 0 := by
  by_contra hne
  have hrank := (rank_eq_card_iff_isUnit_det (cosGram m θ)).2 (Ne.isUnit hne)
  have := rank_cosGram_le m θ
  omega

omit [Fintype ι] in
/-- Complexified factorization `G = Φ diag(m ⊕ m) Φᴴ` over `ℂ` (the entries of `Φ` are real). -/
theorem cosGram_map_ofReal_eq [DecidableEq σ] (m : σ → ℝ) (θ : σ → ι → ℝ) :
    (cosGram m θ).map (fun x : ℝ => (x : ℂ)) =
      (phaseMat θ).map (fun x : ℝ => (x : ℂ)) * diagonal (fun i => ((Sum.elim m m i : ℝ) : ℂ)) *
        ((phaseMat θ).map (fun x : ℝ => (x : ℂ)))ᴴ := by
  ext p q
  simp [cosGram, phaseMat, Matrix.mul_apply, diagonal_apply, Fintype.sum_sum_type, Real.cos_sub,
    ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun k _ => ?_
  ring

/-- **Core PSD lemma, complex coefficients.** With `m ≥ 0` the complexified
Gram matrix is Hermitian positive semidefinite (SI Prop. S5.1: "every Gram matrix `A` … is Hermitian
positive-semidefinite"). -/
theorem cosGram_complex_posSemidef {m : σ → ℝ} (hm : ∀ k, 0 ≤ m k) (θ : σ → ι → ℝ) :
    ((cosGram m θ).map (fun x : ℝ => (x : ℂ))).PosSemidef := by
  classical
  rw [cosGram_map_ofReal_eq]
  refine PosSemidef.mul_mul_conjTranspose_same ?_ _
  refine PosSemidef.diagonal fun i => ?_
  cases i <;> simp [hm]

/-- **Bochner form (SI eq. (S5.3)) of the core lemma.** For every complex coefficient vector `a`,
`Σ_{p,q} a_p conj(a_q) G_{pq} ≥ 0` in `ℂ`'s partial order (i.e. the sum is real and nonnegative). -/
theorem cosGram_complex_nonneg {m : σ → ℝ} (hm : ∀ k, 0 ≤ m k) (θ : σ → ι → ℝ) (a : ι → ℂ) :
    0 ≤ ∑ p, ∑ q, a p * (starRingEnd ℂ) (a q) * ((cosGram m θ p q : ℝ) : ℂ) := by
  have := (cosGram_complex_posSemidef hm θ).dotProduct_mulVec_nonneg (star a)
  simp only [star_star, dotProduct, mulVec, map_apply, Finset.mul_sum, Pi.star_apply] at this
  refine this.trans_eq (Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun q _ => ?_)
  rw [Complex.star_def]
  ring

end Core

/-! ## The FQ intensity kernel (SI Prop. S5.1, finite spectrum) -/

section FQKernel

variable {σ : Type*} [Fintype σ] {d : ℕ}

/-- Profile `κ(δ) = Σ_k m_k cos(2π s ⟨ξ_k, δ⟩)` of the FQ kernel (SI eq. (S5.2) with masses
`m_k = |c_k|²`; manuscript Methods, `K_int(δ)`). Masses `m`, frequencies `ξ_k ∈ ℝᵈ`, scale
`s`. -/
noncomputable def fqProfile (m : σ → ℝ) (ξ : σ → Fin d → ℝ) (s : ℝ) (δ : Fin d → ℝ) : ℝ :=
  ∑ k, m k * Real.cos (2 * π * s * (ξ k ⬝ᵥ δ))

/-- The FQ kernel `K_FQ(u, u') = Σ_k m_k cos(2π s ⟨ξ_k, u − u'⟩)` (SI eq. (S5.2)). -/
noncomputable def fqKernel (m : σ → ℝ) (ξ : σ → Fin d → ℝ) (s : ℝ) (u u' : Fin d → ℝ) : ℝ :=
  ∑ k, m k * Real.cos (2 * π * s * (ξ k ⬝ᵥ (u - u')))

/-- The Gram matrix `A_{pq} = K_FQ(u_p, u_q)` on a finite point family `u : ι → ℝᵈ`
(SI Prop. S5.1, "every Gram matrix `A` with `A_{pq} = K_FQ(u_p, u_q)`"). -/
noncomputable def fqGram {ι : Type*} (m : σ → ℝ) (ξ : σ → Fin d → ℝ) (s : ℝ)
    (u : ι → Fin d → ℝ) : Matrix ι ι ℝ :=
  Matrix.of fun p q => fqKernel m ξ s (u p) (u q)

/-- **Translation invariance** `K_FQ(u, u') = κ(u − u')` (SI Prop. S5.1, first clause). -/
theorem fqKernel_eq_profile (m : σ → ℝ) (ξ : σ → Fin d → ℝ) (s : ℝ) (u u' : Fin d → ℝ) :
    fqKernel m ξ s u u' = fqProfile m ξ s (u - u') := rfl

/-- The FQ Gram matrix is the phase-difference Gram matrix with phases `θ_{kp} = 2π s ⟨ξ_k, u_p⟩`. -/
theorem fqGram_eq_cosGram {ι : Type*} (m : σ → ℝ) (ξ : σ → Fin d → ℝ) (s : ℝ)
    (u : ι → Fin d → ℝ) :
    fqGram m ξ s u = cosGram m (fun k p => 2 * π * s * (ξ k ⬝ᵥ u p)) := by
  ext p q
  simp only [fqGram, fqKernel, of_apply, cosGram_apply, dotProduct_sub, mul_sub]

/-- **SI Proposition S5.1 (finite spectrum), positive-semidefiniteness.** For masses
`m_k ≥ 0` (hypothesis (H1)), any finite family of points `u_p ∈ ℝᵈ`, frequencies `ξ_k ∈ ℝᵈ` and any
scale `s`, the Gram matrix `G_{pq} = Σ_k m_k cos(2π s ⟨ξ_k, u_p − u_q⟩)` is (real symmetric) positive
semidefinite (SI eq. (S5.3) for real coefficients; §S5.4 "for any finite index set … Proposition S5.1
holds verbatim"; manuscript Methods). -/
theorem fqGram_posSemidef {ι : Type*} [Fintype ι] {m : σ → ℝ} (hm : ∀ k, 0 ≤ m k)
    (ξ : σ → Fin d → ℝ) (s : ℝ) (u : ι → Fin d → ℝ) : (fqGram m ξ s u).PosSemidef := by
  rw [fqGram_eq_cosGram]
  exact cosGram_posSemidef hm _

/-- **Explicit sum-of-squares form.** `aᵀ G a = Σ_k m_k [(Σ_p a_p cos θ_{kp})² +
(Σ_p a_p sin θ_{kp})²]` with `θ_{kp} = 2π s ⟨ξ_k, u_p⟩` (SI eq. (S5.4), real form). -/
theorem dotProduct_fqGram_mulVec {ι : Type*} [Fintype ι] (m : σ → ℝ) (ξ : σ → Fin d → ℝ) (s : ℝ)
    (u : ι → Fin d → ℝ) (a : ι → ℝ) :
    a ⬝ᵥ (fqGram m ξ s u *ᵥ a) =
      ∑ k, m k * ((∑ p, a p * Real.cos (2 * π * s * (ξ k ⬝ᵥ u p))) ^ 2 +
        (∑ p, a p * Real.sin (2 * π * s * (ξ k ⬝ᵥ u p))) ^ 2) := by
  rw [fqGram_eq_cosGram, dotProduct_cosGram_mulVec]

/-- The FQ Gram matrix is symmetric (SI Prop. S5.1, "real, even"). -/
theorem fqGram_transpose {ι : Type*} (m : σ → ℝ) (ξ : σ → Fin d → ℝ) (s : ℝ)
    (u : ι → Fin d → ℝ) : (fqGram m ξ s u)ᵀ = fqGram m ξ s u := by
  rw [fqGram_eq_cosGram, cosGram_transpose]

/-- **Complex coefficients — SI eq. (S5.3).** For every `a : ι → ℂ`,
`Σ_{p,q} a_p conj(a_q) K_FQ(u_p, u_q) ≥ 0` (in `ℂ`'s partial order, i.e. it is real and
nonnegative). Equivalently the complexified Gram matrix is Hermitian positive semidefinite. -/
theorem fqGram_complex_nonneg {ι : Type*} [Fintype ι] {m : σ → ℝ} (hm : ∀ k, 0 ≤ m k)
    (ξ : σ → Fin d → ℝ) (s : ℝ) (u : ι → Fin d → ℝ) (a : ι → ℂ) :
    0 ≤ ∑ p, ∑ q, a p * (starRingEnd ℂ) (a q) * (fqKernel m ξ s (u p) (u q) : ℂ) := by
  have h := cosGram_complex_nonneg hm (fun k p => 2 * π * s * (ξ k ⬝ᵥ u p)) a
  rwa [← fqGram_eq_cosGram] at h

/-- **Complex form, matrix version.** The complexified FQ Gram matrix is Hermitian
positive semidefinite (SI Prop. S5.1, "every Gram matrix … is Hermitian positive-semidefinite"). -/
theorem fqGram_complex_posSemidef {ι : Type*} [Fintype ι] {m : σ → ℝ} (hm : ∀ k, 0 ≤ m k)
    (ξ : σ → Fin d → ℝ) (s : ℝ) (u : ι → Fin d → ℝ) :
    ((fqGram m ξ s u).map (fun x : ℝ => (x : ℂ))).PosSemidef := by
  rw [fqGram_eq_cosGram]
  exact cosGram_complex_posSemidef hm _

/-! ### Profile: value at the origin, evenness, continuity -/

/-- `κ(0) = Σ_k m_k`, the total diffracted intensity (SI §S5.3 Remark,
"`κ(0) = K_FQ(u,u) = Σ_k |c_k|² = γ̂_s(ℝᵈ)`"). -/
theorem fqProfile_zero (m : σ → ℝ) (ξ : σ → Fin d → ℝ) (s : ℝ) :
    fqProfile m ξ s 0 = ∑ k, m k := by
  simp [fqProfile]

/-- **Evenness** `κ(−δ) = κ(δ)` (SI Prop. S5.1 "real, even"; SI §S5.3 "real and even in
`δ = u − u'`"). Holds for any real masses. -/
theorem fqProfile_neg (m : σ → ℝ) (ξ : σ → Fin d → ℝ) (s : ℝ) (δ : Fin d → ℝ) :
    fqProfile m ξ s (-δ) = fqProfile m ξ s δ := by
  simp [fqProfile, dotProduct_neg, mul_neg, Real.cos_neg]

/-- **Continuity** of the profile (SI Prop. S5.1 "continuous"; SI §S5.3 "Continuity of
`κ` follows from (H3)" — for a finite spectrum it is a finite sum of continuous functions). -/
theorem continuous_fqProfile (m : σ → ℝ) (ξ : σ → Fin d → ℝ) (s : ℝ) :
    Continuous (fqProfile m ξ s) := by
  unfold fqProfile
  fun_prop

/-- Symmetry of the kernel, `K_FQ(u, u') = K_FQ(u', u)` (SI Prop. S5.1, evenness of `κ`). -/
theorem fqKernel_comm (m : σ → ℝ) (ξ : σ → Fin d → ℝ) (s : ℝ) (u u' : Fin d → ℝ) :
    fqKernel m ξ s u u' = fqKernel m ξ s u' u := by
  rw [fqKernel_eq_profile, fqKernel_eq_profile, ← neg_sub, fqProfile_neg]

/-- With nonnegative masses the profile is maximal at the origin: `|κ(δ)| ≤ κ(0) = Σ_k m_k`
(the normalizing mass of SI §S5.3 Remark; a standard consequence of positive-definiteness). -/
theorem abs_fqProfile_le {m : σ → ℝ} (hm : ∀ k, 0 ≤ m k) (ξ : σ → Fin d → ℝ) (s : ℝ)
    (δ : Fin d → ℝ) : |fqProfile m ξ s δ| ≤ fqProfile m ξ s 0 := by
  rw [fqProfile_zero]
  unfold fqProfile
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun k _ => ?_)
  rw [abs_mul, abs_of_nonneg (hm k)]
  exact mul_le_of_le_one_right (hm k) (Real.abs_cos_le_one _)

/-- The "Re" form of the profile: `κ(δ) = Re Σ_k m_k exp(2π i s ⟨ξ_k, δ⟩)` (SI §S5.3: under (H2)
the "Re" in manuscript eq. (3) is redundant; the identity with `Re` holds unconditionally). -/
theorem fqProfile_eq_re (m : σ → ℝ) (ξ : σ → Fin d → ℝ) (s : ℝ) (δ : Fin d → ℝ) :
    fqProfile m ξ s δ =
      (∑ k, (m k : ℂ) * Complex.exp (((2 * π * s * (ξ k ⬝ᵥ δ) : ℝ) : ℂ) * Complex.I)).re := by
  rw [Complex.re_sum]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [Complex.re_ofReal_mul, Complex.exp_ofReal_mul_I_re]

/-! ### The manuscript's intensity kernel `K_int` -/

/-- Intensity profile `K_int(δ) = Σ_k |c_k|² cos(2π s ⟨ξ_k, δ⟩)` with complex Fourier–Bohr amplitudes
`c_k` (manuscript Methods; SI eq. (S5.1)–(S5.2)). -/
noncomputable def intensityProfile (c : σ → ℂ) (ξ : σ → Fin d → ℝ) (s : ℝ) (δ : Fin d → ℝ) : ℝ :=
  fqProfile (fun k => ‖c k‖ ^ 2) ξ s δ

/-- **Manuscript Methods / SI Prop. S5.1 with (H1) automatic.** For every choice of complex
amplitudes `c_k` (the masses `|c_k|² ≥ 0` need no hypothesis), every scale `s` and every finite point
family, the Gram matrix of `K_int` is positive semidefinite, and `K_int` is even, continuous, with
`K_int(0) = Σ_k |c_k|²`. -/
theorem intensityKernel_props {ι : Type*} [Fintype ι] (c : σ → ℂ) (ξ : σ → Fin d → ℝ) (s : ℝ)
    (u : ι → Fin d → ℝ) :
    (fqGram (fun k => ‖c k‖ ^ 2) ξ s u).PosSemidef ∧
      (∀ δ, intensityProfile c ξ s (-δ) = intensityProfile c ξ s δ) ∧
      Continuous (intensityProfile c ξ s) ∧
      intensityProfile c ξ s 0 = ∑ k, ‖c k‖ ^ 2 :=
  ⟨fqGram_posSemidef (fun _ => sq_nonneg _) ξ s u, fun δ => fqProfile_neg _ ξ s δ,
    continuous_fqProfile _ ξ s, fqProfile_zero _ ξ s⟩

end FQKernel

/-! ## The kriging unification (SI Proposition S6.2) -/

section Kriging

variable {K : ℕ}

/-- The kriging kernel `K_v(t, t') = Σ_k v_k cos(ω_k (t − t'))` on the kept frequencies
(SI Prop. S6.2, display; "§S5, with intensities `v_k` in place of `|c_k|²`"). -/
noncomputable def krigKernel (ω v : Fin K → ℝ) (t t' : ℝ) : ℝ :=
  ∑ k, v k * Real.cos (ω k * (t - t'))

/-- Coefficients of the section `K_v(t, ·)`: `(v_k cos ω_k t, v_k sin ω_k t)` (SI proof of
Prop. S6.2, "the section `K_v(t,·)` … has coefficients `(v_k cos ω_k t, v_k sin ω_k t)`"). -/
noncomputable def krigSection (ω v : Fin K → ℝ) (t : ℝ) : Fin K ⊕ Fin K → ℝ :=
  Sum.elim (fun k => v k * Real.cos (ω k * t)) (fun k => v k * Real.sin (ω k * t))

/-- Coefficients of the `t`-derivative of the section, `∂_t K_v(t, ·)`:
`(−v_k ω_k sin ω_k t, v_k ω_k cos ω_k t)` (SI proof of Prop. S6.2, "differentiating the section in its
argument represents the derivative functionals likewise"). -/
noncomputable def krigSectionDeriv (ω v : Fin K → ℝ) (t : ℝ) : Fin K ⊕ Fin K → ℝ :=
  Sum.elim (fun k => v k * (-(ω k) * Real.sin (ω k * t)))
    (fun k => v k * (ω k * Real.cos (ω k * t)))

/-- The RKHS pairing `⟨f, g⟩_v = Σ_k (a^f_k a^g_k + b^f_k b^g_k) / v_k` on coefficient vectors
(SI proof of Prop. S6.2, "Equip `𝓕` with `⟨f,g⟩_v = …`"). -/
noncomputable def vInner (v : Fin K → ℝ) (c c' : Fin K ⊕ Fin K → ℝ) : ℝ :=
  ∑ k, (c (Sum.inl k) * c' (Sum.inl k) + c (Sum.inr k) * c' (Sum.inr k)) / v k

/-- The section coefficients are `V φ(t)` with `V = diag(v) ⊕ diag(v) = metricW v`. -/
theorem krigSection_eq_mulVec (ω v : Fin K → ℝ) (t : ℝ) :
    krigSection ω v t = metricW v *ᵥ phi ω t := by
  funext i
  cases i <;> simp [krigSection, metricW, mulVec_diagonal]

/-- The derivative-section coefficients are `V φ'(t)`. -/
theorem krigSectionDeriv_eq_mulVec (ω v : Fin K → ℝ) (t : ℝ) :
    krigSectionDeriv ω v t = metricW v *ᵥ dphi ω t := by
  funext i
  cases i <;> simp [krigSectionDeriv, metricW, mulVec_diagonal]

/-- **The section is `K_v(t, ·)` (SI proof of Prop. S6.2).** The comb with the section coefficients
evaluates to the kernel: `r_{K_v(t,·)}(t') = K_v(t, t')` (the identity
`cos ω(t − t') = cos ωt cos ωt' + sin ωt sin ωt'`). -/
theorem rcomb_krigSection (ω v : Fin K → ℝ) (t t' : ℝ) :
    rcomb ω (krigSection ω v t) t' = krigKernel ω v t t' := by
  rw [rcomb_eq_sum, krigKernel, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun k _ => ?_
  simp only [krigSection, Sum.elim_inl, Sum.elim_inr]
  rw [mul_sub, Real.cos_sub]
  ring

/-- Entrywise derivative of the section in its argument: `∂_t K_v(t,·)` has coefficients
`krigSectionDeriv ω v t`. -/
theorem hasDerivAt_krigSection (ω v : Fin K → ℝ) (t : ℝ) (i : Fin K ⊕ Fin K) :
    HasDerivAt (fun x => krigSection ω v x i) (krigSectionDeriv ω v t i) t := by
  have h := (hasDerivAt_phi ω i t).const_mul (Sum.elim v v i)
  cases i with
  | inl k => simpa [krigSection, krigSectionDeriv] using h
  | inr k => simpa [krigSection, krigSectionDeriv] using h

/-- **The derivative section is `∂_t K_v(t, ·)` (SI proof of Prop. S6.2).** For every `t'`,
`t ↦ K_v(t, t')` has derivative `r_{∂K_v(t,·)}(t')` at `t`. -/
theorem hasDerivAt_krigKernel (ω v : Fin K → ℝ) (t t' : ℝ) :
    HasDerivAt (fun x => krigKernel ω v x t') (rcomb ω (krigSectionDeriv ω v t) t') t := by
  have hfun : (fun x => krigKernel ω v x t') =
      fun x => ∑ i, phi ω t' i * krigSection ω v x i := by
    funext x
    rw [← rcomb_krigSection]
    rfl
  rw [hfun]
  exact HasDerivAt.fun_sum fun i _ => (hasDerivAt_krigSection ω v t i).const_mul (phi ω t' i)

/-- **Reproducing property (SI Prop. S6.2 proof).** For `v_k ≠ 0`,
`⟨c, K_v(t, ·)⟩_v = r_c(t) = φ(t) ⬝ᵥ c`: the section reproduces point evaluation. -/
theorem vInner_krigSection {v : Fin K → ℝ} (hv : ∀ k, v k ≠ 0) (ω : Fin K → ℝ)
    (c : Fin K ⊕ Fin K → ℝ) (t : ℝ) : vInner v c (krigSection ω v t) = rcomb ω c t := by
  rw [rcomb_eq_sum, vInner, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun k _ => ?_
  simp only [krigSection, Sum.elim_inl, Sum.elim_inr]
  field_simp [hv k]

/-- **Derivative functionals (SI Prop. S6.2 proof).** For `v_k ≠ 0`,
`⟨c, ∂_t K_v(t, ·)⟩_v = φ'(t) ⬝ᵥ c`, the derivative `r_c'(t)`. -/
theorem vInner_krigSectionDeriv {v : Fin K → ℝ} (hv : ∀ k, v k ≠ 0) (ω : Fin K → ℝ)
    (c : Fin K ⊕ Fin K → ℝ) (t : ℝ) : vInner v c (krigSectionDeriv ω v t) = dphi ω t ⬝ᵥ c := by
  simp only [vInner, dotProduct, Fintype.sum_sum_type, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun k _ => ?_
  simp only [krigSectionDeriv, dphi, Sum.elim_inl, Sum.elim_inr]
  field_simp [hv k]

/-- The derivative functional, as a genuine derivative: `r_c` has derivative
`⟨c, ∂_t K_v(t, ·)⟩_v` at `t` (SI Prop. S6.2 proof, combined with `hasDerivAt_rcomb`). -/
theorem hasDerivAt_rcomb_vInner {v : Fin K → ℝ} (hv : ∀ k, v k ≠ 0) (ω : Fin K → ℝ)
    (c : Fin K ⊕ Fin K → ℝ) (t : ℝ) :
    HasDerivAt (rcomb ω c) (vInner v c (krigSectionDeriv ω v t)) t := by
  rw [vInner_krigSectionDeriv hv]
  exact hasDerivAt_rcomb ω c t

/-- The pairing is symmetric. -/
theorem vInner_comm (v : Fin K → ℝ) (c c' : Fin K ⊕ Fin K → ℝ) : vInner v c c' = vInner v c' c := by
  simp only [vInner, mul_comm (c _)]

/-- **The RKHS norm is the Op-H weighted norm with `w = √v`.** For `v > 0`,
`‖c‖²_v = Σ_k (α_k² + β_k²)/v_k = ‖W⁻¹c‖²` with `W = metricW (√v)` (SI Prop. S6.2, "the operator's
metric enters squared: `‖W⁻¹c‖² = Σ_k (α_k²+β_k²)/w_k²`"). -/
theorem vInner_self_eq_weightedNormSq {v : Fin K → ℝ} (hv : ∀ k, 0 < v k)
    (c : Fin K ⊕ Fin K → ℝ) : vInner v c c = weightedNormSq (fun k => √(v k)) c := by
  rw [weightedNormSq_eq_sum (fun k => Real.sqrt_pos.2 (hv k)), vInner]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [Real.sq_sqrt (hv k).le, sq, sq]

/-- The pairing is positive definite for `v > 0`, so `(𝓕, ⟨·,·⟩_v)` is an inner-product space on the
coefficient vectors (SI Prop. S6.2, "the reproducing-kernel Hilbert space of `K_v` is `𝓕`"). -/
theorem vInner_self_pos {v : Fin K → ℝ} (hv : ∀ k, 0 < v k) {c : Fin K ⊕ Fin K → ℝ}
    (hc : c ≠ 0) : 0 < vInner v c c := by
  rw [vInner_self_eq_weightedNormSq hv]
  have hW := isUnit_det_metricW (fun k => Real.sqrt_pos.2 (hv k))
  refine lt_of_le_of_ne (dotProduct_self_nonneg' _) fun h0 => hc ?_
  have h1 := dotProduct_self_eq_zero.1 h0.symm
  have h2 : metricW (fun k => √(v k)) *ᵥ ((metricW (fun k => √(v k)))⁻¹ *ᵥ c) = 0 := by
    rw [h1, mulVec_zero]
  rwa [mulVec_mulVec, mul_nonsing_inv _ hW, one_mulVec] at h2

/-- `W² = V`: the squared Op-H metric with `w = √v` is `diag(v) ⊕ diag(v)` (SI Prop. S6.2 proof,
"`W² = diag(v) ⊕ diag(v)`"). -/
theorem metricW_sqrt_mul_self {v : Fin K → ℝ} (hv : ∀ k, 0 ≤ v k) :
    metricW (fun k => √(v k)) * metricW (fun k => √(v k)) = metricW v := by
  rw [metricW_mul_metricW]
  congr 1
  funext k
  exact Real.sq_sqrt (hv k)

/-- The Gram matrix of the kriging kernel on a finite node family is positive semidefinite for
`v ≥ 0` (SI Prop. S6.2, "the associated positive-definite kernel (§S5, with intensities `v_k`)"). -/
theorem krigGram_posSemidef {ι : Type*} [Fintype ι] {v : Fin K → ℝ} (hv : ∀ k, 0 ≤ v k)
    (ω : Fin K → ℝ) (t : ι → ℝ) :
    (Matrix.of fun p q => krigKernel ω v (t p) (t q)).PosSemidef := by
  have h : (Matrix.of fun p q => krigKernel ω v (t p) (t q)) =
      cosGram v (fun k p => ω k * t p) := by
    ext p q
    simp only [of_apply, krigKernel, cosGram_apply, mul_sub]
  rw [h]
  exact cosGram_posSemidef hv _

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- The Op-H readout at `η = 0` is its particular part, for every gate value `g`. -/
theorem opHW_eta_zero (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ) (d : ι → ℝ) (g : ℝ) :
    opHW B w d g 0 = partOpHW B w d := by
  rw [opHW_eq_part_add_nullMap, mulVec_zero, smul_zero, add_zero]

/-- **SI Proposition S6.2 (kriging = weighted min-norm interpolation).** Let `v > 0` and
let `B c = d` be any consistent set of value/derivative constraints with full row rank. Then the
Op-H solve at `η = 0` with metric `w = √v` (any gate `g`), `c⋆ = opHW B (√v) d g 0`,
* satisfies the constraints, `B c⋆ = d`;
* minimizes the RKHS norm `‖c‖²_v = ⟨c, c⟩_v` over `{c | B c = d}`;
* is the unique minimizer;
* equals the kriging/representer formula `c⋆ = V Bᵀ (B V Bᵀ)⁻¹ d`, `V = diag(v) ⊕ diag(v)`. -/
theorem kriging_eq_opHW {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B) {v : Fin K → ℝ}
    (hv : ∀ k, 0 < v k) (d : ι → ℝ) (g : ℝ) :
    B *ᵥ opHW B (fun k => √(v k)) d g 0 = d ∧
      (∀ c, B *ᵥ c = d →
        vInner v (opHW B (fun k => √(v k)) d g 0) (opHW B (fun k => √(v k)) d g 0) ≤
          vInner v c c) ∧
      (∀ c, B *ᵥ c = d →
        vInner v c c ≤
          vInner v (opHW B (fun k => √(v k)) d g 0) (opHW B (fun k => √(v k)) d g 0) →
        c = opHW B (fun k => √(v k)) d g 0) ∧
      opHW B (fun k => √(v k)) d g 0 = (metricW v * Bᵀ * (B * metricW v * Bᵀ)⁻¹) *ᵥ d := by
  have hw : ∀ k, 0 < (fun k => √(v k)) k := fun k => Real.sqrt_pos.2 (hv k)
  obtain ⟨hfeas, hmin, -, hform⟩ := partOpHW_isWeightedMinNorm hB hw d
  rw [opHW_eta_zero]
  refine ⟨hfeas, fun c hc => ?_, fun c hc hle => ?_, ?_⟩
  · rw [vInner_self_eq_weightedNormSq hv, vInner_self_eq_weightedNormSq hv]
    exact hmin c hc
  · rw [vInner_self_eq_weightedNormSq hv, vInner_self_eq_weightedNormSq hv] at hle
    exact eq_partOpHW_of_weightedNormSq_le hB hw hc hle
  · rw [hform, metricW_sqrt_mul_self fun k => (hv k).le]

/-- **Representer form of SI Prop. S6.2.** The interpolant is a combination of the kernel-weighted
constraint rows: `c⋆ = Σ_i λ_i V (B i)` with `λ = (B V Bᵀ)⁻¹ d` (the representer / partial-spline
mechanism cited in the SI proof). -/
theorem opHW_sqrt_eq_sum_rows {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B)
    {v : Fin K → ℝ} (hv : ∀ k, 0 < v k) (d : ι → ℝ) (g : ℝ) :
    opHW B (fun k => √(v k)) d g 0 =
      ∑ i, ((B * metricW v * Bᵀ)⁻¹ *ᵥ d) i • (metricW v *ᵥ B i) := by
  rw [(kriging_eq_opHW hB hv d g).2.2.2, ← mulVec_mulVec, ← mulVec_mulVec, mulVec_transpose,
    vecMul_eq_sum, mulVec_sum]
  simp only [mulVec_smul]

/-- **Kriging predictor (SI Prop. S6.2, value constraints).** If every constraint row is a value row
`B i = φ(τ_i)`, the minimal-`‖·‖_v` interpolant is the kriging predictor
`r_{c⋆}(t) = Σ_i λ_i K_v(τ_i, t)`, `λ = (B V Bᵀ)⁻¹ d`: interpolation with the kernel weights `v_k` is
the Op-H solve with `w = √v`. -/
theorem rcomb_opHW_sqrt_eq_kriging {ω : Fin K → ℝ} {τ : ι → ℝ} {B : Matrix ι (Fin K ⊕ Fin K) ℝ}
    (hrows : ∀ i, B i = phi ω (τ i)) (hB : FullRowRank B) {v : Fin K → ℝ} (hv : ∀ k, 0 < v k)
    (d : ι → ℝ) (g : ℝ) (t : ℝ) :
    rcomb ω (opHW B (fun k => √(v k)) d g 0) t =
      ∑ i, ((B * metricW v * Bᵀ)⁻¹ *ᵥ d) i * krigKernel ω v (τ i) t := by
  rw [opHW_sqrt_eq_sum_rows hB hv d g]
  simp only [rcomb, dotProduct_sum, dotProduct_smul, smul_eq_mul]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [hrows i, ← krigSection_eq_mulVec, ← rcomb_krigSection]
  rfl

end Kriging

end FQGNN.Readout

import Mathlib

/-!
# The reconstruction class: affine atom ⊕ FQ exponential-sum comb

Sources:
* Manuscript (Results, eqs. (3)–(5)): the Op-D twisted-Eberlein kernel (3), its
  derivative measure `K^{Dμ}` (4), and the reconstruction class `p̂(t) = A(t) + r(τ)` (5).
* SI §S4.1 (eqs. (S4.1)–(S4.4)): the residual comb, the chord `A`, its
  slope `A'`, the direct-sum claim `Aff₁ ⊕ span{cos ω_k τ, sin ω_k τ}`, and "a residual-class function
  vanishes at both endpoints, so it cannot be affine unless it is `0`".
* SI §S4.6 (Prop. S4.3 and its proof, eq. (S4.10)): the time-linear interpolation of the bracketing
  anchors is the chord; interpolation fractions are invariant under affine time re-parameterization and
  linear interpolation commutes with affine maps.

Shared vocabulary: kept frequencies `ω : Fin K → ℝ`; feature map `phi ω τ` on the index type
`Fin K ⊕ Fin K` (cos part `Sum.inl k`, sin part `Sum.inr k`); derivative map `dphi ω τ`; residual comb
`rcomb ω c τ = phi ω τ ⬝ᵥ c`; chord `chord a b pa pb t`; slope `chordSlope a b pa pb`; reconstruction
`recon ω c a b pa pb t = chord a b pa pb t + rcomb ω c (t − (a+b)/2)`.
-/

namespace FQGNN.Readout

open Real

variable {K : ℕ}

/-! ## Definitions -/

/-- Feature map `φ(τ)` of the residual comb (manuscript eq. (5), SI eq. (S4.1)):
`φ(τ)(inl k) = cos(ω_k τ)`, `φ(τ)(inr k) = sin(ω_k τ)`. -/
noncomputable def phi (ω : Fin K → ℝ) (τ : ℝ) : Fin K ⊕ Fin K → ℝ :=
  Sum.elim (fun k => Real.cos (ω k * τ)) (fun k => Real.sin (ω k * τ))

/-- Derivative feature map `φ'(τ)` (SI eq. (S4.6), the Hermite derivative rows):
`φ'(τ)(inl k) = −ω_k sin(ω_k τ)`, `φ'(τ)(inr k) = ω_k cos(ω_k τ)`. -/
noncomputable def dphi (ω : Fin K → ℝ) (τ : ℝ) : Fin K ⊕ Fin K → ℝ :=
  Sum.elim (fun k => -(ω k) * Real.sin (ω k * τ)) (fun k => ω k * Real.cos (ω k * τ))

/-- Residual comb `r_c(τ) = Σ_k [α_k cos(ω_k τ) + β_k sin(ω_k τ)] = φ(τ) ⬝ᵥ c`, `c = (α; β)`
(manuscript eq. (5), SI eq. (S4.1)). -/
noncomputable def rcomb (ω : Fin K → ℝ) (c : Fin K ⊕ Fin K → ℝ) (τ : ℝ) : ℝ :=
  phi ω τ ⬝ᵥ c

/-- Affine chord through the bracketing anchors, `A(t) = p_a + (t − a)(p_b − p_a)/(b − a)`
(SI eq. (S4.3)). -/
noncomputable def chord (a b pa pb t : ℝ) : ℝ :=
  pa + (t - a) * ((pb - pa) / (b - a))

/-- Chord slope `A' = (p_b − p_a)/(b − a)` (SI eq. (S4.3)). -/
noncomputable def chordSlope (a b pa pb : ℝ) : ℝ :=
  (pb - pa) / (b - a)

/-- Midpoint-centred time `τ = t − (a+b)/2` (manuscript eq. (5), SI eq. (S4.1)). -/
noncomputable def ctime (a b t : ℝ) : ℝ :=
  t - (a + b) / 2

/-- Reconstruction `p̂(t) = A(t) + r_c(τ)`, `τ = t − (a+b)/2` (manuscript eq. (5),
SI eq. (S4.1)). -/
noncomputable def recon (ω : Fin K → ℝ) (c : Fin K ⊕ Fin K → ℝ) (a b pa pb t : ℝ) : ℝ :=
  chord a b pa pb t + rcomb ω c (t - (a + b) / 2)

@[simp] lemma phi_inl (ω : Fin K → ℝ) (τ : ℝ) (k : Fin K) :
    phi ω τ (Sum.inl k) = Real.cos (ω k * τ) := rfl

@[simp] lemma phi_inr (ω : Fin K → ℝ) (τ : ℝ) (k : Fin K) :
    phi ω τ (Sum.inr k) = Real.sin (ω k * τ) := rfl

@[simp] lemma dphi_inl (ω : Fin K → ℝ) (τ : ℝ) (k : Fin K) :
    dphi ω τ (Sum.inl k) = -(ω k) * Real.sin (ω k * τ) := rfl

@[simp] lemma dphi_inr (ω : Fin K → ℝ) (τ : ℝ) (k : Fin K) :
    dphi ω τ (Sum.inr k) = ω k * Real.cos (ω k * τ) := rfl

/-- The reconstruction in terms of the centred time (definitional unfolding of `recon`). -/
lemma recon_eq (ω : Fin K → ℝ) (c : Fin K ⊕ Fin K → ℝ) (a b pa pb t : ℝ) :
    recon ω c a b pa pb t = chord a b pa pb t + rcomb ω c (ctime a b t) := rfl

/-- Explicit cos/sin form of the comb, `r_c(τ) = Σ_k α_k cos(ω_k τ) + Σ_k β_k sin(ω_k τ)`
(manuscript eq. (5), SI eq. (S4.1)). -/
lemma rcomb_eq_sum (ω : Fin K → ℝ) (c : Fin K ⊕ Fin K → ℝ) (τ : ℝ) :
    rcomb ω c τ = ∑ k, Real.cos (ω k * τ) * c (Sum.inl k)
      + ∑ k, Real.sin (ω k * τ) * c (Sum.inr k) := by
  simp [rcomb, dotProduct, Fintype.sum_sum_type]

/-- The comb is linear in its coefficients: additivity. -/
lemma rcomb_add (ω : Fin K → ℝ) (c c' : Fin K ⊕ Fin K → ℝ) (τ : ℝ) :
    rcomb ω (c + c') τ = rcomb ω c τ + rcomb ω c' τ := by
  simp [rcomb, dotProduct_add]

/-- The comb is linear in its coefficients: homogeneity. -/
lemma rcomb_smul (ω : Fin K → ℝ) (a : ℝ) (c : Fin K ⊕ Fin K → ℝ) (τ : ℝ) :
    rcomb ω (a • c) τ = a * rcomb ω c τ := by
  simp [rcomb, dotProduct_smul]

/-- The zero coefficient vector gives the zero comb. -/
@[simp] lemma rcomb_zero (ω : Fin K → ℝ) (τ : ℝ) : rcomb ω 0 τ = 0 := by
  simp [rcomb]

/-! ## The comb is differentiable with derivative `φ'(τ) ⬝ᵥ c` -/

/-- Entrywise derivative of the feature map: `d/dτ φ(τ)_i = φ'(τ)_i`. -/
theorem hasDerivAt_phi (ω : Fin K → ℝ) (i : Fin K ⊕ Fin K) (τ : ℝ) :
    HasDerivAt (fun x => phi ω x i) (dphi ω τ i) τ := by
  cases i with
  | inl k =>
    have h := ((hasDerivAt_id τ).const_mul (ω k)).cos
    simp only [id, mul_one] at h
    simpa [mul_comm] using h
  | inr k =>
    have h := ((hasDerivAt_id τ).const_mul (ω k)).sin
    simp only [id, mul_one] at h
    simpa [mul_comm] using h

/-- The residual comb `r_c(τ) = φ(τ) ⬝ᵥ c` has derivative `φ'(τ) ⬝ᵥ c` at every `τ`
(manuscript eq. (5); SI §S4.2, eq. (S4.5)–(S4.6): the derivative rows `r'(a) = φ'(τ_a) ⬝ᵥ c`). -/
theorem hasDerivAt_rcomb (ω : Fin K → ℝ) (c : Fin K ⊕ Fin K → ℝ) (τ : ℝ) :
    HasDerivAt (rcomb ω c) (dphi ω τ ⬝ᵥ c) τ := by
  have h : HasDerivAt (fun x => ∑ i, phi ω x i * c i) (∑ i, dphi ω τ i * c i) τ :=
    HasDerivAt.fun_sum (fun i _ => (hasDerivAt_phi ω i τ).mul_const (c i))
  exact h

/-- `deriv` form of `hasDerivAt_rcomb`. -/
theorem deriv_rcomb (ω : Fin K → ℝ) (c : Fin K ⊕ Fin K → ℝ) (τ : ℝ) :
    deriv (rcomb ω c) τ = dphi ω τ ⬝ᵥ c :=
  (hasDerivAt_rcomb ω c τ).deriv

/-! ## Chord facts and the S4.6 identity -/

/-- The chord passes through the left anchor: `A(a) = p_a` (SI eq. (S4.3)). -/
@[simp] theorem chord_left (a b pa pb : ℝ) : chord a b pa pb a = pa := by
  simp [chord]

/-- The chord passes through the right anchor: `A(b) = p_b` for `a ≠ b`
(SI eq. (S4.3), gap `a < b`). -/
theorem chord_right {a b : ℝ} (hab : a ≠ b) (pa pb : ℝ) : chord a b pa pb b = pb := by
  have h : b - a ≠ 0 := sub_ne_zero.mpr (Ne.symm hab)
  unfold chord
  field_simp
  ring

/-- The chord has constant derivative `A' = (p_b − p_a)/(b − a)` (SI eq. (S4.3)). -/
theorem hasDerivAt_chord (a b pa pb t : ℝ) :
    HasDerivAt (chord a b pa pb) (chordSlope a b pa pb) t := by
  unfold chordSlope
  have h1 : HasDerivAt (fun x => x - a) 1 t := (hasDerivAt_id t).sub_const a
  have h2 := (h1.mul_const ((pb - pa) / (b - a))).const_add pa
  rw [one_mul] at h2
  exact h2

/-- The chord in slope form, `A(t) = p_a + (t − a) A'`. -/
lemma chord_eq_slope (a b pa pb t : ℝ) :
    chord a b pa pb t = pa + (t - a) * chordSlope a b pa pb := rfl

/-- **SI §S4.6 identity (Prop. S4.3).** The time-linear interpolation of the bracketing
anchors, `((b − t) p_a + (t − a) p_b)/(b − a)`, is exactly the affine atom `A(t)` (manuscript: "the
scored geodesic reference and the affine atom are the same chord in the evaluation chart"). -/
theorem timeLinearInterp_eq_chord {a b : ℝ} (hab : a ≠ b) (pa pb t : ℝ) :
    ((b - t) * pa + (t - a) * pb) / (b - a) = chord a b pa pb t := by
  have h : b - a ≠ 0 := sub_ne_zero.mpr (Ne.symm hab)
  unfold chord
  field_simp
  ring

/-- **SI eq. (S4.10) form of the S4.6 identity.** With the matched time fraction
`w = (t − t_L)/(t_R − t_L)`, the per-coordinate linear interpolation `(1 − w) p_L + w p_R` is the
chord `A(t)`. -/
theorem lerp_fraction_eq_chord {a b : ℝ} (hab : a ≠ b) (pa pb t : ℝ) :
    (1 - (t - a) / (b - a)) * pa + ((t - a) / (b - a)) * pb = chord a b pa pb t := by
  have h : b - a ≠ 0 := sub_ne_zero.mpr (Ne.symm hab)
  unfold chord
  field_simp
  ring

/-- **SI Prop. S4.3 proof, step 1.** An interpolation fraction is invariant under any affine time
re-parameterization `t ↦ α t + β`, `α ≠ 0` (e.g. the run normalization `t̂ = (t − t₀)/T`). -/
theorem fraction_affine_time {a b : ℝ} (hab : a ≠ b) {α : ℝ} (hα : α ≠ 0) (β t : ℝ) :
    ((α * t + β) - (α * a + β)) / ((α * b + β) - (α * a + β)) = (t - a) / (b - a) := by
  have h : b - a ≠ 0 := sub_ne_zero.mpr (Ne.symm hab)
  rw [show (α * t + β) - (α * a + β) = α * (t - a) by ring,
    show (α * b + β) - (α * a + β) = α * (b - a) by ring, mul_div_mul_left _ _ hα]

/-- **SI Prop. S4.3 proof, step 1 (chord form).** The chord is invariant under an affine time
re-parameterization applied to the anchor times and the query time alike. -/
theorem chord_affine_time {a b : ℝ} (hab : a ≠ b) {α : ℝ} (hα : α ≠ 0) (β pa pb t : ℝ) :
    chord (α * a + β) (α * b + β) pa pb (α * t + β) = chord a b pa pb t := by
  have h : b - a ≠ 0 := sub_ne_zero.mpr (Ne.symm hab)
  unfold chord
  rw [show α * t + β - (α * a + β) = α * (t - a) by ring,
    show α * b + β - (α * a + β) = α * (b - a) by ring]
  field_simp

/-- **SI Prop. S4.3 proof, step 2.** Linear interpolation commutes with every affine map of the
positions, `x ↦ m x + q` (e.g. the chart scaling `Φ ↦ Φ/ς` and back). -/
theorem chord_affine_map (a b m q pa pb t : ℝ) :
    chord a b (m * pa + q) (m * pb + q) t = m * chord a b pa pb t + q := by
  unfold chord
  by_cases h : b - a = 0
  · simp [h]
  · field_simp
    ring

/-- **Affine data (SI Prop. 1 / S4.5 setting).** If the data are affine,
`p(t) = u₀ + u₁ t`, then the chord through `(a, p a)`, `(b, p b)` is `p` itself. -/
theorem chord_of_affine {a b : ℝ} (hab : a ≠ b) (u₀ u₁ t : ℝ) :
    chord a b (u₀ + u₁ * a) (u₀ + u₁ * b) t = u₀ + u₁ * t := by
  have h : b - a ≠ 0 := sub_ne_zero.mpr (Ne.symm hab)
  unfold chord
  field_simp
  ring

/-- **Affine data.** On affine data `p(t) = u₀ + u₁ t` the chord slope is `A' = u₁`. -/
theorem chordSlope_of_affine {a b : ℝ} (hab : a ≠ b) (u₀ u₁ : ℝ) :
    chordSlope a b (u₀ + u₁ * a) (u₀ + u₁ * b) = u₁ := by
  have h : b - a ≠ 0 := sub_ne_zero.mpr (Ne.symm hab)
  unfold chordSlope
  field_simp
  ring

/-- **Affine data, same-side finite differences** (SI §S4.1, Boundary-velocity source: "a
same-side finite-difference secant on the interpolated track"). On affine data every secant
`(p t_i − p t_j)/(t_i − t_j)`, `t_i ≠ t_j`, equals the slope `u₁`; hence every finite-difference
boundary velocity equals the chord slope `A'`. -/
theorem finiteDiff_of_affine (u₀ u₁ : ℝ) {ti tj : ℝ} (hij : ti ≠ tj) :
    ((u₀ + u₁ * ti) - (u₀ + u₁ * tj)) / (ti - tj) = u₁ := by
  have h : ti - tj ≠ 0 := sub_ne_zero.mpr hij
  field_simp
  ring

/-- **Affine data (velocity mismatch).** On affine data every same-side finite-difference
velocity matches the chord slope: `v − A' = 0` (the datum of a Hermite derivative row vanishes). -/
theorem finiteDiff_sub_chordSlope_of_affine {a b : ℝ} (hab : a ≠ b) (u₀ u₁ : ℝ) {ti tj : ℝ}
    (hij : ti ≠ tj) :
    ((u₀ + u₁ * ti) - (u₀ + u₁ * tj)) / (ti - tj)
      - chordSlope a b (u₀ + u₁ * a) (u₀ + u₁ * b) = 0 := by
  rw [finiteDiff_of_affine u₀ u₁ hij, chordSlope_of_affine hab, sub_self]

/-- **Affine data (function form).** If `p` is affine, `p t = u₀ + u₁ t` for all `t`, then
`chord a b (p a) (p b) = p` and `chordSlope a b (p a) (p b) = u₁`. -/
theorem chord_eq_of_affine {p : ℝ → ℝ} {u₀ u₁ : ℝ} (hp : ∀ t, p t = u₀ + u₁ * t) {a b : ℝ}
    (hab : a ≠ b) :
    (∀ t, chord a b (p a) (p b) t = p t) ∧ chordSlope a b (p a) (p b) = u₁ := by
  refine ⟨fun t => ?_, ?_⟩
  · rw [hp a, hp b, hp t, chord_of_affine hab]
  · rw [hp a, hp b, chordSlope_of_affine hab]

/-- Derivative of the reconstruction: `p̂'(t) = A' + φ'(τ) ⬝ᵥ c`, `τ = t − (a+b)/2`
(manuscript eq. (5), SI eqs. (S4.1), (S4.5)). -/
theorem hasDerivAt_recon (ω : Fin K → ℝ) (c : Fin K ⊕ Fin K → ℝ) (a b pa pb t : ℝ) :
    HasDerivAt (recon ω c a b pa pb)
      (chordSlope a b pa pb + dphi ω (t - (a + b) / 2) ⬝ᵥ c) t := by
  have h1 : HasDerivAt (fun x => x - (a + b) / 2) 1 t :=
    (hasDerivAt_id t).sub_const ((a + b) / 2)
  have h2 := (hasDerivAt_rcomb ω c (t - (a + b) / 2)).comp t h1
  rw [mul_one] at h2
  exact (hasDerivAt_chord a b pa pb t).add h2

/-- With zero coefficients the reconstruction is the chord (SI Remark S4.2: the chord is the
zero-residual member of the reconstruction class). -/
@[simp] theorem recon_zero (ω : Fin K → ℝ) (a b pa pb t : ℝ) :
    recon ω 0 a b pa pb t = chord a b pa pb t := by
  simp [recon]

/-! ## The Dμ kernel is the derivative of the Op-D kernel -/

/-- One-variable Op-D twisted-Eberlein kernel (manuscript eq. (3)):
`K(δ) = Re Σ_k c_k ĝ_k exp(2π i s ξ_k δ)` with complex amplitudes `c_k`, complex taps `ĝ_k = ĝ(ξ_k)`,
real frequencies `ξ_k` and scale `s`. -/
noncomputable def kernelOpD (c g : Fin K → ℂ) (ξ : Fin K → ℝ) (s δ : ℝ) : ℝ :=
  (∑ k, c k * g k * Complex.exp (2 * π * Complex.I * s * ξ k * δ)).re

/-- One-variable Dμ kernel (manuscript eq. (4)):
`K^{Dμ}(δ) = Re Σ_k (2π i s ξ_k) c_k ĝ_k exp(2π i s ξ_k δ)`. -/
noncomputable def kernelDmu (c g : Fin K → ℂ) (ξ : Fin K → ℝ) (s δ : ℝ) : ℝ :=
  (∑ k, (2 * π * Complex.I * s * ξ k) * c k * g k *
    Complex.exp (2 * π * Complex.I * s * ξ k * δ)).re

/-- **Manuscript eqs. (3)–(4).** The Dμ kernel is the derivative of the Op-D kernel:
`HasDerivAt K (K^{Dμ} δ) δ` for every `δ` ("the derivative measure of the kernel in equation (3): the
same aperiodic comb with each amplitude scaled by its own frequency `2π i s ξ_k`"). -/
theorem hasDerivAt_kernelOpD (c g : Fin K → ℂ) (ξ : Fin K → ℝ) (s δ : ℝ) :
    HasDerivAt (kernelOpD c g ξ s) (kernelDmu c g ξ s δ) δ := by
  have hk : ∀ k, HasDerivAt
      (fun x : ℝ => c k * g k * Complex.exp (2 * π * Complex.I * s * ξ k * x))
      ((2 * π * Complex.I * s * ξ k) * c k * g k *
        Complex.exp (2 * π * Complex.I * s * ξ k * δ)) δ := by
    intro k
    have h1 : HasDerivAt (fun x : ℝ => ((x : ℝ) : ℂ)) ((1 : ℝ) : ℂ) δ :=
      (hasDerivAt_id δ).ofReal_comp
    have h2 := ((h1.const_mul (2 * π * Complex.I * s * ξ k)).cexp).const_mul (c k * g k)
    convert h2 using 1
    push_cast
    ring
  have hsum := HasDerivAt.fun_sum (u := Finset.univ) (fun k _ => hk k)
  have hre := Complex.reCLM.hasFDerivAt.comp_hasDerivAt δ hsum
  exact hre

/-- `deriv` form: `d/dδ K(δ) = K^{Dμ}(δ)` (manuscript eqs. (3)–(4)). -/
theorem deriv_kernelOpD (c g : Fin K → ℂ) (ξ : Fin K → ℝ) (s δ : ℝ) :
    deriv (kernelOpD c g ξ s) δ = kernelDmu c g ξ s δ :=
  (hasDerivAt_kernelOpD c g ξ s δ).deriv

/-- Op-D kernel on `ℝᵈ` (manuscript eq. (3)): `K(δ) = Re Σ_k c_k ĝ_k exp(2π i s ⟨ξ_k, δ⟩)`,
`ξ_k, δ ∈ ℝᵈ`. -/
noncomputable def kernelOpDVec {d : ℕ} (c g : Fin K → ℂ) (ξ : Fin K → Fin d → ℝ) (s : ℝ)
    (δ : Fin d → ℝ) : ℝ :=
  (∑ k, c k * g k * Complex.exp (2 * π * Complex.I * s * ((ξ k ⬝ᵥ δ : ℝ) : ℂ))).re

/-- Dμ kernel on `ℝᵈ` along a direction `e` (manuscript eq. (4) paired with `e`):
`K^{Dμ}_e(δ) = Re Σ_k (2π i s ⟨ξ_k, e⟩) c_k ĝ_k exp(2π i s ⟨ξ_k, δ⟩)`. -/
noncomputable def kernelDmuVec {d : ℕ} (c g : Fin K → ℂ) (ξ : Fin K → Fin d → ℝ) (s : ℝ)
    (e δ : Fin d → ℝ) : ℝ :=
  (∑ k, (2 * π * Complex.I * s * ((ξ k ⬝ᵥ e : ℝ) : ℂ)) * c k * g k *
    Complex.exp (2 * π * Complex.I * s * ((ξ k ⬝ᵥ δ : ℝ) : ℂ))).re

/-- **`d`-dimensional (directional) form; manuscript eqs. (3)–(4).** For every displacement
`δ ∈ ℝᵈ` and direction `e ∈ ℝᵈ`, the directional derivative of the Op-D kernel along `e` is the Dμ kernel
with each amplitude scaled by `2π i s ⟨ξ_k, e⟩`: `d/dh K(δ + h e) = K^{Dμ}_e(δ + h e)`. -/
theorem hasDerivAt_kernelOpDVec {d : ℕ} (c g : Fin K → ℂ) (ξ : Fin K → Fin d → ℝ) (s : ℝ)
    (e δ : Fin d → ℝ) (h : ℝ) :
    HasDerivAt (fun x : ℝ => kernelOpDVec c g ξ s (δ + x • e))
      (kernelDmuVec c g ξ s e (δ + h • e)) h := by
  have hk : ∀ k, HasDerivAt
      (fun x : ℝ => c k * g k *
        Complex.exp (2 * π * Complex.I * s * ((ξ k ⬝ᵥ (δ + x • e) : ℝ) : ℂ)))
      ((2 * π * Complex.I * s * ((ξ k ⬝ᵥ e : ℝ) : ℂ)) * c k * g k *
        Complex.exp (2 * π * Complex.I * s * ((ξ k ⬝ᵥ (δ + h • e) : ℝ) : ℂ))) h := by
    intro k
    have hlin : HasDerivAt (fun x : ℝ => ξ k ⬝ᵥ (δ + x • e)) (ξ k ⬝ᵥ e) h := by
      have h0 := ((hasDerivAt_id h).mul_const (ξ k ⬝ᵥ e)).const_add (ξ k ⬝ᵥ δ)
      rw [one_mul] at h0
      convert h0 using 1
      funext x
      simp [dotProduct_add, dotProduct_smul]
    have h2 := ((hlin.ofReal_comp.const_mul (2 * π * Complex.I * s)).cexp).const_mul (c k * g k)
    convert h2 using 1
    ring
  have hsum := HasDerivAt.fun_sum (u := Finset.univ) (fun k _ => hk k)
  have hre := Complex.reCLM.hasFDerivAt.comp_hasDerivAt h hsum
  exact hre

/-! ## An affine function with two zeros vanishes: the constraints protect the affine atom -/

/-- **Core lemma.** An affine function `u₀ + u₁ τ` vanishing at two distinct points is zero:
`u₀ = 0` and `u₁ = 0`. -/
theorem affine_eq_zero_of_two_zeros {u₀ u₁ τa τb : ℝ} (hab : τa ≠ τb)
    (ha : u₀ + u₁ * τa = 0) (hb : u₀ + u₁ * τb = 0) : u₀ = 0 ∧ u₁ = 0 := by
  have h1 : u₁ * (τa - τb) = 0 := by linear_combination ha - hb
  have hu₁ : u₁ = 0 := (mul_eq_zero.mp h1).resolve_right (sub_ne_zero.mpr hab)
  refine ⟨?_, hu₁⟩
  simpa [hu₁] using ha

/-- **Interval form, any function.** A function `g` that is affine on the interval between
two distinct points and vanishes at both of them vanishes on that whole interval. -/
theorem eqOn_zero_of_affineOn_of_two_zeros {g : ℝ → ℝ} {τa τb u₀ u₁ : ℝ} (hab : τa ≠ τb)
    (ha : g τa = 0) (hb : g τb = 0) (haff : ∀ τ ∈ Set.uIcc τa τb, g τ = u₀ + u₁ * τ) :
    ∀ τ ∈ Set.uIcc τa τb, g τ = 0 := by
  have h := affine_eq_zero_of_two_zeros hab
    ((haff τa Set.left_mem_uIcc).symm.trans ha) ((haff τb Set.right_mem_uIcc).symm.trans hb)
  intro τ hτ
  rw [haff τ hτ, h.1, h.2]
  ring

/-- **SI §S4.1 "Honest scope of the mask".** A residual-class function (an element of
`span{cos ω_k τ, sin ω_k τ}`) that vanishes at two distinct points and is affine is identically zero
("a residual-class function vanishes at both endpoints, so it cannot be affine unless it is `0`"). -/
theorem rcomb_eq_zero_of_affine (ω : Fin K → ℝ) (c : Fin K ⊕ Fin K → ℝ) {τa τb u₀ u₁ : ℝ}
    (hab : τa ≠ τb) (ha : rcomb ω c τa = 0) (hb : rcomb ω c τb = 0)
    (haff : ∀ τ, rcomb ω c τ = u₀ + u₁ * τ) : ∀ τ, rcomb ω c τ = 0 := by
  have h := affine_eq_zero_of_two_zeros hab ((haff τa).symm.trans ha) ((haff τb).symm.trans hb)
  intro τ
  rw [haff τ, h.1, h.2]
  ring

/-- **Interval form.** A residual-class function that vanishes at two distinct points and is
affine on the interval between them vanishes on that whole interval. -/
theorem rcomb_eqOn_zero_of_affineOn (ω : Fin K → ℝ) (c : Fin K ⊕ Fin K → ℝ) {τa τb u₀ u₁ : ℝ}
    (hab : τa ≠ τb) (ha : rcomb ω c τa = 0) (hb : rcomb ω c τb = 0)
    (haff : ∀ τ ∈ Set.uIcc τa τb, rcomb ω c τ = u₀ + u₁ * τ) :
    ∀ τ ∈ Set.uIcc τa τb, rcomb ω c τ = 0 :=
  eqOn_zero_of_affineOn_of_two_zeros hab ha hb haff

/-- **Gap form (SI §S4.1 / Remark S4.2: "the constraints are what protect the affine
atom").** On a gap `a ≠ b`, if the residual comb satisfies the value rows `r(τ_a) = r(τ_b) = 0`
(`τ_a = a − (a+b)/2`, `τ_b = b − (a+b)/2`) and the reconstruction `p̂ = A + r` is affine on the gap,
then `p̂` is the chord `A` on the whole gap. -/
theorem recon_eq_chord_of_affineOn (ω : Fin K → ℝ) (c : Fin K ⊕ Fin K → ℝ) {a b : ℝ} (hab : a ≠ b)
    (pa pb : ℝ) (ha : rcomb ω c (a - (a + b) / 2) = 0) (hb : rcomb ω c (b - (a + b) / 2) = 0)
    {v₀ v₁ : ℝ} (haff : ∀ t ∈ Set.uIcc a b, recon ω c a b pa pb t = v₀ + v₁ * t) :
    ∀ t ∈ Set.uIcc a b, recon ω c a b pa pb t = chord a b pa pb t := by
  have hg : ∀ t ∈ Set.uIcc a b, rcomb ω c (t - (a + b) / 2) =
      (v₀ - pa + a * chordSlope a b pa pb) + (v₁ - chordSlope a b pa pb) * t := by
    intro t ht
    have h := haff t ht
    rw [recon, chord_eq_slope] at h
    linear_combination h
  have hz := eqOn_zero_of_affineOn_of_two_zeros (g := fun t => rcomb ω c (t - (a + b) / 2))
    hab ha hb hg
  intro t ht
  have h := hz t ht
  rw [recon, h, add_zero]

/-! ## The direct sum `Aff₁ ⊕ span{cos ω_k τ, sin ω_k τ}` on a nondegenerate interval

SI §S4.1: "the family `{1, τ} ∪ {cos(ω_k τ), sin(ω_k τ)}` with distinct `ω_k > 0` is linearly independent
on any nondegenerate interval". Proof used here (differentiation + a Lagrange-type product, instead of
the SI's real-analyticity/characters argument): if `u₀ + u₁τ + r_c(τ) ≡ 0` on an open interval, two
derivatives kill the affine part; the operator `−d²/dτ² − ω_j²` maps the comb `r_c` to the comb with
coefficients `(ω_{(i)}² − ω_j²) c_i`; the product over `j ≠ k` isolates mode `k`, and a single mode
`α cos(ωτ) + β sin(ωτ)` vanishing on an open interval has `α = β = 0`. -/

/-- The frequency attached to each coefficient index: `freq ω (inl k) = freq ω (inr k) = ω_k`. -/
def freq (ω : Fin K → ℝ) : Fin K ⊕ Fin K → ℝ :=
  Sum.elim ω ω

/-- Coefficient map of differentiation: `φ'(τ) ⬝ᵥ c = φ(τ) ⬝ᵥ (drot ω c)`, i.e.
`(α_k, β_k) ↦ (ω_k β_k, −ω_k α_k)`. -/
def drot (ω : Fin K → ℝ) (c : Fin K ⊕ Fin K → ℝ) : Fin K ⊕ Fin K → ℝ :=
  Sum.elim (fun k => ω k * c (Sum.inr k)) (fun k => -(ω k) * c (Sum.inl k))

/-- The derivative of a comb is again a comb: `φ'(τ) ⬝ᵥ c = r_{drot c}(τ)`. -/
lemma dphi_dotProduct_eq_rcomb_drot (ω : Fin K → ℝ) (c : Fin K ⊕ Fin K → ℝ) (τ : ℝ) :
    dphi ω τ ⬝ᵥ c = rcomb ω (drot ω c) τ := by
  rw [rcomb_eq_sum]
  simp only [dotProduct, Fintype.sum_sum_type, dphi_inl, dphi_inr, drot, Sum.elim_inl, Sum.elim_inr]
  rw [add_comm]
  congr 1 <;> refine Finset.sum_congr rfl fun k _ => ?_ <;> ring

/-- Second derivative coefficients: `drot (drot c) = −ω² c` (entrywise). -/
lemma drot_drot (ω : Fin K → ℝ) (c : Fin K ⊕ Fin K → ℝ) :
    drot ω (drot ω c) = fun i => -(freq ω i) ^ 2 * c i := by
  funext i
  cases i with
  | inl k => simp only [drot, freq, Sum.elim_inl, Sum.elim_inr]; ring
  | inr k => simp only [drot, freq, Sum.elim_inl, Sum.elim_inr]; ring

/-- Comb form of the derivative: the derivative of `r_c` is `r_{drot c}`. -/
lemma hasDerivAt_rcomb_drot (ω : Fin K → ℝ) (c : Fin K ⊕ Fin K → ℝ) (τ : ℝ) :
    HasDerivAt (rcomb ω c) (rcomb ω (drot ω c) τ) τ := by
  rw [← dphi_dotProduct_eq_rcomb_drot]
  exact hasDerivAt_rcomb ω c τ

/-- A function vanishing on an open interval has vanishing derivative there. -/
lemma deriv_eq_zero_of_eqOn_zero {F F' : ℝ → ℝ} {l r : ℝ}
    (hF : ∀ τ ∈ Set.Ioo l r, HasDerivAt F (F' τ) τ) (h0 : ∀ τ ∈ Set.Ioo l r, F τ = 0) :
    ∀ τ ∈ Set.Ioo l r, F' τ = 0 := by
  intro τ hτ
  have hev : F =ᶠ[nhds τ] fun _ => (0 : ℝ) :=
    Filter.eventuallyEq_of_mem (Ioo_mem_nhds hτ.1 hτ.2) (fun x hx => h0 x hx)
  exact (hF τ hτ).unique ((hasDerivAt_const τ (0 : ℝ)).congr_of_eventuallyEq hev)

/-- A single mode `α cos(ωτ) + β sin(ωτ)`, `ω ≠ 0`, vanishing on a nondegenerate open interval has
`α = β = 0`. -/
theorem cos_sin_coeff_eq_zero {α β ω l r : ℝ} (hω : ω ≠ 0) (hlr : l < r)
    (h : ∀ τ ∈ Set.Ioo l r, α * Real.cos (ω * τ) + β * Real.sin (ω * τ) = 0) : α = 0 ∧ β = 0 := by
  have hd : ∀ τ ∈ Set.Ioo l r, HasDerivAt (fun x => α * Real.cos (ω * x) + β * Real.sin (ω * x))
      (ω * (β * Real.cos (ω * τ) - α * Real.sin (ω * τ))) τ := by
    intro τ _
    have hc := (((hasDerivAt_id τ).const_mul ω).cos).const_mul α
    have hs := (((hasDerivAt_id τ).const_mul ω).sin).const_mul β
    have h' := hc.add hs
    simp only [id, mul_one] at h'
    exact h'.congr_deriv (by ring)
  have hd0 := deriv_eq_zero_of_eqOn_zero hd h
  set τ₀ := (l + r) / 2
  have hτ₀ : τ₀ ∈ Set.Ioo l r := ⟨by simp only [τ₀]; linarith, by simp only [τ₀]; linarith⟩
  have hx := h τ₀ hτ₀
  have hy : β * Real.cos (ω * τ₀) - α * Real.sin (ω * τ₀) = 0 :=
    (mul_eq_zero.mp (hd0 τ₀ hτ₀)).resolve_left hω
  have hsq := Real.cos_sq_add_sin_sq (ω * τ₀)
  constructor
  · linear_combination Real.cos (ω * τ₀) * hx - Real.sin (ω * τ₀) * hy - α * hsq
  · linear_combination Real.sin (ω * τ₀) * hx + Real.cos (ω * τ₀) * hy - β * hsq

/-- The comb `r_c` vanishes on the open interval `(l, r)`. -/
def CombZeroOn (ω : Fin K → ℝ) (l r : ℝ) (c : Fin K ⊕ Fin K → ℝ) : Prop :=
  ∀ τ ∈ Set.Ioo l r, rcomb ω c τ = 0

/-- Vanishing on an interval is preserved by differentiation (`c ↦ drot ω c`). -/
lemma CombZeroOn.diff {ω : Fin K → ℝ} {l r : ℝ} {c : Fin K ⊕ Fin K → ℝ}
    (hc : CombZeroOn ω l r c) : CombZeroOn ω l r (drot ω c) :=
  deriv_eq_zero_of_eqOn_zero (fun τ _ => hasDerivAt_rcomb_drot ω c τ) hc

/-- The operator `−d²/dτ² − μ` acts on comb coefficients as `c_i ↦ (ω_{(i)}² − μ) c_i`, and it preserves
vanishing on an interval. -/
lemma CombZeroOn.scale {ω : Fin K → ℝ} {l r : ℝ} {c : Fin K ⊕ Fin K → ℝ}
    (hc : CombZeroOn ω l r c) (μ : ℝ) :
    CombZeroOn ω l r (fun i => ((freq ω i) ^ 2 - μ) * c i) := by
  have hv : (fun i => ((freq ω i) ^ 2 - μ) * c i) = -(drot ω (drot ω c)) - μ • c := by
    funext i
    simp only [drot_drot, Pi.sub_apply, Pi.neg_apply, Pi.smul_apply, smul_eq_mul]
    ring
  have h2 := hc.diff.diff
  intro τ hτ
  rw [hv]
  have e1 := h2 τ hτ
  have e2 := hc τ hτ
  simp only [rcomb] at e1 e2 ⊢
  rw [dotProduct_sub, dotProduct_neg, dotProduct_smul, e1, e2]
  simp

/-- Iterating `CombZeroOn.scale` over a finite set of frequencies `{ω_j : j ∈ S}`. -/
lemma CombZeroOn.prod {ω : Fin K → ℝ} {l r : ℝ} {c : Fin K ⊕ Fin K → ℝ}
    (hc : CombZeroOn ω l r c) (S : Finset (Fin K)) :
    CombZeroOn ω l r (fun i => (∏ j ∈ S, ((freq ω i) ^ 2 - (ω j) ^ 2)) * c i) := by
  induction S using Finset.induction_on with
  | empty => simpa using hc
  | insert j S hj ih =>
    have h := ih.scale ((ω j) ^ 2)
    have hv : (fun i => (∏ j' ∈ insert j S, ((freq ω i) ^ 2 - (ω j') ^ 2)) * c i) =
        fun i => ((freq ω i) ^ 2 - (ω j) ^ 2) *
          ((∏ j' ∈ S, ((freq ω i) ^ 2 - (ω j') ^ 2)) * c i) := by
      funext i
      rw [Finset.prod_insert hj, mul_assoc]
    rw [hv]
    exact h

/-- Isolating mode `k`: the product over `j ≠ k` turns `r_c` into `P_k (α_k cos ω_k τ + β_k sin ω_k τ)`
with `P_k = ∏_{j ≠ k} (ω_k² − ω_j²)`. -/
lemma rcomb_prod_erase (ω : Fin K → ℝ) (c : Fin K ⊕ Fin K → ℝ) (k : Fin K) (τ : ℝ) :
    rcomb ω (fun i => (∏ j ∈ Finset.univ.erase k, ((freq ω i) ^ 2 - (ω j) ^ 2)) * c i) τ =
      (∏ j ∈ Finset.univ.erase k, ((ω k) ^ 2 - (ω j) ^ 2)) *
        (c (Sum.inl k) * Real.cos (ω k * τ) + c (Sum.inr k) * Real.sin (ω k * τ)) := by
  have hz : ∀ k' : Fin K, k' ≠ k →
      (∏ j ∈ Finset.univ.erase k, ((ω k') ^ 2 - (ω j) ^ 2)) = 0 := fun k' hk' =>
    Finset.prod_eq_zero (Finset.mem_erase.mpr ⟨hk', Finset.mem_univ _⟩) (sub_self _)
  rw [rcomb_eq_sum]
  simp only [freq, Sum.elim_inl, Sum.elim_inr]
  rw [Finset.sum_eq_single k (fun k' _ hk' => by rw [hz k' hk']; ring) (by simp),
    Finset.sum_eq_single k (fun k' _ hk' => by rw [hz k' hk']; ring) (by simp)]
  ring

/-- A comb with distinct positive frequencies that vanishes on a nondegenerate open interval has zero
coefficients (the comb half of SI §S4.1's linear-independence claim). -/
theorem coeff_eq_zero_of_combZeroOn {ω : Fin K → ℝ} (hinj : Function.Injective ω)
    (hpos : ∀ k, 0 < ω k) {l r : ℝ} (hlr : l < r) {c : Fin K ⊕ Fin K → ℝ}
    (hc : CombZeroOn ω l r c) : c = 0 := by
  have hk : ∀ k, c (Sum.inl k) = 0 ∧ c (Sum.inr k) = 0 := by
    intro k
    have hP : (∏ j ∈ Finset.univ.erase k, ((ω k) ^ 2 - (ω j) ^ 2)) ≠ 0 := by
      rw [Finset.prod_ne_zero_iff]
      intro j hj heq
      have hjk : j ≠ k := Finset.ne_of_mem_erase hj
      have h2 : ω k ^ 2 = ω j ^ 2 := sub_eq_zero.mp heq
      have h3 : ω k = ω j := (sq_eq_sq₀ (hpos k).le (hpos j).le).mp h2
      exact hjk (hinj h3).symm
    have hm := hc.prod (Finset.univ.erase k)
    refine cos_sin_coeff_eq_zero (hpos k).ne' hlr (fun τ hτ => ?_)
    have h := hm τ hτ
    rw [rcomb_prod_erase] at h
    exact (mul_eq_zero.mp h).resolve_left hP
  funext i
  cases i with
  | inl k => exact (hk k).1
  | inr k => exact (hk k).2

/-- **SI §S4.1 "[PROVEN] The reconstruction space, defined directly".** With
distinct positive frequencies `ω_k`, if `u₀ + u₁ τ + Σ_k [α_k cos(ω_k τ) + β_k sin(ω_k τ)] = 0` on a
nondegenerate open interval, then `u₀ = u₁ = 0` and every `α_k, β_k = 0`: the family
`{1, τ} ∪ {cos(ω_k τ), sin(ω_k τ)}` is linearly independent on every nondegenerate interval. -/
theorem affine_add_rcomb_eqOn_zero {ω : Fin K → ℝ} (hinj : Function.Injective ω)
    (hpos : ∀ k, 0 < ω k) {l r : ℝ} (hlr : l < r) {u₀ u₁ : ℝ} {c : Fin K ⊕ Fin K → ℝ}
    (h : ∀ τ ∈ Set.Ioo l r, u₀ + u₁ * τ + rcomb ω c τ = 0) : u₀ = 0 ∧ u₁ = 0 ∧ c = 0 := by
  -- first derivative: `u₁ + r_{drot c}` vanishes on the interval
  have h1 : ∀ τ ∈ Set.Ioo l r, u₁ + rcomb ω (drot ω c) τ = 0 := by
    refine deriv_eq_zero_of_eqOn_zero (F := fun τ => u₀ + u₁ * τ + rcomb ω c τ)
      (fun τ _ => ?_) h
    have ha := ((hasDerivAt_id τ).const_mul u₁).const_add u₀
    rw [mul_one] at ha
    exact ha.add (hasDerivAt_rcomb_drot ω c τ)
  -- second derivative: `r_{drot (drot c)}` vanishes on the interval
  have h2 : CombZeroOn ω l r (drot ω (drot ω c)) := by
    refine deriv_eq_zero_of_eqOn_zero (F := fun τ => u₁ + rcomb ω (drot ω c) τ)
      (fun τ _ => ?_) h1
    exact (hasDerivAt_rcomb_drot ω (drot ω c) τ).const_add u₁
  have h3 := coeff_eq_zero_of_combZeroOn hinj hpos hlr h2
  have hc : c = 0 := by
    funext i
    have hi := congrFun h3 i
    rw [drot_drot] at hi
    have hf : freq ω i ≠ 0 := by
      cases i with
      | inl k => exact (hpos k).ne'
      | inr k => exact (hpos k).ne'
    have hne : -(freq ω i) ^ 2 ≠ 0 := neg_ne_zero.mpr (pow_ne_zero 2 hf)
    exact (mul_eq_zero.mp hi).resolve_left hne
  subst hc
  have hA : ∀ τ ∈ Set.Ioo l r, u₀ + u₁ * τ = 0 := fun τ hτ => by
    simpa using h τ hτ
  have hm1 : l + (r - l) / 3 ∈ Set.Ioo l r := ⟨by linarith, by linarith⟩
  have hm2 : l + 2 * (r - l) / 3 ∈ Set.Ioo l r := ⟨by linarith, by linarith⟩
  have hne : l + (r - l) / 3 ≠ l + 2 * (r - l) / 3 := by intro h'; linarith
  obtain ⟨h0, h1'⟩ := affine_eq_zero_of_two_zeros hne (hA _ hm1) (hA _ hm2)
  exact ⟨h0, h1', rfl⟩

/-- **Direct-sum form (SI §S4.1: `Aff₁ ∩ span{cos ω_k τ, sin ω_k τ} = {0}`).** With distinct
positive frequencies, a residual comb that coincides with an affine function on a nondegenerate open
interval is the zero comb, and the affine function is zero. -/
theorem rcomb_eq_affine_on_interval {ω : Fin K → ℝ} (hinj : Function.Injective ω)
    (hpos : ∀ k, 0 < ω k) {l r : ℝ} (hlr : l < r) {u₀ u₁ : ℝ} {c : Fin K ⊕ Fin K → ℝ}
    (h : ∀ τ ∈ Set.Ioo l r, rcomb ω c τ = u₀ + u₁ * τ) : c = 0 ∧ u₀ = 0 ∧ u₁ = 0 := by
  have h' : ∀ τ ∈ Set.Ioo l r, -u₀ + -u₁ * τ + rcomb ω c τ = 0 := fun τ hτ => by
    rw [h τ hτ]; ring
  obtain ⟨h0, h1, hc⟩ := affine_add_rcomb_eqOn_zero hinj hpos hlr h'
  exact ⟨hc, neg_eq_zero.mp h0, neg_eq_zero.mp h1⟩

/-- The spanning family of the reconstruction space `R = Aff₁ ⊕ span{cos ω_k τ, sin ω_k τ}` (SI §S4.1),
as functions on the open interval `(l, r)`: `inl 0 ↦ 1`, `inl 1 ↦ τ`, `inr i ↦ φ(τ)_i`. -/
noncomputable def reconBasis (ω : Fin K → ℝ) (l r : ℝ) :
    Fin 2 ⊕ (Fin K ⊕ Fin K) → (Set.Ioo l r → ℝ) :=
  Sum.elim (fun j τ => (τ : ℝ) ^ (j : ℕ)) (fun i τ => phi ω τ i)

/-- **`LinearIndependent` form; SI §S4.1.** For distinct positive frequencies,
the family `{1, τ} ∪ {cos(ω_k τ), sin(ω_k τ)}` is linearly independent over `ℝ` as functions on any
nondegenerate open interval `(l, r)`, so the sum `Aff₁ ⊕ span{…}` is direct. -/
theorem linearIndependent_reconBasis {ω : Fin K → ℝ} (hinj : Function.Injective ω)
    (hpos : ∀ k, 0 < ω k) {l r : ℝ} (hlr : l < r) :
    LinearIndependent ℝ (reconBasis ω l r) := by
  rw [Fintype.linearIndependent_iff]
  intro g hg
  have h : ∀ τ ∈ Set.Ioo l r,
      g (Sum.inl 0) + g (Sum.inl 1) * τ + rcomb ω (fun i => g (Sum.inr i)) τ = 0 := by
    intro τ hτ
    have e := congrFun hg ⟨τ, hτ⟩
    simp only [Fintype.sum_sum_type, Fin.sum_univ_two, reconBasis, Finset.sum_apply,
      Pi.smul_apply, Sum.elim_inl, Sum.elim_inr, smul_eq_mul, Pi.zero_apply, Fin.val_zero,
      Fin.val_one, pow_zero, pow_one, mul_one] at e
    rw [rcomb, dotProduct_comm]
    simp only [dotProduct, Fintype.sum_sum_type]
    linear_combination e
  obtain ⟨h0, h1, hc⟩ := affine_add_rcomb_eqOn_zero hinj hpos hlr h
  intro i
  cases i with
  | inl j => fin_cases j <;> assumption
  | inr i => exact congrFun hc i

end FQGNN.Readout

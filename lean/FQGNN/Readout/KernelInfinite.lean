import Mathlib
import FQGNN.Readout.Kernel
import FQGNN.Backbone.Kronecker

/-!
# The FQ intensity kernel for an infinite spectrum (SI Prop. S5.1 under (H1)–(H3))

This module extends `FQGNN.Readout.Kernel`, which proves SI Prop. S5.1 for a *finite* spectrum.
Here the spectrum index type `σ` is arbitrary (in particular countably infinite, e.g. the full
module `Λ' ⊆ Lᵀℤⁿ`), and the masses satisfy the two hypotheses of SI §S5.1:

* **(H1) nonnegative masses** `0 ≤ m k` (the masses `|c_k|²`);
* **(H3) finite total mass** `Summable m`, i.e. `γ̂_s(ℝᵈ) = Σ_k m_k < ∞`.

Sources:
* SI §S5.1 (Proposition S5.1 and hypotheses (H1)–(H3)), §S5.3 (proof:
  "Reality and evenness", "Continuity of `κ` follows from (H3) … dominated convergence", eq. (S5.4)
  "the interchange of finite sum and integral being justified by (H3)"), §S5.4 ("For the full
  countably infinite spectrum `Λ'`, finiteness (H3) requires `Σ_k |c_k|² < ∞` … Proposition S5.1
  extends in the limit by dominated convergence in (S5.4)").
* Manuscript, Methods (the intensity kernel `K_int`).

Main results (namespace `FQGNN.Readout`):
* `summable_fqTerm` — the series `Σ_k m_k cos(2π s ⟨ξ_k, δ⟩)` converges for every `δ` (dominated
  by `m`).
* `fqProfileInf_zero`, `fqProfileInf_neg`, `continuous_fqProfileInf`,
  `uniformContinuous_fqProfileInf`, `abs_fqProfileInf_le` — `κ(0) = Σ' m`, evenness, (uniform)
  continuity, `|κ| ≤ κ(0)`.
* `fqProfileInf_eq_fqProfile`, `tendstoUniformly_fqProfile_trunc`,
  `abs_fqProfileInf_sub_trunc_le` — the finite truncations of `Kernel.lean` converge uniformly to
  `κ`, with the explicit tail bound.
* `sum_sum_mul_fqProfileInf` (the sum-of-squares identity (S5.4) for the infinite spectrum),
  `fqGramInf_posSemidef` (**Prop. S5.1, real form**), `fqKernelInf_posSemidef` (positive-definite
  function in Bochner's sense on all of `ℝᵈ`), `fqGramInf_complex_nonneg` and
  `fqGramInf_complex_posSemidef` (**eq. (S5.3), complex coefficients**).
* `tsum_fqTerm_sin_eq_zero`, `fqProfileInf_eq_tsum_cexp` — under (H2) (an involution `k ↦ -k`
  preserving the masses and negating the frequencies) the exponential sum is already real, so the
  "Re" in manuscript eq. (3) is redundant (SI §S5.3); `fqProfileInf_eq_re` holds without (H2).
* `intensityKernelInf_props` — the manuscript's `K_int` with complex amplitudes `c_k`, `(H1)`
  automatic, under `Summable (‖c_k‖²)`.
* `summable_norm_intCast_rpow`, `summable_of_le_norm_rpow`, `fqGramInf_posSemidef_of_decay` —
  SI §S5.4: a decay `m_k ≤ C ‖k‖^{−p}` on the full module `k ∈ ℤⁿ` with `p > n` gives (H3), hence
  Prop. S5.1 for the full spectrum.
* SI §S5.5 (a), "idealized infinite limit": `sum_sum_mul_fqProfileInf_pos`,
  `fqGramInf_posDef_of_dense` (strictly positive masses + dense atoms `{s ξ_k}` ⇒ the Gram matrix of
  distinct anchors is positive **definite**), `fqGramInf_posDef_ambient` (the ambient module
  `Lᵀℤⁿ` with the hypothesis that makes it dense, `Lℝᵈ ∩ ℤⁿ = {0}`), and
  `fqGramInf_ambient_not_posDef` (without that hypothesis the conclusion fails for every choice of
  masses).

`FQGNN.Backbone.Kronecker` is imported for the density of the ambient module
(`dense_spectrumModule_iff`) and two small helper lemmas.
-/

namespace FQGNN.Readout

open Matrix Real Filter Topology
open scoped ComplexOrder

section InfiniteKernel

variable {σ : Type*} {d : ℕ}

/-- The `k`-th term `m_k cos(2π s ⟨ξ_k, δ⟩)` of the FQ kernel profile (SI eq. (S5.2)). -/
noncomputable def fqTerm (m : σ → ℝ) (ξ : σ → Fin d → ℝ) (s : ℝ) (δ : Fin d → ℝ) (k : σ) : ℝ :=
  m k * Real.cos (2 * π * s * (ξ k ⬝ᵥ δ))

/-- Profile `κ(δ) = Σ'_k m_k cos(2π s ⟨ξ_k, δ⟩)` of the FQ kernel for an arbitrary (possibly
infinite) spectrum index `σ` (SI eq. (S5.2) with the sum over the full spectrum `Λ'`, §S5.4).
For a finite `σ` it is `fqProfile` (`fqProfileInf_eq_fqProfile`). -/
noncomputable def fqProfileInf (m : σ → ℝ) (ξ : σ → Fin d → ℝ) (s : ℝ) (δ : Fin d → ℝ) : ℝ :=
  ∑' k, fqTerm m ξ s δ k

/-- The FQ kernel `K_FQ(u, u') = κ(u − u')` for an infinite spectrum (SI eq. (S5.2)). -/
noncomputable def fqKernelInf (m : σ → ℝ) (ξ : σ → Fin d → ℝ) (s : ℝ) (u u' : Fin d → ℝ) : ℝ :=
  fqProfileInf m ξ s (u - u')

/-- The Gram matrix `A_{pq} = K_FQ(u_p, u_q)` on a point family `u : ι → ℝᵈ` (SI Prop. S5.1). -/
noncomputable def fqGramInf {ι : Type*} (m : σ → ℝ) (ξ : σ → Fin d → ℝ) (s : ℝ)
    (u : ι → Fin d → ℝ) : Matrix ι ι ℝ :=
  Matrix.of fun p q => fqKernelInf m ξ s (u p) (u q)

@[simp] lemma fqGramInf_apply {ι : Type*} (m : σ → ℝ) (ξ : σ → Fin d → ℝ) (s : ℝ)
    (u : ι → Fin d → ℝ) (p q : ι) :
    fqGramInf m ξ s u p q = fqProfileInf m ξ s (u p - u q) := rfl

/-! ### Convergence of the series under (H1) + (H3) -/

/-- Each term is dominated by its mass: `‖m_k cos(…)‖ ≤ m_k` under (H1). -/
lemma norm_fqTerm_le {m : σ → ℝ} (hm : ∀ k, 0 ≤ m k) (ξ : σ → Fin d → ℝ) (s : ℝ)
    (δ : Fin d → ℝ) (k : σ) : ‖fqTerm m ξ s δ k‖ ≤ m k := by
  rw [fqTerm, Real.norm_eq_abs, abs_mul, abs_of_nonneg (hm k)]
  exact mul_le_of_le_one_right (hm k) (Real.abs_cos_le_one _)

/-- **Summability (SI §S5.3–S5.4, (H1)+(H3)).** For nonnegative summable masses the series
`Σ_k m_k cos(2π s ⟨ξ_k, δ⟩)` converges (absolutely) for every `δ`: it is dominated by `Σ_k m_k`. -/
theorem summable_fqTerm {m : σ → ℝ} (hm : ∀ k, 0 ≤ m k) (hsum : Summable m)
    (ξ : σ → Fin d → ℝ) (s : ℝ) (δ : Fin d → ℝ) : Summable (fqTerm m ξ s δ) :=
  Summable.of_norm_bounded hsum (norm_fqTerm_le hm ξ s δ)

/-- The sine companion series `Σ_k m_k sin(2π s ⟨ξ_k, δ⟩)` also converges under (H1)+(H3)
(used for the "Re" form, SI §S5.3). -/
theorem summable_fqTerm_sin {m : σ → ℝ} (hm : ∀ k, 0 ≤ m k) (hsum : Summable m)
    (ξ : σ → Fin d → ℝ) (s : ℝ) (δ : Fin d → ℝ) :
    Summable fun k => m k * Real.sin (2 * π * s * (ξ k ⬝ᵥ δ)) := by
  refine Summable.of_norm_bounded hsum fun k => ?_
  rw [Real.norm_eq_abs, abs_mul, abs_of_nonneg (hm k)]
  exact mul_le_of_le_one_right (hm k) (Real.abs_sin_le_one _)

/-! ### Value at the origin, evenness, continuity -/

/-- **SI §S5.3 Remark.** `κ(0) = Σ'_k m_k = γ̂_s(ℝᵈ)`, the total diffracted intensity (the
normalizing mass). Holds for any masses (both sides are `tsum`s of the same family). -/
theorem fqProfileInf_zero (m : σ → ℝ) (ξ : σ → Fin d → ℝ) (s : ℝ) :
    fqProfileInf m ξ s 0 = ∑' k, m k := by
  simp [fqProfileInf, fqTerm]

/-- **Evenness** `κ(−δ) = κ(δ)` (SI Prop. S5.1 "real, even"; SI §S5.3). Holds termwise, for any
masses. -/
theorem fqProfileInf_neg (m : σ → ℝ) (ξ : σ → Fin d → ℝ) (s : ℝ) (δ : Fin d → ℝ) :
    fqProfileInf m ξ s (-δ) = fqProfileInf m ξ s δ := by
  simp [fqProfileInf, fqTerm, dotProduct_neg, mul_neg, Real.cos_neg]

/-- **Translation invariance and symmetry** `K_FQ(u, u') = K_FQ(u', u)` (SI Prop. S5.1). -/
theorem fqKernelInf_comm (m : σ → ℝ) (ξ : σ → Fin d → ℝ) (s : ℝ) (u u' : Fin d → ℝ) :
    fqKernelInf m ξ s u u' = fqKernelInf m ξ s u' u := by
  rw [fqKernelInf, fqKernelInf, ← neg_sub, fqProfileInf_neg]

/-- **Continuity (SI Prop. S5.1 "continuous"; SI §S5.3 "Continuity of `κ` follows from (H3)").**
Under (H1)+(H3) the profile is continuous: it is a uniformly convergent series of continuous
functions (Weierstrass M-test with majorant `m`), which is the dominated-convergence argument of the
SI. -/
theorem continuous_fqProfileInf {m : σ → ℝ} (hm : ∀ k, 0 ≤ m k) (hsum : Summable m)
    (ξ : σ → Fin d → ℝ) (s : ℝ) : Continuous (fqProfileInf m ξ s) := by
  refine continuous_tsum (fun k => ?_) hsum (fun k δ => norm_fqTerm_le hm ξ s δ k)
  unfold fqTerm
  fun_prop

/-- With (H1)+(H3) the profile is bounded by its value at the origin: `|κ(δ)| ≤ κ(0) = Σ'_k m_k`
(SI §S5.3 Remark; "a finite (bounded) measure has a bounded … Fourier transform"). -/
theorem abs_fqProfileInf_le {m : σ → ℝ} (hm : ∀ k, 0 ≤ m k) (hsum : Summable m)
    (ξ : σ → Fin d → ℝ) (s : ℝ) (δ : Fin d → ℝ) :
    |fqProfileInf m ξ s δ| ≤ fqProfileInf m ξ s 0 := by
  rw [fqProfileInf_zero, ← Real.norm_eq_abs]
  exact tsum_of_norm_bounded hsum.hasSum (norm_fqTerm_le hm ξ s δ)

/-- Each term `δ ↦ m_k cos(2π s ⟨ξ_k, δ⟩)` is uniformly continuous (`cos` is 1-Lipschitz and
`δ ↦ 2π s ⟨ξ_k, δ⟩` is a continuous additive map). -/
lemma uniformContinuous_fqTerm (m : σ → ℝ) (ξ : σ → Fin d → ℝ) (s : ℝ) (k : σ) :
    UniformContinuous fun δ => fqTerm m ξ s δ k := by
  let f : (Fin d → ℝ) →+ ℝ :=
    { toFun := fun δ => 2 * π * s * (ξ k ⬝ᵥ δ)
      map_zero' := by simp
      map_add' := fun x y => by simp [dotProduct_add, mul_add] }
  have hf : UniformContinuous f := uniformContinuous_addMonoidHom_of_continuous (by
    change Continuous fun δ => 2 * π * s * (ξ k ⬝ᵥ δ)
    fun_prop)
  exact (Real.lipschitzWith_cos.uniformContinuous.comp hf).const_smul (m k)

/-- Every finite partial sum of the profile series is uniformly continuous. -/
lemma uniformContinuous_finsetSum_fqTerm (m : σ → ℝ) (ξ : σ → Fin d → ℝ) (s : ℝ)
    (t : Finset σ) : UniformContinuous fun δ => ∑ k ∈ t, fqTerm m ξ s δ k := by
  classical
  induction t using Finset.induction_on with
  | empty => simpa using uniformContinuous_const
  | insert k t hk ih =>
    simp only [Finset.sum_insert hk]
    exact (uniformContinuous_fqTerm m ξ s k).add ih

/-- **Uniform continuity (SI §S5.3: "a finite (bounded) measure has a bounded uniformly continuous
Fourier transform").** Under (H1)+(H3) the profile `κ` is uniformly continuous on `ℝᵈ`: it is the
uniform limit of its (uniformly continuous) finite partial sums. -/
theorem uniformContinuous_fqProfileInf {m : σ → ℝ} (hm : ∀ k, 0 ≤ m k) (hsum : Summable m)
    (ξ : σ → Fin d → ℝ) (s : ℝ) : UniformContinuous (fqProfileInf m ξ s) := by
  have h := tendstoUniformly_tsum hsum (fun k δ => norm_fqTerm_le hm ξ s δ k)
  exact h.uniformContinuous
    (Frequently.of_forall fun t => uniformContinuous_finsetSum_fqTerm m ξ s t)

/-! ### Agreement with the finite spectrum and uniform convergence of the truncations -/

/-- For a finite spectrum the infinite-spectrum profile is the finite one of `Kernel.lean`
(SI §S5.4: "For any finite index set, (H3) is automatic and Proposition S5.1 holds verbatim"). -/
theorem fqProfileInf_eq_fqProfile [Fintype σ] (m : σ → ℝ) (ξ : σ → Fin d → ℝ) (s : ℝ) :
    fqProfileInf m ξ s = fqProfile m ξ s := by
  funext δ
  rw [fqProfileInf, tsum_fintype]
  rfl

/-- The profile of the finite truncation of the spectrum to `t : Finset σ` (the enumerated
truncation `Λ'_K` of SI §S5.4) is the partial sum over `t`. -/
theorem fqProfile_trunc_eq_sum (m : σ → ℝ) (ξ : σ → Fin d → ℝ) (s : ℝ) (t : Finset σ)
    (δ : Fin d → ℝ) :
    fqProfile (fun k : t => m k) (fun k : t => ξ k) s δ = ∑ k ∈ t, fqTerm m ξ s δ k := by
  rw [fqProfile, ← Finset.sum_coe_sort t]
  rfl

/-- **SI §S5.4, "extends in the limit".** Under (H1)+(H3), the profiles of the finite truncations
`Λ'_t` (`t : Finset σ`, directed by inclusion) converge **uniformly** on `ℝᵈ` to the profile of the
full spectrum. -/
theorem tendstoUniformly_fqProfile_trunc {m : σ → ℝ} (hm : ∀ k, 0 ≤ m k) (hsum : Summable m)
    (ξ : σ → Fin d → ℝ) (s : ℝ) :
    TendstoUniformly (fun t : Finset σ => fqProfile (fun k : t => m k) (fun k : t => ξ k) s)
      (fqProfileInf m ξ s) atTop := by
  have h := tendstoUniformly_tsum hsum (fun k δ => norm_fqTerm_le hm ξ s δ k)
  simp_rw [← fqProfile_trunc_eq_sum] at h
  exact h

/-- **SI §S5.4, tail bound.** "The truncation … has a tail intensity controlled by the same
decay": the truncation error is bounded by the mass outside the truncation,
`|κ(δ) − κ_t(δ)| ≤ Σ'_{k ∉ t} m_k`, uniformly in `δ`. -/
theorem abs_fqProfileInf_sub_trunc_le {m : σ → ℝ} (hm : ∀ k, 0 ≤ m k) (hsum : Summable m)
    (ξ : σ → Fin d → ℝ) (s : ℝ) (t : Finset σ) (δ : Fin d → ℝ) :
    |fqProfileInf m ξ s δ - fqProfile (fun k : t => m k) (fun k : t => ξ k) s δ| ≤
      ∑' k : ((t : Set σ)ᶜ : Set σ), m k := by
  have hS := summable_fqTerm hm hsum ξ s δ
  rw [fqProfile_trunc_eq_sum, fqProfileInf, ← hS.sum_add_tsum_compl (s := t), add_sub_cancel_left,
    ← Real.norm_eq_abs]
  exact tsum_of_norm_bounded (hsum.subtype _).hasSum fun k => norm_fqTerm_le hm ξ s δ k

/-! ### Positive-semidefiniteness (SI Prop. S5.1, eqs. (S5.3)–(S5.4)) -/

/-- Finite-sum identity behind (S5.4) for one frequency:
`Σ_p Σ_q a_p a_q c cos(x_p − x_q) = c [(Σ_p a_p cos x_p)² + (Σ_p a_p sin x_p)²]`
(`cos(x − y) = cos x cos y + sin x sin y`, SI §S5.5 (a)). -/
lemma sum_sum_mul_cos_sub {P : Type*} (S : Finset P) (a x : P → ℝ) (c : ℝ) :
    ∑ p ∈ S, ∑ q ∈ S, a p * a q * (c * Real.cos (x p - x q)) =
      c * ((∑ p ∈ S, a p * Real.cos (x p)) ^ 2 + (∑ p ∈ S, a p * Real.sin (x p)) ^ 2) := by
  have h1 : (∑ p ∈ S, a p * Real.cos (x p)) ^ 2 =
      ∑ p ∈ S, ∑ q ∈ S, a p * Real.cos (x p) * (a q * Real.cos (x q)) := by
    rw [sq, Finset.sum_mul_sum]
  have h2 : (∑ p ∈ S, a p * Real.sin (x p)) ^ 2 =
      ∑ p ∈ S, ∑ q ∈ S, a p * Real.sin (x p) * (a q * Real.sin (x q)) := by
    rw [sq, Finset.sum_mul_sum]
  rw [h1, h2, ← Finset.sum_add_distrib, Finset.mul_sum]
  refine Finset.sum_congr rfl fun p _ => ?_
  rw [← Finset.sum_add_distrib, Finset.mul_sum]
  refine Finset.sum_congr rfl fun q _ => ?_
  rw [Real.cos_sub]
  ring

/-- **Sum-of-squares identity (SI eq. (S5.4), real coefficients, infinite spectrum).** Under
(H1)+(H3), for every finite family of points `u_p` (`p ∈ S`) and real coefficients `a_p`,
`Σ_{p,q} a_p a_q κ(u_p − u_q) = Σ'_k m_k [(Σ_p a_p cos θ_kp)² + (Σ_p a_p sin θ_kp)²]` with
`θ_kp = 2π s ⟨ξ_k, u_p⟩`. The interchange of the finite sums with the series is the step "justified
by (H3)" in the SI. -/
theorem sum_sum_mul_fqProfileInf {m : σ → ℝ} (hm : ∀ k, 0 ≤ m k) (hsum : Summable m)
    (ξ : σ → Fin d → ℝ) (s : ℝ) {P : Type*} (S : Finset P) (a : P → ℝ) (u : P → Fin d → ℝ) :
    ∑ p ∈ S, ∑ q ∈ S, a p * a q * fqProfileInf m ξ s (u p - u q) =
      ∑' k, m k * ((∑ p ∈ S, a p * Real.cos (2 * π * s * (ξ k ⬝ᵥ u p))) ^ 2 +
        (∑ p ∈ S, a p * Real.sin (2 * π * s * (ξ k ⬝ᵥ u p))) ^ 2) := by
  have hT : ∀ p q, Summable fun k => a p * a q * fqTerm m ξ s (u p - u q) k := fun p q =>
    (summable_fqTerm hm hsum ξ s (u p - u q)).mul_left _
  calc ∑ p ∈ S, ∑ q ∈ S, a p * a q * fqProfileInf m ξ s (u p - u q)
      = ∑ p ∈ S, ∑ q ∈ S, ∑' k, a p * a q * fqTerm m ξ s (u p - u q) k := by
        refine Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun q _ => ?_
        rw [fqProfileInf, tsum_mul_left]
    _ = ∑ p ∈ S, ∑' k, ∑ q ∈ S, a p * a q * fqTerm m ξ s (u p - u q) k := by
        refine Finset.sum_congr rfl fun p _ => ?_
        rw [Summable.tsum_finsetSum fun q _ => hT p q]
    _ = ∑' k, ∑ p ∈ S, ∑ q ∈ S, a p * a q * fqTerm m ξ s (u p - u q) k := by
        rw [Summable.tsum_finsetSum fun p _ => summable_sum fun q _ => hT p q]
    _ = _ := by
        refine tsum_congr fun k => ?_
        rw [← sum_sum_mul_cos_sub]
        refine Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun q _ => ?_
        simp only [fqTerm, dotProduct_sub, mul_sub]

/-- **SI Prop. S5.1 (infinite spectrum, real coefficients), eq. (S5.3).** Under (H1)+(H3),
`Σ_{p,q ∈ S} a_p a_q K_FQ(u_p, u_q) ≥ 0` for every finite family of points and real coefficients. -/
theorem sum_sum_mul_fqProfileInf_nonneg {m : σ → ℝ} (hm : ∀ k, 0 ≤ m k) (hsum : Summable m)
    (ξ : σ → Fin d → ℝ) (s : ℝ) {P : Type*} (S : Finset P) (a : P → ℝ) (u : P → Fin d → ℝ) :
    0 ≤ ∑ p ∈ S, ∑ q ∈ S, a p * a q * fqProfileInf m ξ s (u p - u q) := by
  rw [sum_sum_mul_fqProfileInf hm hsum]
  exact tsum_nonneg fun k => mul_nonneg (hm k) (by positivity)

/-- The quadratic form of the Gram matrix, `aᵀ A a = Σ_{p,q} a_p a_q κ(u_p − u_q)`. -/
theorem dotProduct_fqGramInf_mulVec {ι : Type*} [Fintype ι] (m : σ → ℝ) (ξ : σ → Fin d → ℝ)
    (s : ℝ) (u : ι → Fin d → ℝ) (a : ι → ℝ) :
    a ⬝ᵥ (fqGramInf m ξ s u *ᵥ a) = ∑ p, ∑ q, a p * a q * fqProfileInf m ξ s (u p - u q) := by
  simp only [dotProduct, mulVec, fqGramInf_apply, Finset.mul_sum]
  refine Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun q _ => ?_
  ring

/-- **SI eq. (S5.4), matrix form, infinite spectrum.** `aᵀ A a = Σ'_k m_k [(Σ_p a_p cos θ_kp)² +
(Σ_p a_p sin θ_kp)²]` under (H1)+(H3). -/
theorem dotProduct_fqGramInf_mulVec_eq_tsum {ι : Type*} [Fintype ι] {m : σ → ℝ}
    (hm : ∀ k, 0 ≤ m k) (hsum : Summable m) (ξ : σ → Fin d → ℝ) (s : ℝ) (u : ι → Fin d → ℝ)
    (a : ι → ℝ) :
    a ⬝ᵥ (fqGramInf m ξ s u *ᵥ a) =
      ∑' k, m k * ((∑ p, a p * Real.cos (2 * π * s * (ξ k ⬝ᵥ u p))) ^ 2 +
        (∑ p, a p * Real.sin (2 * π * s * (ξ k ⬝ᵥ u p))) ^ 2) := by
  rw [dotProduct_fqGramInf_mulVec, sum_sum_mul_fqProfileInf hm hsum]

/-- The infinite-spectrum Gram matrix is symmetric (SI Prop. S5.1, "real, even"). -/
theorem fqGramInf_transpose {ι : Type*} (m : σ → ℝ) (ξ : σ → Fin d → ℝ) (s : ℝ)
    (u : ι → Fin d → ℝ) : (fqGramInf m ξ s u)ᵀ = fqGramInf m ξ s u := by
  ext p q
  simp only [transpose_apply, fqGramInf_apply]
  rw [← neg_sub, fqProfileInf_neg]

/-- **SI Proposition S5.1 for the infinite spectrum.** Under (H1) nonnegative
masses and (H3) summable total mass, for every finite family of points `u_p ∈ ℝᵈ`, every family of
frequencies `ξ_k ∈ ℝᵈ` and every scale `s`, the Gram matrix `A_{pq} = K_FQ(u_p, u_q)` is (real
symmetric) positive semidefinite. -/
theorem fqGramInf_posSemidef {ι : Type*} [Fintype ι] {m : σ → ℝ} (hm : ∀ k, 0 ≤ m k)
    (hsum : Summable m) (ξ : σ → Fin d → ℝ) (s : ℝ) (u : ι → Fin d → ℝ) :
    (fqGramInf m ξ s u).PosSemidef := by
  refine PosSemidef.of_dotProduct_mulVec_nonneg ?_ fun a => ?_
  · rw [IsHermitian, conjTranspose_eq_transpose_of_trivial, fqGramInf_transpose]
  · rw [star_trivial, dotProduct_fqGramInf_mulVec]
    exact sum_sum_mul_fqProfileInf_nonneg hm hsum ξ s Finset.univ a u

/-- **Bochner positive-definiteness on all of `ℝᵈ` (SI Prop. S5.1, "positive-definite in the sense
of Bochner").** Under (H1)+(H3), the kernel `K_FQ`, viewed as an (infinite) matrix indexed by
`ℝᵈ × ℝᵈ`, is positive semidefinite in Mathlib's sense (`Matrix.PosSemidef` tests finitely
supported coefficient vectors): i.e. `κ` is a positive-definite function on `ℝᵈ`. -/
theorem fqKernelInf_posSemidef {m : σ → ℝ} (hm : ∀ k, 0 ≤ m k) (hsum : Summable m)
    (ξ : σ → Fin d → ℝ) (s : ℝ) :
    (Matrix.of fun u u' : Fin d → ℝ => fqKernelInf m ξ s u u').PosSemidef := by
  refine ⟨?_, fun x => ?_⟩
  · ext u u'
    simp only [conjTranspose_apply, of_apply, star_trivial]
    exact fqKernelInf_comm m ξ s u' u
  · have h := sum_sum_mul_fqProfileInf_nonneg hm hsum ξ s x.support (fun u => x u) id
    simp only [Finsupp.sum, star_trivial, of_apply, fqKernelInf]
    simpa [mul_assoc, mul_comm, mul_left_comm] using h

/-- Complexification lemma: for a real symmetric family `K` whose real quadratic form is
nonnegative, the Hermitian form `Σ_{p,q} a_p conj(a_q) K_pq` is real and nonnegative for every
complex `a` (write `a = x + i y`: the form is `xᵀKx + yᵀKy`, the cross terms cancel by symmetry). -/
lemma complex_form_nonneg_of_real {ι : Type*} [Fintype ι] (K : ι → ι → ℝ)
    (hK : ∀ p q, K p q = K q p) (hpos : ∀ x : ι → ℝ, 0 ≤ ∑ p, ∑ q, x p * x q * K p q)
    (a : ι → ℂ) : 0 ≤ ∑ p, ∑ q, a p * (starRingEnd ℂ) (a q) * (K p q : ℂ) := by
  rw [Complex.nonneg_iff]
  constructor
  · have hre : (∑ p, ∑ q, a p * (starRingEnd ℂ) (a q) * (K p q : ℂ)).re =
        ∑ p, ∑ q, (a p).re * (a q).re * K p q + ∑ p, ∑ q, (a p).im * (a q).im * K p q := by
      simp only [Complex.re_sum, Complex.mul_re, Complex.conj_re, Complex.conj_im,
        Complex.ofReal_re, Complex.ofReal_im, ← Finset.sum_add_distrib, Complex.mul_im]
      refine Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun q _ => ?_
      ring
    rw [hre]
    exact add_nonneg (hpos _) (hpos _)
  · have him : (∑ p, ∑ q, a p * (starRingEnd ℂ) (a q) * (K p q : ℂ)).im =
        ∑ p, ∑ q, (a p).im * (a q).re * K p q - ∑ p, ∑ q, (a p).re * (a q).im * K p q := by
      simp only [Complex.im_sum, Complex.mul_im, Complex.mul_re, Complex.conj_re, Complex.conj_im,
        Complex.ofReal_re, Complex.ofReal_im, ← Finset.sum_sub_distrib]
      refine Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun q _ => ?_
      ring
    rw [him, Finset.sum_comm (f := fun p q => (a p).re * (a q).im * K p q)]
    rw [eq_comm, sub_eq_zero]
    refine Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun q _ => ?_
    rw [hK]
    ring

/-- **SI eq. (S5.3), complex coefficients, infinite spectrum.** Under (H1)+(H3), for every finite
family of points and every `a : ι → ℂ`, `Σ_{p,q} a_p conj(a_q) K_FQ(u_p, u_q) ≥ 0` in `ℂ`'s partial
order (i.e. the sum is real and nonnegative). -/
theorem fqGramInf_complex_nonneg {ι : Type*} [Fintype ι] {m : σ → ℝ} (hm : ∀ k, 0 ≤ m k)
    (hsum : Summable m) (ξ : σ → Fin d → ℝ) (s : ℝ) (u : ι → Fin d → ℝ) (a : ι → ℂ) :
    0 ≤ ∑ p, ∑ q, a p * (starRingEnd ℂ) (a q) * (fqKernelInf m ξ s (u p) (u q) : ℂ) :=
  complex_form_nonneg_of_real (fun p q => fqKernelInf m ξ s (u p) (u q))
    (fun p q => fqKernelInf_comm m ξ s (u p) (u q))
    (fun x => sum_sum_mul_fqProfileInf_nonneg hm hsum ξ s Finset.univ x u) a

/-- **SI Prop. S5.1, "every Gram matrix … is Hermitian positive-semidefinite" (infinite
spectrum).** Under (H1)+(H3) the complexified Gram matrix is Hermitian positive semidefinite. -/
theorem fqGramInf_complex_posSemidef {ι : Type*} [Fintype ι] {m : σ → ℝ} (hm : ∀ k, 0 ≤ m k)
    (hsum : Summable m) (ξ : σ → Fin d → ℝ) (s : ℝ) (u : ι → Fin d → ℝ) :
    ((fqGramInf m ξ s u).map (fun x : ℝ => (x : ℂ))).PosSemidef := by
  refine PosSemidef.of_dotProduct_mulVec_nonneg ?_ fun x => ?_
  · ext p q
    simp only [conjTranspose_apply, map_apply, fqGramInf_apply, Complex.star_def,
      Complex.conj_ofReal]
    rw [← neg_sub, fqProfileInf_neg]
  · have h := fqGramInf_complex_nonneg hm hsum ξ s u (star x)
    simp only [dotProduct, mulVec, map_apply, Finset.mul_sum, Pi.star_apply, fqGramInf_apply]
    refine h.trans_eq (Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun q _ => ?_)
    simp only [Pi.star_apply, Complex.star_def, Complex.conj_conj, fqKernelInf]
    ring

/-! ### Reality: the "Re" in manuscript eq. (3) and hypothesis (H2) -/

/-- **SI §S5.3 (the "Re" in manuscript eq. (3)).** Under (H1)+(H3),
`κ(δ) = Re Σ'_k m_k exp(i 2π s ⟨ξ_k, δ⟩)` (no symmetry hypothesis needed for the `Re` form). -/
theorem fqProfileInf_eq_re {m : σ → ℝ} (hm : ∀ k, 0 ≤ m k) (hsum : Summable m)
    (ξ : σ → Fin d → ℝ) (s : ℝ) (δ : Fin d → ℝ) :
    fqProfileInf m ξ s δ =
      (∑' k, (m k : ℂ) * Complex.exp (((2 * π * s * (ξ k ⬝ᵥ δ) : ℝ) : ℂ) * Complex.I)).re := by
  have hc : Summable fun k => (m k : ℂ) * Complex.exp (((2 * π * s * (ξ k ⬝ᵥ δ) : ℝ) : ℂ) *
      Complex.I) := by
    refine Summable.of_norm_bounded hsum fun k => ?_
    rw [norm_mul, Complex.norm_exp_ofReal_mul_I, mul_one, Complex.norm_real, Real.norm_eq_abs,
      abs_of_nonneg (hm k)]
  rw [Complex.re_tsum hc, fqProfileInf]
  refine tsum_congr fun k => ?_
  rw [Complex.re_ofReal_mul, Complex.exp_ofReal_mul_I_re, fqTerm]

/-- **(H2) kills the sine series (SI §S5.3, "Reality and evenness").** If an involution `ν` of the
spectrum index (`k ↦ −k`) preserves the masses, `m (ν k) = m k`, and negates the frequencies,
`ξ (ν k) = −ξ k`, then `Σ'_k m_k sin(2π s ⟨ξ_k, δ⟩) = 0` for every `δ`. (No summability needed: the
series equals its own negative.) -/
theorem tsum_fqTerm_sin_eq_zero {m : σ → ℝ} {ξ : σ → Fin d → ℝ} (ν : σ ≃ σ)
    (hmν : ∀ k, m (ν k) = m k) (hξν : ∀ k, ξ (ν k) = -ξ k) (s : ℝ) (δ : Fin d → ℝ) :
    ∑' k, m k * Real.sin (2 * π * s * (ξ k ⬝ᵥ δ)) = 0 := by
  have h := ν.tsum_eq fun k => m k * Real.sin (2 * π * s * (ξ k ⬝ᵥ δ))
  simp only [hmν, hξν, neg_dotProduct, mul_neg, Real.sin_neg, tsum_neg] at h
  linarith

/-- **SI §S5.3: under (H1)–(H3) the exponential sum is already real.** With the (H2) involution
`k ↦ −k`, `Σ'_k m_k exp(i 2π s ⟨ξ_k, δ⟩) = κ(δ)` exactly (as complex numbers): "the exponential
integral is already real and the "Re" in equation (3) of the main text is redundant". -/
theorem fqProfileInf_eq_tsum_cexp {m : σ → ℝ} (hm : ∀ k, 0 ≤ m k) (hsum : Summable m)
    {ξ : σ → Fin d → ℝ} (ν : σ ≃ σ) (hmν : ∀ k, m (ν k) = m k) (hξν : ∀ k, ξ (ν k) = -ξ k)
    (s : ℝ) (δ : Fin d → ℝ) :
    (∑' k, (m k : ℂ) * Complex.exp (((2 * π * s * (ξ k ⬝ᵥ δ) : ℝ) : ℂ) * Complex.I)) =
      (fqProfileInf m ξ s δ : ℂ) := by
  have hc : Summable fun k => (m k : ℂ) * Complex.exp (((2 * π * s * (ξ k ⬝ᵥ δ) : ℝ) : ℂ) *
      Complex.I) := by
    refine Summable.of_norm_bounded hsum fun k => ?_
    rw [norm_mul, Complex.norm_exp_ofReal_mul_I, mul_one, Complex.norm_real, Real.norm_eq_abs,
      abs_of_nonneg (hm k)]
  apply Complex.ext
  · rw [Complex.ofReal_re, fqProfileInf_eq_re hm hsum]
  · rw [Complex.ofReal_im, Complex.im_tsum hc, ← tsum_fqTerm_sin_eq_zero ν hmν hξν s δ]
    refine tsum_congr fun k => ?_
    rw [Complex.im_ofReal_mul, Complex.exp_ofReal_mul_I_im]

/-! ### The manuscript's intensity kernel with complex amplitudes -/

/-- Intensity profile `K_int(δ) = Σ'_k |c_k|² cos(2π s ⟨ξ_k, δ⟩)` for an infinite spectrum
(manuscript Methods; SI eqs. (S5.1)–(S5.2), §S5.4). -/
noncomputable def intensityProfileInf (c : σ → ℂ) (ξ : σ → Fin d → ℝ) (s : ℝ) (δ : Fin d → ℝ) :
    ℝ :=
  fqProfileInf (fun k => ‖c k‖ ^ 2) ξ s δ

/-- **Manuscript Methods / SI Prop. S5.1 for the full spectrum.** For complex amplitudes
`c_k` ((H1) is automatic for the masses `|c_k|²`) with finite total intensity
`Σ_k |c_k|² < ∞` ((H3)), every scale `s` and every finite point family, the Gram matrix of `K_int`
is positive semidefinite, and `K_int` is even, continuous, with `K_int(0) = Σ'_k |c_k|²`. -/
theorem intensityKernelInf_props {ι : Type*} [Fintype ι] (c : σ → ℂ)
    (hc : Summable fun k => ‖c k‖ ^ 2) (ξ : σ → Fin d → ℝ) (s : ℝ) (u : ι → Fin d → ℝ) :
    (fqGramInf (fun k => ‖c k‖ ^ 2) ξ s u).PosSemidef ∧
      (∀ δ, intensityProfileInf c ξ s (-δ) = intensityProfileInf c ξ s δ) ∧
      Continuous (intensityProfileInf c ξ s) ∧
      intensityProfileInf c ξ s 0 = ∑' k, ‖c k‖ ^ 2 :=
  ⟨fqGramInf_posSemidef (fun _ => sq_nonneg _) hc ξ s u, fun δ => fqProfileInf_neg _ ξ s δ,
    continuous_fqProfileInf (fun _ => sq_nonneg _) hc ξ s, fqProfileInf_zero _ ξ s⟩

end InfiniteKernel

/-! ## (H3) from polynomial decay on the full module `ℤⁿ` (SI §S5.4) -/

section Decay

/-- `Σ_{k ∈ ℤⁿ} ‖k‖^{−p} < ∞` for `p > n` (sup norm; the value at `k = 0` is `0^{−p} = 0`). Via
Mathlib's `ZLattice.summable_norm_rpow` for the standard lattice `ℤⁿ ⊆ ℝⁿ`.

Source: SI §S5.4, "the relevant convergence test is `Σ_{k∈ℤⁿ} ‖k‖^{−p} < ∞ ⟺ p > n`" (the `⟸`
direction). -/
theorem summable_norm_intCast_rpow {n : ℕ} {p : ℝ} (hp : (n : ℝ) < p) :
    Summable fun k : Fin n → ℤ => ‖(fun j => (k j : ℝ))‖ ^ (-p) := by
  let b := Pi.basisFun ℝ (Fin n)
  let L := Submodule.span ℤ (Set.range b)
  let bz := b.restrictScalars ℤ
  have hrank : Module.finrank ℤ L = n := by
    rw [Module.finrank_eq_card_basis bz, Fintype.card_fin]
  have hs := ZLattice.summable_norm_rpow L (-p) (by rw [hrank]; linarith)
  have hcoe : ∀ k : Fin n → ℤ, ((bz.equivFun.symm k : L) : Fin n → ℝ) = fun j => (k j : ℝ) := by
    intro k
    funext j
    have h := b.restrictScalars_repr_apply ℤ (bz.equivFun.symm k) j
    have h2 : bz.repr (bz.equivFun.symm k) j = k j := by
      rw [← Module.Basis.equivFun_apply, LinearEquiv.apply_symm_apply]
    rw [h2, Pi.basisFun_repr] at h
    rw [← h]
    simp
  have := (bz.equivFun.symm.toEquiv.summable_iff (f := fun z : L => ‖z‖ ^ (-p))).2 hs
  refine this.congr fun k => ?_
  simp only [Function.comp_apply, LinearEquiv.coe_toEquiv]
  rw [← Submodule.norm_coe, hcoe]

/-- **SI §S5.4: polynomial intensity decay with exponent `p > n` gives (H3).** If nonnegative masses
on the full module index `k ∈ ℤⁿ` satisfy `m_k ≤ C ‖k‖^{−p}` for `k ≠ 0` with `p > n`, then
`Σ_k m_k < ∞` ("a polynomial intensity decay `|c_ξ|² = O(‖k‖^{−p})` is summable … when `p > n`"). -/
theorem summable_of_le_norm_rpow {n : ℕ} {m : (Fin n → ℤ) → ℝ} {C p : ℝ} (hp : (n : ℝ) < p)
    (hm0 : ∀ k, 0 ≤ m k) (hm : ∀ k, k ≠ 0 → m k ≤ C * ‖(fun j => (k j : ℝ))‖ ^ (-p)) :
    Summable m := by
  refine Summable.of_norm_bounded_eventually ((summable_norm_intCast_rpow hp).mul_left C) ?_
  filter_upwards [(Set.finite_singleton (0 : Fin n → ℤ)).compl_mem_cofinite] with k hk
  rw [Real.norm_eq_abs, abs_of_nonneg (hm0 k)]
  exact hm k hk

/-- **SI Prop. S5.1 on the full ambient module under the decay hypothesis of §S5.4.** With the
atoms `ξ_k = Lᵀk` over **all** `k ∈ ℤⁿ` and nonnegative masses decaying like `‖k‖^{−p}`, `p > n`,
every Gram matrix of `K_FQ` is positive semidefinite ("Proposition S5.1 extends in the limit"). -/
theorem fqGramInf_posSemidef_of_decay {n d : ℕ} (L : Matrix (Fin n) (Fin d) ℝ)
    {m : (Fin n → ℤ) → ℝ} {C p : ℝ} (hp : (n : ℝ) < p) (hm0 : ∀ k, 0 ≤ m k)
    (hm : ∀ k, k ≠ 0 → m k ≤ C * ‖(fun j => (k j : ℝ))‖ ^ (-p)) (s : ℝ) {ι : Type*} [Fintype ι]
    (u : ι → Fin d → ℝ) :
    (fqGramInf m (fun k => Lᵀ *ᵥ fun j => (k j : ℝ)) s u).PosSemidef :=
  fqGramInf_posSemidef hm0 (summable_of_le_norm_rpow hp hm0 hm) _ s u

end Decay

/-! ## Strict positive-definiteness in the idealized dense-spectrum limit (SI §S5.5 (a)) -/

section StrictPD

variable {σ : Type*} {d : ℕ}

/-- The character `χ_u(η) = exp(2πi ⟨η, u⟩)` of `ℝᵈ`, as a monoid hom on `Multiplicative ℝᵈ`
(SI §S5.5 (a): "linear independence of the characters `η ↦ e^{2πi⟨η,u_p⟩}`"). -/
noncomputable def fqChar (u : Fin d → ℝ) : Multiplicative (Fin d → ℝ) →* ℂ where
  toFun η := Complex.exp (((2 * π * (η.toAdd ⬝ᵥ u) : ℝ) : ℂ) * Complex.I)
  map_one' := by simp
  map_mul' x y := by
    simp only [toAdd_mul, add_dotProduct, mul_add, Complex.ofReal_add, add_mul, Complex.exp_add]

/-- Distinct points give distinct characters: `χ_u = χ_{u'} → u = u'` (evaluate at
`η = t (u − u')`: then `t ‖u − u'‖² ∈ ℤ` for every real `t`). -/
lemma fqChar_injective : Function.Injective (fqChar (d := d)) := by
  intro u u' h
  have key : ∀ t : ℝ, ∃ z : ℤ, t * ((u - u') ⬝ᵥ (u - u')) = z := by
    intro t
    have h1 := DFunLike.congr_fun h (Multiplicative.ofAdd (t • (u - u')))
    simp only [fqChar, MonoidHom.coe_mk, OneHom.coe_mk, toAdd_ofAdd] at h1
    have h2 : Complex.exp (2 * π * Complex.I * ((t * ((u - u') ⬝ᵥ (u - u')) : ℝ) : ℂ)) = 1 := by
      have : (2 * π * Complex.I * ((t * ((u - u') ⬝ᵥ (u - u')) : ℝ) : ℂ)) =
          ((2 * π * ((t • (u - u')) ⬝ᵥ u) : ℝ) : ℂ) * Complex.I -
            ((2 * π * ((t • (u - u')) ⬝ᵥ u') : ℝ) : ℂ) * Complex.I := by
        simp only [smul_dotProduct, dotProduct_sub, smul_eq_mul]
        push_cast
        ring
      rw [this, Complex.exp_sub, h1, div_self (Complex.exp_ne_zero _)]
    exact (FQGNN.Backbone.cexp_two_pi_I_mul_eq_one_iff _).1 h2
  have h0 : (u - u') ⬝ᵥ (u - u') = 0 := FQGNN.Backbone.eq_zero_of_forall_mul_mem_int key
  exact sub_eq_zero.1 (dotProduct_self_eq_zero.1 h0)

/-- **SI §S5.5 (a), "idealized infinite limit", quadratic-form version.** If every mass is strictly
positive, the masses are summable, and the scaled atoms `{s ξ_k}` are **dense** in `ℝᵈ`, then for
pairwise distinct points `u_p` and any nonzero real coefficients `a`,
`Σ_{p,q} a_p a_q K_FQ(u_p, u_q) > 0`: "`P_a` is continuous and vanishes on a dense set, so
`P_a ≡ 0`; and distinctness of the `u_p` (linear independence of the characters …) forces `a = 0`".
-/
theorem sum_sum_mul_fqProfileInf_pos {m : σ → ℝ} (hm : ∀ k, 0 < m k) (hsum : Summable m)
    (ξ : σ → Fin d → ℝ) (s : ℝ) (hdense : Dense (Set.range fun k => s • ξ k))
    {ι : Type*} [Fintype ι] {u : ι → Fin d → ℝ} (hu : Function.Injective u) {a : ι → ℝ}
    (ha : a ≠ 0) :
    0 < ∑ p, ∑ q, a p * a q * fqProfileInf m ξ s (u p - u q) := by
  have hm0 : ∀ k, 0 ≤ m k := fun k => (hm k).le
  rw [sum_sum_mul_fqProfileInf hm0 hsum ξ s Finset.univ a u]
  set C : σ → ℝ := fun k => ∑ p, a p * Real.cos (2 * π * s * (ξ k ⬝ᵥ u p)) with hCdef
  set S : σ → ℝ := fun k => ∑ p, a p * Real.sin (2 * π * s * (ξ k ⬝ᵥ u p)) with hSdef
  have hF0 : ∀ k, 0 ≤ m k * (C k ^ 2 + S k ^ 2) := fun k => mul_nonneg (hm0 k) (by positivity)
  have hbound : ∀ k, |C k| ≤ ∑ p, |a p| ∧ |S k| ≤ ∑ p, |a p| := by
    intro k
    constructor
    · refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun p _ => ?_)
      rw [abs_mul]
      exact mul_le_of_le_one_right (abs_nonneg _) (Real.abs_cos_le_one _)
    · refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun p _ => ?_)
      rw [abs_mul]
      exact mul_le_of_le_one_right (abs_nonneg _) (Real.abs_sin_le_one _)
  have hFs : Summable fun k => m k * (C k ^ 2 + S k ^ 2) := by
    refine Summable.of_nonneg_of_le hF0 (fun k => ?_) (hsum.mul_right (2 * (∑ p, |a p|) ^ 2))
    obtain ⟨h1, h2⟩ := hbound k
    have hC2 : C k ^ 2 ≤ (∑ p, |a p|) ^ 2 := by
      rw [← sq_abs]
      exact pow_le_pow_left₀ (abs_nonneg _) h1 2
    have hS2 : S k ^ 2 ≤ (∑ p, |a p|) ^ 2 := by
      rw [← sq_abs]
      exact pow_le_pow_left₀ (abs_nonneg _) h2 2
    exact (mul_le_mul_of_nonneg_left (by linarith) (hm0 k) :
      m k * (C k ^ 2 + S k ^ 2) ≤ m k * (2 * (∑ p, |a p|) ^ 2))
  change 0 < ∑' k, m k * (C k ^ 2 + S k ^ 2)
  by_contra hle
  have hzero : ∀ k, m k * (C k ^ 2 + S k ^ 2) = 0 := by
    have h0 : ∑' k, m k * (C k ^ 2 + S k ^ 2) = 0 := le_antisymm (not_lt.1 hle) (tsum_nonneg hF0)
    have := (hasSum_zero_iff_of_nonneg hF0).1 (h0 ▸ hFs.hasSum)
    exact fun k => congrFun this k
  let P : (Fin d → ℝ) → ℂ := fun η =>
    ∑ p, (a p : ℂ) * Complex.exp (((2 * π * (η ⬝ᵥ u p) : ℝ) : ℂ) * Complex.I)
  have hPc : Continuous P := by
    unfold P
    fun_prop
  have hPz : ∀ k, P (s • ξ k) = 0 := by
    intro k
    have hCS : C k ^ 2 + S k ^ 2 = 0 := (mul_eq_zero.1 (hzero k)).resolve_left (hm k).ne'
    have hC : C k = 0 := by nlinarith [sq_nonneg (C k), sq_nonneg (S k)]
    have hS : S k = 0 := by nlinarith [sq_nonneg (C k), sq_nonneg (S k)]
    have harg : ∀ p, 2 * π * ((s • ξ k) ⬝ᵥ u p) = 2 * π * s * (ξ k ⬝ᵥ u p) := by
      intro p
      rw [smul_dotProduct, smul_eq_mul]
      ring
    apply Complex.ext
    · simp only [P, Complex.re_sum, Complex.re_ofReal_mul, Complex.exp_ofReal_mul_I_re, harg,
        Complex.zero_re]
      exact hC
    · simp only [P, Complex.im_sum, Complex.im_ofReal_mul, Complex.exp_ofReal_mul_I_im, harg,
        Complex.zero_im]
      exact hS
  have hP0 : P = fun _ => 0 :=
    hPc.ext_on hdense continuous_const (by rintro _ ⟨k, rfl⟩; exact hPz k)
  have hli := (linearIndependent_monoidHom (Multiplicative (Fin d → ℝ)) ℂ).comp
    (fun p => fqChar (u p)) (fqChar_injective.comp hu)
  rw [Fintype.linearIndependent_iff] at hli
  have hcoef := hli (fun p => (a p : ℂ)) (by
    funext η
    have := congrFun hP0 η.toAdd
    simpa [P, fqChar] using this)
  apply ha
  funext p
  exact_mod_cast hcoef p

/-- **SI §S5.5 (a), idealized limit, matrix form.** Under strictly positive summable masses and a
dense set of scaled atoms `{s ξ_k}`, the Gram matrix on any finite family of **distinct** points is
positive **definite** ("in that idealization `A` is in fact strictly positive-definite for every
finite set of distinct anchors"). -/
theorem fqGramInf_posDef_of_dense {m : σ → ℝ} (hm : ∀ k, 0 < m k) (hsum : Summable m)
    (ξ : σ → Fin d → ℝ) (s : ℝ) (hdense : Dense (Set.range fun k => s • ξ k))
    {ι : Type*} [Fintype ι] {u : ι → Fin d → ℝ} (hu : Function.Injective u) :
    (fqGramInf m ξ s u).PosDef := by
  refine PosDef.of_dotProduct_mulVec_pos ?_ fun a ha => ?_
  · rw [IsHermitian, conjTranspose_eq_transpose_of_trivial, fqGramInf_transpose]
  · rw [star_trivial, dotProduct_fqGramInf_mulVec]
    exact sum_sum_mul_fqProfileInf_pos hm hsum ξ s hdense hu ha

/-- For an aperiodic generator (`Lℝᵈ ∩ ℤⁿ = {0}`, `ker L = 0`) and a nonzero scale `s`, the scaled
ambient module `s · Lᵀℤⁿ` is dense in `ℝᵈ` (from `FQGNN.Backbone.dense_spectrumModule_iff`). -/
theorem dense_range_smul_ambient {n : ℕ} {L : Matrix (Fin n) (Fin d) ℝ}
    (hL : ¬ FQGNN.Backbone.HasNonzeroPeriod L) {s : ℝ} (hs : s ≠ 0) :
    Dense (Set.range fun k : Fin n → ℤ => s • (Lᵀ *ᵥ fun j => (k j : ℝ))) := by
  have hd := (FQGNN.Backbone.dense_spectrumModule_iff L).2 hL
  have himg : (Set.range fun k : Fin n → ℤ => s • (Lᵀ *ᵥ fun j => (k j : ℝ))) =
      (fun x => s • x) '' (FQGNN.Backbone.spectrumModule L : Set (Fin d → ℝ)) := by
    ext y
    simp only [Set.mem_range, Set.mem_image, SetLike.mem_coe,
      FQGNN.Backbone.mem_spectrumModule_iff]
    constructor
    · rintro ⟨k, rfl⟩
      exact ⟨_, ⟨k, rfl⟩, rfl⟩
    · rintro ⟨_, ⟨k, rfl⟩, rfl⟩
      exact ⟨k, rfl⟩
  rw [himg]
  exact (Function.Surjective.denseRange fun y => ⟨s⁻¹ • y, by simp [smul_smul, hs]⟩).dense_image
    (continuous_const_smul s) hd

/-- **SI §S5.5 (a), idealized infinite limit over the ambient module, with the hypothesis made
explicit.** Let the atoms range over the full module `Lᵀℤⁿ` (index `k ∈ ℤⁿ`, `ξ_k = Lᵀk`) with
strictly positive summable masses, `s ≠ 0`, and let `L` be **aperiodic** (no nonzero `p` with
`Lp ∈ ℤⁿ`, equivalently `Lᵀℤⁿ` dense — `FQGNN.Backbone.dense_spectrumModule_iff`). Then the Gram
matrix on any finite family of distinct anchors is positive definite.

The SI justifies density by "`Lᵀℤⁿ` (dense in `ℝᵈ` for `n > d`)"; `n > d` alone does not suffice
(`fqGramInf_ambient_not_posDef`, `FQGNN.Backbone.grPlusExample_spectrumModule_not_dense`). -/
theorem fqGramInf_posDef_ambient {n : ℕ} {L : Matrix (Fin n) (Fin d) ℝ}
    (hL : ¬ FQGNN.Backbone.HasNonzeroPeriod L) {m : (Fin n → ℤ) → ℝ} (hm : ∀ k, 0 < m k)
    (hsum : Summable m) {s : ℝ} (hs : s ≠ 0) {ι : Type*} [Fintype ι] {u : ι → Fin d → ℝ}
    (hu : Function.Injective u) :
    (fqGramInf m (fun k => Lᵀ *ᵥ fun j => (k j : ℝ)) s u).PosDef :=
  fqGramInf_posDef_of_dense hm hsum _ s (dense_range_smul_ambient hL hs) hu

/-- A period of `Φ_L` is a period of the ambient-module kernel profile: if `Lp ∈ ℤⁿ` then every
term `m_k cos(2π ⟨Lᵀk, p⟩)` equals `m_k`, so `κ(p) = κ(0)` (any masses). -/
theorem fqProfileInf_ambient_period {n : ℕ} {L : Matrix (Fin n) (Fin d) ℝ} {p : Fin d → ℝ}
    (hp : ∀ j, ∃ z : ℤ, (L *ᵥ p) j = z) (m : (Fin n → ℤ) → ℝ) (sgn : ℤ) :
    fqProfileInf m (fun k => Lᵀ *ᵥ fun j => (k j : ℝ)) 1 ((sgn : ℝ) • p) =
      fqProfileInf m (fun k => Lᵀ *ᵥ fun j => (k j : ℝ)) 1 0 := by
  choose z hz using hp
  unfold fqProfileInf
  refine tsum_congr fun k => ?_
  have hint : (Lᵀ *ᵥ fun j => (k j : ℝ)) ⬝ᵥ ((sgn : ℝ) • p) =
      ((sgn * ∑ j, k j * z j : ℤ) : ℝ) := by
    rw [dotProduct_smul, mulVec_transpose, ← dotProduct_mulVec, dotProduct, smul_eq_mul]
    simp only [hz]
    push_cast
    ring
  simp only [fqTerm, hint, dotProduct_zero, mul_zero, Real.cos_zero, mul_one]
  rw [mul_comm (2 * π), Real.cos_int_mul_two_pi, mul_one]

/-- **The density hypothesis cannot be dropped (SI §S5.5 (a)).** If `Φ_L` has a nonzero period `p`
(`Lp ∈ ℤⁿ`), then for **any** masses on the full ambient module the Gram matrix at the two distinct
anchors `0, p` (scale `s = 1`) is **not** positive definite: its rows coincide, and `a = (1, −1)` is
an exact annihilator. Combined with `FQGNN.Backbone.grPlusExample_hasNonzeroPeriod`, this happens
for a totally positive `L ∈ Gr₊(2,3)` with ℚ-independent rows. -/
theorem fqGramInf_ambient_not_posDef {n : ℕ} {L : Matrix (Fin n) (Fin d) ℝ} {p : Fin d → ℝ}
    (hp : ∀ j, ∃ z : ℤ, (L *ᵥ p) j = z) (m : (Fin n → ℤ) → ℝ) :
    ¬ (fqGramInf m (fun k => Lᵀ *ᵥ fun j => (k j : ℝ)) 1 ![0, p]).PosDef := by
  intro hpd
  have hne : (![1, -1] : Fin 2 → ℝ) ≠ 0 := by
    intro h
    have := congrFun h 0
    simp at this
  have hpos := (posDef_iff_dotProduct_mulVec.1 hpd).2 hne
  have h1 := fqProfileInf_ambient_period hp m 1
  have h2 := fqProfileInf_ambient_period hp m (-1)
  simp only [Int.cast_one, one_smul, Int.cast_neg, neg_smul] at h1 h2
  simp [dotProduct, mulVec, Fin.sum_univ_two, h1, h2] at hpos

end StrictPD

end FQGNN.Readout

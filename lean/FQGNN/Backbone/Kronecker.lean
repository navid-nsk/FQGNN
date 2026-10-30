import Mathlib
import FQGNN.Backbone.TorusEmbedding
import FQGNN.Backbone.SpectrumModule

/-!
# Kronecker density of the torus lift `Φ_L` (SI §S1.1, "Dense winding from ℚ-independence")

Sources:
* SI §S1.1: "If the rows of `L` are rationally independent (no nonzero `k ∈ ℤⁿ` has `Lᵀk = 0`),
  the flow `{Φ_L(x)}` is dense in `𝕋ⁿ` and equidistributed (Kronecker–Weyl …)"; the image of
  `Φ_L` is "dense in `𝕋ⁿ` iff no nonzero `k ∈ ℤⁿ` has `Lᵀk = 0`".
* SI §S1: the ambient rank-`n` ℤ-module `Lᵀℤⁿ` "is by contrast topologically dense in `ℝᵈ`
  exactly when `Lℝᵈ ∩ ℤⁿ = {0}`".
* Manuscript eq. (1) (the torus lift `Φ_L(x) = exp(2πi Lx) ∈ 𝕋ⁿ`).

Mathlib has neither Kronecker's theorem nor Pontryagin duality for tori, so the density statement is
proved from scratch, through a duality criterion for subgroups of `ℝⁿ`:

* `exists_line_of_forall_small` — a closed subgroup of a finite-dimensional real space whose nonzero
  elements accumulate at `0` contains a whole line `ℝu` (compactness of the unit sphere);
* `exists_intFunctional_of_discrete_span` — a discrete subgroup spanning the space carries a nonzero
  linear functional with integer values on it (Mathlib's `ZLattice` theory);
* `exists_intFunctional_of_not_dense`, `dense_iff_forall_intFunctional` — **duality criterion**: a
  subgroup `G ⊆ ℝⁿ` is dense iff the only linear functional with `f(G) ⊆ ℤ` is `f = 0`
  (the structure theorem of closed subgroups of `ℝⁿ`, in the form needed here).

Main results (namespace `FQGNN.Backbone`):
* `dense_liftGroup` — `Lℝᵈ + ℤⁿ` is dense in `ℝⁿ` when no nonzero `k ∈ ℤⁿ` has `Lᵀk = 0`;
* `closure_range_torusLift` — **Kronecker density**: under the same hypothesis, the image of the
  torus lift is dense in `𝕋ⁿ`, `closure (range Φ_L) = 𝕋ⁿ`;
* `closure_range_torusLift_eq_torus_iff` — the converse also holds: the image of `Φ_L` is dense in
  `𝕋ⁿ` **iff** the rows of `L` are ℚ-linearly independent;
* `closure_range_line_eq_torus2` — the case `n = 2`, `d = 1`: the line
  `t ↦ (e^{2πiat}, e^{2πibt})` is dense in `𝕋²` when `a/b` is irrational;
* `dense_spectrumModule_iff` — the ambient module `M_L = Lᵀℤⁿ` is dense in `ℝᵈ` **iff** `Φ_L` has no
  nonzero period (`Lℝᵈ ∩ ℤⁿ = {0}`); `grPlusExample_*` — a totally positive `L ∈ Gr₊(2,3)` with
  ℚ-independent rows whose module `M_L` is **not** dense (density of `M_L` requires the
  column-space condition; `n > d`, total positivity and ℚ-independence of the rows do not
  suffice).

Only *density* is formalized; the *equidistribution* (Weyl) half of the SI sentence is not (it
needs Haar measure on `𝕋ⁿ` and ergodic averages, and nothing in the development uses it).
-/

noncomputable section

open Matrix Complex Filter Topology

namespace FQGNN.Backbone

/-! ## 1. The line lemma for closed subgroups -/

/-- **Line lemma.** If a closed additive subgroup `D` of a finite-dimensional real normed space has
nonzero elements of arbitrarily small norm, then it contains a whole line `ℝu`, `u ≠ 0`.

Proof: normalize small nonzero elements `g_n` to the unit sphere, extract a convergent subsequence
`g_n/‖g_n‖ → u`, and approximate `t u` by the integer multiples `⌊t/‖g_n‖⌋ g_n ∈ D`.

Source: standard step of the structure theorem for closed subgroups of `ℝⁿ`, used here for
Kronecker's theorem (SI §S1.1). -/
theorem exists_line_of_forall_small {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [FiniteDimensional ℝ E] {D : AddSubgroup E} (hD : IsClosed (D : Set E))
    (h : ∀ ε > 0, ∃ g ∈ D, g ≠ 0 ∧ ‖g‖ < ε) :
    ∃ u : E, u ≠ 0 ∧ ∀ t : ℝ, t • u ∈ D := by
  choose g hgD hg0 hgε using fun n : ℕ => h (1 / ((n : ℝ) + 1)) (by positivity)
  have hr0 : ∀ n, 0 < ‖g n‖ := fun n => norm_pos_iff.2 (hg0 n)
  have hrlim : Tendsto (fun n => ‖g n‖) atTop (𝓝 0) :=
    squeeze_zero (fun n => (hr0 n).le) (fun n => (hgε n).le)
      tendsto_one_div_add_atTop_nhds_zero_nat
  have hv : ∀ n, ‖g n‖⁻¹ • g n ∈ Metric.sphere (0 : E) 1 := by
    intro n
    rw [mem_sphere_zero_iff_norm, norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ (hr0 n).ne']
  obtain ⟨u, hu, φ, hφ, hlim⟩ := (isCompact_sphere (0 : E) 1).tendsto_subseq hv
  refine ⟨u, ?_, fun t => ?_⟩
  · rintro rfl
    simp at hu
  · have hrφ : Tendsto (fun k => ‖g (φ k)‖) atTop (𝓝 0) := hrlim.comp hφ.tendsto_atTop
    have hc : Tendsto (fun k => (⌊t / ‖g (φ k)‖⌋ : ℝ) * ‖g (φ k)‖) atTop (𝓝 t) := by
      refine tendsto_iff_norm_sub_tendsto_zero.2
        (squeeze_zero (fun _ => norm_nonneg _) (fun k => ?_) hrφ)
      have hr := hr0 (φ k)
      have h1 := Int.floor_le (t / ‖g (φ k)‖)
      have h2 := Int.lt_floor_add_one (t / ‖g (φ k)‖)
      have hx : t / ‖g (φ k)‖ * ‖g (φ k)‖ = t := div_mul_cancel₀ t hr.ne'
      have a1 := mul_le_mul_of_nonneg_right h1 hr.le
      have a2 := mul_lt_mul_of_pos_right h2 hr
      rw [hx] at a1 a2
      rw [Real.norm_eq_abs, abs_le]
      constructor <;> nlinarith
    have hmem : ∀ k, ((⌊t / ‖g (φ k)‖⌋ : ℝ) * ‖g (φ k)‖) • (‖g (φ k)‖⁻¹ • g (φ k)) ∈ D := by
      intro k
      rw [smul_smul, mul_assoc, mul_inv_cancel₀ (hr0 (φ k)).ne', mul_one, Int.cast_smul_eq_zsmul]
      exact D.zsmul_mem (hgD _) _
    exact hD.mem_of_tendsto (hc.smul hlim) (Eventually.of_forall hmem)

/-! ## 2. Discrete spanning subgroups carry an integral functional -/

/-- A discrete additive subgroup `D` (nonzero elements have norm `≥ ε > 0`) that spans a nonzero
finite-dimensional real space `E` carries a nonzero linear functional `f : E → ℝ` with
`f(D) ⊆ ℤ`: `D` is a `ℤ`-lattice, a `ℤ`-basis of `D` is an `ℝ`-basis of `E`, and `f` is one of its
coordinate functionals (Mathlib `IsZLattice`, `Basis.ofZLatticeBasis`).

Source: structure theorem for closed subgroups of `ℝⁿ` (used for SI §S1.1). -/
theorem exists_intFunctional_of_discrete_span {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] [FiniteDimensional ℝ E] [Nontrivial E]
    (D : AddSubgroup E) {ε : ℝ} (hε : 0 < ε) (hiso : ∀ g ∈ D, g ≠ 0 → ε ≤ ‖g‖)
    (hspan : Submodule.span ℝ (D : Set E) = ⊤) :
    ∃ f : E →ₗ[ℝ] ℝ, f ≠ 0 ∧ ∀ x ∈ D, ∃ z : ℤ, f x = z := by
  let L : Submodule ℤ E := AddSubgroup.toIntSubmodule D
  have : DiscreteTopology L := by
    apply discreteTopology_of_isOpen_singleton_zero
    have : ({0} : Set L) = Subtype.val ⁻¹' Metric.ball (0 : E) ε := by
      ext x
      simp only [Set.mem_singleton_iff, Set.mem_preimage, mem_ball_zero_iff]
      constructor
      · rintro rfl
        simpa using hε
      · intro hx
        by_contra hne
        have hx0 : (x : E) ≠ 0 := fun h0 => hne (Subtype.ext h0)
        exact absurd (hiso x x.2 hx0) (not_le.2 hx)
    rw [this]
    exact Metric.isOpen_ball.preimage continuous_subtype_val
  have : IsZLattice ℝ L := ⟨by simpa [L] using hspan⟩
  have := ZLattice.module_free ℝ L
  have := ZLattice.module_finite ℝ L
  let b := Module.Free.chooseBasis ℤ L
  let B := b.ofZLatticeBasis ℝ L
  obtain ⟨i⟩ := B.index_nonempty
  refine ⟨B.coord i, ?_, fun x hx => ⟨b.repr ⟨x, hx⟩ i, ?_⟩⟩
  · intro h
    have := LinearMap.congr_fun h (B i)
    simp at this
  · rw [Module.Basis.coord_apply]
    exact b.ofZLatticeBasis_repr_apply ℝ L ⟨x, hx⟩ i

/-! ## 3. The duality criterion for dense subgroups of `ℝⁿ` -/

/-- **Duality criterion, hard direction.** If an additive subgroup `G ⊆ ℝⁿ` is not dense, there is
a nonzero linear functional `f` with `f(G) ⊆ ℤ`.

Proof: let `C = closure G` and `V₀ = {v | ℝv ⊆ C}` (a proper subspace). Map `ℝⁿ → ℝⁿ/V₀ ≅ ℝᵐ` by `ψ`
with a linear section `σ`; then `D = σ⁻¹(C)` is a closed subgroup containing `ψ(C)` and no line, so
it is discrete (`exists_line_of_forall_small`). If `D` spans `ℝᵐ` it carries an integral functional
(`exists_intFunctional_of_discrete_span`); otherwise take a functional killing `span D`. Compose
with `ψ`.

Source: Pontryagin duality for `ℝⁿ/ℤⁿ` in the elementary form needed for Kronecker's theorem
(SI §S1.1). -/
theorem exists_intFunctional_of_not_dense {n : ℕ} (G : AddSubgroup (Fin n → ℝ))
    (hG : ¬ Dense (G : Set (Fin n → ℝ))) :
    ∃ f : (Fin n → ℝ) →ₗ[ℝ] ℝ, f ≠ 0 ∧ ∀ g ∈ G, ∃ z : ℤ, f g = z := by
  classical
  set C := G.topologicalClosure with hC
  have hCc : IsClosed (C : Set (Fin n → ℝ)) := G.isClosed_topologicalClosure
  have hGC : G ≤ C := G.le_topologicalClosure
  let V0 : Submodule ℝ (Fin n → ℝ) :=
    { carrier := {v | ∀ t : ℝ, t • v ∈ C}
      add_mem' := fun ha hb t => by rw [smul_add]; exact C.add_mem (ha t) (hb t)
      zero_mem' := fun t => by rw [smul_zero]; exact C.zero_mem
      smul_mem' := fun c v hv t => by rw [smul_smul]; exact hv (t * c) }
  have hV0C : ∀ v ∈ V0, v ∈ C := fun v hv => by simpa using hv 1
  have hV0top : V0 ≠ ⊤ := by
    intro htop
    apply hG
    rw [dense_iff_closure_eq, ← AddSubgroup.topologicalClosure_coe, Set.eq_univ_iff_forall]
    intro y
    exact hV0C y (htop ▸ Submodule.mem_top)
  let Q := (Fin n → ℝ) ⧸ V0
  let e : Q ≃ₗ[ℝ] (Fin (Module.finrank ℝ Q) → ℝ) := (Module.finBasis ℝ Q).equivFun
  let ψ : (Fin n → ℝ) →ₗ[ℝ] (Fin (Module.finrank ℝ Q) → ℝ) := e.toLinearMap ∘ₗ V0.mkQ
  have hψker : ∀ y, ψ y = 0 ↔ y ∈ V0 := by
    intro y
    rw [show ψ y = e (V0.mkQ y) from rfl, e.map_eq_zero_iff, Submodule.mkQ_apply,
      Submodule.Quotient.mk_eq_zero]
  have hψsurj : Function.Surjective ψ := e.surjective.comp V0.mkQ_surjective
  obtain ⟨σ, hσ⟩ := ψ.exists_rightInverse_of_surjective (LinearMap.range_eq_top.2 hψsurj)
  have hψσ : ∀ x, ψ (σ x) = x := fun x => LinearMap.congr_fun hσ x
  have hσc : Continuous σ := LinearMap.continuous_of_finiteDimensional σ
  let D : AddSubgroup (Fin (Module.finrank ℝ Q) → ℝ) := C.comap σ.toAddMonoidHom
  have hmemD : ∀ x, x ∈ D ↔ σ x ∈ C := fun x => Iff.rfl
  have hDc : IsClosed (D : Set (Fin (Module.finrank ℝ Q) → ℝ)) := hCc.preimage hσc
  have hψD : ∀ y ∈ C, ψ y ∈ D := by
    intro y hy
    rw [hmemD]
    have h0 : σ (ψ y) - y ∈ V0 := (hψker _).1 (by simp [hψσ])
    have := C.add_mem (hV0C _ h0) hy
    simpa using this
  have hnoline : ∀ u, (∀ t : ℝ, t • u ∈ D) → u = 0 := by
    intro u hu
    have hσu : σ u ∈ V0 := fun t => by
      have := (hmemD _).1 (hu t)
      simpa [map_smul] using this
    rw [← hψσ u]
    exact (hψker _).2 hσu
  obtain ⟨ε, hε, hiso⟩ : ∃ ε > 0, ∀ g ∈ D, g ≠ 0 → ε ≤ ‖g‖ := by
    by_contra hcon
    push Not at hcon
    obtain ⟨u, hu0, hu⟩ := exists_line_of_forall_small hDc hcon
    exact hu0 (hnoline u hu)
  have : Nontrivial Q := Submodule.Quotient.nontrivial_iff.2 hV0top
  have : Nontrivial (Fin (Module.finrank ℝ Q) → ℝ) := e.symm.surjective.nontrivial
  obtain ⟨f, hf0, hfD⟩ : ∃ f : (Fin (Module.finrank ℝ Q) → ℝ) →ₗ[ℝ] ℝ,
      f ≠ 0 ∧ ∀ x ∈ D, ∃ z : ℤ, f x = z := by
    by_cases hspan : Submodule.span ℝ (D : Set (Fin (Module.finrank ℝ Q) → ℝ)) = ⊤
    · exact exists_intFunctional_of_discrete_span D hε hiso hspan
    · obtain ⟨f, hf, hle⟩ := Submodule.exists_le_ker_of_lt_top _ (lt_top_iff_ne_top.2 hspan)
      refine ⟨f, hf, fun x hx => ⟨0, ?_⟩⟩
      have := hle (Submodule.subset_span hx)
      simpa using this
  refine ⟨f ∘ₗ ψ, ?_, fun g hg => hfD _ (hψD g (hGC hg))⟩
  intro h0
  apply hf0
  refine LinearMap.ext fun x => ?_
  have := LinearMap.congr_fun h0 (σ x)
  simpa [hψσ] using this

/-- A real number all of whose real multiples are integers is `0`. -/
lemma eq_zero_of_forall_mul_mem_int {a : ℝ} (h : ∀ t : ℝ, ∃ z : ℤ, t * a = z) : a = 0 := by
  by_contra ha
  obtain ⟨z, hz⟩ := h (1 / (2 * a))
  have hz' : (z : ℝ) = 1 / 2 := by
    rw [← hz]
    field_simp
  have h0 : (0 : ℝ) < z := by rw [hz']; norm_num
  have h1 : (z : ℝ) < 1 := by rw [hz']; norm_num
  have h0' : 0 < z := by exact_mod_cast h0
  have h1' : z < 1 := by exact_mod_cast h1
  omega

/-- **Duality criterion, easy direction.** If a nonzero linear functional takes integer values on
`G ⊆ ℝⁿ`, then `G` is not dense (`G` lies in the closed proper set `f⁻¹(ℤ)`). -/
theorem not_dense_of_intFunctional {n : ℕ} (G : AddSubgroup (Fin n → ℝ))
    (f : (Fin n → ℝ) →ₗ[ℝ] ℝ) (hf : f ≠ 0) (hG : ∀ g ∈ G, ∃ z : ℤ, f g = z) :
    ¬ Dense (G : Set (Fin n → ℝ)) := by
  intro hd
  have hclosed : IsClosed (f ⁻¹' Set.range ((↑) : ℤ → ℝ)) :=
    Int.isClosedEmbedding_coe_real.isClosed_range.preimage
      (LinearMap.continuous_of_finiteDimensional f)
  have hsub : (G : Set (Fin n → ℝ)) ⊆ f ⁻¹' Set.range ((↑) : ℤ → ℝ) := by
    intro g hg
    obtain ⟨z, hz⟩ := hG g hg
    exact ⟨z, hz.symm⟩
  have hall : ∀ x, ∃ z : ℤ, f x = z := by
    intro x
    have hx : x ∈ closure (G : Set (Fin n → ℝ)) := hd x
    obtain ⟨z, hz⟩ := closure_minimal hsub hclosed hx
    exact ⟨z, hz.symm⟩
  apply hf
  refine LinearMap.ext fun x => ?_
  refine eq_zero_of_forall_mul_mem_int fun t => ?_
  obtain ⟨z, hz⟩ := hall (t • x)
  exact ⟨z, by rw [← hz, map_smul, smul_eq_mul]⟩

/-- **Duality criterion (both directions).** A subgroup `G ⊆ ℝⁿ` is dense iff the only linear
functional with integer values on `G` is the zero functional. -/
theorem dense_iff_forall_intFunctional {n : ℕ} (G : AddSubgroup (Fin n → ℝ)) :
    Dense (G : Set (Fin n → ℝ)) ↔
      ∀ f : (Fin n → ℝ) →ₗ[ℝ] ℝ, (∀ g ∈ G, ∃ z : ℤ, f g = z) → f = 0 := by
  constructor
  · intro hd f hf
    by_contra hf0
    exact not_dense_of_intFunctional G f hf0 hf hd
  · intro h
    by_contra hnd
    obtain ⟨f, hf0, hf⟩ := exists_intFunctional_of_not_dense G hnd
    exact hf0 (h f hf)

/-! ## 4. Kronecker: `Lℝᵈ + ℤⁿ` is dense in `ℝⁿ` -/

variable {n d : ℕ}

/-- The subgroup `Lℝᵈ + ℤⁿ ⊆ ℝⁿ`: the preimage of the image of `Φ_L` under the covering map
`ℝⁿ → 𝕋ⁿ`. Source: SI §S1.1 (the linear flow `x ↦ Φ_L(x)` on `𝕋ⁿ = ℝⁿ/ℤⁿ`). -/
def liftGroup (L : Matrix (Fin n) (Fin d) ℝ) : AddSubgroup (Fin n → ℝ) where
  carrier := {y | ∃ x : Fin d → ℝ, ∃ k : Fin n → ℤ, y = L *ᵥ x + fun j => (k j : ℝ)}
  add_mem' := by
    rintro _ _ ⟨x, k, rfl⟩ ⟨x', k', rfl⟩
    refine ⟨x + x', k + k', ?_⟩
    funext j
    simp only [mulVec_add, Pi.add_apply, Int.cast_add]
    ring
  zero_mem' := ⟨0, 0, by funext j; simp⟩
  neg_mem' := by
    rintro _ ⟨x, k, rfl⟩
    refine ⟨-x, -k, ?_⟩
    funext j
    simp only [mulVec_neg, Pi.neg_apply, Pi.add_apply, Int.cast_neg]
    ring

/-- A linear functional on `ℝⁿ` is `f(y) = Σ_j y_j f(e_j)`. -/
lemma linearFunctional_apply_eq_sum (f : (Fin n → ℝ) →ₗ[ℝ] ℝ) (y : Fin n → ℝ) :
    f y = ∑ j, y j * f (Pi.single j 1) := by
  rw [LinearMap.pi_apply_eq_sum_univ f y]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [smul_eq_mul]
  congr 2
  funext i
  simp [Pi.single_apply, eq_comm]

/-- **Kronecker density in `ℝⁿ` (SI §S1.1).** If no nonzero `k ∈ ℤⁿ` has `Lᵀk = 0`, the
subgroup `Lℝᵈ + ℤⁿ` is dense in `ℝⁿ`.

Proof: otherwise the duality criterion gives `f ≠ 0` with `f(Lℝᵈ + ℤⁿ) ⊆ ℤ`; integrality on the
subspace `Lℝᵈ` forces `f = 0` there, and `f = ⟨k, ·⟩` with `k_j = f(e_j) ∈ ℤ`, so `Lᵀk = 0`, `k = 0`
and `f = 0`. -/
theorem dense_liftGroup {L : Matrix (Fin n) (Fin d) ℝ}
    (hL : ∀ k : Fin n → ℤ, Lᵀ *ᵥ (fun j => (k j : ℝ)) = 0 → k = 0) :
    Dense (liftGroup L : Set (Fin n → ℝ)) := by
  rw [dense_iff_forall_intFunctional]
  intro f hf
  have hint : ∀ j, ∃ z : ℤ, f (Pi.single j 1) = z := fun j =>
    hf _ ⟨0, Pi.single j 1, by funext i; simp [Pi.single_apply]⟩
  choose k hk using hint
  have hcol : ∀ x, f (L *ᵥ x) = 0 := by
    intro x
    refine eq_zero_of_forall_mul_mem_int fun t => ?_
    obtain ⟨z, hz⟩ := hf (L *ᵥ (t • x)) ⟨t • x, 0, by funext i; simp⟩
    exact ⟨z, by rw [← hz, mulVec_smul, map_smul, smul_eq_mul]⟩
  have hk0 : k = 0 := by
    apply hL
    have hdot : ∀ x, (Lᵀ *ᵥ (fun j => (k j : ℝ))) ⬝ᵥ x = 0 := by
      intro x
      have h := hcol x
      rw [linearFunctional_apply_eq_sum] at h
      simp only [hk] at h
      rw [mulVec_transpose, ← dotProduct_mulVec, ← h, dotProduct]
      exact Finset.sum_congr rfl fun j _ => mul_comm _ _
    have := hdot (Lᵀ *ᵥ (fun j => (k j : ℝ)))
    exact dotProduct_self_eq_zero.1 this
  refine LinearMap.ext fun y => ?_
  rw [linearFunctional_apply_eq_sum]
  simp [hk, hk0]

/-! ## 5. Kronecker density of the torus lift -/

/-- The covering map `ℝⁿ → 𝕋ⁿ`, `y ↦ (e^{2πi y_j})_j`. -/
def expLift (n : ℕ) (y : Fin n → ℝ) : Fin n → ℂ :=
  fun j => cexp (2 * Real.pi * I * (y j : ℂ))

lemma continuous_expLift (n : ℕ) : Continuous (expLift n) := by
  unfold expLift
  fun_prop

/-- `Φ_L(x) = E(Lx)` with `E` the covering map (manuscript eq. (1)). -/
lemma torusLift_eq_expLift (L : Matrix (Fin n) (Fin d) ℝ) (x : Fin d → ℝ) :
    torusLift L x = expLift n (L *ᵥ x) := rfl

/-- The covering map is `ℤⁿ`-periodic. -/
lemma expLift_add_int (y : Fin n → ℝ) (k : Fin n → ℤ) :
    expLift n (y + fun j => (k j : ℝ)) = expLift n y := by
  funext j
  have hk : cexp (2 * Real.pi * I * ((k j : ℝ) : ℂ)) = 1 :=
    (cexp_two_pi_I_mul_eq_one_iff _).2 ⟨k j, rfl⟩
  simp only [expLift, Pi.add_apply, ofReal_add, mul_add, Complex.exp_add, hk, mul_one]

/-- The covering map is onto the torus: every `z ∈ 𝕋ⁿ` is `E(y)` with `y_j = arg(z_j)/(2π)`. -/
lemma exists_expLift_eq {z : Fin n → ℂ} (hz : z ∈ torus n) : ∃ y, expLift n y = z := by
  refine ⟨fun j => arg (z j) / (2 * Real.pi), funext fun j => ?_⟩
  have h := norm_mul_exp_arg_mul_I (z j)
  rw [hz j, ofReal_one, one_mul] at h
  have hπ : (Real.pi : ℂ) ≠ 0 := ofReal_ne_zero.2 Real.pi_ne_zero
  simp only [expLift]
  rw [show (2 * (Real.pi : ℂ) * I * ((arg (z j) / (2 * Real.pi) : ℝ) : ℂ)) = (arg (z j) : ℂ) * I by
    push_cast
    field_simp]
  exact h

/-- The torus `𝕋ⁿ` is closed in `ℂⁿ`. -/
lemma isClosed_torus (n : ℕ) : IsClosed (torus n) := by
  have : torus n = ⋂ j, {z : Fin n → ℂ | ‖z j‖ = 1} := by
    ext z
    simp [torus]
  rw [this]
  exact isClosed_iInter fun j => isClosed_eq (by fun_prop) continuous_const

/-- The image of `Φ_L` lies in the torus, hence so does its closure. -/
lemma closure_range_torusLift_subset (L : Matrix (Fin n) (Fin d) ℝ) :
    closure (Set.range (torusLift L)) ⊆ torus n :=
  closure_minimal (Set.range_subset_iff.2 (torusLift_mem_torus L)) (isClosed_torus n)

/-- **Kronecker's theorem for the torus lift (SI §S1.1, "Dense winding from
ℚ-independence").** If no nonzero `k ∈ ℤⁿ` has `Lᵀk = 0`, the image of `Φ_L : ℝᵈ → 𝕋ⁿ` is dense
in the torus: `closure (range Φ_L) = 𝕋ⁿ`.

Source: SI §S1.1 ("the flow `{Φ_L(x)}` is dense in `𝕋ⁿ` and equidistributed (Kronecker–Weyl …)";
only density is formalized); manuscript eq. (1). -/
theorem closure_range_torusLift {L : Matrix (Fin n) (Fin d) ℝ}
    (hL : ∀ k : Fin n → ℤ, Lᵀ *ᵥ (fun j => (k j : ℝ)) = 0 → k = 0) :
    closure (Set.range (torusLift L)) = torus n := by
  refine (closure_range_torusLift_subset L).antisymm fun z hz => ?_
  obtain ⟨y, rfl⟩ := exists_expLift_eq hz
  have himg : expLift n '' (liftGroup L : Set (Fin n → ℝ)) ⊆ Set.range (torusLift L) := by
    rintro _ ⟨_, ⟨x, k, rfl⟩, rfl⟩
    exact ⟨x, by rw [torusLift_eq_expLift, expLift_add_int]⟩
  have hy : y ∈ closure (liftGroup L : Set (Fin n → ℝ)) := dense_liftGroup hL y
  exact closure_mono himg (image_closure_subset_closure_image (continuous_expLift n) ⟨y, hy, rfl⟩)

/-- **Kronecker density, ℚ-independence form.** If the rows of `L` are ℚ-linearly independent,
`closure (range Φ_L) = 𝕋ⁿ` (SI §S1.1, hypothesis "rows of `L` rationally independent"). -/
theorem closure_range_torusLift_of_linearIndependent {L : Matrix (Fin n) (Fin d) ℝ}
    (hL : LinearIndependent ℚ L.row) : closure (Set.range (torusLift L)) = torus n :=
  closure_range_torusLift ((linearIndependent_rat_iff L).1 hL)

/-- **Kronecker density, approximation form.** If the rows of `L` are ℚ-linearly independent, every
point `z` of the torus is approximated to any accuracy by the flow: `∃ x, ‖Φ_L(x) − z‖ < ε`
(SI §S1.1). -/
theorem exists_dist_torusLift_lt {L : Matrix (Fin n) (Fin d) ℝ} (hL : LinearIndependent ℚ L.row)
    {z : Fin n → ℂ} (hz : z ∈ torus n) {ε : ℝ} (hε : 0 < ε) :
    ∃ x : Fin d → ℝ, dist (torusLift L x) z < ε := by
  rw [← closure_range_torusLift_of_linearIndependent hL] at hz
  obtain ⟨_, ⟨x, rfl⟩, hx⟩ := Metric.mem_closure_iff.1 hz ε hε
  exact ⟨x, by rwa [dist_comm]⟩

/-- The character `χ_k(z) = Π_j z_j^{k_j}` evaluated on the covering map:
`χ_k(E(y)) = exp(2πi ⟨k, y⟩)`. -/
lemma prod_expLift_zpow (k : Fin n → ℤ) (y : Fin n → ℝ) :
    ∏ j, expLift n y j ^ k j =
      cexp (2 * Real.pi * I * ((∑ j, (k j : ℝ) * y j : ℝ) : ℂ)) := by
  simp only [expLift, ← Complex.exp_int_mul, ← Complex.exp_sum]
  congr 1
  push_cast
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  ring

/-- **Converse of Kronecker's theorem.** If the image of `Φ_L` is dense in `𝕋ⁿ`, the rows of `L`
are ℚ-linearly independent: an integer relation `Lᵀk = 0`, `k ≠ 0`, confines the image to the closed
proper subset `{χ_k = 1}` of the torus. -/
theorem linearIndependent_of_closure_range_torusLift {L : Matrix (Fin n) (Fin d) ℝ}
    (h : closure (Set.range (torusLift L)) = torus n) : LinearIndependent ℚ L.row := by
  rw [linearIndependent_rat_iff]
  intro k hk
  by_contra hk0
  obtain ⟨j0, hj0⟩ : ∃ j0, k j0 ≠ 0 := by
    by_contra hall
    push Not at hall
    exact hk0 (funext hall)
  let χ : (Fin n → ℂ) → ℂ := fun z => ∏ j, z j ^ k j
  have hχc : ContinuousOn χ (torus n) := by
    refine continuousOn_finsetProd _ fun j _ => ?_
    refine ((continuous_apply j).continuousOn).zpow₀ _ fun z hz => Or.inl ?_
    intro h0
    have := hz j
    rw [h0, norm_zero] at this
    exact zero_ne_one this
  have hclosed : IsClosed (torus n ∩ χ ⁻¹' {1}) :=
    hχc.preimage_isClosed_of_isClosed (isClosed_torus n) isClosed_singleton
  have hrange : Set.range (torusLift L) ⊆ torus n ∩ χ ⁻¹' {1} := by
    rintro _ ⟨x, rfl⟩
    refine ⟨torusLift_mem_torus L x, ?_⟩
    simp only [Set.mem_preimage, Set.mem_singleton_iff, χ, torusLift_eq_expLift,
      prod_expLift_zpow]
    have hsum : ∑ j, (k j : ℝ) * (L *ᵥ x) j = 0 := by
      have h1 : ∑ j, (k j : ℝ) * (L *ᵥ x) j = (fun j => (k j : ℝ)) ⬝ᵥ (L *ᵥ x) := rfl
      rw [h1, dotProduct_mulVec, ← mulVec_transpose, hk, zero_dotProduct]
    rw [hsum, ofReal_zero, mul_zero, Complex.exp_zero]
  -- the point `E(e_{j0}/(2 k_{j0}))` of the torus has `χ_k = -1`
  let y : Fin n → ℝ := Pi.single j0 (1 / (2 * (k j0 : ℝ)))
  have hy : expLift n y ∈ torus n := fun j => by
    simp only [expLift]
    rw [show (2 * Real.pi * I * (y j : ℂ)) = ((2 * Real.pi * y j : ℝ) : ℂ) * I by push_cast; ring]
    exact Complex.norm_exp_ofReal_mul_I _
  have hmem : expLift n y ∈ torus n ∩ χ ⁻¹' {1} := by
    rw [← h] at hy
    exact closure_minimal hrange hclosed hy
  have hval : χ (expLift n y) = -1 := by
    simp only [χ, prod_expLift_zpow]
    have hsum : ∑ j, (k j : ℝ) * y j = 1 / 2 := by
      rw [Finset.sum_eq_single j0]
      · simp only [y, Pi.single_eq_same]
        have : (k j0 : ℝ) ≠ 0 := by exact_mod_cast hj0
        field_simp
      · intro j _ hj
        simp [y, hj]
      · intro h0
        exact absurd (Finset.mem_univ j0) h0
    rw [hsum]
    have : (2 * Real.pi * I * ((1 / 2 : ℝ) : ℂ)) = Real.pi * I := by push_cast; ring
    rw [this, Complex.exp_pi_mul_I]
  have h1 := hmem.2
  simp only [Set.mem_preimage, Set.mem_singleton_iff] at h1
  rw [hval] at h1
  norm_num at h1

/-- **Kronecker's theorem, exact form.** The image of `Φ_L` is dense in `𝕋ⁿ` **iff** the rows of `L`
are ℚ-linearly independent (equivalently, no nonzero `k ∈ ℤⁿ` has `Lᵀk = 0`).

Source: SI §S1.1 ("dense in `𝕋ⁿ` iff no nonzero `k ∈ ℤⁿ` has `Lᵀk = 0`"). -/
theorem closure_range_torusLift_eq_torus_iff (L : Matrix (Fin n) (Fin d) ℝ) :
    closure (Set.range (torusLift L)) = torus n ↔ LinearIndependent ℚ L.row :=
  ⟨linearIndependent_of_closure_range_torusLift, closure_range_torusLift_of_linearIndependent⟩

/-! ## 6. The case `n = 2`, `d = 1` -/

/-- The scalar generator `(a, b)ᵀ` as a `2 × 1` matrix. -/
def colGen2 (a b : ℝ) : Matrix (Fin 2) (Fin 1) ℝ := Matrix.of fun i _ => ![a, b] i

/-- For `a/b` irrational there is no nonzero integer relation `k₀ a + k₁ b = 0`. -/
lemma colGen2_no_relation {a b : ℝ} (h : Irrational (a / b)) (k : Fin 2 → ℤ)
    (hk : (colGen2 a b)ᵀ *ᵥ (fun j => (k j : ℝ)) = 0) : k = 0 := by
  have hb : b ≠ 0 := by
    rintro rfl
    exact h ⟨0, by simp⟩
  have hrel : (k 0 : ℝ) * a + (k 1 : ℝ) * b = 0 := by
    have := congrFun hk 0
    simpa [colGen2, mulVec, dotProduct, Fin.sum_univ_two, mul_comm] using this
  have hk0 : k 0 = 0 := by
    by_contra hk0
    apply h
    refine ⟨-((k 1 : ℚ) / (k 0 : ℚ)), ?_⟩
    have hk0' : (k 0 : ℝ) ≠ 0 := by exact_mod_cast hk0
    push_cast
    field_simp
    linarith
  have hk1 : k 1 = 0 := by
    rw [hk0, Int.cast_zero, zero_mul, zero_add] at hrel
    have := (mul_eq_zero.1 hrel).resolve_right hb
    exact_mod_cast this
  funext j
  fin_cases j
  · exact hk0
  · exact hk1

/-- **Kronecker, `n = 2`, `d = 1` (the irrational winding line).** If `a/b` is irrational, the line
`t ↦ (e^{2πiat}, e^{2πibt})` is dense in the torus `𝕋²`.

Source: SI §S1.1 in its simplest instance; a corollary of the general theorem (and also of the
density of `ℤa + ℤb` in `ℝ`, `dense_addSubgroupClosure_pair_iff`). -/
theorem closure_range_line_eq_torus2 {a b : ℝ} (h : Irrational (a / b)) :
    closure (Set.range fun t : ℝ =>
      ![cexp (2 * Real.pi * I * ((a * t : ℝ) : ℂ)), cexp (2 * Real.pi * I * ((b * t : ℝ) : ℂ))]) =
      torus 2 := by
  have hset : (Set.range fun t : ℝ =>
      ![cexp (2 * Real.pi * I * ((a * t : ℝ) : ℂ)), cexp (2 * Real.pi * I * ((b * t : ℝ) : ℂ))]) =
      Set.range (torusLift (colGen2 a b)) := by
    ext z
    constructor
    · rintro ⟨t, rfl⟩
      refine ⟨fun _ => t, funext fun j => ?_⟩
      fin_cases j <;> simp [torusLift, colGen2, mulVec, dotProduct]
    · rintro ⟨x, rfl⟩
      refine ⟨x 0, funext fun j => ?_⟩
      have hx : x = fun _ => x 0 := funext fun i => by fin_cases i; rfl
      rw [hx]
      fin_cases j <;> simp [torusLift, colGen2, mulVec, dotProduct]
  rw [hset]
  exact closure_range_torusLift (colGen2_no_relation h)

/-! ## 7. Density of the ambient spectrum module `M_L = Lᵀℤⁿ` -/

/-- **Density of the ambient module ⇔ aperiodicity.** `M_L = Lᵀℤⁿ` is dense in `ℝᵈ` iff `Φ_L` has no
nonzero period, i.e. iff `Lℝᵈ ∩ ℤⁿ = {0}` (with `ker L = 0`); cf. `hasNonzeroPeriod_iff`.

Proof: a nonzero period `p` (`Lp ∈ ℤⁿ`) gives the integral functional `ξ ↦ ⟨p, ξ⟩` on `M_L`
(`⟨p, Lᵀk⟩ = ⟨Lp, k⟩ ∈ ℤ`); conversely an integral functional `f = ⟨w, ·⟩ ≠ 0` on `M_L` has
`(Lw)_j = f(ℓ_j) ∈ ℤ` on the rows `ℓ_j ∈ M_L`, so `w` is a nonzero period.

Source: SI §S1 ("The ambient rank-`n` ℤ-module `Lᵀℤⁿ` … is by contrast topologically dense in
`ℝᵈ` exactly when `Lℝᵈ ∩ ℤⁿ = {0}`"). -/
theorem dense_spectrumModule_iff (L : Matrix (Fin n) (Fin d) ℝ) :
    Dense (spectrumModule L : Set (Fin d → ℝ)) ↔ ¬ HasNonzeroPeriod L := by
  rw [dense_iff_forall_intFunctional, hasNonzeroPeriod_iff]
  constructor
  · rintro h ⟨p, hp0, hp⟩
    let f : (Fin d → ℝ) →ₗ[ℝ] ℝ :=
      { toFun := fun ξ => p ⬝ᵥ ξ
        map_add' := fun x y => dotProduct_add p x y
        map_smul' := fun c x => by simp [dotProduct_smul] }
    have hf : f = 0 := by
      apply h
      intro ξ hξ
      obtain ⟨k, rfl⟩ := (mem_spectrumModule_iff L ξ).1 hξ
      choose m hm using hp
      refine ⟨∑ j, m j * k j, ?_⟩
      change p ⬝ᵥ (Lᵀ *ᵥ fun j => (k j : ℝ)) = _
      rw [dotProduct_mulVec, vecMul_transpose, dotProduct_comm, dotProduct]
      push_cast
      exact Finset.sum_congr rfl fun j _ => by rw [hm j, mul_comm]
    apply hp0
    have := LinearMap.congr_fun hf p
    exact dotProduct_self_eq_zero.1 this
  · intro h f hf
    by_contra hf0
    apply h
    let w : Fin d → ℝ := fun i => f (Pi.single i 1)
    refine ⟨w, ?_, fun j => ?_⟩
    · intro hw
      apply hf0
      refine LinearMap.ext fun y => ?_
      rw [linearFunctional_apply_eq_sum]
      simp [show ∀ i, f (Pi.single i 1) = 0 from fun i => congrFun hw i]
    · obtain ⟨z, hz⟩ := hf (L.row j) (row_mem_spectrumModule L j)
      refine ⟨z, ?_⟩
      rw [← hz, linearFunctional_apply_eq_sum]
      rfl

/-- A totally positive generator with ℚ-independent rows: `L = [[3, 1], [2, 1 + √2], [1, 3]]`. -/
def grPlusExample : Matrix (Fin 3) (Fin 2) ℝ := !![3, 1; 2, 1 + Real.sqrt 2; 1, 3]

/-- `grPlusExample ∈ Gr₊(2,3)`: its minors are `1 + 3√2`, `8` and `5 − √2`, all positive. -/
theorem grPlusExample_inGrPlus2 : InGrPlus2 grPlusExample := by
  have hs : (0 : ℝ) ≤ Real.sqrt 2 := Real.sqrt_nonneg 2
  have hs2 : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  intro i j hij
  fin_cases i <;> fin_cases j <;> simp [minor2, grPlusExample] at hij ⊢ <;> nlinarith

/-- The rows of `grPlusExample` are ℚ-linearly independent (a rank-3 module `M_L`). -/
theorem grPlusExample_rows_linearIndependent : LinearIndependent ℚ grPlusExample.row := by
  rw [linearIndependent_rat_iff]
  intro k hk
  have h0 := congrFun hk 0
  have h1 := congrFun hk 1
  simp [grPlusExample, mulVec, dotProduct, Fin.sum_univ_three] at h0 h1
  -- h0 : 3 k₀ + 2 k₁ + k₂ = 0;  h1 : k₀ + (1 + √2) k₁ + 3 k₂ = 0
  have hk1 : k 1 = 0 := by
    by_contra hk1
    have hk1' : (k 1 : ℝ) ≠ 0 := by exact_mod_cast hk1
    apply irrational_sqrt_two
    refine ⟨-((k 0 : ℚ) + k 1 + 3 * k 2) / k 1, ?_⟩
    push_cast
    field_simp
    linarith
  rw [hk1] at h0 h1
  push_cast at h0 h1
  have hk2 : k 2 = 0 := by
    have : (k 2 : ℝ) = 0 := by linarith
    exact_mod_cast this
  have hk0 : k 0 = 0 := by
    have : (k 0 : ℝ) = 0 := by rw [hk2] at h0; push_cast at h0; linarith
    exact_mod_cast this
  funext j
  fin_cases j
  · exact hk0
  · exact hk1
  · exact hk2

/-- `grPlusExample` has the nonzero period `p = (1, 0)`: `Lp = (3, 2, 1) ∈ ℤ³`. -/
theorem grPlusExample_hasNonzeroPeriod : HasNonzeroPeriod grPlusExample := by
  rw [hasNonzeroPeriod_iff]
  refine ⟨![1, 0], by simp, fun j => ?_⟩
  fin_cases j
  · exact ⟨3, by simp [grPlusExample, mulVec, dotProduct]⟩
  · exact ⟨2, by simp [grPlusExample, mulVec, dotProduct]⟩
  · exact ⟨1, by simp [grPlusExample, mulVec, dotProduct]⟩

/-- **The ambient module need not be dense, even for `L ∈ Gr₊(2,3)` with ℚ-independent rows
(`n = 3 > d = 2`).** For `grPlusExample`, `M_L = Lᵀℤ³` is **not** dense in `ℝ²`.

Density holds exactly under the column-space condition `Lℝᵈ ∩ ℤⁿ = {0}`
(`dense_spectrumModule_iff`), which is not implied by `n > d`, total positivity or ℚ-independence
of the rows (SI §S1: "neither `n > d`, nor positivity of the minors, nor ℚ-independence of the
rows suffices"). -/
theorem grPlusExample_spectrumModule_not_dense :
    InGrPlus2 grPlusExample ∧ LinearIndependent ℚ grPlusExample.row ∧
      ¬ Dense (spectrumModule grPlusExample : Set (Fin 2 → ℝ)) :=
  ⟨grPlusExample_inGrPlus2, grPlusExample_rows_linearIndependent,
    fun hd => (dense_spectrumModule_iff grPlusExample).1 hd grPlusExample_hasNonzeroPeriod⟩

end FQGNN.Backbone

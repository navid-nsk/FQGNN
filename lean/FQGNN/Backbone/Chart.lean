import Mathlib

/-!
# The deployed `d = 2` totally-positive chart (manuscript eq. (10), Proposition 2)

Formalizes the generator side of Proposition 2 ("Always-in-family training, deployed `d = 2`
chart") of the manuscript (Prop. 2 and Methods eq. (10)), as proved in Supplementary Information
(SI) §S2.2–S2.4.

Contents:
* `softplus` and its properties (SI §S2.2(a));
* the normalized cumulative-softplus angles (SI eq. (S2.1)) and `0 < θ₁ < … < θₙ < π/2`;
* the closed-form minor (SI eq. (S2.2)) and Lemma S2.1 (= Prop. 2(i), membership);
* Prop. S2.5(ii), invariance under arbitrary discrete-time updates;
* Prop. S2.5(iii), quantitative interiority with the exact constant of SI eq. (S2.3);
* Lemma S2.2, the chart reaches every point of `Gr₊(2, n)` (`n ≥ 2`).

Indexing conventions. The source indexes rows `i = 1, …, n` and angle weights `l = 0, …, n`.
Here rows are `i : Fin n` (source row `i + 1`), and the `n + 1` angle parameters are
`α : Fin (n + 1) → ℝ`, with `α 0` the source `α₀` (the weight `g₀` present only in the
denominator) and `α l.succ` (for `l : Fin n`) the source `α_{l+1}`. The chart parameters
`φ = (ρ, α, log s) ∈ ℝⁿ × ℝⁿ⁺¹ × ℝ` (`N = 2n + 2` real numbers) form the normed space
`ChartParams n`, whose norm is the sup norm `‖φ‖_∞` used in SI eq. (S2.3).
-/

open Real Finset

namespace FQGNN.Backbone

/-! ## Softplus -/

/-- `softplus t = log (1 + eᵗ)` (SI §S2.2(a), `σ` there; manuscript eq. (10)). -/
noncomputable def softplus (t : ℝ) : ℝ := Real.log (1 + Real.exp t)

/-- `softplus t > 0` for every real `t` (SI §S2.2(a)). -/
theorem softplus_pos (t : ℝ) : 0 < softplus t :=
  Real.log_pos (by linarith [Real.exp_pos t])

/-- `softplus` is strictly increasing (SI §S2.2(a)). -/
theorem softplus_strictMono : StrictMono softplus := by
  intro a b hab
  unfold softplus
  exact Real.log_lt_log (by linarith [Real.exp_pos a]) (by linarith [Real.exp_lt_exp.mpr hab])

/-- `softplus` is monotone. -/
theorem softplus_le_softplus {a b : ℝ} (h : a ≤ b) : softplus a ≤ softplus b :=
  softplus_strictMono.monotone h

/-- The inverse of `softplus` on `(0, ∞)`: `y ↦ log (eʸ − 1)`. -/
noncomputable def softplusInv (y : ℝ) : ℝ := Real.log (Real.exp y - 1)

/-- `softplusInv` is a right inverse of `softplus` on `(0, ∞)` (SI §S2.2(a)). -/
theorem softplus_softplusInv {y : ℝ} (hy : 0 < y) : softplus (softplusInv y) = y := by
  have h1 : 0 < Real.exp y - 1 := by linarith [Real.add_one_lt_exp hy.ne']
  unfold softplus softplusInv
  rw [Real.exp_log h1, add_sub_cancel, Real.log_exp]

/-- **Surjectivity** (SI §S2.2(a): "a smooth strictly increasing bijection
`ℝ → (0, ∞)`"): the range of `softplus` is exactly `(0, ∞)`. -/
theorem range_softplus : Set.range softplus = Set.Ioi 0 := by
  ext y
  constructor
  · rintro ⟨t, rfl⟩
    exact softplus_pos t
  · intro hy
    exact ⟨softplusInv y, softplus_softplusInv hy⟩

/-! ## Normalized cumulative-softplus angles (SI eq. (S2.1)) -/

variable {n : ℕ}

/-- Partial weight `Σ_{l=1}^{i} g_l` of SI eq. (S2.1), `g_l = softplus α_l`
(row `i : Fin n` is source row `i + 1`). -/
noncomputable def partialWeight (α : Fin (n + 1) → ℝ) (i : Fin n) : ℝ :=
  ∑ l ∈ Finset.Iic i, softplus (α l.succ)

/-- Total weight `Σ_{l=0}^{n} g_l` of SI eq. (S2.1). -/
noncomputable def totalWeight (α : Fin (n + 1) → ℝ) : ℝ :=
  ∑ l, softplus (α l)

/-- The normalized cumulative-softplus angle of SI eq. (S2.1):
`θ_i = (π/2) · (Σ_{l=1}^{i} g_l) / (Σ_{l=0}^{n} g_l)`. -/
noncomputable def chartAngle (α : Fin (n + 1) → ℝ) (i : Fin n) : ℝ :=
  π / 2 * (partialWeight α i / totalWeight α)

theorem totalWeight_eq (α : Fin (n + 1) → ℝ) :
    totalWeight α = softplus (α 0) + ∑ l : Fin n, softplus (α l.succ) := by
  unfold totalWeight
  rw [Fin.sum_univ_succ]

theorem totalWeight_pos (α : Fin (n + 1) → ℝ) : 0 < totalWeight α :=
  Finset.sum_pos (fun l _ => softplus_pos (α l)) Finset.univ_nonempty

theorem partialWeight_pos (α : Fin (n + 1) → ℝ) (i : Fin n) : 0 < partialWeight α i :=
  Finset.sum_pos (fun _ _ => softplus_pos _) ⟨i, Finset.mem_Iic.mpr le_rfl⟩

theorem partialWeight_lt_totalWeight (α : Fin (n + 1) → ℝ) (i : Fin n) :
    partialWeight α i < totalWeight α := by
  rw [totalWeight_eq]
  have h : partialWeight α i ≤ ∑ l : Fin n, softplus (α l.succ) :=
    Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _)
      (fun _ _ _ => (softplus_pos _).le)
  linarith [softplus_pos (α 0)]

/-- The increment of the partial weights between rows `i < j` contains the weight `g_j`. -/
theorem partialWeight_add_le (α : Fin (n + 1) → ℝ) {i j : Fin n} (hij : i < j) :
    partialWeight α i + softplus (α j.succ) ≤ partialWeight α j := by
  unfold partialWeight
  have hj : j ∉ Finset.Iic i := by simpa using hij
  rw [add_comm, ← Finset.sum_insert (f := fun l => softplus (α l.succ)) hj]
  refine Finset.sum_le_sum_of_subset_of_nonneg ?_ (fun _ _ _ => (softplus_pos _).le)
  intro l hl
  rcases Finset.mem_insert.mp hl with rfl | hl
  · exact Finset.mem_Iic.mpr le_rfl
  · exact Finset.mem_Iic.mpr ((Finset.mem_Iic.mp hl).trans hij.le)

theorem partialWeight_strictMono (α : Fin (n + 1) → ℝ) : StrictMono (partialWeight α) :=
  fun _ j hij => by linarith [partialWeight_add_le α hij, softplus_pos (α j.succ)]

/-- **Lower end** (SI eq. (S2.1)): `0 < θ_i` for every row (in particular `0 < θ₁`). -/
theorem chartAngle_pos (α : Fin (n + 1) → ℝ) (i : Fin n) : 0 < chartAngle α i :=
  mul_pos (by positivity) (div_pos (partialWeight_pos α i) (totalWeight_pos α))

/-- **Upper end** (SI eq. (S2.1)): `θ_i < π/2` for every row (in particular `θₙ < π/2`):
`g₀ > 0` is present in the denominator but absent from every numerator. -/
theorem chartAngle_lt_pi_div_two (α : Fin (n + 1) → ℝ) (i : Fin n) :
    chartAngle α i < π / 2 := by
  have h : partialWeight α i / totalWeight α < 1 :=
    (div_lt_one (totalWeight_pos α)).mpr (partialWeight_lt_totalWeight α i)
  unfold chartAngle
  nlinarith [Real.pi_pos]

/-- **Strict ordering** (SI eq. (S2.1)): `θ₁ < θ₂ < … < θₙ`. -/
theorem chartAngle_strictMono (α : Fin (n + 1) → ℝ) : StrictMono (chartAngle α) := by
  intro i j hij
  unfold chartAngle
  have := div_lt_div_of_pos_right (partialWeight_strictMono α hij) (totalWeight_pos α)
  nlinarith [Real.pi_pos]

/-- The angles lie in the open ordered chamber (SI §S2.2(b))
`Θₙ = {0 < θ₁ < … < θₙ < π/2}`. -/
theorem chartAngle_mem_chamber (α : Fin (n + 1) → ℝ) :
    StrictMono (chartAngle α) ∧ (∀ i, 0 < chartAngle α i) ∧ ∀ i, chartAngle α i < π / 2 :=
  ⟨chartAngle_strictMono α, chartAngle_pos α, chartAngle_lt_pi_div_two α⟩

/-- Gap bound used in SI Prop. S2.5(iii): for `i < j`,
`θ_j − θ_i ≥ (π/2) · g_j / Σ_l g_l`. -/
theorem chartAngle_sub_ge (α : Fin (n + 1) → ℝ) {i j : Fin n} (hij : i < j) :
    π / 2 * (softplus (α j.succ) / totalWeight α) ≤ chartAngle α j - chartAngle α i := by
  unfold chartAngle
  rw [← mul_sub, ← sub_div]
  have hT := totalWeight_pos α
  gcongr
  linarith [partialWeight_add_le α hij]

/-! ## Polar rows, the closed-form minor (S2.2) and Lemma S2.1 -/

/-- The `d = 2` polar rows of manuscript eq. (10): `L_i = s (r_i cos θ_i, r_i sin θ_i)`. -/
noncomputable def polarRows (s : ℝ) (r θ : Fin n → ℝ) : Matrix (Fin n) (Fin 2) ℝ :=
  Matrix.of fun i => ![s * (r i * Real.cos (θ i)), s * (r i * Real.sin (θ i))]

/-- The `2 × 2` (Plücker) minor `Δ_{ij}` of an `n × 2` matrix on rows `i, j`. -/
def minor2 (L : Matrix (Fin n) (Fin 2) ℝ) (i j : Fin n) : ℝ :=
  L i 0 * L j 1 - L i 1 * L j 0

/-- `Δ_{ij}` is the determinant of the `2 × 2` submatrix on rows `(i, j)` (SI §S2 notation,
`Δ_I` the maximal minor on rows `I`). -/
theorem minor2_eq_det (L : Matrix (Fin n) (Fin 2) ℝ) (i j : Fin n) :
    minor2 L i j = (L.submatrix ![i, j] id).det := by
  rw [Matrix.det_fin_two]
  simp [minor2]

/-- Membership in `Gr₊(2, n)` for a representative `L` (SI §S2, `d = 2`): every `2 × 2` minor
`Δ_{ij}`, `i < j`, is strictly positive. -/
def InGrPlus2 (L : Matrix (Fin n) (Fin 2) ℝ) : Prop :=
  ∀ i j : Fin n, i < j → 0 < minor2 L i j

/-- **Closed form — SI eq. (S2.2)** (manuscript:
"`Δ_{ij} = s² r_i r_j sin(θ_j − θ_i)`"): for polar rows,
`Δ_{ij} = s² r_i r_j sin(θ_j − θ_i)`. No hypotheses are needed. -/
theorem minor2_polarRows (s : ℝ) (r θ : Fin n → ℝ) (i j : Fin n) :
    minor2 (polarRows s r θ) i j = s ^ 2 * r i * r j * Real.sin (θ j - θ i) := by
  simp only [minor2, polarRows, Matrix.of_apply, Matrix.cons_val_zero, Matrix.cons_val_one,
    Real.sin_sub]
  ring

/-- **Positivity — SI Lemma S2.1 / manuscript Prop. 2(i)**: if `s > 0`, every `r_i > 0`,
and `0 < θ₁ < … < θₙ < π/2`, then every minor `Δ_{ij}` (`i < j`) is strictly positive, i.e.
the polar-row matrix represents a point of `Gr₊(2, n)`. -/
theorem polarRows_inGrPlus2 {s : ℝ} {r θ : Fin n → ℝ} (hs : 0 < s) (hr : ∀ i, 0 < r i)
    (hθ : StrictMono θ) (h0 : ∀ i, 0 < θ i) (h1 : ∀ i, θ i < π / 2) :
    InGrPlus2 (polarRows s r θ) := by
  intro i j hij
  rw [minor2_polarRows]
  have hsin : 0 < Real.sin (θ j - θ i) := by
    apply Real.sin_pos_of_pos_of_lt_pi
    · linarith [hθ hij]
    · linarith [h0 i, h1 j, Real.pi_pos]
  have := hr i
  have := hr j
  positivity

/-! ## The chart `C : ℝ^{2n+2} → Gr₊(2,n)` -/

/-- Chart parameters `φ = (ρ, α, log s) ∈ ℝⁿ × ℝⁿ⁺¹ × ℝ` (SI §S2.1 statement of Prop. 2,
`N = 2n + 2`). As a product of sup-normed spaces it carries the sup norm `‖φ‖_∞`. -/
abbrev ChartParams (n : ℕ) := (Fin n → ℝ) × (Fin (n + 1) → ℝ) × ℝ

/-- Radii `r_i = softplus ρ_i` (manuscript eq. (10)). -/
noncomputable def chartRadius (φ : ChartParams n) (i : Fin n) : ℝ := softplus (φ.1 i)

/-- Scale `s = exp (log s)`. -/
noncomputable def chartScale (φ : ChartParams n) : ℝ := Real.exp φ.2.2

/-- The chart map `C(φ) = L(φ)`, rows `L_i = s (r_i cos θ_i, r_i sin θ_i)` (manuscript eq. (10),
SI Lemma S2.1, angles by SI eq. (S2.1)). -/
noncomputable def chartL (φ : ChartParams n) : Matrix (Fin n) (Fin 2) ℝ :=
  polarRows (chartScale φ) (chartRadius φ) (chartAngle φ.2.1)

/-- **Closed-form minor of the chart — SI eq. (S2.2)**:
`Δ_{ij}(L(φ)) = s² r_i r_j sin(θ_j − θ_i)`. -/
theorem minor2_chartL (φ : ChartParams n) (i j : Fin n) :
    minor2 (chartL φ) i j = chartScale φ ^ 2 * chartRadius φ i * chartRadius φ j *
      Real.sin (chartAngle φ.2.1 j - chartAngle φ.2.1 i) :=
  minor2_polarRows _ _ _ i j

/-- **SI Lemma S2.1 (membership) / manuscript Prop. 2(i) / Prop. S2.5(i), first half**:
for **every** parameter value `φ = (ρ, α, log s)`, the chart generator `L(φ)` has every `2 × 2`
minor strictly positive, `L(φ) ∈ Gr₊(2, n)`. -/
theorem chartL_inGrPlus2 (φ : ChartParams n) : InGrPlus2 (chartL φ) :=
  polarRows_inGrPlus2 (Real.exp_pos _) (fun _ => softplus_pos _) (chartAngle_strictMono _)
    (chartAngle_pos _) (chartAngle_lt_pi_div_two _)

/-! ## Prop. S2.5(ii): invariance under arbitrary discrete-time updates -/

/-- The parameter trajectory `φ_{t+1} = T_t(φ_t)` of an arbitrary (time-dependent) update rule
`T : ℕ → ℝᴺ → ℝᴺ` started from `φ₀` (SI Prop. S2.5(ii)). -/
def updateSeq (T : ℕ → ChartParams n → ChartParams n) (φ₀ : ChartParams n) :
    ℕ → ChartParams n
  | 0 => φ₀
  | t + 1 => T t (updateSeq T φ₀ t)

/-- **SI Prop. S2.5(ii) (invariance under arbitrary discrete-time updates)** / manuscript
Prop. 2 ("every gradient step keeps the generator `L` in `Gr₊(2,n)`"): for **any** sequence of
update maps `T_t : ℝᴺ → ℝᴺ` and any start `φ₀`, the iterates satisfy `C(φ_t) ∈ Gr₊(2,n)` for
every `t`. Nothing about descent, step size or smoothness is used. -/
theorem chartL_updateSeq_inGrPlus2 (T : ℕ → ChartParams n → ChartParams n)
    (φ₀ : ChartParams n) (t : ℕ) : InGrPlus2 (chartL (updateSeq T φ₀ t)) :=
  chartL_inGrPlus2 _

/-- **SI Prop. S2.5(ii), history-dependent form**: for an **arbitrary** parameter trajectory
`φ : ℕ → ℝᴺ` (which covers every history-dependent update rule), `C(φ_t) ∈ Gr₊(2,n)` for all
`t`. -/
theorem chartL_traj_inGrPlus2 (φ : ℕ → ChartParams n) (t : ℕ) : InGrPlus2 (chartL (φ t)) :=
  chartL_inGrPlus2 _

/-- **SI Prop. S2.5(ii), gradient-descent instance**: for any "gradient" field
`G : ℕ → ℝᴺ → ℝᴺ` (e.g. `∇_φ 𝓛`, or a momentum/adaptive direction) and step sizes `η_t`,
the iterates `φ_{t+1} = φ_t − η_t G_t(φ_t)` keep `C(φ_t) ∈ Gr₊(2,n)` at every step. -/
theorem chartL_gradientDescent_inGrPlus2 (G : ℕ → ChartParams n → ChartParams n)
    (η : ℕ → ℝ) (φ₀ : ChartParams n) (t : ℕ) :
    InGrPlus2 (chartL (updateSeq (fun k φ => φ - η k • G k φ) φ₀ t)) :=
  chartL_updateSeq_inGrPlus2 _ φ₀ t

/-! ## Prop. S2.5(iii): quantitative interiority, exact constant of (S2.3) -/

/-- The angle-gap constant `δ_B = (π/2) · softplus(−B) / ((n + 1) softplus(B))` of SI
eq. (S2.3). -/
noncomputable def deltaB (n : ℕ) (B : ℝ) : ℝ :=
  π / 2 * (softplus (-B) / ((n + 1) * softplus B))

/-- The interiority constant `m(B) = e^{−2B} softplus(−B)² sin δ_B` of SI eq. (S2.3). -/
noncomputable def interiorityConst (n : ℕ) (B : ℝ) : ℝ :=
  Real.exp (-(2 * B)) * softplus (-B) ^ 2 * Real.sin (deltaB n B)

theorem deltaB_pos (n : ℕ) (B : ℝ) : 0 < deltaB n B := by
  unfold deltaB
  have := softplus_pos (-B)
  have := softplus_pos B
  positivity

theorem deltaB_le (n : ℕ) {B : ℝ} (hB : 0 ≤ B) : deltaB n B ≤ π / 2 := by
  unfold deltaB
  have h1 := softplus_pos (-B)
  have h2 := softplus_pos B
  have h3 : softplus (-B) ≤ softplus B := softplus_le_softplus (by linarith)
  have h4 : softplus (-B) / ((n + 1) * softplus B) ≤ 1 := by
    rw [div_le_one (by positivity)]
    have : (1 : ℝ) ≤ n + 1 := by linarith [(Nat.cast_nonneg n : (0 : ℝ) ≤ n)]
    nlinarith
  nlinarith [Real.pi_pos]

/-- **SI Prop. S2.5(iii), positivity of the constant**: `m(B) > 0` for every `B ≥ 0`. -/
theorem interiorityConst_pos (n : ℕ) {B : ℝ} (hB : 0 ≤ B) : 0 < interiorityConst n B := by
  unfold interiorityConst
  have hsin : 0 < Real.sin (deltaB n B) :=
    Real.sin_pos_of_pos_of_lt_pi (deltaB_pos n B) (by linarith [deltaB_le n hB, Real.pi_pos])
  have := softplus_pos (-B)
  positivity

/-- **SI Prop. S2.5(iii), eq. (S2.3) (quantitative interiority), exact constant**: for
every `B` and every parameter `φ` with `‖φ‖_∞ ≤ B`, every minor satisfies
`Δ_{ij}(L(φ)) ≥ m(B) = e^{−2B} softplus(−B)² sin δ_B`,
`δ_B = (π/2) softplus(−B)/((n+1) softplus B)`.
(For `B < 0` the hypothesis is vacuous; `m(B) > 0` for `B ≥ 0` is `interiorityConst_pos`.) -/
theorem minor2_chartL_ge (φ : ChartParams n) {B : ℝ} (hφ : ‖φ‖ ≤ B) {i j : Fin n}
    (hij : i < j) : interiorityConst n B ≤ minor2 (chartL φ) i j := by
  -- coordinatewise bounds `|x| ≤ B` from the sup norm
  have hρ : ∀ k, |φ.1 k| ≤ B := fun k => by
    have := (norm_le_pi_norm φ.1 k).trans ((norm_fst_le φ).trans hφ)
    rwa [Real.norm_eq_abs] at this
  have hα : ∀ k, |φ.2.1 k| ≤ B := fun k => by
    have := (norm_le_pi_norm φ.2.1 k).trans
      ((norm_fst_le φ.2).trans ((norm_snd_le φ).trans hφ))
    rwa [Real.norm_eq_abs] at this
  have hs' : |φ.2.2| ≤ B := by
    have := (norm_snd_le φ.2).trans ((norm_snd_le φ).trans hφ)
    rwa [Real.norm_eq_abs] at this
  have hB : 0 ≤ B := (abs_nonneg _).trans hs'
  -- scale: `s² ≥ e^{−2B}`
  have hscale : Real.exp (-(2 * B)) ≤ chartScale φ ^ 2 := by
    unfold chartScale
    rw [← Real.exp_nat_mul]
    apply Real.exp_le_exp.mpr
    have := (abs_le.mp hs').1
    push_cast
    linarith
  -- radii: `r_k ≥ softplus(−B)`
  have hrad : ∀ k, softplus (-B) ≤ chartRadius φ k := fun k =>
    softplus_le_softplus (abs_le.mp (hρ k)).1
  -- angle gap: `θ_j − θ_i ≥ δ_B`
  have hT : totalWeight φ.2.1 ≤ (n + 1) * softplus B := by
    unfold totalWeight
    calc ∑ l, softplus (φ.2.1 l) ≤ ∑ _l : Fin (n + 1), softplus B :=
          Finset.sum_le_sum fun l _ => softplus_le_softplus (abs_le.mp (hα l)).2
      _ = (n + 1) * softplus B := by simp
  have hgap : deltaB n B ≤ chartAngle φ.2.1 j - chartAngle φ.2.1 i := by
    refine le_trans ?_ (chartAngle_sub_ge φ.2.1 hij)
    unfold deltaB
    have hTpos := totalWeight_pos φ.2.1
    have hgj : softplus (-B) ≤ softplus (φ.2.1 j.succ) :=
      softplus_le_softplus (abs_le.mp (hα j.succ)).1
    have := softplus_pos (-B)
    have := softplus_pos (φ.2.1 j.succ)
    gcongr
  have hgap' : chartAngle φ.2.1 j - chartAngle φ.2.1 i ≤ π / 2 := by
    linarith [chartAngle_pos φ.2.1 i, chartAngle_lt_pi_div_two φ.2.1 j]
  have hsin : Real.sin (deltaB n B) ≤ Real.sin (chartAngle φ.2.1 j - chartAngle φ.2.1 i) :=
    Real.sin_le_sin_of_le_of_le_pi_div_two
      (by linarith [deltaB_pos n B, Real.pi_pos]) hgap' hgap
  have hsin0 : 0 ≤ Real.sin (deltaB n B) :=
    (Real.sin_pos_of_pos_of_lt_pi (deltaB_pos n B)
      (by linarith [deltaB_le n hB, Real.pi_pos])).le
  have hsp := softplus_pos (-B)
  rw [minor2_chartL]
  unfold interiorityConst
  calc Real.exp (-(2 * B)) * softplus (-B) ^ 2 * Real.sin (deltaB n B)
      = Real.exp (-(2 * B)) * (softplus (-B) * softplus (-B)) * Real.sin (deltaB n B) := by
        ring
    _ ≤ chartScale φ ^ 2 * (chartRadius φ i * chartRadius φ j) *
          Real.sin (chartAngle φ.2.1 j - chartAngle φ.2.1 i) := by
        have hri : 0 < chartRadius φ i := softplus_pos _
        have hrj : 0 < chartRadius φ j := softplus_pos _
        have hr2 : softplus (-B) * softplus (-B) ≤ chartRadius φ i * chartRadius φ j :=
          mul_le_mul (hrad i) (hrad j) hsp.le hri.le
        have h12 : Real.exp (-(2 * B)) * (softplus (-B) * softplus (-B)) ≤
            chartScale φ ^ 2 * (chartRadius φ i * chartRadius φ j) :=
          mul_le_mul hscale hr2 (by positivity) (by positivity)
        exact mul_le_mul h12 hsin hsin0 (by positivity)
    _ = chartScale φ ^ 2 * chartRadius φ i * chartRadius φ j *
          Real.sin (chartAngle φ.2.1 j - chartAngle φ.2.1 i) := by ring

/-- **SI Prop. S2.5(iii), "consequently" clause** (the boundary lies only at parameter infinity):
if along a parameter sequence `φ_k` some minor `Δ_{ij}(L(φ_k))` (`i < j`) tends to `0`, then
`‖φ_k‖_∞` is unbounded. -/
theorem chartParams_unbounded_of_minor2_tendsto_zero (φ : ℕ → ChartParams n) {i j : Fin n}
    (hij : i < j)
    (h : Filter.Tendsto (fun k => minor2 (chartL (φ k)) i j) Filter.atTop (nhds 0)) :
    ¬ ∃ B, ∀ k, ‖φ k‖ ≤ B := by
  rintro ⟨B, hB⟩
  have hB0 : 0 ≤ B := (norm_nonneg _).trans (hB 0)
  have hm := interiorityConst_pos n hB0
  have hle : interiorityConst n B ≤ 0 :=
    ge_of_tendsto h (Filter.Eventually.of_forall fun k => minor2_chartL_ge (φ k) (hB k) hij)
  linarith

/-! ## Lemma S2.2: the chart reaches every point of `Gr₊(2, n)` -/

/-- Minors transform by `det g` under a change of representative `L ↦ L g`
(SI §S2 preamble: "`Δ_I ↦ det(g) Δ_I`"). -/
theorem minor2_mul (L : Matrix (Fin n) (Fin 2) ℝ) (g : Matrix (Fin 2) (Fin 2) ℝ) (i j : Fin n) :
    minor2 (L * g) i j = g.det * minor2 L i j := by
  simp only [minor2, Matrix.mul_apply, Fin.sum_univ_two, Matrix.det_fin_two]
  ring

/-- Extension of an angle sequence by its two chamber walls: `Θ 0 = 0`, `Θ (k+1) = θ_k` for
`k < n`, and `Θ (k+1) = π/2` for `k ≥ n` (the consecutive gaps `d_i` of SI Lemma S2.2 / §S2.2(b)
are `Θ (k+1) − Θ k`). -/
noncomputable def angleExt (θ : Fin n → ℝ) : ℕ → ℝ
  | 0 => 0
  | k + 1 => if h : k < n then θ ⟨k, h⟩ else π / 2

/-- The `n + 1` consecutive gaps of a chamber point are strictly positive (SI §S2.2(b)). -/
theorem angleExt_gap_pos {θ : Fin n → ℝ} (hθ : StrictMono θ) (h0 : ∀ i, 0 < θ i)
    (h1 : ∀ i, θ i < π / 2) {k : ℕ} (hk : k ≤ n) :
    0 < angleExt θ (k + 1) - angleExt θ k := by
  rcases k with _ | k
  · simp only [angleExt]
    split_ifs with h
    · linarith [h0 ⟨0, h⟩]
    · linarith [Real.pi_pos]
  · have hk' : k < n := by omega
    simp only [angleExt, hk', ↓reduceDIte]
    split_ifs with h
    · have hlt : (⟨k, hk'⟩ : Fin n) < ⟨k + 1, h⟩ := Fin.mk_lt_mk.mpr (Nat.lt_succ_self k)
      linarith [hθ hlt]
    · linarith [h1 ⟨k, hk'⟩]

/-- Angle parameters realizing a prescribed chamber point (SI §S2.2(b): set
`(g₁, …, gₙ, g₀) = (d₁, …, dₙ, d_{n+1})` and `α_l = softplus⁻¹ g_l`). -/
noncomputable def anglesToParams (θ : Fin n → ℝ) : Fin (n + 1) → ℝ :=
  Fin.cases (softplusInv (angleExt θ (n + 1) - angleExt θ n))
    (fun l => softplusInv (angleExt θ (l + 1) - angleExt θ l))

theorem sum_Iic_eq_sum_range (f : ℕ → ℝ) (i : Fin n) :
    ∑ l ∈ Finset.Iic i, f l = ∑ k ∈ Finset.range (i + 1), f k := by
  rw [Nat.range_succ_eq_Iic, ← Fin.map_valEmbedding_Iic, Finset.sum_map]
  rfl

/-- **SI §S2.2(b), surjectivity of (S2.1) onto the chamber `Θₙ`**: every
`0 < θ₁ < … < θₙ < π/2` is attained by the normalized cumulative-softplus map. -/
theorem chartAngle_anglesToParams {θ : Fin n → ℝ} (hθ : StrictMono θ) (h0 : ∀ i, 0 < θ i)
    (h1 : ∀ i, θ i < π / 2) : chartAngle (anglesToParams θ) = θ := by
  have hg : ∀ l : Fin n, softplus (anglesToParams θ l.succ) =
      angleExt θ (l + 1) - angleExt θ l := fun l => by
    simp only [anglesToParams, Fin.cases_succ]
    exact softplus_softplusInv (angleExt_gap_pos hθ h0 h1 l.isLt.le)
  have hg0 : softplus (anglesToParams θ 0) = angleExt θ (n + 1) - angleExt θ n := by
    simp only [anglesToParams, Fin.cases_zero]
    exact softplus_softplusInv (angleExt_gap_pos hθ h0 h1 le_rfl)
  have htot : totalWeight (anglesToParams θ) = π / 2 := by
    rw [totalWeight_eq, hg0]
    simp_rw [hg]
    rw [Fin.sum_univ_eq_sum_range (fun k => angleExt θ (k + 1) - angleExt θ k),
      Finset.sum_range_sub]
    simp [angleExt]
  funext i
  have hp : partialWeight (anglesToParams θ) i = θ i := by
    unfold partialWeight
    simp_rw [hg]
    rw [sum_Iic_eq_sum_range (fun k => angleExt θ (k + 1) - angleExt θ k),
      Finset.sum_range_sub]
    simp [angleExt, i.isLt]
  unfold chartAngle
  rw [hp, htot]
  field_simp

/-- Polar form of a vector in the open first quadrant (Step 3 of SI Lemma S2.2). -/
theorem exists_polar_firstQuadrant {x y : ℝ} (hx : 0 < x) (hy : 0 < y) :
    ∃ r θ : ℝ, 0 < r ∧ 0 < θ ∧ θ < π / 2 ∧ r * Real.cos θ = x ∧ r * Real.sin θ = y := by
  refine ⟨‖(⟨x, y⟩ : ℂ)‖, Complex.arg ⟨x, y⟩, ?_, ?_, ?_, Complex.norm_mul_cos_arg _,
    Complex.norm_mul_sin_arg _⟩
  · refine norm_pos_iff.mpr fun h => ?_
    have := congrArg Complex.re h
    simp only [Complex.zero_re] at this
    linarith
  · have h1 : 0 ≤ Complex.arg ⟨x, y⟩ := Complex.arg_nonneg_iff.mpr hy.le
    have h2 : Complex.arg ⟨x, y⟩ ≠ 0 := fun h => by
      have := (Complex.arg_eq_zero_iff.mp h).2
      simp only at this
      linarith
    exact lt_of_le_of_ne h1 (Ne.symm h2)
  · exact Complex.arg_lt_pi_div_two_iff.mpr (Or.inl hx)

/-- **SI Lemma S2.2 (surjectivity; "reaches every point") / Prop. S2.5(i), second half**:
for `n ≥ 2`, every representative `L'` of a point of `Gr₊(2, n)` (all `2 × 2` minors `> 0`) is,
up to an orientation-preserving change of representative `L' ↦ L' g` (`det g > 0`, so
`[L' g] = [L']`), exactly a chart matrix `L(φ)`. The hypothesis `n ≥ 2` is implicit in the source
(`Gr(2, n)` needs `n ≥ 2`). The proof follows Steps 2–3 of the SI, with the explicit
compressing frame `v ↦ (Δ(v, L'ₙ) + ½Δ(L'₁, v), Δ(L'₁, v) + ½Δ(v, L'ₙ))`, `det g = ¾ Δ₁ₙ`,
in place of the angle-lifting Step 1. -/
theorem chartL_surjective (hn : 2 ≤ n) (L' : Matrix (Fin n) (Fin 2) ℝ) (hL : InGrPlus2 L') :
    ∃ φ : ChartParams n, ∃ g : Matrix (Fin 2) (Fin 2) ℝ, 0 < g.det ∧ chartL φ = L' * g := by
  set a : Fin n := ⟨0, by omega⟩
  set b : Fin n := ⟨n - 1, by omega⟩ with hb_def
  have hab : a < b := Fin.mk_lt_mk.mpr (by omega)
  set g : Matrix (Fin 2) (Fin 2) ℝ :=
    !![L' b 1 - L' a 1 / 2, -L' a 1 + L' b 1 / 2; -L' b 0 + L' a 0 / 2, L' a 0 - L' b 0 / 2]
    with hg_def
  have hgdet : g.det = 3 / 4 * minor2 L' a b := by
    rw [hg_def, Matrix.det_fin_two_of, minor2]
    ring
  have hgpos : 0 < g.det := by
    rw [hgdet]
    linarith [hL a b hab]
  have hM0 : ∀ i, (L' * g) i 0 = minor2 L' i b + minor2 L' a i / 2 := fun i => by
    simp [hg_def, Matrix.mul_apply, Fin.sum_univ_two, minor2]
    ring
  have hM1 : ∀ i, (L' * g) i 1 = minor2 L' a i + minor2 L' i b / 2 := fun i => by
    simp [hg_def, Matrix.mul_apply, Fin.sum_univ_two, minor2]
    ring
  have hA : ∀ i, 0 ≤ minor2 L' a i := fun i => by
    rcases eq_or_lt_of_le (show a ≤ i from Fin.le_def.mpr (Nat.zero_le _))
      with h | h
    · rw [← h, minor2]
      ring_nf
      exact le_rfl
    · exact (hL a i h).le
  have hB : ∀ i, 0 ≤ minor2 L' i b := fun i => by
    rcases eq_or_lt_of_le (show i ≤ b from Fin.le_def.mpr (by simp [hb_def]; omega))
      with h | h
    · rw [h, minor2]
      ring_nf
      exact le_rfl
    · exact (hL i b h).le
  have hAB : ∀ i, 0 < minor2 L' a i + minor2 L' i b := fun i => by
    rcases eq_or_lt_of_le (show a ≤ i from Fin.le_def.mpr (Nat.zero_le _))
      with h | h
    · rw [← h]
      linarith [hA a, hL a b hab]
    · linarith [hL a i h, hB i]
  have hx : ∀ i, 0 < (L' * g) i 0 := fun i => by
    rw [hM0]
    linarith [hA i, hB i, hAB i]
  have hy : ∀ i, 0 < (L' * g) i 1 := fun i => by
    rw [hM1]
    linarith [hA i, hB i, hAB i]
  choose r θ hr hθ0 hθ1 hcos hsin using fun i => exists_polar_firstQuadrant (hx i) (hy i)
  have hpolar : L' * g = polarRows 1 r θ := by
    ext i k
    fin_cases k
    · simp [polarRows, hcos]
    · simp [polarRows, hsin]
  have hθmono : StrictMono θ := by
    intro i j hij
    have h := hL i j hij
    have hm : 0 < minor2 (L' * g) i j := by
      rw [minor2_mul]
      exact mul_pos hgpos h
    rw [hpolar, minor2_polarRows] at hm
    by_contra hlt
    have hle := not_lt.mp hlt
    have hs : Real.sin (θ j - θ i) ≤ 0 :=
      Real.sin_nonpos_of_nonpos_of_neg_pi_le (by linarith)
        (by linarith [hθ0 j, hθ1 i, Real.pi_pos])
    have hri := hr i
    have hrj := hr j
    have : (1 : ℝ) ^ 2 * r i * r j * Real.sin (θ j - θ i) ≤ 0 :=
      mul_nonpos_of_nonneg_of_nonpos (by positivity) hs
    linarith
  refine ⟨(fun i => softplusInv (r i), anglesToParams θ, 0), g, hgpos, ?_⟩
  rw [hpolar]
  unfold chartL chartScale chartRadius
  rw [Real.exp_zero, chartAngle_anglesToParams hθmono hθ0 hθ1]
  congr 1
  funext i
  exact softplus_softplusInv (hr i)

/-- **SI Prop. S2.5(i) (membership and completeness), `n ≥ 2`**: the chart image is contained in
`Gr₊(2, n)` (Lemma S2.1) and meets every point of it up to an orientation-preserving change of
representative (Lemma S2.2). -/
theorem chart_image_eq_GrPlus2 (hn : 2 ≤ n) (L' : Matrix (Fin n) (Fin 2) ℝ) :
    InGrPlus2 L' ↔
      ∃ φ : ChartParams n, ∃ g : Matrix (Fin 2) (Fin 2) ℝ, 0 < g.det ∧ chartL φ = L' * g := by
  constructor
  · exact chartL_surjective hn L'
  · rintro ⟨φ, g, hg, hφ⟩ i j hij
    have h := chartL_inGrPlus2 φ i j hij
    rw [hφ, minor2_mul] at h
    exact pos_of_mul_pos_right h hg.le

end FQGNN.Backbone

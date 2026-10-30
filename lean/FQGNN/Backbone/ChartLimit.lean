import FQGNN.Backbone.Chart

/-!
# The boundary of `Gr₊(2,n)` lies at parameter infinity: the limit form of SI Prop. S2.5(iii)

SI Prop. S2.5(iii) proves the quantitative interiority bound (S2.3) and then says:
"Consequently `min_I Δ_I → 0` along a parameter sequence forces `‖φ‖_∞ → ∞`: a radius
`ρ_i → −∞`, the scale `log s → −∞`, or an angle-gap collapse `θ_j − θ_i → 0⁺`, which by (S2.1)
requires some `α_l → −∞` or some `α_m → +∞`."

`FQGNN.Backbone.Chart` proves that `‖φ_k‖_∞` is unbounded
(`chartParams_unbounded_of_minor2_tendsto_zero`). This module proves, for a sequence along which a
fixed minor tends to `0`:

* `tendsto_norm_atTop_of_minor2_tendsto_zero` — the full limit `‖φ_k‖_∞ → ∞`.
* `minor2_chartL_ge_of_oneSided` — a sharpened (S2.3): the same constant `m(B)` bounds the
  minors as soon as `ρ_i ≥ −B`, `log s ≥ −B` and `|α_l| ≤ B`; **no upper bound on `ρ` or
  `log s` is needed**.
* `chart_defect_eventually` — hence, eventually along the sequence, for any threshold `B`, some
  `ρ_i < −B`, or `log s < −B`, or some `α_l < −B`, or some `α_l > B`.
* `chart_coordinate_case_split` — the coordinate case split, in subsequence form:
  one fixed coordinate is unbounded in the stated direction along the sequence (that is, it
  tends to `∓∞` along a subsequence): some `ρ_i` is unbounded below, or `log s` is unbounded
  below, or some `α_l` is unbounded below, or some `α_m` is unbounded above.

Along the whole sequence a single coordinate need not diverge, because a sequence can alternate
between two degeneration mechanisms: setting `ρ₀ = −k` for even `k` and `log s = −k` for odd `k`
(all other coordinates fixed) gives `Δ₀₁ → 0`, yet no coordinate tends to `−∞`. The case split
therefore holds along a subsequence, which is the form proved here.

Sources: SI Prop. S2.5(iii), eq. (S2.3) and the "Consequently" clause; manuscript Prop. 2.
-/

noncomputable section

open Real Filter Topology

namespace FQGNN.Backbone

variable {n : ℕ}

/-- **SI Prop. S2.5(iii), the limit form**: if along a
parameter sequence `φ_k` some minor `Δ_{ij}(L(φ_k))` (`i < j`) tends to `0`, then
`‖φ_k‖_∞ → ∞`.

Proof: for every `B ≥ 0`, `m(B) > 0` and eventually `Δ_{ij}(L(φ_k)) < m(B)`; by (S2.3)
(`minor2_chartL_ge`) such `φ_k` has `‖φ_k‖_∞ > B`.

Source: SI Prop. S2.5(iii), "Consequently `min_I Δ_I → 0` … forces `‖φ‖_∞ → ∞`". -/
theorem tendsto_norm_atTop_of_minor2_tendsto_zero (φ : ℕ → ChartParams n) {i j : Fin n}
    (hij : i < j) (h : Tendsto (fun k => minor2 (chartL (φ k)) i j) atTop (𝓝 0)) :
    Tendsto (fun k => ‖φ k‖) atTop atTop := by
  rw [tendsto_atTop]
  intro B
  set B' := max B 0 with hB'
  have hB'0 : 0 ≤ B' := le_max_right _ _
  have hm := interiorityConst_pos n hB'0
  have hev : ∀ᶠ k in atTop, minor2 (chartL (φ k)) i j < interiorityConst n B' :=
    h (Iio_mem_nhds hm)
  filter_upwards [hev] with k hk
  by_contra hlt
  push Not at hlt
  have hle : ‖φ k‖ ≤ B' := hlt.le.trans (le_max_left _ _)
  exact (minor2_chartL_ge (φ k) hle hij).not_gt hk

/-- **Sharpened (S2.3): only one-sided bounds on `ρ` and `log s` are needed.** If every
radius parameter `ρ_i ≥ −B`, the scale parameter `log s ≥ −B`, and every angle parameter
`|α_l| ≤ B` (which forces `B ≥ 0`), then every minor `Δ_{ij}(L(φ))` (`i < j`) is at least
`m(B)`.

(Large `ρ_i` or large `log s` only enlarge the minors `s² r_i r_j sin(θ_j − θ_i)`; this is what
lets the case split of the source mention only `ρ_i → −∞` and `log s → −∞`.)

Source: SI Prop. S2.5(iii), eq. (S2.3) and the "Consequently" clause; strengthens
`minor2_chartL_ge`. -/
theorem minor2_chartL_ge_of_oneSided (φ : ChartParams n) {B : ℝ}
    (hρ : ∀ i, -B ≤ φ.1 i) (hs : -B ≤ φ.2.2) (hα : ∀ l, |φ.2.1 l| ≤ B) {i j : Fin n}
    (hij : i < j) : interiorityConst n B ≤ minor2 (chartL φ) i j := by
  have hB : 0 ≤ B := (abs_nonneg _).trans (hα 0)
  have hscale : Real.exp (-(2 * B)) ≤ chartScale φ ^ 2 := by
    unfold chartScale
    rw [← Real.exp_nat_mul]
    apply Real.exp_le_exp.mpr
    push_cast
    linarith
  have hrad : ∀ k, softplus (-B) ≤ chartRadius φ k := fun k => softplus_le_softplus (hρ k)
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
  have hri : 0 < chartRadius φ i := softplus_pos _
  have hrj : 0 < chartRadius φ j := softplus_pos _
  have hr2 : softplus (-B) * softplus (-B) ≤ chartRadius φ i * chartRadius φ j :=
    mul_le_mul (hrad i) (hrad j) hsp.le hri.le
  have h12 : Real.exp (-(2 * B)) * (softplus (-B) * softplus (-B)) ≤
      chartScale φ ^ 2 * (chartRadius φ i * chartRadius φ j) :=
    mul_le_mul hscale hr2 (by positivity) (by positivity)
  calc Real.exp (-(2 * B)) * softplus (-B) ^ 2 * Real.sin (deltaB n B)
      = Real.exp (-(2 * B)) * (softplus (-B) * softplus (-B)) * Real.sin (deltaB n B) := by
        ring
    _ ≤ chartScale φ ^ 2 * (chartRadius φ i * chartRadius φ j) *
          Real.sin (chartAngle φ.2.1 j - chartAngle φ.2.1 i) :=
        mul_le_mul h12 hsin hsin0 (by positivity)
    _ = chartScale φ ^ 2 * chartRadius φ i * chartRadius φ j *
          Real.sin (chartAngle φ.2.1 j - chartAngle φ.2.1 i) := by ring

/-- The "defect" disjunction at threshold `B`: some radius parameter `ρ_i < −B`, or the scale
parameter `log s < −B`, or some angle parameter `α_l < −B`, or some `α_l > B`.

Source: SI Prop. S2.5(iii), "Consequently" clause (the four degeneration mechanisms). -/
def ChartDefect (φ : ChartParams n) (B : ℝ) : Prop :=
  (∃ i, φ.1 i < -B) ∨ φ.2.2 < -B ∨ (∃ l, φ.2.1 l < -B) ∨ ∃ l, B < φ.2.1 l

/-- **Eventual defect.** If some minor `Δ_{ij}(L(φ_k))` (`i < j`) tends to `0`, then for every
threshold `B`, eventually some `ρ_i < −B`, or `log s < −B`, or some `α_l < −B`, or some
`α_l > B`. In particular `ρ` and `log s` can only escape to `−∞`; escaping to `+∞` does not
shrink a minor.

Source: SI Prop. S2.5(iii), "Consequently" clause. -/
theorem chart_defect_eventually (φ : ℕ → ChartParams n) {i j : Fin n} (hij : i < j)
    (h : Tendsto (fun k => minor2 (chartL (φ k)) i j) atTop (𝓝 0)) (B : ℝ) :
    ∀ᶠ k in atTop, ChartDefect (φ k) B := by
  set B' := max B 0 with hB'
  have hB'0 : 0 ≤ B' := le_max_right _ _
  have hBB' : B ≤ B' := le_max_left _ _
  have hm := interiorityConst_pos n hB'0
  have hev : ∀ᶠ k in atTop, minor2 (chartL (φ k)) i j < interiorityConst n B' :=
    h (Iio_mem_nhds hm)
  filter_upwards [hev] with k hk
  by_contra hnd
  simp only [ChartDefect, not_or, not_exists, not_lt] at hnd
  obtain ⟨hρ, hs, hαl, hαu⟩ := hnd
  have hge := minor2_chartL_ge_of_oneSided (φ k) (B := B') (fun i => by linarith [hρ i])
    (by linarith) (fun l => abs_le.2 ⟨by linarith [hαl l], by linarith [hαu l]⟩) hij
  exact hge.not_gt hk

/-- **SI Prop. S2.5(iii), the coordinate case split (subsequence form).** If some minor
`Δ_{ij}(L(φ_k))` (`i < j`) tends to `0`, then one fixed coordinate is unbounded in the stated
direction along the sequence (equivalently, it tends to `∓∞` along a subsequence):
* some radius parameter `ρ_i` is unbounded below, or
* the scale parameter `log s` is unbounded below, or
* some angle parameter `α_l` is unbounded below, or
* some angle parameter `α_m` is unbounded above.

(Here "unbounded below" is `∀ B, ∃ᶠ k, x_k < −B`.) Convergence of a single coordinate to `∓∞`
along the full sequence does not hold in general, since the sequence may alternate between
mechanisms (see the module docstring).

Source: SI Prop. S2.5(iii), "Consequently `min_I Δ_I → 0` … : a radius `ρ_i → −∞`, the scale
`log s → −∞`, or an angle-gap collapse …, which by (S2.1) requires some `α_l → −∞` or some
`α_m → +∞`". -/
theorem chart_coordinate_case_split (φ : ℕ → ChartParams n) {i j : Fin n} (hij : i < j)
    (h : Tendsto (fun k => minor2 (chartL (φ k)) i j) atTop (𝓝 0)) :
    (∃ i, ∀ B : ℝ, ∃ᶠ k in atTop, (φ k).1 i < -B) ∨
      (∀ B : ℝ, ∃ᶠ k in atTop, (φ k).2.2 < -B) ∨
      (∃ l, ∀ B : ℝ, ∃ᶠ k in atTop, (φ k).2.1 l < -B) ∨
      (∃ l, ∀ B : ℝ, ∃ᶠ k in atTop, B < (φ k).2.1 l) := by
  by_contra hcon
  simp only [not_or, not_exists, not_forall, Filter.not_frequently, not_lt] at hcon
  obtain ⟨hρ, hs, hαl, hαu⟩ := hcon
  choose Bρ hBρ using hρ
  obtain ⟨Bs, hBs⟩ := hs
  choose Bl hBl using hαl
  choose Bu hBu using hαu
  -- one common threshold dominating all the finitely many bounds
  set B : ℝ := |Bs| + ∑ i, |Bρ i| + ∑ l, |Bl l| + ∑ l, |Bu l| with hBdef
  have hρle : ∀ i, |Bρ i| ≤ ∑ i, |Bρ i| := fun i =>
    Finset.single_le_sum (f := fun i => |Bρ i|) (fun _ _ => abs_nonneg _) (Finset.mem_univ i)
  have hlle : ∀ l, |Bl l| ≤ ∑ l, |Bl l| := fun l =>
    Finset.single_le_sum (f := fun l => |Bl l|) (fun _ _ => abs_nonneg _) (Finset.mem_univ l)
  have hule : ∀ l, |Bu l| ≤ ∑ l, |Bu l| := fun l =>
    Finset.single_le_sum (f := fun l => |Bu l|) (fun _ _ => abs_nonneg _) (Finset.mem_univ l)
  have hs0 : 0 ≤ ∑ i, |Bρ i| := Finset.sum_nonneg fun _ _ => abs_nonneg _
  have hl0 : 0 ≤ ∑ l, |Bl l| := Finset.sum_nonneg fun _ _ => abs_nonneg _
  have hu0 : 0 ≤ ∑ l, |Bu l| := Finset.sum_nonneg fun _ _ => abs_nonneg _
  have hBρ' : ∀ᶠ k in atTop, ∀ i, -Bρ i ≤ (φ k).1 i := eventually_all.2 hBρ
  have hBl' : ∀ᶠ k in atTop, ∀ l, -Bl l ≤ (φ k).2.1 l := eventually_all.2 hBl
  have hBu' : ∀ᶠ k in atTop, ∀ l, (φ k).2.1 l ≤ Bu l := eventually_all.2 hBu
  have hdef := chart_defect_eventually φ hij h B
  obtain ⟨k, hk⟩ := (hBρ'.and (hBs.and (hBl'.and (hBu'.and hdef)))).exists
  obtain ⟨h1, h2, h3, h4, h5⟩ := hk
  rcases h5 with ⟨i, hi⟩ | hsB | ⟨l, hl⟩ | ⟨l, hl⟩
  · have := h1 i
    have := hρle i
    have := le_abs_self (Bρ i)
    have := abs_nonneg Bs
    linarith
  · have := le_abs_self Bs
    linarith
  · have := h3 l
    have := hlle l
    have := le_abs_self (Bl l)
    have := abs_nonneg Bs
    linarith
  · have := h4 l
    have := hule l
    have := le_abs_self (Bu l)
    have := abs_nonneg Bs
    linarith

end FQGNN.Backbone

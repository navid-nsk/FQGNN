import Mathlib
import FQGNN.Common.Pseudoinverse
import FQGNN.Backbone.Chart
import FQGNN.Readout.ExponentialSum

/-!
# The Op-H boundary-Hermite readout

The weighted minimum-norm boundary-Hermite solve of manuscript eq. (6) / SI eq. (S4.7),
`c = W (BW)⁺ d + g · W (I − (BW)⁺(BW)) η`,  `W = diag(w) ⊕ diag(w)`,
and its guarantees: exact interpolation and Hermite velocity match for every `η`, gate and metric
(SI Prop. S4.1), affine reproduction (manuscript Prop. 1 / SI §S4.5), the homogeneity and
max-normalization scale-invariance of the metric (SI Prop. S6.1 (ii)–(iii)), invariance under row
equilibration (SI §S4.2, bullet 1) and the weighted minimum-norm characterization
(SI §S4.3, Prop. S6.2).

Sources:
* Manuscript, Results, eqs. (5)–(6) and Proposition 1 (affine reproduction).
* SI §S4.2–S4.5 (eqs. (S4.5)–(S4.8), Prop. S4.1, Prop. 1 restated with proof and caveats) and
  §S6.1 (Prop. S6.1).

Modelling conventions:
* `FullRowRank B := IsUnit (B * Bᵀ).det` and `pinvR B := Bᵀ (B Bᵀ)⁻¹`, `proj B := 1 − B⁺B`
  (from `FQGNN.Common.Pseudoinverse`). `pinvR` is the Moore–Penrose pseudoinverse only under full row
  rank; the SVD/`rcond` implementation of the paper is not modelled. The genuine Moore–Penrose
  form, with no rank hypothesis, is in `FQGNN.Readout.OpHGeneral`.
* The readout is per output coordinate: `c`, `d`, `p_a`, `p_b`, `v_e` are real scalars per
  coordinate; the curvature gate is computed from the *vector* velocities in an arbitrary normed space
  `E` (the paper uses the two spatial components, which share the gate).
-/

namespace FQGNN.Readout

open Matrix FQGNN.Common

variable {ι : Type*} [Fintype ι] [DecidableEq ι] {K : ℕ}

/-! ## The spectral metric and the operators -/

/-- Spectral metric `W = diag(w) ⊕ diag(w)`: cosine and sine columns of frequency `k` share `w_k`
(manuscript eq. (6); SI eq. (S4.7) and the first bullet of §S4.3). -/
noncomputable def metricW (w : Fin K → ℝ) : Matrix (Fin K ⊕ Fin K) (Fin K ⊕ Fin K) ℝ :=
  diagonal (Sum.elim w w)

/-- Op-H⁰, the Euclidean min-norm readout `c = B⁺ d + g (I − B⁺B) η` (SI §S4.3,
last paragraph "with the uniform metric … reduces to the Euclidean min-norm solve"). -/
noncomputable def opH0 (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) : Fin K ⊕ Fin K → ℝ :=
  pinvR B *ᵥ d + g • (proj B *ᵥ η)

/-- Op-H-W, the weighted min-norm readout `c = W (BW)⁺ d + g · W (I − (BW)⁺(BW)) η`
(manuscript eq. (6); SI eq. (S4.7)). -/
noncomputable def opHW (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) : Fin K ⊕ Fin K → ℝ :=
  metricW w *ᵥ (pinvR (B * metricW w) *ᵥ d) + g • (metricW w *ᵥ (proj (B * metricW w) *ᵥ η))

/-- Particular part `c_part(w) = W (BW)⁺ d` of eq. (S4.7) (SI Prop. S6.1 (ii) notation). -/
noncomputable def partOpHW (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ) (d : ι → ℝ) :
    Fin K ⊕ Fin K → ℝ :=
  metricW w *ᵥ (pinvR (B * metricW w) *ᵥ d)

/-- Null-space map `N(w) = W (I − (BW)⁺(BW))` of eq. (S4.7) (SI Prop. S6.1 (ii) notation). -/
noncomputable def nullMap (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ) :
    Matrix (Fin K ⊕ Fin K) (Fin K ⊕ Fin K) ℝ :=
  metricW w * proj (B * metricW w)

/-! ### Metric lemmas -/

/-- `W` is symmetric (diagonal) (SI §S4.3). -/
theorem metricW_transpose (w : Fin K → ℝ) : (metricW w)ᵀ = metricW w :=
  diagonal_transpose _

/-- `W` is invertible for every metric `w > 0` (SI §S4.4 proof of Prop. S4.1,
"`W` is invertible (`w > 0`)"). -/
theorem isUnit_det_metricW {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) : IsUnit (metricW w).det :=
  isUnit_det_diagonal_of_pos (fun i => by cases i <;> simp [hw])

/-- The uniform metric `w ≡ 1` gives `W = I` (SI §S4.3, last paragraph). -/
theorem metricW_one : metricW (fun _ : Fin K => (1 : ℝ)) = 1 := by
  unfold metricW
  rw [← diagonal_one]
  congr 1
  funext i
  cases i <;> rfl

/-- `W(α w) = α W(w)` (SI Prop. S6.1 (ii)). -/
theorem metricW_smul (α : ℝ) (w : Fin K → ℝ) : metricW (α • w) = α • metricW w := by
  unfold metricW
  rw [← diagonal_smul]
  congr 1
  funext i
  cases i <;> rfl

/-- `W² = diag(w²) ⊕ diag(w²)`. -/
theorem metricW_mul_metricW (w : Fin K → ℝ) :
    metricW w * metricW w = metricW (fun k => w k ^ 2) := by
  unfold metricW
  rw [diagonal_mul_diagonal]
  congr 1
  funext i
  cases i <;> simp [sq]

/-- `diag(w⁻¹) ⊕ diag(w⁻¹)` is a left inverse of `W` for `w > 0`. -/
theorem metricW_inv_mul {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) :
    metricW (fun k => (w k)⁻¹) * metricW w = 1 := by
  unfold metricW
  rw [diagonal_mul_diagonal, ← diagonal_one]
  congr 1
  funext i
  cases i <;> simp [(hw _).ne']

/-- `W⁻¹ = diag(w⁻¹) ⊕ diag(w⁻¹)` for `w > 0` (SI §S4.3 "`‖W⁻¹c‖²`"). -/
theorem inv_metricW {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) :
    (metricW w)⁻¹ = metricW (fun k => (w k)⁻¹) :=
  inv_eq_left_inv (metricW_inv_mul hw)

/-- `B W` has full row rank whenever `B` has and `w > 0`, i.e.
`IsUnit ((BW)(BW)ᵀ).det` (SI §S4.4 proof of Prop. S4.1, "`range(BW) = range(B)` and `BW`
has full row rank `R`"). -/
theorem fullRowRank_mul_metricW {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B)
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) : FullRowRank (B * metricW w) :=
  hB.mul_right (isUnit_det_metricW hw)

/-- The same statement in determinant form: `IsUnit ((B W)(B W)ᵀ).det`. -/
theorem isUnit_det_mul_metricW {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B)
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) :
    IsUnit ((B * metricW w) * (B * metricW w)ᵀ).det :=
  fullRowRank_mul_metricW hB hw

/-! ### Structural identities -/

/-- `opHW = c_part + g · N(w) η` (SI eq. (S4.7) split into its two terms, Prop. S6.1 (ii)). -/
theorem opHW_eq_part_add_nullMap (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ) (d : ι → ℝ)
    (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    opHW B w d g η = partOpHW B w d + g • (nullMap B w *ᵥ η) := by
  unfold opHW partOpHW nullMap
  simp only [mulVec_mulVec]

/-- With the uniform metric `w ≡ 1`, Op-H-W is Op-H⁰ (SI §S4.3, last paragraph: "(S4.7) reduces to
the Euclidean min-norm solve `c = B⁺d + g (I − B⁺B) η`"). -/
theorem opHW_one (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) : opHW B (fun _ => 1) d g η = opH0 B d g η := by
  unfold opHW opH0
  rw [metricW_one, Matrix.mul_one, one_mulVec, one_mulVec]

/-! ## Exact interpolation for every `η`, gate and metric (SI Prop. S4.1) -/

/-- The particular part is feasible: `B (W (BW)⁺ d) = d` (SI §S4.4 proof of Prop. S4.1,
"`B[W(BW)⁺d] = (BW)(BW)⁺d = d`"). -/
theorem mulVec_partOpHW {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B)
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (d : ι → ℝ) : B *ᵥ partOpHW B w d = d := by
  unfold partOpHW
  rw [mulVec_mulVec, mulVec_pinvR_mulVec (fullRowRank_mul_metricW hB hw)]

/-- The null-space channel never touches the constraints: `B (W (I − (BW)⁺(BW)) η) = 0`
(SI §S4.4 proof of Prop. S4.1; manuscript: "`η` ranges over the null space, where it can never
violate the constraints"). In this formalization `pinvR` is the pseudoinverse only under full row
rank, so the hypothesis `FullRowRank B` is used here; the rank-free statement for the genuine
Moore–Penrose inverse is `mulVec_mul_one_sub_mpinv_mulVec` in `FQGNN.Readout.OpHGeneral`. -/
theorem mulVec_nullMap_mulVec {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B)
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (η : Fin K ⊕ Fin K → ℝ) :
    B *ᵥ (nullMap B w *ᵥ η) = 0 := by
  unfold nullMap
  rw [mulVec_mulVec, ← Matrix.mul_assoc, mul_proj (fullRowRank_mul_metricW hB hw), zero_mulVec]

/-- **SI Prop. S4.1 (exact interpolation, every `η`, every gate, every metric).** If `B` has full row rank and `w > 0`, then `B c = d` for the Op-H-W coefficients
(manuscript eq. (6)) for every `d`, `g` and `η`. -/
theorem mulVec_opHW {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B) {w : Fin K → ℝ}
    (hw : ∀ k, 0 < w k) (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    B *ᵥ opHW B w d g η = d := by
  rw [opHW_eq_part_add_nullMap, mulVec_add, mulVec_smul, mulVec_partOpHW hB hw,
    mulVec_nullMap_mulVec hB hw, smul_zero, add_zero]

/-- **Euclidean case** `B (B⁺ d + g P_B η) = d` (SI §S4.3, uniform metric). -/
theorem mulVec_opH0 {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B) (d : ι → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) : B *ᵥ opH0 B d g η = d := by
  rw [← opHW_one]
  exact mulVec_opHW hB (fun _ => one_pos) d g η

/-! ### Geometric corollaries of exact interpolation (row level) -/

omit [Fintype ι] [DecidableEq ι] in
/-- Row `i` of `B c = d` as a dot product. -/
theorem row_dotProduct_eq {B : Matrix ι (Fin K ⊕ Fin K) ℝ} {c : Fin K ⊕ Fin K → ℝ}
    {d : ι → ℝ} (hc : B *ᵥ c = d) (i : ι) : B i ⬝ᵥ c = d i :=
  congrFun hc i

omit [Fintype ι] [DecidableEq ι] in
/-- **Value-row corollary (SI Prop. S4.1, "the reconstruction interpolates the bracketing
anchors").** If `B c = d` and row `i` is the value row `φ(τ)` at the centred time `τ = t − (a+b)/2`
with datum `d i = 0`, then the reconstruction equals the chord at `t`: `p̂(t) = A(t)`. -/
theorem recon_eq_chord_of_valueRow {B : Matrix ι (Fin K ⊕ Fin K) ℝ} {c : Fin K ⊕ Fin K → ℝ}
    {d : ι → ℝ} (hc : B *ᵥ c = d) (ω : Fin K → ℝ) (a b pa pb t : ℝ) {i : ι}
    (hrow : B i = phi ω (ctime a b t)) (hdi : d i = 0) :
    recon ω c a b pa pb t = chord a b pa pb t := by
  have h := row_dotProduct_eq hc i
  rw [hrow, hdi] at h
  rw [recon_eq]
  unfold rcomb
  rw [h, add_zero]

omit [Fintype ι] [DecidableEq ι] in
/-- **Value row at the left anchor (SI Prop. S4.1, `p̂(a) = p(a)`).** -/
theorem recon_left_of_valueRow {B : Matrix ι (Fin K ⊕ Fin K) ℝ} {c : Fin K ⊕ Fin K → ℝ}
    {d : ι → ℝ} (hc : B *ᵥ c = d) (ω : Fin K → ℝ) (a b pa pb : ℝ) {i : ι}
    (hrow : B i = phi ω (ctime a b a)) (hdi : d i = 0) :
    recon ω c a b pa pb a = pa := by
  rw [recon_eq_chord_of_valueRow hc ω a b pa pb a hrow hdi, chord_left]

omit [Fintype ι] [DecidableEq ι] in
/-- **Value row at the right anchor (SI Prop. S4.1, `p̂(b) = p(b)`).** -/
theorem recon_right_of_valueRow {B : Matrix ι (Fin K ⊕ Fin K) ℝ} {c : Fin K ⊕ Fin K → ℝ}
    {d : ι → ℝ} (hc : B *ᵥ c = d) (ω : Fin K → ℝ) {a b : ℝ} (hab : a ≠ b) (pa pb : ℝ) {i : ι}
    (hrow : B i = phi ω (ctime a b b)) (hdi : d i = 0) :
    recon ω c a b pa pb b = pb := by
  rw [recon_eq_chord_of_valueRow hc ω a b pa pb b hrow hdi, chord_right hab]

omit [Fintype ι] [DecidableEq ι] in
/-- **Derivative-row corollary (SI Prop. S4.1, "the Hermite velocity match").** If `B c = d`
and row `i` is the equilibrated derivative row `s • φ'(τ)` at `τ = t − (a+b)/2` with datum
`d i = s (v − A')`, `s ≠ 0`, then `p̂'(t) = v`. -/
theorem hasDerivAt_recon_of_derivRow {B : Matrix ι (Fin K ⊕ Fin K) ℝ} {c : Fin K ⊕ Fin K → ℝ}
    {d : ι → ℝ} (hc : B *ᵥ c = d) (ω : Fin K → ℝ) (a b pa pb t v s : ℝ) (hs : s ≠ 0) {i : ι}
    (hrow : B i = s • dphi ω (ctime a b t)) (hdi : d i = s * (v - chordSlope a b pa pb)) :
    HasDerivAt (recon ω c a b pa pb) v t := by
  have h := row_dotProduct_eq hc i
  rw [hrow, hdi, smul_dotProduct, smul_eq_mul] at h
  have h' : dphi ω (ctime a b t) ⬝ᵥ c = v - chordSlope a b pa pb := mul_left_cancel₀ hs h
  have hd := hasDerivAt_recon ω c a b pa pb t
  unfold ctime at h'
  rwa [h', add_sub_cancel] at hd

/-! ### The boundary-Hermite system (SI eqs. (S4.5)–(S4.6)) -/

/-- Constraint matrix of the boundary-Hermite system (SI eq. (S4.6)): two value rows `φ(τ_a)`,
`φ(τ_b)` (index `Sum.inl 0`, `Sum.inl 1`) and `R_v` included derivative rows `s_e • φ'(τ_e)` at the
row times `te e` (in practice `te e ∈ {a, b}`), equilibrated by `s_e > 0` (SI §S4.2, bullet 1). -/
noncomputable def hermiteB (ω : Fin K → ℝ) (a b : ℝ) {Rv : ℕ} (te s : Fin Rv → ℝ) :
    Matrix (Fin 2 ⊕ Fin Rv) (Fin K ⊕ Fin K) ℝ :=
  Matrix.of (Sum.elim (fun e : Fin 2 => phi ω (ctime a b (![a, b] e)))
    (fun e : Fin Rv => s e • dphi ω (ctime a b (te e))))

/-- Right-hand side of the boundary-Hermite system (SI eqs. (S4.5)–(S4.6)): zeros on the value rows,
the equilibrated velocity mismatches `s_e (v_e − A')` on the included derivative rows. -/
noncomputable def hermiteD (a b pa pb : ℝ) {Rv : ℕ} (s v : Fin Rv → ℝ) : Fin 2 ⊕ Fin Rv → ℝ :=
  Sum.elim 0 (fun e : Fin Rv => s e * (v e - chordSlope a b pa pb))

/-- **SI Prop. S4.1, packaged.** Suppose the boundary-Hermite matrix has full row rank,
`a ≠ b`, `w > 0`, and the equilibration factors are nonzero. Then for every `η` and every gate value
`g`, the Op-H-W reconstruction interpolates both bracketing anchors, `p̂(a) = p_a`, `p̂(b) = p_b`, and
matches every included boundary velocity, `p̂'(t_e) = v_e`. -/
theorem opHW_hermite_interpolation (ω : Fin K → ℝ) {a b : ℝ} (hab : a ≠ b) (pa pb : ℝ)
    {Rv : ℕ} (te s v : Fin Rv → ℝ) (hs : ∀ e, s e ≠ 0)
    (hB : FullRowRank (hermiteB ω a b te s)) {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) :
    let c := opHW (hermiteB ω a b te s) w (hermiteD a b pa pb s v) g η
    recon ω c a b pa pb a = pa ∧ recon ω c a b pa pb b = pb ∧
      ∀ e, HasDerivAt (recon ω c a b pa pb) (v e) (te e) := by
  intro c
  have hc : hermiteB ω a b te s *ᵥ c = hermiteD a b pa pb s v := mulVec_opHW hB hw _ g η
  refine ⟨?_, ?_, fun e => ?_⟩
  · exact recon_left_of_valueRow hc ω a b pa pb (i := Sum.inl 0) rfl rfl
  · exact recon_right_of_valueRow hc ω hab pa pb (i := Sum.inl 1) rfl rfl
  · exact hasDerivAt_recon_of_derivRow hc ω a b pa pb (te e) (v e) (s e) (hs e)
      (i := Sum.inr e) rfl rfl

/-! ## Affine reproduction (manuscript Prop. 1; SI §S4.5) -/

/-- The particular part annihilates `d = 0` with **no** rank hypothesis (SI §S4.5 proof: "the
pseudoinverse of any matrix annihilates the zero vector"). -/
theorem partOpHW_zero (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ) :
    partOpHW B w 0 = 0 := by
  unfold partOpHW
  rw [mulVec_zero, mulVec_zero]

/-- With `d = 0` only the null-space channel survives: `opHW B w 0 g η = g · N(w) η` (SI §S4.5,
caveat 1). No rank hypothesis. -/
theorem opHW_zero_rhs (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ) (g : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) : opHW B w 0 g η = g • (nullMap B w *ᵥ η) := by
  rw [opHW_eq_part_add_nullMap, partOpHW_zero, zero_add]

/-- **Prop. 1 / SI §S4.5, coefficient form.** `d = 0` and `g = 0` give `c = 0`
for every `η`, every `w` and every `B`, with **no rank hypothesis**. -/
theorem opHW_zero_zero (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (η : Fin K ⊕ Fin K → ℝ) : opHW B w 0 0 η = 0 := by
  rw [opHW_zero_rhs, zero_smul]

/-- **SI §S4.5, caveat 1 (gate disabled / no velocity row).** With `d = 0` but an open gate, chord
reproduction holds at `η = 0`: `opHW B w 0 g 0 = 0` for every `g`, with no rank hypothesis. -/
theorem opHW_zero_rhs_zero_eta (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ) (g : ℝ) :
    opHW B w 0 g 0 = 0 := by
  rw [opHW_zero_rhs, mulVec_zero, smul_zero]

/-- **Reconstruction form.** If `d = 0` and `g = 0`, the Op-H-W reconstruction is the chord,
for every `B`, `w`, `η` (no rank hypothesis) (manuscript Prop. 1; SI §S4.5). -/
theorem recon_opHW_zero_zero (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ)
    (η : Fin K ⊕ Fin K → ℝ) (ω : Fin K → ℝ) (a b pa pb t : ℝ) :
    recon ω (opHW B w 0 0 η) a b pa pb t = chord a b pa pb t := by
  rw [opHW_zero_zero, recon_zero]

/-- The curvature gate `g = tanh(softplus(ρ) · d̄)` (SI §S4.3, fourth bullet; manuscript eq. (6)
"`g` is a leak-safe curvature gate read off the boundary-velocity mismatch"). -/
noncomputable def gate (ρ dbar : ℝ) : ℝ :=
  Real.tanh (FQGNN.Backbone.softplus ρ * dbar)

/-- The gate closes on zero mismatch: `gate ρ 0 = 0` for every `ρ` (SI §S4.5 proof,
"`g = tanh(softplus(ρ)·0) = 0` for every `ρ`"). -/
theorem gate_zero (ρ : ℝ) : gate ρ 0 = 0 := by
  rw [gate, mul_zero, Real.tanh_zero]

/-- Relative boundary-velocity mismatch over the `R_v` included rows,
`d̄ = Σ_e ‖v_e − A'‖ / (R_v ‖A'‖ + ε)` (SI §S4.3, fourth bullet), for velocities in a normed space
`E` (the paper: the two spatial components). -/
noncomputable def velMismatch {E : Type*} [NormedAddCommGroup E] {Rv : ℕ} (ε : ℝ)
    (V : Fin Rv → E) (A' : E) : ℝ :=
  (∑ e, ‖V e - A'‖) / (Rv * ‖A'‖ + ε)

/-- `d̄ = 0` when every included velocity equals the chord slope (SI §S4.5 proof). -/
theorem velMismatch_eq_zero {E : Type*} [NormedAddCommGroup E] {Rv : ℕ} (ε : ℝ)
    {V : Fin Rv → E} {A' : E} (hV : ∀ e, V e = A') : velMismatch ε V A' = 0 := by
  unfold velMismatch
  simp [hV]

/-- The deployed gate value (SI §S4.3, fourth bullet): `tanh(softplus(ρ) d̄)` when the gate is
enabled and at least one velocity row is included, `1` otherwise ("With the gate disabled, or with no
included velocity row, `g = 1`"). -/
noncomputable def gateValue {E : Type*} [NormedAddCommGroup E] (enabled : Bool) {Rv : ℕ}
    (ρ ε : ℝ) (V : Fin Rv → E) (A' : E) : ℝ :=
  if enabled = true ∧ 0 < Rv then gate ρ (velMismatch ε V A') else 1

/-- **Gate on affine data.** With the gate enabled, at least one included velocity row, and all
included velocities equal to the chord slope, the gate is `0` for every `ρ` and `ε`
(SI §S4.5 proof). -/
theorem gateValue_eq_zero {E : Type*} [NormedAddCommGroup E] {Rv : ℕ} (hRv : 0 < Rv)
    (ρ ε : ℝ) {V : Fin Rv → E} {A' : E} (hV : ∀ e, V e = A') :
    gateValue true ρ ε V A' = 0 := by
  unfold gateValue
  simp only [true_and, hRv, ↓reduceIte, velMismatch_eq_zero ε hV, gate_zero]

/-- Vector chord slope `A' = (p_b − p_a)/(b − a)` in a real normed space (SI eq. (S4.3)). -/
noncomputable def chordSlopeVec {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (a b : ℝ) (Pa Pb : E) : E :=
  (b - a)⁻¹ • (Pb - Pa)

/-- On an affine track `P t = U₀ + t • U₁`, every same-side finite-difference velocity
`(P t_i − P t_j)/(t_i − t_j)` equals `U₁` (SI §S4.5 proof, "finite differences reproduce the slope
exactly"); in particular the vector chord slope is `U₁`. -/
theorem chordSlopeVec_of_affine {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    {P : ℝ → E} {U₀ U₁ : E} (hP : ∀ t, P t = U₀ + t • U₁) {ti tj : ℝ} (hij : ti ≠ tj) :
    chordSlopeVec tj ti (P tj) (P ti) = U₁ := by
  unfold chordSlopeVec
  rw [hP ti, hP tj, add_sub_add_left_eq_sub, ← sub_smul, smul_smul,
    inv_mul_cancel₀ (sub_ne_zero.mpr hij), one_smul]

/-- The right-hand side of the boundary-Hermite system vanishes on affine data with exact boundary
velocities (SI §S4.5 proof, "every included velocity entry is `v_e − A' = u₁ − u₁ = 0`. Hence
`d = 0`"). -/
theorem hermiteD_eq_zero_of_affine {p : ℝ → ℝ} {u₀ u₁ : ℝ} (hp : ∀ t, p t = u₀ + u₁ * t)
    {a b : ℝ} (hab : a ≠ b) {Rv : ℕ} (s : Fin Rv → ℝ) {v : Fin Rv → ℝ} (hv : ∀ e, v e = u₁) :
    hermiteD a b (p a) (p b) s v = 0 := by
  have hslope := (chord_eq_of_affine hp hab).2
  funext i
  cases i with
  | inl _ => rfl
  | inr e => simp [hermiteD, hv e, hslope]

/-- **Manuscript Prop. 1 (affine reproduction) / SI §S4.5 (restated Prop. 1), for the Op-H
readout of eq. (6).** Let the track be affine and constant-velocity, `P t = U₀ + t • U₁`
in a real normed space `E`, observed in a linear coordinate `π` (each spatial coordinate is solved
separately; the coordinates share the gate). Let the `R_v ≥ 1` included boundary velocities be exact
for that track (`V e = U₁`, e.g. same-side finite differences, `chordSlopeVec_of_affine`), and let the
curvature gate be enabled. Then for **every** `η`, **every** metric `w` (no positivity needed),
every gate parameters `ρ, ε`, every equilibration `s`, every row times `te`, and with **no rank
hypothesis** on the boundary-Hermite matrix, the coefficients vanish, `c = 0`, and the reconstruction
is the exact track: `p̂(t) = π(U₀) + π(U₁) t` for all `t`. -/
theorem prop1_affine_reproduction {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (π : E →ₗ[ℝ] ℝ) {P : ℝ → E} {U₀ U₁ : E} (hP : ∀ t, P t = U₀ + t • U₁)
    {a b : ℝ} (hab : a ≠ b) (ω : Fin K → ℝ) {Rv : ℕ} (hRv : 0 < Rv) (te s : Fin Rv → ℝ)
    {V : Fin Rv → E} (hV : ∀ e, V e = U₁) (w : Fin K → ℝ) (ρ ε : ℝ)
    (η : Fin K ⊕ Fin K → ℝ) :
    let g := gateValue true ρ ε V (chordSlopeVec a b (P a) (P b))
    let c := opHW (hermiteB ω a b te s) w
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
    change opHW (hermiteB ω a b te s) w _ g η = 0
    rw [hd, hg, opHW_zero_zero]
  refine ⟨hc, fun t => ?_⟩
  rw [hc, recon_zero]
  exact (chord_eq_of_affine (p := fun t => π (P t)) hp hab).1 t

/-! ## Homogeneity and max-normalization (SI Prop. S6.1 (ii)–(iii)) -/

/-- **SI Prop. S6.1 (ii), particular part 0-homogeneous:** `c_part(α w) = c_part(w)` for
`α ≠ 0` (no rank hypothesis). -/
theorem partOpHW_smul {α : ℝ} (hα : α ≠ 0) (B : Matrix ι (Fin K ⊕ Fin K) ℝ)
    (w : Fin K → ℝ) (d : ι → ℝ) : partOpHW B (α • w) d = partOpHW B w d := by
  unfold partOpHW
  rw [mulVec_mulVec, mulVec_mulVec, metricW_smul, smul_mul_pinvR_mul_smul hα]

/-- **SI Prop. S6.1 (ii), null-space map 1-homogeneous:** `N(α w) = α N(w)` for `α ≠ 0`
(no rank hypothesis). -/
theorem nullMap_smul {α : ℝ} (hα : α ≠ 0) (B : Matrix ι (Fin K ⊕ Fin K) ℝ)
    (w : Fin K → ℝ) : nullMap B (α • w) = α • nullMap B w := by
  unfold nullMap
  rw [metricW_smul, proj_mul_smul hα, Matrix.smul_mul]

/-- **SI Prop. S6.1 (ii), consequence for the operator.** Scaling the metric by `α ≠ 0` is the
same as scaling the gate: `opHW B (α w) d g η = opHW B w d (α g) η` ("without [the normalization] an
overall inflation of `w` would silently rescale the learned null-space channel"). -/
theorem opHW_smul_metric {α : ℝ} (hα : α ≠ 0) (B : Matrix ι (Fin K ⊕ Fin K) ℝ)
    (w : Fin K → ℝ) (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    opHW B (α • w) d g η = opHW B w d (α * g) η := by
  rw [opHW_eq_part_add_nullMap, opHW_eq_part_add_nullMap, partOpHW_smul hα, nullMap_smul hα,
    smul_mulVec, smul_smul, mul_comm]

/-- Per-gap max-normalization `w ↦ w / max_k w_k` (SI §S4.3, first bullet: "part of the operator
definition"). The maximum over the finite index set is `⨆ k, w k`. -/
noncomputable def normalizeW (w : Fin K → ℝ) : Fin K → ℝ := fun k => w k / ⨆ j, w j

/-- `w` and `α w` (`α > 0`) normalize to the same representative (SI Prop. S6.1 (iii) proof). -/
theorem normalizeW_smul {α : ℝ} (hα : 0 < α) (w : Fin K → ℝ) :
    normalizeW (α • w) = normalizeW w := by
  funext k
  unfold normalizeW
  simp only [Pi.smul_apply, smul_eq_mul]
  rw [← Real.mul_iSup_of_nonneg hα.le, mul_div_mul_left _ _ hα.ne']

/-- The normalized metric is still a metric: `w > 0 ⇒ normalizeW w > 0` (SI §S4.3). -/
theorem normalizeW_pos {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (k : Fin K) : 0 < normalizeW w k :=
  div_pos (hw k) (lt_of_lt_of_le (hw k) (le_ciSup (Finite.bddAbove_range w) k))

/-- The normalized metric is bounded by `1` (SI §S4.3, `w ← w / max_k w_k`). -/
theorem normalizeW_le_one {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (k : Fin K) :
    normalizeW w k ≤ 1 :=
  div_le_one_of_le₀ (le_ciSup (Finite.bddAbove_range w) k)
    ((hw k).le.trans (le_ciSup (Finite.bddAbove_range w) k))

/-- **SI Prop. S6.1 (iii) (the normalization is part of the operator).** After the per-gap
max-normalization the coefficient map is metric-scale-invariant: `c(α w) = c(w)` for every `α > 0`. -/
theorem opHW_normalizeW_smul {α : ℝ} (hα : 0 < α) (B : Matrix ι (Fin K ⊕ Fin K) ℝ)
    (w : Fin K → ℝ) (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    opHW B (normalizeW (α • w)) d g η = opHW B (normalizeW w) d g η := by
  rw [normalizeW_smul hα]

/-! ## Row-scaling invariance (SI §S4.2, bullet 1) -/

/-- **SI §S4.2, bullet 1.** Row operations by any invertible `D` leave
Op-H-W unchanged: `opHW (D B) w (D d) g η = opHW B w d g η` (no rank hypothesis). -/
theorem opHW_mul_left {D : Matrix ι ι ℝ} (hD : IsUnit D.det) (B : Matrix ι (Fin K ⊕ Fin K) ℝ)
    (w : Fin K → ℝ) (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    opHW (D * B) w (D *ᵥ d) g η = opHW B w d g η := by
  unfold opHW
  rw [Matrix.mul_assoc, pinvR_mul_left_mulVec hD, proj_mul_left hD]

/-- **SI §S4.2, bullet 1 ("derivative rows (and their right-hand sides) are equilibrated by
`1/mean|ω|`; row scaling … leaves the min-norm solution and the null space unchanged").** For a
positive diagonal `D = diag(s)`: `opHW (D B) w (D d) g η = opHW B w d g η`. -/
theorem opHW_diagonal_mul {s : ι → ℝ} (hs : ∀ i, 0 < s i) (B : Matrix ι (Fin K ⊕ Fin K) ℝ)
    (w : Fin K → ℝ) (d : ι → ℝ) (g : ℝ) (η : Fin K ⊕ Fin K → ℝ) :
    opHW (diagonal s * B) w (diagonal s *ᵥ d) g η = opHW B w d g η :=
  opHW_mul_left (isUnit_det_diagonal_of_pos hs) B w d g η

/-! ## Weighted minimum-norm characterization (SI §S4.3; SI Prop. S6.2) -/

/-- Weighted squared norm `‖W⁻¹ c‖²` (SI §S4.3, "`min ‖W⁻¹c‖²` subject to `Bc = d`"). -/
noncomputable def weightedNormSq (w : Fin K → ℝ) (c : Fin K ⊕ Fin K → ℝ) : ℝ :=
  ((metricW w)⁻¹ *ᵥ c) ⬝ᵥ ((metricW w)⁻¹ *ᵥ c)

/-- Explicit form `‖W⁻¹c‖² = Σ_k (α_k² + β_k²)/w_k²` (SI §S4.3, first bullet; manuscript after
eq. (6), "`Σ_k |c_k|²/w_k²`"). -/
theorem weightedNormSq_eq_sum {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (c : Fin K ⊕ Fin K → ℝ) :
    weightedNormSq w c = ∑ k, (c (Sum.inl k) ^ 2 + c (Sum.inr k) ^ 2) / w k ^ 2 := by
  unfold weightedNormSq
  rw [inv_metricW hw]
  unfold metricW
  simp only [mulVec_diagonal, dotProduct, Fintype.sum_sum_type, Sum.elim_inl, Sum.elim_inr,
    ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun k _ => ?_
  field_simp

/-- **Closed form (matrix).** `W (BW)⁺ = W² Bᵀ (B W² Bᵀ)⁻¹` for every `B` and `w`
(uses only `Wᵀ = W`). -/
theorem metricW_mul_pinvR_eq (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ) :
    metricW w * pinvR (B * metricW w) =
      (metricW w * metricW w) * Bᵀ * (B * (metricW w * metricW w) * Bᵀ)⁻¹ := by
  unfold pinvR
  rw [transpose_mul, metricW_transpose]
  simp only [Matrix.mul_assoc]

/-- **Closed form (vector).** `W (BW)⁺ d = W² Bᵀ (B W² Bᵀ)⁻¹ d`. -/
theorem partOpHW_eq (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (w : Fin K → ℝ) (d : ι → ℝ) :
    partOpHW B w d =
      ((metricW w * metricW w) * Bᵀ * (B * (metricW w * metricW w) * Bᵀ)⁻¹) *ᵥ d := by
  unfold partOpHW
  rw [mulVec_mulVec, metricW_mul_pinvR_eq]

/-- `W⁻¹ (W (BW)⁺ d) = (BW)⁺ d` for `w > 0` (the substitution `u = W⁻¹c` of SI §S4.3). -/
theorem metricW_inv_mulVec_partOpHW {w : Fin K → ℝ} (hw : ∀ k, 0 < w k)
    (B : Matrix ι (Fin K ⊕ Fin K) ℝ) (d : ι → ℝ) :
    (metricW w)⁻¹ *ᵥ partOpHW B w d = pinvR (B * metricW w) *ᵥ d := by
  unfold partOpHW
  rw [mulVec_mulVec, nonsing_inv_mul _ (isUnit_det_metricW hw), one_mulVec]

omit [Fintype ι] [DecidableEq ι] in
/-- Feasibility transfers through the substitution `u = W⁻¹ c`: `B c = d ⇒ (BW)(W⁻¹c) = d`. -/
theorem mul_metricW_mulVec_inv {B : Matrix ι (Fin K ⊕ Fin K) ℝ} {w : Fin K → ℝ}
    (hw : ∀ k, 0 < w k) {c : Fin K ⊕ Fin K → ℝ} {d : ι → ℝ} (hc : B *ᵥ c = d) :
    (B * metricW w) *ᵥ ((metricW w)⁻¹ *ᵥ c) = d := by
  rw [mulVec_mulVec, Matrix.mul_assoc, mul_nonsing_inv _ (isUnit_det_metricW hw), Matrix.mul_one,
    hc]

/-- **Weighted min-norm (minimality).** For full row rank `B` and `w > 0`, the particular
part `W (BW)⁺ d` has the least weighted norm `‖W⁻¹c‖²` among all solutions of `B c = d`
(SI §S4.3 "the solution of the weighted minimum-norm problem `min ‖W⁻¹c‖²` subject to `Bc = d`"). -/
theorem partOpHW_weightedNormSq_le {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B)
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) {d : ι → ℝ} {c : Fin K ⊕ Fin K → ℝ}
    (hc : B *ᵥ c = d) : weightedNormSq w (partOpHW B w d) ≤ weightedNormSq w c := by
  unfold weightedNormSq
  rw [metricW_inv_mulVec_partOpHW hw]
  exact pinvR_mulVec_dotProduct_le (fullRowRank_mul_metricW hB hw)
    (mul_metricW_mulVec_inv hw hc)

/-- **Weighted min-norm (uniqueness).** A solution of `B c = d` whose weighted norm does not
exceed that of `W (BW)⁺ d` equals it (SI §S4.3). -/
theorem eq_partOpHW_of_weightedNormSq_le {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B)
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) {d : ι → ℝ} {c : Fin K ⊕ Fin K → ℝ}
    (hc : B *ᵥ c = d) (hle : weightedNormSq w c ≤ weightedNormSq w (partOpHW B w d)) :
    c = partOpHW B w d := by
  have hWu := isUnit_det_metricW hw
  unfold weightedNormSq at hle
  rw [metricW_inv_mulVec_partOpHW hw] at hle
  have h := eq_pinvR_mulVec_of_dotProduct_le (fullRowRank_mul_metricW hB hw)
    (mul_metricW_mulVec_inv hw hc) hle
  have h2 : metricW w *ᵥ ((metricW w)⁻¹ *ᵥ c) = metricW w *ᵥ (pinvR (B * metricW w) *ᵥ d) := by
    rw [h]
  rwa [mulVec_mulVec, mul_nonsing_inv _ hWu, one_mulVec] at h2

/-- **Weighted min-norm characterization, bundled (SI §S4.3; SI Prop. S6.2).** For full row rank `B` and `w > 0`, `W (BW)⁺ d` is feasible, is the unique minimizer of
`‖W⁻¹c‖²` on `{c | B c = d}`, and equals `W² Bᵀ (B W² Bᵀ)⁻¹ d`. -/
theorem partOpHW_isWeightedMinNorm {B : Matrix ι (Fin K ⊕ Fin K) ℝ} (hB : FullRowRank B)
    {w : Fin K → ℝ} (hw : ∀ k, 0 < w k) (d : ι → ℝ) :
    B *ᵥ partOpHW B w d = d ∧
      (∀ c, B *ᵥ c = d → weightedNormSq w (partOpHW B w d) ≤ weightedNormSq w c) ∧
      (∀ c, B *ᵥ c = d → weightedNormSq w c = weightedNormSq w (partOpHW B w d) →
        c = partOpHW B w d) ∧
      partOpHW B w d =
        ((metricW w * metricW w) * Bᵀ * (B * (metricW w * metricW w) * Bᵀ)⁻¹) *ᵥ d :=
  ⟨mulVec_partOpHW hB hw d, fun _ hc => partOpHW_weightedNormSq_le hB hw hc,
    fun _ hc heq => eq_partOpHW_of_weightedNormSq_le hB hw hc heq.le, partOpHW_eq B w d⟩

end FQGNN.Readout

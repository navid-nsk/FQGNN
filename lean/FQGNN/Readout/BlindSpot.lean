import Mathlib

/-!
# The exactness / blind-spot trade

Sources:
* SI §S7.2, **Proposition S7.6** ("The exactness/blind-spot trade") and the remark after it: the
  S-curve blind spot is *not* information-theoretically forced; the boundary 1-jet cannot see it,
  the exterior waveform can.

## Part 1 — the abstract trade
A readout is any map `O : Jet → Ext → Out` (boundary 1-jet, exterior data, output). Given a
predicate `JetStraight` on jets and the chord `chordOf : Jet → Out`:
* `ChordExact O S chordOf` : `O` returns the chord on every jet-straight jet, for every exterior;
* `ExteriorBlind O S` : `O` is constant in the exterior on every jet-straight jet;
* `RespondsAt O j` : `O` distinguishes two exteriors at the jet `j`.
Then chord-exact ⇒ exterior-blind (`exteriorBlind_of_chordExact`), and responding at a
jet-straight jet ⇒ not chord-exact (`not_chordExactAt_of_respondsAt`,
`not_chordExact_of_respondsAt`). Both alternatives are realizable (`chordExact_const`,
`gatedReadout_chordExact`, `respondsAt_of_injective`), so the trade is genuine, not vacuous.

## Part 2 — the concrete witness
On the unit gap `[0,1]`, the parallel-tangent S-curve `sCurve u = sin(2πu) − ½ sin(4πu)` has zero
boundary values and zero boundary derivatives, but takes the values `1` at `u = 1/4` and `−1` at
`u = 3/4`. Adding `ε • sCurve` (`ε ≠ 0`) to any line yields a track with the *same*
boundary 1-jet as the line (hence jet-straight) that is not the line. Consequently every
chord-exact readout is wrong on that S-curve track, whatever the exterior data
(`chordExact_misses_sCurve`).
-/

namespace FQGNN.Readout

open Real

section Abstract

variable {Jet Ext Out : Type*}

/-- Chord exactness at one jet: `O j e = chordOf j` for every exterior `e`
(SI Prop. S7.6). -/
def ChordExactAt (O : Jet → Ext → Out) (chordOf : Jet → Out) (j : Jet) : Prop :=
  ∀ e, O j e = chordOf j

/-- Chord exactness on every jet-straight gap, regardless of the exterior data
(SI Prop. S7.6 hypothesis). -/
def ChordExact (O : Jet → Ext → Out) (JetStraight : Jet → Prop) (chordOf : Jet → Out) : Prop :=
  ∀ j, JetStraight j → ChordExactAt O chordOf j

/-- Constancy in the exterior observations on every jet-straight gap (SI Prop. S7.6). -/
def ExteriorBlind (O : Jet → Ext → Out) (JetStraight : Jet → Prop) : Prop :=
  ∀ j, JetStraight j → ∀ e e', O j e = O j e'

/-- The readout responds to exterior evidence at the jet `j`: two exteriors give different outputs
(SI Prop. S7.6, "responsive to the exterior"). -/
def RespondsAt (O : Jet → Ext → Out) (j : Jet) : Prop :=
  ∃ e e', O j e ≠ O j e'

/-- **SI Prop. S7.6, first half.** A readout that returns the exact
chord on every jet-straight gap regardless of the exterior data is constant in the exterior
observations on jet-straight gaps. -/
theorem exteriorBlind_of_chordExact {O : Jet → Ext → Out} {JetStraight : Jet → Prop}
    {chordOf : Jet → Out} (h : ChordExact O JetStraight chordOf) :
    ExteriorBlind O JetStraight := fun j hj e e' => (h j hj e).trans (h j hj e').symm

/-- **SI Prop. S7.6, "cannot distinguish".** Two inputs sharing the
same jet-straight boundary jet (e.g. a straight transit and a parallel-tangent S-curve) receive the
same output from any chord-exact readout, whatever their exterior data. -/
theorem chordExact_cannot_distinguish {O : Jet → Ext → Out} {JetStraight : Jet → Prop}
    {chordOf : Jet → Out} (h : ChordExact O JetStraight chordOf) {j : Jet} (hj : JetStraight j)
    (e₁ e₂ : Ext) : O j e₁ = O j e₂ :=
  exteriorBlind_of_chordExact h j hj e₁ e₂

/-- Pointwise form of the converse (SI Prop. S7.6, second half): a
readout that responds to the exterior at `j` is not chord-exact at `j`. (Jet-straightness is not
even needed for this pointwise form.) -/
theorem not_chordExactAt_of_respondsAt {O : Jet → Ext → Out} {chordOf : Jet → Out} {j : Jet}
    (hr : RespondsAt O j) : ¬ ChordExactAt O chordOf j := by
  rintro hc
  obtain ⟨e, e', hne⟩ := hr
  exact hne ((hc e).trans (hc e').symm)

/-- **SI Prop. S7.6, second half.** An operator that distinguishes
two exteriors on some jet-straight jet gives up chord exactness (on jet-straight gaps). -/
theorem not_chordExact_of_respondsAt {O : Jet → Ext → Out} {JetStraight : Jet → Prop}
    {chordOf : Jet → Out} {j : Jet} (hj : JetStraight j) (hr : RespondsAt O j) :
    ¬ ChordExact O JetStraight chordOf := fun hc =>
  not_chordExactAt_of_respondsAt hr (hc j hj)

/-- Quantitative converse (SI Prop. S7.6): if `O` gives different
outputs for two exteriors at a jet `j`, then at least one of those two outputs differs from the
chord. -/
theorem ne_chord_of_ne {O : Jet → Ext → Out} {chordOf : Jet → Out} {j : Jet} {e e' : Ext}
    (hne : O j e ≠ O j e') : O j e ≠ chordOf j ∨ O j e' ≠ chordOf j := by
  by_contra h
  rw [not_or, not_not, not_not] at h
  exact hne (h.1.trans h.2.symm)

/-- **The trade as an exclusive alternative** (SI Prop. S7.6 proof: "output independent of the
exterior" and "output responsive to the exterior" are exclusive on the same event). -/
theorem not_chordExact_and_respondsAt {O : Jet → Ext → Out} {JetStraight : Jet → Prop}
    {chordOf : Jet → Out} :
    ¬ (ChordExact O JetStraight chordOf ∧ ∃ j, JetStraight j ∧ RespondsAt O j) := by
  rintro ⟨hc, j, hj, hr⟩
  exact not_chordExact_of_respondsAt hj hr hc

/-- Equivalent form (SI Prop. S7.6): chord exactness forces non-responsiveness at every
jet-straight jet. -/
theorem chordExact_imp_not_respondsAt {O : Jet → Ext → Out} {JetStraight : Jet → Prop}
    {chordOf : Jet → Out} (hc : ChordExact O JetStraight chordOf) {j : Jet}
    (hj : JetStraight j) : ¬ RespondsAt O j := fun hr =>
  not_chordExact_of_respondsAt hj hr hc

/-- Non-vacuity, exact side (SI Prop. S7.6): the pure chord readout
`fun j _ => chordOf j` is chord-exact. -/
theorem chordExact_const (JetStraight : Jet → Prop) (chordOf : Jet → Out) :
    ChordExact (fun j (_ : Ext) => chordOf j) JetStraight chordOf := fun _ _ _ => rfl

/-- The conservatively gated readout (the deployed conservative gating of SI
Thm S7.5(ii)): return the chord on jet-straight jets, and an arbitrary exterior-aware readout `F`
otherwise. -/
noncomputable def gatedReadout (JetStraight : Jet → Prop) [DecidablePred JetStraight]
    (chordOf : Jet → Out) (F : Jet → Ext → Out) : Jet → Ext → Out :=
  fun j e => if JetStraight j then chordOf j else F j e

/-- SI Thm S7.5(ii) ("the deployed conservative gating … enforces this identically on jet-straight
gaps") in abstract form. The gated readout is chord-exact for every
exterior-aware fallback `F`; by `exteriorBlind_of_chordExact` it therefore pays the blind spot of
Prop. S7.6. -/
theorem gatedReadout_chordExact (JetStraight : Jet → Prop) [DecidablePred JetStraight]
    (chordOf : Jet → Out) (F : Jet → Ext → Out) :
    ChordExact (gatedReadout JetStraight chordOf F) JetStraight chordOf := by
  intro j hj e
  simp [gatedReadout, hj]

/-- Off the jet-straight event the gated readout is the exterior-aware readout `F` unchanged
(the gate costs nothing on curved gaps; SI §S7.2 remark after Prop. S7.6). -/
theorem gatedReadout_of_not (JetStraight : Jet → Prop) [DecidablePred JetStraight]
    (chordOf : Jet → Out) (F : Jet → Ext → Out) {j : Jet} (hj : ¬ JetStraight j) (e : Ext) :
    gatedReadout JetStraight chordOf F j e = F j e := by
  simp [gatedReadout, hj]

/-- Non-vacuity, responsive side (SI Prop. S7.6): a readout that is injective in the
exterior at `j` responds there as soon as two distinct exteriors exist; hence (by
`not_chordExact_of_respondsAt`) it is not chord-exact if `j` is jet-straight. -/
theorem respondsAt_of_injective {O : Jet → Ext → Out} {j : Jet}
    (hinj : Function.Injective (O j)) {e e' : Ext} (hee : e ≠ e') : RespondsAt O j :=
  ⟨e, e', fun h => hee (hinj h)⟩

end Abstract

section Concrete

/-- The boundary 1-jet of a scalar track on the unit gap `[0,1]`: endpoint values and endpoint
velocities. A jet is straight when the boundary values lie on a line and the included
velocities equal its slope. -/
structure Jet1 where
  /-- value at the left anchor `t = 0` -/
  pa : ℝ
  /-- value at the right anchor `t = 1` -/
  pb : ℝ
  /-- velocity at the left anchor -/
  va : ℝ
  /-- velocity at the right anchor -/
  vb : ℝ

/-- The boundary 1-jet of a track `p` on `[0,1]`. -/
noncomputable def jetOf (p : ℝ → ℝ) : Jet1 := ⟨p 0, p 1, deriv p 0, deriv p 1⟩

/-- Jet-straightness on the unit gap: both included velocities equal the chord slope `pb − pa`
(SI Thm S7.5(ii)). -/
def Jet1.Straight (j : Jet1) : Prop := j.va = j.pb - j.pa ∧ j.vb = j.pb - j.pa

/-- The chord of a jet on `[0,1]`: `A(t) = pa + t (pb − pa)` (manuscript chord atom). -/
def Jet1.chord (j : Jet1) : ℝ → ℝ := fun t => j.pa + t * (j.pb - j.pa)

/-- The parallel-tangent S-curve on the unit gap: `sin(2πu) − ½ sin(4πu)` (witness for
SI Prop. S7.6). -/
noncomputable def sCurve (u : ℝ) : ℝ := Real.sin (2 * π * u) - Real.sin (4 * π * u) / 2

/-- Derivative of the S-curve. -/
lemma hasDerivAt_sCurve (u : ℝ) :
    HasDerivAt sCurve (2 * π * Real.cos (2 * π * u) - 2 * π * Real.cos (4 * π * u)) u := by
  have h1 : HasDerivAt (fun u : ℝ => Real.sin (2 * π * u)) (Real.cos (2 * π * u) * (2 * π)) u := by
    simpa using ((hasDerivAt_id u).const_mul (2 * π)).sin
  have h2 : HasDerivAt (fun u : ℝ => Real.sin (4 * π * u)) (Real.cos (4 * π * u) * (4 * π)) u := by
    simpa using ((hasDerivAt_id u).const_mul (4 * π)).sin
  have h := h1.sub (h2.div_const 2)
  have hf : (fun u : ℝ => Real.sin (2 * π * u)) - (fun u : ℝ => Real.sin (4 * π * u) / 2) =
      sCurve := rfl
  rw [hf] at h
  convert h using 1
  ring

private lemma four_pi_one : 4 * π * 1 = ((4 : ℕ) : ℝ) * π := by push_cast; ring

private lemma four_pi_one' : 4 * π * 1 = ((2 : ℕ) : ℝ) * (2 * π) := by push_cast; ring

/-- The S-curve vanishes at the left anchor. -/
lemma sCurve_zero : sCurve 0 = 0 := by simp [sCurve]

/-- The S-curve vanishes at the right anchor. -/
lemma sCurve_one : sCurve 1 = 0 := by
  have h : Real.sin (4 * π * 1) = 0 := by rw [four_pi_one]; exact Real.sin_nat_mul_pi 4
  simp only [sCurve, h]
  simp

/-- The S-curve has zero velocity at the left anchor. -/
lemma deriv_sCurve_zero : deriv sCurve 0 = 0 := by
  rw [(hasDerivAt_sCurve 0).deriv]; simp

/-- The S-curve has zero velocity at the right anchor. -/
lemma deriv_sCurve_one : deriv sCurve 1 = 0 := by
  rw [(hasDerivAt_sCurve 1).deriv]
  have h1 : Real.cos (2 * π * 1) = 1 := by simp
  have h2 : Real.cos (4 * π * 1) = 1 := by rw [four_pi_one']; exact Real.cos_nat_mul_two_pi 2
  rw [h1, h2]; ring

/-- The S-curve really is an S: it reaches `+1` at `u = 1/4` ... -/
lemma sCurve_quarter : sCurve (1 / 4) = 1 := by
  have h1 : 2 * π * (1 / 4) = π / 2 := by ring
  have h2 : 4 * π * (1 / 4) = π := by ring
  simp only [sCurve, h1, h2, Real.sin_pi_div_two, Real.sin_pi]
  ring

/-- ... and `−1` at `u = 3/4` (opposite lateral excursions: a parallel-tangent S-curve). -/
lemma sCurve_three_quarter : sCurve (3 / 4) = -1 := by
  have h1 : 2 * π * (3 / 4) = π / 2 + π := by ring
  have h2 : 4 * π * (3 / 4) = π + 2 * π := by ring
  simp only [sCurve, h1, h2, Real.sin_add_pi, Real.sin_pi_div_two, Real.sin_add_two_pi,
    Real.sin_pi]
  ring

/-- A line `t ↦ α + β t` on the unit gap. -/
def lineTrack (α β : ℝ) : ℝ → ℝ := fun t => α + β * t

/-- The S-curve track: a line plus an `ε`-scaled parallel-tangent S-curve. -/
noncomputable def sTrack (α β ε : ℝ) : ℝ → ℝ := fun t => α + β * t + ε * sCurve t

/-- A line has constant velocity `β`. -/
lemma hasDerivAt_lineTrack (α β t : ℝ) : HasDerivAt (lineTrack α β) β t := by
  have h := ((hasDerivAt_id t).const_mul β).const_add α
  rw [mul_one] at h
  exact h

/-- **SI Prop. S7.6: the boundary 1-jet cannot see the S-curve.** The S-curve
track has exactly the boundary 1-jet of the line. -/
theorem jetOf_sTrack (α β ε : ℝ) : jetOf (sTrack α β ε) = jetOf (lineTrack α β) := by
  have hd : ∀ t, HasDerivAt (sTrack α β ε)
      (β + ε * (2 * π * Real.cos (2 * π * t) - 2 * π * Real.cos (4 * π * t))) t := fun t =>
    (hasDerivAt_lineTrack α β t).add ((hasDerivAt_sCurve t).const_mul ε)
  have h0 : deriv (sTrack α β ε) 0 = β := by
    rw [(hd 0).deriv, ← (hasDerivAt_sCurve 0).deriv, deriv_sCurve_zero]; ring
  have h1 : deriv (sTrack α β ε) 1 = β := by
    rw [(hd 1).deriv, ← (hasDerivAt_sCurve 1).deriv, deriv_sCurve_one]; ring
  simp only [jetOf, h0, h1, (hasDerivAt_lineTrack α β 0).deriv,
    (hasDerivAt_lineTrack α β 1).deriv]
  simp [sTrack, lineTrack, sCurve_zero, sCurve_one]

/-- The jet of a line is straight. -/
theorem jetOf_lineTrack_straight (α β : ℝ) : (jetOf (lineTrack α β)).Straight := by
  simp only [Jet1.Straight, jetOf, (hasDerivAt_lineTrack α β 0).deriv,
    (hasDerivAt_lineTrack α β 1).deriv]
  constructor <;> simp [lineTrack]

/-- The chord of a line's jet is the line itself (SI Thm S7.5(i) on the unit gap). -/
theorem chord_jetOf_lineTrack (α β : ℝ) : (jetOf (lineTrack α β)).chord = lineTrack α β := by
  funext t
  simp [Jet1.chord, jetOf, lineTrack, mul_comm]

/-- The S-curve track is a genuinely different trajectory from the line (for `ε ≠ 0`). -/
theorem sTrack_ne_lineTrack (α β : ℝ) {ε : ℝ} (hε : ε ≠ 0) : sTrack α β ε ≠ lineTrack α β := by
  intro h
  have := congrFun h (1 / 4)
  simp only [sTrack, lineTrack, sCurve_quarter, mul_one, add_eq_left] at this
  exact hε this

/-- **SI Prop. S7.6, concrete form.** Every readout that is
chord-exact on jet-straight unit gaps (for every exterior) returns, on the parallel-tangent S-curve
track `lineTrack α β + ε • sCurve` (`ε ≠ 0`), the straight line — never the true S-curve —
whatever the
exterior data `e` (which may carry the S-curve's signature). -/
theorem chordExact_misses_sCurve {Ext : Type*} {O : Jet1 → Ext → (ℝ → ℝ)}
    (hO : ChordExact O Jet1.Straight Jet1.chord) (α β : ℝ) {ε : ℝ} (hε : ε ≠ 0) (e : Ext) :
    O (jetOf (sTrack α β ε)) e = lineTrack α β ∧ O (jetOf (sTrack α β ε)) e ≠ sTrack α β ε := by
  have hj : O (jetOf (sTrack α β ε)) e = lineTrack α β := by
    rw [jetOf_sTrack, hO _ (jetOf_lineTrack_straight α β) e, chord_jetOf_lineTrack]
  exact ⟨hj, hj ▸ (sTrack_ne_lineTrack α β hε).symm⟩

/-- **The blind spot is not information-theoretically forced.** An
exterior-aware readout can be exact on both the straight transit and the S-curve when the exterior
distinguishes them — but then, by `not_chordExact_of_respondsAt`, it is not chord-exact on
jet-straight gaps. Witness: the "oracle" readout that returns the exterior track itself
(`O j e = e`) is exact on both tracks and is not chord-exact. -/
theorem exterior_oracle_trade (α β : ℝ) {ε : ℝ} (hε : ε ≠ 0) :
    (fun (_ : Jet1) (e : ℝ → ℝ) => e) (jetOf (sTrack α β ε)) (sTrack α β ε) = sTrack α β ε ∧
      (fun (_ : Jet1) (e : ℝ → ℝ) => e) (jetOf (lineTrack α β)) (lineTrack α β) = lineTrack α β ∧
      ¬ ChordExact (fun (_ : Jet1) (e : ℝ → ℝ) => e) Jet1.Straight Jet1.chord := by
  refine ⟨rfl, rfl, not_chordExact_of_respondsAt (jetOf_lineTrack_straight α β) ?_⟩
  exact ⟨lineTrack α β, sTrack α β ε, (sTrack_ne_lineTrack α β hε).symm⟩

end Concrete

end FQGNN.Readout

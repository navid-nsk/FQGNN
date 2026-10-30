import FQGNN.Backbone.SpectrumModule

/-!
# The ℚ-rank of the extinction-thinned spectrum `Λ'`

`FQGNN.Backbone.SpectrumModule` proves the ℚ-rank theorem for the **ambient** module
`M_L = Lᵀℤⁿ` (`finrank_span_spectrumModule_le`, `finrank_span_spectrumModule_eq_iff`). SI §S1.5(e)
states it for the **thinned** support itself: "the amplitude-carrying spectral support
`Λ' = {Lᵀk : var(k) < d}` is a discrete, nonlattice set whose rational span may have dimension up
to `n` (with `dim_ℚ = n` exactly under ℚ-independence of the rows of `L` [AKKV2024, Thm 2.6(2)])".
This module proves it for `Λ'`: the thinning does not change the rational span, because every
standard basis vector `e_j` has `var(e_j) = 0 < d`, so every generator row `ℓ_j = Lᵀe_j` survives.

Sources:
* SI §S1.5(e), the "[PROVEN — and a distinction we rely on …]" paragraph: `dim_ℚ Λ' ≤ n`,
  `= n` iff the rows are ℚ-independent.
* Manuscript eq. (1) (`Λ' = {Lᵀk : var(k) < d}`).

Main results:
* `signVar_single_one` — `var(e_j) = 0`.
* `row_mem_extinctionSpectrum` — `ℓ_j ∈ Λ'` for `d > 0`.
* `span_extinctionSpectrum_eq` — `span_ℚ Λ' = span_ℚ M_L` (for every `d`, including `d = 0`
  where both are the zero space).
* `qRank_thinned_le`, `qRank_thinned_eq_iff` — `dim_ℚ span_ℚ Λ' ≤ n`, with equality iff the rows
  of `L` are ℚ-linearly independent.
-/

noncomputable section

open Matrix

namespace FQGNN.Backbone

variable {n d : ℕ}

/-- A list of nonnegative integers has no sign changes.

Source: auxiliary for SI §S1.5(e) (`var(k)` of manuscript eq. (1)). -/
theorem signChanges_eq_zero_of_nonneg : ∀ l : List ℤ, (∀ a ∈ l, 0 ≤ a) → signChanges l = 0
  | a :: b :: l, h => by
      have ha : 0 ≤ a := h a (by simp)
      have hb : 0 ≤ b := h b (by simp)
      have ih := signChanges_eq_zero_of_nonneg (b :: l) fun x hx => h x (List.mem_cons_of_mem _ hx)
      have hab : ¬ a * b < 0 := not_lt.2 (mul_nonneg ha hb)
      simp only [signChanges, hab, ite_false, ih]
  | [_], _ => rfl
  | [], _ => rfl

/-- `var(k) = 0` for every `k ∈ ℤⁿ` with nonnegative entries.

Source: auxiliary for SI §S1.5(e); manuscript eq. (1). -/
theorem signVar_eq_zero_of_nonneg {k : Fin n → ℤ} (hk : ∀ j, 0 ≤ k j) : signVar k = 0 := by
  unfold signVar
  apply signChanges_eq_zero_of_nonneg
  intro a ha
  obtain ⟨i, rfl⟩ := List.mem_ofFn.1 (List.mem_of_mem_filter ha)
  exact hk i

/-- `var(e_j) = 0` for a standard basis vector `e_j ∈ ℤⁿ`.

Source: SI §S1.5(e) (the generator rows `ℓ_j = Lᵀe_j` are not thinned out). -/
theorem signVar_single_one (j : Fin n) : signVar (Pi.single j (1 : ℤ)) = 0 := by
  apply signVar_eq_zero_of_nonneg
  intro i
  by_cases h : i = j
  · subst h
    simp
  · simp [h]

/-- **Every generator row survives the thinning.** For `d > 0`, `ℓ_j = Lᵀe_j ∈ Λ'`.

Source: SI §S1.5(e); manuscript eq. (1). -/
theorem row_mem_extinctionSpectrum (L : Matrix (Fin n) (Fin d) ℝ) (hd : 0 < d) (j : Fin n) :
    L.row j ∈ extinctionSpectrum L := by
  refine ⟨Pi.single j 1, by rw [signVar_single_one]; exact hd, ?_⟩
  rw [transpose_mulVec_intCast]
  simp [Pi.single_apply]

/-- **`span_ℚ Λ' = span_ℚ M_L`.** The extinction thinning `var(k) < d` does not change the
rational span of the spectrum. No hypothesis on `d` is needed: for `d > 0` every row lies in
`Λ'` (`row_mem_extinctionSpectrum`); for `d = 0` the ambient space `ℝ⁰` is trivial.

Source: SI §S1.5(e) ("`Λ'` … whose rational span may have dimension up to `n`"). -/
theorem span_extinctionSpectrum_eq (L : Matrix (Fin n) (Fin d) ℝ) :
    Submodule.span ℚ (extinctionSpectrum L) =
      Submodule.span ℚ (spectrumModule L : Set (Fin d → ℝ)) := by
  rcases Nat.eq_zero_or_pos d with hd | hd
  · subst hd
    exact Subsingleton.elim _ _
  apply le_antisymm
  · exact Submodule.span_mono (extinctionSpectrum_subset L)
  · rw [span_spectrumModule]
    apply Submodule.span_mono
    rintro _ ⟨j, rfl⟩
    exact row_mem_extinctionSpectrum L hd j

/-- `span_ℚ Λ' = span_ℚ {ℓ_1, …, ℓ_n}`.

Source: SI §S1.5(e). -/
theorem span_extinctionSpectrum_eq_rows (L : Matrix (Fin n) (Fin d) ℝ) :
    Submodule.span ℚ (extinctionSpectrum L) = Submodule.span ℚ (Set.range L.row) := by
  rw [span_extinctionSpectrum_eq, span_spectrumModule]

/-- **ℚ-rank bound for the thinned spectrum.** `dim_ℚ span_ℚ Λ' ≤ n`.

Source: SI §S1.5(e) ("rational span may have dimension up to `n`"); cf. [AKKV2024, Thm 2.6(2)]. -/
theorem qRank_thinned_le (L : Matrix (Fin n) (Fin d) ℝ) :
    Module.finrank ℚ (Submodule.span ℚ (extinctionSpectrum L)) ≤ n := by
  rw [span_extinctionSpectrum_eq]
  exact finrank_span_spectrumModule_le L

/-- **ℚ-rank equality for the thinned spectrum.** `dim_ℚ span_ℚ Λ' = n` iff the rows of `L` are
ℚ-linearly independent.

Source: SI §S1.5(e) ("with `dim_ℚ = n` exactly under ℚ-independence of the rows of `L`
[AKKV2024, Thm 2.6(2)]"). -/
theorem qRank_thinned_eq_iff (L : Matrix (Fin n) (Fin d) ℝ) :
    Module.finrank ℚ (Submodule.span ℚ (extinctionSpectrum L)) = n ↔
      LinearIndependent ℚ L.row := by
  rw [span_extinctionSpectrum_eq]
  exact finrank_span_spectrumModule_eq_iff L

/-- **ℚ-rank equality, integer-relation form.** `dim_ℚ span_ℚ Λ' = n` iff no nonzero `k ∈ ℤⁿ`
has `Lᵀk = 0` (the form of SI §S1.1).

Source: SI §S1.5(e) and §S1.1. -/
theorem qRank_thinned_eq_iff_int (L : Matrix (Fin n) (Fin d) ℝ) :
    Module.finrank ℚ (Submodule.span ℚ (extinctionSpectrum L)) = n ↔
      ∀ k : Fin n → ℤ, Lᵀ *ᵥ (fun j => (k j : ℝ)) = 0 → k = 0 := by
  rw [qRank_thinned_eq_iff, linearIndependent_rat_iff]

end FQGNN.Backbone

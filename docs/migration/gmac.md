---
audience: public
---

# GMAC: differences from the MATLAB scripts

GMAC (`update_strategies["gmac"]`) is a generalised GIES update. It has no PET equivalent; its sources are the paper, X. Luo (2021). Novel iterative ensemble smoothers derived from a class of generalized cost functions. *Computational Geosciences* 25(3), 1159–1189. [doi:10.1007/s10596-021-10046-1](https://doi.org/10.1007/s10596-021-10046-1) (Appendices A, C and E), and the authors' MATLAB scripts, which are not public. C-GIES's soft constraints follow X. Luo and W. C. Cruz (2022). Data assimilation with soft constraints (DASC) through a generalized iterative ensemble smoother. *Computational Geosciences* 26(3), 571–594. [doi:10.1007/s10596-022-10137-7](https://doi.org/10.1007/s10596-022-10137-7)

## Validation

| Check | Reference | Tolerance | Measured |
|---|---|---|---|
| One identity l2² term = `rlmmac` | `RLMMACUpdate` (its PET parity case), unlocalised and with a distance localization, and a whole GIES run (`tests/workflows/test_gmac.py`) | bit for bit | equal |
| The MATLAB block, codes 10100, 11000, 10001, 01100, 00010 (01100 and 00010 at `tsvdCut = 0.9`) | `tools/gmac_reference/generate.py`, a line-by-line NumPy transcription of `RLM_MAC_GIES.m`'s per-member block; stored as `tests/data/reference/gmac/*.npz` and `.mat` | 1e-10 | 1.1e-15 to 4.7e-15 × max\|step\| |
| The subspace form | `dX (M_D + γ_j M_j)⁻¹ Sᵀ L⁻¹(d_j − y_j)` (full-rank metric), and on the range of a singular one | 1e-8, 1e-7 | |
| The direct form | `RLM_MAC_GIES_backup.m`'s closed form, α = λ tr(M_D)/tr(M_j) | 1e-8 | |
| The l_p^q Hessians | finite differences of the analytic gradients; l1² as l_p² at p = 1 and its limit p → 1⁺ | 1e-6 | |
| Terms on properties | a hand computation: TV of q = x² in secants about P(m̄); `state` and `histogram_entropy` about a given centre; `StackedProperties` against a `PropertyMap` with two outputs; a GIES run with a `state` l1² term on a parameterized property | 1e-12 | |
| Per-member localization | an untapered taper equals the unlocalized per-member step (both forms); equal gains equal `localized_step` (`distance`, `adaptive_per_update`); a localized GIES run resumes bit for bit | 1e-9, 1e-12, exact | |

The `.mat` files hold the block's inputs (`in_ensemble` and `in_sim_data` with the mean model appended, whitened) so that the block can be rerun in MATLAB elsewhere.

## Differences, recorded rather than reproduced

| MATLAB | APEX | Why |
|---|---|---|
| `getHistEntropy.m` : `histogram(x, 100)` bins each sample on its own edges | fixed bins per variable over the ensemble and the mean (`ranges` to fix them) | per-sample bins make the members' and the mean's features incomparable bin by bin |
| `M_k ne / trace(M_k)` for every term, also those with coefficient 0 (`RLM_MAC_GIES.m:237-241`): a zero trace gives NaN | a term of zero trace contributes nothing | NaN would end the run |
| `M_R` cut by the data's `tsvdCut` energy rule (245-255) | with an identity term, a relative eigenvalue floor (`metric_floor = 1e-10`); without one, MATLAB's rule at 0.99 (`metric_energy`, also `>`), and the floor always | with an identity term, `M_R` has N − 1 equal eigenvalues: the energy rule drops an arbitrary few of them for N > 100. Without one, a floor alone keeps eigenvalues ~1e-10 of the largest, whose μ^{-1/2}-scaled directions then dominate the TSVD and γ (the step would be 50× smaller than MATLAB's on smooth fields) |
| the data TSVD cut by `share > tsvdCut` | `energy` keeps the first rank whose share *reaches* the cut (PET's `truncated_svd`, as `rlmmac`) | they differ only at an exact tie |
| `respectBounds` on `[members, mean(members)]`: the mean model is clip(mean) | the mean model is the mean of the clipped members, as `rlmmac` | APEX's GIES loop |
| `M_R_TVL1 = tv' sign(tv_i tv_iᵀ) tv`, an n_s × n_s sign matrix per member | `2 v vᵀ` with `v = Aᵀ sgn(t_j)` | the same matrix (rank one), without the n_s × n_s array (3,960² in a 2-D facies case) |
| `RLM_MAC_HistL1.m` computes Hist l2² | `histogram_entropy` with `l2sq` | its name is misleading |
| enssubspace: TV l2² not trace-normalised (238) | every term's weight is `trace` or `fixed`, as configured | the script's other terms are normalised |
| `getTV2D.m` stacks `[dx; dy]` for 2-D grids | differences along every axis longer than one cell (3-D grids), Eclipse order | the metric does not depend on the features' order |
| perturbations redrawn per call (85-88) | drawn once, as `rlmmac` | APEX's GIES loop |
| the "wavelet" scripts (contourlet coefficients) | not ported: a later parameterization | |

**Fixed weights.** A `fixed` weight multiplies half the Hessian of features whose anomalies are scaled by 1/√(N − 1), as the state's (`identity` with value w is w I, MATLAB's `NLS_coeff_weight * eye(ne)`). MATLAB's `tv_delta_m` and `hist_delta_m` are unscaled, so a MATLAB fixed coefficient c on such a term (the enssubspace script's non-normalised TV l2²) is APEX's `value = c (N − 1)` for l2² (APEX: ½ · value · 2 AᵀA = value DᵀD/(N − 1), MATLAB: c DᵀD). Trace weights are unaffected.

Known hazards, as in MATLAB:
- a trace weight normalises a term whatever its size, so an l1² term of features that differ only by rounding (sgn of ±1e-16) is scaled up to its full weight. Members that are exactly equal in a term's features contribute nothing;
- `lpq` with 1 < p < 2 is singular at 0: exact zeros get the curvature (`floor` · max|x|)^{p−2}, so `floor` decides their weight.

The scripts' other issues (`isfield('options',' writePerm')` always false, "terminating" without a break, the 5Spot `getHistEntropy`'s `inputVec`, the 5Spot `autoAdaLoc` given N + 1 columns) are outside the update and do not reach APEX.


## C-GIES: soft constraints

Sources: X. Luo and W. C. Cruz (2022). Data assimilation with soft constraints (DASC) through a generalized iterative ensemble smoother. *Computational Geosciences* 26(3), 571–594. [doi:10.1007/s10596-022-10137-7](https://doi.org/10.1007/s10596-022-10137-7); and the authors' MATLAB scripts.

| Check | Reference | Tolerance | Measured |
|---|---|---|---|
| The MATLAB update, IN, EQ, IN+EQ (`curvature = "paper"`, direct form, code 100) | `tools/gmac_reference/generate_cgies.py`, a transcription of `GIES_constraint_ensLoc.m` lines 200-315 without `autoAdaLoc`; `tests/data/reference/gmac/cgies_*` (`.npz`, `.mat`) | 1e-10 | 1.2e-15 to 3.0e-15 × max\|step\| |
| A linear constraint, quadratic metric, fixed weight w | GMAC on the observations augmented by (f, 0, I/w): a perfect measurement | 1e-9 | both forms |
| Gradients and Hessians | finite differences of the costs and gradients | 1e-6 | |
| No constraints | `gmac` unchanged | bit for bit | |

| MATLAB / paper | APEX | Why |
|---|---|---|
| The channel's Hessian is negative and the trace weight α = w tr(M_D)/tr(M_eq) negative: the term maximises \|f_eq\| (a toy: 1.02 → 1.31) | `log_channel` `curvature = "absolute_hessian"` (the default): \|H\| and a positive weight; `"paper"`: as written; `"absolute_trace"`: H < 0 with \|tr\| | the paper's form maximises \|f_eq\|; the absolute form pulls toward the constraint, and performed best in tests |
| The barrier at a member beyond the box by a or more: log undefined, the gradient pushes outward; the scripts rely on clipping first | the update is refused, naming the constraint, members, worst violation, causes and fixes | an undefined barrier must not be evaluated |
| `compare_hist`: `hist(m, 50)` bins each member over its own range | fixed bins over `range`, the same for every member and the reference | like with like |
| ε: inside the reciprocal (scripts), outside (the paper's Hessian of the channel function) | inside, as the scripts | the reference transcription |
| Without `autoAdaLoc`, `GIES_constraint_ensLoc.m` uses a stale `tapering_matrix_IN` | no localization: plain step | a script bug |
| Localization: `autoAdaLoc` per source on `coeff × innovation` | `adaptive_per_update`'s `localized_source_steps`, one taper per source | the same scheme; its random shuffle draws from APEX's stream |

---
audience: public
generated: reference
---

# Update strategies: reference

Every one of these 14 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `adaptive_rows`

Adaptive localization per state row, recomputed at every update.

Select it with:

```toml
[method.strategy]
name = "adaptive_rows"
```

| option | type | default | description |
|---|---|---|---|
| `energy` | number | 0.98 | Fraction of the singular-value sum of each group's whitened data anomalies kept, in `(0, 1]` (default 0.98, as `approx`). |
| `max_batch_mb` | number | 256.0 | Memory budget of one batch's `(rows, n_data)` correlations. |
| `threshold` | number or null | | Correlation threshold in `[0, 1)`; default `3 / sqrt(N)` (refused when that is 1 or more, below 10 members). |

Methods that take it: `es`, `esmda`, `gn_enrml`, `lm_enrml`.

| refused by | why |
|---|---|
| `chop` | GIES runs local analysis with RLM-MAC: use strategy local_rlmmac (the approx local_analysis and adaptive_rows run with es, esmda, lm_enrml and gn_enrml; adaptive_rows has no RLM-MAC variant) |
| `gies` | GIES runs local analysis with RLM-MAC: use strategy local_rlmmac (the approx local_analysis and adaptive_rows run with es, esmda, lm_enrml and gn_enrml; adaptive_rows has no RLM-MAC variant) |
| `multilevel_esmda` | multilevel ES-MDA runs the hybrid update only (PET's), not AdaptiveRowsUpdate: leave out [method.strategy], or name hybrid |

## `approx`

Approximate (TSVD) Levenberg-Marquardt update, the ES/ES-MDA analysis at `damping=0`.

Select it with:

```toml
[method.strategy]
name = "approx"
```

| option | type | default | description |
|---|---|---|---|
| `empirical_data_error` | boolean | false | PET's `emp_cov`: use the perturbations' own covariance `E E^T` (`E = L^-1 D Pi`) for `C_d` in the inversion, in the subspace of `U`: with `X0 = s^-1 U^T E`, `X0 X0^T = Q Lambda Q^T` and `d = (1 + lambda) Lambda + 1`, the step is `dX V Q d^-1 Q^T s^-1 U^T L^-1 (D - Y)` (PET `approx_update`, `e65fb10`). Its numerically null singular values are dropped (they would be divided by). PET's distance-localized branch multiplies the anomalies by `sqrt(N - 1)` here, unlike its unlocalized step; APEX uses the same gain factor for both. |
| `energy` | number | 0.98 | Fraction of the singular-value sum of `dY_w` to retain, in `(0, 1]` (PET keyword `energy`, default 0.98). |
| `sensitivities` | boolean | false | PET's adjoint branch: the data anomalies are the simulator's member-mean sensitivity times the state anomalies, `dY = mean_j(dg/dm)_j dX`, in place of the forecast's own. The method requests the sensitivity with each forecast and passes it as `sensitivity` `(n_data, n_state)`. Not with localization. |

Methods that take it: `es`, `esmda`, `gies`, `gn_enrml`, `lm_enrml`.

| refused by | why |
|---|---|
| `chop` | [method] chop: CHOP needs an RLM-MAC strategy (rlmmac), whose gain it tapers per member and whose mean-model forecast its hyper-parameter update uses; got ApproxUpdate |
| `multilevel_esmda` | multilevel ES-MDA runs the hybrid update only (PET's), not ApproxUpdate: leave out [method.strategy], or name hybrid |

## `ert_adaptive`

ERT's adaptive localization (`AdaptiveESMDA`).

Select it with:

```toml
[method.strategy]
name = "ert_adaptive"
```

| option | type | default | description |
|---|---|---|---|
| `max_batch_mb` | number | 256.0 | Memory budget of one batch of state rows. |
| `n_jobs` | integer | 1 | Worker processes of the library's per-row-pattern inversions (default 1: the analysis stays within the run's own core). |
| `threshold` | number or null | | Correlation threshold; default ERT's `3 / sqrt(N)`, which keeps nothing below ten members (refused there). |
| `truncation` | number | 0.99 | Fraction of the singular values kept in the inversion, in `(0, 1]`. |

Methods that take it: `es`, `esmda`, `gn_enrml`.

| refused by | why |
|---|---|
| `chop` | GIES damps its updates, which the ErtAdaptive analysis does not support: use it with es, esmda or gn_enrml |
| `gies` | GIES damps its updates, which the ErtAdaptive analysis does not support: use it with es, esmda or gn_enrml |
| `lm_enrml` | LMEnRML damps its updates, which the ErtAdaptive analysis does not support: use it with es, esmda or gn_enrml |
| `multilevel_esmda` | multilevel ES-MDA runs the hybrid update only (PET's), not ErtAdaptive: leave out [method.strategy], or name hybrid |

## `ert_esmda`

ERT's ES-MDA analysis (`iterative_ensemble_smoother.ESMDA`).

Select it with:

```toml
[method.strategy]
name = "ert_esmda"
```

| option | type | default | description |
|---|---|---|---|
| `max_batch_mb` | number | 256.0 | Memory budget of one batch of state rows. |
| `truncation` | number | 0.99 | Fraction of the singular values kept in the inversion, in `(0, 1]`. |

Methods that take it: `es`, `esmda`, `gn_enrml`.

| refused by | why |
|---|---|
| `chop` | GIES damps its updates, which the ErtEsmda analysis does not support: use it with es, esmda or gn_enrml |
| `gies` | GIES damps its updates, which the ErtEsmda analysis does not support: use it with es, esmda or gn_enrml |
| `lm_enrml` | LMEnRML damps its updates, which the ErtEsmda analysis does not support: use it with es, esmda or gn_enrml |
| `multilevel_esmda` | multilevel ES-MDA runs the hybrid update only (PET's), not ErtEsmda: leave out [method.strategy], or name hybrid |

## `ert_localized`

ERT's localized ES-MDA (`LocalizedESMDA`): the Kalman gain times a given taper.

Select it with:

```toml
[method.strategy]
name = "ert_localized"
```

| option | type | default | description |
|---|---|---|---|
| `max_batch_mb` | number | 256.0 | Memory budget of one batch of state rows. |
| `taper_file` | string or string | required | `.npy`, or `.npz` with `taper`: the taper `(n_state, n_data)` in `[0, 1]`, state rows in layout order and data columns in the analysis data's order (only the shape can be checked: build it for the configuration it is used with). |
| `truncation` | number | 0.99 | Fraction of the singular values kept in the inversion, in `(0, 1]`. |

Methods that take it: `es`, `esmda`, `gn_enrml`.

| refused by | why |
|---|---|
| `chop` | GIES damps its updates, which the ErtLocalized analysis does not support: use it with es, esmda or gn_enrml |
| `gies` | GIES damps its updates, which the ErtLocalized analysis does not support: use it with es, esmda or gn_enrml |
| `lm_enrml` | LMEnRML damps its updates, which the ErtLocalized analysis does not support: use it with es, esmda or gn_enrml |
| `multilevel_esmda` | multilevel ES-MDA runs the hybrid update only (PET's), not ErtLocalized: leave out [method.strategy], or name hybrid |

## `full`

Full LM update with the prior-increment term; needs each member's prior state.

Select it with:

```toml
[method.strategy]
name = "full"
```

| option | type | default | description |
|---|---|---|---|
| `block_sizes` | array or null | | Sizes of the state variables, in state order (the composition passes them), for the per-variable default scaling; one block without them. |
| `energy` | number | 0.98 | Fraction of the singular-value sum kept, for both the data anomalies and the prior anomalies of `Am`, in `(0, 1]` (PET `energy`). The default, 0.98, is the approx update's; PET's LM-EnRML uses 0.95, which `apex pet-import` passes. |
| `state_scaling` | ArrayLike or null | | Explicit scaling per state row `(n_state,)` (PET's `state_scaling`), or an `.npz` holding it as `scaling` (`apex pet-import` writes one), in place of the prior-ensemble standard deviation per variable. |
| `state_scaling_file` | string or string or null | | Explicit scaling per state row `(n_state,)` (PET's `state_scaling`), or an `.npz` holding it as `scaling` (`apex pet-import` writes one), in place of the prior-ensemble standard deviation per variable. |

Methods that take it: `gies`, `gn_enrml`, `lm_enrml`.

| refused by | why |
|---|---|
| `chop` | [method] chop: CHOP needs an RLM-MAC strategy (rlmmac), whose gain it tapers per member and whose mean-model forecast its hyper-parameter update uses; got FullUpdate |
| `es` | ES with the FullUpdate analysis is not supported: it includes the prior-increment term of an iterative method (each member pulled back towards its prior), and ES-MDA's steps have none, each step updating the current ensemble; use the approx strategy, or this one with gies, lm_enrml or gn_enrml |
| `esmda` | ESMDA with the FullUpdate analysis is not supported: it includes the prior-increment term of an iterative method (each member pulled back towards its prior), and ES-MDA's steps have none, each step updating the current ensemble; use the approx strategy, or this one with gies, lm_enrml or gn_enrml |
| `multilevel_esmda` | multilevel ES-MDA runs the hybrid update only (PET's), not FullUpdate: leave out [method.strategy], or name hybrid |

## `gmac`

GMAC: GIES's update with per-member step metrics from composable terms.

Select it with:

```toml
[method.strategy]
name = "gmac"
```

| option | type | default | description |
|---|---|---|---|
| `code` | string or null | | The paper's shorthand instead of `terms` (e.g. `"10100"`). Without `terms` and `code` the metric is one identity `l2sq` term (`code = "100"`): `rlmmac`. |
| `constraints` | array or null | | C-GIES's soft constraints: `{function = {name, options}, metric = {name, options}, weight = {rule = "trace", w} \| {rule = "fixed", value}}`, with functions of `gmac_constraints` and metrics of `gmac_constraint_metrics` (default `log_barrier` for inequalities, `log_channel` for equalities). The regularisation defaults to the identity (`code = "100"`). `metric_rank` then holds the rank kept of `F^T M_eff F`. |
| `energy` | number | 0.95 | Fraction of the singular-value sum of each `S F_j` kept (as `rlmmac`). |
| `form` | string | subspace | `subspace` (the paper's algorithm) or `direct` (`M_D + alpha M_j`). |
| `localize_constraints` | boolean | true | With a localization, taper the constraint terms too, each with its own taper (the paper's §2.3; default), or only the data's (the constraints observe the model itself: their correlation tapers cannot see a pull shared by every member, e.g. a box's toward its centre; the user's option, 2026-10-05). |
| `metric_energy` | number or null | | Also keep only the eigenvalues up to the first whose cumulative share exceeds this fraction of their sum (MATLAB's `> tsvdCut`), in `(0, 1)`. Default: 0.99 when no term is `identity` (nothing bounds such a metric below, and its smallest eigenvalues would dominate the step), else off: with the I term the metric has many equal eigenvalues (all N with I alone), of which the rule would drop an arbitrary few once N > 100. The direct form uses none of `energy`, `metric_floor` and `metric_energy`: it inverts `M_D + alpha M_j` whole. |
| `metric_floor` | number | 1e-10 | Eigenvalues of `M_j` below this fraction of the largest are always dropped (subspace form). |
| `terms` | array or null | | The metric terms: `{transform = {name, options}, norm = {name, options}, weight = {rule = "trace", alpha} \| {rule = "fixed", value}}`. |
| `variables` | array or null | | The variables the code's TV and histogram terms use (the properties' names with a parameterization); default all (gridded, for TV). |

Methods that take it: `gies`, `lm_enrml`.

| refused by | why |
|---|---|
| `chop` | [method] chop: CHOP needs an RLM-MAC strategy (rlmmac), whose gain it tapers per member and whose mean-model forecast its hyper-parameter update uses; got GMACUpdate |
| `es` | ES with the GMACUpdate analysis is not supported: it needs the forecast at the mean state, which only the methods built on GIES's loop make; use the approx strategy, or this one with gies or lm_enrml |
| `esmda` | ESMDA with the GMACUpdate analysis is not supported: it needs the forecast at the mean state, which only the methods built on GIES's loop make; use the approx strategy, or this one with gies or lm_enrml |
| `gn_enrml` | GN-EnRML with the GMACUpdate analysis is not supported: it runs at damping 0, which leaves RLM-MAC unregularised (gamma = lambda * mean(s^2) = 0); use approx, full, subspace, subspace2 or margis, or RLM-MAC with gies or lm_enrml |
| `multilevel_esmda` | multilevel ES-MDA runs the hybrid update only (PET's), not GMACUpdate: leave out [method.strategy], or name hybrid |

## `hybrid`

The multilevel hybrid update: one gain from every fidelity level's anomalies.

Select it with:

```toml
[method.strategy]
name = "hybrid"
```

| option | type | default | description |
|---|---|---|---|
| `energy` | number | 0.98 | Fraction of the singular-value sum kept per level, in `(0, 1]` (default 0.98). |

Methods that take it: `multilevel_esmda`.

| refused by | why |
|---|---|
| `chop` | the hybrid update needs fidelity levels: use multilevel_esmda |
| `es` | the hybrid update needs fidelity levels: use method multilevel_esmda |
| `esmda` | the hybrid update needs fidelity levels: use method multilevel_esmda |
| `gies` | the hybrid update needs fidelity levels: use multilevel_esmda |
| `gn_enrml` | the hybrid update needs fidelity levels: use multilevel_esmda |
| `lm_enrml` | the hybrid update needs fidelity levels: use multilevel_esmda |

## `local_analysis`

Local analysis: groups of state updated with their locally selected data.

Select it with:

```toml
[method.strategy]
name = "local_analysis"
```

| option | type | default | description |
|---|---|---|---|
| `block_sizes` | array or null | | State variable sizes (the composition passes them), for the default selection. |
| `energy` | number | 0.98 | Fraction of the singular-value sum of each group's whitened data anomalies kept, in `(0, 1]` (as the approx update's, default 0.98). |
| `weight` | boolean | false | Also multiply each local increment by the taper `T` (default: use only its support). |

Methods that take it: `es`, `esmda`, `gn_enrml`, `lm_enrml`.

| refused by | why |
|---|---|
| `chop` | GIES runs local analysis with RLM-MAC: use strategy local_rlmmac (the approx local_analysis and adaptive_rows run with es, esmda, lm_enrml and gn_enrml; adaptive_rows has no RLM-MAC variant) |
| `gies` | GIES runs local analysis with RLM-MAC: use strategy local_rlmmac (the approx local_analysis and adaptive_rows run with es, esmda, lm_enrml and gn_enrml; adaptive_rows has no RLM-MAC variant) |
| `multilevel_esmda` | multilevel ES-MDA runs the hybrid update only (PET's), not LocalAnalysisUpdate: leave out [method.strategy], or name hybrid |

## `local_rlmmac`

Local analysis with GIES's RLM-MAC update in each group (for GIES).

Select it with:

```toml
[method.strategy]
name = "local_rlmmac"
```

| option | type | default | description |
|---|---|---|---|
| `block_sizes` | array or null | | State variable sizes (set from the state layout), for the default selection. |
| `energy` | number | 0.95 | Fraction of the singular-value sum of each group's whitened data anomalies kept, in `(0, 1]` (default 0.95, as GIES's). |
| `weight` | boolean | false | Also multiply each local increment by the selection's taper (default false: the taper only decides which data each group uses). |

Methods that take it: `gies`, `lm_enrml`.

| refused by | why |
|---|---|
| `chop` | [method] chop: CHOP needs an RLM-MAC strategy (rlmmac), whose gain it tapers per member and whose mean-model forecast its hyper-parameter update uses; got LocalRLMMACUpdate |
| `es` | ES with the LocalRLMMACUpdate analysis is not supported: it needs the forecast at the mean state, which only the methods built on GIES's loop make; use the approx strategy, or this one with gies or lm_enrml |
| `esmda` | ESMDA with the LocalRLMMACUpdate analysis is not supported: it needs the forecast at the mean state, which only the methods built on GIES's loop make; use the approx strategy, or this one with gies or lm_enrml |
| `gn_enrml` | GN-EnRML with the LocalRLMMACUpdate analysis is not supported: it runs at damping 0, which leaves RLM-MAC unregularised (gamma = lambda * mean(s^2) = 0); use approx, full, subspace, subspace2 or margis, or RLM-MAC with gies or lm_enrml |
| `multilevel_esmda` | multilevel ES-MDA runs the hybrid update only (PET's), not LocalRLMMACUpdate: leave out [method.strategy], or name hybrid |

## `margis`

Marginalised IES of Stordal et al. (2023), PET's `margis`.

Select it with:

```toml
[method.strategy]
name = "margis"
```

| option | type | default | description |
|---|---|---|---|
| `nu` | number or null | | Degrees of freedom of the prior on each type's error scale; default `N - 1` (PET's). Larger trusts `C_d` more; `inf` with `s = 1` gives `subspace2`. |
| `s` | number | 1.0 | Prior scale of the whitened errors, 1 (PET's): `C_d` is right on average. |

Methods that take it: `gies`, `gn_enrml`, `lm_enrml`.

| refused by | why |
|---|---|
| `chop` | [method] chop: CHOP needs an RLM-MAC strategy (rlmmac), whose gain it tapers per member and whose mean-model forecast its hyper-parameter update uses; got MarginalISUpdate |
| `es` | ES with the MarginalISUpdate analysis is not supported: it includes the prior-increment term of an iterative method (each member pulled back towards its prior), and ES-MDA's steps have none, each step updating the current ensemble; use the approx strategy, or this one with gies, lm_enrml or gn_enrml |
| `esmda` | ESMDA with the MarginalISUpdate analysis is not supported: it includes the prior-increment term of an iterative method (each member pulled back towards its prior), and ES-MDA's steps have none, each step updating the current ensemble; use the approx strategy, or this one with gies, lm_enrml or gn_enrml |
| `multilevel_esmda` | multilevel ES-MDA runs the hybrid update only (PET's), not MarginalISUpdate: leave out [method.strategy], or name hybrid |

## `rlmmac`

Regularised Levenberg-Marquardt update for the minimum-average-cost problem (RLM-MAC).

Select it with:

```toml
[method.strategy]
name = "rlmmac"
```

| option | type | default | description |
|---|---|---|---|
| `energy` | number | 0.95 | Fraction of the singular-value sum of `L^-1 dY` to retain, in `(0, 1]` (PET keyword `energy`, default 0.95 for GIES). |
| `scale_data_anomalies` | boolean | true | Scale `dY` by `1/sqrt(N - 1)` like `dX` (default True, as agreed with the method's author). False leaves `dY` unscaled, as does, to reproduce its runs: the singular values grow by `sqrt(N - 1)` and `gamma = lambda * mean(s^2)` with them, so every direction's weight, hence the step, is `1/sqrt(N - 1)` times the default's at the same `lambda`. No `lambda` maps one onto the other (the ratio depends on `s^2 / gamma`), hence the option. False logs `UNSCALED_DATA_ANOMALIES_WARNING`: that update does not follow the theory. |

Methods that take it: `chop`, `gies`, `lm_enrml`.

| refused by | why |
|---|---|
| `es` | ES with the RLMMACUpdate analysis is not supported: it needs the forecast at the mean state, which only the methods built on GIES's loop make; use the approx strategy, or this one with gies or lm_enrml |
| `esmda` | ESMDA with the RLMMACUpdate analysis is not supported: it needs the forecast at the mean state, which only the methods built on GIES's loop make; use the approx strategy, or this one with gies or lm_enrml |
| `gn_enrml` | GN-EnRML with the RLMMACUpdate analysis is not supported: it runs at damping 0, which leaves RLM-MAC unregularised (gamma = lambda * mean(s^2) = 0); use approx, full, subspace, subspace2 or margis, or RLM-MAC with gies or lm_enrml |
| `multilevel_esmda` | multilevel ES-MDA runs the hybrid update only (PET's), not RLMMACUpdate: leave out [method.strategy], or name hybrid |

## `subspace`

Ensemble subspace update: an iterative update of each member's weights.

Select it with:

```toml
[method.strategy]
name = "subspace"
```

| option | type | default | description |
|---|---|---|---|
| `energy` | number | 0.98 | Fraction of the singular-value sum of `S` kept, in `(0, 1]` (PET `energy`; its iterative smoothers default to 0.95, which `apex pet-import` passes). |

Methods that take it: `gies`, `gn_enrml`, `lm_enrml`.

| refused by | why |
|---|---|
| `chop` | [method] chop: CHOP needs an RLM-MAC strategy (rlmmac), whose gain it tapers per member and whose mean-model forecast its hyper-parameter update uses; got SubspaceUpdate |
| `es` | ES with the SubspaceUpdate analysis is not supported: it includes the prior-increment term of an iterative method (each member pulled back towards its prior), and ES-MDA's steps have none, each step updating the current ensemble; use the approx strategy, or this one with gies, lm_enrml or gn_enrml |
| `esmda` | ESMDA with the SubspaceUpdate analysis is not supported: it includes the prior-increment term of an iterative method (each member pulled back towards its prior), and ES-MDA's steps have none, each step updating the current ensemble; use the approx strategy, or this one with gies, lm_enrml or gn_enrml |
| `multilevel_esmda` | multilevel ES-MDA runs the hybrid update only (PET's), not SubspaceUpdate: leave out [method.strategy], or name hybrid |

## `subspace2`

Ensemble-transform IES of Raanes et al. (2019), PET's `subspace2`.

Select it with:

```toml
[method.strategy]
name = "subspace2"
```


Methods that take it: `gies`, `gn_enrml`, `lm_enrml`.

| refused by | why |
|---|---|
| `chop` | [method] chop: CHOP needs an RLM-MAC strategy (rlmmac), whose gain it tapers per member and whose mean-model forecast its hyper-parameter update uses; got TransformUpdate |
| `es` | ES with the TransformUpdate analysis is not supported: it includes the prior-increment term of an iterative method (each member pulled back towards its prior), and ES-MDA's steps have none, each step updating the current ensemble; use the approx strategy, or this one with gies, lm_enrml or gn_enrml |
| `esmda` | ESMDA with the TransformUpdate analysis is not supported: it includes the prior-increment term of an iterative method (each member pulled back towards its prior), and ES-MDA's steps have none, each step updating the current ensemble; use the approx strategy, or this one with gies, lm_enrml or gn_enrml |
| `multilevel_esmda` | multilevel ES-MDA runs the hybrid update only (PET's), not TransformUpdate: leave out [method.strategy], or name hybrid |

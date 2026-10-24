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
| `energy` | number | 0.98 | |
| `max_batch_mb` | number | 256.0 | |
| `threshold` | number or null | | |

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
| `empirical_data_error` | boolean | false | |
| `energy` | number | 0.98 | |
| `sensitivities` | boolean | false | |

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
| `max_batch_mb` | number | 256.0 | |
| `n_jobs` | integer | 1 | |
| `threshold` | number or null | | |
| `truncation` | number | 0.99 | |

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
| `max_batch_mb` | number | 256.0 | |
| `truncation` | number | 0.99 | |

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
| `max_batch_mb` | number | 256.0 | |
| `taper_file` | string or string | required | |
| `truncation` | number | 0.99 | |

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
| `block_sizes` | array or null | | |
| `energy` | number | 0.98 | |
| `state_scaling` | ArrayLike or null | | |
| `state_scaling_file` | string or string or null | | |

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
| `code` | string or null | | |
| `constraints` | array or null | | |
| `energy` | number | 0.95 | |
| `form` | string | subspace | |
| `localize_constraints` | boolean | true | |
| `metric_energy` | number or null | | |
| `metric_floor` | number | 1e-10 | |
| `terms` | array or null | | |
| `variables` | array or null | | |

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
| `energy` | number | 0.98 | |

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
| `block_sizes` | array or null | | |
| `energy` | number | 0.98 | |
| `weight` | boolean | false | |

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
| `block_sizes` | array or null | | |
| `energy` | number | 0.95 | |
| `weight` | boolean | false | |

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
| `nu` | number or null | | |
| `s` | number | 1.0 | |

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
| `energy` | number | 0.95 | |
| `scale_data_anomalies` | boolean | true | |

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
| `energy` | number | 0.98 | |

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

---
audience: public
items:
  - cli:apex optimize
  - cli:apex plugins
  - cli:apex run
  - cli:apex stop
  - config:optimization:objectives
  - plugin:optimizers:enopt
  - plugin:optimizers:ga
  - plugin:optimizers:genopt
  - plugin:optimizers:line_search
  - plugin:optimizers:moead
  - plugin:optimizers:nsga2
  - plugin:optimizers:nsga3
  - plugin:optimizers:pso
  - plugin:optimizers:ropt
  - plugin:optimizers:smcopt
  - plugin:optimizers:trust_region
---

# Optimizers

This page covers 16 of APEX's functionalities: what each is, how to select it, and its options.

## `apex optimize`

Minimise an objective with one of PET's optimizers (EnOpt, GenOpt, SmcOpt, ...).

```bash
apex optimize CONFIG
```

| option | default | description |
|---|---|---|
| `--resume` |  | Continue the optimization run in this directory. |
| `--max-iterations` |  | Stop after this many accepted iterations (resumable). |
| `--progress` (on) / `--no-progress` (off) | true | Show progress on standard error: bars on a terminal, timestamped lines otherwise. |

## `apex plugins`

List every plugin with its options (for clients: --json).

```bash
apex plugins
```

| option | default | description |
|---|---|---|
| `--json` | false | Full descriptions, as JSON. |

## `apex run`

Start a history-matching run.

```bash
apex run CONFIG
```

| option | default | description |
|---|---|---|
| `--max-steps` |  | Stop after this many analysis steps; continue later with `apex resume`. |
| `--progress` (on) / `--no-progress` (off) | true | Show progress on standard error: bars on a terminal, timestamped lines otherwise. |

## `apex stop`

Ask a running run to stop at its next checkpoint (resume continues it).

```bash
apex stop RUN_DIR
```


## `optimization:objectives`

Several scalar objectives optimized together (e.g. by nsga2): each a plugin with options and an optional label. Response objectives share one simulation per member.


## `enopt`

Ensemble optimization (EnOpt): steps along an ensemble estimate of the gradient.

Select it with:

```toml
[optimizer]
name = "enopt"
```

| option | type | default | description |
|---|---|---|---|
| `alpha` | number | 0.1 |  |
| `alpha_cov` | number | 0.001 |  |
| `alpha_maxiter` | integer | 5 |  |
| `beta` | number | 0.0 |  |
| `cov_factor` | number | 0.5 |  |
| `ensemble_hessian` | boolean | false |  |
| `epf` |  |  |  |
| `ftol` | number or null |  |  |
| `gtol` | number | 1e-05 |  |
| `hessian` | boolean | false |  |
| `maxiter` | integer | 100 |  |
| `natural_gradient` | boolean | true |  |
| `ne` | integer | 100 |  |
| `nesterov` | boolean | false |  |
| `normalize` | boolean | true |  |
| `optimizer` | string | GD |  |
| `resample` | integer | 0 |  |
| `step_size` | number or null |  |  |
| `tol` | number or null |  |  |
| `transform` | boolean | false |  |
| `trust_radius` | number | 3.0 |  |
| `xtol` | number | 1e-08 |  |

## `ga`

A real-coded genetic algorithm: SBX crossover, polynomial mutation, tournaments.

Select it with:

```toml
[optimizer]
name = "ga"
```

| option | type | default | description |
|---|---|---|---|
| `constraint_handling` |  | feasibility |  |
| `crossover_prob` | number | 0.9 |  |
| `crossover_prob_var` | number | 0.5 |  |
| `eliminate_duplicates` | boolean | true |  |
| `epf` |  |  |  |
| `eq_tolerance` | number | 0.0001 |  |
| `eta_c` | number | 15.0 |  |
| `eta_m` | number | 20.0 |  |
| `ftol` | number | 1e-06 |  |
| `init` |  | uniform |  |
| `maxiter` | integer | 100 |  |
| `mutation_prob` | number | 0.9 |  |
| `mutation_prob_var` | number or null |  |  |
| `n_offsprings` | integer or null |  |  |
| `pop_size` | integer | 20 |  |
| `stall` | integer | 20 |  |
| `tournament` | integer | 2 |  |
| `transform` | boolean | false |  |
| `xtol` | number | 1e-08 |  |

## `genopt`

Generalized ensemble optimization (GenOpt): EnOpt with marginals and copulas, or CMA.

Select it with:

```toml
[optimizer]
name = "genopt"
```

| option | type | default | description |
|---|---|---|---|
| `alpha` | number | 0.1 |  |
| `alpha_maxiter` | integer | 5 |  |
| `alpha_theta` | number | 0.1 |  |
| `beta` | number | 0.0 |  |
| `cma` | boolean | false |  |
| `cma_alpha_1` | number or null |  |  |
| `cma_alpha_c` | number or null |  |  |
| `cma_alpha_mu` | number or null |  |  |
| `cma_equal_weights` | boolean | true |  |
| `cma_n_mu` | integer or null |  |  |
| `cov_factor` | number | 0.5 |  |
| `epf` |  |  |  |
| `ftol` | number or null |  |  |
| `gtol` | number | 1e-05 |  |
| `marginal` | string | BetaMC |  |
| `maxiter` | integer | 100 |  |
| `ne` | integer | 100 |  |
| `nesterov` | boolean | false |  |
| `normalize` | boolean | true |  |
| `optimizer` | string | GD |  |
| `resample` | integer | 0 |  |
| `separate_mutation_ensemble` | boolean | false |  |
| `step_size` | number or null |  |  |
| `theta` | array or null |  |  |
| `tol` | number or null |  |  |
| `transform` | boolean | false |  |
| `xtol` | number | 1e-08 |  |

## `line_search`

Line-search optimization; options are PET's.

Select it with:

```toml
[optimizer]
name = "line_search"
```

| option | type | default | description |
|---|---|---|---|
| `c1` | number | 0.0001 |  |
| `c2` | number | 0.9 |  |
| `epf` |  |  |  |
| `ftol` | number | 1e-05 |  |
| `gtol` | number | 1e-05 |  |
| `lsmaxiter` | integer | 10 |  |
| `lsmethod` | integer | 1 |  |
| `maxiter` | integer | 100 |  |
| `method` | string | GD |  |
| `recompute_jac` | integer | 0 |  |
| `rho` | number | 0.5 |  |
| `step_size` | number or null |  |  |
| `step_size_adapt` | integer | 1 |  |
| `step_size_max` | number | 100000.0 |  |
| `transform` | boolean | false |  |
| `xtol` | number | 1e-08 |  |

## `moead`

MOEA/D: multi-objective optimization by neighbouring scalar subproblems.

Select it with:

```toml
[optimizer]
name = "moead"
```

| option | type | default | description |
|---|---|---|---|
| `constraint_handling` | string | feasibility |  |
| `crossover_prob` | number | 1.0 |  |
| `crossover_prob_var` | number | 0.5 |  |
| `decomposition` |  | auto |  |
| `epf` |  |  |  |
| `eq_tolerance` | number | 0.0001 |  |
| `eta_c` | number | 20.0 |  |
| `eta_m` | number | 20.0 |  |
| `ftol` | number | 0.0025 |  |
| `init` |  | uniform |  |
| `inner_partitions` | integer or null |  |  |
| `maxiter` | integer | 100 |  |
| `mutation_prob` | number | 0.9 |  |
| `mutation_prob_var` | number or null |  |  |
| `n_neighbors` | integer or null |  |  |
| `n_partitions` | integer or null |  |  |
| `prob_neighbor_mating` | number | 0.9 |  |
| `ref_point` | array or null |  |  |
| `stall` | integer | 30 |  |
| `theta` | number | 5.0 |  |
| `transform` | boolean | false |  |
| `xtol` | number | 1e-08 |  |

## `nsga2`

NSGA-II: multi-objective optimization by non-dominated sorting and crowding.

Select it with:

```toml
[optimizer]
name = "nsga2"
```

| option | type | default | description |
|---|---|---|---|
| `constraint_handling` |  | feasibility |  |
| `crossover_prob` | number | 0.9 |  |
| `crossover_prob_var` | number | 0.5 |  |
| `eliminate_duplicates` | boolean | true |  |
| `epf` |  |  |  |
| `eq_tolerance` | number | 0.0001 |  |
| `eta_c` | number | 15.0 |  |
| `eta_m` | number | 20.0 |  |
| `ftol` | number | 0.0025 |  |
| `init` |  | uniform |  |
| `maxiter` | integer | 100 |  |
| `mutation_prob` | number | 0.9 |  |
| `mutation_prob_var` | number or null |  |  |
| `n_offsprings` | integer or null |  |  |
| `pop_size` | integer | 40 |  |
| `ref_point` | array or null |  |  |
| `stall` | integer | 30 |  |
| `transform` | boolean | false |  |
| `xtol` | number | 1e-08 |  |

## `nsga3`

NSGA-III: many-objective NSGA with survival by reference directions.

Select it with:

```toml
[optimizer]
name = "nsga3"
```

| option | type | default | description |
|---|---|---|---|
| `constraint_handling` |  | feasibility |  |
| `crossover_prob` | number | 1.0 |  |
| `crossover_prob_var` | number | 0.5 |  |
| `eliminate_duplicates` | boolean | true |  |
| `epf` |  |  |  |
| `eq_tolerance` | number | 0.0001 |  |
| `eta_c` | number | 30.0 |  |
| `eta_m` | number | 20.0 |  |
| `ftol` | number | 0.0025 |  |
| `init` |  | uniform |  |
| `inner_partitions` | integer or null |  |  |
| `maxiter` | integer | 100 |  |
| `mutation_prob` | number | 0.9 |  |
| `mutation_prob_var` | number or null |  |  |
| `n_offsprings` | integer or null |  |  |
| `n_partitions` | integer or null |  |  |
| `pop_size` | integer or null |  |  |
| `ref_point` | array or null |  |  |
| `stall` | integer | 30 |  |
| `transform` | boolean | false |  |
| `xtol` | number | 1e-08 |  |

## `pso`

Particle swarm optimization: particles drawn to their own and the swarm's best.

Select it with:

```toml
[optimizer]
name = "pso"
```

| option | type | default | description |
|---|---|---|---|
| `bounds_handling` |  | reflect |  |
| `c1` | number | 1.49618 |  |
| `c2` | number | 1.49618 |  |
| `constraint_handling` |  | feasibility |  |
| `epf` |  |  |  |
| `eq_tolerance` | number | 0.0001 |  |
| `ftol` | number | 1e-06 |  |
| `init` |  | uniform |  |
| `initial_velocity` | string | random |  |
| `maxiter` | integer | 100 |  |
| `pop_size` | integer | 20 |  |
| `stall` | integer | 20 |  |
| `topology` |  | global |  |
| `transform` | boolean | false |  |
| `v_max` | number | 0.2 |  |
| `w` | number | 0.7298 |  |
| `xtol` | number | 1e-08 |  |

## `ropt`

ropt's optimizer backends (SciPy, Dakota) run on APEX's objectives.

Select it with:

```toml
[optimizer]
name = "ropt"
```

| option | type | default | description |
|---|---|---|---|
| `constraint_tolerance` | number | 1e-08 |  |
| `constraints` | boolean | false |  |
| `convergence_tolerance` | number or null |  |  |
| `epf` |  |  |  |
| `max_batches` | integer or null |  |  |
| `max_functions` | integer or null |  |  |
| `maxiter` | integer | 100 |  |
| `method` | string | scipy/slsqp |  |
| `number_of_perturbations` | integer | 5 |  |
| `options` | object or null |  |  |
| `perturbation_magnitude` | number or array or null |  |  |
| `seed` | integer or null |  |  |
| `transform` | boolean | false |  |

## `smcopt`

Sequential Monte Carlo optimization (SmcOpt): resampled ensembles towards the optimum.

Select it with:

```toml
[optimizer]
name = "smcopt"
```

| option | type | default | description |
|---|---|---|---|
| `alpha` | number | 0.1 |  |
| `alpha_maxiter` | integer | 5 |  |
| `best_func` | number or null |  |  |
| `cov_factor` | number | 0.5 |  |
| `epf` |  |  |  |
| `ftol` | number or null |  |  |
| `gtol` | number | 1e-05 |  |
| `inflation_factor` | number | 1.0 |  |
| `maxiter` | integer | 100 |  |
| `ne` | integer | 100 |  |
| `resample` | integer | 0 |  |
| `survival_factor` | number | 1.0 |  |
| `tol` | number or null |  |  |
| `xtol` | number | 1e-08 |  |

## `trust_region`

Trust-region optimization; options are PET's.

Select it with:

```toml
[optimizer]
name = "trust_region"
```

| option | type | default | description |
|---|---|---|---|
| `epf` |  |  |  |
| `eta1` | number | 0.05 |  |
| `eta2` | number | 0.5 |  |
| `ftol` | number | 1e-05 |  |
| `gam1` | number | 0.5 |  |
| `gam2` | number | 1.5 |  |
| `gtol` | number | 1e-05 |  |
| `maxiter` | integer | 100 |  |
| `method` | string | iterative |  |
| `quasi_newton` | boolean | false |  |
| `resample` | boolean | false |  |
| `rho_tol` | number | 1e-06 |  |
| `transform` | boolean | false |  |
| `trust_radius` | number | 1.0 |  |
| `trust_radius_cuts` | integer | 4 |  |
| `trust_radius_max` | number or null |  |  |
| `trust_radius_min` | number or null |  |  |
| `xtol` | number | 1e-08 |  |

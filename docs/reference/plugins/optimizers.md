---
audience: public
generated: reference
---

# Optimizers: reference

Every one of these 11 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `enopt`

Ensemble optimization (EnOpt): steps along an ensemble estimate of the gradient.

Select it with:

```toml
[optimizer]
name = "enopt"
```

| option | type | default | description |
|---|---|---|---|
| `alpha` | number | 0.1 | Initial step size (`step_size` overrides `alpha`). |
| `alpha_cov` | number | 0.001 | Step size of the covariance update. |
| `alpha_maxiter` | integer | 5 | Backtracking trials per resampling attempt. |
| `beta` | number | 0.0 | Momentum of the step and of the covariance update (Adam's first moment rate). |
| `cov_factor` | number | 0.5 | Covariance shrink factor per resampling attempt. |
| `ensemble_hessian` | boolean | false | Estimate the ensemble Hessian (PET passes `hess`): it adapts the covariance and feeds the Steihaug rule. |
| `epf` | | | External penalty options: `r` (required), `r_factor`, `tol_factor`, `conv_crit`, `max_epf_iter` (`EPF_DEFAULTS`). Needs a constrained objective. |
| `ftol` | number or null | | Relative change of the mean objective that counts as converged. |
| `gtol` | number | 1e-05 | Infinity norm of the projected gradient that counts as converged. |
| `hessian` | boolean | false | Precondition the gradient with the Hessian (needs `ensemble_hessian`). |
| `maxiter` | integer | 100 | Iterations per outer EPF iteration. |
| `natural_gradient` | boolean | true | Use the natural gradient `C grad f` (PET's default) instead of `grad f`. |
| `ne` | integer | 100 | Ensemble size (PET's `ne`; default 100). |
| `nesterov` | boolean | false | Estimate the gradient at the momentum-extrapolated point. |
| `normalize` | boolean | true | Scale the gradient and the Hessian by their infinity norms. |
| `optimizer` | string | GD | Step rule: `GD`, `Adam`, `AdaMax` (turns `normalize` off) or `Steihaug` (needs `ensemble_hessian`), or any registered `step_rules` name. |
| `resample` | integer | 0 | Resampling attempts when backtracking fails. |
| `step_size` | number or null | | Initial step size (`step_size` overrides `alpha`). |
| `tol` | number or null | | Improvement of the mean objective a trial needs; also `ftol` when given. |
| `transform` | boolean | false | Optimize in the unit cube of the bounds. |
| `trust_radius` | number | 3.0 | Initial radius of the Steihaug rule. |
| `xtol` | number | 1e-08 | Change of the controls (2-norm, optimizer coordinates) that counts as converged. |

## `ga`

A real-coded genetic algorithm: SBX crossover, polynomial mutation, tournaments.

Select it with:

```toml
[optimizer]
name = "ga"
```

| option | type | default | description |
|---|---|---|---|
| `constraint_handling` | | feasibility | `"feasibility"` (Deb's rules, as pymoo: a smaller constraint violation ranks first, then a smaller objective) or `"epf"` (the external penalty, which needs `epf`). Matters only with a constrained objective. |
| `crossover_prob` | number | 0.9 | SBX: per pair, per variable, the distribution index. |
| `crossover_prob_var` | number | 0.5 | SBX: per pair, per variable, the distribution index. |
| `eliminate_duplicates` | boolean | true | Redraw children equal to a member or to each other. |
| `epf` | | | External penalty options with `constraint_handling = "epf"`: `r` (required), `r_factor`, `tol_factor`, `conv_crit`, `max_epf_iter`. |
| `eq_tolerance` | number | 0.0001 | Equality constraints within this absolute distance of zero count as met under the feasibility rules (default 1e-4). |
| `eta_c` | number | 15.0 | SBX: per pair, per variable, the distribution index. |
| `eta_m` | number | 20.0 | Polynomial mutation: per member, per variable (default `min(0.5, 1/n_x)`), index. |
| `ftol` | number | 1e-06 | Relative improvement of the best member's objective that resets the `stall` count (default 1e-6). |
| `init` | | uniform | How the start population besides `x0` is drawn within the bounds: `"uniform"` (pymoo's random sampling) or `"lhs"` (Latin hypercube). |
| `maxiter` | integer | 100 | Generations at most (per outer EPF iteration with `"epf"`; default 100). |
| `mutation_prob` | number | 0.9 | Polynomial mutation: per member, per variable (default `min(0.5, 1/n_x)`), index. |
| `mutation_prob_var` | number or null | | Polynomial mutation: per member, per variable (default `min(0.5, 1/n_x)`), index. |
| `n_offsprings` | integer or null | | Children per generation (default `pop_size`). |
| `pop_size` | integer | 20 | Members of the population, `x0` included (at least 2, default 20); each generation evaluates `n_offsprings` new members on every realization. |
| `stall` | integer | 20 | Generations in a row without the best member improving by more than `ftol` (relative) after which the run counts as converged (default 20). |
| `tournament` | integer | 2 | Tournament size (pymoo: 2). |
| `transform` | boolean | false | Optimize in the unit cube of the bounds instead of the controls' own units (default off). |
| `xtol` | number | 1e-08 | The run counts as converged when the population's spread, the largest standard deviation of a control (in optimizer coordinates), falls below it (default 1e-8). |

## `genopt`

Generalized ensemble optimization (GenOpt): EnOpt with marginals and copulas, or CMA.

Select it with:

```toml
[optimizer]
name = "genopt"
```

| option | type | default | description |
|---|---|---|---|
| `alpha` | number | 0.1 | As for EnOpt; `cov_factor` shrinks `theta` per resampling attempt. |
| `alpha_maxiter` | integer | 5 | As for EnOpt; `cov_factor` shrinks `theta` per resampling attempt. |
| `alpha_theta` | number | 0.1 | Step size of `theta`. |
| `beta` | number | 0.0 | As for EnOpt; `cov_factor` shrinks `theta` per resampling attempt. |
| `cma` | boolean | false | Adapt the copula's correlation by CMA. |
| `cma_alpha_1` | number or null | | CMA's options (Hansen's defaults when unset); CMA adapts a correlation. |
| `cma_alpha_c` | number or null | | CMA's options (Hansen's defaults when unset); CMA adapts a correlation. |
| `cma_alpha_mu` | number or null | | CMA's options (Hansen's defaults when unset); CMA adapts a correlation. |
| `cma_equal_weights` | boolean | true | CMA's options (Hansen's defaults when unset); CMA adapts a correlation. |
| `cma_n_mu` | integer or null | | CMA's options (Hansen's defaults when unset); CMA adapts a correlation. |
| `cov_factor` | number | 0.5 | As for EnOpt; `cov_factor` shrinks `theta` per resampling attempt. |
| `epf` | | | External penalty options: `r` (required), `r_factor`, `tol_factor`, `conv_crit`, `max_epf_iter` (`EPF_DEFAULTS`). Needs a constrained objective. |
| `ftol` | number or null | | Relative change of the mean objective that counts as converged. |
| `gtol` | number | 1e-05 | Infinity norm of the projected gradient that counts as converged. |
| `marginal` | string | BetaMC | `BetaMC` (PET's default) or `TruncGaussian`: the marginals with a `theta` gradient. |
| `maxiter` | integer | 100 | Iterations per outer EPF iteration. |
| `ne` | integer | 100 | Ensemble size (default 100). |
| `nesterov` | boolean | false | As for EnOpt; `cov_factor` shrinks `theta` per resampling attempt. |
| `normalize` | boolean | true | As for EnOpt; `cov_factor` shrinks `theta` per resampling attempt. |
| `optimizer` | string | GD | Step rule: `GD` or `Adam`. |
| `resample` | integer | 0 | As for EnOpt; `cov_factor` shrinks `theta` per resampling attempt. |
| `separate_mutation_ensemble` | boolean | false | Draw a second ensemble for the mutation gradient, as PET does. |
| `step_size` | number or null | | As for EnOpt; `cov_factor` shrinks `theta` per resampling attempt. |
| `theta` | array or null | | Initial marginal parameter per control (default from the initial variances). |
| `tol` | number or null | | As for EnOpt; `cov_factor` shrinks `theta` per resampling attempt. |
| `transform` | boolean | false | Optimize in the unit cube of the bounds. |
| `xtol` | number | 1e-08 | Change of the controls (2-norm, optimizer coordinates) that counts as converged. |

## `line_search`

Line-search optimization; options are PET's.

Select it with:

```toml
[optimizer]
name = "line_search"
```

| option | type | default | description |
|---|---|---|---|
| `c1` | number | 0.0001 | Wolfe constants. |
| `c2` | number | 0.9 | Wolfe constants. |
| `epf` | | | External penalty options: `r` (required), `r_factor`, `tol_factor`, `conv_crit`, `max_epf_iter` (`EPF_DEFAULTS`). Needs a constrained objective. |
| `ftol` | number | 1e-05 | Relative change of the mean objective that counts as converged. |
| `gtol` | number | 1e-05 | Infinity norm of the projected gradient that counts as converged. |
| `lsmaxiter` | integer | 10 | Trial step lengths per line search. |
| `lsmethod` | integer | 1 | 0 backtracking (Armijo), 1 strong Wolfe. |
| `maxiter` | integer | 100 | Iterations per outer EPF iteration. |
| `method` | string | GD | Search direction: `GD` (steepest descent), `BFGS` or `Newton-CG`. |
| `recompute_jac` | integer | 0 | Gradient re-evaluations (and retries) after a failed line search. |
| `rho` | number | 0.5 | Backtracking factor. |
| `step_size` | number or null | | First step length (default `0.25 / \|\|p\|\|_inf` at the first iteration). |
| `step_size_adapt` | integer | 1 | Later first trials: 0 keep `step_size`, 1 from the last decrease (`2 (f_k - f_k-1) / (g^T p)`), 2 from the slope ratio. |
| `step_size_max` | number | 100000.0 | Largest step length. |
| `transform` | boolean | false | Optimize in the unit cube of the bounds. |
| `xtol` | number | 1e-08 | Change of the controls (2-norm, optimizer coordinates) that counts as converged. |

## `moead`

MOEA/D: multi-objective optimization by neighbouring scalar subproblems.

Select it with:

```toml
[optimizer]
name = "moead"
```

| option | type | default | description |
|---|---|---|---|
| `constraint_handling` | string | feasibility | `epf` for constrained objectives (EPF only); `feasibility` (the default) refuses them. |
| `crossover_prob` | number | 1.0 | SBX (pymoo's MOEA/D: 20, 1, 0.5). |
| `crossover_prob_var` | number | 0.5 | SBX (pymoo's MOEA/D: 20, 1, 0.5). |
| `decomposition` | | auto | `auto` (Tchebicheff for 2 objectives, PBI for more, as pymoo), `tchebicheff` or `pbi`. |
| `epf` | | | As `nsga2`. |
| `eq_tolerance` | number | 0.0001 | As `nsga2`. |
| `eta_c` | number | 20.0 | SBX (pymoo's MOEA/D: 20, 1, 0.5). |
| `eta_m` | number | 20.0 | Polynomial mutation (pymoo: 20, 0.9, `1/n_x`). |
| `ftol` | number | 0.0025 | As `nsga2`. |
| `init` | | uniform | As `nsga2`. |
| `inner_partitions` | integer or null | | Das-Dennis divisions, as `nsga3`'s (default by objectives: 2 -> 19, 3 -> 6, 4 -> 4; from 5, two layers). The population is the number of weights: 20 for 2 objectives, 28 for 3, 20 for 5. |
| `maxiter` | integer | 100 | As `nsga2`. |
| `mutation_prob` | number | 0.9 | Polynomial mutation (pymoo: 20, 0.9, `1/n_x`). |
| `mutation_prob_var` | number or null | | Polynomial mutation (pymoo: 20, 0.9, `1/n_x`). |
| `n_neighbors` | integer or null | | Each subproblem's neighbourhood (default `min(20, max(2, ceil(N / 5)))` for `N` weights; pymoo and the paper: 20). |
| `n_partitions` | integer or null | | Das-Dennis divisions, as `nsga3`'s (default by objectives: 2 -> 19, 3 -> 6, 4 -> 4; from 5, two layers). The population is the number of weights: 20 for 2 objectives, 28 for 3, 20 for 5. |
| `prob_neighbor_mating` | number | 0.9 | Probability of mating within the neighbourhood (pymoo: 0.9). |
| `ref_point` | array or null | | As `nsga2`. |
| `stall` | integer | 30 | As `nsga2`. |
| `theta` | number | 5.0 | PBI's penalty (pymoo: 5). |
| `transform` | boolean | false | As `nsga2`. |
| `xtol` | number | 1e-08 | As `nsga2`. |

## `nsga2`

NSGA-II: multi-objective optimization by non-dominated sorting and crowding.

Select it with:

```toml
[optimizer]
name = "nsga2"
```

| option | type | default | description |
|---|---|---|---|
| `constraint_handling` | | feasibility | `"feasibility"` (Deb's rules) or `"epf"` (the external penalty, with `epf`). |
| `crossover_prob` | number | 0.9 | SBX and polynomial mutation, as `ga`'s. |
| `crossover_prob_var` | number | 0.5 | SBX and polynomial mutation, as `ga`'s. |
| `eliminate_duplicates` | boolean | true | Redraw children equal to a member or to each other. |
| `epf` | | | External penalty options: `r` (required), `r_factor`, `tol_factor`, `conv_crit`, `max_epf_iter`. Only with `constraint_handling = "epf"`. |
| `eq_tolerance` | number | 0.0001 | Equality constraints within it count as met (feasibility rules). |
| `eta_c` | number | 15.0 | SBX and polynomial mutation, as `ga`'s. |
| `eta_m` | number | 20.0 | SBX and polynomial mutation, as `ga`'s. |
| `ftol` | number | 0.0025 | Stop after `stall` generations in a row (default 30) in which the feasible front moved less than `ftol` (default 0.0025, pymoo's tolerance of the measure): `front_movement`, its ideal and nadir points and its IGD to the previous front, in normalised objectives. Before any member is feasible, a generation counts as stalled unless the least violation fell. |
| `init` | | uniform | The initial population besides `x0`: `"uniform"` or `"lhs"`. |
| `maxiter` | integer | 100 | Generations at most (per outer EPF iteration). |
| `mutation_prob` | number | 0.9 | SBX and polynomial mutation, as `ga`'s. |
| `mutation_prob_var` | number or null | | SBX and polynomial mutation, as `ga`'s. |
| `n_offsprings` | integer or null | | Children per generation (default `pop_size`). |
| `pop_size` | integer | 40 | Members (default 40). |
| `ref_point` | array or null | | The hypervolume's reference point, one value per objective (default: from the first feasible members). |
| `stall` | integer | 30 | Stop after `stall` generations in a row (default 30) in which the feasible front moved less than `ftol` (default 0.0025, pymoo's tolerance of the measure): `front_movement`, its ideal and nadir points and its IGD to the previous front, in normalised objectives. Before any member is feasible, a generation counts as stalled unless the least violation fell. |
| `transform` | boolean | false | Optimize in the unit cube of the bounds. |
| `xtol` | number | 1e-08 | The population's spread (the largest standard deviation of a control, optimizer coordinates) that counts as converged. |

## `nsga3`

NSGA-III: many-objective NSGA with survival by reference directions.

Select it with:

```toml
[optimizer]
name = "nsga3"
```

| option | type | default | description |
|---|---|---|---|
| `constraint_handling` | | feasibility | As `nsga2`. |
| `crossover_prob` | number | 1.0 | SBX's distribution index and probability (pymoo's NSGA-III: 30 and 1). |
| `crossover_prob_var` | number | 0.5 | SBX's probability of exchanging each variable within a crossed pair, in `[0, 1]` (pymoo's `prob_var`, default 0.5). |
| `eliminate_duplicates` | boolean | true | Redraw children that equal a member or each other, so no evaluation is spent twice on the same controls (default on). |
| `epf` | | | As `nsga2`. |
| `eq_tolerance` | number | 0.0001 | As `nsga2`. |
| `eta_c` | number | 30.0 | SBX's distribution index and probability (pymoo's NSGA-III: 30 and 1). |
| `eta_m` | number | 20.0 | Polynomial mutation's distribution index (default 20): larger values keep the mutated variables closer to the parent's. |
| `ftol` | number | 0.0025 | As `nsga2`. |
| `init` | | uniform | As `nsga2`. |
| `inner_partitions` | integer or null | | Das-Dennis divisions of each axis, and of an inner layer shrunk halfway to the centre (0: none). Default by objectives: 2 -> 19, 3 -> 6, 4 -> 4 (20, 28, 35 directions); from 5, two layers, 2 and 1 (20 directions for 5, 27 for 6, 44 for 8), as a single layer with fewer partitions than objectives has no interior direction. Deb & Jain (2014) take 12 for three objectives (91 directions), as pymoo's examples do; APEX's are smaller because an evaluation is a simulation. |
| `maxiter` | integer | 100 | As `nsga2`. |
| `mutation_prob` | number | 0.9 | Probability that a child is mutated at all, in `[0, 1]` (default 0.9). |
| `mutation_prob_var` | number or null | | Probability that each variable of a mutated child changes, in `[0, 1]`; by default `min(0.5, 1/n_x)` for `n_x` controls, as pymoo. |
| `n_offsprings` | integer or null | | Children per generation (default `pop_size`). |
| `n_partitions` | integer or null | | Das-Dennis divisions of each axis, and of an inner layer shrunk halfway to the centre (0: none). Default by objectives: 2 -> 19, 3 -> 6, 4 -> 4 (20, 28, 35 directions); from 5, two layers, 2 and 1 (20 directions for 5, 27 for 6, 44 for 8), as a single layer with fewer partitions than objectives has no interior direction. Deb & Jain (2014) take 12 for three objectives (91 directions), as pymoo's examples do; APEX's are smaller because an evaluation is a simulation. |
| `pop_size` | integer or null | | Members (default: the number of directions, rounded up to a multiple of 4; pymoo: the number of directions, warning below it). |
| `ref_point` | array or null | | The reference point of the reported hypervolume, one value per objective (as minimised); by default the first feasible members' nadir plus 10 % of each objective's range, fixed for the run so its hypervolumes compare. |
| `stall` | integer | 30 | As `nsga2`. |
| `transform` | boolean | false | As `nsga2`. |
| `xtol` | number | 1e-08 | As `nsga2`. |

## `pso`

Particle swarm optimization: particles drawn to their own and the swarm's best.

Select it with:

```toml
[optimizer]
name = "pso"
```

| option | type | default | description |
|---|---|---|---|
| `bounds_handling` | | reflect | `"reflect"` or `"clip"`. |
| `c1` | number | 1.49618 | Inertia, cognitive and social weights (Clerc and Kennedy's constriction equivalents). |
| `c2` | number | 1.49618 | Inertia, cognitive and social weights (Clerc and Kennedy's constriction equivalents). |
| `constraint_handling` | | feasibility | `"feasibility"` (Deb's rules, as pymoo: a smaller constraint violation ranks first, then a smaller objective) or `"epf"` (the external penalty, which needs `epf`). Matters only with a constrained objective. |
| `epf` | | | External penalty options with `constraint_handling = "epf"`: `r` (required), `r_factor`, `tol_factor`, `conv_crit`, `max_epf_iter`. |
| `eq_tolerance` | number | 0.0001 | Equality constraints within this absolute distance of zero count as met under the feasibility rules (default 1e-4). |
| `ftol` | number | 1e-06 | Relative improvement of the best member's objective that resets the `stall` count (default 1e-6). |
| `init` | | uniform | How the start population besides `x0` is drawn within the bounds: `"uniform"` (pymoo's random sampling) or `"lhs"` (Latin hypercube). |
| `initial_velocity` | string | random | `"random"` (uniform in `[0, v_max]`, pymoo's) or `"zero"`. |
| `maxiter` | integer | 100 | Generations at most (per outer EPF iteration with `"epf"`; default 100). |
| `pop_size` | integer | 20 | Members of the population, `x0` included (at least 2, default 20); each generation evaluates `pop_size` new members on every realization. |
| `stall` | integer | 20 | Generations in a row without the best member improving by more than `ftol` (relative) after which the run counts as converged (default 20). |
| `topology` | | global | `"global"` (the swarm's best) or `"ring"` (each particle's and its two neighbours' best). |
| `transform` | boolean | false | Optimize in the unit cube of the bounds instead of the controls' own units (default off). |
| `v_max` | number | 0.2 | Velocity limit as a fraction of each control's range (pymoo's 0.2). |
| `w` | number | 0.7298 | Inertia, cognitive and social weights (Clerc and Kennedy's constriction equivalents). |
| `xtol` | number | 1e-08 | The run counts as converged when the population's spread, the largest standard deviation of a control (in optimizer coordinates), falls below it (default 1e-8). |

## `ropt`

ropt's optimizer backends (SciPy, Dakota) run on APEX's objectives.

Select it with:

```toml
[optimizer]
name = "ropt"
```

| option | type | default | description |
|---|---|---|---|
| `constraint_tolerance` | number | 1e-08 | The violation below which a point counts as feasible. |
| `constraints` | boolean | false | Hand a constrained objective's constraints to ropt (default: ignore them, as APEX's optimizers without `epf`). |
| `convergence_tolerance` | number or null | | The backend's tolerance (ropt's `convergence_tolerance`). |
| `epf` | | | Refused: ropt handles constraints itself. |
| `max_batches` | integer or null | | ropt's limits on function evaluations and evaluation batches; `max_batches` bounds the evaluations (approval). |
| `max_functions` | integer or null | | ropt's limits on function evaluations and evaluation batches; `max_batches` bounds the evaluations (approval). |
| `maxiter` | integer | 100 | ropt's `max_iterations` (the backend's iterations) of the run (a resume replays it). |
| `method` | string | scipy/slsqp | ropt's backend method, e.g. `scipy/slsqp`, `scipy/l-bfgs-b`, `scipy/cobyla`, `dakota/optpp_q_newton` (with `ropt-dakota`). |
| `number_of_perturbations` | integer | 5 | Perturbed points per realization of ropt's ensemble gradient. |
| `options` | object or null | | Passed to the backend as its `options` (e.g. SciPy's `ftol`). |
| `perturbation_magnitude` | number or array or null | | Their standard deviation in the optimizer's coordinates (one, or one per control); default ropt's. |
| `seed` | integer or null | | ropt's perturbation seed (default: drawn from the optimization's seed). |
| `transform` | boolean | false | Optimize in the unit cube of the bounds. |

## `smcopt`

Sequential Monte Carlo optimization (SmcOpt): resampled ensembles towards the optimum.

Select it with:

```toml
[optimizer]
name = "smcopt"
```

| option | type | default | description |
|---|---|---|---|
| `alpha` | number | 0.1 | Weight of the particle mean in a step. |
| `alpha_maxiter` | integer | 5 | Backtracking trials per resampling attempt. |
| `best_func` | number or null | | Best objective seen before this run (default: the initial objective). |
| `cov_factor` | number | 0.5 | Covariance shrink factor per resampling attempt. |
| `epf` | | | External penalty options: `r` (required), `r_factor`, `tol_factor`, `conv_crit`, `max_epf_iter` (`EPF_DEFAULTS`). Needs a constrained objective. |
| `ftol` | number or null | | Relative change of the mean objective that counts as converged. |
| `gtol` | number | 1e-05 | Infinity norm of the projected gradient that counts as converged. |
| `inflation_factor` | number | 1.0 | Initial tempering of the weights (it grows with the iteration). |
| `maxiter` | integer | 100 | Iterations per outer EPF iteration. |
| `ne` | integer | 100 | Number of particles (PET's `ne`; default 100). |
| `resample` | integer | 0 | Resampling attempts when backtracking fails. |
| `survival_factor` | number | 1.0 | Fraction of particles drawn anew each iteration (clipped to `[0.1, 1]`). |
| `tol` | number or null | | Improvement a trial needs; also `ftol` when given. |
| `xtol` | number | 1e-08 | Change of the controls (2-norm, optimizer coordinates) that counts as converged. |

## `trust_region`

Trust-region optimization; options are PET's.

Select it with:

```toml
[optimizer]
name = "trust_region"
```

| option | type | default | description |
|---|---|---|---|
| `epf` | | | External penalty options: `r` (required), `r_factor`, `tol_factor`, `conv_crit`, `max_epf_iter` (`EPF_DEFAULTS`). Needs a constrained objective. |
| `eta1` | number | 0.05 | Ratios below which the radius shrinks by `gam1`, and above which (at the boundary) it grows by `gam2`. |
| `eta2` | number | 0.5 | Ratios below which the radius shrinks by `gam1`, and above which (at the boundary) it grows by `gam2`. |
| `ftol` | number | 1e-05 | Relative change of the mean objective that counts as converged. |
| `gam1` | number | 0.5 | Radius factors. |
| `gam2` | number | 1.5 | Radius factors. |
| `gtol` | number | 1e-05 | Infinity norm of the projected gradient that counts as converged. |
| `maxiter` | integer | 100 | Iterations per outer EPF iteration. |
| `method` | string | iterative | Subproblem solver: `iterative` (exact) or `CG-Steihaug`. |
| `quasi_newton` | boolean | false | Approximate the Hessian by BFGS (PET's `hess = "BFGS"`). |
| `resample` | boolean | false | Re-evaluate the gradient and the Hessian after a rejected step. |
| `rho_tol` | number | 1e-06 | Least ratio of actual to predicted decrease for a step to be accepted. |
| `transform` | boolean | false | Optimize in the unit cube of the bounds. |
| `trust_radius` | number | 1.0 | Initial radius. |
| `trust_radius_cuts` | integer | 4 | Radius cuts before a step is rejected. |
| `trust_radius_max` | number or null | | Radius limits (default `100` and `1/1000` times the initial radius); the run stops at the minimum. |
| `trust_radius_min` | number or null | | Radius limits (default `100` and `1/1000` times the initial radius); the run stops at the minimum. |
| `xtol` | number | 1e-08 | Change of the controls (2-norm, optimizer coordinates) that counts as converged. |

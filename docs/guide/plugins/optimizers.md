---
audience: public
items:
  - cli:apex optimize
  - cli:apex plugins
  - cli:apex resume
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

## Goal

By the end of this lesson you will know which optimizer to pick for your problem, how to start and continue an optimization with `apex optimize`, and where to look up each optimizer's options.

Two kinds of things are described here:

- **Commands** you type: `apex optimize`, `apex run`, `apex resume`, `apex stop` and `apex plugins`.
- **Optimizer plugins** you choose by name in your optimization configuration (a TOML file): `enopt`, `genopt`, `line_search`, `smcopt`, `trust_region`, `ga`, `pso`, `ropt`, `nsga2`, `nsga3` and `moead`.

## Prerequisites

- APEX is installed as described in the README. After that installation `apex` is not on your PATH, so every command is typed from the APEX folder as `uv run apex ...`.
- You have an optimization configuration file in TOML. Below it is called `CONFIG`.

## Step 1: Choose an optimizer

Every optimizer has a status. `parity` means it follows PET's optimizer of the same name, and its options are PET's. `experimental` means it is newer and may change.

| plugin | status | what it does | typical use |
|---|---|---|---|
| `enopt` | parity | Ensemble optimization (EnOpt): steps along an ensemble estimate of the gradient. | A good first choice for one objective. |
| `genopt` | parity | Generalized ensemble optimization (GenOpt): EnOpt with marginals and copulas, or CMA. | One objective, when you want a distribution per control (`marginal`) or an adapted correlation (`cma`). |
| `smcopt` | parity | Sequential Monte Carlo optimization (SmcOpt): resampled ensembles towards the optimum. | One objective, ensemble based. |
| `line_search` | parity | Line-search optimization (`method`: `GD`, `BFGS` or `Newton-CG`). | One objective, classic gradient-based search. |
| `trust_region` | parity | Trust-region optimization (`method`: `iterative` or `CG-Steihaug`). | One objective, classic gradient-based search. |
| `ga` | experimental | A real-coded genetic algorithm: SBX crossover, polynomial mutation, tournaments. | One objective, population based. |
| `pso` | experimental | Particle swarm optimization: particles drawn to their own and the swarm's best. | One objective, population based. |
| `ropt` | experimental | ropt's optimizer backends (SciPy, Dakota) run on APEX's objectives. | When you want a backend such as `scipy/slsqp` (the default `method`). |
| `nsga2` | experimental | NSGA-II: multi-objective optimization by non-dominated sorting and crowding. | Several objectives at once. |
| `nsga3` | experimental | NSGA-III: many-objective NSGA with survival by reference directions. | Many objectives. |
| `moead` | experimental | MOEA/D: multi-objective optimization by neighbouring scalar subproblems. | Several objectives at once. |

Several objectives are declared with `optimization:objectives`: a list of scalar objectives optimized together (for example by `nsga2`). Each entry is a plugin with its options and an optional label. Response objectives share one simulation per member, where a member is one candidate set of controls that the optimizer evaluates.

## Step 2: Look up the options

List every plugin with its options, types and defaults:

```bash
uv run apex plugins
```

For the full descriptions as JSON, add `--json`:

```bash
uv run apex plugins --json
```

In the lists below, "unset" means the option has no value by default (`null`). The text says what then happens where it matters.

### Constraints and the external penalty (EPF)

A constrained objective has constraints the controls must satisfy. The external penalty function (EPF) method handles them by solving a series of penalised problems. You turn it on with the `epf` option, which has `r` (required), `r_factor` (default 2), `tol_factor` (0.9), `conv_crit` (1e-5) and `max_epf_iter` (10). Each of these outer EPF iterations runs the optimizer for up to `maxiter` iterations. `epf` needs a constrained objective. For an unconstrained objective there is no outer loop, and `maxiter` is simply the maximum number of iterations (generations for the population methods).

### Options shared by most optimizers

- `maxiter` (integer, default 100): iterations per outer EPF iteration, as described above. For `ga`, `pso`, `nsga2`, `nsga3` and `moead` it counts generations.
- `ftol`: relative change of the mean objective that counts as converged. Defaults: unset for `enopt`, `genopt` and `smcopt`; 1e-05 for `line_search` and `trust_region`; 1e-06 for `ga` and `pso`, where it is the relative improvement of the best member that resets the `stall` count; 0.0025 for `nsga2`, `nsga3` and `moead`, where it is the movement of the feasible front. For `enopt` and `smcopt`, giving `tol` also sets `ftol`.
- `xtol` (default 1e-08): change of the controls that counts as converged. For the population methods it is the spread of the population.
- `gtol` (default 1e-05): infinity norm of the projected gradient that counts as converged. The gradient-based and ensemble optimizers have it.
- `transform` (boolean, default false): optimize in the unit cube of the bounds instead of the controls' own units.
- `epf`: the external penalty options above. `ropt` refuses `epf` because it handles constraints itself.

### Ensemble methods (`enopt`, `genopt`, `smcopt`)

- `ne` (integer, default 100): ensemble size, or the number of particles for `smcopt`.
- `alpha` (default 0.1): initial step size for `enopt` and `genopt`. For `smcopt` it is the weight of the particle mean in a step. For `enopt` and `genopt`, `step_size` (unset by default) overrides it when given.
- `alpha_maxiter` (default 5): backtracking trials per resampling attempt.
- `resample` (default 0): resampling attempts when backtracking fails.
- `cov_factor` (default 0.5): covariance shrink factor per resampling attempt.
- `enopt` also has `beta` (momentum, default 0), `nesterov`, `natural_gradient`, `ensemble_hessian`, `hessian`, `normalize` and `optimizer`. The step rule `optimizer` is `GD` (the default), `Adam`, `AdaMax` or `Steihaug`.
- `genopt` also has `marginal` (`BetaMC`, the default, or `TruncGaussian`), `alpha_theta`, `cma` and the `cma_*` options.
- `smcopt` also has `inflation_factor`, `survival_factor` and `best_func`.

### What `ensemble_hessian`, `hessian` and `normalize` do in `enopt`

- `ensemble_hessian` (boolean, default false): estimate the ensemble Hessian. The estimate adapts the covariance and feeds the `Steihaug` step rule.
- `hessian` (boolean, default false): precondition the gradient with the Hessian. It uses the Hessian estimate, so set `ensemble_hessian` to true as well.
- `normalize` (boolean, default true): scale the gradient and the Hessian by their infinity norms. You normally leave it alone. With `AdaMax` as `optimizer`, `enopt` turns it off for you.

### Gradient-based methods

- `line_search`: `method`, `step_size` (unset by default: the first step length is `0.25 / ||p||_inf`), `step_size_max`, `step_size_adapt`, the Wolfe constants `c1` and `c2`, `rho` (backtracking factor), `lsmaxiter` and `lsmethod` (0 is Armijo backtracking, 1 is strong Wolfe).
- `trust_region`: `method`, `quasi_newton`, `trust_radius` (initial radius, default 1.0), `trust_radius_max` and `trust_radius_min` (unset by default: 100 and 1/1000 times the initial radius; the run stops at the minimum), `trust_radius_cuts`, `rho_tol`, `eta1`, `eta2`, `gam1`, `gam2` and `resample`.

### Population methods (`ga`, `pso`, `nsga2`, `nsga3`, `moead`)

- `pop_size`: number of members (default 20 for `ga` and `pso`, 40 for `nsga2`). For `nsga3` the default follows the number of reference directions.
- `n_offsprings`: children per generation. Unset means `pop_size`.
- `stall`: generations in a row without enough improvement after which the run counts as converged (default 20 for `ga` and `pso`, 30 for the multi-objective ones).
- `init`: `uniform` (the default) or `lhs` (Latin hypercube) for the starting population.
- `constraint_handling`: `feasibility` (the default) or `epf`.
- `crossover_prob`, `crossover_prob_var`, `eta_c`, `mutation_prob`, `mutation_prob_var` and `eta_m` tune crossover and mutation for `ga`, `nsga2`, `nsga3` and `moead`. Where `mutation_prob_var` is unset, it is `min(0.5, 1/n_x)` for `n_x` controls. `eliminate_duplicates` (default true) redraws duplicate children.
- `pso` has its own `w`, `c1`, `c2`, `topology`, `v_max`, `initial_velocity` and `bounds_handling`.
- `nsga2`, `nsga3` and `moead` take a `ref_point` for the hypervolume. Unset means it is taken from the first feasible members. `nsga3` and `moead` also take `n_partitions` and `inner_partitions`. `moead` additionally has `decomposition`, `n_neighbors`, `prob_neighbor_mating` and `theta`.

### `ropt`

- `method` (default `scipy/slsqp`), `max_functions`, `max_batches`, `number_of_perturbations`, `perturbation_magnitude`, `convergence_tolerance`, `seed`, `options`, `constraints` and `constraint_tolerance`. The options `max_functions`, `max_batches`, `convergence_tolerance`, `seed`, `options` and `perturbation_magnitude` are unset by default; the backend's own defaults then apply, except `seed`, which is drawn from the optimization's seed.

## Step 3: Run the optimization

```bash
uv run apex optimize CONFIG
```

| flag | meaning |
|---|---|
| `--max-iterations N` | Stop after N accepted iterations. The run can be resumed. |
| `--resume DIR` | Continue the optimization run in directory DIR. |
| `--progress` / `--no-progress` | Turn progress on (the default) or off. Progress goes to standard error: bars on a terminal, timestamped lines otherwise. |

An accepted iteration is one whose step was accepted by the optimizer, as opposed to a trial step that was rejected, for example during backtracking. It is not the same as `maxiter`, which is the optimizer's own convergence limit.

## Worked example: run in pieces, then continue

Start an optimization but stop after 5 accepted iterations, for example to check that the objective is moving in the right direction:

```bash
uv run apex optimize CONFIG --max-iterations 5
```

The run keeps its state in a run directory. If you do not know where it is, the page on run directories explains where a run's output goes. If you are happy with the first 5 iterations, continue the same run by pointing `--resume` at that directory (`RUN_DIR` stands for it):

```bash
uv run apex optimize --resume RUN_DIR --max-iterations 20
```

`CONFIG` is optional when you resume, since the run already exists. Leave out `--max-iterations` to let the run continue until the optimizer converges.

To run without progress output, for example in a batch job whose log you want to keep clean, use a configuration whose run directory does not hold a run yet:

```bash
uv run apex optimize CONFIG --no-progress
```

## Which command continues which run

- A run started with `apex optimize` is continued with `apex optimize --resume RUN_DIR`.
- A run started with `apex run` is continued with `apex resume RUN_DIR`.

The other commands on this page:

- `apex run CONFIG` starts a history-matching run. It is a different workflow from `apex optimize`. `--max-steps N` stops it after N analysis steps.
- `apex resume RUN_DIR` continues a stopped or interrupted run from its last checkpoint, which is a saved state the run can restart from. `--max-steps N` and `--progress`/`--no-progress` work as for `apex run`.
- `apex stop RUN_DIR` asks a running run to stop at its next checkpoint. `RUN_DIR` is the directory of a running run.

```bash
uv run apex run CONFIG --max-steps 3
uv run apex resume RUN_DIR
uv run apex stop RUN_DIR
```

To stop an optimization early, use `--max-iterations` as shown above.

## Pitfalls

- If the run directory of `CONFIG` already holds an optimization run, `apex optimize CONFIG` stops with an error saying to resume the run or remove it. Use `--resume`, or remove the old run if you want a fresh start.
- `--max-iterations` counts accepted iterations. A run that stops there has not failed; continue it with `--resume`.
- `hessian` in `enopt` works together with `ensemble_hessian`, and the `Steihaug` step rule needs `ensemble_hessian` too. Turn it on in both cases.
- `epf` needs a constrained objective. With `ga`, `pso`, `nsga2` or `nsga3` it only applies when `constraint_handling` is `epf`. With `moead`, `feasibility` (the default) refuses constrained objectives, so use `epf`.
- `ropt` ignores a constrained objective's constraints unless `constraints` is true.
- Default population sizes differ (20 for `ga`, 40 for `nsga2`). Each member costs a simulation, so a large `pop_size` is expensive.
- This page does not show a complete configuration file. Use `uv run apex plugins` for the exact option names and defaults of your APEX version.

## What to read next

Read the pages on the optimization configuration and on objectives to learn how to write `CONFIG`. Then read the page on run directories to see where a run's output goes and how to resume it.

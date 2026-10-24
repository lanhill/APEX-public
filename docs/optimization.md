---
audience: public
---

# Optimization

## Goal

By the end of this lesson you can run an optimization with APEX, choose an optimizer, optimize a simulated objective such as NPV, map a few controls onto a deck with parameterizations, and compare optimizers in a study.

APEX ports PET's `popt`. It provides ensemble optimizers (`enopt`, `genopt`, `smcopt`), deterministic optimizers (`line_search`, `trust_region`), population optimizers (`ga`, `pso`), multi-objective optimizers (`nsga2`, `nsga3`, `moead`) and analytic test objectives. Each is a plugin. Whole optimizations are parity-tested against PET (`docs/migration/pet_parity.md`).

## Prerequisites

- A working APEX checkout. The commands below use `uv run apex ...`.
- For simulated objectives: a deck simulator that APEX supports (`opm_flow`, `eclipse`) and a Mako deck template.
- Familiarity with history-matching configurations: simulators, executors and parameterizations are shared with them (see the guide's [simulators](guide/plugins/simulators.md), [executors](guide/plugins/executors.md) and [parameterizations](guide/plugins/parameterizations.md)).

## Vocabulary used below

- **Controls**: the values the optimizer changes. `n_x` is how many there are.
- **Realization**: one model of a robust optimization. `n_r` is the number of realizations (`[controls] realizations`).
- **Ensemble**: the set of control vectors evaluated together. `n` is the number of members (columns).
- **EPF (external penalty function)**: the way constraints are handled, by adding a penalty. See [Constraints](#constraints).
- **Pareto front**: the members that no other member beats in every objective.
- **Nadir point**: the worst value of each objective among the front's members.
- **Ideal point**: the best value of each objective.
- **Reference point**: a point worse than the front, used to measure the hypervolume.
- **Hypervolume**: the size of the objective space between the front and the reference point. A bigger one is a better front.
- **Representative**: the one front member that `result.json` reports: the member closest to the ideal point, in objectives normalised by their range.

## Step 1: a first run

```bash
uv run apex optimize examples/optimization/rosenbrock_enopt.toml [--max-iterations N]
uv run apex optimize --resume runs/rosenbrock_enopt [--max-iterations N]
uv run apex plugins            # optimizers, objectives and step_rules, with their options and defaults
```

The configuration:

```toml
[optimization]
name = "rosenbrock_enopt"
seed = 2026                    # seeds the perturbations' random stream

[controls]
realizations = 1               # model realizations of robust optimization (PET's num_models)

[[controls.variables]]         # one block per variable; values are concatenated in order
name = "x"
initial = [-1.2, 1.0, -0.5, 0.8]
lower = -2.0
upper = 2.0
variance = 0.001               # initial perturbation variance (ensemble optimizers)

[objective]
name = "rosenbrock"            # or "quadratic"

[optimizer]
name = "enopt"
options = { ne = 20, maxiter = 300, alpha = 0.05, optimizer = "Adam", beta = 0.9 }
```

The run directory is `runs/<name>/` next to the configuration (or `[optimization] output_dir`). It holds:

- `config.json`: the resolved configuration;
- `history.jsonl`: a row for the start, every accepted step and every EPF (external penalty) update, with the controls, the objective, the evaluation counts and the optimizer's own columns;
- `checkpoint.npz`/`.json`: the optimizer's state and its random stream. `--resume` continues bit-identically;
- `result.json`, once the run ends.

`--max-iterations N` stops after N accepted steps in this invocation, and the run can be resumed. One process at a time holds `run.lock`.

**What to look at afterwards.** Open `result.json` for the best controls and the final objective value. Open `history.jsonl` to see how the objective fell, one row per accepted step. Rosenbrock's minimum is 0 at all controls equal to 1, so a good run ends near that.

## Step 2: choose an optimizer

Options keep PET's names and defaults. `apex plugins` lists them.

| Plugin | What | Main options |
|---|---|---|
| `enopt` | EnOpt with Gaussian perturbations: natural gradient, backtracking, resampling; the ensemble Hessian adapts the covariance (Zhang et al. 2023) | `ne`, `alpha`, `beta`, `nesterov`, `optimizer`, `ensemble_hessian`, `hessian`, `alpha_cov`, `resample`, `cov_factor`, `natural_gradient`, `transform` |
| `genopt` | GenOpt: Gaussian-copula perturbations with `BetaMC` or `TruncGaussian` marginals. The marginal's `theta` follows its own gradient, and CMA can adapt the correlation | `ne`, `marginal`, `theta`, `alpha_theta`, `cma`, `optimizer`, `separate_mutation_ensemble` |
| `smcopt` | Sequential Monte Carlo: weighted particles with survival and resampling | `ne`, `alpha`, `inflation_factor`, `survival_factor`, `resample` |
| `line_search` | GD, BFGS or Newton-CG directions, with a Wolfe or backtracking line search | `method`, `lsmethod`, `step_size`, `step_size_adapt`, `c1`, `c2` |
| `trust_region` | Trust region, with the exact or CG-Steihaug subproblem, and the Hessian or BFGS | `method`, `quasi_newton`, `trust_radius`, `eta1`, `eta2`, `gam1`, `gam2` |

**Which one to start with.** For an objective computed by a simulator, start with `enopt`: it estimates the gradient from an ensemble and needs no derivatives. `line_search` and `trust_region` need exact derivatives, so use them only with an objective that provides `gradient` (and `hessian`). For a few controls and several local optima, try `ga` or `pso` below.

All of them take:

- `maxiter`, `ftol`, `xtol` and `gtol` (convergence);
- `transform`, which optimizes in the unit cube of the bounds;
- `epf`, the external penalty options (see below).

### Constraints

EPF turns constraints into a penalty added to the objective. It needs an objective with constraints. It works with the ensemble optimizers, whose estimated gradient includes the penalty, and with the population optimizers below (`constraint_handling = "epf"`). It does not work with `line_search` and `trust_region`, whose exact gradient lacks the penalty. Its options (`r`, `r_factor`, `tol_factor`, `conv_crit`, `max_epf_iter`) are given as `epf = { r = ... }`; `apex plugins` shows the defaults.

### Population optimizers: `ga` and `pso`

Global, derivative-free searches that keep a population, ported from pymoo 0.6.2 (their operators return pymoo's numbers exactly, given the same random generator):

| Plugin | What | Main options |
|---|---|---|
| `ga` | A real-coded genetic algorithm: binary tournaments, SBX crossover, polynomial mutation, elitist survival of the best `pop_size` | `pop_size` (20), `n_offsprings`, `crossover_prob` (0.9), `crossover_prob_var` (0.5), `eta_c` (15), `mutation_prob` (0.9), `mutation_prob_var`, `eta_m` (20), `eliminate_duplicates` |
| `pso` | Particle swarm with an inertia weight, Clerc and Kennedy's constriction-equivalent weights | `pop_size` (20), `w` (0.7298), `c1`, `c2` (1.49618), `topology` (`global` or `ring`), `v_max` (0.2 of each range), `initial_velocity`, `bounds_handling` (`reflect` or `clip`) |

Both also take:

- `init` (`uniform` or `lhs`; `x0` is always a member);
- `stall` (20) and `ftol`: stop when the best has not improved by more than `ftol` (relative) for `stall` generations;
- `xtol`: stop when the population's spread falls below it;
- `maxiter` generations, `transform`;
- `constraint_handling` and `eq_tolerance`.

They need finite bounds on every control.

- **When to use them:** few controls, several local optima, cheap or proxy objectives. For a smooth objective with a narrow curved valley, use `enopt` or a gradient method instead. On Rosenbrock in five dimensions, after 24,000 evaluations `ga` is still near 1 and `pso` near 0.05.
- **Cost:** a generation is `pop_size` members (for `ga`, `n_offsprings`) on every realization, and a run typically takes 30–100 generations. That is 600–5,000 evaluations per realization, against 10–20 per `enopt` iteration.
- **The evaluation bound** is an upper bound, because duplicate elimination and the stall stop can end a run below it. Write `n_new` for the new members per generation (`pop_size`, or `n_offsprings` for `ga`). The bound is `pop_size × n_r` for the start, plus `maxiter × n_new × n_r`. Each EPF outer iteration repeats the generations; an EPF update itself evaluates nothing.
- **Every generation is a checkpoint:** a stop or crash resumes bit-identically from the last generation.
- **Constraints:**
  - `constraint_handling = "feasibility"` (the default, Deb's rules): feasible members beat infeasible ones, and among infeasible ones the smaller violation wins (pymoo's sum, equalities within `eq_tolerance`). An equality constraint needs a tolerance the variation can hit. At pymoo's `1e-4` both optimizers stall on the equality far from the optimum (measured on PET's quadratic); at `1e-2` they reach it, offset by up to the tolerance.
  - `constraint_handling = "epf"` uses the `epf` options. It handles equalities well from `r ≈ 100`.

### Multi-objective: `nsga2`

NSGA-II (Deb et al. 2002), ported from pymoo 0.6.2, returns a Pareto front in one run. It needs a vector objective (below); the single-objective optimizers refuse one, and `nsga2` refuses a scalar one.

| Option | What |
|---|---|
| `pop_size`, `n_offsprings` | population and offspring per generation |
| `crossover_prob`, `crossover_prob_var`, `eta_c`, `mutation_prob`, `mutation_prob_var`, `eta_m`, `eliminate_duplicates` | SBX and polynomial mutation, as `ga` |
| `ref_point` | the hypervolume's reference point. Unset, it is the nadir of the feasible members plus 10 % of their range, fixed when the representative is first feasible. Members beyond it add nothing to the run's hypervolume; give one to fix it |
| `stall`, `ftol` | stop after `stall` generations in a row in which the feasible front moved less than `ftol`. The movement is pymoo's measure: the largest change of the front's ideal and nadir points, and of its IGD to the previous front, in objectives normalised by the front's range. IGD (inverted generational distance) is the average distance from the previous front's points to the nearest point of the new front, so it is small when the front has stopped changing. Before any member is feasible, a generation is progress if the least violation fell |
| `maxiter`, `xtol`, `init`, `transform`, `constraint_handling`, `eq_tolerance`, `epf` | as `ga`; `epf` penalises every objective alike |

The front holds only members that satisfy the constraints, under `epf` too. The EPF outer loop then ends once any member is feasible, so `r` matters only until then. Survival is pymoo's `RankAndCrowding`, and mates are picked by pymoo's binary tournament. Its operators return pymoo's numbers exactly (`docs/migration/pymoo.md`).

**The result is a set.** The run directory gains `pareto.json`, the front at the last checkpoint: each member's controls, its objectives (means over the realizations and per realization) and violation, the objectives' names, the hypervolume and its reference point. `result.json` keeps the representative. `history.jsonl` has, per generation, the representative's objectives (`objective`, the first, and `objective:<name>`), the `hypervolume` (exact for up to four objectives; with more, the front only), `front_size`, `spread` and, if constrained, `feasible_share`.

A vector objective is either:

- one plugin that returns several objectives (the test problems below), in `[objective]`;
- a list of scalar objectives, each with a `label`, in `[[objectives]]` (exactly one of the two forms). Simulated ones share one simulation per member: the union of their summary keys is simulated once.

```toml
[[objectives]]
name = "npv"
label = "npv"
options = { start = "2000-01-01", times = ["2001-01-01", "2002-01-01", "2003-01-01"], wop = 1.0, wwi = 0.02, disc = 0.08, obj_scaling = -1000.0 }

[[objectives]]
name = "field_total"            # a summary total at one time, scaled
label = "water_injected"
options = { vector = "FWIT", time = "2003-01-01", scale = 0.001 }

[optimizer]
name = "nsga2"
options = { pop_size = 20, maxiter = 30, transform = true }
```

This is `examples/toy/toy_reservoir_npv_water/`, on the analytic `toy_reservoir`: it runs in seconds and needs no reservoir simulator. `apex report OPT_DIR --plots` writes the objective, the hypervolume over the generations and the front (a scatter for two objectives, parallel coordinates for more), one figure per file, PNG and PDF. Validation refuses an optimizer that does not match its objectives.

**Statistics over realizations.** In a robust optimization (`[model] realizations_file`), every member is simulated on every realization, and each `[[objectives]]` entry is ranked by its `statistic` over them:

| `statistic` | Ranked |
|---|---|
| `mean` (default) | the mean |
| `std` | the sample standard deviation: the risk (at least 2 realizations) |
| `worst` | the largest value. Every objective is minimised, so for −NPV it is the realization with the lowest NPV |
| `p1` … `p99`, e.g. `p10`, `p50`, `p90` | that percentile of the objective as minimised. For −NPV, `p90` is NPV's 10th percentile, a pessimistic value |

The same objective may be listed with several statistics at no extra simulation cost: each member is simulated once per realization, and every entry reads the same simulations. An entry without a `label` is shown as `<name>_<statistic>`, or as `<name>` for the mean. `pareto.json` and `result.json` record `objective_statistics`; the front's `objectives` are the ranked statistics, and `objectives_per_realization` every value. `examples/toy/toy_reservoir_robust_npv/` trades NPV's mean against its standard deviation over 5 realizations (in the file, the `options` of both entries are identical and filled in):

```toml
[[objectives]]
name = "npv"
label = "npv_mean"
statistic = "mean"
options = { ... }

[[objectives]]
name = "npv"
label = "npv_std"
statistic = "std"
options = { ... }        # the same options: the same simulations
```

A statistic other than `mean` is for the multi-objective optimizers only.

**Cost.** A front needs a larger population than one optimum. The bound is `pop_size × n_r + maxiter × n_offsprings × n_r` evaluations. For `pop_size = 40`, `n_offsprings = 40`, `maxiter = 150` and one realization, that is 40 + 150 × 40 = 6,040.

### Many objectives: `nsga3` and `moead`

Both take the vector objectives of `nsga2` and share its results (`pareto.json`, the representative, the front-movement stall), checkpoints and bit-identical resume.

- **`nsga3`** (Deb & Jain 2014). NSGA-II's crowding loses its pressure beyond about three objectives, where nearly every member is non-dominated. NSGA-III keeps the population spread along reference directions on the unit simplex (Das-Dennis): it normalises the objectives by an adaptive hyperplane, associates each member with its nearest direction, and fills the last front from the least crowded directions. Parents are picked at random among feasible members, else by violation.
- **`moead`** (Zhang & Li 2007) solves one scalar subproblem per weight vector (Tchebicheff for two objectives, PBI with θ = 5 for more). Each child comes from two parents of its subproblem's neighbourhood (the `n_neighbors` nearest weights, with probability `prob_neighbor_mating`, else from anywhere). A generation makes one child per subproblem and evaluates them as one batch; each child then replaces the neighbours whose decomposed value it improves. Constraints only through EPF (`constraint_handling = "epf"`).

The number of reference directions (or weights) is set by `n_partitions` and `inner_partitions`; the population follows the directions (`pop_size` for `nsga3`). The defaults are smaller than pymoo's and the papers', because here an evaluation is a reservoir simulation on every realization. The front is therefore coarser. Raise `n_partitions` (and `pop_size`) for a denser front. `apex plugins` shows the defaults.

Examples: `examples/toy/multi_objective/dtlz1/`, `dtlz2/` and `dtlz2_5/` (five objectives, NSGA-III against NSGA-II), and `zdt1/` (MOEA/D).

## Objectives

| Plugin | Function |
|---|---|
| `rosenbrock` | SciPy's Rosenbrock function, with gradient and Hessian |
| `quadratic` | `0.5 ‖x − 1‖²`, with constraints for EPF |
| `rastrigin` | `10 n + Σ x_i² − 10 cos(2π x_i)`: many local minima, global 0 at 0 |
| `ackley` | Ackley's function: a flat outer region with many local minima, global 0 at 0 |

Vector-valued test problems for `nsga2` and its relatives are taken from pymoo 0.6.2 (`pymoo/problems/multi/`; Blank & Deb 2020). Their examples are in `examples/toy/multi_objective/`.

| Plugin | Problem | n_x | Objectives | Front |
|---|---|---|---|---|
| `sch` | Schaffer 1985 | 1 | 2 | convex |
| `fon` | Fonseca & Fleming 1995 | 3 | 2 | non-convex |
| `kur` | Kursawe 1991 | 3 | 2 | disconnected |
| `zdt1`, `zdt2`, `zdt3` | Zitzler, Deb & Thiele 2000 | ≥ 2 (examples: 10) | 2 | convex, non-convex, disconnected |
| `dtlz1` | Deb, Thiele, Laumanns & Zitzler 2005 | ≥ `n_obj` (examples: 7) | `n_obj` (3) | linear, the simplex `Σ f = 0.5` |
| `dtlz2` | Deb, Thiele, Laumanns & Zitzler 2005 | ≥ `n_obj` (examples: 12, 14) | `n_obj` (3) | the unit sphere's positive part |
| `bnh` | Binh & Korn 1997 | 2 | 2 | convex |
| `tnk` | Tanaka et al. 1995 | 2 | 2 | disconnected |
| `srn` | Srinivas & Deb 1994 | 2 | 2 | convex |

`bnh`, `tnk` and `srn` have inequality constraints.

**Writing a new objective.** It is a class with `__call__(controls, realizations) -> values`, registered in `objectives`:

- `controls` has shape `(n_x, n)`: `n_x` control values for each of `n` ensemble members, one member per column. The values are in the controls' own units (the units of the `[controls]` bounds), not the unit cube that `transform` uses;
- `realizations` has shape `(n,)`: the model realization of each column;
- it returns `(n,)` values.

`gradient`/`hessian` add exact derivatives (needed by `line_search` and `trust_region`), and `constraints` adds the EPF constraints. A vector objective (`VectorObjective`) also declares `objective_names` and returns `(n_obj, n)`.

## Step 3: a simulated objective (NPV)

An objective of a simulator's responses (registry `response_objectives`) runs each column of controls as one simulation. The member's state is the control blocks by name, so a deck template reads `${injbhp[0]}`. The simulators and executors are the history-matching ones (`opm_flow`, `eclipse`, `process`, ...), with their retries and the machine-wide resource pools (Eclipse licences, cores).

`examples/optimization/3well/enopt.toml` is PET's popt tutorial. Its folder holds the configuration, the deck template `deck.mako` and, for the robust variant, `models.npz`. The objective part of it:

```toml
[objective]                    # -NPV in millions: minimising it maximises NPV
name = "npv"
options = { start = "1994-02-09", times = ["1995-01-01", "1996-01-01", "1997-01-01", "1998-01-01", "1999-01-01"], wop = 283.05, wwp = 37.74, wwi = 12.58, disc = 0.08, obj_scaling = -1.0e6 }

[simulator]
name = "opm_flow"
options = { deck_template_file = "deck.mako" }

[execution]
name = "process"
options = { workers = 4 }

[model]                        # optional: robust optimization over fixed model realizations
realizations_file = "models.npz"   # arrays (n_values, n_realizations) by name, e.g. permx

[output]
keep_simulations = "failed"    # all, failed (default) or none
```

Run it as in step 1, with a few iterations first:

```bash
uv run apex optimize examples/optimization/3well/enopt.toml --max-iterations 3
```

A successful run prints its progress and writes `history.jsonl` and `result.json` as in step 1. Here the objective is −NPV in millions, so it becomes more negative as NPV improves. Simulation folders are in `simulations/eval_<id>/`.

**How the deck reads the controls.** The template is a Mako file. A control block named `injbhp` (after the parameterization in step 4) is a list with one value per report step. A sketch of an injector record per report step:

```mako
% for k in range(5):
WCONINJE
  'INJ1' 'WATER' 'OPEN' 'BHP' 1* 1* ${injbhp[k]} /
/
TSTEP
  365 /
% endfor
```

The `npv` objective's `times` list the report dates, so five dates means five steps and `k` runs from 0 to 4.

| Plugin | Function | Options |
|---|---|---|
| `npv` | PET's NPV: `Σ_k (ΔFOPT r_o + ΔFGPT r_g − ΔFWPT r_w − ΔFWIT r_wi) / (1 + d)^(days_k / 365)`, divided by `obj_scaling` | `start` (the deck's START, where the totals are zero; checked against the summary's `TIME` at the first report), `times` (report dates), `wop`, `wgp`, `wwp`, `wwi`, `disc`, `obj_scaling`, `from_time` (one of `times`: count only the steps after it and discount from it, for a forecast after a history; not in PET, unset by default) |
| `ren_npv` | PET's `ren_npv`: as `npv`, but injection is priced well by well, `wwi` for `injectors` and `wrenwi` for `renewable_injectors`, from their WWIT | as `npv` (without FWIT), `injectors`, `renewable_injectors`, `wrenwi` |
| `ecalc_npv` | `npv` with an emission cost from eCalc (needs eCalc and Java, below) | as `npv`, `wem`, `ecalc_model_file`, `time_series` |
| `ecalc_emissions` | APEX's own: the emission eCalc computes, in tonnes over the steps, discounted by `disc` (default 0), divided by `obj_scaling` (positive); a second objective next to `npv` for `nsga2` (`examples/toy/toy_reservoir_npv_co2/`) | `start`, `times`, `ecalc_model_file`, `time_series`, `emission`, `disc`, `obj_scaling` |
| `ecalc_pareto_npv` | PET's `ecalc_pareto_npv`: `ecalc_npv` combined with an emission objective | as `ecalc_npv`, one of `wemc`, `w` |
| `ren_npv_co2` | PET's `ren_npv_co2`: `ecalc_npv` where eCalc also gets `THP_MAX` and `WIND_POWER` | as `ecalc_npv`, `injectors`, `wind_power_file` |

The eCalc objectives need eCalc (Equinor's `libecalc`), in APEX's default environment (`uv sync --locked`); when APEX is a dependency of your own project, name the `ecalc` extra. It brings a Java-based thermodynamics library, so Java must be installed. The simulations' rates replace the model's time series in memory, and eCalc's periods must be the report steps, which is checked. Unlike PET, `ren_npv_co2` discounts from the start and dates each step's rates at its start, as the other NPVs do (`docs/migration/pet_parity.md`).

**The evaluation journal.** `evaluations.jsonl` in the run directory records every simulated evaluation, with its value and responses. Each entry is keyed by the bytes of its controls, its realization, and a hash of the objective and simulator sections, the files they name (the deck template) and the model file. A resume after any of those changed therefore simulates again. A key that is in the journal is never simulated again. A resumed run restarts the iteration it was in and redraws the same controls; it finds them in the journal and does not simulate them again. The same holds for a point that the loop evaluates twice. A failed simulation stops the run. The batch's successful members stay in the journal, so a resume after the cause is fixed simulates only the failed one again.

With `[model] realizations_file`, column `r` of each array goes into the simulations of realization `r`, and `[controls] realizations` follows the file.

A new response objective has `response_keys()`, the `DataKey`s it needs (summary vectors at report dates), and `evaluate(responses)`, which maps `(n_keys, n)` to `(n,)`.

## Step 4: controls through parameterizations

`[[parameterizations]]` map the optimizer's controls to what the deck reads, applied in order. They are the same plugins as a history match's. This page uses the two in the table below; the guide's [parameterizations](guide/plugins/parameterizations.md) page covers the others.

- The control blocks and the `[model]` arrays are the variables.
- Each parameterization consumes some and adds its outputs.
- What nothing consumes passes through.
- The journal's key includes the parameterizations and the files they name, so a change simulates again.

```toml
[[controls.variables]]        # 3 normalized values: the optimizer sees few, well-scaled controls
name = "u"
initial = [0.5, 0.5, 0.5]
lower = 0.0
upper = 1.0
variance = 0.01

[[parameterizations]]         # onto the injectors' pressure range
name = "control_scaling"
options = { variable = "u", lower = 150.0, upper = 400.0, output = "bhp_knots" }

[[parameterizations]]         # held over 10 report steps: the deck reads ${injbhp[k]}
name = "control_schedule"
options = { variable = "bhp_knots", knots = [0, 3, 6], n_steps = 10, output = "injbhp" }
```

| Plugin | Function | Options |
|---|---|---|
| `control_schedule` | `K` values over `n_steps` report steps: each held from its knot to the next (`hold`), or interpolated between knots and held after the last (`linear`) | `variable`, `knots` (first step of each value, from 0), `n_steps`, `interpolation`, `output` |
| `control_scaling` | `lower + (upper − lower) u` for normalized `u` | `variable`, `lower`, `upper` (scalars or one per value), `output` |

The optimizer sees the controls, not what they map to. `[controls]` bounds and `variance` are therefore in the controls' own units: in the example `u` is in `[0, 1]`, so a variance of `0.01` is a standard deviation of 10 % of the range.

**Several wells.** One `control_schedule` spreads one block over time, so give each well its own block, its own scaling and its own schedule, each with its own `output`. A complete example with two injectors:

```toml
[optimization]
name = "two_injectors"
seed = 2026

[controls]
realizations = 1

[[controls.variables]]
name = "u1"
initial = [0.5, 0.5]
lower = 0.0
upper = 1.0
variance = 0.01

[[controls.variables]]
name = "u2"
initial = [0.5, 0.5]
lower = 0.0
upper = 1.0
variance = 0.01

[[parameterizations]]
name = "control_scaling"
options = { variable = "u1", lower = 150.0, upper = 400.0, output = "bhp1_knots" }

[[parameterizations]]
name = "control_schedule"
options = { variable = "bhp1_knots", knots = [0, 3], n_steps = 5, output = "injbhp1" }

[[parameterizations]]
name = "control_scaling"
options = { variable = "u2", lower = 150.0, upper = 400.0, output = "bhp2_knots" }

[[parameterizations]]
name = "control_schedule"
options = { variable = "bhp2_knots", knots = [0, 3], n_steps = 5, output = "injbhp2" }

[objective]
name = "npv"
options = { start = "1994-02-09", times = ["1995-01-01", "1996-01-01", "1997-01-01", "1998-01-01", "1999-01-01"], wop = 283.05, wwp = 37.74, wwi = 12.58, disc = 0.08, obj_scaling = -1.0e6 }

[simulator]
name = "opm_flow"
options = { deck_template_file = "deck.mako" }

[execution]
name = "process"
options = { workers = 4 }

[optimizer]
name = "enopt"
options = { ne = 10, maxiter = 5, alpha = 0.2 }
```

The deck then reads `${injbhp1[k]}` for the first well and `${injbhp2[k]}` for the second.

**Counting report steps.** `n_steps` must equal the number of report steps the deck reads, and nothing checks it. Count them in the deck: one per `TSTEP` entry or per date in `DATES` after the start. They should also match the objective's `times`. In the example, `times` lists 5 dates, so `n_steps = 5`. A block whose size does not fit its map is refused when the configuration is built, by mapping the initial controls.

## Step 5: check the cost before you run

Optimizations run as jobs of the job service, as history-matching runs do. Validate first to see the evaluation bound:

```python
from pathlib import Path

from apex.client import Client

client = Client()  # or Client.connect(url, token)
plan = client.validate_optimization(Path("enopt.toml")).optimization
print(plan)  # the plan: the evaluation bound and the resources it needs
job = client.submit_optimization(Path("enopt.toml"), max_iterations=3)
client.wait(job.id)
client.optimization_summary(Path(client.optimizations()[-1]["path"]))
client.resume(job.id)  # an optimization_resume job
```

- **Where it runs.** An unset `[optimization] output_dir` places it in `<root>/optimizations/<name>/<id>`, where the id hashes the configuration. Submitting the same configuration again finds the finished result, or continues an unfinished one. Web-UI users' optimizations go to their UI folder.
- **Admission and approval.** The simulator's resources (licences, cores) are reserved as for runs. Each optimizer bounds its evaluations (`max_evaluations`). With a simulated objective the bound is counted in simulator runs, and approval (above 20,000) and quotas use it. The journal makes the actual count smaller.
- **Stop.** `apex stop` stops a run at its next checkpoint, and the run can be resumed.
- **Tools.** The tools are `validate`, `submit_optimization` and `optimization_summary` (`docs/tools.md`).
- **Agents.** `apex agent monitor|report OPT_DIR` give the optimization's health and narrative (`docs/agents.md`). For multi-objective runs they judge the fronts.

## Step 6: compare optimizers in a study

A **study of optimizations** compares optimizers and settings over seeds. A cell is one run: one case, method and variant, for one replicate.

```toml
[study]
kind = "optimization"
name = "3well_optimizers"
seed = 7                              # derives the seeds of replicates after the first
replicates = 3                        # seeds: the case's own [optimization] seed, then derived from [study] seed
metrics = ["objective:final", "objective:best", "evaluations:all"]   # also iterations:all; fronts: hypervolume:final, front_size:final

[study.cases]
threewell = "enopt.toml"

[study.methods.enopt]                 # complete [optimizer] sections
name = "enopt"
options = { ne = 10, maxiter = 5, alpha = 0.2 }

[study.methods.smcopt]
name = "smcopt"
options = { ne = 10, maxiter = 5 }

[[study.variants]]
name = "base"

[[study.variants]]                    # patches anything but [optimizer] and the identity
name = "cheap_water"
patch = { objective = { options = { wwi = 6.0 } } }
```

Save it as `study.toml` and run:

```bash
uv run apex study run study.toml
uv run apex study status 3well_optimizers
uv run apex study compare 3well_optimizers
```

`apex study run` takes the study file; `status` and `compare` take the study's name, its `[study] name`.

The comparison has each method and variant's mean and spread over the replicates, and paired differences with Wilcoxon tests. The first replicate uses the case's own seed, so it reproduces a plain `apex optimize` of that case; the others are derived from `[study] seed`, so the study is reproducible.

`metrics` takes only what the comparison reports: `objective:final`, `objective:best`, `evaluations:all`, `iterations:all`, and for several objectives `hypervolume:final` and `front_size:final`. A bare name means `name:all`, so `evaluations` and `iterations` work alone, but `objective` and `hypervolume` need their label. Any other label, such as `objective:mean`, is refused when the study is validated, and by `apex study compare --metric`.

- `objective:final` is the representative's first objective, by its statistic over the realizations (the mean unless `statistic` says otherwise).
- `hypervolume:final` uses one reference point for the whole study: `[study] reference_point`, else the nadir of every finished cell's front plus 10 %. The cells must have the same objectives. It is computed only for fronts of up to four objectives; with more, its column stays empty.

## Drafted from a history-matching case

The web UI's assistant (tool `draft_optimization`), or `apex agent draft-optimization CASE --forecast-end DATE --wop ... --wwp ... --wwi ... --disc ... --producer-bhp LOW,HIGH` on the command line, drafts an optimization from a history-matching case on a deck simulator (`opm_flow`, `eclipse`). The case's history continues into a forecast with controls, and the objective is the forecast's NPV.

- **Wells.** They are read from the deck and its includes: producers from `WCONHIST`/`WCONPROD`, injectors from `WCONINJE`/`WCONINJH`. Wells last set `SHUT`, or that never produce or inject, are left out. (In the 5Spots example case, the well `PROD4` is left out for that reason.)
- **Controls.** Each producer gets a bottom-hole pressure `bhp_<WELL>` (bounds from the user; or `producer_control = "oil_rate"`: `orat_<WELL>`, 0 to twice the history's largest). Each injector gets a rate `rate_<WELL>` (0 to twice the history's largest). They are piecewise constant over `control_periods` equal periods up to `forecast_end`. `enopt` works in the unit cube of the bounds (`transform = true`).
- **The deck.** The case's template gets `WCONPROD`/`WCONINJE` and `DATES` per period before `END`, `${case_dir}` resolved, and any missing `FOPT`/`FGPT`/`FWPT`/`FWIT` in SUMMARY. It is written as `<name>_deck.mako` next to the draft; the case's files are not changed.
- **The model.** `members` columns of the posterior (`checkpoints/posterior.npz`) of a finished run of the case go to `<name>_model.npz` as `[model] realizations_file`: robust optimization over the history-matched ensemble. Without such a run the draft is refused. A web-UI user's draft takes only their own runs.
- **Economics.** `npv` counts from the history's end (`from_time`). `wop`, `wwp`, `wwi` and `disc` (and the producers' pressure bounds) must be given; the draft is refused, naming them, otherwise. `wgp = 0` and `obj_scaling = -1e6` are assumed when not given, and the draft's header lists every assumption.
- **Refused** when the schedule cannot be read (template expressions in well controls, `TSTEP`, unordered `DATES`); injectors other than water keep their history's last control.
- **Checked.** The draft is validated and costed (simulator runs) as `apex validate` would, and deleted if invalid. Its paths are relative to its folder.

## ropt's optimizers in APEX

With the `ert` extra, `[optimizer] name = "ropt"` runs ropt's backends on APEX's objectives, analytic or simulated.

```toml
[optimizer]
name = "ropt"
options = { method = "scipy/slsqp", maxiter = 50, number_of_perturbations = 8, perturbation_magnitude = 0.001, max_batches = 400 }
```

- **Steps.** ropt drives its iterations. Each evaluated point that improves the mean objective is an accepted step, logged and checkpointed.
- **Gradients.** They are ropt's ensemble estimates: `number_of_perturbations` points per realization, with `perturbation_magnitude` in the optimizer's coordinates (the unit cube with `transform = true`).
- **Approval.** The evaluation bound needs `max_batches`: the start, then at most `max_batches` batches of `n_realizations × (number_of_perturbations + 1)` evaluations.
- **Resume.** A resume replays ropt's run from the start with the same seed: the journal gives every simulated point back, the replay is checked against the checkpoint, and the run continues exactly as the uninterrupted one.
- **Constraints.** With `constraints = true`, the objective's constraints become ropt's nonlinear constraints. A point is feasible within `constraint_tolerance`. Feasible points rank first, so an infeasible start first reduces its violation. `epf` is refused.
- **Other options.** `max_functions`, `convergence_tolerance`, `seed`, and `options` passed to the backend.

## Pitfalls

- **A wrong `n_steps`** is not caught. Count the report steps in the deck and compare with `times`.
- **EPF with `line_search` or `trust_region`** does not work: their exact gradient lacks the penalty. Use an ensemble or population optimizer for constrained problems.
- **`ga`, `pso` and the multi-objective optimizers need finite bounds** on every control and many evaluations. Check the bound with validation before you submit.
- **Bounds and `variance` are in the controls' own units**, not in the units the deck sees.
- **Equality constraints with `ga`/`pso`:** a tolerance that is too tight stalls the run.
- **A statistic other than `mean`** is for the multi-objective optimizers only.
- **A failed simulation stops the run.** Fix the cause and resume: only the failed member is simulated again.
- **Changing the deck template, the model file or the objective** invalidates the journal's keys, so everything is simulated again.

## Differences from PET

These are listed in `docs/migration/pet_parity.md`. In short:

- **The baseline.** The ensemble gradient subtracts `f(x_k)`. PET subtracts the last evaluated point, which after a resample is a rejected trial.
- **Ensembles.** `genopt` draws one ensemble per iteration, not two.
- **Two PET bugs are not copied:**
  - the truncated Gaussian's `theta` gradient, which PET takes in the mean;
  - CG-Steihaug's boundary step at negative curvature.

## What to read next

- The guide's [simulators](guide/plugins/simulators.md), [executors](guide/plugins/executors.md) and [parameterizations](guide/plugins/parameterizations.md) pages.
- `docs/migration/pet_parity.md` and `docs/migration/pymoo.md`: how the ports were checked.
- `docs/tools.md` and `docs/agents.md`: the tools and the monitor/report agents for optimizations.

---
audience: public
items:
  - cli:apex optimize
  - cli:apex plugins
  - cli:apex run
  - cli:apex schema
  - config:optimization:objectives
  - plugin:gmac_constraint_metrics:quadratic
  - plugin:objectives:ackley
  - plugin:objectives:bnh
  - plugin:objectives:dtlz1
  - plugin:objectives:dtlz2
  - plugin:objectives:fon
  - plugin:objectives:kur
  - plugin:objectives:quadratic
  - plugin:objectives:rastrigin
  - plugin:objectives:rosenbrock
  - plugin:objectives:sch
  - plugin:objectives:srn
  - plugin:objectives:tnk
  - plugin:objectives:zdt1
  - plugin:objectives:zdt2
  - plugin:objectives:zdt3
  - plugin:optimizers:moead
  - plugin:optimizers:nsga2
  - plugin:optimizers:nsga3
---

# Objectives

## Goal

An objective is the function APEX minimises. The `objectives` registry holds 18 built-in plugins, mostly classic benchmark problems with known optima or known Pareto fronts. This page helps you choose one, set its only option, and pair it with an optimizer.

A few terms first:

- A **control** is a variable the optimizer changes.
- A **Pareto front** (or just *front*) is the set of best trade-offs in a multi-objective problem. Along it, no objective can be improved without making another worse.

The `quadratic` plugin of the separate `gmac_constraint_metrics` registry (`D(x) = 1/2 ||x||^2`, gradient `x`, Hessian `I`) is a measurement metric, not an objective. Do not confuse it with the objective `quadratic` below.

## Prerequisites

- A working APEX installation, run from the APEX folder with `uv run apex ...`.
- An optimization configuration file in TOML. `apex run` starts history-matching runs, so it is not the command for this page. Optimization runs use `apex optimize`.
- A rough idea of what you want to test: a single-objective optimizer (one minimum) or a multi-objective one (a front of trade-offs).

## Step 1: Pick a plugin

List every plugin with its options at any time:

```bash
uv run apex plugins
```

Add `--json` for the full descriptions as JSON.

### Single-objective functions

These have one minimum, so you can check whether an optimizer finds it. None has an option, so no objective option sets the number of controls.

| Plugin | What it is |
| --- | --- |
| `ackley` | Ackley's function. Minimum 0 at 0, a nearly flat outer region with many local minima. |
| `rastrigin` | `10 n + sum x_i^2 - 10 cos(2 pi x_i)`. Minimum 0 at 0. |
| `rosenbrock` | SciPy's Rosenbrock function `sum 100 (x_i+1 - x_i^2)^2 + (1 - x_i)^2`. Minimum 0 at 1. |
| `quadratic` | PET's quadratic `f(x) = (x - 1)^T A (x - 1)` with `A = I / 2`. Minimum 0 at `x = 1`. |

### Multi-objective problems without constraints

| Plugin | What it is |
| --- | --- |
| `sch` | `f1 = x^2`, `f2 = (x - 2)^2` on `[-1000, 1000]`. The Pareto-optimal points are `x` in [0, 2]. |
| `fon` | FON with three variables, on `[-4, 4]`. Non-convex front. |
| `kur` | KUR with three variables, on `[-5, 5]`. Disconnected front (pymoo's form). |
| `zdt1` | `f2 = g (1 - sqrt(f1 / g))`. Convex front `f2 = 1 - sqrt(f1)`. |
| `zdt2` | `f2 = g (1 - (f1 / g)^2)`. Non-convex front `f2 = 1 - f1^2`. |
| `zdt3` | `f2 = g (1 - sqrt(f1/g) - f1/g sin(10 pi f1))`. Five disconnected pieces. |
| `dtlz1` | A linear front, the simplex `sum f = 0.5`, surrounded by many local fronts. |
| `dtlz2` | The positive part of the unit sphere. |

### Multi-objective problems with constraints

| Plugin | What it is |
| --- | --- |
| `bnh` | `f1 = 4 x1^2 + 4 x2^2`, `f2 = (x1-5)^2 + (x2-5)^2`, with two constraints. |
| `srn` | Two quadratic objectives on `[-20, 20]^2`, with two constraints. |
| `tnk` | `f = (x1, x2)` on `[0, pi]^2`, with two constraints. Disconnected front. |

The constraints are part of the problem. They are not something you configure on the objective, but the optimizer must be able to deal with them (see Step 3).

## Step 2: Set options, if the plugin has any

Most plugins take no options. These have none: `ackley`, `bnh`, `fon`, `kur`, `quadratic`, `rastrigin`, `rosenbrock`, `sch`, `srn`, `tnk`, `zdt1`, `zdt2` and `zdt3`. Any option you add to them is rejected, because no extra keys are allowed.

Only two plugins have an option:

| Plugin | Option | Type | Default |
| --- | --- | --- | --- |
| `dtlz1` | `n_obj` | integer | `3` |
| `dtlz2` | `n_obj` | integer | `3` |

`n_obj` is the number of objectives, at least 2. The controls of your configuration must number at least `n_obj`: the first `n_obj - 1` place a point on the front, and the rest set its distance from it. So with `n_obj = 4` you need at least four controls. For `dtlz2`, the summary mentions `n_x` = 12 (the number of controls) as the default for three objectives.

## Step 3: Pair the objective with an optimizer

`apex optimize` minimises an objective with one of PET's optimizers (EnOpt, GenOpt, SmcOpt, ...). For the single-objective functions, use one of those.

For the multi-objective problems, use a multi-objective optimizer: `nsga2`, `nsga3` or `moead`. All three are experimental.

- `nsga2` and `nsga3` have `constraint_handling` with two values: `feasibility` (the default, Deb's rules) or `epf` (an external penalty, configured in `epf`, where `r` is required).
- `moead` refuses constrained objectives under its default `feasibility`. For `bnh`, `srn` or `tnk`, set `constraint_handling` to `epf`.

Several objectives can also be listed together in the `optimization:objectives` configuration entry, which is an array: each entry is a plugin with its options and an optional label. Objectives that are responses share one simulation per member.

## Worked example

Here is the objective part of an optimization configuration: the `dtlz2` problem with four objectives instead of the default three. Each entry of the `objectives` array names a plugin, and its `options` table holds the plugin's options.

```toml
[[objectives]]
name = "dtlz2"

[objectives.options]
n_obj = 4
```

This shows only the objective. The controls and the optimizer go in other sections of the same file, and you need at least four controls here. To see every section an optimization configuration accepts, print its JSON Schema:

```bash
uv run apex schema optimization
```

To keep the default of three objectives, leave the options out:

```toml
[[objectives]]
name = "dtlz2"
```

Once the complete file is ready, run it. Add `--max-iterations` to stop after a number of accepted iterations. You can continue later with `--resume` and the run's directory:

```bash
uv run apex optimize optimization.toml --max-iterations 5
```

```bash
uv run apex optimize --resume path/to/run_directory
```

Progress is shown on standard error by default; `--no-progress` turns it off.

## Pitfalls

- **Adding options to plugins that have none.** Only `dtlz1` and `dtlz2` accept `n_obj`. Every other objective takes no options.
- **Too few controls for `n_obj`.** `dtlz1` and `dtlz2` need at least `n_obj` controls. Raise the number of controls when you raise `n_obj`.
- **Expecting a single best answer from a multi-objective problem.** Problems like `zdt1` or `sch` have a front of trade-offs, not one minimum. Judge results by how well they cover the front.
- **Misreading the shape of the front.** `zdt1` is convex, `zdt2` is non-convex, `zdt3` has five disconnected pieces, and `kur` and `tnk` have disconnected fronts. An optimizer that does well on one may do badly on another.
- **Forgetting the constraints.** `bnh`, `srn` and `tnk` each have two constraints. Check `constraint_handling` on your optimizer, especially for `moead`.
- **Many local optima.** `ackley` and `rastrigin` have many local minima, and `dtlz1` has many local fronts. A run that stalls may be stuck in one of them rather than broken.
- **Changing `n_obj` and comparing against old results.** The problem changes with the number of objectives, so results are only comparable at the same `n_obj`.

## What to read next

- The other plugin pages in the guide, to choose and tune an optimizer (`nsga2`, `nsga3`, `moead`) for your objective.
- The configuration reference, for the other sections of your TOML file.

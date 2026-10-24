---
audience: public
items:
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
---

# Objectives

## Goal

An objective is the function APEX optimizes. The `objectives` registry holds 15 built-in plugins, mostly classic benchmark problems with known optima or known Pareto fronts. This page helps you choose one and shows how to set the only options any of them have.

## Prerequisites

- A working APEX installation and a configuration file in TOML.
- A rough idea of what you want to test: a single-objective optimizer (one minimum) or a multi-objective one (a front of trade-offs).

## Step 1: Pick a plugin

### Single-objective functions

These have one minimum, so you can check whether an optimizer finds it.

| Plugin | What it is |
| --- | --- |
| `ackley` | Ackley's function. Minimum 0 at 0, a nearly flat outer region with many local minima. |
| `rastrigin` | `10 n + sum x_i^2 - 10 cos(2 pi x_i)`. Minimum 0 at 0. |
| `rosenbrock` | SciPy's Rosenbrock function `sum 100 (x_i+1 - x_i^2)^2 + (1 - x_i)^2`. Minimum 0 at 1. |
| `quadratic` | PET's quadratic `f(x) = (x - 1)^T A (x - 1)` with `A = I / 2`. Minimum 0 at `x = 1`. |

### Multi-objective problems without constraints

| Plugin | What it is |
| --- | --- |
| `sch` | `f1 = x^2`, `f2 = (x - 2)^2` on `[-1000, 1000]`. The front is for `x` in [0, 2]. |
| `fon` | FON (n = 3) on `[-4, 4]`. Non-convex front. |
| `kur` | KUR (n = 3) on `[-5, 5]`. Disconnected front (pymoo's form). |
| `zdt1` | `f2 = g (1 - sqrt(f1 / g))`. Convex front `f2 = 1 - sqrt(f1)`. |
| `zdt2` | `f2 = g (1 - (f1 / g)^2)`. Non-convex front `f2 = 1 - f1^2`. |
| `zdt3` | `f2 = g (1 - sqrt(f1/g) - f1/g sin(10 pi f1))`. Five disconnected pieces. |
| `dtlz1` | A linear front, the simplex `sum f = 0.5`, with `11^k - 1` local fronts. |
| `dtlz2` | The positive part of the unit sphere. |

### Multi-objective problems with constraints

| Plugin | What it is |
| --- | --- |
| `bnh` | `f1 = 4 x1^2 + 4 x2^2`, `f2 = (x1-5)^2 + (x2-5)^2`, with two constraints. |
| `srn` | Two quadratic objectives on `[-20, 20]^2`, with two constraints. |
| `tnk` | `f = (x1, x2)` on `[0, pi]^2`, with two constraints. Disconnected front. |

## Step 2: Set options, if the plugin has any

Most plugins take no options at all. These have none: `ackley`, `bnh`, `fon`, `kur`, `quadratic`, `rastrigin`, `rosenbrock`, `sch`, `srn`, `tnk`, `zdt1`, `zdt2` and `zdt3`. Any option you add to them is rejected, because no extra keys are allowed.

Only two plugins have an option:

| Plugin | Option | Type | Default |
| --- | --- | --- | --- |
| `dtlz1` | `n_obj` | integer | `3` |
| `dtlz2` | `n_obj` | integer | `3` |

`n_obj` is the number of objectives. For `dtlz2`, the default of 3 objectives goes with `n_x` = 12.

## Worked example

The objective is the `[objective]` section of an optimization configuration: `name` is the plugin, and `[objective.options]` holds its options. Suppose you want the `dtlz2` problem with four objectives instead of the default three:

```toml
[objective]
name = "dtlz2"

[objective.options]
n_obj = 4
```

To keep the default, leave the option out: `[objective]` with `name = "dtlz2"` alone gives `n_obj = 3`. The same shape works for `dtlz1`:

```toml
[objective]
name = "dtlz1"

[objective.options]
n_obj = 3
```

A multi-objective problem such as `dtlz2` needs a multi-objective optimizer (`nsga2`, `nsga3` or `moead`). The examples catalogue has complete configurations, for example `toy/multi_objective/dtlz2`; `apex schema optimization` prints every section an optimization configuration accepts.

## Pitfalls

- **Adding options to plugins that have none.** Only `dtlz1` and `dtlz2` accept `n_obj`. Every other objective takes no options.
- **Expecting a single best answer from a multi-objective problem.** Problems like `zdt1` or `sch` have a front of trade-offs, not one minimum. Judge results by how well they cover the front.
- **Misreading the shape of the front.** `zdt1` is convex, `zdt2` is non-convex, `zdt3` has five disconnected pieces, and `kur` and `tnk` have disconnected fronts. An optimizer that does well on one may do badly on another.
- **Forgetting the constraints.** `bnh`, `srn` and `tnk` each have two constraints, so your optimizer has to handle them.
- **Many local optima.** `ackley` and `rastrigin` have many local minima, and `dtlz1` has `11^k - 1` local fronts. A run that stalls may be stuck in one of them rather than broken.
- **Changing `n_obj` and comparing against old results.** The problem changes with the number of objectives, so results are only comparable at the same `n_obj`.

## What to read next

- The other plugin pages in the guide, to pair an objective with an optimizer.
- The configuration reference, for the other sections of your TOML file.

---
audience: public
items:
  - cli:apex plugins
  - cli:apex schema
  - config:history_match:method
  - plugin:gmac_constraint_metrics:l1sq
  - plugin:gmac_constraint_metrics:log_barrier
  - plugin:gmac_constraint_metrics:log_channel
  - plugin:gmac_constraint_metrics:quadratic
  - plugin:gmac_constraints:box
  - plugin:gmac_constraints:histogram
  - plugin:gmac_norms:l1sq
  - plugin:methods:gies
  - plugin:objectives:quadratic
  - plugin:update_strategies:gmac
---

# Gmac constraint metrics

## Goal

The `gmac_constraint_metrics` registry holds four plugins: `quadratic`, `l1sq`, `log_barrier` and `log_channel`. Each defines a function `D(x)` that scores how far a constraint residual `x` is from being satisfied. After this lesson you will know what each plugin computes, which options it takes, and which one fits which kind of constraint.

The metrics are C-GIES's: X. Luo and W. C. Cruz (2022). Data assimilation with soft constraints (DASC) through a generalized iterative ensemble smoother. *Computational Geosciences* 26(3), 571–594. [doi:10.1007/s10596-022-10137-7](https://doi.org/10.1007/s10596-022-10137-7)

| Plugin | `D(x)` | Meant for |
|---|---|---|
| `quadratic` | `1/2 ||x||^2` | a perfect measurement (see step 1) |
| `l1sq` | `||x||_1^2` | a squared L1 norm |
| `log_barrier` | `-sum log(x + a)` | inequalities, with `x = -h >= 0` |
| `log_channel` | `sum log(|x| + b)` | equalities |

A similarly named `l1sq` also exists in the `gmac_norms` registry, and `objectives` has its own `quadratic`. They are different plugins. This page covers only `gmac_constraint_metrics`.

## Prerequisites

- You know what a constraint residual is. A constraint is written as a function `h` of the state. The residual `x` is the vector the metric scores:
  - **Equality** (`h = 0`): `x = h`. For the `histogram` constraint, `h` is `counts(x) - counts_ref`, so `x` is the difference between your histogram and the reference one. `x = 0` means the constraint holds.
  - **Inequality** (`h <= 0`): `x = -h`, and `x >= 0` means the constraint holds. For the `box` constraint, `h = [lower - x; x - upper]`. With `lower = 100`, `upper = 15000` and a value of `500`, `h = (-400, -14500)` and the residual is `x = (400, 14500)`.
- You can run `apex` from the APEX folder with `uv run`. To list the plugins and their options:

```bash
uv run apex plugins
```

For full descriptions as JSON, add `--json`:

```bash
uv run apex plugins --json
```

## Steps

### 1. Pick the plugin for your constraint

- **Inequality constraints:** use `log_barrier`. It is defined on `x = -h >= 0`.
- **Equality constraints:** use `log_channel`.
- **A perfect measurement:** use `quadratic`. This means a quantity you treat as exactly known, so that the plain squared distance `1/2 ||x||^2` is the score. Its gradient is `x` and its Hessian is the identity. Choose `quadratic` instead of `log_channel` only when that is how you want to treat the quantity. For an ordinary equality such as a histogram match, use `log_channel`.
- **Squared L1 norm:** use `l1sq`.

You can also skip this choice. Without a `metric` on a constraint, an inequality gets `log_barrier` and an equality gets `log_channel`, both with their default option values. Set `metric` only to pick a different plugin or to change an option.

### 2. Know what each plugin gives you

Below, every operation acts entry by entry on `x`.

- `quadratic`: gradient `x`, Hessian `I`. No options.
- `l1sq`: gradient `2 ||x||_1 sgn(x)`, Hessian `2 sgn(x) sgn(x)^T`. No options.
- `log_barrier`: `D(x) = -sum log(x + a)`, with gradient `-1 / (x + a)`. One option, `a`.
- `log_channel`: `D(x) = sum log(|x| + b)`, with gradient `sgn(x) / (|x| + b)`. Its curvature (the Hessian, or its trace) is chosen by `curvature`. Options `b`, `epsilon` and `curvature`.

### 3. Set the options (only where there are any)

| Plugin | Option | Type | Default | Allowed values |
|---|---|---|---|---|
| `log_barrier` | `a` | number | `0.1` | any number |
| `log_channel` | `b` | number | `0.1` | any number |
| `log_channel` | `epsilon` | number | `0.001` | any number |
| `log_channel` | `curvature` | string | `absolute_hessian` | `absolute_hessian`, `absolute_trace`, `paper` |

What they control:

- `a` is the offset in `log_barrier`. The paper uses `0.1`. It sets how large the term is at the boundary `x = 0`.
- `b` and `epsilon` are offsets of `log_channel`. The paper's scripts use `0.1` and `0.001`. `epsilon` is used by `curvature = "paper"`.
- `curvature` selects the metric:
  - `absolute_hessian` (default): the Hessian's absolute value, with a positive weight.
  - `absolute_trace`: the negative Hessian with a positive weight. It may become indefinite.
  - `paper`: the negative Hessian with the paper's negative trace weight.

No plugin accepts options other than those listed. `quadratic` and `l1sq` accept none.

### 4. Put the metric in the configuration

A metric is chosen per constraint, as the `metric` of an entry of the `gmac` strategy's `constraints`. The surrounding keys are not specific to the metric:

- `function` is the constraint itself (here `histogram` or `box`).
- `weight` is the weight of the constraint's term. It is either `{ rule = "trace", w = ... }` or `{ rule = "fixed", value = ... }`.
- The strategy is `gmac`, run by the `gies` method.

See [GMAC and C-GIES](../../gmac.md) for them. Only the `metric` line is what this page is about.

An equality with `log_channel`, every option written out at its default:

```toml
[method]
name = "gies"

[method.strategy]
name = "gmac"

[[method.strategy.options.constraints]]
function = { name = "histogram", options = { variable = "permx", range = [100.0, 15000.0], bins = 50, reference_file = "truth.npz" } }
metric = { name = "log_channel", options = { b = 0.1, epsilon = 0.001, curvature = "absolute_hessian" } }
weight = { rule = "trace", w = 0.5 }
```

An inequality with `log_barrier`:

```toml
[method]
name = "gies"

[method.strategy]
name = "gmac"

[[method.strategy.options.constraints]]
function = { name = "box", options = { bounds = { permx = [100.0, 15000.0] } } }
metric = { name = "log_barrier", options = { a = 0.1 } }
weight = { rule = "trace", w = 0.5 }
```

To see what the `[method]` section accepts, print its schema:

```bash
uv run apex schema history_match
```

## Worked example

**`l1sq`.** Take `x = (1, -2, 0)`.

1. `||x||_1 = 1 + 2 + 0 = 3`, so `D(x) = 3^2 = 9`.
2. `sgn(x) = (1, -1, 0)`.
3. The gradient is `2 * 3 * sgn(x) = (6, -6, 0)`.
4. The Hessian is `2 sgn(x) sgn(x)^T`:

```
[[ 2, -2, 0],
 [-2,  2, 0],
 [ 0,  0, 0]]
```

**`log_barrier`.** With the default `a = 0.1`, one entry contributes `-log(x + 0.1)`:

- `x = 0.9`: `-log(1) = 0`.
- `x = 0` (the boundary): `-log(0.1) ≈ 2.303`.
- With `a = 0.01`, the same boundary gives `-log(0.01) ≈ 4.605`.

So the term grows as `x` approaches `0`, and a smaller `a` makes the wall at the boundary higher.

**`log_channel`.** With the defaults `b = 0.1` and `x = (0.9, -0.1, 0)`, the entries contribute `log(|x| + 0.1)`:

- `log(1.0) = 0`
- `log(0.2) ≈ -1.609`
- `log(0.1) ≈ -2.303`

The sum is `D(x) ≈ -3.912`. The score falls as entries approach `0`, which is where an equality is satisfied.

## Pitfalls

- **Using the wrong sign for inequalities.** `log_barrier` is defined with `x = -h >= 0`. Pass the negated constraint value, not `h` itself.
- **Setting options on `quadratic` or `l1sq`.** These plugins have no options, and unknown options are not accepted.
- **Mixing up `a` and `b`.** `a` belongs to `log_barrier`, which shifts `x`. `b` belongs to `log_channel`, which shifts `|x|`.
- **Misspelling `curvature`.** Only `absolute_hessian`, `absolute_trace` and `paper` are valid. Note that `absolute_trace` may become indefinite.
- **Expecting `l1sq` to be smooth at zero.** Its gradient and Hessian use `sgn(x)`, which is `0` where an entry is `0`, as in the worked example.
- **Writing `metric` when you do not need to.** The defaults already give `log_barrier` for inequalities and `log_channel` for equalities.

## What to read next

- [GMAC and C-GIES](../../gmac.md): how the `gmac` strategy uses `constraints`, `function` and `weight`. Then try a small case with one constraint and a chosen `metric`.
- `uv run apex plugins` shows every plugin with its options, and `uv run apex schema history_match` shows what the TOML file accepts.

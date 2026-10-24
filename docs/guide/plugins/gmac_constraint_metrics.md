---
audience: public
items:
  - plugin:gmac_constraint_metrics:l1sq
  - plugin:gmac_constraint_metrics:log_barrier
  - plugin:gmac_constraint_metrics:log_channel
  - plugin:gmac_constraint_metrics:quadratic
---

# Gmac constraint metrics

## Goal

The `gmac_constraint_metrics` registry holds four plugins. Each one defines a function `D(x)` that scores how far a constraint residual `x` is from being satisfied. After this lesson you will know what each plugin computes, which options it takes, and which one fits which kind of constraint. The metrics are C-GIES's: X. Luo and W. C. Cruz (2022). Data assimilation with soft constraints (DASC) through a generalized iterative ensemble smoother. *Computational Geosciences* 26(3), 571–594. [doi:10.1007/s10596-022-10137-7](https://doi.org/10.1007/s10596-022-10137-7)

| Plugin | `D(x)` | Meant for |
|---|---|---|
| `quadratic` | `1/2 ||x||^2` | a perfect measurement |
| `l1sq` | `||x||_1^2` | a squared L1 norm |
| `log_barrier` | `-sum log(x + a)` | inequalities, with `x = -h >= 0` |
| `log_channel` | `sum log(|x| + b)` | equalities |

## Prerequisites

- You have a constraint residual `x` (a vector) that you want to score.
- You know whether the constraint is an equality or an inequality. For an inequality, `h` is the constraint value and `x = -h` must be `>= 0`.

## Steps

### 1. Pick the plugin for your constraint

- **Equality constraints:** use `log_channel`. For a perfectly measured quantity, `quadratic` is the plugin described for that case.
- **Inequality constraints:** use `log_barrier`. It is defined on `x = -h >= 0`.
- **Squared L1 norm:** use `l1sq`.

### 2. Know what each plugin gives you

- `quadratic`: gradient `x`, Hessian `I`. It has no options.
- `l1sq`: gradient `2 ||x||_1 sgn(x)`, Hessian `2 sgn(x) sgn(x)^T`. It has no options.
- `log_barrier`: one option, `a`.
- `log_channel`: options `b`, `epsilon` and `curvature`.

### 3. Set the options (only where there are any)

| Plugin | Option | Type | Default | Allowed values |
|---|---|---|---|---|
| `log_barrier` | `a` | number | `0.1` | any number |
| `log_channel` | `b` | number | `0.1` | any number |
| `log_channel` | `epsilon` | number | `0.001` | any number |
| `log_channel` | `curvature` | string | `absolute_hessian` | `absolute_hessian`, `absolute_trace`, `paper` |

No plugin accepts options other than those listed. `quadratic` and `l1sq` accept none.

A metric is chosen per constraint, as the `metric` of an entry of the `gmac` strategy's `constraints` (see [GMAC and C-GIES](../../gmac.md)). An equality with `log_channel`, every option written out at its default:

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
[[method.strategy.options.constraints]]
function = { name = "box", options = { bounds = { permx = [100.0, 15000.0] } } }
metric = { name = "log_barrier", options = { a = 0.1 } }
weight = { rule = "trace", w = 0.5 }
```

Without `metric`, an inequality gets `log_barrier` and an equality `log_channel`.

## Worked example

Take `x = (1, -2, 0)` and the `l1sq` plugin.

1. `||x||_1 = 1 + 2 + 0 = 3`, so `D(x) = 3^2 = 9`.
2. `sgn(x) = (1, -1, 0)`.
3. The gradient is `2 * 3 * sgn(x) = (6, -6, 0)`.
4. The Hessian is `2 sgn(x) sgn(x)^T`:

```
[[ 2, -2, 0],
 [-2,  2, 0],
 [ 0,  0, 0]]
```

Now a `log_barrier` check with the default `a = 0.1`. If a single entry has `x = 0.9`, then `x + a = 1` and its contribution is `-log(1) = 0`. As `x` gets closer to `0`, `x + a` approaches `a` and the term grows. The option `a` therefore sets how large the term is at the boundary `x = 0`.

## Pitfalls

- **Using the wrong sign for inequalities.** `log_barrier` is defined with `x = -h >= 0`. Pass the negated constraint value, not `h` itself.
- **Setting options on `quadratic` or `l1sq`.** These plugins have no options, and unknown options are not accepted.
- **Mixing up `a` and `b`.** `a` belongs to `log_barrier`, which shifts `x`. `b` belongs to `log_channel`, which shifts `|x|`.
- **Misspelling `curvature`.** Only `absolute_hessian`, `absolute_trace` and `paper` are valid.
- **Expecting `l1sq` to be smooth at zero.** Its gradient and Hessian use `sgn(x)`, which is `0` where an entry is `0`, as in the worked example.

## What to read next

- The other plugin pages in the guide's plugins section, to see where these metrics are used.
- The reference listing for `gmac_constraint_metrics`, for the full option schema of each plugin.

---
audience: public
items:
  - cli:apex optimize
  - cli:apex run
  - cli:apex stop
  - plugin:optimizers:enopt
  - plugin:optimizers:genopt
  - plugin:step_rules:adam
  - plugin:step_rules:adamax
  - plugin:step_rules:gd
  - plugin:step_rules:steihaug
---

# Step rules

A step rule decides how an optimizer turns the gradient information at the current point into the next move. APEX ships four: `gd`, `adam`, `adamax` and `steihaug`. This page shows what each one does, how the optimizer passes it its settings, and how to choose one in a configuration.

## Goal

By the end of this page you will be able to:

- choose between `gd`, `adam`, `adamax` and `steihaug`;
- select one through the `enopt` or `genopt` optimizer and set its step size, momentum and trust radius;
- avoid the common mistakes when switching rules.

## Prerequisites

- An optimization configuration whose optimizer is `enopt` or `genopt`. These are the optimizers that take a step rule.
- A basic idea of what a step size and momentum are in gradient-based optimization.

## The four step rules

| Plugin | What it does | Typical reason to pick it |
| --- | --- | --- |
| `gd` | Gradient descent with momentum: `v = beta v_old - alpha g`, `x_new = x + v`. | The simplest rule, and the default. With no momentum it is plain gradient descent. |
| `adam` | Adam: bias-corrected first and second moments of the gradient. | An adaptive step, when one global step size suits the controls badly. |
| `adamax` | AdaMax: Adam with the infinity norm for the second moment (a running maximum of `abs(g)`). | Adam's behaviour, but with each control moving by at most about the step size per step. |
| `steihaug` | CG-Steihaug: the truncated conjugate-gradient trust-region step. | A trust-region step that uses an ensemble Hessian. Only for `enopt`, and it costs the Hessian estimate. |

If you have no reason to choose otherwise, start with `gd`, and switch to an adaptive rule or `steihaug` only if progress is poor.

## How the optimizer passes values to the rule

A step rule is not a section of its own. The optimizer chooses it with its option `optimizer`, and gives it two values:

- the rule's `step_size` is the optimizer's `alpha` (or `step_size`, which overrides `alpha`);
- the rule's `momentum` is the optimizer's `beta`.

So the defaults you get are the optimizer's, not the rule's own:

| Optimizer option | Default (`enopt`) | Becomes |
| --- | --- | --- |
| `alpha` | `0.1` | the rule's `step_size` |
| `beta` | `0.0` | the rule's `momentum` |
| `trust_radius` | `3.0` | `steihaug`'s starting radius |

`genopt` has `alpha` (`0.1`) and `beta` (`0.0`) too, but no `trust_radius`.

The rule's own defaults, shown in the tables below, only describe the rule on its own. They are overridden whenever an optimizer supplies `alpha` and `beta`.

### Which optimizer takes which rule

- `enopt` takes `"GD"`, `"Adam"`, `"AdaMax"` and `"Steihaug"`. Its default is `"GD"`.
- `genopt` takes `"GD"` and `"Adam"` only. Its default is `"GD"`.

### What `normalize` means

`normalize` (default `true`) scales the gradient, and the Hessian when there is one, by their infinity norms (the largest absolute entry). You normally leave it alone. Choosing `"AdaMax"` turns it off for you, because AdaMax already works with the infinity norm.

### Steihaug and the ensemble Hessian

The optimizer keeps an ensemble of candidate control vectors drawn from a covariance, which says how widely, and in which directions, the candidates are spread around the current point. The gradient is estimated from this ensemble. The Hessian is the matrix of second derivatives, which describes the curvature of the objective. The ensemble Hessian is an estimate of that curvature computed from the same ensemble.

`ensemble_hessian` (default `false`) makes `enopt` compute this estimate. The estimate is used to adapt the covariance, and it is what the Steihaug rule works with. This is why `"Steihaug"` needs `ensemble_hessian = true`.

For `"Steihaug"`, the `step_size` the rule receives is its initial trust radius (the largest move it allows at first). `enopt` always takes that value from `trust_radius` (default `3.0`), never from `alpha`. The rule's own default `step_size` of `1.0` is therefore not used under `enopt`.

About `beta`: the Steihaug rule does not use `momentum`, so `beta` has no effect on the Steihaug step. With `ensemble_hessian = true`, `beta` still acts as the momentum of the covariance update. If you use `"Steihaug"` you can leave `beta` at its default.

### Options of each rule

Columns: the option, the rule's own default, and what sets it from a configuration.

**`gd`**

| Option | Rule default | Set by |
| --- | --- | --- |
| `step_size` | `0.1` | optimizer's `alpha` |
| `momentum` | `0.0` | optimizer's `beta` |

In the `gd` formula, `step_size` is `alpha` and `momentum` is `beta`. Backtracking (shrinking a step that does not improve the objective) shrinks the momentum together with the step size.

**`adam`**

| Option | Rule default | Set by |
| --- | --- | --- |
| `step_size` | `0.1` | optimizer's `alpha` |
| `momentum` | `0.9` | optimizer's `beta` |
| `beta2` | `0.999` | fixed, cannot be set |

**`adamax`**

| Option | Rule default | Set by |
| --- | --- | --- |
| `step_size` | `0.1` | optimizer's `alpha` |
| `momentum` | `0.9` | optimizer's `beta` |
| `beta2` | `0.999` | fixed, cannot be set |

`adamax` takes the same options as `adam`. Each control moves by at most about `step_size` per step, and backtracking shrinks it for a trial.

**`steihaug`**

| Option | Rule default | Set by |
| --- | --- | --- |
| `step_size` | `1.0` | `enopt`'s `trust_radius` (default `3.0`) |
| `momentum` | `0.0` | unused |
| `maxiter` | `1000000.0` | fixed, cannot be set |
| `epsilon` | `1e-08` | fixed, cannot be set |
| `delta_max` | `100000.0` | fixed, cannot be set |

Steihaug's `maxiter` bounds its inner conjugate-gradient iterations, and you cannot set it. It is a different thing from the optimizer's own `maxiter` (default `100`), which you set in `[optimizer.options]` like any other option. The optimizer's `maxiter` is the number of iterations in each pass of the external penalty (EPF) loop, which the optimizer uses to handle constraints. Without a constrained objective there is just one pass, so it is simply the iteration limit.

## Steps

1. Pick the rule (see the table above).
2. In `[optimizer.options]`, set `optimizer` to `"GD"`, `"Adam"`, `"AdaMax"` or `"Steihaug"`.
3. Set `alpha` for the step size and `beta` for the momentum. Anything you leave out keeps the optimizer's default (`0.1` and `0.0`).
4. For `"Steihaug"` (with `enopt` only), also set `ensemble_hessian = true`, and `trust_radius` if you want a starting radius other than `3.0`.
5. Save the file and run it, giving the configuration file as the argument:

```bash
uv run apex optimize config.toml
```

To try the rule on a few iterations first, stop early with `--max-iterations`. It stops after that many accepted iterations:

```bash
uv run apex optimize config.toml --max-iterations 3
```

The run keeps its results in a run directory. To continue it later, pass that directory to `--resume`:

```bash
uv run apex optimize --resume path/to/run_dir
```

If you run `apex optimize` again with the same configuration, it stops with an error saying the run directory already holds an optimization run, and that you should resume it or remove it. Either resume it with `--resume` as above, or delete that directory if you want to start over, for example after changing the rule.

To check that the rule is the one you meant, re-read the `optimizer` line of the configuration you passed. Then run two short runs of three iterations, one with each rule (remove or resume the first run directory in between, as described above), and compare the progress lines and the objective values they report. Two different rules normally give different objective values after the same number of iterations. If the numbers are identical, the rule you set may not be the one in use.

## Worked example

Suppose you start from plain gradient descent and want to add momentum and a smaller step. With `enopt`'s defaults, `alpha = 0.1` and `beta = 0.0`, so you override both:

```toml
[optimizer]
name = "enopt"

[optimizer.options]
optimizer = "GD"
alpha = 0.05
beta = 0.9
```

The update is now `v = 0.9 v_old - 0.05 g`, followed by `x_new = x + v`.

To try an adaptive rule instead, set `optimizer` to `"Adam"` and lower the step. Adam's momentum is the optimizer's `beta`, which defaults to `0.0`, so set it to Adam's usual `0.9`:

```toml
[optimizer]
name = "enopt"

[optimizer.options]
optimizer = "Adam"
alpha = 0.01
beta = 0.9
```

`"AdaMax"` (with `enopt`) takes the same values. For the trust-region step, with `enopt`, enable the ensemble Hessian and start from a radius of 1.0 instead of the default 3.0. The optimizer's own `maxiter` is set in the same place:

```toml
[optimizer]
name = "enopt"

[optimizer.options]
optimizer = "Steihaug"
ensemble_hessian = true
trust_radius = 1.0
maxiter = 50
```

These blocks show the optimizer part of a configuration. See [Optimization](../../optimization.md) for the other sections it needs.

## Pitfalls

- **The rule's own defaults do not apply to `step_size` and `momentum`.** They come from the optimizer's `alpha` and `beta`. Since `beta` defaults to `0.0`, `"Adam"` without `beta` runs with no momentum. Set `beta` when you switch rules.
- **`beta2`, and Steihaug's own `maxiter`, `epsilon` and `delta_max`, cannot be set.** The optimizers have no options for them, so they keep the values in the tables. The optimizer's `maxiter` is a different option and can be set.
- **`genopt` takes `"GD"` or `"Adam"` only.** `"AdaMax"` and `"Steihaug"` belong to `enopt`, and `"Steihaug"` also needs `ensemble_hessian = true`.
- **For `"Steihaug"`, `alpha` is not the starting step.** The starting radius is `trust_radius`.
- **A step size that works for one rule may not suit another.** Re-tune `alpha` when you switch rules.
- **Do not mix up `"Adam"` and `"AdaMax"`.** They take the same values, so a copied configuration runs without error but does a different computation.
- **Running the same configuration twice fails.** The second run stops because the run directory already holds a run. Resume it with `--resume` or remove the directory.

## What to read next

- [Optimization](../../optimization.md), for the rest of an optimization configuration.
- The other plugin pages in this guide, to see how step rules fit with the rest of your configuration.

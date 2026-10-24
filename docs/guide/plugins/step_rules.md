---
audience: public
items:
  - plugin:step_rules:adam
  - plugin:step_rules:adamax
  - plugin:step_rules:gd
  - plugin:step_rules:steihaug
---

# Step rules

A step rule decides how an optimizer turns the gradient information at the current point into the next move. APEX ships four: `gd`, `adam`, `adamax` and `steihaug`. This page shows what each one does, which options it accepts, and how to configure it.

## Goal

By the end of this page you will be able to:

- choose between `gd`, `adam`, `adamax` and `steihaug`;
- choose them through the `enopt` or `genopt` optimizer, with its `alpha`, `beta` and `trust_radius`;
- avoid the common mistakes when changing them.

## Prerequisites

- An optimization configuration with `enopt` or `genopt` as its `[optimizer]`: they take the step rule.
- A basic idea of what a step size and momentum are in gradient-based optimization.

## The four step rules

| Plugin | What it does |
| --- | --- |
| `gd` | Gradient descent with momentum: `v = beta v_old - alpha g`, `x_new = x + v`. |
| `adam` | Adam: bias-corrected first and second moments of the gradient. |
| `adamax` | AdaMax: Adam with the infinity norm for the second moment. |
| `steihaug` | CG-Steihaug: the truncated conjugate-gradient trust-region step. |

### Options and defaults

A step rule is not a section of its own: the optimizer chooses it with its option `optimizer` (any case), and passes it two values. Its `step_size` is the optimizer's `alpha` and its `momentum` the optimizer's `beta`.

- `enopt` takes all four: `"GD"`, `"Adam"`, `"AdaMax"` and `"Steihaug"`. For `"Steihaug"` it needs `ensemble_hessian = true`, and the step starts from its `trust_radius` instead of `alpha`. When `enopt` updates its covariance (with an ensemble Hessian), `beta` is also the momentum of that covariance step.
- `genopt` takes `"GD"` and `"Adam"` only. It has no `trust_radius`, so `"AdaMax"` and `"Steihaug"` are `enopt`'s alone.

The rules' other options below (`beta2`, and Steihaug's own `maxiter`, `epsilon` and `delta_max`) keep their defaults. Steihaug's `maxiter` bounds its inner conjugate-gradient iterations; it is not the optimizer's `maxiter`, which you set as usual.

**`gd`**

| Option | Default |
| --- | --- |
| `step_size` | `0.1` |
| `momentum` | `0.0` |

In the `gd` formula, `step_size` plays the role of alpha and `momentum` the role of beta. With the default `momentum` of `0.0` you get plain gradient descent.

**`adam`**

| Option | Default |
| --- | --- |
| `step_size` | `0.1` |
| `momentum` | `0.9` |
| `beta2` | `0.999` |

**`adamax`**

| Option | Default |
| --- | --- |
| `step_size` | `0.1` |
| `momentum` | `0.9` |
| `beta2` | `0.999` |

`adamax` takes the same options as `adam`. It differs only in using the infinity norm for the second moment.

**`steihaug`**

| Option | Default |
| --- | --- |
| `step_size` | `1.0` |
| `momentum` | `0.0` |
| `maxiter` | `1000000.0` |
| `epsilon` | `1e-08` |
| `delta_max` | `100000.0` |

## Steps

1. Pick the rule. `gd` is the simplest, and `adam` and `adamax` are the adaptive choices. Use `steihaug` when you want a trust-region step.
2. In `[optimizer.options]`, set `optimizer` to the rule.
3. Set `alpha` for the step size and `beta` for the momentum. For `"Steihaug"` (with `enopt` only), also set `ensemble_hessian = true`, and `trust_radius` for the starting radius. Anything you leave out keeps the optimizer's default.
4. Save the file and run it with `apex optimize`.

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

To try an adaptive rule instead, name `Adam` and lower the step. Adam's momentum is then the optimizer's `beta`, so set it to Adam's usual 0.9:

```toml
[optimizer]
name = "enopt"

[optimizer.options]
optimizer = "Adam"
alpha = 0.01
beta = 0.9
```

`"AdaMax"` (with `enopt`) takes the same values. For the trust-region step, with `enopt`, start from a radius of 1.0:

```toml
[optimizer]
name = "enopt"

[optimizer.options]
optimizer = "Steihaug"
ensemble_hessian = true
trust_radius = 1.0
```

## Pitfalls

- **The rule's own defaults do not apply to `step_size` and `momentum`.** They come from the optimizer's `alpha` and `beta` (`beta` defaults to 0.0, so `"Adam"` without `beta` runs with no momentum). Set `beta` when you switch rules.
- **`beta2`, and Steihaug's own `maxiter`, `epsilon` and `delta_max`, cannot be set** from a configuration: they keep the defaults in the tables above. The optimizer's `maxiter` is a different option and can be set.
- **`genopt` takes `"GD"` or `"Adam"` only**, and `"Steihaug"` needs `enopt` with `ensemble_hessian = true`.
- **A step size that works for one rule may not suit another.** Re-tune `step_size` when you switch rules.
- **Do not mix up `Adam` and `AdaMax`.** They take the same values, so a copied configuration runs without error but does a different computation.

## What to read next

- The other plugin pages in this guide, to see how step rules fit with the rest of your configuration.
- [Optimization](../../optimization.md), for the rest of an optimization configuration.

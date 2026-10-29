---
audience: public
items:
  - cli:apex plugins
  - cli:apex resume
  - cli:apex run
  - cli:apex stop
  - config:history_match:method
  - plugin:damping_schemes:rlmmac
  - plugin:gmac_constraint_metrics:log_barrier
  - plugin:gmac_constraint_metrics:log_channel
  - plugin:gmac_constraints:box
  - plugin:gmac_constraints:histogram
  - plugin:gmac_constraints:linear
  - plugin:gmac_constraints:tv
  - plugin:gmac_transforms:tv
  - plugin:methods:chop
  - plugin:methods:es
  - plugin:methods:esmda
  - plugin:methods:gies
  - plugin:methods:gn_enrml
  - plugin:methods:lm_enrml
  - plugin:methods:multilevel_esmda
  - plugin:simulators:linear
  - plugin:update_strategies:gmac
  - plugin:update_strategies:hybrid
  - plugin:update_strategies:rlmmac
---

# Gmac constraints

The `gmac_constraints` registry holds four constraint plugins: `box`, `histogram`, `linear` and `tv`. Each one states a condition that the variables should satisfy. The `gmac` update strategy uses them as entries of its `constraints` option (C-GIES's soft constraints).

## Goal

"Soft" means the condition is not imposed as a hard restriction. Each constraint enters the `gmac` update as one more term. An entry has three parts:

- a `function`: one of the plugins on this page, as `{ name, options }`;
- optionally a `metric`: `{ name, options }` from the `gmac_constraint_metrics` registry;
- a `weight`: either `{ rule = "trace", w = ... }` or `{ rule = "fixed", value = ... }`.

If you leave out `metric`, the default depends on the kind of condition:

- **Inequalities** use `log_barrier`, the barrier `D(x) = -sum log(x + a)` with `x = -h >= 0` (option `a`, default `0.1`). The inequalities are `box` and `linear` with `kind = "inequality"`.
- **Equalities** use `log_channel`, the channel `D(x) = sum log(|x| + b)` (options `b`, `epsilon`, `curvature`). The equalities are `histogram`, `tv` and `linear` with `kind = "equality"`.

This page does not explain how to choose a weight. The `trace` and `fixed` rules, and what `w` and `value` do, are described in [GMAC and C-GIES](../../gmac.md). Read that page before you settle on a number.

By the end of this lesson you will be able to:

- pick the right constraint plugin for a condition you want to impose;
- write the options for `box`, `histogram`, `linear` and `tv` in TOML;
- check option names with `apex plugins` and start a run with `apex run`;
- avoid the common mistakes.

## Prerequisites

- You know the names of the variables you want to constrain: state variables, or properties that have a parameterization. These are the names your own problem defines in its run configuration.
- You have a run configuration file in TOML, which is the file you pass to `apex run`. The blocks on this page go into it.
- In that file, `[method] name` is `gies` or `lm_enrml`, and `[method.strategy] name` is `gmac`.
- For `linear`, an `.npz` file holding the linear system (its layout is given below). For `histogram`, reference counts, either as a list of numbers or in an `.npy` or `.npz` file.

## How constraints are written

Each plugin defines a constraint function of the variables `x`. For `box`, the condition applies to each variable separately. For `linear`, `x` is the named `variables` stacked in the order you list them. `histogram` looks at one variable, and `tv` at gridded variables.

A *gridded variable* is a variable that is defined on a grid. When you leave `variables` out of `tv`, it selects every variable that has a grid.

- `box` gives `h = [lower - x; x - upper]`. The semicolon means the two parts are stacked into one vector. The condition is `lower <= x <= upper`.
- `histogram` states `counts(x) - counts_ref = 0`. Here `counts(x)` is the histogram of the variable, with `bins` bins over `range`, and `counts_ref` is the target histogram.
- `linear` states `A x - b`, where `A` is a matrix and `b` a vector. It is an equality (`= 0`) or an inequality (`<= 0`).
- `tv` gives `f = TV(x)`, and the condition is `f = 0`: the total variation of the gridded variables vanishes.

| Plugin | Condition it expresses | Kind |
| --- | --- | --- |
| `box` | `lower <= x <= upper` per variable | inequality |
| `histogram` | the histogram of a variable equals a reference one | equality |
| `linear` | a linear system `A x - b` | equality or inequality, set by `kind` |
| `tv` | the total variation of gridded variables vanishes | equality |

All four plugins reject unknown options, so a misspelled option name is an error and is not silently ignored.

## Steps

1. Decide which condition you need from the table above.
2. Write the options for that plugin, using the tables below. Required options must be present.
3. Run `apex plugins` to list every plugin with its options. Add `--json` for the full descriptions as JSON. Use it to check exact option names.
4. Make sure every variable name is a state variable of your problem, or a property that has a parameterization.
5. Put the entry in `constraints` of the `gmac` strategy, with a `weight`, and start the run with `apex run`.

```bash
uv run apex plugins
uv run apex plugins --json
```

### `box`

| Option | Type | Default | Notes |
| --- | --- | --- | --- |
| `bounds` | table mapping a variable name to `[lower, upper]` | none (required) | the bounds of each variable |

Write each entry as `name = [lower, upper]`, e.g. `options = { bounds = { temperature = [250.0, 320.0] } }`.

### `histogram`

| Option | Type | Default | Notes |
| --- | --- | --- | --- |
| `variable` | string | none (required) | the variable whose histogram is constrained |
| `range` | list of numbers | none (required) | `[low, high]` of the bins |
| `bins` | integer | `50` | number of bins |
| `reference_file` | path or null | `null` | an `.npy`, or an `.npz` with the variable's array (e.g. the truth), whose histogram is the target |
| `counts` | list of numbers or null | `null` | or the target counts per bin |

Give the target either as `counts` (one number per bin) or as a `reference_file`. With a `reference_file`, the target is the histogram of the array stored in the file, for example the truth. This is the simplest choice, because APEX computes the target histogram from the array with the same `bins` and `range`, so you do not have to decide how to scale the counts yourself.

### `linear`

| Option | Type | Default | Notes |
| --- | --- | --- | --- |
| `matrix_file` | path | none (required) | an `.npz` file holding the system |
| `kind` | `"equality"` or `"inequality"` | `"equality"` | `A x - b = 0` or `A x - b <= 0` |
| `variables` | list of strings or null | `null` | the variables `A` acts on, stacked in this order; default all |

The `.npz` file holds an array named `matrix` of shape `(n_c, n)`, where `n_c` is the number of constraints and `n` the length of the stacked variables. It can also hold an array named `rhs` of shape `(n_c,)`, which is `b`. If `rhs` is absent it is zeros. The columns of `matrix` follow the variables in the order of `variables`. Always list `variables` explicitly, so that the column order of `matrix` is the one you wrote.

### `tv`

| Option | Type | Default | Notes |
| --- | --- | --- | --- |
| `variables` | list of strings or null | `null` | the gridded variables; default every one with a grid |

`tv` has no required option. If you leave `variables` out, it applies to every variable that has a grid. Name the variables to restrict it to some of them.

## Worked example

A constraint is an entry of the `gmac` strategy's `constraints`. The `weight` values below (`rule = "trace"`, `w = 0.5`) are placeholders; see [GMAC and C-GIES](../../gmac.md) for how to choose them. The variable names (`temperature`, `pressure`, `field`) and the file name are placeholders too; replace them with your own.

This block keeps `temperature` between 250 and 320 and `pressure` between 0.5 and 2.0 with `box`, and matches the `temperature` distribution to reference counts with `histogram` (20 bins, 20 counts):

```toml
[method]
name = "gies"

[method.strategy]
name = "gmac"

[[method.strategy.options.constraints]]
function = { name = "box", options = { bounds = { temperature = [250.0, 320.0], pressure = [0.5, 2.0] } } }
weight = { rule = "trace", w = 0.5 }

[[method.strategy.options.constraints]]
function = { name = "histogram", options = { variable = "temperature", range = [250.0, 320.0], bins = 20, counts = [1, 2, 4, 8, 12, 15, 16, 15, 12, 8, 6, 5, 4, 3, 2, 2, 1, 1, 1, 1] } }
weight = { rule = "trace", w = 0.5 }
```

To add an inequality `A x - b <= 0` read from `system.npz` (with `matrix` and optionally `rhs` inside), use a `linear` entry:

```toml
[method]
name = "gies"

[method.strategy]
name = "gmac"

[[method.strategy.options.constraints]]
function = { name = "linear", options = { matrix_file = "system.npz", kind = "inequality", variables = ["temperature", "pressure"] } }
weight = { rule = "trace", w = 0.5 }
```

To make the total variation of a gridded variable vanish, use `tv`:

```toml
[method]
name = "gies"

[method.strategy]
name = "gmac"

[[method.strategy.options.constraints]]
function = { name = "tv", options = { variables = ["field"] } }
weight = { rule = "trace", w = 0.5 }
```

These blocks show only the `[method]` part. A real run configuration also holds the other sections of your problem. Once the file is complete, start it with:

```bash
uv run apex run config.toml
```

Progress is shown by default on standard error; pass `--no-progress` to turn it off. If the run stops or is interrupted, `uv run apex resume <run_dir>` continues it from its last checkpoint, where `<run_dir>` is the folder of that run. `apex run` reports an error if the run folder already holds a run: resume it, or start a new one with another `[run]` name or `[run]` `output_dir` in the configuration.

## Pitfalls

- **Missing required options.** `box` needs `bounds`, `histogram` needs `variable` and `range`, and `linear` needs `matrix_file`.
- **Forgetting the reference for `histogram`.** Both `reference_file` and `counts` default to `null`. Set one of them.
- **Wrong layout of the `linear` file.** It is an `.npz` with `matrix` of shape `(n_c, n)` and optionally `rhs` of shape `(n_c,)`. The columns follow the stacked `variables`.
- **Putting `bounds` in the wrong place.** For `box`, the entries `name = [lower, upper]` belong in the function's `options.bounds`, not in the constraint entry itself.
- **Wrong `kind` for `linear`.** The default is `"equality"`. Set `kind = "inequality"` if you mean `A x - b <= 0`.
- **Typos in option names.** A stray key is an error. `uv run apex plugins` shows the exact names.
- **Variable names that do not match your problem.** Check each name in `bounds`, `variable` and `variables`: they must be state variables, or properties with a parameterization.
- **Starting a run in a folder that already holds one.** `apex run` refuses; resume it with `apex resume`, or change `[run]` name or `output_dir`. To ask a running run to stop at its next checkpoint, use `uv run apex stop <run_dir>`.
- **Using `gmac` with an unsupported method.** `gmac` works with the `gies` and `lm_enrml` methods. It is not supported with `es`, `esmda` or `gn_enrml`, and `chop` needs the `rlmmac` strategy. `multilevel_esmda` runs only the `hybrid` update, so leave out `[method.strategy]` there or name `hybrid`.

## What to read next

- [GMAC and C-GIES](../../gmac.md), for `metric`, `weight` and how constraints enter the update.
- The other plugin pages in the plugins section of this guide, to see how constraints fit with the rest of a run.

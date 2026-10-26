---
audience: public
items:
  - cli:apex plugins
  - cli:apex resume
  - cli:apex run
  - cli:apex stop
  - config:history_match:method
  - plugin:gmac_constraints:box
  - plugin:gmac_constraints:histogram
  - plugin:gmac_constraints:linear
  - plugin:gmac_constraints:tv
  - plugin:update_strategies:gmac
---

# Gmac constraints

The `gmac_constraints` registry holds four constraint plugins: `box`, `histogram`, `linear` and `tv`. Each one states a condition that the variables should satisfy. They are used by the `gmac` update strategy.

## Goal

These constraints are C-GIES's soft constraints: X. Luo and W. C. Cruz (2022). Data assimilation with soft constraints (DASC) through a generalized iterative ensemble smoother. *Computational Geosciences* 26(3), 571-594. [doi:10.1007/s10596-022-10137-7](https://doi.org/10.1007/s10596-022-10137-7)

"Soft" means the condition is not imposed as a hard restriction. Each constraint enters the `gmac` update as one more term, with a `metric` and a `weight` that set how it is measured and how strongly it counts. Both are described in [GMAC and C-GIES](../../gmac.md).

By the end of this lesson you will be able to:

- pick the right constraint plugin for a condition you want to impose;
- write the options for `box`, `histogram`, `linear` and `tv` in TOML;
- check option names with `apex plugins` and start a run with `apex run`;
- avoid the common mistakes.

## Prerequisites

- You know the names of the variables you want to constrain.
- You have a run configuration file in TOML, which is the file you pass to `apex run`. The blocks on this page go into it.
- A `[method]` section that uses the `gmac` strategy with `gies` or `lm_enrml` (see the pitfalls).
- For `linear`, a file holding the linear system. For `histogram`, reference counts, either in a file or as a list of numbers.

## How constraints are written

Each plugin defines a constraint function of the variables `x`:

- `box` gives `h = [lower - x; x - upper]`. The semicolon means the two parts are stacked into one vector. The condition is `lower <= x <= upper`.
- `histogram` states `counts(x) - counts_ref = 0`.
- `linear` states `A x - b`, where `A` is a matrix and `b` a vector. It is an equality (`= 0`) or an inequality (`<= 0`).
- `tv` gives `f = TV(x)`, and the condition is `f = 0`.

| Plugin | Condition it expresses |
| --- | --- |
| `box` | `lower <= x <= upper` per variable |
| `histogram` | the histogram of a variable equals a reference one |
| `linear` | a linear system `A x - b`, equality or inequality |
| `tv` | the total variation of gridded variables vanishes |

Total variation (`TV`) measures how much a field changes from one grid point to the next. A field with zero total variation does not change at all, so `tv` pushes the named gridded variables toward a constant field.

All four plugins reject unknown options, so a misspelled option name is an error and is not silently ignored.

## Steps

1. Decide which condition you need from the table above.
2. Write the options for that plugin, using the tables below. Required options must be present.
3. Run `apex plugins` to list every plugin with its options. Add `--json` for the full descriptions as JSON. Use it to check exact option names.
4. Make sure every variable name matches a variable in your problem.
5. Start the run with `apex run config.toml`, where `config.toml` is your configuration file.

### `box`

| Option | Type | Default | Notes |
| --- | --- | --- | --- |
| `bounds` | table mapping a variable name to a list of numbers | none (required) | the bounds of each variable |

Write each entry as `name = [lower, upper]`, e.g. `options = { bounds = { temperature = [250.0, 320.0] } }`.

### `histogram`

| Option | Type | Default | Notes |
| --- | --- | --- | --- |
| `variable` | string | none (required) | the variable whose histogram is constrained |
| `range` | list of numbers | none (required) | the range the histogram covers, written `[min, max]` |
| `bins` | integer | `50` | number of bins |
| `reference_file` | path or null | `null` | file holding the reference histogram |
| `counts` | list of numbers or null | `null` | the reference counts given directly |

The histogram of `variable`, taken over `range` with `bins` bins, is compared with reference counts. Give the reference through `reference_file` or `counts`. When you give `counts`, supply one count per bin, as in the example below. This page does not describe the layout of a `reference_file`; to avoid guessing, use `counts` unless you already have such a file.

### `linear`

| Option | Type | Default | Notes |
| --- | --- | --- | --- |
| `matrix_file` | path | none (required) | file holding the linear system `A x - b` |
| `kind` | `"equality"` or `"inequality"` | `"equality"` | `A x - b = 0` or `A x - b <= 0` |
| `variables` | list of strings or null | `null` | the variables the system acts on |

This page does not specify the contents or format of `matrix_file`, so you need to prepare that file following the reference for `linear`. Name the variables in `variables` in the order the columns of `A` use them. Setting `variables` explicitly is the safest choice.

### `tv`

| Option | Type | Default | Notes |
| --- | --- | --- | --- |
| `variables` | list of strings or null | `null` | the gridded variables to constrain |

`tv` has no required option, but in practice you name the gridded variables in `variables`.

## Worked example

A constraint is an entry of the `gmac` strategy's `constraints`. Its `function` is one of these plugins, and it comes with a `weight`. The `weight` values below (`rule = "trace"`, `w = 0.5`) are placeholders: what they mean and how to choose them is explained in [GMAC and C-GIES](../../gmac.md). The variable names (`temperature`, `pressure`, `field`) and the file name are placeholders too; replace them with your own.

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

To add an inequality `A x - b <= 0` read from a file, append a `linear` entry:

```toml
[method]
name = "gies"

[method.strategy]
name = "gmac"

[[method.strategy.options.constraints]]
function = { name = "linear", options = { matrix_file = "system.npz", kind = "inequality", variables = ["temperature", "pressure"] } }
weight = { rule = "trace", w = 0.5 }
```

To force a gridded variable toward a constant field, use `tv`:

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

To stop after a few analysis steps, add `--max-steps`, for example `apex run config.toml --max-steps 5`, then continue with `apex resume <run_dir>`, where `<run_dir>` is the directory of the run. To ask a running run to stop at its next checkpoint, use `apex stop <run_dir>`; `apex resume` continues it. Progress is shown by default; pass `--no-progress` to turn it off.

## Pitfalls

- **Missing required options.** `box` needs `bounds`, `histogram` needs `variable` and `range`, and `linear` needs `matrix_file`.
- **Forgetting the reference for `histogram`.** Both `reference_file` and `counts` default to `null`. Set one of them.
- **Putting `bounds` in the wrong place.** For `box`, the entries `name = [lower, upper]` belong in the function's `options.bounds`, not in the constraint entry itself.
- **Wrong `kind` for `linear`.** The default is `"equality"`. Set `kind = "inequality"` if you mean `A x - b <= 0`.
- **Typos in option names.** A stray key is an error. `apex plugins` shows the exact names.
- **Variable names that do not match your problem.** Check each name in `bounds`, `variable` and `variables`.
- **Using `gmac` with an unsupported method.** `gmac` works with the `gies` and `lm_enrml` methods. It is not supported with `es`, `esmda` or `gn_enrml`, and `chop` needs the `rlmmac` strategy. `multilevel_esmda` runs only the hybrid update, so leave out `[method.strategy]` there or name `hybrid`.

## What to read next

- [GMAC and C-GIES](../../gmac.md), for `metric`, `weight` and how constraints enter the update.
- The other plugin pages in the plugins section of this guide, to see how constraints fit with the rest of a run.

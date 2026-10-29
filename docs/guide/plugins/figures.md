---
audience: public
items:
  - cli:apex plugins
  - cli:apex predict
  - cli:apex report
  - cli:apex run
  - cli:apex stop
  - cli:apex summary
  - plugin:figures:data_bands
  - plugin:figures:field_layers
  - plugin:figures:field_maps
  - plugin:figures:metric_distribution
  - plugin:figures:objective_distribution
  - plugin:figures:state_histogram
  - plugin:gmac_constraints:box
  - plugin:metrics:normalized_data_mismatch
---

# Figures

This page introduces the six `figures` plugins: `data_bands`, `field_layers`, `field_maps`, `metric_distribution`, `objective_distribution` and `state_histogram`. Each one draws plots that help you judge how a history-matching run behaved.

## Goal

By the end you will know what each figure plugin draws, which options it accepts (names, types and defaults), and how to write the figures of a run to disk with `apex report`.

## Prerequisites

- A run directory, `RUN_DIR`, made with `apex run`. A run can be stopped early with `--max-steps` or with `apex stop` and continued later. A prediction directory made with `apex predict` can also be given to `apex report`.
- The names of the state variables, data types and metric you want to plot. They are plain strings and must match the names in your own setup.

To list every plugin with its options, run:

```bash
uv run apex plugins
uv run apex plugins --json
```

`apex plugins` lists every plugin together with its options, including the six figure plugins. With `--json` you get the full option descriptions as JSON, which is the form meant for clients. Use it to check the exact names, types and defaults on your installation against this page.

## The six plugins

| Plugin | What it draws |
|---|---|
| `data_bands` | Per data type and well, the initial and final ensemble forecasts as percentile bands. |
| `field_layers` | Per gridded state variable and layer: the true, prior, posterior, change and std maps. |
| `field_maps` | Per geophysical field datum and vintage: observed, prior, posterior and mismatch maps. |
| `metric_distribution` | Box and violin plots of a per-member metric over the accepted iterates. |
| `objective_distribution` | The objective over all data, as box and violin plots. |
| `state_histogram` | A state variable's values at the initial and the final iterate, one histogram per member. |

No option is shared by all six. Each plugin accepts only its own options, and an unknown option is rejected.

## Steps

1. Decide which question you want answered, then pick the plugin from the table.
2. Read the plugin's options below, in particular any required one (`metric` for `metric_distribution`, `variable` for `state_histogram`).
3. Run `apex report` on the run directory with `--plots`. It prints the data mismatch history and performance metrics of the run and also writes the result figures as PNG and PDF.
4. Use `--output` to choose another directory for the figures (the default is `RUN_DIR/report`), and `--font-size` to set the font size of the figures in points (default 12.0). `--no-plots` turns the figures off again, which is the default.
5. To compare several runs in one table instead, use `apex summary`, which lists every metric at every step. Its `--output` writes the Markdown (and a CSV next to it) to a file, and `--title` sets the report title.

The command-line flags of `apex report` are the only ones this page can vouch for: they do not take plugin options. How a figure plugin's own options (`metric`, `variable`, `kinds` and so on) are chosen, and which plugins `--plots` draws by default, is covered on the [Figures](../../figures.md) page, together with the presets.

## Options

### `data_bands`

- `source` (`predicted` or `auxiliary`, default `predicted`): `predicted` plots the forecasts of the observations. `auxiliary` plots the forecasts saved beside them that were never assimilated (a model-error correction's residuals).
- `observations` (true, false or unset, default unset): whether to draw the observations on the bands. Left unset, they are drawn for `predicted` only.
- `datatypes` (list of strings, default empty): only these data types, for example `["WOPR:BR-P-1"]`. Empty means all scalar ones.

### `field_layers`

- `variables` (list of strings, default empty): the variables to draw. Empty means every one with a grid.
- `layers` (list of integers, default empty): 1-based layers. Empty means all.
- `kinds` (list of strings): which maps to draw. The default is all six values below. `ref` needs the true model.
- `exponentiate` (list of strings or unset, default unset): variables shown as the exponential of their values, which suits ln-permeabilities. When unset, the variables whose name contains `perm` are exponentiated. The means are computed from the exponentiated members.

| `kinds` value | Map |
|---|---|
| `ref` | The true (reference) field |
| `prior` | The prior ensemble |
| `post` | The posterior ensemble |
| `diff` | The change from prior to posterior |
| `std_prior` | The std map of the prior |
| `std_post` | The std map of the posterior |

### `field_maps`

- `datatypes` (list of strings, default empty): the field data to draw. Empty means every one with a map.
- `kinds` (list of strings): which maps to draw. The default is `obs`, `prior`, `post` and `mismatch`, all four.

| `kinds` value | Map |
|---|---|
| `obs` | The observed field |
| `prior` | The prior prediction |
| `post` | The posterior prediction |
| `mismatch` | The mismatch between observed and predicted |

### `metric_distribution`

- `metric` (string, **required**): the recorded metric, for example `rmse`, `ssim`, `model_change` or `normalized_data_mismatch`.
- `labels` (list of strings, default empty): which labels of that metric to draw. Empty means every one, with `all` last. A label of the form `TYPE:WELL` is written `TYPE WELL` in the plot.
- `kinds` (list of strings, default the list `["violin", "box"]`): `violin` and/or `box`, in the order drawn.
- `prefix` (string, default empty, which means the metric name is used): the start of the file names and the word in the y-label. The files are `<prefix>_<label>_violinplot` and `<prefix>_<label>_boxplot`, and the y-label is `<prefix> (<label>)`.
- `font_size` (number, default 28.0): tick and axis label size.

### `objective_distribution`

- `metric` (`data_mismatch` or `normalized_data_mismatch`, default `data_mismatch`): the objective's metric.
- `font_size` (number, default 30.0): tick and axis label size.

### `state_histogram`

- `variable` (string, **required**): the state variable, for example `nn_params`, a model-error network's parameters.
- `bins` (integer, default 50): the number of shared bins.
- `font_size` (number, default 16.0): title, axis label and tick size.
- `prefix` (string, default `hist`): the file prefix.

## Worked example

Suppose a run in `runs/case1` has finished. Write its figures to `runs/case1/figs` with a font size of 14 points:

```bash
uv run apex report runs/case1 --plots --output runs/case1/figs --font-size 14
```

The command prints the data mismatch history and performance metrics, and writes the figures, as PNG and PDF, to `runs/case1/figs`. Without `--output` they go to `runs/case1/report`. Open the PNG files with any image viewer; the PDF files hold the same figures.

To compare this run with others in one Markdown table:

```bash
uv run apex summary runs --output runs-summary.md --title "Case 1 runs"
```

Here `runs` is a folder holding runs, which `apex summary` searches.

## Pitfalls

- **Missing required option.** `metric_distribution` needs `metric` and `state_histogram` needs `variable`.
- **Names the run does not have.** Use metric, variable and data type names that exist in your own setup.
- **Wrong value for `objective_distribution`'s `metric`.** Only `data_mismatch` and `normalized_data_mismatch` are allowed. `metric_distribution`'s `metric` is a free string instead.
- **Wrong types.** `layers` holds integers, not strings. `bins` is an integer. `font_size` is a number.
- **Similar `kinds` options.** In `field_layers` and `field_maps` the `kinds` are map kinds. In `metric_distribution` they are the plot styles `violin` and `box`. Use the values listed for each plugin.
- **`ref` without a true model.** The `ref` map of `field_layers` needs the true model.
- **Forgetting `--plots`.** `apex report` only prints the numbers unless you pass `--plots`.

## What to read next

- [Figures](../../figures.md), for the plotter agent, the presets, how plugin options are chosen and how to add a figure plugin.
- `apex plugins --json`, to see the full description of every option on your installation.

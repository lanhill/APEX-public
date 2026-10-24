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
---

# Figures

This page introduces the six `figures` plugins: `data_bands`, `field_layers`, `field_maps`, `metric_distribution`, `objective_distribution` and `state_histogram`. Each one draws plots that help you judge how a history-matching run behaved.

## Goal

By the end you will know what each figure plugin draws, which options it accepts (names, types and defaults), and how to write the figures of a run to disk with `apex report`.

## Prerequisites

- A history-matching run directory, `RUN_DIR`, made with `apex run` (a run stopped early with `--max-steps` or with `apex stop` also works). A prediction directory made with `apex predict` can be given too.
- The names of the state variables, data types and metric you want to plot. They are plain strings and must match the names in your own setup.

To list every plugin with its options, run:

```bash
apex plugins
apex plugins --json
```

The `--json` flag prints the full descriptions as JSON.

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
2. Check the plugin's options below, in particular any required one (`metric` for `metric_distribution`, `variable` for `state_histogram`).
3. Print the data mismatch history and performance metrics of the run, and write the figures as PNG and PDF, with `apex report` and `--plots`.
4. Use `--output` to choose another directory for the figures (the default is `RUN_DIR/report`), and `--font-size` to set the font size in points (default 12.0). `--no-plots` turns the figures off again, which is the default.
5. To compare several runs in one table instead, use `apex summary`, which lists every metric at every step. Its `--output` writes the Markdown (and a CSV next to it) to a file, and `--title` sets the report title.

## Options

### `data_bands`

- `source` (`predicted` or `auxiliary`, default `predicted`): which forecasts are plotted.
- `observations` (true, false or unset, default unset).
- `datatypes` (list of strings, default empty): the data types, for example `["oil_rate"]`.

### `field_layers`

- `variables` (list of strings, default empty): the gridded state variables.
- `layers` (list of integers, default empty): the layers.
- `kinds` (list of strings): which maps to draw. The default is all six values below.
- `exponentiate` (list of strings or unset, default unset): variable names.

| `kinds` value | Map |
|---|---|
| `ref` | The true (reference) field |
| `prior` | The prior ensemble |
| `post` | The posterior ensemble |
| `diff` | The change from prior to posterior |
| `std_prior` | The std map of the prior |
| `std_post` | The std map of the posterior |

### `field_maps`

- `datatypes` (list of strings, default empty): the geophysical field data to map.
- `kinds` (list of strings): which maps to draw. The default is `obs`, `prior`, `post`, `mismatch`.

| `kinds` value | Map |
|---|---|
| `obs` | The observed field |
| `prior` | The prior prediction |
| `post` | The posterior prediction |
| `mismatch` | The mismatch between observed and predicted |

### `metric_distribution`

- `metric` (string, **required**): the per-member metric to plot. It is a free string, so use a metric name from your own setup.
- `labels` (list of strings, default empty).
- `kinds` (list of strings, default `violin`, `box`): the plot styles.
- `prefix` (string, default empty).
- `font_size` (number, default 28.0).

### `objective_distribution`

- `metric` (`data_mismatch` or `normalized_data_mismatch`, default `data_mismatch`).
- `font_size` (number, default 30.0).

### `state_histogram`

- `variable` (string, **required**): the state variable.
- `bins` (integer, default 50): the number of histogram bins.
- `font_size` (number, default 16.0).
- `prefix` (string, default `hist`).

`font_size` is accepted by `metric_distribution`, `objective_distribution` and `state_histogram`. It is a plain number; larger values give larger text.

## Worked example

Suppose a run in `runs/case1` has finished. Write its figures to `runs/case1/figs` with a font size of 14 points:

```bash
apex report runs/case1 --plots --output runs/case1/figs --font-size 14
```

The command prints the data mismatch history and performance metrics, and writes the figures, as PNG and PDF, to `runs/case1/figs`. Without `--output` they go to `runs/case1/report`.

To compare this run with others in one Markdown table:

```bash
apex summary runs --output runs-summary.md --title "Case 1 runs"
```

Here `runs` is a folder holding runs, which `apex summary` searches.

## Pitfalls

- **Missing required option.** `metric_distribution` needs `metric` and `state_histogram` needs `variable`.
- **Names the run does not have.** Use metric, variable and data type names that exist in your own setup.
- **Wrong value for `objective_distribution`'s `metric`.** Only `data_mismatch` and `normalized_data_mismatch` are allowed. `metric_distribution`'s `metric` is a free string instead.
- **Wrong types.** `layers` holds integers, not strings. `bins` is an integer. `font_size` is a number.
- **Similar `kinds` options.** In `field_layers` and `field_maps` the `kinds` are map kinds. In `metric_distribution` they are the plot styles `violin` and `box`. Use the values listed for each plugin.
- **Forgetting `--plots`.** `apex report` only prints the numbers unless you pass `--plots`.

## What to read next

- The other plugin pages in this guide, to see what produces the results these figures plot.
- [Figures](../../figures.md), for the plotter agent, the presets and how to add a figure plugin.

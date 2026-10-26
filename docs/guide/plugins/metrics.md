---
audience: public
items:
  - cli:apex plugins
  - cli:apex predict
  - cli:apex report
  - cli:apex resume
  - cli:apex run
  - cli:apex schema
  - cli:apex stop
  - plugin:metrics:constraint_cost
  - plugin:metrics:facies_brier
  - plugin:metrics:facies_log_score
  - plugin:metrics:facies_mismatch
  - plugin:metrics:model_change
  - plugin:metrics:normalized_data_mismatch
  - plugin:metrics:rmse
  - plugin:metrics:ssim
---

# Metrics

This page covers the eight plugins in the `metrics` registry: `constraint_cost`, `facies_brier`, `facies_log_score`, `facies_mismatch`, `model_change`, `normalized_data_mismatch`, `rmse` and `ssim`. It also shows the commands you use to run a history match and read its metrics: `apex run`, `apex resume`, `apex stop`, `apex report`, `apex plugins` and `apex schema`. `apex predict` is mentioned at the end.

## Goal

By the end of this page you will know:

- what each metric measures,
- which options each plugin accepts, with their types and defaults,
- how to add metrics to a TOML run configuration,
- how to start, stop, resume and report on a run.

## Prerequisites

- A run configuration, which is a TOML file. `apex run` reads it and starts a history-matching run. The metric entries on this page are only part of that file. To see everything the file accepts, run `apex schema history_match`. It prints the JSON Schema of the configuration.
- For the metrics that compare against the true model (`rmse`, `ssim`, `facies_mismatch`), a true model to compare with.
- For the facies metrics (`facies_brier`, `facies_log_score`, `facies_mismatch`), facies variables in your configuration. Use the `variables` option to name them.

## The metrics

| Plugin | What it measures |
| --- | --- |
| `normalized_data_mismatch` | `mean(((d_obs - y_j) / sigma)^2)` against the observed values. `d_obs` is the observed data, `y_j` is the simulated data of ensemble member `j`, and `sigma` normalizes each difference. |
| `model_change` | Change of each member's parameters since the ensemble it was updated from. |
| `rmse` | RMSE of each member's parameters against the true model. |
| `ssim` | Structural similarity of each member's fields to the true model's. |
| `facies_brier` | Brier score (Brier, 1950) of the ensemble's facies probabilities. |
| `facies_log_score` | Logarithmic score (Good, 1952) of the ensemble's facies probabilities. |
| `facies_mismatch` | Fraction of cells whose facies differs from the true model's. |
| `constraint_cost` | Each member's `D(0 - c(m_j))` and the violated inequality rows. `m_j` is member `j`. |

Values of different metrics are on different scales. Compare a metric's values only between runs that use the same metric with the same options.

## Options

Each plugin accepts only the options listed here. A default of `null` means the option has no value unless you set one. Leave it out of the TOML file to keep `null`.

### `normalized_data_mismatch`

No options.

### `constraint_cost`

| Option | Type | Default | Notes |
| --- | --- | --- | --- |
| `function` | table | none | Required. |
| `metric` | table, or `null` | `null` | Optional. |

Both options are tables. The keys inside them are not described on this page. Run `apex schema history_match` to see what the configuration accepts there.

### `model_change`, `rmse`, `facies_brier`, `facies_log_score`, `facies_mismatch`

These five take the same two options.

| Option | Type | Default |
| --- | --- | --- |
| `variables` | list of strings, or `null` | `null` |
| `actnum` | array, or `null` | `null` |

`variables` is a list of variable names, for example `variables = ["poro", "permx"]`. `actnum` is an array that marks the active cells. Leave both at `null` unless you need them.

### `ssim`

`ssim` takes `variables` and `actnum` as above, plus these options.

| Option | Type | Default | Notes |
| --- | --- | --- | --- |
| `grid` | list of integers, or `null` | `null` | The shape of your grid, one integer per dimension, for example `[100, 100, 20]`. |
| `order` | `"F"` or `"C"` | `"F"` | Use capital letters. |
| `win_size` | integer | `7` | |
| `per_layer` | true/false | `false` | Off by default. |
| `data_range` | number, or `null` | `null` | A single number. |

## Steps

1. **Pick the metrics for your question.** Use `normalized_data_mismatch` to compare simulated and observed data. Use `rmse` or `ssim` to compare parameters or fields with the true model. Use `facies_brier`, `facies_log_score` and `facies_mismatch` for facies. Use `model_change` to see how far members have moved from the ensemble they were updated from. Use `constraint_cost` to check constraints.
2. **Write the plugins into your run configuration.** Each metric is one `[[metrics]]` entry (an array of tables, so the brackets are doubled). `name` is the plugin and `options` holds its options. Put only the options listed above in `options`. Confirm the layout against `apex schema history_match`.
3. **Start the run** with `apex run config.toml`. Add `--max-steps N` to stop after `N` analysis steps, the update steps of the history match. Add `--no-progress` to hide the progress display.
4. **Stop or continue.** `apex stop RUN_DIR` asks a running run to stop at its next checkpoint. A run started with `--max-steps` stops by itself after `N` steps. In both cases `apex resume RUN_DIR` continues from the last checkpoint. `RUN_DIR` is the directory of the run. `apex resume` also accepts `--max-steps` and `--no-progress`.
5. **Read the results** with `apex report RUN_DIR`. It prints the data mismatch history and the performance metrics of the run.

## Worked example

Suppose you want the normalized data mismatch, the parameter RMSE, the facies mismatch and the structural similarity on a grid of 100 x 100 x 20 cells. These are the metric entries of the run configuration:

```toml
[[metrics]]
name = "normalized_data_mismatch"

[[metrics]]
name = "rmse"

[[metrics]]
name = "facies_mismatch"

[[metrics]]
name = "ssim"
options = { grid = [100, 100, 20], order = "F", win_size = 7, per_layer = false }
```

The values of `order`, `win_size` and `per_layer` are the defaults, so you can leave them out. `variables` and `actnum` stay at `null`.

Start the run, stopping after 5 analysis steps. Then continue it and print the report. Here the run directory is called `my_run`. Use the directory of your own run.

```bash
uv run apex run config.toml --max-steps 5
uv run apex resume my_run
uv run apex report my_run
```

Add `--plots` to also write the figures (PNG and PDF). They go to `my_run/report` unless you pass `--output`. `--font-size` sets the font size in points (default 12).

```bash
uv run apex report my_run --plots --output figures --font-size 14
```

To list all plugins with their options, run `apex plugins`. Add `--json` for the full descriptions as JSON.

## Pitfalls

- **Options belong to their plugin.** `grid`, `order`, `win_size`, `per_layer` and `data_range` belong to `ssim` only. `normalized_data_mismatch` takes no options.
- **`constraint_cost` requires `function`.** Without it the plugin is not configured.
- **`order` takes only two values.** Use `"F"` or `"C"`, in capitals.
- **The scores are different measures.** Do not compare a `facies_brier` value with a `facies_log_score` value as if they were on the same scale.
- **Changing options between runs.** If you change `win_size`, `grid` or any other option, do not compare the new values with earlier ones.

## What to read next

- Run `apex plugins` to see every plugin with its options.
- Run `apex schema history_match` to see the whole run configuration. `apex schema` also accepts `prediction`, `study`, `optimization`, `optimization_study` and `service`.
- To forecast the ensembles of a history-matching run beyond history, use `apex predict` with a prediction configuration. `apex schema prediction` shows what that file accepts. `apex report` also accepts the directory of a prediction.
- The other plugin pages in this guide.

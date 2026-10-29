---
audience: public
items:
  - cli:apex agent plot
  - cli:apex plugins
  - cli:apex report
  - cli:apex run
  - cli:apex stop
  - plugin:figure_presets:brugge_classic
  - plugin:metrics:model_change
---

# Figure presets

## Goal

By the end of this page you will know what the `figure_presets` registry is for, what the `brugge_classic` plugin draws, and how to switch on its options: `histogram_variable`, `normalized` and `model_change`. You will draw the figures of a run with `uv run apex agent plot`.

## Prerequisites

- APEX installed as the README describes. Type every command from the APEX folder, starting with `uv run apex`.
- A finished or stopped run, which you can pass to the command as a run directory, a run id or a job id. A run starts with `uv run apex run CONFIG.toml`. A running run can be asked to stop at its next checkpoint with `uv run apex stop RUN_DIR`.

## What `brugge_classic` is

`brugge_classic` is a plugin in the `figure_presets` registry. A figure preset is a named, ready-made set of result figures. `brugge_classic` is the classic figure set of an ensemble history-matching study, arranged in folders such as `nn_hist/`.

A preset is not part of a run configuration file. You ask for it when you draw the figures of a run, with `uv run apex agent plot` and its `--preset` option. With `--preset`, no model is asked.

To see every plugin with its options, run `uv run apex plugins`. Add `--json` for the full descriptions as JSON.

## Options

The plugin accepts only the options below. Any other option name is rejected.

| Option | Type | Default | What it does |
| --- | --- | --- | --- |
| `histogram_variable` | string | `nn_params` | The state variable of `nn_hist/`. |
| `normalized` | boolean | `false` | Also draws `normalized_data_mismatch` (the objective, and per data type and well). |
| `model_change` | boolean | `false` | Also draws `model_change`. |

The `model_change` figure is based on the `model_change` plugin of the `metrics` registry. That metric measures the change of each member's parameters since the ensemble it was updated from.

## Steps

1. Decide whether the defaults suit your run. If they do, name the preset alone.
2. To switch on `normalized` or `model_change`, add `--with` followed by the option name. Repeat `--with` once per option.
3. Optionally choose where the figures go with `--output`. Without it they go to `RUN_DIR/report/plots`.
4. Open the output folder and look at the figures. The histogram figures are in `nn_hist/`.

`--with` only switches an option on. The command line shows no way to give `histogram_variable` a value, so the default, `nn_params`, applies when you name the preset.

## Worked example

The classic set with every default:

```bash
uv run apex agent plot RUN_DIR --preset brugge_classic
```

The same set, with `normalized` and `model_change` switched on:

```bash
uv run apex agent plot RUN_DIR --preset brugge_classic --with normalized --with model_change
```

The same set, written to a folder of your choice:

```bash
uv run apex agent plot RUN_DIR --preset brugge_classic --output my_figures
```

Replace `RUN_DIR` with your run directory, run id or job id. If your runs are not in the default services root (`$APEX_HOME`, else `~/.apex/service`), add `--root` with the right folder.

## Pitfalls

- **Defaults apply when you stay silent.** If you expect `normalized` or `model_change` to be on, switch them on with `--with`.
- **Unknown options are rejected.** The plugin accepts only `histogram_variable`, `normalized` and `model_change`. A misspelled name is an error, not a silent no-op.
- **`--preset` and `--request` are different routes.** `--preset` draws a named preset without asking a model. `--request` describes in words what to draw.
- **A preset name is not a file.** Do not put `brugge_classic` in a run configuration; it is only given to `apex agent plot`.

## What to read next

- [Figures](../../figures.md), for the plotter agent and the figure plugins.
- `uv run apex report RUN_DIR --plots`, which prints the data mismatch history and performance metrics of a run and also writes its result figures (PNG and PDF).

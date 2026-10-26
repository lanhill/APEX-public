---
audience: public
items:
  - plugin:figure_presets:brugge_classic
---

# Figure presets

## Goal

By the end of this page you will know what the `figure_presets` registry is for, what the `brugge_classic` plugin produces, and how to set its three options: `histogram_variable`, `normalized` and `model_change`.

## Prerequisites

- A finished or stopped history-matching run directory, `RUN_DIR`.
- matplotlib, in APEX's default environment (`uv sync --locked`); when APEX is a dependency of your own project, name the `plots` extra.

## What `brugge_classic` is

`brugge_classic` is a plugin in the `figure_presets` registry. It produces the classic figure set of an ensemble history-matching study, organised in its folders. It is not part of a configuration file: the plotter agent draws it, `apex agent plot RUN_DIR --preset brugge_classic`, or the assistant's `plot_run`, or the REST route `POST /v1/agents/plot` with `"preset": "brugge_classic"`.

## Options

The plugin accepts only the options below. Any other option name is rejected.

| Option | Type | Default |
| --- | --- | --- |
| `histogram_variable` | string | `nn_params` |
| `normalized` | boolean | `false` |
| `model_change` | boolean | `false` |

- `histogram_variable` is the variable used for the histogram figures. It defaults to `nn_params`.
- `normalized` is a true/false setting and is off by default.
- `model_change` is a true/false setting and is off by default.

## Steps

1. Decide whether the defaults suit your run. If they do, name the preset alone.
2. Switch on `normalized` or `model_change` with `--with`, once per option.
3. Through REST, give the same switches as `preset_options`, e.g. `{"model_change": true}`. They take true or false only, so `histogram_variable` keeps its default, `nn_params`, when the preset is named.
4. Look for the figures in the preset's folders under `RUN_DIR/report/plots/`.

## Worked example

The classic set with every default:

```bash
uv run apex agent plot RUN_DIR --preset brugge_classic
```

With `normalized` and `model_change` switched on:

```bash
uv run apex agent plot RUN_DIR --preset brugge_classic --with normalized --with model_change
```

Through REST, the same: `POST /v1/agents/plot` with `{"run": "RUN_ID", "preset": "brugge_classic", "preset_options": {"normalized": true, "model_change": true}}`, where `RUN_ID` is the run's id from `GET /v1/runs` (not its folder).

## Pitfalls

- **Booleans are not strings.** In `preset_options`, write `"normalized": true`, not `"normalized": "true"`.
- **Unknown options are rejected.** The plugin accepts only `histogram_variable`, `normalized` and `model_change`. A misspelled name is an error, not a silent no-op.
- **`histogram_variable` is a string**, the name of a state variable of the run, as in `"nn_params"`.
- **Defaults apply when you stay silent.** If you expect `normalized` or `model_change` to be on, switch them on explicitly.

## What to read next

- The other pages in the plugins section of this guide, for the remaining plugin registries.
- [Figures](../../figures.md), for the plotter agent and the figure plugins.

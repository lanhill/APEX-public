---
audience: public
---

# Figures and the plotter agent

## Goal

By the end of this page you can draw a history-matching run's result figures with the plotter agent, from the command line, the web UI or the assistant. You will know the figure plugins and presets it chooses from, and how to add your own.

## Prerequisites

- A history-matching run with at least one scored forecast (`apex run CONFIG.toml`).
- matplotlib, in APEX's default environment (`uv sync --locked`); when APEX is a dependency of your own project, name the `plots` extra.
- For the model to choose the figures, a model set in `[agents.llm]` (see [runtime agents](agents.md)). Without one, the agent's rules choose them.

## Step 1: draw the classic set

```bash
uv run apex agent plot RUN_DIR --preset brugge_classic
```

The figures go to `RUN_DIR/report/plots/`, one figure per file, as PNG and PDF. The run's page in the web UI shows them under *Figures*. `--output DIR` writes them elsewhere. `summary.json` lists every figure drawn, and why any was skipped.

The `brugge_classic` preset writes the classic figure set of an ensemble history-matching study, in these folders:

| Folder | Figures |
|---|---|
| `obj/` | The objective (the data mismatch over all data) per accepted iterate, as a box plot and a violin plot |
| `data_mismatch/` | The data mismatch per data type and well (`WOPR BR-P-1`, ...) and over all data |
| `rmse/`, `mssim/` | The model's RMSE and mean SSIM against the truth, per variable and over all |
| `simData/` | Per data type and well, the initial and the final ensemble forecasts as percentile bands, with the observations |
| `residual/` | The same bands for forecasts saved beside the data but never assimilated: the residuals of a model-error characterization (MEC) run |
| `models/` | Per gridded variable and layer: the true, prior and posterior means, their difference, and the prior and posterior standard deviations |
| `nn_hist/` | The histogram of a state variable at the initial and final iterate, e.g. the MEC network's parameters `nn_params` |

Two groups are drawn only when you ask for them:

- `normalized`: the normalized data mismatch (the mean square of the residuals against the observations, scaled by their errors), as `obj_normalized/` and `normalized_data_mismatch/`.
- `model_change`: the change between iterates, per variable and over all, as `model_change/`. Every run records it.

Ask with `--with normalized` or `--with model_change` after `--preset brugge_classic`, or in words: *"the classic set with the model change"*. Through REST, give `"preset_options": {"model_change": true}`.

A figure the run cannot give is skipped and reported, with the reason, in `summary.json` and by the command: no truth means no `rmse`, `mssim` or reference map; no residuals means no `residual/`; a metric the run's `[[metrics]]` leave out means none of its figures.

## Step 2: ask in words

```bash
uv run apex agent plot RUN_DIR --request "the RMSE box plots and the layer maps"
```

The agent reads what the run holds (its metrics and their labels, the variables with grids, the data types, whether it saved residuals) and every figure plugin's options. It then chooses figures and their options for the request.

- **With a model**, code checks the model's choice against the run before anything is drawn. An unknown figure, an option the plugin refuses, a metric the run did not record, or more than `[agents.plotter] max_figures` requests sends the plan back to the model with the reasons.
- **Without a model** (`--no-llm`), or when the model fails, the rules match the request's whole words (and their plurals): objective, mismatch, normalized, rmse, ssim, model change, forecast or wells, residual or MEC, map or layer, histogram, and seismic, AVO, gravity, displacement, subsidence, time shift or 4D (the geophysical maps). A request naming the classic set (classic, preset, all figures, full set) draws `brugge_classic`, with `normalized` and `model_change` switched on when the request names them. With no request, or none of these words, they draw `brugge_classic` as it is.

The decision is recorded in `RUN_DIR/agents/`, like every agent's.

## Step 3: in the web UI and the assistant

- On a run's page, *Draw figures* takes a request in words, or none for the classic set. Tick *ask the model* to let the model choose. Reload the page to see the figures under *Figures*.
- The assistant's tool `plot_run` draws a run's figures for a request or a preset, e.g. *"Draw the box plots of my run's RMSE"*.
- REST: `POST /v1/agents/plot` with `run`, and optionally `request`, `preset`, `preset_options` and `model`.

These three draw in the background: they answer once the figures are chosen, and the figures appear as they are drawn. `report/plots/summary.json` says `drawing` until all are done. A second drawing of the same run is refused until the first ends. So is a drawing beyond `[agents.plotter] max_drawings` on the machine. The command line draws in the foreground.

Guests use the plotter only if their access list names it (`[agents] names = ["plotter"]`), and only on their own runs.

## The figure plugins

Figures are plugins, chosen by name, in the registry `figures`. Presets are named sets of them, in `figure_presets`. `apex plugins` lists both with their options, and so does the web UI's *Plugins* page under *Figures*.

| Plugin | What it draws | Main options |
|---|---|---|
| `metric_distribution` | Box and violin plots of a per-member metric over the accepted iterates, 20 × 16 in at 500 dpi | `metric`, `labels`, `kinds` (`violin`, `box`), `prefix` |
| `objective_distribution` | The objective as a box and a violin plot | `font_size` |
| `data_bands` | Per data type and well, initial and final percentile bands (0–100 light grey, 10–90 grey, the mean orange, observations red), with days since the deck's `START` on the x axis | `source` (`predicted`, `auxiliary`), `observations`, `datatypes` |
| `field_layers` | Per gridded variable and layer, `ref`, `prior`, `post`, `diff`, `std_prior` and `std_post` maps (`jet`, inactive cells blank); ln-permeabilities shown as permeabilities | `variables`, `layers`, `kinds`, `exponentiate` |
| `state_histogram` | A state variable's values at the initial and final iterate, one histogram per member | `variable`, `bins`, `prefix` |
| `field_maps` | Per geophysical field datum (AVO maps, time shifts, gravity, seafloor displacement) and vintage: the observed map, the prior and posterior means of the forecast, and the posterior mean minus the observed (`jet`; maps over the grid's columns, j downwards, untraced ones blank, a thinned survey on its own lattice; stations as points at their x and y in m, y upwards). Whole traces and data compressed by a transform have no map | `datatypes`, `kinds` (`obs`, `prior`, `post`, `mismatch`) |

The accepted iterates are numbered 0, 1, 2, … in order. A rejected trial takes no number.

## Settings

`[agents.plotter]` in `service.toml`:

| Setting | Default | Meaning |
|---|---|---|
| `max_figures` | 40 | The most figure requests one plot may draw (a preset counts as all its requests, including those the run cannot give) |
| `dpi_scale` | 1.0 | The figures' resolution relative to the plugins' own (at most 2); lower draws faster and smaller files |
| `max_drawings` | 2 | Background drawings running at once on the machine, over all runs and users; another is refused until one ends |

The classic set of a large case is large: for a 139 × 48 × 9 grid with 70 data types, about 160 box and violin plots of 10000 × 8000 pixels, and several hundred maps and band figures. Lower `dpi_scale` to save time and space.

## Adding a figure

A figure plugin is a class with `applies(view)` and `draw(view, output_dir, *, dpi_scale)`, registered in `apex.core.plugins.figures`. The view (`apex.core.figures.RunView`) gives the run's iterations, iterates (state, forecasts and auxiliary forecasts), metrics, truth, observations and grids. `applies` returns why the figure cannot be drawn for a run, or `None`. `draw` writes PNG and PDF twins and returns the PNG paths. A package can register plugins through the entry-point group `apex.figures`, and presets through `apex.figure_presets`.

## Pitfalls

- **The figures are of the members only.** A mean-model forecast (GIES) is kept apart and never drawn as a member.
- **"Final" and "posterior" mean the run's posterior.** With cross validation, that is the iterate of least cross-validation mismatch, not the last.
- **The final band figure keeps the initial one's y limits**, so a converged posterior is easy to compare with the prior; posterior bands outside them are cut.
- **`Sm3/Day` is the y label of every rate and of the water cut.** This is the classic set's labelling.

## What to read next

- [Runtime agents](agents.md): the other agents, and how a model's answer is held to the facts.
- [Run summaries](summaries.md): every metric at every step, as tables.

---
audience: public
items:
  - cli:apex plugins
  - cli:apex report
  - cli:apex resume
  - cli:apex run
  - cli:apex stop
  - cli:apex summary
  - config:history_match:prior
  - plugin:localizations:adaptive_prior
  - plugin:model_error_models:pg_lstm
  - plugin:objectives:sch
  - plugin:surrogate_designs:run
---

# Model error models

## Goal

This page shows how to set up the `pg_lstm` plugin from the `model_error_models` registry. `pg_lstm` provides physics-guided LSTM well residual models, one per well. A residual is the correction a well's model adds to the simulated data of that well. By the end you will have a complete `[model_error]` section, and you will know what each optional setting controls. [MEC](../../mec.md) describes the method and the rest of the configuration a run needs, including the declaration of the state variable that holds the LSTM parameters and its prior (the `[prior]` section, where `[prior.variables.<name>]` sets a generator for one variable).

## Prerequisites

- A history schedule file for your model, which you will point `schedule_file` at. It holds one COMPDAT, WCONHIST and WCONINJH block per report step.
- The report steps you want the well models to act at. Report steps are numbered from 1 in the order of the schedule's blocks: step 1 is the first block, step 2 the second, and so on. Each step has a report time, which is also the time of an observation.
- The names of the producer and injector wells you want to cover.
- When you use `causal_mask` or `save_inputs`, or when your simulator takes its wells from its own input instead of a schedule file: the ISO date of each report step.
- A run configuration file (TOML) that already holds the rest of the run setup described in [MEC](../../mec.md). This page only covers the `[model_error]` section of that file.

To see every option of the plugin from the command line, run this from the APEX folder:

```bash
uv run apex plugins
```

Add `--json` to get the full descriptions as JSON:

```bash
uv run apex plugins --json
```

## Steps

1. **Add the section.** Create a `[model_error]` section with `name = "pg_lstm"`, and put its options in `[model_error.options]`.
2. **Set `report_steps`.** It is a list of integers: the 1-based report steps (see the prerequisites) at which the well models act. Default: `[]`, which names no step, so list the steps you need.
3. **Point to the schedule.** `schedule_file` is the path to your history schedule (a string or a path). Default: unset. Leave it out only on a simulator whose wells come from its own input rather than from a schedule; in that case `report_dates` becomes required (step 10).
4. **List the wells.**
   - `producers`: a list of producer well names. Default: `[]`.
   - `injectors`: a list of injector well names. Default: `[]`.
5. **Choose the data used for each well type (optional).** These are the data types of the wells, given as strings.
   - `producer_data`: default `["WOPR", "WBHP", "WWCT"]`.
   - `injector_data`: default `["WBHP"]`.
6. **Tune the LSTM and its state (optional).**
   - `hidden_size`: the hidden size of each LSTM, an integer. Default: `20`.
   - `state_variable`: the name of the state variable that holds the parameters of every well's LSTM, producers first and then injectors, each in torch's layout. Default: `"nn_params"`. Change it only if your configuration declares the LSTM parameters under another name; the name here must match the state variable declared in your run configuration (see [MEC](../../mec.md)).
7. **Decide how shut wells are treated (optional).** `shut_wells` accepts two values. Default: `"zero"`.
   - `"zero"`: there is no residual while a well is shut, except for `WBHP`.
   - `"ignore"`: residuals are produced at every report time, even when the well is shut.
8. **Set the phases (optional).** `phases` is a list of strings naming the phases of the deck, one or more of `oil`, `water` and `gas`. They set the dynamic features, namely each phase's density and saturation. Without oil, its density and saturation are 0. Default: `["oil", "water"]`.
9. **Give an initial schedule (optional).** `initial_schedule_file` is schedule text that is read before `schedule_file`. Use it for the deck's own SCHEDULE keywords that come before its included history, for example a first COMPDAT. Leave it unset (the default) if the deck has none.
10. **Give the report dates.** `report_dates` is a list of ISO date strings, one per report step, in order. Default: `[]`. It is optional in general, but you must give it when you use `causal_mask` or `save_inputs`, or when you leave out `schedule_file`, because the plugin must place each datum at its report step before any run. The report dates of every case in your run are checked against this list, so it must agree with them.
11. **Declare the causal structure (optional).** `causal_mask` is a boolean. Default: `false`. When `true`, it tells the run's localization how the well models are built: a well's LSTM acts on its own outputs at every report time, and the output layer for report time `t` (each output's row and bias) acts on that output at `t` only. It requires the `adaptive_prior` localization, see below.
12. **Save the inputs (optional).** `save_inputs` is a boolean. Default: `false`. When `true`, the plugin also answers the inputs of the well models and saves them with the residuals. There is one answer per well, per report date and per input feature, named `MEC:X<j>:<well>`, where `<j>` is the index of the feature among the well's inputs and `<well>` is the well name. The features depend on your `phases` and data lists. The idea is a two-stage use: first do a run of the prior with `save_inputs = true`, so that its results hold the inputs next to the residuals. A per-step prior, which works through the report steps in sequence, then takes those saved inputs for its sequential pass. [MEC](../../mec.md) covers how to set up that prior.

No other options are accepted: unknown keys are rejected.

### Using `causal_mask` with `adaptive_prior`

`adaptive_prior` is a plugin of the `localizations` registry: adaptive localization whose taper is built once from the prior. You select it in the localization section of your run configuration. To use the mask on its own, list the state variable of the LSTM parameters in its `unlocalized` option. Those variables are then updated without localization, so only the mask applies:

```toml
[localization]
name = "adaptive_prior"

[localization.options]
unlocalized = ["nn_params"]
```

This is useful because single network weights correlate too weakly with the data for the adaptive threshold, which would otherwise switch almost all of them off. The other `adaptive_prior` options (`threshold_scale`, `taper`, `threshold`, `cutoff`) keep their defaults here. See [MEC](../../mec.md) for how it fits into the full configuration.

## Worked example

This section configures `pg_lstm` for two producers and one injector. It keeps the default data lists and sets the LSTM size, the shut-well handling and the report dates explicitly.

```toml
[model_error]
name = "pg_lstm"

[model_error.options]
schedule_file = "schedule.sch"
report_steps = [1, 2, 3, 4, 5]
report_dates = ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01"]
producers = ["PROD1", "PROD2"]
injectors = ["INJ1"]
producer_data = ["WOPR", "WBHP", "WWCT"]
injector_data = ["WBHP"]
hidden_size = 20
state_variable = "nn_params"
shut_wells = "zero"
phases = ["oil", "water"]
save_inputs = true
```

Here report step 1 is the first block of `schedule.sch` and falls on 2020-01-01, step 2 the second block on 2020-02-01, and so on. The well names, dates, file name and report steps are placeholders. Replace them with values from your own case. A shorter section, relying on the defaults:

```toml
[model_error]
name = "pg_lstm"
options = { schedule_file = "schedule.sch", report_steps = [1, 2, 3], producers = ["PROD1"] }
```

Put the section in your run configuration (the rest of that file comes from [MEC](../../mec.md)) and start the run with `apex run`. Add `--max-steps` with a number to stop after that many analysis steps, or `--no-progress` to turn the progress display off. This command stops after two analysis steps:

```bash
uv run apex run config.toml --max-steps 2
```

To ask a running run to stop at its next checkpoint, give `apex stop` the run directory, the directory of that run:

```bash
uv run apex stop path/to/run
```

`apex resume` continues a stopped or interrupted run from its last checkpoint. Give it the same run directory. It also accepts `--max-steps` and `--no-progress`:

```bash
uv run apex resume path/to/run
```

After a run, inspect its results with `apex report` and `apex summary`:

```bash
uv run apex report path/to/run --plots
uv run apex summary path/to/run
```

`uv run apex report` prints the data mismatch history and the performance metrics of a run; `--plots` also writes the figures (to `RUN_DIR/report` unless you give `--output`). `uv run apex summary` lists every metric at every step.

## Pitfalls

- **Empty `report_steps`.** It defaults to an empty list, so list the report steps you want the well models to act at. `schedule_file` is needed too, unless your simulator takes its wells from its own input and you give `report_dates`.
- **Empty well lists.** `producers` and `injectors` both default to empty lists, so list the wells you want modelled.
- **Wrong value for `shut_wells`.** Only `"zero"` and `"ignore"` are valid.
- **Non-integer report steps.** `report_steps` takes integers, not dates or strings. They are 1-based.
- **Missing or inconsistent `report_dates`.** `causal_mask` and `save_inputs` need `report_dates`, with one ISO date per report step in order. The report dates of each case are checked against them, so keep the list identical to your cases' dates.
- **`causal_mask` without `adaptive_prior`.** `causal_mask` needs the `adaptive_prior` localization. Without `nn_params` in its `unlocalized` list, the adaptive threshold may switch off nearly all LSTM parameters.
- **Phases that do not match your deck.** The phases set the dynamic features, so list the phases your deck really uses.
- **Typos in option names.** Unknown keys are rejected.
- **A `state_variable` that does not match.** The default is `"nn_params"`. The name must be the state variable your run configuration declares for the LSTM parameters.

## What to read next

- [MEC](../../mec.md), for the method and the rest of the configuration it needs, including the per-step prior.
- The other plugin pages in the guide's plugins section. For example, `sch` is a small test objective in the `objectives` registry, and `run` is a `surrogate_designs` plugin that reuses the forecasts a history-matching run already holds. Both take no options.
- `uv run apex plugins`, for the full list of options and defaults of every plugin.

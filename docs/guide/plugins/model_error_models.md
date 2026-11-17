---
audience: public
items:
  - cli:apex plugins
  - cli:apex report
  - cli:apex run
  - cli:apex summary
  - plugin:model_error_models:pg_lstm
  - plugin:objectives:sch
  - plugin:surrogate_designs:run
---

# Model error models

## Goal

This page shows how to set up the `pg_lstm` plugin from the `model_error_models` registry. `pg_lstm` provides physics-guided LSTM well residual models, one per well. By the end you will have a complete `[model_error]` section for it, with its required options, and you will know what each optional setting controls. [MEC](../../mec.md) describes the method and the rest of the configuration it needs, such as the prior of the state variable that holds the LSTM parameters.

## Prerequisites

- A history schedule file for your model, which you will point `schedule_file` at. It holds one COMPDAT, WCONHIST and WCONINJH block per report step.
- The 1-based report steps of the report times of the well models, which are the observation times.
- The names of the producer and injector wells you want to cover.
- For `causal_mask` or `save_inputs` only: the ISO date of each report step.

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
2. **Set the two required options.**
   - `schedule_file`: the path to your history schedule (a string or a path).
   - `report_steps`: a list of integers. These are 1-based report steps of the report times, which are the observation times.
3. **List the wells.**
   - `producers`: a list of producer well names. Default: `[]`.
   - `injectors`: a list of injector well names. Default: `[]`.
4. **Choose the data used for each well type (optional).** These are the data types of the wells, given as strings.
   - `producer_data`: default `["WOPR", "WBHP", "WWCT"]`.
   - `injector_data`: default `["WBHP"]`.
5. **Tune the LSTM and its state (optional).**
   - `hidden_size`: the hidden size of the LSTM, an integer. A larger value gives each well's model more capacity, and a smaller one gives it less. Default: `20`.
   - `state_variable`: the name of the state variable that holds the parameters of every well's LSTM, producers first and then injectors, each in torch's layout. Default: `"nn_params"`. Change it only if you have a reason to.
6. **Decide how shut wells are treated (optional).** `shut_wells` accepts two values. Default: `"zero"`.
   - `"zero"` follows the manuscript. There is no residual while a well is shut, except for `WBHP`.
   - `"ignore"` gives residuals at every report time, even when the well is shut.
7. **Set the phases (optional).** `phases` is a list of strings naming the phases of the deck, one or more of `oil`, `water` and `gas`. They set the dynamic features, namely each phase's density and saturation. Without oil, its density and saturation are 0. Default: `["oil", "water"]`.
8. **Give an initial schedule (optional).** `initial_schedule_file` is schedule text that is read before `schedule_file`. Use it for the deck's own SCHEDULE keywords that come before its included history, for example a first COMPDAT. Leave it unset (the default, null) if the deck has none.
9. **Give the report dates (optional, but needed for the next two options).** `report_dates` is a list of ISO date strings, one per report step, in order. Default: `[]`. `causal_mask` and `save_inputs` must place a datum at its report step before any run, so they need it. Every case's own report dates are checked against this list.
10. **Declare the causal structure (optional).** `causal_mask` is a boolean. Default: `false`. When `true`, it declares the structure of the well models to the run's localization: a well's LSTM acts on its own outputs at every report time, and the output layer for report time `t` (each output's row and bias) acts on that output at `t` only. It needs `adaptive_prior`. Putting `nn_params` in its `unlocalized` list gives the mask alone.
11. **Save the inputs (optional).** `save_inputs` is a boolean. Default: `false`. When `true`, the plugin also answers the inputs, `MEC:X<j>:<well>` at each report date (feature `j` of the well's inputs), and saves them with the residuals. A run of the prior then gives a per-step prior's sequential pass its inputs.

No other options are accepted: unknown keys are rejected.

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

The well names, dates, file name and report steps are placeholders. Replace them with values from your own case. Only `schedule_file` and `report_steps` are required, so a short section is:

```toml
[model_error]
name = "pg_lstm"
options = { schedule_file = "schedule.sch", report_steps = [1, 2, 3], producers = ["PROD1"] }
```

Put the section in your run configuration and start the run with `apex run`. Add `--max-steps` to stop after a number of analysis steps, and `--no-progress` to turn the progress display off:

```bash
uv run apex run config.toml
```

After a run, inspect its results with `apex report` and `apex summary`:

```bash
uv run apex report path/to/run
uv run apex summary path/to/run
```

`uv run apex report` prints the data mismatch history and the performance metrics of a run. Add `--plots` to also write the figures. `uv run apex summary` lists every metric at every step.

## Pitfalls

- **Missing required options.** Leaving out `schedule_file` or `report_steps` makes the section invalid.
- **Empty well lists.** `producers` and `injectors` both default to empty lists, so list the wells you want modelled.
- **Wrong value for `shut_wells`.** Only `"zero"` and `"ignore"` are valid.
- **Non-integer report steps.** `report_steps` takes integers, not dates or strings. They are 1-based.
- **Missing `report_dates`.** `causal_mask` and `save_inputs` need `report_dates`, with one ISO date per report step in order. Each case's own report dates are checked against them.
- **`causal_mask` without `adaptive_prior`.** `causal_mask` needs `adaptive_prior`.
- **Phases that do not match your deck.** The phases set the dynamic features, so list the phases your deck really uses.
- **Typos in option names.** Unknown keys are rejected.
- **Changing `state_variable` without need.** The default is `"nn_params"`. The state variable you name must be the one that holds the LSTM parameters.

## What to read next

- [MEC](../../mec.md), for the method and the rest of the configuration it needs.
- The other plugin pages in the guide's plugins section. For example, `sch` is a small test objective in the `objectives` registry, and `run` is a `surrogate_designs` plugin that reuses the forecasts a history-matching run already holds. Both take no options.
- `uv run apex plugins`, for the full list of options and defaults of every plugin.

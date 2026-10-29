---
audience: public
items:
  - cli:apex plugins
  - cli:apex report
  - cli:apex summary
  - plugin:model_error_models:pg_lstm
  - plugin:objectives:sch
---

# Model error models

## Goal

This page shows how to set up the `pg_lstm` plugin from the `model_error_models` registry. `pg_lstm` provides physics-guided LSTM well residual models, one per well. By the end you will have a complete `[model_error]` section for it, with its required options, and you will know what each optional setting controls. [MEC](../../mec.md) describes the method and the rest of the configuration it needs, such as the prior of the networks' state variable.

## Prerequisites

- A history schedule file for your model, which you will point `schedule_file` at. It holds one COMPDAT, WCONHIST and WCONINJH block per report step.
- The 1-based report steps of the network's report times, which are the observation times.
- The names of the producer and injector wells you want to cover.

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
   - `report_steps`: a list of integers. These are 1-based report steps of the network's report times, which are the observation times.
3. **List the wells.**
   - `producers`: a list of producer well names. Default: `[]`.
   - `injectors`: a list of injector well names. Default: `[]`.
4. **Choose the data used for each well type (optional).** These are the data types of the wells, given as strings.
   - `producer_data`: default `["WOPR", "WBHP", "WWCT"]`.
   - `injector_data`: default `["WBHP"]`.
5. **Tune the network and its state (optional).**
   - `hidden_size`: the hidden size of the LSTM, an integer. A larger value gives each well's network more capacity, and a smaller one gives it less. Default: `20`.
   - `state_variable`: the name of the state variable that holds the parameters of every well's network, producers first and then injectors, each in torch's layout. Default: `"nn_params"`. Change it only if you have a reason to.
6. **Decide how shut wells are treated (optional).** `shut_wells` accepts two values. Default: `"zero"`.
   - `"zero"` follows the manuscript. There is no residual while a well is shut, except for `WBHP`.
   - `"ignore"` gives residuals at every report time, even when the well is shut.
7. **Set the phases (optional).** `phases` is a list of strings naming the phases of the deck. Oil is required. The phases set the dynamic features, namely each phase's density and saturation. Default: `["oil", "water"]`.
8. **Give an initial schedule (optional).** `initial_schedule_file` is schedule text that is read before `schedule_file`. Use it for the deck's own SCHEDULE keywords that come before its included history, for example a first COMPDAT. Leave it unset (the default, null) if the deck has none.

No other options are accepted: unknown keys are rejected.

## Worked example

This section configures `pg_lstm` for two producers and one injector. It keeps the default data lists and sets the network size and the shut-well handling explicitly.

```toml
[model_error]
name = "pg_lstm"

[model_error.options]
schedule_file = "schedule.sch"
report_steps = [1, 2, 3, 4, 5]
producers = ["PROD1", "PROD2"]
injectors = ["INJ1"]
producer_data = ["WOPR", "WBHP", "WWCT"]
injector_data = ["WBHP"]
hidden_size = 20
state_variable = "nn_params"
shut_wells = "zero"
phases = ["oil", "water"]
```

The well names, file name and report steps are placeholders. Replace them with values from your own case. Only `schedule_file` and `report_steps` are required, so a short section is:

```toml
[model_error]
name = "pg_lstm"
options = { schedule_file = "schedule.sch", report_steps = [1, 2, 3], producers = ["PROD1"] }
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
- **Phases that do not match your deck.** Oil is required in `phases`. The phases set the dynamic features, so list the phases your deck really uses.
- **Typos in option names.** Unknown keys are rejected.
- **Changing `state_variable` without need.** The default is `"nn_params"`. The state variable you name must be the one that holds the networks' parameters.

## What to read next

- [MEC](../../mec.md), for the method and the rest of the configuration it needs.
- The other plugin pages in the guide's plugins section. For example, `sch` is a small test objective in the `objectives` registry, and its page shows a plugin that takes no options.
- `uv run apex plugins`, for the full list of options and defaults of every plugin.

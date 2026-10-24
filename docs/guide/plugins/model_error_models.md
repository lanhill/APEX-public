---
audience: public
items:
  - plugin:model_error_models:pg_lstm
---

# Model error models

## Goal

This page shows how to set up the `pg_lstm` plugin from the `model_error_models` registry. `pg_lstm` provides physics-guided LSTM well residual models, one per well. By the end you will have a complete `[model_error]` section for it, with its required options, and know what each optional setting controls. [MEC](../../mec.md) describes the method and the rest of the configuration it needs (the networks' state variable, its prior, the deck's restarts and summary).

## Prerequisites

- A schedule file for your model, which you will point `schedule_file` at.
- The list of report steps (integers) the model error model should work on.
- The names of the producer and injector wells you want it to cover. The deck's SUMMARY section must request `WBHP` for each of them, observed or not.

## Steps

1. **Add the section.** Create a `[model_error]` section with `name = "pg_lstm"`, and its options in `[model_error.options]`.
2. **Set the two required options.**
   - `schedule_file`: the path to your schedule file (a string or path).
   - `report_steps`: a list of integers naming the report steps.
3. **List the wells.**
   - `producers`: a list of producer well names. Default: `[]`.
   - `injectors`: a list of injector well names. Default: `[]`.
4. **Choose the data used for each well type (optional).**
   - `producer_data`: a list of strings. Default: `["WOPR", "WBHP", "WWCT"]`.
   - `injector_data`: a list of strings. Default: `["WBHP"]`.
5. **Tune the network and its state (optional).**
   - `hidden_size`: an integer. Default: `20`.
   - `state_variable`: a string. Default: `"nn_params"`.
6. **Decide how shut wells are treated (optional).** `shut_wells` accepts `"zero"` or `"ignore"`. Default: `"zero"`.
7. **Set the phases (optional).** `phases` is a list of strings. Default: `["oil", "water"]`.
8. **Give an initial schedule (optional).** `initial_schedule_file` is a path or string. Default: unset (null).

At least one producer or injector is required. No other options are accepted: unknown keys are rejected.

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

The well names, file name and report steps are placeholders. Replace them with values from your own case. Only `schedule_file`, `report_steps` and one well are required, so a short valid section is:

```toml
[model_error]
name = "pg_lstm"
options = { schedule_file = "schedule.sch", report_steps = [1, 2, 3], producers = ["PROD1"] }
```

## Pitfalls

- **Missing required options.** Leaving out `schedule_file` or `report_steps` makes the section invalid.
- **Empty well lists.** `producers` and `injectors` both default to empty lists, and at least one well is required. List the wells you want modelled.
- **No WBHP in the summary.** Every listed well's `WBHP` is a network input; `apex validate` refuses a deck whose SUMMARY section lacks it.
- **Wrong value for `shut_wells`.** Only `"zero"` and `"ignore"` are valid.
- **Non-integer report steps.** `report_steps` takes integers, not dates or strings.
- **Typos in option names.** Unknown keys are not accepted.
- **Changing `state_variable` without need.** The default is `"nn_params"`. Change it only if you have a reason to.

## What to read next

- The other plugin pages in the guide's plugins section, to see how settings in one registry relate to the rest of your configuration.
- The reference entry for `pg_lstm`, for the full list of options and defaults, and [MEC](../../mec.md).

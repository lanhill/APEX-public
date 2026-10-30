---
audience: public
items:
  - cli:apex optimize
  - cli:apex plugins
  - cli:apex resume
  - cli:apex run
  - cli:apex stop
  - config:optimization:objectives
  - plugin:optimizers:nsga2
  - plugin:response_objectives:ecalc_emissions
  - plugin:response_objectives:ecalc_npv
  - plugin:response_objectives:ecalc_pareto_npv
  - plugin:response_objectives:field_total
  - plugin:response_objectives:npv
  - plugin:response_objectives:ren_npv
  - plugin:response_objectives:ren_npv_co2
---

# Response objectives

## Goal

A response objective is the number an optimization tries to improve, computed from the simulation's results. This page shows how to choose one of the seven `response_objectives` plugins, set its options, combine several of them, and avoid common mistakes.

| Plugin | What it computes | Status |
|---|---|---|
| `npv` | NPV of the field's oil and gas production, water production and water injection. | parity |
| `ren_npv` | NPV with water injection priced per well: by grid or by renewable power. | parity |
| `ecalc_npv` | NPV of oil, gas, water and the CO2 emitted, with the emissions computed by eCalc. | parity |
| `ecalc_pareto_npv` | NPV with eCalc's CO2, combined with an emission objective. | parity |
| `ren_npv_co2` | NPV with eCalc's CO2, where wind power and injection pressure enter eCalc. | parity |
| `ecalc_emissions` | The emission eCalc computes, in tonnes over the steps: a second objective. | validated |
| `field_total` | A field total at one time, scaled: for example the water injected. | validated |

Here "the steps" are the periods between the dates you list in `times`.

To see every plugin with its options, run `uv run apex plugins`. Add `--json` for full descriptions as JSON.

## Prerequisites

- An optimization configuration file in TOML, run with `uv run apex optimize`. (`apex run` starts a history-matching run and is not used here.)
- A simulator set up in the same configuration, because a response objective is computed from simulation results. Take it from your existing configuration.
- The dates at which the objective is evaluated, as ISO dates such as `2025-01-01`. For `field_total`, also a cumulative summary vector that your simulation reports.
- For the eCalc-based plugins (`ecalc_npv`, `ecalc_pareto_npv`, `ren_npv_co2`, `ecalc_emissions`): an eCalc model file (YAML), given through `ecalc_model_file`.
- For `ren_npv_co2`: a wind power file, given through `wind_power_file` (see below).

## Steps

1. **Pick the plugin.** Use `npv` for a plain economic objective. Use `ren_npv` if water injection is priced per well. Use an eCalc plugin if CO2 should be part of the result. Use `field_total` for a single scaled total, and `ecalc_emissions` when the emission itself should be an objective.
2. **Declare it in the optimization configuration.** For one objective use `[objective]` with `name` and its options in `[objective.options]`. For several objectives at once use the `optimization:objectives` list: one `[[objectives]]` entry each, with a plugin `name`, its `options` and an optional `label`. Response objectives in that list share one simulation per member, so adding one does not multiply the simulation cost.
3. **Choose the optimizer.** Several objectives are optimized together by a multi-objective optimizer such as `nsga2` (NSGA-II, status experimental). Its options include `pop_size` (default `40`), `maxiter` (default `100`, generations at most) and `stall` (default `30`). The Optimization page shows where the optimizer is configured.
4. **Set the required options** below. Unknown options are rejected, so use only the names shown here.
5. **Set the optional options** you need. Anything you leave out takes its default.
6. **Run it:** `uv run apex optimize config.toml`. Use `--max-iterations N` to stop after N accepted iterations (the run can be resumed), and `--resume DIR` to continue the optimization run in directory `DIR`. Progress is shown by default; `--no-progress` turns it off.

### Options shared by the NPV plugins

`npv`, `ren_npv`, `ecalc_npv`, `ecalc_pareto_npv` and `ren_npv_co2` take:

- `start` (string, required): the simulation's start date, in ISO format (`2025-01-01`). It must be the start date of the simulation deck. The eCalc-based plugins check it against the summary's time, and discounting counts the days from it.
- `times` (list of strings, required): the report dates in ISO format, increasing and after `start`. Each is the end of one step, and the simulator must report at each.
- `wop` (number, required): the oil price per unit of `FOPT`.
- `wgp` (number, default `0.0`): the gas price per unit of `FGPT`. The default leaves gas out of the value.
- `wwp` and `wwi` (numbers, default `0.0`): the cost per unit of water produced (`FWPT`) and injected (`FWIT`). The default makes them free.
- `disc` (number, default `0.0`): the discount rate per year, for example `0.08` for 8 %. The default does not discount.
- `obj_scaling` (number, default `1.0`): divides the NPV. Optimizers minimise, so a negative value maximises NPV. With `obj_scaling = -1.0e6` the objective is the NPV divided by minus one million, so minimising it maximises the NPV, and the value is in millions. For example an NPV of 250 000 000 gives an objective of `-250`.

### Per-plugin options

- `npv`: adds `from_time` (string or none, default none). It must be one of `times`. NPV then counts only the steps after it, with the discount counted from it. Use it for a forecast that follows a history, whose production is not part of the decision. By default NPV counts from `start`.
- `ren_npv`: adds `from_time` (as for `npv`), `injectors`, `renewable_injectors`, `wwi` and `wrenwi`. The water injected (`WWIT`) by the wells in `injectors` costs `wwi` per unit. The water injected by the wells in `renewable_injectors` (powered by renewable energy) costs `wrenwi` per unit. The two lists default to empty and `wrenwi` defaults to `0.0`.
- `ecalc_npv`: adds `ecalc_model_file` and `wem` (both required), `time_series` (default none) and `emission` (default `"co2"`). The model's `START` and `END` must be `start` and the last of `times`, and its time series reads `OIL_PROD`, `GAS_PROD` and `WATER_INJ` (per day). `wem` is the cost per tonne of CO2. `time_series` names the model's time-series file that the simulations replace; by default the model's only one. `emission` is the emission priced at `wem`, by its name in the model.
- `ecalc_pareto_npv`: takes the options of `ecalc_npv`, plus `wemc` and `w` (numbers or none, default none). Set one of them. `wemc` is the CO2 cost correction per tonne. `w` is the weight of the emission intensity, in `[0, 1]`. Here `wem` is the cost per tonne of the emission eCalc computes, and `time_series` is required when the model has several.
- `ren_npv_co2`: requires `ecalc_model_file`, `injectors`, `wind_power_file` and `wem`, and takes the optional `time_series` and `emission`. `injectors` are the injection wells whose `WTHP` gives `THP_MAX`. `wind_power_file` is a `.npy` file holding the wind power available in each report step, as an array of shape `(n_steps,)`.
- `ecalc_emissions`: requires `start`, `times` and `ecalc_model_file`, used as for `ecalc_npv`. Optional: `disc` (default `0.0`), `obj_scaling` (default `1.0`), `time_series` and `emission` (default `"co2"`). It has no price weights. `obj_scaling` divides the tonnes (for example `1000` for kilotonnes) and must be positive, because emissions are minimised.
- `field_total`: requires `vector` (a cumulative total such as `FWIT`, `FOPT` or `FGPT`) and `time` (an ISO date). Optional: `scale` (number, default `1.0`), which multiplies the total, for example `1e-6` for millions.

## Worked example

A plain NPV with `npv`, valuing oil and charging for water. `obj_scaling = -1.0e6` makes the optimizer maximise the NPV, in millions:

```toml
[objective]
name = "npv"

[objective.options]
start = "2025-01-01"
times = ["2026-01-01", "2027-01-01", "2028-01-01"]
wop = 500.0
wwp = 50.0
wwi = 10.0
disc = 0.08
obj_scaling = -1.0e6
```

To count CO2 too, switch to `ecalc_npv`:

```toml
[objective]
name = "ecalc_npv"

[objective.options]
start = "2025-01-01"
times = ["2026-01-01", "2027-01-01", "2028-01-01"]
ecalc_model_file = "model/ecalc_model.yaml"
wop = 500.0
wem = 60.0
disc = 0.08
obj_scaling = -1.0e6
emission = "co2"
```

For `ecalc_pareto_npv`, set `w` (or `wemc` instead):

```toml
[objective]
name = "ecalc_pareto_npv"

[objective.options]
start = "2025-01-01"
times = ["2026-01-01", "2027-01-01", "2028-01-01"]
ecalc_model_file = "model/ecalc_model.yaml"
wop = 500.0
wem = 60.0
w = 0.5
obj_scaling = -1.0e6
```

For `ren_npv`, put the grid-powered wells in `injectors` and the renewable-powered wells in `renewable_injectors`. The well names are examples; use the wells of your model:

```toml
[objective]
name = "ren_npv"

[objective.options]
start = "2025-01-01"
times = ["2026-01-01", "2027-01-01", "2028-01-01"]
wop = 500.0
wwp = 50.0
wwi = 10.0
wrenwi = 2.0
injectors = ["INJ1"]
renewable_injectors = ["INJ2"]
disc = 0.08
obj_scaling = -1.0e6
```

For `ren_npv_co2`, give the injectors and the wind power file:

```toml
[objective]
name = "ren_npv_co2"

[objective.options]
start = "2025-01-01"
times = ["2026-01-01", "2027-01-01", "2028-01-01"]
ecalc_model_file = "model/ecalc_model.yaml"
injectors = ["INJ1", "INJ2"]
wind_power_file = "model/wind_power.npy"
wop = 500.0
wem = 60.0
obj_scaling = -1.0e6
```

To count only a forecast after a history, add `from_time = "2026-01-01"` to the options of `npv` (it is one of `times`).

Finally, several objectives together: NPV and the emission as two objectives, plus a `field_total`. Pair this list with a multi-objective optimizer such as `nsga2`, configured as described on the Optimization page:

```toml
[[objectives]]
name = "npv"
label = "NPV"
options = { start = "2025-01-01", times = ["2026-01-01", "2027-01-01", "2028-01-01"], wop = 500.0, wwp = 50.0, wwi = 10.0, disc = 0.08, obj_scaling = -1.0e6 }

[[objectives]]
name = "ecalc_emissions"
label = "Emissions"
options = { start = "2025-01-01", times = ["2026-01-01", "2027-01-01", "2028-01-01"], ecalc_model_file = "model/ecalc_model.yaml", obj_scaling = 1000.0 }

[[objectives]]
name = "field_total"
label = "Water injected"
options = { vector = "FWIT", time = "2028-01-01", scale = 1.0e-6 }
```

Then run `uv run apex optimize config.toml`, and later `uv run apex optimize config.toml --resume DIR` to continue a run.

## Pitfalls

- **Missing required options.** `wem` is required for `ecalc_npv`, `ecalc_pareto_npv` and `ren_npv_co2`. `ren_npv_co2` also requires `injectors` and `wind_power_file`.
- **Misspelled option names.** Every plugin rejects options it does not define. A typo is an error, not an ignored line.
- **Two injector lists in `ren_npv`.** `injectors` and `renewable_injectors` are separate lists with their own weights (`wwi` and `wrenwi`). List each well in the one that matches how it is powered.
- **Both `wemc` and `w`.** Set one of them for `ecalc_pareto_npv`; do not leave the choice implicit.
- **Dates that do not match.** `start` must be the simulation's start date, `times` must be increasing and after it, and the simulator must report at each. The eCalc model's `START` and `END` must be `start` and the last of `times`.
- **Forgotten defaults.** `wgp`, `wwp` and `wwi` default to `0.0`, so a term you do not set does not contribute. `emission` defaults to `"co2"`.
- **Mixing up similar plugins.** `ecalc_emissions` returns the emission itself, while `ecalc_npv` prices it in an NPV through `wem`. `field_total` is not an NPV and has no discount.
- **Several objectives with a single-objective optimizer.** A `[[objectives]]` list is meant for an optimizer such as `nsga2` that handles several objectives together.
- **Direction of `obj_scaling`.** It divides the NPV, so a negative value maximises it. For `ecalc_emissions` it must stay positive.
- **Maturity.** `ecalc_emissions` and `field_total` are marked `validated`; the other five are marked `parity`. `nsga2` is marked `experimental`.

## What to read next

- The Optimization page, for the rest of an optimization configuration and where the optimizer is set.
- The eCalc documentation, for how to prepare the model file used by `ecalc_model_file`.
- `uv run apex plugins`, or the reference entry for each plugin, for the full list of options.

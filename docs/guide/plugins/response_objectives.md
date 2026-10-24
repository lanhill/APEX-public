---
audience: public
items:
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

A response objective defines the single number (or, for some plugins, the pair of numbers) that an optimization is trying to improve. This page shows how to choose one of the seven `response_objectives` plugins, set its options, and avoid the usual mistakes.

The plugins are:

| Plugin | What it computes |
|---|---|
| `npv` | NPV of the field's oil and gas production, water production and water injection. |
| `ren_npv` | NPV with water injection priced per well: by grid or by renewable power. |
| `ecalc_npv` | NPV of oil, gas, water and the CO2 emitted, by eCalc. |
| `ecalc_pareto_npv` | NPV with eCalc's CO2, combined with an emission objective (two variants). |
| `ren_npv_co2` | NPV with eCalc's CO2, where wind power and injection pressure enter eCalc. |
| `ecalc_emissions` | The emission eCalc computes, in tonnes over the steps: a second objective. |
| `field_total` | A field total at one time, scaled: for example the water injected. |

## Prerequisites

- A working APEX configuration file in TOML.
- The simulation vectors and time points you want the objective evaluated over.
- For the eCalc-based plugins (`ecalc_npv`, `ecalc_pareto_npv`, `ren_npv_co2`, `ecalc_emissions`): an eCalc model file, given through `ecalc_model_file`, and Java. eCalc (Equinor's `libecalc`) is in APEX's default environment (`uv sync --locked`; the `ecalc` extra when APEX is a dependency of your own project), and it brings a Java-based thermodynamics library, so Java must be installed.
- For `ren_npv_co2`: a wind power file, given through `wind_power_file`.

## Steps

1. **Pick the plugin.** Use `npv` for a plain economic objective. Use `ren_npv` if injection should be priced per well. Use an eCalc plugin if CO2 should be part of the result. Use `field_total` for a single scaled total, and `ecalc_emissions` when the emission itself should be an objective.
2. **Choose it in the optimization configuration**: `[objective]` with `name = "npv"` (for example) and its options in `[objective.options]`; for several objectives at once, one `[[objectives]]` entry each. A response objective also needs the run's `[simulator]`.
3. **Set the required options** (listed below for each plugin). Unknown options are rejected, so use only the names shown here.
4. **Set the optional options** you need. Anything you leave out takes its default.

### Options shared by the NPV plugins

`npv`, `ren_npv`, `ecalc_npv`, `ecalc_pareto_npv` and `ren_npv_co2` all take:

- `start` (string, required) and `times` (list of strings, required): the start and the time points of the evaluation.
- `wop` (number, required): the oil price, per unit of FOPT.
- `wgp` (number, default `0.0`): the gas price, per unit of FGPT.
- `wwp`, `wwi` (numbers, default `0.0`): the costs of produced and injected water, per unit of FWPT and FWIT. They are costs: NPV subtracts them, so give them as positive numbers (a negative cost would count water as revenue).
- `disc` (number, default `0.0`): the yearly discount rate.
- `obj_scaling` (number, default `1.0`): NPV is divided by it. Optimizers minimise, so give it a negative value to maximise NPV, for example `-1.0e6` for -NPV in millions. With the default `1.0` the optimization minimises NPV.

### Per-plugin options

- `npv`: adds `from_time` (string or none, default none).
- `ren_npv`: adds `from_time` (default none), `wrenwi` (number, default `0.0`), `injectors` (list of strings, default empty) and `renewable_injectors` (list of strings, default empty).
- `ecalc_npv`: requires `ecalc_model_file` and `wem` (number: the cost per unit of the emission, subtracted like the water costs). Optional: `time_series` (string or none, default none) and `emission` (string, default `"co2"`).
- `ecalc_pareto_npv`: the same as `ecalc_npv`, plus `wemc` (number or none, default none) and `w` (number or none, default none).
- `ren_npv_co2`: requires `ecalc_model_file`, `injectors` (list of strings), `wind_power_file` and `wem`. Optional: `time_series` and `emission`, as for `ecalc_npv`.
- `ecalc_emissions`: requires `start`, `times` and `ecalc_model_file`. Optional: `disc` (default `0.0`), `obj_scaling` (default `1.0`), `time_series` (default none) and `emission` (default `"co2"`). It has no price coefficients.
- `field_total`: requires `vector` (string) and `time` (string). Optional: `scale` (number, default `1.0`).

## Worked example

First, a plain NPV with `npv`, valuing oil and charging for water production and injection. Prices and costs are positive, and `obj_scaling = -1.0e6` makes the optimizer maximise NPV, in millions:

```toml
[objective]
name = "npv"

[objective.options]
start = "2025-01-01"
times = ["2026-01-01", "2027-01-01", "2028-01-01"]
wop = 500.0
wgp = 0.0
wwp = 50.0
wwi = 10.0
disc = 0.08
obj_scaling = -1.0e6
```

The objective is built from the `times` you list, starting at `start`, which must be the simulation's start (the deck's `START`). `wgp` is spelled out here for clarity, although `0.0` is its default. The example `optimization/3well` (PET's tutorial) prices the same way.

To count CO2 as well, switch to `ecalc_npv`. It needs the eCalc model file and `wem`:

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

Finally, `field_total` gives one scaled total at one time, typically as one of several objectives, here beside the NPV (as optimized by `nsga2`). The vector name below is only an example; use a vector that exists in your simulation:

```toml
[[objectives]]
name = "npv"
options = { start = "2025-01-01", times = ["2026-01-01", "2027-01-01", "2028-01-01"], wop = 500.0, wwp = 50.0, wwi = 10.0, disc = 0.08, obj_scaling = -1.0e6 }

[[objectives]]
name = "field_total"
options = { vector = "FWIT", time = "2028-01-01", scale = 1.0 }
```

## Pitfalls

- **Missing required options.** `wem` is required for `ecalc_npv`, `ecalc_pareto_npv` and `ren_npv_co2`, even if you only care about the oil term. `ren_npv_co2` also requires `injectors` and `wind_power_file`.
- **Misspelled option names.** Every plugin rejects options it does not define. A typo is an error, not an ignored line.
- **Two injector lists in `ren_npv`.** `injectors` and `renewable_injectors` are separate lists, both empty by default. Wells you want priced as renewable-powered belong in `renewable_injectors`, with `wrenwi` set. The two options are not aliases.
- **Forgotten defaults.** The coefficients `wgp`, `wwp` and `wwi` default to `0.0`, so a term you do not set does not contribute. `emission` defaults to `"co2"`.
- **Mixing up similar plugins.** `ecalc_emissions` returns the emission itself, while `ecalc_npv` folds the emission into an NPV through `wem`. `field_total` is not an NPV and has no discount.
- **The pareto variants.** `ecalc_pareto_npv` needs exactly one of `wemc` (a CO2 cost correction per tonne) and `w` (a weight); both default to none, and giving neither or both is an error.
- **Signs.** Prices, costs and `wem` are positive; NPV subtracts the costs. Optimizers minimise, so a negative `obj_scaling` maximises NPV.
- **Maturity.** `ecalc_emissions` and `field_total` are marked `validated`; the other five are marked `parity`.

## What to read next

- [Optimization](../../optimization.md), for the rest of an optimization configuration.
- The eCalc documentation, for how to prepare the model file used by `ecalc_model_file`.
- The reference entry for each plugin, for the full list of options.

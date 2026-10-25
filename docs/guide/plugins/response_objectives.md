---
audience: public
items:
  - cli:apex optimize
  - cli:apex plugins
  - cli:apex run
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

A response objective defines the number an optimization tries to improve, computed from the simulation's results. This page shows how to choose one of the seven `response_objectives` plugins, set its options, combine several of them, and avoid common mistakes.

| Plugin | What it computes | Status |
|---|---|---|
| `npv` | NPV of the field's oil and gas production, water production and water injection. | parity |
| `ren_npv` | NPV with water injection priced per well: by grid or by renewable power. | parity |
| `ecalc_npv` | NPV of oil, gas, water and the CO2 emitted, with the emissions computed by eCalc. | parity |
| `ecalc_pareto_npv` | NPV with eCalc's CO2, combined with an emission objective (two variants). | parity |
| `ren_npv_co2` | NPV with eCalc's CO2, where wind power and injection pressure enter eCalc. | parity |
| `ecalc_emissions` | The emission eCalc computes, in tonnes over the steps: a second objective. | validated |
| `field_total` | A field total at one time, scaled: for example the water injected. | validated |

Here "the steps" are the evaluation time points you list in `times`.

To see every plugin with its options, run `apex plugins`. Add `--json` for full descriptions as JSON.

## Prerequisites

- An optimization configuration file in TOML. You run it with `apex optimize`. (`apex run` is for history matching and is not used here.)
- A `[simulator]` section in the same configuration. A response objective is evaluated from simulation results, so the optimization needs the simulator that produces them. Copy it from your existing run configuration or see the simulator documentation.
- The time points you want the objective evaluated at, and, for `field_total`, a vector name that exists in your simulation.
- For the eCalc-based plugins (`ecalc_npv`, `ecalc_pareto_npv`, `ren_npv_co2`, `ecalc_emissions`): an eCalc model file, given through `ecalc_model_file`. eCalc brings a Java-based library, so Java must be installed.
- For `ren_npv_co2`: a wind power file, given through `wind_power_file`.

## Steps

1. **Pick the plugin.** Use `npv` for a plain economic objective. Use `ren_npv` if injection should be priced per well. Use an eCalc plugin if CO2 should be part of the result. Use `field_total` for a single scaled total, and `ecalc_emissions` when the emission itself should be an objective.
2. **Declare it in the optimization configuration.** For one objective use `[objective]` with `name = "npv"` (for example) and its options in `[objective.options]`. For several objectives at once, use one `[[objectives]]` entry each (the `optimization:objectives` list). Each entry is a plugin with its options and an optional `label`. Response objectives in the list share one simulation per ensemble member, so adding one does not multiply the simulation cost.
3. **Choose a suitable optimizer.** Several objectives are optimized together by a multi-objective optimizer such as `nsga2` (NSGA-II, status experimental). Its options include `pop_size` (default `40`) and `maxiter` (default `100`). See the Optimization page for where the optimizer is configured.
4. **Set the required options** listed below. Unknown options are rejected, so use only the names shown here.
5. **Set the optional options** you need. Anything you leave out takes its default.
6. **Run it:** `apex optimize config.toml`. Use `--max-iterations N` to stop after N accepted iterations, and `--resume DIR` to continue the run in that directory. Progress is shown by default; `--no-progress` turns it off.

### Options shared by the NPV plugins

`npv`, `ren_npv`, `ecalc_npv`, `ecalc_pareto_npv` and `ren_npv_co2` all take:

- `start` (string, required) and `times` (list of strings, required): the start and the time points of the evaluation.
- `wop` (number, required): the weight for oil production.
- `wgp` (number, default `0.0`): the weight for gas production.
- `wwp`, `wwi` (numbers, default `0.0`): the weights for water production and water injection. Treat them as costs and give positive numbers; a negative value would count water as revenue.
- `disc` (number, default `0.0`): the discount rate.
- `obj_scaling` (number, default `1.0`): a scaling of the objective. Optimizers minimise, so a negative value such as `-1.0e6` is the usual way to maximise NPV, in millions. With the default `1.0` the optimization minimises NPV.

### Per-plugin options

- `npv`: adds `from_time` (string or none, default none), an optional time point.
- `ren_npv`: adds `from_time` (default none), `wrenwi` (number, default `0.0`), `injectors` (list of strings, default empty) and `renewable_injectors` (list of strings, default empty). `wrenwi` is the injection weight for the renewable-powered wells, next to `wwi`.
- `ecalc_npv`: requires `ecalc_model_file` and `wem` (number: the weight for the emission). Optional: `time_series` (string or none, default none) and `emission` (string, default `"co2"`).
- `ecalc_pareto_npv`: takes the same options as `ecalc_npv`, plus `wemc` (number or none, default none) and `w` (number or none, default none). The two options select between PET's two variants of combining NPV with an emission objective. Set one of them; the plugin reference describes each variant.
- `ren_npv_co2`: requires `ecalc_model_file`, `injectors` (list of strings, the injection wells), `wind_power_file` and `wem`. Optional: `time_series` and `emission`, as for `ecalc_npv`.
- `ecalc_emissions`: requires `start`, `times` and `ecalc_model_file`. Optional: `disc` (default `0.0`), `obj_scaling` (default `1.0`), `time_series` (default none) and `emission` (default `"co2"`). It has no price weights.
- `field_total`: requires `vector` (string) and `time` (string). Optional: `scale` (number, default `1.0`).

The facts for this page do not describe the contents of `time_series` or `wind_power_file`, nor the valid values of `emission` other than the default `"co2"`. Check the eCalc documentation and the plugin reference before changing them.

## Worked example

First, a plain NPV with `npv`, valuing oil and charging for water production and injection. `obj_scaling = -1.0e6` makes the optimizer maximise NPV, in millions:

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

The objective is built from the `times` you list, starting at `start`, which should be the simulation's start. `wgp` is spelled out for clarity, although `0.0` is its default.

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

For `ren_npv`, list the injection wells in `injectors` and the wells you want priced as renewable-powered in `renewable_injectors`, with `wrenwi` set. The well names below are examples; use the wells of your model:

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

Finally, several objectives together: NPV and the emission itself as two objectives, with `ecalc_emissions` as the second. Use a multi-objective optimizer such as `nsga2`:

```toml
[[objectives]]
name = "npv"
label = "NPV"
options = { start = "2025-01-01", times = ["2026-01-01", "2027-01-01", "2028-01-01"], wop = 500.0, wwp = 50.0, wwi = 10.0, disc = 0.08, obj_scaling = -1.0e6 }

[[objectives]]
name = "ecalc_emissions"
label = "Emissions"
options = { start = "2025-01-01", times = ["2026-01-01", "2027-01-01", "2028-01-01"], ecalc_model_file = "model/ecalc_model.yaml" }
```

A `field_total` entry can be added in the same way, for example `options = { vector = "FWIT", time = "2028-01-01", scale = 1.0 }`. Use a vector that exists in your simulation.

## Pitfalls

- **Missing required options.** `wem` is required for `ecalc_npv`, `ecalc_pareto_npv` and `ren_npv_co2`, even if you only care about the oil term. `ren_npv_co2` also requires `injectors` and `wind_power_file`.
- **Misspelled option names.** Every plugin rejects options it does not define. A typo is an error, not an ignored line.
- **Two injector lists in `ren_npv`.** `injectors` and `renewable_injectors` are separate lists, both empty by default. They are not aliases.
- **Forgotten defaults.** `wgp`, `wwp` and `wwi` default to `0.0`, so a term you do not set does not contribute. `emission` defaults to `"co2"`.
- **Mixing up similar plugins.** `ecalc_emissions` returns the emission itself, while `ecalc_npv` folds the emission into an NPV through `wem`. `field_total` is not an NPV and has no discount.
- **The pareto variants.** `ecalc_pareto_npv` has two optional options, `wemc` and `w`, both none by default. Decide which variant you want and set the matching one; do not leave the choice implicit.
- **Several objectives with a single-objective optimizer.** A list of `[[objectives]]` is meant for an optimizer such as `nsga2` that handles several objectives together.
- **Signs.** Give prices, costs and `wem` as positive numbers. Optimizers minimise, so a negative `obj_scaling` maximises NPV.
- **Maturity.** `ecalc_emissions` and `field_total` are marked `validated`; the other five are marked `parity`. `nsga2` is marked `experimental`.

## What to read next

- The Optimization page, for the rest of an optimization configuration and where the optimizer is set.
- The eCalc documentation, for how to prepare the model file used by `ecalc_model_file`.
- The reference entry for each plugin, or `apex plugins`, for the full list of options.

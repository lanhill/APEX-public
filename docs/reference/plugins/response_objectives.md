---
audience: public
generated: reference
---

# Response objectives: reference

Every one of these 7 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `ecalc_emissions`

The emission eCalc computes, in tonnes over the steps: a second objective.

Select it with:

```toml
[objective]
name = "ecalc_emissions"
```

| option | type | default | description |
|---|---|---|---|
| `disc` | number | 0.0 | Discount rate per year (default 0: plain tonnes). |
| `ecalc_model_file` | string | required | As `ecalc_npv`. |
| `emission` | string | co2 | As `ecalc_npv`. |
| `obj_scaling` | number | 1.0 | Divides the tonnes (e.g. 1000 for kilotonnes); positive, as emissions are minimised. |
| `start` | string | required | As `ecalc_npv`. |
| `time_series` | string or null | | As `ecalc_npv`. |
| `times` | array | required | As `ecalc_npv`. |

## `ecalc_npv`

NPV of oil, gas, water and the CO2 emitted, the emissions computed by eCalc.

Select it with:

```toml
[objective]
name = "ecalc_npv"
```

| option | type | default | description |
|---|---|---|---|
| `disc` | number | 0.0 | As `npv`. |
| `ecalc_model_file` | string | required | The eCalc model (YAML), whose START and END are `start` and the last time and whose time series reads `OIL_PROD`, `GAS_PROD` and `WATER_INJ` (per day). |
| `emission` | string | co2 | The emission priced at `wem` (its name in the model; default `co2`). |
| `obj_scaling` | number | 1.0 | As `npv`. |
| `start` | string | required | The simulation's start and the report dates (ISO), as `npv`. |
| `time_series` | string or null | | The model's time-series file the simulations replace (default: its only one). |
| `times` | array | required | The simulation's start and the report dates (ISO), as `npv`. |
| `wem` | number | required | Cost per tonne of CO2. |
| `wgp` | number | 0.0 | Prices and costs per unit of FOPT, FGPT, FWPT and FWIT. |
| `wop` | number | required | Prices and costs per unit of FOPT, FGPT, FWPT and FWIT. |
| `wwi` | number | 0.0 | Prices and costs per unit of FOPT, FGPT, FWPT and FWIT. |
| `wwp` | number | 0.0 | Prices and costs per unit of FOPT, FGPT, FWPT and FWIT. |

## `ecalc_pareto_npv`

NPV with eCalc's CO2, combined with an emission objective (PET's two variants).

Select it with:

```toml
[objective]
name = "ecalc_pareto_npv"
```

| option | type | default | description |
|---|---|---|---|
| `disc` | number | 0.0 | Discount rate per year (e.g. 0.08 for 8 %), applied by the days since `start`; must be above -1, and the default 0 does not discount. |
| `ecalc_model_file` | string | required | The eCalc model (YAML), whose START and END are `start` and the last of `times` and whose time series reads the step rates (per day). |
| `emission` | string | co2 | The emission priced at `wem`, by its name in the model's fuels and emitters (case-insensitive; default `co2`, PET's cases used `co2_fuel_gas`). |
| `obj_scaling` | number | 1.0 | Divides the NPV (e.g. 1e6 for millions); any nonzero number, default 1. |
| `start` | string | required | The simulation's start date (ISO, e.g. `2025-01-01`): the deck's START, checked against the summary's TIME; discounting counts the days from it. |
| `time_series` | string or null | | The name of the model's time series that the simulated rates replace; by default the model's only one (required when it has several). |
| `times` | array | required | The report dates (ISO), increasing and after `start`; each is the end of one step whose rates eCalc gets. |
| `w` | number or null | | The weight of the emission intensity, in `[0, 1]`. |
| `wem` | number | required | Cost per tonne of the emission eCalc computes. |
| `wemc` | number or null | | The CO2 cost correction per tonne (one of `wemc` and `w`). |
| `wgp` | number | 0.0 | Gas price per unit of FGPT; default 0 leaves gas out of the value. |
| `wop` | number | required | Oil price per unit of FOPT (PET's `wop`). |
| `wwi` | number | 0.0 | Cost per unit of water injected (FWIT); default 0 makes it free. |
| `wwp` | number | 0.0 | Cost per unit of water produced (FWPT); default 0 makes it free. |

## `field_total`

A field total at one time, scaled: e.g. the water injected.

Select it with:

```toml
[objective]
name = "field_total"
```

| option | type | default | description |
|---|---|---|---|
| `scale` | number | 1.0 | Multiplies the total (e.g. `1e-6` for millions). |
| `time` | string | required | The report time (ISO date). |
| `vector` | string | required | The summary vector, a cumulative total such as `FWIT`, `FOPT` or `FGPT`. |

## `npv`

NPV of the field's oil and gas production, water production and water injection.

Select it with:

```toml
[objective]
name = "npv"
```

| option | type | default | description |
|---|---|---|---|
| `disc` | number | 0.0 | Yearly discount rate. |
| `from_time` | string or null | | One of `times` from which NPV counts: only the steps after it, with the discount from it. For a forecast after a history, whose production is not the decision's. Default: from `start`, as PET. |
| `obj_scaling` | number | 1.0 | Divides NPV; negative to maximise it (PET's tutorial: `-1e6`). |
| `start` | string | required | The simulation's start date (ISO), where the totals are zero. |
| `times` | array | required | Report dates (ISO), increasing and after `start`; the simulator must report at each. |
| `wgp` | number | 0.0 | Oil and gas prices, per unit of FOPT and FGPT. |
| `wop` | number | required | Oil and gas prices, per unit of FOPT and FGPT. |
| `wwi` | number | 0.0 | Costs of produced and injected water, per unit of FWPT and FWIT. |
| `wwp` | number | 0.0 | Costs of produced and injected water, per unit of FWPT and FWIT. |

## `ren_npv`

NPV with water injection priced per well: by grid or by renewable power.

Select it with:

```toml
[objective]
name = "ren_npv"
```

| option | type | default | description |
|---|---|---|---|
| `disc` | number | 0.0 | As `npv`. |
| `from_time` | string or null | | As `npv`. |
| `injectors` | array | | Wells whose injected water (WWIT) costs `wwi`. |
| `obj_scaling` | number | 1.0 | As `npv`. |
| `renewable_injectors` | array | | Wells whose injected water costs `wrenwi` (powered by renewable energy). |
| `start` | string | required | As `npv`. |
| `times` | array | required | As `npv`. |
| `wgp` | number | 0.0 | As `npv`. |
| `wop` | number | required | As `npv`. |
| `wrenwi` | number | 0.0 | Cost per unit of water injected by `renewable_injectors` (powered by renewable energy). |
| `wwi` | number | 0.0 | Cost per unit of water injected by `injectors`. |
| `wwp` | number | 0.0 | As `npv`. |

## `ren_npv_co2`

NPV with eCalc's CO2, where wind power and injection pressure enter eCalc.

Select it with:

```toml
[objective]
name = "ren_npv_co2"
```

| option | type | default | description |
|---|---|---|---|
| `disc` | number | 0.0 | Discount rate per year (e.g. 0.08 for 8 %), applied by the days since `start`; must be above -1, and the default 0 does not discount. |
| `ecalc_model_file` | string | required | The eCalc model (YAML), whose START and END are `start` and the last of `times` and whose time series reads the step rates (per day). |
| `emission` | string | co2 | The emission priced at `wem`, by its name in the model's fuels and emitters (case-insensitive; default `co2`, PET's cases used `co2_fuel_gas`). |
| `injectors` | array | required | The injectors whose WTHP gives `THP_MAX`. |
| `obj_scaling` | number | 1.0 | Divides the NPV (e.g. 1e6 for millions); any nonzero number, default 1. |
| `start` | string | required | The simulation's start date (ISO, e.g. `2025-01-01`): the deck's START, checked against the summary's TIME; discounting counts the days from it. |
| `time_series` | string or null | | The name of the model's time series that the simulated rates replace; by default the model's only one (required when it has several). |
| `times` | array | required | The report dates (ISO), increasing and after `start`; each is the end of one step whose rates eCalc gets. |
| `wem` | number | required | Cost per tonne of the emission eCalc computes. |
| `wgp` | number | 0.0 | Gas price per unit of FGPT; default 0 leaves gas out of the value. |
| `wind_power_file` | string | required | `.npy` of the wind power available in each report step `(n_steps,)`. |
| `wop` | number | required | Oil price per unit of FOPT (PET's `wop`). |
| `wwi` | number | 0.0 | Cost per unit of water injected (FWIT); default 0 makes it free. |
| `wwp` | number | 0.0 | Cost per unit of water produced (FWPT); default 0 makes it free. |

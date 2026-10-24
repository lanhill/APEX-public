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
| `disc` | number | 0.0 | |
| `ecalc_model_file` | string or string | required | |
| `emission` | string | co2 | |
| `obj_scaling` | number | 1.0 | |
| `start` | string | required | |
| `time_series` | string or null | | |
| `times` | array | required | |

## `ecalc_npv`

NPV of oil, gas, water and the CO2 emitted, the emissions computed by eCalc.

Select it with:

```toml
[objective]
name = "ecalc_npv"
```

| option | type | default | description |
|---|---|---|---|
| `disc` | number | 0.0 | |
| `ecalc_model_file` | string or string | required | |
| `emission` | string | co2 | |
| `obj_scaling` | number | 1.0 | |
| `start` | string | required | |
| `time_series` | string or null | | |
| `times` | array | required | |
| `wem` | number | required | |
| `wgp` | number | 0.0 | |
| `wop` | number | required | |
| `wwi` | number | 0.0 | |
| `wwp` | number | 0.0 | |

## `ecalc_pareto_npv`

NPV with eCalc's CO2, combined with an emission objective (PET's two variants).

Select it with:

```toml
[objective]
name = "ecalc_pareto_npv"
```

| option | type | default | description |
|---|---|---|---|
| `disc` | number | 0.0 | |
| `ecalc_model_file` | string or string | required | |
| `emission` | string | co2 | |
| `obj_scaling` | number | 1.0 | |
| `start` | string | required | |
| `time_series` | string or null | | |
| `times` | array | required | |
| `w` | number or null | | |
| `wem` | number | required | |
| `wemc` | number or null | | |
| `wgp` | number | 0.0 | |
| `wop` | number | required | |
| `wwi` | number | 0.0 | |
| `wwp` | number | 0.0 | |

## `field_total`

A field total at one time, scaled: e.g. the water injected.

Select it with:

```toml
[objective]
name = "field_total"
```

| option | type | default | description |
|---|---|---|---|
| `scale` | number | 1.0 | |
| `time` | string | required | |
| `vector` | string | required | |

## `npv`

NPV of the field's oil and gas production, water production and water injection.

Select it with:

```toml
[objective]
name = "npv"
```

| option | type | default | description |
|---|---|---|---|
| `disc` | number | 0.0 | |
| `from_time` | string or null | | |
| `obj_scaling` | number | 1.0 | |
| `start` | string | required | |
| `times` | array | required | |
| `wgp` | number | 0.0 | |
| `wop` | number | required | |
| `wwi` | number | 0.0 | |
| `wwp` | number | 0.0 | |

## `ren_npv`

NPV with water injection priced per well: by grid or by renewable power.

Select it with:

```toml
[objective]
name = "ren_npv"
```

| option | type | default | description |
|---|---|---|---|
| `disc` | number | 0.0 | |
| `from_time` | string or null | | |
| `injectors` | array | | |
| `obj_scaling` | number | 1.0 | |
| `renewable_injectors` | array | | |
| `start` | string | required | |
| `times` | array | required | |
| `wgp` | number | 0.0 | |
| `wop` | number | required | |
| `wrenwi` | number | 0.0 | |
| `wwi` | number | 0.0 | |
| `wwp` | number | 0.0 | |

## `ren_npv_co2`

NPV with eCalc's CO2, where wind power and injection pressure enter eCalc.

Select it with:

```toml
[objective]
name = "ren_npv_co2"
```

| option | type | default | description |
|---|---|---|---|
| `disc` | number | 0.0 | |
| `ecalc_model_file` | string or string | required | |
| `emission` | string | co2 | |
| `injectors` | array | required | |
| `obj_scaling` | number | 1.0 | |
| `start` | string | required | |
| `time_series` | string or null | | |
| `times` | array | required | |
| `wem` | number | required | |
| `wgp` | number | 0.0 | |
| `wind_power_file` | string or string | required | |
| `wop` | number | required | |
| `wwi` | number | 0.0 | |
| `wwp` | number | 0.0 | |

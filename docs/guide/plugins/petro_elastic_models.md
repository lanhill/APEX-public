---
audience: public
items:
  - cli:apex plugins
  - plugin:petro_elastic_models:soft_sand
  - plugin:petro_elastic_models:standard
---

# Petro elastic models

## Goal

By the end of this page you will know what the two plugins in the `petro_elastic_models` registry, `standard` and `soft_sand`, are for, which options control each, and how to override them in a TOML configuration.

- `standard` corresponds to SimulatorWrap's `standardrp`. It uses fluid fits, mixed minerals (clay and quartz) and a power-law dry rock.
- `soft_sand` corresponds to SimulatorWrap's `softsandrp`. It is a soft-sand rock model with Batzle & Wang fluids and Brie mixing by default.

## Prerequisites

- An APEX configuration file in TOML format that you can edit.
- Values for your own rock and fluids, if you do not want the defaults. Every option has a default, so you only set what differs.

## Steps

1. List the plugins and their options to see what is available:

   ```bash
   apex plugins
   ```

   Add `--json` to get the full descriptions as JSON.
2. Choose it as the `pem` of the run's `[geophysics]` section: `[geophysics.pem]` with `name = "standard"` or `name = "soft_sand"`, and its options in `[geophysics.pem.options]`. AVO data need a petro-elastic model.
3. Set only the options you want to change. Unlisted options keep their defaults.
4. Use only the option names in the tables below. Neither plugin accepts extra keys, so a misspelled name is an error and is not ignored.
5. Run `apex plugins` again to check that the plugin is listed with the options you expect.

This page does not cover how a model is chosen for a particular run. See the configuration reference for that.

### A note on units

The defaults are on the scale of SI values: moduli are of order 10^9 to 10^10 and densities of order 10^3. For example, the default quartz bulk modulus is `37000000000.0`. Give your own values in the same units and on the same scale as the defaults. Pressure options such as `min_effective_pressure` follow the same rule.

## The `standard` plugin

### Rock and porosity options

| Option | Type | Default |
|---|---|---|
| `critical_porosity` | number | `0.4` |
| `clay_volume_intercept` | number | `0.7` |
| `clay_volume_slope` | number | `1.58` |
| `min_effective_pressure` | number | `10000.0` |

`critical_porosity` is the porosity limit used by the rock model. `clay_volume_intercept` and `clay_volume_slope` are the two parameters that set the clay volume. `min_effective_pressure` is the lowest effective pressure the model uses.

### Mineral options

| Option | Type | Default |
|---|---|---|
| `clay_bulk_modulus` | number | `14900000000.0` |
| `clay_shear_modulus` | number | `1950000000.0` |
| `clay_density` | number | `2540.0` |
| `quartz_bulk_modulus` | number | `37000000000.0` |
| `quartz_shear_modulus` | number | `44000000000.0` |
| `quartz_density` | number | `2650.0` |

### Dry-rock (pack) options

The dry rock is described by a pack with a power-law dependence on effective pressure. The options give its moduli, a reference pressure and the exponent.

| Option | Type | Default |
|---|---|---|
| `pack_bulk_modulus` | number | `3310000000.0` |
| `pack_shear_modulus` | number | `2840000000.0` |
| `pack_reference_pressure` | number | `8800000.0` |
| `pack_exponent` | number | `0.233` |

### Fluid fits

`fluid_fits` is either left out (the default is `null`) or a table keyed by phase. The allowed phase names are `oil`, `water` and `gas`, in lowercase. Each phase maps to a list of exactly four numbers. This page does not define what the four numbers mean, so take them, in order, from the fluid fit you are using.

### Worked example for `standard`

This configuration lowers the critical porosity, changes the pack exponent, and supplies fit coefficients for oil and water. The four numbers per phase are placeholders that show the shape. Replace them with your own fit values.

```toml
[geophysics.pem]
name = "standard"

[geophysics.pem.options]
critical_porosity = 0.38
quartz_density = 2650.0
pack_exponent = 0.25
fluid_fits = { oil = [1.0, 2.0, 3.0, 4.0], water = [1.5, 2.5, 3.5, 4.5] }
```

Everything not listed, such as the clay properties and the other pack options, keeps its default. `gas` is not given in `fluid_fits`, so no gas entry is supplied.

## The `soft_sand` plugin

### Rock and mineral options

| Option | Type | Default |
|---|---|---|
| `critical_porosity` | number | `0.36` |
| `coordination_number` | number | `9.0` |
| `mineral_bulk_modulus` | number | `37000000000.0` |
| `mineral_shear_modulus` | number | `44000000000.0` |
| `mineral_density` | number | `2650.0` |
| `min_effective_pressure` | number | `10000.0` |
| `porosity_ratio_cap` | number or `null` | `0.9` |

### Fluid options

| Option | Type | Default |
|---|---|---|
| `temperature_c` | number | `37.0` |
| `salinity` | number | `0.035` |
| `gas` | `"co2"` or `"methane"` | `"co2"` |
| `gas_model` | `"span_wagner"`, `"batzle_wang"` or `null` | `null` |
| `oil_reference_density` | number or `null` | `null` |
| `extrapolate_gas_correlation` | boolean | `false` |
| `eos_density_range` | list of two numbers | `[1.0, 1200.0]` |
| `temperature` | `"deck"` or `null` | `null` |
| `temperature_range` | list of two numbers or `null` | `null` |
| `dissolved_co2` | `"none"` or `"lebedev2014"` | `"none"` |
| `dissolved_co2_min_sg` | number | `1e-06` |

`temperature_c` is the fixed temperature. Setting `temperature = "deck"` is the only non-null value allowed for `temperature`.

### Mixing options

| Option | Type | Default |
|---|---|---|
| `mixing` | `"brie"`, `"wood"`, `"patchy"` or `"patchiness"` | `"brie"` |
| `brie_exponent` | number | `5.0` |
| `patchiness` | number | `0.5` |

`brie_exponent` belongs to the `brie` mixing, and `patchiness` belongs to the patchy mixing choices.

### Worked example for `soft_sand`

This configuration uses methane at a higher temperature and switches to patchy mixing.

```toml
[geophysics.pem]
name = "soft_sand"

[geophysics.pem.options]
gas = "methane"
temperature_c = 60.0
mixing = "patchy"
patchiness = 0.3
critical_porosity = 0.38
```

All other options keep their defaults, including `dissolved_co2 = "none"`.

## Pitfalls

- **Misspelled or unknown options are rejected.** Use only the names listed on this page.
- **Wrong list length.** Each `fluid_fits` phase needs exactly four numbers, and `eos_density_range` and `temperature_range` need exactly two.
- **Wrong phase name.** In `fluid_fits`, only `oil`, `water` and `gas` are accepted.
- **Wrong choice value.** `gas`, `gas_model`, `mixing`, `dissolved_co2` and `temperature` accept only the values in the tables.
- **Unit mismatch.** Entering a modulus in GPa where the default is in Pa gives a wildly wrong rock. Keep your values on the scale of the defaults.
- **Mixing the two plugins' options.** `standard` has no `mixing` or `gas` option, and `soft_sand` has no `clay_*`, `quartz_*` or `pack_*` options. Each plugin accepts only its own names.
- **`fluid_fits = {}`.** An empty table is a valid table, but this page does not describe its effect. To use no fits, leave `fluid_fits` out so that it keeps its default `null`.

## What to read next

- The other pages in the plugins section of the guide, for the rest of the plugin registries.
- The [configuration reference](../../reference/configuration/history_match.md), for the rest of `[geophysics]` and its `[[geophysics.data]]` entries.

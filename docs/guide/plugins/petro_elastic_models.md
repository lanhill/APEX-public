---
audience: public
items:
  - cli:apex plugins
  - cli:apex resume
  - cli:apex run
  - cli:apex stop
  - plugin:petro_elastic_models:soft_sand
  - plugin:petro_elastic_models:standard
---

# Petro elastic models

## Goal

By the end of this page you will know what the two plugins in the `petro_elastic_models` registry, `standard` and `soft_sand`, are for, which options control each, and how to override them in a TOML configuration.

- `standard` corresponds to SimulatorWrap's `standardrp`. It uses fluid fits, mixed minerals (clay and quartz) and a power-law dry rock.
- `soft_sand` corresponds to SimulatorWrap's `softsandrp`. It is a soft-sand rock model with Batzle & Wang fluids and Brie mixing by default.

## Prerequisites

- An APEX run configuration in TOML format that you can edit.
- Values for your own rock and fluids, if you do not want the defaults. Every option has a default, so you only set what differs.

## Steps

1. List the plugins and their options to see what is available:

   ```bash
   uv run apex plugins
   ```

   Add `--json` to get the full descriptions as JSON:

   ```bash
   uv run apex plugins --json
   ```
2. Select a plugin by adding a `[geophysics.pem]` table to the run configuration, with `name = "standard"` or `name = "soft_sand"`. Put its options in `[geophysics.pem.options]`. AVO data need a petro-elastic model.
3. Set only the options you want to change. Unlisted options keep their defaults.
4. Use only the option names in the tables below. Neither plugin accepts extra keys, so a misspelled name is an error and is not ignored.
5. Check the edited file by starting the run for a single analysis step:

   ```bash
   uv run apex run config.toml --max-steps 1
   ```

   `--max-steps` stops the run after that many analysis steps. Continue it later with `apex resume`, giving the run directory:

   ```bash
   uv run apex resume path/to/run_dir
   ```

   Add `--no-progress` to either command if you do not want progress output on standard error. To ask a running run to stop at its next checkpoint, use `apex stop` with the run directory; `apex resume` continues it. Note that `apex plugins` only lists what each plugin accepts. It does not read your configuration, so it cannot confirm your edits.

This page does not cover how a model is chosen for a particular run. See the configuration reference for that.

### A note on units

Most options are in SI units: moduli and pressures in Pa (moduli are of order 10^9 to 10^10, for example the default quartz bulk modulus is `37000000000.0`), densities in kg/m³, temperatures in °C. The one exception is `fluid_fits` of `standard`, which works in MPa (see below). Give your own values in the units of the option you set.

## The `standard` plugin

### Rock and porosity options

| Option | Type | Default |
|---|---|---|
| `critical_porosity` | number | `0.4` |
| `clay_volume_intercept` | number | `0.7` |
| `clay_volume_slope` | number | `1.58` |
| `min_effective_pressure` | number | `10000.0` |

- `critical_porosity` is the porosity at which the dry rock takes the pack's moduli. It must be in `(0, 1)`.
- `clay_volume_intercept` and `clay_volume_slope` define the clay volume fraction of each cell as `clay_volume_intercept - clay_volume_slope * porosity`. With the defaults that is `0.7 - 1.58 * porosity`. A cell where the result leaves `[0, 1]` is refused, so check that your porosities keep it inside that range.
- `min_effective_pressure` is the lowest effective pressure in Pa (overburden minus pore pressure). Cells below it are raised to it and counted. It must be positive; the default is 0.01 MPa.

### Mineral options

| Option | Type | Default |
|---|---|---|
| `clay_bulk_modulus` | number | `14900000000.0` |
| `clay_shear_modulus` | number | `1950000000.0` |
| `clay_density` | number | `2540.0` |
| `quartz_bulk_modulus` | number | `37000000000.0` |
| `quartz_shear_modulus` | number | `44000000000.0` |
| `quartz_density` | number | `2650.0` |

The moduli are in Pa and the densities in kg/m³.

### Dry-rock (pack) options

The dry rock is described by a grain pack whose moduli grow with effective pressure as `(P_eff / P_ref)^exponent`.

| Option | Type | Default |
|---|---|---|
| `pack_bulk_modulus` | number | `3310000000.0` |
| `pack_shear_modulus` | number | `2840000000.0` |
| `pack_reference_pressure` | number | `8800000.0` |
| `pack_exponent` | number | `0.233` |

`pack_bulk_modulus` and `pack_shear_modulus` are the pack's moduli in Pa at `pack_reference_pressure`, which is the effective pressure in Pa (default 8.8 MPa) at which the pack has those moduli. `pack_exponent` is the power of the growth.

### Fluid fits

`fluid_fits` is either left out (the default, `null`) or a table keyed by phase. The allowed phase names are `oil`, `water` and `gas`, in lowercase. Each phase maps to a list of exactly four numbers, `[a, b, k0, k1]`, where `P` is the pore pressure in MPa:

- density in kg/m³ is `a + b * P`;
- bulk modulus in MPa is `k0 + k1 * P`.

So `a` is a density, `b` its change per MPa, `k0` a bulk modulus in MPa and `k1` its change per MPa. A phase you do not give keeps the reference's fit, so you can override just one phase.

### Worked example for `standard`

This configuration lowers the critical porosity, changes the pack exponent, and supplies a fit for water only. The four numbers are illustrative: water at about 1000 kg/m³ with a small increase per MPa, and a bulk modulus of about 2300 MPa. Replace them with values from your own fluid fit. Oil and gas keep the reference's fits.

```toml
[geophysics.pem]
name = "standard"

[geophysics.pem.options]
critical_porosity = 0.38
quartz_density = 2650.0
pack_exponent = 0.25
fluid_fits = { water = [1000.0, 0.5, 2300.0, 8.0] }
```

Everything not listed, such as the clay properties and the other pack options, keeps its default.

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
| `porosity_ratio_cap` | number | `0.9` |

- `critical_porosity` is the porosity at which the dry rock takes the Hertz-Mindlin pack's moduli, in `(0, 1)`.
- `coordination_number` is the average number of contacts per grain in the pack. It must be positive.
- `mineral_bulk_modulus`, `mineral_shear_modulus` (in Pa) and `mineral_density` (in kg/m³) describe the single mineral; the defaults are quartz. The Poisson's ratio that follows from the two moduli must be in `(-1, 0.5)`.
- `coordination_number`, the mineral moduli and `mineral_density` may also be set per member by a state variable.
- `porosity_ratio_cap` caps the ratio `porosity / critical_porosity` in the dry-rock interpolation. A value must be in `(0, 1]`, and the default is `0.9`. The plugin's description says that leaving it unset means no cap; this page does not cover how to unset it in TOML, so leave it out to keep the default of `0.9`.
- `min_effective_pressure` is the lowest effective pressure in Pa (overburden minus pore pressure). Cells below it are raised to it and counted. It must be positive; the default is 0.01 MPa.

### Fluid options

| Option | Type | Default |
|---|---|---|
| `temperature_c` | number | `37.0` |
| `salinity` | number | `0.035` |
| `gas` | `"co2"` or `"methane"` | `"co2"` |
| `gas_model` | `"span_wagner"`, `"batzle_wang"` or unset | unset |
| `oil_reference_density` | number or unset | unset |
| `extrapolate_gas_correlation` | boolean | `false` |
| `eos_density_range` | list of two numbers | `[1.0, 1200.0]` |
| `temperature` | `"deck"` or unset | unset |
| `temperature_range` | list of two numbers or unset | unset |
| `dissolved_co2` | `"none"` or `"lebedev2014"` | `"none"` |
| `dissolved_co2_min_sg` | number | `1e-06` |

What each option does:

- `temperature_c` is the reservoir temperature in °C for every cell, within 0 to 100. Change it for a different reservoir temperature. It is not used for cells whose temperature comes from the deck (see `temperature`).
- `salinity` is the brine's salt content as a weight fraction, in `[0, 0.35)`. With `dissolved_co2 = "lebedev2014"` it can be at most 0.1.
- `gas` selects the gas phase, `"co2"` or `"methane"`.
- `gas_model` selects how the gas modulus is computed. `"span_wagner"` is CO₂'s equation of state at the simulated density and needs the `co2` extra. `"batzle_wang"` is the reference's correlation. When unset, CO₂ uses Span & Wagner unless `extrapolate_gas_correlation` is set, and methane uses Batzle & Wang.
- `extrapolate_gas_correlation` allows Batzle & Wang's gas modulus outside its fitted range. With `false`, cells with gas outside that range are refused. Set it to `true` only together with `gas_model = "batzle_wang"`.
- `oil_reference_density` is the oil's density at standard conditions in kg/m³, below 1080, used for the dead-oil velocity. When unset, the simulator's in-situ oil density is used, which is an approximation.
- `eos_density_range` is `[low, high]`, the CO₂ densities in kg/m³ spanned by the equation of state's table, with `0 < low < high`.
- `temperature` accepts only `"deck"`. It makes each cell take its temperature from the simulation deck: a temperature-depth table, or a thermal run's restart. Use it when the temperature varies in your model. When unset, `temperature_c` is used everywhere.
- `temperature_range` only matters with `temperature = "deck"`. It is `[low, high]` in °C, within 0 to 100, and gives the temperatures over which CO₂'s table is built. When unset it is taken from the deck's depth table, but a thermal run must give it.
- `dissolved_co2 = "lebedev2014"` corrects the brine's velocity in cells with free CO₂ for the CO₂ dissolved in it. It applies to full saturation only and to CO₂ only. `"none"` applies no correction.
- `dissolved_co2_min_sg` is the gas saturation (`sg`) above which a cell's brine is corrected for dissolved CO₂. It is in `[0, 1)`.

### Mixing options

| Option | Type | Default |
|---|---|---|
| `mixing` | `"brie"`, `"wood"`, `"patchy"` or `"patchiness"` | `"brie"` |
| `brie_exponent` | number | `5.0` |
| `patchiness` | number | `0.5` |

`mixing` selects how the fluids are mixed:

| `mixing` | Meaning | Option it reads |
|---|---|---|
| `"brie"` | Brie's mixing of liquid and gas | `brie_exponent` (positive; larger values approach uniform mixing) |
| `"wood"` | Uniform saturation, the lower bound | none |
| `"patchy"` | Patchy saturation, the upper bound | none |
| `"patchiness"` | Between the two bounds | `patchiness`, the weight of the patchy bound against Wood's, in `[0, 1]` |

The difference between `"patchy"` and `"patchiness"` is that `"patchy"` is always the upper bound, while `"patchiness"` lets you choose a point between Wood's bound (`patchiness = 0`) and the patchy bound (`patchiness = 1`). `"patchy"` and `"patchiness"` take two fluid phases. `patchiness` may also be set per member by a state variable.

### Worked example for `soft_sand`

This configuration uses methane at a higher temperature and mixes the fluids part way between the Wood and patchy bounds.

```toml
[geophysics.pem]
name = "soft_sand"

[geophysics.pem.options]
gas = "methane"
temperature_c = 60.0
mixing = "patchiness"
patchiness = 0.3
critical_porosity = 0.38
```

All other options keep their defaults, including `dissolved_co2 = "none"`. With methane and no `gas_model`, Batzle & Wang's correlation is used for the gas.

## Pitfalls

- **Misspelled or unknown options are rejected.** Use only the names listed on this page.
- **Wrong list length.** Each `fluid_fits` phase needs exactly four numbers, and `eos_density_range` and `temperature_range` need exactly two.
- **Wrong phase name.** In `fluid_fits`, only `oil`, `water` and `gas` are accepted.
- **Wrong choice value.** `gas`, `gas_model`, `mixing`, `dissolved_co2` and `temperature` accept only the values in the tables.
- **Unit mismatch.** Entering a modulus in GPa where the option is in Pa gives a wildly wrong rock. Note that `fluid_fits` bulk moduli are in MPa and use the pore pressure in MPa.
- **Clay volume outside `[0, 1]`.** In `standard`, a cell where `clay_volume_intercept - clay_volume_slope * porosity` leaves `[0, 1]` is refused.
- **Setting `patchiness` with another `mixing`.** `patchiness` is only the weight with `mixing = "patchiness"`, and `brie_exponent` belongs to `"brie"`.
- **Gas outside the correlation's range.** With Batzle & Wang's gas modulus, cells with gas outside its fitted range are refused unless `extrapolate_gas_correlation` is `true`, which needs `gas_model = "batzle_wang"`.
- **Salinity with dissolved CO₂.** `dissolved_co2 = "lebedev2014"` needs `salinity` of at most 0.1, and works for CO₂ only.
- **Mixing the two plugins' options.** `standard` has no `mixing` or `gas` option, and `soft_sand` has no `clay_*`, `quartz_*` or `pack_*` options. Each plugin accepts only its own names.
- **Expecting `apex plugins` to check your file.** It lists the available options only. Use `apex run` with `--max-steps` to try your configuration.

## What to read next

- The other pages in the plugins section of the guide, for the rest of the plugin registries.
- The [configuration reference](../../reference/configuration/history_match.md), for the rest of `[geophysics]` and its `[[geophysics.data]]` entries.

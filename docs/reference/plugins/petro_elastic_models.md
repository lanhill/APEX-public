---
audience: public
generated: reference
---

# Petro elastic models: reference

Every one of these 2 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `soft_sand`

SimulatorWrap's `softsandrp`: soft sand, Batzle & Wang fluids, Brie mixing.

Select it with:

```toml
[geophysics.pem]
name = "soft_sand"
```

| option | type | default | description |
|---|---|---|---|
| `brie_exponent` | number | 5.0 | |
| `coordination_number` | number | 9.0 | |
| `critical_porosity` | number | 0.36 | |
| `dissolved_co2` | string | none | |
| `dissolved_co2_min_sg` | number | 1e-06 | |
| `eos_density_range` | array | [1.0, 1200.0] | |
| `extrapolate_gas_correlation` | boolean | false | |
| `gas` | string | co2 | |
| `gas_model` | string or null | | |
| `min_effective_pressure` | number | 10000.0 | |
| `mineral_bulk_modulus` | number | 37000000000.0 | |
| `mineral_density` | number | 2650.0 | |
| `mineral_shear_modulus` | number | 44000000000.0 | |
| `mixing` | string | brie | |
| `oil_reference_density` | number or null | | |
| `patchiness` | number | 0.5 | |
| `porosity_ratio_cap` | number or null | 0.9 | |
| `salinity` | number | 0.035 | |
| `temperature` | string or null | | |
| `temperature_c` | number | 37.0 | |
| `temperature_range` | array or null | | |

## `standard`

SimulatorWrap's `standardrp`: fluid fits, mixed minerals, power-law dry rock.

Select it with:

```toml
[geophysics.pem]
name = "standard"
```

| option | type | default | description |
|---|---|---|---|
| `clay_bulk_modulus` | number | 14900000000.0 | |
| `clay_density` | number | 2540.0 | |
| `clay_shear_modulus` | number | 1950000000.0 | |
| `clay_volume_intercept` | number | 0.7 | |
| `clay_volume_slope` | number | 1.58 | |
| `critical_porosity` | number | 0.4 | |
| `fluid_fits` | object or null | | |
| `min_effective_pressure` | number | 10000.0 | |
| `pack_bulk_modulus` | number | 3310000000.0 | |
| `pack_exponent` | number | 0.233 | |
| `pack_reference_pressure` | number | 8800000.0 | |
| `pack_shear_modulus` | number | 2840000000.0 | |
| `quartz_bulk_modulus` | number | 37000000000.0 | |
| `quartz_density` | number | 2650.0 | |
| `quartz_shear_modulus` | number | 44000000000.0 | |

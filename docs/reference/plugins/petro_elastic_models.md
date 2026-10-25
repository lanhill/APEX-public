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
| `brie_exponent` | number | 5.0 | The exponent of Brie's mixing of liquid and gas (positive, default 5); larger values approach uniform (Wood) mixing. |
| `coordination_number` | number | 9.0 | The average number of contacts per grain in the pack (positive, default 9); may be set per member by a state variable. |
| `critical_porosity` | number | 0.36 | The porosity at which the dry rock takes the Hertz-Mindlin pack's moduli, in `(0, 1)` (default 0.36). |
| `dissolved_co2` | string | none | `"lebedev2014"` corrects the brine's velocity in cells with free CO₂ for the CO₂ dissolved in it (full saturation only; CO₂ only); `"none"` (default) does not. |
| `dissolved_co2_min_sg` | number | 1e-06 | The gas saturation above which a cell's brine is corrected for dissolved CO₂, in `[0, 1)` (default 1e-6). |
| `eos_density_range` | array | [1.0, 1200.0] | The CO₂ densities in kg/m³ spanned by the equation of state's table, `[low, high]` with `0 < low < high` (default `[1, 1200]`). |
| `extrapolate_gas_correlation` | boolean | false | Allow Batzle & Wang's gas modulus outside its fitted range (default false: cells with gas are refused there). Only with `gas_model = "batzle_wang"`. |
| `gas` | string | co2 | The gas phase: `"co2"` (default) or `"methane"`. |
| `gas_model` | string or null | | The gas modulus: `"span_wagner"`, CO₂'s equation of state at the simulated density (needs the `co2` extra), or `"batzle_wang"`, the reference's correlation. Unset: Span & Wagner for CO₂, unless `extrapolate_gas_correlation` is set; Batzle & Wang for methane. |
| `min_effective_pressure` | number | 10000.0 | The lowest effective pressure in Pa (overburden minus pore pressure); cells below it are raised to it and counted (positive, default 0.01 MPa). |
| `mineral_bulk_modulus` | number | 37000000000.0 | The mineral's moduli in Pa (defaults 37 GPa and 44 GPa, quartz); their Poisson's ratio must be in `(-1, 0.5)`. May be set per member by a state variable. |
| `mineral_density` | number | 2650.0 | The mineral's density in kg/m³ (default 2650); may be set per member by a state variable. |
| `mineral_shear_modulus` | number | 44000000000.0 | The mineral's moduli in Pa (defaults 37 GPa and 44 GPa, quartz); their Poisson's ratio must be in `(-1, 0.5)`. May be set per member by a state variable. |
| `mixing` | string | brie | How the fluids are mixed: `"brie"` (default), `"wood"` (uniform saturation, the lower bound), `"patchy"` (the upper bound) or `"patchiness"` (between the two by `patchiness`). The last two take two fluid phases. |
| `oil_reference_density` | number or null | | The oil's density at standard conditions in kg/m³, below 1080, for the dead-oil velocity; unset uses the simulator's in-situ oil density, an approximation. |
| `patchiness` | number | 0.5 | With `mixing = "patchiness"`, the weight of the patchy bound against Wood's, in `[0, 1]` (default 0.5); may be set per member by a state variable. |
| `porosity_ratio_cap` | number or null | 0.9 | The cap on `porosity / critical_porosity` in the dry-rock interpolation, in `(0, 1]` (default 0.9); unset for no cap. |
| `salinity` | number | 0.035 | The brine's salt content as a weight fraction, in `[0, 0.35)` (default 0.035); at most 0.1 with `dissolved_co2 = "lebedev2014"`. |
| `temperature` | string or null | | `"deck"` takes each cell's temperature from the deck (a temperature-depth table, or a thermal run's restart); unset (default) uses `temperature_c` everywhere. |
| `temperature_c` | number | 37.0 | The reservoir temperature in °C for every cell, within 0-100 (default 37); not used for the cells with `temperature = "deck"`. |
| `temperature_range` | array or null | | With `temperature = "deck"`, the temperatures in °C `[low, high]`, within 0-100, over which CO₂'s table is built; taken from the deck's depth table when unset, but a thermal run must give it. |

## `standard`

SimulatorWrap's `standardrp`: fluid fits, mixed minerals, power-law dry rock.

Select it with:

```toml
[geophysics.pem]
name = "standard"
```

| option | type | default | description |
|---|---|---|---|
| `clay_bulk_modulus` | number | 14900000000.0 | The clay's moduli in Pa (defaults 14.9 GPa and 1.95 GPa). |
| `clay_density` | number | 2540.0 | The clay's density in kg/m³ (default 2540). |
| `clay_shear_modulus` | number | 1950000000.0 | The clay's moduli in Pa (defaults 14.9 GPa and 1.95 GPa). |
| `clay_volume_intercept` | number | 0.7 | The clay volume fraction is `clay_volume_intercept - clay_volume_slope * porosity` (defaults 0.7 and 1.58); a cell where it leaves `[0, 1]` is refused. |
| `clay_volume_slope` | number | 1.58 | The clay volume fraction is `clay_volume_intercept - clay_volume_slope * porosity` (defaults 0.7 and 1.58); a cell where it leaves `[0, 1]` is refused. |
| `critical_porosity` | number | 0.4 | The porosity at which the dry rock takes the pack's moduli, in `(0, 1)` (default 0.4). |
| `fluid_fits` | object or null | | Phase (`oil`, `water`, `gas`) to `[a, b, k0, k1]`: density `a + b P` in kg/m³ and bulk modulus `k0 + k1 P` in MPa, `P` the pore pressure in MPa. A phase not given keeps the reference's fit. |
| `min_effective_pressure` | number | 10000.0 | The lowest effective pressure in Pa (overburden minus pore pressure); cells below it are raised to it and counted (positive, default 0.01 MPa). |
| `pack_bulk_modulus` | number | 3310000000.0 | The grain pack's moduli in Pa at `pack_reference_pressure` (defaults 3.31 GPa and 2.84 GPa). |
| `pack_exponent` | number | 0.233 | The power of the pack moduli's growth with effective pressure, `(P_eff / P_ref)^exponent` (default 0.233). |
| `pack_reference_pressure` | number | 8800000.0 | The effective pressure in Pa at which the pack has those moduli (default 8.8 MPa). |
| `pack_shear_modulus` | number | 2840000000.0 | The grain pack's moduli in Pa at `pack_reference_pressure` (defaults 3.31 GPa and 2.84 GPa). |
| `quartz_bulk_modulus` | number | 37000000000.0 | The quartz's moduli in Pa (defaults 37 GPa and 44 GPa). |
| `quartz_density` | number | 2650.0 | The quartz's density in kg/m³ (default 2650). |
| `quartz_shear_modulus` | number | 44000000000.0 | The quartz's moduli in Pa (defaults 37 GPa and 44 GPa). |

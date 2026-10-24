---
audience: public
generated: reference
---

# Configuring history match: reference

Every one of these 16 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `history_match:case`

`[case]`: a synthetic case study (known true model) or a real one.

A synthetic case needs `truth_file`: an `.npz` with one array per state variable,
the parameters that generated the observations. Metrics against the true model
(`rmse`, `ssim`) run only in synthetic cases.

| option | type | default | description |
|---|---|---|---|
| `kind` | string | synthetic | |
| `truth_file` | string or null | | |

## `history_match:data_transforms`


## `history_match:execution`

A registry name plus keyword options for its factory.

`diagnostics`, for plugins that produce diagnostics (the method, its
strategy and localization), switches this plugin's items on or off and **overrides**
`[output] diagnostics` for it: `{ taper = true }` saves the taper even if
`[output]` does not list it, `{ taper = false }` suppresses it even if it does.
`max_mb` and `every` override `[output] diagnostics_max_mb` and
`diagnostics_every` for this plugin. The run logs the resulting plan and warns when
a plugin switch contradicts `[output]`.

| option | type | default | description |
|---|---|---|---|
| `diagnostics` | object or null | | Per-plugin diagnostics switches (name = true/false, max_mb, every); they override [output] diagnostics for this plugin. |
| `name` | string | required | |
| `options` | object | | |

## `history_match:failures`

`[failures]`: what to do when member simulations fail.

| option | type | default | description |
|---|---|---|---|
| `min_success_fraction` | number | 0.9 | Drop failed members and continue while at least this fraction of the current ensemble succeeds; otherwise abort. |

## `history_match:geophysics`

Geophysical data (AVO) forecast from the deck's runs.

| option | type | default | description |
|---|---|---|---|
| `data` | array | required | |
| `grid_file` | string | required | An EGRID of the deck (e.g. the truth's or a base-case run's): its active columns fix the traces and its cells the gravity and displacement kernels, the same for every member. |
| `max_field_values` | integer | 1000000 | Most field values per member (every field at every vintage) without a data transform: beyond it each field needs one (e.g. wavelet), as whole traces would fill the checkpoints and memory. |
| `overburden_pressure` | OverburdenPressureGradient or OverburdenPressureFile or null | | For AVO data, the overburden (lithostatic) pressure, for the effective pressure: { gradient = 22600.0, surface = 101325.0 } (Pa/m, Pa: 2300 kg/m³ * 9.81 m/s² over the cell centre's depth, so 45.3 MPa at 2000 m), or one value per cell from a file, { file = "overburden.npz", key = "obvalues", unit = "bar" }. |
| `pem` | string or PluginSection or null | | A plugin of `petro_elastic_models`: `standard` or `soft_sand` (for AVO data). |

**`GeophysicsDataSection`**: `[[geophysics.data]]`: one kind of geophysical data, at its vintages.

| option | type | default | description |
|---|---|---|---|
| `datatype` | string | required | The observations' datatype: AVO fields are `<datatype>_<angle>`, e.g. avo_10, avo_12p5; gravity and displacement fields are `<datatype>`. |
| `kind` | string | required | 'avo', 'gravity' or 'seafloor_displacement' (onshore too). |
| `options` | AVOOptions or GravityOptions or DisplacementOptions | required | |
| `vintages` | array | required | ISO dates of the surveys; each needs a restart. |

**`OverburdenPressureFile`**: The overburden pressure of each cell, from a file (the reference's form).

| option | type | default | description |
|---|---|---|---|
| `file` | string | required | An .npz or .npy file: one value per active cell, or per grid cell in Eclipse (F) order. |
| `key` | string or null | | Array name in an .npz; default its first. |
| `unit` | string | Pa | The values' pressure unit. |

**`OverburdenPressureGradient`**: The overburden pressure as lithostatic: `surface + gradient * depth`.

`depth` is each cell centre's true vertical depth (m, positive down, from the run's
INIT). Example::

 # Rock above the reservoir with an average density of 2300 kg/m³ gives
 # gradient = 2300 kg/m³ * 9.81 m/s² ≈ 22,600 Pa/m (≈ 1 psi/ft),
 # so a cell at 2000 m has p_ob ≈ 101,325 + 22,600 * 2000 ≈ 45.3 MPa.
 overburden_pressure = { gradient = 22600.0, surface = 101325.0 } # Pa/m, Pa

| option | type | default | description |
|---|---|---|---|
| `gradient` | number | required | Pa per m of depth: the overburden's mean density times g, e.g. 2300 kg/m³ * 9.81 m/s² ≈ 22600 Pa/m (≈ 1 psi/ft), so that a cell at 2000 m has 101325 + 22600 * 2000 ≈ 45.3 MPa. |
| `surface` | number | 101325.0 | Pa at depth 0 (atmospheric by default; offshore, depth 0 is the sea surface and the water column is part of the gradient's average). |

**`PluginSection`**: A registry name plus keyword options for its factory.

`diagnostics`, for plugins that produce diagnostics (the method, its
strategy and localization), switches this plugin's items on or off and **overrides**
`[output] diagnostics` for it: `{ taper = true }` saves the taper even if
`[output]` does not list it, `{ taper = false }` suppresses it even if it does.
`max_mb` and `every` override `[output] diagnostics_max_mb` and
`diagnostics_every` for this plugin. The run logs the resulting plan and warns when
a plugin switch contradicts `[output]`.

| option | type | default | description |
|---|---|---|---|
| `diagnostics` | object or null | | Per-plugin diagnostics switches (name = true/false, max_mb, every); they override [output] diagnostics for this plugin. |
| `name` | string | required | |
| `options` | object | | |

## `history_match:method`

`[method]`: an ensemble algorithm from the `methods` registry and its strategy.

Iterative methods also take `max_iterations` and an optional `schedule` table
(`DampingSchedule` fields).

| option | type | default | description |
|---|---|---|---|
| `damping` | string or PluginSection or null | | Damping scheme from the damping_schemes registry: rlmmac (GIES's default) or lm (LM-EnRML's default, PET's rule), with its options; for GIES and LM-EnRML. Without it, [method.schedule] gives the default scheme's options. |
| `diagnostics` | object or null | | Per-plugin diagnostics switches (name = true/false, max_mb, every); they override [output] diagnostics for this plugin. |
| `localization` | string or PluginSection or null | | Localization plugin passed to the strategy, e.g. `adaptive` (prior-only, alias `adaptive_prior`), `adaptive_per_update` or `distance`; its `block_sizes` come from the state layout. Unset (the default): no localization, except the local-analysis strategies (local_analysis, local_rlmmac, adaptive_rows), which select data by prior correlation. |
| `max_iterations` | integer | 10 | |
| `name` | string | required | |
| `options` | object | | |
| `schedule` | object or null | | |
| `strategy` | string or PluginSection or null | | |

**`PluginSection`**: A registry name plus keyword options for its factory.

`diagnostics`, for plugins that produce diagnostics (the method, its
strategy and localization), switches this plugin's items on or off and **overrides**
`[output] diagnostics` for it: `{ taper = true }` saves the taper even if
`[output]` does not list it, `{ taper = false }` suppresses it even if it does.
`max_mb` and `every` override `[output] diagnostics_max_mb` and
`diagnostics_every` for this plugin. The run logs the resulting plan and warns when
a plugin switch contradicts `[output]`.

| option | type | default | description |
|---|---|---|---|
| `diagnostics` | object or null | | Per-plugin diagnostics switches (name = true/false, max_mb, every); they override [output] diagnostics for this plugin. |
| `name` | string | required | |
| `options` | object | | |

## `history_match:metrics`

Performance metrics; default `normalized_data_mismatch`, plus `rmse` in a synthetic case (`ssim` needs its grid, so it must be configured).


## `history_match:model_error`

A model-error (MEC) model added to the simulator's output, e.g. `pg_lstm`; its parameters are the state variable it names (default `nn_params`).


## `history_match:observations`

`[observations]`: scalar data from a long-format CSV and/or field data from files.

The CSV has columns `time,datatype,value,variance`. Fields follow the CSV's data.

| option | type | default | description |
|---|---|---|---|
| `fields` | array | | |
| `file` | string or null | | |

**`FieldObservationSection`**: `[[observations.fields]]`: one field datum (e.g. a seismic vintage) from a file.

The file (`.npz` or `.npy`) holds the field's values over its active cells; they
become the keys `(time, datatype, 0..n-1)`.

| option | type | default | description |
|---|---|---|---|
| `datatype` | string | required | |
| `file` | string | required | |
| `key` | string or null | | Array name in an .npz; default its first. |
| `time` | string | required | |
| `variance` | number | required | Error variance of every value; a data transform (e.g. wavelet) may replace it with its own estimate. |

## `history_match:output`

`[output]`: optional extra results written during a run.

Every run saves, per step, the ensemble and its forecast (`checkpoints/iter_<k>.npz`,
for iterative methods the accepted iterates), per-member data mismatches and metrics
(`data_mismatch_<k>.npz`) and the history. Perturbed observations are saved per step for
ES-MDA (redrawn each step) and once for iterative methods. …

| option | type | default | description |
|---|---|---|---|
| `diagnostics` | array | | Diagnostics to save (checkpoints/diagnostics_<k>.npz), e.g. ["taper", "strategy.singular_values"]; default none. A bare name applies to every plugin producing it, 'role.name' to one role (method, strategy, localization). A plugin section's own diagnostics table overrides this for that plugin. |
| `diagnostics_every` | integer | 1 | Save diagnostics every k-th iteration. A plugin's diagnostics.every overrides it. |
| `diagnostics_max_mb` | number | 200.0 | Size limit per item and step (MB); larger items are skipped, noted in the step's index. A plugin's diagnostics.max_mb overrides it. |
| `keep_simulations` | string | all | Which simulator work directories (runs/<name>/simulations/iter_<k>/) to keep: all (default); failed (only failed members'); last (only the latest forecast's); failed_and_last (the latest forecast's and every failed member's); none. Results never depend on them. |
| `save_raw_forecasts` | boolean | false | With data transforms: also save the untransformed forecasts (raw_predicted) in each saved step. |
| `save_rejected_trials` | boolean | false | Iterative methods: also save each rejected trial ensemble and forecast (checkpoints/trial_<k>.npz). |

## `history_match:parameterizations`

Maps from the state to the properties the simulator reads, applied in order, e.g. `pgs_truncation`; none: the state is passed as is.


## `history_match:prior`

`[prior]`: how the initial ensemble is generated.

`generator` applies to every state variable unless `[prior.variables.<name>]`
overrides it. Generators are plugins: `gaussian` (independent normals), `file`
(stored `.npz`), `gaussian_field` (variogram + Cholesky, PET's sampler) and
`geostatspy_sgsim` (GeostatsPy). Realisations are clipped to the state bounds.

| option | type | default | description |
|---|---|---|---|
| `ensemble_size` | integer or null | | Members; optional when every variable is read from a file. |
| `generator` | string or PluginSection | | |
| `groups` | object | | Variables drawn together by one joint generator, e.g. the latent fields of a plurigaussian simulation (`gstools_pgs`). |
| `variables` | object | | |

**`PluginSection`**: A registry name plus keyword options for its factory.

`diagnostics`, for plugins that produce diagnostics (the method, its
strategy and localization), switches this plugin's items on or off and **overrides**
`[output] diagnostics` for it: `{ taper = true }` saves the taper even if
`[output]` does not list it, `{ taper = false }` suppresses it even if it does.
`max_mb` and `every` override `[output] diagnostics_max_mb` and
`diagnostics_every` for this plugin. The run logs the resulting plan and warns when
a plugin switch contradicts `[output]`.

| option | type | default | description |
|---|---|---|---|
| `diagnostics` | object or null | | Per-plugin diagnostics switches (name = true/false, max_mb, every); they override [output] diagnostics for this plugin. |
| `name` | string | required | |
| `options` | object | | |

**`PriorGroupSection`**: `[prior.groups.<name>]`: a joint generator and the state variables it draws.

| option | type | default | description |
|---|---|---|---|
| `diagnostics` | object or null | | Per-plugin diagnostics switches (name = true/false, max_mb, every); they override [output] diagnostics for this plugin. |
| `name` | string | required | |
| `options` | object | | |
| `variables` | array | required | |

## `history_match:run`

`[run]`: identity and reproducibility.

| option | type | default | description |
|---|---|---|---|
| `name` | string | required | |
| `output_dir` | string or null | | Run directory; default `runs/<name>` next to the config. |
| `seed` | integer | required | Root seed for all random streams. |

## `history_match:simulator`

A registry name plus keyword options for its factory.

`diagnostics`, for plugins that produce diagnostics (the method, its
strategy and localization), switches this plugin's items on or off and **overrides**
`[output] diagnostics` for it: `{ taper = true }` saves the taper even if
`[output]` does not list it, `{ taper = false }` suppresses it even if it does.
`max_mb` and `every` override `[output] diagnostics_max_mb` and
`diagnostics_every` for this plugin. The run logs the resulting plan and warns when
a plugin switch contradicts `[output]`.

| option | type | default | description |
|---|---|---|---|
| `diagnostics` | object or null | | Per-plugin diagnostics switches (name = true/false, max_mb, every); they override [output] diagnostics for this plugin. |
| `name` | string | required | |
| `options` | object | | |

## `history_match:state`

`[state]`.

| option | type | default | description |
|---|---|---|---|
| `variables` | array | required | |

**`StateVariableSection`**: `[[state.variables]]`: one state block.

| option | type | default | description |
|---|---|---|---|
| `grid` | array or null | | Field shape [nx, ny, nz] (Eclipse cell order) when the variable is a grid field; used by SSIM and parameter maps. |
| `lower` | number or null | | |
| `name` | string | required | |
| `size` | integer | required | |
| `upper` | number or null | | |

## `history_match:validation`

`[validation]`: cross validation (CV) and test data for iterative methods.

`cv` data are forecast at every iteration but never assimilated; their mean-square
normalized data mismatch picks the posterior (the iterate where it is lowest) and, with the
`patience` rule, stops the run early. Only iterative methods (e.g. GIES) take CV;
ES-MDA's fixed schedule does not. `test` data are only scored, to check that CV
works. `history_matching` defaults to every datum in neither. **Field data (e.g.
seismic) are always history-matching data** and cannot be selected for cv or test. …

| option | type | default | description |
|---|---|---|---|
| `cv` | DataSelection or null | | The cross-validation data: forecast at every iteration, never assimilated, their mismatch picks the posterior. A data selection, e.g. { wells = ["PRO3"] } or { keywords = ["WWPR"] }. |
| `history_matching` | DataSelection or null | | The data assimilated (a data selection: wells, datatypes, keywords, times); default every scalar datum in neither cv nor test. |
| `stopping` | string or PluginSection or null | | CV stopping rule; default { name = "patience", options = { patience = 2 } }, or "select" (short for { name = "select" }: run to the usual criteria). Needs cv. |
| `test` | DataSelection or null | | Data only scored, never assimilated nor used to stop (a data selection, as cv), to check that cross validation works. |

**`DataSelection`**: Which scalar data a role of `[validation]` takes.

A datum `KEYWORD:WELL` at a report time is selected if it matches any of `wells`,
`datatypes` and `keywords` (and one of `times`, when given). A datatype without
`:WELL` is its own well for `wells`. At least one of `wells`, `datatypes` and
`keywords` must be given.

| option | type | default | description |
|---|---|---|---|
| `datatypes` | array | | Exact datatypes, e.g. ['WOPR:PRO3']. |
| `keywords` | array | | Keywords for every well, e.g. ['WWPR']. |
| `times` | array or null | | Restrict the selection to these times, as the observations name them. |
| `wells` | array | | Wells, e.g. ['PRO3']. |

**`PluginSection`**: A registry name plus keyword options for its factory.

`diagnostics`, for plugins that produce diagnostics (the method, its
strategy and localization), switches this plugin's items on or off and **overrides**
`[output] diagnostics` for it: `{ taper = true }` saves the taper even if
`[output]` does not list it, `{ taper = false }` suppresses it even if it does.
`max_mb` and `every` override `[output] diagnostics_max_mb` and
`diagnostics_every` for this plugin. The run logs the resulting plan and warns when
a plugin switch contradicts `[output]`.

| option | type | default | description |
|---|---|---|---|
| `diagnostics` | object or null | | Per-plugin diagnostics switches (name = true/false, max_mb, every); they override [output] diagnostics for this plugin. |
| `name` | string | required | |
| `options` | object | | |

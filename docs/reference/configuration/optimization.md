---
audience: public
generated: reference
---

# Configuring optimization: reference

Every one of these 10 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `optimization:controls`

`[controls]`: the variables and the model realizations of robust optimization.

| option | type | default | description |
|---|---|---|---|
| `realizations` | integer | 1 | Model realizations (PET's num_models) the objective averages. |
| `variables` | array | required | |

**`ControlVariable`**: `[[controls.variables]]`: one named block of controls.

| option | type | default | description |
|---|---|---|---|
| `initial` | number or array | required | Initial value(s) of the block. |
| `lower` | number or null | | Lower bound of every value. |
| `name` | string | required | |
| `upper` | number or null | | Upper bound of every value. |
| `variance` | number or null | | Initial perturbation variance (needed by the ensemble optimizers). |

## `optimization:execution`

Where the simulations run (with [simulator]).

| option | type | default | description |
|---|---|---|---|
| `diagnostics` | object or null | | Per-plugin diagnostics switches (name = true/false, max_mb, every); they override [output] diagnostics for this plugin. |
| `name` | string | required | |
| `options` | object | | |

## `optimization:model`

`[model]`: what every simulation reads besides the controls.

| option | type | default | description |
|---|---|---|---|
| `realizations_file` | string or null | | An .npz of model arrays (n_values, n_realizations) by name, e.g. permx: column r goes into the simulations of realization r (robust optimization; sets [controls] realizations). |

## `optimization:objective`

The objective: one plugin (a scalar one, or one that returns several, such as the test problem zdt1). Give either this or [[objectives]].

| option | type | default | description |
|---|---|---|---|
| `diagnostics` | object or null | | Per-plugin diagnostics switches (name = true/false, max_mb, every); they override [output] diagnostics for this plugin. |
| `name` | string | required | |
| `options` | object | | |

## `optimization:objectives`

Several scalar objectives optimized together (e.g. by nsga2): each a plugin with options and an optional label. Response objectives share one simulation per member.


## `optimization:optimization`

`[optimization]`: identity and reproducibility.

| option | type | default | description |
|---|---|---|---|
| `name` | string | required | |
| `output_dir` | string or null | | Run directory; default `runs/<name>` next to the config. |
| `seed` | integer | required | Root seed of the perturbations' random stream. |

## `optimization:optimizer`

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

## `optimization:output`

`[output]`: what an optimization keeps besides its history.

| option | type | default | description |
|---|---|---|---|
| `keep_simulations` | string | failed | Simulation folders kept in simulations/: all, the failed ones, or none. |

## `optimization:parameterizations`

Maps from the controls to what the simulator reads, applied in order (control_schedule, control_scaling, ...); none: the controls are passed as is.


## `optimization:simulator`

The simulator of a response objective such as npv.

| option | type | default | description |
|---|---|---|---|
| `diagnostics` | object or null | | Per-plugin diagnostics switches (name = true/false, max_mb, every); they override [output] diagnostics for this plugin. |
| `name` | string | required | |
| `options` | object | | |

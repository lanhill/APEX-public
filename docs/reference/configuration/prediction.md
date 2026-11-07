---
audience: public
generated: reference
---

# Configuring prediction: reference

Every one of these 5 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `prediction:controls`

A control scenario every member is given.

| option | type | default | description |
|---|---|---|---|
| `parameterizations` | array | | Parameterizations of the controls only (e.g. control_schedule), in order. |
| `variables` | array | required | |

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

**`PredictionControlVariable`**: `[[controls.variables]]` of a prediction: one named block of fixed controls.

| option | type | default | description |
|---|---|---|---|
| `lower` | number or null | | Lower bound of every value. |
| `name` | string | required | |
| `upper` | number or null | | Upper bound of every value. |
| `values` | number or array | required | The block's value(s), one per control. |

## `prediction:execution`

Where members run; default: the run's [execution].

| option | type | default | description |
|---|---|---|---|
| `diagnostics` | object or null | | Per-plugin diagnostics switches (name = true/false, max_mb, every); they override [output] diagnostics for this plugin. |
| `name` | string | required | |
| `options` | object | | |

## `prediction:prediction`

`[prediction]`.

| option | type | default | description |
|---|---|---|---|
| `check_history` | boolean | true | Fail unless the forecast reproduces the run's stored forecasts over the history period. |
| `ensembles` | array | | 'posterior', 'prior', 'iteration:<k>' and, in synthetic cases, 'truth'. |
| `include_mean_model` | boolean | false | Also forecast each ensemble's mean model. |
| `name` | string | required | |
| `output_dir` | string or null | | Default: <run_dir>/predictions/<name>. |
| `outputs` | | required | |
| `percentile_convention` | string | percentile | 'percentile': p10 is the 10th percentile; 'exceedance': SPE-PRMS labels (P90 is the value exceeded with 90 % probability, the 10th percentile). |
| `run_dir` | string | required | The history-matching run to forecast from. |
| `scores` | array | | Forecast scores against the truth (synthetic cases, with 'truth'). |

**`PredictionOutputsSection`**: `[prediction.outputs]`: which series to forecast.

| option | type | default | description |
|---|---|---|---|
| `times` | string or array | report | 'report' (every report time) or a list of ISO dates. |
| `vectors` | array | required | Summary vectors or shell-style patterns, e.g. 'WOPR:*'. |

## `prediction:simulator`

`[simulator]` of a prediction: changes to the run's simulator.

Without `name`, `options` update the run's simulator options (e.g. another deck
template). With another `name`, the simulator is replaced and takes only `options`.

| option | type | default | description |
|---|---|---|---|
| `name` | string or null | | |
| `options` | object | | |

## `prediction:surrogate`

Forecast the members with a fitted surrogate.

| option | type | default | description |
|---|---|---|---|
| `model_file` | string | required | The surrogate's manifest (surrogate.json). |
| `outside` | string | stop | A member outside the model's domain stops the prediction, or is run by the simulator. |
| `workers` | integer or null | | Process workers for the surrogate (None: serial). |

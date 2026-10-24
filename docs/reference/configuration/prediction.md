---
audience: public
generated: reference
---

# Configuring prediction: reference

Every one of these 3 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

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

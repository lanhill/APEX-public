---
audience: public
generated: reference
---

# Figures: reference

Every one of these 6 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `data_bands`

Per data type and well, the initial and final ensemble forecasts as percentile bands.

| option | type | default | description |
|---|---|---|---|
| `datatypes` | array | | Only these data types (`WOPR:BR-P-1`; default all scalar ones). |
| `observations` | boolean or null | | Draw the observations (default: for `predicted` only). |
| `source` | string | predicted | `predicted`: the forecasts of the observations; `auxiliary`: the forecasts saved beside them but never assimilated (a model-error correction's residuals, MEC). |

## `field_layers`

Per gridded state variable and layer: the true, prior, posterior, change and std maps.

| option | type | default | description |
|---|---|---|---|
| `exponentiate` | array or null | | Variables shown as `exp` of their values (ln-permeabilities; default those whose name contains `perm`, as the reference). The means are of the exponentiated members. |
| `kinds` | array | ["ref", "prior", "post", "diff", "std_prior", "std_post"] | The maps to draw (default: all six; `ref` needs the true model). |
| `layers` | array | | 1-based layers (default: all). |
| `variables` | array | | The variables to draw (default: every one with a grid). |

## `field_maps`

Per geophysical field datum and vintage: observed, prior, posterior and mismatch maps.

| option | type | default | description |
|---|---|---|---|
| `datatypes` | array | | The field data to draw (default: every one with a map). |
| `kinds` | array | ["obs", "prior", "post", "mismatch"] | The maps to draw (default: all four). |

## `metric_distribution`

Box and violin plots of a per-member metric over the accepted iterates.

| option | type | default | description |
|---|---|---|---|
| `font_size` | number | 28.0 | Tick and axis label size. |
| `kinds` | array | ["violin", "box"] | `violin` and/or `box`, in the order drawn. |
| `labels` | array | | Its labels to draw (default: every one, `all` last). |
| `metric` | string | required | The recorded metric (`rmse`, `ssim`, `model_change`, `normalized_data_mismatch`, ...). |
| `prefix` | string | | The file prefix and y-label word (default: the metric): files `<prefix>_<label>_violinplot` and `<prefix>_<label>_boxplot`, y label `<prefix> (<label>)`; a label `TYPE:WELL` is written `TYPE WELL`. |

## `objective_distribution`

The objective over all data, as box and violin plots.

| option | type | default | description |
|---|---|---|---|
| `font_size` | number | 30.0 | Tick and axis label size. |
| `metric` | string | data_mismatch | The objective's metric. |

## `state_histogram`

A state variable's values at the initial and the final iterate, one histogram per member.

| option | type | default | description |
|---|---|---|---|
| `bins` | integer | 50 | Number of shared bins. |
| `font_size` | number | 16.0 | Title, axis label and tick size. |
| `prefix` | string | hist | The file prefix. |
| `variable` | string | required | The state variable (e.g. `nn_params`, a model-error network's parameters). |

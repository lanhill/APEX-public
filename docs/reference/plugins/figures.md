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
| `datatypes` | array | | |
| `observations` | boolean or null | | |
| `source` | string | predicted | |

## `field_layers`

Per gridded state variable and layer: the true, prior, posterior, change and std maps.

| option | type | default | description |
|---|---|---|---|
| `exponentiate` | array or null | | |
| `kinds` | array | ["ref", "prior", "post", "diff", "std_prior", "std_post"] | |
| `layers` | array | | |
| `variables` | array | | |

## `field_maps`

Per geophysical field datum and vintage: observed, prior, posterior and mismatch maps.

| option | type | default | description |
|---|---|---|---|
| `datatypes` | array | | |
| `kinds` | array | ["obs", "prior", "post", "mismatch"] | |

## `metric_distribution`

Box and violin plots of a per-member metric over the accepted iterates.

| option | type | default | description |
|---|---|---|---|
| `font_size` | number | 28.0 | |
| `kinds` | array | ["violin", "box"] | |
| `labels` | array | | |
| `metric` | string | required | |
| `prefix` | string | | |

## `objective_distribution`

The objective over all data, as box and violin plots.

| option | type | default | description |
|---|---|---|---|
| `font_size` | number | 30.0 | |
| `metric` | string | data_mismatch | |

## `state_histogram`

A state variable's values at the initial and the final iterate, one histogram per member.

| option | type | default | description |
|---|---|---|---|
| `bins` | integer | 50 | |
| `font_size` | number | 16.0 | |
| `prefix` | string | hist | |
| `variable` | string | required | |

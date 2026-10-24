---
audience: public
generated: reference
---

# Parameterizations: reference

Every one of these 4 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `control_scaling`

Normalized controls in [0, 1] mapped linearly onto their physical bounds.

Select it with:

```toml
[[parameterizations]]
name = "control_scaling"
```

| option | type | default | description |
|---|---|---|---|
| `lower` | number or array | required | |
| `output` | string or null | | |
| `upper` | number or array | required | |
| `variable` | string | required | |

## `control_schedule`

A control block's values spread over the report steps of its periods.

Select it with:

```toml
[[parameterizations]]
name = "control_schedule"
```

| option | type | default | description |
|---|---|---|---|
| `interpolation` | string | hold | |
| `knots` | array | required | |
| `n_steps` | integer | required | |
| `output` | string or null | | |
| `variable` | string | required | |

## `facies_distance`

Facies from signed distance fields (`argmax`), and properties by facies.

Select it with:

```toml
[[parameterizations]]
name = "facies_distance"
```

| option | type | default | description |
|---|---|---|---|
| `facies` | array | required | |
| `fields` | array | required | |
| `property_fields` | object or null | | |
| `values` | object or null | | |

## `pgs_truncation`

Facies from latent fields by a fixed truncation rule, and properties by facies.

Select it with:

```toml
[[parameterizations]]
name = "pgs_truncation"
```

| option | type | default | description |
|---|---|---|---|
| `extent` | number | 3.0 | |
| `facies` | array | required | |
| `fields` | array | required | |
| `property_fields` | object or null | | |
| `proportions` | array or null | | |
| `rule_file` | string or string or null | | |
| `values` | object or null | | |

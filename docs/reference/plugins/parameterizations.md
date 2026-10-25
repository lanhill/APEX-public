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
| `lower` | number or array | required | The physical value at 0 and at 1: scalars, or one per value of the block. |
| `output` | string or null | | The name the simulation reads (default: `variable`). |
| `upper` | number or array | required | The physical value at 0 and at 1: scalars, or one per value of the block. |
| `variable` | string | required | The normalized control block. |

## `control_schedule`

A control block's values spread over the report steps of its periods.

Select it with:

```toml
[[parameterizations]]
name = "control_schedule"
```

| option | type | default | description |
|---|---|---|---|
| `interpolation` | string | hold | `hold` (piecewise constant) or `linear`. |
| `knots` | array | required | The first report step of each value (0-based, increasing, starting at 0). |
| `n_steps` | integer | required | The report steps of the schedule. |
| `output` | string or null | | The name the simulation reads, `(n_steps,)` (default: `variable`). |
| `variable` | string | required | The control block (or an earlier parameterization's output) of `len(knots)` values. |

## `facies_distance`

Facies from signed distance fields (`argmax`), and properties by facies.

Select it with:

```toml
[[parameterizations]]
name = "facies_distance"
```

| option | type | default | description |
|---|---|---|---|
| `facies` | array | required | Facies names. |
| `fields` | array | required | One distance state variable per facies, in code order (`fields[k]` is facies `facies[k]`'s). |
| `property_fields` | object or null | | Properties by facies, as for `pgs_truncation`. |
| `values` | object or null | | Properties by facies, as for `pgs_truncation`. |

## `pgs_truncation`

Facies from latent fields by a fixed truncation rule, and properties by facies.

Select it with:

```toml
[[parameterizations]]
name = "pgs_truncation"
```

| option | type | default | description |
|---|---|---|---|
| `extent` | number | 3.0 | A lithotype array (`.npy` of integer codes, one axis per field) laid over `[-extent, extent]` standard deviations on each axis (default 3; edge bins extend to infinity), or, for one field, the facies' target `proportions` (a truncated Gaussian). |
| `facies` | array | required | Facies names; facies code `k` is `facies[k]`. |
| `fields` | array | required | The latent state variables, one per rule axis, in order. |
| `property_fields` | object or null | | `{property: {facies: state variable}}`: properties taken cell by cell from a state variable of the facies there, e.g. `{permx = {sand = "permx_sand", ...}}` (updated with the latent fields). A property is in `values` or here, not both. |
| `proportions` | array or null | | A lithotype array (`.npy` of integer codes, one axis per field) laid over `[-extent, extent]` standard deviations on each axis (default 3; edge bins extend to infinity), or, for one field, the facies' target `proportions` (a truncated Gaussian). |
| `rule_file` | string or string or null | | A lithotype array (`.npy` of integer codes, one axis per field) laid over `[-extent, extent]` standard deviations on each axis (default 3; edge bins extend to infinity), or, for one field, the facies' target `proportions` (a truncated Gaussian). |
| `values` | object or null | | `{facies: {property: value}}`: constant properties per facies, e.g. `{sand = {poro = 0.25, permx = 500}, shale = {poro = 0.05, permx = 1}}`. |

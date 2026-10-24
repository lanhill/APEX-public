---
audience: public
generated: reference
---

# Localizations: reference

Every one of these 6 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `adaptive`

Adaptive localization with the taper built once from the prior (PET `ini_only`).

Select it with:

```toml
[method.localization]
name = "adaptive"
```

| option | type | default | description |
|---|---|---|---|
| `cutoff` | number or null | | |
| `taper` | string | hard | |
| `threshold` | string | universal | |
| `threshold_scale` | number | 1.0 | |

## `adaptive_per_update`

Automatic and adaptive localization, recomputed at every update (PET `autoadaloc`).

Select it with:

```toml
[method.localization]
name = "adaptive_per_update"
```

| option | type | default | description |
|---|---|---|---|
| `cutoff` | number or null | | |
| `taper` | string | hard | |
| `threshold` | string | universal | |
| `threshold_scale` | number | 1.0 | |

## `adaptive_prior`

An alias of 'adaptive'. Adaptive localization with the taper built once from the prior (PET `ini_only`).

Select it with:

```toml
[method.localization]
name = "adaptive_prior"
```

| option | type | default | description |
|---|---|---|---|
| `cutoff` | number or null | | |
| `taper` | string | hard | |
| `threshold` | string | universal | |
| `threshold_scale` | number | 1.0 | |

## `adaptive_projected`

Upstream PET's adaptive localization (`autoadaloc`) in the projected-data space.

Select it with:

```toml
[method.localization]
name = "adaptive_projected"
```

| option | type | default | description |
|---|---|---|---|
| `cutoff` | number | 0.3 | |
| `taper` | string | hard | |
| `threshold` | string | adaptive | |

## `distance`

Distance-based localization around well locations (PET `dist_loc`).

Select it with:

```toml
[method.localization]
name = "distance"
```

| option | type | default | description |
|---|---|---|---|
| `anisotropy` | | [1.0, 0.0] | |
| `grid` | array | required | |
| `positions` | object | required | |
| `radius` | number | required | |
| `taper` | string | gaspari_cohn | |
| `unlisted` | string | error | |

## `distance_entries`

Upstream PET's distance localization (`distance_loc`), one entry per data and variable.

Select it with:

```toml
[method.localization]
name = "distance_entries"
```

| option | type | default | description |
|---|---|---|---|
| `actnum_file` | string or string or null | | |
| `entries` | array | required | |
| `grid` | array | required | |

**`Entry`**: One localization entry (PET: one row of its entries CSV, or one dict of `entries`).

`data`, `time` and `parameter` name what it localizes (`"*"`: all, the default);
`taper` is the kernel. `position` `[i, j, k]` is the centre cell, 1-based as in
Eclipse decks; `radius` the kernel's length in cells; `z_range` `"all"` (PET
`":"`, the default) puts the kernel in every layer, an integer `r` in layers
``k - r .. …

| option | type | default | description |
|---|---|---|---|
| `anisotropy` | array | | |
| `data` | string | | |
| `parameter` | string | | |
| `position` | array | | |
| `radius` | number | | |
| `taper` | string | | |
| `taper_file` | string | | |
| `time` | string | | |
| `z_range` | string or integer | | |

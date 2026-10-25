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
| `cutoff` | number or null | | The fixed threshold on <code>&#124;correlation&#124;</code>, in `(0, 1)`; by default `3/sqrt(N)` for `N` members (0.3 at 100), three standard deviations of a sample correlation where there is none. Only with `threshold = "fixed"`. |
| `taper` | string | hard | How correlations are kept: `"hard"` (default) keeps those above the threshold and drops the rest, `"soft"` scales them down smoothly with Gaspari-Cohn, and `"sigmoid"` uses a steep logistic step. |
| `threshold` | string | universal | `"universal"` (default): a threshold per state variable and datum, estimated from correlations with a shuffled ensemble and growing with the variable's size; `"fixed"`: `cutoff` everywhere. With many cells and few members the universal threshold can keep almost nothing; a fixed one does not. |
| `threshold_scale` | number | 1.0 | Multiplier of the universal threshold (positive, default 1); larger keeps fewer correlations. Not with `threshold = "fixed"`. |

## `adaptive_per_update`

Automatic and adaptive localization, recomputed at every update (PET `autoadaloc`).

Select it with:

```toml
[method.localization]
name = "adaptive_per_update"
```

| option | type | default | description |
|---|---|---|---|
| `cutoff` | number or null | | The fixed threshold, in `(0, 1)`; by default `3/sqrt(N)` for `N` members, three standard deviations of a sample correlation where there is none (Vossepoel, Evensen & van Leeuwen 2025, eq. 6; 0.3 at 100). Only with `threshold = "fixed"`, which also takes no `threshold_scale`. The `hard` taper keeps exactly <code>&#124;rho&#124; &gt; cutoff</code>; `soft` (Gaspari-Cohn of <code>(1 - &#124;rho&#124;) / (1 - cutoff)</code>) reaches 0 only at <code>&#124;rho&#124; = 2 cutoff - 1</code>, so below 0.5 it keeps a little of every correlation. |
| `taper` | string | hard | `"hard"` (default), `"soft"` or `"sigmoid"` (PET `type`). |
| `threshold` | string | universal | `"universal"` (default): `threshold_scale * sqrt(2 ln n_block) * sigma` from the shuffled correlations; `"fixed"`: `cutoff` for every datum and cell. The universal threshold grows with the block's size, so with many cells and few members it can keep almost nothing (0.45 for 18,000 cells and 100 members); a fixed one does not. |
| `threshold_scale` | number | 1.0 | Multiplier of the universal threshold (PET `autoadaloc` value, `nstd`). |

## `adaptive_prior`

An alias of 'adaptive'. Adaptive localization with the taper built once from the prior (PET `ini_only`).

Select it with:

```toml
[method.localization]
name = "adaptive_prior"
```

| option | type | default | description |
|---|---|---|---|
| `cutoff` | number or null | | The fixed threshold on <code>&#124;correlation&#124;</code>, in `(0, 1)`; by default `3/sqrt(N)` for `N` members (0.3 at 100), three standard deviations of a sample correlation where there is none. Only with `threshold = "fixed"`. |
| `taper` | string | hard | How correlations are kept: `"hard"` (default) keeps those above the threshold and drops the rest, `"soft"` scales them down smoothly with Gaspari-Cohn, and `"sigmoid"` uses a steep logistic step. |
| `threshold` | string | universal | `"universal"` (default): a threshold per state variable and datum, estimated from correlations with a shuffled ensemble and growing with the variable's size; `"fixed"`: `cutoff` everywhere. With many cells and few members the universal threshold can keep almost nothing; a fixed one does not. |
| `threshold_scale` | number | 1.0 | Multiplier of the universal threshold (positive, default 1); larger keeps fewer correlations. Not with `threshold = "fixed"`. |

## `adaptive_projected`

Upstream PET's adaptive localization (`autoadaloc`) in the projected-data space.

Select it with:

```toml
[method.localization]
name = "adaptive_projected"
```

| option | type | default | description |
|---|---|---|---|
| `cutoff` | number | 0.3 | The multiplier of `sigma` (`"adaptive"`) or the threshold itself (`"fixed"`, in `(0, 1)`); PET `cutoff` (also `nstd`), default 0.3. Unused by `"universal"`. |
| `taper` | string | hard | `"hard"` (default), `"soft"` or `"sigmoid"` (PET `type`: `hard`, `soft`, `sigm`). |
| `threshold` | string | adaptive | `"adaptive"` (default): `cutoff * sigma`; `"fixed"`: `cutoff`; `"universal"`: `sqrt(2 ln n_block) * sigma` (PET `threshold`). |

## `distance`

Distance-based localization around well locations (PET `dist_loc`).

Select it with:

```toml
[method.localization]
name = "distance"
```

| option | type | default | description |
|---|---|---|---|
| `anisotropy` | | [1.0, 0.0] | `[ratio, angle in degrees]`; `[1, 0]` is isotropic. |
| `grid` | array | required | `[nx, ny, nz]`; each state variable is one grid property in Eclipse order (`i` fastest), so every block size must equal `nx * ny * nz`. |
| `positions` | object | required | Well name to completion cells `[[i, j, k], ...]`, 1-based as in Eclipse decks. |
| `radius` | number | required | Localization length `c` in cells (the taper vanishes beyond `2c` for Gaspari-Cohn, beyond `c` for Furrer-Bengtsson). |
| `taper` | string | gaspari_cohn | `"gaspari_cohn"` (PET `gc`) or `"furrer_bengtsson"` (PET `fb`: spherical correlation `rho` turned into `N rho^2 / ((N + 1) rho^2 + 1)`). |
| `unlisted` | string | error | What to do with data of wells missing from `positions`: `"error"` (default) refuses the run, so a misspelt well cannot silently drop its data; `"zero"` leaves the state untouched by them. |

## `distance_entries`

Upstream PET's distance localization (`distance_loc`), one entry per data and variable.

Select it with:

```toml
[method.localization]
name = "distance_entries"
```

| option | type | default | description |
|---|---|---|---|
| `actnum_file` | string or null | | Optional `.npy`/`.npz` of active cells (one boolean per grid cell, Eclipse order); the tapers are then taken over the active cells (PET `actnum`). |
| `entries` | array | required | The entries, in order: a later entry replaces an earlier one for the (data, time, variable) it names, as in PET. |
| `grid` | array | required | `[nx, ny, nz]`; each state variable named by an entry is a grid property in Eclipse order (`i` fastest) over all cells, or over the active cells of `actnum_file`. |

**`Entry`**: One localization entry (PET: one row of its entries CSV, or one dict of `entries`).

Parameters
----------
data
 The data type it localizes (`"*"`: all, the default).
time
 The data time it localizes (`"*"`: all, the default).
parameter
 The state variable it localizes (`"*"`: all, the default).
taper
 The kernel: `gaspari_cohn`, `furrer_bengtsson`, `region`, or `import` (read
 from `taper_file`).
position
 The centre cell `[i, j, k]`, 1-based as in Eclipse decks.
radius
 The kernel's length, in cells. …

| option | type | default | description |
|---|---|---|---|
| `anisotropy` | array | | `[ratio, angle in degrees]`: the rotated offset's first (`j`) component is divided by the ratio, so a ratio of 2 doubles the reach along `j` (PET's docstring says it halves it; its code divides, as here). |
| `data` | string | | The data type it localizes (`"*"`: all, the default). |
| `parameter` | string | | The state variable it localizes (`"*"`: all, the default). |
| `position` | array | | The centre cell `[i, j, k]`, 1-based as in Eclipse decks. |
| `radius` | number | | The kernel's length, in cells. |
| `taper` | string | | The kernel: `gaspari_cohn`, `furrer_bengtsson`, `region`, or `import` (read from `taper_file`). |
| `taper_file` | string | | The `import` kernel's taper (`.npy`/`.npz`: one value per grid cell, flat in Eclipse order or shaped `(nx, ny, nz)`), relative to the configuration's folder. |
| `time` | string | | The data time it localizes (`"*"`: all, the default). |
| `z_range` | string or integer | | `"all"` (PET `":"`, the default) puts the kernel in every layer; an integer `r` in layers `k - r .. k + r`. |

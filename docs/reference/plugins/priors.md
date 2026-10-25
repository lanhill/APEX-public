---
audience: public
generated: reference
---

# Priors: reference

Every one of these 6 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `file`

Read a stored ensemble `(size, n_ensemble)` from an `.npz` file.

Select it with:

```toml
[prior.generator]
name = "file"
```

| option | type | default | description |
|---|---|---|---|
| `ensemble_file` | string | required | `.npz` with one array per variable. |
| `key` | string or null | | Array name; defaults to the variable's name. |

## `gaussian`

Independent normals `N(mean, std^2)` for every entry (no spatial correlation).

Select it with:

```toml
[prior.generator]
name = "gaussian"
```

| option | type | default | description |
|---|---|---|---|
| `mean` | | 0.0 | Scalar, or one value per state entry. |
| `std` | number | 1.0 | Standard deviation, positive. |

## `gaussian_field`

Gaussian random field from a variogram, sampled layer by layer (PET's prior sampler).

Select it with:

```toml
[prior.generator]
name = "gaussian_field"
```

| option | type | default | description |
|---|---|---|---|
| `azimuth` | number or array | 0.0 | Direction of the major axis, degrees counter-clockwise from the i-axis, scalar or per layer. |
| `conditioning` | object or null | | Hard data, `{cells = [[i, j, k], ...], values = [...], error_std = 0}`: 1-based Eclipse cells and the values there, in the state's space (ln of a log variable's values). Every realisation is conditioned to them by kriging its residuals with the same variogram (exact for this generator); with `error_std = 0` it takes the values at their cells, with a positive one (a number, or one per value) it is drawn around them. Values outside the variable's bounds are clipped as the rest. Data improbable under the variogram (rough values close together under a smooth model) are honoured too, with the field swinging far outside the prior: that is logged as a warning, not refused (the geomodeler refuses such data). |
| `exponent` | number or array | 1.0 | Exponent of the `stable` model. |
| `grid` | array | required | `[nx, ny, nz]`; `nx * ny * nz` must equal the variable's size. |
| `major_range` | number or array | 10.0 | Ranges in cells along and across the major axis, scalar or per layer; `minor_range` defaults to `major_range`. A range of 0 means uncorrelated. |
| `mean` | | 0.0 | Scalar, or one value per cell (e.g. PET's `priormean.npz`, loaded by the composition from `mean_file`). |
| `minor_range` | number or array or null | | Ranges in cells along and across the major axis, scalar or per layer; `minor_range` defaults to `major_range`. A range of 0 means uncorrelated. |
| `variance` | number or array | 1.0 | Sill, scalar or per layer. |
| `variogram` | string or array | spherical | One of `spherical`, `exponential`, `gaussian`, `cubic`, `stable`, scalar or per layer. PET's `sph`/`cub` are `spherical`/`cubic`; PET's `exp` is `stable` with `exponent = 1.9`. |

## `geostatspy_sgsim`

Unconditional sequential Gaussian simulation with GeostatsPy (`geostatspy` extra).

Select it with:

```toml
[prior.generator]
name = "geostatspy_sgsim"
```

| option | type | default | description |
|---|---|---|---|
| `azimuth` | number or array | 0.0 | Direction of the major axis, degrees counter-clockwise from the i-axis, scalar or per layer. |
| `conditioning` | object or null | | Hard data, `{cells = [[i, j, k], ...], values = [...], error_std = 0}`: 1-based Eclipse cells and the values there, in the state's space. The realisations are conditioned by kriging their residuals with the variogram, not by SGSIM's own search, so that this generator and `gaussian_field` honour data alike. SGSIM's fields have that covariance only approximately (its search neighbourhood), so the conditioning is approximate away from the data; at the data's cells the values are honoured exactly (`error_std = 0`), or drawn around them with a positive one. |
| `grid` | array | required | `[nx, ny, nz]`, all positive; `nx * ny * nz` must equal the variable's size. |
| `major_range` | number or array | 10.0 | Ranges in cells along and across the major axis, scalar or per layer; both must be positive with `major_range >= minor_range` (GSLIB's convention). `minor_range` defaults to `major_range` (an isotropic field). |
| `max_neighbours` | integer | 20 | Number of previously simulated nodes SGSIM conditions each node on (at least 1, default 20), searched within the major range. More neighbours reproduce the variogram more closely at a higher cost. |
| `mean` | | 0.0 | Scalar, or one value per cell in Eclipse order (e.g. loaded from `mean_file`). |
| `minor_range` | number or array or null | | Ranges in cells along and across the major axis, scalar or per layer; both must be positive with `major_range >= minor_range` (GSLIB's convention). `minor_range` defaults to `major_range` (an isotropic field). |
| `variance` | number or array | 1.0 | Sill of the field, scalar or per layer; the standardised SGSIM field is scaled by its square root. |
| `variogram` | string or array | spherical | One of `spherical` (default), `exponential`, `gaussian`, scalar or per layer. |

## `gstools_pgs`

The latent fields of a plurigaussian simulation, optionally conditioned to well facies.

Select it with:

```toml
[prior.generator]
name = "gstools_pgs"
```

| option | type | default | description |
|---|---|---|---|
| `cell_size` | array | [1.0, 1.0, 1.0] | `[dx, dy, dz]`, the cell size in the units of `len_scale`; default 1. |
| `conditioning` | object or null | | `{cells = [[i, j, k], ...], facies = [code, ...]}`, 1-based Eclipse cells and the facies observed there; every realisation honours them. |
| `extent` | number | 3.0 | The truncation rule, as for `pgs_truncation` (the composition checks they match). |
| `fields` | array | required | One covariance model per latent field, in the order of the group's variables: `{model = "Gaussian", len_scale = 10 or [10, 5(2)], angles = 0 or [...]}` (GSTools' `CovModel` arguments; variance 1). `model` is one of `MODELS`. |
| `gibbs_sweeps` | integer | 100 | Iterations of the conditioning sampler (exact HMC and a coordinate sweep per field; `condition_latent`), default 100. |
| `proportions` | array or null | | The truncation rule, as for `pgs_truncation` (the composition checks they match). |
| `rule_file` | string or null | | The truncation rule, as for `pgs_truncation` (the composition checks they match). |

## `mpslib_snesim`

SNESIM facies realisations (MPSlib), as one signed distance field per facies.

Select it with:

```toml
[prior.generator]
name = "mpslib_snesim"
```

| option | type | default | description |
|---|---|---|---|
| `cell_size` | array | [1.0, 1.0, 1.0] | `[dx, dy, dz]`, for the hard data's coordinates and the distances. |
| `conditioning` | object or null | | `{cells = [[i, j, k], ...], facies = [code, ...]}`: 1-based Eclipse cells and the facies observed there (hard data); every realisation honours them. |
| `distance_band` | number or null | | Clip the signed distances to `[-band, band]`, in the units of `cell_size` (default: unclipped; a facies absent from a layer is at minus the layer's diagonal). |
| `entropy_factor` | number | 4.0 | The preferential path's entropy factor. |
| `n_cond` | integer | 16 | Most conditioning data per cell (0: all). |
| `n_min_node_count` | integer | 0 | Least count of a pattern for its conditional probability (0: no limit). |
| `n_multiple_grids` | integer | 3 | Multiple grids (coarse to fine; MPSlib counts from 0). |
| `program` | string or null | | MPSlib's `mps_snesim_tree` (default: the one installed with `scikit-mps`). |
| `random_path` | string | random | `random` (1) or `preferential` (2, MPSlib's default, with `entropy_factor`). |
| `soft_probability_file` | string or null | | Soft data: a `.npy` grid of facies probabilities `(nx, ny, nz, K)` (e.g. from seismic), NaN in cells without soft data. MPSlib combines each cell's probabilities with the training image's conditional ones (`P ∝ P_TI · P_soft`), so they shift the facies rather than fix them, unless they are 0 or 1. MPSlib reads the first `K - 1` and takes the last as one minus their sum (each cell is renormalised before writing). With the `preferential` path MPSlib also copies them to nearby coarse-grid nodes; with `random` they act at their own cell only. A realisation whose facies has soft probability 0 somewhere is refused: where the soft data contradict the training image MPSlib returns a probability as the facies code (`SNESIM.cpp`, its conflict branch; right only by chance for two facies). Hard data take precedence. A pdf is certain or has no zeros (refused otherwise). Needs a program that reads every soft category (checked). |
| `template_size` | array | [8, 7, 1] | The search template `[tx, ty, tz]`. |
| `timeout_s` | number | 3600.0 | Seconds the program may take. |
| `training_image_file` | string | required | The training image: `.npy` codes `[i, j(k)]` or a GSLIB/EAS grid file. |

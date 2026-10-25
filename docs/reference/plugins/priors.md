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
| `ensemble_file` | string or string | required | |
| `key` | string or null | | |

## `gaussian`

Independent normals `N(mean, std^2)` for every entry (no spatial correlation).

Select it with:

```toml
[prior.generator]
name = "gaussian"
```

| option | type | default | description |
|---|---|---|---|
| `mean` | | 0.0 | |
| `std` | number | 1.0 | |

## `gaussian_field`

Gaussian random field from a variogram, sampled layer by layer (PET's prior sampler).

Select it with:

```toml
[prior.generator]
name = "gaussian_field"
```

| option | type | default | description |
|---|---|---|---|
| `azimuth` | number or array | 0.0 | |
| `conditioning` | object or null | | |
| `exponent` | number or array | 1.0 | |
| `grid` | array | required | |
| `major_range` | number or array | 10.0 | |
| `mean` | | 0.0 | |
| `minor_range` | number or array or null | | |
| `variance` | number or array | 1.0 | |
| `variogram` | string or array | spherical | |

## `geostatspy_sgsim`

Unconditional sequential Gaussian simulation with GeostatsPy (`geostatspy` extra).

Select it with:

```toml
[prior.generator]
name = "geostatspy_sgsim"
```

| option | type | default | description |
|---|---|---|---|
| `azimuth` | number or array | 0.0 | |
| `conditioning` | object or null | | |
| `grid` | array | required | |
| `major_range` | number or array | 10.0 | |
| `max_neighbours` | integer | 20 | |
| `mean` | | 0.0 | |
| `minor_range` | number or array or null | | |
| `variance` | number or array | 1.0 | |
| `variogram` | string or array | spherical | |

## `gstools_pgs`

The latent fields of a plurigaussian simulation, optionally conditioned to well facies.

Select it with:

```toml
[prior.generator]
name = "gstools_pgs"
```

| option | type | default | description |
|---|---|---|---|
| `cell_size` | array | [1.0, 1.0, 1.0] | |
| `conditioning` | object or null | | |
| `extent` | number | 3.0 | |
| `fields` | array | required | |
| `gibbs_sweeps` | integer | 100 | |
| `proportions` | array or null | | |
| `rule_file` | string or string or null | | |

## `mpslib_snesim`

SNESIM facies realisations (MPSlib), as one signed distance field per facies.

Select it with:

```toml
[prior.generator]
name = "mpslib_snesim"
```

| option | type | default | description |
|---|---|---|---|
| `cell_size` | array | [1.0, 1.0, 1.0] | |
| `conditioning` | object or null | | |
| `distance_band` | number or null | | |
| `entropy_factor` | number | 4.0 | |
| `n_cond` | integer | 16 | |
| `n_min_node_count` | integer | 0 | |
| `n_multiple_grids` | integer | 3 | |
| `program` | string or null | | |
| `random_path` | string | random | |
| `soft_probability_file` | string or string or null | | |
| `template_size` | array | [8, 7, 1] | |
| `timeout_s` | number | 3600.0 | |
| `training_image_file` | string or string | required | |

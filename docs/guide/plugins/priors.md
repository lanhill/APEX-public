---
audience: public
items:
  - cli:apex plugins
  - cli:apex resume
  - cli:apex run
  - cli:apex stop
  - config:history_match:prior
  - plugin:priors:file
  - plugin:priors:gaussian
  - plugin:priors:gaussian_field
  - plugin:priors:geostatspy_sgsim
  - plugin:priors:gstools_pgs
  - plugin:priors:mpslib_snesim
---

# Priors

## Goal

A prior plugin generates the initial ensemble of your model parameters. This page helps you choose one of the six prior plugins, shows the options each accepts, and gives a short example for the ones you are most likely to use:

- `file`
- `gaussian`
- `gaussian_field`
- `geostatspy_sgsim`
- `gstools_pgs`
- `mpslib_snesim`

`priors` is the name of the plugin registry that holds these plugins. It is not a section of your configuration. The prior is configured in the `[prior]` section of a `history_match` run configuration (`history_match:prior`).

## How the `[prior]` section works

- `generator` is the plugin used for every state variable. A variable listed under `[prior.variables.<name>]` uses its own plugin instead.
- `ensemble_size` is the number of members, an integer of at least 2. You can leave it out when every variable is read from a file.
- `[prior.groups.<name>]` draws several variables together with one joint generator. For example, it draws the latent fields of a plurigaussian simulation with `gstools_pgs`. A group has a `name` (the plugin), its `variables` (a list with at least one variable name) and its `options`.
- A plugin is written as a table with `name` (the plugin) and `options` (a table with the plugin's options). Options go under `options`, as in the examples below. A plugin section may also have a `diagnostics` table, which this page does not cover.
- Realisations are clipped to the bounds of the state.

## Prerequisites

- A `history_match` run configuration in TOML. This is the file you pass to `apex run`. This page covers only its prior part. The other sections of the configuration are described elsewhere in the guide.
- For `file`, a stored ensemble in an `.npz` file.
- For `gaussian_field` and `geostatspy_sgsim`, the grid size `[nx, ny, nz]` of the variable. The product `nx * ny * nz` must equal the variable's size.
- For `geostatspy_sgsim`, the `geostatspy` extra must be installed.
- For `mpslib_snesim`, a training image file.
- For `gstools_pgs`, one covariance model per latent field.

## Some terms

- **Variogram**: a description of how quickly values become less alike with distance.
- **Range**: the distance over which values stay correlated, measured in cells.
- **Facies**: discrete rock types, such as sand or shale, rather than a continuous value.
- **Conditioning**: forcing every realisation to honour observed values, such as well data.
- **Latent field**: a smooth underlying Gaussian field that a plurigaussian simulation turns into facies.
- **Signed distance field**: for each facies, a field of distances to that facies. `mpslib_snesim` returns one per facies.

## Choosing a plugin

| Plugin | What it produces | Required option |
|---|---|---|
| `file` | A stored ensemble of shape `(size, n_ensemble)` read from an `.npz` file. | `ensemble_file` |
| `gaussian` | Independent normals `N(mean, std^2)` for every entry, with no spatial correlation. | none |
| `gaussian_field` | A Gaussian random field from a variogram, sampled layer by layer. It is the prior sampler used by PET. | `grid` |
| `geostatspy_sgsim` | Sequential Gaussian simulation with GeostatsPy. | `grid` |
| `gstools_pgs` | The latent fields of a plurigaussian simulation, optionally conditioned to well facies. | `fields` |
| `mpslib_snesim` | SNESIM facies realisations (MPSlib), as one signed distance field per facies. | `training_image_file` |

How to decide:

- Use `file` when you already have an ensemble you want to reuse.
- Use `gaussian` when neighbouring entries need no correlation.
- Use `gaussian_field` or `geostatspy_sgsim` for smooth, spatially correlated continuous properties.
- Use `gstools_pgs` or `mpslib_snesim` when the property is facies.
- To honour observed values, use the `conditioning` option. All of `gaussian_field`, `geostatspy_sgsim`, `gstools_pgs` and `mpslib_snesim` have it, and its content differs, as described below.

## Steps

1. List the plugins and their options with `uv run apex plugins`. Add `--json` to get the full descriptions as JSON. Look for the plugins in the `priors` registry.
2. Pick the plugin from the table above.
3. In your `history_match` configuration, set `name` in `[prior.generator]` (or in a `[prior.variables.<name>]` section) to the plugin name, and put its settings in the `options` table.
4. Set the plugin's required option, if it has one. Every option you leave out keeps its default, listed below.
5. Check the spelling of option names. These plugins reject unknown options.
6. Start the run with `uv run apex run config.toml`, where `config.toml` is your run configuration. Useful flags are `--max-steps` (stop after that many analysis steps) and `--no-progress` (switch off progress output on standard error, which is on by default; `--progress` switches it on again).
7. To stop a running run at its next checkpoint, use `uv run apex stop <run_dir>`, where `<run_dir>` is the directory of the running run. Continue it later with `uv run apex resume <run_dir>`, which also accepts `--max-steps` and `--no-progress`.

## Plugin options

### `file`

| Option | Type | Default |
|---|---|---|
| `ensemble_file` | string or path | required |
| `key` | string or null | `null` |

`ensemble_file` is an `.npz` file with one array per variable. `key` is the name of the array to read. When it is `null`, the plugin reads the array named like the variable. Set `key` when the array in the file has a different name from the variable.

```toml
[prior]

[prior.generator]
name = "file"

[prior.generator.options]
ensemble_file = "prior_ensemble.npz"
key = "permx"
```

### `gaussian`

| Option | Type | Default |
|---|---|---|
| `mean` | number or array-like | `0.0` |
| `std` | number (positive) | `1.0` |

`mean` is a scalar or one value per state entry. Every entry is drawn independently from a normal distribution with this mean and `std`. The number of entries is the size of the state variable, and `ensemble_size` sets the number of members. There is no `grid` option, because no spatial structure is used.

```toml
[prior]
ensemble_size = 100

[prior.generator]
name = "gaussian"

[prior.generator.options]
mean = 0.0
std = 0.5
```

### `gaussian_field` and `geostatspy_sgsim`

These two share the same core options:

| Option | Type | Default |
|---|---|---|
| `grid` | list of integers `[nx, ny, nz]` | required |
| `mean` | number or array-like | `0.0` |
| `variance` | number or list of numbers | `1.0` |
| `major_range` | number or list of numbers | `10.0` |
| `minor_range` | number, list of numbers or null | `null` |
| `azimuth` | number or list of numbers | `0.0` |
| `variogram` | string or list of strings | `"spherical"` |
| `conditioning` | table or null | `null` |

What the options mean:

- `mean` is a scalar or one value per cell.
- `variance` is the sill of the field.
- `major_range` and `minor_range` are the ranges in cells along and across the major axis. When `minor_range` is `null`, it takes the value of `major_range`, so the field is isotropic.
- `azimuth` is the direction of the major axis in degrees, counter-clockwise from the i-axis.
- `variogram` is one of the names below.
- A list means one value per layer. For a `grid` of `[50, 50, 5]`, `variance = [1.0, 1.0, 0.5, 0.5, 0.25]` gives each of the five layers its own sill. A single value applies to all layers.

Variogram names:

- `gaussian_field`: `spherical`, `exponential`, `gaussian`, `cubic` or `stable`. PET's `sph` and `cub` are `spherical` and `cubic`. PET's `exp` is `stable` with `exponent = 1.9`.
- `geostatspy_sgsim`: `spherical`, `exponential` or `gaussian`.

Extra options:

- `gaussian_field` also has `exponent` (number or list of numbers, default `1.0`), the exponent of the `stable` model. A `major_range` of `0` means uncorrelated.
- `geostatspy_sgsim` also has `max_neighbours` (integer, default `20`, at least 1), the number of previously simulated nodes each node is conditioned on. More neighbours reproduce the variogram more closely at a higher cost. Here both ranges must be positive, with `major_range >= minor_range`.

`conditioning` has the form `{cells = [[i, j, k], ...], values = [...], error_std = 0}`:

- `cells` are 1-based Eclipse cells.
- `values` are the values there, in the state's space. For a log variable, that is the natural logarithm of the values.
- With `error_std = 0`, the realisations take exactly those values at those cells. With a positive `error_std` (a number, or one per value), they are drawn around the values.
- For `gaussian_field`, the conditioning is exact. For `geostatspy_sgsim`, it is approximate away from the data.

### `gstools_pgs`

| Option | Type | Default |
|---|---|---|
| `fields` | list of tables | required |
| `rule_file` | path or null | `null` |
| `proportions` | list of numbers or null | `null` |
| `extent` | number | `3.0` |
| `conditioning` | table of lists or null | `null` |
| `cell_size` | list of numbers | `[1.0, 1.0, 1.0]` |
| `gibbs_sweeps` | integer | `100` |

- `fields` has one covariance model per latent field, in the order of the group's variables. Each is a table such as `{model = "Gaussian", len_scale = 10, angles = 0}`. `len_scale` is a number or a list like `[10, 5]` or `[10, 5, 2]`. `angles` is a number or a list. The variance is 1.
- `rule_file`, `proportions` and `extent` describe the truncation rule. The same rule is used by `pgs_truncation`, and the composition checks that they match. This page does not describe the rule file format.
- `conditioning` has the form `{cells = [[i, j, k], ...], facies = [code, ...]}`: 1-based Eclipse cells and the facies observed there. Every realisation honours them.
- `cell_size` is `[dx, dy, dz]` in the units of `len_scale`.
- `gibbs_sweeps` is the number of iterations of the conditioning sampler.

Because the fields are drawn together, use a group:

```toml
[prior]
ensemble_size = 100

[prior.groups.facies]
name = "gstools_pgs"
variables = ["latent_1", "latent_2"]

[prior.groups.facies.options]
fields = [
  { model = "Gaussian", len_scale = 10, angles = 0 },
  { model = "Gaussian", len_scale = 20, angles = 0 },
]
```

The variable names `latent_1` and `latent_2` are examples. Use the names of the state variables of your model.

### `mpslib_snesim`

The result is one signed distance field per facies.

| Option | Type | Default |
|---|---|---|
| `training_image_file` | string or path | required |
| `conditioning` | table of lists or null | `null` |
| `template_size` | list of integers | `[8, 7, 1]` |
| `n_multiple_grids` | integer | `3` |
| `n_cond` | integer | `16` |
| `n_min_node_count` | integer | `0` |
| `random_path` | string | `"random"` |
| `entropy_factor` | number | `4.0` |
| `cell_size` | list of numbers | `[1.0, 1.0, 1.0]` |
| `distance_band` | number or null | `null` |
| `soft_probability_file` | path or null | `null` |
| `program` | string or null | `null` |
| `timeout_s` | number | `3600.0` |

- `training_image_file` is the training image: a `.npy` file of facies codes indexed `[i, j(, k)]`, or a GSLIB/EAS grid file.
- `conditioning` has the form `{cells = [[i, j, k], ...], facies = [code, ...]}`, with 1-based Eclipse cells and the facies observed there. These are hard data, and every realisation honours them.
- `template_size` is the search template `[tx, ty, tz]`.
- `n_multiple_grids` is the number of multiple grids, from coarse to fine.
- `n_cond` is the most conditioning data per cell (`0` means all).
- `n_min_node_count` is the least count of a pattern for its conditional probability (`0` means no limit).
- `random_path` is `random` or `preferential`. The `preferential` path uses `entropy_factor`.
- `cell_size` is `[dx, dy, dz]`, used for the coordinates of the hard data and for the distances.
- `distance_band` clips the signed distances to `[-band, band]`, in the units of `cell_size`. By default they are not clipped.
- `soft_probability_file` is a `.npy` grid of facies probabilities with shape `(nx, ny, nz, K)`, with NaN in cells without soft data. Soft data shift the facies rather than fix them, unless the probabilities are 0 or 1. Hard data take precedence.
- `program` is MPSlib's `mps_snesim_tree`. By default, it is the one installed with `scikit-mps`.
- `timeout_s` is the number of seconds the program may take.

```toml
[prior]
ensemble_size = 100

[prior.generator]
name = "mpslib_snesim"

[prior.generator.options]
training_image_file = "training_image.npy"
random_path = "random"
```

## Worked example

You want a spatially correlated Gaussian prior for a 50 x 50 x 5 model. Choose `gaussian_field`. Select it in the `[prior.generator]` section and give it these values:

```toml
[prior]
ensemble_size = 100

[prior.generator]
name = "gaussian_field"

[prior.generator.options]
grid = [50, 50, 5]
major_range = 20.0
minor_range = 10.0
azimuth = 45.0
variogram = "spherical"
```

This block is only the prior part of the configuration. The rest of the `history_match` configuration is described in other pages of the guide.

What you get: a Gaussian random field on that grid. The `mean`, `variance` and `exponent` options keep their defaults (`0.0`, `1.0` and `1.0`). The correlation range is 20 cells along the major axis and 10 cells across it. The major axis points 45 degrees counter-clockwise from the i-axis. Leaving out `minor_range` would give an isotropic field with a range of 20 cells.

To see the options of every plugin, run:

```bash
uv run apex plugins
```

To try the configuration on a short run, start it with a limit on the number of analysis steps, then continue it with `apex resume` and the directory of the run:

```bash
uv run apex run config.toml --max-steps 1
uv run apex resume <run_dir> --max-steps 1
```

## Pitfalls

- `priors` is a registry, not a section. Do not write a `[priors]` section. Choose the plugin with `name` in `[prior.generator]`, `[prior.variables.<name>]` or `[prior.groups.<name>]`.
- Required options: `ensemble_file` for `file`, `grid` for `gaussian_field` and `geostatspy_sgsim`, `fields` for `gstools_pgs`, `training_image_file` for `mpslib_snesim`. `gaussian` needs none.
- These plugins reject unknown options, so a misspelled name is an error. The tables inside `fields` for `gstools_pgs` are the exception, because they accept arbitrary keys.
- `nx * ny * nz` of `grid` must equal the size of the variable.
- `geostatspy_sgsim` needs the `geostatspy` extra. For `geostatspy_sgsim`, `major_range` must not be smaller than `minor_range`.
- `max_neighbours` exists only on `geostatspy_sgsim`, and `exponent` only on `gaussian_field`. Using either on another plugin is an error.
- `conditioning` cells are 1-based Eclipse cells. For a log variable, `values` are natural logarithms.
- `gaussian` has no spatial correlation. Do not use it where you need smooth fields.
- `cell_size` defaults to `[1.0, 1.0, 1.0]` for `gstools_pgs` and `mpslib_snesim`. Set it if your cells have a different size.

## What to read next

- The other pages in the plugins section of the guide.
- The guide pages on `history_match` configurations, for the sections around `[prior]`.
- `uv run apex plugins --json`, for the full option descriptions of every plugin.

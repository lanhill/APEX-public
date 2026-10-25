---
audience: public
items:
  - cli:apex plugins
  - cli:apex resume
  - cli:apex run
  - cli:apex stop
  - plugin:priors:file
  - plugin:priors:gaussian
  - plugin:priors:gaussian_field
  - plugin:priors:geostatspy_sgsim
  - plugin:priors:gstools_pgs
  - plugin:priors:mpslib_snesim
---

# Priors

## Goal

A prior plugin generates the initial ensemble of your model parameters. This page helps you choose one of the six prior plugins and shows which options each one accepts:

- `file`
- `gaussian`
- `gaussian_field`
- `geostatspy_sgsim`
- `gstools_pgs`
- `mpslib_snesim`

`priors` is the name of the plugin registry that holds these plugins. It is not a section of your configuration. You choose a plugin with `name = "..."` in the `[prior.generator]` section of a `history_match` configuration, or in a `[prior.variables.*]` section of a `history_match` configuration.

## Prerequisites

- A run configuration in TOML. This is the file you pass to `apex run`.
- For `file`, a stored ensemble in an `.npz` file.
- For `gaussian_field` and `geostatspy_sgsim`, the grid size of the model you want to sample.
- For `geostatspy_sgsim`, the `geostatspy` extra must be installed.
- For `mpslib_snesim`, a training image file.
- For `gstools_pgs`, a description of the latent fields in `fields`.

## Some terms

- **Variogram**: a description of how quickly values become less alike with distance. In `gaussian_field` and `geostatspy_sgsim` it is set by the options `variogram`, `variance`, `major_range`, `minor_range` and `azimuth`.
- **Facies**: discrete rock types, such as sand or shale, rather than a continuous value.
- **Conditioning**: forcing realisations to honour observed values, such as well facies.
- **Latent field**: a smooth underlying Gaussian field that a plurigaussian simulation turns into facies.
- **Signed distance field**: for each facies, a field of distances to that facies. `mpslib_snesim` returns one per facies.

## Choosing a plugin

| Plugin | What it produces | Required option |
|---|---|---|
| `file` | A stored ensemble of shape `(size, n_ensemble)` read from an `.npz` file. | `ensemble_file` |
| `gaussian` | Independent normals `N(mean, std^2)` for every entry, with no spatial correlation. | none |
| `gaussian_field` | A Gaussian random field from a variogram, sampled layer by layer. It is the prior sampler used by PET. | `grid` |
| `geostatspy_sgsim` | An unconditional sequential Gaussian simulation with GeostatsPy. | `grid` |
| `gstools_pgs` | The latent fields of a plurigaussian simulation, optionally conditioned to well facies. | `fields` |
| `mpslib_snesim` | SNESIM facies realisations (MPSlib), as one signed distance field per facies. | `training_image_file` |

How to decide:

- Use `file` when you already have an ensemble you want to reuse.
- Use `gaussian` when neighbouring entries need no correlation.
- Use `gaussian_field` or `geostatspy_sgsim` for smooth, spatially correlated continuous properties.
- Use `gstools_pgs` or `mpslib_snesim` when the property is facies. Only these two have a `conditioning` option.

## Steps

1. List the plugins and their options with `apex plugins`. Add `--json` to get the full descriptions as JSON. Look for the plugins in the `priors` registry.
2. Pick the plugin from the table above.
3. In your `history_match` configuration, set `name` in `[prior.generator]` (or in a `[prior.variables.*]` section) to the plugin name.
4. Set the plugin's required option, if it has one. Every option you leave out keeps its default, listed below.
5. Check the spelling of option names. These plugins reject unknown options.
6. Start the run with `apex run config.toml`, where `config.toml` is your run configuration. Useful flags are `--max-steps` (stop after that many analysis steps) and `--no-progress` (switch off progress output on standard error, which is on by default; `--progress` switches it on again).
7. To stop a running run at its next checkpoint, use `apex stop <run_dir>`. Continue it later with `apex resume <run_dir>`, which also accepts `--max-steps` and `--no-progress`.

### `file`

| Option | Type | Default |
|---|---|---|
| `ensemble_file` | string or path | required |
| `key` | string or null | `null` |

`ensemble_file` is the `.npz` file holding the stored ensemble, with shape `(size, n_ensemble)`. `key` is optional and defaults to `null`. The plugin description does not say what `key` selects inside the file, so leave it at `null` unless you know you need it.

### `gaussian`

| Option | Type | Default |
|---|---|---|
| `mean` | number (or array-like) | `0.0` |
| `std` | number | `1.0` |

Every entry is drawn independently from a normal distribution with this `mean` and `std`. With the defaults, that is a standard normal.

### `gaussian_field` and `geostatspy_sgsim`

These two share the same variogram options:

| Option | Type | Default |
|---|---|---|
| `grid` | list of integers | required |
| `mean` | number (or array-like) | `0.0` |
| `variance` | number or list of numbers | `1.0` |
| `major_range` | number or list of numbers | `10.0` |
| `minor_range` | number, list of numbers or null | `null` |
| `azimuth` | number or list of numbers | `0.0` |
| `variogram` | string or list of strings | `"spherical"` |

`gaussian_field` also has `exponent` (number or list of numbers, default `1.0`).

`geostatspy_sgsim` also has `max_neighbours` (integer, default `20`).

The options `variance`, `major_range`, `minor_range`, `azimuth`, `variogram` and `exponent` accept either a single value or a list. Single values are the safe choice: the published descriptions do not say what each list element applies to. `gaussian_field` samples layer by layer, but how a list relates to those layers is not documented. The default `minor_range` is `null`; its effect, and the units and reference direction of `azimuth`, are also not documented. If you want a second range, set `minor_range` explicitly.

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

`fields` is a list with one table per latent field. The tables accept arbitrary keys, so APEX does not check them against a fixed list. `conditioning` is a table whose values are lists; leave it as `null` for an unconditioned simulation. The published descriptions do not define the keys of a latent-field table, the rule file format, or the contents of the `conditioning` lists, so this page cannot give a complete `gstools_pgs` example.

### `mpslib_snesim`

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

The result is one signed distance field per facies. `training_image_file` is the only option you must set. The published descriptions do not explain the tuning options, the training image format, the contents of `conditioning`, or the allowed values of `random_path`. Start from the defaults and change one option at a time.

## Worked example

You want a spatially correlated Gaussian prior for a 50 x 50 x 5 model. Choose `gaussian_field`. Select it in the `[prior.generator]` section of your `history_match` configuration and give it these values:

```toml
[prior.generator]
name = "gaussian_field"
grid = [50, 50, 5]
major_range = 20.0
minor_range = 10.0
azimuth = 0.0
variogram = "spherical"
```

This block is only the prior part. A full `history_match` configuration needs more, such as the model and the data. The options `mean`, `variance` and `exponent` keep their defaults (`0.0`, `1.0` and `1.0`). The result is a Gaussian random field on that grid, with a major range of `20.0` and a minor range of `10.0`.

To confirm that the plugin and its options exist, run `apex plugins`. This command lists what is available; it does not check your own configuration file. To test the configuration, start it with `apex run config.toml --max-steps 1`, which stops after one analysis step. Continue later with `apex resume <run_dir>`.

## Pitfalls

- `priors` is a registry, not a section. Do not write a `[priors]` section. Choose the plugin with `name` in `[prior.generator]` or `[prior.variables.*]`.
- Required options: `ensemble_file` for `file`, `grid` for `gaussian_field` and `geostatspy_sgsim`, `fields` for `gstools_pgs`, `training_image_file` for `mpslib_snesim`. `gaussian` needs none.
- These plugins reject unknown options, so a misspelled name is an error. The one exception is the tables inside `fields` for `gstools_pgs`, which accept arbitrary keys.
- `geostatspy_sgsim` is unconditional and needs the `geostatspy` extra.
- `max_neighbours` exists only on `geostatspy_sgsim`, and `exponent` only on `gaussian_field`. Using either on another plugin is an error.
- `gaussian` has no spatial correlation. Do not use it where you need smooth fields.
- `cell_size` defaults to `[1.0, 1.0, 1.0]` for `gstools_pgs` and `mpslib_snesim`. Set it if your cells have a different size.

## What to read next

- The other pages in the plugins section of the guide.
- The guide pages on `history_match` configurations, for the sections around `[prior.generator]`.
- `apex plugins --json`, for the full option descriptions of every plugin.

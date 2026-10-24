---
audience: public
items:
  - cli:apex plugins
  - cli:apex run
  - cli:apex stop
  - plugin:parameterizations:control_scaling
  - plugin:parameterizations:control_schedule
  - plugin:parameterizations:facies_distance
  - plugin:parameterizations:pgs_truncation
---

# Parameterizations

A parameterization turns the quantities APEX updates into something your model can use. APEX ships four parameterization plugins in the `parameterizations` registry:

- `control_scaling` maps normalized controls onto physical bounds.
- `control_schedule` spreads a control block's values over report steps.
- `facies_distance` builds facies from signed distance fields (using `argmax`) and assigns properties by facies.
- `pgs_truncation` builds facies from latent fields with a fixed truncation rule and assigns properties by facies.

## Goal

By the end of this page you will be able to:

- choose the right parameterization plugin;
- write a complete, valid configuration section for each one;
- give each facies its properties with `values` or `property_fields`;
- check the plugin options with `apex plugins` and test a configuration with `apex run`.

## Prerequisites

- An APEX run configuration file in TOML.
- For the control plugins: the name of the control variable.
- For the facies plugins: the names of the fields your model provides and the facies you want.

## Key terms

- **Facies**: a named rock type, such as `channel` or `shale`.
- **Signed distance field**: a field giving, for each cell, a distance to a facies body. `facies_distance` picks, in each cell, the field with the largest value (`argmax`) to decide the facies.
- **Latent field**: an underlying field that is not a facies itself. `pgs_truncation` cuts latent fields with a fixed truncation rule to get facies. With a suitable rule, a small number of latent fields can give a larger number of facies, for example two fields giving three facies.
- **Truncation rule**: the fixed rule that says which facies a cell gets, given the values of the latent fields. `rule_file` names a NumPy `.npy` array of integer facies codes (code `k` is `facies[k]`) with one axis per latent field; it is laid over `[-extent, extent]` standard deviations on each axis, and the edge bins extend to infinity. With one latent field, `proportions` (the facies' target proportions, a truncated Gaussian) can replace the file.

## Choosing a plugin

| Plugin | Use it when | Status |
| --- | --- | --- |
| `control_scaling` | You have normalized controls that must map onto physical lower and upper bounds. | stable |
| `control_schedule` | A control block's values must be spread over report steps. | stable |
| `facies_distance` | Your model provides signed distance fields. | no status tag |
| `pgs_truncation` | Your model provides latent fields and a truncation rule. | no status tag |

To find out which kind of fields your model provides, look at the field names and descriptions in your model documentation. If you cannot tell, ask whoever prepared the model set-up.

## Options

All four plugins reject any option not listed here, so a typo fails instead of being ignored.

### `control_scaling`

| Option | Type | Default | Notes |
| --- | --- | --- | --- |
| `variable` | string | required | The control variable. |
| `lower` | number or list of numbers | required | Lower physical bound. |
| `upper` | number or list of numbers | required | Upper physical bound. |
| `output` | string or none | none | Optional output name. |

### `control_schedule`

| Option | Type | Default | Notes |
| --- | --- | --- | --- |
| `variable` | string | required | The control variable. |
| `knots` | list of integers | required | The knot positions. |
| `n_steps` | integer | required | Number of report steps. |
| `output` | string or none | none | Optional output name. |
| `interpolation` | `hold` or `linear` | `hold` | How values are carried between knots. |

### `facies_distance`

| Option | Type | Default | Notes |
| --- | --- | --- | --- |
| `fields` | list of strings | required | Names of the signed distance fields. |
| `facies` | list of strings | required | Names of the facies. |
| `values` | table of tables of numbers | none | Constant properties per facies: `{facies: {property: value}}`. |
| `property_fields` | table of tables of strings | none | Properties taken cell by cell from a state variable per facies: `{property: {facies: variable}}`. |

### `pgs_truncation`

| Option | Type | Default | Notes |
| --- | --- | --- | --- |
| `fields` | list of strings | required | Names of the latent fields. |
| `facies` | list of strings | required | Names of the facies. |
| `rule_file` | string (a path) | none | The truncation rule, a `.npy` array of facies codes. |
| `proportions` | list of numbers | none | With one latent field, the facies' target proportions instead of `rule_file`. |
| `extent` | number | `3.0` | The rule array covers `[-extent, extent]` standard deviations on each axis. |
| `values` | table of tables of numbers | none | Constant properties per facies: `{facies: {property: value}}`. |
| `property_fields` | table of tables of strings | none | Properties taken cell by cell from a state variable per facies: `{property: {facies: variable}}`. |

### Property tables: `values` and `property_fields`

Both options are tables of tables, so each takes two levels of keys, and the two are nested the other way round:

- `values` is `{facies: {property: value}}`: the outer keys are facies names, the inner keys property names, each with a number. Every cell of a facies gets that number.
- `property_fields` is `{property: {facies: variable}}`: the outer keys are property names, the inner keys facies names, each naming a state variable. Every cell takes the value of the variable of its facies there, and those variables are updated with the latent fields.

The facies names must be those in `facies`, and the property names those the simulator reads (for example `poro`, `permx`). A property is given by `values` or by `property_fields`, not both. The plugin's outputs are `facies` (the codes) and every property.

## Steps

1. Pick the plugin that matches what you have.
2. Add a `[[parameterizations]]` entry with the plugin's `name`, for example `name = "facies_distance"`, and its options in `options`. `[[parameterizations]]` is an array of tables, so its brackets are doubled: one entry per map, applied in order. The control plugins go in an optimization configuration (`apex schema optimization`), the facies plugins in a history-matching one (`apex schema history_match`).
3. Set the required options for that plugin (see the tables above).
4. Add optional options only where you need them. Omitted options keep their defaults.
5. Run `apex plugins` to list every plugin with its status and options. Add `--json` for the full descriptions. Use it to check option names, types and defaults against your file. A plugin's status (for example `stable`) is shown in this listing.
6. Test with `apex run config.toml --max-steps 1`. The plugin options are checked when the run starts. A missing required option, a misspelt option or a value of the wrong type gives an error instead of a silent default. `--max-steps` stops the run after that many analysis steps, and you can continue later with `apex resume`. Add `--no-progress` to hide the progress display.
7. To stop a run that is in progress, use `apex stop` with the run directory. The run stops at its next checkpoint.

## Worked example

Control plugins first, in an optimization configuration. This scales a control onto physical bounds:

```toml
[[parameterizations]]
name = "control_scaling"
options = { variable = "rate", lower = 0.0, upper = 500.0 }
```

`lower` and `upper` can also be lists of numbers, one per value of the control block. `rate` is a placeholder; use your own control variable.

This spreads a control block of three values over 12 report steps, with linear interpolation between the knots (the first report step of each value):

```toml
[[parameterizations]]
name = "control_schedule"
options = { variable = "rate", knots = [0, 4, 8], n_steps = 12, interpolation = "linear" }
```

Leave out `interpolation` to get the default, `hold`. With both entries, list `control_schedule` first and give `control_scaling` its `output` as `variable`, since the maps apply in order.

Now facies, in a history-matching configuration. This builds three facies from three signed distance fields (one state variable per facies, in the order of `facies`) and gives each a constant porosity:

```toml
[[parameterizations]]
name = "facies_distance"

[parameterizations.options]
fields = ["d_channel", "d_levee", "d_shale"]
facies = ["channel", "levee", "shale"]

[parameterizations.options.values]
channel = { poro = 0.30 }
levee = { poro = 0.20 }
shale = { poro = 0.05 }
```

`[parameterizations.options]` holds the options of the entry just above it. To take a property from state variables instead of fixed numbers, use `property_fields`. The variable names are placeholders for state variables of your own configuration.

```toml
[[parameterizations]]
name = "facies_distance"

[parameterizations.options]
fields = ["d_channel", "d_levee", "d_shale"]
facies = ["channel", "levee", "shale"]

[parameterizations.options.property_fields.permx]
channel = "perm_channel"
levee = "perm_levee"
shale = "perm_shale"
```

With `pgs_truncation`, two latent fields are cut by the truncation rule in `rule_file` (a 2-D `.npy` array of the codes 0, 1 and 2) to give three facies. The file name is a placeholder.

```toml
[[parameterizations]]
name = "pgs_truncation"

[parameterizations.options]
fields = ["gaussian_1", "gaussian_2"]
facies = ["sand", "silt", "clay"]
rule_file = "truncation_rule.npy"

[parameterizations.options.values]
sand = { poro = 0.28 }
silt = { poro = 0.15 }
clay = { poro = 0.04 }
```

Save a configuration as `config.toml`, then check the plugins and test the run:

```bash
apex plugins
apex run config.toml --max-steps 1
```

If the plugin options are wrong, the run stops with an error right away. Fix the option named in the message and run again. Output from `apex plugins` is a list of plugins with their status and options; the exact layout can differ, so look for the plugin name and compare its options with your file.

## Pitfalls

- **Missing required options.** Each plugin has required options (see the tables). Leaving one out is an error.
- **Misspelt or unknown options.** The plugins reject options they do not define.
- **Mixing up the plugins' inputs.** `facies_distance` expects signed distance fields. `pgs_truncation` expects latent fields.
- **Inconsistent names.** Facies names in `values` and `property_fields` must match the names in `facies`.
- **Nesting the property tables the wrong way.** `values` is keyed by facies first (`sand = { poro = 0.28 }`), `property_fields` by property first (`[parameterizations.options.property_fields.permx]`).
- **Wrong value types.** `values` holds numbers, `property_fields` holds strings, `knots` holds integers.
- **Run directory already in use.** Starting a run whose run directory already holds a run fails with a message such as `... already holds a run; resume it or choose another`. Either continue it with `apex resume`, or choose another run directory in your configuration.
- **Assuming optional settings are required.** `extent` is optional and defaults to `3.0`. `interpolation` is optional and defaults to `hold`. `output` is optional in both control plugins.

## What to read next

- The other plugin guides in this section, for the registries that work alongside parameterizations.
- The configuration reference, for how sections are laid out in a full APEX file.

---
audience: public
items:
  - cli:apex plugins
  - cli:apex report
  - cli:apex resume
  - cli:apex run
  - cli:apex schema
  - cli:apex stop
  - config:history_match:parameterizations
  - config:optimization:parameterizations
  - plugin:parameterizations:control_scaling
  - plugin:parameterizations:control_schedule
  - plugin:parameterizations:facies_distance
  - plugin:parameterizations:pgs_truncation
---

# Parameterizations

A parameterization turns the quantities APEX updates into something your simulator can read. APEX ships four parameterization plugins in the `parameterizations` registry:

- `control_scaling` maps normalized controls in [0, 1] linearly onto physical bounds.
- `control_schedule` spreads a control block's values over the report steps of its periods.
- `facies_distance` builds facies from signed distance fields (using `argmax`) and assigns properties by facies.
- `pgs_truncation` builds facies from latent fields with a fixed truncation rule and assigns properties by facies.

The `parameterizations` section is a list of maps applied in order. If you leave it out, the state (or the controls) is passed to the simulator as is.

## Goal

By the end of this page you will be able to:

- choose the right parameterization plugin;
- write a complete configuration section for each one;
- chain `control_schedule` and `control_scaling`;
- give each facies its properties with `values` or `property_fields`;
- check the plugin options with `apex plugins` and test a configuration with `apex run`.

## Prerequisites

- An APEX run configuration file in TOML.
- For the control plugins: the name of the control variable, in an optimization configuration.
- For the facies plugins: the state variables your history-matching configuration defines for the fields, and the facies you want.

## Key terms

- **Facies**: a named rock type, such as `channel` or `shale`.
- **State variable**: a variable that APEX updates in a history-matching run. Every name you give in `fields`, and every variable name in `property_fields`, must be a state variable defined elsewhere in your configuration (`apex schema history_match` prints what that file accepts).
- **Signed distance field**: a field giving, for each cell, a distance to a facies body. `facies_distance` picks, in each cell, the field with the largest value (`argmax`) to decide the facies. The i-th entry of `fields` belongs to the i-th entry of `facies`, so keep both lists in the same order.
- **Latent field**: an underlying field that is not a facies itself. `pgs_truncation` cuts latent fields with a fixed truncation rule to get facies. Two fields can, for example, give three facies.
- **Truncation rule**: the fixed rule that says which facies a cell gets, given the latent field values. See the `pgs_truncation` section below.
- **Knot**: for `control_schedule`, a report step at which a value of the control block starts. The worked example uses `knots = [0, 4, 8]`: three values, starting at report steps 0, 4 and 8.

## Choosing a plugin

| Plugin | Use it when | Status |
| --- | --- | --- |
| `control_scaling` | You have normalized controls that must map onto physical lower and upper bounds. | stable |
| `control_schedule` | A control block's values must be spread over report steps. | stable |
| `facies_distance` | Your model provides signed distance fields. | no status tag |
| `pgs_truncation` | Your model provides latent fields and a truncation rule. | no status tag |

If you cannot tell which kind of fields your model provides, ask whoever prepared the model set-up.

## Options

All four plugins reject any option not listed here, so a typo is an error instead of being ignored.

### `control_scaling`

| Option | Type | Default | Notes |
| --- | --- | --- | --- |
| `variable` | string | required | The control variable to scale. |
| `lower` | number or list of numbers | required | Lower physical bound. |
| `upper` | number or list of numbers | required | Upper physical bound. |
| `output` | string or none | none | Optional output name. |

### `control_schedule`

| Option | Type | Default | Notes |
| --- | --- | --- | --- |
| `variable` | string | required | The control variable to spread. |
| `knots` | list of integers | required | The report steps at which the values of the control block start. |
| `n_steps` | integer | required | Number of report steps. |
| `output` | string or none | none | Optional name under which the scheduled values are provided, so a later map can use it as its `variable`. |
| `interpolation` | `hold` or `linear` | `hold` | How values are carried between knots. |

### `facies_distance`

| Option | Type | Default | Notes |
| --- | --- | --- | --- |
| `fields` | list of strings | required | Names of the signed distance fields (state variables), in the order of `facies`. |
| `facies` | list of strings | required | Names of the facies. |
| `values` | table of tables of numbers | none | Constant properties per facies: `{facies: {property: value}}`. |
| `property_fields` | table of tables of strings | none | Properties taken cell by cell from a state variable per facies: `{property: {facies: variable}}`. |

### `pgs_truncation`

| Option | Type | Default | Notes |
| --- | --- | --- | --- |
| `fields` | list of strings | required | Names of the latent fields (state variables). |
| `facies` | list of strings | required | Names of the facies. |
| `rule_file` | string (a path) | none | The truncation rule, a `.npy` array of facies codes. |
| `proportions` | list of numbers | none | With one latent field, the facies' target proportions instead of `rule_file`. |
| `extent` | number | `3.0` | The rule array covers `[-extent, extent]` standard deviations on each axis. |
| `values` | table of tables of numbers | none | Constant properties per facies: `{facies: {property: value}}`. |
| `property_fields` | table of tables of strings | none | Properties taken cell by cell from a state variable per facies: `{property: {facies: variable}}`. |

### The truncation rule: `rule_file` and `proportions`

`rule_file` names a NumPy `.npy` array of integer facies codes. Code `k` means `facies[k]`. The array has one axis per latent field: axis 0 belongs to the first entry of `fields`, axis 1 to the second, and so on. Each axis is laid over `[-extent, extent]` standard deviations, and the edge bins extend to infinity. A cell's latent values pick a bin on each axis, and the code stored there is the cell's facies.

For two fields and three facies, a small rule could be a 3x3 array:

```python
import numpy as np

rule = np.array(
    [
        [0, 0, 1],
        [0, 1, 2],
        [1, 2, 2],
    ],
    dtype=int,
)
np.save("truncation_rule.npy", rule)
```

With one latent field, `proportions` can replace the file. List one proportion per facies, in the order of `facies`, summing to 1. They are the target proportions of a truncated Gaussian.

### Property tables: `values` and `property_fields`

Both options are tables of tables, nested the other way round:

- `values` is `{facies: {property: value}}`: outer keys are facies names, inner keys property names, each with a number. Every cell of a facies gets that number.
- `property_fields` is `{property: {facies: variable}}`: outer keys are property names, inner keys facies names, each naming a state variable that you define in your configuration. Every cell takes, from the variable of its facies, the value at that cell. APEX updates these variables as it updates the other state variables.

The facies names must be those in `facies`, and the property names those the simulator reads (for example `poro`, `permx`). A property is given by `values` or by `property_fields`, not both. The plugin's outputs are `facies` (the codes) and every property.

## Steps

1. Pick the plugin that matches what you have.
2. Add a `[[parameterizations]]` entry with the plugin's `name` and its options in `options`. `[[parameterizations]]` is an array of tables, so its brackets are doubled: one entry per map, applied in order. The control plugins go in an optimization configuration (`apex schema optimization`), the facies plugins in a history-matching one (`apex schema history_match`).
3. Set the required options for that plugin (see the tables above).
4. Add optional options only where you need them. Omitted options keep their defaults.
5. Run `apex plugins` to list every plugin with its options. Add `--json` for the full descriptions. Use it to check option names, types and defaults against your file.
6. Test with `apex run config.toml --max-steps 1`. `--max-steps` stops the run after that many analysis steps. Add `--no-progress` to hide the progress display.
7. To stop a run that is in progress, pass its run directory to `apex stop`. The run stops at its next checkpoint. Continue it later with `apex resume`, which also takes the run directory.

```bash
apex stop RUN_DIR
apex resume RUN_DIR
```

`RUN_DIR` stands for the directory of your run.

## Worked example

Control plugins first, in an optimization configuration. This scales a control onto physical bounds:

```toml
[[parameterizations]]
name = "control_scaling"
options = { variable = "rate", lower = 0.0, upper = 500.0 }
```

`lower` and `upper` can also be lists of numbers. `rate` is a placeholder; use your own control variable.

This spreads a control block of three values over 12 report steps, with linear interpolation between the knots:

```toml
[[parameterizations]]
name = "control_schedule"
options = { variable = "rate", knots = [0, 4, 8], n_steps = 12, interpolation = "linear" }
```

Leave out `interpolation` to get the default, `hold`.

To use both, list `control_schedule` first, because the maps apply in order. Give it `output = "rate_steps"`. Then give `control_scaling` `variable = "rate_steps"`, so it scales the scheduled values instead of the raw control:

```toml
[[parameterizations]]
name = "control_schedule"
options = { variable = "rate", knots = [0, 4, 8], n_steps = 12, output = "rate_steps" }

[[parameterizations]]
name = "control_scaling"
options = { variable = "rate_steps", lower = 0.0, upper = 500.0 }
```

The data flow is: `rate` goes into `control_schedule`, which provides `rate_steps`; `control_scaling` reads `rate_steps`. The name `rate_steps` is your choice; it only has to be the same in both entries. When `control_scaling` is used alone, its `variable` is the control itself, as in the first example.

Now facies, in a history-matching configuration. This builds three facies from three signed distance fields and gives each a constant porosity. `d_channel`, `d_levee` and `d_shale` are state variables of your configuration, in the same order as the facies:

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

`[parameterizations.options]` holds the options of the entry just above it. To take a property from state variables instead of fixed numbers, use `property_fields`. The variable names are placeholders for state variables you define yourself.

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

With `pgs_truncation`, two latent fields are cut by the rule in `rule_file` (the 3x3 array saved above, with codes 0, 1 and 2) to give three facies:

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

With one latent field, use `proportions` instead of a file. The three proportions follow the order of `facies` and sum to 1:

```toml
[[parameterizations]]
name = "pgs_truncation"

[parameterizations.options]
fields = ["gaussian_1"]
facies = ["sand", "silt", "clay"]
proportions = [0.5, 0.3, 0.2]

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

If the run reports an error about an option, fix the option named in the message and run again. The layout of the `apex plugins` listing can differ, so look for the plugin name and compare its options with your file.

## Pitfalls

- **Missing required options.** Each plugin has required options (see the tables). Leaving one out is an error.
- **Misspelt or unknown options.** The plugins reject options they do not define.
- **Mixing up the plugins' inputs.** `facies_distance` expects signed distance fields. `pgs_truncation` expects latent fields.
- **Fields and facies in different orders.** For `facies_distance`, the i-th field belongs to the i-th facies.
- **Undefined state variables.** Names in `fields` and `property_fields` must be state variables defined in your configuration.
- **Inconsistent names.** Facies names in `values` and `property_fields` must match the names in `facies`.
- **Nesting the property tables the wrong way.** `values` is keyed by facies first (`sand = { poro = 0.28 }`), `property_fields` by property first (`[parameterizations.options.property_fields.permx]`).
- **Wrong value types.** `values` holds numbers, `property_fields` holds strings, `knots` holds integers.
- **Chaining with the wrong name.** When `control_schedule` has an `output`, the `variable` of the following `control_scaling` must be that same name.
- **Assuming optional settings are required.** `extent` is optional and defaults to `3.0`. `interpolation` is optional and defaults to `hold`. `output` is optional in both control plugins.

## What to read next

- The other plugin guides in this section, for the registries that work alongside parameterizations.
- The configuration reference, for how sections are laid out in a full APEX file.

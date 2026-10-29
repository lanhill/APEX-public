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
  - plugin:gmac_constraints:linear
  - plugin:parameterizations:control_scaling
  - plugin:parameterizations:control_schedule
  - plugin:parameterizations:facies_distance
  - plugin:parameterizations:pgs_truncation
  - plugin:simulators:linear
---

# Parameterizations

A parameterization turns the quantities APEX updates (the state in a history match, the controls in an optimization) into the properties your simulator reads. The `parameterizations` section exists in both the `history_match` and the `optimization` configurations. It is a list of maps applied in order. If you leave it out, the state (or the controls) is passed to the simulator as is.

APEX ships four plugins in the `parameterizations` registry:

- `control_scaling`: normalized controls in [0, 1], mapped linearly onto their physical bounds.
- `control_schedule`: a control block's values spread over the report steps of its periods.
- `facies_distance`: facies from signed distance fields (`argmax`), and properties by facies.
- `pgs_truncation`: facies from latent fields by a fixed truncation rule, and properties by facies.

Other registries also have a plugin called `linear` (in `gmac_constraints` and in `simulators`). They are not parameterizations and are not covered here.

## Goal

By the end of this page you will be able to:

- choose the right parameterization plugin;
- write a complete `[[parameterizations]]` entry for each one;
- chain `control_schedule` and `control_scaling`;
- give each facies its properties with `values` or `property_fields`;
- check plugin options with `apex plugins` and test a configuration with `apex run`.

## Prerequisites

- A run configuration file in TOML.
- For the control plugins: an optimization configuration with a control block.
- For the facies plugins: a history-matching configuration whose state variables hold the fields you want to use (see "Key terms").

## Key terms

- **Facies**: a named rock type, such as `channel` or `shale`. In the options, facies code `k` is `facies[k]`.
- **State variable**: a variable APEX updates in a history-matching run. Every name in `fields`, and every variable named in `property_fields`, must be a state variable of your configuration. This page does not define state variables. Run `uv run apex schema history_match` to print the JSON Schema of what a history-matching file accepts, and define them there.
- **Signed distance field**: for `facies_distance`, one distance state variable per facies. The cell gets the facies of the field with the largest value (`argmax`). `fields[k]` belongs to `facies[k]`, so keep both lists in the same order.
- **Latent field**: for `pgs_truncation`, a state variable that is cut by a truncation rule to give facies.
- **Knot**: for `control_schedule`, the first report step (0-based) of each value of the control block. The knots increase and start at 0.

## Choosing a plugin

| Plugin | Use it when | Status |
| --- | --- | --- |
| `control_scaling` | Your controls are normalized in [0, 1] and must map onto physical bounds. | stable |
| `control_schedule` | A control block's values must be spread over report steps. | stable |
| `facies_distance` | Your state holds one signed distance field per facies. | no status tag |
| `pgs_truncation` | Your state holds latent fields and you have a truncation rule (or one field and target proportions). | no status tag |

## Options

All four plugins reject options they do not define, so a typo is an error.

### `control_scaling`

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `variable` | string | required | The normalized control block. |
| `lower` | number or list of numbers | required | The physical value at 0: a scalar, or one per value of the block. |
| `upper` | number or list of numbers | required | The physical value at 1: a scalar, or one per value of the block. |
| `output` | string or none | none | The name the simulation reads. When omitted, it is `variable`. |

### `control_schedule`

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `variable` | string | required | The control block (or an earlier parameterization's output) of `len(knots)` values. |
| `knots` | list of integers | required | The first report step of each value (0-based, increasing, starting at 0). |
| `n_steps` | integer | required | The report steps of the schedule. |
| `output` | string or none | none | The name the simulation reads, with `(n_steps,)` values. When omitted, it is `variable`. |
| `interpolation` | `hold` or `linear` | `hold` | `hold` is piecewise constant. |

### `facies_distance`

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `fields` | list of strings | required | One distance state variable per facies, in code order. |
| `facies` | list of strings | required | Facies names. |
| `values` | table of tables of numbers | none | Constant properties per facies, as for `pgs_truncation`. |
| `property_fields` | table of tables of strings | none | Properties by facies taken from state variables, as for `pgs_truncation`. |

### `pgs_truncation`

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `fields` | list of strings | required | The latent state variables, one per rule axis, in order. |
| `facies` | list of strings | required | Facies names; facies code `k` is `facies[k]`. |
| `rule_file` | string (a path) or none | none | A lithotype array (see below). |
| `proportions` | list of numbers or none | none | For one field, the facies' target proportions (a truncated Gaussian). |
| `extent` | number | `3.0` | The rule array is laid over `[-extent, extent]` standard deviations on each axis. |
| `values` | table of tables of numbers | none | Constant properties per facies. |
| `property_fields` | table of tables of strings | none | Properties taken cell by cell from a state variable of the facies there. |

### The truncation rule: `rule_file` or `proportions`

Both options default to none. You describe the rule with one of them: `rule_file`, or, for a single field, `proportions`. Give exactly one. This page does not describe what happens if you give both or neither, so do not rely on either.

`rule_file` is a lithotype array: an `.npy` file of integer codes with one axis per field. It is laid over `[-extent, extent]` standard deviations on each axis, and the edge bins extend to infinity. A relative path is not documented to resolve from the config file's folder, so use an absolute path if the run cannot find the file.

For two fields and three facies, a small rule is a 3x3 array of codes 0, 1 and 2:

```python
import numpy as np

rule = np.array([[0, 0, 1], [0, 1, 2], [1, 2, 2]], dtype=int)
np.save("truncation_rule.npy", rule)
```

With one field, `proportions` gives the facies' target proportions, one per facies in the order of `facies`.

### Property tables: `values` and `property_fields`

The two tables are nested in opposite ways:

- `values` is `{facies: {property: value}}`: constant properties per facies, for example `{sand = {poro = 0.25, permx = 500}, shale = {poro = 0.05, permx = 1}}`.
- `property_fields` is `{property: {facies: state variable}}`: properties taken cell by cell from a state variable of the facies there, for example `{permx = {sand = "permx_sand", ...}}`. These variables are updated with the latent fields.

A property is in `values` or in `property_fields`, not both.

## Steps

1. Pick the plugin that matches your fields (see the table above).
2. Add a `[[parameterizations]]` entry with the plugin's `name` and its `options`. The brackets are doubled because it is an array of tables: one entry per map, applied in order. Control plugins go in an optimization configuration (`uv run apex schema optimization` prints what it accepts), facies plugins in a history-matching one.
3. Set the required options of that plugin.
4. Add optional options only where you need them. Omitted options keep their defaults.
5. Run `uv run apex plugins` to list every plugin with its options, and add `--json` for the full descriptions. Compare option names and types with your file.
6. Test with `uv run apex run config.toml --max-steps 1`. `--max-steps` stops the run after that many analysis steps. `--no-progress` hides the progress display.
7. To stop a running run, pass its run directory to `apex stop`. It stops at its next checkpoint. `apex resume` continues it from its last checkpoint.

```bash
uv run apex stop runs/my_run
uv run apex resume runs/my_run
```

`runs/my_run` is a placeholder for the directory of your run. `apex stop`, `apex resume` and `apex report` all take it as `run_dir`. Use the directory of the run you started.

To see the data mismatch history and performance metrics of a run, use `apex report`. Add `--plots` to write figures; `--output` sets their directory (default: `RUN_DIR/report`) and `--font-size` their font size (default 12.0 points):

```bash
uv run apex report runs/my_run --plots
```

## Worked example

Control plugins first, in an optimization configuration. This scales a normalized control block onto physical bounds:

```toml
[[parameterizations]]
name = "control_scaling"
options = { variable = "rate", lower = 0.0, upper = 500.0 }
```

`rate` is a placeholder for your control block. `lower` and `upper` can also be lists with one value per value of the block. Without `output`, the simulation reads the result under the name `rate`. To give it another name, set `output`, for example `output = "rate_physical"`.

This spreads a block of three values over 12 report steps, starting at steps 0, 4 and 8, with linear interpolation:

```toml
[[parameterizations]]
name = "control_schedule"
options = { variable = "rate", knots = [0, 4, 8], n_steps = 12, interpolation = "linear" }
```

To use both, list `control_schedule` first, because maps are applied in order. Its `variable` may be the control block, and a later map may read an earlier map's output. So give `control_schedule` the `output` `rate_steps`, and make `control_scaling` read it:

```toml
[[parameterizations]]
name = "control_schedule"
options = { variable = "rate", knots = [0, 4, 8], n_steps = 12, output = "rate_steps" }

[[parameterizations]]
name = "control_scaling"
options = { variable = "rate_steps", lower = 0.0, upper = 500.0 }
```

The data flow is: `rate` goes into `control_schedule`, which outputs `rate_steps` (12 values); `control_scaling` reads `rate_steps`. Since `control_scaling` has no `output`, the simulation reads the scaled values as `rate_steps`. The name `rate_steps` is your choice, but it must be the same in both entries.

Now facies, in a history-matching configuration. This builds three facies from three signed distance fields and gives each a constant porosity. `d_channel`, `d_levee` and `d_shale` are placeholders for state variables you define, in the same order as `facies`:

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

`[parameterizations.options]` holds the options of the entry just above it. To take a property from state variables instead of fixed numbers, use `property_fields`:

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

With `pgs_truncation`, two latent fields are cut by the rule saved above to give three facies:

```toml
[[parameterizations]]
name = "pgs_truncation"

[parameterizations.options]
fields = ["gaussian_1", "gaussian_2"]
facies = ["sand", "silt", "clay"]
rule_file = "/path/to/truncation_rule.npy"

[parameterizations.options.values]
sand = { poro = 0.28 }
silt = { poro = 0.15 }
clay = { poro = 0.04 }
```

With one latent field, use `proportions` instead of a file, one per facies in the order of `facies`:

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

Put an entry in your `config.toml`, together with the other sections it needs, then check the plugins and test the run:

```bash
uv run apex plugins
uv run apex run config.toml --max-steps 1
```

If the run reports an error about an option, fix the option named in the message and run again.

## Pitfalls

- **Missing required options.** Each plugin has required options (see the tables). Leaving one out is an error.
- **Misspelt or unknown options.** The plugins reject options they do not define.
- **Mixing up the inputs.** `facies_distance` expects one distance field per facies. `pgs_truncation` expects latent fields cut by a rule.
- **Fields and facies in different orders.** For `facies_distance`, `fields[k]` belongs to `facies[k]`.
- **Undefined state variables.** Names in `fields` and `property_fields` must be state variables of your configuration.
- **Giving both `rule_file` and `proportions`, or neither.** Give exactly one.
- **Nesting the property tables the wrong way.** `values` is keyed by facies first, `property_fields` by property first.
- **The same property in both tables.** A property is in `values` or in `property_fields`, not both.
- **Chaining with the wrong name.** When `control_schedule` has an `output`, the `variable` of the following `control_scaling` must be that name.
- **Wrong value types.** `values` holds numbers, `property_fields` holds strings, `knots` holds integers.

## What to read next

- The other plugin guides in this section, for the registries that work alongside parameterizations.
- The configuration reference, for how sections are laid out in a full APEX file.

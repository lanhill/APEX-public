---
audience: public
items:
  - cli:apex plugins
  - cli:apex run
  - cli:apex summary
  - cli:apex validate
  - plugin:simulators:eclipse
  - plugin:simulators:ert
  - plugin:simulators:linear
  - plugin:simulators:opm_flow
  - plugin:simulators:toy_reservoir
---

# Simulators

A simulator plugin turns the inputs of one member (one realization of the ensemble) into outputs. This page lists the five simulator plugins, their options and defaults, and how to configure one.

## Goal

By the end of this page you will be able to:

- choose the right simulator plugin for your case;
- write the simulator section of a configuration in TOML;
- check it with `apex validate` before you start a run with `apex run`.

## Prerequisites

- An APEX run configuration (TOML) in which you can add a simulator section.
- For `eclipse`: Eclipse (E100 or E300) and its `eclrun` launcher installed.
- For `opm_flow`: OPM Flow installed (the executable is called `flow` by default).
- For `ert`: an ERT case file.
- For `eclipse` and `opm_flow`: a deck template file, that is, your simulator input deck stored as a template. Values you want APEX to fill in are passed through `template_values`.

## The plugins at a glance

Run `apex plugins` to list every plugin with its status and options. Add `--json` for the full descriptions as JSON.

| Plugin | What it does |
| --- | --- |
| `eclipse` | Runs Eclipse (E100 or E300) through `eclrun` for one member in its work directory. |
| `ert` | Runs an ERT case's forward model through the `ert` executor. |
| `linear` | Computes `y = f(G x)` for a fixed matrix `G`, where `f` is the identity or `tanh`. |
| `opm_flow` | Runs OPM Flow for one member in its work directory. |
| `toy_reservoir` | Returns field totals of a toy waterflood, for optimization tests and demos. Its status is `validated`. |

Only `toy_reservoir` reports a status in `apex plugins`; the other four report none.

## Steps

1. Pick the plugin from the table above.
2. Add a `[simulator]` section to your configuration. Its `name` is the plugin name, and its settings go in a `[simulator.options]` table.
3. Set every required option for that plugin (listed below). Leave the others at their defaults unless you need to change them.
4. Use only the options listed for that plugin. Each plugin rejects options it does not define.
5. Check the file with `apex validate your-config.toml`. This validates the configuration and plans its cost; nothing is simulated.
6. Start the run with `apex run your-config.toml`. Use `--max-steps N` to stop after N analysis steps; you can continue later with `apex resume`.

## Options

### `eclipse`

Required: `deck_template_file`.

| Option | Type | Default |
| --- | --- | --- |
| `deck_template_file` | string (path) | required |
| `version` | `e100` or `e300` | `e100` |
| `executable` | string | `eclrun` |
| `eclrun_args` | list of strings | `[]` |
| `timeout_s` | number | `3600.0` |
| `keep_files` | boolean | `true` |
| `licenses` | integer or none | none |
| `template_values` | table or none | none |
| `transient_patterns` | list of strings | `[]` |
| `levels` | list of tables or none | none |
| `case_dir` | string (path) or none | none |

### `opm_flow`

Required: `deck_template_file`.

| Option | Type | Default |
| --- | --- | --- |
| `deck_template_file` | string (path) | required |
| `executable` | string | `flow` |
| `flow_args` | list of strings | `[]` |
| `timeout_s` | number | `3600.0` |
| `keep_files` | boolean | `true` |
| `template_values` | table or none | none |
| `transient_patterns` | list of strings | `[]` |
| `levels` | list of tables or none | none |
| `case_dir` | string (path) or none | none |

### `ert`

Required: `case_file`, `fields`, `eclbase`.

| Option | Type | Default |
| --- | --- | --- |
| `case_file` | string (path) | required |
| `fields` | table of tables of strings | required |
| `eclbase` | string | required |
| `responses` | `summary` or `storage` | `summary` |
| `results_program` | list of strings | `[]` |
| `step_parameters` | table of strings or none | none |
| `parameter_step` | string or none | none |
| `skip_forward_steps` | integer | `0` |

`case_file` points to your ERT case (`.ert`). `fields` maps each state variable to a table with its ERT name (`keyword`, its GRDECL keyword) and the output `file` the forward model reads in the run path, for example `fields = { poro = { keyword = "PORO", file = "poro.grdecl" } }`. `eclbase` is the case's `ECLBASE`, the simulator's case name in the run path (`<IENS>` and `<ITER>` are replaced). `step_parameters` maps state variables to the ERT names of parameters that ERT's own writers put in the run path (`GEN_KW` groups, surfaces, transformed fields), and `parameter_step` is the forward-model step that writes them. `skip_forward_steps` leaves the case's first forward-model steps (those that make `FORWARD_INIT` parameters) out of every batch. `apex ert history-match` sets all of these from the case for you.

### `linear`

Required: `outputs`, `variables`.

| Option | Type | Default |
| --- | --- | --- |
| `outputs` | list of lists of strings | required |
| `variables` | list of strings | required |
| `fields` | list of lists (each item a string or integer) | `[]` |
| `matrix` | array-like or none | none |
| `forward_file` | string (path) or none | none |
| `nonlinearity` | `none` or `tanh` | `none` |
| `fail_members` | list of integers | `[]` |
| `fail_above` | number or none | none |
| `flaky_members` | table of integers or none | none |
| `delay_s` | number | `0.0` |
| `resources` | table of integers or none | none |
| `levels` | list of tables or none | none |
| `write_files` | boolean | `false` |

The plugin computes `y = f(G x)`. `variables` names the inputs `x`, `outputs` names the results `y`, and `nonlinearity` chooses `f` (`none` is the identity). The matrix `G` is given either inline with `matrix` or from a file with `forward_file`; both default to none. `fail_members`, `fail_above`, `flaky_members` and `delay_s` have names that suggest they are for testing failure handling and slow runs, but their exact behaviour is not described here. Check `apex plugins --json`.

### `toy_reservoir`

No required options. All have defaults.

| Option | Type | Default |
| --- | --- | --- |
| `start` | string | `2000-01-01` |
| `control` | string | `inj` |
| `oil_rate` | number | `20.0` |
| `water_rate` | number | `1.0` |

`start` is a date string, `oil_rate` and `water_rate` are numbers, and `control` is a string naming the control.

### Optional settings shared by the deck-based plugins

- `timeout_s`: a number of seconds, default `3600.0`.
- `keep_files`: `true` by default, so member files are kept.
- `template_values`: a table of values for your deck template.
- `transient_patterns`: a list of strings, empty by default. Each is a case-insensitive regular expression: a failed run whose log matches one counts as transient and is retried by the executor, on top of the built-in licence and file-system patterns.
- `levels`: a list of tables.
- `case_dir`: a path.
- `licenses` (`eclipse` only): an integer.
- `eclrun_args` (`eclipse`) and `flow_args` (`opm_flow`): lists of extra command-line arguments for the launcher.

## Worked example

The `toy_reservoir` plugin is the quickest to try because it needs no external program. This section changes the oil rate and keeps the other defaults:

```toml
[simulator]
name = "toy_reservoir"

[simulator.options]
start = "2000-01-01"
control = "inj"
oil_rate = 25.0
water_rate = 1.0
```

For Eclipse, the minimal section sets the required deck template and, if needed, the version and timeout:

```toml
[simulator]
name = "eclipse"

[simulator.options]
deck_template_file = "model/DECK.DATA.tmpl"
version = "e300"
timeout_s = 7200.0
```

The `opm_flow` section looks the same, but without `version`:

```toml
[simulator]
name = "opm_flow"

[simulator.options]
deck_template_file = "model/DECK.DATA.tmpl"
timeout_s = 7200.0
keep_files = false
```

These examples show only the simulator section. The rest of your run configuration goes in the same file. After editing, run `apex validate your-config.toml` to confirm the section is accepted, for example to catch a wrong path or a misspelled option.

## Pitfalls

- **Wrong section layout.** The plugin name goes in `name` under `[simulator]`, and its settings go under `[simulator.options]`. Do not write a section named after the plugin.
- **Missing required options.** `eclipse` and `opm_flow` need `deck_template_file`; `ert` needs `case_file`, `fields` and `eclbase`; `linear` needs `outputs` and `variables`.
- **Unknown options.** Every simulator rejects options it does not define. Do not carry an `eclipse` option such as `version` over to `opm_flow`.
- **Fixed choices.** `version` accepts only `e100` or `e300`. `nonlinearity` accepts only `none` or `tanh`. `responses` accepts only `summary` or `storage`.
- **Timeouts.** `timeout_s` defaults to `3600.0` seconds for `eclipse` and `opm_flow`. Raise it for long runs.
- **Disk use.** `keep_files` defaults to `true`, so member files are kept. Set it to `false` if you do not need them.
- **Executable names.** `executable` defaults to `eclrun` for `eclipse` and `flow` for `opm_flow`. Set it only if your installation uses a different name.
- **Demo plugin.** `toy_reservoir` is meant for optimization tests and demos, not for real reservoir models.

## What to read next

- Run `apex plugins --json` for the full description of every option.
- Use `apex validate` on your configuration, then `apex run`, and inspect results with `apex summary` on the run directory.
- The other plugin pages in this guide, for the components that use a simulator's outputs.

---
audience: public
items:
  - cli:apex ert history-match
  - cli:apex plugins
  - cli:apex report
  - cli:apex resume
  - cli:apex run
  - cli:apex stop
  - cli:apex summary
  - cli:apex validate
  - config:history_match:method
  - config:history_match:prior
  - plugin:executors:ert
  - plugin:gmac_constraints:linear
  - plugin:gmac_transforms:state
  - plugin:methods:multilevel_esmda
  - plugin:priors:file
  - plugin:simulators:eclipse
  - plugin:simulators:ert
  - plugin:simulators:linear
  - plugin:simulators:opm_flow
  - plugin:simulators:toy_reservoir
---

# Simulators

A simulator plugin turns the inputs of one member (one realization of the ensemble) into outputs. This page covers the simulator plugins, their options and defaults, and how to configure one.

Some names also exist in other plugin registries, where they are not simulators: `ert` is also an executor, `linear` is also a `gmac_constraints` plugin, `file` is also a prior generator, and `state` is a `gmac_transforms` plugin. This page covers only the registry `simulators`.

## Goal

By the end of this page you will be able to:

- choose the right simulator plugin for your case;
- write the simulator section of a configuration in TOML;
- check it with `apex validate` before you start a run with `apex run`.

## Prerequisites

- An APEX run configuration (TOML) in which you can add a simulator section.
- For `eclipse`: Eclipse (E100 or E300) and its `eclrun` launcher installed.
- For `opm_flow`: OPM Flow installed (the executable is called `flow` by default).
- For `ert`: an ERT case file (`.ert`).
- For `eclipse` and `opm_flow`: a deck template file, that is, your simulator input deck (`.DATA`) stored as a Mako template. See "Deck templates" below.

## The plugins at a glance

Run `uv run apex plugins` to list every plugin with its options. Add `--json` for the full descriptions as JSON.

| Plugin | What it does |
| --- | --- |
| `eclipse` | Runs Eclipse (E100 or E300) through `eclrun` for one member in its work directory. |
| `ert` | Runs an ERT case's forward model, run by ERT through the `ert` executor. |
| `linear` | Computes `y = f(G x)` for a fixed matrix `G`, where `f` is the identity or `tanh`. |
| `opm_flow` | Runs OPM Flow for one member in its work directory. |
| `toy_reservoir` | Returns field totals of a toy waterflood, for optimization tests and demos. Its status is `validated`. |

Only `toy_reservoir` reports a status; the other four report none.

## Steps

1. Pick the plugin from the table above.
2. Add a `[simulator]` section to your configuration. Its `name` is the plugin name, and its settings go in a `[simulator.options]` table.
3. Set every required option for that plugin (listed below). Leave the others at their defaults unless you need to change them.
4. Use only the options listed for that plugin. Each plugin rejects options it does not define.
5. Check the file with `uv run apex validate your-config.toml`. This validates the configuration and plans its cost; nothing is simulated. Add `--json` to get the validation as JSON, or `--root` to choose the services root (default: `$APEX_HOME`, else `~/.apex/service`).
6. Start the run with `uv run apex run your-config.toml`.

### Stopping and continuing a run

A run works in analysis steps and can be stopped and continued from its last checkpoint. You pass the run's directory (`RUN_DIR`) to the commands below.

- `uv run apex run your-config.toml --max-steps 3` stops after 3 analysis steps; continue later with `apex resume`.
- `uv run apex stop RUN_DIR` asks a running run to stop at its next checkpoint.
- `uv run apex resume RUN_DIR` continues a stopped or interrupted run from its last checkpoint. It also accepts `--max-steps`.
- `--no-progress` turns off the progress display on standard error (it is on by default) for `apex run` and `apex resume`.

To look at results, use `uv run apex report RUN_DIR` (data mismatch history and performance metrics; add `--plots` to also write figures, `--output` to choose their directory) or `uv run apex summary PATHS` (every metric at every step). `PATHS` can be run directories or folders holding runs, which are searched, so you can point `apex summary` at a folder if you do not know the exact run directory.

## Options

### Deck templates

For `eclipse` and `opm_flow`, `deck_template_file` is a Mako template of the `.DATA` deck. Two things are passed to it:

- `${case_dir}` is the folder named by the option `case_dir` (default: the template's folder). Use it in `INCLUDE` paths so that the deck finds its data files:

```
INCLUDE
  '${case_dir}/grid.grdecl' /
```

- `template_values` is a table of extra fixed values passed to the template.

If the template lives in one folder and the data files in another, set `case_dir` to the data folder:

```toml
[simulator]
name = "opm_flow"

[simulator.options]
deck_template_file = "templates/DECK.DATA.tmpl"
case_dir = "model"
```

Here `${case_dir}/grid.grdecl` resolves to `model/grid.grdecl`, while the template stays in `templates`.

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

- `version`: `e100` is black oil (`eclrun eclipse`), `e300` is compositional.
- `executable`: the `eclrun` command, looked up on `PATH`.
- `eclrun_args`: extra `eclrun` arguments before the program, for example `["--version=2023.1"]` or `["--nocleanup"]`.
- `timeout_s`: wall-clock limit per run; exceeding it kills the run and fails the member.
- `keep_files`: keeps the member's deck and outputs. With `false`, everything is deleted after reading except the log, the deck, the `.PRT` and `.ECLEND` reports and the summary files.
- `licenses`: the number of Eclipse licences available. At most this many members run at once; none means no cap.

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

- `executable`: the OPM Flow command, looked up on `PATH`.
- `flow_args`: extra command-line arguments, for example `["--threads-per-process=1"]`.
- `timeout_s`: wall-clock limit per run; exceeding it fails the member.
- `keep_files`: with `false`, everything is deleted after reading except the log and the summary files, to save disk space.

### Options shared by `eclipse` and `opm_flow`

- `transient_patterns`: extra case-insensitive regular expressions. A failed run whose log matches one is transient, and the executor retries it. This is on top of built-in licence and file-system patterns.
- `levels`: see "Fidelity levels" below. You do not set it here.

### Fidelity levels

A multilevel ensemble splits its members over several models of different fidelity: a cheap coarse model and the expensive fine model. Each part is a level. The `multilevel_esmda` method (ES-MDA's schedule on such an ensemble) is configured in `[method]` with one table per level, coarsest first. Each level table has `members` (at least 2) and `weight`; the weights are normalised to sum 1, and the members of all levels add up to `[prior] ensemble_size`. A level may also have its own `deck_template_file`, or, for `eclipse`, an `upscale` table that coarsens the grid with Eclipse's `COARSEN`. OPM Flow has no `COARSEN`, so a coarse `opm_flow` level needs its own deck. A level with no simulator key runs the fine model.

APEX passes these levels on to the simulator, so the simulator's `levels` option stays at none. Leave it alone unless you use `multilevel_esmda`, and then set the levels in `[method]`, not in `[simulator.options]`.

A complete example with two levels (15 + 5 = 20 members); the `[prior]` here reads a stored ensemble with the `file` generator, so it needs one array per state variable in the `.npz`:

```toml
[prior]
ensemble_size = 20

[prior.generator]
name = "file"

[prior.generator.options]
ensemble_file = "prior.npz"

[method]
name = "multilevel_esmda"

[method.options]
inflation = [2.0, 2.0]

[[method.options.levels]]
members = 15
weight = 0.3
deck_template_file = "templates/COARSE.DATA.tmpl"

[[method.options.levels]]
members = 5
weight = 0.7

[simulator]
name = "opm_flow"

[simulator.options]
deck_template_file = "templates/FINE.DATA.tmpl"
```

The `inflation` factors `alpha_i` must satisfy sum of `1 / alpha_i` = 1, so `[2.0, 2.0]` is valid. The other sections a run needs (such as the data to match) are not shown; `uv run apex validate` reports what is missing from your file.

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

Terms: the run path is the directory in which ERT runs one member's forward model. A state variable is a quantity APEX updates, such as porosity.

- `case_file`: the ERT configuration (`.ert`).
- `fields`: one entry per field state variable. The key is the state variable's name in APEX; its value is a table with two strings: `keyword`, the field's ERT name (its GRDECL keyword), and `file`, the file in the run path that APEX writes the field to and the forward model reads. In the worked example below, the state variable `poro` has keyword `PORO` and file `poro.grdecl`.
- `eclbase`: the case's `ECLBASE`, the simulator's case name in the run path (`<IENS>` and `<ITER>` are replaced).
- `responses`: with `summary`, each realization's summary is read. With `storage`, the batch keeps the case's `OBS_CONFIG` and the responses of every observation type are read from the batch's ERT storage.
- `results_program`: with `storage`, the program that reads the responses (`python -m apex.integrations.ert.observations`). Leave it at its default; it is filled in when APEX builds the configuration from an ERT case (see below).
- `step_parameters`: for parameters that are not plain fields written through `fields`, but that ERT's own writers put into the run path: `GEN_KW` groups, surfaces, and fields in other formats or with transforms. The table maps each state variable (or property) to its ERT name. The values are saved in `apex_parameters.npz`.
- `parameter_step`: the forward-model step that writes those parameters (`APEX_WRITE_PARAMETERS(<CASE>=...)`), put before the case's first `FORWARD_MODEL`. Leave both `step_parameters` and `parameter_step` at none if all your variables go through `fields`.
- `skip_forward_steps`: the number of the case's first `FORWARD_MODEL` steps to leave out of every batch. These are steps that make `FORWARD_INIT` parameters, run once for the prior.

The `ert` executor (a different plugin, in the registry `executors`) runs a forecast's members as one ERT ensemble experiment, locally. Its options are `workers` (default `1`), `retries` (default `0`), `timeout_s` (default `86400.0`) and `ert_executable` (default none).

#### Letting APEX read the case

Instead of writing the `ert` options yourself, you can run:

```bash
uv run apex ert history-match my_case.ert
```

This history-matches the ERT case with APEX's methods; ERT runs the forward model. The case supplies the fields, observations and steps. Your APEX choices come from the `apex.toml` next to the case, or from a file you pass after the case. Choose this route when you already have a complete ERT case and want APEX's methods on it; write the `ert` options by hand when you want to keep a configuration for `apex run`. The `apex.toml` holds one `[methods.<label>]` section per method to run, with a label of your choice. Flags:

- `--methods a,b` runs only the listed `[methods.<label>]` (default: all).
- `--max-steps N` stops after N analysis steps; run the same command again to resume.
- `--no-progress` hides progress output (`--progress` is the default).
- `--no-write-back` stops APEX adding the iterates to the case's ERT storage (by default they are added, so you can see them in ERT's GUI).

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

The plugin computes `y = f(G x)`:

- `variables` lists the state variables stacked, in this order, into `x`. For example `["a", "b"]`.
- `outputs` has one `[time, datatype]` label for each scalar row of `G`, in row order. For example `[["1", "oil"], ["2", "oil"]]` is two rows.
- `fields` has `[time, datatype, size]` for each field output. Each adds `size` further rows of `G`, after the scalar ones, with keys `(time, datatype, 0..size-1)`.
- `matrix` or `forward_file` gives `G` with shape `(n_outputs, n_state)`: `matrix` directly, `forward_file` as an `.npz` holding it as `matrix`. Neither is marked required, but you need one of them.
- `nonlinearity` chooses `f`: `none` is the identity, `tanh` applies tanh.

The remaining options are test aids; leave them at their defaults for normal use:

- `fail_members`: member ids whose runs fail with `SimulationError`.
- `fail_above`: runs whose state has a value above this fail, so failures appear mid-run as iterates move.
- `flaky_members`: member id to the number of times its run fails transiently before it succeeds, to test the executors' retries.
- `delay_s`: seconds each run waits before answering.
- `resources`: machine-wide tokens each run holds, for example `{ cpu = 1 }`, to test admission.
- `write_files`: gives each run a work directory and writes `output.txt` there, to test work-directory handling such as `[output] keep_simulations`.
- `levels`: filled in for multilevel ensembles (see "Fidelity levels"); per level `first`, `size` and optionally its own `G` (`matrix` or `forward_file`). You normally do not set it.

### `toy_reservoir`

No required options. All have defaults.

| Option | Type | Default |
| --- | --- | --- |
| `start` | string | `2000-01-01` |
| `control` | string | `inj` |
| `oil_rate` | number | `20.0` |
| `water_rate` | number | `1.0` |

`start` is the simulation's start as an ISO date; `TIME` is the days since. `control` is the name of the controls block (the member's state key). `oil_rate` gives the yearly oil per square root of pressure, and `water_rate` the water per unit of pressure. Raising `oil_rate` from `20.0` to `25.0` therefore scales the oil totals by 1.25 at the same pressure.

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

A small `linear` simulator with two variables and two outputs, `G` given inline:

```toml
[simulator]
name = "linear"

[simulator.options]
variables = ["a", "b"]
outputs = [["1", "oil"], ["2", "oil"]]
matrix = [[1.0, 0.0], [0.5, 2.0]]
nonlinearity = "none"
```

For Eclipse, set the required deck template and, if needed, the version and timeout:

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

An `ert` section sets its three required options. The state variable `poro` is written as the GRDECL keyword `PORO` to the file `poro.grdecl` in each run path:

```toml
[simulator]
name = "ert"

[simulator.options]
case_file = "my_case.ert"
eclbase = "MODEL"

[simulator.options.fields.poro]
keyword = "PORO"
file = "poro.grdecl"
```

These examples show only the simulator section; the rest of your run configuration goes in the same file. After editing, run:

```bash
uv run apex validate your-config.toml
```

This confirms the file is accepted and plans its cost, and it catches mistakes such as a misspelled option. Then start with:

```bash
uv run apex run your-config.toml
```

## Pitfalls

- **Wrong section layout.** The plugin name goes in `name` under `[simulator]`, and its settings go under `[simulator.options]`. Do not write a section named after the plugin.
- **Missing required options.** `eclipse` and `opm_flow` need `deck_template_file`; `ert` needs `case_file`, `fields` and `eclbase`; `linear` needs `outputs` and `variables`, and in practice `matrix` or `forward_file`.
- **Unknown options.** Every simulator rejects options it does not define. Do not carry an `eclipse` option such as `version` over to `opm_flow`.
- **Fixed choices.** `version` accepts only `e100` or `e300`. `nonlinearity` accepts only `none` or `tanh`. `responses` accepts only `summary` or `storage`.
- **Timeouts.** `timeout_s` defaults to `3600.0` seconds for `eclipse` and `opm_flow`. Raise it for long runs.
- **Disk use.** `keep_files` defaults to `true`, so member files are kept. Set it to `false` if you do not need them.
- **Executable names.** `executable` defaults to `eclrun` for `eclipse` and `flow` for `opm_flow`. Set it only if your installation uses a different name.
- **Setting `levels` by hand.** It is filled in from the levels of `multilevel_esmda`; leave it alone otherwise.
- **Demo plugin.** `toy_reservoir` is meant for optimization tests and demos, not for real reservoir models.

## What to read next

- Run `uv run apex plugins --json` for the full description of every option.
- Use `uv run apex validate` on your configuration, then `uv run apex run`, and inspect results with `uv run apex report` or `uv run apex summary` on the run directory.
- The other plugin pages in this guide, for the components that use a simulator's outputs: the `[prior]` generators (such as `file`) and the `[method]` algorithms (such as `multilevel_esmda`).

---
audience: public
items:
  - cli:apex plugins
  - cli:apex resume
  - cli:apex run
  - cli:apex stop
  - plugin:executors:ert
  - plugin:executors:process
  - plugin:executors:serial
---

# Executors

## Goal

By the end of this page you will know which three executors exist in APEX (`serial`, `process` and `ert`), which options each one accepts, and how to write them in a TOML run configuration. You will also learn how to inspect plugins with `apex plugins`, and how to stop and continue a run with `apex run --max-steps`, `apex stop` and `apex resume`.

## Prerequisites

- A working APEX installation, so that the `apex` command is available.
- A run configuration file in TOML. This is the file you pass to `apex run` as its `config` argument.

## Background: what an executor is

A history-matching run adjusts a set of models so that they agree better with observed data. Each model in that set is a **member**. The run proceeds in **analysis steps**. After a step the run reaches a **checkpoint**, where its state is saved, so a run can stop and be continued later.

A **plugin** is a named, replaceable part of APEX. A **registry** is the group of plugins of one kind. The `executors` registry holds the plugins that run members; each of them is an **executor**. Members are run as part of a **forecast**.

| Executor | What it does |
| --- | --- |
| `serial` | Runs members one after another in the calling process. |
| `process` | Runs members in a pool of worker processes. |
| `ert` | Runs a forecast's members as one ERT ensemble experiment, locally. |

Each executor accepts only the options listed below, and an unknown option is rejected. A run names its executor in the `[execution]` section of its configuration: `name` is the executor and `options` holds its options.

## Options

### `serial`

| Option | Type | Default |
| --- | --- | --- |
| `retries` | integer | `2` |
| `retry_delay_s` | number | `30.0` |

### `process`

| Option | Type | Default |
| --- | --- | --- |
| `workers` | integer or null | `null` |
| `retries` | integer | `2` |
| `retry_delay_s` | number | `30.0` |

The default of `workers` is `null`. TOML has no null value, so to keep that default, leave the line out. To pick a value yourself, set an integer.

### `ert`

| Option | Type | Default |
| --- | --- | --- |
| `workers` | integer | `1` |
| `retries` | integer | `0` |
| `timeout_s` | number | `86400.0` |
| `ert_executable` | string or null | `null` |

The default of `ert_executable` is `null`. To keep it, leave the line out. To set it, give a string such as `"ert"`.

This page gives names, types and defaults only. It does not define the exact rules behind `retries`, `retry_delay_s` or `timeout_s` (for example, what the timeout covers or what happens when it is reached). To read the full description of every option as installed, run `apex plugins --json`.

## Steps

1. Run `apex plugins` to list every plugin with its options. Check that `serial`, `process` and `ert` appear. Add `--json` for full descriptions.
2. Open the TOML file that you pass to `apex run`.
3. In the `[execution]` section, set `name` to the executor you want, for example `name = "process"`.
4. Set only the options you want to change, in `options`. Each option has a default, so you can leave any of them out, or give no `options` to keep all defaults.
5. Save the file and start the run with `apex run config.toml`, using the name of your file.

## Worked example

Suppose you want a pool of four worker processes, more retries than the default and a shorter delay between retries:

```toml
[execution]
name = "process"
options = { workers = 4, retries = 5, retry_delay_s = 10.0 }
```

Start the run:

```bash
apex run config.toml
```

To keep every default of `serial`, give its name alone:

```toml
[execution]
name = "serial"
```

An `ert` section that sets all of its options:

```toml
[execution]
name = "ert"
options = { workers = 2, retries = 1, timeout_s = 3600.0, ert_executable = "ert" }
```

### Working through a long run in pieces

`--max-steps` makes `apex run` stop after the given number of analysis steps. This command **starts a new run** and stops it after three steps:

```bash
apex run config.toml --max-steps 3
```

Do not repeat this command to carry on. Each run lives in its own folder, and if that folder already holds a run, `apex run` refuses to start and exits with an error. The error tells you your options: resume the run with `apex resume`, or start a new one in another folder (another `[run]` name, or `[run]` `output_dir`, in the configuration), or move the old folder away first. This page does not cover the `[run]` section further.

To continue a stopped or interrupted run from its last checkpoint, pass the directory of that run to `apex resume`. Here `runs/my_run` stands for the directory of your own run; use the directory your run actually uses:

```bash
apex resume runs/my_run
```

`apex resume` also accepts `--max-steps`:

```bash
apex resume runs/my_run --max-steps 3
```

The option is described only as stopping after this many analysis steps. Watch the progress output to see where the run stops.

To stop a run that is still going, open a second terminal and give `apex stop` the directory of that running run. It asks the run to stop at its next checkpoint, so it may not stop at once. `apex resume` continues it later:

```bash
apex stop runs/my_run
```

Both `apex run` and `apex resume` show progress on standard error: bars on a terminal, timestamped lines otherwise. Add `--no-progress` to switch this off (`--progress` switches it on, which is the default).

## Pitfalls

- **Wrong type.** `workers` and `retries` must be integers, so write `3`, not `3.5`. `retry_delay_s` and `timeout_s` may be decimal numbers.
- **Unknown options.** Each executor accepts only its own options. For example `retry_delay_s` is not an option of `ert`, and `timeout_s` is not an option of `serial` or `process`.
- **Null defaults.** `workers` of `process` and `ert_executable` of `ert` default to `null`. TOML has no null value, so leave the line out instead of writing a value.
- **Different defaults.** `ert` has `retries = 0`, while `serial` and `process` have `retries = 2`.
- **One at a time.** Because `serial` runs members one after another, a slow member delays all the members after it.
- **One executor per run.** `[execution]` names one executor; its options go in `options`.
- **Run folder already in use.** Running `apex run` again with a configuration whose run folder already holds a run fails with an error. Use `apex resume` with the run directory to continue it, or give the new run another `[run]` name or `output_dir`.

## What to read next

Run `apex plugins` or `apex plugins --json` to explore the options of every plugin from the command line. Then continue with the other pages under Plugins in this guide.

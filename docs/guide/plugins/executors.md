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
  - plugin:simulators:ert
---

# Executors

## Goal

By the end of this page you will know which three executors APEX has (`serial`, `process` and `ert`), which options each accepts, and how to write them in a TOML run configuration. You will also learn to inspect plugins with `apex plugins`, and to stop and continue a run with `apex run --max-steps`, `apex stop` and `apex resume`.

## Prerequisites

- A working APEX installation. After the README's installation, type every command as `uv run apex ...` from the APEX folder.
- A run configuration file in TOML. You pass it to `apex run` as its `config` argument.

## Background: what an executor is

A history-matching run adjusts a set of models so that they agree better with observed data. Each model in that set is a **member**. The run proceeds in **analysis steps**. After a step the run reaches a **checkpoint**, where its state is saved, so a run can stop and be continued later.

A **plugin** is a named, replaceable part of APEX, and a **registry** is the group of plugins of one kind. The `executors` registry holds the plugins that run members; each is an **executor**. A **forecast** is a batch of members that the run hands to the executor to simulate. A **simulator** is the plugin, in the `simulators` registry, that describes the model program being run for each member; the executor decides how and how many of those simulations run at once.

| Executor | What it does |
| --- | --- |
| `serial` | Runs members one after another in the calling process. |
| `process` | Runs members in a pool of worker processes. |
| `ert` | Runs a forecast's members as one ERT ensemble experiment, locally. |

ERT is an external program that runs an ensemble of simulations. The `ert` executor calls it, so the `ert` program must be available (see `ert_executable` below). ERT calls a member a *realization*; on this page the two words mean the same thing.

The `ert` executor goes together with the `ert` simulator plugin, which describes an ERT case's forward model and is run by ERT through the `ert` executor. This page covers only the executors.

Each executor accepts only the options listed below, and an unknown option is rejected. A run names its executor in the `[execution]` section of its configuration: `name` is the executor and `options` holds its options.

## Options

### `serial`

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `retries` | integer | `2` | Retries of a member's transient failures. |
| `retry_delay_s` | number | `30.0` | The first wait before a retry, in seconds. |

### `process`

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `workers` | integer or null | `null` | Number of worker processes. When left out, it is the CPU count. |
| `retries` | integer | `2` | Retries of a member's transient failures. |
| `retry_delay_s` | number | `30.0` | The first wait before a retry, in seconds. |

The default of `workers` is `null`, and then the pool has as many processes as the machine has CPUs. TOML has no null value, so to keep that default, leave the line out. To pick a value yourself, set an integer.

Two things to keep in mind when you choose `workers`:

- Simulators that launch multi-threaded or MPI programs use several cores per run, so budget those cores within `workers`. Example: a machine has 16 CPUs and each simulation uses 4 cores. Set `workers = 4`, so that 4 runs at once use the 16 cores.
- A simulator can declare `max_parallel_runs`, the most runs it allows at once (for example, Eclipse's `licenses`). That value caps `workers`: there are never more runs at once than the simulator allows, even if `workers` is larger.

### `ert`

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `workers` | integer | `1` | Realizations ERT runs at once (ERT's `MAX_RUNNING`). |
| `retries` | integer | `0` | Resubmissions of a failed realization (ERT's `MAX_SUBMIT` is 1 + `retries`). |
| `timeout_s` | number | `86400.0` | Longest a batch may take. |
| `ert_executable` | string or null | `null` | The `ert` program. When left out, APEX uses the one in its own environment, else the one on `PATH`. |

To set `ert_executable`, give a string such as `"ert"`. To keep the default, leave the line out, because TOML has no null value.

### About retries

`retries` is how many times a member's transient failure is tried again. `retry_delay_s` sets the first wait before a retry. The option descriptions say nothing more about the waits that follow, so do not assume that they are the same or that they grow.

To read the full description of every option as installed, run `uv run apex plugins --json`.

## Steps

1. Run `uv run apex plugins` to list every plugin with its options. Look for `serial`, `process` and `ert` among the executors. If one is missing, your installation differs from this page; run it again with `--json` and check the registry of each entry.
2. Open the TOML file that you pass to `apex run`.
3. In the `[execution]` section, set `name` to the executor you want, for example `name = "process"`.
4. Set only the options you want to change, in `options`. Each option has a default, so you can leave any out, or give no `options` to keep all defaults.
5. Save the file and start the run with `uv run apex run config.toml`, using the name of your file.

```bash
uv run apex plugins
```

## Worked example

Suppose you want a pool of four worker processes, more retries than the default and a shorter first delay between retries:

```toml
[execution]
name = "process"
options = { workers = 4, retries = 5, retry_delay_s = 10.0 }
```

Start the run:

```bash
uv run apex run config.toml
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

`--max-steps` makes `apex run` stop after the given number of analysis steps. This command starts a run and stops it after three steps:

```bash
uv run apex run config.toml --max-steps 3
```

The run is stored in a folder that comes from the `[run]` section of your configuration (its `name`, or `output_dir`). If that folder already holds a run, `apex run` refuses to start and prints an error that names the folder. You then have three choices: resume that run with `apex resume`, start a new run in another folder (another `[run]` name or `output_dir`), or move the old folder away. The folder named in that message is the run directory you give to `apex resume` and `apex stop`.

To continue a stopped or interrupted run from its last checkpoint, pass its directory to `apex resume`. Here `runs/my_run` stands for your own run directory:

```bash
uv run apex resume runs/my_run
```

`apex resume` also accepts `--max-steps`:

```bash
uv run apex resume runs/my_run --max-steps 3
```

The option is described only as "stop after this many analysis steps; continue later with `apex resume`". The description does not say whether the count starts at the beginning of the run or at the checkpoint you resume from. Do not rely on either reading: after a resume with `--max-steps`, watch the progress output to see at which step the run stopped, and plan the next `apex resume` from that.

To stop a run that is still going, open a second terminal and give `apex stop` the directory of that running run. It asks the run to stop at its next checkpoint, so it may not stop at once. `apex resume` continues it later:

```bash
uv run apex stop runs/my_run
```

Both `apex run` and `apex resume` show progress on standard error: bars on a terminal, timestamped lines otherwise. Add `--no-progress` to switch this off (`--progress` switches it on, which is the default).

## Pitfalls

- **Wrong type.** `workers` and `retries` must be integers, so write `3`, not `3.5`. `retry_delay_s` and `timeout_s` may be decimal numbers.
- **Unknown options.** Each executor accepts only its own options. For example `retry_delay_s` is not an option of `ert`, and `timeout_s` is not an option of `serial` or `process`. `serial` has no `workers` option either.
- **Null defaults.** `workers` of `process` and `ert_executable` of `ert` default to `null`. TOML has no null value, so leave the line out instead of writing a value.
- **Different defaults.** `ert` has `retries = 0` and `workers = 1`, while `serial` and `process` have `retries = 2`.
- **Same name, two registries.** `ert` exists both as an executor and as a simulator. The `[execution]` section takes the executor.
- **Too many workers.** `process` never runs more members at once than the simulator allows, and simulators that use several cores need their share within `workers`.
- **One at a time.** Because `serial` runs members one after another, a slow member delays all the members after it.
- **Starting over an existing run.** `apex run` does not overwrite a folder that already holds a run; resume it or choose another folder.
- **One executor per run.** `[execution]` names one executor; its options go in `options`.

## What to read next

Read the simulators page in the Plugins section of this guide next, especially if you use the `ert` executor, which goes with the `ert` simulator. Run `uv run apex plugins --json` any time to see the full description of every plugin option.

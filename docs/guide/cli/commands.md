---
audience: public
items:
  - cli:apex approve
  - cli:apex data-split
  - cli:apex gc
  - cli:apex mcp
  - cli:apex optimize
  - cli:apex pet-import
  - cli:apex plugins
  - cli:apex predict
  - cli:apex qaqc
  - cli:apex report
  - cli:apex resume
  - cli:apex run
  - cli:apex schema
  - cli:apex stop
  - cli:apex submit
  - cli:apex summary
  - cli:apex validate
  - cli:apex version
  - config:service:approval
  - config:service:guests
---

# The apex commands

## Goal

This page is a tour of the `apex` commands: `apex approve`, `apex data-split`, `apex gc`, `apex mcp`, `apex optimize`, `apex pet-import`, `apex plugins`, `apex predict`, `apex qaqc`, `apex report`, `apex resume`, `apex run`, `apex schema`, `apex stop`, `apex submit`, `apex summary`, `apex validate` and `apex version`. By the end you will know what each command does, which arguments it needs, and which flags change its behavior.

## Prerequisites

- `apex` installed and available on your command line. Check with `apex version`.
- For most commands, a TOML configuration of your case. This page does not describe the contents of a configuration. To see what a configuration kind accepts, use `apex schema`.

## How the options work

- An **argument** is positional: you write its value without a flag.
- An **option** is given with a flag, such as `--output`.
- Some options are on/off switches. The first flag turns the switch on and the other flag turns it off, for example `--plots` and `--no-plots`. They are two positions of one switch, not aliases.
- Several commands take `--root`, the services root: the folder that the services (job submission, the run catalogue, approvals) use. Its default is `$APEX_HOME`, otherwise `~/.apex/service`. Override it only if you keep your services somewhere else.
- Several commands take `--progress` / `--no-progress`. Progress is shown on standard error: bars on a terminal, timestamped lines otherwise. It is on by default.

## Words used on this page

- **Analysis step**: one step of a history-matching run. `--max-steps` counts these. A small value such as `1` or `2` makes the run pause early, which is handy for a first look; you then continue with `apex resume`.
- **Checkpoint**: a point from which a run can be continued. `apex stop` and `apex resume` work with checkpoints.
- **Iterate**: the state of the run at one iteration. The **posterior** is the result of the run, and it is what `apex qaqc` checks unless you ask for another iterate.
- **Accepted iteration**: an iteration of an optimization that the optimizer accepted. `--max-iterations` counts these, whereas `--max-steps` of `apex run` counts analysis steps of a history-matching run.
- **ES**: ensemble smoother, the update that `apex qaqc` uses when its damping is `0`.
- **PET**: the toolkit whose history-matching configurations `apex pet-import` converts, and whose optimizers `apex optimize` uses.

## The commands at a glance

| Command | What it does |
|---|---|
| `apex version` | Print the installed APEX version. |
| `apex plugins` | List every plugin with its status and options. |
| `apex schema` | Print the JSON Schema of a configuration kind. |
| `apex validate` | Validate a configuration and plan its cost; nothing is simulated. |
| `apex run` | Start a history-matching run. |
| `apex submit` | Run a configuration, or resume a run, in the job service's background. |
| `apex stop` | Ask a running run to stop at its next checkpoint. |
| `apex resume` | Continue a stopped or interrupted run from its last checkpoint. |
| `apex predict` | Forecast ensembles of a history-matching run beyond history. |
| `apex report` | Print the data mismatch history and performance metrics of a run. |
| `apex summary` | Summarise runs: every metric at every step. |
| `apex qaqc` | Check the quality (QA/QC) of an iterate of a run: coverage, Mahalanobis scores, Kalman gains and movement. |
| `apex data-split` | Explore how to split production data for cross validation. |
| `apex optimize` | Minimise an objective with one of PET's optimizers. |
| `apex pet-import` | Convert a PET history-matching configuration into an APEX case. |
| `apex approve` | Approve a pending submission. |
| `apex gc` | Forget runs whose folder is gone, and prune what the service keeps. |
| `apex mcp` | Serve APEX's tools to an MCP host. |

## Inspecting APEX

### `apex version`

Prints the installed APEX version. It has no options.

### `apex plugins`

Lists every plugin with its status and options. Add `--json` (off by default) to get the full descriptions as JSON, which is the form meant for clients.

### `apex schema`

Prints the JSON Schema of a configuration kind. The optional argument `kind` is one of `history_match`, `prediction` or `study`; the default is `history_match`.

```bash
uv run apex schema prediction
```

## Checking, running and controlling a run

### `apex validate`

Validates a configuration and plans its cost. Nothing is simulated, so it is a safe first step.

- `config` (required): a run, prediction or study (TOML).
- `--json`: print the validation as JSON.
- `--root`: the services root.

### `apex run`

Starts a history-matching run.

- `config` (required): the run configuration (TOML).
- `--max-steps`: stop after this many analysis steps; continue later with `apex resume`.
- `--progress` / `--no-progress`: show or hide progress (default: on).

### `apex submit`

Runs a configuration, or resumes a run, in the background of the job service, so the command does not have to stay open while the work goes on. Without `--wait` the command returns at once; the job carries on in the background. To follow it, find its run directory and look at it with `apex report` or `apex summary` (`apex summary` also searches folders for runs).

- `config` (optional): a run, prediction or optimization configuration.
- `--resume`: resume this run directory instead of starting a configuration.
- `--kind`: `run`, `prediction` or `optimization`. Default: taken from the file.
- `--max-steps`: stop after this many steps (iterations of an optimization).
- `--code`: which copy of the APEX code the job runs with. It takes one value: `live`, `frozen` or a commit. `live` uses the code as it is now. `frozen` uses the copy of the last commit (for a resume, the run's own copy), so later edits do not affect the job. A commit, for example its hash, uses the code of that commit. Default: the `[jobs]` setting `cli_code`, which is `frozen`.
- `--priority`: `high`, `normal` or `low`. Default: `normal`.
- `--yes`: approve a large job without asking.
- `--approval-token`: a token approved with `apex approve`.
- `--force-new` / `--no-force-new`: run again even if an identical run exists. Default: off.
- `--wait` / `--no-wait`: wait for the job to end. Default: off.
- `--root`: the services root.

```bash
uv run apex submit case/run.toml --priority low --wait
uv run apex submit case/run.toml --code live
```

The first command submits the run in `case/run.toml` with low priority and waits for the job to end. The second runs the job with the code as it is now instead of the frozen copy.

### `apex stop`

Asks a running run to stop at its next checkpoint. The run therefore stops when it reaches that checkpoint, not necessarily at the moment you type the command. `apex resume` continues it. The one argument, `run_dir` (required), is the directory of a running run.

### `apex resume`

Continues a stopped or interrupted run from its last checkpoint.

- `run_dir` (required): the directory of the run to continue.
- `--max-steps`: stop after this many analysis steps; continue later with `apex resume` again.
- `--progress` / `--no-progress`: show or hide progress (default: on).

### `apex predict`

Forecasts the ensembles of a history-matching run beyond history. Do the history matching first with `apex run`, because the forecast is of such a run.

- `config` (required): the prediction configuration (TOML).
- `--progress` / `--no-progress`: show or hide progress (default: on).

This page does not list the contents of a prediction configuration. To write one, print what it accepts with `apex schema prediction`; the schema also shows how the file refers to the run to forecast. Save your file (below it is called `case/predict.toml`), check it, then forecast:

```bash
uv run apex schema prediction
uv run apex validate case/predict.toml
uv run apex predict case/predict.toml
```

The first command shows what a prediction configuration accepts, the second checks your file and plans its cost, and the third makes the forecast. `apex report` also accepts the directory of a prediction, so you can look at the result with it.

### `apex optimize`

Minimises an objective with one of PET's optimizers (EnOpt, GenOpt, SmcOpt, ...). The optimization configuration (TOML) is where you set up the problem; its contents are not covered on this page.

- `config` (optional): the optimization configuration (TOML).
- `--resume`: continue the optimization run in this directory.
- `--max-iterations`: stop after this many accepted iterations (resumable).
- `--progress` / `--no-progress`: show or hide progress (default: on).

```bash
uv run apex optimize case/optimize.toml --max-iterations 5
uv run apex optimize --resume path/to/optimization_run
```

The first command stops after 5 accepted iterations. The second continues the optimization run in the directory you give; replace the path with the directory of your own optimization run.

## Looking at results

### `apex report`

Prints the data mismatch history and the performance metrics of a run.

- `run_dir` (required): the directory of a run or of a prediction.
- `--plots` / `--no-plots`: also write the result figures (PNG and PDF). Default: off.
- `--output`: the directory for the figures. Default: `RUN_DIR/report`.
- `--font-size`: font size of the figures in points. Default: `12.0`.

### `apex summary`

Summarises runs, with every metric at every step (iterates, forecasts, optimization steps).

- `paths` (required): run directories, or folders holding runs (these are searched).
- `--output`: write the Markdown here, and the CSV next to it, instead of standard output.
- `--title`: the title of the report. Default: `Results summary`.

Because folders are searched, `apex summary` is also a way to find the run directories inside a folder.

### `apex qaqc`

Checks the quality (QA/QC) of an iterate of a history-matching run. The report covers coverage, Mahalanobis scores, Kalman gains and movement (PET). Use it after a run to see how an iterate behaves; by default it checks the posterior, the result of the run.

- `run_dir` (required): the directory of a history-matching run.
- `--iteration`: the iterate to check, given as one number after the flag, for example `--iteration 2`. Default: the posterior. To see which iterates your run has, look at the steps listed by `apex report` or `apex summary` for that run.
- `--output`: the directory for the report. Default: `RUN_DIR/qaqc`.
- `--damping`: the lambda, that is, the damping of the Kalman gains. `0` means ES (ensemble smoother), with no damping. Default: `0.0`.
- `--font-size`: font size of the figures in points. Default: `12.0`.

```bash
uv run apex qaqc runs/case_run --damping 0.5 --output runs/case_run/qaqc_damped
```

This checks the posterior of the run in `runs/case_run` with a damping of `0.5` and writes the report to `runs/case_run/qaqc_damped`. Replace the path with the directory of your own run.

## Cross validation and PET

### `apex data-split`

Explores how to split production data for cross validation. In cross validation you hold back a group of wells (a CV set), match the model to the rest, and check how well it predicts the held-back wells. This command explores candidate CV sets and writes results and figures.

- `config` (required): the run configuration (TOML) of the case.
- `--prior-run`: optional. The directory of a run of the same case that has a prior forecast, that is, a forecast made from the prior ensemble. It is used to take the redundancy of the ensemble into account, that is, how much the ensemble members repeat each other.
- `--max-cv`: the largest candidate CV set, in wells.
- `--output`: the directory for the results. Default: `CONFIG dir/data_split`, that is, a `data_split` folder in the directory of the configuration.
- `--font-size`: font size of the figures in points. Default: `12.0`.

```bash
uv run apex data-split case/run.toml --max-cv 3 --output case/split --font-size 14
uv run apex data-split case/run.toml --prior-run runs/prior_forecast --max-cv 3
```

The first command explores candidate splits of at most 3 wells for the case in `case/run.toml` and writes results and figures (font size 14) to `case/split`. The second also passes a run with a prior forecast; replace `runs/prior_forecast` with the directory of your own such run.

### `apex pet-import`

Converts a PET history-matching configuration into an APEX case.

- `pet_config` (required): the PET TOML configuration.
- `output_dir` (required): the directory for the APEX case.
- `--prior`: a stored prior ensemble (`.npz`), if the configuration samples one.
- `--simulator`: the APEX simulator for the runfile. Default: `opm_flow`.
- `--seed`: the run seed. Default: the seed of the configuration.
- `--truth`: the true model (`.npz`, one array per variable). Giving it makes the case synthetic.
- `--pet-geometry` / `--intended-geometry`: one switch for the prior correlation geometry. `--pet-geometry` (the default) uses what PET samples; `--intended-geometry` turns that off and uses what PET's keywords describe.
- `--grid-file`: an EGRID of the deck, for geophysical data. Its active cells fix the AVO traces and the gravity and displacement stations.
- `--avo-angles-checked` / `--no-avo-angles-checked`: import multi-angle AVO data that is known to hold each angle's own traces. SimulatorWrap's data does not: it gives every angle the last one's traces. Default: off.

```bash
uv run apex pet-import pet/config.toml apex_case --seed 7
```

This reads `pet/config.toml`, fixes the seed at 7 and writes the APEX case to `apex_case`. The simulator stays `opm_flow`. Replace the paths with your own. If your PET configuration samples a stored prior, add `--prior` with the path to that `.npz` file.

## Housekeeping and integration

### `apex approve`

Approves a pending submission. Afterwards you submit it again with its token.

- `token` (required): the pending approval's token. It identifies the submission that is waiting for approval.
- `--root`: the services root.

Some submissions need a person's approval before they run. The `[approval]` section of the service configuration (`service:approval`) sets when. It has two options, both integers of at least 1:

- `max_runs`: a limit on the number of runs in a submission. Default: `20`.
- `max_simulator_runs`: a limit on the number of simulator runs in a submission. Default: `20000`.

A submission that goes beyond these limits counts as a large job and waits for approval. This page does not say where the service configuration file lives, so check your installation for it.

```toml
[approval]
max_runs = 20
max_simulator_runs = 20000
```

This is the section with both options at their defaults.

To get a large job started you have these options of `apex submit`:

- `--yes` approves the job on the spot, without asking.
- `--approval-token` takes the token of a pending approval that you approved with `apex approve`.

The token route works in three steps:

1. Submit the configuration. A submission that needs approval is held as a pending approval, which has a token. Note that token.
2. Approve it with `apex approve`.
3. Submit the same configuration again, giving the token to `--approval-token`.

```bash
uv run apex submit case/run.toml
uv run apex approve TOKEN
uv run apex submit case/run.toml --approval-token TOKEN
```

Replace `TOKEN` with the token of the pending approval for your submission and the path with your own configuration.

### `apex gc`

Forgets runs whose folder is gone, and prunes what the service keeps. The switches below add further clean-up and are all off unless you ask for them.

- `--root`: the services root.
- `--chats` / `--no-chats`: also apply the chat history's limits now (`[chat]` `retention_days` and `max_history_mb`). Default: off.
- `--guests` / `--no-guests`: also apply the guests' limits now (`[guests]` `retention_days` and `max_storage_mb`; guests' work only). Default: off.
- `--code` / `--no-code`: also remove frozen copies of APEX that no job or process uses and that are older than `[jobs]` `keep_code_days`, and run the installed maintenance tasks. Default: off.
- `--dry-run` / `--no-dry-run`: with `--code`, list what would go and remove nothing. Default: off.

```bash
uv run apex gc --code --dry-run
```

This lists the frozen copies the clean-up would remove, without removing anything. Run it first, then repeat without `--dry-run` if the list is what you expect.

The guests' limits are set in the `[guests]` section of the service configuration (`service:guests`). They apply only to work that a guest of the web UI submitted; the owner's work and that of other local programs (the CLI, the Python client) is never pruned. The section has two options, both numbers greater than 0:

- `retention_days`: how long guests' work is kept, in days. Default: `30.0`.
- `max_storage_mb`: how much guests' work is kept, in MB. Default: `500.0`.

```toml
[guests]
retention_days = 30.0
max_storage_mb = 500.0
```

This is the section with both options at their defaults.

### `apex mcp`

Serves APEX's tools to an MCP host, that is, a client such as Claude Code that calls those tools. MCP support is in APEX's default environment (`uv sync --locked`); when APEX is a dependency of your own project, name the `mcp` extra.

- `--http`: serve Streamable HTTP at `HOST:PORT`, written as the value after the flag. Default: stdio.
- `--client`: the client name, for quotas and the call log. Default: `mcp`.
- `--model`: the model driving the calls, for the log.
- `--budget-runs`: the number of simulator runs this session may submit.
- `--root`: the services root.

With stdio (the default), the host starts the command itself and talks to it through standard input and output, so no address is needed. With `--http`, the command runs as a server that listens at `HOST:PORT`, and a host connects to that address. How to register either kind of server is set in the host's own settings.

```bash
uv run apex mcp --http 127.0.0.1:8770 --client myclient --budget-runs 10
```

This serves over HTTP at `127.0.0.1:8770` (a port of its own: 8765 is the service's in the README, 8000 `mkdocs serve`'s), names the client `myclient` for quotas and the call log, and lets the session submit at most 10 simulator runs. Without `--http`, the command serves over stdio.

## Worked example: a run from start to finish

```bash
uv run apex validate case/run.toml
uv run apex run case/run.toml --max-steps 2
uv run apex resume runs/case_run
uv run apex report runs/case_run --plots
uv run apex qaqc runs/case_run
uv run apex summary runs --output summary.md
```

1. `apex validate` checks the configuration and plans its cost without simulating anything.
2. `apex run` starts the run and stops after 2 analysis steps.
3. `apex resume` continues from the last checkpoint. It needs the directory of your run. The path `runs/case_run` is a placeholder: this page does not fix where a run directory is created, so use the directory of your own run. If you do not know it, point `apex summary` at the folder that holds your runs; it searches the folder and lists the runs it finds.
4. `apex report` prints the mismatch history and metrics and writes the figures to `runs/case_run/report`.
5. `apex qaqc` checks the posterior and writes its report to `runs/case_run/qaqc`.
6. `apex summary` searches `runs` for runs and writes a Markdown summary to `summary.md`, with a CSV next to it.

If a run is still going and you want it to pause, use `apex stop runs/case_run`, then `apex resume runs/case_run` later.

## Pitfalls

- Positional arguments are required where listed: for example `config` for `apex run`, `apex predict`, `apex validate` and `apex data-split`, and both `pet_config` and `output_dir` for `apex pet-import`.
- `apex optimize` and `apex submit` take `config` as optional, because you can continue with `--resume` instead.
- `--pet-geometry` and `--intended-geometry`, like `--chats` and `--no-chats`, are the on and off positions of one switch, not aliases.
- `--max-steps` and `--max-iterations` only pause the work. Continue with `apex resume` or `apex optimize --resume`.
- `apex stop` stops the run at its next checkpoint, so the run may continue for a while after you ask.
- `apex gc` switches `--chats`, `--guests` and `--code` are off unless you ask for them; use `--dry-run` with `--code` to look before you remove.
- `--dry-run` only has an effect together with `--code`.
- `--code` of `apex submit` takes one value (`live`, `frozen` or a commit); it is not the same switch as `--code` of `apex gc`.
- `--max-cv` counts wells, not data points.
- `--truth` is for synthetic cases and expects one array per variable.
- `--avo-angles-checked` is only for data you know holds each angle's own traces. Leave it off for SimulatorWrap's data.
- `apex mcp` needs MCP support: in APEX's default environment (`uv sync --locked`); when APEX is a dependency of your own project, name the `mcp` extra.

## What to read next

- The pages of this guide on the configuration sections, to understand the TOML files that `apex run`, `apex predict` and `apex validate` take.
- The output of `apex schema` for the kind of configuration you want to write.
- The output of `apex plugins` to see which plugins and options are available.

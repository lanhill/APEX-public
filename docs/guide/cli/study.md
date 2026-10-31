---
audience: public
items:
  - cli:apex approve
  - cli:apex predict
  - cli:apex report
  - cli:apex run
  - cli:apex stop
  - cli:apex study compare
  - cli:apex study run
  - cli:apex study status
  - cli:apex submit
  - cli:apex validate
  - config:optimization_study:study
  - config:service:approval
  - config:service:service
  - config:study:study
---

# The apex study commands

## Goal

By the end of this lesson you will be able to write a small study specification, check it with `apex validate`, submit it with `apex study run`, follow it with `apex study status`, and compare its variants with `apex study compare`. You will also know when `apex approve` is needed, and how `apex study run` differs from `apex run`, `apex predict`, `apex report`, `apex stop` and `apex submit`.

## Prerequisites

- APEX installed, with commands run from the APEX folder as `uv run apex ...`.
- A history-matching configuration (TOML) for each case you want in the study.
- A services root. Every command below that takes `--root` looks there. If you leave `--root` out, APEX uses `$APEX_HOME`, and if that is not set, `~/.apex/service`.
- Optionally, a `service.toml` in the services root. It can set defaults such as the number of jobs at once. Without it the built-in defaults named below apply.

## The commands

| Command | What it does |
| --- | --- |
| `apex validate` | Validates a run, prediction, optimization or study configuration and plans its cost. Nothing is simulated. |
| `apex study run` | Validates and submits a study. Its runs go on in the background. |
| `apex study status` | Shows a study's state and the state of every run. |
| `apex study compare` | Compares a study's variants at each run's posterior. |
| `apex approve` | Approves a pending submission. You then submit it again with its token. |

Related commands that are not used to submit a study:

- `apex run` starts one history-matching run from a run configuration. `--max-steps` stops it after that many analysis steps, and `apex resume` continues it later.
- `apex predict` forecasts the ensembles of a history-matching run beyond history.
- `apex report` prints the data mismatch history and performance metrics of a run directory, and with `--plots` writes figures.
- `apex stop` asks a running run to stop at its next checkpoint.
- `apex submit` runs a single configuration (or resumes a run directory with `--resume`) in the job service's background.

## Step 1: write a study specification

A study specification is a TOML file with a `[study]` section. It runs every **case** (a history-matching configuration) with every **variant** (a change applied to every case), and optionally repeats each combination with several seeds.

The options of `[study]`:

| Option | Meaning |
| --- | --- |
| `name` | Required. Letters, digits, `_`, `.` and `-` only. `apex study status` and `apex study compare` take this name. |
| `cases` | Required. A table of case name to the path of its configuration. At least one case. |
| `variants` | Required. At least one `[[study.variants]]` entry, each with a `name` and an optional `patch`. |
| `seed` | Default 0. The seeds of replicates 1, 2, ... derive from it. |
| `replicates` | Default 1. Seeds per case and variant. |
| `metrics` | `'metric:label'` entries to compare at each run's posterior. An unknown metric, or a label none of the cases has, is refused at validation. |
| `simulator` | `opm_flow` or `eclipse`. Runs every case's deck on that simulator. Unset: each case keeps its own. |
| `methods` | Name to algorithm, each complete (`[study.methods.<name>]`). Unset: each case's own method. |
| `predictions` | Prediction configurations without `run_dir`, run after each run. |
| `annotations` | Table of string values. |
| `budget` | `[study.budget]` with `max_simulator_runs`. The study is refused before anything runs when the plan exceeds it. |

A variant's `patch` is a JSON merge patch (RFC 7386) over each case's configuration.

Here is a complete study with one case and two variants:

```toml
[study]
name = "my-study"
seed = 0
replicates = 2

[study.cases]
base = "cases/base.toml"

[study.budget]
max_simulator_runs = 5000

[[study.variants]]
name = "baseline"

[[study.variants]]
name = "short"

[study.variants.patch.method]
max_iterations = 3
```

Save it as `my-study.toml`. The path `cases/base.toml` is a placeholder for your own history-matching configuration.

An optimization study uses `kind = "optimization"` in `[study]`. Its cases are optimization configurations, and its variants patch those. Its `metrics` can be `objective:final`, `objective:best`, `evaluations:all` and `iterations:all`; with several objectives also `hypervolume:final` and `front_size:final`. It also accepts `reference_point` for the hypervolume. Its `methods` entries are complete `[optimizer]` sections.

```toml
[study]
kind = "optimization"
name = "my-opt-study"
metrics = ["objective:final", "objective:best"]

[study.cases]
base = "cases/opt.toml"

[[study.variants]]
name = "baseline"
```

## Step 2: check it with `apex validate`

`apex validate` takes one required argument, `config`: a run, prediction, optimization or study TOML. It validates the file and plans its cost without simulating anything, so it is a safe first step.

```bash
uv run apex validate my-study.toml
```

- `--json`: print the validation as JSON. Off by default.
- `--root`: the services root.

If the file is not valid, the command reports what is wrong. Fix the file and run it again.

## Step 3: submit with `apex study run`

`apex study run` takes one required argument, `study`, the path to the study specification. It validates the specification and submits the study. The runs continue in the background.

```bash
uv run apex study run my-study.toml
```

Options:

- `--yes`: approve a large study without asking. Off by default.
- `--wait` / `--no-wait`: an on/off switch. `--wait` waits for the study to finish; `--no-wait` turns waiting off. The default is off.
- `--max-concurrent-jobs`: jobs at once. Default: the value in `service.toml`, else 2.
- `--priority`: `high`, `normal` or `low`. Default `normal`.
- `--approval-token`: a token approved with `apex approve`.
- `--code`: `live`, `frozen` (the last commit, for every cell) or a commit. Default: `cli_code` in the `[jobs]` section of `service.toml`, else `frozen`.
- `--root`: the services root.

`service.toml` lives in the services root (`~/.apex/service`, or `$APEX_HOME` if set). Its `[service]` section has `max_concurrent_jobs` (default 2), `poll_s` (default 1.0), `max_retries` (default 2) and `routines` (default true).

### When approval is needed

The `[approval]` section of `service.toml` sets when a person must approve a submission. Both options are integers of at least 1:

```toml
[approval]
max_runs = 20
max_simulator_runs = 20000
```

These are the defaults. A study beyond these limits is "large" and needs approval. There are two ways forward:

1. Pass `--yes` to approve it as you submit:

```bash
uv run apex study run my-study.toml --yes
```

2. Have the pending submission approved with `apex approve`, which takes the pending approval's token (required; `--root` is optional). Then submit the same study again with that token:

```bash
uv run apex approve <token>
uv run apex study run my-study.toml --approval-token <token>
```

Replace `<token>` with your pending approval's token, and use the same `--root` for both commands.

## Step 4: check progress with `apex study status`

`apex study status` takes the study's name (the `name` in `[study]`, here `my-study`) and shows the state of the study and of every run.

```bash
uv run apex study status my-study
```

Add `--root` if your services root is not the default. The name need not match the file name.

## Step 5: compare variants with `apex study compare`

`apex study compare` takes the study's name and compares its variants at each run's posterior.

- `--metric`: given as `metric:label`. Repeat it to compare several metrics.
- `--json`: print the comparison as JSON. Off by default.
- `--root`: the services root.

For an optimization study such as `my-opt-study`, a metric is written like this:

```bash
uv run apex study compare my-opt-study --metric objective:final --metric objective:best
```

For a history-matching study, the `metrics` you list in `[study]` are written as `metric:label`, and the label is a state variable or data type of your cases. A metric a run does not record (for example rmse in a real case) is left out of that run's comparison.

## Worked example

You have `my-study.toml` from Step 1. Check it, then run it at high priority with four jobs at once and wait for it to finish:

```bash
uv run apex validate my-study.toml
uv run apex study run my-study.toml --priority high --max-concurrent-jobs 4 --wait
```

Because of `--wait`, the command returns when the study has finished. If the study is large, add `--yes`, or approve a token and add `--approval-token <token>` as shown above.

Without `--wait`, the command returns right after submission, and you can check on it from another terminal:

```bash
uv run apex study status my-study
```

Once the runs have finished, compare the variants:

```bash
uv run apex study compare my-study --json
```

## Pitfalls

- **`--wait` is off by default.** Without it, `apex study run` returns as soon as the study is submitted. The runs are still going. Use `apex study status` to see where they are.
- **`--no-wait` is not a separate option.** It is the off flag of the same switch as `--wait`.
- **`apex study run` takes a file; the other two take a name.** `apex study status` and `apex study compare` need the `name` from `[study]`.
- **Large studies need approval.** Pass `--yes`, or approve with `apex approve` and submit again with `--approval-token`.
- **Use the same `--root` everywhere.** If you submitted with a non-default `--root`, pass the same value to `apex study status`, `apex study compare` and `apex approve`.
- **Repeat `--metric` for each metric.** Each use takes one `metric:label` value.
- **`--code` changes what the jobs run.** By default the jobs use the last commit (`frozen`), not your uncommitted edits. Use `--code live` for the live code.
- **Study names are restricted.** Use only letters, digits, `_`, `.` and `-` in `name`, for the study and for each variant.
- **A study over its budget is refused.** If the plan exceeds `max_simulator_runs` in `[study.budget]`, nothing runs. Check the planned cost with `apex validate`.
- **An invalid file stops at `apex validate`.** Fix the errors it reports before you submit.

## What to read next

- The page on approvals, for `apex approve` and the tokens used with `--approval-token`.
- The reference for the study specification, for the full list of `[study]` options.
- The page on `apex run` and `apex report`, for working with a single run and its figures.

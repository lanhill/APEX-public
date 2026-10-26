---
audience: public
items:
  - cli:apex approve
  - cli:apex predict
  - cli:apex run
  - cli:apex study compare
  - cli:apex study run
  - cli:apex study status
  - cli:apex validate
  - config:service:approval
---

# The apex study commands

## Goal

By the end of this lesson you will be able to check a study with `apex validate`, submit it with `apex study run`, follow it with `apex study status`, and compare its variants with `apex study compare`. You will also know how `apex approve` fits in when a submission needs approval.

## Prerequisites

- A study specification, which is a TOML file. Its contents are not covered here; this lesson only shows how to check and submit one. Throughout, `my-study.toml` is a placeholder for your own file.
- A services root. Every command below that takes `--root` looks there. If you leave `--root` out, APEX uses `$APEX_HOME`, and if that is not set, `~/.apex/service`.
- Optionally, a service configuration file, `service.toml`. Some defaults below (the number of jobs at once and the code used) can come from it. If it does not set them, the built-in defaults named below apply.

## The commands

| Command | What it does |
| --- | --- |
| `apex validate` | Validates a configuration (a run, a prediction or a study) and plans its cost. Nothing is simulated. |
| `apex study run` | Validates and submits a study. Its runs go on in the background. |
| `apex study status` | Shows a study's state and the state of every run. |
| `apex study compare` | Compares a study's variants at each run's posterior. |
| `apex approve` | Approves a pending submission. You then submit it again with its token. |

`apex run` and `apex predict` are different commands. `apex run` starts a single history-matching run from a run configuration, and `apex predict` forecasts the ensembles of a history-matching run beyond history. Neither is used to submit a study.

## Step 1: check the study with `apex validate`

`apex validate` takes one required argument, `config`, the path to your TOML file (a run, a prediction or a study). It checks the file and plans its cost without simulating anything, so it is a safe first step.

```bash
uv run apex validate my-study.toml
```

- `--json`: print the validation as JSON. Off by default.
- `--root`: the services root.

If the file is not valid, the command prints an error that starts with `error: invalid configuration`, followed by what is wrong, and it exits with status 1. Fix the file and run the command again.

## Step 2: submit with `apex study run`

`apex study run` takes one required argument, `study`, the path to the study specification (TOML). It validates the specification and submits the study. The runs continue in the background.

```bash
uv run apex study run my-study.toml
```

Options:

- `--yes`: approve a large study without being asked. Off by default.
- `--wait` / `--no-wait`: an on/off switch. `--wait` makes the command wait for the study to finish; `--no-wait` turns waiting off. The default is off, so the command returns once the study is submitted.
- `--max-concurrent-jobs`: how many jobs run at once. If you do not set it, the value comes from `service.toml`, and otherwise it is 2.
- `--priority`: `high`, `normal` or `low`. The default is `normal`.
- `--approval-token`: a token approved with `apex approve`.
- `--code`: which code the jobs use. The values are `live`, `frozen` (the last commit) or a commit. If you do not set it, the value comes from `cli_code` in the `[jobs]` section of `service.toml`, and otherwise it is `frozen`. This page does not say which repository the commit belongs to or how a commit is written; check your project's reference before passing one.
- `--root`: the services root.

### When approval is needed

The `[approval]` section of the service configuration (`service:approval`) sets when a person must approve a submission. It has two options, both integers of at least 1:

| Option | Default |
| --- | --- |
| `max_runs` | 20 |
| `max_simulator_runs` | 20000 |

Here is that section written out with its default values, as it would appear in `service.toml`:

```toml
[approval]
max_runs = 20
max_simulator_runs = 20000
```

A study that goes beyond these limits is a "large" study. This page does not show the exact text `apex study run` prints in that case, so read its message when you submit. There are two ways forward:

1. Pass `--yes` to approve it as you submit.
2. Have the pending submission approved with `apex approve <token>`, where `token` is the pending approval's token (a required argument; `--root` is optional). Then submit the study again with `--approval-token`.

The order is: submit the study, get the pending approval's token, approve that token, then submit the same study again with the token.

```bash
uv run apex approve <token>
uv run apex study run my-study.toml --approval-token <token>
```

Replace `<token>` with the token of your pending approval. Who is allowed to approve is a matter of how your service is run; the command itself only needs the token and the same services root.

## Step 3: check progress with `apex study status`

`apex study status` takes the study's name as its one required argument, `name`. It shows the state of the study and of every run in it.

```bash
uv run apex study status my-study
```

Add `--root` if your services root is not the default. The study's name is set by the study specification, so it need not match the file name. `my-study` here is a placeholder: use your own study's name here and in the next step. Read the state of each run in the output; the comparison in the next step is meant for runs that have finished.

## Step 4: compare variants with `apex study compare`

`apex study compare` takes the study's name, `name`, and compares the study's variants at each run's posterior. The variants are the alternatives your study defines, and the posterior is where each run ends up after history matching.

- `--metric`: given as `metric:label`. You can repeat the option to compare several metrics.
- `--json`: print the comparison as JSON. Off by default.
- `--root`: the services root.

```bash
uv run apex study compare my-study --json
```

The valid metric names and labels depend on your study, and this page does not list them or state what is compared when `--metric` is left out. Take them from your study's reference material.

## Worked example

You have `my-study.toml`. You want to check it, then run it at high priority with four jobs at once, and wait for it to finish.

```bash
uv run apex validate my-study.toml
uv run apex study run my-study.toml --priority high --max-concurrent-jobs 4 --wait
```

Because of `--wait`, the command returns when the study has finished. If the study is large, you will need approval first: either add `--yes`, or approve a token and add `--approval-token <token>` as shown above.

If you had left out `--wait`, the command would return right after submission. You could then check on it from another terminal:

```bash
uv run apex study status my-study
```

When the status shows that the runs are done, compare the variants:

```bash
uv run apex study compare my-study
```

Here `my-study` stands for your study's name.

## Pitfalls

- **`--wait` is off by default.** Without it, `apex study run` returns as soon as the study is submitted. The runs are still going. Use `apex study status` to see where they are.
- **`--no-wait` is not a separate option.** It is the off flag for the same switch as `--wait`.
- **`apex study run` takes a file; the other two take a name.** `apex study status` and `apex study compare` need the study's name.
- **Large studies need approval.** Pass `--yes`, or approve with `apex approve` and submit again with `--approval-token`.
- **Use the same `--root` everywhere.** If you submitted with a non-default `--root`, pass the same value to `apex study status`, `apex study compare` and `apex approve`, or they will look in the default location.
- **Repeat `--metric` for each metric.** Each use takes one `metric:label` value.
- **`--code` changes what the jobs run.** By default the jobs use the last commit (`frozen`), not your uncommitted edits. Use `--code live` if you want the live code.
- **An invalid file stops at `apex validate`.** Fix the errors it reports before you submit.

## What to read next

- The page on approvals, for `apex approve` and the tokens used with `--approval-token`.
- The reference for the study specification, for what goes in the TOML file you pass to `apex study run`, and for the study's name and metrics.

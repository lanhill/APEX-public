---
audience: public
items:
  - cli:apex surrogate build
  - cli:apex surrogate check
---

# The `apex surrogate` commands

## Goal

By the end of this lesson you will be able to:

- build a surrogate with `apex surrogate build`: full runs of a design, then the fit, its domain and its gate;
- continue or limit a build with `--resume` and `--max-runs`;
- test an existing surrogate on new full runs with `apex surrogate check`, and read its pass/fail result.

## Prerequisites

- APEX installed as described in the README. `apex` is not on your PATH, so every command below is run from the APEX folder as `uv run apex ...`.
- For `build`: a surrogate build configuration, a TOML file.
- For `check`: a surrogate you have already built, with its manifest `surrogate.json`, and either a history-matching configuration or a history-matching run directory to draw test members from.

## Step 1: Build a surrogate with `apex surrogate build`

`apex surrogate build` performs the full runs of a design, then fits the surrogate and determines its domain and its gate.

```bash
uv run apex surrogate build build.toml
```

| Argument or flag | Meaning | Default |
|---|---|---|
| `config` (argument) | The surrogate build, a TOML file. | none |
| `--resume DIR` | Continue the build in the directory `DIR`. | none |
| `--max-runs N` | Run at most `N` full runs now. A resumed build continues the rest. | no limit |
| `--progress` / `--no-progress` | Show or hide progress on standard error (bars, or timestamped lines). `--progress` turns it on, `--no-progress` turns it off. | on |

Full runs are the expensive part. To do a build in portions, cap the number of runs per invocation with `--max-runs`, then continue the same build with `--resume`:

```bash
uv run apex surrogate build build.toml --max-runs 20
uv run apex surrogate build --resume path/to/build-directory
```

## Step 2: Gate an existing surrogate with `apex surrogate check`

`apex surrogate check` gates an existing surrogate on new full runs. The model file never changes: the command only tests the model, it does not refit it.

```bash
uv run apex surrogate check path/to/surrogate.json --against path/to/history-matching.toml
```

| Argument or flag | Meaning | Default |
|---|---|---|
| `model` (argument, required) | The model's manifest, `surrogate.json`. | none |
| `--against` (required) | A history-matching configuration (members are drawn from its prior) or a run directory (members of its posterior). | none |
| `--runs N` | How many new full runs. | 10 |
| `--seed N` | Seed of the members drawn or chosen. | 0 |
| `--iteration N` | Use a run's iterate instead of its posterior. | none |
| `--min-success-fraction F` | Fail unless this share of the runs succeeds. | 0.9 |
| `--min-inside F` | Fail unless this share of the successful runs is inside the model's domain. | 0.8 |
| `--max-runs N` | Run at most `N` full runs now. | no limit |
| `--progress` / `--no-progress` | Show or hide progress on standard error. | on |

Two thresholds decide the outcome:

- the share of runs that succeed must reach `--min-success-fraction`;
- the share of the successful runs that lie inside the model's domain must reach `--min-inside`.

If either share falls short, the check fails.

## Worked example

You have built a surrogate and want to know whether it still holds for the posterior of a finished history-matching run.

1. Check it on 10 new full runs (the default) against the run directory, which tests members of its posterior:

   ```bash
   uv run apex surrogate check path/to/surrogate.json --against path/to/run-directory
   ```

2. To test the model on a particular iterate instead of the posterior, add `--iteration`:

   ```bash
   uv run apex surrogate check path/to/surrogate.json --against path/to/run-directory --iteration 3
   ```

3. To test it on members drawn from the prior of a history-matching configuration, with more runs, a different seed and stricter thresholds:

   ```bash
   uv run apex surrogate check path/to/surrogate.json --against path/to/history-matching.toml --runs 20 --seed 1 --min-success-fraction 0.95 --min-inside 0.9
   ```

4. To keep a single invocation short, cap the full runs it performs and hide progress output:

   ```bash
   uv run apex surrogate check path/to/surrogate.json --against path/to/run-directory --max-runs 5 --no-progress
   ```

## Pitfalls

- `--against` is required for `check`. A configuration gives members from its prior, a run directory gives members of its posterior. Pick the one that matches the question you are asking.
- `--iteration` replaces the posterior with a run's iterate, so it applies when `--against` is a run directory.
- `check` never modifies the model file. If the check fails, build a new surrogate with `apex surrogate build`.
- `--max-runs` caps the full runs done now. A smaller number than the full design means the work is not complete after that invocation. For `build`, continue it with `--resume`.
- With only a few runs, each failure moves the shares a lot. With the default 10 runs and `--min-success-fraction` 0.9, a single failed run is the most the check allows.
- `--no-progress` only hides progress on standard error. It does not change the result.

## What to read next

- The other pages of the command-line guide, for the history-matching runs that supply the configuration or run directory used with `--against`.

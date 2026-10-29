---
audience: public
items:
  - cli:apex plugins
  - cli:apex report
  - cli:apex resume
  - cli:apex run
  - cli:apex schema
  - cli:apex stop
  - config:history_match:method
  - plugin:damping_schemes:lm
  - plugin:damping_schemes:rlmmac
  - plugin:methods:gies
  - plugin:methods:lm_enrml
  - plugin:update_strategies:adaptive_rows
  - plugin:update_strategies:local_analysis
  - plugin:update_strategies:local_rlmmac
  - plugin:update_strategies:rlmmac
---

# Damping schemes

A damping scheme is a plugin in the `damping_schemes` registry. It controls the damping value λ (lambda) that the iterative methods use to limit the size of their steps. APEX provides two schemes: `lm` and `rlmmac`. Each has its own options, listed below with types, defaults and what they do.

## Goal

By the end of this lesson you will know:

- which scheme belongs to which method,
- what λ does and how each option changes it,
- how to write a `[method.damping]` block, start a run with it, stop and resume it, and check the result.

## Prerequisites

- A run configuration in TOML that you can pass to `apex run`. To see everything such a file accepts, run `uv run apex schema history_match` (`history_match` is the default kind). The damping settings live in its `[method]` section, which the schema describes as `history_match:method`.
- You know whether you run LM-EnRML (`lm_enrml`) or GIES (`gies`, RLM-MAC). That decides the scheme.

## How damping works

In each analysis the method makes a trial step. λ limits the size of that step: a larger λ gives a more cautious update. After each trial the scheme adjusts λ:

- **A rejected trial** (the data mismatch got worse) raises λ by `factor`, so the next trial is more cautious.
- **An accepted trial** lowers λ. In `lm`, λ is divided by `factor`, but only when the spread of the data mismatch (how widely the mismatch varies) also fell. In `rlmmac`, λ is multiplied by `reduction` (default `0.9`); if you leave `reduction` unset, it is divided by `factor` instead.
- **`minimum`** is a floor: λ is not reduced once it is at or below it.

## The two schemes

| Plugin | Used by | Default scheme of | PET equivalent |
|---|---|---|---|
| `lm` | LM-EnRML | `lm_enrml` | `[dataassim.iteration]` |
| `rlmmac` | GIES (RLM-MAC) | `gies` | `iteration` |

The last column is only for readers who come from PET. If you do not use PET, ignore it.

Both schemes accept only the options below. An unknown or misspelled option is rejected.

## Options of `lm`

| Option | Type | Default | Meaning |
|---|---|---|---|
| `initial` | number, or `"auto"` | `100.0` | λ of the first analysis (positive). `"auto"` sets it to half the prior's mean data mismatch per datum, so it adapts to your data instead of using a fixed number. |
| `factor` | number | `5.0` | Divide λ by this after an accepted trial whose data mismatch spread also fell; multiply by it after a rejected one. Greater than 1. |
| `minimum` | number | `0.01` | λ is not reduced once it is at or below this. |
| `maximum` | number | `10000000000.0` | The run stops once a trial's λ reaches this. Must be above `minimum`. |
| `data_mismatch_tolerance` | number | `0.01` | The run has converged when an improving trial changes the mean data mismatch by a relative amount below this. Positive. |
| `converge_on_rejection` | boolean | `false` | If `true`, a worse trial that is within the tolerance also stops the run. If `false`, that trial is rejected and λ is raised. |
| `max_stalled_rejections` | integer | `3` | Stop after this many consecutive rejected trials within the tolerance. At least 1. |
| `confirm_convergence` | boolean | `true` | If `true`, a small improvement stops the run only when λ is at its `minimum` or the previous accepted trial was small too; otherwise the trial is accepted and λ relaxed. If `false`, the first small improvement stops the run. |

## Options of `rlmmac`

| Option | Type | Default | Meaning |
|---|---|---|---|
| `initial` | number | `1.0` | The damping of the first analysis. Must be positive. |
| `factor` | number | `2.0` | Multiply the damping by this after a rejected trial. Greater than 1. |
| `minimum` | number | not set (no floor) | The damping is not reduced once it is at or below this. |
| `reduction` | number | `0.9` | Multiply the damping by this after an accepted trial; in `(0, 1]`. Unset, the damping is divided by `factor` instead. |
| `data_mismatch_tolerance` | number | `0.01` | Convergence when an improving trial changes the mean data mismatch by a relative amount below this. Positive. |
| `converge_on_rejection` | boolean | `false` | If `true`, a worse trial within the tolerance also stops the run. If `false`, it is rejected and the damping raised. |
| `max_stalled_rejections` | integer | `3` | Stop after this many consecutive rejected trials within the tolerance. At least 1. |
| `confirm_convergence` | boolean | `false` | If `true`, a small improvement stops the run only when the damping is at its `minimum` or the previous accepted trial was small too. If `false`, the first small improvement stops it. Set `true` for a run that seems to stop too early. |

TOML has no null value, so you cannot write an option as "unset". Leave the line out: `minimum` then has no floor, and `reduction` takes its default `0.9`.

Three differences matter in practice:

- Only `lm` has `maximum`, and only `lm` accepts `"auto"` for `initial`.
- Only `rlmmac` has `reduction`, and its `minimum` is unset by default.
- `confirm_convergence` is `true` by default in `lm` and `false` in `rlmmac`. With `false`, the first small improvement ends the run; with `true`, APEX asks for more evidence first.

## The method's own stopping limit: `max_inner_iterations`

`max_inner_iterations` is not a damping option. It is an option of the method `lm_enrml`, so it goes in `[method.options]`, not in `[method.damping.options]`. It is an integer, default `10`: the run stops after that many rejected trials in a row.

The two limits work together. Every rejected trial multiplies λ by `factor`, so with the `lm` scheme λ usually reaches `maximum` before ten rejections in a row have happened, and `maximum` ends the run first. With the `rlmmac` scheme there is no `maximum`, so the tolerance, the iteration budget (`max_iterations`) or `max_inner_iterations` ends it.

## The update strategy is separate

The damping scheme is not the same as the update strategy in `[method.strategy]`. The name `rlmmac` exists in both registries (`damping_schemes` and `update_strategies`) and the two have different options. The strategy section is optional in the configuration (it defaults to unset), so the examples below leave it out and set only the damping. If you do choose a strategy:

- The `rlmmac` strategy has the options `energy` (number, default `0.95`: the fraction of the singular-value sum of `L^-1 dY` that is kept) and `scale_data_anomalies` (boolean, default `true`). It runs with `gies` and `lm_enrml`.
- The `local_rlmmac` strategy is the RLM-MAC update with local analysis. It also runs with `gies` and `lm_enrml`.
- Localization (`[method.localization]`) is a separate, optional topic. If you leave it out, there is no localization, except in the local-analysis strategies (`local_analysis`, `local_rlmmac`, `adaptive_rows`), which select data by prior correlation. You can skip it for this lesson.

## Steps

1. Decide which scheme matches your method: `lm` for LM-EnRML, `rlmmac` for GIES.
2. In your run configuration, add `[method.damping]` with `name = "lm"` or `name = "rlmmac"`, and put the scheme's options in `[method.damping.options]`.
3. If your configuration has a `[method.schedule]` table: without `[method.damping]`, that table gives the options of the method's default scheme. When you move to `[method.damping]`, move those values there and remove `[method.schedule]`, so the damping options are in one place.
4. Start with the defaults and change one option at a time. A sensible first change is `initial`. If runs stop too early, set `confirm_convergence = true` (for `rlmmac`). If a run keeps rejecting trials, a larger `initial` or `factor` makes the steps more cautious; for `lm`, `maximum` bounds how far λ can grow before the run stops.
5. Start the run:

   ```bash
   uv run apex run run.toml
   ```

   Progress is shown on standard error: bars on a terminal, timestamped lines otherwise. `--no-progress` turns it off; `--progress` is the default.

   **The run directory.** Each run lives in its own folder, whose location comes from the `[run]` section of your configuration (`name`, and optionally `output_dir`). The folder is the `<run_dir>` used by the commands below. If the folder already holds a run, `apex run` refuses to start and its error message names the folder. Then either resume that run (see the next step), or start a new one by giving the configuration another `[run]` `name` (or `output_dir`), or after moving the old folder away.
6. To work in pieces, stop after a number of analysis steps (one analysis step is one update of the ensemble). This needs a run folder that does not hold a run yet:

   ```bash
   uv run apex run run.toml --max-steps 3
   ```

   Continue from the last checkpoint with the run directory as argument (it also accepts `--max-steps` and `--progress`/`--no-progress`):

   ```bash
   uv run apex resume <run_dir>
   ```

   To ask a running run to stop at its next checkpoint, use `uv run apex stop <run_dir>`.
7. If the configuration has an unknown or misspelled option, the run is rejected. Compare your option names with the tables above (or run `uv run apex plugins`), correct them and start again.
8. When the run has finished, look at the data mismatch history and metrics:

   ```bash
   uv run apex report <run_dir>
   ```

   Add `--plots` to also write the figures (to `RUN_DIR/report` unless you give `--output`). A healthy run shows the data mismatch falling over the iterations. If it stops after very few iterations, see the pitfalls.

## Worked example: `lm`

Only `max_stalled_rejections` differs from the damping defaults here (`2` instead of `3`). Every damping option is written out so you can see the full set, and `max_inner_iterations` is set in `[method.options]` at its default.

```toml
[method]
name = "lm_enrml"
max_iterations = 10

[method.options]
max_inner_iterations = 10

[method.damping]
name = "lm"

[method.damping.options]
initial = 100.0
factor = 5.0
minimum = 0.01
maximum = 10000000000.0
data_mismatch_tolerance = 0.01
converge_on_rejection = false
max_stalled_rejections = 2
confirm_convergence = true
```

To let APEX choose the starting value, replace the `initial` line with `initial = "auto"`.

This snippet shows only the `[method]` section. It belongs in your run configuration alongside the other sections it needs (`uv run apex schema history_match` lists them).

## Worked example: `rlmmac`

This block sets `minimum`, which is unset by default, to `0.001`, and sets `confirm_convergence` to `true` so that the first small improvement does not end the run. The other options are at their defaults.

```toml
[method]
name = "gies"
max_iterations = 10

[method.damping]
name = "rlmmac"

[method.damping.options]
initial = 1.0
factor = 2.0
minimum = 0.001
reduction = 0.9
data_mismatch_tolerance = 0.01
converge_on_rejection = false
max_stalled_rejections = 3
confirm_convergence = true
```

Run this configuration, and then a copy with another `[run]` `name` and the default `confirm_convergence = false`, so the two runs do not share a folder. Compare `uv run apex report <run_dir>` for both: look at how many iterations each ran and at the final data mismatch. A longer run is worth keeping only if the final mismatch is clearly lower.

## Pitfalls

- **Mixing the schemes' options.** Do not give `maximum` to `rlmmac`, or `reduction` to `lm`. They do not exist there, and unknown options are rejected.
- **`"auto"` is only for `lm`.** For `rlmmac`, `initial` must be a number. `"auto"` is a string and needs quotes; numbers must not be quoted.
- **Different defaults.** `initial` is `100.0` for `lm` but `1.0` for `rlmmac`, and `factor` is `5.0` versus `2.0`. Do not copy values from one scheme to the other without thinking.
- **A run that stops too early.** The defaults for `confirm_convergence` differ. In `rlmmac` the first small improvement stops the run; set `confirm_convergence = true` if that seems premature.
- **Changing several stopping options at once.** `data_mismatch_tolerance`, `converge_on_rejection`, `max_stalled_rejections` and `confirm_convergence` all decide when the run ends. Change one at a time and compare with `apex report`.
- **Two plugins named `rlmmac`.** One is in `damping_schemes` (`[method.damping]`), the other in `update_strategies` (`[method.strategy]`). They have different options.
- **Leftover `[method.schedule]`.** Keep damping options in `[method.damping.options]`, not split between the two places.
- **Starting a second run in the same folder.** `apex run` refuses when the run folder already holds a run. Resume it with `uv run apex resume <run_dir>`, or give the new run another `[run]` `name`.
- **Putting `max_inner_iterations` in the damping options.** It belongs to `lm_enrml` and goes in `[method.options]`; in `[method.damping.options]` it is an unknown option and is rejected.

## What to read next

- `uv run apex plugins --json` for the full option descriptions of every plugin (without `--json` you get the list of plugins with their options).
- `uv run apex schema history_match` for everything a run configuration accepts.
- The other plugin pages in this guide, especially the update strategies and the reference for your chosen method (`lm_enrml` or `gies`).

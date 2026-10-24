---
audience: public
items:
  - cli:apex run
  - cli:apex stop
  - plugin:damping_schemes:lm
  - plugin:damping_schemes:rlmmac
  - plugin:update_strategies:rlmmac
---

# Damping schemes

A damping scheme is a plugin in the `damping_schemes` registry. APEX provides two: `lm` and `rlmmac`. Each one has its own set of options, and this page lists them with their types and defaults.

## Goal

By the end of this lesson you will know:

- which scheme belongs to which method,
- which options each scheme accepts, with types and defaults,
- how to write a complete settings block for each scheme and start a run with it.

## Prerequisites

- You have a run configuration in TOML that you can pass to `apex run`.
- You know whether you are running LM-EnRML or GIES (RLM-MAC), because that decides which scheme you use.

## The two schemes

| Plugin | Used by | PET equivalent |
|---|---|---|
| `lm` | LM-EnRML | `[dataassim.iteration]` |
| `rlmmac` | GIES (RLM-MAC) | `iteration` |

The `rlmmac` damping scheme goes with the update strategy that has the same name, `rlmmac` in the `update_strategies` registry (the regularised Levenberg-Marquardt update for the minimum-average-cost problem). That strategy has its own options, `energy` (number, default `0.95`) and `scale_data_anomalies` (boolean, default `true`), and it is supplied with `localization`.

Neither damping scheme accepts options other than those listed below. In the option schemas, additional properties are not allowed, so an unknown option is an error.

## Options of `lm`

| Option | Type | Default |
|---|---|---|
| `initial` | number, or the string `"auto"` | `100.0` |
| `factor` | number | `5.0` |
| `minimum` | number | `0.01` |
| `maximum` | number | `10000000000.0` |
| `data_mismatch_tolerance` | number | `0.01` |
| `converge_on_rejection` | boolean | `false` |
| `max_stalled_rejections` | integer | `3` |
| `confirm_convergence` | boolean | `true` |

## Options of `rlmmac`

| Option | Type | Default |
|---|---|---|
| `initial` | number | `1.0` |
| `factor` | number | `2.0` |
| `minimum` | number or unset | unset |
| `data_mismatch_tolerance` | number | `0.01` |
| `reduction` | number or unset | `0.9` |
| `converge_on_rejection` | boolean | `false` |
| `max_stalled_rejections` | integer | `3` |
| `confirm_convergence` | boolean | `false` |

Three differences matter in practice:

- Only `lm` has `maximum`, and only `lm` accepts `"auto"` for `initial`.
- Only `rlmmac` has `reduction`. Its `minimum` has no default value.
- `confirm_convergence` is on by default in `lm` and off in `rlmmac`. With it on, a small improvement made while λ is above `minimum` does not stop the run: λ is relaxed, and the run stops after two small improvements in a row. A large λ makes every step small, so one small improvement says little about how close the run is to a minimum. `lm` starts at λ = 100, where a step can be about 1 % of a full one, so it needs the confirmation; `rlmmac` starts at 1 and lowers λ after each accepted step, so its steps stay a large part of a full step (in their leading directions), and keeping it off keeps GIES's results as they were. Turn it on for an `rlmmac` run that seems to stop too early. `false` stops on the first small improvement, as PET does.

This page does not describe in detail how `factor`, `reduction`, `data_mismatch_tolerance`, `converge_on_rejection` and `max_stalled_rejections` act during a run. The safe approach is to keep their defaults and change one option at a time, comparing runs, when you have a reason to.

## Steps

1. Decide which scheme matches your method: `lm` for LM-EnRML, `rlmmac` for GIES.
2. In your run configuration, add the section `[method.damping]` with `name = "lm"` or `name = "rlmmac"`, and put the scheme's options in `[method.damping.options]`. Leave out `[method.schedule]`: it cannot be given together with `[method.damping]`.
3. Start with the defaults and set only the options you want to change.
4. Set `initial` first. For `lm` you may use the string `"auto"` instead of a number.
5. Use `minimum` (and, for `lm`, `maximum`) to bound the damping.
6. Start the run:

   ```bash
   apex run run.toml
   ```

   Use `--max-steps N` to stop after `N` analysis steps. `--progress` (the default) and `--no-progress` turn the progress display on standard error on or off. To end a run at its next checkpoint, use `apex stop <run_dir>`, where `run_dir` is the directory of the running run.

## Worked example: `lm`

Only `max_stalled_rejections` differs from the defaults here (`2` instead of `3`). Every other option is written out at its default value so you can see the full set.

```toml
[method]
name = "lm_enrml"

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
```

To use `"auto"` for the starting value, replace the `initial` line with `initial = "auto"`.

## Worked example: `rlmmac`

This block sets `minimum`, which has no default, to `0.001`. All other options keep their defaults. It also sets the options of the `rlmmac` update strategy at their defaults.

```toml
[method]
name = "gies"

[method.strategy]
name = "rlmmac"
options = { energy = 0.95, scale_data_anomalies = true }

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
```

These snippets show only the `[method]` section. The rest of your run configuration is not shown.

## Pitfalls

- **Mixing the schemes' options.** Do not give `maximum` to `rlmmac`, or `reduction` to `lm`. Neither exists there, and unknown options are rejected.
- **`"auto"` is only for `lm`.** For `rlmmac`, `initial` must be a number.
- **Different defaults.** `initial` is `100.0` for `lm` but `1.0` for `rlmmac`, and `factor` is `5.0` versus `2.0`. Do not copy values from one scheme to the other without thinking.
- **Quoting.** `"auto"` is a string and needs quotes. Numbers must not be quoted.
- **Changing the stopping behaviour.** `converge_on_rejection` is `false` by default and `max_stalled_rejections` is `3`. Change them deliberately, and compare against a run with the defaults.
- **Confusing the two `rlmmac` plugins.** The name `rlmmac` exists both in `damping_schemes` (`[method.damping]`) and in `update_strategies` (`[method.strategy]`). They have different options.

## What to read next

- The other plugin pages in this guide, for the rest of the setup around the damping scheme, including the `update_strategies` registry.
- The reference for your chosen method (LM-EnRML or GIES).

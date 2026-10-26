---
audience: public
items:
  - cli:apex plugins
  - cli:apex run
  - cli:apex schema
  - cli:apex stop
  - config:history_match:method
  - plugin:damping_schemes:lm
  - plugin:damping_schemes:rlmmac
  - plugin:methods:lm_enrml
  - plugin:update_strategies:rlmmac
---

# Damping schemes

A damping scheme is a plugin in the `damping_schemes` registry. It controls the damping value λ that the iterative methods use to limit the size of their steps. APEX provides two schemes: `lm` and `rlmmac`. Each has its own options, and this page lists them with their types and defaults.

The options `initial`, `factor`, `minimum` and `maximum` all concern λ. `initial` is its starting value, and `minimum` and `maximum` are its bounds. Not every scheme has every option (see the tables below).

## Goal

By the end of this lesson you will know:

- which scheme belongs to which method,
- which options each scheme accepts, with types and defaults,
- how to write a settings block for each scheme, start a run with it, and find the full option descriptions.

## Prerequisites

- You have a run configuration in TOML that you can pass to `apex run`. To see everything such a file accepts, run `apex schema history_match` (`history_match` is the default kind).
- You know whether you are running LM-EnRML (`lm_enrml`) or GIES (`gies`, RLM-MAC), because that decides which scheme you use.

## The two schemes

| Plugin | Used by | PET equivalent |
|---|---|---|
| `lm` | LM-EnRML | `[dataassim.iteration]` |
| `rlmmac` | GIES (RLM-MAC) | `iteration` |

The last column is only for readers who come from PET, the tool whose iteration settings these schemes follow. If you do not use PET, ignore it.

The `rlmmac` damping scheme goes with the update strategy of the same name, `rlmmac` in the `update_strategies` registry (the regularised Levenberg-Marquardt update for the minimum-average-cost problem). That strategy has its own options, `energy` (number, default `0.95`) and `scale_data_anomalies` (boolean, default `true`). It works with `gies` and `lm_enrml`.

The strategy can take an optional localization plugin, set in `[method.localization]`. You do not need to set it for this lesson: when it is left out there is no localization, except for the local-analysis strategies (`local_analysis`, `local_rlmmac`, `adaptive_rows`), which select data by prior correlation.

Neither damping scheme accepts options other than those listed below. An unknown option is an error.

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
| `minimum` | number | none (no lower bound set) |
| `data_mismatch_tolerance` | number | `0.01` |
| `reduction` | number | `0.9` |
| `converge_on_rejection` | boolean | `false` |
| `max_stalled_rejections` | integer | `3` |
| `confirm_convergence` | boolean | `false` |

TOML has no null value, so you cannot write `reduction` as "unset". Leave the line out and the default `0.9` applies. Likewise, leave out `minimum` and no minimum is set.

Three differences matter in practice:

- Only `lm` has `maximum`, and only `lm` accepts `"auto"` for `initial`.
- Only `rlmmac` has `reduction`. Its `minimum` has no default value.
- `confirm_convergence` is on by default in `lm` and off in `rlmmac`. If you want the other behaviour, set it explicitly. When you change it, compare the final results of a run with and without the change.

## Finding out what each option does

This page lists names, types and defaults. For the full description of every option of every plugin, run:

```bash
uv run apex plugins --json
```

`--json` prints the full descriptions as JSON (without it you get the list of plugins with their options). The safe approach is to keep the defaults and change one option at a time, comparing runs, when you have a reason to.

## Steps

1. Decide which scheme matches your method: `lm` for LM-EnRML, `rlmmac` for GIES.
2. In your run configuration, add the section `[method.damping]` with `name = "lm"` or `name = "rlmmac"`, and put the scheme's options in `[method.damping.options]`.
3. If your configuration already has a `[method.schedule]` table: without `[method.damping]`, that table gives the options of the method's default scheme (`rlmmac` for GIES, `lm` for LM-EnRML). When you move to `[method.damping]`, move those values there and remove `[method.schedule]`, so the damping options are set in one place.
4. Start with the defaults and set only the options you want to change. Begin with `initial`, the starting λ (for `lm` you may use `"auto"`). If you want to bound λ, set `minimum` (and, for `lm`, `maximum`) at the same time. Bounds are worth adding only if you have seen λ go somewhere you do not want; otherwise the defaults are fine.
5. Start the run:

   ```bash
   uv run apex run run.toml
   ```

   Progress is shown on standard error: bars on a terminal, timestamped lines otherwise. `--no-progress` turns it off (`--progress` is the default). Use `--max-steps N` to stop after `N` analysis steps; continue later with `apex resume`. To end a run at its next checkpoint, use `apex stop <run_dir>`, where `run_dir` is the directory of the running run.
6. Check that the run starts without a configuration error. An unknown or misspelled option is reported as an error before the run proceeds.

## Worked example: `lm`

Only `max_stalled_rejections` differs from the defaults here (`2` instead of `3`). Every option is written out, so you can see the full set.

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
confirm_convergence = true
```

To use `"auto"` for the starting value, replace the `initial` line with `initial = "auto"`.

## Worked example: `rlmmac`

This block sets `minimum`, which has no default, to `0.001`. All other options are written at their defaults. It also sets the options of the `rlmmac` update strategy at their defaults.

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
confirm_convergence = false
```

These snippets show only the `[method]` section; they belong in your run configuration alongside the other sections it needs (run `apex schema history_match` to see them). The `rlmmac` strategy block is not required by the damping scheme in general, but with `gies` it is the matching strategy, as shown.

## Pitfalls

- **Mixing the schemes' options.** Do not give `maximum` to `rlmmac`, or `reduction` to `lm`. Neither exists there, and unknown options are rejected.
- **`"auto"` is only for `lm`.** For `rlmmac`, `initial` must be a number.
- **Different defaults.** `initial` is `100.0` for `lm` but `1.0` for `rlmmac`, and `factor` is `5.0` versus `2.0`. Do not copy values from one scheme to the other without thinking.
- **Quoting.** `"auto"` is a string and needs quotes. Numbers must not be quoted.
- **Changing the stopping behaviour.** `converge_on_rejection`, `max_stalled_rejections` and `confirm_convergence` differ between the schemes. Change them deliberately, and compare against a run with the defaults.
- **Confusing the two `rlmmac` plugins.** The name `rlmmac` exists both in `damping_schemes` (`[method.damping]`) and in `update_strategies` (`[method.strategy]`). They have different options.
- **Leftover `[method.schedule]`.** Options left there are not a substitute for `[method.damping.options]`; keep your damping options in one place.

## What to read next

- The other plugin pages in this guide, for the rest of the setup around the damping scheme, including the `update_strategies` registry.
- The reference for your chosen method (LM-EnRML or GIES).
- `apex plugins --json` for full option descriptions, and `apex schema history_match` for everything a run configuration accepts.

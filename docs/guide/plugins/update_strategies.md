---
audience: public
items:
  - cli:apex plugins
  - cli:apex run
  - cli:apex validate
  - plugin:damping_schemes:rlmmac
  - plugin:update_strategies:adaptive_rows
  - plugin:update_strategies:approx
  - plugin:update_strategies:ert_adaptive
  - plugin:update_strategies:ert_esmda
  - plugin:update_strategies:ert_localized
  - plugin:update_strategies:full
  - plugin:update_strategies:gmac
  - plugin:update_strategies:hybrid
  - plugin:update_strategies:local_analysis
  - plugin:update_strategies:local_rlmmac
  - plugin:update_strategies:margis
  - plugin:update_strategies:rlmmac
  - plugin:update_strategies:subspace
  - plugin:update_strategies:subspace2
---

# Update strategies

## Goal

This page lists the `update_strategies` plugins and shows how to configure one. By the end you will be able to:

- list the available plugins with `apex plugins`,
- find the options of a plugin and their defaults,
- write a valid TOML section for it,
- check the configuration with `apex validate` before you start a run with `apex run`.

The plugins covered are `adaptive_rows`, `approx`, `ert_adaptive`, `ert_esmda`, `ert_localized`, `full`, `gmac`, `hybrid`, `local_analysis`, `local_rlmmac`, `margis`, `rlmmac`, `subspace` and `subspace2`. The `rlmmac` damping scheme is covered at the end, because it shares its name with an update strategy.

## Prerequisites

- An APEX configuration file in TOML.
- A rough idea of which kind of update your workflow needs. The table below gives a one-line description of each plugin.

## See what is installed

List every plugin with its status and options:

```bash
apex plugins
```

For full descriptions as JSON, add `--json`:

```bash
apex plugins --json
```

## The strategies at a glance

| Plugin | What it is |
|---|---|
| `adaptive_rows` | Adaptive localization per state row, recomputed at every update. |
| `approx` | Approximate (truncated SVD, TSVD) Levenberg-Marquardt (LM) update. It is the ES/ES-MDA analysis at `damping=0`. |
| `ert_adaptive` | ERT's adaptive localization (`AdaptiveESMDA`). Status: validated. |
| `ert_esmda` | ERT's ES-MDA analysis (`iterative_ensemble_smoother.ESMDA`). Status: validated. |
| `ert_localized` | ERT's localized ES-MDA (`LocalizedESMDA`): the Kalman gain times a given taper. Status: validated. |
| `full` | Full LM update. It needs each member's prior state. |
| `gmac` | GIES's generalised update. |
| `hybrid` | The multilevel hybrid update (PET `analysis = "hybrid"`). |
| `local_analysis` | Local analysis: grouped local `approx` updates. |
| `local_rlmmac` | Local analysis with GIES's RLM-MAC update in each group. |
| `margis` | Marginalised IES of Stordal et al. (2023), PET's `margis`. |
| `rlmmac` | Regularised Levenberg-Marquardt update for the minimum-average-cost problem (RLM-MAC). |
| `subspace` | Ensemble subspace (weight-space) update. |
| `subspace2` | Ensemble-transform IES of Raanes et al. (2019), PET's `subspace2`. |

## Which methods take which strategy

A method refuses a strategy it cannot run when the configuration is checked (`apex validate`, or the start of `apex run`), before anything is simulated, and the refusal says why. This table is what APEX's composition accepts (`chop` with its default inner method, `gies`); the reference page of each strategy lists the refusals in APEX's words.

| Strategy | `es` | `esmda` | `gies` | `lm_enrml` | `gn_enrml` | `multilevel_esmda` | `chop` |
|---|---|---|---|---|---|---|---|
| `adaptive_rows` | yes | yes | no | yes | yes | no | no |
| `approx` | yes | yes | yes | yes | yes | no | no |
| `ert_adaptive` | yes | yes | no | no | yes | no | no |
| `ert_esmda` | yes | yes | no | no | yes | no | no |
| `ert_localized` | yes | yes | no | no | yes | no | no |
| `full` | no | no | yes | yes | yes | no | no |
| `gmac` | no | no | yes | yes | no | no | no |
| `hybrid` | no | no | no | no | no | yes | no |
| `local_analysis` | yes | yes | no | yes | yes | no | no |
| `local_rlmmac` | no | no | yes | yes | no | no | no |
| `margis` | no | no | yes | yes | yes | no | no |
| `rlmmac` | no | no | yes | yes | no | no | yes |
| `subspace` | no | no | yes | yes | yes | no | no |
| `subspace2` | no | no | yes | yes | yes | no | no |

The main reasons for a "no":

- `full`, `subspace`, `subspace2` and `margis` include the prior-increment term of an iterative method (each member pulled back towards its prior). ES-MDA's steps have no such term: each step updates the current ensemble. Use `approx` with `es` and `esmda`.
- `rlmmac`, `local_rlmmac` and `gmac` need the forecast of the mean model, which only the methods built on GIES's loop make. `gn_enrml` runs at damping 0, which leaves them unregularised.
- `gies` runs local analysis with RLM-MAC: use `local_rlmmac` instead of `local_analysis` or `adaptive_rows`.
- The ERT strategies do not damp their updates, so `gies` and `lm_enrml` refuse them.
- `multilevel_esmda` runs `hybrid` only, and `hybrid` needs its fidelity levels.

Abbreviations: LM is Levenberg-Marquardt, IES is iterative ensemble smoother, ES-MDA is ensemble smoother with multiple data assimilation, and RLM-MAC is the regularised LM minimum-average-cost update.

## Steps

1. Pick a plugin from the table.
2. Add the section `[method.strategy]` to your configuration, with `name = "<name>"`, where `<name>` is the plugin name, and its options in `options = { ... }` (or a `[method.strategy.options]` section). Check the table above that your method takes it.
3. Set only the options you want to change. An option you leave out keeps its default.
4. Do not add options that are not listed below. Every section rejects unknown options.
5. Run `apex plugins` to check the plugin and its options.
6. Run `apex validate` on your configuration. It validates the configuration and plans its cost; nothing is simulated.

```bash
apex validate config.toml
```

   Add `--json` to get the validation as JSON, or `--root` to set the services root (default: `$APEX_HOME`, else `~/.apex/service`).
7. Start the run with `apex run`:

```bash
apex run config.toml
```

   `--max-steps N` stops after N analysis steps; you can continue later with `apex resume`. Progress is shown on standard error by default; `--no-progress` turns it off and `--progress` turns it on.

### About `energy`, `truncation` and `max_batch_mb`

- `energy` appears in many plugins, with a default of `0.98` or `0.95`. It belongs to the truncated (SVD-based) updates. This page does not state its valid range or the exact effect of changing it. Use `apex plugins --json` to read the full description of the option in your installation, and change it from the default only deliberately.
- `truncation` is the corresponding setting of the ERT plugins. Its default is `0.99`.
- `max_batch_mb` is a number with default `256.0`, in the plugins that have it.
- `apex plugins --json` gives the full description of each option.

### About `localization`

`approx`, `gmac`, `local_analysis`, `local_rlmmac` and `rlmmac` take a localization. You do not set it among the strategy's options: choose it in its own section, `[method.localization]`, and APEX gives it to the strategy. `gmac` is also given `property_map` and `state_variables` by APEX, so you do not set them either. `ert_localized` instead reads a taper from a file that you name with `taper_file`. The taper is the weighting applied to the Kalman gain. This page does not describe the file format, so take it from whoever prepares your localization data.

### Options

**`adaptive_rows`**

- `energy`: number, default `0.98`.
- `threshold`: number, or unset. Default: unset.
- `max_batch_mb`: number, default `256.0`.

**`approx`**

- `energy`: number, default `0.98`.
- `empirical_data_error`: true or false, default `false`.
- `sensitivities`: true or false, default `false`.

**`ert_adaptive`**

- `truncation`: number, default `0.99`.
- `threshold`: number, or unset. Default: unset.
- `n_jobs`: integer, default `1`.
- `max_batch_mb`: number, default `256.0`.

**`ert_esmda`**

- `truncation`: number, default `0.99`.
- `max_batch_mb`: number, default `256.0`.

**`ert_localized`**

- `taper_file`: a path. **Required**, no default.
- `truncation`: number, default `0.99`.
- `max_batch_mb`: number, default `256.0`.

**`full`**

- `energy`: number, default `0.98`.
- `block_sizes`: list of integers, or unset. Default: unset.
- `state_scaling`: an array of values, or unset. Default: unset.
- `state_scaling_file`: a path, or unset. Default: unset.

`full` needs each member's prior state, that is, the state of each ensemble member before the update. Make sure your workflow provides it. `state_scaling` and `state_scaling_file` are two ways to give the state scaling, inline or from a file. Set only one of them, because this page does not define which wins if both are set.

**`gmac`**

- `terms`: list of tables, or unset. Default: unset.
- `code`: string, or unset. Default: unset.
- `variables`: list of strings, or unset. Default: unset.
- `energy`: number, default `0.95`.
- `form`: `"subspace"` or `"direct"`, default `"subspace"`.
- `metric_floor`: number, default `1e-10`.
- `metric_energy`: number, or unset. Default: unset.
- `constraints`: list of tables, or unset. Default: unset.
- `localize_constraints`: true or false, default `true`.

The tables inside `terms` and `constraints` can hold any keys, and this page does not define their contents. Start from the `gmac` section of an existing configuration, and check it with `apex validate`.

**`hybrid`**

- `energy`: number, default `0.98`.

**`local_analysis`**

- `energy`: number, default `0.98`.
- `block_sizes`: list of integers, or unset. Default: unset.
- `weight`: true or false, default `false`.

**`local_rlmmac`**

- `energy`: number, default `0.95`.
- `block_sizes`: list of integers, or unset. Default: unset.
- `weight`: true or false, default `false`.

**`margis`**

- `nu`: number, or unset. Default: unset.
- `s`: number, default `1.0`.

**`rlmmac`**

- `energy`: number, default `0.95`.
- `scale_data_anomalies`: true or false, default `true`.

**`subspace`**

- `energy`: number, default `0.98`.

**`subspace2`**

- No options. Leave the section empty.

### The `rlmmac` damping scheme

The name `rlmmac` also exists as a damping scheme, chosen in its own section, `[method.damping]` with `name = "rlmmac"`. It is separate from the `rlmmac` update strategy. Its options:

- `initial`: number, default `1.0`.
- `factor`: number, default `2.0`.
- `minimum`: number, or unset. Default: unset.
- `data_mismatch_tolerance`: number, default `0.01`.
- `reduction`: number, or unset. Default: `0.9`.
- `converge_on_rejection`: true or false, default `false`.
- `max_stalled_rejections`: integer, default `3`.

## Worked example

First, `full` with a lower `energy`, for `lm_enrml`:

```toml
[method]
name = "lm_enrml"

[method.strategy]
name = "full"
options = { energy = 0.95 }
```

Options you did not write keep their defaults. Here `state_scaling` and `state_scaling_file` stay unset, and `block_sizes` comes from `[[state.variables]]`.

Next, `rlmmac` with the new option turned off, together with its damping scheme:

```toml
[method]
name = "gies"

[method.strategy]
name = "rlmmac"
options = { energy = 0.95, scale_data_anomalies = false }

[method.damping]
name = "rlmmac"
options = { initial = 1.0, factor = 2.0, max_stalled_rejections = 3 }
```

A strategy without options needs only its name:

```toml
[method]
name = "gies"

[method.strategy]
name = "subspace2"
```

`ert_localized` has a required option, `taper_file`. The path below is a placeholder; replace it with your own taper file:

```toml
[method]
name = "esmda"
options = { inflation = [4.0, 4.0, 4.0, 4.0] }

[method.strategy]
name = "ert_localized"
options = { taper_file = "path/to/your/taper_file", truncation = 0.99 }
```

Finally, run `apex plugins` to see the plugin and its options, and `apex validate` on your configuration file to check it without simulating anything:

```bash
apex plugins
apex validate config.toml
```

## Pitfalls

- **Unknown options are rejected.** `hybrid` and `subspace` accept only `energy`, `rlmmac` accepts only `energy` and `scale_data_anomalies`, and `subspace2` accepts no options at all.
- **`ert_localized` needs `taper_file`.** It has no default, so the section is invalid without it.
- **Different defaults.** `energy` defaults to `0.95` in `gmac`, `local_rlmmac` and `rlmmac`. The other plugins with `energy` default to `0.98`. The ERT plugins use `truncation` with `0.99`. If you switch plugins, set the value explicitly if you want to keep it.
- **`full` needs each member's prior state.** The iterative methods (`gies`, `lm_enrml`, `gn_enrml`) give it; `es` and `esmda` refuse `full` (see the table).
- **Do not put `localization` among the options.** Choose it in `[method.localization]`; APEX gives it to the strategy.
- **Two things called `rlmmac`.** `[method.strategy] name = "rlmmac"` and `[method.damping] name = "rlmmac"` have different options.
- **Types matter.** `block_sizes` is a list of integers such as `[10, 10, 5]`, not a single number. `weight`, `sensitivities` and `scale_data_anomalies` are TOML booleans, not strings. `form` in `gmac` must be exactly `"subspace"` or `"direct"`.
- **Do not set both `state_scaling` and `state_scaling_file`.** This page does not define which one wins.

## What to read next

- The other plugin pages in the plugins section of this guide.
- The configuration reference, for the full layout of a TOML file.

---
audience: public
items:
  - cli:apex plugins
  - cli:apex resume
  - cli:apex run
  - cli:apex stop
  - cli:apex validate
  - config:history_match:method
  - plugin:damping_schemes:rlmmac
  - plugin:methods:es
  - plugin:methods:gn_enrml
  - plugin:methods:lm_enrml
  - plugin:methods:multilevel_esmda
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

## Terms used on this page

- **Update strategy**: the analysis that turns the forecast ensemble into an updated ensemble. You choose it in `[method.strategy]`.
- **Method**: the ensemble algorithm in `[method]`, for example `es`, `esmda`, `gies`, `lm_enrml`, `gn_enrml`, `multilevel_esmda` or `chop`. Each method accepts only some strategies.
- **`gies`**: the APEX method that runs the RLM-MAC update in its loop.
- **`chop`**: a method that tapers, per member, the gain of an `rlmmac` strategy. It works only with the `rlmmac` strategy.
- **ERT**: the tool whose analysis the `ert_*` plugins reproduce.
- **PET**: a reference tool. APEX names `margis` and `subspace2` after the PET rules they follow.
- **LM**: Levenberg-Marquardt. **IES**: iterative ensemble smoother. **ES-MDA**: ensemble smoother with multiple data assimilation. **RLM-MAC**: the regularised LM update for the minimum-average-cost problem. **TSVD**: truncated singular value decomposition.

## See what is installed

List every plugin with its options:

```bash
uv run apex plugins
```

For full descriptions as JSON, add `--json`:

```bash
uv run apex plugins --json
```

## The strategies at a glance

| Plugin | What it is |
|---|---|
| `adaptive_rows` | Adaptive localization per state row, recomputed at every update. |
| `approx` | Approximate (TSVD) Levenberg-Marquardt update. It is the ES/ES-MDA analysis at `damping=0`. |
| `ert_adaptive` | ERT's adaptive localization (`AdaptiveESMDA`). Status: validated. |
| `ert_esmda` | ERT's ES-MDA analysis (`iterative_ensemble_smoother.ESMDA`). Status: validated. |
| `ert_localized` | ERT's localized ES-MDA (`LocalizedESMDA`): the Kalman gain times a given taper. Status: validated. |
| `full` | Full LM update with the prior-increment term. It needs each member's prior state. |
| `gmac` | The update of `gies` with per-member step metrics built from composable terms. |
| `hybrid` | The multilevel hybrid update: one gain from every fidelity level's anomalies. |
| `local_analysis` | Local analysis: groups of state updated with their locally selected data. |
| `local_rlmmac` | Local analysis with the RLM-MAC update of `gies` in each group. |
| `margis` | Marginalised IES of Stordal et al. (2023), PET's `margis`. |
| `rlmmac` | Regularised Levenberg-Marquardt update for the minimum-average-cost problem (RLM-MAC). |
| `subspace` | Ensemble subspace update: an iterative update of each member's weights. |
| `subspace2` | Ensemble-transform IES of Raanes et al. (2019), PET's `subspace2`. |

## Which methods take which strategy

A method refuses a strategy it cannot run when the configuration is checked (`apex validate`, or the start of `apex run`), before anything is simulated, and the refusal says why. In the table, "yes" means the combination is accepted.

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
- `rlmmac`, `local_rlmmac` and `gmac` need the forecast at the mean state, which only the methods built on the loop of `gies` make. `gn_enrml` runs at damping 0, which leaves them unregularised.
- `gies` runs local analysis with RLM-MAC: use `local_rlmmac` instead of `local_analysis` or `adaptive_rows`.
- The ERT strategies do not support damped updates, so `gies` and `lm_enrml` refuse them.
- `chop` needs an `rlmmac` strategy.
- `multilevel_esmda` runs `hybrid` only (or leave out `[method.strategy]`), and `hybrid` needs fidelity levels, so it needs `multilevel_esmda`.

## Steps

1. Pick a plugin from the tables.
2. Add the section `[method.strategy]` to your configuration, with `name = "<name>"`, where `<name>` is the plugin name, and its options in `options = { ... }` (or a `[method.strategy.options]` section). Check in the table that your method takes it.
3. Set only the options you want to change. An option you leave out keeps its default.
4. Do not add options that are not listed below. Every section rejects unknown options.
5. Run `apex plugins` to check the plugin and its options.
6. Run `apex validate` on your configuration. It validates the configuration and plans its cost; nothing is simulated.

```bash
uv run apex validate config.toml
```

   Add `--json` to get the validation as JSON, or `--root` to set the services root (default: `$APEX_HOME`, else `~/.apex/service`).
7. Start the run with `apex run`:

```bash
uv run apex run config.toml
```

   `--max-steps N` stops after N analysis steps; you can continue later with `apex resume`. Progress is shown on standard error by default; `--no-progress` turns it off and `--progress` turns it on.

### About `energy`, `truncation` and `max_batch_mb`

- `energy` appears in many plugins, with a default of `0.98` or `0.95`. It belongs to the truncated (SVD-based) updates. This page does not state its valid range or the exact effect of changing it. Use `apex plugins --json` to read the full description in your installation, and change it from the default only deliberately.
- `truncation` is the corresponding setting of the ERT plugins. Its default is `0.99`.
- `max_batch_mb` is a number with default `256.0`, in the plugins that have it.

### About `localization`

`approx`, `gmac`, `local_analysis`, `local_rlmmac` and `rlmmac` take a localization. You do not set it among the strategy's options: choose it in its own section, `[method.localization]`, and APEX gives it to the strategy. Examples of localization plugins are `adaptive`, `adaptive_per_update` and `distance`. Their `block_sizes` come from the state layout. If you leave `[method.localization]` out, no localization is used, except by `local_analysis`, `local_rlmmac` and `adaptive_rows`, which select data by prior correlation.

`gmac` is also given `property_map` and `state_variables` by APEX, so you do not set them either. `ert_localized` instead reads a taper from a file that you name with `taper_file`. The taper is the weighting applied to the Kalman gain. This page does not describe the file format, so take it from whoever prepares your localization data.

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

"Unset" means you leave the option out of the TOML. `full` needs each member's prior state, that is, the state of each ensemble member before the update. Make sure your workflow provides it. `state_scaling` and `state_scaling_file` are two ways to give the state scaling, inline or from a file. Set only one of them, because this page does not define which wins if both are set. This page does not describe what an unset `block_sizes` does; if you need blocks, set it as a list of integers such as `[10, 10, 5]` and check the result with `apex validate`.

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

- No options. Leave the section without options.

### The `rlmmac` damping scheme

The name `rlmmac` also exists as a damping scheme, chosen in its own section, `[method.damping]` with `name = "rlmmac"`. It is the default damping scheme of `gies` and is separate from the `rlmmac` update strategy. Its options:

- `initial`: number, default `1.0`.
- `factor`: number, default `2.0`.
- `minimum`: number, or unset. Default: unset.
- `data_mismatch_tolerance`: number, default `0.01`.
- `reduction`: number, default `0.9`. If you leave it out, `0.9` is used. The option can also be unset internally, but TOML has no null value, and this page does not describe a way to switch the reduction off.
- `converge_on_rejection`: true or false, default `false`.
- `max_stalled_rejections`: integer, default `3`.
- `confirm_convergence`: true or false, default `false`.

## Worked example

First, `full` with a lower `energy`, for `lm_enrml`:

```toml
[method]
name = "lm_enrml"

[method.strategy]
name = "full"
options = { energy = 0.95 }
```

Options you did not write keep their defaults. Here `block_sizes`, `state_scaling` and `state_scaling_file` stay unset.

Next, `rlmmac` with `scale_data_anomalies` turned off, together with its damping scheme:

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
name = "es"

[method.strategy]
name = "ert_localized"
options = { taper_file = "path/to/your/taper_file", truncation = 0.99 }
```

Finally, run `apex plugins` to see the plugin and its options, and `apex validate` on your configuration file to check it without simulating anything:

```bash
uv run apex plugins
uv run apex validate config.toml
```

## Pitfalls

- **Unknown options are rejected.** `hybrid` and `subspace` accept only `energy`, `rlmmac` accepts only `energy` and `scale_data_anomalies`, and `subspace2` accepts no options at all.
- **`ert_localized` needs `taper_file`.** It has no default, so the section is invalid without it.
- **Different defaults.** `energy` defaults to `0.95` in `gmac`, `local_rlmmac` and `rlmmac`. The other plugins with `energy` default to `0.98`. The ERT plugins use `truncation` with `0.99`. If you switch plugins, set the value explicitly if you want to keep it.
- **`full` needs each member's prior state.** The iterative methods (`gies`, `lm_enrml`, `gn_enrml`) accept it; `es` and `esmda` refuse `full` (see the table).
- **Do not put `localization` among the options.** Choose it in `[method.localization]`; APEX gives it to the strategy.
- **Two things called `rlmmac`.** `[method.strategy] name = "rlmmac"` and `[method.damping] name = "rlmmac"` have different options.
- **Types matter.** `block_sizes` is a list of integers such as `[10, 10, 5]`, not a single number. `weight`, `sensitivities` and `scale_data_anomalies` are TOML booleans, not strings. `form` in `gmac` must be exactly `"subspace"` or `"direct"`.
- **Do not set both `state_scaling` and `state_scaling_file`.** This page does not define which one wins.

## What to read next

- The other plugin pages in the plugins section of this guide.
- The configuration reference, for the full layout of a TOML file.

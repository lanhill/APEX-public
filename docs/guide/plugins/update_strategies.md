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
  - plugin:localizations:adaptive
  - plugin:localizations:adaptive_per_update
  - plugin:localizations:distance
  - plugin:methods:chop
  - plugin:methods:es
  - plugin:methods:esmda
  - plugin:methods:gies
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

This page shows how to choose and configure an update strategy, the `[method.strategy]` part of the `[method]` section of a run configuration (`history_match:method`). By the end you will be able to:

- list the available plugins with `apex plugins`,
- pick a strategy that your method accepts,
- write a valid TOML section for it,
- check the configuration with `apex validate` before you start a run with `apex run`.

The strategies covered are `adaptive_rows`, `approx`, `ert_adaptive`, `ert_esmda`, `ert_localized`, `full`, `gmac`, `hybrid`, `local_analysis`, `local_rlmmac`, `margis`, `rlmmac`, `subspace` and `subspace2`. The `rlmmac` damping scheme is covered at the end, because it shares its name with a strategy.

## Prerequisites

- An APEX configuration file in TOML, with a `[method]` section.
- A rough idea of which method you use: `es`, `esmda`, `gies`, `lm_enrml`, `gn_enrml`, `multilevel_esmda` or `chop`.

## Terms used on this page

- **Update strategy**: the analysis that turns the forecast ensemble into an updated ensemble. You choose it in `[method.strategy]`.
- **Method**: the ensemble algorithm in `[method]`. Each method accepts only some strategies.
- **`gies`**: the iterative smoother whose loop runs the RLM-MAC update (regularised Levenberg-Marquardt for the minimum-average-cost problem) with damped strategy updates.
- **`chop`**: a method that wraps another method (its option `inner`, default `"gies"`) and estimates localization and gain hyper-parameters per member. It needs the `rlmmac` strategy.
- **ERT**: the tool whose analysis the `ert_*` strategies reproduce. **PET**: a reference tool; `margis` and `subspace2` follow its rules of the same names.
- **Singular-value energy**: many strategies truncate a singular value decomposition. They keep the leading directions that together hold a given fraction of the sum of the singular values (see `energy` below).
- **Block**: a state variable. The state is a vector made of several variables laid out one after the other, and `block_sizes` lists the length of each, in state order.
- **Fidelity level**: one of the groups of ensemble members, each run with its own simulator model (for example a coarser grid), used by `multilevel_esmda`.

## See what is installed

List every plugin with its options:

```bash
uv run apex plugins
```

For the full descriptions as JSON, add `--json`:

```bash
uv run apex plugins --json
```

## The strategies at a glance

| Plugin | What it is |
|---|---|
| `adaptive_rows` | Adaptive localization per state row, recomputed at every update. |
| `approx` | Approximate (TSVD) Levenberg-Marquardt update; the ES/ES-MDA analysis at `damping=0`. |
| `ert_adaptive` | ERT's adaptive localization (`AdaptiveESMDA`). Status: validated. |
| `ert_esmda` | ERT's ES-MDA analysis (`iterative_ensemble_smoother.ESMDA`). Status: validated. |
| `ert_localized` | ERT's localized ES-MDA (`LocalizedESMDA`): the Kalman gain times a given taper. Status: validated. |
| `full` | Full LM update with the prior-increment term; needs each member's prior state. |
| `gmac` | GMAC: the update of `gies` with per-member step metrics from composable terms. |
| `hybrid` | The multilevel hybrid update: one gain from every fidelity level's anomalies. |
| `local_analysis` | Local analysis: groups of state updated with their locally selected data. |
| `local_rlmmac` | Local analysis with the RLM-MAC update of `gies` in each group. |
| `margis` | Marginalised IES of Stordal et al. (2023), PET's `margis`. |
| `rlmmac` | Regularised Levenberg-Marquardt update for the minimum-average-cost problem (RLM-MAC). |
| `subspace` | Ensemble subspace update: an iterative update of each member's weights. |
| `subspace2` | Ensemble-transform IES of Raanes et al. (2019), PET's `subspace2`. |

## Which methods take which strategy

A method refuses a strategy it cannot run when the configuration is checked (`apex validate`, or the start of `apex run`), before anything is simulated, and the message says why. "yes" means the combination is accepted.

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

The reasons for a "no":

- `full`, `subspace`, `subspace2` and `margis` include the prior-increment term of an iterative method (each member is pulled back towards its prior). ES-MDA's steps have no such term: each step updates the current ensemble. With `es` and `esmda`, use `approx`.
- `rlmmac`, `local_rlmmac` and `gmac` need the forecast at the mean state, which only the methods built on the loop of `gies` make. `gn_enrml` runs at damping 0, which leaves them unregularised.
- **Localized updates with `gies`:** choose `local_rlmmac`. `gies` refuses `local_analysis` and `adaptive_rows`. (`adaptive_rows` has no RLM-MAC variant; with `lm_enrml` it is accepted.)
- The ERT strategies do not support damped updates, so `gies` and `lm_enrml` refuse them. Use them with `es`, `esmda` or `gn_enrml`.
- `chop` needs the `rlmmac` strategy, because it tapers that strategy's gain per member.
- `multilevel_esmda` runs only the `hybrid` update. Leave out `[method.strategy]`, or name `hybrid`. In turn, `hybrid` needs fidelity levels, which only `multilevel_esmda` defines (see the example below).

## Steps

1. Pick a strategy from the tables that your method accepts.
2. Add `[method.strategy]` with `name = "<plugin>"`, and put its options in `options = { ... }`.
3. Set only the options you want to change; the rest keep their defaults. Unknown options are rejected.
4. Run `apex plugins` to read the options in your installation.
5. Check the configuration. Nothing is simulated:

```bash
uv run apex validate config.toml
```

   Add `--json` to get the validation as JSON, or `--root` to set the services root (default: `$APEX_HOME`, else `~/.apex/service`).
6. Start the run:

```bash
uv run apex run config.toml
```

   `--max-steps N` stops after N analysis steps. Progress goes to standard error by default; `--no-progress` turns it off and `--progress` turns it on.
7. To stop a running run at its next checkpoint, and to continue it later:

```bash
uv run apex stop path/to/run_dir
uv run apex resume path/to/run_dir
```

   `apex resume` also takes `--max-steps` and `--progress`/`--no-progress`.

### The shared options `energy`, `truncation` and `max_batch_mb`

- `energy` (a number in `(0, 1]`) is the fraction of the singular-value sum that is kept. A lower value drops more of the weak directions, so the update is smoother and cheaper but uses less of the data. A value of `1` keeps everything. The defaults are `0.98` for `adaptive_rows`, `approx`, `full`, `hybrid`, `local_analysis` and `subspace`, and `0.95` for `gmac`, `local_rlmmac` and `rlmmac` (GIES's value). When you switch strategy, set `energy` explicitly if you want to keep a value.
- `truncation` (a number in `(0, 1]`, default `0.99`) is the same idea for the ERT strategies: the fraction of the singular values kept in the inversion.
- `max_batch_mb` (default `256.0`) is the memory budget of one batch of state rows, in the strategies that have it. Lower it if memory is short.

### Localization

`approx`, `gmac`, `local_analysis`, `local_rlmmac` and `rlmmac` take a localization. Do not put it in the strategy's `options`: choose it in its own section, `[method.localization]`, and APEX passes it to the strategy. Localization plugins include `adaptive` (taper built once from the prior), `adaptive_per_update` (recomputed at every update) and `distance` (around well locations). Their `block_sizes` come from the state layout. With no `[method.localization]`, nothing is localized, except that `local_analysis`, `local_rlmmac` and `adaptive_rows` select data by prior correlation.

`gmac` also receives `property_map` and `state_variables` from APEX; you do not set them.

### Options

**`adaptive_rows`**

- `energy`: number, default `0.98`.
- `threshold`: correlation threshold in `[0, 1)`. Unset by default, which means `3 / sqrt(N)` for `N` members. Below 10 members that default is 1 or more and the run is refused, so set `threshold` yourself. A higher threshold keeps fewer correlations.
- `max_batch_mb`: number, default `256.0`.

**`approx`**

- `energy`: number, default `0.98`.
- `empirical_data_error`: true or false, default `false`. If true, the inversion uses the perturbations' own covariance for the data error instead of the nominal one (PET's `emp_cov`).
- `sensitivities`: true or false, default `false`. If true, the data anomalies come from the simulator's member-mean sensitivity times the state anomalies. Not with a localization.

**`ert_adaptive`**

- `truncation`: number, default `0.99`.
- `threshold`: correlation threshold. Unset by default: ERT's `3 / sqrt(N)`, which keeps nothing below ten members (the run is refused there; set `threshold` for small ensembles).
- `n_jobs`: integer, default `1`. Worker processes for the per-row-pattern inversions; 1 keeps the analysis within the run's own core.
- `max_batch_mb`: number, default `256.0`.

**`ert_esmda`**

- `truncation`: number, default `0.99`.
- `max_batch_mb`: number, default `256.0`.

**`ert_localized`**

- `taper_file`: a path. **Required**, no default. A `.npy` file, or an `.npz` file holding an array named `taper`. The array has shape `(n_state, n_data)` with values in `[0, 1]`. Rows follow the state layout order and columns follow the order of the analysis data. APEX can only check the shape, so build the file for the configuration you use it with.
- `truncation`: number, default `0.99`.
- `max_batch_mb`: number, default `256.0`.

**`full`**

- `energy`: number, default `0.98`.
- `block_sizes`: list of integers, unset by default. APEX passes the sizes of the state variables, so you normally leave it out. Without it the whole state is one block for the default scaling.
- `state_scaling`: an explicit scaling per state row (or an `.npz` holding it as `scaling`), replacing the default prior-ensemble standard deviation per variable. Unset by default.
- `state_scaling_file`: a path to such a file. Unset by default.

`full` needs each member's prior state, the state before any update. The iterative methods (`gies`, `lm_enrml`, `gn_enrml`) provide it. `state_scaling` and `state_scaling_file` are two ways of giving the same scaling; set only one.

**`gmac`**

- `terms`: list of tables, unset by default. Each table has `transform = {name, options}`, `norm = {name, options}` and `weight`, which is `{rule = "trace", alpha = ...}` or `{rule = "fixed", value = ...}`.
- `code`: string, unset by default. The paper's shorthand used instead of `terms`, for example `"10100"`. With neither `terms` nor `code`, the metric is one identity `l2sq` term (`code = "100"`), which is `rlmmac`.
- `variables`: list of strings, unset by default (all variables). The variables that the TV and histogram terms of the code use.
- `energy`: number, default `0.95`.
- `form`: `"subspace"` (the paper's algorithm, default) or `"direct"` (`M_D + alpha M_j`, inverted whole).
- `metric_floor`: number, default `1e-10`. Eigenvalues of the step metric below this fraction of the largest are dropped (subspace form).
- `metric_energy`: number in `(0, 1)`, unset by default. Keeps only the eigenvalues up to the first whose cumulative share exceeds this fraction. When unset, it is 0.99 if no term is `identity`, and off otherwise.
- `constraints`: list of tables, unset by default. Soft constraints (C-GIES): each has `function = {name, options}`, `metric = {name, options}` and `weight` (`{rule = "trace", w = ...}` or `{rule = "fixed", value = ...}`). The regularisation then defaults to the identity.
- `localize_constraints`: true or false, default `true`. With a localization, the constraint terms are tapered too.

The names inside `transform`, `norm`, `function` and `metric` come from registries that `apex plugins` lists in your installation. The simplest start is `code`, as in the example below.

**`hybrid`**

- `energy`: number, default `0.98` (kept per level).

**`local_analysis`**

- `energy`: number, default `0.98`.
- `block_sizes`: list of integers, unset by default; APEX passes the state variable sizes for the default data selection.
- `weight`: true or false, default `false`. If true, each local increment is also multiplied by the taper; by default only its support is used.

**`local_rlmmac`**

- `energy`: number, default `0.95`.
- `block_sizes`: list of integers, unset by default; set from the state layout.
- `weight`: true or false, default `false`. By default the taper only decides which data each group uses.

**`margis`**

- `nu`: number, unset by default, which means `N - 1` (PET's). It is the degrees of freedom of the prior on each data type's error scale. Larger values trust the data error covariance more.
- `s`: number, default `1.0`. The prior scale of the whitened errors: 1 says the data error covariance is right on average.

**`rlmmac`**

- `energy`: number, default `0.95`.
- `scale_data_anomalies`: true or false, default `true`. If false, the data anomalies are left unscaled to reproduce PET-style runs; the update then does not follow the theory and a warning is logged.

**`subspace`**

- `energy`: number, default `0.98`.

**`subspace2`**

- No options.

### The method's own options and the damping scheme

- `gies` has `max_trials_per_iteration` (integer, unset by default): retry a rejected trial within the same iteration, with more damping, up to this many trials. Unset, every trial is one iteration.
- `lm_enrml` and `gn_enrml` have `max_inner_iterations` (integer, default `10`): the number of rejected trials in a row after which the run stops.
- `max_iterations` (default `10`, at least 1) in `[method]` applies to the iterative methods.

The name `rlmmac` also exists as a damping scheme, chosen in `[method.damping]` with `name = "rlmmac"`. It is the default of `gies`, and is separate from the `rlmmac` strategy. Its options:

- `initial`: number, default `1.0`. The damping of the first analysis; positive.
- `factor`: number, default `2.0`. The damping is multiplied by it after a rejected trial; greater than 1.
- `minimum`: number, unset by default (no floor). The damping is not reduced once it is at or below this.
- `data_mismatch_tolerance`: number, default `0.01`. The run has converged when an improving trial changes the mean data mismatch by a relative amount below this.
- `reduction`: number in `(0, 1]`, default `0.9`. The damping is multiplied by it after an accepted trial.
- `converge_on_rejection`: true or false, default `false`. If true, a worse trial within the tolerance also stops the run; otherwise it is rejected and the damping raised.
- `max_stalled_rejections`: integer, default `3`. Stop after this many consecutive rejected trials within the tolerance.
- `confirm_convergence`: true or false, default `false`. If true, a small improvement stops the run only when the damping is at its `minimum` or the previous accepted trial was small too. Set it to true when a run seems to stop too early.

## Worked example

`full` with a lower `energy`, for `lm_enrml`:

```toml
[method]
name = "lm_enrml"
max_iterations = 10

[method.strategy]
name = "full"
options = { energy = 0.95 }
```

Options you did not write keep their defaults. A lower `energy` than `0.98` keeps fewer singular directions.

`gies` with the `rlmmac` strategy, an unscaled variant of the data anomalies, and its damping scheme:

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

`gmac` with the paper's shorthand:

```toml
[method]
name = "gies"

[method.strategy]
name = "gmac"
options = { code = "10100", energy = 0.95, form = "subspace" }
```

`ert_localized` with its required `taper_file` (the path is a placeholder for your own `.npy` or `.npz` file):

```toml
[method]
name = "es"

[method.strategy]
name = "ert_localized"
options = { taper_file = "path/to/taper.npy", truncation = 0.99 }
```

`multilevel_esmda` defines the fidelity levels that `hybrid` needs. Levels are listed coarsest first; the `members` of all levels must add up to the ensemble size of your prior (here 100); the `weight`s are normalised to sum 1. Leaving out `[method.strategy]` gives `hybrid`:

```toml
[method]
name = "multilevel_esmda"

[method.options]
inflation = [2.0, 2.0]
levels = [
  { members = 80, weight = 0.7 },
  { members = 20, weight = 0.3 },
]

[method.strategy]
name = "hybrid"
options = { energy = 0.98 }
```

Finally, list the plugins and check a configuration without simulating anything:

```bash
uv run apex plugins
uv run apex validate config.toml
```

## Pitfalls

- **Unknown options are rejected.** `hybrid` and `subspace` accept only `energy`; `subspace2` accepts no options.
- **`ert_localized` needs `taper_file`.** It has no default, so the section is invalid without it, and the file's shape must match your state and data.
- **Different defaults.** `energy` is `0.95` in `gmac`, `local_rlmmac` and `rlmmac`, and `0.98` elsewhere. The ERT strategies use `truncation` with `0.99`.
- **Small ensembles and `threshold`.** The default `3 / sqrt(N)` is refused below ten members for `adaptive_rows` and `ert_adaptive`.
- **`es` and `esmda` refuse `full`.** Use `approx` with them.
- **Localization with `gies`.** Use `local_rlmmac`; `local_analysis` and `adaptive_rows` are refused.
- **Do not put `localization` among the strategy options.** Use `[method.localization]`.
- **Two things called `rlmmac`.** `[method.strategy]` and `[method.damping]` with `name = "rlmmac"` have different options.
- **Types matter.** `block_sizes` is a list of integers such as `[10, 10, 5]`; `weight`, `sensitivities` and `scale_data_anomalies` are TOML booleans; `form` must be exactly `"subspace"` or `"direct"`.
- **Set only one of `state_scaling` and `state_scaling_file`.**

## What to read next

- The other plugin pages in the plugins section of this guide, for methods, localizations and damping schemes.
- The configuration reference, for the full layout of a TOML file.

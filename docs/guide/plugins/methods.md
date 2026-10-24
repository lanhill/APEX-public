---
audience: public
items:
  - cli:apex plugins
  - cli:apex report
  - cli:apex resume
  - cli:apex run
  - cli:apex stop
  - plugin:methods:chop
  - plugin:methods:es
  - plugin:methods:esmda
  - plugin:methods:gies
  - plugin:methods:gn_enrml
  - plugin:methods:lm_enrml
  - plugin:methods:multilevel_esmda
---

# Methods

This page covers the plugins in the `methods` registry: `chop`, `es`, `esmda`, `gies`, `gn_enrml`, `lm_enrml` and `multilevel_esmda`. Each one is a history-matching method, meaning an algorithm that adjusts an ensemble of model parameters until the model output matches your observed data. You choose one in the method section of your TOML run configuration.

## Goal

By the end of this lesson you will be able to:

- list the method plugins and their options with `apex plugins`,
- choose a method and know which options it accepts,
- start, stop, resume and report on a run with `apex run`, `apex stop`, `apex resume` and `apex report`.

## Prerequisites

- A complete APEX run configuration file in TOML. The data, model and other sections are described on other pages. This page covers only the method.
- APEX installed, so that the `apex` command works.

## Steps

1. Run `apex plugins` to list every plugin with its status and options. Add `--json` for the full descriptions as JSON.
2. Pick a method from the guide below.
3. In the `[method]` section of your configuration, set `name` to the plugin you chose.
4. Set the options of that plugin, and only those. Option schemas reject unknown options.
5. Start the run with `apex run config.toml`.

### The seven methods

Every method here except `chop` has the registry status `parity`; `chop` has none, as it is not a port of a PET method. That is the status `apex plugins` reports. It is informational: you never set it, and it does not change how you configure a method.

| Plugin | What it does | When to pick it | Options |
|---|---|---|---|
| `chop` | CHOP (continuous hyper-parameter optimization): wraps `gies` (the default) or `lm_enrml` and estimates hyper-parameters of its update, such as localization length scales, during the history match (X. Luo, W. C. Cruz, X.-L. Zhang and H. Xiao (2023). Hyper-parameter optimization for improving the performance of localization in an iterative ensemble smoother. *Geoenergy Science and Engineering* 231, 212404. [doi:10.1016/j.geoen.2023.212404](https://doi.org/10.1016/j.geoen.2023.212404)). | When you would otherwise tune localization length scales or inflation factors by hand. | `inner`, `components`, `hyper_localization`, `max_batch_mb`, and the inner method's own options |
| `es` | Ensemble smoother: one single update of the ensemble using all the data at once (ES-MDA with one step at `alpha = 1`). | The simplest and cheapest choice, for a first pass or a nearly linear problem. | none |
| `esmda` | ES-MDA (ensemble smoother with multiple data assimilation): several smaller updates, each using a covariance inflation factor `alpha_i` that makes the observation noise look larger. The factors come from a fixed schedule. | When a single update is too coarse and you want a few gentler updates. | `inflation` (required) |
| `gies` | GIES, an iterative smoother (X. Luo (2021). Novel iterative ensemble smoothers derived from a class of generalized cost functions. *Computational Geosciences* 25(3), 1159–1189. [doi:10.1007/s10596-021-10046-1](https://doi.org/10.1007/s10596-021-10046-1)). It keeps the perturbed observations fixed and damps the updates it makes to the strategy. | When you want an iterative method that limits how far each update moves. | `max_trials_per_iteration` |
| `gn_enrml` | GN-EnRML: the loop of `lm_enrml`, plus a step length that scales each update. | An iterative method where you want control of the step taken. | `max_inner_iterations` |
| `lm_enrml` | LM-EnRML: the loop of `gies`, with a simplified (approximate) update and Levenberg-Marquardt damping, which shrinks steps when the fit gets worse. | A damped iterative method for strongly nonlinear problems. | `max_inner_iterations` |
| `multilevel_esmda` | ES-MDA that works with several fidelity levels (versions of the model of differing accuracy and cost). | When you have models of different fidelity. | `inflation` (required), `levels` (required) |

### Options

| Plugin | Option | Type | Default | Required |
|---|---|---|---|---|
| `chop` | `inner` | string | `gies` | no |
| `chop` | `components` | array of objects | `[{ name = "length_scale", mode = "data" }]` | no |
| `chop` | `hyper_localization` | string or object | `same` | no |
| `chop` | `max_batch_mb` | number | 256.0 | no |
| `esmda` | `inflation` | array of numbers | none | yes |
| `gies` | `max_trials_per_iteration` | integer or null | null | no |
| `gn_enrml` | `max_inner_iterations` | integer | 10 | no |
| `lm_enrml` | `max_inner_iterations` | integer | 10 | no |
| `multilevel_esmda` | `inflation` | array of numbers | none | yes |
| `multilevel_esmda` | `levels` | array of objects | none | yes |

The `es` plugin has no options.

`chop` passes every option it does not know itself to its inner method, so a `chop` method's options may also hold, for example, `max_trials_per_iteration` for `gies`. CHOP needs the `rlmmac` strategy and replaces a localization, so leave out the method's `localization` section. Its components and hyper-parameter localization are described on the [CHOP page](../../chop.md).

TOML has no null value. To get the default (null) for `max_trials_per_iteration`, leave the key out. Likewise, leaving out `max_inner_iterations` gives 10.

#### The levels of `multilevel_esmda`

`levels` is a list of tables, one per fidelity level, **coarsest first**. Each level holds consecutive members of the ensemble:

| Key | Required | Meaning |
|---|---|---|
| `members` | yes | The level's number of members, at least 2. The levels' members must add up to `[prior] ensemble_size`. |
| `weight` | yes | The level's weight in the hybrid gain, non-negative. The weights are normalised to sum 1. |
| `upscale` | no | Grid coarsening of the level's deck (`method`, `dim`, `max_trunc`, ...), for `eclipse`. |
| `deck_template_file` | no | A coarse deck template of the level's own, for `opm_flow` or `eclipse`. |

A level with neither `upscale` nor `deck_template_file` runs the fine model. The rules:

- `inflation` is ES-MDA's schedule, as for `esmda`: one factor per step, and the sum of 1/α must be 1 (`[2, 2]`, `[4, 4, 4, 4]`). Its length is the number of steps and has nothing to do with the number of levels.
- The only update strategy is `hybrid`, the default: leave out `[method.strategy]`.

```toml
[prior]
ensemble_size = 100

[method]
name = "multilevel_esmda"

[method.options]
inflation = [4.0, 4.0, 4.0, 4.0]
levels = [
  { members = 70, weight = 0.3, deck_template_file = "coarse.mako" },
  { members = 30, weight = 0.7 },
]
```

Here members 0–69 run the coarse deck and members 70–99 the fine one.

### Items APEX supplies

APEX itself hands each method some items. They are not options of the method, so never set them there:

- `lower_bounds`, `strategy` and `upper_bounds`: supplied to all six methods other than `chop`, which passes them on to its inner method.
- `inner_method` and `block_sizes`: supplied to `chop`. The inner method comes from the `inner` option.
- `max_iterations` and `schedule`: also supplied to `gies`, `gn_enrml` and `lm_enrml`. These come from the rest of your configuration, not from the method options. The `inflation` array of `esmda` and `multilevel_esmda` is a separate thing: it is a method option that you set yourself.

## Worked example

This example picks the plain ensemble smoother, `es`, which needs no options. It is a fragment: your `config.toml` must also contain the other required sections (data, model and so on).

```toml
[method]
name = "es"
```

Start the run:

```bash
apex run config.toml
```

Progress goes to standard error: bars on a terminal, timestamped lines otherwise. Use `--no-progress` to hide it (`--progress` is the default).

```bash
apex run config.toml --no-progress
```

An analysis step is one update of the ensemble. To stop after a few steps, add `--max-steps`:

```bash
apex run config.toml --max-steps 2
```

Continue from the last checkpoint by giving `apex resume` the run directory:

```bash
apex resume RUN_DIR --max-steps 2
```

`apex resume` accepts the same `--max-steps` and `--progress`/`--no-progress` flags. To stop a run that is still going, at its next checkpoint, use:

```bash
apex stop RUN_DIR
```

When a run has finished or stopped, print its data mismatch history and performance metrics, and optionally write figures (PNG and PDF):

```bash
apex report RUN_DIR --plots
```

Figures go to `RUN_DIR/report` unless you pass `--output`. `--font-size` sets the font size in points (default 12).

### Example: CHOP

GIES with its localization length scales estimated by CHOP, one per datum:

```toml
[method]
name = "chop"
max_iterations = 20
strategy = { name = "rlmmac", options = { energy = 0.99 } }

[method.options]
inner = "gies"
max_trials_per_iteration = 4
components = [{ name = "length_scale", mode = "data" }]
```

## Pitfalls

- Starting `apex run` when the run directory already holds a run. APEX refuses with an error. Use `apex resume` on that directory, or choose another run directory.
- Leaving out `inflation` (for `esmda` and `multilevel_esmda`) or `levels` (for `multilevel_esmda`). They are required.
- Giving `inflation` as a single number. It must be an array of numbers.
- Giving `levels` as anything other than an array of objects.
- Setting `lower_bounds`, `strategy`, `upper_bounds`, `max_iterations` or `schedule` as method options. APEX supplies them, and unknown options are rejected.
- Setting any option on `es`. It accepts none.
- Setting `max_inner_iterations` on a method other than `gn_enrml` and `lm_enrml`.
- Passing a non-integer to `max_inner_iterations` or `max_trials_per_iteration`.
- Looking for a null in TOML. Omit the key instead.
- Giving `chop` a strategy other than `rlmmac`, an inner method other than `gies` or `lm_enrml` (`gn_enrml` runs on GIES's loop but refuses `rlmmac`), or a `localization` section. CHOP refuses all three.

## What to read next

- The other pages in the plugins guide, for the rest of your run configuration.
- The output of `apex plugins --json`, for the full description of each method, including the keys a `multilevel_esmda` level accepts.

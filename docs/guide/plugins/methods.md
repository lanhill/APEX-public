---
audience: public
items:
  - cli:apex plugins
  - cli:apex predict
  - cli:apex report
  - cli:apex resume
  - cli:apex run
  - cli:apex schema
  - cli:apex stop
  - config:history_match:method
  - config:history_match:prior
  - plugin:damping_schemes:lm
  - plugin:damping_schemes:rlmmac
  - plugin:methods:chop
  - plugin:methods:es
  - plugin:methods:esmda
  - plugin:methods:gies
  - plugin:methods:gn_enrml
  - plugin:methods:lm_enrml
  - plugin:methods:multilevel_esmda
  - plugin:update_strategies:hybrid
  - plugin:update_strategies:rlmmac
---

# Methods

This page covers the plugins in the `methods` registry: `chop`, `es`, `esmda`, `gies`, `gn_enrml`, `lm_enrml` and `multilevel_esmda`. Each one is a history-matching method: an algorithm that adjusts an ensemble of model parameters until the model output matches your observed data. You choose one in the `[method]` section of your TOML run configuration.

## Goal

By the end of this lesson you will be able to:

- list the method plugins and their options with `apex plugins`,
- choose a method and know which options it accepts and where to set them,
- start, stop, resume and report on a run with `apex run`, `apex stop`, `apex resume` and `apex report`.

## Prerequisites

- A complete APEX run configuration file in TOML. The data, model and other sections are described on other pages. This page covers only `[method]`.
- APEX installed as described in the README. `apex` is not on your PATH, so every command here is typed from the APEX folder as `uv run apex ...`.

## Some words used on this page

- **Analysis step**: one update of the ensemble. `es` makes one step. `esmda` makes one step per inflation factor. An iterative method (`gies`, `gn_enrml`, `lm_enrml`) makes one step per iteration, up to `max_iterations`.
- **Update strategy**: the formula that turns the data mismatch into an update of the ensemble. You set it in `[method.strategy]`, by name (`strategy = "rlmmac"`) or with options. Two examples are `rlmmac` and `hybrid`. Run `uv run apex plugins` to list the strategies (the `update_strategies` registry) with their options. The facts for this page do not state a default strategy for each method, except that `multilevel_esmda` uses `hybrid`.
- **Localization**: a way to limit each datum's influence to the parameters that are plausibly related to it. You set it in `[method.localization]`. If you leave it out, no localization is used, except by the local-analysis strategies (`local_analysis`, `local_rlmmac`, `adaptive_rows`), which select data by prior correlation.
- **Damping**: shrinking the steps of an iterative method when the fit gets worse. You choose a damping scheme in `[method.damping]`: `rlmmac` (the default for `gies`) or `lm` (the default for `lm_enrml`). It applies to `gies` and `lm_enrml`.
- **Checkpoint**: the state a run saves as it goes. `apex resume` continues from the last one, and `apex stop` takes effect at the next one.

## Steps

1. Run `uv run apex plugins` to list every plugin with its options. Add `--json` for the full descriptions as JSON.
2. Pick a method from the guide below.
3. In the `[method]` section of your configuration, set `name` to the plugin you chose.
4. Put the plugin's own options in `[method.options]`, and only those. Option schemas reject unknown options.
5. Set the settings that belong to `[method]` itself directly under `[method]`: `strategy`, `localization`, `damping`, `schedule` and `max_iterations` (default 10, at least 1). Iterative methods use `max_iterations` and `schedule`. See "Damping and schedule" below for what `damping` and `schedule` look like.
6. Start the run with `uv run apex run config.toml`. Add `--max-steps N` to stop after N analysis steps; you can continue later with `apex resume`.

To see everything a configuration file accepts, run `uv run apex schema`. Its argument is one of `history_match` (the default), `prediction`, `study`, `optimization`, `optimization_study` or `service`.

### The seven methods

Every method here except `chop` has the registry status `parity`; `chop` has none. That status is informational: you never set it.

| Plugin | What it does | When to pick it | Options |
|---|---|---|---|
| `chop` | CHOP (continuous hyper-parameter optimization): wraps `gies` (the default) or another inner method and estimates hyper-parameters of its update, such as localization length scales, per member during the history match (X. Luo, W. C. Cruz, X.-L. Zhang and H. Xiao (2023). Hyper-parameter optimization for improving the performance of localization in an iterative ensemble smoother. *Geoenergy Science and Engineering* 231, 212404. [doi:10.1016/j.geoen.2023.212404](https://doi.org/10.1016/j.geoen.2023.212404)). | When you would otherwise tune localization length scales by hand. | `inner`, `components`, `hyper_localization`, `max_batch_mb` |
| `es` | Ensemble smoother: a single update of the ensemble using all the data at once (ES-MDA's single step at `alpha = 1`). | The simplest and cheapest choice, for a first pass or a nearly linear problem. | none |
| `esmda` | ES-MDA (ensemble smoother with multiple data assimilation): several smaller updates with a fixed schedule of covariance inflation factors `alpha_i`. You list the factors in `inflation`. | When a single update is too coarse and you want a few gentler updates. | `inflation` (required) |
| `gies` | GIES, an iterative smoother (X. Luo (2021). Novel iterative ensemble smoothers derived from a class of generalized cost functions. *Computational Geosciences* 25(3), 1159–1189. [doi:10.1007/s10596-021-10046-1](https://doi.org/10.1007/s10596-021-10046-1)). It keeps the same perturbed observations in every iteration and applies damped strategy updates, so each step is limited. | When you want an iterative method that limits how far each update moves. | `max_trials_per_iteration` |
| `gn_enrml` | GN-EnRML: an iterative smoother with Gauss-Newton steps scaled by a step length. | An iterative method where you want control of the step taken. | `max_inner_iterations` |
| `lm_enrml` | LM-EnRML: an iterative smoother with Levenberg-Marquardt damping of its steps. | A damped iterative method for strongly nonlinear problems. | `max_inner_iterations` |
| `multilevel_esmda` | ES-MDA's schedule on an ensemble split over several fidelity levels (versions of the model of differing accuracy and cost). | When you have models of different fidelity. | `inflation` (required), `levels` (required) |

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

TOML has no null value. To get the default (null) for `max_trials_per_iteration`, leave the key out. Likewise, leaving out `max_inner_iterations` gives 10.

#### The `inflation` list

`inflation` (for `esmda` and `multilevel_esmda`) holds the factors `alpha_i`, one per analysis step, and you set it yourself. The length of the list is the number of steps. For `multilevel_esmda` the sum of 1/`alpha_i` must equal 1. Check it by adding the reciprocals:

- `[2.0, 2.0]`: 1/2 + 1/2 = 1, so two steps.
- `[4.0, 4.0, 4.0, 4.0]`: four times 1/4 = 1, so four steps.

The `es` method is the single-step case at `alpha = 1`.

#### The `chop` options

- `inner` is the name of the wrapped method (default `gies`); its options follow.
- `components` lists what CHOP estimates, each entry a table with a `name` and a `mode`, from the `chop_components` registry. The default, `{ name = "length_scale", mode = "data" }`, is the paper's: one length scale per datum.
- `hyper_localization` says how the hyper-parameter update is localized: `"same"` (each component's own taper, the default), `"none"`, or the name of a registered localization (or `{ name = ..., options }`).
- `max_batch_mb` is the memory of one row block of the model update's per-member tapers.

#### The `strategy` of each method

- `rlmmac` works with `gies`, `lm_enrml` and `chop`. APEX rejects it for `es`, `esmda`, `gn_enrml` and `multilevel_esmda`. For `gn_enrml` the error suggests `approx`, `full`, `subspace`, `subspace2` or `margis`; for `es` and `esmda` it suggests `approx`.
- `hybrid` works with `multilevel_esmda` only, and it is that method's only strategy: leave out `[method.strategy]`, or name `hybrid`.

#### Damping and schedule

Both go directly under `[method]`. `damping` picks a damping scheme and its options. `schedule` is a table of the default scheme's options, used when you give no `damping`. The scheme options below are real ones; defaults are in parentheses.

- `rlmmac` scheme: `initial` (1), `factor` (2, greater than 1), `minimum` (unset), `reduction` (0.9), `data_mismatch_tolerance` (0.01), `converge_on_rejection` (false), `max_stalled_rejections` (3), `confirm_convergence` (false).
- `lm` scheme: `initial` (100, or `"auto"`), `factor` (5), `minimum` (0.01), `maximum` (1e10), `data_mismatch_tolerance` (0.01), `converge_on_rejection` (false), `max_stalled_rejections` (3), `confirm_convergence` (true).

For example, `lm_enrml` with the `lm` scheme, starting at an automatic damping:

```toml
[method]
name = "lm_enrml"
max_iterations = 15

[method.damping]
name = "lm"

[method.damping.options]
initial = "auto"
factor = 5.0
```

And `gies` with the default scheme's options given through `schedule`:

```toml
[method]
name = "gies"

[method.schedule]
initial = 1.0
factor = 2.0
```

#### The levels of `multilevel_esmda`

`levels` is a list of tables, one per fidelity level, **coarsest first**. Each level holds consecutive members of the ensemble:

| Key | Required | Meaning |
|---|---|---|
| `members` | yes | The level's number of members, at least 2. The levels' members must add up to `[prior] ensemble_size`. |
| `weight` | yes | The level's weight in the hybrid gain, non-negative. APEX normalises the weights to sum 1, so any non-negative numbers work. |
| `upscale` | no | Grid coarsening of the level's simulator (`method`, `dim`, `max_trunc`, and more). |
| `deck_template_file` | no | A deck template of the level's own (a coarse model). |
| `forward_file` | no | The linear simulator's G of the level, an `.npz` file. |
| `matrix` | no | The linear simulator's G, inline. |

A level with no simulator key runs the fine model. The *hybrid gain* is the one gain, used to update the whole ensemble, that is built from the anomalies (deviations from the mean) of every level; the weights say how much each level counts. Run `uv run apex plugins --json` for the full description of the level keys.

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

The members add up to 70 + 30 = 100, the `ensemble_size`. Members 0–69 run the coarse deck and members 70–99 the fine one. The weights 0.3 and 0.7 happen to sum to 1; weights of 3 and 7 would give the same result after normalisation.

### Items APEX supplies

APEX itself hands each method some items from the rest of your configuration. They are not options of the method, so never put them in `[method.options]`:

- `lower_bounds`, `strategy` and `upper_bounds`: supplied to `es`, `esmda`, `gies`, `gn_enrml`, `lm_enrml` and `multilevel_esmda`. You choose the strategy in `[method.strategy]`; the bounds are not set on the method.
- `inner_method` and `block_sizes`: supplied to `chop`. The inner method comes from the `inner` option.
- `max_iterations` and `schedule`: also supplied to `gies`, `gn_enrml` and `lm_enrml`. You set them directly under `[method]`, not in `[method.options]`.

## Worked example

This example uses `esmda` with two steps. It is a fragment: your `config.toml` must also contain the other required sections (data, model and so on).

```toml
[method]
name = "esmda"

[method.options]
inflation = [2.0, 2.0]
```

Start the run, stopping after one analysis step. `--max-steps 1` makes `apex run` stop after one step, so you can inspect the run and continue later:

```bash
uv run apex run config.toml --max-steps 1
```

Progress goes to standard error: bars on a terminal, timestamped lines otherwise. Add `--no-progress` to hide it (`--progress` is the default).

A run lives in its own run directory. The `[run]` section of your configuration sets it (the `name` or `output_dir`). Because of `--max-steps`, this run has stopped with work left. Continue it from its last checkpoint by giving `apex resume` that directory:

```bash
uv run apex resume RUN_DIR
```

Replace `RUN_DIR` with your run directory. `apex resume` accepts the same `--max-steps` and `--progress`/`--no-progress` flags. To ask a run that is still going to stop at its next checkpoint, open a second terminal in the APEX folder (the first one is busy with `apex run`) and type:

```bash
uv run apex stop RUN_DIR
```

The run stops at its next checkpoint; `apex resume RUN_DIR` continues it later, and `apex report` can already print what it has so far.

When a run has finished or stopped, print its data mismatch history and performance metrics, and optionally write figures (PNG and PDF):

```bash
uv run apex report RUN_DIR --plots
```

Figures go to `RUN_DIR/report` unless you pass `--output`. `--font-size` sets the font size in points (default 12).

### Example: CHOP

GIES with its localization length scales estimated by CHOP, one per datum (`mode = "data"`). `max_iterations` and `strategy` are `[method]` settings, so they sit directly under `[method]`; the rest are options of `chop`. Here the `rlmmac` strategy keeps `energy = 0.99` of the singular-value sum (default 0.95):

```toml
[method]
name = "chop"
max_iterations = 20
strategy = { name = "rlmmac", options = { energy = 0.99 } }

[method.options]
inner = "gies"
components = [{ name = "length_scale", mode = "data" }]
```

## Pitfalls

- Leaving out `inflation` (for `esmda` and `multilevel_esmda`) or `levels` (for `multilevel_esmda`). They are required.
- Giving `inflation` as a single number. It must be an array of numbers; for `multilevel_esmda` their 1/`alpha_i` must sum to 1.
- Giving `levels` as anything other than an array of objects, or with members that do not add up to `[prior] ensemble_size`.
- Setting `lower_bounds`, `strategy`, `upper_bounds`, `max_iterations` or `schedule` in `[method.options]`. Set `strategy`, `max_iterations` and `schedule` directly under `[method]`; unknown options are rejected.
- Setting any option on `es`. It accepts none.
- Setting `max_inner_iterations` on a method other than `gn_enrml` and `lm_enrml`.
- Passing a non-integer to `max_inner_iterations` or `max_trials_per_iteration`.
- Looking for a null in TOML. Omit the key instead.
- Choosing the `rlmmac` strategy with `es`, `esmda`, `gn_enrml` or `multilevel_esmda`. Use it with `gies`, `lm_enrml` or `chop`.
- Choosing `hybrid` with anything but `multilevel_esmda`. APEX rejects it and says to use `multilevel_esmda`.

## What to read next

- The other pages in the plugins guide, for the rest of your run configuration.
- `uv run apex plugins --json`, for the full description of each method, including the keys a `multilevel_esmda` level accepts.
- `uv run apex schema`, for what the `[method]` section of the configuration accepts.
- `uv run apex predict` with a prediction configuration, to forecast the ensembles of a finished run beyond history; `apex report` also accepts a prediction's directory.

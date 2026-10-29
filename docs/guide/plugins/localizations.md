---
audience: public
items:
  - cli:apex plugins
  - cli:apex predict
  - cli:apex resume
  - cli:apex run
  - cli:apex schema
  - cli:apex stop
  - config:history_match:method
  - plugin:damping_schemes:rlmmac
  - plugin:localizations:adaptive
  - plugin:localizations:adaptive_per_update
  - plugin:localizations:adaptive_prior
  - plugin:localizations:adaptive_projected
  - plugin:localizations:distance
  - plugin:localizations:distance_entries
  - plugin:methods:gn_enrml
  - plugin:methods:lm_enrml
  - plugin:update_strategies:adaptive_rows
  - plugin:update_strategies:approx
  - plugin:update_strategies:local_analysis
  - plugin:update_strategies:local_rlmmac
  - plugin:update_strategies:rlmmac
---

# Localizations

Localization limits how far an observation may influence a parameter during an update. Localization plugins live in the `localizations` registry. This page covers all six: `adaptive`, `adaptive_per_update`, `adaptive_prior`, `adaptive_projected`, `distance` and `distance_entries`.

To list every plugin with its options, run:

```bash
uv run apex plugins
uv run apex plugins --json
```

`--json` gives the full descriptions as JSON.

## Goal

By the end of this page you will be able to:

- choose a localization plugin;
- select it in `[method.localization]` and set its options;
- avoid the common mistakes in the `distance` and `distance_entries` options.

## Prerequisites

- An APEX run configuration file in TOML (the file you pass to `apex run`). Localization goes in its `[method]` section (`history_match:method` in the configuration reference).
- For `distance` and `distance_entries`, your grid dimensions `nx`, `ny` and `nz`. Cell positions are 1-based, as in Eclipse decks.

APEX supplies `block_sizes` to every plugin on this page, and `variable_names` to `distance_entries`. Do not set these in your configuration. All six plugins reject options they do not know, so a misspelled option name is an error and is not silently ignored.

## Choosing a plugin

The four adaptive plugins have no positions to set: you choose a taper and a threshold. The two distance plugins use kernels around cells that you place and size yourself.

| Plugin | What it is (PET name) | Use it when |
|---|---|---|
| `adaptive` | adaptive localization with the taper built once from the prior (`ini_only`) | you want the taper computed once, at the start. Try this first |
| `adaptive_prior` | an alias of `adaptive`: identical behaviour and options | you meet this name in older material; otherwise write `adaptive` |
| `adaptive_per_update` | automatic and adaptive localization, recomputed at every update (`autoadaloc`) | you want the taper recomputed at every update; try it second |
| `adaptive_projected` | upstream PET's adaptive localization (`autoadaloc`) in the projected-data space | you need to reproduce PET's own `autoadaloc` settings (`threshold = "adaptive"` and `cutoff` as a multiplier of `sigma`) |
| `distance` | distance-based localization around well locations (`dist_loc`) | one radius and taper for a set of named wells |
| `distance_entries` | upstream PET's distance localization (`distance_loc`), one entry per data and variable | you want per-entry control of centre, reach, shape and target |

Rule of thumb between the two `autoadaloc` plugins: `adaptive_per_update` uses a threshold of the form `threshold_scale * sqrt(2 ln n_block) * sigma` and is the one with a `threshold_scale`; `adaptive_projected` has no `threshold_scale` and works in PET's projected-data space, a term this page does not define beyond that. Start with `adaptive_per_update` unless you must match PET.

A run has one localization, chosen in `[method.localization]`: `name` is the plugin, and its options go in `[method.localization.options]`. It is passed to the update strategy, so `[method.strategy]` must name a strategy that receives a localization: `approx`, `rlmmac`, `local_analysis` and `local_rlmmac` do. `adaptive_rows` is not passed one, so do not combine it with `[method.localization]`. The snippets below use `approx` with `lm_enrml` (the iterative methods `lm_enrml` and `gn_enrml` both use a strategy in the same way).

Without `[method.localization]` the update is not localized, except by the local-analysis strategies (`local_analysis`, `local_rlmmac`, `adaptive_rows`), which select data by prior correlation. These strategies belong to the `update_strategies` registry.

Note that `rlmmac` is also the name of a damping scheme (`damping_schemes`), a different plugin from the `rlmmac` update strategy. A localization is passed to the strategy, not to the damping scheme.

## Steps for the adaptive plugins

The adaptive plugins are correlation-based localization, from:

- X. Luo, T. Bhakta and G. Nævdal (2018). Correlation-based adaptive localization with applications to ensemble-based 4D-seismic history matching. *SPE Journal* 23(2), 396–427. [doi:10.2118/185936-PA](https://doi.org/10.2118/185936-PA)
- X. Luo and T. Bhakta (2020). Automatic and adaptive localization for ensemble-based history matching. *Journal of Petroleum Science and Engineering* 184, 106559. [doi:10.1016/j.petrol.2019.106559](https://doi.org/10.1016/j.petrol.2019.106559)

`adaptive`, `adaptive_per_update` and `adaptive_prior` share the same four optional options:

| Option | Type | Default | Allowed values |
|---|---|---|---|
| `threshold_scale` | number | `1.0` | positive; larger keeps fewer correlations; not with `threshold = "fixed"` |
| `taper` | string | `"hard"` | `"hard"`, `"soft"`, `"sigmoid"` |
| `threshold` | string | `"universal"` | `"universal"`, `"fixed"` |
| `cutoff` | number or unset | unset | in `(0, 1)`; only with `threshold = "fixed"` |

### Choosing the threshold

A correlation is kept only if it is above a threshold. `threshold` chooses how that threshold is found:

- `"universal"` (default): a threshold per state variable and datum, estimated from correlations with a shuffled ensemble. It grows with the size of the variable. For `adaptive_per_update` it is `threshold_scale * sqrt(2 ln n_block) * sigma`; with 18,000 cells and 100 members it comes to 0.45, so only strong correlations pass. With many cells and few members it can keep almost nothing.
- `"fixed"`: the threshold is `cutoff` everywhere, whatever the size of the variable. If `cutoff` is unset, it is `3/sqrt(N)` for `N` members (0.3 at 100), three standard deviations of a sample correlation where there is none. A fixed threshold takes no `threshold_scale`.

Use `"universal"` when you want the threshold to follow the size of each variable. Use `"fixed"` when you want one predictable cutoff, or when the universal one keeps almost nothing.

### Choosing the taper

With the default `"hard"` taper, exactly the correlations above the threshold are kept (`|rho| > cutoff`). `"soft"` scales them down smoothly with Gaspari-Cohn, and `"sigmoid"` uses a steep logistic step. For `adaptive_per_update` with a fixed threshold, the `soft` taper reaches 0 only at `|rho| = 2 cutoff - 1`, so with a cutoff below 0.5 it keeps a little of every correlation. Start with `"hard"`, then change one option at a time and compare the results.

### Steps

1. Add `[method.localization]` with the plugin's name, for example `name = "adaptive_per_update"`.
2. Add `[method.localization.options]` and set only the options where you want to depart from the defaults.

```toml
[method]
name = "lm_enrml"

[method.strategy]
name = "approx"

[method.localization]
name = "adaptive_per_update"

[method.localization.options]
taper = "soft"
```

Here only `taper` changes, from `"hard"` to `"soft"`. This shows how to write an option; it is not a recommendation.

A fixed threshold with a cutoff of your own:

```toml
[method]
name = "lm_enrml"

[method.strategy]
name = "approx"

[method.localization]
name = "adaptive"

[method.localization.options]
threshold = "fixed"
cutoff = 0.25
```

Leave `cutoff` out to get the default `3/sqrt(N)`.

### `adaptive_projected`

`adaptive_projected` has three optional options (it has no `threshold_scale`):

| Option | Type | Default | Allowed values |
|---|---|---|---|
| `threshold` | string | `"adaptive"` | `"adaptive"`: `cutoff * sigma`; `"fixed"`: `cutoff`; `"universal"`: `sqrt(2 ln n_block) * sigma` |
| `cutoff` | number | `0.3` | the multiplier of `sigma` (`"adaptive"`) or the threshold itself (`"fixed"`, in `(0, 1)`); unused by `"universal"` |
| `taper` | string | `"hard"` | `"hard"`, `"soft"`, `"sigmoid"` |

```toml
[method]
name = "lm_enrml"

[method.strategy]
name = "approx"

[method.localization]
name = "adaptive_projected"

[method.localization.options]
threshold = "adaptive"
cutoff = 0.3
taper = "sigmoid"
```

Here `threshold` and `cutoff` are written at their defaults only to show the layout; `taper` is the one that changes.

## Steps for `distance`

Each state variable is one grid property in Eclipse order (`i` fastest), so every block size must equal `nx * ny * nz`.

| Option | Type | Default | Notes |
|---|---|---|---|
| `grid` | list of integers | required | the grid size, `[nx, ny, nz]` |
| `positions` | table of lists of positions | required | each key is a well name; its value is the list of its completion cells, each `[i, j, k]` |
| `radius` | number | required | localization length `c` in cells; the taper vanishes beyond `2c` for Gaspari-Cohn, beyond `c` for Furrer-Bengtsson |
| `taper` | string | `"gaspari_cohn"` | `"gaspari_cohn"` or `"furrer_bengtsson"` |
| `anisotropy` | pair of numbers | `[1.0, 0.0]` | `[ratio, angle in degrees]`; `[1, 0]` is isotropic. Leave it at the default at first |
| `unlisted` | string | `"error"` | `"error"` or `"zero"` |

Each position is three integers `[i, j, k]`, 1-based as in Eclipse decks, so the first cell is `[1, 1, 1]`. The keys of `positions` are well names, spelled exactly as the wells appear in your data.

`unlisted` says what happens to the data of a well that is missing from `positions`:

- `"error"` (default) refuses the run, so a misspelt well cannot silently drop its data;
- `"zero"` leaves the state untouched by that well's data. It is useful when you deliberately want some wells not to update the state through this localization.

Keep the default until you have such a reason.

```toml
[method]
name = "lm_enrml"

[method.strategy]
name = "approx"

[method.localization]
name = "distance"

[method.localization.options]
grid = [40, 40, 10]
radius = 8
taper = "gaspari_cohn"

[method.localization.options.positions]
WELL1 = [[20, 20, 5]]
```

Replace `WELL1` with a well name from your own data.

## Steps for `distance_entries`

### What a kernel is

A kernel is a weighting around a centre cell. Each entry in `entries` defines one kernel, and its `taper` option chooses it: `gaspari_cohn`, `furrer_bengtsson`, `region` or `import`.

- `gaspari_cohn` and `furrer_bengtsson` are defined by a centre `position` and a `radius`.
- `import` takes the weights from a file of your own (`taper_file`), one value per grid cell, instead of from a centre and radius.
- `region` is also available; this page does not describe its weights, so check the results before relying on it.

The `radius` is the kernel's length in cells, not in metres. Treat the radii in the examples as illustrations and try values of your own.

### Plugin options

| Option | Type | Default | Notes |
|---|---|---|---|
| `grid` | list of integers | required | the grid size, `[nx, ny, nz]` |
| `entries` | list of entries | required | in order: a later entry replaces an earlier one for the (data, time, variable) it names, as in PET |
| `actnum_file` | path | none | optional `.npy` or `.npz` file of active cells: one boolean per grid cell, in Eclipse order. When given, the tapers are taken over the active cells only (PET `actnum`) |

The facts state that `taper_file` is resolved relative to the configuration's folder; they do not say this for `actnum_file`, so prefer an absolute path there.

### Entry options

No entry option is required by the schema, and `taper` has no default. Always set:

- for `gaspari_cohn` and `furrer_bengtsson`: `taper`, `position` and `radius`;
- for `import`: `taper` and `taper_file`.

What happens when one is missing is not specified, so do not leave them out.

| Option | Type | Default | Meaning |
|---|---|---|---|
| `data` | string | `"*"` | the data type it localizes; `"*"` means all |
| `time` | string | `"*"` | the data time it localizes; `"*"` means all |
| `parameter` | string | `"*"` | the state variable it localizes; `"*"` means all |
| `taper` | string | none | `gaspari_cohn`, `furrer_bengtsson`, `region` or `import` |
| `position` | list of integers | none | the centre cell `[i, j, k]`, 1-based, as in Eclipse decks |
| `radius` | number | none | the kernel's length, in cells |
| `z_range` | string or integer | `"all"` | `"all"` puts the kernel in every layer; an integer `r` puts it in layers `k - r` to `k + r`, where `k` is the third number of `position` |
| `anisotropy` | list of numbers | none | `[ratio, angle in degrees]` |
| `taper_file` | string | none | the `import` kernel's taper, relative to the configuration's folder |

The schema does not say how a layer range that runs past the first or last layer is handled, so choose `r` so that `k - r` and `k + r` stay inside the grid.

`data`, `time` and `parameter` must match the data types, times and state variable names used in your own setup, spelled exactly. Unless you know the exact time name, leave `time` at `"*"`. For example, `parameter = "pressure"` applies an entry only to a state variable called `pressure`, if your setup has one.

### Anisotropy

The offset of a cell is its displacement from the centre cell `position`. With `anisotropy = [ratio, angle]`, the offset is first rotated by the angle (in degrees). Then its `j` component is divided by the ratio. Dividing makes distances along `j` count as smaller, so the kernel reaches further along `j`: a ratio of 2 doubles the reach along `j`. (PET's own docstring says the reach is halved; its code divides, as APEX does.)

Numeric check: with `angle = 0` there is no rotation. A cell 4 cells away along `j` with ratio 2 then has an effective distance of 2 cells.

The direction in which a positive angle rotates follows upstream PET and is not stated here. Start with `angle = 0` to see the effect of the ratio alone.

### Steps

1. Add `[method.localization]` with `name = "distance_entries"`, and set `grid` in `[method.localization.options]`.
2. Add one `[[method.localization.options.entries]]` table for each rule.
3. In each entry, choose the `taper`, then give `position` and `radius`.
4. Narrow the entry with `data`, `time` and `parameter` if it should not apply to everything.
5. Optionally add `z_range` and `anisotropy`.

### The `import` kernel

With `taper = "import"`, the weights come from `taper_file`, so write only `taper` and `taper_file`, plus any of `data`, `time` and `parameter`. Leave `position` and `radius` out.

The file is `.npy` or `.npz` with one value per grid cell: either flat in Eclipse order or shaped `(nx, ny, nz)`. The shaped form is easier to get right. The path is relative to the folder containing the configuration file, not to the directory where you start `apex run`.

How the values are interpreted is defined by PET and not stated here. The all-ones file below only shows the file layout.

```python
import numpy as np

weights = np.ones((40, 40, 10))  # one value per cell, shape (nx, ny, nz)
np.save("taper.npy", weights)
```

```toml
[method]
name = "lm_enrml"

[method.strategy]
name = "approx"

[method.localization]
name = "distance_entries"

[method.localization.options]
grid = [40, 40, 10]

[[method.localization.options.entries]]
taper = "import"
taper_file = "taper.npy"
```

## Worked example

A 40 × 40 × 10 grid with one kernel centred in cell `[20, 20, 5]`. The kernel has a radius of 8 cells and extends two layers above and below layer 5 (layers 3 to 7). The `j` component of the offset is divided by 2, so the reach along `j` is doubled. `data`, `time` and `parameter` keep their default `"*"`, so the entry applies to everything.

```toml
[method]
name = "lm_enrml"

[method.strategy]
name = "approx"

[method.localization]
name = "distance_entries"

[method.localization.options]
grid = [40, 40, 10]

[[method.localization.options.entries]]
taper = "gaspari_cohn"
position = [20, 20, 5]
radius = 8
z_range = 2
anisotropy = [2, 0]
```

To cover all layers, write `z_range = "all"` or leave `z_range` out.

Run it for two analysis steps as a first test, then continue:

```bash
uv run apex run config.toml --max-steps 2
uv run apex resume path/to/run_dir
```

### Two entries that do not overlap

Each entry below is narrowed to a different variable, so they never apply to the same target. Replace `"pressure"` and `"saturation"` with state variable names from your own setup.

```toml
[method]
name = "lm_enrml"

[method.strategy]
name = "approx"

[method.localization]
name = "distance_entries"

[method.localization.options]
grid = [40, 40, 10]

[[method.localization.options.entries]]
parameter = "pressure"
taper = "gaspari_cohn"
position = [20, 20, 5]
radius = 8
z_range = 2
anisotropy = [2, 45]

[[method.localization.options.entries]]
parameter = "saturation"
taper = "furrer_bengtsson"
position = [10, 30, 3]
radius = 5
```

The first entry also rotates the offset by 45 degrees before the division.

### When entries overlap

Two entries overlap when they can apply to the same data, time and variable and their kernels cover some of the same cells. Because `"*"` matches everything, an entry that leaves `data`, `time` or `parameter` at the default overlaps with every entry covering the same cells. A later entry replaces an earlier one for the (data, time, variable) it names. To keep the outcome predictable, narrow each entry as in the example above.

## Pitfalls

- **Positions are 1-based.** The first cell is `[1, 1, 1]`, not `[0, 0, 0]`.
- **`adaptive` and `adaptive_prior` are the same plugin.** Do not expect them to differ.
- **A large variable can make the universal threshold strict.** If almost nothing is kept, lower `threshold_scale` or use `threshold = "fixed"`.
- **A fixed threshold takes no `threshold_scale`.**
- **Anisotropy widens the reach along `j`.** A ratio of 2 doubles it.
- **`z_range` takes two kinds of value.** Use the string `"all"` or an integer.
- **`import` needs a file.** Set `taper_file`, with the path relative to the folder of the configuration file.
- **Always set `taper` in an entry.** It has no default and the schema does not require it.
- **Required options.** `distance_entries` needs `grid` and `entries`; `distance` needs `grid`, `positions` and `radius`.
- **Unknown options are errors.** All six plugins reject options not listed on this page.
- **Do not set supplied values.** `block_sizes` and `variable_names` come from APEX.
- **Names must match.** Values for `data`, `time`, `parameter` and the keys of `positions` that match nothing in your setup will not localize what you intend. With `distance`, a missing well stops the run unless `unlisted = "zero"`.
- **Do not combine `adaptive_rows` with a localization.** It is not passed one.
- **Default `"*"` overlaps.** An entry without `data`, `time` or `parameter` applies to everything.

## What to read next

- `apex plugins` lists every plugin with its options; add `--json` for full descriptions.
- `apex schema` prints the JSON Schema of a configuration kind. Its argument `kind` is one of `history_match` (the default), `prediction`, `study`, `optimization`, `optimization_study` or `service`:

```bash
uv run apex schema history_match
```

- `apex run` with your configuration file uses the localization you chose. `--max-steps` stops after a given number of analysis steps; `--no-progress` turns off the progress display that is on by default (`--progress`).
- `apex stop` with the run directory of a running run asks it to stop at its next checkpoint; `apex resume` continues it from its last checkpoint:

```bash
uv run apex stop path/to/run_dir
uv run apex resume path/to/run_dir --max-steps 5
```

- `apex predict` with a prediction configuration forecasts the ensembles of a history-matching run beyond history:

```bash
uv run apex predict prediction.toml
```

- The [configuration reference](../../reference/configuration/history_match.md), for the rest of a run configuration.

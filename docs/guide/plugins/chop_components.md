---
audience: public
items:
  - cli:apex plugins
  - cli:apex resume
  - cli:apex run
  - cli:apex stop
  - config:history_match:prior
  - plugin:chop_components:data_weights
  - plugin:chop_components:gain_inflation
  - plugin:chop_components:length_scale
---

# CHOP components

## Goal

By the end of this page you will know what the three plugins in the `chop_components` registry do, how to select one, and how to choose values for its options:

- `data_weights`: per-member weights of the innovations in the gain.
- `gain_inflation`: per-member multiplicative inflation of the gain.
- `length_scale`: P-AutoAdaLoc, per-member Gaspari-Cohn length scales of a correlation-based taper.

Every ensemble member `j` carries its own hyper-parameters: extra numbers that tune the update and are not model parameters. They start as draws from a uniform prior between `low` and `high`. As the run proceeds, a *hyper-parameter update* adjusts them. It works on the scale set by `transform`, and it uses how strongly each hyper-parameter is correlated, across the members, with each datum. For `data_weights` and `gain_inflation`, `threshold` decides which of those correlations count (step 6).

### Symbols used below

| Symbol | Meaning |
|---|---|
| `K` | the gain of the update |
| `r_j` | the innovation of member `j`: the mismatch between the observed data and that member's simulated data |
| `dm_j` | the update applied to member `j` |
| `w_j` | member `j`'s weights (`data_weights`) |
| `D_j` | member `j`'s inflation factors (`gain_inflation`), applied element by element to `K` |
| `o` | element-wise product |
| `N` | the ensemble size |

The plugins act as follows:

- `data_weights`: `dm_j = K diag(w_j) r_j`. Each datum's innovation is scaled by a weight before the gain is applied.
- `gain_inflation`: `dm_j = (D_j o K) r_j`. The gain itself is scaled by member `j`'s factors. The `mode` decides what a factor belongs to: the whole gain (`scalar`), a state variable (`variable`), or a state variable and a data type (`variable_datatype`).
- `length_scale`: a taper multiplies the correlations used in the update (see below).

A *datum* is a single observed value. A *data type* is a group of data of the same kind.

### What localization and the taper are

With a limited number of members, the ensemble shows spurious correlations between a state variable and data that have nothing to do with it. *Localization* damps these by multiplying correlations with a *taper*, a number between 0 and 1 that falls off as the correlation gets weaker. The Gaspari-Cohn function is a standard taper shape. In `length_scale` the taper is computed from correlations, not from physical distance, so a length scale is a plain number in correlation space and has no physical unit. Each member gets its own length scale. A longer length scale lets more correlations through. A shorter one cuts more of them.

## Prerequisites

- APEX installed as described in the README, with commands run from the APEX folder as `uv run apex ...`.
- A run configuration (TOML) whose role takes a plugin from the `chop_components` registry. The configuration reference for that role gives the exact section header.
- A rough idea of the ensemble size `N`, because the `threshold` option depends on it.

## How you select a component

A plugin section has a `name` and, optionally, an `options` table. You choose the component with `name` and give its options in `options`. The valid names are `data_weights`, `gain_inflation` and `length_scale`. The keys of the section look like this:

```text
name = "gain_inflation"
options = { transform = "logit" }
```

Put these keys under the section header that the role's configuration reference shows. Every option below has a default, so a plugin works with only its `name` set. TOML has no null value: to get the default of `lower` or `upper`, leave the option out.

To list every plugin with its options, run:

```
uv run apex plugins
```

For full descriptions as JSON:

```
uv run apex plugins --json
```

## Steps

1. **Pick the plugin** by what you want to vary per member:
   - `data_weights` for how strongly each datum's innovation enters the update.
   - `gain_inflation` for scaling the gain itself.
   - `length_scale` for how far localization reaches.
2. **Pick the `mode`**, which sets how many hyper-parameters each member gets. A finer mode gives more freedom but more hyper-parameters to estimate from `N` members. Order of the modes for each plugin, from coarse to fine:
   - `data_weights`: `datatype` < `datum`
   - `gain_inflation`: `scalar` < `variable` < `variable_datatype`
   - `length_scale`: `scalar` < `data`

   With a small ensemble, start with a coarser mode.
3. **Keep or change the prior range** `low` and `high`. Each member's values are drawn uniformly from it. Widen it to let members differ more; narrow it to keep them close to each other. For `length_scale` the default range is `0.23` to `0.43`. To let more correlations through, shift the range up; to cut more, shift it down.
4. **Choose the `transform`**, the scale on which the hyper-parameter update works:
   - `none`: natural values. Any value below `floor` (default `1e-06`) is raised to `floor`. This is the default and fine for most cases.
   - `log`: the update works on the logarithm of the value, so values stay positive and a step is a multiplicative change. Going from `0.5` to `1.0` then counts the same as going from `1.0` to `2.0`. Choose it for a strictly positive quantity that could differ by an order of magnitude, such as data weights for data of very different quality. It needs `low > 0`.
   - `logit`: the update keeps values inside the interval between `lower` and `upper`. Choose it when values must never leave a known range.
5. **Set `lower` and `upper`** only if you use `logit` and the defaults do not suit you. They must satisfy `lower < low <= high < upper`. By default, with `range = high - low`, `lower = max(0, low - range)` and `upper = high + range`.
6. **Tune `threshold`** (for `data_weights` and `gain_inflation`). The update keeps the correlation between a hyper-parameter and a datum only when `abs(corr) > threshold / sqrt(N)`. Correlations estimated from `N` members carry sampling noise of roughly `1/sqrt(N)`, so the cut-off shrinks as the ensemble grows. A larger `threshold` keeps fewer correlations. A smaller one keeps more, including weak ones that may be noise. For `length_scale`, tune `correlation` and `factor` instead.
7. **Run** as usual, for example:

```
uv run apex run config.toml --max-steps 2
```

   `--max-steps` stops the run after that many analysis steps (an analysis step is one update of the ensemble against the data). The run stops after two steps and can be continued. Progress is shown on standard error by default; `--no-progress` turns it off.

8. **Continue later** by giving `apex resume` the directory of the run:

```
uv run apex resume path/to/run_dir
```

   `apex resume` continues from the last checkpoint and accepts `--max-steps` and `--no-progress` too. To ask a running run to stop at its next checkpoint, use `uv run apex stop path/to/run_dir`; `apex resume` continues it.

## Options of `data_weights`

| Option | Type | Default | Meaning |
|---|---|---|---|
| `mode` | `datatype` or `datum` | `datatype` | One weight per data type, or one per datum, for each member. |
| `low` | number | `0.5` | Lower end of the uniform prior. |
| `high` | number | `1.5` | Upper end of the uniform prior. |
| `transform` | `none`, `log` or `logit` | `none` | Scale of the hyper-parameter update. |
| `floor` | number | `1e-06` | Smallest value kept with `transform = "none"`. |
| `lower` | number | derived from `low` and `high` | `logit` lower bound. |
| `upper` | number | derived from `low` and `high` | `logit` upper bound. |
| `threshold` | number | `3.0` | Cut-off in `abs(corr) > threshold / sqrt(N)`. |

## Options of `gain_inflation`

| Option | Type | Default | Meaning |
|---|---|---|---|
| `mode` | `variable`, `scalar` or `variable_datatype` | `variable` | One factor per state variable, a single factor, or one per state variable and data type. |
| `low` | number | `0.5` | Lower end of the uniform prior. |
| `high` | number | `1.5` | Upper end of the uniform prior. |
| `transform` | `none`, `log` or `logit` | `none` | Scale of the hyper-parameter update. |
| `floor` | number | `1e-06` | Smallest value kept with `transform = "none"`. |
| `lower` | number | derived from `low` and `high` | `logit` lower bound. |
| `upper` | number | derived from `low` and `high` | `logit` upper bound. |
| `threshold` | number | `3.0` | Same correlation cut-off as for `data_weights`. |

## Options of `length_scale`

| Option | Type | Default | Meaning |
|---|---|---|---|
| `mode` | `data` or `scalar` | `data` | One length scale per datum, or one for all data, for each member. |
| `correlation` | `prior` or `current` | `prior` | Taper computed from the initial ensemble's correlations, or from correlations recomputed before each update. |
| `factor` | number | `2.0` | Positive multiplier of the length scales in the taper. The default 2 is the value in the published code of the method; 1 gives the formula as published in the paper. |
| `low` | number | `0.23` | Lower end of the uniform prior. |
| `high` | number | `0.43` | Upper end of the uniform prior. |
| `transform` | `none`, `log` or `logit` | `none` | Scale of the hyper-parameter update. |
| `floor` | number | `1e-06` | Smallest value kept with `transform = "none"`. |
| `lower` | number | derived from `low` and `high` | `logit` lower bound. |
| `upper` | number | derived from `low` and `high` | `logit` upper bound. |

`length_scale` has no `threshold` option. Keep `factor = 2` unless you want the formula as written in the paper, in which case set `factor = 1`. Use `correlation = "current"` if the correlations may change a lot during the run and you want the taper to follow them; the default `prior` keeps the taper tied to the initial ensemble.

## Worked examples

### `gain_inflation` with `logit`

Suppose you choose `gain_inflation` and set `transform = "logit"`, leaving everything else at its default:

```text
name = "gain_inflation"
options = { transform = "logit" }
```

- `mode` is `variable`, so each member gets one inflation factor per state variable.
- The prior is uniform from `low = 0.5` to `high = 1.5`, so the range is 1.
- `lower` and `upper` are left out, so they are derived: `lower = max(0, 0.5 - 1) = 0` and `upper = 1.5 + 1 = 2.5`.
- The condition `lower < low <= high < upper` holds: `0 < 0.5 <= 1.5 < 2.5`. The updated factors therefore stay between 0 and 2.5.
- With `N = 100` members and the default `threshold = 3.0`, the cut-off is `3 / sqrt(100) = 0.3`. A hyper-parameter's correlation with a datum is kept only if its absolute value exceeds 0.3.

To widen or narrow the allowed interval, set `lower` and `upper` yourself, keeping the inequality above.

### `data_weights` with `log`

```text
name = "data_weights"
options = { mode = "datum", transform = "log", low = 0.1, high = 10.0 }
```

- Each member gets one weight per datum, drawn uniformly from 0.1 to 10.
- `low = 0.1` is above 0, so `log` is allowed.
- The update works on the logarithm, so a weight can move from 0.1 to 1 as easily as from 1 to 10, and it can never become negative.
- With `N = 25` members the cut-off is `3 / sqrt(25) = 0.6`. Raise `threshold` for a stricter cut-off, lower it to keep more correlations.

### `length_scale` with a shifted range

```text
name = "length_scale"
options = { mode = "scalar", low = 0.3, high = 0.6 }
```

- Each member gets one length scale for all data, drawn from 0.3 to 0.6 instead of the default 0.23 to 0.43.
- The range sits higher, so the tapers let more correlations through than with the defaults.
- `factor` stays at `2.0` and `correlation` at `prior`.

## Pitfalls

- **`transform = "log"` with `low <= 0`.** `log` needs `low > 0`.
- **Bad `logit` bounds.** `lower` and `upper` must satisfy `lower < low <= high < upper`.
- **Expecting `floor` to act on every transform.** It applies only with `transform = "none"`.
- **Setting `threshold` on `length_scale`.** That plugin has no such option. Only `data_weights` and `gain_inflation` have it.
- **Treating `mode` values as interchangeable.** The valid values differ per plugin: `datatype`/`datum`, `variable`/`scalar`/`variable_datatype`, and `data`/`scalar`.
- **Setting `factor` to zero or a negative number** in `length_scale`. It must be positive.
- **Reading `low` and `high` as hard limits.** They define the uniform prior the members start from. The bounds during the update are set by `transform`, `floor`, `lower` and `upper`.
- **Mixing up the default ranges.** `length_scale` draws from `0.23` to `0.43`, while the other two plugins draw from `0.5` to `1.5`.

## What to read next

- The configuration reference for the role that uses `chop_components`, to find the section where you set `name`.
- The other plugin pages in this guide, for the rest of the plugin registries.

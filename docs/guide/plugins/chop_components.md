---
audience: public
items:
  - cli:apex plugins
  - cli:apex run
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

In each case every ensemble member `j` carries its own hyper-parameters. They are drawn from a uniform prior, and a hyper-parameter update then adjusts them as the run proceeds.

### Symbols used below

| Symbol | Meaning |
|---|---|
| `K` | the gain of the update |
| `r_j` | the innovation of member `j`: the mismatch between the observed data and that member's simulated data |
| `dm_j` | the update applied to member `j` |
| `w_j` | member `j`'s weights (`data_weights`) |
| `D_j` | member `j`'s inflation factors (`gain_inflation`) |
| `o` | element-wise product |

The plugins act as follows:

- `data_weights`: `dm_j = K diag(w_j) r_j`
- `gain_inflation`: `dm_j = (D_j o K) r_j`

A *datum* is a single observed value. A *data type* is a group of data of the same kind.

## Prerequisites

- A working APEX installation, with commands run from the APEX folder as `uv run apex ...`.
- A run configuration (TOML) with a role that takes a plugin from the `chop_components` registry.
- A rough idea of the ensemble size `N`, because the `threshold` option depends on it.

## How you select a component

`chop_components` is a plugin registry, not a configuration section, so do not write a `[chop_components]` section. You choose the plugin with `name = "..."` in the plugin section of the role that uses it, and you give its options there too. The valid names are `data_weights`, `gain_inflation` and `length_scale`. Take the section's name from the configuration reference for that role.

Every option below has a default, so a plugin works with only its `name` set.

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
2. **Pick the `mode`**, which sets how many hyper-parameters each member gets. A finer mode (`datum`, `variable_datatype`, `data`) gives more freedom but more hyper-parameters to estimate from `N` members. A coarser mode (`datatype`, `scalar`) gives fewer. With a small ensemble, start with the coarser mode.
3. **Keep or change the prior range** `low` and `high`. Each member's values are drawn uniformly from it.
4. **Choose the `transform`**, the scale on which the hyper-parameter update works:
   - `none`: natural values, floored at `floor` (default `1e-06`). A reasonable default.
   - `log`: suits strictly positive quantities; needs `low > 0`.
   - `logit`: suits values that must stay inside an interval, between `lower` and `upper`.
5. **Set `lower` and `upper`** only if you use `logit` and the defaults do not suit you. They must satisfy `lower < low <= high < upper`. By default, with `range = high - low`, `lower = max(0, low - range)` and `upper = high + range`. The clamp at 0 keeps the lower bound non-negative.
6. **Tune `threshold`** (for `data_weights` and `gain_inflation`). The update keeps a hyper-parameter's correlation with a datum only when `abs(corr) > threshold / sqrt(N)`. A larger `threshold` keeps fewer correlations. A smaller one keeps more, including weak ones that may be sampling noise. For `length_scale`, tune `correlation` and `factor` instead.
7. **Run** as usual, for example:

```
uv run apex run config.toml --max-steps 2
```

   `--max-steps` stops after that many analysis steps, and you continue later with `apex resume`. Progress is shown on standard error by default; `--no-progress` turns it off.

## Options of `data_weights`

| Option | Type | Default | Meaning |
|---|---|---|---|
| `mode` | `datatype` or `datum` | `datatype` | One weight per data type, or one per datum, for each member. |
| `low` | number | `0.5` | Lower end of the uniform prior. |
| `high` | number | `1.5` | Upper end of the uniform prior. |
| `transform` | `none`, `log` or `logit` | `none` | Scale of the hyper-parameter update. |
| `floor` | number | `1e-06` | Smallest value kept with `transform = "none"`. |
| `lower` | number or null | null | `logit` lower bound. |
| `upper` | number or null | null | `logit` upper bound. |
| `threshold` | number | `3.0` | The update keeps a hyper-parameter's correlation with a datum only when `abs(corr) > threshold / sqrt(N)`. |

## Options of `gain_inflation`

| Option | Type | Default | Meaning |
|---|---|---|---|
| `mode` | `variable`, `scalar` or `variable_datatype` | `variable` | One factor per state variable, a single factor, or one per state variable and data type. |
| `low` | number | `0.5` | Lower end of the uniform prior. |
| `high` | number | `1.5` | Upper end of the uniform prior. |
| `transform` | `none`, `log` or `logit` | `none` | Scale of the hyper-parameter update. |
| `floor` | number | `1e-06` | Smallest value kept with `transform = "none"`. |
| `lower` | number or null | null | `logit` lower bound. |
| `upper` | number or null | null | `logit` upper bound. |
| `threshold` | number | `3.0` | Same correlation cut-off as for `data_weights`. |

## Options of `length_scale`

| Option | Type | Default | Meaning |
|---|---|---|---|
| `mode` | `data` or `scalar` | `data` | One length scale per datum, or one for all data, for each member. |
| `correlation` | `prior` or `current` | `prior` | Taper computed from the initial ensemble's correlations, or from correlations recomputed before each update. |
| `factor` | number | `2.0` | Multiplier of the length scales in the taper. It must be positive. The default 2 is the published code's value; 1 gives the paper's formula. |
| `low` | number | `0.23` | Lower end of the uniform prior. |
| `high` | number | `0.43` | Upper end of the uniform prior. |
| `transform` | `none`, `log` or `logit` | `none` | Scale of the hyper-parameter update. |
| `floor` | number | `1e-06` | Smallest value kept with `transform = "none"`. |
| `lower` | number or null | null | `logit` lower bound. |
| `upper` | number or null | null | `logit` upper bound. |

`length_scale` has no `threshold` option.

Keep `factor = 2` to match the published code. Set `factor = 1` if you want the paper's formula.

## Worked example: `gain_inflation` with `logit`

Suppose you choose `gain_inflation` and set `transform = "logit"`, leaving everything else at its default.

- `mode` is `variable`, so each member gets one inflation factor per state variable.
- The prior is uniform on `low = 0.5` to `high = 1.5`, so `range = 1`.
- `lower` and `upper` are null, so they are derived: `lower = max(0, 0.5 - 1) = 0` and `upper = 1.5 + 1 = 2.5`.
- The condition `lower < low <= high < upper` holds: `0 < 0.5 <= 1.5 < 2.5`. The updated factors therefore stay between 0 and 2.5.
- With `N = 100` members and the default `threshold = 3.0`, the cut-off is `3 / sqrt(100) = 0.3`. A hyper-parameter's correlation with a datum is kept only if its absolute value exceeds 0.3.

To widen or narrow the allowed interval, set `lower` and `upper` yourself, keeping the inequality above.

## Pitfalls

- **Writing a `[chop_components]` section.** It is a registry, not a section. Select the plugin with `name` in its role's section.
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

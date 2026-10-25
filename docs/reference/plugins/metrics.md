---
audience: public
generated: reference
---

# Metrics: reference

Every one of these 8 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `constraint_cost`

Each member's `D(0 - c(m_j))` and violated inequality rows.

Select it with:

```toml
[[metrics]]
name = "constraint_cost"
```

| option | type | default | description |
|---|---|---|---|
| `function` | object | required | `{name, options}` of `gmac_constraints` (e.g. `box`, `histogram`). |
| `metric` | object or null | | `{name, options}` of `gmac_constraint_metrics`; default `log_barrier` for inequalities, `log_channel` for equalities. |

## `facies_brier`

Brier score of the ensemble's facies probabilities (Brier, 1950).

Select it with:

```toml
[[metrics]]
name = "facies_brier"
```

| option | type | default | description |
|---|---|---|---|
| `actnum` | ArrayLike or null | | Active-cell indicator (non-zero = active); default all cells. Only active cells are scored, for variables with as many values. Configurations give `actnum_file`. |
| `variables` | array or null | | Facies variables to score (integer codes from 0); default `facies`. |

## `facies_log_score`

Logarithmic score of the ensemble's facies probabilities (Good, 1952).

Select it with:

```toml
[[metrics]]
name = "facies_log_score"
```

| option | type | default | description |
|---|---|---|---|
| `actnum` | ArrayLike or null | | Active-cell indicator (non-zero = active); default all cells. Only active cells are scored, for variables with as many values. Configurations give `actnum_file`. |
| `variables` | array or null | | Facies variables to score (integer codes from 0); default `facies`. |

## `facies_mismatch`

Fraction of cells whose facies differs from the true model's.

Select it with:

```toml
[[metrics]]
name = "facies_mismatch"
```

| option | type | default | description |
|---|---|---|---|
| `actnum` | ArrayLike or null | | Active-cell indicator (non-zero = active); default all cells. Configurations give `actnum_file`. |
| `variables` | array or null | | Facies properties to score; default `["facies"]`. |

## `model_change`

Change of each member's parameters since the ensemble it was updated from.

Select it with:

```toml
[[metrics]]
name = "model_change"
```

| option | type | default | description |
|---|---|---|---|
| `actnum` | ArrayLike or null | | Active-cell indicator (non-zero = active), as for `ModelRMSE`. |
| `variables` | array or null | | State variables to score; default all. |

## `normalized_data_mismatch`

Normalized data mismatch `mean(((d_obs - y_j) / sigma)^2)` against the observed values.

Select it with:

```toml
[[metrics]]
name = "normalized_data_mismatch"
```


## `rmse`

RMSE of each member's parameters against the true model.

Select it with:

```toml
[[metrics]]
name = "rmse"
```

| option | type | default | description |
|---|---|---|---|
| `actnum` | ArrayLike or null | | Active-cell indicator (non-zero = active); default all cells. It applies to the variables with as many values (e.g. the grid fields, not scalar parameters); named `variables` must all match it. Configurations give `actnum_file`. |
| `variables` | array or null | | State variables to score; default all. |

## `ssim`

Structural similarity of each member's fields to the true model's.

Select it with:

```toml
[[metrics]]
name = "ssim"
```

| option | type | default | description |
|---|---|---|---|
| `actnum` | ArrayLike or null | | Active-cell indicator over the grid; inactive cells are filled with the active mean. Configurations give `actnum_file`. |
| `data_range` | number or null | | Fixed value range; default the range of the truth and the ensemble together. |
| `grid` | array or null | | Field shape `[nx, ny, nz]` (or 1-2 dimensions), including inactive cells; default each variable's own `grid` (`[[state.variables]]`). |
| `order` | string | F | Cell order of the state vectors, `"F"` (Eclipse, x fastest; default) or `"C"`. |
| `per_layer` | boolean | false | Average 2-D SSIMs over the layers (last grid axis) instead of one 3-D SSIM. |
| `variables` | array or null | | State variables to score; default every variable with a grid (or, with `grid`, every variable with `prod(grid)` values). |
| `win_size` | integer | 7 | Odd SSIM window (>= 3), at most every (per-layer: every horizontal) grid dimension. |

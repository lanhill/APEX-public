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
| `function` | object | required | |
| `metric` | object or null | | |

## `facies_brier`

Brier score of the ensemble's facies probabilities (Brier, 1950).

Select it with:

```toml
[[metrics]]
name = "facies_brier"
```

| option | type | default | description |
|---|---|---|---|
| `actnum` | ArrayLike or null | | |
| `variables` | array or null | | |

## `facies_log_score`

Logarithmic score of the ensemble's facies probabilities (Good, 1952).

Select it with:

```toml
[[metrics]]
name = "facies_log_score"
```

| option | type | default | description |
|---|---|---|---|
| `actnum` | ArrayLike or null | | |
| `variables` | array or null | | |

## `facies_mismatch`

Fraction of cells whose facies differs from the true model's.

Select it with:

```toml
[[metrics]]
name = "facies_mismatch"
```

| option | type | default | description |
|---|---|---|---|
| `actnum` | ArrayLike or null | | |
| `variables` | array or null | | |

## `model_change`

Change of each member's parameters since the ensemble it was updated from.

Select it with:

```toml
[[metrics]]
name = "model_change"
```

| option | type | default | description |
|---|---|---|---|
| `actnum` | ArrayLike or null | | |
| `variables` | array or null | | |

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
| `actnum` | ArrayLike or null | | |
| `variables` | array or null | | |

## `ssim`

Structural similarity of each member's fields to the true model's.

Select it with:

```toml
[[metrics]]
name = "ssim"
```

| option | type | default | description |
|---|---|---|---|
| `actnum` | ArrayLike or null | | |
| `data_range` | number or null | | |
| `grid` | array or null | | |
| `order` | string | F | |
| `per_layer` | boolean | false | |
| `variables` | array or null | | |
| `win_size` | integer | 7 | |

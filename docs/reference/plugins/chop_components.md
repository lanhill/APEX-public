---
audience: public
generated: reference
---

# CHOP components: reference

Every one of these 3 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `data_weights`

Per-member weights of the innovations in the gain: `dm_j = K diag(w_j) r_j`.

| option | type | default | description |
|---|---|---|---|
| `floor` | number | 1e-06 | The smallest value kept with `transform = "none"` (default 1e-6). |
| `high` | number | 1.5 | The range of the uniform prior each member's weights is drawn from (defaults 0.5 and 1.5). |
| `low` | number | 0.5 | The range of the uniform prior each member's weights is drawn from (defaults 0.5 and 1.5). |
| `lower` | number or null | | `logit`'s bounds, with `lower < low <= high < upper`; by default `low` and `high` widened by their range, `lower` never below 0. |
| `mode` | string | datatype | `datatype` (default) gives each member one weight per data type, `datum` one per datum. |
| `threshold` | number | 3.0 | The hyper-update keeps a hyper-parameter's correlation with a datum only when `abs(corr) > threshold / sqrt(N)`, `N` the ensemble size (default 3). |
| `transform` | string | none | The scale the hyper-parameter update works on: `none` (natural values, floored at `floor`), `log` (needs `low > 0`) or `logit` (kept within `lower` and `upper`). |
| `upper` | number or null | | `logit`'s bounds, with `lower < low <= high < upper`; by default `low` and `high` widened by their range, `lower` never below 0. |

## `gain_inflation`

Per-member multiplicative inflation of the gain: `dm_j = (D_j o K) r_j`.

| option | type | default | description |
|---|---|---|---|
| `floor` | number | 1e-06 | The smallest value kept with `transform = "none"` (default 1e-6). |
| `high` | number | 1.5 | The range of the uniform prior each member's inflation factors is drawn from (defaults 0.5 and 1.5). |
| `low` | number | 0.5 | The range of the uniform prior each member's inflation factors is drawn from (defaults 0.5 and 1.5). |
| `lower` | number or null | | `logit`'s bounds, with `lower < low <= high < upper`; by default `low` and `high` widened by their range, `lower` never below 0. |
| `mode` | string | variable | `variable` (default) gives each member one inflation factor per state variable, `scalar` one factor, `variable_datatype` one per state variable and data type. |
| `threshold` | number | 3.0 | The hyper-update keeps a hyper-parameter's correlation with a datum only when `abs(corr) > threshold / sqrt(N)`, `N` the ensemble size (default 3). |
| `transform` | string | none | The scale the hyper-parameter update works on: `none` (natural values, floored at `floor`), `log` (needs `low > 0`) or `logit` (kept within `lower` and `upper`). |
| `upper` | number or null | | `logit`'s bounds, with `lower < low <= high < upper`; by default `low` and `high` widened by their range, `lower` never below 0. |

## `length_scale`

P-AutoAdaLoc: per-member Gaspari-Cohn length scales of a correlation-based taper.

| option | type | default | description |
|---|---|---|---|
| `correlation` | string | prior | The correlations the taper is computed from: `prior` (default) those of the initial ensemble, `current` those recomputed before each update. |
| `factor` | number | 2.0 | The multiplier of the length scales in the taper (positive; default 2, the published code's; 1 is the paper's formula). |
| `floor` | number | 1e-06 | The smallest value kept with `transform = "none"` (default 1e-6). |
| `high` | number | 0.43 | The range of the uniform prior each member's length scales is drawn from (defaults 0.23 and 0.43). |
| `low` | number | 0.23 | The range of the uniform prior each member's length scales is drawn from (defaults 0.23 and 0.43). |
| `lower` | number or null | | `logit`'s bounds, with `lower < low <= high < upper`; by default `low` and `high` widened by their range, `lower` never below 0. |
| `mode` | string | data | `data` (default) gives each member one length scale per datum, `scalar` one length scale for all data. |
| `transform` | string | none | The scale the hyper-parameter update works on: `none` (natural values, floored at `floor`), `log` (needs `low > 0`) or `logit` (kept within `lower` and `upper`). |
| `upper` | number or null | | `logit`'s bounds, with `lower < low <= high < upper`; by default `low` and `high` widened by their range, `lower` never below 0. |

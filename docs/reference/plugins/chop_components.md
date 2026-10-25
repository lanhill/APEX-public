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
| `floor` | number | 1e-06 | |
| `high` | number | 1.5 | |
| `low` | number | 0.5 | |
| `lower` | number or null | | |
| `mode` | string | datatype | |
| `threshold` | number | 3.0 | |
| `transform` | string | none | |
| `upper` | number or null | | |

## `gain_inflation`

Per-member multiplicative inflation of the gain: `dm_j = (D_j o K) r_j`.

| option | type | default | description |
|---|---|---|---|
| `floor` | number | 1e-06 | |
| `high` | number | 1.5 | |
| `low` | number | 0.5 | |
| `lower` | number or null | | |
| `mode` | string | variable | |
| `threshold` | number | 3.0 | |
| `transform` | string | none | |
| `upper` | number or null | | |

## `length_scale`

P-AutoAdaLoc: per-member Gaspari-Cohn length scales of a correlation-based taper.

| option | type | default | description |
|---|---|---|---|
| `correlation` | string | prior | |
| `factor` | number | 2.0 | |
| `floor` | number | 1e-06 | |
| `high` | number | 0.43 | |
| `low` | number | 0.23 | |
| `lower` | number or null | | |
| `mode` | string | data | |
| `transform` | string | none | |
| `upper` | number or null | | |

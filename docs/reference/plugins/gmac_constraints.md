---
audience: public
generated: reference
---

# Gmac constraints: reference

Every one of these 4 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `box`

Box constraints `lower <= x <= upper` per variable: `h = [lower - x; x - upper]`.

| option | type | default | description |
|---|---|---|---|
| `bounds` | object | required | |

## `histogram`

The histogram of a variable equals a reference one: `counts(x) - counts_ref = 0`.

| option | type | default | description |
|---|---|---|---|
| `bins` | integer | 50 | |
| `counts` | array or null | | |
| `range` | array | required | |
| `reference_file` | string or string or null | | |
| `variable` | string | required | |

## `linear`

A linear system `A x - b` (equality `= 0` or inequality `<= 0`).

| option | type | default | description |
|---|---|---|---|
| `kind` | string | equality | |
| `matrix_file` | string or string | required | |
| `variables` | array or null | | |

## `tv`

The total variation of gridded variables vanishes: `f = TV(x) = 0`.

| option | type | default | description |
|---|---|---|---|
| `variables` | array or null | | |

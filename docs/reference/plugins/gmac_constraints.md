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
| `bounds` | object | required | `{variable: [lower, upper]}` (state variables, or properties with a parameterization). |

## `histogram`

The histogram of a variable equals a reference one: `counts(x) - counts_ref = 0`.

| option | type | default | description |
|---|---|---|---|
| `bins` | integer | 50 | Number of bins (MATLAB: 50). |
| `counts` | array or null | | Or the target counts per bin. |
| `range` | array | required | `[low, high]` of the bins. |
| `reference_file` | string or string or null | | An `.npy`, or an `.npz` with the variable's array (e.g. the truth), whose histogram is the target. |
| `variable` | string | required | The variable (state, or property with a parameterization). |

## `linear`

A linear system `A x - b` (equality `= 0` or inequality `<= 0`).

| option | type | default | description |
|---|---|---|---|
| `kind` | string | equality | `equality` or `inequality`. |
| `matrix_file` | string or string | required | An `.npz` with `matrix` `(n_c, n)` over the stacked `variables` and, optionally, `rhs` `(n_c,)` (default zeros). |
| `variables` | array or null | | The variables `A` acts on, stacked in this order; default all. |

## `tv`

The total variation of gridded variables vanishes: `f = TV(x) = 0`.

| option | type | default | description |
|---|---|---|---|
| `variables` | array or null | | The gridded variables; default every one with a grid. |

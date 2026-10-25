---
audience: public
generated: reference
---

# Gmac constraint metrics: reference

Every one of these 4 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `l1sq`

`D(x) = ||x||_1^2`: gradient `2 ||x||_1 sgn(x)`, Hessian `2 sgn(x) sgn(x)^T`.


## `log_barrier`

The barrier `D(x) = -sum log(x + a)` of inequalities, `x = -h >= 0`.

| option | type | default | description |
|---|---|---|---|
| `a` | number | 0.1 | The offset (the paper: 0.1). |

## `log_channel`

The channel `D(x) = sum log(|x| + b)` of equalities.

| option | type | default | description |
|---|---|---|---|
| `b` | number | 0.1 | The offsets (the paper's scripts: 0.1, 0.001; `epsilon` is used by `paper`). |
| `curvature` | string | absolute_hessian | The metric: `absolute_hessian` (default), the Hessian's absolute value with a positive weight; `absolute_trace`, the negative Hessian with a positive weight (it may become indefinite); or `paper`, the negative Hessian with the paper's negative trace weight. |
| `epsilon` | number | 0.001 | The offsets (the paper's scripts: 0.1, 0.001; `epsilon` is used by `paper`). |

## `quadratic`

`D(x) = 1/2 ||x||^2`: gradient `x`, Hessian `I` (a perfect measurement).

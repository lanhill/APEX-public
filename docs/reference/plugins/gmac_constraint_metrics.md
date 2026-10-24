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
| `a` | number | 0.1 | |

## `log_channel`

The channel `D(x) = sum log(|x| + b)` of equalities.

| option | type | default | description |
|---|---|---|---|
| `b` | number | 0.1 | |
| `curvature` | string | absolute_hessian | |
| `epsilon` | number | 0.001 | |

## `quadratic`

`D(x) = 1/2 ||x||^2`: gradient `x`, Hessian `I` (a perfect measurement).

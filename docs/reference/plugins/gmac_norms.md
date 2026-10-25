---
audience: public
generated: reference
---

# Gmac norms: reference

Every one of these 3 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `l1sq`

`‖x‖₁²`: Hessian `2 sgn(x) sgn(x)ᵀ`, rank one, so `M_j = 2 v_j v_jᵀ`.


## `l2sq`

`‖x‖₂²`: Hessian `2I`, so the metric `2AᵀA` is the same for every member.


## `lpq`

`‖x‖_p^q` with its exact Hessian (the manuscript's App. C).

| option | type | default | description |
|---|---|---|---|
| `floor` | number | 1e-06 | For `1 < p < 2`, <code>&#124;x&#124;^{p-2}</code> is taken at <code>max(&#124;x&#124;, floor · max&#124;x&#124;)</code> (it is infinite at 0). |
| `p` | number | 2.0 | At least 1. `p = q = 1` (`‖x‖₁`) is refused: its Hessian is 0 almost everywhere, so it adds nothing. |
| `q` | number | 2.0 | At least 1. `p = q = 1` (`‖x‖₁`) is refused: its Hessian is 0 almost everywhere, so it adds nothing. |

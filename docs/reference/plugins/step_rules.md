---
audience: public
generated: reference
---

# Step rules: reference

Every one of these 4 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `adam`

Adam: bias-corrected first and second moments of the gradient.

| option | type | default | description |
|---|---|---|---|
| `beta2` | number | 0.999 | `b2`. |
| `momentum` | number | 0.9 | `b1`. PET's EnOpt and GenOpt pass their momentum `beta` here (default 0). |
| `step_size` | number | 0.1 | `alpha`. |

## `adamax`

AdaMax: Adam with the infinity norm for the second moment.

| option | type | default | description |
|---|---|---|---|
| `beta2` | number | 0.999 | `b2`, the decay of the running maximum of `\|g\|`, in `[0, 1)` (default 0.999). |
| `momentum` | number | 0.9 | `b1`, the decay of the gradient's running mean, in `[0, 1)` (0 uses the current gradient alone). Set from the optimizer's momentum `beta`, as PET's EnOpt and GenOpt do (EnOpt's default 0). |
| `step_size` | number | 0.1 | `alpha`, the base step length in optimizer coordinates (default 0.1); each control moves by at most about this much per step, and backtracking shrinks it for a trial. |

## `gd`

Gradient descent with momentum: `v = beta v_old - alpha g`, `x_new = x + v`.

| option | type | default | description |
|---|---|---|---|
| `momentum` | number | 0.0 | `beta`; backtracking shrinks it with the step size. |
| `step_size` | number | 0.1 | `alpha`. |

## `steihaug`

CG-Steihaug: the truncated conjugate-gradient trust-region step.

| option | type | default | description |
|---|---|---|---|
| `delta_max` | number | 100000.0 | CG iterations, relative residual tolerance, largest radius. |
| `epsilon` | number | 1e-08 | CG iterations, relative residual tolerance, largest radius. |
| `maxiter` | number | 1000000.0 | CG iterations, relative residual tolerance, largest radius. |
| `momentum` | number | 0.0 | Unused (the common factory signature of the step rules). |
| `step_size` | number | 1.0 | Initial trust radius `delta0`. PET's EnOpt builds it with 3.0, whatever its `alpha`. |

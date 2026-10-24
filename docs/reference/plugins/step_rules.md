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
| `beta2` | number | 0.999 | |
| `momentum` | number | 0.9 | |
| `step_size` | number | 0.1 | |

## `adamax`

AdaMax: Adam with the infinity norm for the second moment.

| option | type | default | description |
|---|---|---|---|
| `beta2` | number | 0.999 | |
| `momentum` | number | 0.9 | |
| `step_size` | number | 0.1 | |

## `gd`

Gradient descent with momentum: `v = beta v_old - alpha g`, `x_new = x + v`.

| option | type | default | description |
|---|---|---|---|
| `momentum` | number | 0.0 | |
| `step_size` | number | 0.1 | |

## `steihaug`

CG-Steihaug: the truncated conjugate-gradient trust-region step.

| option | type | default | description |
|---|---|---|---|
| `delta_max` | number | 100000.0 | |
| `epsilon` | number | 1e-08 | |
| `maxiter` | number | 1000000.0 | |
| `momentum` | number | 0.0 | |
| `step_size` | number | 1.0 | |

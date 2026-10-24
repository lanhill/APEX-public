---
audience: public
generated: reference
---

# Methods: reference

Every one of these 7 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `chop`

Estimate a wrapped method's localization and gain hyper-parameters per member as it runs.

Select it with:

```toml
[method]
name = "chop"
```

| option | type | default | description |
|---|---|---|---|
| `components` | array | [{"name": "length_scale", "mode": "data"}] | |
| `hyper_localization` | string or object | same | |
| `inner` | string | gies | The wrapped method's name (its options follow). |
| `max_batch_mb` | number | 256.0 | |

## `es`

Ensemble smoother: one update with all the data, ES-MDA's single step at `alpha = 1`.

Select it with:

```toml
[method]
name = "es"
```


## `esmda`

ES-MDA with a fixed schedule of covariance inflation factors `alpha_i`.

Select it with:

```toml
[method]
name = "esmda"
```

| option | type | default | description |
|---|---|---|---|
| `inflation` | array | required | |

## `gies`

GIES iterative smoother: fixed perturbed observations, damped strategy updates.

Select it with:

```toml
[method]
name = "gies"
```

| option | type | default | description |
|---|---|---|---|
| `max_trials_per_iteration` | integer or null | | |

## `gn_enrml`

GN-EnRML: an iterative smoother with Gauss-Newton steps scaled by a step length.

Select it with:

```toml
[method]
name = "gn_enrml"
```

| option | type | default | description |
|---|---|---|---|
| `max_inner_iterations` | integer | 10 | |

## `lm_enrml`

LM-EnRML: an iterative smoother with Levenberg-Marquardt damping of its steps.

Select it with:

```toml
[method]
name = "lm_enrml"
```

| option | type | default | description |
|---|---|---|---|
| `max_inner_iterations` | integer | 10 | |

## `multilevel_esmda`

Multilevel ES-MDA: ES-MDA's schedule on an ensemble split over fidelity levels.

Select it with:

```toml
[method]
name = "multilevel_esmda"
```

| option | type | default | description |
|---|---|---|---|
| `inflation` | array | required | |
| `levels` | array | required | |

**`Level`**: One fidelity level: its members and weight, and what its simulator runs.

Levels are listed coarsest first and hold consecutive member ids; their members add up
to `[prior] ensemble_size`. A level with no simulator key runs the fine model.

| option | type | default | description |
|---|---|---|---|
| `deck_template_file` | string | | A deck template of the level's own (a coarse model). |
| `forward_file` | string | | The linear simulator's G of the level (an .npz). |
| `matrix` | array | | The linear simulator's G, inline. |
| `members` | integer | required | The level's number of members N_l (at least 2). |
| `upscale` | object | | The simulator's grid coarsening (method, dim, max_trunc, ...). |
| `weight` | number | required | The level's weight w_l in the hybrid gain (non-negative; the weights are normalised to sum 1, as PET's). |

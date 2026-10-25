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
| `components` | array | [{"name": "length_scale", "mode": "data"}] | The estimated components: `[{name = "length_scale", mode = "data", ...}, ...]` (registry `chop_components`); default the paper's, one length scale per datum. |
| `hyper_localization` | string or object | same | How the hyper-parameter update is localized: `"same"` (each component's own taper, the default), `"none"`, or a registered localization's name (or `{name = ..., options}`) applied to the hyper-parameter block. |
| `inner` | string | gies | The wrapped method's name (its options follow). |
| `max_batch_mb` | number | 256.0 | The memory of one row block of the model update's per-member tapers. |

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
| `inflation` | array | required | Inflation factors, one per step (PET `mda.inflation_param`). |

## `gies`

GIES iterative smoother: fixed perturbed observations, damped strategy updates.

Select it with:

```toml
[method]
name = "gies"
```

| option | type | default | description |
|---|---|---|---|
| `max_trials_per_iteration` | integer or null | | Retry a rejected trial within the same iteration, from the same accepted iterate with the increased damping, up to this many trials. An accepted trial ends the iteration. When every trial is rejected, the damping is multiplied by `factor` once more, raised to at least a numeric `schedule.initial` (no floor for an `"auto"` start, which only the `lm` scheme has), and the iteration ends without an update. These are the inner retries of the author's MATLAB RLM-MAC (MATLAB `RLM_average_cost.m`: `maxInnerIter = 5` gives at most 4 trials). Unset (the default), every trial is an iteration, as in . Every trial keeps its own number in the results (`iter_<k>`/`trial_<k>`) either way. The convergence test is the schedule's in both modes: a rejected trial whose data mismatch changed by less than `data_mismatch_tolerance` stops the run only with `converge_on_rejection` (); by default it is rejected and retried with more damping, as the MATLAB RLM-MAC, which tests the minimum reduction on accepted steps only. An exhausted iteration adds no record; its extra damping shows in the next trial's record. |

## `gn_enrml`

GN-EnRML: an iterative smoother with Gauss-Newton steps scaled by a step length.

Select it with:

```toml
[method]
name = "gn_enrml"
```

| option | type | default | description |
|---|---|---|---|
| `max_inner_iterations` | integer | 10 | Rejected trials in a row after which the run stops (`max_inner_iter`, 10). |

## `lm_enrml`

LM-EnRML: an iterative smoother with Levenberg-Marquardt damping of its steps.

Select it with:

```toml
[method]
name = "lm_enrml"
```

| option | type | default | description |
|---|---|---|---|
| `max_inner_iterations` | integer | 10 | Rejected trials in a row after which the run stops (`max_inner_iter`, default 10); with the `lm` scheme `maximum` usually stops it first (`rlmmac` has none: then the tolerance, the iteration budget or these rejections do). |

## `multilevel_esmda`

Multilevel ES-MDA: ES-MDA's schedule on an ensemble split over fidelity levels.

Select it with:

```toml
[method]
name = "multilevel_esmda"
```

| option | type | default | description |
|---|---|---|---|
| `inflation` | array | required | The ES-MDA inflation factors `alpha_i` (sum of `1 / alpha_i` equal to 1). |
| `levels` | array | required | One table per level (`Level`), coarsest first as PET's: `members` (`N_l`) and `weight` (`w_l`; normalised to sum 1, as PET). Other keys (`upscale`, `deck_template_file`) are for the simulator (the composition passes them). |

**`Level`**: One fidelity level: its members and weight, and what its simulator runs. Levels are listed coarsest first and hold consecutive member ids; their members add up to `[prior] ensemble_size`. A level with no simulator key runs the fine model.

| option | type | default | description |
|---|---|---|---|
| `deck_template_file` | string | | A deck template of the level's own (a coarse model). |
| `forward_file` | string | | The linear simulator's G of the level (an .npz). |
| `matrix` | array | | The linear simulator's G, inline. |
| `members` | integer | required | The level's number of members N_l (at least 2). |
| `upscale` | object | | The simulator's grid coarsening (method, dim, max_trunc, ...). |
| `weight` | number | required | The level's weight w_l in the hybrid gain (non-negative; the weights are normalised to sum 1, as PET's). |

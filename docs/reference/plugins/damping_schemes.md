---
audience: public
generated: reference
---

# Damping schemes: reference

Every one of these 2 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `lm`

LM-EnRML's damping scheme, `damping_schemes["lm"]` (PET `[dataassim.iteration]`).

Select it with:

```toml
[method.damping]
name = "lm"
```

| option | type | default | description |
|---|---|---|---|
| `confirm_convergence` | boolean | true | A small improvement stops the run only when `lambda` is at its `minimum` or the previous accepted trial was small too (default true; otherwise the trial is accepted and `lambda` relaxed). False stops at the first small improvement. |
| `converge_on_rejection` | boolean | false | Also stop on a worse trial within the tolerance (default false: it is rejected and `lambda` raised). |
| `data_mismatch_tolerance` | number | 0.01 | The run has converged when an improving trial changes the mean data mismatch by a relative amount below this (positive, default 0.01). |
| `factor` | number | 5.0 | Divide `lambda` by this after an accepted trial whose data mismatch spread also fell, and multiply by it after a rejected one (greater than 1, default 5). |
| `initial` | number or string | 100.0 | The damping `lambda` of the first analysis (positive, default 100), or `"auto"` for half the prior's mean data mismatch per datum. |
| `max_stalled_rejections` | integer | 3 | Stop after this many consecutive rejected trials within the tolerance (at least 1, default 3). |
| `maximum` | number | 10000000000.0 | Stop once a trial's `lambda` reaches this (above `minimum`, default 1e10). |
| `minimum` | number | 0.01 | `lambda` is not reduced once it is at or below this (default 0.01). |

## `rlmmac`

GIES's (RLM-MAC's) damping scheme, `damping_schemes["rlmmac"]` (PET `iteration`).

Select it with:

```toml
[method.damping]
name = "rlmmac"
```

| option | type | default | description |
|---|---|---|---|
| `confirm_convergence` | boolean | false | A small improvement stops the run only when the damping is at its `minimum` or the previous accepted trial was small too (default false: the first small improvement stops it). Set true for a run that seems to stop too early. |
| `converge_on_rejection` | boolean | false | Also stop on a worse trial within the tolerance (default false: it is rejected and the damping raised). |
| `data_mismatch_tolerance` | number | 0.01 | The run has converged when an improving trial changes the mean data mismatch by a relative amount below this (positive, default 0.01). |
| `factor` | number | 2.0 | Multiply the damping by this after a rejected trial (greater than 1, default 2). |
| `initial` | number | 1.0 | The damping of the first analysis (default 1); must be positive. |
| `max_stalled_rejections` | integer | 3 | Stop after this many consecutive rejected trials within the tolerance (at least 1, default 3). |
| `minimum` | number or null | | The damping is not reduced once it is at or below this (unset by default: no floor). |
| `reduction` | number or null | 0.9 | Multiply the damping by this after an accepted trial, in `(0, 1]` (default 0.9); leaving it unset divides by `factor` instead. |

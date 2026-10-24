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
| `confirm_convergence` | boolean | true | |
| `converge_on_rejection` | boolean | false | |
| `data_mismatch_tolerance` | number | 0.01 | |
| `factor` | number | 5.0 | |
| `initial` | number or string | 100.0 | |
| `max_stalled_rejections` | integer | 3 | |
| `maximum` | number | 10000000000.0 | |
| `minimum` | number | 0.01 | |

## `rlmmac`

GIES's (RLM-MAC's) damping scheme, `damping_schemes["rlmmac"]` (PET `iteration`).

Select it with:

```toml
[method.damping]
name = "rlmmac"
```

| option | type | default | description |
|---|---|---|---|
| `confirm_convergence` | boolean | false | |
| `converge_on_rejection` | boolean | false | |
| `data_mismatch_tolerance` | number | 0.01 | |
| `factor` | number | 2.0 | |
| `initial` | number | 1.0 | |
| `max_stalled_rejections` | integer | 3 | |
| `minimum` | number or null | | |
| `reduction` | number or null | 0.9 | |

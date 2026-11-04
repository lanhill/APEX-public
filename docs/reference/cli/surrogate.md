---
audience: public
generated: reference
---

# The apex surrogate commands: reference

Every one of these 2 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `apex surrogate build`

Build a surrogate: full runs of a design, then the fit, its domain and its gate.

```bash
uv run apex surrogate build CONFIG
```

| option | default | description |
|---|---|---|
| `--resume` | | Continue the build in this directory. |
| `--max-runs` | | Run at most this many full runs now (resume continues). |
| `--progress` (on) / `--no-progress` (off) | true | Show progress on standard error (bars, or timestamped lines). |

## `apex surrogate check`

Gate an existing surrogate on new full runs; the model file never changes.

```bash
uv run apex surrogate check MODEL
```

| option | default | description |
|---|---|---|
| `--against` | | A history-matching configuration (draws from its prior) or run directory (members of its posterior). |
| `--runs` | 10 | How many new full runs. |
| `--seed` | 0 | Seed of the members drawn or chosen. |
| `--iteration` | | A run's iterate instead of its posterior. |
| `--min-success-fraction` | 0.9 | Fail unless this share of the runs succeeds. |
| `--min-inside` | 0.8 | Fail unless this share of the successful runs is inside the model's domain. |
| `--max-runs` | | Run at most this many full runs now (resume continues). |
| `--progress` (on) / `--no-progress` (off) | true | Show progress on standard error (bars, or timestamped lines). |

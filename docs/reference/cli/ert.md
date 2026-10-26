---
audience: public
generated: reference
---

# The apex ert commands: reference

Every one of these 6 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `apex ert compare`

Compare APEX's method with ERT's on one case, over seeds: a study.

```bash
uv run apex ert compare CONFIG
```

| option | default | description |
|---|---|---|
| `--mode` | es_mda | ERT's run model: es_mda or ensemble_smoother. |
| `--weights` | | ES-MDA weights, e.g. 4,4,4,4 (default: ERT's 4,2,1). |
| `--apex-method` | | APEX's method to compare, with its defaults (default: the case's). |
| `--replicates` | 3 | Seeds per method. |
| `--name` | | The study's name. |
| `--yes` | false | Approve without asking. |
| `--wait` (on) / `--no-wait` (off) | false | Wait for the study to finish. |

## `apex ert convert`

Write CASE as a native APEX case that runs without ERT.

```bash
uv run apex ert convert CASE OUT_DIR
```

| option | default | description |
|---|---|---|
| `--method` | esmda | APEX's method in the configuration. |
| `--overwrite` (on) / `--no-overwrite` (off) | false | Replace a case already in OUT_DIR. |

## `apex ert export`

Write CONFIG as a self-contained ERT case (case.ert, prior, observations, grid).

```bash
uv run apex ert export CONFIG OUT_DIR
```

| option | default | description |
|---|---|---|
| `--ert-defaults` (on) / `--no-ert-defaults` (off) | false | Keep ERT's outlier rejection and STD_CUTOFF (default: off). |
| `--overwrite` (on) / `--no-overwrite` (off) | false | Replace a case already in OUT_DIR. |

## `apex ert history-match`

History-match an ERT case with APEX's methods; ERT runs the forward model.

```bash
uv run apex ert history-match CASE APEX_TOML
```

| option | default | description |
|---|---|---|
| `--methods` | | Comma-separated [methods.<label>] to run (default: all). |
| `--max-steps` | | Stop after this many analysis steps; run the same command again to resume. |
| `--progress` (on) / `--no-progress` (off) | true | Show progress on standard error. |
| `--write-back` (on) / `--no-write-back` (off) | true | Add the iterates to the case's ERT storage (ERT's GUI). |

## `apex ert import`

Replay ERT's iterations into an APEX run directory (reports, summaries, comparisons).

```bash
uv run apex ert import CASE_DIR RUN_DIR
```


## `apex ert run`

Run ERT on an exported case, locally (ERT 26.x, the `ert` extra).

```bash
uv run apex ert run CASE_DIR
```

| option | default | description |
|---|---|---|
| `--mode` | es_mda | ERT's experiment: ensemble_experiment, ensemble_smoother, es_mda, test_run. |
| `--weights` | | ES-MDA weights, e.g. 4,4,4,4 (mode es_mda). |
| `--dry-run` (on) / `--no-dry-run` (off) | false | Print the command only. |

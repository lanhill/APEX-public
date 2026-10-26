---
audience: public
generated: reference
---

# The apex commands: reference

Every one of these 18 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `apex approve`

Approve a pending submission (then submit it again with its token).

```bash
uv run apex approve TOKEN
```

| option | default | description |
|---|---|---|
| `--root` | | Services root (default: $APEX_HOME, else ~/.apex/service). |

## `apex data-split`

Explore how to split production data for cross validation.

```bash
uv run apex data-split CONFIG
```

| option | default | description |
|---|---|---|
| `--prior-run` | | A run of the case with a prior forecast, for ensemble redundancy. |
| `--max-cv` | | Largest candidate CV set, in wells. |
| `--output` | | Directory for the results (default: CONFIG dir/data_split). |
| `--font-size` | 12.0 | Font size of the figures (points). |

## `apex gc`

Forget runs whose folder is gone; prune what the service keeps.

```bash
uv run apex gc
```

| option | default | description |
|---|---|---|
| `--root` | | Services root (default: $APEX_HOME, else ~/.apex/service). |
| `--chats` (on) / `--no-chats` (off) | false | Also apply the chat history's limits now ([chat] retention_days and max_history_mb). |
| `--guests` (on) / `--no-guests` (off) | false | Also apply the guests' limits now ([guests] retention_days and max_storage_mb; guests' work only). |
| `--dry-run` (on) / `--no-dry-run` (off) | false | With --code: list what would go, remove nothing. |
| `--code` (on) / `--no-code` (off) | false | Also remove frozen copies of APEX no job or process uses that are older than [jobs] keep_code_days, and run the installed maintenance tasks. |

## `apex mcp`

Serve APEX's tools to an MCP host (Claude Code, ...); mcp extra, installed by default.

```bash
uv run apex mcp
```

| option | default | description |
|---|---|---|
| `--http` | | Serve Streamable HTTP at HOST:PORT, behind the API token of `apex service token` (default: stdio). |
| `--client` | mcp | Client name, for quotas and the call log. |
| `--model` | | The model driving the calls, for the log. |
| `--budget-runs` | | Simulator runs this session may submit. |
| `--root` | | Services root (default: $APEX_HOME, else ~/.apex/service). |

## `apex optimize`

Minimise an objective with one of PET's optimizers (EnOpt, GenOpt, SmcOpt, ...).

```bash
uv run apex optimize CONFIG
```

| option | default | description |
|---|---|---|
| `--resume` | | Continue the optimization run in this directory. |
| `--max-iterations` | | Stop after this many accepted iterations (resumable). |
| `--progress` (on) / `--no-progress` (off) | true | Show progress on standard error: bars on a terminal, timestamped lines otherwise. |

## `apex pet-import`

Convert a PET history-matching configuration into an APEX case.

```bash
uv run apex pet-import PET_CONFIG OUTPUT_DIR
```

| option | default | description |
|---|---|---|
| `--prior` | | Stored prior ensemble (.npz), if the config samples one. |
| `--simulator` | opm_flow | APEX simulator for the runfile. |
| `--seed` | | Run seed (default: the config's). |
| `--truth` | | True model (.npz, one array per variable): a synthetic case. |
| `--pet-geometry` (on) / `--intended-geometry` (off) | true | Prior correlation geometry: what PET samples, or what its keywords describe. |
| `--grid-file` | | An EGRID of the deck, for geophysical data: its active cells fix the AVO traces and the gravity and displacement stations. |
| `--avo-angles-checked` (on) / `--no-avo-angles-checked` (off) | false | Import multi-angle AVO data known to hold each angle's own traces (SimulatorWrap's do not: it gives every angle the last one's). |

## `apex plugins`

List every plugin with its options (for clients: --json).

```bash
uv run apex plugins
```

| option | default | description |
|---|---|---|
| `--json` | false | Full descriptions, as JSON. |

## `apex predict`

Forecast ensembles of a history-matching run beyond history.

```bash
uv run apex predict CONFIG
```

| option | default | description |
|---|---|---|
| `--progress` (on) / `--no-progress` (off) | true | Show progress on standard error: bars on a terminal, timestamped lines otherwise. |

## `apex qaqc`

QA/QC of an iterate: coverage, Mahalanobis scores, Kalman gains, movement (PET).

```bash
uv run apex qaqc RUN_DIR
```

| option | default | description |
|---|---|---|
| `--iteration` | | Iterate to check (default: the posterior). |
| `--output` | | Directory for the report (default: RUN_DIR/qaqc). |
| `--damping` | 0.0 | lambda of the Kalman gains (0: ES). |
| `--font-size` | 12.0 | Font size of the figures (points). |

## `apex report`

Print the data mismatch history and performance metrics of a run; optionally write figures.

```bash
uv run apex report RUN_DIR
```

| option | default | description |
|---|---|---|
| `--plots` (on) / `--no-plots` (off) | false | Also write the result figures (PNG and PDF). |
| `--output` | | Directory for the figures (default: RUN_DIR/report). |
| `--font-size` | 12.0 | Font size of the figures (points). |

## `apex resume`

Continue a stopped or interrupted run from its last checkpoint.

```bash
uv run apex resume RUN_DIR
```

| option | default | description |
|---|---|---|
| `--max-steps` | | Stop after this many analysis steps; continue later with `apex resume`. |
| `--progress` (on) / `--no-progress` (off) | true | Show progress on standard error: bars on a terminal, timestamped lines otherwise. |

## `apex run`

Start a history-matching run.

```bash
uv run apex run CONFIG
```

| option | default | description |
|---|---|---|
| `--max-steps` | | Stop after this many analysis steps; continue later with `apex resume`. |
| `--progress` (on) / `--no-progress` (off) | true | Show progress on standard error: bars on a terminal, timestamped lines otherwise. |

## `apex schema`

Print the JSON Schema of a configuration kind (what its TOML file accepts).

```bash
uv run apex schema KIND
```


## `apex stop`

Ask a running run to stop at its next checkpoint (resume continues it).

```bash
uv run apex stop RUN_DIR
```


## `apex submit`

Run a configuration, or resume a run, in the job service's background.

```bash
uv run apex submit CONFIG
```

| option | default | description |
|---|---|---|
| `--resume` | | Resume this run directory instead. |
| `--kind` | | run, prediction or optimization (default: from the file). |
| `--max-steps` | | Stop after this many steps (iterations of an optimization). |
| `--code` | | live, frozen (the last commit; a resume: the run's own) or a commit (default: [jobs] cli_code, frozen). |
| `--priority` | normal | high, normal or low. |
| `--yes` | false | Approve a large job without asking. |
| `--approval-token` | | A token approved with `apex approve`. |
| `--force-new` (on) / `--no-force-new` (off) | false | Run again even if an identical run exists. |
| `--wait` (on) / `--no-wait` (off) | false | Wait for the job to end. |
| `--client` | user | Client name, for quotas and grants (`apex grant`). |
| `--annotation` | | An annotation 'key=value' (repeatable); 'experiment=NAME' lets a grant of --client for that experiment approve the job. |
| `--root` | | Services root (default: $APEX_HOME, else ~/.apex/service). |

## `apex summary`

Summarise runs: every metric at every step (iterates, forecasts, optimization steps).

```bash
uv run apex summary PATHS
```

| option | default | description |
|---|---|---|
| `--output` | | Write the Markdown here (and the CSV next to it) instead of stdout. |
| `--title` | Results summary | Title of the report. |

## `apex validate`

Validate a configuration and plan its cost; nothing is simulated.

```bash
uv run apex validate CONFIG
```

| option | default | description |
|---|---|---|
| `--json` | false | The validation, as JSON. |
| `--root` | | Services root (default: $APEX_HOME, else ~/.apex/service). |

## `apex version`

Print the installed APEX version.

```bash
uv run apex version
```

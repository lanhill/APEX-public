---
audience: public
generated: reference
---

# Executors: reference

Every one of these 3 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `ert`

Runs a forecast's members as one ERT ensemble experiment, locally.

Select it with:

```toml
[execution]
name = "ert"
```

| option | type | default | description |
|---|---|---|---|
| `ert_executable` | string or null | | The `ert` program; default the one in APEX's environment, else on `PATH`. |
| `retries` | integer | 0 | Resubmissions of a failed realization (`MAX_SUBMIT` = 1 + retries). |
| `timeout_s` | number | 86400.0 | Longest a batch may take. |
| `workers` | integer | 1 | Realizations ERT runs at once (`MAX_RUNNING`). |

## `process`

Runs members in a pool of worker processes.

Select it with:

```toml
[execution]
name = "process"
```

| option | type | default | description |
|---|---|---|---|
| `retries` | integer | 2 | Retries of a member's transient failures, and the first wait. |
| `retry_delay_s` | number | 30.0 | Retries of a member's transient failures, and the first wait. |
| `workers` | integer or null | | Number of worker processes; defaults to the CPU count. Simulators that launch multi-threaded or MPI programs should budget their own cores within this. A simulator that declares `max_parallel_runs` (e.g. Eclipse's `licenses`) caps it: never more runs at once than the simulator allows. |

## `serial`

Runs members one after another in the calling process.

Select it with:

```toml
[execution]
name = "serial"
```

| option | type | default | description |
|---|---|---|---|
| `retries` | integer | 2 | Retries of a member's transient failures, and the first wait. |
| `retry_delay_s` | number | 30.0 | Retries of a member's transient failures, and the first wait. |

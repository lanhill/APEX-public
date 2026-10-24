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
| `ert_executable` | string or null | | |
| `retries` | integer | 0 | |
| `timeout_s` | number | 86400.0 | |
| `workers` | integer | 1 | |

## `process`

Runs members in a pool of worker processes.

Select it with:

```toml
[execution]
name = "process"
```

| option | type | default | description |
|---|---|---|---|
| `retries` | integer | 2 | |
| `retry_delay_s` | number | 30.0 | |
| `workers` | integer or null | | |

## `serial`

Runs members one after another in the calling process.

Select it with:

```toml
[execution]
name = "serial"
```

| option | type | default | description |
|---|---|---|---|
| `retries` | integer | 2 | |
| `retry_delay_s` | number | 30.0 | |

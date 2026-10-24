---
audience: public
generated: reference
---

# The apex jobs commands: reference

Every one of these 3 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `apex jobs list`

List the jobs, oldest first: id, kind, state, code, where.

```bash
apex jobs list
```

| option | default | description |
|---|---|---|
| `--state` | | Only jobs in this state (repeatable). |
| `--last` | 20 | Only the last N jobs (0: all). |
| `--root` | | Services root (default: $APEX_HOME, else ~/.apex/service). |

## `apex jobs show`

Show a job's record and its last events, as JSON.

```bash
apex jobs show JOB
```

| option | default | description |
|---|---|---|
| `--events` | 10 | The last N events too. |
| `--root` | | Services root (default: $APEX_HOME, else ~/.apex/service). |

## `apex jobs stop`

Stop a job: a queued one is cancelled, a running run stops at its next checkpoint.

```bash
apex jobs stop JOB
```

| option | default | description |
|---|---|---|
| `--root` | | Services root (default: $APEX_HOME, else ~/.apex/service). |

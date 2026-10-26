---
audience: public
generated: reference
---

# The apex runs commands: reference

Every one of these 2 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `apex runs list`

List the catalogued runs (oldest first).

```bash
uv run apex runs list
```

| option | default | description |
|---|---|---|
| `--annotation` | | Only runs with this annotation, 'key=value'. |
| `--root` | | Services root (default: $APEX_HOME, else ~/.apex/service). |

## `apex runs show`

Show a run's summary, as JSON.

```bash
uv run apex runs show RUN
```

| option | default | description |
|---|---|---|
| `--root` | | Services root (default: $APEX_HOME, else ~/.apex/service). |

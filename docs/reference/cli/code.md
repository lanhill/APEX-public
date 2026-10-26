---
audience: public
generated: reference
---

# The apex code commands: reference

Every one of these 3 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `apex code freeze`

Make (or complete) the frozen copy of a commit; print its folder.

```bash
uv run apex code freeze COMMIT
```

| option | default | description |
|---|---|---|
| `--root` | | Services root (default: $APEX_HOME, else ~/.apex/service). |

## `apex code list`

List the frozen copies, and what uses each.

```bash
uv run apex code list
```

| option | default | description |
|---|---|---|
| `--root` | | Services root (default: $APEX_HOME, else ~/.apex/service). |

## `apex code remove`

Remove a frozen copy (refused while a job or a process uses it).

```bash
uv run apex code remove COMMIT
```

| option | default | description |
|---|---|---|
| `--root` | | Services root (default: $APEX_HOME, else ~/.apex/service). |

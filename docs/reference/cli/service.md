---
audience: public
generated: reference
---

# The apex service commands: reference

Every one of these 5 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `apex service reindex`

Rebuild the run catalogue from the run directories.

```bash
uv run apex service reindex
```

| option | default | description |
|---|---|---|
| `--root` | | Services root (default: $APEX_HOME, else ~/.apex/service). |

## `apex service start`

Start the job service of a root in the background (one per root).

```bash
uv run apex service start
```

| option | default | description |
|---|---|---|
| `--root` | | Services root (default: $APEX_HOME, else ~/.apex/service). |
| `--http` | | Also serve the REST API at HOST:PORT (the api extra, installed by default). Keep 127.0.0.1 unless a TLS proxy is in front. |

## `apex service status`

Show whether the service runs, and the jobs by state.

```bash
uv run apex service status
```

| option | default | description |
|---|---|---|
| `--root` | | Services root (default: $APEX_HOME, else ~/.apex/service). |

## `apex service stop`

Stop the job service (running jobs go on; their workers record their end).

```bash
uv run apex service stop
```

| option | default | description |
|---|---|---|
| `--root` | | Services root (default: $APEX_HOME, else ~/.apex/service). |
| `--timeout` | | Seconds to wait for it to end (default 30); exit code 2 if it is still stopping then. |

## `apex service token`

Print the REST API's bearer token (created on first use, readable by you only).

```bash
uv run apex service token
```

| option | default | description |
|---|---|---|
| `--root` | | Services root (default: $APEX_HOME, else ~/.apex/service). |
| `--rotate` (on) / `--no-rotate` (off) | false | Replace the token (old clients are refused). |
| `--ui` | false | A guest's UI token instead: they see only their own work, within <root>/ui_access.toml. |
| `--guest` | guest | The guest (with --ui); each guest has a token of their own. |
| `--revoke` (on) / `--no-revoke` (off) | false | Delete the guest's token (with --ui); refused at once. |
| `--list-guests` | false | List the guests who have a token, and when each expires. |
| `--owner` | false | The owner's web-UI token (with --ui): it expires, unlike the API token, which the web UI refuses. |

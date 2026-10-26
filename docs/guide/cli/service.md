---
audience: public
items:
  - cli:apex service reindex
  - cli:apex service start
  - cli:apex service status
  - cli:apex service stop
  - cli:apex service token
---

# The apex service commands

Run these commands from the APEX folder: `uv run` runs them in APEX's environment, as the README's installation sets it up.

## Goal

By the end of this lesson you will be able to start and stop the APEX job service for a root, check what it is doing, hand out access tokens for the REST API and the web UI, and rebuild the run catalogue if it gets out of step with the run directories.

The commands covered are `apex service start`, `apex service status`, `apex service stop`, `apex service token` and `apex service reindex`.

## Prerequisites

- APEX is installed and the `apex` command works in your shell.
- You know which services root you want to work with. Every command takes `--root`. If you leave it out, APEX uses `$APEX_HOME`, and if that is not set, `~/.apex/service`.
- To serve the REST API with `--http`, the REST API's packages, in APEX's default environment (`uv sync --locked`); when APEX is a dependency of your own project, name the `api` extra.

## The idea: one service per root

A root is a directory that holds the job service's state: its runs, its jobs and its tokens. Each root has at most one job service, running in the background. All five commands act on one root, chosen with `--root`.

## Steps

### 1. Start the service: `apex service start`

```bash
uv run apex service start
```

This starts the job service of the root in the background.

| Option | What it does |
| --- | --- |
| `--root` | The services root (default: `$APEX_HOME`, else `~/.apex/service`). |
| `--http` | Also serves the REST API at `HOST:PORT`. Needs the REST API's packages (the `api` extra, in the default environment). |

To serve the REST API as well, give a host and port:

```bash
uv run apex service start --http 127.0.0.1:8765
```

Keep the host at `127.0.0.1` unless a TLS proxy is in front of the service.

### 2. Check on it: `apex service status`

```bash
uv run apex service status
```

This shows whether the service is running and lists the jobs by state. It takes only `--root`.

### 3. Get a token: `apex service token`

The REST API uses a bearer token. Print it with:

```bash
uv run apex service token
```

The token is created the first time you ask for it and can be read by you only.

| Option | What it does |
| --- | --- |
| `--root` | The services root. |
| `--rotate` / `--no-rotate` | `--rotate` replaces the token, and old clients are refused. With `--ui` (and `--guest NAME` or `--owner`) it issues a new web-UI token, e.g. after the old one expired. `--no-rotate` turns it off again. The default is off. |
| `--ui` | Prints a guest's UI token instead. A guest sees only their own work, within `<root>/ui_access.toml`. |
| `--guest` | Names the guest, used together with `--ui`. Each guest has a token of their own. The default is `guest`. |
| `--revoke` / `--no-revoke` | `--revoke` deletes the guest's token (used with `--ui`), and it is refused at once. `--no-revoke` turns it off. The default is off. |
| `--list-guests` | Lists the guests who have a token and when each expires. |
| `--owner` | Prints the owner's web-UI token (used with `--ui`). It expires, unlike the API token, which the web UI refuses. |

Common cases:

```bash
# Replace the API token
uv run apex service token --rotate

# A token for a guest called alice
uv run apex service token --ui --guest alice

# Take that access away again
uv run apex service token --ui --guest alice --revoke

# See who has a token, and when each expires
uv run apex service token --list-guests

# Your own token for the web UI
uv run apex service token --ui --owner
```

### 4. Stop the service: `apex service stop`

```bash
uv run apex service stop
```

This stops the job service. Jobs that are already running go on, and their workers record their end themselves.

| Option | What it does |
| --- | --- |
| `--root` | The services root. |
| `--timeout` | Seconds to wait for the service to end (default 30). The exit code is 2 if it is still stopping by then. |

```bash
uv run apex service stop --timeout 60
```

### 5. Rebuild the catalogue: `apex service reindex`

```bash
uv run apex service reindex
```

This rebuilds the run catalogue from the run directories. It takes only `--root`.

## Worked example

You want a service for a project root at `/data/project-a`, with the REST API on the local machine and a guest who may look at their own work.

```bash
# 1. Start the service for this root, with the REST API on localhost
uv run apex service start --root /data/project-a --http 127.0.0.1:8765

# 2. Confirm it is running and see the jobs by state
uv run apex service status --root /data/project-a

# 3. Print the API token for your client
uv run apex service token --root /data/project-a

# 4. Give a guest a UI token (the guest sees only their own work)
uv run apex service token --root /data/project-a --ui --guest alice

# 5. Later, check who has a token and when each expires
uv run apex service token --root /data/project-a --list-guests

# 6. Stop the service, waiting up to a minute
uv run apex service stop --root /data/project-a --timeout 60
```

If you later suspect the catalogue no longer matches the run directories, rebuild it:

```bash
uv run apex service reindex --root /data/project-a
```

## Pitfalls

- **Forgetting `--root`.** Without it, every command works on `$APEX_HOME`, or `~/.apex/service` if that is unset. If you use several roots, pass the same `--root` to every command.
- **Exposing the API.** Keep `--http` on `127.0.0.1` unless a TLS proxy is in front of it.
- **`--http` fails to serve.** It needs the REST API's packages: in APEX's default environment (`uv sync --locked`); when APEX is a dependency of your own project, name the `api` extra.
- **Rotating the token.** `--rotate` replaces the token, and clients that hold the old one are refused. Give them the new one.
- **Revoking is immediate.** After `--revoke`, the guest's token is refused at once.
- **`--guest`, `--revoke` and `--owner` need `--ui`.** They apply to UI tokens, not to the API token.
- **Using the API token in the web UI.** The web UI refuses it. Use `--ui --owner` for your own web-UI token, which expires.
- **Expecting `stop` to end jobs.** Running jobs keep going. Their workers record their end.
- **Exit code 2 from `stop`.** The service was still stopping when the `--timeout` ran out. Check with `apex service status`, or wait longer with a larger `--timeout`.

## What to read next

- The other pages of the command-line guide, for the commands that submit and inspect jobs once the service is running.
- The page on the REST API, for how to use the bearer token you printed.
- The page on the web UI, for how guests and owners sign in with their UI tokens.

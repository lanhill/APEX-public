---
audience: public
items:
  - cli:apex run
  - cli:apex service reindex
  - cli:apex service start
  - cli:apex service status
  - cli:apex service stop
  - cli:apex service token
  - cli:apex stop
  - cli:apex submit
  - config:service:service
  - plugin:gmac_transforms:state
---

# The apex service commands

Run these commands from the APEX folder. `uv run` runs them in APEX's environment, as the README's installation sets it up.

## Goal

By the end of this lesson you can start and stop the APEX job service for a root, check what it is doing, send it a job with `apex submit`, hand out access tokens for the REST API and the web UI, and rebuild the run catalogue.

The commands covered are `apex service start`, `apex service status`, `apex service stop`, `apex service token`, `apex service reindex`, `apex submit`, `apex run` and `apex stop`.

## Prerequisites

- APEX is installed as the README describes, so `uv run apex ...` works from the APEX folder.
- You know which services root you want. Every `apex service` command, and `apex submit`, takes `--root`. If you leave it out, APEX uses `$APEX_HOME`, and if that is not set, `~/.apex/service`.
- To serve the REST API with `--http`, you need the `api` extra. It is installed by default.
- A run configuration in TOML, if you want to submit a job.

## The idea: one service per root

Each root has at most one job service, running in the background. The service takes jobs, runs them with workers and keeps a **run catalogue**: its list of runs, built from the run directories. All the commands below act on the root you choose with `--root`.

`apex run` is different. It starts a history-matching run in your terminal, without the service. `apex submit` hands the same kind of work to the service.

## Steps

### 1. Start the service: `apex service start`

```bash
uv run apex service start
```

This starts the job service of the root in the background.

| Option | What it does |
| --- | --- |
| `--root` | The services root (default: `$APEX_HOME`, else `~/.apex/service`). |
| `--http` | Also serves the REST API at `HOST:PORT`. Needs the `api` extra, installed by default. |
| `--wait` | Seconds to wait for the service to be up: 60 by default, 120 with `--http` (the REST API must answer too). `$APEX_SERVICE_START_TIMEOUT` sets the default; `--wait` wins over it. |

To serve the REST API too, give a host and port:

```bash
uv run apex service start --http 127.0.0.1:8765
```

Keep the host at `127.0.0.1` unless a TLS proxy is in front. A TLS proxy is a program that sits in front of a service and adds encryption (HTTPS) to its traffic. With `127.0.0.1` only programs on your own machine can reach the API.

### 2. Check on it: `apex service status`

```bash
uv run apex service status
```

This shows whether the service runs and lists the jobs by state. It takes only `--root`.

### 3. Submit a job: `apex submit`

```bash
uv run apex submit run.toml
```

This runs a configuration in the job service's background. You can also resume a run directory with `--resume`.

| Option | What it does |
| --- | --- |
| `config` | A run, prediction or optimization configuration. |
| `--resume` | Resume this run directory instead. |
| `--kind` | `run`, `prediction` or `optimization` (default: taken from the file). |
| `--max-steps` | Stop after this many steps (iterations of an optimization). |
| `--code` | `live`, `frozen` (the last commit; for a resume, the run's own) or a commit. The default is `[jobs] cli_code`, which is `frozen`. |
| `--priority` | `high`, `normal` or `low`. The default is `normal`. |
| `--yes` | Approve a large job without asking. |
| `--approval-token` | A token approved with `apex approve`. |
| `--force-new` / `--no-force-new` | `--force-new` runs again even if an identical run exists. `--no-force-new` turns it off. The default is off. |
| `--wait` / `--no-wait` | `--wait` waits for the job to end. `--no-wait` turns it off. The default is off. |
| `--client` | Client name, for quotas and grants (`apex grant`). The default is `user`. |
| `--annotation` | An annotation `key=value`, repeatable. `experiment=NAME` lets a grant of `--client` for that experiment approve the job. |
| `--root` | The services root. |

```bash
uv run apex submit run.toml --priority high --wait
```

### 4. Run in the foreground and stop a run

`apex run` starts a history-matching run directly:

```bash
uv run apex run run.toml --max-steps 5
```

`--max-steps` stops after that many analysis steps, and you continue later with `apex resume`. `--progress` (the default) shows progress on standard error: bars on a terminal, timestamped lines otherwise. `--no-progress` turns it off.

To ask a running run to stop at its next checkpoint, give `apex stop` its run directory. Resuming continues it.

```bash
uv run apex stop path/to/run_dir
```

### 5. Get a token: `apex service token`

The REST API uses a bearer token. Start the service with `--http` first, since without it there is no API to use the token on. Send the token with your requests as a bearer token (an `Authorization: Bearer ...` header). Print it with:

```bash
uv run apex service token
```

The token is created the first time you ask for it and can be read by you only.

| Option | What it does |
| --- | --- |
| `--root` | The services root. |
| `--rotate` / `--no-rotate` | `--rotate` replaces the token, and old clients are refused. `--no-rotate` turns it off. The default is off. |
| `--ui` | Prints a guest's UI token instead of the API token. The guest sees only their own work, within `<root>/ui_access.toml`. |
| `--guest` | Names the guest, used with `--ui`. Each guest has a token of their own. The default is `guest`. |
| `--revoke` / `--no-revoke` | `--revoke` deletes the guest's token (used with `--ui`), and it is refused at once. `--no-revoke` turns it off. The default is off. |
| `--list-guests` | Lists the guests who have a token, and when each expires. |
| `--owner` | Prints the owner's web-UI token (used with `--ui`). It expires. The API token does not work in the web UI, because the web UI refuses it. |

Common cases:

```bash
# Replace the API token (clients with the old one are refused)
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

The web UI is the one place the API token is not accepted. If a web-UI token has expired, run the `--ui --owner` command again for your own, or `--ui --guest NAME` for a guest, and use what it prints.

### 6. Stop the service: `apex service stop`

```bash
uv run apex service stop
```

This stops the job service. Jobs that are already running go on, and their workers record their end.

| Option | What it does |
| --- | --- |
| `--root` | The services root. |
| `--timeout` | Seconds to wait for it to end (default 30). The exit code is 2 if it is still stopping then. |

```bash
uv run apex service stop --timeout 60
```

### 7. Rebuild the catalogue: `apex service reindex`

```bash
uv run apex service reindex
```

This rebuilds the run catalogue from the run directories. Use it if you think the catalogue no longer matches the run directories. It takes only `--root`.

## Service settings

The `[service]` section has four options. This block shows each with its default:

```toml
[service]
max_concurrent_jobs = 2
poll_s = 1.0
max_retries = 2
routines = true
```

- `max_concurrent_jobs` is an integer, at least 1.
- `poll_s` is a number greater than 0.
- `max_retries` is an integer, at least 0.
- `routines` is true or false.

## Worked example

You want a service for a project root at `/data/project-a`, with the REST API on the local machine, one job submitted, and a guest who may look at their own work.

```bash
# 1. Start the service for this root, with the REST API on localhost
uv run apex service start --root /data/project-a --http 127.0.0.1:8765

# 2. Confirm it is running and see the jobs by state
uv run apex service status --root /data/project-a

# 3. Submit a configuration to the service
uv run apex submit run.toml --root /data/project-a

# 4. Print the API token for your client
uv run apex service token --root /data/project-a

# 5. Give a guest a UI token (the guest sees only their own work)
uv run apex service token --root /data/project-a --ui --guest alice

# 6. Later, check who has a token and when each expires
uv run apex service token --root /data/project-a --list-guests

# 7. Stop the service, waiting up to a minute
uv run apex service stop --root /data/project-a --timeout 60
```

If you later suspect the catalogue no longer matches the run directories, rebuild it:

```bash
uv run apex service reindex --root /data/project-a
```

## Pitfalls

- **Forgetting `--root`.** Without it, every command works on `$APEX_HOME`, or `~/.apex/service` if that is unset. If you use several roots, pass the same `--root` to every command.
- **Exposing the API.** Keep `--http` on `127.0.0.1` unless a TLS proxy is in front of it.
- **A slow start on a loaded machine.** If `apex service start` says the service was not up in time, start it again with a longer wait, for example `--wait 300`. The error says how long it waited, and `service.log` in the root records the attempt.
- **Rotating the token.** `--rotate` replaces the token, and clients that hold the old one are refused. Give them the new one.
- **Revoking is immediate.** After `--revoke`, the guest's token is refused at once.
- **`--guest`, `--revoke` and `--owner` go with `--ui`.** They apply to UI tokens, not to the API token.
- **Using the API token in the web UI.** The web UI refuses it. Use `--ui --owner` for your own web-UI token, which expires.
- **Expecting `stop` to end jobs.** Running jobs keep going. Their workers record their end.
- **Exit code 2 from `apex service stop`.** The service was still stopping when `--timeout` ran out. Run `uv run apex service status` to see whether it runs now. If it is still stopping, run `apex service stop` again with a larger `--timeout`.
- **Mixing up `apex run` and `apex submit`.** `apex run` works in your terminal. `apex submit` hands the job to the service, so the service must be running.

## A note on `state`

`state` is not a service command. It is a plugin of the `gmac_transforms` registry that gives the raw state, or some of its variables, with the option `variables` (a list of names; the default is all). It appears in this reference only because the page inventory lists it.

## What to read next

- The other pages of the command-line guide, for the commands that inspect jobs once the service is running.
- The page on the REST API, for how to use the bearer token you printed.
- The page on the web UI, for how guests and owners sign in with their UI tokens.

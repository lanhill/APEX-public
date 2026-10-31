---
audience: public
items:
  - cli:apex approve
  - cli:apex predict
  - cli:apex resume
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

By the end of this lesson you can start and stop the APEX job service for a root, check what it is doing, send it a job with `apex submit`, approve a job that needs approval, hand out access tokens for the REST API and the web UI, and rebuild the run catalogue. You also learn how the foreground commands `apex run`, `apex predict`, `apex resume` and `apex stop` differ from the service.

The commands covered are `apex service start`, `apex service status`, `apex service stop`, `apex service token`, `apex service reindex`, `apex submit`, `apex approve`, `apex run`, `apex predict`, `apex resume` and `apex stop`.

## Prerequisites

- APEX is installed as the README describes, so `uv run apex ...` works from the APEX folder.
- You know which services root you want. Every `apex service` command, `apex submit` and `apex approve` takes `--root`. If you leave it out, APEX uses `$APEX_HOME`, and if that is not set, `~/.apex/service`.
- To serve the REST API with `--http`, you need the `api` extra. It is installed by default.
- A configuration file in TOML, if you want to submit a job: a run, prediction or optimization configuration. In the examples it is called `run.toml`.

## The idea: one service per root

Each root has at most one job service, running in the background. The service takes jobs, runs them with workers and keeps a **run catalogue**: its list of runs, built from the run directories. All the commands below act on the root you choose with `--root`.

The foreground commands work differently. `apex run` starts a history-matching run in your terminal, without the service. `apex submit` hands the same kind of work to the service.

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
| `--wait` | Seconds to wait for the service to be up: 60 by default, 120 with `--http`, or the value of `$APEX_SERVICE_START_TIMEOUT`. A loaded machine may need more. |

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

This runs a configuration in the job service's background, so the service must be running. You can also resume a run directory in the service with `--resume`.

| Option | What it does |
| --- | --- |
| `config` | A run, prediction or optimization configuration. |
| `--resume` | Resume this run directory instead. |
| `--kind` | `run`, `prediction` or `optimization` (default: taken from the file). |
| `--max-steps` | Stop after this many steps (iterations of an optimization). |
| `--code` | Which code the job uses: `live`, `frozen` (the last commit; for a resume, the run's own) or a commit. The default is the `[jobs] cli_code` setting, which is `frozen`. |
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

With `--code frozen` the job uses the last commit, so it is not affected by edits you make afterwards. With `--code live` it uses your code as it is. Pass a commit to pin one exactly.

### 4. Approve a large job: `apex approve`

A large job needs approval before it runs. There are two ways to give it:

- Approve at once: add `--yes` to `apex submit`, which approves a large job without asking.
- Approve by token: a submission that is waiting for approval has a pending approval with a token. Approve it with `apex approve`, then submit it again with that token.

```bash
uv run apex approve TOKEN
uv run apex submit run.toml --approval-token TOKEN
```

`apex approve` takes the pending approval's token as its argument and, like the other service commands, `--root`.

### 5. Run in the foreground, resume and stop

`apex run` starts a history-matching run directly, in your terminal:

```bash
uv run apex run run.toml --max-steps 5
```

`--max-steps` stops after that many analysis steps. `--progress` (the default) shows progress on standard error: bars on a terminal, timestamped lines otherwise. `--no-progress` turns it off.

`apex resume` continues a stopped or interrupted run from its last checkpoint. It takes the run's directory, and also `--max-steps` and `--progress`/`--no-progress`:

```bash
uv run apex resume path/to/run_dir
```

`apex resume` works in your terminal. To continue a run in the job service instead, use `apex submit --resume path/to/run_dir`.

`apex predict` forecasts the ensembles of a history-matching run beyond history. It takes a prediction configuration (TOML) and `--progress`/`--no-progress`:

```bash
uv run apex predict predict.toml
```

To ask a running run to stop at its next checkpoint, give `apex stop` its run directory. Resuming continues it.

```bash
uv run apex stop path/to/run_dir
```

### 6. Get a token: `apex service token`

The REST API uses a bearer token. Start the service with `--http` first, since without it there is no API to use the token on. Print the token with:

```bash
uv run apex service token
```

The token is created the first time you ask for it and can be read by you only. Send it with each request in an `Authorization: Bearer` header. For example, with the service started on `127.0.0.1:8765`:

```bash
TOKEN=$(uv run apex service token)
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8765/
```

Add the path of the endpoint you want to call to the URL.

| Option | What it does |
| --- | --- |
| `--root` | The services root. |
| `--rotate` / `--no-rotate` | `--rotate` replaces the token, and old clients are refused. `--no-rotate` turns it off. The default is off. |
| `--ui` | Prints a guest's UI token instead of the API token. The guest sees only their own work, within `<root>/ui_access.toml`. |
| `--guest` | Names the guest, used with `--ui`. Each guest has a token of their own. The default is `guest`. |
| `--revoke` / `--no-revoke` | `--revoke` deletes the guest's token (used with `--ui`), and it is refused at once. `--no-revoke` turns it off. The default is off. |
| `--list-guests` | Lists the guests who have a token, and when each expires. |
| `--owner` | Prints the owner's web-UI token (used with `--ui`). Unlike the API token, it expires, and the web UI refuses the API token. |

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

The web UI refuses the API token, so use a UI token there. If a web-UI token has expired, run the `--ui --owner` command again for your own, or `--ui --guest NAME` for a guest, and use what it prints.

### 7. Stop the service: `apex service stop`

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

### 8. Rebuild the catalogue: `apex service reindex`

```bash
uv run apex service reindex
```

This rebuilds the run catalogue from the run directories. Use it if you think the catalogue no longer matches the run directories. It takes only `--root`.

## Service settings

The `[service]` section of `service.toml`, which lives in the services root (`~/.apex/service`, or `$APEX_HOME` if set), has four options. This block shows each with its default:

```toml
[service]
max_concurrent_jobs = 2
poll_s = 1.0
max_retries = 2
routines = true
```

- `max_concurrent_jobs` is an integer, at least 1. It limits how many jobs the service runs at the same time.
- `poll_s` is a number greater than 0, in seconds.
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

# 5. Use it against the API
curl -H "Authorization: Bearer $(uv run apex service token --root /data/project-a)" http://127.0.0.1:8765/

# 6. Give a guest a UI token (the guest sees only their own work)
uv run apex service token --root /data/project-a --ui --guest alice

# 7. Later, check who has a token and when each expires
uv run apex service token --root /data/project-a --list-guests

# 8. Stop the service, waiting up to a minute
uv run apex service stop --root /data/project-a --timeout 60
```

If you later suspect the catalogue no longer matches the run directories, rebuild it:

```bash
uv run apex service reindex --root /data/project-a
```

## Pitfalls

- **Forgetting `--root`.** Without it, every command works on `$APEX_HOME`, or `~/.apex/service` if that is unset. If you use several roots, pass the same `--root` to every command.
- **Exposing the API.** Keep `--http` on `127.0.0.1` unless a TLS proxy is in front of it.
- **A slow start on a loaded machine.** If the service is not up in time, start it again with a longer wait, for example `--wait 300`.
- **Rotating the token.** `--rotate` replaces the token, and clients that hold the old one are refused. Give them the new one.
- **Revoking is immediate.** After `--revoke`, the guest's token is refused at once.
- **`--guest`, `--revoke` and `--owner` go with `--ui`.** They apply to UI tokens, not to the API token.
- **Using the API token in the web UI.** The web UI refuses it. Use `--ui --owner` for your own web-UI token, which expires.
- **Expecting `stop` to end jobs.** `apex service stop` stops the service, and running jobs keep going. Their workers record their end. To stop one run, use `apex stop` with its run directory.
- **Exit code 2 from `apex service stop`.** The service was still stopping when `--timeout` ran out. Run `uv run apex service status` to see whether it runs now. If it is still stopping, run `apex service stop` again with a larger `--timeout`.
- **Mixing up `apex run` and `apex submit`.** `apex run` works in your terminal. `apex submit` hands the job to the service, so the service must be running. The same holds for `apex resume` (terminal) and `apex submit --resume` (service).
- **Submitting again without the token.** After `apex approve`, pass the token to `apex submit` with `--approval-token`, or the job is still waiting for approval.

## Not a service command: `state`

`state` is not a service command. It is a plugin in the `gmac_transforms` registry, so look for it with the transforms rather than here.

## What to read next

- The other pages of the command-line guide, for the commands that inspect jobs once the service is running.
- The page on the REST API, for the endpoints you can call with the bearer token.
- The page on the web UI, for how guests and owners use their UI tokens.

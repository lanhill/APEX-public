---
audience: public
---

# The REST API

## Goal

By the end of this lesson you can start the APEX REST API, get a token, validate and submit a configuration over HTTP, follow the job, and handle the approval and file-editing flows. The API serves APEX's service calls to planners on other machines, clients in other languages, and the web UI.

## Prerequisites

- The REST API's packages: in APEX's default environment (`uv sync --locked`); when APEX is a dependency of your own project, name the `api` extra.
- The job service, described in [the job service](service.md). The API runs inside it.
- Configuration files on the machine where the service runs. All paths in requests name files on that machine.

## Steps

### 1. Start the service with the API

```bash
apex service start --http 127.0.0.1:8765   # the scheduler and the API, in one process
apex service token                          # print the bearer token (created on first use)
apex service token --rotate                 # a new token; the old one is refused at once
apex service stop
```

`<root>` below means the service's data directory, where it keeps its state, such as `api_token` and `ui_access.toml`.

### 2. Choose how clients reach it

- **Same machine:** keep `127.0.0.1`, as above.
- **Another machine, simplest:** open an SSH tunnel, `ssh -L 8765:127.0.0.1:8765 host`, and connect to `127.0.0.1:8765` locally.
- **Another machine, direct:** give an explicit host, for example `apex service start --http 0.0.0.0:8765`, and put a reverse proxy that terminates TLS in front of it. Configuring the proxy itself is outside this page.

### 3. Authenticate

Every `/v1` request needs `Authorization: Bearer <token>`. The owner's API token is in `<root>/api_token`, readable by its user only. Print it with `apex service token`.

**Guests** are people you let into the web UI without giving them the owner's token. The owner creates each one on the service's machine:

```bash
apex service token --ui --guest alice     # create alice's token and print it
apex service token --list-guests          # list the guests, with each expiry
```

Send the printed token to the guest. They send it as `Authorization: Bearer <token>`. A guest sees only their own jobs, runs, studies, predictions and chats, within `<root>/ui_access.toml` (see docs/ui.md, "Who sees what"). Hidden things answer `404`, refused actions `403`. `GET /v1/whoami` returns the role, the guest and the allowed actions.

Two kinds of owner token exist:

- **The API token** (`apex service token`) is for scripts. It does not expire.
- **The owner's web-UI token** (`apex service token --ui --owner [--rotate]`) is for the browser. The web UI marks its requests with the header `X-APEX-Client: ui`. The API token is refused on such requests (title "Web-UI token required"), so scripts should not send that header.

A guest's token and the owner's web-UI token expire `[ui] token_days` after they were issued (default 7). `whoami` gives `expires`.

### 4. Validate, then submit

The right column names the Python calls, `ApexService`'s and the clients' (`Client`, `RemoteClient`), not the tools of LLM clients. The tools (`apex tools`, the MCP server at `/mcp`) have one `validate` for every kind of configuration: runs, studies, predictions and optimizations; see [the tools](tools.md).

| Method and path | Service and client call |
|---|---|
| `GET /v1/plugins`, `GET /v1/schemas/{kind}` | `plugins()`, `schema(kind)` |
| `POST /v1/configs:validate` | `validate`, `validate_study` or `validate_optimization` |
| `POST /v1/runs`, `/v1/studies`, `/v1/predictions` | `submit_run`, `submit_study`, `submit_prediction`: `202 Accepted`, with `Location: /v1/jobs/{id}` |
| `POST /v1/optimizations` | `submit_optimization`: `optimization_file`, or `config` with `base_dir`; `max_iterations`, `annotations`, `priority`, `client`, `approval_token`. `202`, with the job |
| `GET /v1/jobs` (`?state=`, cursor), `GET /v1/jobs/{id}` | the queue, `status` |
| `GET /v1/events` | every job's events: `?since=` (service-wide numbers), or SSE |
| `GET /v1/runs` (`?annotation=k=v`, cursor), `GET /v1/runs/{id}` | the catalogue, `run_summary` |
| `GET /v1/studies/{name}`, `GET /v1/studies/{name}/comparison` (`?metric=`) | `study_status`, `compare` |
| `GET /v1/predictions/summary?path=` | `prediction_summary` |
| `GET /v1/optimizations`, `GET /v1/optimizations/summary?path=` | the caller's optimization jobs, `optimization_summary` |

**Which kind of configuration.** For `POST /v1/configs:validate`, give one of: `config_file` (a path on the service's machine), `config` with `base_dir`, or `config_text` (TOML, not saved) with `base_dir`. A study file is recognised by its `[study]` table, an optimization by its `[optimization]` and `[optimizer]` tables.

**Paths** are the service's. `config_file`, `study_file`, `prediction_file`, `optimization_file`, `base_dir` and `path` (and an agent route's `optimization`) all name files on the machine where the service runs: absolute, or relative to the workspace (see step 8; `..` is refused). **A relative path is always read as a workspace path**, never against the service's working directory: send absolute paths for files elsewhere. The Python client (`Client.connect`) resolves the `Path`s it is given where it runs and sends them absolute. A configuration sent as a document (`config`) resolves its relative paths against `base_dir`.

A web-UI user's answers (a guest's) carry no server paths: paths under the workspace come back relative to it (the only root itself as `(workspace)`), the service's root as `<service>`, and other known folders as placeholders: `<code>` (frozen copies), `<resources>`, `<apex>` (APEX's source), `<python>` (the virtual environment and its site-packages), `<tmp>` and `<home>`. System folders (`/usr`, `/opt`, ...) are never replaced. The routes take `(workspace)` and `<service>/...` back; the other placeholders are for reading only, since no guest's work lies under them. A file's contents (`GET /v1/files/content`) are never rewritten, nor are JSON keys.

### 5. Follow a job

Submitting returns `202` with `Location: /v1/jobs/{id}`. Read `GET /v1/jobs/{id}` for its state. With the Python client, `client.follow(job.id)` streams the job's events until it ends (see the worked example).

Lists return `{"items": [...], "next_cursor": ...}`. Pass `next_cursor` back as `?cursor=` to get the next page. It is `null` on the last page. `limit` sets the page size.

### 6. Handle approval

A submission above the approval limits is refused. The limits are set in `service.toml [approval]`, so read that file for the current values.

1. Submit as usual. If it is above the limits you get `403`, and the response carries a pending `token`.
2. A person runs `apex approve TOKEN` on a terminal. This is deliberately not available over HTTP.
3. Submit the same request again, adding `"approval_token": "TOKEN"`.

A submission with an `experiment` annotation may instead pass on a *grant*. A grant is a standing permission that a person created with `apex grant create` on the service's machine. No HTTP route creates, changes or lists grants.

### 7. The agents

`GET /v1/agents` lists the runtime agents: name, summary, command, the chat tool that runs each, and `visibility` (`public` or `private`, as `service.toml [agents.visibility]` may set; guests get only public ones).

| Route | Body | Answer |
|---|---|---|
| `POST /v1/agents/monitor` | `{run}` or `{optimization}` (a directory), `model?` | the run monitor's report (status, recommendation, findings); it never stops anything |
| `POST /v1/agents/diagnose` | `{run, model?}` | the diagnosis: verdict, causes with evidence, recommended changes with patches |
| `POST /v1/agents/report` | `{run}`, `{study}` or `{optimization}`, `model?` | the narrative report, Markdown (its numbers checked against the facts) |
| `POST /v1/agents/plan` | `{case, goal, budget?, model?}` | a study draft written to the user's UI folder (`UI/<user>/drafts/planner/`), with its variants and cost |

A `?` after a field name means the field is optional. `model` is a boolean. `run`, `study` and `optimization` identify what to look at.

- By default each route runs the agent's rules, which is quick. `model: true` also asks the model of `[agents.llm]`. If that model is unavailable, the rules answer and `reason` says so.
- Every answer carries `decided_by` and `reason`.
- None of the routes submits, stops or resumes anything.

`GET /v1/ui/pages` lists the web-UI pages installed packages add that the caller may see: `{"pages": [{name, title, url}]}`, each served at `/ui/pages/<name>/`. A page for the owner only is never listed to a guest.
- UI users see only the agents their list names, and only on their own runs and studies.

### 8. Edit configuration files

The web UI has an editor that reads and writes configuration files on the service's machine. The API routes behind it are:

| Route | Purpose |
|---|---|
| `GET /v1/workspace` | the workspace's roots: each one's name (the prefix of relative paths) and folder (names only for a web-UI guest) |
| `GET /v1/files?path=` | list files |
| `GET /v1/files/content?path=` | read a file |
| `PUT /v1/files/content` | write a file |

The editor is confined by `[workspace] roots` in `service.toml` (default `<root>/configs`). If the service runs under a `[policy]`, the roots must be set explicitly and lie under the policy's `input_roots`.

- Paths are relative to the workspace. With one root, relative to it. With several, prefixed by each root's folder name. Absolute paths under a root are accepted too. Every route that takes a path takes these relative paths.
- Only `.toml` files of at most 1 MB, under a root (after resolving symlinks and `..`). Anything else is `403`.

The read-then-write cycle uses `sha256` to avoid overwriting someone else's change:

1. `GET /v1/files/content?path=case/config.toml` returns `{path, text, sha256}`.
2. Edit `text`, then `PUT /v1/files/content` with `{path, text, sha256}`, where `sha256` is the value you read (`null` for a new file).
3. If the file changed since (or already exists, for a new one), you get `409` with its current `sha256`. Read the file again, merge your edit into the new text, and `PUT` with the new `sha256`.

## Worked example

The commands below validate a configuration, submit it, and read the job. `TOKEN` is the output of `apex service token`.

```bash
TOKEN=$(apex service token)

# 1. Validate a file that lives on the service's machine
curl -s -X POST http://127.0.0.1:8765/v1/configs:validate \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"config_file": "/data/cases/m5spots/config.toml"}'

# 2. Submit it as a run
curl -s -i -X POST http://127.0.0.1:8765/v1/runs \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"config_file": "/data/cases/m5spots/config.toml", "annotations": {"planner": "x"}}'

# 3. Read the job named in the Location header
curl -s http://127.0.0.1:8765/v1/jobs/JOB_ID -H "Authorization: Bearer $TOKEN"
```

The submission answers `202 Accepted` with `Location: /v1/jobs/JOB_ID`. Submitting a study, prediction or optimization works the same way, with the route and body fields from the table in step 4 (for an optimization: `optimization_file`, optionally `max_iterations`, `annotations` and `approval_token`).

The same flow in Python, with the client:

```python
from pathlib import Path
from apex.client import Client

client = Client.connect("http://127.0.0.1:8765")  # token: $APEX_API_TOKEN, or token=...
check = client.validate(Path("/data/cases/m5spots/config.toml"))  # a path on the service's machine
job = client.submit_run(Path("/data/cases/m5spots/config.toml"), annotations={"planner": "x"})
for event in client.follow(job.id):  # Server-Sent Events, until the job ends
    print(event.index, event.message)
print(client.run_summary(job.run_ids[0]).posterior_iteration)
```

`Client.connect` returns a `RemoteClient` with the in-process `Client`'s methods, so a program written against one mostly runs against the other. Some calls differ:

- **Remote only:** `follow` (SSE), `runs` (the catalogue, with an annotation filter) and `artifact` (a run's files).
- **Run summaries:** on a remote client, `run_summary` takes a run id, as above. Reading a summary from a run directory is local only.

**Errors** are the in-process ones: `ConfigError` for invalid input, `ApprovalRequired` (with `.token`) when a person must approve, and `KeyError` for an unknown job.

## Errors on the wire

Errors are RFC 9457 problem documents (`application/problem+json`) with `type`, `title`, `status` and `detail`.

| Status | When |
|---|---|
| `400` | A failed request (an APEX error), or a bad cursor |
| `403` | Approval required (the document has `token`), or a file the editor may not touch |
| `404` | No such job, run or file |
| `409` | A workspace file changed since it was read |
| `422` | An invalid configuration, or a request that does not match its schema |

To tell the two `422` cases apart: for an invalid configuration, read `detail` for the validation messages. For a request that does not match its schema, read `errors: [{location, message}]`, where `location` names the offending field. Both name the offending keys and say what is wrong with them, but never repeat the values given, so an error cannot show the contents of a file that is not a configuration.

## Pitfalls

- Paths in a request name files on the **service's** machine, not yours.
- Binding a host other than `127.0.0.1` exposes the API. Use a TLS-terminating reverse proxy, or an SSH tunnel.
- `apex service token --rotate` refuses the old token at once. Update every script.
- Web-UI tokens expire. Send the API token from scripts and do not send `X-APEX-Client: ui`.
- Resubmitting after approval needs the same request plus `approval_token`.
- After a `409` on `PUT`, re-read the file and merge. Do not resend the old `sha256`.

## What to read next

- [The job service](service.md), for starting, stopping and configuring the service.
- docs/ui.md, "Who sees what", for guest access.
- docs/tools.md, "Other MCP hosts".

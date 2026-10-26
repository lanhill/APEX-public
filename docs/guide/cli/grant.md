---
audience: public
items:
  - cli:apex approve
  - cli:apex grant create
  - cli:apex grant list
  - cli:apex grant revoke
  - cli:apex resume
  - cli:apex run
  - cli:apex stop
---

# The apex grant commands

Run these commands from the APEX folder: `uv run` runs them in APEX's environment, as the README's installation sets it up.

## Goal

Some submissions wait for approval before they go ahead. You can approve a single pending submission with `apex approve`, which takes the pending approval's token. After approving, you submit it again with that token. A grant approves in advance instead: it covers a budget of simulator runs for one client's experiment until the grant expires.

In this lesson you will create a grant with `apex grant create`, check it with `apex grant list`, and end it early with `apex grant revoke`. You will also see how `apex stop` and `apex resume` relate to a revoke.

## Terms used on this page

- **Client**: the name of whoever sends work in, for example `planner`.
- **Submission**: a piece of work a client sends in. It is labelled with an `experiment` annotation. The grant's `experiment` argument must match that annotation, so ask whoever runs the client which value it uses.
- **Simulator run**: one run of the simulator. The grant's budget is a total number of these.
- **Grant**: an advance approval for one client and one `experiment` value, limited by a budget and an expiry time.

## Prerequisites

- A services root. By default this is `$APEX_HOME`, or `~/.apex/service` if that is not set. Every grant command accepts `--root` to point at a different one.
- The name of the client the grant is for (for example `planner`).
- The exact value of the `experiment` annotation on that client's submissions.
- A budget (a total number of simulator runs) and a lifetime for the grant.

## Steps

### 1. Create a grant with `apex grant create`

| Name | Kind | Required | Meaning |
|---|---|---|---|
| `client` | argument | yes | The client the grant is for (e.g. `planner`). |
| `experiment` | argument | yes | The submissions' `experiment` annotation. |
| `--max-simulator-runs` | option | yes | The total budget of simulator runs. |
| `--expires` | option | yes | How long the grant lasts: `7d`, `12h` or `30m`. At most `30d`. |
| `--root` | option | no | Services root (default: `$APEX_HOME`, else `~/.apex/service`). |

```bash
uv run apex grant create planner my-experiment --max-simulator-runs 200 --expires 7d
```

While the grant is active, it approves work from `planner` annotated `my-experiment` ahead of time, up to the budget of 200 simulator runs. Once the expiry passes, the grant no longer approves anything.

After creating a grant, run `apex grant list` (next step) to confirm it is there and to see its id.

### 2. Check the grants with `apex grant list`

`apex grant list` shows the active grants. For each one you see the id, client, experiment, used/budget and expiry. In `used/budget`, the budget is the number you gave to `--max-simulator-runs`, and `used` is how much of it has been used so far.

```bash
uv run apex grant list
```

Add `--all` to also include expired and revoked grants. `--all` is an on switch and is off by default.

```bash
uv run apex grant list --all
```

### 3. Revoke a grant with `apex grant revoke`

Give `apex grant revoke` the grant's id, which you copy from the id column of `apex grant list`. The grant is revoked at once.

```bash
uv run apex grant revoke <grant_id>
```

To check the result, run `apex grant list --all`, which includes revoked grants.

Revoking only stops the grant from approving more. Submissions already let through before the revoke are not stopped.

### 4. Stop work that is already running

To stop a run that is already going, ask it to stop at its next checkpoint with `apex stop`. It takes the directory of the running run:

```bash
uv run apex stop <run_dir>
```

To continue a stopped or interrupted run from its last checkpoint, give `apex resume` the same run directory:

```bash
uv run apex resume <run_dir>
```

### Approving one submission with `apex approve`

For a single pending submission, give `apex approve` the pending approval's token, then submit it again with that token:

```bash
uv run apex approve <token>
```

Like the grant commands, it accepts `--root` if your services root is not the default.

## Worked example

The client `planner` is about to run an experiment annotated `my-experiment`, and you want to allow up to 200 simulator runs for the next 12 hours.

1. Create the grant:

   ```bash
   uv run apex grant create planner my-experiment --max-simulator-runs 200 --expires 12h
   ```

2. Confirm it is there:

   ```bash
   uv run apex grant list
   ```

   There is one row per active grant, with the columns id, client, experiment, used/budget and expiry. Your new grant shows `planner` and `my-experiment`, and its budget is 200.

3. Later, the experiment is called off. Copy the grant's id from the id column of the list and revoke it:

   ```bash
   uv run apex grant revoke <grant_id>
   ```

4. To see it again afterwards, include revoked and expired grants:

   ```bash
   uv run apex grant list --all
   ```

If your services root is not the default, add `--root /path/to/service` to each command.

## Pitfalls

- **Revoking does not stop work already started.** Submissions already let through before the revoke carry on. Use `apex stop` with the run directory to ask a running run to stop.
- **The budget is a limit.** The budget is the total number of simulator runs the grant covers. Watch `used/budget` in `apex grant list`.
- **The experiment must match.** The `experiment` argument must be the same text as the submissions' `experiment` annotation.
- **Two options are required on `create`.** `--max-simulator-runs` and `--expires` must both be given. `--root` is optional.
- **Lifetime limit.** `--expires` takes `7d`, `12h` or `30m` style values, and at most `30d`.
- **Expired and revoked grants are hidden.** `apex grant list` shows only active grants unless you add `--all`.
- **Wrong root, empty list.** If a grant seems to be missing, check that you use the same `--root` (or `$APEX_HOME`) as when you created it.

## What to read next

- `apex run` starts a history-matching run from a TOML configuration. Its `--max-steps` option stops after a number of analysis steps, and `--progress` / `--no-progress` turn progress output on or off.
- `apex resume` also accepts `--max-steps` and `--progress` / `--no-progress`.
- `apex stop` asks a running run to stop at its next checkpoint; `apex resume` continues it.

The other pages in the CLI section of the guide cover these commands in more detail.

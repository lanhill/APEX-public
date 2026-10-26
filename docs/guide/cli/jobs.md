---
audience: public
items:
  - cli:apex jobs list
  - cli:apex jobs show
  - cli:apex jobs stop
---

# The apex jobs commands

## Goal

By the end of this lesson you will be able to see which jobs APEX knows about, inspect one job in detail, and stop a job you no longer want. Three commands cover this: `apex jobs list`, `apex jobs show` and `apex jobs stop`.

## Prerequisites

- APEX is installed and the `apex` command works in your shell.
- You know where your services root is. By default it is `$APEX_HOME`, or `~/.apex/service` if that variable is not set. All three commands accept `--root` to point somewhere else.

## Steps

### 1. List jobs with `apex jobs list`

`apex jobs list` prints the jobs, oldest first. Each line gives the id, kind, state, code and where.

Options:

| Option | Meaning | Default |
|---|---|---|
| `--state` | Only jobs in this state. You can repeat it. | none (all states) |
| `--last` | Only the last N jobs. `0` means all. | `20` |
| `--root` | Services root. | `$APEX_HOME`, else `~/.apex/service` |

Examples:

```bash
uv run apex jobs list
uv run apex jobs list --last 5
uv run apex jobs list --last 0
uv run apex jobs list --state running --state queued
```

### 2. Inspect a job with `apex jobs show`

`apex jobs show` prints a job's record and its last events, as JSON. It takes one required argument, `job`: a job id, or just its first characters.

Options:

| Option | Meaning | Default |
|---|---|---|
| `job` (argument) | A job id or its first characters. Required. | none |
| `--events` | How many of the last events to include. | `10` |
| `--root` | Services root. | `$APEX_HOME`, else `~/.apex/service` |

Examples:

```bash
uv run apex jobs show 3f9a
uv run apex jobs show 3f9a --events 50
```

### 3. Stop a job with `apex jobs stop`

`apex jobs stop` stops a job. What happens depends on where the job is:

- A queued job is cancelled.
- A running job stops at its next checkpoint, not instantly.

It takes the required argument `job` (an id or its first characters) and the optional `--root`.

```bash
uv run apex jobs stop 3f9a
```

## Worked example

You started several jobs and want to stop the one that is still waiting.

1. Look at the jobs that are queued or running:

   ```bash
   uv run apex jobs list --state queued --state running
   ```

2. Pick the job from the id column. Say its id starts with `3f9a`. Check its record and recent events:

   ```bash
   uv run apex jobs show 3f9a --events 20
   ```

3. Stop it:

   ```bash
   uv run apex jobs stop 3f9a
   ```

4. Confirm with the list again, or with `show`:

   ```bash
   uv run apex jobs show 3f9a
   ```

If the job was queued it is now cancelled. If it was running, it ends when it reaches its next checkpoint, so its state may not change at once.

## Pitfalls

- **Only the last 20 jobs are listed by default.** An older job may be missing from `apex jobs list`. Use `--last 0` to see all of them.
- **A short id must be unambiguous.** `job` accepts the first characters of an id; give enough characters to identify one job.
- **Stopping a running job is not immediate.** It stops at the next checkpoint.
- **Use the same root.** If your jobs live under a non-default services root, pass the same `--root` to every command, otherwise you will look at a different set of jobs.
- **`show` prints JSON.** Its output is meant to be read or piped into a JSON tool, not formatted as a table.

## What to read next

Read the other pages in the command-line section of this guide to see how jobs are started, and return here whenever you need to check on or stop one.

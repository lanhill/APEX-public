---
audience: public
---

# The APEX job service

APEX runs history-matching runs, predictions and studies as **jobs**. Jobs run in the background, in worker processes that outlive the command that submitted them. This page shows where a job's data lives, how the service schedules jobs, and how approval, grants and policy limit what a submission may do.

## Goal

By the end you will be able to:
- start, check and stop the service;
- work out a job's reservation and predict whether it will start;
- get a large submission approved, by hand or ahead of time with a grant;
- limit what submissions may read, run and write.

## Prerequisites

- APEX installed, with the `apex` command on your path.
- A services root you can write to.

The **services root** keeps everything about the jobs. It is `$APEX_HOME`, by default `~/.apex/service`, and holds:
- `apex.db`: jobs, events, the run catalogue, approvals and quota use;
- `service.toml`: settings;
- `runs/` and `studies/`: the runs submitted without a directory of their own;
- `jobs/<id>/worker.log`: each job's log.

`apex.db` is SQLite in WAL mode with `synchronous = NORMAL`. A crashed process loses nothing, and the file cannot be corrupted. A power loss or an operating-system crash may lose the last seconds of writes. Dead jobs are then resumed or failed as usual, `apex service reindex` rebuilds the run catalogue from the run folders, and a grant created just before can be created again.

## Running the service

`apex service start` runs one scheduler for the root. It starts what fits, resumes jobs whose worker died (up to `max_retries`), and sends notifications. Clients only queue work. Stopping the service does not stop running jobs.

```bash
uv run apex service start            # or: --root DIR; --http HOST:PORT also serves the REST API (api.md)
uv run apex service status           # running?, and the jobs by state
uv run apex service stop
uv run apex service reindex          # rebuild the run catalogue from the run directories
```

`stop` waits up to 30 s (change it with `--timeout`). If the service is still stopping by then, it says so and exits with code 2 (not 1, which is an error). A script can then wait longer and check with `apex service status`:

```bash
uv run apex service stop --timeout 30
if [ $? -eq 2 ]; then
  uv run apex service stop --timeout 120     # try again with a longer wait
fi
uv run apex service status                   # confirm it has stopped
```

## Submitting from the command line

`apex submit` queues one configuration: a history-matching run, a prediction or an optimization. It tells which from the file's sections; `--kind` settles a file that has more than one. The job goes through the same queue, pools, approval and notifications as a study or the REST API. An identical run is reused. The run's folder is the one `apex run` (or `apex optimize`) would use.

```bash
uv run apex submit CONFIG.toml [--max-steps N] [--wait]     # prints the job id and the run's folder
uv run apex submit --resume RUN_DIR                          # continue a stopped run, even one started with apex run
uv run apex jobs list [--state running] [--last 20]          # id, kind, state, code, folder
uv run apex jobs show JOB                                    # the record and its last events (a prefix of the id will do)
uv run apex jobs stop JOB                                    # queued: cancelled; running: stops at the next checkpoint
```

Large submissions ask for approval, as `apex study run` does (`--yes`, or `--approval-token`). By default a job submitted from the command line runs **frozen code** (next section). `--code live` runs the checkout instead.

## Scheduling

A job starts only when its **reservation** fits the machine's pools beside the jobs already running.

- **Runs at once** is how many runs of the job go on at the same time.
- A run's **tokens** are the units it takes from each pool while it runs. (These are pool tokens, not the approval tokens described below.)
- The reservation is the runs at once times each run's tokens.

Worked example: a job runs 4 runs at once, and each run takes `{eclipse: 2, cpu: 2}`. Its reservation is 4 x `{eclipse: 2, cpu: 2}` = `{eclipse: 8, cpu: 8}`. It starts only when 8 `eclipse` and 8 `cpu` tokens are free beside the running jobs.

- **Priorities** are `high`, `normal` and `low`, for example `apex study run --priority high`.
- **Quotas** apply per client, that is, per `client` name of a submission. Set them in a `[quotas.NAME]` section of `service.toml`, as `[quotas.planner-x]` does in the settings example below. The client sets its own name.
- A job that must wait keeps its place in the queue, so smaller jobs behind it do not keep it waiting for ever.

## Approval

Two counts decide whether a submission needs a person's approval:
- **simulator runs** are the individual simulations the whole submission may perform: more than `[approval] max_simulator_runs` (20,000) needs approval;
- **runs** are the runs the submission starts: more than `max_runs` (20) needs approval.

If APEX cannot work out the possible number of simulator runs, the cost is **unknown**.

How approval happens:
- The command that submits asks you at the prompt, and `--yes` answers yes without asking. For example, add `--yes` to a submitting command such as `apex study run`.
- Without a terminal (a script, CI), or when you answer no, `apex submit` submits nothing, prints the pending approval token and exits with status 1. A person runs `apex approve TOKEN`, and the script runs the same command again with `--approval-token TOKEN`.
- A program that submits gets `ApprovalRequired` with a token. A person runs `apex approve TOKEN`, and the program submits again with `approval_token=TOKEN`.
- Tool (MCP) clients are programs that use APEX through tools. They get their tokens when they run `validate`, which checks a submission without running it.
- A run that is reused as it is (an identical run already complete, queued or running) simulates nothing, so it needs no approval.

## Grants

A **grant** approves an experiment's spending ahead. It is for one client and one experiment (the submission's `experiment` annotation), with a total budget of simulator runs and an expiry.

```bash
uv run apex grant create planner-x exp-042 --max-simulator-runs 60000 --expires 7d   # 7d, 12h or 30m; at most 30d
uv run apex grant list [--all]
uv run apex grant revoke ID
```

Walkthrough:
1. Create the grant with the command above.
2. Submit as client `planner-x` with the `experiment` annotation set to `exp-042`. The grant stands in for the owner's approval, so nobody runs `apex approve` (a tool client still passes `validate`'s token, as for every submission):
   - `apex submit CONFIG --client planner-x --annotation experiment=exp-042`;
   - over REST, `"client": "planner-x"` and `"annotations": {"experiment": "exp-042"}` in the body;
   - as a tool client named `planner-x` (`apex mcp --client planner-x`), `validate` with `annotations={"experiment": "exp-042"}`, which answers `"approval": "granted"`, then the submit tool with that token and the same `annotations`.
3. If the submission needs approval and its cost fits the budget left, it passes. The cost is charged at once, and the approval is recorded `by grant <id>`.
4. Run `apex grant list`. The grant's `used` value has grown by the cost. Add `--all` to see grants that are no longer active.

Rules:
- An unknown cost never uses a grant. When no grant applies, `ApprovalRequired` says why (expired, revoked, budget left).
- The budget counts only submissions over the `[approval]` limits. Smaller ones need no approval and are not charged, so a grant's `used` is not the experiment's total spend.
- `apex submit` tries a grant before it asks, so a covered submission is not asked about. At the prompt of other commands, a person's explicit no is final: a grant does not override it.
- The client sets its own name and annotation. A grant bounds how much and how long it may spend unasked, not what it runs.
- For tool (MCP) clients, `validate` only checks the grant; the submission with the same `experiment` annotation charges it. A token validated without the annotation is still covered if the submission carries it and the budget holds.

## Policy

A service runs outside its clients' sandboxes. `[policy]` in `service.toml` limits what a submission may make it do:
- **read:** `input_roots`;
- **execute:** `code_roots` for deck templates and executables, plus `executables`, the program names allowed through `PATH`;
- **write:** `output_roots`.

The policy is off by default. Once you set one of `input_roots`, `code_roots` and `output_roots`, set all three. `executables` is an extra list on top of them. Minimal example:

```toml
[policy]
input_roots = ["/cases/*/inputs"]
code_roots = ["/opt/apex-code"]
output_roots = ["/cases/*/runs/apex"]
executables = ["flow"]
```

- **Roots** are absolute, may use globs (`/cases/*/runs/apex`), and are matched after resolving symlinks.
- **Timing:** the checks run before anything is resolved, read or run (`validate` reports them), and the worker checks each job again just before it runs.
- **Your part:** keep `code_roots`, `PATH` and the output roots unwritable for the clients.

**Tool clients are confined even without a policy.** A tool client (MCP, `apex tools call`, an agent loop) acts as the owner, so the files it names, and every input file their configurations name, must lie in the workspace's roots (`[workspace] roots`, by default `<root>/configs`), in `[tools] roots`, or in the service's own `runs/`, `studies/` and `optimizations/`; its `overrides` may not set templates, programs, their arguments or hosts. Under a policy, the policy's roots apply instead. The tools page's "Path rules" has the details. Widen it with:

```toml
[tools]
roots = ["/data/cases"]          # more folders whose files tool clients may name
```

## Which code a job runs

A job's worker imports APEX when it starts, and so does every process it starts later, including the automatic resume after a worker dies. By default that is the checkout as it is at that moment. If you edit the checkout while a long run goes on, the run can stop with an error, or mix two versions of the code.

A job can instead be **pinned** to a commit. It then runs from a **frozen copy**: a git worktree of the repository at that commit, with an environment of its own. Edits of the checkout never reach the job, nor its automatic resumes.

- `apex submit` and `apex study run` freeze by default (`[jobs] cli_code`, or `$APEX_CLI_CODE`). Pass `--code live`, `--code frozen` or `--code COMMIT` to choose. A study is frozen once, so every cell runs the same commit.
- In Python and over the REST API, a submission takes `code`: `None` or `"live"` (the default), `"frozen"`, or a commit. Only the owner may set it; a web-UI guest's request with a `code` is refused. `"frozen"` pins the repository's HEAD. For a resume or a prediction, it pins the commit the run started on, as long as the run started from a clean checkout. That way a resumed run stays bit-identical.
- A resume that names no code keeps the code the run last ran. A run submitted frozen and resumed from the web UI continues on its own commit.
- A pin is always a commit. Uncommitted changes are not in the copy; the command line says so.
- Only commits that know about pinning can be pinned. A worker refuses a job store whose records it would misread, and the job fails at once with the reason.
- A running service must know about pinning too. Restart the service after updating APEX; until then, pinned submissions are refused with that advice.
- A copy gets the extras installed where APEX runs (`[jobs] code_extras` to choose), and its worker runs in the copy's environment, programs such as `ert` included.

```bash
uv run apex code freeze [COMMIT]        # make (or complete) a commit's copy; prints its folder
uv run apex code list                   # the copies, and the jobs or processes using each
uv run apex code remove COMMIT          # refused while a job or a process uses it
uv run apex gc --code [--dry-run]       # remove copies not pinned for [jobs] keep_code_days, and unused
```

The service also removes unused copies daily: those no queued or running job pins and no process runs, last pinned more than `keep_code_days` (2) ago. The newest of `master` is always kept. Only copies APEX made are removed, and never one holding files git does not know; a copy you made by hand goes by hand. A resume pinned to a removed commit makes its copy again (a few minutes). A copy takes about 2 GB, most of it the environment.

## Settings (`service.toml`)

```toml
[service]
max_concurrent_jobs = 2
poll_s = 1.0
max_retries = 2
routines = true                  # run the routines installed packages add (false: none)

[jobs]                           # which code jobs run
cli_code = "frozen"              # jobs submitted from the command line ($APEX_CLI_CODE overrides)
code_dir = "/data/apex-code"     # where frozen copies live (default <root>/code)
code_extras = ["reservoir", "plots"]   # extras synced into each copy (default: those installed here)
keep_code_days = 2

[approval]
max_simulator_runs = 20000
max_runs = 20

[notifications]                  # optional, when a job ends
command = "notify-send 'APEX job ended'"
# webhook = "https://..."

[quotas.planner-x]               # optional, per client
max_concurrent_jobs = 1
max_simulator_runs_per_day = 50000

[agents.llm]                     # optional: the model of APEX's runtime agents (agents.md)
provider = "openai_compatible"
model = "qwen3:32b"
base_url = "http://localhost:11434/v1"

[agents.monitor]                 # optional: the run monitor (agents.md)
auto = false                     # true: the service checks running jobs, and notifies
collapse_spread_ratio = 0.05
max_failed_fraction = 0.2
```

## The catalogue

```bash
uv run apex runs list [--annotation study=NAME]
uv run apex runs show RUN_ID            # its summary, as JSON
uv run apex gc                          # forget entries whose directory was deleted (never deletes runs)
```

## Pitfalls

- Setting only one of the three policy roots lists. Set all three.
- Expecting a grant to cover a submission with an unknown cost. It never does.
- Expecting `used` on a grant to equal the experiment's total spend. Only submissions over the approval limits are charged.
- Treating exit code 2 from `apex service stop` as a failure. It means the service is still stopping.
- Confusing pool tokens (a run's share of a pool) with approval tokens (from `ApprovalRequired`).
- Expecting uncommitted changes to reach a job submitted from the command line. It runs the last commit; commit first, or pass `--code live`.

## What to read next

- `api.md` for the REST API served by `--http`.
- `agents.md` for the runtime agents and the run monitor.

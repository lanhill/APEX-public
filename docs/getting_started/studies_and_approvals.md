---
audience: public
---

# Getting started: studies, the job service and approvals

## Goal

By the end of this page you have compared two history-matching methods over several seeds in a study, followed it as background jobs, run APEX's job service from the terminal, and approved a submission that is over the service's cost limits.

The case is the linear-Gaussian example of [Your first history-matching run](first_run.md). It runs in seconds and needs no reservoir simulator.

## Before you start

- APEX installed as the README describes. Run every command below from the APEX folder.

## Step 1: a study of two methods

A **study** runs every combination of cases, methods and variants, several times with different seeds (the replicates), and compares the results. The example has one, `examples/linear_gaussian/study.toml`:

- `[study]`: its name (`linear_methods`), 3 replicates, the seed the later replicates' seeds come from (the first keeps the case's own), and the metrics to compare;
- `[study.cases]`: the case, the example's ES-MDA configuration;
- `[study.methods.esmda]` and `[study.methods.gies]`: each a complete `[method]` section;
- `[[study.variants]]`: a change to the case for every run, here an ensemble of 100 members instead of 500.

Check its cost first, then run it:

```bash
uv run apex validate examples/linear_gaussian/study.toml
uv run apex study run examples/linear_gaussian/study.toml --wait
```

The plan ends with the study's total:

```text
study: 6 runs, at most 7,923 simulator runs
```

`apex study run` submits the 6 runs as background jobs. Without `--wait` it returns at once; with it, it waits and prints `study complete`. The runs go to the services root (`$APEX_HOME`, by default `~/.apex/service`), in `studies/linear_methods/`. Jobs submitted from the command line run a frozen copy of APEX's code by default, so that editing the APEX folder never changes a running job. The first submission makes the copy, a git worktree of the commit with an environment of its own, which takes a few minutes and about 2 GB; uncommitted changes are not in it. `--code live` runs the folder as it is ([The job service](../service.md#which-code-a-job-runs)).

## Step 2: follow and compare it

```bash
uv run apex study status linear_methods
uv run apex jobs list
uv run apex study compare linear_methods
```

`apex study status` counts the runs by state (`6 complete`), and `apex jobs list` lists every job with its state. `apex study compare` gives each method's mean and spread over the replicates, and the paired difference with a Wilcoxon test:

```text
linear, rmse
  esmda/small                      n=3  0.20814 ± 0.005739
  gies/small                       n=3  0.20274 ± 0.005936
  gies/small - esmda/small: -0.005401 ± 0.00658, Wilcoxon p = 0.75
```

With 3 replicates the test cannot show a difference (`compare` says so); a real comparison needs more. `apex study run` takes the study file, while `status` and `compare` take the study's name, its `[study] name`.

## Step 3: the job service

Jobs run with or without APEX's job service. Without it, the command that submits them starts the first ones, and each job starts the next as it ends. With the service running, the service schedules them, resumes a job whose worker died, and sends notifications:

```bash
uv run apex service start
uv run apex service status
uv run apex service stop
```

`apex service status` says whether it runs and counts the jobs by state. Stopping the service does not stop running jobs. [The job service](../service.md) has its settings, and `uv run apex service start --http 127.0.0.1:8765` also serves the web UI ([The web UI and its AI assistant](web_ui.md)).

## Step 4: approving a submission

A submission needs a person's approval when it may take more simulator runs, or start more runs, than the limits in `service.toml` (`[approval] max_simulator_runs`, 20,000 by default, and `max_runs`, 20). At a terminal, the command asks you, and your yes is the approval. Answer no, or submit from a script, and you get a pending token to approve instead.

To see it on a small run, lower the limit in your own services root's `service.toml` (by default `~/.apex/service/service.toml`; create the file if it is missing):

```toml
[approval]
max_simulator_runs = 1000
```

The example's ES-MDA run may take 2,500 simulator runs, so it is now over the limit:

```bash
uv run apex submit examples/linear_gaussian/config_esmda.toml
```

It asks `This may take 2500 simulator runs in 1 runs. Run it? [y/N]`. Answer `y`, and it is approved and submitted. Answer `n`, and you get the token:

```text
not submitted: this submission may take 2500 simulator runs in 1 runs, which need approval (limits: 1000 simulator runs, 20 runs)
pending approval token: 4c60b46a3781c1ad
approve it with `apex approve 4c60b46a3781c1ad`, then run this command again with --approval-token 4c60b46a3781c1ad
```

Approve the token you were given, then submit again with it:

```bash
uv run apex approve 4c60b46a3781c1ad
uv run apex submit examples/linear_gaussian/config_esmda.toml --approval-token 4c60b46a3781c1ad
```

A token is bound to that submission and its cost, can be used once, and expires after 24 hours. Remove the `[approval]` lines afterwards to go back to the default limits. [Grants](../guide/cli/grant.md) approve a budget in advance, for a client's experiment.

## Next steps

- [Optimization](../optimization.md): studies of optimizers, compared the same way.
- [The agents from the command line](agents_cli.md): the planner writes a study for a goal and a budget.

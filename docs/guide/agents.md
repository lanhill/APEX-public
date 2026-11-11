---
audience: public
items:
  - agent:case-converter
  - agent:diagnosis
  - agent:geomodeler
  - agent:monitor
  - agent:planner
  - agent:plotter
  - agent:report
  - cli:apex agent diagnose
  - cli:apex agent monitor
  - cli:apex agent plan
  - cli:apex agent plot
  - cli:apex agent report
  - cli:apex approve
  - cli:apex ert import
  - cli:apex report
  - cli:apex resume
  - cli:apex run
  - cli:apex stop
  - cli:apex submit
  - config:history_match:prior
  - config:history_match:validation
  - config:optimization_study:study
  - config:service:service
  - config:study:study
  - plugin:figure_presets:brugge_classic
  - plugin:surrogate_designs:run
---

# APEX's agents

An APEX agent is a command that reads your case or your run and gives you something to review: a converted case, a prior, a plan, a health check, a diagnosis, figures or a report. Agents advise and draft. You decide what to apply or launch.

This page covers APEX's seven public agents: what each does, how to start it, and what you get.

## Goal

By the end you will know which agent to use at each stage of a history-matching study and how to start it.

## Prerequisites

- APEX is installed as described in the README. Type every command from the APEX folder as `uv run apex ...`.
- For the commands that take `RUN`: a run directory, a run id or a job id. The run directory is the folder your run writes its results to. Use it, or the id, wherever this page says `RUN`.
- The services root is the folder where APEX's job service keeps its state, including `service.toml`. It is `$APEX_HOME`, else `~/.apex/service`. The commands `uv run apex agent diagnose`, `uv run apex agent monitor`, `uv run apex agent plan`, `uv run apex agent plot`, `uv run apex agent report`, `uv run apex approve` and `uv run apex submit` accept `--root` to point at another services root.

## What a model receives

An agent can ask a language model for help. The model receives facts from your runs. Read [what a provider receives and how to keep it local](../agents.md#what-a-provider-receives-and-how-to-keep-it-local) before you point an agent at a hosted provider.

The agents that take a run or a case (`diagnosis`, `monitor`, `planner`, `plotter`, `report`) accept `--no-llm`: rules only, no model is asked. The provider and model default to `provider` and `model` in `[agents.llm]`. You can override them with `--provider`, `--model`, `--base-url` and `--api-key-env`. `--api-key-env` names the variable holding the key, never the key itself. If your model has no native tool calling, add `--no-tool-calls` and the agent prompts for JSON instead.

## The agents in the order you meet them

### `case-converter`

Turns a simulator's case (an OPM Flow or Eclipse deck with its history) into an APEX history-matching case that validates, with its observations, data errors, state and prior.

```bash
uv run apex agent convert import DECK
```

`DECK` is your simulator deck. This agent may send facts from your case to the model.

### `geomodeler`

Gives a case's property fields a prior: continuous Gaussian fields or facies by proportion, from your values or a reference field. Each prior is checked on a drawn sample.

```bash
uv run apex agent geomodel CONFIG
```

`CONFIG` is your case configuration. This agent may send facts from your case to the model.

The prior lives in the `[prior]` section of the case (`history_match:prior`). Its `generator` is a plugin: `gaussian` (independent normals), `file` (stored `.npz`), `gaussian_field` (variogram and Cholesky) or `geostatspy_sgsim`. A minimal complete `[prior]`:

```toml
[prior]
ensemble_size = 100
generator = "gaussian"
```

`ensemble_size` is the number of members (at least 2). A `[prior.variables.<name>]` section overrides the generator for one state variable. Realisations are clipped to the state bounds.

### `planner`

Plans a study (variants, replicates, metrics) for a case and a goal within a budget, as a draft you review, revise and launch.

- The case is a history-matching or optimization configuration.
- `--goal` says what the experiment should find out. It is required.
- The budget is the number of simulator runs for the whole study. `--budget` sets it. Without it, the approval limit applies: the limit on simulator runs set in `service.toml` in the services root, above which a submission needs approval.
- The draft is written beside the case, or where `--out` says.

```bash
uv run apex agent plan CONFIG --goal "Find out which of three permeability priors gives the best forecast" --budget 200 --no-llm
```

A study is a `[study]` section (`study:study`; `optimization_study:study` with `kind = "optimization"`). Open the draft, change what you disagree with, then launch it with `--submit`. A small study of the kind the planner drafts looks like this:

```toml
[study]
name = "compare-priors"
replicates = 2

[study.budget]
max_simulator_runs = 200

[study.cases]
base = "case.toml"

[[study.variants]]
name = "bigger-ensemble"
patch = { prior = { ensemble_size = 200 } }
```

The budget is checked before anything runs; a plan that exceeds it is refused.

#### Launching with approval

`--submit` submits the study only after you confirm. Without it, nothing is submitted.

1. Ask to submit: `uv run apex agent plan CONFIG --goal "Compare two priors" --submit`. If the study needs approval, the submission stays pending and you get a token.
2. Approve it: `uv run apex approve TOKEN`.
3. Submit again with the token: `uv run apex agent plan CONFIG --goal "Compare two priors" --approval-token TOKEN`. The token implies `--submit`.

### `monitor`

Checks a run's health: collapse, failures, damping, stagnation, overfitting and cross validation. It recommends continue, stop or investigate. Cross validation is set in the `[validation]` section (`history_match:validation`), with `cv` and `test` data and a `stopping` rule.

```bash
uv run apex agent monitor RUN
```

- `--watch` checks whenever a new forecast is scored, until the job ends. `--no-watch` turns it off. `--interval` sets the seconds between looks (default 60).
- `--act` offers to stop the job when a stop is recommended. `--no-act` turns it off. It applies to running iterative runs only, and you are asked to confirm first. You can also stop a run yourself with `uv run apex stop RUN_DIR`; `uv run apex resume RUN_DIR` continues it.
- After "investigate", run `diagnosis` on the same run.

### `diagnosis`

Explains what went wrong in a run (QA/QC of the posterior, failed members) and what to change, with validated configuration patches. It is advisory: it changes nothing itself.

```bash
uv run apex agent diagnose RUN --no-llm
```

Add `--json` to print the result as JSON. Review each patch, and edit your run configuration yourself if you agree with it.

### `plotter`

Draws a run's result figures: metric box and violin plots, forecast bands per well, layer maps and histograms, or a whole preset such as `brugge_classic`. See [Figures and the plotter agent](../figures.md).

With words (a model is asked; without `--request` you get the classic set):

```bash
uv run apex agent plot RUN --request "forecast bands for each well"
```

With a preset (no model):

```bash
uv run apex agent plot RUN --preset brugge_classic --with normalized
```

`--with` switches on a preset's option; repeat it for several. The `brugge_classic` options are `normalized` (also draws `normalized_data_mismatch`), `model_change` (also draws `model_change`) and `histogram_variable` (the state variable of `nn_hist/`, default `nn_params`). Figures go to `RUN/report/plots`, or to the folder given by `--output`.

### `report`

Writes a narrative report of a run, optimization, surrogate build or study, with every number checked against the facts of the run.

For a run:

```bash
uv run apex agent report RUN
```

For a study:

```bash
uv run apex agent report --study NAME
```

`--out` says where to write the Markdown.

## Worked example

You have an OPM Flow deck and want a first study.

1. Convert it: `uv run apex agent convert import DECK`.
2. Give its property fields a prior: `uv run apex agent geomodel CONFIG`.
3. Plan a study: `uv run apex agent plan CONFIG --goal "Compare two priors" --no-llm`. Review the draft.
4. Start a run: `uv run apex run CONFIG`. The run writes a run directory; use it as `RUN` below. (`--max-steps N` is optional: it stops after N analysis steps, the ensemble updates of the method, and `uv run apex resume RUN_DIR` continues later.)
5. Watch it: `uv run apex agent monitor RUN --watch`.
6. If the verdict is "investigate" or the run fails: `uv run apex agent diagnose RUN`.
7. Draw figures: `uv run apex agent plot RUN --preset brugge_classic`.
8. Write it up: `uv run apex agent report RUN`.

Without a model, `uv run apex report RUN_DIR` prints the data mismatch history and performance metrics of a run. Add `--plots` to also write the figures (`--no-plots` turns them off).

## Related commands

- `uv run apex submit CONFIG` runs a configuration, or with `--resume` continues one, in the job service's background. The job service reads `[service]` of `service.toml` (`service:service`): `max_concurrent_jobs` (default 2), `poll_s` (1.0), `max_retries` (2) and `routines` (true).
- `uv run apex ert import CASE_DIR` replays the iterations ERT ran on an exported case into an APEX run directory, so the agents can read it.
- The surrogate design `run` reuses the forecasts a history-matching run already holds, with no simulator runs.

## Pitfalls

- In the usage `apex agent report RUN | --study NAME`, the `|` means "either", not a shell pipe. Type one form.
- `RUN` is a run directory, a run id or a job id, not a free-form name.
- A plan is a draft. Nothing is submitted unless you pass `--submit` and confirm; a pending approval needs `uv run apex approve TOKEN` and a second call with `--approval-token`.
- `--preset` draws without a model. `--request` asks one.
- To keep your data local, use `--no-llm` or a local provider; see the link above.

## What to read next

- [What a provider receives and how to keep it local](../agents.md#what-a-provider-receives-and-how-to-keep-it-local)
- [Figures and the plotter agent](../figures.md)

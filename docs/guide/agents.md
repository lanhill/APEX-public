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
  - cli:apex report
  - cli:apex run
  - cli:apex stop
---

# APEX's agents

An APEX agent is a command that reads your case or your run and gives you something to review: a converted case, a prior, a plan, a health check, a diagnosis, figures or a report. Agents advise and draft. You decide what to apply or launch.

This page covers APEX's seven public agents: what each does, how to start it, and what you get.

## Goal

By the end you will know which agent to use at each stage of a history-matching study and how to start it.

## Prerequisites

- APEX is installed as described in the README. `apex` is not on your PATH, so every command is typed from the APEX folder as `uv run apex ...`.
- For the commands that take `RUN`: a run directory, a run id or a job id. A run directory is the folder `apex run` writes for your run.
- Agents look for the services root in `$APEX_HOME`, else `~/.apex/service`. Pass `--root` to use another one.

## What a model receives

An agent can ask a language model for help. The model receives facts from your runs. Read [what a provider receives and how to keep it local](../agents.md#what-a-provider-receives-and-how-to-keep-it-local) before you point an agent at a hosted provider.

The agents that take a run or a case (`diagnose`, `monitor`, `plan`, `plot`, `report`) accept `--no-llm`: rules only, no model is asked. The provider and model default to `provider` and `model` in `[agents.llm]`. You can override them with `--provider`, `--model`, `--base-url` and `--api-key-env`. `--api-key-env` names the variable holding the key, never the key itself. If your model has no native tool calling, add `--no-tool-calls` and the agent prompts for JSON instead.

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

### `planner`

Plans a study (variants, replicates, metrics) for a case and a goal within a budget, as a draft you review, revise and launch.

- The case is a history-matching or optimization configuration.
- The budget is the number of simulator runs for the whole study. `--budget` sets it. Without it, the approval limit of `service.toml` in the services root applies.
- The draft is written beside the case, or where `--out` says.
- `--submit` submits the study only after you confirm. If approval is pending, `apex approve` yields a token. Pass it with `--approval-token` to submit.

```bash
uv run apex agent plan CONFIG --goal "Find out which of three permeability priors gives the best forecast" --budget 200 --no-llm
```

Open the written draft, change what you disagree with, then launch it.

### `monitor`

Checks a run's health: collapse, failures, damping, stagnation, overfitting and cross validation. It recommends continue, stop or investigate.

```bash
uv run apex agent monitor RUN
```

- `--watch` checks whenever a new forecast is scored, until the job ends. `--interval` sets the seconds between looks (default 60).
- `--act` offers to stop the job when a stop is recommended. It applies to running iterative runs only, and you are asked to confirm first. You can also stop a run yourself with `uv run apex stop RUN_DIR`; `uv run apex resume` continues it.
- After "investigate", run `diagnose` on the same run.

### `diagnosis`

Explains what went wrong in a run (QA/QC of the posterior, failed members) and what to change, with validated configuration patches. It is advisory: it changes nothing itself.

```bash
uv run apex agent diagnose RUN --no-llm
```

Add `--json` to print the result as JSON. Review each patch and edit your configuration if you agree.

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

`--with` switches on a preset's option; repeat it for several. Figures go to `RUN/report/plots`, or to the folder given by `--output`.

### `report`

Writes a narrative report of a run or a study, with every number checked against the facts of the run.

For a run:

```bash
uv run apex agent report RUN
```

For a study (use one or the other):

```bash
uv run apex agent report --study NAME
```

`--out` says where to write the Markdown.

## Worked example

You have an OPM Flow deck and want a first study.

1. Convert it: `uv run apex agent convert import DECK`.
2. Give its property fields a prior: `uv run apex agent geomodel CONFIG`.
3. Plan a study: `uv run apex agent plan CONFIG --goal "Compare two priors" --no-llm`. Review the draft.
4. Start a run: `uv run apex run CONFIG`. `--max-steps` stops after that many analysis steps.
5. Watch it: `uv run apex agent monitor RUN --watch`.
6. If the verdict is "investigate" or the run fails: `uv run apex agent diagnose RUN`.
7. Draw figures: `uv run apex agent plot RUN --preset brugge_classic`.
8. Write it up: `uv run apex agent report RUN`.

Without a model, `uv run apex report RUN_DIR` prints the data mismatch history and performance metrics of a run. Add `--plots` to also write the figures.

## Pitfalls

- In the usage `apex agent report RUN | --study NAME`, the `|` means "either", not a shell pipe. Type one form.
- `RUN` is a run directory, a run id or a job id, not a free-form name.
- A plan is a draft. Nothing is submitted unless you pass `--submit` and confirm.
- `--preset` draws without a model. `--request` asks one.
- To keep your data local, use `--no-llm` or a local provider; see the link above.

## What to read next

- [What a provider receives and how to keep it local](../agents.md#what-a-provider-receives-and-how-to-keep-it-local)
- [Figures and the plotter agent](../figures.md)

---
audience: public
---

# Runtime agents and LLM providers

## Goal

By the end of this page you can point APEX's runtime agents at a language model, ask the agent a question, and check a run with the monitor, the diagnosis agent, the planner, the report writer and the plotter. You will also know how the agents keep a model's answer honest.

APEX's runtime agents run on any language model: hosted services with the OpenAI chat API, or open-source models served locally by Ollama, vLLM, llama.cpp or LM Studio. One interface, `apex.core.llm.LLMProvider`, and APEX's own agent loop drive them all the same way. No agent framework is involved. Frameworks can still drive APEX from outside, through the tools (see [the tools page](tools.md)).

For a step-by-step start with Claude Code or Ollama's Gemma, see [the agents from the command line](getting_started/agents_cli.md) and [the web UI and its AI assistant](getting_started/web_ui.md).

## Prerequisites

- A working APEX install and a services root with a `service.toml`.
- A model you can reach: either a local server, or a hosted service and its API key in an environment variable.
- For the monitor, diagnosis and report agents: a finished or running run directory, run id or job id.
- For the planner: a case configuration file (a `.toml` you already use with APEX).

Every agent also works without a model (`--no-llm`), using rules only.

## Step 1: choose a model

The model is set in `[agents.llm]` of the services root's `service.toml`:

```toml
[agents.llm]
provider = "openai_compatible"
model = "qwen3:32b"
base_url = "http://localhost:11434/v1"
api_key_env = "OLLAMA_API_KEY"
temperature = 0.0
timeout_s = 120
max_tokens = 2048
```

- `provider` defaults to `openai_compatible`. It covers Ollama, vLLM, llama.cpp's server, LM Studio and hosted services that speak the OpenAI chat API. A new model or server needs no code, only `base_url` and `model`.
- `api_key_env` names the environment variable that holds the key, if the server needs one. Keys are never written to a file or a log.
- `max_tokens` is the limit per answer.
- Optional settings: `max_context` (the model's context, in tokens; with `ollama`, 65536 by default, and each request asks Ollama for what the conversation needs, in powers of two, at least `options = { min_context = 32768 }` and never more than the model's own context; a larger `max_context` lets long reports through but costs a local model memory), `tool_calls = false` (the model has no native tool calling, so APEX prompts for JSON), `json_mode = true` (the server can constrain answers to JSON) and `options` (the provider's own options, e.g. `options = { keep_alive = "30m" }`).

Run `apex agent providers` to list the providers your install offers.

### What a provider receives, and how to keep it local

An agent sends the model what it needs to judge a run: facts that APEX's code computes from your runs and cases, such as configuration settings, data-mismatch and metric values, QA/QC results, the names of wells and data types, and the paths of runs and cases. The web UI's assistant also sends your chat messages and its tools' answers. Where this goes depends on the provider:

- A **hosted service** (any `base_url` that is not your own machine or network, `anthropic`, or `claude_code`) receives these facts on its servers, under that service's terms. Use one only for data you may share with it.
- A **local server** (Ollama, vLLM, llama.cpp or LM Studio on `localhost` or your own network) keeps everything on machines you control. Set `base_url` to it, as in the example above.
- **No model at all:** `--no-llm` on `apex agent monitor`, `diagnose`, `plan`, `report` and `plot` uses the agent's rules only, and `apex agent draft-optimization` never asks a model. These agents fall back to their rules whenever the model is unavailable or its answer is invalid. `apex agent ask` and the web UI's assistant need a model: without one they say so and answer nothing.

API keys go only into a request's authentication, never into prompts or traces (a trace records the name of the key's variable, not the key). Every call is recorded, prompt included, in an `agents` folder: the run's, the optimization's or the study's for the agents that judge them, `<services root>/chats/<chat id>/` for the web UI's assistant, and `<services root>` for `apex agent ask` and the planner. So you can always see what was sent.

## Step 2: ask the agent

```bash
apex agent ask "Which localization methods does APEX have, and which is the default?"
apex agent ask "Validate /data/cases/m5spots/config.toml and tell me its cost" --model glm-4.6
apex agent ask "..." --no-tool-calls           # a model without native tool calling
```

The agent answers with APEX's tools, the same tools that MCP hosts and external planners use. So approval applies to it as to any client: it cannot submit work above the service's approval limits until a person approves the pending token with `apex approve TOKEN`. The limits are the ones in `service.toml`. A token is issued when `validate` accepts work that is above them.

## Step 3: check a run with the monitor

`apex agent monitor` checks whether a run is healthy and whether it should stop. `RUN` is a run directory, a run id or a job id.

```bash
apex agent monitor examples/linear_gaussian/runs/linear_gies  # once: rules and the model's judgement
apex agent monitor 20260929T071500-3f2a --watch   # a job: again after every newly scored forecast
apex agent monitor RUN --no-llm                   # rules only, no model
apex agent monitor RUN --watch --act              # also stop the job on a stop recommendation
apex agent monitor RUN --json                     # everything, for scripts
```

`--watch` keeps checking a job; press Ctrl+C to end it.

### Words used below

- A **trial** is a proposed update of an iterative method. It is **accepted** if it improves the data mismatch and **rejected** if not. An **iterate** is the state after an accepted trial.
- The **data mismatch** measures how far the simulated data is from the observations. The **normalized** data mismatch is its mean square, about 1 at the noise level.
- The **monitor's recommendation** is one of `continue` (nothing to do), `investigate` (look at the findings yourself) or `stop` (the run should end). A `stop` is only meaningful for a running iterative method.
- The **status** of a report is `info`, `warning` or `critical`.

### How it works

1. **Facts.** Code reads the run directory. It computes the accepted data mismatch trajectory, the spread of each variable relative to the prior's, the failed members, and the number of consecutive rejected trials at the end of the run. It also takes the normalized data mismatch, the damping and, with cross validation, the CV curve.
2. **Rules.** Named thresholds turn the facts into findings. Change them in `[agents.monitor]` of `service.toml`.

   | Finding | When (default threshold) |
   |---|---|
   | `ensemble_collapse` | a variable's spread < 0.05 of the prior's |
   | `member_failures` | more than 20 % of the members failed |
   | `damping_runaway` | 5 rejected trials in a row (iterative methods) |
   | `stagnation` | accepted data mismatch changed < 1 % (relative) between the last 3 accepted iterates (iterative methods) |
   | `overfitting` | normalized data mismatch < 0.5: the noise is fitted |
   | `poor_fit` | a finished run's normalized data mismatch > 10 000 |
   | `cv_past_minimum` | no CV improvement for 2 accepted iterates |

   For example, a mismatch of 2.00 that is still 1.99 three accepted iterates later has changed by 0.5 %, so `stagnation` fires.
3. **Judgement.** The model gets the facts, the rules' report and read-only tools. It returns its own report, with evidence for each finding. If it fails, or no model is set, the rules decide. The model's report is then held to the facts:
   - a finding with a rule's code that the rules did not find is dropped;
   - a model's `stop` that no rule finding backs becomes `investigate`.

   Each correction is printed (`corrected: ...`).
4. **Acting.** The monitor is advisory. With `--act`, a `stop` recommendation for a running iterative run asks you at the terminal first, showing the findings. If you answer yes, the job stops at its next checkpoint. A no, or no answer (e.g. under `nohup`), leaves it running.

### In the job service

With `[agents.monitor] auto = true` in `service.toml`, the job service runs the monitor for you. It checks every running history-matching job after each newly scored forecast, in a background thread, so scheduling never waits for the model. Each report goes into the job's events as `monitor_report`. When the status is in `notify_on` (default warning and critical), it calls your `[notifications]` command or webhook. This mode is advisory only: it never stops a job.

```toml
[agents.monitor]
auto = true
notify_on = ["warning", "critical"]

[notifications]
command = "notify-send 'APEX monitor' 'A job needs a look'"
```

## Step 4: diagnose, plan and report

These three agents work like the monitor: code computes the facts, named rules decide without a model, and the model's answer is held to the facts before you see it. All three take `--no-llm` (rules only) and the model options of `apex agent ask`.

### Diagnosis: what went wrong, and what to change

```bash
apex agent diagnose examples/linear_gaussian/runs/linear_gies  # a run directory, run id or job id
apex agent diagnose RUN --json                 # the facts, both diagnoses, the corrections
```

The facts are the monitor's facts, plus the QA/QC of the posterior (see `apex qaqc`), the failed members grouped by reason, and the method, strategy and localization. In the QA/QC, a **Mahalanobis score** measures how far one datum lies from the forecast, in units of its uncertainty. The **gain ranking** orders the data by how much they moved the parameters.

Thresholds are in `[agents.diagnosis]`:

| Cause | When (default threshold) |
|---|---|
| `data_outside_forecast` | more than 20 % of a data type's observations outside the forecast range widened by two observation-error standard deviations, and more than chance would put there (binomial test at `coverage_significance`, 0.01) |
| `model_deficiency` | a Mahalanobis score above 10.83 (the χ²(1) 99.9th percentile) |
| `overcorrection` | more than 10 % (`overcorrection_percent`) of a group's parameters moved far from the prior |
| `recurring_failure` | 3 or more members failed for the same reason |

The monitor's findings are also causes. Recommendations address the causes, e.g. adaptive localization for a collapse or an overcorrection, `[validation]` for overfitting, checking data errors and the prior for a model deficiency (never dropping the data), a larger initial damping for a damping runaway.

A recommendation that is a setting carries a **JSON merge patch** over the run's configuration, the same kind a study variant uses. Every patch is applied to the run's configuration and validated by the service before it is shown; patches that fail are dropped, with the error, and the advice stays. A cause with a rule's code that the rules did not find is dropped.

The agent changes nothing. To use a patch, put it in a study variant (see the planner below), or in the web UI ask the assistant to draft a run with it (see below).

### Planning: an experiment for a goal

```bash
apex agent plan CASE.toml --goal "does adaptive localization improve the RMSE?" \
    --budget 20000                              # simulator runs for the whole study
apex agent plan CASE.toml --goal "..." --submit # submit after you confirm
apex agent plan CASE.toml --goal "..." --approval-token TOKEN   # after `apex approve TOKEN`
```

Replace `CASE.toml` with one of your own case configurations.

- **Facts:** the case's validated plan (the simulator runs one run takes), its method and localization, whether it is synthetic, the plugins, and the budget (default: the approval limit of `service.toml`).
- **The model proposes** the algorithms to compare, if any, as complete method entries written from their own schemas. It also proposes variants (JSON merge patches over the case; the baseline has none), replicates and metrics.
- **Held to the facts:** code builds the study, validates every run with the service and checks the total cost against the budget. An invalid or over-budget proposal goes back to the model with the errors, up to `[agents.planner] max_revisions` (2) times. After that, the rules plan: the baseline, and adaptive localization if the case has none, with as many replicates as the budget allows (at most `max_replicates`, 5). Replicates are repeated runs with different random seeds. They are **paired**: the baseline and each variant use the same seeds, so a difference between them comes from the variant and not from the seed.
- **Output:** a study TOML beside the case (set the path with `--out`), with the hypothesis, the reasoning, the simulator choices and the cost in its header. Run it with `apex study run`.
- **Spending:** only with `--submit`, after you confirm at the terminal. Submission goes through the tools: `validate` issues an approval token, and `submit_study` uses it. Above the service's approval limits the token is pending: the command prints `apex approve TOKEN`, and you submit again with `--approval-token`.

### Reports: a narrative of the results

```bash
apex agent report RUN
apex agent report --study NAME
apex agent report RUN --out summary.md
```

The facts are, for a run, its summary, the metrics per step and the monitor's and diagnosis rules' findings; for a study, its comparison. The model writes the sections and lists every number the prose relies on, with its source. A key number that matches no fact is dropped and listed under *Corrections* at the end of the report, so a reader knows which prose to check. Without a model, a template reports each metric from the prior to the posterior, the health findings, and a study's differences.

### Figures: the plotter

```bash
apex agent plot RUN --preset brugge_classic
apex agent plot RUN --request "the RMSE box plots and the layer maps"
```

The plotter draws a run's result figures into `RUN/report/plots/`, where the run's page in the web UI shows them. The facts are what the run holds and every figure plugin's options. The model, or without one the rules, chooses figures or a preset, and code refuses a choice the run cannot give before anything is drawn. See [figures](figures.md).

### Bringing your own case: the case-converter

```bash
apex agent convert import CASE.DATA --out my_case
apex agent convert import CASE.DATA --out my_case --history observed.csv --request "match porosity with ES-MDA"
apex agent convert skills
```

The case-converter turns an OPM Flow or Eclipse deck into an APEX history-matching case that validates. It writes the configuration, a deck template, the observations and the deck's included files into `my_case/`. Nothing is simulated or submitted: run the case with `apex run my_case/config.toml` once you have read it.

How the case is put together:

- **Facts.** Code reads the deck: grid, per-cell properties, wells, report dates and the observed history. The history comes from `--history` (a CSV with columns `time,datatype,value`, or a summary file), or else from the deck's own `WCONHIST`/`WCONINJH`. A history rate set at one report date is observed at the next, where the simulator reports it.
- **Choices.** The model chooses the state variables, the observations with their data errors, and the method. It can only name properties set as arrays and vectors that have data, and code refuses anything else.
- **Rules.** Without a model, ln PERMX and PORO become the state. Every rate and pressure with data becomes an observation. Data errors are 10 % of each value, with a floor of 1 % of the vector's largest value. The method is GIES with `rlmmac` and `adaptive_prior`, with 100 members and 10 iterations (`[agents.case_converter]`).
- **Cells that are 0 in the deck stay 0.** Inactive cells and barriers are not matched.
- **Validation.** The draft is validated. Errors go back to the model up to `max_revisions` times (3), and then the rules convert.

Every value the converter assumed is listed in `my_case/CONVERSION.md`: the bounds, and the prior's spread and correlation range. Read it before running. A converted case is `real`: its prior is centred on the deck's values, so replace it with a prior of your own where you have one.

Synthetic cases are created from a description:

```bash
apex agent convert create --out box --request "a 30 x 30 x 4 five-spot, 200 mD, monthly for three years"
apex agent convert observe box   # if you answered no to the truth run
```

- **The description.** The model turns the request into a case description: a box grid, the wells and their controls, the report steps, an oil-water fluid, and the statistics of ln PERMX and PORO. Code checks it, for example that the wells lie inside the grid. Without a model, the rules give a five-spot: 21 × 21 × 3 cells, four producers and an injector, monthly reports for two years.
- **What is written.** The converter writes the deck, a prior of Gaussian fields from those statistics, and a truth drawn from the same prior with a seed of its own.
- **The observations.** After you confirm (or with `--yes`), the truth runs once on the simulator, and its well rates and pressures, with noise, become the observations. A vector that stays 0 is not observed. The case is then `synthetic` and validates; `case_spec.json` keeps its description.

PET configurations go both ways, with no model:

```bash
apex agent convert from-pet CONFIG.toml apex_case   # as apex pet-import
apex agent convert to-pet apex_case/config.toml pet_case   # pet.toml, data.csv, var.csv, the runfile, the prior
```

An APEX case written for PET reads back to the same APEX case, and PET's own reader accepts it. What PET cannot run as APEX does is refused with the reason, never approximated:
- data transforms, geophysics, parameterizations, model error and cross validation;
- localization and damping;
- GIES: the PET fork that runs it has a configuration format of its own, which is not written yet;
- APEX's own methods, strategies and options PET has no place for;
- template values, one-sided bounds and joint priors;
- variogram priors PET's sampler would not reproduce. Only isotropic fields, or fields anisotropic along j, on a square grid are written; give a stored ensemble otherwise.

LM-EnRML and GN-EnRML run in both, but they stop by different rules, so the converter notes that the number of iterations may differ.

ERT cases go both ways too (`apex ert convert` and `apex ert export`, with the `ert` extra):

```bash
apex agent convert from ert CASE.ert apex_case
apex agent convert to ert apex_case/config.toml ert_case
```

Each simulator, and PET and ERT, has a converter skill; `apex agent convert skills` lists them with what each can do. A package that adds a simulator adds its skill (entry point `apex.converter_skills`).

### Giving a case a prior: the geomodeler

```bash
apex agent geomodel my_case/config.toml --reference my_case/reference.npz
apex agent geomodel my_case/config.toml --request "ln PERMX around 5 with a 10-cell range, porosity 0.2 ± 0.03"
```

The geomodeler gives a case's state variables a prior. It works on every variable not read from a stored ensemble, and writes the prior into a new configuration beside the old one (`config_prior.toml`).

- **Skills.** `continuous` draws Gaussian fields of properties such as ln PERMX and PORO. `facies_pluri` draws facies by proportion from a latent field, conditioned to the wells' facies, with each facies' property values.
- **Parameters.** Your values come first. Then come the mean and spread of a reference field (`--reference`: an `.npz` of arrays in state space, e.g. the deck's values). Anything else is proposed and listed as assumed.
- **Rules.** Without a model, each variable gets a Gaussian field around the reference (or the middle of its bounds), with a range of a quarter of the grid (`[agents.geomodeler]`).
- **The drawn sample is checked.** Code draws a small sample of the new prior and compares its mean and spread, or its facies proportions, with what was asked. Bounds that clip a field show up here. A prior that fails goes back to the model to revise.

A run never calls the geomodeler: a prior is always a recorded section or file, so runs stay reproducible.

### Optimizations

The monitor, diagnosis, planning and report agents also handle optimization directories. Their rules read the accepted steps, the objective at the start, at its best and last, the simulated and failed evaluations, and the optimizer's message. For several objectives (`nsga2`) the run is judged by its front. Thresholds are in `[agents.monitor]`, e.g. `optimization_stall_window` and `optimization_stall_tolerance` (default 0.1 %). A model's finding with a rule's code that the rules did not find is dropped. See [optimization](optimization.md) for the optimizers and their settings.

## Step 5: the assistant in the web UI

The web UI's assistant (see [the web UI](ui.md)) is a conversation on the same loop. You type a request in the chat, for example:

> a config to run history matching on Norne with rlmmac and adaptive_prior

The assistant, not you, calls the tools below. You see its answer and cards. A card that proposes work (`propose_action`) is only launched when you press it.

The assistant uses:
- the last `[chat] context_turns` messages (default 20);
- `[chat] max_model_calls` (10) model calls per answer, plus one per failed tool call up to `refusal_turns` (4);
- the read-only registry tools (tools that list and describe APEX's plugins, runs and schemas; see [the tools page](tools.md) for the full list);
- the agents as local tools: `monitor_run`, `diagnose_run`, `write_report` and `list_agents`, and `case_info` (a case's configuration, data, grid and well completions).

For planning, `plan_study` checks the design with the planning agent's code and writes a draft, `revise_draft` changes it, and `propose_action` shows a card that only you can launch. `draft_run(run, patch)` writes a run's configuration with a change, such as a diagnosis's recommended patch, as a validated draft. `find_cases(query, folder)` lists the configurations in the workspace whose path contains the query, with each one's kind (run, study, prediction, optimization), method, damping, localization and simulator.

For optimizations, `draft_optimization` writes an optimization drafted from a history-matching case (see [optimization](optimization.md#drafted-from-a-history-matching-case)). You must give prices, the discount rate and the producers' pressure bounds in your message, for example *"draft an optimization of NPV from this case, oil price 70, gas price 3, discount rate 8 %, producer BHP bounds 150 to 250 bar"*. Without them the draft is refused, naming what is missing.

## Step 6: visibility

Each agent is `public` or `private`. The web UI and its assistant show and name only public ones. `service.toml` overrides an agent's default:

```toml
[agents.visibility]
monitor = "private"
```

`GET /v1/agents` lists every agent with its visibility. Guests get only the public ones.

## Worked example

Check a finished run and ask what to change, with no model and no spending:

```bash
apex agent monitor examples/linear_gaussian/runs/linear_gies --no-llm
apex agent diagnose examples/linear_gaussian/runs/linear_gies --no-llm
apex agent report examples/linear_gaussian/runs/linear_gies --no-llm
```

The monitor prints its findings, status and recommendation. Suppose it reports `stagnation`: the data mismatch stopped moving. The diagnosis then lists causes with recommendations and, where a recommendation is a setting, a validated patch. Copy that patch into a study variant, plan the study with `apex agent plan CASE.toml --goal "..."` and run it with `apex study run`. The report writes a narrative of the result. Once this works, add a model in `[agents.llm]` and repeat without `--no-llm`.

## Pitfalls

- **A model without native tool calling.** Set `tool_calls = false` in `[agents.llm]`, or pass `--no-tool-calls`. APEX then prompts for JSON.
- **Small context windows.** If answers are cut or refused, set `max_context` and `max_tokens` to what the model supports.
- **A missing key.** `api_key_env` names a variable, not the key. The variable must be set in the environment that runs APEX.
- **Trusting the model over the rules.** A model's finding the rules did not find is dropped, and a model's `stop` that no rule backs becomes `investigate`. Read the printed `corrected:` lines.
- **Expecting the monitor to act.** It only advises. `--act` asks you first, and `auto = true` in the job service never stops a job.
- **Spending by accident.** `apex agent plan` only writes a study. It spends budget only with `--submit` (and your confirmation), or `--approval-token`.

## How the loop copes with weaker models

- **Tool calls.** The loop uses native tool calling where the model has it. Otherwise it uses *prompted JSON*: the tool schemas go in the prompt, and the model answers `{"tool": ..., "arguments": {...}}` or `{"answer": {...}}`. A refused call is recorded like any call, and its reason goes back to the model.
- **Validation.** A pydantic model validates every answer before APEX acts on it. An invalid answer gets its error back and is retried a limited number of times.
- **Fallback.** A rule-based fallback decides when the provider is unreachable, times out, or refuses, when the answers stay invalid, or when the turns run out. Every task has such a fallback, and its thresholds are named configuration. The fallback is recorded as having decided.
- **Context.** The conversation is trimmed to the model's context, oldest tool results first.
- **Last turn.** The last turn offers no tools and asks for the answer, so a task always ends with a decision.
- **Turns.** A task has a limited number of model calls (`max_turns`). With `refusal_turns`, each refused tool call adds one, up to that many.

## What to read next

- [The tools page](tools.md): the tools the agents and outside frameworks call.
- [The web UI](ui.md): the assistant and the agents' cards.
- [Optimization](optimization.md): the optimizers the agents monitor, diagnose and plan.

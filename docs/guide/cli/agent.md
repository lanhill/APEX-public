---
audience: public
items:
  - cli:apex agent ask
  - cli:apex agent convert create
  - cli:apex agent convert from
  - cli:apex agent convert from-pet
  - cli:apex agent convert import
  - cli:apex agent convert observe
  - cli:apex agent convert skills
  - cli:apex agent convert to
  - cli:apex agent convert to-pet
  - cli:apex agent diagnose
  - cli:apex agent draft-optimization
  - cli:apex agent geomodel
  - cli:apex agent monitor
  - cli:apex agent plan
  - cli:apex agent plot
  - cli:apex agent providers
  - cli:apex agent report
  - cli:apex approve
  - cli:apex ert import
  - cli:apex optimize
  - cli:apex pet-import
  - cli:apex predict
  - cli:apex report
  - cli:apex resume
  - cli:apex run
  - cli:apex stop
  - cli:apex submit
  - cli:apex summary
  - config:history_match:prior
  - config:optimization_study:study
  - config:prediction:prediction
  - config:service:service
  - config:study:study
  - plugin:executors:ert
  - plugin:figure_presets:brugge_classic
  - plugin:metrics:model_change
  - plugin:optimizers:enopt
  - plugin:response_objectives:npv
  - plugin:simulators:ert
  - plugin:simulators:opm_flow
  - plugin:surrogate_designs:run
---

# The `apex agent` commands

## Goal

By the end of this lesson you will know what each `apex agent` command does and which ones can run on rules alone, with no model. You will use them to ask questions, check runs, understand failures, draw figures, write reports, plan studies, convert cases, give a case a prior and draft optimizations. You will also meet the commands that work alongside them: `apex run`, `apex resume`, `apex stop`, `apex submit`, `apex approve`, `apex optimize` and `apex report`.

The agent commands are `apex agent ask`, `apex agent monitor`, `apex agent diagnose`, `apex agent plot`, `apex agent report`, `apex agent plan`, `apex agent draft-optimization`, `apex agent geomodel`, `apex agent providers`, and the converters `apex agent convert create`, `apex agent convert from`, `apex agent convert from-pet`, `apex agent convert import`, `apex agent convert observe`, `apex agent convert skills`, `apex agent convert to` and `apex agent convert to-pet`.

The lesson also shows three configuration sections: `study:study` (the `[study]` section of a study), `optimization_study:study` (`[study]` with `kind = "optimization"`) and `service:service` (the `[service]` section).

## Prerequisites

- A working `apex` installation. Type every command from the APEX folder as `uv run apex ...`.
- A services root. By default this is `$APEX_HOME`, otherwise `~/.apex/service`. `service.toml` lives there. Most agent commands accept `--root` to point somewhere else (see the note under Step 2).
- For the commands that consult a model: a provider and a model. Unless you pass `--provider` and `--model`, APEX uses the `provider` and `model` options in the `[agents.llm]` section of your configuration.
- For the commands that look at a run: a run directory, a run id or a job id. You get a run by starting one with `uv run apex run CONFIG`, or in the background with `uv run apex submit CONFIG`.

## Step 1: See which providers exist

```bash
uv run apex agent providers
```

This lists the LLM providers. It takes no options. Use the names it prints with `--provider`.

## Step 2: Choose how the model is reached

Most agent commands share the same model options:

| Option | Meaning |
|---|---|
| `--provider` | Provider. Default: `provider` in `[agents.llm]`. |
| `--model` | Model. Default: `model` in `[agents.llm]`. |
| `--base-url` | Server address. Set it when the model is served somewhere other than the provider's usual address. Leave it out otherwise. |
| `--api-key-env` | The name of the environment variable that holds the key. Never put the key itself here. |
| `--no-tool-calls` | Use this when the model has no native tool calling; APEX then prompts it for JSON. Default: off. |

The commands `apex agent convert create`, `apex agent convert import`, `apex agent diagnose`, `apex agent geomodel`, `apex agent monitor`, `apex agent plan`, `apex agent plot` and `apex agent report` also accept `--no-llm`. It means rules only: no model is asked. Default: off, so a model is asked. `apex agent draft-optimization` never uses a model.

**What the model receives.** An agent sends the model facts computed from your runs and cases: configuration settings, data-mismatch and metric values, well and data-type names, and paths. A hosted provider (`anthropic`, `claude_code`, or a `--base-url` outside your own machines) receives them on its servers, under its terms. A local server (Ollama, vLLM, llama.cpp on `localhost` or your network) keeps them on your machines, and `--no-llm` sends nothing. See [what a provider receives, and how to keep it local](../../agents.md#what-a-provider-receives-and-how-to-keep-it-local).

**About `--root`.** All agent commands accept `--root` except `apex agent providers` and `apex agent convert skills`. Of the other commands in this lesson, `apex submit` and `apex approve` accept it too, but `apex run`, `apex resume`, `apex stop`, `apex optimize` and `apex report` do not: they work on the directory or configuration you give them.

## Step 3: Ask a question with `apex agent ask`

```bash
uv run apex agent ask "Which runs have finished?"
```

The `question` argument is required. The agent may use APEX's tools to answer.

- `--tools` offers APEX's tools (the default). `--no-tools` turns them off, so the model answers on its own.
- `--max-turns` limits the number of model calls. Default: 8.
- `--client` is the client name used for quotas and the call log. Default: `agent`.
- `--root`, `--provider`, `--model`, `--base-url`, `--api-key-env` and `--no-tool-calls` work as described above.

## Step 4: Check health with `apex agent monitor`

```bash
uv run apex agent monitor RUN_ID
```

`apex agent monitor` checks the health of a run, an optimization or a surrogate build. It applies rules and adds a model's judgement (with `--no-llm`, only the rules). The `target` is a run directory, a run id or a job id; the path of the run's directory (the one you would give to `apex resume`) always works.

Read the result as an assessment of whether the job looks healthy. For a running iterative run (a run that proceeds in iterations), it also says whether stopping is recommended.

What to do next:

- If the job looks healthy, let it finish.
- If it does not, run `apex agent diagnose` (Step 5) to learn what to change.
- If it recommends a stop, stop the run (see below). Change the configuration if needed, then continue with `apex resume` or start again with `apex run`.

Options:

- `--watch` checks again whenever a new forecast is scored, until the job ends. `--interval` sets the seconds between looks (default 60) and only matters together with `--watch`. `--no-watch` turns watching off (the default).
- `--act` offers to stop the job when the check recommends a stop. This applies to running iterative runs only, and you are asked to confirm first. `--no-act` turns it off (the default).
- `--no-llm` gives a rules-only check.
- `--json` prints the result as JSON, which is handy for scripts.

You can also stop a run yourself with `apex stop RUN_DIR`. It stops at the next checkpoint, and `apex resume RUN_DIR` continues it later:

```bash
uv run apex stop RUN_DIR
uv run apex resume RUN_DIR
```

`apex resume` also accepts `--max-steps` (stop after this many analysis steps) and `--progress` / `--no-progress` (progress on standard error; default on).

## Step 5: Understand a failure with `apex agent diagnose`

```bash
uv run apex agent diagnose RUN_ID --no-llm
```

`apex agent diagnose` explains what went wrong in a run and what to change. It is advisory: it changes nothing. It takes the same `target` as `monitor`, and supports `--no-llm`, `--json` and the model options.

To act on the advice, edit the run's configuration as suggested and start the run again with `apex run CONFIG`. Treat the advice as a suggestion to check, not a final verdict.

## Step 6: Draw figures with `apex agent plot`

```bash
uv run apex agent plot RUN_ID --preset brugge_classic
```

`apex agent plot` draws a run's result figures. You choose what to draw in one of three ways:

- `--preset NAME` draws a named preset, for example `brugge_classic`. No model is used.
- `--with` switches on an option of a preset. Repeat it for several, as in `--with model_change --with normalized` (for `brugge_classic`).
- `--request "..."` says in words what to draw. If you give neither, the classic set is drawn.

Figures go to `RUN/report/plots` unless you set `--output`. `--no-llm` gives a rules-only run.

The `brugge_classic` preset is the classic figure set of an ensemble history-matching study. Its options are `histogram_variable` (the state variable of the `nn_hist/` figures; default `nn_params`), `normalized` (also draw `normalized_data_mismatch`; default false) and `model_change` (also draw `model_change`; default false). The `model_change` figure comes from the `model_change` metric, which scores the change of each member's parameters since the ensemble it was updated from.

A related command is `apex report RUN_DIR`. It prints the data mismatch history and performance metrics of a run or a prediction. With `--plots` it also writes the figures (PNG and PDF) to `--output` (default `RUN_DIR/report`); `--no-plots` turns that off (the default), and `--font-size` sets the font in points (default 12.0).

```bash
uv run apex report RUN_DIR --plots --font-size 14
```

## Step 7: Write a report with `apex agent report`

```bash
uv run apex agent report RUN_ID --out report.md
```

`apex agent report` writes a narrative report, in Markdown, of a run, an optimization, a surrogate build or a study.

- Give a `target` (a run, optimization or surrogate build directory, a run id or a job id) to report on one of those, or `--study` to report on a study instead. Both are optional on the command line, but you need one of them.
- `--out` says where to write the Markdown.
- `--no-llm` gives a rules-only report.

To report on a study, write the study first (Step 8 shows how to write one) and then point `--study` at it:

```bash
uv run apex agent report --study study.toml --out study_report.md
```

## Step 8: Plan a study with `apex agent plan`

```bash
uv run apex agent plan case.toml --goal "Find out which parameters the history constrains"
```

`apex agent plan` proposes a study for a case and a goal. It submits the study only if you confirm.

- `config` (required) is the case: a history-matching or optimization configuration.
- `--goal` (required) states what the experiment should find out.
- `--budget` is the number of simulator runs for the whole study. By default it is the approval limit of `service.toml`, the file in your services root. A submission above that limit needs approval before it starts.
- `--out` is where the study is written. By default it goes beside the case.
- `--submit` submits the study after you confirm, through APEX's tools. `--no-submit` turns it off (the default).
- `--approval-token` submits with a token that `apex approve` approved after an earlier `--submit` was left pending. It implies `--submit`.
- `--no-llm` and `--json` are also available.

### The approval sequence

The first `--submit` needs no token. If the submission needs no approval, it starts right after you confirm. If it does need approval (for example, the budget is above the approval limit), it is left pending instead of starting, and you get a token for it. The sequence is then:

1. Run `apex agent plan` with `--submit` and confirm when asked. The pending submission and its token are reported.
2. Approve it with `apex approve TOKEN`.
3. Submit again with `--approval-token TOKEN`.

```bash
uv run apex agent plan case.toml --goal "Find out which parameters the history constrains" --submit
uv run apex approve TOKEN
uv run apex agent plan case.toml --goal "Find out which parameters the history constrains" --approval-token TOKEN
```

The plain `apex submit` command works the same way. It runs a configuration, or resumes a run with `--resume`, in the job service's background. Useful options:

- `--kind`: `run`, `prediction`, `optimization` or `surrogate` (default: by the file).
- `--max-steps`: stop after this many steps (iterations of an optimization, member runs of a surrogate build).
- `--code`: `live`, `frozen` (the last commit; for a resume, the run's own) or a commit. Default: `cli_code` in `[jobs]`, which is `frozen`.
- `--priority`: `high`, `normal` or `low`. Default `normal`.
- `--yes`: approve a large job without asking.
- `--approval-token`: a token approved with `apex approve`.
- `--wait` / `--no-wait`: wait for the job to end. Default off.
- `--force-new` / `--no-force-new`: run again even if an identical run exists. Default off.
- `--client`: the client name, for quotas and grants. Default `user`.
- `--annotation key=value`: repeatable; `experiment=NAME` lets a grant of `--client` for that experiment approve the job.
- `--root`.

```bash
uv run apex submit case.toml --priority low --wait
```

A surrogate build can use the `run` surrogate design, which takes the forecasts a history-matching run already holds (its `run_dir`), so no simulator runs.

### What a study file looks like

A study is described by a `[study]` section. It needs:

- a `name`;
- at least one case under `cases` (a case name mapped to a history-matching configuration);
- at least one `[[study.variants]]` entry. A variant is one change applied to every case. Its optional `patch` is a JSON merge patch laid over each case's configuration.

Other options:

- `seed` (default 0): the seeds of replicates 1, 2, ... derive from it.
- `replicates` (default 1): seeds per case and variant.
- `metrics`: a list of `'metric:label'` to compare at each run's posterior.
- `simulator`: `opm_flow` or `eclipse`; runs every case's deck on that simulator instead of the case's own.
- `methods`: a name mapped to a complete algorithm. Without it, each case keeps its own.
- `predictions`: prediction configurations without `run_dir`, run after each run.

A budget goes in `[study.budget]`: the study is refused before anything runs when the plan exceeds `max_simulator_runs`.

This study runs one case twice with its own settings (`base`) and again with a patch that limits the method to 5 iterations (`short`):

```toml
[study]
name = "compare_seeds"
seed = 0
replicates = 2

[study.budget]
max_simulator_runs = 200

[study.cases]
case_a = "case/config.toml"

[[study.variants]]
name = "base"

[[study.variants]]
name = "short"
patch = { method = { max_iterations = 5 } }
```

An optimization study is the same `[study]` section with `kind = "optimization"`. Its cases are optimization configurations, its `metrics` are such as `objective:final`, `objective:best`, `evaluations:all` and `iterations:all`, and its `methods` are complete `[optimizer]`-style sections.

```toml
[study]
kind = "optimization"
name = "compare_optimizers"
metrics = ["objective:final", "evaluations:all"]

[study.cases]
npv = "case/optimization.toml"

[[study.variants]]
name = "base"
```

The job service itself is configured in `[service]`, with `max_concurrent_jobs` (default 2), `poll_s` (default 1.0), `max_retries` (default 2) and `routines` (default true):

```toml
[service]
max_concurrent_jobs = 2
poll_s = 1.0
max_retries = 2
routines = true
```

## Step 9: Convert cases with `apex agent convert`

The `convert` commands translate cases between formats. Those marked "nothing runs" only write files.

Start by listing the converter skills and what they can do:

```bash
uv run apex agent convert skills
```

- `apex agent convert to SKILL SOURCE OUTPUT` translates a case into a skill's format (for example `pet` or `ert`). Nothing runs.
- `apex agent convert from SKILL SOURCE OUTPUT` translates a case from a skill's format. Nothing runs.
- `apex agent convert to-pet CONFIG OUTPUT` writes an APEX case as a PET case (`pet.toml`, data, runfile, prior). Nothing runs.
- `apex agent convert from-pet PET_CONFIG OUTPUT` writes a PET case (a `.toml`) as an APEX case, as `apex pet-import` does. Nothing runs.
- `apex agent convert import CASE --out OUTPUT` converts a case, such as an OPM Flow or Eclipse `.DATA` deck, into an APEX history-matching case that validates. Nothing runs. `--history` gives observed data (a CSV with `time,datatype,value`, or a summary file; default: the deck's `WCONHIST`/`WCONINJH`), `--request` says in words what to match, `--name` names the run, `--skill` picks the converter skill, and `--json` prints the result as JSON.
- `apex agent convert create --out OUTPUT` creates a synthetic case: deck, truth, prior and observations from one truth run. `--request` describes the case in words (default: a five-spot), `--name` names it, `--skill` sets the simulator's skill (default `opm_flow`), and `--observe` / `--no-observe` runs the truth once now to make the observations (default on). `--yes` runs the truth without asking.
- `apex agent convert observe CASE` runs a created case's truth once and writes its observations, for a case `convert create` wrote with `--no-observe`. `--yes` runs it without asking.

`create` and `import` also take `--no-llm` and the model options. If you have no deck of your own, `apex agent convert create` is the way to get one to practise on.

```bash
uv run apex agent convert create --out synthetic_case --no-llm --yes
```

The simulators a converted case can name include `opm_flow` (it needs a `deck_template_file`) and `ert` (it needs a `case_file`, `fields` and `eclbase`); an `ert` case runs through the `ert` executor, which runs a forecast's members as one ERT ensemble experiment, locally.

## Step 10: Give a case a prior with `apex agent geomodel`

```bash
uv run apex agent geomodel config.toml --request "Smooth permeability with a long correlation length"
```

`apex agent geomodel` gives a case's state variables a prior, checked on a drawn sample. It writes a new configuration to `--out` (default `<config>_prior.toml`).

- `--request` states the prior you want, in words (for the model).
- `--reference` is an `.npz` of reference fields in state space, such as the deck's values. The prior shares their mean, std and variogram without reproducing them.
- `--data` is hard data to estimate from and condition to: a CSV of `i,j,k,property,value` (1-based cells; values in the deck's units; for the property `facies`, codes).
- `--no-llm` gives rules only, and the model options apply.

## Step 11: Draft an optimization with `apex agent draft-optimization`

`apex agent draft-optimization` drafts an optimization of the forecast of a history-matched case. It does not use a model.

**Inputs.**

- `case` (required): the history-matching case, a deck simulator, with a finished history match (its posterior is the model). You need a deck; if you have none, create a case with `apex agent convert create` (Step 9), which also observes its truth; if you passed `--no-observe` or declined its question, run `apex agent convert observe` next. Then run its history match.
- `--forecast-end` (required): the last date of the forecast, in ISO form (`YYYY-MM-DD`). It must come after the history. The history ends at the last date of the deck's `WCONHIST`/`WCONINJH` data (or of the observed data the case was built with); choose a later date.

**Economics.** APEX has no default for prices, so choose them yourself. Use your deck's units for volumes and your own currency. These match the `wop`, `wgp`, `wwp`, `wwi`, `disc` and `obj_scaling` options of the `npv` response objective:

- `--wop`: oil price per unit of FOPT, the cumulative oil produced.
- `--wwp`: cost per unit of produced water.
- `--wwi`: cost per unit of injected water.
- `--disc`: yearly discount rate, as a fraction (0.08 is 8 %).
- `--wgp`: gas price. By default gas is not priced.
- `--obj-scaling`: divides the NPV (net present value). The default, -1e6, expresses the NPV in millions and makes the optimizer maximise it.

**Controls and model.**

- `--producer-control`: put producers on `bhp` or `oil_rate`. Default: `bhp`.
- `--producer-bhp`: the producers' pressure bounds as `LOW,HIGH`, used with bhp control, in your deck's pressure unit. Check that unit in your deck before choosing bounds.
- `--control-periods`: the number of periods the forecast is split into. Default: 4.
- `--run`: the history-matching run whose posterior is the model (id, name or path). The posterior holds the model realisations that agree with the history after matching. Without it, the case's latest finished run is used; with none, the draft is refused.
- `--members`: how many posterior members are used. Default: 10.
- `--optimizer`: the optimizer plugin. Default: `enopt`, which steps along an ensemble estimate of the gradient.

**Output.** `--name` names the draft (default `<run>_npv`), `--out` sets the folder (default: beside the case), and `--json` prints the result as JSON.

Open the draft in its folder and review the economics and bounds. Then pass the draft's optimization configuration (the `.toml` it wrote there) to `apex optimize`:

```bash
uv run apex optimize OPTIMIZATION_CONFIG --max-iterations 5
```

`apex optimize` minimises an objective with one of PET's optimizers (EnOpt, GenOpt, SmcOpt and others). `--max-iterations` stops after that many accepted iterations; `--resume DIR` continues an optimization run in a directory; `--progress` / `--no-progress` controls progress output (default on).

## Worked example

Suppose a history-matching run `hm_run` has finished and you want to review it, then draft an optimization of its forecast. The prices below are illustrative; use your own, in your deck's units.

1. Check which providers are available:

   ```bash
   uv run apex agent providers
   ```

2. Check the run's health without a model:

   ```bash
   uv run apex agent monitor hm_run --no-llm
   ```

3. If something looks wrong, ask for a diagnosis:

   ```bash
   uv run apex agent diagnose hm_run --no-llm
   ```

4. Draw the classic figures:

   ```bash
   uv run apex agent plot hm_run --preset brugge_classic --no-llm
   ```

5. Write the report:

   ```bash
   uv run apex agent report hm_run --out hm_run_report.md --no-llm
   ```

6. Draft an optimization using the run's posterior. Here the history is assumed to end before 2032, and the pressure bounds 100,250 are assumed to suit the deck's pressure unit; adjust both:

   ```bash
   uv run apex agent draft-optimization case/config.toml \
     --forecast-end 2032-12-31 \
     --wop 70 --wwp 5 --wwi 2 --disc 0.08 \
     --run hm_run --members 10 \
     --producer-control bhp --producer-bhp 100,250 \
     --name hm_run_npv
   ```

   The draft is written beside the case, because `--out` is not given.

7. Review the draft, then run a short first try of its optimization configuration:

   ```bash
   uv run apex optimize OPTIMIZATION_CONFIG --max-iterations 3
   ```

   Continue it later with `uv run apex optimize --resume DIR`.

Every agent command above ran with `--no-llm` or without a model, so nothing left your machine. Drop `--no-llm` to get the model's judgement; then the model receives facts from your runs (see Step 2).

## Pitfalls

- **Passing the key itself.** `--api-key-env` takes the name of a variable, never the key.
- **Mixing up the switches.** `--no-tool-calls` turns on prompting for JSON; it is not the same as `--no-tools`, which stops the agent from being offered APEX's tools.
- **Expecting a model to be asked with `--no-llm`.** It gives rules only.
- **Passing `--root` to `apex run`, `apex resume`, `apex stop`, `apex optimize` or `apex report`.** They do not accept it.
- **Forgetting `--forecast-end`.** `apex agent draft-optimization` requires it, and the date must come after the history.
- **Using `--producer-bhp` with `oil_rate` control.** The bounds are for bhp control.
- **Expecting `apex agent plan` to submit by itself.** It submits only after you confirm, and only with `--submit` or `--approval-token`.
- **Looking for a token that never came.** A token exists only when the submission needed approval; otherwise the first `--submit` already started the study.
- **Using `--act` outside its scope.** It only offers to stop running iterative runs, and asks you first.
- **Treating the advice as final.** `apex agent diagnose` is advisory.
- **Using `--interval` alone.** It only matters together with `--watch`.
- **Running `apex agent report` with no target.** Give a `target` or `--study`.
- **Expecting a converter to run the simulator.** `apex agent convert to`, `from`, `to-pet`, `from-pet` and `import` run nothing; only `create` and `observe` run a truth, and they ask first unless you pass `--yes`.

## What to read next

- The pages on history matching and optimization, to prepare the cases that `plan` and `draft-optimization` work on, and on `apex run` and `apex optimize`.
- The page on the service configuration, for the `[agents.llm]` section, the `[service]` section and the approval limit in `service.toml`.
- The page on approvals, for `apex approve` and how a pending submission is approved.
- [What a provider receives, and how to keep it local](../../agents.md#what-a-provider-receives-and-how-to-keep-it-local), before you point an agent at a hosted model.

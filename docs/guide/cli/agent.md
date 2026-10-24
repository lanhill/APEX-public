---
audience: public
items:
  - cli:apex agent ask
  - cli:apex agent diagnose
  - cli:apex agent draft-optimization
  - cli:apex agent monitor
  - cli:apex agent plan
  - cli:apex agent plot
  - cli:apex agent providers
  - cli:apex agent report
  - cli:apex approve
  - cli:apex optimize
  - cli:apex report
  - cli:apex run
  - cli:apex stop
---

# The `apex agent` commands

## Goal

By the end of this lesson you will know what each `apex agent` command does, which ones need a language model and which can run on rules alone, and how to use them to ask questions, check on runs, understand failures, draw figures, write reports, plan studies and draft optimizations.

The commands covered are `apex agent ask`, `apex agent diagnose`, `apex agent draft-optimization`, `apex agent monitor`, `apex agent plan`, `apex agent plot`, `apex agent providers` and `apex agent report`. The lesson also touches `apex approve`, `apex optimize`, `apex report`, `apex run` and `apex stop`, which work alongside them.

## Prerequisites

- A working `apex` installation and a services root. By default this is `$APEX_HOME`, otherwise `~/.apex/service`. Every command except `apex agent providers` accepts `--root` to point somewhere else.
- For the commands that consult a model: a provider and a model. Unless you pass `--provider` and `--model`, APEX uses the `provider` and `model` options in the `[agents.llm]` section of your configuration.
- For the commands that look at a run: a run directory, a run id or a job id. You get a run by starting one with `apex run CONFIG`.

## Step 1: See which providers exist

```bash
apex agent providers
```

This lists the LLM providers. It takes no options. Use the names it prints with `--provider`.

## Step 2: Choose how the model is reached

Most agent commands share the same model options:

| Option | Meaning |
|---|---|
| `--provider` | Provider. Default: `provider` in `[agents.llm]`. |
| `--model` | Model. Default: `model` in `[agents.llm]`. |
| `--base-url` | Address of the model server. Set it when the model is served somewhere other than the provider's usual address, for example a local server: `--base-url http://localhost:8000`. Leave it out otherwise. |
| `--api-key-env` | The name of the environment variable that holds the key. Never put the key itself here. |
| `--no-tool-calls` | Use this when the model has no native tool calling; APEX then prompts it for JSON. `--no-no-tool-calls` switches it back off. Default: off. |

The commands `diagnose`, `monitor`, `plan`, `plot` and `report` also accept `--no-llm`. It means rules only: no model is asked. `--no-no-llm` switches it off again. Default: off, so a model is asked. `apex agent draft-optimization` never uses a model.

**What the model receives.** An agent sends the model facts computed from your runs and cases: configuration settings, data-mismatch and metric values, well and data-type names, and paths. A hosted provider (`anthropic`, `claude_code`, or a `--base-url` outside your own machines) receives them on its servers, under its terms; a local server (Ollama, vLLM, llama.cpp on `localhost` or your network) keeps them on your machines, and `--no-llm` sends nothing. See [what a provider receives, and how to keep it local](../../agents.md#what-a-provider-receives-and-how-to-keep-it-local).

## Step 3: Ask a question with `apex agent ask`

```bash
apex agent ask "Which runs have finished?"
```

The `question` argument is required. The agent may use APEX's tools to answer.

- `--tools` offers APEX's tools (the default). `--no-tools` turns them off, so the model answers on its own.
- `--max-turns` limits the number of model calls. Default: 8.
- `--client` is the client name used for quotas and the call log. Default: `agent`.
- `--root`, `--provider`, `--model`, `--base-url`, `--api-key-env` and `--no-tool-calls` work as described above.

## Step 4: Check a run with `apex agent monitor`

```bash
apex agent monitor RUN_ID
```

`apex agent monitor` checks a run's health. It applies rules to the run's facts and adds a model's judgement (with `--no-llm`, only the rules). The `target` is a run directory, a run id or a job id. Read the result as an assessment of whether the run looks healthy, and, for a running iterative run, whether stopping is recommended.

- `--watch` checks again whenever a new forecast is scored, until the job ends. `--interval` sets the seconds between looks (default 60) and only matters together with `--watch`. `--no-watch` turns watching off.
- `--act` offers to stop the job when the check recommends a stop. This applies to running iterative runs only. APEX asks you to confirm before anything is stopped, so nothing stops without your answer. `--no-act` turns it off. Default: off. You can also stop a run yourself at any time with `apex stop RUN_DIR`; it stops at the next checkpoint, and you can continue it later.
- `--no-llm` gives a rules-only check.
- `--json` prints the result as JSON, which is handy for scripts.

## Step 5: Understand a failure with `apex agent diagnose`

```bash
apex agent diagnose RUN_ID --no-llm
```

`apex agent diagnose` explains what went wrong in a run and what to change. It is advisory: it changes nothing. It takes the same `target` as `monitor`, and supports `--no-llm`, `--json` and the model options.

To act on the advice, edit the run's configuration as suggested and start the run again with `apex run CONFIG`. Treat the advice as a suggestion to check, not a final verdict.

## Step 6: Draw figures with `apex agent plot`

```bash
apex agent plot RUN_ID --preset brugge_classic
```

`apex agent plot` draws a run's result figures. The `target` is a run directory, a run id or a job id. You choose what to draw in one of three ways:

- `--preset NAME` draws a named preset, for example `brugge_classic`. No model is used.
- `--with` switches on an option of a preset. Repeat it for several, as in `--with model_change --with normalized` (for `brugge_classic`).
- `--request "..."` says in words what to draw. If you give neither `--preset` nor `--request`, the classic set is drawn.

Figures go to `RUN/report/plots` unless you set `--output`. `--no-llm` gives a rules-only run, and the model options apply as usual.

A related, model-free command is `apex report RUN_DIR`. It prints the data mismatch history and performance metrics of a run, and with `--plots` it also writes the figures (PNG and PDF) to `--output` (default `RUN_DIR/report`), with `--font-size` setting the font in points (default 12).

## Step 7: Write a report with `apex agent report`

```bash
apex agent report RUN_ID --out report.md
```

`apex agent report` writes a narrative report, in Markdown, of a run or a study.

- Give a `target` (run directory, run id or job id) for a run, or `--study` to report on a study instead. Both are optional on the command line, but you need one of them.
- `--study` refers to a study that `apex agent plan` proposed. Use the study you wrote with it (see Step 8), for example `apex agent report --study STUDY --out study_report.md`.
- `--out` says where to write the Markdown.
- `--no-llm` gives a rules-only report.

## Step 8: Plan a study with `apex agent plan`

```bash
apex agent plan case.toml --goal "Find out which parameters the history constrains"
```

`apex agent plan` proposes a study for a case and a goal. It submits the study only if you confirm.

- `config` (required) is the case: a history-matching or optimization configuration.
- `--goal` (required) states what the experiment should find out.
- `--budget` is the number of simulator runs for the whole study. By default it is the approval limit in `service.toml`.
- `--out` is where the study is written. By default it goes beside the case.
- `--submit` submits the study after you confirm. The submission goes through APEX's own tools, which is the same route the agent uses to start jobs. `--no-submit` turns it off.
- `--approval-token` submits with a token that `apex approve` has approved. It implies `--submit`.
- `--no-llm` and `--json` are also available.

### The approval sequence

A submission may be held until someone approves it. When that happens, it stays pending instead of starting. The sequence is:

1. Run `apex agent plan case.toml --goal "..." --submit` and confirm when asked. If the submission needs approval, it is left pending and you get a token for it.
2. Approve it with `apex approve TOKEN`.
3. Submit again with the token:

   ```bash
   apex agent plan case.toml --goal "..." --approval-token TOKEN
   ```

## Step 9: Draft an optimization with `apex agent draft-optimization`

`apex agent draft-optimization` drafts an optimization of the forecast of a history-matched case. It does not use a model.

Required:

- `case`: the history-matching case's configuration (TOML), for example `case/config.toml`. Its simulator must run a deck (`opm_flow` or `eclipse`, with a `deck_template_file`): the draft continues that deck's schedule into a forecast.
- `--forecast-end`: the last date of the forecast, written `YYYY-MM-DD`. It must come after the end of the history, so look in your deck for where the history ends and choose a later date.

Economics. APEX never invents them: without `--wop`, `--wwp`, `--wwi` and `--disc` (and, with bhp control, `--producer-bhp`) the draft is refused, naming what is missing. Only `--wgp` and `--obj-scaling` are optional. Use the same units as your deck for volumes and your own currency for prices and costs:

- `--wop`: oil price per unit of FOPT. FOPT is the field oil production total, the cumulative oil produced.
- `--wwp`: cost per unit of produced water.
- `--wwi`: cost per unit of injected water.
- `--disc`: yearly discount rate, as a fraction (0.08 is 8 %).
- `--wgp`: gas price. By default gas is not priced.
- `--obj-scaling`: divides the NPV. NPV is the net present value: income from oil minus water costs, discounted over time. The optimizers minimise, so the default of -1e6 turns maximising the NPV into minimising, and it also expresses the NPV in millions. Change it only if you want another scale, and keep the negative sign if you still want to maximise.

Controls and model (optional, except `--producer-bhp` with bhp control):

- `--producer-control`: put producers on `bhp` (bottom-hole pressure) or `oil_rate`. Default: `bhp`.
- `--producer-bhp`: the producers' pressure bounds as `LOW,HIGH`, in the pressure unit of your deck. Required with bhp control (the default).
- `--control-periods`: the number of periods the forecast is split into, each with its own control value. Default: 4.
- `--run`: the history-matching run whose posterior is the model (id, name or path). The posterior is the set of model realisations that agree with the history after matching. Give it, so the draft uses your matched models.
- `--members`: how many of those posterior members are used. Default: 10.
- `--optimizer`: the optimizer plugin. Default: `enopt`.

Output: `--name` names the draft (default `<run>_npv`), `--out` sets the folder (default: beside the case), and `--json` prints the result as JSON.

The result is a draft optimization written to that folder. Open it, review the economics and bounds, and then pass its optimization configuration to `apex optimize`. You can limit a first try with `apex optimize CONFIG --max-iterations N` and continue later with `--resume DIR`.

## Worked example

Suppose a history-matching run `hm_run` has finished and you want to review it, then draft an optimization of its forecast. The prices below are illustrative; use your own.

1. Check which providers are available:

   ```bash
   apex agent providers
   ```

2. Check the run's health without a model:

   ```bash
   apex agent monitor hm_run --no-llm
   ```

3. If something looks wrong, ask for a diagnosis:

   ```bash
   apex agent diagnose hm_run --no-llm
   ```

4. Draw the classic figures:

   ```bash
   apex agent plot hm_run --preset brugge_classic
   ```

5. Write the report:

   ```bash
   apex agent report hm_run --out hm_run_report.md
   ```

6. Draft an optimization using the run's posterior. Here the history is assumed to end before 2032 and the pressure bounds are in the deck's pressure unit:

   ```bash
   apex agent draft-optimization case/config.toml \
     --forecast-end 2032-12-31 \
     --wop 70 --wwp 5 --wwi 2 --disc 0.08 \
     --run hm_run --members 10 \
     --producer-control bhp --producer-bhp 100,250 \
     --name hm_run_npv
   ```

   The draft is written beside the case, because `--out` is not given.

7. Review the draft, then run it with `apex optimize`.

## Pitfalls

- **Passing the key itself.** `--api-key-env` takes the name of a variable, never the key.
- **Mixing up the switches.** `--no-tool-calls` turns on prompting for JSON; it is not the same as `--no-tools`, which stops the agent from being offered APEX's tools. `--no-no-tool-calls` and `--no-no-llm` are the off switches for their options, not aliases.
- **Expecting a model to be asked with `--no-llm`.** It gives rules only.
- **Forgetting `--forecast-end`.** `apex agent draft-optimization` requires it, and the date must come after the history.
- **Using `--producer-bhp` with `oil_rate` control.** The bounds are for bhp control.
- **Expecting `apex agent plan` to submit by itself.** It submits only after you confirm, and only with `--submit` or `--approval-token`.
- **Using `--act` outside its scope.** It only offers to stop running iterative runs, and asks you first.
- **Treating the advice as final.** `apex agent diagnose` is advisory.
- **Using `--interval` alone.** It only matters together with `--watch`.
- **Running `apex agent report` with no target.** Give a `target` or `--study`.

## What to read next

- The pages on history matching and optimization, to prepare the cases that `plan` and `draft-optimization` work on, and on `apex run` and `apex optimize`.
- The page on the service configuration, for the `[agents.llm]` section and the approval limit in `service.toml`.
- The page on approvals, for `apex approve` and how a pending submission is approved.

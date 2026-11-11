---
audience: public
items:
  - cli:apex approve
  - cli:apex gc
  - cli:apex optimize
  - cli:apex plugins
  - cli:apex predict
  - cli:apex report
  - cli:apex resume
  - cli:apex run
  - cli:apex stop
  - cli:apex study compare
  - cli:apex submit
  - cli:apex summary
  - cli:apex validate
  - config:history_match:method
  - config:optimization:objectives
  - config:prediction:prediction
  - config:service:approval
  - plugin:methods:esmda
  - plugin:methods:gies
  - plugin:methods:gn_enrml
  - plugin:methods:lm_enrml
  - plugin:priors:file
  - plugin:surrogate_designs:file
  - plugin:surrogate_designs:run
  - plugin:update_strategies:full
  - webui:agents
  - webui:editor
  - webui:examples
  - webui:job
  - webui:jobs
  - webui:optimization
  - webui:optimizations
  - webui:plugins
  - webui:prediction
  - webui:run
  - webui:runs
  - webui:studies
---

# The web UI

## Goal

By the end of this lesson you will know what each page of the APEX web UI shows, how to move between the pages, and how to follow a piece of work from an example configuration to its results.

## Prerequisites

- A running APEX web UI that you can open in your browser.
- Nothing else is needed for the main path. The worked example is done in the browser, and the `examples` page gives you a first configuration.
- Optionally, a terminal in the APEX folder, for the steps marked optional. Type every command on this page as `uv run apex ...` from the APEX folder.
- Work is started from the web UI itself (the `editor` page's *submit*). The `jobs` page lists only jobs you submitted from the web UI. A run started in a terminal with `uv run apex run` is not a job, and a job handed over with `uv run apex submit` from a terminal is not listed there either.

The pages `jobs`, `runs`, `optimizations` and `studies` are empty until work has been submitted.

## Words used on this page

- **Services root**: the folder where APEX keeps its service data, including the record of your jobs. The default is `$APEX_HOME` if that variable is set, else `~/.apex/service`. The service's settings file, `service.toml`, lives there.
  - To see what your terminal uses, run `echo "$APEX_HOME"`. If it prints a path, that path is the root. If it prints an empty line, the variable is unset and the default applies.
  - The web UI must use the same root as your terminal. Ask whoever started the web UI which `APEX_HOME` it was started with (or whether it was unset), and give your terminal the same value.
  - The commands `apex submit`, `apex validate`, `apex study compare`, `apex approve` and `apex gc` accept `--root` to name the root; without it they take `$APEX_HOME`, else the default.
- **Workspace folder**: the folder that holds your configurations. The `editor` page lists the configurations in it, and `examples` copies files into it. Terminal commands that take a configuration need the path of the file, for example the path of your saved copy.
- **Workflow**: the kind of work a configuration is for, for example a history-matching run or a prediction. The `examples` page groups its catalogue by workflow, and the `plugins` page also groups plugins by workflow.
- **Registry**: a named group of plugins within a workflow, as the `plugins` page shows them. For example, `file` is a plugin of the `priors` registry (it reads a stored ensemble from an `.npz` file), and `full` is a plugin of the `update_strategies` registry (a full LM update with the prior-increment term). The `surrogate_designs` registry has plugins named `file` and `run` too; `run` takes the forecasts a history-matching run already holds, with no simulator runs.
- **History-matching run**: a run that adjusts an ensemble of models so that it matches observed data. It proceeds in **analysis steps**. `apex run` starts one in your terminal. Its `[method]` section names the ensemble algorithm from the `methods` registry (`name` is required) and takes `max_iterations` (default 10) for iterative methods. The methods you will meet are `esmda` (a fixed schedule of inflation factors), `gies`, `gn_enrml` and `lm_enrml`; the `history_match:method` configuration is that `[method]` section.
- **Data mismatch**: how far the simulated results are from the observed data. A run is doing its job when the mismatch goes down from step to step.
- **Checkpoint**: a point where a run can be stopped and later continued. You see its effect when a stop request takes effect.
- **Run directory**: the directory of one run. The commands `apex stop`, `apex resume` and `apex report` take it as an argument. `apex summary` takes run directories, or folders holding runs, which it searches. The web UI pages do not need it; only the optional terminal commands do.
- **Annotation**: a `key=value` label attached to a job when it is submitted with `uv run apex submit --annotation key=value`. The `runs` page can be filtered by annotation.
- **Prediction**: a forecast of the ensembles of a history-matching run beyond the history. `apex predict` makes one from a configuration whose `[prediction]` section (the `prediction:prediction` configuration) names the run (`run_dir`), the `ensembles` to forecast (such as `posterior` or `prior`) and the `vectors` to output.
- **Optimization**: work that improves an objective over several steps. `apex optimize` runs one in a terminal. An optimization configuration can list several objectives in its `optimization:objectives` entry, each a plugin with options and an optional label. The `optimization` page shows the objective over its steps, and its front when there are several objectives.
- **Study**: a set of variants that you compare. `apex study compare NAME` compares a study's variants at each run's posterior. The `studies` page lists your studies and their state. `apex validate` accepts a study configuration.
- **Job**: a unit of submitted work. It has a kind, a state and progress.
- **Approval**: a large submission is held until it is approved. The service's `[approval]` section (`service:approval`) sets when: `max_simulator_runs` (default 20000) and `max_runs` (default 20). The editor then shows an `apex approve TOKEN` command. `apex approve` takes the pending approval's token (and `--root` for the services root). After approval, submit again.

How to configure and start optimizations and studies is covered in later lessons. `apex summary` already reports optimization steps.

## How the UI is organised

Every page has its own route, written after a `#` in the address bar. For example, the list of jobs is at `#/jobs`. Some pages describe one specific item (a job, a run, an optimization). You reach those by opening an entry from the matching list page, and the UI fills in the item for you (the `{job}`, `{run}` and `{optimization}` parts of the routes below).

The `optimization` route ends in `?path={optimization}`. The UI fills in the `path` value. You never need to type it. Open the entry from `optimizations` instead.

### Setting up and configuring

| Page | Route | What it shows |
|---|---|---|
| `examples` | `#/examples` | The examples catalogue by workflow. Copy one into your workspace folder. |
| `editor` | `#/editor` | The workspace's configurations. Open, edit, validate and save a TOML file. |
| `plugins` | `#/plugins` | Every plugin by workflow and registry, with its options and their defaults. |
| `agents` | `#/agents` | APEX's public agents: what each does and how to run it from the web UI. |

![The examples page](img/webui-examples.png)

![The editor page](img/webui-editor.png)

![The plugins page](img/webui-plugins.png)

The plugins page picture was taken with some optional add-on packages installed, so the counts on its tabs may be a little higher than the counts on your own `plugins` page. A plugin from an add-on appears only once that add-on is installed.

![The agents page](img/webui-agents.png)

The editor and agents pictures may differ in detail from the pages you see; the table above describes what each page is for.

### Following work in progress

| Page | Route | What it shows |
|---|---|---|
| `jobs` | `#/jobs` | Every job you submitted from the web UI: its kind, state, progress and when it ran. Open one to follow it. |
| `job` | `#/jobs/{job}` | One job's live events and progress, with Stop and Resume, and links to its runs. |

![The jobs page](img/webui-jobs.png)

![A single job](img/webui-job.png)

### Looking at results

| Page | Route | What it shows |
|---|---|---|
| `runs` | `#/runs` | The runs you can see, filterable by annotation, with their method and state. |
| `run` | `#/runs/{run}` | One run's summary and its performance metrics at every step, as charts. |
| `optimizations` | `#/optimizations` | Your optimizations and their state. |
| `optimization` | `#/optimizations?path={optimization}` | One optimization's objective over its steps, and its front with several objectives. |
| `studies` | `#/studies` | Your studies and their state. |
| `prediction` | `#/prediction` | A prediction's percentiles and forecast scores. |

![The runs page](img/webui-runs.png)

![A single run](img/webui-run.png)

![The optimizations page](img/webui-optimizations.png)

![A single optimization](img/webui-optimization.png)

![The studies page](img/webui-studies.png)

![The prediction page](img/webui-prediction.png)

## Steps

1. Open the web UI in your browser and start at `examples` if you are new, or at `jobs` if you already have work going.
2. To configure something, use `examples` to copy a configuration into your workspace folder. The copy is then among the workspace's configurations that you can open in `editor`.
3. While editing, look up plugin names, options and defaults in `plugins` instead of guessing them. The same list is available in a terminal:

   ```bash
   uv run apex plugins --json
   ```

   Without `--json` it lists every plugin with its options; with `--json` you get full descriptions as JSON.
4. Validate the file before you rely on it. The `editor` page lets you validate a TOML file. You can also check a saved file from the terminal:

   ```bash
   uv run apex validate config.toml
   ```

   Replace `config.toml` with the path of your file. `apex validate` accepts a run, prediction, optimization, study or surrogate build configuration, reports any problem it finds, and plans the cost without simulating anything. Add `--json` to get the result as JSON, and `--root` to name the services root.
5. Start the work from the `editor`: save the file, then press *submit*. The service runs it as a job, whatever its kind. If the submission is large, the editor shows an `apex approve TOKEN` command instead of starting it: the submission is waiting for approval. Run that command from the APEX folder, replacing `TOKEN` with the token shown:

   ```bash
   uv run apex approve TOKEN
   ```

   Then submit again. In a terminal, `uv run apex run config.toml` runs a history match in that terminal, without a job, and `uv run apex submit config.toml` hands a configuration to the job service in the background. Neither appears in the web UI. `apex submit` also takes `--wait` (wait for the job to end), `--priority` (`high`, `normal` or `low`) and `--resume` (resume a run, optimization or build directory).
6. Find your work in `jobs` and open it to reach `job`, where you can watch live events and progress.
7. From `job`, follow the links to its runs, then open a run in `run` to see its charts.
8. For other kinds of results, use `optimizations` (then `optimization`), `studies` and `prediction`.
   - The route of `prediction` has no item. Open the page at `#/prediction`; it shows a prediction's percentiles and forecast scores.
   - To inspect a prediction you made with `apex predict` from the terminal, give its directory to `apex report`, which accepts the directory of a run or of a prediction.

## Worked example: from an example to a result

This example takes you from the catalogue to the charts of a history-matching run, in the browser: `examples` (copy) → `editor` (edit, validate, save, submit) → `jobs` → `job` → `run`. Only the optional last steps use a terminal.

1. Go to `#/examples` and pick an example under a **history-matching** workflow. Do not choose a prediction or a study example: the later steps follow a history-matching run. Use the example's copy control to copy it into your workspace folder.
2. Go to `#/editor`, and click the name of the file you just copied to open it. Now make one small, safe change, or none at all:
   - Look in the file for the strategy section, `[method.strategy]`. If it names the `full` strategy, you can change its option `energy`. On `plugins`, `energy` of `full` is a number in `(0, 1]` with default `0.98`: the fraction of the singular-value sum kept. Any value in that range keeps the run the same size.
   - If the file names another strategy, or has no `[method.strategy]` section, make no change: submit the copy as it is.
   - If the `energy` line is already in the section, change its value; if not, add the line to the section.

   ```text
   before:  energy = 0.98      (or no energy line at all)
   after:   energy = 0.95
   ```

   Avoid options that set sizes, such as an ensemble size: a large value can make the run much longer, and a large submission is held for approval. `full` only works with the methods `gies`, `lm_enrml` and `gn_enrml`; with `esmda` it is refused, so keep the strategy the example already uses if it is not `full`.

   You have done it right when the line uses the exact option name shown on `plugins` and the editor's validation (next step) reports no problem. Another option you will meet is `ensemble_file` of the `file` priors plugin: it is required, and it is the path of an `.npz` file with one array per variable.
3. Validate the file in the editor and fix anything it reports. Save the file when it validates.
4. With the saved file open in the editor, press *submit*. If the editor reports an error, fix the file, save it and submit again. If it shows an `apex approve TOKEN` command, run `uv run apex approve TOKEN` from the APEX folder (with the token shown in place of `TOKEN`), then submit again. A refused submission does not appear in `jobs`.
5. Go to `#/jobs`. Each row shows a job's kind, state, progress and when it ran. Find the row whose kind matches what you submitted (a history-matching run) and whose time is the most recent; if several rows look alike, open each and check that its live events refer to your work. If the list is empty, look at the editor for an error from the previous step.
6. On the `job` page, watch the live events and the progress. How long a run takes depends on the example, so wait until the run links appear, or until the progress stops changing and the events say the job has ended, before you read the charts.
7. Optional: try Stop and Resume. Stop does not act at once: it asks the run to stop at its next checkpoint, so the run may keep working for a while. Keep watching the live events and progress on the `job` page until they stop moving; a slow stop is normal. Then press Resume to continue it from its last checkpoint, and wait until the events and progress move again. Use these buttons on the `job` page so that you keep following the work there. From a terminal, `uv run apex stop RUN_DIR` asks a running run to stop and `uv run apex resume RUN_DIR` continues it; `--max-steps` on either `apex run` or `apex resume` stops after that many analysis steps.
8. Follow the links from the job to a `run` page. The links may not be there at the start; if the live events keep moving, wait and check again after some analysis steps have finished. If the job has stopped or failed and still shows no run links, read the last live events and the error on the `job` page. On the `run` page, look at the charts of the performance metrics and find the data mismatch: for a finished run it should go down from the first step to the last. If it does not, check the events for warnings.
9. To see all your runs, go to `#/runs`. The list shows each run's method and state. Open an entry to reach its `run` page.
10. Optional, in a terminal: for a written overview, run from the APEX folder:

    ```bash
    uv run apex summary RUN_DIR --output report.md --title "My run"
    ```

    `RUN_DIR` can be one or more run directories, or a folder that holds runs, which `apex summary` searches. It reports every metric at every step. The Markdown goes to `report.md` and a CSV file is written next to it. Without `--output` the Markdown is printed; without `--title` the title is "Results summary". This step and the next need a run directory; if you do not have one, stay with the charts on the `run` page.
11. Optional, in a terminal: print the data mismatch history and performance metrics of one run, and write its figures:

    ```bash
    uv run apex report RUN_DIR --plots
    ```

    `RUN_DIR` is the directory of a run or of a prediction. `--plots` also writes the result figures as PNG and PDF, into `RUN_DIR/report` unless you give `--output`. Use `--font-size` to change the font size of the figures (default 12 points).

## Pitfalls

- **Empty lists.** `jobs`, `runs`, `optimizations` and `studies` only show what you submitted. An empty list usually means nothing has been submitted yet, or the submission was refused. Work started in a terminal (`apex run`, `apex optimize`, `apex submit`) is not shown in `jobs`.
- **Different services roots.** If the terminal and the web UI use different roots, your terminal commands will not see the web UI's work. Give both the same `APEX_HOME`, or pass `--root` to the commands that accept it.
- **A submission waiting for approval.** It does not run until `apex approve TOKEN` has been run; submit again afterwards.
- **Runs that disappear.** `apex gc` forgets runs whose folder is gone and prunes what the service keeps. With `--chats` and `--guests` it also applies the chat history's and the guests' limits, and with `--code` it removes old frozen copies of APEX. Its `--dry-run` switch only has an effect together with `--code`, where it lists what would go and removes nothing.
- **Starting work from a terminal.** `apex run` runs in the terminal, not as a job of the service, so the web UI never shows it. Submit from the `editor` to follow work in the web UI. Likewise `apex optimize` and `apex predict` run in the terminal.
- **Pages that need an item.** `job`, `run` and `optimization` describe a single item, so open them from their list page instead of typing a route by hand.
- **Wrong example type.** The worked example follows a history-matching run. A prediction or study example gives another kind of job.
- **Wrong folder or command in the terminal.** Type `uv run apex ...` from the APEX folder.
- **Saving without validating.** Validate before you submit anything, so mistakes show up early.
- **Guessing option names.** Check `plugins` for the exact names, types and defaults.
- **A strategy that does not fit the method.** `full` is refused by `esmda`; use it with `gies`, `lm_enrml` or `gn_enrml`.
- **Examples are copies.** `examples` copies a configuration into your workspace folder. Edit your copy, not the catalogue.
- **Stop is not instant.** A run stops at its next checkpoint, so it may keep going for a while after you ask. A stopped run can be continued with Resume on its `job` page.
- **Run directories for the terminal commands.** `apex stop`, `apex resume` and `apex report` need the directory of a run (`apex report` also accepts a prediction's). Only `apex summary` searches folders.

## What to read next

- The lessons on writing and validating a TOML configuration, to get more from `editor`.
- The lessons on plugins and their options, alongside the `plugins` page.
- The lessons on running work and following jobs, to go deeper on `jobs` and `job`.
- The lessons on optimizations, studies and predictions, to start them and to read `optimization`, `studies` and `prediction` with confidence.

---
audience: public
generated: reference
---

# The assistant's tools: reference

Every one of these 21 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `case_info`

Read a case: its configuration (method, state variables with their grids and bounds, each variable's prior generator and options, simulator), its data (types, wells, times, and the observation errors' sigma per type) and, from its deck, the grid and each well's completion cells (1-based [i, j, k], the format of distance localization's positions). Read-only.


## `compare_study`

Compare a study's methods and variants at each run's posterior.


## `diagnose_run`

Diagnose a run: causes (with evidence) and recommended changes, with validated patches.


## `draft_optimization`

Draft an optimization from a history-matching case: its wells' controls over a forecast after the history, NPV, robust over a run's posterior; validated and costed. Ask the user for wop, wwp, wwi, disc and the producers' pressure bounds first; never invent them.


## `draft_run`

Write a run's configuration with a change (a JSON merge patch, e.g. a diagnosis's recommended patch, or the method, damping and localization the user asks for), from a run or a case's configuration, as a draft of this chat, validated and costed; then propose_action launches it as a new run.


## `find_cases`

Find configurations in the workspace by part of their path (e.g. 'norne'): each one's path, kind (run, study, prediction, optimization), method, damping, localization and simulator. Read-only; use it before asking the user for a path.


## `get_schema`

Get the JSON Schema of a configuration, or one section of it. Examples: get_schema(kind="study"), get_schema(kind="history_match", section="prior").


## `job_events`

Get a job's events from number since (use next_since to continue).


## `job_status`

Get a job's state. Example: job_status(job_id="20260928T160000-ab12cd34").


## `list_agents`

List APEX's agents, what each does and its command.


## `list_plugins`

List methods, localizations, simulators and other plugins. Example: list_plugins(registry="localizations").


## `list_runs`

List runs, optionally by annotation. Example: list_runs(annotations={"study": "loc"}).


## `monitor_run`

Check a run's health with the run monitor's rules, and a diagnosis (advisory; it never stops anything). Call the run healthy only when 'healthy' is true (null: not confirmed, say so).


## `plan_study`

Plan a study for a case and a goal: a draft in this chat's drafts folder, validated and costed (give variants to design it yourself; errors come back to revise). A history-matching case gives a history-matching study; an optimization configuration a study of optimizations (comparing optimizers). For APEX against ERT's localization, the design is ES-MDA methods differing in strategy (adaptive_rows, ert_adaptive, approx, ert_esmda), not engine = 'ert' (which takes no strategy); without a design, the rules plan it. …


## `plot_run`

Draw a run's result figures (PNG and PDF) in the background into its report/plots folder, shown on the run's page as they are drawn: what the request asks for (metric box/violin plots, forecast bands per well, layer maps, histograms) or a preset (brugge_classic: the whole classic set, minutes for a large case).


## `prediction_summary`

Get a finished prediction's statistics and scores.


## `propose_action`

Propose launching a draft (a study, or an optimization): the user sees a card to review, launch or discard it. You cannot launch it yourself.


## `read_draft`

Read a draft of this chat (its TOML), e.g. to check or explain it before revising it.


## `revise_draft`

Replace a draft's TOML (validated and costed; an invalid one is not saved).


## `run_summary`

Get a run's results: iterations, data mismatch mean and std, metrics.


## `write_report`

Write the narrative report of a run or a study (Markdown; numbers checked against the facts): a summary, the run's setup from its configuration, its results and health, and with compare_with a comparison with another run; save = true also saves it in this chat's drafts folder.

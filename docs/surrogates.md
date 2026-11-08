---
audience: public
---

# Surrogate models

!!! note "No surrogate model plugin is included yet"
    APEX's core ships the surrogate interface and the workflows on this page, but no surrogate model plugin yet: models come from installed packages, and `apex plugins` lists those in the registry `surrogate_models`. A model implements the protocol `apex.core.surrogates.SurrogateModel` and is registered through the entry-point group `apex.surrogate_models`; to write one, see [Writing and registering your own plugin](getting_started/writing_a_plugin.md).

## Goal

A surrogate model is a fast model fitted to full simulator runs of one case, which can later stand in for the simulator. `apex surrogate build` makes one: it runs a design of full simulations, fits a surrogate plugin to some of them, and gates it on the others, which it never saw. `apex surrogate check` gates an existing model on new full runs. After this page you can build a surrogate of a history-matching case, read its gate, check it again later, resume a build that stopped, and history-match the case with the surrogate making the forecasts, checked by the simulator. You can also build one over an optimization's controls and optimize with it, the simulator checking its accepted steps and evaluating the optimum.

Using a surrogate in predictions comes later. A model is only ever used inside the domain where it was validated and within its gate's tolerance; outside, it refuses to answer.

## Prerequisites

- A history-matching or optimization configuration (`CASE.toml`) whose simulator you can run. A surrogate stands in for its whole simulator stack: the simulator, its parameterizations and its geophysics.
- A surrogate plugin. `apex plugins` lists the registry `surrogate_models`; plugins come from installed packages.
- The keys a surrogate answers need times that are step indices or ISO dates.

## Terms used on this page

- **Snapshot:** one full run of the design: what the simulator received and what it answered.
- **Training runs:** the snapshots the plugin is fitted on.
- **Held-out runs:** the snapshots the gate judges the fitted model on. They are drawn first, from the seed, so they stay the same whatever the number of training runs.
- **Domain:** where the model was validated. It is built from the model's coordinates of the training and held-out inputs: a box per coordinate (optionally their convex hull or their Mahalanobis ellipsoid), a bound on how far an input may lie from the model's reduced input space, and the keys and times it was gated on.
- **Gate:** the accuracy test on the held-out runs (below).

## Steps

1. Write a build configuration next to the case:

   ```toml
   [surrogate]
   name = "box_surrogate"
   model = "PLUGIN"                 # a name from the surrogate_models registry
   options = {}                     # the plugin's options
   seed = 7
   # extent = "box"                 # or "hull", "mahalanobis"

   [case]
   config_file = "CASE.toml"
   outputs = "observations"         # or { vectors = ["WOPR:P1"], times = ["2020-01-01"] }

   [design]
   plan = "prior"                   # "prior" | "lhs" | "file" | "run"
   training = 80
   held_out = 20

   [gate]
   sigma_ratio = 0.5
   ```

   `training` and `held_out` are counted separately: this design runs the simulator 100 times.

2. Check what it will cost; nothing runs:

   ```bash
   uv run apex validate BUILD.toml
   ```

3. Build it. The full runs go through the case's `[execution]` (or the build's own `[execution]`), as `apex run` does:

   ```bash
   uv run apex surrogate build BUILD.toml
   ```

   The command prints the gate per datatype and exits with status 1 if the gate fails.

4. Check the model on new full runs, drawn from the case's prior or taken from a history-matching run's posterior:

   ```bash
   uv run apex surrogate check runs/box_surrogate/model/surrogate.json --against CASE.toml --runs 10
   uv run apex surrogate check runs/box_surrogate/model/surrogate.json --against runs/my_gies_run --runs 10
   ```

   The model file never changes. The verdict goes to `runs/box_surrogate/checks/<id>/`, and `apex summary` reports the latest check with the build.

   - A member equal to one of the model's own runs (training or held-out) is left out and reported as trained on. If fewer than `--runs` members are new to the model, the check is refused: a stored prior returns its first columns whatever the seed, so check a stored-prior case against a run directory instead.
   - Members outside the model's domain are reported and not gated: the model refuses them.
   - The check passes only if the gate passes, at least `--min-success-fraction` of the runs succeed (0.9 by default, as a build) and at least `--min-inside` of the successful runs lie inside the domain (0.8 by default). A check that fails on too few successful runs is still recorded, as failed, and becomes the model's latest check.
   - Choosing `--min-inside`: the domain is only as wide as the design's runs, so new draws from the same prior often fall outside a small design's box. With 25 runs in a few coordinates, 30–50 % of prior draws can lie outside, and the default 0.8 then fails the check: this is the check working (the model cannot answer those members), not a fault of the model. A larger design, or checking against members like those the model will be asked about (a run's posterior, which lies inside the prior's bulk), raises the share. Lower `--min-inside` only to judge the model's accuracy where it is valid, knowing that the members outside will be refused when it is used.

5. Summarise builds with `uv run apex summary runs/` (see [Run summaries](summaries.md)).

## Designs

- `prior`: draws from the case's prior generators, so fields keep their geostatistics. A prior that gives stored columns (`file`) is refused: use plan `file` with its ensemble. In any design, a held-out run equal to a training run is refused.
- `lhs`: a Latin hypercube over the state, one point per stratum of each coordinate. The box is `[design.box]` per state variable (`x = [lo, hi]` for every entry), else the variables' bounds, which must then be finite. It suits a few scalar inputs; draw fields from their prior.
- `file`: columns of a stored ensemble, `ensemble_file`, an `.npz` with one `(size, n)` array per state variable.
- `run`: the forecasts a history-matching run already holds (`run_dir`, its `iterations`, `"all"` by default). No simulator runs. The run must be of the same simulator stack, and its stored forecasts must be the keys' raw values. Members whose stored forecasts are not finite at every key are recorded as failed and left out.

With `[[parameterizations]]` the design is drawn in state space; the surrogate's inputs are the properties the simulator reads.

For an optimization's case the design is over its controls: `lhs` (the controls' bounds, or `[design.box]`) and `file` (an ensemble of controls) apply, `prior` and `run` do not. With `[model] realizations_file` (robust optimization), design member `m` is given realization `m mod n`, so the surrogate learns the realizations too.

Design plans are plugins of the registry `surrogate_designs` (entry-point group `apex.surrogate_designs`). The four above are core's own. Packages may add adaptive plans, which choose training runs from a fitted surrogate; such a plan takes its settings from `[design] options`.

An adaptive build runs in rounds:
1. The held-out runs and the plan's initial training runs are drawn from the prior. The held-out runs are drawn first and fixed by the seed, so the gate stays an unbiased test.
2. A pool of candidate prior draws is drawn.
3. The build runs the members, fits the surrogate on the training runs so far, and runs the batch of candidates the plan chooses. This repeats until `training` runs are made.
4. The build then fits and gates the model as any other.

The batches chosen are recorded, so an interrupted build resumes to the same choices and byte-identical model files. Because the surrogate chose the runs, an adaptive build's runs depend on its plugin and options. Changing them is refused in the same build directory, where a core plan would refit on the runs already made.

## A coarse stack

Some surrogates correct a coarse simulation of each member instead of predicting from the inputs alone. Their builds name a coarse stack in `[coarse]`:

```toml
[coarse]
options = { deck_template_file = "coarse/deck.mako" }   # merged over the case's simulator options

[[coarse.parameterizations]]
name = "block_average"
options = { dims = [50, 50, 1], factors = [2, 2, 1], means = { poro = "arithmetic", permx = "arithmetic" } }
```

- **The coarse stack** is the case's, with `simulator` (default: the case's) and `options` merged over the case's simulator options. Its `parameterizations` are applied after the case's.
- **`block_average`** averages fields over blocks of `factors` cells onto the coarser grid. Each variable takes its own mean:
  - `arithmetic` for porosity, and for ln k, which gives k's geometric mean;
  - `geometric` or `harmonic` for positive variables.
- **Every design run** also runs on the coarse stack, in the build's `coarse/` folder, so a build makes twice the simulator runs. The snapshots keep both stacks' outputs.
- **Coarse-only members:** `[design] coarse_only = N` adds N members run on the coarse stack only, for surrogates that learn from many cheap coarse runs and fewer fine ones (pretrained on the coarse runs, then fine-tuned on the fine ones):
  - they are drawn from the same plan after the held-out and training runs, so those stay the same for a given seed; with plan `file`, they are the next columns of the ensemble, which must have enough;
  - they run in `coarse/` after the paired coarse runs; `apex validate` counts them among the simulator runs, the build's summary reports `coarse_only_runs`, and a resumed build reuses those already made;
  - the snapshots keep their inputs and coarse outputs apart (`coarse_only_inputs`, `coarse_only_outputs`) and the plugin's fit receives them; they are never gated;
  - a failed one is recorded with the design's failures and counted against `min_success_fraction` on its own;
  - `coarse_only` needs `[coarse]` and a surrogate that trains on coarse runs (below; another's fit would not read them), and is refused with plan `run`; with N > 0 it joins the build's fingerprint, so changing it is refused in the same build directory;
  - a coarse-only run that fails is reported apart, as the summary's `coarse_only_failed` (`failed` lists the runs the fit and the gate lost); `min_success_fraction` applies to them on their own.
- **Surrogates that train on coarse runs** declare the capability `trains_on_coarse`. They take a `[coarse]` stack (and need one) without correcting a coarse forecast: they learn from the coarse runs in training only, so history matching runs no coarse stack for them, a `[policy]` judges none of that stack's files in a run, and `on_check_failure = "refit"` is refused for them.
- **The gate** judges the corrected prediction. The validated domain bounds the coarse forecast's coordinates as well as the inputs'.
- **The coarse stack's provenance** joins the model's, and `[coarse]` joins the build's fingerprint.
- **The model's manifest names its coarse stack,** with its file options relative to the model's folder. History matching and `apex surrogate check` rebuild the stack from there over the case. They refuse it if it is not the stack the model was built with.
- **In history matching,** each surrogate forecast first runs every member on the coarse stack:
  - the runs use the run's own executor, so pools, licences, retries and `keep_simulations` apply as for the simulator's runs;
  - they go in `simulations/coarse_<iteration>/`;
  - then the domain check and the corrected prediction follow.
- **A member whose coarse run fails** is a failed member, counted against `min_success_fraction`.
- **Counting:**
  - `apex validate` counts one coarse run per surrogate evaluation among the simulator runs;
  - the run's summary reports `coarse_runs`;
  - the timing reports the coarse runs apart (`coarse`), and adds their cost to each corrected prediction's.
- **A refit** pairs the checked runs with their coarse runs, made once.
- **`apex surrogate check`** runs each member on both stacks, in the check's folder and `coarse/` in it, and gates the corrected prediction.
- **Under a `[policy]`,** the files the manifest's coarse stack names are checked as the case's own simulator's are.
- **Not yet supported:**
  - a coarse stack with an adaptive or `run` design;
  - reusing a forecast's coarse runs after a resume: they are made again, with the same results.

## Early-start surrogates

Some surrogates continue a member's own run instead of replacing it: each member is simulated up to a report date, and the model predicts the later report dates from the member's data up to that date.

- **Running to a date:** a member can carry a stop date (`MemberInput.stop_at`, an ISO report date). `opm_flow` and `eclipse` honour it by cutting the rendered deck after that report date (a `DATES` record, or a `TSTEP` step reaching it) and closing it with `END`. The `SCHEDULE` section's includes are first written into the deck in place of their `INCLUDE` records (an include is text inclusion, so the run is the same), so a schedule kept in include files is cut too; the rest of the text before the cut is the deck's own. The date must be a report date of the deck's schedule, in whole days. Includes named by a relative path or a `PATHS` alias are left as they are, and when they hold report dates before the date the deck is refused, as is one that skips or restarts before it; a schedule include that is not UTF-8 text is left as it is, and refused by name if it holds report dates. Executors refuse a stop date for a simulator that cannot honour it, so no run silently goes to the end of its schedule.
- **The protocol:** such a model is a `core.surrogates.EarlyStartSurrogate`. It names the keys it reads up to its date (`early_keys()`), places a member's early data in the validated domain (`early_coordinates`), and predicts from them (`predict_from_early`); it returns the early keys unchanged.
- **The domain** bounds the early data rather than the inputs, so a member whose early behaviour lies outside the training runs' is refused.
- **The gate** uses the held-out runs' own values up to the date as their early data, so it judges exactly what a member would get, with no extra run. It scores only the keys after the date: the early ones are the member's own values.
- **In history matching,** each surrogate forecast first runs every member up to the model's date:
  - the runs use the run's own simulator and executor, so pools, licences, retries and `keep_simulations` apply as for the simulator's runs; they go in `simulations/early_<iteration>/`, and are asked only for the model's early keys;
  - then the domain check and the prediction follow; a member whose early run fails is a failed member, and one outside the domain runs whole;
  - the checks, and the posterior's comparison in the summary, gate only the keys after the date; data transforms make keys of their own, which are gated whole, early part included (the run logs `surrogate_early_transformed`);
  - `apex validate` counts one early run per surrogate evaluation among the simulator runs (members outside the domain, or whose early run fails, make one more each), the run's summary reports `early_runs`, and the timing reports them apart (`early`);
  - the run's simulator must stop at report dates (`opm_flow`, `eclipse`; not with model error or geophysics), its executor must carry the stop date to it (executors that rebuild members on other hosts do not), the date must be a report date of the deck, and some assimilated or cv observation must lie after it; otherwise the run is refused.
- **Predictions** take early-start surrogates too (below, *Predictions with a surrogate*). Optimizations refuse them, since their controls change the whole schedule.

## State fields

A field surrogate, one that predicts state fields as well as data, is built on snapshots of those fields. The build's `[case]` names them:

```toml
[case]
config_file = "case.toml"
fields = ["PRESSURE", "SWAT"]               # restart arrays
field_times = ["2025-01-01", "2026-01-01"]  # ISO dates
```

- **Deck simulators** (`opm_flow`, `eclipse`): every run reads the fields from its restart files at those dates, on the deck's whole grid (`DIMENS` or `SPECGRID`) in Eclipse order, with inactive cells 0. The deck must write a restart at each date with those arrays (`RPTRST`).
- **Other simulators** must answer the field keys themselves; give `field_cells`.
- **The snapshots** keep the fields beside the data (`snapshots.npz`), covered by its digest; the fields join the build's fingerprint.
- **The gate** judges each predicted field by one pooled relative RMS error over its cells, dates and held-out runs: the RMS error over the RMS change from the first date for a pressure, over the field's range for other fields. Cells that are 0 throughout (inactive) are left out. The tolerance is `[gate] relative`'s for the field's kind (0.05 for pressures and saturations). `apex surrogate check` runs and gates the fields too.
- **The deck's restarts** are checked before any run, as far as its text tells: a field date with no restart is refused.
- **Refused:**
  - a plugin that predicts no fields;
  - fields with a `[coarse]` stack, an adaptive plan or plan `run`;
  - a case with `[geophysics]`.
- **History matching** still refuses field observations with a surrogate. Here the fields serve predictions and figures.

## The gate

- With `outputs = "observations"`, per datatype the RMS over the held-out runs and keys of `(surrogate − simulator) / σ` must be at most `sigma_ratio` (0.5 by default), σ being the observation error's standard deviation. The surrogate's error must lie well inside the data noise the methods assume.
- With `outputs = "responses"` (an optimization's objective's keys), relative errors as below, and the held-out runs' objectives must keep their order: Spearman's rank correlation of the surrogate's objective against the simulator's is at least `[gate] min_rank_correlation` (0.95). With several objectives, each one's.
- With listed vectors and times, relative errors by kind of quantity: pressures against their change from the first value, rates and other quantities against their magnitude; tolerances in `[gate] relative` (`pressure`, `rate`, `plume`, `mass`, `other`; 0.05, 0.05, 0.1, 0.01, 0.05 by default) and `[gate] kinds` where a vector's mnemonic would mislead.

A case that transforms its field data (`[[data_transforms]]`) cannot have its observations as outputs yet: list scalar vectors instead.

## Resuming and refitting

Each finished run is saved at once. Running the same configuration again (or `apex surrogate build --resume runs/NAME`) runs only the missing members, and the model files it writes are identical, byte for byte, to an uninterrupted build's. `--max-runs N` runs at most N full runs now. A configuration whose plugin, options or gate changed is fitted again on the runs already made; one whose case, design or seed changed is refused in the same directory, naming what differs (for example `stack.simulator_version` or `design`).

A build refuses a case with `[model_error]`, and stacks whose coupling reads each step's restart files.

## The build's directory

```text
runs/<name>/
  config.json, state.json, provenance.json, events.jsonl, run.lock
  design.npz          the design: member ids and inputs of the held-out and training runs
  members/            one file per finished full run
  snapshots.jsonl     each finished run's outcome (failed runs are recorded with their message)
  snapshots.npz       the snapshots the model was fitted and gated on
  simulations/        work directories ([output] keep_simulations: "failed" by default)
  model/surrogate.json, model/surrogate.npz   the model: a JSON manifest and plain arrays
  model/surrogate.cost.json   the full runs' times (mean, median, total), for the runs that use it
  gate.json, summary.json
  checks/<id>/        the checks of the model
```

A model's files execute no code when they are read: the manifest is JSON and the arrays are numbers. Its identity is the SHA-256 of both; any changed byte gives another identity. The plugin's options that name files (`*_file`, `*_dir`, `*_path`) are written in the manifest relative to the model's folder, so a build and its input files moved together keep the model's identity, and a refit in history matching finds the files there. The cost file is not part of the identity, since run times differ between builds that give identical models. It names the model it belongs to, and a run passes over another model's file. Copy it with the model so that runs with the model can compare their times with the full runs. A build whose members run whole on other hosts gets their results back untimed, so it writes no cost file and logs `surrogate_untimed`.

## History matching with a surrogate

A `[surrogate]` section in a history-matching configuration makes every forecast the method asks for with a built model, on this machine; the configured `[simulator]` and `[execution]` run only the checks, the fallbacks and the final forecast of the posterior. The method, its strategy, localization and damping are unchanged: the surrogate stands in for the simulator stack only.

```toml
[surrogate]
model_file = "runs/box_surrogate/model/surrogate.json"
check_every = 3            # iterations 3, 6, 9, ... are also checked by the simulator (0: none)
check_members = 10         # members checked then (or "all"), drawn from the stream "surrogate"
on_check_failure = "stop"  # "stop" | "refit" | "simulator"
outside = "stop"           # members outside the model's domain: "stop" | "simulator"
max_refits = 2
# refit_held_out = 0.5     # a refit's share of the checked runs held out for its gate
# variance_in_errors = false   # ES-MDA, ES: add the model's predictive variance to C_d
# sigma_ratio = 0.3        # the checks' tolerance: may only tighten the model's gate
# workers = 4              # local processes for the surrogate's forecasts (none: in this process)
# max_evaluations = 50000  # the run's bound on surrogate evaluations
```

1. Validate the run: `uv run apex validate CASE.toml`. The plan counts the surrogate's evaluations apart from the simulator's runs (the checks, the posterior's forecast, and every forecast where the simulator may take over); only the simulator's runs count towards approval. The run is refused when:
   - the model did not pass its gate, or its last check failed;
   - it was built for another simulator stack or case;
   - the run observes a key (any role: history matching, cross validation, test) the model was not gated on;
   - the case has `[model_error]`, fidelity levels, the adjoint update (`sensitivities`), an executor running members on other hosts (remote mode; use the executor's other mode), or `[[data_transforms]]` on field data (or any, with `refit` or `variance_in_errors`);
   - the plan would exceed `max_evaluations`;
   - `sigma_ratio` would loosen the model's gate;
   - `variance_in_errors` names a model without a predictive variance, or an iterative method (GIES, LM-EnRML, GN-EnRML, CHOP draw their perturbations once and judge trials on the observation errors alone). With ES-MDA and ES, the members' mean predictive variance per datum is added to the observation errors of each step, in its perturbations and gain; the variance is that step's, and diagonal (correlated surrogate errors are left out).
2. Run it as any run: `uv run apex run CASE.toml`.
3. Read `summary.json`'s `surrogate` section, or `uv run apex summary runs/NAME`.

What happens during the run:

- **Every decision is made on the surrogate's forecasts:** the acceptance of trials and their damping, the stopping criteria and, with `[validation]`, the cross-validation rule and its choice of the minimum-CV iterate. The update uses the surrogate's forecast of every member, so that the members' errors stay alike.
- **Members outside the domain** stop the run with the members, the coordinates outside and by how much (`outside = "stop"`), or are run by the simulator and recorded as such (`outside = "simulator"`). They are never dropped.
- **Checks:** at the check iterations, `check_members` members are run by the simulator too and gated against the surrogate's forecasts, with observation errors (`sigma_ratio`, in the space the method sees), the assimilated data and the cross-validation data apart. A check that fails:
  - `stop` (default): the run stops before the update, at its last checkpoint, with the event `surrogate_check_failed` and the errors per datatype. Set `on_check_failure` in the run's `config.json` and resume. A rebuilt model is another model (another identity), which a resume refuses: start a new run with it.
  - `refit`: the checked runs join the model's snapshots, the plugin is fitted again and gated on the held-out runs plus a share of the new ones, and the iteration's forecast is repeated with the refit. After `max_refits` refits a failing check stops the run.
  - `simulator`: the rest of the run uses the simulator.

  After a refit or a switch, an iterative method's accepted iterate (and its mean model) is forecast again by the new source before the trial is judged, so that the trial is compared with an iterate of the same source (event `surrogate_rescored`, `checkpoints/rescored_<k>.npz`). With `[validation]`, the stopping rule then starts again from that iterate: iterates scored by the old source are no longer candidates for the posterior.
- **The posterior is always the simulator's:** when the run completes, the posterior iterate (the last accepted one, or the minimum-CV iterate with `[validation]`) is forecast by the simulator. `posterior.npz`, the run's reported data mismatch and metrics of the posterior, and the forecasts predictions start from are the simulator's. `summary.json` gives both forecasts' scores (data mismatch, normalized data mismatch, and the cross-validation mismatch), their difference, and whether the difference is beyond the model's gate. Its `data_mismatch_reduction` compares the simulator's posterior with the prior's forecast, whose source `prior_source` names (`data_mismatch_reduction_basis`); `surrogate_data_mismatch_reduction` compares the surrogate's posterior with the same forecast. Members the simulator forecast already are not run again.
- **The cost:** `summary.json`'s `surrogate.timing` compares the forecasts' times with the full model's, and `apex summary` states it in one note:
  - `surrogate` and `simulator`: the forecasts, members, wall seconds, and worker-seconds per member, over every session of the run, including a forecast repeated after an interrupted session.
    - Worker-seconds are wall time multiplied by the members run at once: the executor's workers, capped by the simulator's licences. They include waiting and retries, and a process pool's start-up, so they overstate the time of a short forecast.
    - `simulator` is the run's own full runs: checks, members outside the domain, the posterior's forecast, and runs after a switch. Its failed members are counted too.
    - An executor that does not state its workers gives wall time only.
  - `full_model`: the build's mean, median and total time per full run, from `model/surrogate.cost.json`, or from the build's journal for an older model.
  - `speedup`: the full model's mean time over the surrogate's worker-seconds per member.
  - `simulator_equivalent_worker_seconds`: what the surrogate's members would have taken as full runs, against `run_worker_seconds`. The build's total is a one-off cost beside both.

  The times are measured: they come from the run's events, never from its checkpoints, and they do not change its results.
- **Resume** is bit-identical to an uninterrupted run, across checks and refits. It refuses a model file whose identity differs from the one the run started with.

What the run directory adds:

```text
runs/<name>/
  checkpoints/iter_<k>.npz     forecast_source per member ("surrogate" or "simulator") and surrogate_id
  checkpoints/check_<k>.npz    a check: the checked members' simulator and surrogate forecasts, the gate
  checkpoints/rescored_<k>.npz the accepted iterate forecast again after the source changed at k
  checkpoints/full_<k>.npz     the simulator's forecast of the posterior iterate k, scored
  surrogates/refit_<k>/        a refit: its snapshot set and model
  events.jsonl                 surrogate_forecast, surrogate_check, surrogate_check_failed,
                               surrogate_refit, surrogate_switched, surrogate_rescored,
                               surrogate_outside, surrogate_posterior
```

`history.json` records each forecast's source, and `apex report --plots` marks the surrogate's steps (`S`, or `M` where some members were simulated). Run directories written before surrogates existed are read as forecast by the simulator.

## Optimization with a surrogate

1. Build a surrogate of the optimization (its configuration as `[case] config_file`, `outputs = "responses"`, `[design] plan = "lhs"`), and read its gate.
2. Add a `[surrogate]` section to the optimization's configuration:

   ```toml
   [surrogate]
   model_file = "runs/opt_surrogate/model/surrogate.json"
   check_every = 3            # accepted steps 3, 6, 9, ... are checked by the simulator (0: none)
   on_check_failure = "stop"  # "stop" | "simulator"
   outside = "stop"           # controls outside the model's domain: "stop" | "simulator"
   # relative = { rate = 0.02 }   # the checks' tolerances: may only tighten the model's gate
   # workers = 4              # local processes for the surrogate's evaluations
   # max_evaluations = 50000  # the run's bound on surrogate evaluations
   ```

3. Validate it: `uv run apex validate OPT.toml`. The plan gives the surrogate's evaluations apart from the simulator's runs (the checks and the optimum's evaluations, and every evaluation when the simulator may take over); only the simulator's runs count towards approval. The run is refused when the model did not pass its gate or its last check failed, was built for another simulator stack, or on other inputs (controls, parameterizations, `[model]` arrays), or does not answer the objective's keys.
4. Run it: `uv run apex optimize OPT.toml`.

What happens during the run:

- **Every evaluation the optimizer asks for is the surrogate's**, journalled in `evaluations.jsonl` with `source = "surrogate"` under a key that includes the model's identity. A run without the surrogate, or with another model, never reuses them. The simulator's evaluations (`source = "simulator"`) are journalled under the plain run's keys, so within the run directory (a resume, or the restart after a switch to the simulator) they are reused as a plain run's would be.
- **Controls outside the domain** stop the run at its last checkpoint (`outside = "stop"`), or are evaluated by the simulator (`outside = "simulator"`).
- **Checks:** every `check_every` accepted steps, the accepted controls are evaluated by the simulator at every realization and gated against the surrogate's values by relative errors, before the step's checkpoint. The history row records the check (`surrogate_check`), and `surrogate_state.json` keeps every check. A check that fails:
  - `stop` (default): the run stops at its last checkpoint. Set `on_check_failure = "simulator"` in the run's `config.json` and resume. A rebuilt model is another model (another identity): a run refuses to resume with it, so start a new optimization.
  - `simulator`: the optimization starts again from its start with the simulator for every evaluation. It does not start from its last checkpoint, which holds the surrogate's objective values. From the start it is the plain run with the same seed, and the evaluations the simulator made already are reused from the journal. A resume keeps the switch.
- **The optimum is the simulator's:** the result's controls are evaluated by the simulator at every realization; `result.json`'s `objective` is the simulator's, `surrogate_objective` the surrogate's, and its `surrogate` section counts the evaluations of each source, the checks and whether the run switched. A multi-objective run's whole front is evaluated again, and `pareto.json` holds the simulator's objectives (and hypervolume) with the surrogate's beside them.
- **Resume** is bit-identical to an uninterrupted run, checks included, for APEX's optimizers and for ropt's.
- **Checking the model later:** `uv run apex surrogate check MODEL --against OPT.toml --runs N` draws `N` controls by a Latin hypercube within the box the build's design spanned, which the model's manifest records (so a copied model can be checked; older models read it from their build's `design.npz`). With `[surrogate] extent = "hull"` or `"mahalanobis"` the domain is narrower than that box, the more so the more controls: runs outside it are simulated but not gated, the check warns, and its summary says how many; too few inside (`--min-inside`) fail the check, which makes the model unusable until rebuilt. Prefer the box extent for optimizations with many controls.
- **In a study's comparison**, `objective:final`, `front_size:final` and `hypervolume:final` are the simulator's (the front without the members the simulator's values dominate); `objective:best` comes from the history, which holds the surrogate's values.

## Predictions with a surrogate

A prediction (`apex predict`) can forecast its ensembles with a surrogate whose keys reach the prediction's horizon: one built on series that run past the history (`[case] outputs = { vectors = [...], times = [...] }` in the build, on the prediction's own deck), or an early-start surrogate that continues each member's own run.

```toml
[surrogate]
model_file = "runs/to_2030/model/surrogate.json"
outside = "stop"     # members outside the model's domain: "stop" | "simulator"
# workers = 4        # process workers for the surrogate (default: serial)
```

- **The axis** is the model's: `[prediction.outputs] vectors` are matched against the model's datatypes, and `times = "report"` means the model's report times, in time order. A time that is not one of them (beyond its last, its horizon) or a vector it does not answer is refused.
- **The deck** must be the build's: a prediction whose simulator stack differs from the one the model was built on (another deck, other controls, another forward model, other simulator options such as a timeout) is refused, naming what differs. Build the model on a case without `[geophysics]` or `[model_error]`: a prediction forecasts the simulator's series alone.
- **Members:** each ensemble's members (and mean models) are predicted by the model on its own executor, and a member with non-finite values stops the prediction; an early-start model's members first run on the simulator up to its date (through the prediction's executor, in `simulations/<ensemble>_early/`). A member outside the domain stops the prediction, or with `outside = "simulator"` runs whole by the simulator; a member whose early run fails is a failed member (a prediction stopped during the early runs makes them again on resume: they are not saved). The surrogate failing a member stops the prediction.
- **The true model** of a synthetic case is always forecast by the simulator: it is what the scores measure the ensemble against.
- **The history check** compares the members the simulator ran exactly, as without a surrogate. The surrogate's members are checked within the model's gate (`sigma_ratio` of the run's observation errors), and an early-start model's early keys exactly; a failure fails the prediction.
- **Records:** `state.json` holds each member's source (`member_sources`), `provenance.json` the model and its identity (a rebuilt model is another prediction: delete the folder or give the prediction another name), and `apex summary` says how many members each source forecast.
- **Not supported:** coarse-corrected surrogates (their coarse stack would have to run to the horizon), and run directories of a model-error run (refused for every prediction).

### A control scenario

A prediction's control scenario (future BHP or rate targets every member is given; see [Predictions: a control scenario](predictions.md#a-control-scenario)) needs a surrogate built over the controls:

- **How the model is built:** over the controls, on an optimization-shaped case of the forecast:
  - the prediction's deck template and simulator options;
  - its `[[controls.variables]]` with bounds, and their parameterizations;
  - the run's realizations as `[model] realizations_file` (e.g. its posterior);
  - outputs `{ vectors, times }` over the history's report times and the forecast's: the history check compares the history's.

  The design varies the controls jointly with the realizations. The prediction then checks that the model's inputs are the run's state (after its parameterizations) and the scenario's controls (after theirs), by name and size, and that each control, as the model sees it (after the parameterizations), lies within the range the design spanned, entry by entry: a surrogate does not extrapolate the controls, and another schedule can put the same values elsewhere. The model's manifest records those inputs and ranges, so a copied model needs no build directory. A model built without controls is refused with a scenario, and a model built over controls without one. The realizations are checked by name and size only: that they are the run's (and not another run's of the same shape) is the user's choice, and members far from them are outside the model's domain. The model has to take the controls as inputs over time to follow a change of schedule.


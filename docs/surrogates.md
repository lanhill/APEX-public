---
audience: public
---

# Surrogate models

!!! note "No surrogate model plugin is included yet"
    APEX's core ships the surrogate interface and the workflows on this page, but no surrogate model plugin yet: models come from installed packages, and `uv run apex plugins` lists those in the registry `surrogate_models`. A model implements the protocol `apex.core.surrogates.SurrogateModel` and is registered through the entry-point group `apex.surrogate_models`; to write one, see [Writing and registering your own plugin](getting_started/writing_a_plugin.md).

## Goal

A surrogate model is a fast model fitted to full simulator runs of one case, which can later stand in for the simulator. `apex surrogate build` makes one: it runs a design of full simulations, fits a surrogate plugin to some of them, and gates it on the others, which it never saw. `apex surrogate check` gates an existing model on new full runs.

After this page you can build a surrogate of a history-matching case, read its gate, check it again later, resume a build that stopped, and history-match the case with the surrogate making the forecasts, checked by the simulator. You can also build one over an optimization's controls and optimize with it, and use one in predictions (see [Predictions with a surrogate](#predictions-with-a-surrogate) below). A model is only ever used inside the domain where it was validated and within its gate's tolerance; outside, it refuses to answer.

## Prerequisites

- A history-matching or optimization configuration (`CASE.toml`) whose simulator you can run. A surrogate stands in for its whole simulator stack: the simulator, its parameterizations and its geophysics.
- A surrogate plugin. `uv run apex plugins` lists the registry `surrogate_models`; plugins come from installed packages.
- Times you can name. A surrogate answers *keys*: a key is one output vector of the case, such as `WOPR:P1`, at the times you ask for. Those times must be step indices or ISO dates (for example `2020-01-01`). Check the `times` you list in the build's `[case] outputs`, or the report dates of the observations when you use `outputs = "observations"`; if a time is neither a step index nor an ISO date, change it before building.

## Terms used on this page

- **Snapshot:** one full run of the design: what the simulator received and what it answered.
- **Training runs:** the snapshots the plugin is fitted on.
- **Held-out runs:** the snapshots the gate judges the fitted model on. They are drawn first, from the seed, so they stay the same whatever the number of training runs.
- **Reduced input space:** the compact coordinates the model itself uses to describe an input (for a field, far fewer numbers than cells). The domain is measured in these coordinates.
- **Mahalanobis ellipsoid:** an ellipsoid around the training and held-out points in those coordinates, shaped by how the points spread, so it is longer along directions where the runs varied more.
- **Domain:** where the model was validated. It is a box per coordinate (or, with `extent`, their convex hull or their Mahalanobis ellipsoid), a bound on how far an input may lie from the model's reduced input space, and the keys and times it was gated on. A member outside it is refused, never answered.
- **Gate:** the accuracy test on the held-out runs (see [The gate](#the-gate)).

## Steps

1. Write a build configuration next to the case and save it as `BUILD.toml`:

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

   `extent` chooses the shape of the domain. The default box is the widest: an input is accepted if each coordinate lies within the range the design covered. `"hull"` and `"mahalanobis"` are narrower, so more members are refused. Choose them when you want the model to answer only close to where the runs actually were; the narrower the domain, the more members a later history match or check will find outside it.

2. Check what it will cost; nothing runs:

   ```bash
   uv run apex validate BUILD.toml
   ```

   The plan counts the simulator runs the build will make (100 for the file above). Decide from that count whether the cost is acceptable, or whether the design should be smaller, before you build.

3. Build it. The full runs go through the case's `[execution]` (or the build's own `[execution]`), as `apex run` does:

   ```bash
   uv run apex surrogate build BUILD.toml
   ```

   The command prints the gate per datatype and exits with status 1 if the gate fails. A datatype passes when its error is within the tolerance (with `outputs = "observations"`, an RMS of `(surrogate − simulator) / σ` at most `sigma_ratio`). A passing build prints every datatype within its tolerance and exits with status 0.

   **If the gate fails,** the model is not usable. Typical remedies, then rebuild:
   - add training runs (raise `training`): the run files already made are kept, so only the missing members are simulated;
   - change the plugin's `options`: they are refitted on the runs already made;
   - use a design that fits the case better (`prior` for fields, `lhs` for a few scalar inputs);
   - loosen `sigma_ratio` only if the observation errors really allow it: the surrogate's error should lie well inside the data noise.

   A build also requires at least 90 % of its design's runs to succeed by default; a build whose runs mostly fail is refused rather than fitted.

4. Check the model on new full runs, drawn from the case's prior or taken from a history-matching run's posterior:

   ```bash
   uv run apex surrogate check runs/box_surrogate/model/surrogate.json --against CASE.toml --runs 10
   uv run apex surrogate check runs/box_surrogate/model/surrogate.json --against runs/my_gies_run --runs 10
   ```

   The model file never changes. The verdict goes to `runs/box_surrogate/checks/<id>/`, and `apex summary` reports the latest check with the build.

   - A member equal to one of the model's own runs (training or held-out) is left out and reported as trained on. If fewer than `--runs` members are new to the model, the check is refused: a stored prior returns its first columns whatever the seed, so check a stored-prior case against a run directory instead.
   - Members outside the model's domain are reported and not gated: the model refuses them.
   - The check passes only if the gate passes, at least `--min-success-fraction` of the runs succeed (0.9 by default) and at least `--min-inside` of the successful runs lie inside the domain (0.8 by default). A check that fails on too few successful runs is still recorded, as failed, and becomes the model's latest check.
   - Choosing `--min-inside`: the domain is only as wide as the design's runs, so new draws from the same prior often fall outside a small design's box. With 25 runs in a few coordinates, 30–50 % of prior draws can lie outside, and the default 0.8 then fails the check: this is the check working (the model cannot answer those members), not a fault of the model. A larger design, or checking against members like those the model will be asked about (a run's posterior, which lies inside the prior's bulk), raises the share. Lower `--min-inside` only to judge the model's accuracy where it is valid, knowing that the members outside will be refused when it is used.

5. Summarise builds with `uv run apex summary runs/` (see [Run summaries](summaries.md)).

## A worked example

Suppose a history-matching case `CASE.toml` has 20 uncertain scalar inputs and you want a surrogate for the observed data.

1. Save the file from step 1 as `BUILD.toml`, with `model` set to a name that `uv run apex plugins` lists.
2. Run `uv run apex validate BUILD.toml`: it reports 100 simulator runs. You accept that cost.
3. Run `uv run apex surrogate build BUILD.toml`. The gate prints per datatype. Suppose one datatype is above `sigma_ratio`: raise `training` to 160 in `BUILD.toml` and run the same command again; the first 100 runs are reused.
4. When the exit status is 0, run `uv run apex surrogate check runs/box_surrogate/model/surrogate.json --against CASE.toml --runs 10`. If it fails only on `--min-inside`, the design is small for the prior: enlarge it, or check against a run's posterior.
5. Add the `[surrogate]` section of [History matching with a surrogate](#history-matching-with-a-surrogate) to `CASE.toml` and run `uv run apex run CASE.toml`.

## Designs

- `prior`: draws from the case's prior generators, so fields keep their geostatistics. A prior that gives stored columns (`file`) is refused: use plan `file` with its ensemble. In any design, a held-out run equal to a training run is refused.
- `lhs`: a Latin hypercube over the state, one point per stratum of each coordinate. The box is `[design.box]` per state variable (`x = [lo, hi]` for every entry), else the variables' bounds, which must then be finite. It suits a few scalar inputs; draw fields from their prior.
- `file`: columns of a stored ensemble, `ensemble_file`, an `.npz` with one `(size, n)` array per state variable.
- `run`: the forecasts a history-matching run already holds. No simulator runs. The run must be of the same simulator stack, and its stored forecasts must be the keys' raw values. Members whose stored forecasts are not finite at every key are recorded as failed and left out. The held-out members are taken first from the run's members, chosen by the seed, and the training members come from the rest.

  ```toml
  [surrogate]
  name = "from_run"
  model = "PLUGIN"
  options = {}
  seed = 7

  [case]
  config_file = "CASE.toml"
  outputs = "observations"

  [design]
  plan = "run"
  run_dir = "runs/my_gies_run"
  iterations = "all"        # the default: every iteration's forecasts
  training = 80
  held_out = 20

  [gate]
  sigma_ratio = 0.5
  ```

With `[[parameterizations]]` the design is drawn in state space; the surrogate's inputs are the properties the simulator reads.

For an optimization's case the design is over its controls: `lhs` (the controls' bounds, or `[design.box]`) and `file` (an ensemble of controls) apply, `prior` and `run` do not. With `[model] realizations_file` (robust optimization), design member `m` is given realization `m mod n`, so the surrogate learns the realizations too.

Design plans are plugins of the registry `surrogate_designs` (entry-point group `apex.surrogate_designs`). The four above are core's own. Packages may add adaptive plans, which choose training runs from a fitted surrogate; such a plan takes its settings from `[design] options`.

An adaptive build runs in rounds:
1. The held-out runs and the plan's initial training runs are drawn from the prior. The held-out runs are drawn first and fixed by the seed, so the gate stays an unbiased test.
2. A pool of candidate prior draws is drawn.
3. The build runs the members, fits the surrogate on the training runs so far, and runs the batch of candidates the plan chooses. This repeats until `training` runs are made.
4. The build then fits and gates the model as any other.

The batches chosen are recorded, so an interrupted build resumes to the same choices and byte-identical model files. Because the surrogate chose the runs, an adaptive build's runs depend on its plugin and options. Changing them is refused in the same build directory, where a core plan would refit on the runs already made.

## Extra calibration runs

A surrogate that declares `uses_calibration_runs` (one with a spread to calibrate: a Laplace posterior, `uncertainty = "laplace"`, or a deep ensemble) gets extra full runs for its spread. They are used only to fit the spread's calibration (a Laplace spread's two scales, an ensemble's factor per datatype): never trained on, never used for early stopping, never gated on. Calibrating on the validation runs instead reuses runs that also chose the training's stopping: for a Laplace spread that made it about 20 % too narrow on new runs; for an ensemble's factors no bias was measured, and the runs only remove the reuse (users who do not want their cost set `calibration = 0`).

- **Default:** when `[design]` does not set `calibration`, the build adds them itself: **30 runs for a case of up to 20,000 grid cells, 20 for a larger one**. The cells are the deck's `DIMENS`/`SPECGRID` (every cell, active or not); without a readable grid, the largest state variable's size counts.
- **Told before it runs:** `apex validate` (and the REST plan, so the web UI and the assistant) counts them among the simulator runs and states the line `N extra calibration runs for the surrogate's spread (<rule>); set [design] calibration = K to change, 0 to calibrate on the validation runs instead`. The build's log says it again at the start (event `surrogate_calibration_runs`). Approval thresholds and quotas count them.
- **To change the count:** `[design] calibration = K`. `calibration = 0` adds none: the spread is then calibrated on the validation runs, with a warning (`laplace_calibration_on_validation` for a Laplace spread, `spread_calibration_on_validation` for an ensemble's).
- **Where they come from:** drawn after every other run (held-out, training, coarse-only), from the same plan, so the other runs are those of the build without them. With plans `file` and `run` they are further columns of the ensemble (with `run` at no cost, and the plan's line says so); the automatic count is capped at the columns left after the other runs (the plan says so), and a given count beyond them is refused. They run on the fine stack only.
- **Adaptive plans** take none yet: the automatic count is 0 there, the plan says so, and the spread is calibrated on the validation runs with the warning. An explicit `calibration = K > 0` with an adaptive plan is refused, as is one for a surrogate that does not declare `uses_calibration_runs`.
- **Kept:** the snapshots hold them apart (`calibration_inputs`, `calibration_outputs`, `calibration_ids`), and refits keep them. A failed calibration run is reported apart (`calibration_failed`). If fewer than `min_success_fraction` of them succeed, the build is not lost (the design keeps their ids, the summary reports `calibration_runs = 0` and `calibration_fallback`): the spread is calibrated on the validation runs instead, with a warning (`surrogate_calibration_runs_failed`). The build's summary reports `calibration_runs`.
- **Asked at validation:** whether a surrogate wants them is asked of the plugin, so `apex validate` creates it: a plugin refusing its options fails there, before any run, not only in the build.
- **Unchanged:** builds of surrogates that do not declare it draw none, and their configurations, fingerprints and files are those of before. The calibration runs are never part of a build's fingerprint (they change no other run), so models built before reload and predict unchanged, and a complete build is found without its plugin.
- **In an existing build folder:** turning the option on adds the calibration runs there, reusing the runs already made, and refits; `calibration = 0` drops them. A surrogate build made before calibration runs existed (a Laplace one, and since the ensembles' extension a deep ensemble's) gets them on its next build and refits, as its plan and a resume's charge count them. With a drawn plan (`prior`, `lhs`) members are reused by their ids, so changing the count once calibration runs were made there is refused (keep the count, give 0, or build in another folder), though returning to the count they were made with (after 0) reuses them; if too many fail, the summary's `calibration_fallback` says the spread fell back to the validation runs; with `file` and `run` an id is its ensemble's column, and the count may change.

## A coarse stack

Some surrogates correct a coarse simulation of each member instead of predicting from the inputs alone. Their builds name a coarse stack in `[coarse]`:

```toml
[surrogate]
name = "corrected"
model = "PLUGIN"
options = {}
seed = 7

[case]
config_file = "CASE.toml"
outputs = "observations"

[design]
plan = "prior"
training = 80
held_out = 20

[gate]
sigma_ratio = 0.5

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
  - a failed one is reported apart, as the summary's `coarse_only_failed` (`failed` lists the runs the fit and the gate lost), and `min_success_fraction` applies to them on their own;
  - `coarse_only` needs `[coarse]` and a surrogate that trains on coarse runs, and is refused with plan `run`; with N > 0 it joins the build's fingerprint, so changing it is refused in the same build directory.
- **Surrogates that train on coarse runs** declare the capability `trains_on_coarse`. They take a `[coarse]` stack (and need one) without correcting a coarse forecast: they learn from the coarse runs in training only, so history matching runs no coarse stack for them, a `[policy]` judges none of that stack's files in a run, and `on_check_failure = "refit"` is refused for them.
- **The gate** judges the corrected prediction. The validated domain bounds the coarse forecast's coordinates as well as the inputs'.
- **The model's manifest names its coarse stack,** with its file options relative to the model's folder. History matching and `apex surrogate check` rebuild the stack from there over the case, and refuse it if it is not the stack the model was built with.
- **In history matching,** each surrogate forecast first runs every member on the coarse stack, with the run's own executor (pools, licences, retries and `keep_simulations` apply), in `simulations/coarse_<iteration>/`; then the domain check and the corrected prediction follow. A member whose coarse run fails is a failed member, counted against `min_success_fraction`.
- **Counting:** `apex validate` counts one coarse run per surrogate evaluation among the simulator runs; the run's summary reports `coarse_runs`; the timing reports the coarse runs apart (`coarse`) and adds their cost to each corrected prediction's.
- **A refit** pairs the checked runs with their coarse runs, made once.
- **`apex surrogate check`** runs each member on both stacks, in the check's folder and `coarse/` in it, and gates the corrected prediction.
- **Not yet supported:** a coarse stack with an adaptive or `run` design; reusing a forecast's coarse runs after a resume (they are made again, with the same results).

## Early-start surrogates

Some surrogates continue a member's own run instead of replacing it: each member is simulated up to a report date, and the model predicts the later report dates from the member's data up to that date.

- **Running to a date:** a member can carry a stop date (`MemberInput.stop_at`, an ISO report date). `opm_flow` and `eclipse` honour it by cutting the rendered deck after that report date (a `DATES` record, or a `TSTEP` step reaching it) and closing it with `END`. The `SCHEDULE` section's includes are first written into the deck in place of their `INCLUDE` records, so a schedule kept in include files is cut too. The date must be a report date of the deck's schedule, in whole days. Includes named by a relative path or a `PATHS` alias are left as they are, and when they hold report dates before the date the deck is refused, as is one that skips or restarts before it; a schedule include that is not UTF-8 text is left as it is, and refused by name if it holds report dates. Executors refuse a stop date for a simulator that cannot honour it, so no run silently goes to the end of its schedule.
- **The protocol:** such a model is a `core.surrogates.EarlyStartSurrogate`. It names the keys it reads up to its date (`early_keys()`), places a member's early data in the validated domain (`early_coordinates`), and predicts from them (`predict_from_early`); it returns the early keys unchanged.
- **The domain** bounds the early data rather than the inputs, so a member whose early behaviour lies outside the training runs' is refused.
- **The gate** uses the held-out runs' own values up to the date as their early data, so it judges exactly what a member would get, with no extra run. It scores only the keys after the date.
- **In history matching,** each surrogate forecast first runs every member up to the model's date, with the run's own simulator and executor, in `simulations/early_<iteration>/`, asked only for the model's early keys:
  - then the domain check and the prediction follow; a member whose early run fails is a failed member, and one outside the domain runs whole;
  - the checks, and the posterior's comparison in the summary, gate only the keys after the date; data transforms make keys of their own, which are gated whole (the run logs `surrogate_early_transformed`);
  - `apex validate` counts one early run per surrogate evaluation among the simulator runs (members outside the domain, or whose early run fails, make one more each), the run's summary reports `early_runs`, and the timing reports them apart (`early`);
  - the run's simulator must stop at report dates (`opm_flow`, `eclipse`; not with model error or geophysics), its executor must carry the stop date to it (executors that rebuild members on other hosts do not), the date must be a report date of the deck, and some history-matching or cross-validation observation must lie after it; otherwise the run is refused.
- **Predictions** take early-start surrogates too. Optimizations refuse them, since their controls change the whole schedule.

## State fields

A field surrogate, one that predicts state fields as well as data, is built on snapshots of those fields. The build's `[case]` names them:

```toml
[surrogate]
name = "field_surrogate"
model = "PLUGIN"                 # a plugin that predicts fields
options = {}
seed = 7

[case]
config_file = "case.toml"
outputs = "observations"
fields = ["PRESSURE", "SWAT"]               # restart arrays
field_times = ["2025-01-01", "2026-01-01"]  # ISO dates

[design]
plan = "prior"
training = 80
held_out = 20

[gate]
sigma_ratio = 0.5
```

- **Deck simulators** (`opm_flow`, `eclipse`): every run reads the fields from its restart files at those dates, on the deck's whole grid (`DIMENS` or `SPECGRID`) in Eclipse order, with inactive cells 0. The deck must write a restart at each date with those arrays (`RPTRST`).
- **Other simulators** must answer the field keys themselves; give `field_cells`.
- **The snapshots** keep the fields beside the data (`snapshots.npz`), covered by its digest; the fields join the build's fingerprint.
- **The gate** judges each predicted field by one pooled relative RMS error over its cells, dates and held-out runs: the RMS error over the RMS change from the first date for a pressure, over the field's range for other fields. Cells that are 0 throughout (inactive) are left out. The tolerance is `[gate] relative`'s for the field's kind (0.05 for pressures and saturations). `apex surrogate check` runs and gates the fields too.
- **The deck's restarts** are checked before any run, as far as its text tells: a field date with no restart is refused.
- **Refused:** a plugin that predicts no fields; fields with a `[coarse]` stack, an adaptive plan or plan `run`; a case with `[geophysics]`.
- **History matching** still refuses field observations with a surrogate. Here the fields serve predictions and figures.

## The gate

- With `outputs = "observations"`, per datatype the RMS over the held-out runs and keys of `(surrogate − simulator) / σ` must be at most `sigma_ratio` (0.5 by default), σ being the observation error's standard deviation. The surrogate's error must lie well inside the data noise the methods assume.
- With `outputs = "responses"` (an optimization's objective's keys), relative errors as below, and the held-out runs' objectives must keep their order: Spearman's rank correlation of the surrogate's objective against the simulator's is at least `[gate] min_rank_correlation` (0.95). With several objectives, each one's.
- With listed vectors and times, relative errors by kind of quantity: pressures against their change from the first value, rates and other quantities against their magnitude; tolerances in `[gate] relative` (`pressure`, `rate`, `plume`, `mass`, `other`; 0.05, 0.05, 0.1, 0.01, 0.05 by default) and `[gate] kinds` where a vector's mnemonic would mislead.

A case that transforms its field data (`[[data_transforms]]`) cannot have its observations as outputs yet: list scalar vectors instead.

## Resuming and refitting

Each finished run is saved at once. Running the same configuration again (or `uv run apex surrogate build --resume runs/NAME`) runs only the missing members, and the model files it writes are identical, byte for byte, to an uninterrupted build's. `--max-runs N` runs at most N full runs now. A configuration whose plugin, options or gate changed is fitted again on the runs already made; one whose case, design or seed changed is refused in the same directory, naming what differs (for example `stack.simulator_version` or `design`).

A build refuses a case with `[model_error]`, and stacks whose coupling reads each step's restart files.

## Builds and checks as jobs

The job service runs builds and checks in the background, as it runs history matches and optimizations:

```bash
uv run apex submit build.toml                 # a build: recognised by its [surrogate] and [design] tables
uv run apex submit build.toml --max-steps 20  # at most 20 member runs now; the rest later
uv run apex submit --resume runs/NAME         # continue a stopped build
uv run apex jobs list
uv run apex jobs stop JOB                     # a build stops between waves of member runs
```

- **Approval.** A build is gated by its simulator runs (training and held-out runs, twice with a coarse stack), a check by its `runs`, as any submission is: above the service's limits it needs approval or a budget grant. An approval is for the build's content (its configuration and its case's, with the files they name), whatever its folder: an edited build needs a new one.
- **Where it goes.** `apex submit` places a build beside its file (`runs/<name>`), as `apex surrogate build` does. Submitted through the client, the REST API (`POST /v1/surrogates`, `POST /v1/surrogates:check`) or the tools (`submit_surrogate`, `surrogate_summary`) without an `output_dir`, it goes to `<root>/surrogates/<name>/<id>`, named by its content: submitting the same build again finds it there instead of running it again. A web-UI user's build goes to their own folder.
- **Stopping.** A stopped build ends after its current wave of member runs: four times the executor's parallel width, capped by the simulator's (an Eclipse licence count). An executor that does not say its width runs the design in one batch, which a stop cannot interrupt. Every finished member is kept, and resuming (`uv run apex submit --resume BUILD_DIR`, or the job's resume) writes the same model files as an uninterrupted build, on the same code. A stopped check is submitted again. `uv run apex stop BUILD_DIR` stops a build running as a job; a build run by `apex surrogate build` stops with Ctrl-C and resumes the same way.
- **Checks** write their verdict beside the model, and a model's last verdict decides whether runs may use it: a web-UI user checks only models of their own folder, and under a `[policy]` the verdicts' folder must lie in the output roots. A check runs against a history-matching or optimization configuration, or a run.
- **Pinned code.** A job pinned to a commit whose worker cannot run surrogate jobs is refused when it is submitted.
- **Under a `[policy]`** a build is judged with its case's configuration, which it runs, and a model file with what its manifest names (its plugin's file options and its snapshot file).
- **In the web UI** the Surrogates page lists your builds and checks; a build's page shows its runs, its gate on the held-out runs datatype by datatype, its fit and prediction times, and a form that checks the model on new runs of a case. Submit a build from the Editor.
- **The monitor and report agents** can read a build (`apex agent monitor`, `apex agent report`). Their rules say whether the gate failed or the latest check failed (then runs refuse the model), whether more than `[agents.monitor] surrogate_max_failed_share` (5 %) of the design's runs failed, whether a prediction is less than `surrogate_min_speedup` (10) times faster than the median full run, and how many runs a stopped build has pending. They advise only: neither stops a build.

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
   - the run observes a key (in any role: history matching, cross validation, test) the model was not gated on;
   - the case has `[model_error]`, fidelity levels, the adjoint update (`sensitivities`), an executor running members on other hosts (remote mode; use the executor's other mode), or `[[data_transforms]]` on field data (or any, with `refit` or `variance_in_errors`);
   - the plan would exceed `max_evaluations`;
   - `sigma_ratio` would loosen the model's gate;
   - `variance_in_errors` names a model without a predictive variance, or an iterative method (GIES, LM-EnRML, GN-EnRML, CHOP draw their perturbations once and judge trials on the observation errors alone). With ES-MDA and ES, the members' mean predictive variance per datum is added to the observation errors of each step, in its perturbations and gain; the variance is that step's, and diagonal.
2. Run it as any run: `uv run apex run CASE.toml`.
3. Read `summary.json`'s `surrogate` section, or `uv run apex summary runs/NAME`.

What happens during the run:

- **Every decision is made on the surrogate's forecasts:** the acceptance of trials and their damping, the stopping criteria and, with `[validation]`, the cross-validation rule and its choice of the minimum-CV iterate. The update uses the surrogate's forecast of every member, so that the members' errors stay alike.
- **Members outside the domain** stop the run with the members, the coordinates outside and by how much (`outside = "stop"`), or are run by the simulator and recorded as such (`outside = "simulator"`). They are never dropped.
- **Checks:** at the check iterations, `check_members` members are run by the simulator too and gated against the surrogate's forecasts, with observation errors (`sigma_ratio`, in the space the method sees), the history-matching data and the cross-validation data apart. A check that fails:
  - `stop` (default): the run stops before the update, at its last checkpoint, with the event `surrogate_check_failed` and the errors per datatype. Set `on_check_failure` in the run's `config.json` and resume. A rebuilt model is another model (another identity), which a resume refuses: start a new run with it.
  - `refit`: the checked runs join the model's snapshots, the plugin is fitted again and gated on the held-out runs plus a share of the new ones, and the iteration's forecast is repeated with the refit. After `max_refits` refits a failing check stops the run.
  - `simulator`: the rest of the run uses the simulator.

  After a refit or a switch, an iterative method's accepted iterate (and its mean model) is forecast again by the new source before the trial is judged (event `surrogate_rescored`, `checkpoints/rescored_<k>.npz`). With `[validation]`, the stopping rule then starts again from that iterate: iterates scored by the old source are no longer candidates for the posterior.
- **The posterior is always the simulator's:** when the run completes, the posterior iterate (the last accepted one, or the minimum-CV iterate with `[validation]`) is forecast by the simulator. `posterior.npz`, the run's reported data mismatch and metrics of the posterior, and the forecasts predictions start from are the simulator's. `summary.json` gives both forecasts' scores (data mismatch, normalized data mismatch, and the cross-validation mismatch), their difference, and whether the difference is beyond the model's gate. Its `data_mismatch_reduction` compares the simulator's posterior with the prior's forecast, whose source `prior_source` names (`data_mismatch_reduction_basis`); `surrogate_data_mismatch_reduction` compares the surrogate's posterior with the same forecast.
- **The cost:** `summary.json`'s `surrogate.timing` compares the forecasts' times with the full model's, and `apex summary` states it in one note:
  - `surrogate` and `simulator`: the forecasts, members, wall seconds, and worker-seconds per member, over every session of the run. Worker-seconds are wall time multiplied by the members run at once, and include waiting and retries, so they overstate the time of a short forecast. `simulator` is the run's own full runs: checks, members outside the domain, the posterior's forecast, and runs after a switch.
  - `full_model`: the build's mean, median and total time per full run, from `model/surrogate.cost.json`.
  - `speedup`: the full model's mean time over the surrogate's worker-seconds per member.
  - `simulator_equivalent_worker_seconds`: what the surrogate's members would have taken as full runs, against `run_worker_seconds`. The build's total is a one-off cost beside both.

  The times are measured from the run's events and do not change its results.
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

`history.json` records each forecast's source, and `apex report --plots` marks the surrogate's steps (`S`, or `M` where some members were simulated).

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

3. Validate it: `uv run apex validate OPT.toml`. The plan gives the surrogate's evaluations apart from the simulator's runs; only the simulator's runs count towards approval. The run is refused when the model did not pass its gate or its last check failed, was built for another simulator stack, or on other inputs (controls, parameterizations, `[model]` arrays), or does not answer the objective's keys.
4. Run it: `uv run apex optimize OPT.toml`.

What happens during the run:

- **Every evaluation the optimizer asks for is the surrogate's**, journalled in `evaluations.jsonl` with `source = "surrogate"` under a key that includes the model's identity. A run without the surrogate, or with another model, never reuses them. The simulator's evaluations (`source = "simulator"`) are journalled under the plain run's keys, so within the run directory they are reused as a plain run's would be.
- **Controls outside the domain** stop the run at its last checkpoint (`outside = "stop"`), or are evaluated by the simulator (`outside = "simulator"`).
- **Checks:** every `check_every` accepted steps, the accepted controls are evaluated by the simulator at every realization and gated against the surrogate's values by relative errors, before the step's checkpoint. The history row records the check (`surrogate_check`), and `surrogate_state.json` keeps every check. A check that fails:
  - `stop` (default): the run stops at its last checkpoint. Set `on_check_failure = "simulator"` in the run's `config.json` and resume. A rebuilt model is another model: a run refuses to resume with it, so start a new optimization.
  - `simulator`: the optimization starts again from its start with the simulator for every evaluation, as the plain run with the same seed; the evaluations the simulator made already are reused from the journal. A resume keeps the switch.
- **The optimum is the simulator's:** the result's controls are evaluated by the simulator at every realization; `result.json`'s `objective` is the simulator's, `surrogate_objective` the surrogate's, and its `surrogate` section counts the evaluations of each source, the checks and whether the run switched. A multi-objective run's whole front is evaluated again, and `pareto.json` holds the simulator's objectives (and hypervolume) with the surrogate's beside them.
- **Resume** is bit-identical to an uninterrupted run, checks included.
- **Checking the model later:** `uv run apex surrogate check MODEL --against OPT.toml --runs N` draws `N` controls by a Latin hypercube within the box the build's design spanned, which the model's manifest records. With `extent = "hull"` or `"mahalanobis"` the domain is narrower than that box, the more so the more controls: runs outside it are simulated but not gated, the check warns, and its summary says how many; too few inside (`--min-inside`) fail the check, which makes the model unusable until rebuilt. Prefer the box extent for optimizations with many controls.
- **In a study's comparison**, `objective:final`, `front_size:final` and `hypervolume:final` are the simulator's; `objective:best` comes from the history, which holds the surrogate's values.

## Predictions with a surrogate

A prediction (`apex predict`) can forecast its ensembles with a surrogate whose keys reach the prediction's horizon: one built on series that run past the history (`[case] outputs = { vectors = [...], times = [...] }` in the build, on the prediction's own deck), or an early-start surrogate that continues each member's own run.

```toml
[surrogate]
model_file = "runs/to_2030/model/surrogate.json"
outside = "stop"     # members outside the model's domain: "stop" | "simulator"
# workers = 4        # process workers for the surrogate (default: serial)
```

- **The axis** is the model's: `[prediction.outputs] vectors` are matched against the model's datatypes, and `times = "report"` means the model's report times, in time order. A time that is not one of them (beyond its horizon) or a vector it does not answer is refused.
- **The deck** must be the build's: a prediction whose simulator stack differs from the one the model was built on (another deck, other controls, another forward model, other simulator options such as a timeout) is refused, naming what differs. Build the model on a case without `[geophysics]` or `[model_error]`.
- **Members:** each ensemble's members (and mean models) are predicted by the model on its own executor, and a member with non-finite values stops the prediction; an early-start model's members first run on the simulator up to its date, in `simulations/<ensemble>_early/`. A member outside the domain stops the prediction, or with `outside = "simulator"` runs whole by the simulator; a member whose early run fails is a failed member (a prediction stopped during the early runs makes them again on resume). The surrogate failing a member stops the prediction.
- **The true model** of a synthetic case is always forecast by the simulator: it is what the scores measure the ensemble against.
- **The history check** compares the members the simulator ran exactly. The surrogate's members are checked within the model's gate (`sigma_ratio` of the run's observation errors), and an early-start model's early keys exactly; a failure fails the prediction.
- **Records:** `state.json` holds each member's source (`member_sources`), `provenance.json` the model and its identity (a rebuilt model is another prediction: delete the folder or give the prediction another name), and `apex summary` says how many members each source forecast.
- **Not supported:** coarse-corrected surrogates (their coarse stack would have to run to the horizon), and run directories of a model-error run.

### A control scenario

A prediction's control scenario (future BHP or rate targets every member is given; see [Predictions: a control scenario](predictions.md#a-control-scenario)) needs a surrogate built over the controls:

- **How the model is built:** over the controls, on an optimization-shaped case of the forecast: the prediction's deck template and simulator options; its `[[controls.variables]]` with bounds, and their parameterizations; the run's realizations as `[model] realizations_file` (for example its posterior); outputs `{ vectors, times }` over the history's report times and the forecast's.
- **What the prediction checks:** the design varies the controls jointly with the realizations. The prediction checks that the model's inputs are the run's state (after its parameterizations) and the scenario's controls (after theirs), by name and size, and that each control lies within the range the design spanned, entry by entry: a surrogate does not extrapolate the controls. The model's manifest records those inputs and ranges, so a copied model needs no build directory. A model built without controls is refused with a scenario, and a model built over controls without one. The model has to take the controls as inputs over time to follow a change of schedule.

## Pitfalls

- Building without `uv run apex validate BUILD.toml` first: you may pay for far more runs than you expected.
- Treating a failed `--min-inside` as a bad model. It means the check's members lie outside the validated domain; widen the design or check against a run's posterior.
- Changing the case, design or seed in an existing build folder: it is refused. Use a new folder; only the plugin, its options and the gate may change in place.
- Rebuilding a model and resuming a run that used the old one: the rebuilt model is another model, and the resume refuses it.
- Using `outputs = "observations"` with `[[data_transforms]]` on field data: list scalar vectors instead.

## What to read next

- [Run summaries](summaries.md), to read a build's gate and cost.
- [Predictions: a control scenario](predictions.md#a-control-scenario), for forecasting with a model built over controls.
- [Writing and registering your own plugin](getting_started/writing_a_plugin.md), to add a surrogate model or design plan.

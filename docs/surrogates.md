---
audience: public
---

# Surrogate models

## Goal

A surrogate model is a fast model fitted to full simulator runs of one case, which can later stand in for the simulator. `apex surrogate build` makes one: it runs a design of full simulations, fits a surrogate plugin to some of them, and gates it on the others, which it never saw. `apex surrogate check` gates an existing model on new full runs. After this page you can build a surrogate of a history-matching case, read its gate, check it again later, resume a build that stopped, and history-match the case with the surrogate making the forecasts, checked by the simulator.

Using a surrogate in optimization or predictions comes later. A model is only ever used inside the domain where it was validated and within its gate's tolerance; outside, it refuses to answer.

## Prerequisites

- A history-matching configuration (`CASE.toml`) whose simulator you can run. A surrogate stands in for its whole simulator stack: the simulator, its parameterizations and its geophysics.
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

## The gate

- With `outputs = "observations"`, per datatype the RMS over the held-out runs and keys of `(surrogate − simulator) / σ` must be at most `sigma_ratio` (0.5 by default), σ being the observation error's standard deviation. The surrogate's error must lie well inside the data noise the methods assume.
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
  - `stop` (default): the run stops before the update, at its last checkpoint, with the event `surrogate_check_failed` and the errors per datatype. Set `on_check_failure` in the run's `config.json` (or rebuild the model) and resume.
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

---
audience: public
---

# Surrogate models

## Goal

A surrogate model is a fast model fitted to full simulator runs of one case, which can later stand in for the simulator. `apex surrogate build` makes one: it runs a design of full simulations, fits a surrogate plugin to some of them, and gates it on the others, which it never saw. `apex surrogate check` gates an existing model on new full runs. After this page you can build a surrogate of a history-matching case, read its gate, check it again later, and resume a build that stopped.

Using a surrogate inside history matching, optimization or predictions comes later. A model is only ever used inside the domain where it was validated and within its gate's tolerance; outside, it refuses to answer.

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
  gate.json, summary.json
  checks/<id>/        the checks of the model
```

A model's files execute no code when they are read: the manifest is JSON and the arrays are numbers. Its identity is the SHA-256 of both; any changed byte gives another identity.

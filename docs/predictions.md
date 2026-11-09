---
audience: public
---

# Predictions

A prediction forecasts a finished history-matching run's ensembles beyond the history. It reads the run and never changes it; its results go to `RUN_DIR/predictions/<name>/`.

```bash
apex predict PREDICTION.toml       # resumable: a finished member is not forecast again
apex report RUN_DIR/predictions/NAME --plots
apex summary RUN_DIR/predictions/NAME
```

```toml
[prediction]
name = "to_2030"
run_dir = "runs/case"                 # the history-matching run
ensembles = ["posterior", "prior"]    # also "iteration:<k>", and "truth" in a synthetic case

[prediction.outputs]
vectors = ["WOPR:*", "FOPT"]          # patterns, at every report time ("report") or given times
times = "report"

[simulator]                            # the forecast's deck: the history, then the forecast
name = "opm_flow"
options = { deck_template_file = "deck_forecast.mako" }
```

- **The ensembles:** `posterior` is the run's selected iterate (the minimum-CV one with cross validation, else the last), `prior` its prior, `iteration:<k>` any accepted iterate, and `truth` the true model of a synthetic case.
- **Its run must exist:** a prediction of a `run_dir` that holds no run (no `config.json`; a typo, or a run not started yet) is refused before anything is made. `apex validate` accepts its form and says in a note that the run is missing; a study's predictions wait for their runs.
- **The history check:** a prediction fails unless it reproduces the run's stored forecasts over the history (`check_history`, on by default), so the forecast deck must keep the run's schedule up to the last observation.
- **Percentiles:** `p10` is the 10th percentile, unless `percentile_convention = "exceedance"` (the SPE-PRMS labels).
- **Scores:** against the truth, over the times after the last observation (`coverage_p10_p90`, `crps`, `mean_error`, `spread`).
- **Provenance:** `provenance.json` records APEX's version, the host, the input files' hashes, and the control scenario, if any.

## A control scenario

After a history match the wells are often operated differently: they leave history mode for BHP or rate control, and their targets change. A prediction can set those future controls as a **control scenario**: fixed values that every member, and the true model, is given besides its state, and that the deck template reads.

```toml
[simulator.options]
deck_template_file = "../model/deck_forecast.mako"   # e.g.  PROD1 OPEN BHP 5* ${bhp[0]} /

[[controls.variables]]
name = "bhp"
values = [130.0, 140.0, 120.0]   # one value per control; optional lower and upper bounds are checked

[[controls.parameterizations]]   # optional: maps the controls only, as an optimization's
name = "control_schedule"
options = { variable = "bhp", output = "bhp_steps", knots = [0, 60], n_steps = 122 }
```

- **The controls are not part of the state.** The history match never sees them, and they are left out of the scores and metrics. They come after the run's own parameterizations. A control named as a state variable or property, or as a name the deck template is given otherwise (`member_id`, `case_dir`, `level`, `coarse`, the simulator's `template_values`), is refused.
- **When it is checked:** once the run exists, `apex validate` and a submission check it all: the form, the names, the parameterizations and whether the simulator reads the controls (the simulator is built, nothing is simulated). The prediction's start checks them again, before anything is simulated or written. A prediction of a run that does not exist yet (a study's) is checked at its start.
- **The simulator must read them.** A simulator that declares the names it reads refuses a scenario it would ignore, before anything is simulated:
  - a deck simulator (`opm_flow`, `eclipse`) reads a control when its template names it, e.g. `${bhp[0]}` (a template that includes another template, or reaches its values through `context`, cannot be checked: a warning says so);
  - the analytic simulators read only their own state names (`linear` its `variables`, `toy_reservoir` its controls block);
  - a simulator that does not declare what it reads gets a warning that the scenario may not reach its forecast;
  - a batch simulator that writes only its declared parameters (ERT's) is refused.
- **A scenario is part of the prediction:** another scenario is another prediction, so give it another name. A prediction without `[controls]` is unchanged.
- **Where they belong in the deck:** only after the history, so that the history check still reproduces the run. Template expressions are rendered in the deck template itself, not in the files it includes, so write the forecast's schedule in the template.
- **Where it is recorded:** the scenario, after its parameterizations, is in `provenance.json`; `apex summary` names it in the prediction's notes.
- **An example:** `examples/prediction_controls` history-matches a small box on OPM Flow in about a minute and forecasts it under two scenarios of the producers' bottom-hole pressures, with its forecast template.
- **With a surrogate:** the model must be built over the controls; see [Predictions with a surrogate](surrogates.md#predictions-with-a-surrogate).

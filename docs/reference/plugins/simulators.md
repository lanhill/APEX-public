---
audience: public
generated: reference
---

# Simulators: reference

Every one of these 5 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `eclipse`

Runs Eclipse (E100 or E300) through `eclrun` for one member in its work directory.

Select it with:

```toml
[simulator]
name = "eclipse"
```

| option | type | default | description |
|---|---|---|---|
| `case_dir` | string or string or null | | The folder `${case_dir}` names in the template (its `INCLUDE` paths), for every level; default the template's folder. |
| `deck_template_file` | string or string | required | Mako template of the `.DATA` deck (as for `opm_flow`). |
| `eclrun_args` | array | | Extra `eclrun` arguments before the program, e.g. `["--version=2023.1"]` or `["--nocleanup"]` (PET's default). |
| `executable` | string | eclrun | The `eclrun` command (default `eclrun`, looked up on `PATH`). |
| `keep_files` | boolean | true | Keep the member's deck and outputs. `False` deletes everything but the log, the deck, the `.PRT`/`.ECLEND` reports and the summary files after reading. |
| `levels` | array or null | | Fidelity levels of a multilevel ensemble, filled in from `[method] levels` (`multilevel_esmda`) rather than set here: one table per level, coarsest first, with its first member and size, and optionally its own `deck_template_file` or an `upscale` table that coarsens the grid with Eclipse's `COARSEN`. |
| `licenses` | integer or null | | The number of Eclipse licences available. At most this many members run at once (the process executor honours `max_parallel_runs`); `None` means no cap. |
| `template_values` | object or null | | Extra fixed values passed to the template. |
| `timeout_s` | number | 3600.0 | Wall-clock limit per run; exceeding it kills the run and fails the member. |
| `transient_patterns` | array | | Extra case-insensitive regular expressions: a failed run whose log matches one is transient, and the executor retries it (on top of the built-in licence and file-system patterns, `apex.simulation.deck.TRANSIENT_PATTERNS`). |
| `version` | string | e100 | `"e100"` (black oil, `eclrun eclipse`) or `"e300"` (compositional). |

## `ert`

An ERT case's forward model, run by ERT through the `ert` executor.

Select it with:

```toml
[simulator]
name = "ert"
```

| option | type | default | description |
|---|---|---|---|
| `case_file` | string or string | required | The ERT configuration (`.ert`). |
| `eclbase` | string | required | The case's `ECLBASE`: the simulator's case name in the run path (`<IENS>`, `<ITER>` are replaced). |
| `fields` | object | required | `{state variable: {keyword, file}}`: each field's ERT name (its GRDECL keyword) and the output file the forward model reads in the run path. |
| `parameter_step` | string or null | | The forward-model step that writes them (`APEX_WRITE_PARAMETERS(<CASE>=...)`), put before the case's first `FORWARD_MODEL`. |
| `responses` | string | summary | `"summary"` (default): each realization's summary is read (`read`). `"storage"`: the batch keeps the case's `OBS_CONFIG`, and the responses of every observation type are read from the batch's ERT storage (`read_batch`). |
| `results_program` | array | | For `"storage"`: the program that reads them (`python -m apex.integrations.ert.observations`, set by the case's builder). |
| `skip_forward_steps` | integer | 0 | The case's first `FORWARD_MODEL` steps to leave out of every batch: those that make `FORWARD_INIT` parameters, run once for the prior (`ert_forward_init`). |
| `step_parameters` | object or null | | `{state variable (or property): ERT name}` of the parameters ERT's own writers put into the run path: `GEN_KW` groups, surfaces, and fields in other formats or with transforms. They are saved in `apex_parameters.npz`. |

## `linear`

`y = f(G x)` for a fixed matrix `G`, with `f` the identity or `tanh`.

Select it with:

```toml
[simulator]
name = "linear"
```

| option | type | default | description |
|---|---|---|---|
| `delay_s` | number | 0.0 | Seconds each run waits before answering, to test scheduling and recovery with runs that take time. |
| `fail_above` | number or null | | Runs whose state has a value above this raise `SimulationError`: failures that appear mid-run, as iterates move (deterministic, so resume-safe). |
| `fail_members` | array | | Member ids whose runs raise `SimulationError`, to exercise failure handling. |
| `fields` | array | | `[time, datatype, size]` of each field output: `size` further rows of `G`, after the scalar ones, with keys `(time, datatype, 0..size-1)`. |
| `flaky_members` | object or null | | Member id -> how many times its run fails transiently (a `TransientSimulationError`, as a licence checkout) before it succeeds; to test the executors' retries. Keys may be strings (TOML tables). |
| `forward_file` | string or string or null | | `G` itself `(n_outputs, n_state)`, or an `.npz` holding it as `matrix`. |
| `levels` | array or null | | Fidelity levels of a multilevel ensemble (the composition passes them): per level `first`, `size` and optionally its own `G` (`matrix` or `forward_file`), e.g. a biased coarse model; a level without one uses `G`. Grid upscaling does not apply to this model. |
| `matrix` | ArrayLike or null | | `G` itself `(n_outputs, n_state)`, or an `.npz` holding it as `matrix`. |
| `nonlinearity` | string | none | `"none"` or `"tanh"`. |
| `outputs` | array | required | `[time, datatype]` label of each scalar row of `G`, in row order. |
| `resources` | object or null | | Machine-wide tokens each run holds, e.g. `{"cpu": 1}`, to test admission; none by default. |
| `variables` | array | required | State variables stacked, in this order, into `x`. |
| `write_files` | boolean | false | Give each run a work directory and write `output.txt` there (the values), to test work-directory handling such as retention (`[output] keep_simulations`). |

## `opm_flow`

Runs OPM Flow for one member in its work directory.

Select it with:

```toml
[simulator]
name = "opm_flow"
```

| option | type | default | description |
|---|---|---|---|
| `case_dir` | string or string or null | | The folder `${case_dir}` names in the template (its `INCLUDE` paths), for every level; default the template's folder. A copied example sets it to its own folder, so that its data files are read while the template stays in the examples root. |
| `deck_template_file` | string or string | required | Mako template of the `.DATA` deck. |
| `executable` | string | flow | OPM Flow command (default `flow`, looked up on `PATH`). |
| `flow_args` | array | | Extra command-line arguments, e.g. `["--threads-per-process=1"]`. |
| `keep_files` | boolean | true | Keep the member's deck and outputs (`False` deletes everything but the log and the summary files after reading, to save disk space). |
| `levels` | array or null | | Fidelity levels of a multilevel ensemble, filled in from `[method] levels` (`multilevel_esmda`) rather than set here: one table per level, coarsest first, with its first member and size, and optionally its own `deck_template_file`. OPM Flow has no `COARSEN`, so a coarse level needs its own deck, not `upscale`. |
| `template_values` | object or null | | Extra fixed values passed to the template. |
| `timeout_s` | number | 3600.0 | Wall-clock limit per run; exceeding it fails the member. |
| `transient_patterns` | array | | Extra case-insensitive regular expressions: a failed run whose log matches one is transient, and the executor retries it (on top of the built-in licence and file-system patterns, `apex.simulation.deck.TRANSIENT_PATTERNS`). |

## `toy_reservoir`

Field totals of a toy waterflood, for optimization tests and demos (/0043).

Select it with:

```toml
[simulator]
name = "toy_reservoir"
```

| option | type | default | description |
|---|---|---|---|
| `control` | string | inj | The name of the controls block (the member's state key). |
| `oil_rate` | number | 20.0 | Yearly oil per square root of pressure, and water per unit of pressure. |
| `start` | string | 2000-01-01 | The simulation's start (ISO date); `TIME` is the days since. |
| `water_rate` | number | 1.0 | Yearly oil per square root of pressure, and water per unit of pressure. |

---
audience: public
---

# APEX and ERT

**Goal.** Use APEX together with Equinor's [ERT](https://github.com/equinor/ert): export an APEX case to ERT, run ERT on it, import the results back, run APEX's methods from ERT, and run ERT's methods inside APEX studies.

## Prerequisites

- A working APEX installation and a configuration file (`CONFIG.toml`).
- ERT, installed as the optional `ert` extra. It is large (about 100 packages, including PyQt6) and GPL-3.0-only, so a distribution that includes it is GPL-3.0-only.
- The extra is ERT 26 (26.0.1 or later within 26.x). A case whose storage an older ERT wrote is migrated to ERT 26's format the first time APEX writes to it, as ERT itself does, after which an older ERT cannot open it.

```bash
uv sync --locked --extra ert   # in APEX's folder; repeat any other extras you use, e.g. --extra co2 --extra ert
```

Each `uv sync` installs exactly what it is given, so name every extra you need in one command. When APEX is a dependency of your own project, name the `ert` extra there, as the README's *Using APEX from another project* shows (`apex[ert]`, installed from APEX's repository, not from PyPI, whose `apex` is an unrelated package).

ERT finds APEX's plugin by itself (entry point `ert`). The plugin adds the forward-model step `APEX_RENDER_DECK`.

### Terms used below

- **State variable**: a quantity APEX updates, for example a grid field such as permeability.
- **Prior**: the ensemble of state values before any data is assimilated.
- **Realization / member**: one ensemble member. ERT calls it a realization; APEX calls it a member.
- **Iterate**: one step of an iterative update. Each iterate has its own parameters and responses.
- **C_d**: the observation error covariance. APEX's ES-MDA inflates it per iteration; "un-inflated" means the original errors.
- **Study**: a set of runs (cases, seeds, methods) launched with `apex study run` and compared with `apex study compare`.

## Step 1: Export a case

```bash
uv run apex ert export CONFIG.toml OUT_DIR [--ert-defaults] [--overwrite]
```

`OUT_DIR` then holds a self-contained ERT case:

| File | What it is |
|---|---|
| `case.ert` | The ERT configuration: `NUM_REALIZATIONS`, a `FIELD` per grid variable (`INIT_FILES` from APEX's prior, bounds as `MIN`/`MAX`), a `SUMMARY` per simulated key, `OBS_CONFIG`, `FORWARD_MODEL APEX_RENDER_DECK` then `FLOW`, `QUEUE_SYSTEM LOCAL` with `MAX_RUNNING` from the workers, and `RANDOM_SEED` from the run's seed |
| `prior/<FIELD>_<i>.grdecl` | APEX's prior, in Eclipse cell order |
| `grid/<CASE>.GRID` | The grid's dimensions (`DIMENS`) |
| `observations/obs.txt`, `observations.csv` | ERT's summary observations: `keyword`, `value`, `error` (the standard deviation), `date`, `well` |
| `observations/held_out.csv` | The data of a `[validation]` split, with their roles. ERT simulates them (their keys are in `SUMMARY`) but does not assimilate them (they are not in `OBS_CONFIG`) |
| `apex_export.json` | What `APEX_RENDER_DECK` needs: the deck template, its values, each field's state variable |

By default the case switches off ERT's outlier rejection (`ENKF_ALPHA`) and keeps all but exactly constant responses (`STD_CUTOFF 1e-30`), as APEX does. `--ert-defaults` keeps ERT's own settings instead.

### What can be exported

OPM Flow cases of grid fields, and also:

- **Eclipse decks** (the `eclipse` simulator). ERT's `ECLIPSE100` or `ECLIPSE300` step runs after `APEX_RENDER_DECK`. The release comes from the `eclipse` simulator's `eclrun_args` option, for example `eclrun_args = ["--version=<release>"]`; without it, export refuses the case. The same simulator has a `licenses` option. Both options are set in the case's simulator section of the APEX configuration, not in a history-match section.
- **Scalar state variables.** They travel together as one `SURFACE APEXSCAL`, a 1 × n binary Irap surface, with APEX's prior in `prior/APEXSCAL_<i>.irap`. `APEX_RENDER_DECK` and the import read it back.
- **Parameterizations.** The state (the latent fields) is exported, and `APEX_RENDER_DECK` maps it to the properties before rendering, so ERT updates what APEX updates.
- **Data transforms.** ERT assimilates the transformed data (`GEN_DATA APEX_TRANSFORMED`, with a `GENERAL_OBSERVATION` of the transformed observations). The only built-in transform, `wavelet`, compresses field data. The deck simulators do not produce field data, so today no built-in transform can be used in an export.

### What is refused

- Field data, which no deck simulator produces.
- Simulators other than `opm_flow` and `eclipse`.
- Grid fields on different grids.

## Step 2: Run ERT

```bash
uv run apex ert run OUT_DIR --mode es_mda --weights 4,4,4,4   # or ensemble_smoother, ensemble_experiment, test_run
```

This runs `ert <mode> case.ert` in `OUT_DIR`, with APEX's environment first on `PATH`. ERT users can run the same command themselves.

Results are in ERT's `storage/` and in `runs/realization-<i>/iter-<k>/`, where `<i>` is the realization number and `<k>` the iteration. You can open them in the ERT GUI, or import them into APEX (next step).

## Step 3: Import ERT's results into APEX

Run the import first; the report needs the run directory the import creates.

```bash
uv run apex ert import OUT_DIR [RUN_DIR]     # default RUN_DIR: OUT_DIR/apex_run
uv run apex report OUT_DIR/apex_run          # also: apex summary, comparisons, agents
```

The import replays ERT's iterations through APEX's own workflow. Each iterate's parameters are forecast by a stand-in simulator that returns the summary ERT simulated for exactly those parameters. The run directory then looks like an APEX run's: data mismatch, performance metrics, the held-out data's scores, `summary.json`, and `ert_import.json` (`source = "ert"`). A realization for which ERT has no summary fails in the replay too.

## The deck

`APEX_RENDER_DECK` renders the case's Mako deck in each run path, from the field files ERT wrote there, using APEX's own renderer. ERT stores fields as float32, so a field ERT owns reaches the deck with about 7 significant digits; the deck can differ from APEX's double-precision rendering in the last printed digit.

## History-match an ERT case with APEX's methods

An ERT user keeps their case and adds an `apex.toml` next to it, with APEX's method and, optionally, `[run]` (name, seed), `[validation]`, `[[metrics]]`, `[output]` and `[prior]`.

```toml
[run]
name = "gies"

[method]
name = "gies"
strategy = { name = "rlmmac" }
localization = { name = "adaptive_prior" }
```

Save a workflow file next to `case.ert`, for example `apex_history_match`, naming the job and the path to your `apex.toml`:

```text
APEX_HISTORY_MATCH ./apex.toml
```

Add this line to `case.ert` to load it:

```text
LOAD_WORKFLOW apex_history_match APEX_HISTORY_MATCH
```

Then run it from ERT, or from APEX:

```bash
uv run ert workflow APEX_HISTORY_MATCH case.ert
uv run apex ert history-match case.ert [apex.toml] [--max-steps N]
```

`--max-steps N` stops after N analysis steps and prints `stopped (resume: run the same command again)`. To continue, run the same command again (with or without `--max-steps`): the run in `apex_runs/<name>/` is continued from its last checkpoint, not started anew. `ert workflow APEX_HISTORY_MATCH case.ert` continues it the same way.

### What APEX runs

APEX runs its own loop, including checkpoints, cross validation, rejected trials and damping. ERT runs the case's forward model for every forecast, as one `ert ensemble_experiment`: the executor is `ert` with the simulator `ert`. The run goes to `apex_runs/<name>/` next to the case, and a run that is already there is continued.

Afterwards, the iterates are added to the case's ERT storage as the experiment `apex:<name>`, with one ensemble per iterate. They show in ERT's GUI next to ERT's own experiments. `apex ert history-match --no-write-back` skips this.

The parameters are written exactly, not through ERT's float32 storage.

### Several methods in one workflow

To compare APEX's methods in ERT's GUI, list them as `[methods.<label>]` instead of one `[method]`. Each entry is a complete `[method]`:

```toml
[run]
name = "compare"

[methods.gies]
name = "gies"
strategy = { name = "rlmmac" }

[methods.esmda]
name = "esmda"
options = { inflation = [4, 4, 4, 4] }

[methods.esmda_adaptive]
name = "esmda"
options = { inflation = [4, 4, 4, 4] }
localization = { name = "adaptive_prior" }
```

The workflow file and the `LOAD_WORKFLOW` line are the same as for a single method, and you run the same command: `ert workflow APEX_HISTORY_MATCH case.ert`. The methods run in order:

- Each method runs into `apex_runs/compare/<label>/` (for example `apex_runs/compare/gies/`), with the same seed and so the same prior.
- Each run is written back as the experiment `apex:compare:<label>` (for example `apex:compare:gies`), which is the name you see in ERT's GUI.
- A method that fails does not stop the others; the workflow fails at the end and names it.
- `apex ert history-match case.ert [apex.toml] --methods gies,esmda` runs only those labels.

### What APEX takes from the case

| From the case | In APEX |
|---|---|
| each `FIELD` | a state variable, named in lower case |
| `MIN` / `MAX` | the variable's bounds |
| `INIT_FILES` | the prior, unless `apex.toml` has a `[prior]` |
| summary observations | the data; `variance = error²` |
| `NUM_REALIZATIONS` | the ensemble size |
| `MAX_RUNNING` | the number of workers |
| `RANDOM_SEED` | the seed, unless `[run] seed` is set |

The configuration APEX builds, and its inputs, are in `apex_runs/<name>.inputs/`.

### Parameters

- **`FIELD`**: any format (GRDECL, binary GRDECL, ROFF) and any `INIT_TRANSFORM`/`OUTPUT_TRANSFORM`.
- **`GEN_KW`**: each group is one state variable in ERT's standard-normal space.
- **`SURFACE`**: the values of the surface, with the prior from `INIT_FILES`.
- **`FORWARD_INIT:True`**: tell APEX how many of the case's first `FORWARD_MODEL` steps generate the parameters (rather than simulate) by setting the `ert` simulator's `skip_forward_steps` option. Those steps are run once to build the prior and left out of every forecast.

### Limits

- `DESIGN_MATRIX` is refused.
- Only local runs are supported.
- RFT, breakthrough and seismic responses are not written back to ERT's storage; summaries and `GEN_DATA` are.
- A deck must use APEX's member id, which is written as `APEX_MEMBER` in each run path, and not ERT's realization number. The two can differ, so if your deck uses the realization number to pick a file or a seed, switch it to the member id.

## An ERT case as a native APEX case

```bash
uv run apex ert convert case.ert native/ [--method esmda]
uv run apex run native/config.toml
```

For a case whose forward model is one step (`FLOW`, `ECLIPSE100` or `ECLIPSE300`) on a `DATA_FILE`, this writes an APEX case that runs on `opm_flow` or `eclipse` without ERT. The `DATA_FILE` becomes `model/deck.mako`, the `FIELD`s and `GEN_KW` groups become the state, and summary observations become APEX's CSV. Templates ERT writes but the deck does not include are listed in `convert.json`.

Cases with other forward-model steps, `SURFACE`, other field formats, field transforms, `FORWARD_INIT` or other observation types are refused. Keep those on `APEX_HISTORY_MATCH`.

## ERT's methods in APEX's studies

A study is run with `apex study run <study file>`. A method entry in the study names the engine, ERT's run model and its options:

```toml
[study.methods.esmda]                 # APEX's ES-MDA
name = "esmda"
options = { inflation = [4, 4, 4, 4] }
strategy = { name = "approx" }

[study.methods.ert_esmda]             # ERT's ES-MDA, on the same cases, seeds and priors
engine = "ert"
name = "es_mda"                       # or ensemble_smoother
options = { weights = [4, 4, 4, 4] }
```

An ERT cell exports the cell's configuration, runs ERT in `<run_dir>.ert/`, and imports the result into the cell's run directory. `apex study compare` and reports then read it like any run.

Limits:

- Comparisons are statistical: ERT perturbs the observations itself and stores float32.
- ERT has no resume, so a stopped ERT cell starts again.
- Cases that export refuses are refused before anything runs.

For one case there is a shortcut:

```bash
uv run apex ert compare config.toml --weights 4,4,4,4 --replicates 3 [--apex-method gies] [--yes] [--wait]
```

It builds a study that pits the case's own APEX method (or `--apex-method`) against ERT's run model, with `--replicates 3` giving three replicates of each. `--yes` submits the study without asking, and `--wait` makes the command wait for it to finish. Compare the results with `apex study compare <name>_apex_vs_ert`.

## ERT's smoothers inside APEX's methods

ERT's analysis library (`iterative_ensemble_smoother`, installed with the `ert` extra) provides three update strategies for APEX's `es` and `esmda`:

```toml
[method]
name = "esmda"
options = { inflation = [4, 4, 4, 4] }
strategy = { name = "ert_esmda", options = { truncation = 0.99 } }
```

The other two are `ert_adaptive` (option `threshold`) and `ert_localized` (option `taper_file`, for example `taper.npz`). Use one by replacing the `strategy` line.

APEX still draws the perturbations and inflates `C_d`; the library only computes the step. So seeds, checkpoints and resume behave as with APEX's own strategies.

- `ert_adaptive` recomputes each state row's update over the data it correlates with, `|corr| > threshold`. The same method runs natively, without the `ert` extra, as `adaptive_rows`.
- `ert_localized` uses a taper file whose shape is `n_state x n_data`: one row per state entry and one column per datum. Only the shape is checked, so you must build the taper for exactly the configuration it is used with.
- A localization prepared from the prior, such as `adaptive_prior`, computes its taper once from the prior at the first update and reuses it for later updates, as in an APEX run.

## Pitfalls

- Running `apex report` on `OUT_DIR/apex_run` before `apex ert import` fails, because the run directory does not exist yet.
- Putting `eclrun_args` or `licenses` in a history-match section: they belong to the `eclipse` simulator.
- Using ERT's realization number in a deck instead of the member id `APEX_MEMBER`.
- Passing an `ert_localized` taper of the wrong shape.

## What to read next

- [Studies](guide/cli/study.md), for `apex study run` and `apex study compare`.
- [Cross validation](guide/plugins/stopping_rules.md#cross-validation), for `[validation]` splits and their stopping rules.

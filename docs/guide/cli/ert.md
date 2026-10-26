---
audience: public
items:
  - cli:apex ert compare
  - cli:apex ert convert
  - cli:apex ert export
  - cli:apex ert history-match
  - cli:apex ert import
  - cli:apex ert run
---

# The apex ert commands

## Goal

The `apex ert` commands connect APEX to ERT. After this lesson you will be able to:

- hand an APEX configuration to ERT as an ERT case (`apex ert export`),
- run ERT on that case locally (`apex ert run`),
- bring ERT's iterations back into an APEX run directory (`apex ert import`),
- history-match an existing ERT case with APEX's methods while ERT runs the forward model (`apex ert history-match`),
- turn an ERT case into a native APEX case that runs without ERT (`apex ert convert`),
- compare APEX's method with ERT's over several seeds (`apex ert compare`).

## Prerequisites

- APEX installed, with an APEX run configuration (TOML) for the case you want to work on.
- For `apex ert run`: ERT 26.x, installed through the `ert` extra.
- For `apex ert convert` and `apex ert history-match`: an ERT case (a `.ert` file). For `apex ert convert` it must be on FLOW or ECLIPSE100/300.

## Which command do I need?

| Command | What it does |
|---|---|
| `apex ert export` | Writes an APEX configuration as a self-contained ERT case (`case.ert`, prior, observations, grid). |
| `apex ert run` | Runs ERT locally on a folder written by `apex ert export`. |
| `apex ert import` | Replays ERT's iterations into an APEX run directory (reports, summaries, comparisons). |
| `apex ert history-match` | History-matches an ERT case with APEX's methods; ERT runs the forward model. |
| `apex ert convert` | Writes an ERT case as a native APEX case that runs without ERT. |
| `apex ert compare` | Compares APEX's method with ERT's on one case, over seeds, as a study. |

## Steps

### 1. Export an APEX configuration to ERT

```bash
uv run apex ert export CONFIG OUT_DIR
```

- `CONFIG` is the APEX run configuration (TOML). `OUT_DIR` is the folder for the ERT case.
- `--ert-defaults` keeps ERT's outlier rejection and `STD_CUTOFF`. `--no-ert-defaults` turns that off. The default is off.
- `--overwrite` replaces a case already in `OUT_DIR`. `--no-overwrite` is the default.

### 2. Run ERT on the exported case

```bash
uv run apex ert run CASE_DIR
```

- `CASE_DIR` is a folder written by `apex ert export`.
- `--mode` chooses ERT's experiment: `ensemble_experiment`, `ensemble_smoother`, `es_mda` or `test_run`. The default is `es_mda`.
- `--weights` sets the ES-MDA weights, for example `4,4,4,4`. It applies to mode `es_mda`.
- `--dry-run` only prints the command. `--no-dry-run` is the default.

### 3. Import ERT's iterations into APEX

```bash
uv run apex ert import CASE_DIR [RUN_DIR]
```

- `CASE_DIR` is an exported case that ERT has run.
- `RUN_DIR` is the APEX run directory. If you leave it out, APEX uses `CASE_DIR/apex_run`.

### 4. History-match an ERT case with APEX's methods

```bash
uv run apex ert history-match CASE [APEX_TOML]
```

- `CASE` is an ERT case (`.ert`): fields, observations, steps.
- `APEX_TOML` holds APEX's choices. If you leave it out, APEX uses `apex.toml` next to `CASE`.
- `--methods` takes a comma-separated list of `[methods.<label>]` sections to run. The default is all of them.
- `--max-steps` stops after that many analysis steps. Rerun the command to continue.
- `--progress` shows progress on standard error (default on); `--no-progress` turns it off.
- `--write-back` adds the iterates to the case's ERT storage, so you can see them in ERT's GUI (default on); `--no-write-back` turns it off.

### 5. Convert an ERT case to a native APEX case

```bash
uv run apex ert convert CASE OUT_DIR
```

- `CASE` is an ERT case (`.ert`). `OUT_DIR` is the folder for the native APEX case.
- `--method` sets APEX's method in the configuration. The default is `esmda`.
- `--overwrite` replaces a case already in `OUT_DIR`. `--no-overwrite` is the default.

### 6. Compare APEX with ERT over seeds

```bash
uv run apex ert compare CONFIG
```

- `CONFIG` is an APEX history-matching configuration.
- `--mode` is ERT's run model: `es_mda` (default) or `ensemble_smoother`.
- `--weights` sets the ES-MDA weights, for example `4,4,4,4`. The default is ERT's `4,2,1`.
- `--apex-method` names the APEX method to compare, with its defaults. The default is the case's own method.
- `--replicates` sets the seeds per method. The default is 3.
- `--name` names the study.
- `--yes` approves without asking.
- `--wait` waits for the study to finish. `--no-wait` is the default.

## Worked example: round trip through ERT

You have an APEX configuration `history_match.toml` and want ERT to run it, then look at the result in APEX.

1. Export the case. Replace the file and folder names with your own:

   ```bash
   uv run apex ert export history_match.toml ert_case
   ```

2. Check what ERT would do, without running it:

   ```bash
   uv run apex ert run ert_case --mode es_mda --weights 4,4,4,4 --dry-run
   ```

3. Run it for real:

   ```bash
   uv run apex ert run ert_case --mode es_mda --weights 4,4,4,4
   ```

4. Replay ERT's iterations into APEX. With no `RUN_DIR`, the result goes to `ert_case/apex_run`:

   ```bash
   uv run apex ert import ert_case
   ```

5. To see how APEX's method fares against ERT's on the same configuration, over three seeds each, start a study and wait for it:

   ```bash
   uv run apex ert compare history_match.toml --replicates 3 --name first-comparison --wait
   ```

## Pitfalls

- `apex ert run` needs a folder written by `apex ert export`. It does not accept a plain `.ert` case from elsewhere.
- `apex ert run` needs ERT 26.x from the `ert` extra. Use `--dry-run` to see the command before you commit to a run.
- `--weights` only has an effect in the `es_mda` mode.
- `apex ert export` and `apex ert convert` do not replace an existing case in `OUT_DIR` unless you pass `--overwrite`.
- `apex ert export` leaves ERT's outlier rejection and `STD_CUTOFF` off unless you pass `--ert-defaults`.
- `apex ert history-match` writes the iterates back into the case's ERT storage by default. Pass `--no-write-back` if you do not want that.
- `apex ert history-match` can stop early with `--max-steps`. Rerun the command to continue.
- `apex ert compare` asks for approval before it starts unless you pass `--yes`. It also returns without waiting unless you pass `--wait`.
- The `apex ert import` and `apex ert run` commands work on an exported case folder (`CASE_DIR`). `apex ert history-match` and `apex ert convert` take the ERT case itself (`CASE`). Do not mix them up.

## What to read next

- The page on APEX history-matching configurations, for the TOML that `apex ert export` and `apex ert compare` read.
- The page on run directories and reports, for what you can do with the output of `apex ert import`.

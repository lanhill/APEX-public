---
audience: public
items:
  - plugin:converter_skills:eclipse
  - plugin:converter_skills:ert
  - plugin:converter_skills:opm_flow
  - plugin:converter_skills:pet
---

# Converter skills

## Goal

Converter skills let APEX read the simulation and workflow files you already have and turn them into APEX cases. Some of them also write results back out. After this lesson you will know which converter fits which file, and which command to start from.

## Prerequisites

- APEX is installed as described in the README. `apex` is not on your PATH, so every command below is run from the APEX folder as `uv run apex ...`.
- You have an input file of one of the kinds in the table below.
- For ERT cases, the `ert` extra is installed: `uv sync --locked --extra ert` in APEX's folder (see the [ERT page](../../ert.md)).

## The converters

Four converter skills are available. None of them has any options, so there is nothing to configure in a TOML file. You choose a converter by the kind of file you have.

| Plugin | Input | What it does |
| --- | --- | --- |
| `eclipse` | Eclipse decks (E100/E300, `.DATA`) | Converts the deck to an APEX case on `eclipse` with `apex agent convert import --skill eclipse`. |
| `opm_flow` | OPM Flow decks (`.DATA`) | Converts the deck to an APEX case on `opm_flow` with `apex agent convert import --skill opm_flow`. |
| `ert` | ERT cases (`.ert`) | Converts to an APEX case with `apex ert convert`, and back with `apex ert export`. |
| `pet` | PET configurations (`.toml`) | Converts to an APEX case with `apex pet-import`, and back with `apex agent convert to-pet`. |

Both deck formats use the `.DATA` extension. The skill you pick decides which simulator the converted case runs on: `eclipse` for Eclipse decks, `opm_flow` for OPM Flow decks.

## Steps

1. Identify your file type from the table. For example, a `.ert` file points to `ert`, and a PET `.toml` file points to `pet`.
2. Run the matching command. The paths below are placeholders for your own files, and the output folder must be new:

   ```bash
   # an OPM Flow deck (use --skill eclipse for an Eclipse deck)
   uv run apex agent convert import CASE.DATA --out apex_case --skill opm_flow
   # an ERT case
   uv run apex ert convert case.ert apex_case
   # a PET configuration
   uv run apex pet-import pet.toml apex_case
   ```

   A deck needs its observed data: give `--history` a CSV (`time,datatype,value`) or a summary file, or leave it out when the deck itself has `WCONHIST`/`WCONINJH`. The deck converter asks the model of your agent settings for its choices; see [what a provider receives](../../agents.md#what-a-provider-receives-and-how-to-keep-it-local), or add `--no-llm` to use its rules alone. Step 9 of the [agent commands](../cli/agent.md) covers it in full. Add `--help` to any command to see all its options.
3. Each command writes the APEX case into the output folder, with its configuration in `apex_case/config.toml`. Check it with `uv run apex validate apex_case/config.toml` before you run anything.
4. To go back to the original format, give the APEX configuration and a new folder:

   ```bash
   uv run apex ert export apex_case/config.toml ert_case
   uv run apex agent convert to-pet apex_case/config.toml pet_case
   ```

## Worked example: ERT round trip

Suppose you have an ERT case in `case.ert` and want to work on it in APEX, then hand it back.

1. Convert it to an APEX case in a new folder:

   ```bash
   uv run apex ert convert case.ert apex_case
   ```

2. Check the case, then make your changes to `apex_case/config.toml`:

   ```bash
   uv run apex validate apex_case/config.toml
   ```

3. Export it again as an ERT case:

   ```bash
   uv run apex ert export apex_case/config.toml ert_case
   ```

   `ert_case` is a self-contained ERT case that ERT can run. It is APEX's own rendering of the case (its `case.ert`, prior, observations and grid), not your original `.ert` file.

The PET round trip is the same, with `uv run apex pet-import pet.toml apex_case` on the way in and `uv run apex agent convert to-pet apex_case/config.toml pet_case` on the way back. It needs a PET configuration of your own: see the pitfall on `to-pet` below.

## Pitfalls

- Eclipse and OPM Flow decks both end in `.DATA`. The extension does not tell them apart, so make sure you know which simulator the deck was written for before choosing `eclipse` or `opm_flow`.
- `ert` and `pet` are not interchangeable. `pet` takes PET configurations, which are `.toml` files, not ERT cases.
- The converter skills have no TOML options; what you can choose is in the commands' flags (`--help`). If a conversion looks wrong, check the input file first.
- `apex ert convert` takes ERT cases whose forward model is one `FLOW`, `ECLIPSE100` or `ECLIPSE300` step on a `DATA_FILE`, and refuses others. `apex ert export` refuses cases with field data, data transforms together with a `[validation]` split, and model-error models. Each refusal states its reason.
- `apex pet-import` writes a case on `opm_flow` by default; give `--simulator eclipse` for an Eclipse case.
- `apex agent convert to-pet` writes only what PET can run as APEX does, so the way back is meant for cases PET can run, typically ones that came from PET. It refuses, with the reason: simulators other than OPM Flow and Eclipse, data transforms and field data (not yet exported), geophysical data, parameterizations, model-error models, cross validation (`[validation]`), localization, damping, methods PET lacks (such as GIES and CHOP), method options PET has no place for, template values, one-sided bounds, joint priors, priors other than a stored ensemble or a variogram PET's sampler reproduces, and a few template, observation and file-layout limits, each stated in its refusal. None of the examples shipped with APEX is such a case, so try it on a case you imported from PET.
- Do not type `apex ...` on its own. It is not on the PATH, so always start with `uv run` from the APEX folder.

## What to read next

Continue with the other plugin pages in this guide, such as the pages for the simulators your converted cases run on.

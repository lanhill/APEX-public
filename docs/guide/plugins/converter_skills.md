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

## The converters

Four converter skills are available. None of them has any options, so there is nothing to configure in a TOML file. You choose a converter by the kind of file you have.

| Plugin | Input | What it does |
| --- | --- | --- |
| `eclipse` | Eclipse decks (E100/E300, `.DATA`) | Reads the deck and converts it to an APEX case on `eclipse`. |
| `opm_flow` | OPM Flow decks (`.DATA`) | Reads the deck and converts it to an APEX case on `opm_flow`. |
| `ert` | ERT cases (`.ert`) | Converts to an APEX case with `apex ert convert`, and back with `apex ert export`. |
| `pet` | PET configurations (`.toml`) | Converts to an APEX case with `apex pet-import`, and back. |

Both deck formats use the `.DATA` extension. The skill you pick decides which simulator the converted case runs on: `eclipse` for Eclipse decks, `opm_flow` for OPM Flow decks.

## Steps

1. Identify your file type from the table. For example, a `.ert` file points to `ert`, and a PET `.toml` file points to `pet`.
2. For `ert` and `pet`, run the matching command. Start with its help text to see what it accepts:

   ```bash
   uv run apex ert convert --help
   uv run apex pet-import --help
   ```

3. Run the conversion on your file, using the arguments shown in the help.
4. Open the APEX case that was produced and check it before you run anything.
5. To go back to the original format, use `uv run apex ert export` for ERT. The `pet` skill also converts back to a PET configuration.

## Worked example: ERT round trip

Suppose you have an ERT case in a `.ert` file and want to work on it in APEX, then hand it back.

1. Convert it to an APEX case:

   ```bash
   uv run apex ert convert --help
   ```

   Use the arguments listed there to point the command at your `.ert` file.
2. Inspect the APEX case and make your changes.
3. Export it again as an ERT case:

   ```bash
   uv run apex ert export --help
   ```

   Again, follow the listed arguments.

The PET workflow is the same, with `uv run apex pet-import` for the way in.

## Pitfalls

- Eclipse and OPM Flow decks both end in `.DATA`. The extension does not tell them apart, so make sure you know which simulator the deck was written for before choosing `eclipse` or `opm_flow`.
- `ert` and `pet` are not interchangeable. `pet` takes PET configurations, which are `.toml` files, not ERT cases.
- The converter skills have no options, so there is no TOML section to tune. If a conversion looks wrong, check the input file first.
- Do not type `apex ...` on its own. It is not on the PATH, so always start with `uv run` from the APEX folder.

## What to read next

Continue with the other plugin pages in this guide, such as the pages for the simulators your converted cases run on.

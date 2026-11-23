---
audience: public
items:
  - plugin:case_state_readers:eclipse
  - plugin:case_state_readers:opm_flow
---

# Case state readers

## Goal

By the end of this lesson you will know what a case state reader does in APEX, which two readers exist (`opm_flow` and `eclipse`), and how to choose between them.

## Prerequisites

- A finished simulation deck run that left its grid and state files on disk: the `EGRID` file, the `INIT` file and the restart files.
- APEX installed as described in the README. Run `apex` commands from the APEX folder as `uv run apex ...`.

## What a case state reader is

A case state reader opens the output of a deck run and gives the rest of APEX the run's state. Each reader in this family supplies one thing: the `case`. Plugins that need the grid, the static properties or the dynamic state of a run work from that `case`.

The readers in this registry are:

| Plugin | What it is |
|---|---|
| `opm_flow` | Reads a deck run's `EGRID`, `INIT` and restarts. Values stay in the deck's units, and restarts are addressed by step. |
| `eclipse` | An alias of `opm_flow`. It reads the same files in the same way. |

## Steps

1. **Check the run's files.** Confirm that the run directory contains the `EGRID`, `INIT` and restart files for the case.
2. **Pick a reader.** Use `opm_flow`, or `eclipse` if you prefer that name. Because `eclipse` is an alias, the choice changes only the name you write.
3. **Leave the options alone.** Neither plugin has any options. Their option schemas are empty and accept nothing extra, so there is nothing to tune.
4. **Work in the deck's units.** Whatever units the deck was written in are the units you get back. The reader does not convert them.
5. **Ask for restarts by step.** The restarts are indexed by simulation step, not by date or by time.

## Worked example

Suppose a deck run has finished and its directory holds an `EGRID` file, an `INIT` file and restart files for several steps.

- With `opm_flow` selected, the reader exposes the grid from the `EGRID`, the static properties from the `INIT`, and the restart for any step you ask for. A pressure value is returned in whatever pressure unit the deck used, for example bar if the deck is in METRIC.
- With `eclipse` selected, you get exactly the same `case`, with the same units and the same step-based restarts.

If you mix names between two configurations, the results do not differ, because both names point to the same reader.

## Pitfalls

- **Treating `eclipse` as a different reader.** It is only an alias of `opm_flow`. It does not add features, options or a different file format.
- **Looking for options.** There are none. Adding extra settings to either plugin is not accepted, because the option schema allows no additional properties.
- **Assuming unit conversion.** Values come back in the deck's units. If you compare two runs, check that their decks use the same unit system.
- **Mixing up step and time.** Restarts are chosen by step. A step number is not a date.
- **Missing files.** The reader needs the `EGRID`, `INIT` and restart files from the run. A run that stopped before writing them cannot be read.

## What to read next

Continue with the other plugin pages in the guide's plugins section. They show which plugins consume the `case` that these readers supply.

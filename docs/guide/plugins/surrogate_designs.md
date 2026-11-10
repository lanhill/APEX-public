---
audience: public
items:
  - cli:apex plugins
  - cli:apex run
  - plugin:priors:file
  - plugin:surrogate_designs:file
  - plugin:surrogate_designs:lhs
  - plugin:surrogate_designs:prior
  - plugin:surrogate_designs:run
---

# Surrogate designs

A surrogate design plan decides which input points a surrogate is built from. APEX has four, registered in `surrogate_designs`: `prior`, `lhs`, `file` and `run`. They are different plans, each taking its points from a different source. This page helps you choose one.

## Goal

By the end of this page you will know:

- where each design plan gets its points,
- which plan fits which situation,
- how to list the plans and confirm that they take no options of their own.

## Prerequisites

- APEX installed as described in the README. Run commands from the APEX folder as `uv run apex ...`.
- A configuration for a case, started with `uv run apex run <config>`, where you choose a design plan by name.

## The four plans

| Plan | Where the points come from |
|------|----------------------------|
| `prior` | Draws from the case's prior generators, so fields keep their geostatistics. |
| `lhs` | A Latin hypercube over the state, or over an optimization's controls (`[design.box]`). A Latin hypercube spreads points evenly: each variable's range is cut into slices and every slice is used once. |
| `file` | Columns of a stored ensemble, either of the state or of controls (`ensemble_file`). |
| `run` | The forecasts a history-matching run already holds (`run_dir`). No simulator runs are needed. |

## Options

None of the four plans has options of its own. The option set of each is empty and unknown options are rejected, so do not add option keys under a design plan. What a plan needs, such as `ensemble_file` for `file` or `run_dir` for `run`, is named in the plan's description above.

## Steps

1. Decide where your design points should come from:
   - the case's priors, with realistic fields: `prior`;
   - an even, space-filling spread over a box of state or control values: `lhs`;
   - points you already stored in an ensemble file: `file`;
   - forecasts that a history-matching run already holds, so that no new simulations are needed: `run`.
2. Write that name where your configuration asks for a surrogate design plan, and leave the plan's options empty.
3. List the plans to check the name and the empty option set:

```bash
uv run apex plugins
```

For full descriptions in machine-readable form, add `--json`:

```bash
uv run apex plugins --json
```

4. Start the run as usual:

```bash
uv run apex run <config>
```

To run only part of the work and continue later, add `--max-steps N` (stop after N analysis steps; continue with `apex resume`). Progress is shown on standard error; `--no-progress` turns it off.

## Worked example

Suppose you have finished a history-matching run and want a surrogate without paying for new simulations. The plan that fits is `run`: it uses the forecasts that run already holds (`run_dir`).

Now suppose you want to explore an optimization's controls evenly. The plan that fits is `lhs`: it builds a Latin hypercube over the controls in `[design.box]`.

In both cases, run `uv run apex plugins` and find the plan under `surrogate_designs`. You will see that its option set is empty, so the plan name is all you write for it.

## Pitfalls

- **Treating the names as synonyms.** Each plan draws its points from a different source. Switching from `lhs` to `prior` changes the design.
- **Adding options.** The design plans have no options, and extra keys are rejected.
- **Confusing `file` in two registries.** The `file` plan in `surrogate_designs` has no options. The `file` plugin in `priors` is a different plugin: it reads a stored ensemble `(size, n_ensemble)` from an `.npz` file, requires `ensemble_file` and accepts an optional `key` (the array name; it defaults to the variable's name).
- **Expecting `run` to launch simulations.** It only reuses forecasts already held by a history-matching run.

## What to read next

- The other pages in the plugins section of the guide, for the surrogate plugins that use a design plan and for the `priors` plugins.
- The guide's configuration pages, for where plugin choices are written in your configuration.

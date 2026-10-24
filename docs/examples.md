---
audience: public
---

# Examples and toy problems

## Goal

`examples/` holds runnable examples, from analytic toy problems that run in seconds to reservoir cases. This lesson shows how examples are grouped, how to browse and copy them, and how to add your own.

The service lists the examples as a catalogue (`GET /v1/examples`), and the web UI's **Examples** page shows one tab per workflow. A web-UI user copies an example into their own folder and opens it in the Editor (`POST /v1/examples:copy`).

## Prerequisites

- A running APEX service with the examples root set (see "The service's examples root" below).
- To copy from the web UI: `[actions] examples = true` in `ui_access.toml`. Every action but `delete` is off by default. Listing stays allowed without it.
- To add an example: a checkout of APEX, so you can create a folder under `examples/` and run `tests/workflows/test_examples.py`.

## Grouping by workflow

An example is a folder with an `example.toml` manifest. The manifest groups it, so a new toy problem, or a new workflow, needs a folder and a manifest and no code:

```toml
[example]
title = "ZDT1: a convex front"
summary = "Zitzler, Deb and Thiele's first problem with 10 variables: front f2 = 1 - sqrt(f1)."
source = "pymoo 0.6.2 (github.com/anyoptimization/pymoo, Apache-2.0; Blank & Deb 2020), problems/multi/zdt.py; Zitzler, Deb & Thiele (2000)"
tags = ["toy", "multi-objective", "analytic"]
cost = "seconds"          # seconds | minutes | hours: what the card says
requires = []             # extras or programs it needs, e.g. ["reservoir", "opm_flow"]

[[example.entries]]       # one per configuration, each for one workflow
workflow = "optimization" # the Plugins page's ids: history_matching, prediction, optimization, ...
config = "zdt1_nsga2.toml"
title = "NSGA-II"         # optional: what this entry shows, when there are several
```

- **`workflow`** is the id the Plugins page uses (`history_matching`, `prediction`, `optimization`, `simulation`). An example may serve several workflows, with one entry each. The page shows a tab for every workflow that has an example, in the Plugins page's order. An id the page does not know gets a tab of its own after them, so a new workflow needs no UI change.
- **`source`** credits where the problem or case comes from, and the card shows it. The analytic multi-objective problems are taken from pymoo 0.6.2 (SCH and FON, which pymoo 0.6.2 lacks, are written from their papers).
- The folder layout is free. By convention, toy problems live in `examples/toy/<group>/<name>/`, and reservoir cases at the top level. The id is the folder's path relative to `examples/`, e.g. `toy/multi_objective/zdt1`.
- A folder with its own manifest inside another example is a separate example with its own id. It is not part of the outer example's copy. For instance, copying `optimization` does not include `3well`. To get it, copy `optimization/3well` by that id.
- Keep an example's inputs inside its own folder. A reference to a file outside it is rewritten to that file's path in the examples root, which web-UI guests cannot read (only the root's deck templates are trusted, see below).

## The catalogue

| Example | Workflow | What |
|---|---|---|
| `toy/multi_objective/{sch,fon,kur,zdt1,zdt2,zdt3,dtlz2,bnh,tnk,srn}` | optimization | NSGA-II on the analytic test problems, seconds each; `zdt1` also by MOEA/D, `dtlz2` also by NSGA-III and MOEA/D |
| `toy/multi_objective/dtlz1` | optimization | DTLZ1's linear front with many local fronts, by NSGA-III and MOEA/D |
| `toy/multi_objective/dtlz2_5` | optimization | DTLZ2 with five objectives: NSGA-III against NSGA-II |
| `toy/toy_reservoir_npv_water` | optimization | NSGA-II: NPV against the water injected, on the analytic `toy_reservoir` simulator |
| `toy/toy_reservoir_robust_npv` | optimization | NSGA-II over 5 model realizations: NPV's mean against its standard deviation |
| `toy/toy_reservoir_npv_co2` | optimization | NSGA-II: NPV against the CO2 that eCalc computes from the injection pumps' fuel (needs eCalc, in the default environment, and Java, which eCalc's thermodynamics library runs on) |
| `toy/rastrigin` | optimization | GA and PSO on Rastrigin |
| `linear_gaussian` | history matching | a linear-Gaussian case whose posterior is known exactly (Kalman) |
| `optimization` | optimization | Rosenbrock with EnOpt and with trust region (PET's popt) |
| `optimization/3well` | optimization | PET's popt tutorial: NPV on OPM Flow |
| `3dbox_production` | history matching | The 3dBox case on OPM Flow: ES-MDA and GIES, with and without adaptive localization, on the wells' rates, the log-permeability and porosity uncertain (hours) |
| `3dbox_production_avo` | history matching | The 3dBox AVO case on OPM Flow: ES-MDA and GIES, with and without adaptive localization, on the wells' rates and wavelet-compressed RMS maps of two angle stacks at two vintages, the log-permeability and porosity uncertain (hours) |

## Copying an example

`copy_example` copies an example into `<destination>/<id>`. A second copy gets `-2`, `-3`, ... and never overwrites. From the web UI, the destination is the user's UI folder (`UI/owner/examples/` or `UI/guests/<name>/examples/`).

### What is copied and what is not

- Programs and notebooks (`.py`, `.ipynb`) are not copied.
- Deck templates (`.mako`) are code and are not copied, so a user cannot edit one. The copy keeps pointing at the original template in the examples root.
- Everything else, including the data files (grid, schedule, PVT), is copied and is the user's to edit.

### How the copy finds the template

Every copied `.toml` is rewritten so that its reference to a template, and any reference to a file outside the example's folder, becomes the absolute path of that file in the examples root. Comments and the rest of the text are kept. Schematically (the paths here are placeholders):

```text
in the original example:  the configuration names the template  deck.mako
in the copy:              the same reference reads              /path/to/examples/<id>/deck.mako
```

### Which includes read which files

`case_dir` is a simulator option. It names the folder from which the deck's `${case_dir}` includes are read. When a copied configuration's template stays in the examples root, the copy gets `case_dir` set to the copy's own folder, for example `"."`. A configuration that already sets `case_dir` keeps its value.

So two kinds of include behave differently:

- An include written with `${case_dir}` resolves to the **copy**. If a deck includes `${case_dir}/<file>`, the simulation reads `<file>` in your copy, and editing it changes the simulation.
- Any other include in the template is read beside the template, in the **examples root**. Your copy of such a file is not read by the deck.

If you want a data file to be editable, the deck must reach it through `${case_dir}`.

### Trust and access

- The server treats the deck templates in the examples root as its own files, which is why they may be used even though guests cannot edit them. Other files in the root get no such treatment. They are checked by the same access rules as any user file.
- If the examples root is inside the web UI's folders, or contains them (for example, `root` set to a folder that holds `UI/`), nothing in it is trusted. The templates are then checked like any other file, and copies that depend on them may be refused. Keep the root outside the UI folders.
- The id must be the example's own, a plain relative path as listed. An id with a detour such as `../` is refused, and a copy never lands outside the user's `examples/` folder.
- A UI user's copies count against `[guests] max_storage_mb` with everything else in their folder. Guest retention prunes submissions, not copies, so users delete their copies themselves in the Editor.

## The service's examples root

In `service.toml`:

```toml
[examples]
root = "/path/to/examples"   # default: the checkout's examples/ when APEX runs from a checkout
enabled = true               # false: no catalogue
```

An installed APEX (not a checkout) has no catalogue until `root` is set.

## Worked example: add a new toy problem

1. Create the folder `examples/toy/<group>/<name>/`. Its id will be `toy/<group>/<name>`.
2. Put the configuration in it, for example `zdt1_nsga2.toml`, and keep every input it needs inside this folder.
3. Add `example.toml` beside it, like the manifest above. Set `workflow` to a Plugins page id, `config` to the file name, and `cost` to `"seconds"` if it runs quickly.
4. Run `tests/workflows/test_examples.py`. It checks every manifest, validates every entry's configuration, and runs the cheap toy optimizations (`cost = "seconds"`, nothing required beyond installed extras such as `ecalc`) for a few generations. Your example is tested as soon as its manifest exists.
5. Check the catalogue (`GET /v1/examples`) or the **Examples** page. Your example should appear under its workflow's tab, with the `source` credit on its card.
6. Copy it from the web UI and open the copy in the Editor to confirm that it runs.

## Pitfalls

- Copying fails or does nothing for a web-UI user: `[actions] examples = true` is probably missing in `ui_access.toml`.
- No catalogue on an installed APEX: set `root` in `[examples]`.
- A copied data file seems to be ignored: the deck probably reads it beside the template, not through `${case_dir}`.
- Copying `optimization` does not give you `3well`. Copy `optimization/3well` separately.
- An input kept outside the example's folder is rewritten to a root path that guests cannot read. Keep inputs inside the folder.
- `.py` and `.ipynb` files are not copied, so do not make an example depend on them in a copy.

## What to read next

- The Plugins page, for the workflow ids used in `workflow`.
- The Editor, to open, edit and run your copy.
- The access settings in `ui_access.toml`, for what web-UI users and guests may do.

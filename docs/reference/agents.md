---
audience: public
generated: reference
---

# APEX's agents: reference

Every one of these 5 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `diagnosis`

Explains what went wrong in a run (QA/QC of the posterior, failed members) and what to change, with validated configuration patches.

Run it with `apex agent diagnose RUN`.

## `monitor`

Checks a run's health: collapse, failures, damping, stagnation, overfitting, cross validation; recommends continue, stop or investigate.

Run it with `apex agent monitor RUN`.

## `planner`

Plans a study (variants, replicates, metrics) for a case and a goal within a budget, as a draft you review, revise and launch.

Run it with `apex agent plan CONFIG --goal TEXT`.

## `plotter`

Draws a run's result figures: metric box and violin plots, forecast bands per well, layer maps and histograms, or a whole preset such as brugge_classic.

Run it with `apex agent plot RUN [--request TEXT] [--preset NAME]`.

## `report`

Writes a narrative report of a run or a study, every number checked against the facts.

Run it with `apex agent report RUN | --study NAME`.

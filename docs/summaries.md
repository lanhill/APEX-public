---
audience: public
---

# Run summaries

## Goal

`apex summary` reports what every run under a folder did, step by step. For each performance metric at every step, it gives statistics over the ensemble. After this lesson you can summarise one run or many, read the Markdown and CSV output, and interpret the history-matching, prediction and optimization summaries.

## Prerequisites

- One or more finished runs. A *run directory* (`RUN_DIR`) is the folder of one run.
- A *case folder* (`CASE_DIR`) is any folder that holds several run directories.
- A shell where you can call `uv run apex`.

## Terms used on this page

- **Ensemble / member:** a run keeps a set of model realisations; each one is a member. Statistics are taken over the members.
- **Step:** one row position in the run's history. Step 0 is the prior, the starting ensemble.
- **Iterate:** the ensemble after an update that the method accepted.
- **Trial:** an attempted update. A rejected trial is not kept as an iterate.
- **Label:** a name that a metric's value is reported under, for example one value per data type or per well. The label `all` is the combined value.

## Steps

1. Summarise one run. The Markdown goes to stdout:

   ```bash
   uv run apex summary RUN_DIR
   ```

2. Summarise several places at once and write files. `--output` writes the Markdown to the path you give, and a CSV next to it (`results.csv`). Create the output folder first if it does not exist:

   ```bash
   uv run apex summary CASE_DIR other/runs --output summary/results.md
   ```

3. The paths may be run directories or folders that hold runs. Folders are searched, but not the inside of a run. The one exception is predictions, described below.

4. The case-study runner writes one summary per case when you pass `--report`: `<case>/summary.md` and `summary.csv`.

## Reading a history-matching summary

- **Steps.** Step 0 is the prior. Each later step is an accepted iterate. Rejected trials are not listed as steps. They are counted in the `rejected` column of the step that follows them.
- **`control`.** The method's control parameter for that step, for example the damping, or the step length for GN-EnRML.
- **Metrics.** Every metric the run recorded is included. If the run did not record `model_change`, it is computed from the saved iterates, and a note says so. It is computed only between consecutive accepted iterates. If the run has an `actnum_file` (the file that marks which cells are active) on its `rmse` metric, only those cells are used.
- **Result.** The *result* is the posterior: the last step, or, if the run used cross validation, the step that cross validation selected.
- **Markdown.** An overview has one row per run, showing the first step and the result. Each run then gets a table of the member mean ± standard deviation of the *headline* metrics. A metric is a headline metric if it has an `all` label, or if it has at most three labels. Either condition is enough.
- **CSV.** One row per run, metric, label and step, with mean, standard deviation, median, P10 and P90. It includes the per-data-type and per-well values.

## Reading a prediction summary

A prediction directory (`RUN_DIR/predictions/<name>`) belongs to a run. When you summarise the folder containing that run, the search finds it on its own and no extra flag is needed.

- **Steps.** Each step is one ensemble forecast, labelled by name.
- **Result.** The result is the posterior forecast.
- **Final values.** `final_value` gives each vector's forecast at the last time, over the members: mean ± std in the tables, and median, P10 and P90 in the CSV. A *vector* here is one forecast quantity, such as one well's rate.
- **Scores.** When a truth forecast (the reference forecast the ensemble is compared with) is available, these scores over the prediction period are listed per vector and as their mean over vectors (`all`): `coverage_p10_p90`, `crps`, `mean_error` and `spread`. A score is a single value, not a statistic over members. It has no ±, and its std, median and percentiles are empty in the CSV.

## Reading an optimization summary

An optimization directory (from `apex optimize`, or a job's folder under `optimizations/<name>/` in the services root) is summarised one step per row of its history: `start`, each accepted step (`step 1`, `step 2`, ...) and, with an external penalty, each `epf_update`.

- **members** is the number of objective evaluations the step took, and **control** the optimizer's step size, where it has one.
- **objective** (label `all`) is the optimizer's objective. Every optimizer minimises, so a maximised quantity such as NPV appears negated, and a note says so.
- With several objectives, `objective` also has one label per objective, and `hypervolume` and `front_size` describe the front.
- A finished robust optimization adds the objective's statistics over the realizations at its last step (label `realizations`).
- The result is the last step of a finished run. A run that has not finished has no result yet.

## Worked example

Suppose `cases/a` and `cases/b` each hold history-matching runs.

```bash
mkdir -p summary
uv run apex summary cases/a cases/b --output summary/results.md
```

This writes `summary/results.md` and `summary/results.csv`. Open the Markdown overview to compare the first step and the result of each run. Then read each run's table for the mean ± standard deviation of its headline metrics. Use the CSV when you need the median, P10 or P90, or the per-well values.

## Other workflows

Summaries are produced per workflow, and every workflow's summaries are written by the same renderer. Each summary has steps and statistics per metric, label and step, so the Markdown and CSV layout stays the same.

## Pitfalls

- Pointing the command at the inside of a run does nothing useful, because folders are searched but not the inside of a run. Pass the run directory itself or a folder that holds runs.
- A `rejected` count belongs to the step after the rejected trials. Do not look for rejected trials as their own rows.
- Forecast scores have no ± and empty std, median and percentile columns in the CSV. This is expected.
- If `model_change` was not recorded, the summary computes it and adds a note. Check that note before comparing runs.

## What to read next

Read the pages on running a history-matching case, on predictions and on optimization to produce the runs and forecasts that this command summarises.

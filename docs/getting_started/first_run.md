---
audience: public
---

# Getting started: your first history-matching run

## Goal

By the end of this page you have run a complete history match in a terminal, from a copy of an example to its summary, report and figures. You will read the run's configuration, check it with `apex validate`, run part of it, stop and resume it, and compare two methods on the same case.

The case is APEX's linear-Gaussian example: a linear model `y = G x` with 3 unknowns and 8 data, a Gaussian prior and Gaussian data errors. It needs no reservoir simulator and runs in seconds. Because the model is linear and Gaussian, its exact answer (the Kalman posterior) is known, so you can see how close the methods come.

## Before you start

- APEX installed as the README describes. Run every command below from the APEX folder.

## Step 1: copy the example

Work on a copy, so that the example stays as it is:

```bash
cp -r examples/linear_gaussian my_first_case
rm -rf my_first_case/runs    # runs you made in the example, e.g. by the README's step 6
uv run python my_first_case/make_case.py
```

`my_first_case` is a folder of your own: `git status` lists it as untracked, and an APEX update leaves it alone.

`make_case.py` writes the case's files next to itself: the forward model `G` (`forward.npz`), the observed data with their error variances (`observations.csv`) and the true model that generated them (`truth.npz`). It also prints the exact answer to compare with later:

```text
truth:               [ 0.5 -1.   0.8]
exact posterior mean [ 0.4205 -1.0471  0.7041]
exact posterior std  [0.2602 0.1569 0.1932]
```

## Step 2: read the configuration

A run is described by one TOML file. Open `my_first_case/config_esmda.toml`; its sections are:

- `[run]`: the run's name (`linear_esmda`), which names its folder, and the random seed. The same seed gives the same results.
- `[[state.variables]]`: the unknowns, here one variable `x` of size 3.
- `[prior]`: the ensemble, 500 members drawn from a standard normal distribution.
- `[observations]`: the data, from `observations.csv`.
- `[case]`: a synthetic case, with the true model in `truth.npz`, so that runs also report how far the members are from the truth (RMSE).
- `[simulator]`: the model that turns each member into predicted data, here the linear model in `forward.npz`.
- `[execution]`: where the members run, here 4 processes on this machine.
- `[method]`: the history-matching method, ES-MDA in 4 steps (`inflation = [4, 4, 4, 4]`), with its update strategy.

File names in a configuration are relative to the configuration's own folder, so the copy reads its own files. `uv run apex schema history_match` prints everything such a file may hold, as a long JSON Schema (`> schema.json` keeps it in a file).

## Step 3: validate it

```bash
uv run apex validate my_first_case/config_esmda.toml
```

`apex validate` checks the configuration without running anything, and prints the run's plan (your run's id, a hash of the configuration and its folder, will differ):

```text
run 70496daf5296: esmda, 500 members, 3 unknowns
  data: history_matching 8
  at most 5 forecasts, 2,500 simulator runs (linear, 4 at once)
  about 0.1 MB per checkpointed iterate, 0.1 MB of memory per forecast
```

A forecast runs every member once: the prior's and one after each of the 4 steps. On a real case, the plan tells you the cost before you start.

## Step 4: run part of it

```bash
uv run apex run my_first_case/config_esmda.toml --max-steps 2
```

`--max-steps 2` stops the run after two steps. The run's folder is `my_first_case/runs/linear_esmda`, next to its configuration. The table shows the data mismatch falling from the prior (step 0):

```text
iter  data mismatch mean  data mismatch std*  members  accepted      damping
----------------------------------------------------------------------------
   0         253.594         225.7      500       yes
   1         27.4167         13.12      500       yes
   2         22.0431         9.328      500       yes
* over the members: the population std (ddof = 0), as the methods use it; `apex summary` gives the sample std (ddof = 1)

stopped (resume to continue); 500 members in the last saved iterate (not yet the posterior)
```

Every step is saved as a checkpoint in the run's folder. A long run can also be stopped from another terminal with `uv run apex stop my_first_case/runs/linear_esmda`: it stops at its next checkpoint.

## Step 5: resume it

```bash
uv run apex resume my_first_case/runs/linear_esmda
```

The run continues from its last checkpoint to the end; a resumed run gives exactly the same results as one that never stopped. The table ends with:

```text
   3         19.8628         8.393      500       yes
   4         18.9286         7.748      500       yes

complete; 500 members in the posterior
```

Running the same configuration again with `apex run` is refused while its folder holds a run: resume it, or give the new run another `[run] name`.

## Step 6: read the results

```bash
uv run apex summary my_first_case/runs/linear_esmda
```

The summary gives every metric at every step, as member means ± standard deviations (the sample standard deviation, so slightly larger than the run's table), from the prior (step 0) to the result:

```text
| step | members | rejected | control | data_mismatch | normalized_data_mismatch | rmse | model_change |
| 0 | 500 | 0 | - | 253.6 ± 226 | 31.01 ± 28.38 | 1.177 ± 0.4687 | - |
| 4 (result) | 500 | 0 | - | 18.93 ± 7.755 | 1.387 ± 0.2701 | 0.201 ± 0.09464 | 0.104 ± 0.0485 |
```

- `data_mismatch` is what the method minimises: each member's misfit to the data, weighted by the data errors.
- `normalized_data_mismatch` is the mean square of the residuals to the observations, each scaled by its data error. Near 1, the members fit the data about as well as the data errors allow.
- `rmse` is the members' distance from the true model, which only a synthetic case has.
- `model_change` is how much the members changed in the last step.

For the figures:

```bash
uv run apex report my_first_case/runs/linear_esmda --plots
```

They are written as PNG and PDF files in the run's `report/` folder: `metrics/` (the metrics at each step), `health/` (the ensemble's spread and the members' data mismatch) and `data/` (the predicted data against the observations, before and after).

## Step 7: compare another method

The example also has GIES, an iterative method, in `config_gies.toml`, with its own run name (`linear_gies`):

```bash
uv run apex run my_first_case/config_gies.toml
uv run apex summary my_first_case/runs
```

Given a folder, `apex summary` tabulates every run it finds, so the two methods are side by side:

```text
| run | method | status | steps | forecasts | result | data_mismatch | normalized_data_mismatch | rmse | model_change |
| linear_esmda | esmda (approx) | complete | 4 | 5 | 4 | 253.6 → 18.93 | 31.01 → 1.387 | 1.177 → 0.201 | - → 0.104 |
| linear_gies | gies (rlmmac) | converged | 8 | 9 | 8 | 253.6 → 13.13 | 31.01 → 1.429 | 1.177 → 0.207 | - → 0.002344 |
```

GIES chooses its number of steps and stops when it has converged (`converged`), while ES-MDA always takes the number of steps its configuration gives.

## Next steps

- [The web UI and its AI assistant](web_ui.md): run the same example from the browser.
- [The agents from the command line](agents_cli.md): check, diagnose and report this run with a language model.
- [The history-matching notebook](../guide/notebooks/history_matching.ipynb): the same example in Python, with the exact posterior compared.
- [Writing a plugin](writing_a_plugin.md): your own metric or method, run on this example.

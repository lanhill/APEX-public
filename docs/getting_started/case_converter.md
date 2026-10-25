---
audience: public
---

# Getting started: bring your own case with the case-converter

## Goal

By the end of this page you have turned a reservoir simulator's deck and its observed history into an APEX history-matching case, read what was assumed, given the case a better prior, and run it. You will also create a synthetic case from a description in words, and do the same from the web UI.

The case-converter reads an OPM Flow or Eclipse deck (`.DATA`) and writes an APEX case that validates: the configuration, a deck template, the observations and the deck's included files. It never runs a simulation itself. The page uses the deck that ships with APEX, `examples/3dbox_production/BOX.DATA`: a 60 × 60 × 5 three-phase box with three producers and three water injectors, as a user would bring it.

## Prerequisites

- APEX installed as the README describes. Run every command below from its folder.
- [OPM Flow](https://opm-project.org) on `PATH`, for Step 4 and Step 5 only: converting needs no simulator, running the case does.
- A language model is optional. Without one, add `--no-llm` and the converter decides by its rules. To set a model up, see Step 1 of [the agents from the command line](agents_cli.md).

## Step 1: convert the example deck

```bash
uv run apex agent convert import examples/3dbox_production/BOX.DATA \
    --history examples/3dbox_production/observations.csv --out my_box --no-llm
```

It takes under a minute and prints what it chose:

```text
written: .../my_box/config.toml (skill opm_flow)
state: PERMX (log), PORO (none)
observations: 9 vectors; method gies, 100 members
notes and assumed values: .../my_box/CONVERSION.md
valid
plan: {...}
```

- **The deck.** The `opm_flow` skill reads it: the grid, the per-cell properties, the wells and the report dates. Its included files (`pvt.txt`, `Schdl.sch`) are copied into `my_box/`.
- **The history.** `--history` takes a CSV with the columns `time,datatype,value` (here the example's observed rates, with noise; its `variance` column is not used, the converter sets the errors), or a summary file (`.SMSPEC`). Without it, the converter reads the deck's own `WCONHIST`/`WCONINJH`.
- **The rules.** ln PERMX and PORO become the state. Every rate and pressure with data becomes an observation, with an error of 10 % of each value (at least 1 % of the vector's largest value). The method is GIES with adaptive localization, 100 members and 10 iterations.
- **The output folder** must be new or empty. A relative path such as `my_box` is taken from where you run the command.

## Step 2: read what was assumed

Open `my_box/CONVERSION.md`. Every value the converter had to assume is listed there, for example:

```text
- permx: bounds [-4.21, 8.871] from the deck's values (assumed)
- permx: a stationary Gaussian field with the deck's mean 5.309 and variance 0.7229 of its logarithm, range 15 cells (assumed: ask the geomodeler to estimate it)
```

The prior is one stationary field per property, with the deck's mean and variance, not the deck's values cell by cell. A converted case is `real`: it has no true model, so its metrics are the data mismatches only.

With a model, say what you want in words, and the converter follows it where the deck allows:

```bash
uv run apex agent convert import examples/3dbox_production/BOX.DATA \
    --history examples/3dbox_production/observations.csv --out my_box_esmda \
    --request "match porosity only, with ES-MDA in 4 steps and 50 members"
```

The model's choices are checked against the deck: it can name only properties set as arrays and vectors that have data. A draft that does not validate goes back to the model up to 3 times. If the model fails, the rules convert, and the first note says that your request was not applied and why.

## Step 3: give it a better prior (optional)

The geomodeler writes a new configuration beside the case, with a prior for its fields, and checks it on a drawn sample:

```bash
uv run apex agent geomodel my_box/config.toml --no-llm
uv run apex agent geomodel my_box/config.toml --request "ln PERMX around 5 with a 20-cell range, porosity 0.2 ± 0.03"
```

The result is `my_box/config_prior.toml` (or `--out`). With `--data`, a CSV of hard data at cells (`i,j,k,property,value`), the geomodeler estimates the variogram from them. See [the geomodeler](../geomodeler.md).

## Step 4: validate and run

```bash
uv run apex validate my_box/config.toml            # the plan: members, forecasts, simulator runs
uv run apex run my_box/config.toml --max-steps 1   # the prior and one iteration first: about 200 simulations
uv run apex report my_box/runs/box --plots        # the data mismatch and figures
```

The full run is up to 1,121 simulations of this 18,000-cell model, on 10 workers by default, so try one step first. If you did Step 3, run `my_box/config_prior.toml` instead: it has the same run name, so run one of the two, or change `[run] name` in one. The run is named after the deck (`box`; `--name` sets another). `apex resume my_box/runs/box` continues it from its last checkpoint.

## Step 5: create a synthetic case from words

Without a deck, describe the case:

```bash
uv run apex agent convert create --out box --request "a 30 x 30 x 4 five-spot, 200 mD, monthly for three years"
```

The model turns the request into a description: a box grid, the wells and their controls, the report steps, the fluid (`oil_water`, or `co2_brine` for CO2 storage on OPM Flow), and the statistics of ln PERMX and porosity. Without a model (`--no-llm`), you get the rules' five-spot: 21 × 21 × 3 cells, four producers around an injector, monthly reports for two years. The converter then draws a true model from the prior, asks before running the truth once on OPM Flow, and writes its observations with noise. `box/CONVERSION.md` lists every value of the case, assumed or asked for.

If you answered no to the truth run (or gave `--no-observe`), run it later:

```bash
uv run apex agent convert observe box
```

A created case is `synthetic`: it has a true model, so its runs also report how close the members come to it (RMSE).

## Step 6: the same from the web UI

1. Start the web UI as [its getting-started page](web_ui.md) describes.
2. Open **Examples**, choose the **History matching** tab, and on **3dBox: production data on OPM Flow** click one of its **Copy (…) to my folder** buttons. Each copies the whole example, `BOX.DATA` and `observations.csv` included, into `UI/<you>/examples/3dbox_production` (a second copy gets `-2`).
3. Open **Agents**, and under **Bring a case** give the deck and its history by their workspace paths, as the Editor lists them, e.g. `UI/<you>/examples/3dbox_production/BOX.DATA` and `UI/<you>/examples/3dbox_production/observations.csv`, where `UI/<you>` is `UI/owner` for the owner and `UI/guests/<name>` for a guest (the deck field's placeholder shows yours). Click **Import the deck**.
4. The case is a draft in `UI/<you>/drafts/case-converter/`. Open its `config.toml` in the **Editor** to review, validate and submit it.

In the web UI, a deck may include only files its user can see there (for a guest, their own folder and the folders the owner shares), and cases are limited to 200,000 cells by default; larger ones are converted on the command line. **Bring a case** also creates synthetic cases from a description; only the owner can run a created case's truth there. The AI assistant can do the same in a chat (its tools `convert_case` and `create_case`), as drafts you launch.

## Pitfalls

- **An output folder that is not empty is refused.** Give a new one each time.
- **A path the web UI cannot find.** It takes workspace paths (`UI/<you>/...`), not paths relative to your folder.
- **A request that seems ignored.** Read the first note in `CONVERSION.md`: without a model, or when the model failed, the rules decide and the note says so.
- **Assumed values left as they are.** The bounds, the prior's spread and its correlation range come from the deck's values or defaults. Replace them with what you know, by hand or with the geomodeler, before a long run.

## What to read next

- [Runtime agents and LLM providers](../agents.md): the case-converter's rules and settings (`[agents.case_converter]`), and the other agents.
- [The geomodeler](../geomodeler.md): priors from statistics, hard data or a reference field.
- [The web UI](../ui.md): every page, guests and their folders.

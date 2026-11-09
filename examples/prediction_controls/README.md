# Predictions under a control scenario (OPM Flow) example

After a history match, the wells are often operated differently than in their history. This example forecasts a history-matched ensemble under two such **control scenarios**, which set the producers' bottom-hole pressures after the history ([Predictions](../../docs/predictions.md#a-control-scenario)).

- **The case:** a 20×20×2 box of 30 m cells from 2355 m, one water injector (INJ1, 300 bar) and two producers (PRO1, PRO2, 90 bar), every well on bottom-hole pressure, with three years of history reported quarterly (`deck.mako`). The fluids are the 3dBox case's (`pvt.txt`).
- **The history match** (`config_esmda.toml`): ES-MDA with 4 steps of α = 4 and 30 members on the 60 rate observations (`observations.csv`), with the log-permeability of the 800 cells uncertain and the true model in `truth.npz`.
- **The predictions:** two years after the history, from the posterior, the prior and the truth.
  - `predict_hold.toml`: the producers stay at 90 bar.
  - `predict_bhp.toml`: PRO1 is drawn down to 60 bar and PRO2 held back at 120 bar.

  Each runs `deck_forecast.mako`, which keeps `deck.mako`'s history unchanged (so the prediction's history check passes) and then reads the scenario `bhp = [PRO1, PRO2]` as `${bhp[0]}` and `${bhp[1]}`.

```bash
uv run apex run examples/prediction_controls/config_esmda.toml        # the history match first
uv run apex predict examples/prediction_controls/predict_hold.toml    # then each scenario
uv run apex predict examples/prediction_controls/predict_bhp.toml
uv run apex summary examples/prediction_controls/runs/control_scenario_esmda/predictions/bhp
uv run apex report examples/prediction_controls/runs/control_scenario_esmda/predictions/bhp --plots
```

It needs OPM Flow on `PATH` and the `reservoir` extra (installed by `uv sync`); the figures also need the `plots` extra. On 4 cores the history match and the two predictions take about two minutes together.

## What to expect

Measured on 2026-11-09 (30 members; member means; RMSE of ln PERMX against the truth):

| | prior | after ES-MDA |
|---|---|---|
| normalized data mismatch | 1,051 | 2.6 |
| RMSE | 1.375 | 1.207 (the best of the 4 steps) |

The posterior's mean rates over the two forecast years (sm³/day):

| | `predict_hold.toml` (90, 90 bar) | `predict_bhp.toml` (60, 120 bar) |
|---|---|---|
| PRO1 oil | 98 | 132 |
| PRO2 oil | 79 | 51 |
| PRO1 water | 1,261 | 2,173 |
| PRO2 water | 2,163 | 1,120 |

## What to look at

- **The scenario is recorded:** `apex summary` names it in the prediction's notes, and the prediction's `provenance.json` holds it (`controls`).
- **The scenarios differ only after the history:** both predictions reproduce the run's own forecasts up to 2025-01-01; after it, PRO1 produces more and PRO2 less under `predict_bhp.toml` than under `predict_hold.toml`.
- **A scenario must reach the deck:** a control the forecast's template does not name is refused before anything is simulated, and so is a value outside its `lower` and `upper` bounds.

`prepare.py` wrote the true model, the observations and `deck_forecast.mako`; the files are committed, so it only needs rerunning to refresh them.

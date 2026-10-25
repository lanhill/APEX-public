# 3dBox with production data (OPM Flow) example

The 3dBox case history-matched to the wells' production data. The case is the APEX authors' own, set up for [PET](https://github.com/Python-Ensemble-Toolbox/PET), and is shipped with APEX under GPL-3.0-or-later:
- a 60×60×5 three-phase model of 30 m cells from 2355 m, with 3 water injectors and 3 producers;
- 36,000 unknowns, the log-permeability and the porosity of every cell, and 100 members;
- 1,197 rate observations: the oil and water rates of the 3 producers and the water injection rates of the 3 injectors, on the first of every month from February 2022 to December 2032 and at 2027-06-24 and 2032-12-14 (133 report times);
- every well on bottom-hole pressure control (producers at 90 bar, injectors at 300 bar), so every rate is a response of the model;
- two methods, each with adaptive localization (`adaptive_prior` with the fixed threshold 3/√N = 0.3): ES-MDA with 4 steps of α = 4 (`config_esmda.toml`), and GIES with RLM-MAC for 10 iterations (`config_gies.toml`); and the same two without localization, for comparison (`config_esmda_noloc.toml`, `config_gies_noloc.toml`);
- the metrics: normalized data mismatch, and RMSE and SSIM (per layer) of ln PERMX and porosity against the true model.

The same case with AVO maps as well is [`3dbox_production_avo`](../3dbox_production_avo/README.md): the same deck and the same rates, drawn with the same noise, plus RMS maps at the two vintages.

```bash
uv run apex validate examples/3dbox_production/config_esmda.toml   # the plan and sizes
uv run apex run examples/3dbox_production/config_esmda.toml        # ES-MDA, on 10 cores
uv run apex run examples/3dbox_production/config_gies.toml         # GIES, on 10 cores
uv run python examples/3dbox_production/synthetic_demo.py          # all four, with tables and figures
```

It needs OPM Flow on `PATH` and the `reservoir` extra (installed by `uv sync`); the demonstration also needs the `plots` extra. A run takes a while: 500 simulations for ES-MDA and up to 1,121 for GIES.

`synthetic_demo.py` (and the notebook `synthetic_demo.ipynb`, its steps one by one) runs the four configurations into `results/`, tabulates the prior and final metrics (`results/comparison/final_metrics.csv`) and the metrics at every kept iteration (`results/comparison/metrics_by_iteration_<run>.csv`) and draws the figures: the metrics over the iterations for the four runs, and per run the observed and simulated rates with the true model's, and per layer the true model and the ensembles' means and spreads (`results/<run>/report`). `--plots-only` redraws them from finished runs.

## What to expect

Measured on 2026-10-21 (100 members; member means of the normalized data mismatch, 1 being the noise level; RMSE and SSIM against the truth; spread relative to the prior's):

| | prior | ES-MDA | GIES | ES-MDA, no localization | GIES, no localization |
|---|---|---|---|---|---|
| normalized data mismatch | 491,000 | 73,100 | 162,000 | 711,000 | 94,200 |
| RMSE of ln PERMX / porosity | 1.345 / 0.082 | 1.563 / 0.084 | 1.335 / 0.082 | 4.286 / 0.177 | 1.560 / 0.090 |
| SSIM of ln PERMX / porosity | 0.322 / 0.153 | 0.286 / 0.155 | 0.327 / 0.161 | 0.018 / 0.012 | 0.293 / 0.143 |
| spread kept, ln PERMX / porosity | 100 % / 100 % | 103 % / 99 % | 99 % / 100 % | 33 % / 17 % | 92 % / 93 % |

The notebook also tabulates these metrics at every kept iteration of each run. No member failed in these runs.

| File | Origin |
|---|---|
| `pvt.txt`, `Schdl.sch` | the case's PVT tables and well schedule |
| `deck.mako` | the case's PET deck, reporting monthly (`DATES`) with restarts only at the AVO example's vintages, with the grid as `DX`/`DY`/`DZ`/`TOPS`, the porosity from the state (or `poro.inc` without one), ln PERMX capped at `permx_cap` (`template_values`; 8 here, the PET deck's 6 by default), and `${case_dir}` includes |
| `truth.npz`, `poro.inc` | the case's true model (log-permeability and porosity; `poro.inc` holds the porosity too) |
| `observations.csv` | the truth's monthly rates with noise: a standard deviation of 10 % of the rate, at least 8 (a rate of 0 observed as 0, variance 10²; a small rate may come out negative) |
| `true_rates.csv` | the truth's noise-free rates (for the demonstration's figures) |
| `BOX.DATA` | `deck.mako` rendered with the true model and ln PERMX capped at 8: a plain OPM Flow deck to try the case-converter on, with `observations.csv` as its history (docs/getting_started/case_converter.md); re-render it after changing the template or the truth (a test checks it) |
| `config_esmda.toml`, `config_gies.toml`, `config_esmda_noloc.toml`, `config_gies_noloc.toml` | written by `prepare.py` |
| `prior_ensemble.npz` | the 100 prior members (float32), drawn by `prepare.py`; the same in both 3dBox examples, each folder standing alone |
| `prepare.py` | regenerates the prior, observations and configurations, running the truth once |
| `synthetic_demo.py`, `synthetic_demo.ipynb` | the four runs, their table and figures |

The prior, `prior_ensemble.npz`, is drawn once by `prepare.py` with APEX's `gaussian_field` sampler and shared by both 3dBox examples: ln PERMX with mean 5.0 (the truth's is 5.3) and standard deviation 1, and the porosity with mean 0.20 (the truth's is 0.21) and standard deviation 0.05, correlated 0.7 with ln PERMX (the truth's correlation is 0.95), both with a Gaussian variogram of practical range 17 cells, about as smooth as the truth, and clipped to the bounds (ln PERMX −3 to 10, porosity 0.01–0.40). 14 cells of the true porosity are below the prior's lower bound (the truth's smallest is 0.005).

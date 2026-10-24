# 3dBox with production and AVO data (OPM Flow) example

The 3dBox AVO case, run through APEX's geophysics. The case is the APEX authors' own, set up for the AVO simulator of [PET](https://github.com/Python-Ensemble-Toolbox/PET) (SimulatorWrap's `flow_avo`), and is shipped with APEX under GPL-3.0-or-later:
- a 60×60×5 three-phase model of 30 m cells from 2355 m, with 3 water injectors and 3 producers;
- 36,000 unknowns, the log-permeability and the porosity of every cell, and 100 members;
- 1,197 rate observations: the 9 rates (oil and water of the producers, water injection) on the first of every month from February 2022 to December 2032 and at the two vintages (133 report times), the same as `3dbox_production`'s;
- RMS maps of the 10° and 20° angle stacks over the reservoir's time window, at PET's days 2000 and 4000 (4 × 3,600 values), wavelet-compressed (`[[data_transforms]]` `wavelet`, Daubechies-2 at 3 levels, one per angle stack): the methods see 1,394 coefficients;
- two methods, each with adaptive localization (`adaptive_prior` with the fixed threshold 3/√N = 0.3): ES-MDA with 4 steps of α = 4 (`config_esmda.toml`), and GIES with RLM-MAC for 10 iterations (`config_gies.toml`); and the same two without localization, for comparison (`config_esmda_noloc.toml`, `config_gies_noloc.toml`);
- the metrics: normalized data mismatch, and RMSE and SSIM (per layer) of ln PERMX and porosity against the true model.

```bash
uv run apex validate examples/3dbox_production_avo/config_esmda.toml   # the plan, sizes and restart check
uv run apex run examples/3dbox_production_avo/config_esmda.toml        # ES-MDA, on 10 cores
uv run apex run examples/3dbox_production_avo/config_gies.toml         # GIES, on 10 cores
uv run python examples/3dbox_production_avo/synthetic_demo.py          # all four, with tables and figures
```

`synthetic_demo.py` (and the notebook `synthetic_demo.ipynb`, its steps one by one) runs the four configurations into `results/`, tabulates the prior and final metrics (`results/comparison/final_metrics.csv`) and the metrics at every kept iteration (`results/comparison/metrics_by_iteration_<run>.csv`) and draws the figures: the metrics over the iterations for the four runs, and per run the observed and simulated rates with the true model's, and per layer the true model and the ensembles' means and spreads (`results/<run>/report`). `--plots-only` redraws them from finished runs.

A run takes a while: 500 simulations for ES-MDA and up to 1,121 for GIES, each a few seconds to a minute.

## What to expect

Measured on 2026-10-21 (100 members; member means of the normalized data mismatch, 1 being the noise level, the AVO maps' over their kept wavelet coefficients; RMSE and SSIM against the truth; spread relative to the prior's):

| | prior | ES-MDA | GIES | ES-MDA, no localization | GIES, no localization |
|---|---|---|---|---|---|
| normalized data mismatch | 228,500 | 59,500 | 79,200 | 896,000 | 11,500 |
| ... AVO 10° / 20° | 2,637 / 3,610 | 1,534 / 2,566 | 2,233 / 3,117 | 23,060 / 45,830 | 1,849 / 3,034 |
| RMSE of ln PERMX / porosity | 1.345 / 0.082 | 1.450 / 0.079 | 1.323 / 0.081 | 4.771 / 0.179 | 1.756 / 0.088 |
| SSIM of ln PERMX / porosity | 0.322 / 0.153 | 0.285 / 0.197 | 0.331 / 0.172 | 0.022 / 0.030 | 0.266 / 0.145 |
| spread kept, ln PERMX / porosity | 100 % / 100 % | 97 % / 89 % | 99 % / 98 % | 0.5 % / 0.2 % | 39 % / 37 % |

The notebook also tabulates these metrics at every kept iteration of each run. The runs assimilate 2,591 data: 1,197 rates and 1,394 wavelet coefficients of the maps. The localization uses a fixed threshold because its default (universal) one is about 0.45 here (18,000 cells at 100 members).

- No member failed in these runs.
- 14 cells of the true porosity are below the prior's lower bound of 0.01 (the truth's smallest is 0.005); clipped members cannot reach them.

It needs OPM Flow on `PATH` and the `reservoir` extra (installed by `uv sync`).

| File | Origin |
|---|---|
| `pvt.txt`, `Schdl.sch` | the case's PVT tables and well schedule |
| `deck.mako`, `deck_pet.mako` | the case's PET deck (`RUNFILE.mako`): `deck.mako` reports monthly with restarts only at the vintages; `deck_pet.mako` keeps PET's schedule (10 steps of 400 days, a restart at each) for the integration test's SimulatorWrap comparisons; both with the grid as `DX`/`DY`/`DZ`/`TOPS` (the same box as its `Grid.grdecl`), the porosity from the state (or `poro.inc` without one), ln PERMX capped at `permx_cap` (`template_values`; 8 here, the PET deck's 6 by default), and `${case_dir}` includes |
| `truth.npz`, `poro.inc` | the case's true model, `true_property_values.npz` (log-permeability and porosity; `poro.inc` holds the porosity too) |
| `REF.EGRID` | the truth's grid: its active columns fix the AVO traces |
| `observations.csv`, `avo_*.npy` | the truth's rates and maps, with noise: rates with a standard deviation of 10 % of the rate, at least the case's PET `setup.py`'s 8 (a rate of 0 observed as 0, variance 10²; a small rate may come out negative), maps 5 % of their mean \|value\| |
| `true_response.npz` | the truth's maps without noise (checked by the integration test) |
| `config_esmda.toml`, `config_gies.toml`, `config_esmda_noloc.toml`, `config_gies_noloc.toml` | written by `prepare.py`, with the maps' noise variances |
| `prior_ensemble.npz` | the 100 prior members (float32), drawn by `prepare.py`; the same in both 3dBox examples, each folder standing alone |
| `true_rates.csv` | the truth's noise-free rates (for the demonstration's figures) |
| `synthetic_demo.py`, `synthetic_demo.ipynb` | the four runs, their table and figures |
| `prepare.py` | regenerates the prior, observations, maps and configurations from the files above, running the truth once |

Differences from the case's PET set-up (`3D_IES.pipt`, `setup.py`):
- PET assimilates only the AVO data; APEX adds the wells' rates, to show production and seismic data matched together.
- PET's AVO data are the whole traces with 10 % relative noise per value, compressed by wavelets (`COMPRESS`); APEX's are RMS maps of each angle stack, one noise variance per map (5 % of its mean |value|), also wavelet-compressed.
- PET runs GIES (RLM-MAC) with AutoAdaLoc localization on the log-permeability; APEX runs ES-MDA and GIES with adaptive localization at a fixed threshold, on the log-permeability and the porosity (PET fixes the porosity at the truth's).
- The PET set-up's prior mean is a file, `priormean.npz`, that its folder lacks; APEX's prior is drawn by `prepare.py` (below).
- PET's rates are every 400 days; APEX's are monthly.
- PET's rate noise is 8 sm³/day whatever the rate (0.1 % of a 7,000 sm³/day rate, far narrower than any 100-member ensemble's spread); APEX's is 10 % of the rate, with 8 as the floor.
- The deck caps ln PERMX at 8, not PET's 6 (the truth reaches 6.9).
- The overburden pressure is lithostatic, 22,600 Pa/m below 1 atm (54 MPa at 2400 m), instead of PET's constant 320 bar.
- The synthesis samples the exact interface times (`sampling = "interfaces"`, the default; PET's is `"nearest"`).
- PET's code returns the last angle's traces for every angle (`pet_parity.md`, G7); APEX computes each angle.
- The deck's restarts hold the phases' densities and pore volumes (`RPTRST BASIC=2 DEN RPORV`; PET's had `FIP RPORV`), which gravity and seafloor displacement data read (the integration test checks both against SimulatorWrap's on the truth); it leaves out PET's unused VFP, aquifer and fault dimensions (no effect on the run).

The prior, `prior_ensemble.npz`, is drawn once by `prepare.py` with APEX's `gaussian_field` sampler and shared by both 3dBox examples: ln PERMX with mean 5.0 (the truth's is 5.3) and standard deviation 1, and the porosity with mean 0.20 (the truth's is 0.21) and standard deviation 0.05, correlated 0.7 with ln PERMX (the truth's correlation is 0.95), both with a Gaussian variogram of practical range 17 cells, about as smooth as the truth, and clipped to the bounds (ln PERMX −3 to 10, porosity 0.01–0.40). 14 cells of the true porosity are below the prior's lower bound (the truth's smallest is 0.005).

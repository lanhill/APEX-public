# Reference results

Regression and parity tests compare against the files in this directory. Never edit them by hand. Regenerate them, and review the diff, only when a change in results is intended.

| Directory | Contents | How to regenerate |
|---|---|---|
| `pet/` | PET outputs for parity tests. Each `.npz` holds `in_*` inputs, `out_*` outputs and `meta_*` provenance. | `env -u PYTHONPATH .venv-pet/bin/python tools/pet_reference/generate.py` for upstream cases, `.venv-pet-fork/bin/python` for GIES cases (setup: see [Reference environments](#reference-environments)). `meta_pet_source` records which. The `wavelet_*` and `pet_import_compress` cases also need PyWavelets in `.venv-pet` (generated with 1.8.0); `pet_import_compress` also uses pandas (3.0.6) and feeds PET pickled tables, since PET cannot read `.npz` cells from CSV under pandas 3 (see `docs/migration/pet_parity.md`). `metrics_rmse_ssim` runs in `.venv-pet-fork` and needs PET-fork's `Plotting/` plus scikit-image (0.26.0) and openpyxl. The `popt_*` cases run whole PET optimizations with a recording `RandomState` (seeded from the case's seed) in place of the ensembles' `rng`, and store its draws as `in_draw_<k>_<kind>`; SciPy 1.18.1 in both environments. `popt_npv` runs PET's `npv` from `fd17979` (gone upstream), read from a PET git checkout: `PET_GIT_CHECKOUT=<checkout> env -u PYTHONPATH .venv-pet/bin/python tools/pet_reference/generate.py popt_npv`; `in_npv_source` records the file's blob hash. `popt_ren_npv`, `popt_ecalc_npv`, `popt_ecalc_pareto_npv_wemc`, `popt_ecalc_pareto_npv_w` and `popt_ren_npv_co2` run PET's functions of the same commit the same way (`in_source`). The eCalc ones have libecalc, ecalc_cli and pqdm replaced by the generator's stand-ins: `ECALC_STUB`, CO2 linear in the rates handed to eCalc (`in_stub`), and a serial map. |
| `gmac/` | GMAC: a NumPy transcription of the user's MATLAB `RLM_MAC_GIES.m` per-member block (no MATLAB licence on the host), `.npz` and `.mat` with `in_*`, `out_iterated_ensemble`, `meta_seed`, `meta_code`. | `uv run python tools/gmac_reference/generate.py` (NumPy and SciPy of the APEX environment; seeds 101-105 in its `CASES`). See `docs/migration/gmac.md`. The `cgies_*` cases: `uv run python tools/gmac_reference/generate_cgies.py`, a transcription of `GIES_constraint_ensLoc.m`'s update (seeds 201-203). |
| `rlmmac/` | `rlmmac` against the author's MATLAB RLM-MAC: a NumPy transcription of the analysis of `RLM_MAC.m` (scaled anomalies) and `RLM_average_cost.m` (unscaled), no MATLAB licence on the host; `.npz` and `.mat` with `in_*` (ensemble and forecasts with the mean model's as the last column, perturbed data, `C`, λ, `tsvdData`) and `out_<version>_ensemble`, `_svd_pd`, `_alpha`. | `uv run python tools/rlmmac_reference/generate.py` (NumPy and SciPy; seed 20261013). See the GIES notes in `docs/migration/pet_parity.md`. |
| `chop/` | CHOP: one trial of the author's `RLM_MAC_CHOP.m` per-member-taper model update and its `update_hyper.m` / `update_hyper_no_loc.m` (5SpotExample/run_scripts) on a small linear case; a line-by-line NumPy transcription (`tools/chop_reference/transcription.py`), as this host's MATLAB licence was refused (error -39,147). `chop_step.npz`: `in_*` in MATLAB's layout (a mean column last), `out_*` members only, `meta_seed`, `meta_backend`. | `uv run python tools/chop_reference/generate.py` (seed 20261010); `--backend matlab --run-scripts DIR` runs the MATLAB itself (`chop_reference.m`) once licensed. |

The model error characterization cases in `pet/`, `mec_lstm_forward` and `mec_brugge_features`, come from PET-fork's `Examples/.../Brugge/ml_model_error.py`. That file is not in the exported pet-source, and the published runs used the checkout's working copy, which has uncommitted changes. So `generate.py` imports it from `$PET_FORK_CHECKOUT` (default `../PET-fork`), stubs `torchviz`, runs in `.venv-pet-fork` (torch 2.14 CPU), and records the file's SHA-256 as `in_module_sha256`. `mec_brugge_features` also reads that checkout's Eclipse run `model_error/BRUGGE_REF_FAULTS` and schedule `include/Hist`. Regenerate with `env -u PYTHONPATH -u DISPLAY .venv-pet-fork/bin/python tools/pet_reference/generate.py mec_lstm_forward mec_brugge_features`.

For any other reference file, add a row stating the tool, its version or commit, and the seed used.

## Reference environments

The generators run outside APEX's environment, which never installs PET. Clear `PYTHONPATH` when running PET's generators: a PET checkout on it would shadow the installed version, and they refuse to run while it is set. The `gmac`, `rlmmac` and `chop` references are made in APEX's own environment.

```bash
# Upstream PET at the reference commit. SimulatorWrap's references (tools/geophysics_reference)
# also need SimulatorWrap (subsurface, d29dc57), mako, resdata and resfo-utilities in it.
uv venv .venv-pet --python 3.12
uv pip install --python .venv-pet/bin/python "PET @ git+https://github.com/Python-Ensemble-Toolbox/PET@e65fb1058411c8dad7ca70a09d501e0b18075e75"

# pymoo, for the population and multi-objective optimizers
uv venv .venv-pymoo --python 3.12 && uv pip install --python .venv-pymoo/bin/python "pymoo==0.6.2"
env -u PYTHONPATH .venv-pymoo/bin/python tools/pymoo_reference/generate.py
```

GIES and its localizations are ported from PET-fork at commit `1348861`, which is not public: its references are stored here, and regenerating them needs that source in a second environment, `.venv-pet-fork` (numpy, scipy, scikit-image and openpyxl, with the source's `pipt`, `misc`, `ensemble`, `input_output`, `simulator`, `popt` and `Plotting` folders on its path). Each generator case declares which PET it needs, and a run generates only the cases of the PET it finds.


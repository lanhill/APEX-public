---
audience: public
---

# MEC: model error characterization with physics-guided LSTMs

## Goal

A reservoir simulator is never the true system: unmapped faults, wrong relative permeabilities or a coarse grid leave errors that a history match otherwise pushes into the estimated properties. Model error characterization (MEC) adds a data-driven model-error model `r(m, θ)` to the simulator, and estimates its parameters `θ` together with the reservoir model `m`. By the end of this page you will be able to:

- history-match with the effective simulator `g(m) + r(m, θ)`;
- set up the PG-LSTM model-error model for a deck's wells;
- find the residuals `r` of every saved ensemble.

The method is the manuscript by X. Luo, "Model error characterization for ensemble-based history matching through a physics guided long short-term memory network".

## How it works

- **One network per well.** Each well gets its own PG-LSTM: an LSTM over the report times, followed at each report time by a fully connected layer, a sigmoid and a scale. Its residual lies in `(−u, u)`, with `u = 0.9999 ×` the largest observed value of that well and data type.
- **Inputs from Peaceman's well model.** At every report time and every block the well perforates, the network reads:
  - DX, DY, DZ, PERMX, PERMY, PERMZ and PORO from the member's INIT file;
  - OIL_DEN, WAT_DEN, SWAT, SOIL and WBHP − PRESSURE from its restart file.
- **θ is state.** The parameters of all the networks form one state variable (`nn_params` by default). The method updates it with the reservoir properties. The deck never sees it.
- **The effective simulator.** Each member's run is the usual simulation plus the residuals read from that run's files, so the data mismatch the method minimises is that of `g + r`.

## Running it

```toml
[[state.variables]]               # the networks' parameters
name = "nn_params"
size = 352920                     # the model's count; apex validate reports it if wrong

[prior.variables.nn_params]       # N(0, 0.001^2), the manuscript's initialization
name = "gaussian"
options = { mean = 0.0, std = 0.001 }

[model_error]
name = "pg_lstm"

[model_error.options]
schedule_file = "../model/include/Hist"   # the history schedule (COMPDAT, WCONHIST/WCONINJH, TSTEP or DATES)
report_steps = [1, 2, 3, 4, 5, 6, 10, 20, 23, 24, 25, 26, 27, 35, 36, 60, 80, 100, 126, 127]
producers = ["BR-P-1", "BR-P-2"]          # ... every well with data
injectors = ["BR-I-1"]
producer_data = ["WOPR", "WBHP", "WWCT"]
injector_data = ["WBHP"]
hidden_size = 20
shut_wells = "zero"               # or "ignore" (the published runs)
```

The rest of the configuration is unchanged. MEC works with any method and localization. It needs the following:

- **A deck simulator** (`opm_flow` or `eclipse`): the residuals are read from each run's files.
- **Restarts with densities at the report steps.** Put `RPTRST BASIC=2 DEN /` in the deck's SCHEDULE section.
- **The bottom-hole pressure of every listed well in the summary.** Each network's inputs include the well's `WBHP` minus the pressure of its perforated blocks, so the deck's SUMMARY section must request `WBHP` for every well in `producers` and `injectors`, whether or not `WBHP` is observed. `WBHP` followed by `/` on its own line requests it for every well. `apex validate` reads the SUMMARY section and refuses a deck that lacks it, before anything is simulated.
- **One report time per observation time.** `report_steps` are the observation times' report steps, 1-based. The network has one output layer per report time.

Leave the model-error variable out of the truth file of a synthetic case: it has no true value. RMSE, SSIM and model change score the reservoir properties only.

## Options

| Option | Default | Meaning |
|---|---|---|
| `schedule_file` | required | The schedule the wells' perforations and status are read from. Report steps are counted from TSTEP and DATES; COMPDAT adds connections, as in Eclipse (shut ones count); a well's status is its last WCONHIST, WCONPROD, WCONINJH, WCONINJE or well-level WELOPEN record. |
| `initial_schedule_file` | none | Schedule text read first: the deck's own SCHEDULE keywords before its included history (e.g. a first COMPDAT). |
| `phases` | `oil, water` | The deck's phases: with gas, GAS_DEN and SGAS are features too. |
| `report_steps` | required | 1-based report steps of the network's report times. |
| `producers`, `injectors` | none | Well names; at least one well. |
| `producer_data`, `injector_data` | `WOPR, WBHP, WWCT`; `WBHP` | Their data types; each must be observed. |
| `hidden_size` | 20 | The LSTM's hidden size. |
| `state_variable` | `nn_params` | The state variable that holds θ: producers' networks first, then injectors', each in torch's parameter order. |
| `shut_wells` | `zero` | `zero`: no residual for a shut well's rates (WBHP is kept). `ignore`: residuals at every report time, as the published runs of the reference implementation (the method author's Python code) (its code never zeroes them). |

## What a run saves

Every saved ensemble (`checkpoints/iter_<k>.npz`) also holds:

- `auxiliary_predicted`: the residuals `r`, one row per datum, with keys `MEC:<type>:<well>`;
- `auxiliary_keys`: the names of those keys.

`predicted` is the effective forecast `g + r`, so `g` itself is `predicted − auxiliary_predicted`.

## Limits

- No predictions beyond the history: the network has one output layer per history report time.
- No parameterizations, adjoint update, multilevel methods (fidelity levels) or ERT export yet. A multilevel run's coarse level would feed the networks coarse-grid inputs, so one θ would give different residuals per level.
- An output whose observations are never positive cannot be modelled: its scale u would not be positive.
- The θ ensemble can be large: 352,920 values for Brugge's 30 wells. An adaptive taper over the whole state is then about 6 GB.

## Parity with the reference implementation

**The Brugge comparison's first iterations differ.** APEX's MEC + AdaLoc run on OPM Flow reduced the data mismatch from 1.04e9 to 5.4e8 in its first iteration; the reference implementation's published run went to 9.7e8, and then decreased steadily for 20 iterations. Two known differences explain it, not the case set-up:
- At the same λ, APEX's RLM-MAC step is √(N−1) ≈ 10× the reference implementation's: APEX scales the data anomalies by 1/√(N−1) and the reference implementation's localised branch does not (see the GIES notes in the [PET migration page](migration/pet_parity.md)). Recomputed from the reference run's saved prior forecast, observations and taper, the reference implementation's formula reproduces its iteration-1 increment exactly, and APEX's gives 10.1× it. `schedule.initial ≈ 100` gives APEX the published run's first step size, but not its later ones: replaying the reference's own iterates, APEX's step at λ·102 grows from 1× the reference's step at λ (iteration 1) to 2.5× (iteration 3) and √102 ≈ 10.1× (iterations 16–20), as λ falls and fewer directions are damped. To reproduce the published run, set `[method.strategy.options] scale_data_anomalies = false` with `schedule.initial = 1`.
- The reference implementation tapered only the first 60,048 of the 352,920 `nn_params` rows (its 3-D `GRID` taper defect, also in the [migration page](migration/pet_parity.md)); APEX tapers all of them, so its `nn_params` moved less.

Ported from the method author's Python implementation of the Brugge example (the parity references record its file's SHA-256); upstream PET has no counterpart.

| Reference implementation | APEX | Status | Tolerance | Notes |
|---|---|---|---|---|
| `LSTMWellModel.forward`, `get_nn_params`/`assign_nn_params` | `algorithms/model_error/lstm.py` (`WellNetwork`, `well_residuals`) | parity | 1e-6 × u (prior scale θ ~ N(0, 0.001²); measured 1e-7), 1e-4 × u (saturated gates; measured 3.4e-5) | NumPy forward pass in float32, torch's parameter layout and gate order. Case `mec_lstm_forward` (a producer and an injector, 20 report times, hidden size 20). The tolerances are float32 rounding in different summation orders: against a float64 evaluation, torch is within 6e-6 u and APEX within 3e-5 u in the saturated regime. |
| `ModelErrorProd` (schedule reading, `get_props_at_wells`, input sequences, `set_obs_ranges`) and `add_me_residual` | `simulation/model_error.py` (`well_schedules`, `CaseFiles`, `well_features`, `ModelErrorSimulator`), `workflows/model_error.py` (`model_error_models["pg_lstm"]`) | parity (opt-in) | features 1e-6 relative (measured 5.9e-8); residuals 1e-5 × u (measured 2.4e-7) | Case `mec_brugge_features`: the reference implementation's input sequences and residuals of all 30 Brugge wells on its Eclipse reference run, against APEX reading the same schedule and run (`tests/parity/test_mec_brugge.py`, opt-in: it needs the reference run's files); the block union and order match exactly, and the 30 networks have 352,920 parameters, the manuscript's count. Differences: a general schedule walk (TSTEP/DATES; COMPDAT adds connections and WELOPEN sets a well's status, as in Eclipse; on Brugge's schedule, which re-lists every connection, this is the reference implementation's `rpt_idx`-th block, and the well status matches the reference implementation's for all 30 wells); SOIL read when the restart has it (the reference implementation then fails with an undefined name). **Defect in the reference implementation:** `forward` zeroes a shut well's outputs with `if not well_open[ts]`, but `initialize_lstm_models` stores a list of per-output flags there, and a non-empty list is always true, so residuals are never zeroed, also for rates of shut wells (data 0 with variance 1e-6). APEX's `shut_wells = "zero"` (default) zeroes them, as the manuscript says; `"ignore"` reproduces the published runs. |

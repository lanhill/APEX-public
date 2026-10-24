# Golden workflow outputs

References for `tests/workflows/test_golden_outputs.py`: the outputs of representative history-matching runs, recorded from the code *before* the generic driver. They check that the refactor reproduces today's results.

Each `workflow_<case>.npz` holds:
- the posterior and member ids;
- the misfit means, acceptance flags and per-member misfits of every forecast;
- the names of the result files the run saved;
- a SHA-256 `digest` of all of the above.

- Values are compared to `rtol = 1e-9`, because bit-exact results differ between CPU types.
- `APEX_GOLDEN_EXACT=1` also checks the digests, on the machine that recorded them.
- Recorded on 2026-09-27 from the old drivers at `290916f` (Intel i9-10900K, AVX2; numpy 2.5.3), with `APEX_RECORD_GOLDEN=1 uv run pytest tests/workflows/test_golden_outputs.py`. The two `gies_midrun_failures*` cases need `LinearModel`'s `fail_above`, which was added for them, so they were recorded from `290916f` plus the new `simulation/analytic.py` only. The numerics review re-checked all 11 cases that way, bit-exactly.
- Not covered: the history's `damping`, and the contents of the arrays saved alongside each ensemble (`reference_prediction`, per-step `perturbed_observations`); only their file names are compared. `tests/workflows/test_saved_results.py` checks those.
- Re-record only when a change of results is intended, and say so in the commit.

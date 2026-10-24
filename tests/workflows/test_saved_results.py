"""What a run saves per step: ensembles, forecasts, perturbed observations and statistics."""

import itertools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import pytest
from typer.testing import CliRunner

from apex.cli.main import app
from apex.core.errors import ConfigError
from apex.services import composition
from apex.storage.run_directory import RunDirectory


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path
    observations: np.ndarray
    variances: np.ndarray


CaseFactory = Callable[..., Case]

GIES = (
    'name = "gies"\nstrategy = { name = "rlmmac", options = { energy = 1.0 } }\n'
    "schedule = { data_mismatch_tolerance = 1e-9 }\nmax_iterations = 12"
)
#: With the tanh model, GIES's steps overshoot now and then, so some trials are rejected.
TANH = 'nonlinearity = "tanh"'
ESMDA = (
    'name = "esmda"\noptions = { inflation = [4, 4, 4, 4] }\n'
    'strategy = { name = "approx", options = { energy = 1.0 } }'
)


def _checkpoints(case: Case) -> Path:
    [run_dir] = (case.directory / "runs").iterdir()  # named after the case
    return run_dir / "checkpoints"


def _files(case: Case, prefix: str) -> set[int]:
    return {int(p.stem.split("_")[1]) for p in _checkpoints(case).glob(f"{prefix}_*.npz")}


@pytest.mark.parametrize("method", [ESMDA, GIES], ids=["esmda", "gies"])
def test_the_once_drawn_perturbations_are_saved_once(linear_case: CaseFactory, method: str) -> None:
    """GIES's perturbed observations, or ES-MDA's un-inflated scoring draw (enObs_conv)."""
    case = linear_case(ensemble_size=20, method=method, simulator_extra="fail_members = [5]")
    result = composition.start(case.config)
    run = RunDirectory(case.directory / "runs" / "case")

    saved = run.load_arrays("perturbed_observations")
    # Drawn for every member before the prior forecast, member 5 included.
    np.testing.assert_array_equal(saved["member_ids"], np.arange(20))
    assert saved["perturbed_observations"].shape == (8, 20)
    # ES-MDA: each step's own perturbations; GIES (RLM-MAC): the mean-model forecast.
    per_step = {"perturbed_observations"} if method == ESMDA else set()
    extra = set() if method == ESMDA else {"reference_prediction"}
    for k in _files(case, "iter"):
        arrays = run.load_arrays(f"iter_{k}")
        expected = {"state", "predicted", "member_ids"} | extra | (per_step if k > 0 else set())
        assert set(arrays) == expected
    assert 5 not in result.member_ids


def test_esmda_saves_the_perturbations_each_step_drew(linear_case: CaseFactory) -> None:
    """Each step's saved perturbations reproduce that step's update (PET's enObs)."""
    from apex.algorithms.data_assimilation import ESMDA, ApproxUpdate, DataErrorCovariance

    case = linear_case(ensemble_size=20, simulator_extra="fail_members = [5]")
    composition.start(case.config)
    run = RunDirectory(case.directory / "runs" / "case")
    esmda = ESMDA(inflation=[4.0] * 4, strategy=ApproxUpdate(energy=1.0))
    data_error = DataErrorCovariance.diagonal(case.variances)

    draws = []
    for k in range(1, 5):
        before, after = run.load_arrays(f"iter_{k - 1}"), run.load_arrays(f"iter_{k}")
        perturbed = after["perturbed_observations"]
        assert perturbed.shape == (8, 19)
        np.testing.assert_array_equal(after["member_ids"], before["member_ids"])
        posterior = esmda.update(k - 1, before["state"], before["predicted"], perturbed, data_error)
        np.testing.assert_allclose(posterior, after["state"], rtol=1e-12, atol=1e-12)
        draws.append(perturbed)
    # Redrawn every step, with the inflated C_d: spread about sqrt(alpha) sigma.
    assert all(not np.array_equal(a, b) for a, b in itertools.pairwise(draws))
    spread = np.std(draws[0] - case.observations[:, None], axis=1) / np.sqrt(case.variances)
    assert 1.0 < spread.mean() < 3.0  # sqrt(4) = 2


def test_gies_saves_every_accepted_iterate_and_optionally_the_rejected_trials(
    linear_case: CaseFactory,
) -> None:
    plain = linear_case(name="plain", ensemble_size=20, method=GIES, simulator_extra=TANH)
    trials = linear_case(
        name="trials",
        ensemble_size=20,
        method=GIES,
        simulator_extra=TANH,
        failures="[output]\nsave_rejected_trials = true",
    )
    result = composition.start(plain.config)
    composition.start(trials.config)

    accepted = {r.iteration for r in result.history if r.accepted}
    rejected = {r.iteration for r in result.history if not r.accepted}
    assert rejected, "the case should reject some trials"
    assert _files(plain, "iter") == accepted  # iteration 0 is the prior
    assert _files(plain, "trial") == set()
    assert _files(trials, "iter") == accepted
    assert _files(trials, "trial") == rejected

    # Rejected trials are scored on their own ensemble and forecast.
    run_trials = RunDirectory(trials.directory / "runs" / "trials")
    truth = np.load(trials.directory / "truth.npz")["x"]
    for k in rejected:
        trial = run_trials.load_arrays(f"trial_{k}")
        rmse = np.sqrt(np.mean((trial["state"] - truth[:, None]) ** 2, axis=0))
        np.testing.assert_allclose(
            run_trials.load_arrays(f"data_mismatch_{k}")["metric:rmse:x"], rmse, rtol=1e-12
        )

    # Each saved forecast is the one scored in the history.
    run = RunDirectory(plain.directory / "runs" / "plain")
    perturbed = run.load_arrays("perturbed_observations")["perturbed_observations"]
    for record in result.history:
        if record.accepted:
            predicted = run.load_arrays(f"iter_{record.iteration}")["predicted"]
            data_mismatch = np.sum((perturbed - predicted) ** 2 / plain.variances[:, None], axis=0)
            assert data_mismatch.mean() == pytest.approx(record.data_mismatch_mean, rel=1e-12)
    final = run.load_arrays(f"iter_{max(accepted)}")
    np.testing.assert_array_equal(final["state"], result.posterior)


def test_default_metrics_of_a_synthetic_case(linear_case: CaseFactory) -> None:
    """Normalized data mismatch against the observations and model RMSE against the true model."""
    case = linear_case(ensemble_size=30)
    result = composition.start(case.config)
    run = RunDirectory(case.directory / "runs" / "case")
    truth = np.load(case.directory / "truth.npz")["x"]

    for record in result.history:
        stats = run.load_arrays(f"data_mismatch_{record.iteration}")
        saved = run.load_arrays(f"iter_{record.iteration}")
        residual = (case.observations[:, None] - saved["predicted"]) / np.sqrt(case.variances)[
            :, None
        ]
        mismatch = np.mean(residual**2, axis=0)  # PET-fork's mean square
        rmse = np.sqrt(np.mean((saved["state"] - truth[:, None]) ** 2, axis=0))
        np.testing.assert_allclose(
            stats["metric:normalized_data_mismatch:all"], mismatch, rtol=1e-12
        )
        np.testing.assert_allclose(stats["metric:normalized_data_mismatch:y"], mismatch, rtol=1e-12)
        np.testing.assert_allclose(stats["metric:rmse:x"], rmse, rtol=1e-12)
        np.testing.assert_allclose(stats["metric:rmse:all"], rmse, rtol=1e-12)
        assert record.metrics is not None
        assert record.metrics["rmse"]["x"] == pytest.approx(rmse.mean(), rel=1e-12)
        assert record.metrics["normalized_data_mismatch"]["all"] == pytest.approx(
            mismatch.mean(), rel=1e-12
        )
        # One data type, so its data mismatch share is the whole data mismatch.
        np.testing.assert_allclose(
            stats["data_mismatch_by_type"][0], stats["data_mismatch"], rtol=1e-12
        )
        assert record.data_mismatch_by_type is not None
        assert record.data_mismatch_by_type["y"] == pytest.approx(record.data_mismatch_mean)
    first, last = result.history[0].metrics, result.history[-1].metrics
    assert first is not None
    assert last is not None
    assert last["rmse"]["all"] < first["rmse"]["all"]
    summary = json.loads((case.directory / "runs" / "case" / "summary.json").read_text())
    assert summary["final_metrics"] == last

    report = CliRunner().invoke(app, ["report", str(case.directory / "runs" / "case")])
    assert report.exit_code == 0, report.output
    assert "metrics, member means" in report.stdout
    assert "rmse" in report.stdout


def test_a_real_case_scores_the_data_only(linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=20)
    text = case.config.read_text().replace(
        '[case]\ntruth_file = "truth.npz"', '[case]\nkind = "real"'
    )
    case.config.write_text(text)
    result = composition.start(case.config)
    assert result.history[0].metrics is not None
    assert set(result.history[0].metrics) == {"data_mismatch", "normalized_data_mismatch"}

    case.config.write_text(text + '\n[[metrics]]\nname = "rmse"\n')
    with pytest.raises(ConfigError, match="real case study does not have"):
        composition.build_run(composition.load_config(case.config))


def test_rejected_trials_and_dropped_members_resume_identically(linear_case: CaseFactory) -> None:
    extra = f"fail_members = [4]\n{TANH}"
    output = "[output]\nsave_rejected_trials = true"
    whole = linear_case(
        name="whole",
        ensemble_size=20,
        method=GIES,
        simulator_extra=extra,
        failures=output,
    )
    parts = linear_case(
        name="parts",
        ensemble_size=20,
        method=GIES,
        simulator_extra=extra,
        failures=output,
    )
    reference = composition.start(whole.config)
    for _ in range(12):
        run = (
            composition.start(parts.config, max_steps=1)
            if _ == 0
            else composition.resume(parts.directory / "runs" / "parts", max_steps=1)
        )
        if run.complete:
            break
    np.testing.assert_array_equal(run.posterior, reference.posterior)
    assert run.history == reference.history


def test_configured_metrics_with_labels_and_an_actnum_file(linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=10)
    np.savez(case.directory / "actnum.npz", actnum=np.array([1, 1, 0]))
    metrics = (
        '\n[[metrics]]\nname = "rmse"\n'
        '\n[[metrics]]\nname = "rmse"\nlabel = "rmse_active"\n'
        'options = { actnum_file = "actnum.npz" }\n'
        '\n[[metrics]]\nname = "ssim"\noptions = { grid = [3], win_size = 3 }\n'
    )
    case.config.write_text(case.config.read_text() + metrics)
    result = composition.start(case.config)
    run = RunDirectory(case.directory / "runs" / "case")

    last = result.history[-1].metrics
    assert last is not None
    # Configured metrics replace the defaults; model_change is always recorded.
    assert set(last) == {"data_mismatch", "rmse", "rmse_active", "ssim", "model_change"}
    saved = run.load_arrays(f"iter_{result.history[-1].iteration}")["state"]
    truth = np.load(case.directory / "truth.npz")["x"]
    active = np.sqrt(np.mean((saved[:2] - truth[:2, None]) ** 2, axis=0))
    assert last["rmse_active"]["x"] == pytest.approx(active.mean(), rel=1e-12)

    case.config.write_text(case.config.read_text() + '\n[[metrics]]\nname = "rmse"\n')
    with pytest.raises(ConfigError, match="twice"):
        composition.build_run(composition.load_config(case.config))


def test_a_crash_before_state_json_resumes_identically(
    linear_case: CaseFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The process dies after a step's files but before its state.json: resume redoes it."""
    whole = linear_case(name="whole", ensemble_size=20, method=GIES, simulator_extra=TANH)
    parts = linear_case(name="parts", ensemble_size=20, method=GIES, simulator_extra=TANH)
    reference = composition.start(whole.config)
    run_dir = parts.directory / "runs" / "parts"
    composition.start(parts.config, max_steps=1)

    def crash(self: RunDirectory, state: object) -> None:
        raise KeyboardInterrupt("killed while writing state.json")

    with monkeypatch.context() as patched:
        patched.setattr(RunDirectory, "write_state", crash)
        with pytest.raises(KeyboardInterrupt):
            composition.resume(run_dir, max_steps=1)
    # The step's algorithm state was written, but state.json still names the previous one,
    # which was kept.
    saved = json.loads((run_dir / "state.json").read_text())
    files = {p.stem for p in (run_dir / "checkpoints").glob("algorithm_state_*.npz")}
    assert saved["algorithm_file"] in files
    assert len(files) == 2

    resumed = composition.resume(run_dir)

    np.testing.assert_array_equal(resumed.posterior, reference.posterior)
    assert resumed.history == reference.history


def test_gies_keeps_one_resume_pointer_and_saves_the_mean_model_forecast(
    linear_case: CaseFactory,
) -> None:
    case = linear_case(ensemble_size=20, method=GIES, simulator_extra=TANH)
    result = composition.start(case.config)
    checkpoints = _checkpoints(case)
    state = json.loads((checkpoints.parent / "state.json").read_text())

    # One algorithm state, the one state.json names; no old bookkeeping files.
    assert [p.stem for p in checkpoints.glob("algorithm_state_*.npz")] == [state["algorithm_file"]]
    assert not list(checkpoints.glob("current_*.npz"))
    assert not (checkpoints / "method_state.npz").exists()
    run = RunDirectory(checkpoints.parent)
    forward = np.load(case.directory / "forward.npz")["matrix"]
    for k in _files(case, "iter"):
        saved = run.load_arrays(f"iter_{k}")
        # RLM-MAC's reference: the forecast of the ensemble-mean model, tanh(G mean(x)).
        expected = np.tanh(forward @ saved["state"].mean(axis=1))
        np.testing.assert_allclose(saved["reference_prediction"], expected, rtol=1e-12)
    assert result.converged is not None


def test_bad_metric_options_fail_before_any_simulation(linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=10)
    case.config.write_text(
        case.config.read_text()
        + '\n[[metrics]]\nname = "ssim"\noptions = { grid = [5, 5], win_size = 3 }\n'
    )
    with pytest.raises(ConfigError, match="no state variable has 25 cells"):
        composition.build_run(composition.load_config(case.config))
    assert not (case.directory / "runs").exists()


def test_a_run_keeps_its_own_copy_of_the_truth(linear_case: CaseFactory) -> None:
    whole = linear_case(name="whole", ensemble_size=20)
    parts = linear_case(name="parts", ensemble_size=20)
    reference = composition.start(whole.config)
    composition.start(parts.config, max_steps=2)
    (parts.directory / "truth.npz").unlink()  # moved away mid-run

    resumed = composition.resume(parts.directory / "runs" / "parts")

    assert resumed.history == reference.history


@pytest.mark.parametrize("method", [ESMDA, GIES], ids=["esmda", "gies"])
def test_resumed_runs_update_from_c_ordered_arrays(
    linear_case: CaseFactory, method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Uninterrupted and resumed steps hand the method C-ordered arrays.

    Same values in another layout are not enough: on AVX-512 machines NumPy and OpenBLAS
    sum Fortran-ordered arrays in another order, and a resumed GIES run then drifted by
    2e-16 from an uninterrupted one (seen on CI, AMD EPYC 9V74). The drift itself only
    shows on such hardware, so the layout is checked directly.
    """
    from apex.algorithms.data_assimilation import ESMDA as EsmdaMethod
    from apex.algorithms.data_assimilation import GIES as GiesMethod

    layouts: list[tuple[str, bool]] = []

    def record(**arrays: object) -> None:
        for name, array in arrays.items():
            if isinstance(array, np.ndarray) and array.ndim == 2:
                layouts.append((name, array.flags.c_contiguous))

    if method == GIES:
        original = GiesMethod.update

        def spy(
            self: GiesMethod,
            state: np.ndarray,
            predicted: np.ndarray,
            perturbed: np.ndarray,
            *args: object,
            **kwargs: object,
        ) -> np.ndarray:
            record(state=state, predicted=predicted, perturbed=perturbed)
            return original(self, state, predicted, perturbed, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(GiesMethod, "update", spy)
    else:
        original_analysis = EsmdaMethod.analysis

        def spy_analysis(
            self: EsmdaMethod,
            step: int,
            state: np.ndarray,
            predicted: np.ndarray,
            *args: object,
            **kwargs: object,
        ) -> object:
            record(state=state, predicted=predicted)
            return original_analysis(self, step, state, predicted, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(EsmdaMethod, "analysis", spy_analysis)
    # Member 3 fails, so the loop also holds arrays with members dropped ([:, ok]).
    case = linear_case(
        ensemble_size=20, method=method, simulator_extra=f"fail_members = [3]\n{TANH}"
    )
    composition.start(case.config, max_steps=2)  # the uninterrupted steps
    composition.resume(case.directory / "runs" / "case")  # and the resumed ones

    assert layouts
    assert [name for name, c_ordered in layouts if not c_ordered] == []


def test_model_change_compares_each_forecast_with_its_starting_iterate(
    linear_case: CaseFactory,
) -> None:
    """Accepted iterates against the previous accepted one, trials against their start."""
    case = linear_case(
        ensemble_size=20,
        method=GIES,
        simulator_extra=TANH,
        failures='[output]\nsave_rejected_trials = true\n\n[[metrics]]\nname = "model_change"',
    )
    result = composition.start(case.config)
    run = RunDirectory(case.directory / "runs" / "case")

    assert "metric:model_change:all" not in run.load_arrays("data_mismatch_0")
    start = run.load_arrays("iter_0")["state"]
    for record in result.history[1:]:
        kind = "iter" if record.accepted else "trial"
        state = run.load_arrays(f"{kind}_{record.iteration}")["state"]
        expected = np.linalg.norm(state - start, axis=0) / np.sqrt(state.shape[0])
        got = run.load_arrays(f"data_mismatch_{record.iteration}")["metric:model_change:all"]
        np.testing.assert_allclose(got, expected, rtol=1e-14)
        if record.accepted:
            start = state
    assert any(not r.accepted for r in result.history), "the case should reject some trials"
    iterations, _, _ = composition.load_results(run.path).metric("model_change")
    assert iterations == [r.iteration for r in result.history[1:]]
    figures = composition.write_report_figures(run.path, case.directory / "report")
    assert any("model_change" in f.name for f in figures)


@pytest.mark.parametrize("method", [GIES, ESMDA], ids=["gies", "esmda"])
def test_model_change_follows_members_across_drops_and_resume(
    linear_case: CaseFactory, method: str
) -> None:
    """Members dropped mid-run are matched by id, and a resumed run records the same values."""
    extra = 'nonlinearity = "tanh"\nfail_above = 1.6'
    metrics = '[failures]\nmin_success_fraction = 0.5\n\n[[metrics]]\nname = "model_change"'
    whole = linear_case(
        name="whole", ensemble_size=24, method=method, simulator_extra=extra, failures=metrics
    )
    parts = linear_case(
        name="parts", ensemble_size=24, method=method, simulator_extra=extra, failures=metrics
    )
    reference = composition.start(whole.config)
    composition.start(parts.config, max_steps=2)
    composition.resume(parts.directory / "runs" / "parts")

    sizes = [r.n_members for r in reference.history]
    assert sizes[-1] < sizes[0], "members should be dropped in an update"
    run = RunDirectory(whole.directory / "runs" / "whole")
    resumed = RunDirectory(parts.directory / "runs" / "parts")
    start = run.load_arrays("iter_0")
    for record in reference.history[1:]:
        scored = run.load_arrays(f"data_mismatch_{record.iteration}")
        again = resumed.load_arrays(f"data_mismatch_{record.iteration}")
        np.testing.assert_array_equal(
            scored["metric:model_change:all"], again["metric:model_change:all"]
        )
        if not record.accepted:
            continue
        now = run.load_arrays(f"iter_{record.iteration}")
        columns = [list(start["member_ids"]).index(m) for m in now["member_ids"]]
        expected = np.linalg.norm(now["state"] - start["state"][:, columns], axis=0)
        np.testing.assert_allclose(
            scored["metric:model_change:all"], expected / np.sqrt(3), rtol=1e-14
        )
        start = now


def test_model_change_is_always_recorded_over_the_rmse_actnum(linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=10)
    np.savez(case.directory / "actnum.npz", actnum=np.array([1, 0, 1]))
    case.config.write_text(
        case.config.read_text()
        + '\n[[metrics]]\nname = "rmse"\noptions = { actnum_file = "actnum.npz" }\n'
    )

    built = composition.build_metrics(composition.load_config(case.config))
    result = composition.start(case.config)

    assert set(built) == {"rmse", "model_change"}
    np.testing.assert_array_equal(built["model_change"].active, [True, False, True])  # type: ignore[attr-defined]
    run = RunDirectory(case.directory / "runs" / "case")
    before, after = run.load_arrays("iter_0"), run.load_arrays("iter_1")
    expected = np.linalg.norm((after["state"] - before["state"])[[0, 2]], axis=0) / np.sqrt(2)
    np.testing.assert_allclose(
        run.load_arrays("data_mismatch_1")["metric:model_change:x"], expected, rtol=1e-14
    )
    assert result.history[0].metrics is not None
    assert "model_change" not in result.history[0].metrics  # not defined for the prior

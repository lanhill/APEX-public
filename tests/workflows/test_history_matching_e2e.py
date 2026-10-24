"""End-to-end history matching on generated linear-Gaussian cases (no reservoir simulator)."""

import itertools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pytest
from typer.testing import CliRunner

from apex.cli.main import app
from apex.core.errors import ConfigError, WorkflowError
from apex.services import composition
from apex.storage.run_directory import RunDirectory


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path

    def exact_posterior(self) -> tuple[np.ndarray, np.ndarray]: ...

    def least_squares(self, perturbed: np.ndarray) -> np.ndarray: ...


CaseFactory = Callable[..., Case]

GIES_METHOD = (
    'name = "gies"\nstrategy = { name = "rlmmac", options = { energy = 1.0 } }\n'
    "schedule = { data_mismatch_tolerance = 1e-9 }\nmax_iterations = 40"
)


def _run_dir(case: Case) -> Path:
    return case.directory / "runs" / "case"


def test_esmda_run_recovers_the_exact_linear_gaussian_posterior(linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=3000)

    result = composition.start(case.config)

    mean, covariance = case.exact_posterior()
    # Monte Carlo error of the mean ~ sqrt(var / N) < 0.01; 0.04 leaves margin for the
    # O(1/N) sampling error of the gains.
    np.testing.assert_allclose(result.posterior.mean(axis=-1), mean, atol=0.04)
    np.testing.assert_allclose(np.cov(result.posterior), covariance, atol=0.04)
    assert result.complete
    assert [r.iteration for r in result.history] == [0, 1, 2, 3, 4]
    summary = json.loads((_run_dir(case) / "summary.json").read_text())
    assert summary["data_mismatch_reduction"] > 0.9
    posterior_file = RunDirectory(_run_dir(case)).load_arrays("posterior")
    np.testing.assert_array_equal(posterior_file["x"], result.posterior)


def test_gies_run_converges_to_the_least_squares_fit(linear_case: CaseFactory) -> None:
    # The default reduction (0.9) lowers the damping gently: a tighter tolerance and more
    # iterations reach the fit.
    method = GIES_METHOD.replace("1e-9", "1e-13").replace(
        "max_iterations = 40", "max_iterations = 80"
    )
    case = linear_case(method=method, ensemble_size=30)

    result = composition.start(case.config)

    saved = RunDirectory(_run_dir(case)).load_arrays("perturbed_observations")
    np.testing.assert_array_equal(saved["member_ids"], result.member_ids)
    perturbed = saved["perturbed_observations"]
    np.testing.assert_allclose(
        result.posterior, case.least_squares(perturbed), rtol=1e-4, atol=1e-6
    )
    accepted = [r.data_mismatch_mean for r in result.history if r.accepted]
    assert all(b < a for a, b in itertools.pairwise(accepted))
    assert result.converged


@pytest.mark.parametrize("method", ["esmda", "gies"])
def test_resumed_run_is_identical_to_an_uninterrupted_one(
    linear_case: CaseFactory, method: str
) -> None:
    method_text = GIES_METHOD if method == "gies" else None
    kwargs: dict[str, Any] = {"method": method_text} if method_text else {}
    whole = linear_case(name="whole", ensemble_size=40, **kwargs)
    parts = linear_case(name="parts", ensemble_size=40, **kwargs)

    reference = composition.start(whole.config)
    first = composition.start(parts.config, max_steps=2)
    resumed = composition.resume(parts.directory / "runs" / "parts")

    assert not first.complete
    np.testing.assert_array_equal(resumed.posterior, reference.posterior)
    assert resumed.history == reference.history


def test_process_executor_gives_the_same_result_as_serial(linear_case: CaseFactory) -> None:
    serial = linear_case(name="serial", ensemble_size=24)
    pooled = linear_case(
        name="pooled", ensemble_size=24, execution='name = "process"\noptions = { workers = 3 }'
    )

    np.testing.assert_array_equal(
        composition.start(pooled.config).posterior, composition.start(serial.config).posterior
    )


def test_failed_members_are_dropped_and_recorded(linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=40, simulator_extra="fail_members = [3, 17]")

    result = composition.start(case.config)

    assert result.posterior.shape == (3, 38)
    assert 3 not in result.member_ids
    assert 17 not in result.member_ids
    assert {r.n_members for r in result.history} == {38}
    events = [
        json.loads(line) for line in (_run_dir(case) / "events.jsonl").read_text().splitlines()
    ]
    failed = {e["member_id"] for e in events if e.get("event") == "member_failed"}
    assert failed == {3, 17}


def test_too_many_failures_abort_the_run(linear_case: CaseFactory) -> None:
    case = linear_case(
        ensemble_size=20,
        simulator_extra="fail_members = [0, 1, 2]",
        failures="[failures]\nmin_success_fraction = 0.9",
    )

    with pytest.raises(WorkflowError, match="only 17 of 20"):
        composition.start(case.config)


def test_invalid_configurations_raise_config_errors(linear_case: CaseFactory) -> None:
    case = linear_case(method='name = "no_such_method"')

    with pytest.raises(ConfigError, match="unknown method"):
        composition.start(case.config)

    case.config.write_text(case.config.read_text().replace("seed = 2026", "seed = 2026\ntypo = 1"))
    with pytest.raises(ConfigError, match="typo"):
        composition.load_config(case.config)


def test_starting_twice_in_the_same_directory_is_refused(linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=10)
    composition.start(case.config)

    with pytest.raises(WorkflowError, match="already holds a run"):
        composition.start(case.config)


def test_cli_run_resume_and_report(linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=20)
    runner = CliRunner()

    started = runner.invoke(app, ["run", str(case.config), "--max-steps", "1"])
    resumed = runner.invoke(app, ["resume", str(_run_dir(case))])
    reported = runner.invoke(app, ["report", str(_run_dir(case))])
    failed = runner.invoke(app, ["run", str(case.config)])

    assert started.exit_code == 0, started.output
    assert "stopped (resume to continue)" in started.stdout
    assert resumed.exit_code == 0, resumed.output
    assert "complete" in resumed.stdout
    assert reported.stdout.count("\n") >= 6  # header, rule, iterations 0-4
    assert failed.exit_code == 1
    assert "already holds a run" in failed.output


@pytest.mark.parametrize("method", ["esmda", "gies"])
def test_report_writes_data_mismatch_and_data_match_figures(
    linear_case: CaseFactory, method: str
) -> None:
    kwargs: dict[str, Any] = {"method": GIES_METHOD} if method == "gies" else {}
    case = linear_case(ensemble_size=20, **kwargs)
    # A 3x1 "grid" for x, so the parameter maps are drawn too.
    case.config.write_text(
        case.config.read_text().replace("size = 3\n", "size = 3\ngrid = [3, 1]\n", 1)
    )
    composition.start(case.config)

    result = CliRunner().invoke(app, ["report", str(_run_dir(case)), "--plots"])

    assert result.exit_code == 0, result.output
    report = _run_dir(case) / "report"
    names = (
        "metrics/data_mismatch",
        "metrics/metric_normalized_data_mismatch_all",
        "metrics/metric_rmse_x",
        "data/y_initial",
        "data/y_final",
        "fields/x_true_layer1",
        "fields/x_final_std_layer1",
        "histograms/x_prior",
        "histograms/x_final",
    )
    for name in names:
        figure = report / f"{name}.png"
        assert figure.stat().st_size > 10_000
        assert figure.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert (report / f"{name}.pdf").read_bytes()[:5] == b"%PDF-"
    table = (report / "metrics" / "metrics_summary.csv").read_text().splitlines()
    assert table[0] == "metric,label,iteration,mean,std,median,p10,p90"
    assert any(row.startswith("rmse,x,0,") for row in table)


def _localized_gies(localization: str) -> str:
    return (
        'name = "gies"\nstrategy = { name = "rlmmac", options = { energy = 1.0 } }\n'
        f"localization = {localization}\n"
        "schedule = { data_mismatch_tolerance = 1e-9 }\nmax_iterations = 12"
    )


@pytest.mark.parametrize(
    "localization",
    [
        '{ name = "adaptive_per_update", options = { threshold_scale = 1.0, taper = "soft" } }',
        '{ name = "adaptive", options = { taper = "hard" } }',  # the default: prior only
    ],
)
def test_localized_gies_runs_and_resumes_identically(
    linear_case: CaseFactory, localization: str
) -> None:
    whole = linear_case(name="whole", ensemble_size=30, method=_localized_gies(localization))
    parts = linear_case(name="parts", ensemble_size=30, method=_localized_gies(localization))

    reference = composition.start(whole.config)
    composition.start(parts.config, max_steps=3)
    resumed = composition.resume(parts.directory / "runs" / "parts")

    accepted = [r.data_mismatch_mean for r in reference.history if r.accepted]
    assert accepted[-1] < 0.5 * accepted[0]
    np.testing.assert_array_equal(resumed.posterior, reference.posterior)
    assert resumed.history == reference.history


@pytest.mark.parametrize("localization", ["adaptive_per_update", "adaptive", "adaptive_prior"])
def test_localized_esmda_runs_and_resumes_identically(
    linear_case: CaseFactory, localization: str
) -> None:
    method = (
        'name = "esmda"\noptions = { inflation = [2, 2] }\n'
        'strategy = { name = "approx", options = { energy = 1.0 } }\n'
        f'localization = {{ name = "{localization}", options = {{ taper = "soft" }} }}'
    )
    whole = linear_case(name="whole", ensemble_size=30, method=method)
    parts = linear_case(name="parts", ensemble_size=30, method=method)

    reference = composition.start(whole.config)
    composition.start(parts.config, max_steps=1)
    resumed = composition.resume(parts.directory / "runs" / "parts")

    np.testing.assert_array_equal(resumed.posterior, reference.posterior)
    assert reference.history[-1].data_mismatch_mean < reference.history[0].data_mismatch_mean
    # The prior taper is part of the algorithm's checkpointed state.
    [state_file] = (parts.directory / "runs" / "parts" / "checkpoints").glob(
        "algorithm_state_*.npz"
    )
    saved = np.load(state_file)
    # "adaptive" is the prior-only method (2026-09-28), as its alias "adaptive_prior".
    assert ("method_state:taper" in saved) == (localization != "adaptive_per_update")


@pytest.mark.parametrize(
    ("name", "options"),
    [
        ("esmda", "inflation = [2, 2]"),
        ("lm_enrml", "max_iterations = 3"),
        ("gn_enrml", "max_iterations = 3"),
    ],
)
def test_projected_localization_runs_and_resumes_identically(
    linear_case: CaseFactory, name: str, options: str
) -> None:
    """Upstream PET's ``autoadaloc`` (rank-r) with the approx analysis of each method."""
    method = (
        f'name = "{name}"\noptions = {{ {options} }}\n'
        'strategy = { name = "approx", options = { energy = 0.99 } }\n'
        'localization = { name = "adaptive_projected", options = { taper = "soft" } }'
    )
    whole = linear_case(name="whole", ensemble_size=30, method=method)
    parts = linear_case(name="parts", ensemble_size=30, method=method)

    reference = composition.start(whole.config)
    composition.start(parts.config, max_steps=1)
    resumed = composition.resume(parts.directory / "runs" / "parts")

    np.testing.assert_array_equal(resumed.posterior, reference.posterior)
    assert resumed.history == reference.history
    accepted = [r.data_mismatch_mean for r in reference.history if r.accepted]
    assert accepted[-1] < accepted[0]


@pytest.mark.parametrize(
    ("name", "options"),
    [("esmda", "inflation = [2, 2]"), ("lm_enrml", "max_iterations = 3")],
)
def test_entry_distance_localization_runs_and_resumes_identically(
    linear_case: CaseFactory, name: str, options: str
) -> None:
    """Upstream PET's ``distance_loc`` entries, the variable names supplied from [state]."""
    method = (
        f'name = "{name}"\noptions = {{ {options} }}\n'
        'strategy = { name = "approx", options = { energy = 0.99 } }\n'
        'localization = { name = "distance_entries", options = { grid = [3, 1, 1], entries = ['
        '{ parameter = "x", taper = "gaspari_cohn", position = [2, 1, 1], radius = 1.5 }] } }'
    )
    whole = linear_case(name="whole", ensemble_size=30, method=method)
    parts = linear_case(name="parts", ensemble_size=30, method=method)

    reference = composition.start(whole.config)
    composition.start(parts.config, max_steps=1)
    resumed = composition.resume(parts.directory / "runs" / "parts")

    np.testing.assert_array_equal(resumed.posterior, reference.posterior)
    assert resumed.history == reference.history
    accepted = [r.data_mismatch_mean for r in reference.history if r.accepted]
    assert accepted[-1] < accepted[0]


def test_an_imported_taper_file_is_relative_to_the_configuration(
    linear_case: CaseFactory,
) -> None:
    method = (
        'name = "esmda"\noptions = { inflation = [2, 2] }\n'
        'strategy = { name = "approx", options = { energy = 0.99 } }\n'
        'localization = { name = "distance_entries", options = { grid = [3, 1, 1], entries = ['
        '{ taper = "import", taper_file = "mask.npy" }] } }'
    )
    case = linear_case(name="imported", ensemble_size=30, method=method)
    np.save(case.directory / "mask.npy", np.array([1.0, 0.0, 1.0]))  # x[1] is not updated

    result = composition.start(case.config)

    prior = RunDirectory(case.directory / "runs" / "imported").load_arrays("iter_0")["state"]
    np.testing.assert_array_equal(result.posterior[1], prior[1])
    assert not np.array_equal(result.posterior[0], prior[0])


def test_runs_saved_before_the_localization_stream_still_resume(linear_case: CaseFactory) -> None:
    # Simulate a run directory written before the localization stream existed: its
    # state.json has no state for it, which then starts fresh (deterministically).
    whole = linear_case(name="whole", ensemble_size=30, method=GIES_METHOD)
    parts = linear_case(name="parts", ensemble_size=30, method=GIES_METHOD)
    reference = composition.start(whole.config)
    composition.start(parts.config, max_steps=2)
    run_dir = parts.directory / "runs" / "parts"
    state = json.loads((run_dir / "state.json").read_text())
    del state["random_streams"]["localization"]
    (run_dir / "state.json").write_text(json.dumps(state))

    resumed = composition.resume(run_dir)

    # Unlocalized GIES never draws from the new stream, so the result is unchanged.
    np.testing.assert_array_equal(resumed.posterior, reference.posterior)


def _with_prior(case: Case, prior: str, state_extra: str = "") -> None:
    text = case.config.read_text()
    start, end = text.index("[prior]"), text.index("[observations]")
    text = text[:start] + prior + "\n\n" + text[end:]
    if state_extra:
        text = text.replace('name = "x"\nsize = 3\n', f'name = "x"\nsize = 3\n{state_extra}\n')
    case.config.write_text(text)


def test_per_variable_prior_generators_and_bounds(linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=40)
    _with_prior(
        case,
        '[prior]\nensemble_size = 40\n\n[prior.variables.x]\nname = "gaussian_field"\n'
        "options = { grid = [3, 1, 1], mean = 5.0, variance = 4.0, major_range = 2.0 }",
        state_extra="upper = 6.0",
    )
    run = composition.build_run(composition.load_config(case.config))

    prior = composition.draw_prior(run)

    assert prior.shape == (3, 40)
    assert prior.max() <= 6.0  # clipped to the state bound
    assert prior.mean() == pytest.approx(5.0, abs=1.0)
    corr = np.corrcoef(prior)
    assert corr[0, 1] > corr[0, 2]  # spatially correlated: neighbours more alike


def test_stored_prior_ensembles_are_not_clipped(linear_case: CaseFactory) -> None:
    case = linear_case()
    stored = np.arange(30.0).reshape(3, 10)
    np.savez(case.config.parent / "stored.npz", x=stored)
    _with_prior(
        case,
        '[prior]\ngenerator = { name = "file", options = { ensemble_file = "stored.npz" } }',
        state_extra="upper = 6.0",
    )
    run = composition.build_run(composition.load_config(case.config))

    # As PET's importstate: a stored ensemble is used as is.
    np.testing.assert_array_equal(composition.draw_prior(run), stored)


@pytest.mark.parametrize(
    ("prior", "message"),
    [
        ('[prior]\nensemble_size = 10\n\n[prior.variables.y]\nname = "gaussian"', "unknown state"),
        ('[prior]\nensemble_size = 10\ngenerator = { name = "kriging" }', "unknown priors"),
        ('[prior]\ngenerator = { name = "gaussian" }', "ensemble_size"),
    ],
)
def test_prior_configuration_errors_are_reported(
    linear_case: CaseFactory, prior: str, message: str
) -> None:
    case = linear_case()
    _with_prior(case, prior)

    with pytest.raises(ConfigError, match=message):
        composition.start(case.config)


@pytest.mark.parametrize("method", ["esmda", "gies"])
def test_resuming_a_complete_run_changes_nothing(linear_case: CaseFactory, method: str) -> None:
    # Regression: a finished ES-MDA run was checkpointed without a "done" marker, so its
    # resume forecast the last posterior again and appended a duplicate history record.
    kwargs = {"method": GIES_METHOD} if method == "gies" else {}
    case = linear_case(ensemble_size=24, **kwargs)
    run_dir = case.directory / "runs" / "case"
    finished = composition.start(case.config)
    history = RunDirectory(run_dir).history_dicts()

    resumed = composition.resume(run_dir)

    assert RunDirectory(run_dir).history_dicts() == history
    np.testing.assert_array_equal(resumed.posterior, finished.posterior)


def test_gies_damping_follows_a_configured_reduction_factor(linear_case: CaseFactory) -> None:
    # [method] schedule = { reduction = 0.9 }: lambda * 0.9 after an accepted trial and
    # * factor (2) after a rejected one, as the MATLAB RLM-MAC (RLM_average_cost.m).
    method = (
        'name = "gies"\nstrategy = { name = "rlmmac", options = { energy = 1.0 } }\n'
        "schedule = { initial = 1.0, factor = 2.0, reduction = 0.9, "
        "data_mismatch_tolerance = 1e-9 }\n"
        "max_iterations = 8"
    )
    case = linear_case(ensemble_size=24, method=method, simulator_extra='nonlinearity = "tanh"')
    composition.start(case.config)
    trials = RunDirectory(case.directory / "runs" / "case").history_dicts()[1:]
    assert trials[0]["damping"] == 1.0
    for before, after in itertools.pairwise(trials):
        expected = before["damping"] * (0.9 if before["accepted"] else 2.0)
        assert after["damping"] == pytest.approx(expected, rel=1e-12)
    assert any(r["accepted"] for r in trials[1:])
    assert any(not r["accepted"] for r in trials)  # both branches are exercised


TANH = 'nonlinearity = "tanh"'
RETRYING_GIES = (
    'name = "gies"\noptions = { max_trials_per_iteration = 2 }\n'
    'strategy = { name = "rlmmac", options = { energy = 1.0 } }\n'
    "schedule = { initial = 1.0, factor = 2.0, reduction = 0.9, data_mismatch_tolerance = 1e-9 }\n"
    "max_iterations = 6"
)


def _expected_dampings(
    accepted: list[bool], limit: int, *, reduction: float = 0.9, initial: float = 1.0
) -> tuple[list[float], int]:
    """The MATLAB RLM-MAC damping of each trial, and the outer iterations they complete.

    MATLAB RLM_average_cost.m: * reduction after an acceptance, * 2 after a rejection,
    and when ``limit`` trials in a row are rejected, * 2 once more (at least the initial)
    and the iteration ends without an update.
    """
    damping, trials, outer, used = initial, 0, 0, []
    for ok in accepted:
        used.append(damping)
        trials += 1
        if ok:
            damping, outer, trials = damping * reduction, outer + 1, 0
        else:
            damping *= 2.0
            if trials >= limit:
                damping, outer, trials = max(damping * 2.0, initial), outer + 1, 0
    return used, outer


@pytest.mark.parametrize("limit", [1, 2])
def test_gies_retries_rejected_trials_within_an_iteration(
    linear_case: CaseFactory, limit: int
) -> None:
    # Measured on this case: trial 5 is rejected and trial 6 accepted. With limit 2 that is
    # one iteration (8 trials for 6 iterations); with limit 1 the rejection ends its
    # iteration (damping * 2 twice, no update), so every trial is an iteration (6).
    method = RETRYING_GIES.replace(
        "max_trials_per_iteration = 2", f"max_trials_per_iteration = {limit}"
    )
    case = linear_case(ensemble_size=24, method=method, simulator_extra=TANH)
    result = composition.start(case.config)
    trials = RunDirectory(case.directory / "runs" / "case").history_dicts()[1:]
    accepted = [bool(r["accepted"]) for r in trials]
    dampings, outer = _expected_dampings(accepted, limit=limit)
    assert [r["damping"] for r in trials] == pytest.approx(dampings, rel=1e-12)
    assert not result.converged
    assert outer == 6  # the budget counts iterations, not trials
    assert len(trials) == {1: 6, 2: 8}[limit]
    assert [r["iteration"] for r in trials] == list(range(1, len(trials) + 1))


def test_exhausted_retries_raise_the_damping_to_the_initial_one(linear_case: CaseFactory) -> None:
    # An aggressive reduction (0.1) overshoots: each accepted trial is followed by two
    # rejected ones (measured: A r r A r r A r r). The second rejection ends the iteration,
    # and 0.2 * 2 * 2 = 0.8 is raised to the initial damping, 1.0.
    method = RETRYING_GIES.replace("reduction = 0.9", "reduction = 0.1")
    case = linear_case(ensemble_size=24, method=method, simulator_extra=TANH)
    composition.start(case.config)
    run_dir = case.directory / "runs" / "case"
    trials = RunDirectory(run_dir).history_dicts()[1:]
    accepted = [bool(r["accepted"]) for r in trials]
    assert accepted == [True, False, False] * 3
    dampings, outer = _expected_dampings(accepted, limit=2, reduction=0.1)
    assert [r["damping"] for r in trials] == pytest.approx(dampings, rel=1e-12)
    assert trials[3]["damping"] == 1.0  # clamped: 0.8 raised to the initial damping
    # The method's own count: 3 accepted and 3 exhausted iterations, 9 trials.
    assert json.loads((run_dir / "state.json").read_text())["algorithm"]["outer"] == outer == 6
    # An exhausted iteration keeps the accepted iterate: only iterations 1, 4 and 7 are saved.
    saved = sorted(p.stem for p in (run_dir / "checkpoints").glob("iter_*.npz"))
    assert saved == ["iter_0", "iter_1", "iter_4", "iter_7"]


@pytest.mark.parametrize("reduction", ["0.9", "0.1"])  # 0.1: through exhausted iterations
def test_gies_with_retries_resumes_bit_identically(
    linear_case: CaseFactory, reduction: str
) -> None:
    method = RETRYING_GIES.replace("reduction = 0.9", f"reduction = {reduction}")
    whole_case = linear_case(name="whole", ensemble_size=24, method=method, simulator_extra=TANH)
    whole = composition.start(whole_case.config)
    parts_case = linear_case(name="parts", ensemble_size=24, method=method, simulator_extra=TANH)
    run_dir = parts_case.directory / "runs" / "parts"
    composition.start(parts_case.config, max_steps=1)
    for _ in range(12):  # one trial at a time, through the retries
        if json.loads((run_dir / "state.json").read_text())["algorithm"]["phase"] == "done":
            break
        composition.resume(run_dir, max_steps=1)
    resumed = composition.resume(run_dir)
    np.testing.assert_array_equal(resumed.posterior, whole.posterior)
    assert (
        RunDirectory(run_dir).history_dicts()
        == RunDirectory(whole_case.directory / "runs" / "whole").history_dicts()
    )


def test_a_run_from_before_the_reduction_default_resumes_by_halving(
    linear_case: CaseFactory,
) -> None:
    """Until 2026-10-13 rlmmac divided the damping by its factor after an accepted step; the
    default is 0.9. Such runs must resume as they began."""
    halving = GIES_METHOD.replace("schedule = { ", "schedule = { reduction = 0.5, ")
    whole = linear_case(name="whole", ensemble_size=24, method=halving, simulator_extra=TANH)
    parts = linear_case(name="parts", ensemble_size=24, method=halving, simulator_extra=TANH)
    reference = composition.start(whole.config)
    composition.start(parts.config, max_steps=3)
    run_dir = parts.directory / "runs" / "parts"
    # Make it look like a run started before the change: no reduction, snapshot version 3.
    snapshot = json.loads((run_dir / "config.json").read_text())
    del snapshot["method"]["schedule"]["reduction"]
    (run_dir / "config.json").write_text(json.dumps(snapshot))
    (run_dir / "snapshot_version").write_text("3\n")

    resumed = composition.resume(run_dir)

    np.testing.assert_array_equal(resumed.posterior, reference.posterior)
    dampings = [r.damping for r in RunDirectory(run_dir).history()]
    assert dampings == [
        r.damping for r in RunDirectory(whole.directory / "runs" / "whole").history()
    ]
    assert 0.25 in dampings  # halved: 1, 0.5, 0.25 on the first acceptances


def test_a_run_from_before_the_rename_resumes_with_its_per_update_localization(
    linear_case: CaseFactory,
) -> None:
    """Until 2026-09-28 "adaptive" meant per update; such runs must resume as they began."""
    localization = '{ name = "adaptive_per_update", options = { taper = "soft" } }'
    # An explicit reduction, which the snapshot keeps: only the localization is migrated.
    method = _localized_gies(localization).replace(
        "schedule = { ", "schedule = { reduction = 0.9, "
    )
    whole = linear_case(name="whole", ensemble_size=30, method=method)
    parts = linear_case(name="parts", ensemble_size=30, method=method)
    reference = composition.start(whole.config)
    composition.start(parts.config, max_steps=3)
    run_dir = parts.directory / "runs" / "parts"
    # Make it look like a run started before the rename: the old name, no version marker.
    snapshot = run_dir / "config.json"
    snapshot.write_text(snapshot.read_text().replace('"adaptive_per_update"', '"adaptive"'))
    (run_dir / "snapshot_version").unlink()

    resumed = composition.resume(run_dir)

    np.testing.assert_array_equal(resumed.posterior, reference.posterior)


def _tamper_accepted_data_mismatch(run_dir: Path, relative: float) -> None:
    state_file = run_dir / "state.json"
    state = json.loads(state_file.read_text())
    state["algorithm"]["data_mismatch"] *= 1 + relative
    state_file.write_text(json.dumps(state))


def test_gies_resume_refuses_an_inconsistent_accepted_data_mismatch(
    linear_case: CaseFactory,
) -> None:
    """Resume recomputes GIES's accepted data mismatch from the saved forecast and perturbed
    observations; a stored value far from it (a changed formula, corrupted arrays) would
    silently change the next accept/reject decision, so resume refuses it.
    """
    case = linear_case(method=GIES_METHOD, ensemble_size=20)
    run_dir = _run_dir(case)
    composition.start(case.config, max_steps=2)
    _tamper_accepted_data_mismatch(run_dir, 1e-6)

    with pytest.raises(RuntimeError, match="inconsistent"):
        composition.resume(run_dir)


def test_gies_resume_tolerates_rounding_of_the_accepted_data_mismatch(
    linear_case: CaseFactory, caplog: pytest.LogCaptureFixture
) -> None:
    """A rounding-sized difference (another CPU or library) is logged, and the stored value
    is used, so the resumed run equals an uninterrupted one.
    """
    whole = linear_case(name="whole", method=GIES_METHOD, ensemble_size=20)
    uninterrupted = composition.start(whole.config)
    case = linear_case(method=GIES_METHOD, ensemble_size=20)
    run_dir = _run_dir(case)
    composition.start(case.config, max_steps=2)
    _tamper_accepted_data_mismatch(run_dir, 1e-15)

    resumed = composition.resume(run_dir)

    assert any(
        getattr(r, "event", None) == "data_mismatch_recomputed_differs" for r in caplog.records
    )
    np.testing.assert_array_equal(resumed.posterior, uninterrupted.posterior)


def test_es_approaches_the_kalman_posterior(linear_case: CaseFactory) -> None:
    """Independent reference: for a linear-Gaussian model the ES update with perturbed
    observations is the Kalman update; a large ensemble approaches its posterior."""
    case = linear_case(method='name = "es"', ensemble_size=2000)

    result = composition.start(case.config)

    mean, covariance = case.exact_posterior()
    # Monte Carlo errors with 2000 members: the mean's is sqrt(var / N) <= 0.006, so 0.03 is
    # ~5 sigma; a covariance entry's is at most sqrt(2 / N) var ~ 0.002 (largest posterior
    # variance 0.068), so 0.01 is ~5 sigma. The run is seeded, so this is deterministic.
    np.testing.assert_allclose(result.posterior.mean(axis=-1), mean, atol=0.03)
    np.testing.assert_allclose(np.cov(result.posterior), covariance, atol=0.01)
    assert [r.iteration for r in result.history] == [0, 1]  # prior and one update
    assert result.complete

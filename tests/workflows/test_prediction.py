"""Predictions from finished runs, on generated linear-Gaussian cases.

The prediction "deck" is the case's linear model with four more report times: its first
eight rows are the history model, so every forecast is known exactly (``G_pred @ x``).
"""

import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import pytest
from typer.testing import CliRunner

from apex.algorithms.metrics.forecast import crps
from apex.cli.main import app
from apex.core.errors import ConfigError, WorkflowError
from apex.services import composition, prediction

N_HISTORY, N_FUTURE = 8, 4


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path
    forward: np.ndarray


CaseFactory = Callable[..., Case]


def _prediction(
    case: Case,
    *,
    body: str = "",
    ensembles: str = '["posterior", "prior", "truth"]',
    perturb_history: bool = False,
    simulator_extra: str = "",
    name: str = "future",
) -> tuple[Path, np.ndarray]:
    """Write the prediction model and configuration; return its path and ``G_pred``."""
    rng = np.random.default_rng(7)
    history = case.forward.copy()
    if perturb_history:
        history[3] *= 1.01  # a "deck" whose history differs from the run's
    g_pred = np.vstack([history, rng.standard_normal((N_FUTURE, history.shape[1]))])
    np.savez(case.directory / f"forward_{name}.npz", matrix=g_pred)
    outputs = ", ".join(f'["t{i}", "y"]' for i in range(N_HISTORY + N_FUTURE))
    path = case.directory / f"{name}.toml"
    path.write_text(
        f"""
[prediction]
name = "{name}"
run_dir = "runs/case"
ensembles = {ensembles}
{body}

[prediction.outputs]
vectors = ["y"]

[simulator]
options = {{ forward_file = "forward_{name}.npz", outputs = [{outputs}] {simulator_extra} }}
"""
    )
    return path, g_pred


def _posterior_and_prior(case: Case) -> tuple[np.ndarray, np.ndarray]:
    results = composition.load_results(case.directory / "runs" / "case")
    posterior = results.iterate(results.posterior_iteration).state["x"]
    return posterior, results.iterate(0).state["x"]


@pytest.fixture
def finished(linear_case: CaseFactory) -> Case:
    case = linear_case(ensemble_size=30)
    composition.start(case.config)
    return case


def test_every_member_is_forecast_exactly(finished: Case) -> None:
    path, g_pred = _prediction(finished, body="include_mean_model = true")

    result = prediction.predict(path)

    posterior, prior = _posterior_and_prior(finished)
    forecast = result.forecasts["posterior"]
    assert forecast.times == tuple(f"t{i}" for i in range(12))
    assert forecast.vectors == ("y",)
    np.testing.assert_allclose(forecast.values[0], g_pred @ posterior, rtol=1e-13)
    np.testing.assert_allclose(result.forecasts["prior"].values[0], g_pred @ prior, rtol=1e-13)
    assert forecast.mean_model is not None
    np.testing.assert_allclose(forecast.mean_model[0], g_pred @ posterior.mean(axis=1), rtol=1e-13)
    truth = np.load(finished.directory / "truth.npz")["x"]
    assert result.truth is not None
    np.testing.assert_allclose(result.truth[0], g_pred @ truth, rtol=1e-13)
    # The history ends at t7, the last observation time.
    assert result.first_prediction_time == N_HISTORY


def test_statistics_and_scores_are_written(finished: Case) -> None:
    path, _ = _prediction(finished)

    result = prediction.predict(path)

    out = finished.directory / "runs" / "case" / "predictions" / "future"
    rows = list(csv.DictReader((out / "statistics.csv").open()))
    assert len(rows) == 2 * 12  # two ensembles, one vector, twelve times
    row = next(r for r in rows if r["ensemble"] == "posterior" and r["time"] == "t10")
    values = result.forecasts["posterior"].values[0, 10]
    assert float(row["p10"]) == pytest.approx(np.percentile(values, 10))
    assert float(row["p90"]) == pytest.approx(np.percentile(values, 90))
    assert float(row["std"]) == pytest.approx(values.std(ddof=1))
    # Scores over the prediction period (t8-t11) only, each as its kernel computes it.
    assert result.truth is not None
    future = slice(N_HISTORY, None)
    expected = crps(result.forecasts["prior"].values[0, future], result.truth[0, future])
    assert result.scores["prior"]["y"]["crps"] == pytest.approx(expected)
    assert json.loads((out / "scores.json").read_text()) == result.scores
    # The posterior of the linear-Gaussian case predicts better than the prior.
    assert result.scores["posterior"]["y"]["crps"] < result.scores["prior"]["y"]["crps"]
    assert result.scores["posterior"]["y"]["spread"] < result.scores["prior"]["y"]["spread"]
    assert (out / "provenance.json").exists()


def test_a_deck_that_does_not_reproduce_the_history_fails(finished: Case) -> None:
    path, _ = _prediction(finished, perturb_history=True)

    with pytest.raises(WorkflowError, match=r"does not reproduce the history match.*y@t3"):
        prediction.predict(path)


def test_the_check_can_be_switched_off(finished: Case) -> None:
    path, _ = _prediction(finished, perturb_history=True, body="check_history = false")

    assert prediction.predict(path).forecasts["posterior"].values.shape == (1, 12, 30)


def test_resume_forecasts_only_the_missing_members(finished: Case) -> None:
    path, _ = _prediction(finished)
    first = prediction.predict(path)
    members = finished.directory / "runs" / "case" / "predictions" / "future" / "members"
    removed = [members / f"posterior_member_{i}.npz" for i in (2, 5, 11)]
    for file in removed:
        file.unlink()
    kept = {file: file.stat().st_mtime_ns for file in members.glob("*.npz")}

    second = prediction.predict(path)

    assert all(file.exists() for file in removed)
    assert {file: file.stat().st_mtime_ns for file in kept} == kept  # not forecast again
    np.testing.assert_array_equal(
        second.forecasts["posterior"].values, first.forecasts["posterior"].values
    )


def test_a_changed_configuration_is_refused(finished: Case) -> None:
    path, _ = _prediction(finished)
    prediction.predict(path)
    path.write_text(
        path.read_text().replace('name = "future"', 'name = "future"\ncheck_history = false')
    )

    with pytest.raises(ConfigError, match="another configuration"):
        prediction.predict(path)


def test_explicit_times_skip_the_pilot_and_nothing_to_check_fails(finished: Case) -> None:
    # Only future times: the history check would compare nothing, which is an error...
    path, _ = _prediction(finished, ensembles='["posterior"]')
    path.write_text(
        path.read_text().replace('vectors = ["y"]', 'vectors = ["y"]\ntimes = ["t9", "t11"]')
    )
    with pytest.raises(WorkflowError, match="found nothing to compare"):
        prediction.predict(path)

    # ... unless the check is switched off.
    path, g_pred = _prediction(
        finished, ensembles='["posterior"]', body="check_history = false", name="future_only"
    )
    path.write_text(
        path.read_text().replace('vectors = ["y"]', 'vectors = ["y"]\ntimes = ["t9", "t11"]')
    )
    result = prediction.predict(path)

    posterior, _ = _posterior_and_prior(finished)
    np.testing.assert_allclose(result.forecasts["posterior"].values[0], g_pred[[9, 11]] @ posterior)


def test_a_prediction_refuses_to_resume_after_the_run_changed(linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=20)
    composition.start(case.config, max_steps=2)  # stopped: its posterior is iteration 2
    path, _ = _prediction(case, ensembles='["posterior"]')
    prediction.predict(path)
    composition.resume(case.directory / "runs" / "case")  # the posterior is now iteration 4

    with pytest.raises(WorkflowError, match=r"posterior ensemble.*changed"):
        prediction.predict(path)


def test_failed_members_are_dropped_and_too_many_fail_the_prediction(finished: Case) -> None:
    path, _ = _prediction(finished, simulator_extra=", fail_members = [0, 4]")

    result = prediction.predict(path)

    forecast = result.forecasts["posterior"]
    assert forecast.failed_member_ids.tolist() == [0, 4]  # member 0 failed as the pilot too
    assert 0 not in forecast.member_ids
    assert forecast.values.shape[-1] == 28

    many = ", fail_members = [" + ", ".join(str(i) for i in range(3, 11)) + "]"
    path, _ = _prediction(finished, simulator_extra=many, name="failing")
    with pytest.raises(WorkflowError, match="only 22 of 30 members"):
        prediction.predict(path)


def test_exceedance_convention_relabels_the_percentiles(finished: Case) -> None:
    path, _ = _prediction(finished, body='percentile_convention = "exceedance"')

    result = prediction.predict(path)

    out = finished.directory / "runs" / "case" / "predictions" / "future"
    row = next(csv.DictReader((out / "statistics.csv").open()))
    assert {"P90", "P50", "P10"} <= set(row)
    assert "p10" not in row
    values = result.forecasts[row["ensemble"]].values[0, 0]
    assert float(row["P90"]) == pytest.approx(np.percentile(values, 10))  # exceeded 90 %


def test_configuration_errors_are_actionable(finished: Case, tmp_path: Path) -> None:
    path, _ = _prediction(finished, ensembles='["posterior", "iteration:99"]')
    with pytest.raises(ConfigError, match="iteration:99"):
        prediction.predict(path)
    path, _ = _prediction(finished, ensembles='["posterior", "best"]', name="bad")
    with pytest.raises(ConfigError, match="unknown ensembles"):
        prediction.predict(path)


def test_cli_predicts_and_reports(finished: Case) -> None:
    path, _ = _prediction(finished)
    out = finished.directory / "runs" / "case" / "predictions" / "future"

    predicted = CliRunner().invoke(app, ["predict", str(path)])
    reported = CliRunner().invoke(app, ["report", str(out), "--plots"])

    assert predicted.exit_code == 0, predicted.output
    assert "forecast on t11" in predicted.stdout
    assert "crps" in predicted.stdout
    assert "ensemble forecast" not in predicted.stdout  # progress goes to standard error
    # A prediction's forecasts have no iteration: none is shown. The pilot ran member 0.
    assert "posterior (pilot) forecast: 1/1 members ok" in predicted.stderr
    assert "posterior forecast: 29/29 members ok" in predicted.stderr
    assert reported.exit_code == 0, reported.output
    for figure in ("forecasts/y_posterior", "forecasts/y_comparison", "scores/crps"):
        assert (out / "report" / f"{figure}.png").exists()
        assert (out / "report" / f"{figure}.pdf").exists()


def test_predictions_of_a_cross_validated_run(well_case: Callable[..., Case]) -> None:
    """The posterior is the minimum-CV iterate; the check covers the CV wells too.

    The run also drops a member that fails (member 3), which the prediction never runs.
    """
    case = well_case(
        validation='[validation]\ncv = { wells = ["P7", "P8"] }\nstopping = { name = "select" }',
        simulator_extra="fail_members = [3]",
        ensemble_size=12,
    )
    composition.start(case.config)
    results = composition.load_results(case.directory / "runs" / "case")
    assert results.selected_iteration is not None
    forward = np.load(case.directory / "forward.npz")["matrix"]
    rng = np.random.default_rng(5)
    # History rows are the run's (well-major, t0-t3); t4 and t5 are new.
    rows, outputs = [], []
    for w in range(8):
        for t in range(6):
            rows.append(forward[4 * w + t] if t < 4 else rng.standard_normal(forward.shape[1]) / 5)
            outputs.append(f'["t{t}", "WOPR:P{w + 1}"]')
    np.savez(case.directory / "forward_pred.npz", matrix=np.array(rows))
    path = case.directory / "predict.toml"
    path.write_text(
        f"""
[prediction]
name = "future"
run_dir = "runs/case"
ensembles = ["posterior", "prior", "truth"]

[prediction.outputs]
vectors = ["WOPR:*"]

[simulator]
options = {{ forward_file = "forward_pred.npz", outputs = [{", ".join(outputs)}] }}
"""
    )

    result = prediction.predict(path)

    posterior = results.iterate(results.selected_iteration)
    forecast = result.forecasts["posterior"]
    assert 3 not in forecast.member_ids
    np.testing.assert_array_equal(forecast.member_ids, posterior.member_ids)
    g_pred = np.array(rows)
    expected = (g_pred @ posterior.state["x"]).reshape(8, 6, -1)  # vectors sorted P1-P8
    np.testing.assert_allclose(forecast.values, expected, rtol=1e-12)
    assert result.first_prediction_time == 4
    assert result.scores["posterior"]["WOPR:P1"]["coverage_p10_p90_nominal"] == pytest.approx(
        0.8 * 10 / 12
    )
    events = [
        json.loads(line)
        for line in (case.directory / "runs/case/predictions/future/events.jsonl").open()
    ]
    checked = {e["ensemble"]: e["compared"] for e in events if e.get("event") == "history_check"}
    # Every scalar datum, CV wells included: 8 wells x 4 times x 11 members.
    assert checked["posterior"] == 8 * 4 * 11


def test_time_rules_for_dates_and_labels() -> None:
    from apex.workflows.prediction import first_prediction_time, time_label

    # Any ISO form of a date meets the summary's YYYY-MM-DD.
    assert time_label("20240311") == time_label("2024-03-11") == "2024-03-11"
    assert time_label("t3") == "t3"
    dates = ["2024-01-01", "2024-06-01", "2025-01-01"]
    assert first_prediction_time(dates, ["20240115", "2024-06-01"]) == 2
    assert first_prediction_time(dates, ["2026-01-01"]) == 3  # no prediction period
    assert first_prediction_time(["t0", "t1", "t2"], ["t0", "t1"]) == 2  # by position
    assert first_prediction_time(dates, []) == 0


def test_non_finite_values_fail_the_member() -> None:
    from apex.core.errors import SimulationError
    from apex.core.simulation import MemberInput, SeriesRequest
    from apex.workflows.prediction import series_values

    class Broken:
        def run(self, member: MemberInput, keys: object) -> np.ndarray:
            return np.array([1.0, np.nan])

    with pytest.raises(SimulationError, match="non-finite"):
        series_values(Broken(), MemberInput(0, {}), SeriesRequest(("y",), ("t0", "t1")))


def test_a_multilevel_run_is_forecast_on_the_fine_model(linear_case: CaseFactory) -> None:
    """Every member runs the prediction's (fine) model; the history check skips
    the coarse level's members, whose stored forecasts came from the coarse operator."""
    case = linear_case(
        ensemble_size=12,
        method='name = "multilevel_esmda"\noptions = { inflation = [2, 2], levels = ['
        '{ members = 4, weight = 0.3, forward_file = "coarse.npz" }, '
        "{ members = 8, weight = 0.7 }] }",
    )
    np.savez(case.directory / "coarse.npz", matrix=1.2 * case.forward + 0.1)
    composition.start(case.config)
    path, g_pred = _prediction(case, ensembles='["posterior"]')

    result = prediction.predict(path)  # the history check passes on the fine members

    posterior, _ = _posterior_and_prior(case)
    np.testing.assert_allclose(
        result.forecasts["posterior"].values[0], g_pred @ posterior, rtol=1e-12
    )
    assert composition.coarse_member_ids(composition.load_config(case.config)) == set(range(4))


def test_a_prediction_that_failed_its_check_keeps_its_members(finished: Case) -> None:
    """Finished members are never deleted automatically (the review, 2026-10-20).

    A corrected rerun is refused with their count and the exact command to delete them.
    """
    path, _ = _prediction(finished, perturb_history=True)
    with pytest.raises(WorkflowError, match="does not reproduce the history match"):
        prediction.predict(path)
    out = finished.directory / "runs" / "case" / "predictions" / "future"
    members = sorted((out / "members").glob("*.npz"))
    assert members  # the check needs every member first
    assert json.loads((out / "state.json").read_text())["history_check"] == "failed"

    path.write_text(
        path.read_text().replace('name = "future"', 'name = "future"\ncheck_history = false')
    )
    with pytest.raises(
        ConfigError, match=rf"{len(members)} finished member.*history check failed.*rm -r"
    ):
        prediction.predict(path)
    assert sorted((out / "members").glob("*.npz")) == members

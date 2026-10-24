"""The rename of 2026-09-30: ``misfit`` became ``data_mismatch``, and the mean-square
``data_mismatch`` became ``normalized_data_mismatch``.

Old run directories are read in the new names (their bit-identical resume is checked by
``test_legacy_resume.py`` and the golden outputs by ``test_golden_outputs.py``); these tests
cover the pieces: records, score files, snapshots and configurations.
"""

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import pytest

from apex.core.errors import ConfigError
from apex.services import composition
from apex.services.api import summarise_run
from apex.storage.run_directory import RunDirectory, upgrade_record, upgrade_scores
from apex.storage.run_results import RunResults

GIES = (
    'name = "gies"\nstrategy = { name = "rlmmac", options = { energy = 1.0 } }\n'
    "schedule = { TOLERANCE = 1e-9 }\nmax_iterations = 6"
)


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path


CaseFactory = Callable[..., Case]


def test_an_old_record_is_read_in_the_new_names() -> None:
    old = {
        "iteration": 2, "misfit_mean": 10.0, "misfit_std": 1.0, "n_members": 5,
        "accepted": True, "misfit_by_type": {"y": 10.0},
        "metrics": {"data_mismatch": {"all": 1.2}, "cv_data_mismatch": {"all": 2.0},
                    "rmse": {"x": 0.3}},
    }  # fmt: skip

    new = upgrade_record(old)

    assert new["data_mismatch_mean"] == 10.0
    assert new["data_mismatch_by_type"] == {"y": 10.0}
    assert new["metrics"] == {
        "normalized_data_mismatch": {"all": 1.2},
        "cv_normalized_data_mismatch": {"all": 2.0},
        "rmse": {"x": 0.3},
        "data_mismatch": {"all": 10.0, "y": 10.0},  # the objective, as new runs record it
    }
    assert upgrade_record(new) == new  # a new record is left alone


def test_an_old_score_file_is_read_in_the_new_names() -> None:
    old = {
        "misfit": np.ones(3), "misfit_by_type": np.full((1, 3), 2.0),
        "datatypes": np.array(["y"]), "metric:data_mismatch:all": np.zeros(3),
        "metric:rmse:x": np.zeros(3),
    }  # fmt: skip

    new = upgrade_scores(old)

    assert set(new) == {
        "data_mismatch", "data_mismatch_by_type", "datatypes",
        "metric:normalized_data_mismatch:all", "metric:rmse:x",
        "metric:data_mismatch:all", "metric:data_mismatch:y",
    }  # fmt: skip
    np.testing.assert_array_equal(new["metric:data_mismatch:all"], np.ones(3))
    np.testing.assert_array_equal(new["metric:data_mismatch:y"], np.full(3, 2.0))
    np.testing.assert_array_equal(new["metric:normalized_data_mismatch:all"], np.zeros(3))


def test_the_objective_is_recorded_as_data_mismatch(linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=20)
    result = composition.start(case.config)
    run = RunDirectory(case.directory / "runs" / "case")

    last = result.history[-1]
    assert last.metrics is not None
    assert last.metrics["data_mismatch"]["all"] == pytest.approx(last.data_mismatch_mean)
    assert last.data_mismatch_by_type is not None
    assert last.metrics["data_mismatch"]["y"] == pytest.approx(last.data_mismatch_by_type["y"])
    scores = run.load_arrays(f"data_mismatch_{last.iteration}")
    np.testing.assert_array_equal(scores["metric:data_mismatch:all"], scores["data_mismatch"])
    # The mean square is about the objective divided by the number of data (C_d diagonal),
    # against the observed rather than the perturbed values: the same order of magnitude.
    assert 0 < last.metrics["normalized_data_mismatch"]["all"] < last.data_mismatch_mean


def test_a_run_directory_of_the_old_format_is_summarised(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    case = linear_case(ensemble_size=20)
    composition.start(case.config)
    new = case.directory / "runs" / "case"
    old = tmp_path / "old"
    shutil.copytree(new, old)
    # Write it back in the old names: history fields, score files, metric keys, snapshot.
    history = json.loads((old / "history.json").read_text())
    for record in history:
        record["misfit_mean"] = record.pop("data_mismatch_mean")
        record["misfit_std"] = record.pop("data_mismatch_std")
        record["misfit_by_type"] = record.pop("data_mismatch_by_type")
        metrics = record["metrics"]
        metrics.pop("data_mismatch")
        metrics["data_mismatch"] = metrics.pop("normalized_data_mismatch")
    (old / "history.json").write_text(json.dumps(history))
    for path in (old / "checkpoints").glob("data_mismatch_*.npz"):
        with np.load(path) as data:
            arrays = {k: data[k] for k in data.files if k != "metric:data_mismatch:all"}
        arrays = {k: v for k, v in arrays.items() if not k.startswith("metric:data_mismatch:")}
        arrays["misfit"] = arrays.pop("data_mismatch")
        arrays["misfit_by_type"] = arrays.pop("data_mismatch_by_type")
        arrays["metric:data_mismatch:all"] = arrays.pop("metric:normalized_data_mismatch:all")
        arrays["metric:data_mismatch:y"] = arrays.pop("metric:normalized_data_mismatch:y")
        np.savez(path.with_name(path.name.replace("data_mismatch_", "misfit_")), **arrays)
        path.unlink()
    (old / "snapshot_version").write_text("2\n")

    results = RunResults(
        old, composition.build_layout(composition.snapshot_config(RunDirectory(old)))
    )
    reference = RunResults(
        new, composition.build_layout(composition.snapshot_config(RunDirectory(new)))
    )

    assert [r.data_mismatch_mean for r in results.history] == [
        r.data_mismatch_mean for r in reference.history
    ]
    old_means = results.metric_means("normalized_data_mismatch")[1]
    assert old_means == reference.metric_means("normalized_data_mismatch")[1]
    assert results.metric_means("data_mismatch")[1] == reference.metric_means("data_mismatch")[1]
    assert results.metric_means("data_mismatch", "y")[1] == pytest.approx(
        reference.metric_means("data_mismatch", "y")[1], rel=1e-12
    )
    # Studies compare posterior metrics: the objective is found for the old run too.
    old_summary, new_summary = summarise_run(old), summarise_run(new)
    assert old_summary.posterior_metrics["data_mismatch"] == pytest.approx(
        new_summary.posterior_metrics["data_mismatch"], rel=1e-12
    )
    assert old_summary.posterior_metrics["normalized_data_mismatch"] == pytest.approx(
        new_summary.posterior_metrics["normalized_data_mismatch"], rel=1e-12
    )


def test_an_old_snapshot_keeps_its_mean_square_metric(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    config = {
        "run": {"name": "r", "seed": 1}, "state": {"variables": [{"name": "x", "size": 1}]},
        "prior": {"ensemble_size": 5}, "observations": {"file": str(tmp_path / "o.csv")},
        "simulator": {"name": "linear"}, "method": {"name": "esmda"},
        "case": {"kind": "real"}, "metrics": [{"name": "data_mismatch"}],
    }  # fmt: skip
    (run / "config.json").write_text(json.dumps(config))
    (run / "snapshot_version").write_text("2\n")

    snapshot = composition.snapshot_config(RunDirectory(run))

    assert snapshot.metrics is not None
    assert [m.name for m in snapshot.metrics] == ["normalized_data_mismatch"]


def test_the_old_tolerance_name_still_configures(linear_case: CaseFactory) -> None:
    old = linear_case(
        name="old", ensemble_size=20, method=GIES.replace("TOLERANCE", "misfit_tolerance")
    )
    new = linear_case(
        name="new", ensemble_size=20, method=GIES.replace("TOLERANCE", "data_mismatch_tolerance")
    )

    a, b = composition.start(old.config), composition.start(new.config)

    assert [r.data_mismatch_mean for r in a.history] == [r.data_mismatch_mean for r in b.history]
    both = GIES.replace("TOLERANCE", "misfit_tolerance = 1e-9, data_mismatch_tolerance")
    with pytest.raises(ConfigError, match="not both"):
        composition.start(linear_case(name="both", ensemble_size=20, method=both).config)


def test_data_mismatch_in_metrics_is_the_objective(linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=20, failures='[[metrics]]\nname = "data_mismatch"')

    result = composition.start(case.config)

    metrics = result.history[-1].metrics
    assert metrics is not None
    assert set(metrics) == {"data_mismatch", "model_change"}
    assert metrics["data_mismatch"]["all"] == pytest.approx(result.history[-1].data_mismatch_mean)


def test_data_mismatch_in_metrics_warns_in_plain_words(
    linear_case: CaseFactory, caplog: pytest.LogCaptureFixture
) -> None:
    """Release tester: the warning read like a changelog (a date and an ADR number)."""
    case = linear_case(ensemble_size=20, failures='[[metrics]]\nname = "data_mismatch"')

    with caplog.at_level("WARNING", logger="apex.services.composition"):
        composition.build_metrics(composition.load_config(case.config))

    (message,) = [r.getMessage() for r in caplog.records if "[[metrics]]" in r.getMessage()]
    assert "the objective the methods minimise" in message
    assert "normalized_data_mismatch" in message
    assert "ADR" not in message
    assert "2026" not in message


@pytest.mark.parametrize("label", ["data_mismatch", "cv_normalized_data_mismatch", "test_x"])
def test_reserved_metric_names_are_refused(linear_case: CaseFactory, label: str) -> None:
    section = f'[[metrics]]\nname = "rmse"\nlabel = "{label}"'
    case = linear_case(ensemble_size=10, failures=section)

    with pytest.raises(ConfigError, match="reserved"):
        composition.start(case.config)


def test_a_configuration_naming_data_mismatch_validates(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    from apex.services.api import ApexService

    case = linear_case(ensemble_size=10, failures='[[metrics]]\nname = "data_mismatch"')

    validation = ApexService(tmp_path / "service").validate(case.config)

    assert validation.valid, validation.errors

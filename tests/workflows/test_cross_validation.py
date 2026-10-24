"""Cross validation for iterative ensemble smoothers, end to end.

Independent reference (no PET equivalent): the linear case of ``well_case`` overfits by
construction. GIES without localization, with 10 members for 30 unknowns, keeps
lowering the mismatch of the assimilated wells P1-P6. Meanwhile the mean CV data
mismatch of the held-out wells P7-P8 reaches its minimum at iteration 2 and rises after
it, and so does the RMSE against the truth. This reproduces the V shape and the
"double validation" of Luo & Ranazzi (§3.1) on a problem whose truth is known.
"""

import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Protocol

import numpy as np
import pytest

from apex.core.errors import ConfigError
from apex.domain.results import HistoryMatchResult
from apex.services import composition
from apex.storage.run_directory import RunDirectory


class WellCase(Protocol):
    """What the ``well_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path


WellCaseFactory = Callable[..., WellCase]

CV = '[validation]\ncv = { wells = ["P7", "P8"] }'
SELECT = f'{CV}\nstopping = {{ name = "select" }}'
#: The CV minimum of the case (see the module docstring).
MINIMUM = 2


def _start(case: WellCase, max_steps: int | None = None) -> HistoryMatchResult:
    return composition.start(case.config, max_steps=max_steps)


def _run_dir(case: WellCase, name: str = "case") -> Path:
    return case.directory / "runs" / name


def _accepted(run_dir: Path) -> list[dict[str, object]]:
    return [r for r in RunDirectory(run_dir).history_dicts() if r["accepted"]]


def _cv_curve(run_dir: Path) -> list[float]:
    return [r["metrics"]["cv_normalized_data_mismatch"]["all"] for r in _accepted(run_dir)]  # type: ignore[index]


def test_select_keeps_the_minimum_cv_iterate(well_case: WellCaseFactory) -> None:
    case = well_case(validation=SELECT)
    result = _start(case)
    run_dir = _run_dir(case)

    curve = _cv_curve(run_dir)
    assert int(np.argmin(curve)) == MINIMUM
    assert len(curve) > MINIMUM + 3  # ran on: select never stops early
    assert result.selected_iteration == MINIMUM
    assert not result.stopped_by_cv
    saved = RunDirectory(run_dir).load_arrays(f"iter_{MINIMUM}")
    np.testing.assert_array_equal(result.posterior, saved["state"])
    posterior = RunDirectory(run_dir).load_arrays("posterior")
    np.testing.assert_array_equal(posterior["x"], saved["state"])
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["validation"]["selected_iteration"] == MINIMUM
    assert summary["validation"]["stopped_by_cv"] is False


def test_the_cv_minimum_is_the_rmse_minimum(well_case: WellCaseFactory) -> None:
    # "Double validation": the truth confirms the iterate that CV picked.
    case = well_case(validation=SELECT)
    _start(case)
    rmse = [r["metrics"]["rmse"]["all"] for r in _accepted(_run_dir(case))]  # type: ignore[index]
    assert int(np.argmin(rmse)) == MINIMUM
    assert rmse[-1] > rmse[MINIMUM]  # overfitting: later iterates are further from the truth


def test_patience_stops_after_the_minimum(well_case: WellCaseFactory) -> None:
    case = well_case(validation=CV)  # default rule: patience 2
    result = _start(case)
    accepted = _accepted(_run_dir(case))
    assert accepted[-1]["iteration"] == MINIMUM + 2
    assert result.selected_iteration == MINIMUM
    assert result.stopped_by_cv
    assert result.complete
    selected = RunDirectory(_run_dir(case)).load_arrays(f"iter_{MINIMUM}")
    np.testing.assert_array_equal(result.posterior, selected["state"])


def test_the_algorithm_sees_only_history_matching_data(well_case: WellCaseFactory) -> None:
    case = well_case(validation=CV)
    _start(case)
    run = RunDirectory(_run_dir(case))
    for record in run.history_dicts():
        assert set(record["data_mismatch_by_type"]) == {f"WOPR:P{w}" for w in range(1, 7)}
        assert set(record["metrics"]["cv_normalized_data_mismatch"]) == {
            "all",
            "WOPR:P7",
            "WOPR:P8",
        }
    saved = run.load_arrays("iter_1")
    assert saved["predicted"].shape == (24, saved["member_ids"].size)
    assert saved["cv_predicted"].shape == (8, saved["member_ids"].size)
    split = json.loads((run.checkpoints / "data_split.json").read_text())
    assert split["counts"] == {"history_matching": 24, "cv": 8, "test": 0, "unused": 0}
    assert split["ratio_history_matching_to_cv"] == 3.0
    perturbed = run.load_arrays("perturbed_observations")["perturbed_observations"]
    assert perturbed.shape[0] == 24


def test_cv_mismatch_is_the_mean_square_over_the_cv_data(well_case: WellCaseFactory) -> None:
    case = well_case(validation=CV)
    _start(case)
    run = RunDirectory(_run_dir(case))
    rows = [line.split(",") for line in (case.directory / "observations.csv").read_text().split()]
    cv = [(float(v), float(s2)) for _, dt, v, s2 in rows[1:] if dt in ("WOPR:P7", "WOPR:P8")]
    observed, variances = np.array(cv).T
    saved = run.load_arrays("iter_1")
    expected = (((observed[:, None] - saved["cv_predicted"]) ** 2) / variances[:, None]).mean(0)
    np.testing.assert_allclose(
        run.load_arrays("data_mismatch_1")["metric:cv_normalized_data_mismatch:all"],
        expected,
        rtol=1e-12,
    )


@pytest.mark.parametrize("stops", [[3], [1, 1, 1]])
def test_resume_inside_the_patience_window_is_bit_identical(
    well_case: WellCaseFactory, stops: list[int]
) -> None:
    whole_case = well_case(validation=CV, name="whole")
    whole = _start(whole_case)
    case = well_case(validation=CV, name="parts")
    _start(case, max_steps=stops[0])
    for stop in stops[1:]:
        composition.resume(_run_dir(case, "parts"), max_steps=stop)
    resumed = composition.resume(_run_dir(case, "parts"))

    np.testing.assert_array_equal(resumed.posterior, whole.posterior)
    assert resumed.selected_iteration == whole.selected_iteration == MINIMUM
    assert resumed.stopped_by_cv
    assert (
        RunDirectory(_run_dir(case, "parts")).history_dicts()
        == RunDirectory(_run_dir(whole_case, "whole")).history_dicts()
    )


def test_resuming_a_run_stopped_by_cv_forecasts_nothing(well_case: WellCaseFactory) -> None:
    case = well_case(validation=CV)
    first = _start(case)
    history = RunDirectory(_run_dir(case)).history_dicts()
    again = composition.resume(_run_dir(case))
    assert RunDirectory(_run_dir(case)).history_dicts() == history
    np.testing.assert_array_equal(again.posterior, first.posterior)
    assert again.stopped_by_cv
    assert again.selected_iteration == MINIMUM


def test_without_cv_the_method_runs_unchanged(well_case: WellCaseFactory) -> None:
    plain = well_case(name="plain")
    no_cv = well_case(name="no_cv", validation='[validation]\ntest = { wells = ["P8"] }')
    result = _start(plain)
    assert result.selected_iteration is None
    assert not (_run_dir(plain, "plain") / "checkpoints" / "data_split.json").exists()
    tested = _start(no_cv)
    assert tested.selected_iteration is None
    assert not tested.stopped_by_cv
    history = RunDirectory(_run_dir(no_cv, "no_cv")).history_dicts()
    assert "test_normalized_data_mismatch" in history[0]["metrics"]
    assert "cv_normalized_data_mismatch" not in history[0]["metrics"]
    # Test data only ever get scored: the run goes on to its own convergence test.
    assert history[-1]["iteration"] > MINIMUM + 2


def test_test_data_are_scored_but_never_decide(well_case: WellCaseFactory) -> None:
    with_test = well_case(
        name="with_test", validation=f'{CV}\ntest = {{ datatypes = ["WOPR:P6"] }}'
    )
    run_dir = _run_dir(with_test, "with_test")
    result = _start(with_test)
    history = RunDirectory(run_dir).history_dicts()
    assert set(history[0]["metrics"]["test_normalized_data_mismatch"]) == {"all", "WOPR:P6"}
    assert result.selected_iteration == int(np.argmin(_cv_curve(run_dir)))


def test_es_mda_refuses_cross_validation(well_case: WellCaseFactory) -> None:
    esmda = (
        'name = "esmda"\noptions = { inflation = [4, 4, 4, 4] }\n'
        'strategy = { name = "approx", options = { energy = 1.0 } }'
    )
    with pytest.raises(ConfigError, match="fixed schedule"):
        _start(well_case(method=esmda, validation=CV))
    # Test data alone are fine with any method.
    result = _start(
        well_case(
            name="esmda_test", method=esmda, validation='[validation]\ntest = { wells = ["P8"] }'
        )
    )
    assert result.complete


@pytest.mark.parametrize(
    ("validation", "message"),
    [
        ('[validation]\ncv = { wells = ["P9"] }', "matches no data"),
        ('[validation]\ncv = { wells = ["P7"] }\ntest = { datatypes = ["WOPR:P7"] }', "overlap"),
        ('[validation]\nstopping = { name = "select" }', "needs a cv"),
        ('[validation]\ncv = { wells = ["P7"] }\nstopping = { name = "nope" }', "nope"),
        (
            '[validation]\ncv = { wells = ["P7"] }\nstopping = { name = "patience", '
            "options = { patience = 0 } }",
            "patience",
        ),
    ],
)
def test_invalid_validation_sections_are_refused(
    well_case: WellCaseFactory, validation: str, message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        _start(well_case(validation=validation))


def _compression_cases() -> ModuleType:
    # The field-data case builders of test_data_compression_e2e.py (tests/ is not a package).
    path = Path(__file__).with_name("test_data_compression_e2e.py")
    spec = importlib.util.spec_from_file_location("_compression_cases_cv", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compressed_field_data_stay_in_history_matching(tmp_path: Path) -> None:
    compression = _compression_cases()
    validation = '[validation]\ncv = { datatypes = ["y"], times = ["t3"] }'
    config = compression._write_case(
        tmp_path / "field",
        transform=f"{compression._wavelet_section()}\n\n{validation}",
        method=compression.GIES,
    )
    result = composition.start(config)
    run = RunDirectory(tmp_path / "field" / "runs" / "case")
    history = run.history_dicts()
    assert set(history[0]["data_mismatch_by_type"]) == {"y", "wavelet(avo)"}
    assert set(history[0]["metrics"]["cv_normalized_data_mismatch"]) == {"all", "y"}
    split = json.loads((run.checkpoints / "data_split.json").read_text())
    assert split["keys"]["cv"] == ["y@t3"]
    assert "avo@t1" in split["keys"]["history_matching"]
    assert result.selected_iteration is not None


@pytest.mark.parametrize(
    "selection",
    ['cv = { datatypes = ["avo"] }', 'test = { wells = ["avo"] }', 'cv = { keywords = ["avo"] }'],
)
def test_field_data_cannot_be_held_out(tmp_path: Path, selection: str) -> None:
    compression = _compression_cases()
    config = compression._write_case(
        tmp_path / "field",
        transform=f"{compression._wavelet_section()}\n\n[validation]\n{selection}",
        method=compression.GIES,
    )
    with pytest.raises(ConfigError, match="field data are used for history matching only"):
        composition.start(config)


def test_a_transform_that_changes_held_out_data_is_refused() -> None:
    # Held-out data must reach the analysis space as observed.
    from apex.domain.data_split import DataSplit
    from apex.domain.observations import DataKey, ObservationSet
    from apex.workflows.history_matching import _roles

    keys = [DataKey("t0", "WOPR:P1"), DataKey("t0", "WOPR:P2")]
    raw = ObservationSet(keys, [1.0, 2.0], [0.1, 0.1])
    split = DataSplit(history_matching=(keys[0],), cv=(keys[1],))
    rows, held = _roles(raw, raw, split)
    assert rows is not None
    assert rows.tolist() == [0]
    assert held["cv"][0].tolist() == [1]
    scaled = ObservationSet(keys, [1.0, 4.0], [0.1, 0.1])
    with pytest.raises(ValueError, match="changes cv datum"):
        _roles(raw, scaled, split)
    with pytest.raises(ValueError, match="removes cv datum"):
        _roles(raw, ObservationSet(keys[:1], [1.0], [0.1]), split)


def test_the_data_split_command_ranks_wells_and_writes_its_report(
    well_case: WellCaseFactory, tmp_path: Path
) -> None:
    from typer.testing import CliRunner

    from apex.cli.main import app

    case = well_case(validation=CV)
    _start(case)  # a prior forecast with held-out wells, for the redundancy
    output = tmp_path / "split"
    outcome = CliRunner().invoke(
        app,
        [
            "data-split",
            str(case.config),
            "--prior-run",
            str(_run_dir(case)),
            "--max-cv",
            "3",
            "--output",
            str(output),
        ],
    )
    assert outcome.exit_code == 0, outcome.output
    assert "[validation]" in outcome.output
    names = {p.name for p in output.iterdir()}
    assert {"groups.csv", "correlation.csv", "scores.csv", "candidates.toml"} <= names
    assert {"correlation.png", "scores.png", "redundancy.png", "redundancy.pdf"} <= names
    scores = (output / "scores.csv").read_text().splitlines()
    assert len(scores) == 1 + 8  # header and one row per well, every well has redundancy
    assert all(row.split(",")[3] for row in scores[1:])
    assert (output / "candidates.toml").read_text().count("# [validation]") == 3


def test_reports_of_cv_runs_plot_every_role_and_the_selected_posterior(
    well_case: WellCaseFactory,
) -> None:
    from typer.testing import CliRunner

    from apex.cli.main import app

    case = well_case(validation=f'{CV}\ntest = {{ wells = ["P6"] }}')
    _start(case)
    outcome = CliRunner().invoke(app, ["report", str(_run_dir(case)), "--plots"])
    assert outcome.exit_code == 0, outcome.output
    assert f"final = {MINIMUM}, selected by cross validation" in outcome.output
    data = {p.stem for p in (_run_dir(case) / "report" / "data").glob("*.png")}
    # Assimilated, CV and test wells all get their data figures.
    assert {f"P{w}_WOPR_final" for w in range(1, 9)} <= data
    metrics = {p.stem for p in (_run_dir(case) / "report" / "metrics").glob("*.png")}
    assert {
        "metric_cv_normalized_data_mismatch_all",
        "metric_test_normalized_data_mismatch_all",
    } <= metrics


#: Seed 12 with members failing above 2.8: two members fail at iteration 3 (9 -> 7). A
#: mean over each iterate's own members would select iteration 3, only because the two
#: failed members had large CV mismatches; on the survivors, iteration 2 stays the minimum.
FAILING = {
    "seed": 12,
    "simulator_extra": "fail_above = 2.8",
    "validation": f"{SELECT}\n\n[failures]\nmin_success_fraction = 0.5",
}


def test_cv_compares_iterates_on_the_same_members(well_case: WellCaseFactory) -> None:
    # Regression (numerics review): the rule compared means over different member sets.
    case = well_case(**FAILING)
    result = _start(case)
    run = RunDirectory(_run_dir(case))
    accepted = _accepted(_run_dir(case))
    assert [r["n_members"] for r in accepted][2:4] == [9, 7]
    survivors = run.load_arrays(f"data_mismatch_{accepted[-1]['iteration']}")["member_ids"]
    on_survivors = []
    for record in accepted:
        saved = run.load_arrays(f"data_mismatch_{record['iteration']}")
        columns = [saved["member_ids"].tolist().index(m) for m in survivors]
        on_survivors.append(saved["metric:cv_normalized_data_mismatch:all"][columns].mean())
    own_members = _cv_curve(_run_dir(case))
    assert int(np.argmin(own_members)) == 3  # what the old rule would have selected
    assert result.selected_iteration == int(np.argmin(on_survivors)) == 2
    # The posterior is the selected ensemble, with the members it had then.
    assert result.member_ids.size == 9


def test_cv_with_failing_members_resumes_bit_identically(well_case: WellCaseFactory) -> None:
    whole_case = well_case(name="whole", **FAILING)
    whole = _start(whole_case)
    case = well_case(name="parts", **FAILING)
    _start(case, max_steps=2)
    composition.resume(_run_dir(case, "parts"), max_steps=2)
    resumed = composition.resume(_run_dir(case, "parts"))
    np.testing.assert_array_equal(resumed.posterior, whole.posterior)
    assert resumed.selected_iteration == whole.selected_iteration
    parts_history = RunDirectory(_run_dir(case, "parts")).history_dicts()
    assert parts_history == RunDirectory(_run_dir(whole_case, "whole")).history_dicts()

"""Whole time series from simulators, for predictions."""

import datetime as dt
import stat
from pathlib import Path

import numpy as np
import pytest

from apex.core.errors import SimulationError
from apex.core.simulation import MemberInput, SeriesRequest, match_vectors
from apex.simulation.analytic import LinearModel
from apex.simulation.deck import read_series
from apex.simulation.eclipse import EclipseSimulator


@pytest.fixture
def summary(tmp_path: Path) -> Path:
    """``MODEL`` with WOPR:P1, WOPR:P2 and FOPT at three reports (Feb, Mar, Apr 2024)."""
    from resdata.summary import Summary

    directory = tmp_path / "summary"
    directory.mkdir()
    writer = Summary.writer(str(directory / "MODEL"), dt.datetime(2024, 1, 1), 1, 1, 1)
    writer.add_variable("WOPR", wgname="P1")
    writer.add_variable("WOPR", wgname="P2")
    writer.add_variable("FOPT")
    for step, days in ((1, 31), (2, 60), (3, 91)):
        t = writer.add_t_step(step, days)
        t["WOPR:P1"], t["WOPR:P2"], t["FOPT"] = 100.0 * step, 10.0 * step, 1000.0 * step
    writer.fwrite()
    return directory / "MODEL"


def test_patterns_resolve_sorted_and_case_sensitive() -> None:
    available = ["WOPR:P2", "FOPT", "WOPR:P1", "WWPR:P1"]

    assert match_vectors(["WOPR:*", "FOPT"], available) == ("FOPT", "WOPR:P1", "WOPR:P2")
    with pytest.raises(ValueError, match="'wopr:\\*'"):
        match_vectors(["wopr:*"], available)


def test_series_are_read_at_every_report_date(summary: Path) -> None:
    series = read_series(summary, SeriesRequest(("WOPR:*", "FOPT")))

    assert series.times == ("2024-02-01", "2024-03-01", "2024-04-01")
    assert series.vectors == ("FOPT", "WOPR:P1", "WOPR:P2")
    assert series.values.dtype == np.float64
    np.testing.assert_array_equal(
        series.values, [[1000, 2000, 3000], [100, 200, 300], [10, 20, 30]]
    )


def test_series_at_requested_dates_and_errors(summary: Path) -> None:
    series = read_series(summary, SeriesRequest(("FOPT",), ("2024-04-01", "2024-02-01")))

    np.testing.assert_array_equal(series.values, [[3000, 1000]])
    with pytest.raises(SimulationError, match="2024-02-15: not report dates"):
        read_series(summary, SeriesRequest(("FOPT",), ("2024-02-15",)))
    with pytest.raises(SimulationError, match="no vector matches 'WWCT:\\*'"):
        read_series(summary, SeriesRequest(("WWCT:*",)))


def test_eclipse_returns_series_from_its_run(tmp_path: Path, summary: Path) -> None:
    template = tmp_path / "deck.mako"
    template.write_text("% for v in permx:\n${v}\n% endfor\n")
    eclrun = tmp_path / "fake_eclrun"
    eclrun.write_text(
        "#!/bin/sh\nprintf ' Error summary\\n Errors                 0\\n' > MODEL.ECLEND\n"
        f"touch MODEL.PRT\ncp {summary}.SMSPEC {summary}.UNSMRY .\n"
    )
    eclrun.chmod(eclrun.stat().st_mode | stat.S_IEXEC)
    work_dir = tmp_path / "member_0"
    work_dir.mkdir()

    series = EclipseSimulator(template, executable=str(eclrun), keep_files=False).run_series(
        MemberInput(0, {"permx": np.ones(2)}, work_dir), SeriesRequest(("WOPR:P1",))
    )

    np.testing.assert_array_equal(series.values, [[100, 200, 300]])
    assert (work_dir / "MODEL.SMSPEC").exists()  # pruning keeps the summary


def test_linear_model_returns_its_outputs_as_series() -> None:
    model = LinearModel(
        outputs=[["t0", "a"], ["t1", "a"], ["t0", "b"], ["t1", "b"]],
        variables=["x"],
        matrix=[[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 3.0]],
    )
    member = MemberInput(0, {"x": np.array([1.0, 2.0])})

    series = model.run_series(member, SeriesRequest(("*",)))

    assert (series.times, series.vectors) == (("t0", "t1"), ("a", "b"))
    np.testing.assert_array_equal(series.values, [[1.0, 2.0], [2.0, 6.0]])
    np.testing.assert_array_equal(
        model.run_series(member, SeriesRequest(("b",), ("t1",))).values, [[6.0]]
    )
    with pytest.raises(SimulationError, match="no output at t9"):
        model.run_series(member, SeriesRequest(("a",), ("t9",)))


def test_report_steps_on_one_day_are_refused(tmp_path: Path) -> None:
    from resdata.summary import Summary

    writer = Summary.writer(str(tmp_path / "MODEL"), dt.datetime(2024, 1, 1), 1, 1, 1)
    writer.add_variable("FOPT")
    for step, days in ((1, 31.0), (2, 31.5)):
        writer.add_t_step(step, days)["FOPT"] = float(step)
    writer.fwrite()

    with pytest.raises(SimulationError, match="several report steps on 2024-02-01"):
        read_series(tmp_path / "MODEL", SeriesRequest(("FOPT",)))


def test_values_are_the_report_steps_not_interpolated_at_midnight(tmp_path: Path) -> None:
    """Regression (2026-10-11): report steps that end within a day were read wrongly.

    resdata's ``numpy_vector(report_only=True)`` interpolates at the report *dates*, i.e.
    at midnight, so a report step ending at 12:00 got a value between its last ministep and
    the one before (the Brugge schedule's report times end at e.g. 23:45:36).
    """
    from resdata.summary import Summary

    from apex.domain.observations import DataKey
    from apex.simulation.deck import read_summary

    writer = Summary.writer(str(tmp_path / "MODEL"), dt.datetime(2024, 1, 1), 1, 1, 1)
    writer.add_variable("WBHP", wgname="P1")
    # Report 1 ends at day 1.5 (2024-01-02 12:00) after a ministep at day 1.0; report 2
    # ends at day 3.5.
    for step, days, value in ((1, 1.0, 100.0), (1, 1.5, 200.0), (2, 2.5, 300.0), (2, 3.5, 400.0)):
        writer.add_t_step(step, days)["WBHP:P1"] = value
    writer.fwrite()

    keys = [DataKey("2024-01-02", "WBHP:P1"), DataKey("2024-01-04", "WBHP:P1")]
    np.testing.assert_array_equal(read_summary(tmp_path / "MODEL", keys), [200.0, 400.0])
    series = read_series(tmp_path / "MODEL", SeriesRequest(("WBHP:P1",)))
    np.testing.assert_array_equal(series.values, [[200.0, 400.0]])

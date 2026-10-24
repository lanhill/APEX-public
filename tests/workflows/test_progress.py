"""Progress reporting of whole runs, on generated linear-Gaussian cases."""

import io
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pytest
from rich.console import Console
from typer.testing import CliRunner

from apex.cli.main import app
from apex.cli.progress import LineProgress, RichProgress, describe_record
from apex.core.progress import IterationProgress
from apex.domain.results import IterationRecord
from apex.services import composition


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path


CaseFactory = Callable[..., Case]

GIES_RETRIES = (
    'name = "gies"\noptions = { max_trials_per_iteration = 3 }\n'
    'strategy = { name = "rlmmac", options = { energy = 1.0 } }\n'
    "schedule = { data_mismatch_tolerance = 1e-9 }\nmax_iterations = 4"
)


@dataclass
class Recorder:
    """A reporter that records every call, in order, and each kind of call by type."""

    calls: list[str] = field(default_factory=list)
    started: list[tuple[str, IterationProgress | None]] = field(default_factory=list)
    positions: list[IterationProgress] = field(default_factory=list)
    forecasts: list[tuple[int | None, str, int]] = field(default_factory=list)
    members: list[tuple[int, bool]] = field(default_factory=list)
    finished: list[tuple[int, int]] = field(default_factory=list)
    records: list[IterationRecord] = field(default_factory=list)
    status: list[str] = field(default_factory=list)

    def run_started(self, *, algorithm: str, progress: IterationProgress | None) -> None:
        self.calls.append("run_started")
        self.started.append((algorithm, progress))

    def iteration_progress(self, progress: IterationProgress) -> None:
        self.calls.append("iteration_progress")
        self.positions.append(progress)

    def forecast_started(self, *, iteration: int | None, label: str, n_members: int) -> None:
        self.calls.append("forecast_started")
        self.forecasts.append((iteration, label, n_members))

    def member_finished(self, *, member_id: int, ok: bool) -> None:
        self.calls.append("member_finished")
        self.members.append((member_id, ok))

    def forecast_finished(self, *, n_ok: int, n_failed: int, seconds: float) -> None:
        self.calls.append("forecast_finished")
        self.finished.append((n_ok, n_failed))

    def record_scored(self, record: IterationRecord) -> None:
        self.calls.append("record_scored")
        self.records.append(record)

    def run_finished(self, *, status: str) -> None:
        self.calls.append("run_finished")
        self.status.append(status)


def test_esmda_reports_every_forecast_member_and_step(linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=20)
    recorder = Recorder()

    result = composition.start(case.config, progress=recorder)

    assert recorder.calls[0] == "run_started"
    assert recorder.started == [("ESMDA", IterationProgress(0, 4, "prior"))]
    assert recorder.calls[-1] == "run_finished"
    assert recorder.status == ["complete"]
    # The prior and four steps: 20 members each, every one reported once per forecast.
    assert recorder.forecasts == [(k, "ensemble", 20) for k in range(5)]
    assert recorder.finished == [(20, 0)] * 5
    members = recorder.members
    assert len(members) == 100
    assert sorted(members[:20]) == [(i, True) for i in range(20)]
    positions = recorder.positions
    assert [p.completed for p in positions] == [0, 1, 2, 3, 4]
    assert positions[-1] == IterationProgress(4, 4, "done")
    assert recorder.records == list(result.history)


PROCESS = 'name = "process"\noptions = { workers = 2 }'


def test_progress_does_not_change_results(linear_case: CaseFactory) -> None:
    """History and every checkpoint are bit-identical with and without a reporter."""
    kwargs = {"method": GIES_RETRIES, "ensemble_size": 10, "execution": PROCESS}
    plain = linear_case(name="plain", **kwargs)
    reported = linear_case(name="reported", **kwargs)
    composition.start(plain.config)
    composition.start(reported.config, progress=Recorder())

    runs = [case.directory / "runs" / case.directory.name for case in (plain, reported)]
    assert (runs[0] / "history.json").read_text() == (runs[1] / "history.json").read_text()
    saved = sorted(p.name for p in (runs[0] / "checkpoints").glob("*.npz"))
    assert saved == sorted(p.name for p in (runs[1] / "checkpoints").glob("*.npz"))
    for name in saved:
        with (
            np.load(runs[0] / "checkpoints" / name) as a,
            np.load(runs[1] / "checkpoints" / name) as b,
        ):
            assert a.files == b.files
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key], err_msg=f"{name}:{key}")


def test_gies_without_retries_counts_trial_iterates(linear_case: CaseFactory) -> None:
    method = GIES_RETRIES.replace("options = { max_trials_per_iteration = 3 }\n", "")
    recorder = Recorder()

    composition.start(linear_case(method=method, ensemble_size=10).config, progress=recorder)

    running = [p for p in recorder.positions if p.current not in ("prior", "done")]
    assert all(p.current == f"iteration {p.completed + 1}" for p in running)
    assert recorder.positions[-1] == IterationProgress(4, 4, "done")
    # RLM-MAC also forecasts the mean model: the prior's, then each accepted iterate's.
    labels = [label for _, label, _ in recorder.forecasts]
    assert labels[:2] == ["ensemble", "mean model"]
    assert labels.count("mean model") == 1 + sum(r.accepted for r in recorder.records[1:])


class _Broken(Recorder):
    def member_finished(self, *, member_id: int, ok: bool) -> None:
        raise BrokenPipeError("terminal closed")


def test_a_failing_display_is_dropped_and_the_run_completes(
    linear_case: CaseFactory, caplog: pytest.LogCaptureFixture
) -> None:
    broken = _Broken()

    result = composition.start(linear_case(ensemble_size=10).config, progress=broken)

    assert result.complete
    assert broken.forecasts == [(0, "ensemble", 10)]  # dropped at the first failure
    assert any(getattr(r, "event", None) == "progress_failed" for r in caplog.records)


def test_gies_reports_trials_within_its_outer_iteration_budget(
    linear_case: CaseFactory,
) -> None:
    recorder = Recorder()

    composition.start(linear_case(method=GIES_RETRIES, ensemble_size=10).config, progress=recorder)

    positions = recorder.positions
    assert all(p.total == 4 and p.completed <= 4 for p in positions)
    assert [p.completed for p in positions] == sorted(p.completed for p in positions)
    running = [p.current for p in positions if p.current not in ("prior", "done")]
    assert running
    assert all("trial" in text and "of at most 3" in text for text in running)
    assert positions[-1].current == "done"


def test_resume_reports_from_the_checkpointed_step(linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=10)
    first = composition.start(case.config, max_steps=2)
    assert not first.complete
    recorder = Recorder()

    composition.resume(case.directory / "runs" / "case", progress=recorder)

    # Steps 1 and 2 are done; step 3 is the pending forecast.
    assert recorder.started == [("ESMDA", IterationProgress(2, 4, "step 3"))]
    assert [c[0] for c in recorder.forecasts] == [3, 4]


def test_cli_prints_progress_lines_with_data_mismatch_mean_and_std(
    linear_case: CaseFactory,
) -> None:
    case = linear_case(ensemble_size=10)

    result = CliRunner().invoke(app, ["run", str(case.config)])
    quiet = CliRunner().invoke(app, ["run", str(linear_case(name="quiet").config), "--no-progress"])

    assert result.exit_code == 0, result.output
    # Not a terminal: timestamped lines on standard error, the tables on standard output.
    lines = result.stderr.splitlines()
    assert any("ensemble forecast 0: 10/10 members ok" in line for line in lines)
    assert any("ESMDA iter 4: data mismatch" in line and " ± " in line for line in lines)
    assert lines[-1].endswith("ESMDA done (4 of ≤4 iterations): complete")
    assert "data_mismatch" not in quiet.stderr


def test_describe_record_gives_data_mismatch_mean_plus_minus_std() -> None:
    record = IterationRecord(
        3, 12100.0, 3210.0, 99, accepted=False, damping=2.0,
        metrics={"cv_normalized_data_mismatch": {"all": 17518.5}},
    )  # fmt: skip

    assert describe_record(record) == (
        "data mismatch 1.21e+04 ± 3.21e+03 (rejected), 99 members; CV mismatch 1.752e+04"
    )
    assert (
        describe_record(IterationRecord(0, 1.5, 0.25, 10)) == "data mismatch 1.5 ± 0.25, 10 members"
    )


@pytest.mark.parametrize("total", [4, None])
def test_displays_render_a_run(total: int | None) -> None:
    """Both displays accept a whole sequence of calls, with or without a budget."""
    out = io.StringIO()
    record = IterationRecord(1, 2.0, 0.5, 3)
    rich = RichProgress(Console(file=out, force_terminal=True, width=150))
    with rich:
        for display in (rich, LineProgress(io.StringIO())):
            display.run_started(algorithm="X", progress=IterationProgress(0, total, "prior"))
            display.forecast_started(iteration=1, label="ensemble", n_members=3)
            for member in range(3):
                display.member_finished(member_id=member, ok=member != 1)
            display.forecast_finished(n_ok=2, n_failed=1, seconds=65.0)
            display.record_scored(record)
            display.iteration_progress(IterationProgress(1, total, "done"))
            display.run_finished(status="converged")

    assert "X converged" in out.getvalue()
    assert "1 failed" in out.getvalue()


def test_line_progress_reports_failed_members_and_duration() -> None:
    out = io.StringIO()
    display = LineProgress(out)
    display.run_started(algorithm="GIES", progress=IterationProgress(6, 20, "iteration 7"))
    display.forecast_started(iteration=9, label="ensemble", n_members=100)
    display.forecast_finished(n_ok=99, n_failed=1, seconds=1450.0)

    assert (
        out.getvalue()
        .splitlines()[-1]
        .endswith(
            "GIES iteration 7 (6 of ≤20 iterations done): ensemble forecast 9: 99/100 members ok "
            "(1 failed) in 24m10s"
        )
    )


def test_run_directory_events_record_each_forecast(linear_case: CaseFactory) -> None:
    import json

    case = linear_case(ensemble_size=10)
    composition.start(case.config)

    events = [
        json.loads(line) for line in (case.directory / "runs" / "case" / "events.jsonl").open()
    ]
    done = [e for e in events if e.get("event") == "forecast_done"]
    assert [(e["iteration"], e["label"], e["n_ok"], e["n_failed"]) for e in done] == [
        (k, "ensemble", 10, 0) for k in range(5)
    ]
    assert all(e["seconds"] >= 0 for e in done)

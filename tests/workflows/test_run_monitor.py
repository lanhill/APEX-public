"""The run monitor agent, on linear cases and scripted models (no live model).

Independent references: the data mismatch trajectory is the run's history, the spread ratio is
recomputed here from the saved ensembles, and failed members and rejected trials are
counted from the history.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pytest
from typer.testing import CliRunner

from apex.agents.monitor import (
    READ_ONLY_TOOLS,
    MonitorReport,
    RunMonitor,
    RunObservation,
    observe,
    reconcile,
    rules,
)
from apex.agents.providers.stub import StubProvider
from apex.cli.main import app
from apex.services import composition
from apex.services.api import ApexService
from apex.services.settings import MonitorSection
from apex.storage.run_directory import RunDirectory

TIMEOUT = 180.0
GIES = (
    'name = "gies"\nstrategy = { name = "rlmmac", options = { energy = 1.0 } }\n'
    "schedule = { data_mismatch_tolerance = 1e-9 }\nmax_iterations = 12"
)
#: With the tanh model GIES rejects some trials; members whose state exceeds 2 fail.
FAILING = 'nonlinearity = "tanh"\nfail_above = 2.0'


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path


CaseFactory = Callable[..., Case]


def _observation(**fields: Any) -> RunObservation:  # noqa: ANN401
    """A running GIES run with nothing wrong, changed by ``fields``."""
    base: dict[str, Any] = {
        "run_dir": "/runs/x", "name": "x", "method": "gies", "iterative": True,
        "state": "running", "job_id": "j1", "forecasts": 5, "accepted": 5, "rejected": 0,
        "rejections_in_a_row": 0, "members_prior": 50, "members_now": 50,
        "failed_fraction": 0.0, "data_mismatch_iterations": [0, 1, 2, 3, 4],
        "data_mismatch_means": [1000.0, 400.0, 200.0, 120.0, 90.0], "last_relative_change": 0.25,
        "normalized_data_mismatch": 1.4, "spread_ratio": {"permx": 0.4}, "damping": 0.5,
        "initial_damping": 1.0,
    }  # fmt: skip
    return RunObservation(**{**base, **fields})


def _codes(report: MonitorReport) -> set[str]:
    return {f.code for f in report.findings}


# ---------------------------------------------------------------------- rules


def test_a_healthy_running_run_continues() -> None:
    report = rules(_observation())

    assert (report.status, report.recommendation, report.findings) == ("healthy", "continue", [])


@pytest.mark.parametrize(
    ("fields", "code", "recommendation"),
    [
        ({"spread_ratio": {"permx": 0.04, "poro": 0.3}}, "ensemble_collapse", "investigate"),
        ({"members_now": 39, "failed_fraction": 0.22}, "member_failures", "investigate"),
        ({"rejections_in_a_row": 5, "damping": 32.0}, "damping_runaway", "stop"),
        ({"data_mismatch_means": [100.0, 99.5, 99.2, 99.0]}, "stagnation", "stop"),
        ({"normalized_data_mismatch": 0.4}, "overfitting", "stop"),
        (
            {
                "cv_mismatch": [5.0, 3.0, 3.2, 3.4],
                "cv_minimum_iteration": 1,
                "accepted_since_cv_minimum": 2,
            },
            "cv_past_minimum",
            "stop",
        ),
    ],
)
def test_each_finding_at_its_threshold(
    fields: dict[str, Any], code: str, recommendation: str
) -> None:
    report = rules(_observation(**fields))

    assert code in _codes(report)
    assert report.recommendation == recommendation


def test_just_inside_the_thresholds_nothing_is_found() -> None:
    t = MonitorSection()
    report = rules(
        _observation(
            spread_ratio={"permx": t.collapse_spread_ratio},
            members_now=40,
            failed_fraction=t.max_failed_fraction,
            rejections_in_a_row=t.max_consecutive_rejections - 1,
            normalized_data_mismatch=t.overfit_mismatch,
        )
    )

    assert report.findings == []


def test_a_fixed_schedule_is_never_told_to_stop() -> None:
    # ES-MDA's theory needs every step: the same findings only ask for a look.
    report = rules(_observation(method="esmda", iterative=False, normalized_data_mismatch=0.3))

    assert _codes(report) == {"overfitting"}
    assert report.recommendation == "investigate"


def test_with_cv_overfitting_is_left_to_the_stopping_rule() -> None:
    report = rules(_observation(normalized_data_mismatch=0.3, cv_mismatch=[5.0, 4.0],
                                cv_minimum_iteration=1,
                                accepted_since_cv_minimum=0))  # fmt: skip

    assert report.recommendation == "investigate"


def test_a_finished_run_needs_nothing_unless_it_fits_poorly() -> None:
    assert rules(_observation(state="complete")).recommendation == "none"
    assert rules(_observation(state="complete", normalized_data_mismatch=186.0)).findings == []
    poor = rules(_observation(state="complete", normalized_data_mismatch=2e4))
    assert _codes(poor) == {"poor_fit"}
    assert poor.recommendation == "investigate"


def test_a_finished_run_that_left_the_prior_unchanged_is_flagged() -> None:
    """Tester finding 34: LM-EnRML stopped after one heavily damped step, 252 -> 251.8."""
    stuck = rules(
        _observation(
            state="complete", data_mismatch_means=[252.0, 251.8],
            data_mismatch_iterations=[0, 1], normalized_data_mismatch=5.0,
        )
    )  # fmt: skip

    assert _codes(stuck) == {"no_progress"}
    assert "251.8" in stuck.findings[0].evidence
    assert stuck.recommendation != "none"
    # 15 % lower is progress; a run still going is judged by the other rules.
    assert rules(_observation(state="complete", data_mismatch_means=[100.0, 85.0])).findings == []
    assert "no_progress" not in _codes(rules(_observation(data_mismatch_means=[252.0, 251.8])))
    # A prior that already fits the data (normalized mismatch near 1) has nothing to gain.
    fits = _observation(state="complete", data_mismatch_means=[52.0, 51.8],
                        normalized_data_mismatch=1.1)  # fmt: skip
    assert rules(fits).findings == []


def test_thresholds_are_configurable() -> None:
    strict = MonitorSection(collapse_spread_ratio=0.5)

    assert "ensemble_collapse" in _codes(rules(_observation(), strict))


def _model_report(recommendation: str, *codes: str) -> MonitorReport:
    return MonitorReport.model_validate({
        "status": "warning", "recommendation": recommendation, "summary": "the model's view",
        "findings": [
            {"code": c, "severity": "warning", "evidence": f"model says {c}"} for c in codes
        ],
    })  # fmt: skip


def test_model_findings_that_the_facts_contradict_are_dropped() -> None:
    # The live check of 2026-09-29: qwen3.8-27b reported ensemble_collapse and stagnation
    # for a spread of 0.3 of the prior's and a 16.7% data mismatch change within the window.
    facts = _observation(
        data_mismatch_means=[1000.0, 300.0, 120.0, 100.0, 99.6, 99.3], normalized_data_mismatch=0.4
    )
    rules_report = rules(facts)
    assert _codes(rules_report) == {"overfitting"}
    model = _model_report("stop", "overfitting", "stagnation", "ensemble_collapse", "slow_wells")

    report, changes = reconcile(model, rules_report)

    assert _codes(report) == {"overfitting", "slow_wells"}  # its own code stays
    assert report.recommendation == "stop"  # overfitting is stop-worthy and found
    assert [c.split(" (")[0] for c in changes] == [
        "dropped stagnation",
        "dropped ensemble_collapse",
    ]
    # The model's summary may still mention them: it says what was corrected.
    assert report.summary.endswith(
        "[Corrected: stagnation, ensemble_collapse dropped, not supported by the facts.]"
    )


def test_a_stop_without_support_becomes_investigate() -> None:
    rules_report = rules(_observation())  # nothing found

    report, changes = reconcile(_model_report("stop", "damping_runaway"), rules_report)

    assert (report.findings, report.recommendation) == ([], "investigate")
    assert changes[-1] == "stop became investigate: no stop-worthy finding is left"


def test_a_consistent_model_report_is_unchanged() -> None:
    rules_report = rules(_observation(rejections_in_a_row=6, damping=64.0))
    model = _model_report("stop", "damping_runaway")  # the model may also leave findings out

    assert reconcile(model, rules_report) == (model, [])


# ---------------------------------------------------------------------- observation


def test_the_observation_of_an_esmda_run(linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=30)
    result = composition.start(case.config)
    run_dir = case.directory / "runs" / "case"

    o = observe(run_dir)

    history = result.history
    assert (o.method, o.iterative, o.state) == ("esmda", False, "complete")
    assert o.data_mismatch_means == [r.data_mismatch_mean for r in history]
    assert (o.forecasts, o.accepted, o.rejected, o.failed_fraction) == (5, 5, 0, 0.0)
    assert history[-1].metrics is not None
    assert o.normalized_data_mismatch == history[-1].metrics["normalized_data_mismatch"]["all"]
    run = RunDirectory(run_dir)
    prior, last = run.load_arrays("iter_0")["state"], run.load_arrays("iter_4")["state"]
    ratio = np.std(last, axis=-1, ddof=1).mean() / np.std(prior, axis=-1, ddof=1).mean()
    assert o.spread_ratio == {"x": pytest.approx(ratio, rel=1e-12)}
    assert rules(o).recommendation in ("none", "investigate")  # never stop


def test_the_observation_counts_failures_and_rejections(linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=24, method=GIES, simulator_extra=FAILING)
    result = composition.start(case.config)

    o = observe(case.directory / "runs" / "case")

    history = result.history
    assert o.iterative
    assert o.rejected == sum(not r.accepted for r in history) > 0
    assert o.members_prior == 24  # the ensemble size, although the prior forecast lost some
    assert history[0].n_members < 24
    assert o.members_now == history[-1].n_members < 24
    assert o.failed_fraction == pytest.approx(1 - o.members_now / 24)
    trailing = 0
    for record in reversed(history):
        if record.accepted:
            break
        trailing += 1
    assert o.rejections_in_a_row == trailing
    assert o.damping == history[-1].damping


# ---------------------------------------------------------------------- the agent


def _report_json(recommendation: str = "none") -> str:
    return json.dumps({
        "status": "healthy", "recommendation": recommendation,
        "findings": [
            {"code": "spread_ok", "severity": "info", "evidence": "spread 0.4 of the prior"}
        ],
        "summary": "The run converged and kept its spread.",
    })  # fmt: skip


def test_the_model_reports_with_read_only_tools_and_traces(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    case = linear_case(ensemble_size=20)
    composition.start(case.config)
    run_dir = case.directory / "runs" / "case"
    stub = StubProvider([_report_json()])
    monitor = RunMonitor(ApexService(tmp_path / "service"), stub)

    result = monitor.check(run_dir)

    assert result.decided_by == "model"
    assert result.report.findings[0].code == "spread_ok"
    [request] = stub.requests
    assert [t.name for t in request.tools] == list(READ_ONLY_TOOLS)  # no stop_job
    task = request.messages[1].content
    assert '"data_mismatch_means"' in task
    assert "threshold rules' report" in task
    assert result.trace.parent == run_dir / "agents"
    assert len(list((run_dir / "agents").glob("monitor-*-call.json"))) == 1


def test_the_monitor_corrects_the_model_and_traces_it(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    case = linear_case(ensemble_size=20)
    composition.start(case.config)
    run_dir = case.directory / "runs" / "case"
    stub = StubProvider([_model_report("none", "ensemble_collapse").model_dump_json()])

    result = RunMonitor(ApexService(tmp_path / "service"), stub).check(run_dir)

    assert "ensemble_collapse" not in _codes(result.rules_report)
    assert "ensemble_collapse" not in _codes(result.report)
    assert result.corrections[0].startswith("dropped ensemble_collapse")
    [record] = (run_dir / "agents").glob("monitor-*-action.json")
    assert json.loads(record.read_text())["corrections"] == list(result.corrections)


def test_an_invalid_model_report_falls_back_to_the_rules(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    case = linear_case(ensemble_size=20)
    composition.start(case.config)
    stub = StubProvider(['{"status": "fine"}'] * 3)
    monitor = RunMonitor(ApexService(tmp_path / "service"), stub)

    result = monitor.check(case.directory / "runs" / "case")

    assert result.decided_by == "fallback"
    assert result.report == result.rules_report
    decision = json.loads(result.trace.read_text())
    assert decision["decided_by"] == "fallback"
    assert decision["thresholds"] == MonitorSection().model_dump(mode="json")


def test_without_a_model_the_rules_decide(tmp_path: Path, linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=20)
    composition.start(case.config)

    result = RunMonitor(ApexService(tmp_path / "service"), None).check(
        case.directory / "runs" / "case"
    )

    assert result.decided_by == "rules"
    assert json.loads(result.trace.read_text())["model"] is None


def _yes(asked: list[str]) -> Callable[[str], bool]:
    """A person who says yes, noting each question."""

    def confirm(question: str) -> bool:
        asked.append(question)
        return True

    return confirm


def _stoppable(**fields: Any) -> RunObservation:  # noqa: ANN401
    """A running GIES run whose facts justify a stop: 6 rejections in a row, overfitting."""
    return _observation(**{"rejections_in_a_row": 6, "damping": 64.0,
                           "normalized_data_mismatch": 0.3,
                           **fields})  # fmt: skip


def _acting_monitor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, observation: RunObservation
) -> tuple[RunMonitor, list[str]]:
    import apex.agents.monitor as monitor_module

    service = ApexService(tmp_path / "service")
    stopped: list[str] = []
    monkeypatch.setattr(service, "stop", stopped.append)
    monkeypatch.setattr(monitor_module, "observe", lambda *a, **k: observation)
    stop = _model_report("stop", "overfitting").model_dump_json()
    monitor = RunMonitor(service, StubProvider([stop]))
    monkeypatch.setattr(monitor, "resolve", lambda target: (tmp_path / "run", "j1"))
    monkeypatch.setattr(service.jobs, "record", lambda job: {"state": observation.state})
    return monitor, stopped


def test_acting_stops_a_running_iterative_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monitor, stopped = _acting_monitor(tmp_path, monkeypatch, _stoppable())
    asked: list[str] = []

    result = monitor.check("j1", act=True, confirm=_yes(asked))

    assert stopped == ["j1"]
    assert result.action.startswith("stopped job j1")
    [question] = asked  # a person was asked first, with the findings
    assert "recommends stopping job j1" in question
    assert "overfitting" in question
    [action] = (tmp_path / "run" / "agents").glob("monitor-*-action.json")
    assert json.loads(action.read_text())["action"] == result.action


@pytest.mark.parametrize(
    ("fields", "why"),
    [
        ({"method": "esmda", "iterative": False}, "fixed schedule"),
        ({"state": "complete"}, "complete"),
    ],
)
def test_acting_refuses_fixed_schedules_and_finished_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fields: dict[str, Any], why: str
) -> None:
    monitor, stopped = _acting_monitor(tmp_path, monkeypatch, _stoppable(**fields))
    asked: list[str] = []

    result = monitor.check("j1", act=True, confirm=_yes(asked))

    assert (stopped, asked) == ([], [])  # refused before anyone is asked
    assert result.action.startswith("refused")
    assert why in result.action


@pytest.mark.parametrize(
    ("confirm", "action"),
    [(lambda q: False, "declined: job j1 was not stopped"),
     (None, "refused: no one to confirm the stop")],
)  # fmt: skip
def test_a_stop_needs_a_persons_yes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    confirm: Callable[[str], bool] | None,
    action: str,
) -> None:
    monitor, stopped = _acting_monitor(tmp_path, monkeypatch, _stoppable())

    result = monitor.check("j1", act=True, confirm=confirm)

    assert (stopped, result.action) == ([], action)
    [record] = (tmp_path / "run" / "agents").glob("monitor-*-action.json")
    assert json.loads(record.read_text())["action"] == action


@pytest.mark.parametrize(("answer", "expected"), [("y\n", True), ("n\n", False), ("", False)])
def test_the_cli_asks_at_the_terminal(answer: str, expected: bool) -> None:
    """``--act`` asks before stopping; no answer (EOF, e.g. under nohup) is a no."""
    import typer

    from apex.cli.agent_commands import _confirm

    probe = typer.Typer()
    answers: list[bool] = []

    @probe.command()
    def run() -> None:
        answers.append(_confirm("Stop it?"))

    CliRunner().invoke(probe, [], input=answer)

    assert answers == [expected]


def test_without_act_nothing_is_stopped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monitor, stopped = _acting_monitor(tmp_path, monkeypatch, _stoppable())

    result = monitor.check("j1", confirm=lambda q: True)

    assert result.report.recommendation == "stop"
    assert (stopped, result.action) == ([], "none")


def test_runs_are_found_by_job_and_run_id(tmp_path: Path, linear_case: CaseFactory) -> None:
    service = ApexService(tmp_path / "service")
    handle = service.submit_run(linear_case(ensemble_size=10).config)
    assert service.wait(handle.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
    monitor = RunMonitor(service, None)

    by_job = monitor.resolve(handle.id)
    by_id = monitor.resolve(handle.run_ids[0][:12])

    assert by_job == (by_id[0], handle.id)
    assert by_id[1] == handle.id
    observation = monitor.check(handle.id).observation
    assert observation.state == "complete"
    # The facts carry the run id, which run_summary needs (the model had to guess it).
    assert observation.run_id == handle.run_ids[0]


# ---------------------------------------------------------------------- CLI


def test_the_cli_reports_on_a_run(tmp_path: Path, linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=20)
    composition.start(case.config)
    run_dir = str(case.directory / "runs" / "case")
    root = str(tmp_path / "service")
    runner = CliRunner()

    text = runner.invoke(app, ["agent", "monitor", run_dir, "--no-llm", "--root", root])
    as_json = runner.invoke(
        app, ["agent", "monitor", run_dir, "--no-llm", "--json", "--root", root]
    )
    watched = runner.invoke(
        app, ["agent", "monitor", run_dir, "--no-llm", "--watch", "--interval", "1", "--root", root]
    )

    assert text.exit_code == 0, text.output
    assert text.stdout.splitlines()[0].split(":")[0] in ("healthy", "warning")
    document = json.loads(as_json.stdout)
    assert document["decided_by"] == "rules"
    assert document["observation"]["method"] == "esmda"
    assert watched.exit_code == 0  # a finished run without a job: one check, then done
    missing = runner.invoke(app, ["agent", "monitor", "nothing-here", "--no-llm", "--root", root])
    assert missing.exit_code == 1
    assert "not a run directory" in missing.stderr


def test_apex_report_writes_the_health_figures(linear_case: CaseFactory) -> None:
    gies = linear_case(name="g", ensemble_size=24, method=GIES, simulator_extra=FAILING)
    composition.start(gies.config)
    esmda = linear_case(name="e", ensemble_size=20)
    composition.start(esmda.config)

    iterative = composition.write_report_figures(gies.directory / "runs" / "g")
    fixed = composition.write_report_figures(esmda.directory / "runs" / "e")

    def health(paths: list[Path]) -> set[str]:
        return {p.stem for p in paths if p.parent.name == "health"}

    assert health(iterative) == {"spread", "data_mismatch_change", "members", "damping"}
    assert health(fixed) == {"spread", "data_mismatch_change", "members"}  # no damping for ES-MDA

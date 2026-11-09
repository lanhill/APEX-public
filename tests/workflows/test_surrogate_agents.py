"""The run monitor and the report agent on surrogate builds.

The case is linear and the surrogate the test-only affine map of ``surrogate_cases``, as in
the build's own tests. The facts are computed from the build's folder; the rules are
checked on built folders and, for verdicts a small linear case cannot give (a failed gate,
a failed check), on facts set by hand.
"""

import importlib.util
import os
import threading
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

from apex.agents.monitor import Finding, MonitorReport
from apex.agents.surrogate import (
    CheckFact,
    SurrogateFacts,
    SurrogateMonitor,
    reconcile,
    surrogate_facts,
    surrogate_report,
    surrogate_rules,
)
from apex.cli.worker import run_job
from apex.core.errors import ConfigError
from apex.core.jobs import JobRecord
from apex.core.plugins import surrogate_models
from apex.services import surrogates
from apex.services.api import ApexService
from apex.services.jobs import LocalJobRunner
from apex.services.settings import MonitorSection

PLUGIN = "test_affine_agents"
TIMEOUT = 120.0


def _load_cases() -> ModuleType:
    path = Path(__file__).with_name("surrogate_cases.py")
    spec = importlib.util.spec_from_file_location("_surrogate_cases_agents", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cases = _load_cases()


@pytest.fixture(autouse=True)
def _registered(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    surrogate_models.register(PLUGIN)(cases.Affine)

    def spawn(self: LocalJobRunner, record: JobRecord) -> int:
        def work() -> None:
            run_job(LocalJobRunner(self.root, role="worker"), record["id"])

        threading.Thread(target=work, daemon=True).start()
        return os.getpid()

    monkeypatch.setattr(LocalJobRunner, "_spawn", spawn)
    try:
        yield
    finally:
        surrogate_models.remove(PLUGIN)


def _build(folder: Path) -> Path:
    cases.write_case(folder, name="base")
    built: Path = cases.write_build(folder, PLUGIN, case="base.toml")
    return built


def test_a_complete_build_is_healthy_and_its_report_holds_its_numbers(tmp_path: Path) -> None:
    service = ApexService(tmp_path / "service")
    handle = service.submit_surrogate_build(_build(tmp_path))
    assert service.wait(handle.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
    path = Path(service.surrogates()[0]["path"])

    monitor = SurrogateMonitor(service, None)
    # Found by its folder and by its job, with the job's own word on its state.
    assert monitor.resolve(path)[1] is not None
    assert monitor.resolve(handle.id)[0] == path
    check = monitor.check(handle.id)
    facts = check.facts
    assert facts.state == "complete"
    assert facts.job_id == handle.id
    assert (facts.planned_runs, facts.finished_runs, facts.failed_runs) == (40, 40, 0)
    assert facts.gate_passed is True
    assert facts.gate
    assert all(d.passed for d in facts.gate)
    assert check.decided_by == "rules"
    assert check.report.status == "healthy"
    assert check.report.recommendation == "none"
    assert (path / "agents").is_dir()  # the decision is recorded beside the build

    result = surrogate_report(service, None, path)
    assert f"Surrogate build {facts.name}" in result.text
    assert "40 of 40 design runs" in result.text
    assert "passed its gate" in result.text


def test_a_stopped_build_is_reported_with_its_pending_runs(tmp_path: Path) -> None:
    service = ApexService(tmp_path / "service")
    built = surrogates.build(_build(tmp_path), max_runs=7)
    facts = SurrogateMonitor(service, None).facts(built.path)
    assert facts.job_id is None
    assert facts.state == "stopped"
    assert facts.finished_runs == 7
    assert facts.pending_runs == 33
    report = surrogate_rules(facts, MonitorSection())
    assert [f.code for f in report.findings] == ["stopped_pending"]
    assert report.status == "healthy"  # information only


def test_the_counts_come_from_the_journal_not_a_stale_summary(tmp_path: Path) -> None:
    """The review of 2026-11-09: a summary is the last stop's. An adaptive design's pending
    count holds its undrawn runs, a coarse stage's its coarse runs, and a resume runs on
    after it: the design's runs are counted from the build's journal."""
    import json

    from apex.storage.surrogate_directory import SUMMARY, MemberRecord, SurrogateDirectory

    service = ApexService(tmp_path / "service")
    built = surrogates.build(_build(tmp_path), max_runs=7)
    directory = SurrogateDirectory(built.path)
    summary = json.loads((built.path / SUMMARY).read_text())
    # As an adaptive design stopped after its first batch writes it: 10 training runs drawn
    # of 30, and every undrawn one counted as pending.
    (built.path / SUMMARY).write_text(json.dumps({**summary, "training_runs": 10,
                                                  "pending": 43}))  # fmt: skip
    # A resume running on: two more runs journalled, one failed.
    directory.record([MemberRecord(100, True, 1.0), MemberRecord(101, False, None, "boom")])
    facts = SurrogateMonitor(service, None).facts(built.path)
    assert (facts.planned_runs, facts.finished_runs, facts.failed_runs) == (40, 9, 1)
    assert facts.pending_runs == 31
    assert facts.coarse_runs is None

    # Every full run done and the build stopped in its coarse stage: what is left is coarse.
    directory.record([MemberRecord(200 + i, True, 1.0) for i in range(31)])
    coarse = SurrogateDirectory(built.path / surrogates.COARSE)
    coarse.members.mkdir(parents=True)
    coarse.record([MemberRecord(i, True, 1.0) for i in range(12)])
    (built.path / SUMMARY).write_text(json.dumps({**summary, "pending": 28}))
    facts = SurrogateMonitor(service, None).facts(built.path)
    assert (facts.finished_runs, facts.pending_runs) == (40, 0)
    assert (facts.coarse_runs, facts.coarse_pending) == (12, 28)
    [finding] = surrogate_rules(facts, MonitorSection()).findings
    assert finding.code == "stopped_pending"
    assert "28 coarse runs" in finding.evidence
    # A resume running the coarse stage: the stopped summary's coarse count is not current.
    running = surrogate_facts(built.path, {"id": "j", "state": "running"})
    assert running.coarse_pending == 0

    # After a fit, a run lost to its failed coarse run counts as failed (journalled ok).
    (built.path / SUMMARY).write_text(json.dumps({**summary, "pending": 0, "failed": [101, 5]}))
    assert SurrogateMonitor(service, None).facts(built.path).failed_runs == 2


def test_the_monitor_refuses_what_is_not_a_build(tmp_path: Path) -> None:
    from apex.agents.surrogate import is_build_dir

    service = ApexService(tmp_path / "service")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ConfigError, match="not a surrogate build"):
        SurrogateMonitor(service, None).resolve(tmp_path / "empty")
    (tmp_path / "empty" / "state.json").write_text("{torn")  # a run's dispatch asks too
    assert not is_build_dir(tmp_path / "empty")


def test_a_build_its_cli_runs_is_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No job, a stopped summary, and the folder's lock held by the CLI's build: running."""
    from apex.storage.locks import exclusive

    built = surrogates.build(_build(tmp_path), max_runs=7)
    with exclusive(built.path, "surrogate build"):
        assert surrogate_facts(built.path, None).state == "running"
    assert surrogate_facts(built.path, None).state == "stopped"
    # The lock file names a holder that has ended: the facts do not probe the lock (a probe
    # takes it for an instant, and a CLI build starting then would be refused). Any flock
    # here would be a probe, whichever helper made it.
    import fcntl
    import json
    import socket
    import subprocess

    from apex.storage import locks

    ended = subprocess.Popen(["true"])
    ended.wait()
    (built.path / locks.LOCK_FILE).write_text(
        json.dumps({"pid": ended.pid, "host": socket.gethostname(), "since": "-"})
    )
    probes: list[int] = []
    real = fcntl.flock

    def probe(descriptor: int, operation: int) -> None:
        probes.append(operation)
        real(descriptor, operation)

    monkeypatch.setattr(fcntl, "flock", probe)
    assert surrogate_facts(built.path, None).state == "stopped"
    assert probes == []


def _facts(**changes: object) -> SurrogateFacts:
    base: dict[str, object] = {
        "path": "/b", "name": "lin", "state": "complete", "plugin": "pod_rbf", "plan": "prior",
        "planned_runs": 40, "finished_runs": 40, "failed_runs": 0, "failed_share": 0.0,
        "pending_runs": 0, "gate_passed": True,
    }  # fmt: skip
    return SurrogateFacts.model_validate({**base, **changes})


def test_the_rules_name_a_failed_gate_a_failed_check_failures_and_a_small_speedup() -> None:
    thresholds = MonitorSection()
    report = surrogate_rules(
        _facts(
            failed_runs=4, failed_share=0.1, gate_passed=False,
            gate_failures=["WOPR: sigma_ratio error 0.8 > 0.5"],
            prediction_seconds=1.0, member_seconds=1.0, simulator_median_seconds=5.0, speedup=5.0,
            last_check=CheckFact(id="c", sequence=2, against="hm.toml", runs=10, succeeded=10,
                                 inside=4, passed=False),
        ),
        thresholds,
    )  # fmt: skip
    codes = {f.code: f.severity for f in report.findings}
    assert codes == {
        "failed_runs": "warning", "gate_failed": "critical", "check_failed": "critical",
        "small_speedup": "info",
    }  # fmt: skip
    assert report.status == "critical"
    assert report.recommendation == "investigate"
    check = next(f for f in report.findings if f.code == "check_failed")
    assert "4 lay inside the domain" in check.evidence

    # Below the thresholds: nothing to say; a running build goes on.
    quiet = surrogate_rules(
        _facts(state="running", failed_runs=1, failed_share=0.025, gate_passed=None,
               speedup=50.0, prediction_seconds=0.1, simulator_median_seconds=5.0),
        thresholds,
    )  # fmt: skip
    assert quiet.findings == []
    assert quiet.recommendation == "continue"


def test_the_models_report_is_held_to_the_rules() -> None:
    rules = surrogate_rules(_facts(), MonitorSection())
    invented = MonitorReport(
        status="critical", recommendation="stop",
        findings=[Finding(code="gate_failed", severity="critical", evidence="made up"),
                  Finding(code="odd_timing", severity="info", evidence="its own")],
        summary="The model's view.",
    )  # fmt: skip
    held, changes = reconcile(invented, rules)
    assert [f.code for f in held.findings] == ["odd_timing"]  # not a rule's code: kept
    assert held.recommendation == "investigate"
    assert len(changes) == 2


def test_the_rest_routes_monitor_and_report_a_build(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from apex.api.app import create_app
    from apex.services.auth import api_token

    root = tmp_path / "service"
    root.mkdir()
    service = ApexService(root)
    handle = service.submit_surrogate_build(_build(tmp_path))
    assert service.wait(handle.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
    path = service.surrogates()[0]["path"]
    http = TestClient(create_app(root))
    owner = {"Authorization": f"Bearer {api_token(root)}"}
    monitored = http.post("/v1/agents/monitor", headers=owner, json={"surrogate": path})
    assert monitored.status_code == 200, monitored.text
    assert monitored.json()["report"]["status"] == "healthy"
    reported = http.post("/v1/agents/report", headers=owner, json={"surrogate": path})
    assert reported.status_code == 200, reported.text
    assert "passed its gate" in reported.json()["markdown"]
    both = http.post("/v1/agents/report", headers=owner, json={"surrogate": path, "run": "x"})
    assert both.status_code == 422
    two = http.post("/v1/agents/monitor", headers=owner, json={"surrogate": path, "run": "x"})
    assert two.status_code == 422

    # A guest allowed both agents never reaches the owner's build through them.
    from apex.services.auth import ui_token

    (root / "ui_access.toml").write_text(
        '[workspace]\nfolders = ["*"]\n[agents]\nnames = ["monitor", "report"]\n'
    )
    guest = {"Authorization": f"Bearer {ui_token(root, create=True, guest='g')}"}
    for route in ("/v1/agents/monitor", "/v1/agents/report"):
        hidden = http.post(route, headers=guest, json={"surrogate": path})
        assert hidden.status_code == 404, (route, hidden.text)


def test_a_build_the_agents_read_can_still_be_deleted_and_pruned(tmp_path: Path) -> None:
    """The monitor and the report write the build's agents/ and report/ (the coder's review):
    a deletion and guest retention still judge the folder a build's."""
    from apex.services import deletion, guest_retention

    service = ApexService(tmp_path / "service")
    handle = service.submit_surrogate_build(_build(service.workspace().roots[0]))
    assert service.wait(handle.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
    folder = Path(service.surrogates()[0]["path"])
    SurrogateMonitor(service, None).check(folder)
    report = folder / "report" / "narrative.md"  # where `apex agent report` writes it
    report.parent.mkdir()
    report.write_text(surrogate_report(service, None, folder).text)
    assert deletion.unsafe_build(folder, [service.root]) is None
    # The plan judges the folder a build's (it refuses it only as outside the workspace,
    # where the service's own folder is).
    blocked = deletion.plan(service, handle.id).blocked
    assert all("outside the workspace" in reason for reason in blocked), blocked
    unit = guest_retention.Submission("guest:g", handle.id, guest_retention._parse("2026-01-01"))
    unit.jobs, unit.dirs, unit.builds = [handle.id], [folder], [folder]
    deleted, _, _ = guest_retention.delete(service, unit)
    assert deleted
    assert not folder.exists()


def test_the_speedup_counts_a_coarse_run_and_none_for_an_early_start_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The coder's review: a coarse-corrected model runs the coarse stack per member, an
    early-start model the simulator up to its date; the speedup counts the first, and is
    not given for the second (its early run is not timed)."""
    import json

    from apex.storage.surrogate_directory import SUMMARY, MemberRecord, SurrogateDirectory

    service = ApexService(tmp_path / "service")
    built = surrogates.build(_build(tmp_path))
    summary = json.loads((built.path / SUMMARY).read_text())
    (built.path / SUMMARY).write_text(json.dumps(
        {**summary, "prediction_seconds": 0.5, "simulator_median_seconds": 50.0}))  # fmt: skip
    plain = SurrogateMonitor(service, None).facts(built.path)
    assert (plain.member_simulation, plain.speedup) == ("none", 100.0)

    coarse = SurrogateDirectory(built.path / surrogates.COARSE)
    coarse.members.mkdir(parents=True)
    coarse.record([MemberRecord(i, True, s) for i, s in enumerate((4.0, 4.5, 30.0))])
    # Coarse runs count only for a model that corrects a coarse forecast.
    assert SurrogateMonitor(service, None).facts(built.path).speedup == 100.0
    monkeypatch.setattr(cases.Affine, "predict_corrected", lambda *a: None, raising=False)
    facts = SurrogateMonitor(service, None).facts(built.path)
    assert facts.member_simulation == "coarse"
    assert facts.member_seconds == pytest.approx(5.0)  # 0.5 + the coarse runs' median 4.5
    assert facts.speedup == pytest.approx(10.0)
    [finding] = surrogate_rules(facts, MonitorSection(surrogate_min_speedup=20.0)).findings
    assert "with its coarse run" in finding.evidence

    monkeypatch.delattr(cases.Affine, "predict_corrected")
    monkeypatch.setattr(cases.Affine, "predict_from_early", lambda *a: None, raising=False)
    early = SurrogateMonitor(service, None).facts(built.path)
    assert (early.member_simulation, early.speedup) == ("early", None)
    assert surrogate_rules(early, MonitorSection(surrogate_min_speedup=1e9)).findings == []

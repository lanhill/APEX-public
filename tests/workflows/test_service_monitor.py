"""The run monitor inside the job service: ``[agents.monitor] auto``.

A real run is marked running, so the service's hook checks it; the model is the stub
with no replies, so the rules decide (no live model is used).
"""

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import pytest

from apex.agents.watch import ServiceMonitor, service_hook
from apex.services.api import ApexService
from apex.services.daemon import serve_jobs
from apex.services.jobs import LocalJobRunner

TIMEOUT = 180.0


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path


CaseFactory = Callable[..., Case]


def _root(tmp_path: Path, extra: str = "") -> Path:
    root = tmp_path / "service"
    root.mkdir()
    log = tmp_path / "notified.txt"
    (root / "service.toml").write_text(
        '[agents.llm]\nprovider = "stub"\n\n'
        # A threshold every run meets, so the report is a warning (ensemble_collapse).
        "[agents.monitor]\nauto = true\ncollapse_spread_ratio = 0.99\n" + extra + "\n\n"
        "[notifications]\n"
        f"command = \"sh -c 'echo {{event}}: {{job}} {{status}} {{findings}} >> {log}'\"\n"
    )
    return root


def _running_job(root: Path, case: Case) -> tuple[LocalJobRunner, str]:
    service = ApexService(root)
    handle = service.submit_run(case.config)
    assert service.wait(handle.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
    runner = LocalJobRunner(root, role="service")
    runner.update(handle.id, state="running")  # as if its worker were still at it
    return runner, handle.id


def test_the_service_reports_on_running_jobs_and_notifies_once(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    root = _root(tmp_path)
    runner, job = _running_job(root, linear_case(ensemble_size=20))
    hook = ServiceMonitor(root)

    hook(runner)
    hook.drain()
    hook(runner)  # no new forecast since: no second check
    hook.drain()

    reports = [e for e in ApexService(root).events(job) if e.event == "monitor_report"]
    assert len(reports) == 1
    fields = reports[0].fields
    assert (fields["status"], fields["recommendation"]) == ("warning", "investigate")
    assert fields["findings"] == ["ensemble_collapse"]
    assert fields["decided_by"] == "fallback"  # the stub had no reply: the rules decided
    assert Path(fields["trace"]).is_file()
    lines = (tmp_path / "notified.txt").read_text().splitlines()
    # The job's own end is notified too (an empty event); the monitor's report once.
    assert [x for x in lines if x.startswith("monitor_report:")] == [
        f"monitor_report: {job} warning ensemble_collapse"
    ]
    assert any(x.startswith(f": {job}") for x in lines)
    assert runner.record(job)["state"] == "running"  # advisory: nothing was stopped
    hook.close()


def test_notify_on_filters_statuses(tmp_path: Path, linear_case: CaseFactory) -> None:
    root = _root(tmp_path, 'notify_on = ["critical"]')
    runner, job = _running_job(root, linear_case(ensemble_size=20))
    hook = ServiceMonitor(root)

    hook(runner)
    hook.drain()

    assert [e.event for e in ApexService(root).events(job)].count("monitor_report") == 1
    lines = (tmp_path / "notified.txt").read_text().splitlines()  # the job's end only
    assert not [x for x in lines if x.startswith("monitor_report:")]  # a warning, not critical
    hook.close()


def test_a_routine_that_fails_to_start_or_on_a_tick_never_stops_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Routines come from entry points; a broken one is logged and
    skipped, a failing one does not keep the others from running, and all are closed."""
    from apex.services import routines as module

    calls: list[str] = []

    class Good:
        def __call__(self, runner: object) -> None:
            calls.append("good")

        def close(self) -> None:
            calls.append("closed")

    def failing(runner: object) -> None:
        raise RuntimeError("tick failed")

    class Entry:
        def __init__(self, name: str, load: object) -> None:
            self.name, self.value, self._load = name, name, load

        def load(self) -> object:
            if isinstance(self._load, Exception):
                raise self._load
            return self._load

    entries = [
        Entry("broken", ImportError("no module")),
        Entry("off", lambda root: None),
        Entry("failing", lambda root: failing),
        Entry("good", lambda root: Good()),
    ]
    monkeypatch.setattr(module, "entry_points", lambda group: entries)
    hooks = module.ServiceHooks(module.routines(tmp_path))
    assert len(hooks.hooks) == 2
    hooks(LocalJobRunner(tmp_path / "service"))
    hooks.close()
    assert calls == ["good", "closed"]


def test_the_service_loop_calls_its_hook_and_survives_its_failures(tmp_path: Path) -> None:
    root = tmp_path / "service"
    root.mkdir()
    (root / "service.toml").write_text("[service]\npoll_s = 0.05\n")
    stop, calls = threading.Event(), []

    def failing(runner: LocalJobRunner) -> None:
        calls.append(runner)
        if len(calls) >= 3:
            stop.set()
        raise RuntimeError("a broken hook")

    serve_jobs(root, stop, failing)

    assert len(calls) >= 3  # it went on after the first failure
    assert not (root / "service.pid").exists()


def test_the_service_closes_its_hook_when_it_stops(tmp_path: Path) -> None:
    """Routines with threads of their own are closed at the service's end."""
    root = tmp_path / "service"
    root.mkdir()
    (root / "service.toml").write_text("[service]\npoll_s = 0.05\n")
    stop, closed = threading.Event(), []

    class Hook:
        def __call__(self, runner: LocalJobRunner) -> None:
            stop.set()

        def close(self) -> None:
            closed.append(True)

    serve_jobs(root, stop, Hook())
    assert closed == [True]


def test_the_monitor_is_off_unless_asked_and_routines_can_be_switched_off(tmp_path: Path) -> None:
    """Core's own hook is the run monitor (``[agents.monitor] auto``); the routines installed
    packages add are none with ``[service] routines = false``."""
    root = tmp_path / "service"
    root.mkdir()
    (root / "service.toml").write_text("[service]\nroutines = false\n")
    assert service_hook(root) is None
    (root / "service.toml").write_text(
        '[service]\nroutines = false\n[agents.llm]\nprovider = "stub"\n'
        "[agents.monitor]\nauto = true\n"
    )
    hook = service_hook(root)
    assert hook is not None
    assert [type(h).__name__ for h in hook.hooks] == [ServiceMonitor.__name__]
    hook.close()

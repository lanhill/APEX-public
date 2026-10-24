"""Jobs pinned to a frozen copy of APEX, with real worker processes.

The copy is a worktree of a small git repository whose interpreter is a wrapper of this one
that records every start: what is checked is that a pinned job's worker, and its automatic
resume, start from the copy, and the refusals. That the copy then imports its own code is
plain Python (its environment's editable install), checked on the site.
"""

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

import pytest

from apex.cli import worker
from apex.core.errors import ConfigError
from apex.services import composition
from apex.services.api import ApexService
from apex.services.code import CodeCopies, CodeError
from apex.services.jobs import LocalJobRunner, alive
from apex.services.settings import load_settings
from apex.storage.sqlite_store import JOB_PROTOCOL

TIMEOUT = 240.0
GIES = (
    'name = "gies"\nstrategy = { name = "rlmmac", options = { energy = 1.0 } }\n'
    "schedule = { data_mismatch_tolerance = 1e-9 }\nmax_iterations = 6"
)


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path


CaseFactory = Callable[..., Case]


def _wait_for(condition: Callable[[], bool], timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.org", *args],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    return done.stdout.strip()


class RecordingSync:
    """Gives each copy an interpreter that records its starts, then runs this one."""

    def __init__(self, starts: Path) -> None:
        self.starts = starts

    def __call__(self, copy: Path, extras: Sequence[str], log: Path) -> None:
        python = copy / ".venv" / "bin" / "python"
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_text(
            f'#!/bin/sh\necho "{copy.name} $*" >> "{self.starts}"\nexec "{sys.executable}" "$@"\n'
        )
        python.chmod(0o755)


@pytest.fixture
def service(tmp_path: Path) -> ApexService:
    """A service whose frozen copies come from a small repository, two commits deep."""
    repo = tmp_path / "repo"
    store = repo / "src" / "apex" / "storage" / "sqlite_store.py"
    store.parent.mkdir(parents=True)
    (repo / ".gitignore").write_text(".venv/\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    for version in ("first", "second"):
        store.write_text(f'"""{version}"""\n\nJOB_PROTOCOL = {JOB_PROTOCOL}\n')
        _git(repo, "add", ".")
        _git(repo, "commit", "-q", "-m", version)
    service = ApexService(tmp_path / "service")
    service.code = CodeCopies(tmp_path / "code", repo, sync=RecordingSync(tmp_path / "starts"))
    return service


def _starts(service: ApexService) -> list[str]:
    path = service.code.directory.parent / "starts"
    return path.read_text().splitlines() if path.exists() else []


def test_a_pinned_job_and_its_automatic_resume_run_from_the_frozen_copy(
    service: ApexService, linear_case: CaseFactory
) -> None:
    # Slow enough (about a second per forecast) that the kill comes before the end.
    case = linear_case(method=GIES, ensemble_size=10, simulator_extra="delay_s = 0.1")
    config = composition.load_config(case.config)
    output = config.run.output_dir
    assert output is not None
    head = service.code.resolve("HEAD")

    handle = service.submit_run(config, code="frozen")
    _wait_for(lambda: (output / "state.json").exists())
    pid = service.jobs.record(handle.id)["pid"]
    os.kill(pid, signal.SIGKILL)  # a crash: no end recorded
    _wait_for(lambda: not alive(pid))
    status = service.wait(handle.id, timeout_s=TIMEOUT, poll_s=0.2)

    assert status.state == "complete", service.events(handle.id)
    record = service.jobs.record(handle.id)
    assert record["kind"] == "resume"  # recovered...
    assert record["code"] == {
        "commit": head,
        "python": str(service.code.path(head) / ".venv/bin/python"),
    }
    # ...on the same code: both workers started from the copy.
    assert _starts(service) == [f"{head} -m apex.cli.worker {service.root} {handle.id}"] * 2
    assert (output / "checkpoints" / "posterior.npz").exists()


def test_live_jobs_and_finished_runs_freeze_nothing(
    service: ApexService, linear_case: CaseFactory
) -> None:
    config = composition.load_config(linear_case().config)
    handle = service.submit_run(config)  # live by default
    assert service.wait(handle.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
    again = service.submit_run(config, code="frozen")  # reused as it is

    assert again.reused
    assert service.jobs.record(handle.id)["code"] is None
    assert service.jobs.record(again.id)["code"] is None
    assert not service.code.directory.exists()


def test_frozen_resumes_continue_on_the_run_s_own_commit(
    service: ApexService, tmp_path: Path
) -> None:
    first = service.code.resolve("HEAD~1")
    run = tmp_path / "run"
    run.mkdir()
    head = service.code.resolve("HEAD")

    def pinned(code: str | None, run_dir: Path | None = None) -> str | None:
        pin = service.pin_code(code, run_dir)
        return None if pin is None else pin["commit"]

    def started_on(commit: str) -> None:
        record = {"start": {"apex": {"commit": commit, "dirty": False}}, "resumes": []}
        (run / "provenance.json").write_text(json.dumps(record))

    started_on(first + "0" * 28)
    assert pinned("frozen", run) == first
    assert pinned("frozen") == head
    assert pinned("live", run) is None
    assert pinned(None, run) is None
    # A commit that cannot be pinned (here: not in the repository) falls back to HEAD.
    started_on("f" * 40)
    assert pinned("frozen", run) == head
    with pytest.raises(CodeError):
        service.pin_code("f" * 12)


def test_a_job_whose_copy_is_gone_fails_with_the_reason(service: ApexService) -> None:
    pin = service.code.ensure()
    pin.python.unlink()  # removed by hand after the submission

    # Spawning fails before any worker, so the spec does not matter.
    job = service.jobs.submit("prediction", {"config": "{}"}, code=pin.as_json())

    record = service.jobs.record(job)
    assert record["state"] == "failed"
    assert "is gone" in record["error"]
    assert _starts(service) == []


def test_a_worker_of_another_job_protocol_refuses_the_job(tmp_path: Path) -> None:
    runner = LocalJobRunner(tmp_path / "service", role="worker")
    job = runner.submit("prediction", {"config": "{}"}, start=False)
    with runner.store.locked():
        runner.store._execute(
            "UPDATE meta SET value = ? WHERE key = 'job_protocol'", (str(JOB_PROTOCOL + 1),)
        )

    worker.run_job(runner, job)

    record = runner.record(job)
    assert record["state"] == "failed"
    assert f"job protocol {JOB_PROTOCOL} and the job store {JOB_PROTOCOL + 1}" in record["error"]


def test_a_live_job_started_by_a_pinned_worker_runs_live_code(
    service: ApexService, linear_case: CaseFactory
) -> None:
    service.jobs.max_concurrent_jobs = 1
    first = linear_case(name="first", ensemble_size=10, simulator_extra="delay_s = 0.02")
    pinned = service.submit_run(composition.load_config(first.config), code="frozen")
    live = service.submit_run(composition.load_config(linear_case(name="second").config))
    store = service.jobs.store
    assert store.job(live.id)["state"] == "queued"

    # No scheduling here: the pinned worker starts the queued job as it ends (embedded mode).
    _wait_for(lambda: store.job(live.id)["state"] != "queued", timeout=120)
    assert service.wait(live.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"

    assert service.jobs.record(pinned.id)["state"] == "complete"
    assert service.jobs.record(live.id)["code"] is None
    # Only the pinned job's worker started from the copy; the live one from this interpreter.
    assert [line.split()[-1] for line in _starts(service)] == [pinned.id]


def test_a_resume_keeps_the_code_its_run_ran(
    service: ApexService, linear_case: CaseFactory
) -> None:
    config = composition.load_config(linear_case().config)
    first = service.submit_run(config, code="frozen", max_steps=1)
    assert service.wait(first.id, timeout_s=TIMEOUT, poll_s=0.2).state == "stopped"
    pin = service.jobs.record(first.id)["code"]

    resumed = service.resume(first.id)  # no code named: the web UI's button, Client.resume

    assert service.jobs.record(resumed.id)["code"] == pin
    assert service.wait(resumed.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
    live = service.resume(resumed.id, code="live")  # unless the caller says otherwise
    assert service.jobs.record(live.id)["code"] is None


def test_a_pin_a_running_older_service_would_ignore_is_refused(service: ApexService) -> None:
    # This process plays a running service that declared nothing (code from before 0062).
    (service.root / "service.pid").write_text(f"{os.getpid()}\n")
    try:
        with pytest.raises(CodeError, match="predates frozen jobs"):
            service.pin_code("frozen")
        assert service.pin_code("live") is None
        service.jobs.store.declare_service(os.getpid())
        assert service.pin_code("frozen") is not None
    finally:
        (service.root / "service.pid").unlink()


def test_a_worker_that_cannot_start_records_why(tmp_path: Path) -> None:
    root = tmp_path / "service"
    runner = LocalJobRunner(root)
    job = runner.submit("prediction", {"config": "{}"}, start=False)
    (root / "service.toml").write_text("[service\n")  # unreadable

    subprocess.run(
        [sys.executable, "-m", "apex.cli.worker", str(root), job], capture_output=True, check=False
    )

    record = runner.store.job(job)
    assert record["state"] == "failed"
    assert "the worker could not start" in record["error"]


def test_a_worker_reads_settings_newer_code_wrote(tmp_path: Path) -> None:
    (tmp_path / "service.toml").write_text(
        "[service]\nmax_retries = 5\nfuture_option = 1\n\n[future_section]\nx = 1\n"
    )
    with pytest.raises(ConfigError, match="future_option"):
        load_settings(tmp_path)
    settings = load_settings(tmp_path, lenient=True)
    assert settings.service.max_retries == 5
    assert LocalJobRunner(tmp_path, role="worker").settings.service.max_retries == 5

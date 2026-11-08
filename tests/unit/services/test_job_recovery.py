"""A dead worker's own last words in its job's record (release tester H34)."""

import subprocess
from pathlib import Path

from apex.services.jobs import WORKER_START_MARKER, LocalJobRunner, worker_cause

TRACEBACK = """[2026-11-08 19:42:50] starting
Traceback (most recent call last):
  File "worker.py", line 89, in run_job
    record = runner.record(job_id)
  File "sqlite_store.py", line 285, in job
    raise KeyError(f"no job {job_id!r}")
KeyError: "no job 'j1'"
"""


def _dead_pid() -> int:
    process = subprocess.Popen(["true"])
    process.wait()
    return process.pid


def test_a_dead_workers_exception_is_its_jobs_error(tmp_path: Path) -> None:
    """'its worker died' alone hid a KeyError: the job's error quotes the worker's traceback's
    last line, once the automatic resumes are used up."""
    runner = LocalJobRunner(tmp_path)
    retries = runner.settings.service.max_retries
    runner.store.insert_job({"id": "j1", "kind": "optimization", "state": "running",
                             "created": "2026-11-08", "spec": {}, "pid": _dead_pid(),
                             "retries": retries})  # fmt: skip
    (tmp_path / "jobs" / "j1").mkdir(parents=True)
    (tmp_path / "jobs" / "j1" / "worker.log").write_text(TRACEBACK)

    runner.schedule()

    record = runner.record("j1")
    assert record["state"] == "failed"
    assert record["error"] == (
        f"the worker process died ({retries} automatic resumes used): KeyError: \"no job 'j1'\""
    )


def test_a_worker_killed_from_outside_leaves_no_cause(tmp_path: Path) -> None:
    log = tmp_path / "worker.log"
    assert worker_cause(log) is None  # no log at all
    log.write_text("[2026-11-08] evaluations forecast: 10/10 members ok\n")
    assert worker_cause(log) is None  # killed by a signal: no traceback
    log.write_text(TRACEBACK + "[2026-11-08] retried\nTraceback (most recent call last):\n")
    assert worker_cause(log) is None  # cut off mid-traceback: nothing certain
    log.write_text("x" * 50_000 + TRACEBACK)
    assert worker_cause(log) == "KeyError: \"no job 'j1'\""  # read from the log's end


def test_only_the_latest_attempts_traceback_counts(tmp_path: Path) -> None:
    """Retries append to one log (the review): an earlier attempt's exception is not the cause
    of a later attempt killed from outside."""
    log = tmp_path / "worker.log"
    log.write_text(f"{WORKER_START_MARKER} 1 ===\n{TRACEBACK}{WORKER_START_MARKER} 2 ===\n"
                   "[2026-11-08] evaluations forecast: 10/10 members ok\n")  # fmt: skip
    assert worker_cause(log) is None


def test_the_exception_line_is_found_under_a_long_message_and_later_lines(tmp_path: Path) -> None:
    log = tmp_path / "worker.log"
    log.write_text(
        f"{WORKER_START_MARKER} 1 ===\n"
        "Traceback (most recent call last):\n"
        '  File "worker.py", line 9, in main\n'
        "    run()\n"
        "pydantic_core._pydantic_core.ValidationError: 1 validation error for Config\n"
        "method.name\n"
        "  Input should be a valid string [type=string_type]\n"
        "    For further information visit https://errors.pydantic.dev\n"
        "[2026-11-08] logging shut down\n"
    )
    assert worker_cause(log) == (
        "pydantic_core._pydantic_core.ValidationError: 1 validation error for Config"
    )


def test_a_retried_job_carries_its_workers_cause(tmp_path: Path) -> None:
    """Before the resumes are used up: the cause is in the job_recovered event."""
    runner = LocalJobRunner(tmp_path)
    runner.store.insert_job({"id": "j2", "kind": "optimization", "state": "running",
                             "created": "2026-11-08", "spec": {}, "pid": _dead_pid(),
                             "retries": 0, "after": ["never-ends"]})  # fmt: skip
    (tmp_path / "jobs" / "j2").mkdir(parents=True)
    (tmp_path / "jobs" / "j2" / "worker.log").write_text(TRACEBACK.replace("'j1'", "'j2'"))

    runner._recover()

    [recovered] = [e for e in runner.events("j2") if e.event == "job_recovered"]
    assert recovered.fields["cause"] == "KeyError: \"no job 'j2'\""
    assert runner.record("j2")["state"] == "queued"

"""The job service and its store, with real worker processes.

Recovery, admission, the service process, approval, quotas and notifications, on the
generated linear-Gaussian cases (``delay_s`` makes runs take time).
"""

import datetime as dt
import os
import signal
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import pytest
from typer.testing import CliRunner

from apex.cli.main import app
from apex.core.errors import ConfigError
from apex.services import composition
from apex.services.api import ApexService
from apex.services.approval import ApprovalRequired
from apex.services.jobs import alive, service_pid

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


def test_a_killed_worker_is_resumed_and_its_result_is_bit_identical(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    service = ApexService(tmp_path / "service")
    case = linear_case(method=GIES, ensemble_size=10, simulator_extra="delay_s = 0.03")
    config = composition.load_config(case.config)
    output = config.run.output_dir
    assert output is not None
    handle = service.submit_run(config)
    # Kill the worker once the run has a checkpoint (no end recorded: a crash).
    _wait_for(lambda: (output / "state.json").exists())
    pid = service.jobs.record(handle.id)["pid"]
    os.kill(pid, signal.SIGKILL)
    _wait_for(lambda: not alive(pid))

    status = service.wait(handle.id, timeout_s=TIMEOUT, poll_s=0.2)

    assert status.state == "complete", service.events(handle.id)
    record = service.jobs.record(handle.id)
    assert record["retries"] == 1
    assert record["kind"] == "resume"
    assert "job_recovered" in [e.event for e in service.events(handle.id)]
    other = config.model_copy(
        update={"run": config.run.model_copy(update={"output_dir": case.directory / "whole"})}
    )
    whole = service.submit_run(other, force_new=True)
    assert service.wait(whole.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
    with (
        np.load(output / "checkpoints" / "posterior.npz") as a,
        np.load(case.directory / "whole" / "checkpoints" / "posterior.npz") as b,
    ):
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])


def test_a_job_waits_until_its_reservation_fits(tmp_path: Path, linear_case: CaseFactory) -> None:
    # The tests' cpu pool holds 64: two jobs of 40 runs at once cannot run together.
    service = ApexService(tmp_path / "service", max_concurrent_jobs=4)
    # Each forecast takes 0.5 s (members in parallel), so a job runs for seconds: long
    # enough for the timestamps (to the second) below to tell overlap from succession.
    extra = "delay_s = 0.5\nresources = { cpu = 1 }"
    execution = 'name = "process"\noptions = { workers = 40 }'
    first = service.submit_run(
        linear_case(name="a", ensemble_size=40, simulator_extra=extra, execution=execution).config
    )
    # Another run, not an identical one: that would be reused, queued as it is.
    other = (
        'name = "esmda"\noptions = { inflation = [2, 4, 8, 8] }\n'
        'strategy = { name = "approx", options = { energy = 1.0 } }'
    )
    second = service.submit_run(
        linear_case(
            name="b", ensemble_size=40, simulator_extra=extra, execution=execution, method=other
        ).config
    )

    assert service.jobs.record(first.id)["reservation"] == {"cpu": 40}
    _complete(service, first.id)
    _complete(service, second.id)
    # 40 + 40 > 64: the second started only once the first had ended. (Not "the second is
    # queued" right after its submission: on a loaded CI runner the first had sometimes
    # ended already, the flake of 2026-10-14.)
    ran = [service.jobs.record(j.id) for j in (first, second)]
    assert ran[0]["started"] < ran[0]["finished"] <= ran[1]["started"], ran


def _complete(service: ApexService, job: str) -> None:
    """Wait for a job and assert it completed, saying why if not (an intermittent failure
    of the job above, under a loaded test suite, left no reason to read)."""
    status = service.wait(job, timeout_s=TIMEOUT, poll_s=0.2)
    events = [(e.event, e.message) for e in service.events(job)][-5:]
    assert status.state == "complete", f"{status.state}: {status.error}; last events: {events}"


def test_the_service_process_runs_what_clients_queue(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    root = tmp_path / "service"
    runner = CliRunner()
    started = runner.invoke(app, ["service", "start", "--root", str(root)])
    assert started.exit_code == 0, started.output
    try:
        pid = service_pid(root)
        assert pid is not None
        client = ApexService(root)
        handle = client.submit_run(linear_case(ensemble_size=10).config)
        # The client does not start it itself: the service does, as its parent.
        status = client.wait(handle.id, timeout_s=TIMEOUT, poll_s=0.2)
        assert status.state == "complete"
        shown = runner.invoke(app, ["service", "status", "--root", str(root)])
        assert f"running (pid {pid})" in shown.stdout
        assert "1 complete" in shown.stdout
    finally:
        stopped = runner.invoke(app, ["service", "stop", "--root", str(root)])
    assert stopped.exit_code == 0, stopped.output
    assert service_pid(root) is None


def test_approval_tokens_gate_large_submissions(tmp_path: Path, linear_case: CaseFactory) -> None:
    service = ApexService(tmp_path / "service", approval_threshold_runs=50)
    case = linear_case(ensemble_size=20)  # ES-MDA: 5 forecasts x 20 = 100 simulator runs

    with pytest.raises(ApprovalRequired, match="need approval") as refused:
        service.submit_run(case.config)
    token = refused.value.token
    with pytest.raises(ConfigError, match="still pending"):
        service.submit_run(case.config, approval_token=token)
    service.approvals.approve(token, by="test")
    handle = service.submit_run(case.config, approval_token=token)
    assert service.wait(handle.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
    with pytest.raises(ConfigError, match="already used"):
        service.submit_run(case.config, approval_token=token, force_new=True)
    # The identical run, complete, is reused: it runs nothing, so it needs no approval
    # (release tester N8, 2026-10-20).
    assert service.submit_run(case.config).reused
    # An interactive yes is recorded as an approval.
    other = linear_case(name="other", ensemble_size=20)
    assert service.submit_run(other.config, approve=lambda cost: True).id


def test_studies_with_many_runs_need_approval_and_the_cli_approves(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    root = tmp_path / "service"
    root.mkdir()
    (root / "service.toml").write_text("[approval]\nmax_runs = 2\n")
    case = linear_case(ensemble_size=10)
    study = case.directory / "study.toml"
    study.write_text(
        '[study]\nname = "many"\nreplicates = 2\n\n[study.cases]\nlinear = "config.toml"\n\n'
        '[[study.variants]]\nname = "a"\n\n[[study.variants]]\nname = "b"\n'
        "patch = { method = { options = { inflation = [2, 2] } } }\n"
    )
    service = ApexService(root)

    # The CLI says how in its own form (release tester C9): --approval-token, not the API's.
    asked = CliRunner().invoke(app, ["study", "run", str(study), "--root", str(root)], input="n\n")
    assert asked.exit_code == 1
    assert "--approval-token" in asked.output
    assert "approval_token=" not in asked.output
    with pytest.raises(ApprovalRequired, match="4 runs") as refused:
        service.submit_study(study)  # 4 runs > max_runs = 2, although cheap
    approved = CliRunner().invoke(app, ["approve", refused.value.token, "--root", str(root)])
    assert approved.exit_code == 0, approved.output
    handle = service.submit_study(study, approval_token=refused.value.token)
    assert service.wait(handle.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"


def test_a_grant_covers_a_study_with_its_experiment_annotation(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """A study's annotations reach the gate; it is charged once, as a whole."""
    root = tmp_path / "service"
    root.mkdir()
    (root / "service.toml").write_text("[approval]\nmax_runs = 2\n")
    case = linear_case(ensemble_size=10)
    study = case.directory / "study.toml"
    study.write_text(
        '[study]\nname = "granted"\nreplicates = 2\nannotations = { experiment = "E1" }\n\n'
        '[study.cases]\nlinear = "config.toml"\n\n'
        '[[study.variants]]\nname = "a"\n\n[[study.variants]]\nname = "b"\n'
        "patch = { method = { options = { inflation = [2, 2] } } }\n"
    )
    service = ApexService(root)
    grant = service.grants.create("planner", "E1", 1000, dt.timedelta(hours=1), by="test")

    handle = service.submit_study(study, client="planner")  # 4 runs > max_runs = 2

    assert service.wait(handle.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
    record = service.grants.store.grant(grant["id"])
    assert record is not None
    # 2 runs of 5 x 10 and 2 of 3 x 10 simulator runs, charged once for the study.
    assert len(record["uses"]) == 1
    assert record["used_simulator_runs"] == 160


def test_quotas_refuse_too_large_jobs_and_notifications_run(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    root = tmp_path / "service"
    root.mkdir()
    marks = tmp_path / "marks"
    marks.mkdir()
    (root / "service.toml").write_text(
        f'[notifications]\ncommand = "touch {marks}/{{job}}.{{state}}"\n\n'
        "[quotas.planner]\nmax_simulator_runs_per_day = 60\n"
    )
    service = ApexService(root)
    case = linear_case(ensemble_size=20)

    with pytest.raises(ConfigError, match="more than client 'planner' may use in a day"):
        service.submit_run(case.config, client="planner")  # 100 > 60
    handle = service.submit_run(case.config, client="user")
    assert service.wait(handle.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
    _wait_for(lambda: (marks / f"{handle.id}.complete").exists())


def test_runs_are_listed_reindexed_and_forgotten(tmp_path: Path, linear_case: CaseFactory) -> None:
    root = tmp_path / "service"
    service = ApexService(root)
    handle = service.submit_run(linear_case(ensemble_size=10).config, annotations={"study": "s"})
    assert service.wait(handle.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
    runner = CliRunner()

    listed = runner.invoke(app, ["runs", "list", "--annotation", "study=s", "--root", str(root)])
    assert handle.run_ids[0][:12] in listed.stdout
    shown = runner.invoke(app, ["runs", "show", handle.run_ids[0][:12], "--root", str(root)])
    assert '"state": "complete"' in shown.stdout
    reindexed = runner.invoke(app, ["service", "reindex", "--root", str(root)])
    assert "indexed" in reindexed.stdout
    run_dir = service.jobs.record(handle.id)["run_dirs"][0]
    import shutil

    shutil.rmtree(run_dir)
    forgot = runner.invoke(app, ["gc", "--root", str(root)])
    assert "forgot 1 runs" in forgot.stdout


def test_every_approval_token_can_be_given_to_apex_approve(tmp_path: Path) -> None:
    """Regression: URL-safe tokens could begin with "-" (1 in 64), which the CLI read as an
    option ("Usage: apex approve [OPTIONS] TOKEN"). Tokens are now hexadecimal.
    """
    service = ApexService(tmp_path / "service")
    tokens = [
        service.approvals.request(f"p{i}", {"simulator_runs": 10, "runs": 1}, requested_by="t")
        for i in range(200)
    ]

    assert all(t.isalnum() and not t.startswith("-") for t in tokens)
    approved = CliRunner().invoke(app, ["approve", tokens[0], "--root", str(tmp_path / "service")])
    assert approved.exit_code == 0, approved.output


def test_an_invalid_service_toml_is_one_line_not_a_traceback(tmp_path: Path) -> None:
    """Regression: `apex service status` printed a traceback for a `service.toml` with
    `[agents.monitor] auto = true` on one line (invalid TOML); the entry point now turns
    any APEX error into one line and exit code 1.
    """
    import subprocess
    import sys

    root = tmp_path / "service"
    root.mkdir()
    (root / "service.toml").write_text("[agents.monitor] auto = true\n")

    done = subprocess.run(
        [sys.executable, "-c", "from apex.cli.main import cli; cli()",
         "service", "status", "--root", str(root)],
        capture_output=True, text=True, check=False,
    )  # fmt: skip

    assert done.returncode == 1
    assert "Traceback" not in done.stderr
    assert done.stderr.strip().startswith(f"error: invalid {root / 'service.toml'}")


def test_a_slow_stop_is_reported_as_still_stopping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review of 2026-10-05: `apex service stop` said "did not stop" after 10 s while the
    service was still ending its work; it waits longer and says it is still stopping."""
    from apex.cli import services_commands
    from apex.services import jobs

    killed: list[int] = []
    monkeypatch.setattr(jobs, "service_pid", lambda root: 4242)  # it never ends here
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append(pid))
    monkeypatch.setattr(services_commands, "STOP_WAIT_S", 0.3)

    result = CliRunner().invoke(app, ["service", "stop", "--root", str(tmp_path)])

    assert killed == [4242]
    assert result.exit_code == services_commands.STILL_STOPPING  # not an error's 1
    assert "still stopping" in result.output
    assert "did not stop" not in result.output


def test_a_service_that_does_not_start_in_time_is_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """2026-10-06: under a loaded test suite the service took ~30 s to start; `service start`
    gave up after 10 s and left it running unattended (stray services for hours)."""
    import subprocess
    import sys

    started: list[subprocess.Popen[bytes]] = []
    real = subprocess.Popen

    def never_ready(command: list[str], **kwargs: object) -> subprocess.Popen[bytes]:
        # A process that runs but never writes its pid, in its own session as the service.
        process = real([sys.executable, "-c", "import time; time.sleep(120)"],
                       start_new_session=True)  # fmt: skip
        started.append(process)
        return process

    monkeypatch.setattr("apex.cli.services_commands.subprocess.Popen", never_ready)
    monkeypatch.setenv("APEX_SERVICE_START_TIMEOUT", "1")
    result = CliRunner().invoke(app, ["service", "start", "--root", str(tmp_path / "svc")])
    assert result.exit_code != 0
    assert "did not start" in result.output
    # The release tester's H27: the error names the wait and the way to give more, and the
    # log is not empty.
    assert "within 1 s" in result.output
    assert "--wait" in result.output
    assert "no process is left" in result.output
    log = (tmp_path / "svc" / "service.log").read_text()
    assert "launching the service" in log
    assert "did not start" in log
    assert started
    assert started[0].wait(timeout=15) is not None  # stopped, not left running

    started.clear()
    monkeypatch.setenv("APEX_SERVICE_START_TIMEOUT", "300")  # --wait comes first
    waited = CliRunner().invoke(app, ["service", "start", "--root", str(tmp_path / "svc"),
                                      "--wait", "1"])  # fmt: skip
    assert "within 1 s" in waited.output
    assert started[0].wait(timeout=15) is not None


def test_a_service_that_exits_at_once_says_its_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess
    import sys

    real = subprocess.Popen

    def exits(command: list[str], **kwargs: object) -> subprocess.Popen[bytes]:
        return real([sys.executable, "-c", "raise SystemExit(3)"], start_new_session=True)

    monkeypatch.setattr("apex.cli.services_commands.subprocess.Popen", exits)
    result = CliRunner().invoke(app, ["service", "start", "--root", str(tmp_path / "svc"),
                                      "--wait", "10"])  # fmt: skip
    assert result.exit_code != 0
    assert "exited with code 3" in result.output
    assert "exited with code 3" in (tmp_path / "svc" / "service.log").read_text()


def test_the_start_wait_is_longer_with_the_rest_api(monkeypatch: pytest.MonkeyPatch) -> None:
    from apex.cli import services_commands

    monkeypatch.delenv("APEX_SERVICE_START_TIMEOUT", raising=False)
    assert services_commands._start_timeout() == services_commands.START_TIMEOUT_S
    assert services_commands._start_timeout(True) == services_commands.HTTP_START_TIMEOUT_S
    monkeypatch.setenv("APEX_SERVICE_START_TIMEOUT", "300")
    assert services_commands._start_timeout(True) == 300.0


def test_a_free_reuse_that_must_run_after_all_is_gated(
    tmp_path: Path, linear_case: CaseFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review of release tester N8: a reuse let through as free (its job queued or running)
    whose job ended before ``_reuse`` looked again was restarted without approval."""
    service = ApexService(tmp_path / "service", approval_threshold_runs=50)
    case = linear_case(ensemble_size=20)  # 100 simulator runs
    first = service.submit_run(case.config, approve=lambda cost: True)
    assert service.wait(first.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
    run_dir = Path(service.jobs.record(first.id)["run_dirs"][0])
    # As if its job had failed before its first checkpoint: a reuse starts it again.
    (run_dir / "summary.json").unlink()
    (run_dir / "state.json").unlink()
    monkeypatch.setattr(ApexService, "_free_reuse", lambda self, identity, path: True)

    with pytest.raises(ApprovalRequired, match="need approval"):
        service.submit_run(case.config)

"""``apex submit``, ``apex jobs`` and ``apex code``, through the embedded service."""

import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

import pytest
import typer
from typer.testing import CliRunner

from apex.cli.job_commands import detect_kind
from apex.cli.main import app
from apex.core.errors import ConfigError
from apex.services import code as code_module
from apex.services.api import ApexService
from apex.storage.sqlite_store import JOB_PROTOCOL

N_DATA = 8  # the linear case's (conftest.py)
GIES = (
    'name = "gies"\nstrategy = { name = "rlmmac", options = { energy = 1.0 } }\n'
    "schedule = { data_mismatch_tolerance = 1e-9 }\nmax_iterations = 6"
)


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path


CaseFactory = Callable[..., Case]
runner = CliRunner()


def apex(*args: str) -> str:
    result = runner.invoke(app, list(args))
    assert result.exit_code == 0, result.output
    return result.output


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.org", *args],
        check=True, capture_output=True,
    )  # fmt: skip


def _job(output: str) -> str:
    """The id on the line ``<kind> job <id>: ...`` of apex submit's output."""
    found = re.search(r"^\w+ job (\S+?):? ", output, re.MULTILINE)
    assert found is not None, output
    return found.group(1)


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A services root whose frozen copies come from a small repository, synced by a stub."""
    repo = tmp_path / "repo"
    source = repo / "src" / "apex" / "storage" / "sqlite_store.py"
    source.parent.mkdir(parents=True)
    source.write_text(f'"""Store."""\n\nJOB_PROTOCOL = {JOB_PROTOCOL}\n')
    (repo / ".gitignore").write_text(".venv/\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "first")

    def sync(copy: Path, extras: Sequence[str], log: Path) -> None:
        python = copy / ".venv" / "bin" / "python"
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
        python.chmod(0o755)

    monkeypatch.setattr(code_module, "uv_sync", sync)
    monkeypatch.delenv("APEX_CLI_CODE")  # the default, frozen
    root = tmp_path / "service"
    root.mkdir()
    (root / "service.toml").write_text(
        f'[jobs]\nrepository = "{repo}"\ncode_dir = "{tmp_path / "code"}"\n'
    )
    return root


def test_a_run_is_submitted_frozen_by_default_and_listed(
    root: Path, linear_case: CaseFactory
) -> None:
    case = linear_case()
    repo = root.parent / "repo"
    (repo / "src" / "apex" / "storage" / "sqlite_store.py").write_text("# an edit\n")

    result = runner.invoke(app, ["submit", str(case.config), "--wait", "--root", str(root)])

    assert result.exit_code == 0, result.output
    assert "1 uncommitted changes are not in it (src/apex/storage/sqlite_store.py)" in result.output
    job = _job(result.output)
    record = ApexService(root).jobs.record(job)
    assert record["state"] == "complete"
    commit = record["code"]["commit"]
    assert f"code {commit}" in result.output
    assert f"folder: {case.directory / 'runs' / 'case'}" in result.output  # as apex run
    listed = apex("jobs", "list", "--root", str(root))
    assert f"{job}  run" in listed
    assert commit in listed
    assert commit in apex("code", "list", "--root", str(root))

    again = apex("submit", str(case.config), "--root", str(root))
    assert "(an identical run: reused)" in again
    assert '"state": "complete"' in apex("jobs", "show", job[:20], "--root", str(root))


def test_a_run_started_outside_the_service_is_resumed_by_its_folder(
    root: Path, linear_case: CaseFactory
) -> None:
    case = linear_case(method=GIES, ensemble_size=10)
    apex("run", str(case.config), "--max-steps", "1")
    run_dir = case.directory / "runs" / "case"

    output = apex("submit", "--resume", str(run_dir), "--code", "live", "--wait",
                  "--root", str(root))  # fmt: skip

    assert "resume job" in output
    assert "job complete" in output
    record = ApexService(root).jobs.record(_job(output))
    assert record["code"] is None
    # A run the service knows resumes through its last job.
    again = apex("submit", "--resume", str(run_dir), "--code", "live", "--root", str(root))
    assert "resume job" in again


def test_predictions_and_optimizations_are_told_apart_by_their_sections(
    root: Path, linear_case: CaseFactory
) -> None:
    case = linear_case()
    apex("submit", str(case.config), "--code", "live", "--wait", "--root", str(root))
    outputs = ", ".join(f'["t{i}", "y"]' for i in range(N_DATA))
    prediction = case.directory / "future.toml"
    prediction.write_text(
        '[prediction]\nname = "future"\nrun_dir = "runs/case"\nensembles = ["posterior"]\n'
        '[prediction.outputs]\nvectors = ["y"]\n'
        f'[simulator]\noptions = {{ forward_file = "forward.npz", outputs = [{outputs}] }}\n'
    )
    optimization = case.directory / "rosenbrock.toml"
    optimization.write_text(
        '[optimization]\nname = "rosen"\nseed = 1\n'
        '[[controls.variables]]\nname = "x"\ninitial = [-1.2, 1.0]\nlower = -2.0\n'
        "upper = 2.0\nvariance = 0.001\n"
        '[objective]\nname = "rosenbrock"\n'
        '[optimizer]\nname = "enopt"\noptions = { ne = 10, maxiter = 3 }\n'
    )
    assert detect_kind(case.config) == "run"
    assert detect_kind(prediction) == "prediction"
    assert detect_kind(optimization) == "optimization"

    predicted = apex("submit", str(prediction), "--code", "live", "--wait", "--root", str(root))
    optimized = apex("submit", str(optimization), "--code", "live", "--wait", "--root", str(root))

    assert predicted.startswith("prediction job")
    assert optimized.startswith("optimization job")
    assert "job complete" in predicted
    assert "job complete" in optimized
    # Placed as apex optimize places it.
    assert f"folder: {case.directory / 'runs' / 'rosen'}" in optimized


def test_a_file_of_no_known_kind_is_refused(tmp_path: Path) -> None:
    neither = tmp_path / "x.toml"
    neither.write_text("[run]\nname = 'x'\n")
    with pytest.raises(ConfigError, match="give --kind"):
        detect_kind(neither)
    # The exception, not Rich's error panel, which CI colours and wraps.
    result = runner.invoke(app, ["submit", "--root", str(tmp_path / "root")], standalone_mode=False)
    assert isinstance(result.exception, typer.BadParameter)
    assert "--resume RUN_DIR" in str(result.exception)


def test_a_queued_job_is_stopped_by_apex_jobs_stop(root: Path, linear_case: CaseFactory) -> None:
    (root / "service.toml").write_text(
        (root / "service.toml").read_text() + "[service]\nmax_concurrent_jobs = 1\n"
    )
    slow = linear_case(name="slow", ensemble_size=10, simulator_extra="delay_s = 0.05")
    first = _job(apex("submit", str(slow.config), "--code", "live", "--root", str(root)))
    queued = _job(apex("submit", str(linear_case().config), "--code", "live", "--root", str(root)))

    assert "cancelled" in apex("jobs", "stop", queued, "--root", str(root))
    service = ApexService(root)
    assert service.wait(first, timeout_s=240, poll_s=0.2).state == "complete"
    assert service.jobs.record(queued)["state"] == "cancelled"


def test_every_cell_of_a_frozen_study_runs_one_commit(root: Path, linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=20)
    study = case.directory / "study.toml"
    study.write_text(
        '[study]\nname = "pinned"\nseed = 3\nreplicates = 2\n'
        'metrics = ["normalized_data_mismatch:all"]\n'
        '[study.cases]\nlinear = "config.toml"\n'
        '[[study.variants]]\nname = "four"\n'
        '[[study.variants]]\nname = "two"\n'
        "patch = { method = { options = { inflation = [2, 2] } } }\n"
    )

    apex("study", "run", str(study), "--wait", "--root", str(root))  # frozen by default

    service = ApexService(root)
    cells = [r for r in service.jobs.all_records() if r["kind"] == "run"]
    assert len(cells) == 4
    assert all(r["state"] == "complete" for r in cells)
    assert len({r["code"]["commit"] for r in cells}) == 1


def test_without_a_git_checkout_the_default_runs_live_code(
    root: Path, linear_case: CaseFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(code_module, "default_repository", lambda: None)
    (root / "service.toml").write_text("")  # no [jobs] repository either
    case = linear_case()

    output = apex("submit", str(case.config), "--wait", "--root", str(root))

    assert "does not run from a git checkout, so the job runs live code" in output
    assert "code live" in output
    other = linear_case(name="other", ensemble_size=30)  # not a reuse: it must run
    refused = runner.invoke(
        app, ["submit", str(other.config), "--code", "frozen", "--root", str(root)]
    )
    assert refused.exit_code == 1, refused.output
    assert "set [jobs] repository" in refused.output


def _approval_root(tmp_path: Path) -> Path:
    """A services root (live code) where more than 10 simulator runs need approval."""
    root = tmp_path / "limits"
    root.mkdir()
    (root / "service.toml").write_text("[approval]\nmax_simulator_runs = 10\n")
    return root


def test_without_a_terminal_the_pending_token_is_printed(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """Release tester N7 (2026-10-20): without a terminal's answer the question printed
    "Aborted." and no token, so scripts could not take the token route."""
    root = _approval_root(tmp_path)
    config = str(linear_case(ensemble_size=20).config)  # 100 simulator runs

    refused = runner.invoke(app, ["submit", config, "--root", str(root)], input="")

    assert refused.exit_code == 1, refused.output
    found = re.search(r"pending approval token: (\w+)", refused.output)
    assert found is not None, refused.output
    token = found.group(1)
    assert f"apex approve {token}" in refused.output
    assert f"--approval-token {token}" in refused.output
    assert "approval_token=" not in refused.output  # the API's form (release tester C9)
    assert ApexService(root).jobs.all_records() == []
    apex("approve", token, "--root", str(root))
    submitted = apex("submit", config, "--approval-token", token, "--wait", "--root", str(root))
    assert "complete" in submitted

    # Release tester N8: the identical run is reused, and running nothing needs no approval.
    again = runner.invoke(app, ["submit", config, "--root", str(root)], input="")
    assert again.exit_code == 0, again.output
    assert "(an identical run: reused)" in again.output
    assert "approval" not in again.output


def test_a_grant_approves_a_submission_of_its_client_and_experiment(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """Release tester N4 (2026-10-20): `apex submit` had no --client or --annotation, so a
    grant (docs/service.md's walkthrough) could not apply to it."""
    import datetime as dt

    root = _approval_root(tmp_path)
    service = ApexService(root)
    grant = service.grants.create("planner-x", "exp-042", 3000, dt.timedelta(days=1), by="t")
    config = str(linear_case(ensemble_size=20).config)

    result = runner.invoke(
        app,
        ["submit", config, "--client", "planner-x", "--annotation", "experiment=exp-042",
         "--root", str(root)],
        input="",
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    record = service.jobs.record(_job(result.output))
    assert record["annotations"]["experiment"] == "exp-042"
    assert record["client"] == "planner-x"
    [used] = service.grants.records()
    assert used["id"] == grant["id"]
    assert used["used_simulator_runs"] == 100
    bad = runner.invoke(app, ["submit", config, "--annotation", "experiment", "--root", str(root)])
    assert bad.exit_code != 0
    assert "key=value" in bad.output

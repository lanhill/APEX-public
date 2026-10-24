"""Notification commands (the run monitor's fields)."""

from pathlib import Path

import pytest

from apex.services.notify import notify
from apex.services.settings import NotificationSection


def test_one_command_serves_job_ends_and_monitor_reports(tmp_path: Path) -> None:
    """Regression: a command with a monitor field ({status}) raised KeyError at a job's end."""
    settings = NotificationSection(command=f"touch {tmp_path}/{{job}}.{{state}}.{{status}}")
    record = {"id": "j1", "kind": "run", "state": "complete"}

    notify(settings, record)  # a job ended: no {status}
    notify(settings, {**record, "state": "running"}, {"status": "warning"})

    assert sorted(p.name for p in tmp_path.iterdir()) == ["j1.complete.", "j1.running.warning"]


def test_field_values_stay_single_arguments(tmp_path: Path) -> None:
    # A summary with spaces and shell characters is one argument, never shell text.
    out = tmp_path / "out.txt"
    script = tmp_path / "echo.sh"
    script.write_text(f'#!/bin/sh\nprintf "%s|" "$@" > {out}\n')
    script.chmod(0o755)
    settings = NotificationSection(command=f"{script} {{job}} {{summary}}")

    notify(
        settings, {"id": "j1", "kind": "run", "state": "running"}, {"summary": "a; rm -rf ~ $(x)"}
    )

    assert out.read_text() == "j1|a; rm -rf ~ $(x)|"


def test_a_failing_command_is_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    script = tmp_path / "fail.sh"
    script.write_text("#!/bin/sh\necho 'Teams webhook answered HTTP 404' >&2\nexit 1\n")
    script.chmod(0o755)

    notify(NotificationSection(command=str(script)), {"id": "j1", "kind": "run", "state": "failed"})

    [record] = [r for r in caplog.records if getattr(r, "event", None) == "notify_failed"]
    assert "exited with 1: Teams webhook answered HTTP 404" in record.getMessage()

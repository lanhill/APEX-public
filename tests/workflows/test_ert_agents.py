"""APEX's agents for ERT users: what APEX_DIAGNOSE and APEX_REPORT run."""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import pytest

from apex.core.errors import ConfigError
from apex.integrations.ert import agents
from apex.services import composition


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path


CaseFactory = Callable[..., Case]


def test_diagnosis_and_report_are_written_into_the_run(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    case = linear_case(ensemble_size=20)
    composition.start(case.config)
    run_dir = case.directory / "runs" / "case"

    diagnosis = agents.diagnose(run_dir, root=tmp_path / "service")
    report = agents.report(run_dir, root=tmp_path / "service")

    assert diagnosis == run_dir / "agents" / "diagnosis.md"
    assert diagnosis.read_text().startswith(f"# Diagnosis of {run_dir.name}")
    assert "Decided by rules" in diagnosis.read_text()
    assert report == run_dir / "report" / "narrative.md"
    assert report.read_text().strip()


def test_an_exported_case_is_imported_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = tmp_path / "exported"
    case.mkdir()
    (case / "apex_export.json").write_text("{}")
    imported: list[Path] = []

    def import_case(case_dir: Path, run_dir: Path) -> None:
        imported.append(run_dir)
        run_dir.mkdir()
        (run_dir / "state.json").write_text("{}")

    monkeypatch.setattr("apex.integrations.ert.importer.import_case", import_case)
    assert agents.run_directory(case) == case / "apex_run"
    assert agents.run_directory(case) == case / "apex_run"  # imported once
    assert imported == [case / "apex_run"]
    with pytest.raises(ConfigError, match="neither"):
        agents.run_directory(tmp_path)

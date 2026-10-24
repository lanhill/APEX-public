"""The release tester's first findings on APEX-public (2026-10-19), each as a test."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import pytest
from typer.testing import CliRunner

from apex.cli.main import app
from apex.services import composition


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    config: Path
    directory: Path


CaseFactory = Callable[..., Case]


def test_plugins_and_schema_need_no_service_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A service.toml of another installation, with agents this one does not have.
    home = tmp_path / "home"
    home.mkdir()
    (home / "service.toml").write_text("[agents.no_such_agent]\nauto = true\n")
    monkeypatch.setenv("APEX_HOME", str(home))
    assert CliRunner().invoke(app, ["plugins"]).exit_code == 0
    result = CliRunner().invoke(app, ["schema", "optimization"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["title"] == "OptimizationConfig"


@pytest.mark.parametrize("kind", ["optimization_study", "service"])
def test_every_documented_configuration_kind_has_a_schema(kind: str) -> None:
    result = CliRunner().invoke(app, ["schema", kind])
    assert result.exit_code == 0, result.output


def test_an_unknown_schema_lists_the_kinds() -> None:
    result = CliRunner().invoke(app, ["schema", "nonsense"])
    assert result.exit_code != 0
    assert "optimization" in result.output


def test_a_stale_agents_section_names_what_to_remove(tmp_path: Path) -> None:
    from apex.core.errors import ConfigError
    from apex.services.settings import load_settings

    # An agent of a package not installed here (the public release lacks private ones).
    (tmp_path / "service.toml").write_text("[agents.no_such_agent]\nauto = true\n")
    with pytest.raises(
        ConfigError, match=r"\[agents\.no_such_agent\].*Remove or correct those sections"
    ):
        load_settings(tmp_path)


def test_qaqc_of_an_unsaved_iteration_is_a_plain_error(linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=20)
    composition.start(case.config)
    run = next((case.directory / "runs").iterdir())
    result = CliRunner().invoke(app, ["qaqc", str(run), "--iteration", "99"])
    assert result.exit_code == 1
    assert "has no saved iteration 99; its iterations are 0," in result.output
    assert "Traceback" not in result.output


def test_the_report_lists_each_metric_once_and_says_which_std(linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=20)
    composition.start(case.config)
    run = next((case.directory / "runs").iterdir())
    result = CliRunner().invoke(app, ["report", str(run)])
    assert result.exit_code == 0, result.output
    rows = [
        line.split()[:2] for line in result.output.splitlines() if line.startswith("data_mismatch")
    ]
    assert len(rows) == len({tuple(r) for r in rows})  # no row twice
    assert "population std (ddof = 0)" in result.output


def test_a_run_folder_is_refused_with_how_to_start_another(linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=20)
    composition.start(case.config)
    result = CliRunner().invoke(app, ["run", str(case.config), "--no-progress"])
    assert result.exit_code != 0
    assert "[run] name" in result.output


def test_each_data_mismatch_label_has_a_figure_of_its_own(linear_case: CaseFactory) -> None:
    # The data types' shares once wrote over the total's figure (same file name).
    pytest.importorskip("matplotlib")
    case = linear_case(ensemble_size=20)
    composition.start(case.config)
    run = next((case.directory / "runs").iterdir())
    result = CliRunner().invoke(app, ["report", str(run), "--plots"])
    assert result.exit_code == 0, result.output
    written = [line for line in result.output.splitlines() if "metrics/" in line]
    assert len(written) == len(set(written))  # no file written twice
    metrics = run / "report" / "metrics"
    assert (metrics / "data_mismatch.png").is_file()
    assert list(metrics.glob("metric_data_mismatch_*.png"))  # the shares, apart


def test_the_service_schema_is_the_command_lines_only(tmp_path: Path) -> None:
    # It names every installed agent's settings, private ones too: never through
    # the API, the tools or the web UI.
    from apex.core.errors import ConfigError
    from apex.services.api import ApexService

    with pytest.raises(ConfigError, match="unknown schema 'service'"):
        ApexService(tmp_path).schema("service")  # type: ignore[arg-type]
    assert CliRunner().invoke(app, ["schema", "service"]).exit_code == 0

"""The release tester's second findings on APEX-public (2026-10-19), each as a test."""

import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from apex.cli.main import app


def test_service_start_refuses_a_port_another_program_uses(tmp_path: Path) -> None:
    # Before: "started" was printed, the server then failed to bind and the service ended,
    # while requests (and their token) went to whatever held the port.
    with socket.socket() as other:
        other.bind(("127.0.0.1", 0))
        other.listen()
        port = other.getsockname()[1]
        result = CliRunner().invoke(
            app, ["service", "start", "--root", str(tmp_path), "--http", f"127.0.0.1:{port}"]
        )
    assert result.exit_code != 0
    assert "Another program may use that port" in result.output
    assert not (tmp_path / "service.log").exists()  # nothing was started


@pytest.mark.parametrize(
    "example",
    [
        "examples/optimization/rosenbrock_enopt.toml",
        "examples/toy/multi_objective/zdt1/zdt1_nsga2.toml",
    ],
)
def test_validate_takes_optimizations(example: str) -> None:
    root = Path(__file__).resolve().parents[2]
    result = CliRunner().invoke(app, ["validate", str(root / example)])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("optimization ")


def test_a_plugin_whose_module_is_missing_is_flagged_unavailable() -> None:
    from types import MappingProxyType

    from apex.services.plugins import _unavailable

    class Needs:
        requires = MappingProxyType({"no_such_module_for_apex": "nothing"})

    class Has:
        requires = MappingProxyType({"numpy": "core"})

    assert _unavailable(Needs) == "needs no_such_module_for_apex (the nothing extra)"
    assert _unavailable(Has) is None
    assert _unavailable(object) is None  # declares nothing


def test_an_analytic_optimization_reports_its_evaluations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The report said "0 evaluations" for every analytic objective (no journal).
    import shutil

    monkeypatch.setenv("APEX_HOME", str(tmp_path / "home"))
    root = Path(__file__).resolve().parents[2]
    config = tmp_path / "rosenbrock_enopt.toml"
    shutil.copy(root / "examples/optimization/rosenbrock_enopt.toml", config)
    ran = CliRunner().invoke(app, ["optimize", str(config), "--max-iterations", "3",
                                   "--no-progress"])  # fmt: skip
    assert ran.exit_code == 0, ran.output
    assert "steps accepted), " in ran.output
    report = CliRunner().invoke(app, ["report", str(tmp_path / "runs" / "rosenbrock_enopt")])
    assert report.exit_code == 0, report.output
    line = report.output.splitlines()[0]
    count = int(line.rsplit(", ", 1)[1].split()[0])
    assert count > 0
    assert line.endswith("objective evaluations")


def test_the_monitor_falls_back_to_its_rules_without_a_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, linear_case: Callable[..., Any]
) -> None:
    # [agents.llm] names a provider without its model: the rules answer, with a warning
    # (the REST routes' behaviour), instead of the command failing.
    from apex.services import composition

    home = tmp_path / "home"
    home.mkdir()
    (home / "service.toml").write_text('[agents.llm]\nprovider = "openai_compatible"\n')
    monkeypatch.setenv("APEX_HOME", str(home))
    case = linear_case(ensemble_size=20)
    composition.start(case.config)
    run = next((case.directory / "runs").iterdir())
    result = CliRunner().invoke(app, ["agent", "monitor", str(run)])
    assert result.exit_code == 0, result.output
    assert "the agent's rules answer" in result.output
    assert ": nothing to do" in result.output or ": let it continue" in result.output


def test_the_start_check_asks_the_port_itself_not_a_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from apex.cli.services_commands import _answers

    class Health(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = json.dumps({"status": "ok"}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Health)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")  # nothing listens there
        monkeypatch.delenv("no_proxy", raising=False)
        assert _answers("127.0.0.1", server.server_address[1])
    finally:
        server.shutdown()


def test_service_start_refuses_port_zero(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["service", "start", "--root", str(tmp_path), "--http", "127.0.0.1:0"]
    )
    assert result.exit_code != 0
    assert "port from 1 to 65535" in result.output

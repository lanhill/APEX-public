"""``apex agent`` on a scripted model configured in ``service.toml``."""

import json
from pathlib import Path

from typer.testing import CliRunner

from apex.cli.main import app


def _root(tmp_path: Path, llm: str) -> Path:
    root = tmp_path / "service"
    root.mkdir()
    (root / "service.toml").write_text(f"[agents.llm]\n{llm}\n")
    return root


def test_ask_answers_with_the_configured_model_and_traces_it(tmp_path: Path) -> None:
    root = _root(
        tmp_path,
        'provider = "stub"\nmodel = "scripted"\n'
        'options = { replies = [\'{"answer": "adaptive, the prior-only default"}\'] }',
    )

    result = CliRunner().invoke(app, ["agent", "ask", "Which localization?", "--root", str(root)])

    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == "adaptive, the prior-only default"
    [decision] = (root / "agents").glob("ask-*-decision.json")
    trace = json.loads(decision.read_text())
    assert (trace["model"], trace["decided_by"]) == ("scripted", "model")


def test_ask_falls_back_when_the_model_is_unreachable(tmp_path: Path) -> None:
    # Port 9 (discard) on the loopback: nothing listens there, so the connection is refused.
    root = _root(tmp_path, 'model = "qwen3:32b"\nbase_url = "http://127.0.0.1:9/v1"')

    result = CliRunner().invoke(app, ["agent", "ask", "Hello?", "--root", str(root)])

    assert result.exit_code == 0, result.output
    assert "no answer" in result.stdout
    assert "fallback: the provider failed" in result.stderr


def test_ask_refuses_a_provider_without_its_model(tmp_path: Path) -> None:
    root = _root(tmp_path, 'provider = "ollama"')

    result = CliRunner().invoke(app, ["agent", "ask", "Hello?", "--root", str(root)])

    assert result.exit_code == 1
    assert "'ollama' needs model" in result.stderr


def test_providers_are_listed() -> None:
    result = CliRunner().invoke(app, ["agent", "providers"])

    assert result.exit_code == 0
    names = [line.split()[0] for line in result.stdout.splitlines()]
    assert names == ["anthropic", "claude_code", "ollama", "openai_compatible", "stub"]

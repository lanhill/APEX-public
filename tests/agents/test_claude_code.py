"""The ``claude_code`` provider, with a fake CLI: no Claude Code is run.

The recorded answer is the shape ``claude -p --output-format json`` gave on 2026-09-29
(Claude Code 2.1.284). The opt-in smoke test at the end runs the real CLI.
"""

import json
import os
import stat
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import BaseModel

from apex.agents.loop import AgentLoop, Fallback
from apex.agents.providers import create_provider
from apex.agents.providers.claude_code import ClaudeCodeProvider, Run
from apex.agents.trace import CallRecorder
from apex.core.llm import LLMError, LLMInvalidOutput, LLMTimeout, LLMUnavailable, Message
from apex.services.api import ApexService
from apex.services.settings import LLMSection
from apex.services.tools import ToolSession


def _answer(result: str, **fields: object) -> str:
    return json.dumps({
        "type": "result", "subtype": "success", "is_error": False, "result": result,
        "stop_reason": "end_turn", "num_turns": 1, "session_id": "72e134ce-44ac",
        "usage": {"input_tokens": 453, "output_tokens": 9}, **fields,
    })  # fmt: skip


@pytest.fixture(autouse=True)
def cache_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep the CLI's working directories out of the real ~/.cache."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return tmp_path / "cache"


class FakeCLI:
    """Records each call and answers from a script."""

    def __init__(self, *answers: Run | Exception) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[list[str], str, float, str]] = []
        self.envs: list[Mapping[str, str]] = []

    def __call__(
        self, argv: list[str], stdin: str, timeout_s: float, cwd: str, env: Mapping[str, str]
    ) -> Run:
        self.calls.append((argv, stdin, timeout_s, cwd))
        self.envs.append(env)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


CONVERSATION = [
    Message("system", "You monitor runs."),
    Message("user", "How is the run?"),
    Message("assistant", '{"tool": "run_summary", "arguments": {"run_id": "abc"}}'),
    Message("user", 'Result of run_summary: {"ok": true}', name="run_summary"),
]


def test_the_cli_runs_as_a_bare_model_in_an_empty_directory(cache_home: Path) -> None:
    cli = FakeCLI(Run(0, _answer('{"ok": true}'), ""))
    provider = ClaudeCodeProvider("opus", effort="low", runner=cli)

    completion = provider.complete(CONVERSATION, max_tokens=900, temperature=0.0)

    [(argv, stdin, timeout_s, cwd)] = cli.calls
    assert argv[:4] == ["claude", "-p", "--output-format", "json"]
    expected = {
        "--tools": "",
        "--setting-sources": "",
        "--system-prompt": "You monitor runs.",
        "--model": "opus",
        "--effort": "low",
    }
    for flag, value in expected.items():
        assert argv[argv.index(flag) + 1] == value
    assert {
        "--strict-mcp-config", "--safe-mode", "--no-session-persistence", "--disable-slash-commands"
    } <= set(argv)  # fmt: skip
    assert "[user]\nHow is the run?" in stdin
    assert "[tool result (run_summary)]" in stdin
    assert "You monitor runs." not in stdin  # the system prompt goes in --system-prompt
    assert timeout_s == 180.0  # one attempt's
    assert not Path(cwd).exists()  # a temporary, empty directory, removed afterwards
    base = cache_home / "apex" / "claude"
    assert Path(cwd).parent == base  # under a folder only the user may write
    assert stat.S_IMODE(base.stat().st_mode) == 0o700
    assert completion.text == '{"ok": true}'
    assert (completion.usage.input_tokens, completion.usage.output_tokens) == (453, 9)
    assert completion.raw_id == "72e134ce-44ac"
    assert not provider.capabilities.tool_calls  # APEX's loop runs the tools


@pytest.mark.parametrize(
    ("answer", "error"),
    [
        (FileNotFoundError("claude"), LLMUnavailable),
        (subprocess.TimeoutExpired("claude", 300), LLMTimeout),
        (Run(1, "", "Invalid API key · Please run /login"), LLMUnavailable),
        (Run(0, _answer("Claude usage limit reached", is_error=True), ""), LLMUnavailable),
        # The subscription's weekly limit (2026-10-09): an outage of every agent on it.
        (
            Run(
                0,
                _answer("You've hit your weekly limit · resets 11am (Europe/Paris)", is_error=True),
                "",
            ),
            LLMUnavailable,
        ),
        (Run(0, _answer("the request was refused", is_error=True), ""), LLMError),
        (Run(2, "not json", "error: unknown option '--safe-mode'"), LLMUnavailable),
        (Run(2, "not json", "Error: something else broke"), LLMError),
        # The CLI's error when Claude attempts a native tool call (it has none): the model
        # is there and is asked again (2026-10-01, the web UI's assistant).
        (
            Run(
                0,
                _answer(
                    "The model's tool call could not be parsed (retry also failed).", is_error=True
                ),
                "",
            ),
            LLMInvalidOutput,
        ),
    ],
)
def test_failures_map_to_llm_errors(answer: Run | Exception, error: type[LLMError]) -> None:
    cli = FakeCLI(answer, answer)
    provider = ClaudeCodeProvider(runner=cli)

    with pytest.raises(error) as raised:
        provider.complete(CONVERSATION[:2], max_tokens=8, temperature=0.0)

    assert type(raised.value) is error
    # Only a hang is tried again; any other failure is final at once.
    assert len(cli.calls) == (2 if isinstance(answer, subprocess.TimeoutExpired) else 1)


class Default(BaseModel):
    """The answer of the loop test."""

    default_localization: str


def test_the_loop_runs_apex_tools_for_claude(tmp_path: Path) -> None:
    cli = FakeCLI(
        Run(0, _answer('{"tool": "list_plugins", "arguments": {"registry": "localizations"}}'), ""),
        Run(0, _answer('{"answer": {"default_localization": "adaptive"}}'), ""),
    )
    session = ToolSession(ApexService(tmp_path / "service"), client="agent-test")
    loop = AgentLoop(
        ClaudeCodeProvider(runner=cli), CallRecorder(tmp_path, "t"), session=session,
        tools=["list_plugins"],
    )  # fmt: skip

    outcome = loop.run(
        "Which is the default localization?",
        Default,
        Fallback("-", lambda: Default(default_localization="-")),
    )

    assert not outcome.fell_back
    assert outcome.decision.default_localization == "adaptive"
    assert outcome.tool_calls == [
        {"tool": "list_plugins", "arguments": {"registry": "localizations"}, "ok": True}
    ]
    second_stdin = cli.calls[1][1]
    assert "Result of list_plugins" in second_stdin
    assert "adaptive_per_update" in second_stdin  # the dispatcher's result went back


def test_the_cli_gets_no_keys_or_session_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    # An ANTHROPIC_API_KEY would move the calls from the Claude Code login to API billing,
    # and a parent Claude Code session's variables would hand the child its channel.
    variables = {
        "ANTHROPIC_API_KEY": "sk-ant", "ANTHROPIC_AUTH_TOKEN": "t", "ANTHROPIC_BASE_URL": "http://x",
        "CLAUDECODE": "1", "CLAUDE_CODE_MESSAGING_SOCKET": "/tmp/s", "CLAUDE_EFFORT": "max",
        "GROQ_API_KEY": "g", "SSH_AUTH_SOCK": "/tmp/a", "PATH": "/usr/bin",
    }  # fmt: skip
    for name, value in variables.items():
        monkeypatch.setenv(name, value)
    cli = FakeCLI(Run(0, _answer("ok"), ""))
    ClaudeCodeProvider(runner=cli).complete(CONVERSATION[:2], max_tokens=8, temperature=0.0)
    [env] = cli.envs
    for name in variables:
        if name != "PATH":
            assert name not in env, name
    assert env["PATH"] == "/usr/bin"
    assert env["HOME"] == os.environ["HOME"]  # where the Claude Code login lives


def test_no_working_directory_makes_the_provider_unavailable(cache_home: Path) -> None:
    cache_home.mkdir()
    cache_home.chmod(0o500)  # read-only: the work folder cannot be made
    try:
        with pytest.raises(LLMUnavailable, match="no working directory"):
            ClaudeCodeProvider(runner=FakeCLI(Run(0, _answer("ok"), ""))).complete(
                CONVERSATION[:2], max_tokens=8, temperature=0.0
            )
    finally:
        cache_home.chmod(0o700)


def test_a_relative_xdg_cache_home_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from apex.agents.providers.claude_code import work_dir

    monkeypatch.setenv("XDG_CACHE_HOME", "relative/cache")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert work_dir() == tmp_path / "home" / ".cache" / "apex" / "claude"


def test_an_existing_work_folder_is_tightened(cache_home: Path) -> None:
    base = cache_home / "apex" / "claude"
    base.mkdir(parents=True, mode=0o755)
    base.chmod(0o755)
    ClaudeCodeProvider(runner=FakeCLI(Run(0, _answer("ok"), ""))).complete(
        CONVERSATION[:2], max_tokens=8, temperature=0.0
    )
    assert stat.S_IMODE(base.stat().st_mode) == 0o700


def test_the_provider_is_configured_by_agents_llm() -> None:
    provider = create_provider(
        LLMSection(provider="claude_code", model="opus", options={"effort": "high"})
    )

    assert provider.describe() == {
        "provider": "claude_code", "model": "opus", "command": "claude", "effort": "high",
    }  # fmt: skip
    assert create_provider(LLMSection(provider="claude_code")).model_id == "sonnet"


@pytest.mark.skipif(
    not os.environ.get("APEX_TEST_CLAUDE_CODE"),
    reason="opt-in: set APEX_TEST_CLAUDE_CODE=1 to run the real Claude Code CLI",
)
def test_claude_code_answers_through_the_loop(tmp_path: Path) -> None:
    class Ok(BaseModel):
        ok: bool

    outcome = AgentLoop(ClaudeCodeProvider(), CallRecorder(tmp_path, "smoke")).run(
        'Reply with the JSON object {"ok": true}.', Ok, Fallback("-", lambda: Ok(ok=False))
    )
    assert not outcome.fell_back, outcome.reason
    assert outcome.decision.ok


def test_a_hung_call_is_tried_again() -> None:
    """A test session through the web UI (2026-10-05): calls answer in seconds, but now and then the
    CLI hung until the 300 s timeout. An attempt has 120 s, and a hung one is retried."""
    cli = FakeCLI(subprocess.TimeoutExpired("claude", 120), Run(0, _answer("fine"), ""))
    provider = ClaudeCodeProvider(runner=cli)

    completion = provider.complete(CONVERSATION[:2], max_tokens=8, temperature=0.0)

    assert completion.text == "fine"
    assert len(cli.calls) == 2
    never = ClaudeCodeProvider(runner=FakeCLI(subprocess.TimeoutExpired("claude", 1)),
                               hang_retries=0)  # fmt: skip
    with pytest.raises(LLMTimeout, match="1 attempt"):
        never.complete(CONVERSATION[:2], max_tokens=8, temperature=0.0)

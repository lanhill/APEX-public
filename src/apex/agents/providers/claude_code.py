"""The ``claude_code`` provider: Claude through the Claude Code CLI.

It runs ``claude -p`` (headless), so agents use the Claude subscription or login of the
person running APEX, and need no API key. Claude Code is used as the model only:

- none of its own tools (``--tools ""``), MCP servers (``--strict-mcp-config``), settings or
  hooks (``--setting-sources ""``) or skills, and no saved session;
- ``--safe-mode``, which also turns off ``CLAUDE.md`` files, skills, installed plugins, hooks
  and output styles; admin-managed (policy) settings still apply;
- its system prompt is replaced by the agent's (``--system-prompt``);
- it runs in a new, empty directory under ``$XDG_CACHE_HOME/apex/claude`` (default
  ``~/.cache``), which only the user may write, so no project's ``CLAUDE.md`` is near it;
- its environment is APEX's without keys and credentials (so an ``ANTHROPIC_API_KEY`` cannot
  move the calls from the Claude Code login to API billing) and without a parent Claude Code
  session's variables (``CLAUDECODE``, ``CLAUDE_CODE_*``: its messaging socket, its effort).
  For API billing, use the ``anthropic`` provider instead.

Tools stay APEX's: the provider declares no native tool calling, so the agent loop uses
prompted JSON and runs every tool call through its own dispatcher (approval tokens,
budgets, traces). The conversation goes to ``claude`` on stdin as a transcript. The CLI has
no temperature, seed or output-token options, so those are not sent. Every call takes a
few seconds to start, and counts against the account's Claude Code usage.
"""

import json
import logging
import os
import re
import signal
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from apex.core.llm import (
    Capabilities,
    Completion,
    LLMError,
    LLMInvalidOutput,
    LLMTimeout,
    LLMUnavailable,
    Message,
    ToolSpec,
    Usage,
)
from apex.core.plugins import llm_providers

NAME = "claude_code"

#: Words in the CLI's errors that mean "not now" (login, limits, overload): the agent falls
#: back to its rule rather than treating the request as wrong.
# "You've hit your weekly limit · resets 11am" (2026-10-09): the subscription's limits.
_UNAVAILABLE = ("login", "logged in", "authenticat", "rate limit", "usage limit", "overloaded",
                "credit", "network", "connection", "timed out", "weekly limit", "hit your limit",
                "session limit")  # fmt: skip


@dataclass(frozen=True)
class Run:
    """What running the CLI gave: exit code, stdout and stderr."""

    returncode: int
    stdout: str
    stderr: str


#: Runs the CLI: ``(argv, stdin, timeout_s, cwd, env) -> Run``. Tests pass a fake.
Runner = Callable[[list[str], str, float, str, Mapping[str, str]], Run]

#: Variables the CLI never gets: Anthropic's (a key or an endpoint would move the calls from
#: the Claude Code login to API billing), a parent Claude Code session's, and agent sockets.
_DROPPED = re.compile(
    r"^(ANTHROPIC_.*|CLAUDECODE|CLAUDE_CODE_.*|CLAUDE_EFFORT|CLAUDE_PID|SSH_AUTH_SOCK)$"
)
#: Names that hold secrets (other providers' and services' keys): never passed on.
_SECRET = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)


def cli_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """Return the CLI's environment: ``environ`` without keys and session variables."""
    return {k: v for k, v in environ.items() if not _DROPPED.match(k) and not _SECRET.search(k)}


def work_dir() -> Path:
    """Return the folder for the CLI's empty working directories (only the user may write it)."""
    cache = os.environ.get("XDG_CACHE_HOME", "")
    # The XDG specification: a relative value is ignored.
    root = Path(cache) if cache and Path(cache).is_absolute() else Path.home() / ".cache"
    base = root / "apex" / "claude"
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    base.chmod(0o700)  # also when it existed already
    return base


def _subprocess(
    argv: list[str], stdin: str, timeout_s: float, cwd: str, env: Mapping[str, str]
) -> Run:
    done = subprocess.run(
        argv, input=stdin, capture_output=True, text=True, timeout=timeout_s, cwd=cwd,
        env=dict(env), check=False,
    )  # fmt: skip
    return Run(done.returncode, done.stdout, done.stderr)


class CallStopped(LLMError):
    """The CLI was terminated because the task's caller asked to stop (the web UI's Stop)."""


#: How often a running CLI is checked for a stop request.
STOP_POLL_S = 0.25


def _stoppable(
    argv: list[str],
    stdin: str,
    timeout_s: float,
    cwd: str,
    env: Mapping[str, str],
    should_stop: Callable[[], bool],
) -> Run:
    """Run the CLI as :func:`_subprocess`, but kill it as soon as ``should_stop`` is true.

    Raises
    ------
    CallStopped
        When stopped.
    subprocess.TimeoutExpired
        After ``timeout_s``, as :func:`_subprocess`.
    """
    deadline = time.monotonic() + timeout_s
    # Its own process group, so that a stop kills the CLI's children too (they could hold
    # its pipes open, and the last read would then wait for them).
    with subprocess.Popen(
        argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=cwd, env=dict(env), start_new_session=True,
    ) as process:  # fmt: skip
        given: str | None = stdin
        while True:
            try:
                out, err = process.communicate(given, timeout=STOP_POLL_S)
                return Run(process.returncode, out, err)
            except subprocess.TimeoutExpired:
                given = None  # the input went with the first call (retrying loses nothing)
                stopped = should_stop()
                if stopped or time.monotonic() > deadline:
                    _kill_group(process)
                    if stopped:
                        raise CallStopped(f"{NAME}: stopped by the caller") from None
                    raise subprocess.TimeoutExpired(argv, timeout_s) from None


def _kill_group(process: "subprocess.Popen[str]") -> None:
    """Kill a process and its group, and collect what is left within a few seconds."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()
    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        logger.warning("the CLI's pipes stayed open after it was killed",
                       extra={"event": "claude_code_pipes_open"})  # fmt: skip


def transcript(messages: Sequence[Message]) -> str:
    """Return the conversation (without system messages) as the prompt for ``claude -p``."""
    parts = []
    for message in messages:
        if message.role == "system":
            continue
        if message.role == "tool" or (message.role == "user" and message.name):
            label = f"tool result ({message.name or 'tool'})"
        else:
            label = message.role
        parts.append(f"[{label}]\n{message.content}")
    return (
        "The conversation so far:\n\n"
        + "\n\n".join(parts)
        + "\n\nReply with the assistant's next message only."
    )


logger = logging.getLogger(__name__)
#: One attempt's time limit. A call answers in seconds (3-10 s in a first test session's
#: round, prompts of 20-32k characters); the 300 s timeouts of that round (issues #41, #89,
#: #98) were the CLI hanging at random, so a shorter attempt (still ample for a slow,
#: high-effort call) and a retry answer sooner.
DEFAULT_TIMEOUT_S = 180.0
DEFAULT_HANG_RETRIES = 1


class ClaudeCodeProvider:
    """Claude through ``claude -p`` (see the module docstring)."""

    def __init__(
        self,
        model: str = "sonnet",
        command: str = "claude",
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_context: int = 200_000,
        effort: str | None = None,
        runner: Runner | None = None,
        hang_retries: int = DEFAULT_HANG_RETRIES,
    ) -> None:
        self.model = model
        self.command = command
        self.timeout_s = timeout_s
        self.hang_retries = max(0, int(hang_retries))
        self.effort = effort
        self._capabilities = Capabilities(
            tool_calls=False, json_mode=False, max_context=max_context
        )
        self._run = runner or _subprocess
        #: Set by the caller for a task it may stop (the web UI's Stop): the default runner
        #: then kills the CLI at once when it turns true (a test's runner is used as given).
        self.should_stop: Callable[[], bool] | None = None

    @property
    def model_id(self) -> str:
        """The model alias or name given to ``claude --model`` (``sonnet``, ``opus``, ...)."""
        return self.model

    @property
    def provider_name(self) -> str:
        """``claude_code``."""
        return NAME

    @property
    def capabilities(self) -> Capabilities:
        """No native tool calling: APEX's loop prompts for JSON and runs the tools itself."""
        return self._capabilities

    def describe(self) -> dict[str, Any]:
        """Return the settings for traces (no credentials are involved)."""
        return {
            "provider": NAME,
            "model": self.model,
            "command": self.command,
            "effort": self.effort,
        }

    def argv(self, system: str) -> list[str]:
        """Return the command line of one call."""
        argv = [
            self.command, "-p", "--output-format", "json",
            "--tools", "", "--strict-mcp-config", "--setting-sources", "", "--safe-mode",
            "--no-session-persistence", "--disable-slash-commands",
            "--system-prompt", system, "--model", self.model,
        ]  # fmt: skip
        if self.effort is not None:
            argv += ["--effort", self.effort]
        return argv

    def _attempts(self, argv: list[str], stdin: str, cwd: str, env: Mapping[str, str]) -> Run:
        """Run the CLI, again after an attempt that hung (it answers in seconds when well)."""
        for attempt in range(1, self.hang_retries + 2):
            try:
                if self._run is _subprocess and self.should_stop is not None:
                    return _stoppable(argv, stdin, self.timeout_s, cwd, env, self.should_stop)
                return self._run(argv, stdin, self.timeout_s, cwd, env)
            except subprocess.TimeoutExpired:
                if attempt > self.hang_retries:
                    raise
                _hung(attempt, self.timeout_s)
        raise AssertionError("unreachable")  # pragma: no cover

    def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        response_model: type[BaseModel] | None = None,
        max_tokens: int,
        temperature: float,
        seed: int | None = None,
    ) -> Completion:
        """Return Claude's answer (see :class:`apex.core.llm.LLMProvider`)."""
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        argv = self.argv(system or "You are a helpful assistant.")
        env = cli_environment(os.environ)
        try:
            # A new, empty directory: no project's CLAUDE.md or settings are near it.
            workspace = tempfile.TemporaryDirectory(prefix="call-", dir=work_dir())
        except OSError as error:
            raise LLMUnavailable(f"{NAME}: no working directory for the CLI: {error}") from error
        try:
            with workspace as cwd:
                run = self._attempts(argv, transcript(messages), cwd, env)
        except FileNotFoundError as error:
            raise LLMUnavailable(
                f"{NAME}: {self.command!r} not found; install Claude Code, or set "
                '[agents.llm] options = {{ command = "/path/to/claude" }}'
            ) from error
        except subprocess.TimeoutExpired as error:
            tries = 1 + self.hang_retries
            raise LLMTimeout(
                f"{NAME}: no answer in {self.timeout_s:.0f} s ({tries} attempt(s))"
            ) from error
        except OSError as error:  # the CLI cannot be run, or its directory not removed
            raise LLMUnavailable(f"{NAME}: {error}") from error
        return parse_run(run)


def _hung(attempt: int, timeout_s: float) -> None:
    logger.warning(
        "%s: no answer in %.0f s (attempt %d): the CLI hung; trying again", NAME, timeout_s,
        attempt, extra={"event": "llm_hang_retry", "provider": NAME, "attempt": attempt},
    )  # fmt: skip


def parse_run(run: Run) -> Completion:
    """Return the completion of a ``claude -p --output-format json`` run."""
    try:
        body = json.loads(run.stdout) if run.stdout.strip() else None
    except json.JSONDecodeError:
        body = None
    if not isinstance(body, dict):
        detail = (run.stderr or run.stdout).strip()[:500] or f"exit code {run.returncode}"
        raise _error(detail)
    if body.get("is_error") or body.get("subtype") != "success":
        raise _error(str(body.get("result") or body.get("subtype") or "error")[:500])
    usage = body.get("usage") or {}
    return Completion(
        text=str(body.get("result") or ""),
        usage=Usage(usage.get("input_tokens"), usage.get("output_tokens")),
        raw_id=body.get("session_id"),
        stop_reason=body.get("stop_reason"),
    )


def _error(detail: str) -> LLMError:
    text = detail.lower()
    if "unknown option" in text:
        # An older CLI without a flag the provider needs, such as --safe-mode.
        return LLMUnavailable(
            f"{NAME}: {detail}; upgrade Claude Code (the provider needs --safe-mode; tested "
            "with 2.1.285)"
        )
    if "could not be parsed" in text:
        # The CLI's own error when Claude attempts a native tool call (it has none here):
        # the model is there, and asked again in APEX's JSON form it usually complies.
        return LLMInvalidOutput(f"{NAME}: {detail}")
    kind = LLMUnavailable if any(word in text for word in _UNAVAILABLE) else LLMError
    return kind(f"{NAME}: {detail}")


@llm_providers.register(NAME)
def claude_code(
    model: str = "sonnet",
    command: str = "claude",
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_context: int = 200_000,
    effort: str | None = None,
    hang_retries: int = DEFAULT_HANG_RETRIES,
) -> ClaudeCodeProvider:
    """Return a provider running Claude through the Claude Code CLI.

    ``timeout_s`` is one attempt's; an attempt with no answer by then is hung (calls take
    seconds) and is tried again, ``hang_retries`` times.
    """
    return ClaudeCodeProvider(model, command, timeout_s, max_context, effort,
                              hang_retries=hang_retries)  # fmt: skip

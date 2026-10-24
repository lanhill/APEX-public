"""Traces of runtime agents: every model call and every decision.

A :class:`CallRecorder` writes one JSON file per model call to ``<directory>/agents/``:
the provider's settings (never keys), the model id, the time, the tokens used, a SHA-256
of the full request, the request itself (the prompt), the answer, and its tool calls. The
decision an agent reaches gets its own file: its inputs, the parsed decision, whether the
rule-based fallback decided and why, and the fallback's rule and thresholds. Each file is
also logged as a structured event (``llm_call``, ``agent_decision``), without the prompt.

The directory is the run's or the job's (or ``<services root>/agents`` for a person's
questions), so that a decision is kept with what it was about.
"""

import dataclasses
import datetime as dt
import hashlib
import itertools
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from apex.core.llm import Completion, LLMProvider, Message, ToolSpec

logger = logging.getLogger(__name__)


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds")


def request_document(
    provider: LLMProvider,
    messages: Sequence[Message],
    tools: Sequence[ToolSpec],
    response_model: type[BaseModel] | None,
    max_tokens: int,
    temperature: float,
    seed: int | None,
) -> dict[str, Any]:
    """Return a request as data: what the hash and the trace cover."""
    return {
        "model": provider.model_id,
        "messages": [dataclasses.asdict(m) for m in messages],
        "tools": [t.name for t in tools],
        "tool_schemas": {t.name: dict(t.parameters) for t in tools},
        "response_schema": None if response_model is None else response_model.model_json_schema(),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
    }


def request_hash(document: Mapping[str, Any]) -> str:
    """Return the SHA-256 of a request document (canonical JSON)."""
    text = json.dumps(document, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


class CallRecorder:
    """Writes an agent's traces (see the module docstring).

    Parameters
    ----------
    directory
        Where ``agents/`` is created: a run's or job's directory, or a services root.
    agent
        The agent's name, the prefix of its files.
    """

    def __init__(self, directory: Path, agent: str) -> None:
        self.directory = Path(directory) / "agents"
        self.agent = agent
        self._stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%f")
        self._counter = itertools.count(1)
        self.calls: list[str] = []
        #: The advice in force for the decision, ``"<advisor>:<id>"`` (:mod:`apex.core.advice`).
        self.advice: list[str] = []

    def _write(self, kind: str, document: Mapping[str, Any]) -> tuple[str, Path]:
        self.directory.mkdir(parents=True, exist_ok=True)
        name = f"{self.agent}-{self._stamp}-{next(self._counter):03d}-{kind}"
        path = self.directory / f"{name}.json"
        path.write_text(json.dumps(document, indent=2, default=str))
        return name, path

    def record_call(
        self,
        provider: LLMProvider,
        request: Mapping[str, Any],
        *,
        completion: Completion | None,
        error: Exception | None,
        duration_s: float,
    ) -> str:
        """Record one model call; return its id (the file's name)."""
        digest = request_hash(request)
        answer = None
        if completion is not None:
            answer = {
                "text": completion.text,
                "tool_calls": [dataclasses.asdict(c) for c in completion.tool_calls],
                "raw_id": completion.raw_id,
                "stop_reason": completion.stop_reason,
            }
        usage = completion.usage if completion is not None else None
        document = {
            "kind": "llm_call",
            "agent": self.agent,
            "time": _now(),
            "provider": provider.describe(),
            "model": provider.model_id,
            "duration_s": round(duration_s, 3),
            "usage": None if usage is None else dataclasses.asdict(usage),
            "request_sha256": digest,
            "request": dict(request),
            "answer": answer,
            "error": None if error is None else f"{type(error).__name__}: {error}",
        }
        call, path = self._write("call", document)
        self.calls.append(call)
        logger.info(
            "llm call",
            extra={
                "event": "llm_call",
                "agent": self.agent,
                "call": call,
                "provider": provider.provider_name,
                "model": provider.model_id,
                "request_sha256": digest,
                "input_tokens": None if usage is None else usage.input_tokens,
                "output_tokens": None if usage is None else usage.output_tokens,
                "tool_calls": [c.name for c in completion.tool_calls] if completion else [],
                "error": document["error"],
                "trace": str(path),
            },
        )
        return call

    def record_decision(
        self,
        provider: LLMProvider | None,
        *,
        task: str,
        decision: BaseModel,
        fell_back: bool,
        reason: str | None,
        rule: str,
        thresholds: Mapping[str, Any],
        tool_calls: Sequence[Mapping[str, Any]],
    ) -> Path:
        """Record an agent's decision; return the file."""
        document = {
            "kind": "agent_decision",
            "agent": self.agent,
            "time": _now(),
            "provider": None if provider is None else provider.describe(),
            "model": None if provider is None else provider.model_id,
            "task": task,
            "decision": decision.model_dump(mode="json"),
            # Without a provider the rules decided: no model was asked.
            "decided_by": "fallback" if fell_back else "model" if provider else "rules",
            "fallback_reason": reason,
            "fallback_rule": rule,
            "thresholds": dict(thresholds),
            "tool_calls": list(tool_calls),
            "llm_calls": list(self.calls),
            "advice": list(self.advice),
        }
        _, path = self._write("decision", document)
        logger.info(
            "agent decision",
            extra={
                "event": "agent_decision",
                "agent": self.agent,
                "model": document["model"],
                "decided_by": document["decided_by"],
                "fallback_reason": reason,
                "trace": str(path),
            },
        )
        return path

    def record_action(self, action: str, **fields: Any) -> Path:  # noqa: ANN401
        """Record what an agent did, or refused to do, after a decision; return the file."""
        document = {"kind": "agent_action", "agent": self.agent, "time": _now(),
                    "action": action, **fields}  # fmt: skip
        _, path = self._write("action", document)
        logger.info(
            "agent action: %s",
            action,
            extra={"event": "agent_action", "agent": self.agent, "action": action,
                   "trace": str(path)},
        )  # fmt: skip
        return path

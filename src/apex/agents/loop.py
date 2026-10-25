"""APEX's own agent loop: plan, call tools, validate, decide, record.

:meth:`AgentLoop.run` gives a model a task, lets it call APEX's tools (the registry of
through its dispatcher, so approval tokens, budgets, bounded results and the call
log apply as for any client), and returns a decision validated by a pydantic model. It
works with weak models as with strong ones:

- **Tools:** native tool calling where the provider has it; otherwise *prompted JSON*: the
  tool schemas go in the prompt, the model answers ``{"tool": ..., "arguments": {...}}``
  or ``{"answer": {...}}``, and the arguments are validated with the tool's model.
- **Validation:** every final answer is validated with ``response_model`` before APEX acts
  on it. An invalid answer, or an invalid prompted tool call, gets its error back, up to
  ``max_retries`` times in a row.
- **Fallback:** when the provider fails (unavailable, timed out, refused), when answers stay
  invalid, or when the turns run out, the task's rule-based :class:`Fallback` decides. Its
  rule and thresholds are recorded with the decision.
- **Context:** the conversation is trimmed to the provider's context, oldest tool results
  first.
- **Traces:** every call and the decision are recorded (:mod:`apex.agents.trace`).

The loop has no framework dependency; frameworks can drive APEX from outside through the
tools instead.
"""

import contextvars
import json
import logging
import math
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ValidationError

from apex.agents.streaming import AnswerText
from apex.agents.trace import CallRecorder, request_document
from apex.core.errors import ApexError
from apex.core.llm import (
    Completion,
    LLMError,
    LLMInvalidOutput,
    LLMProvider,
    LLMTimeout,
    LLMUnavailable,
    Message,
    StreamingProvider,
    ToolCall,
    ToolSpec,
)
from apex.services.tools import TOOLS, ToolSession, bound, call

logger = logging.getLogger(__name__)

#: Characters per token when estimating a conversation's size. English prose gives 3-4,
#: but agent conversations are dense JSON with numbers and hex ids: measured on the M-5Spots
#: monitor's requests to Groq (qwen3.8-27b, 2026-09-29), 2.2 and 1.7. 1.5 errs on the side of
#: fitting; with a large context it only trims a little earlier.
CHARS_PER_TOKEN = 1.5

#: What a trimmed tool result is replaced with.
TRIMMED = '{"trimmed": "an older tool result, dropped to fit the model\'s context"}'

SYSTEM = (
    "You are an agent of APEX, a platform for ensemble history matching of reservoir "
    "models. Use the tools to find what you need; never guess values a tool can give. "
    "Validate before you submit anything, and submit only with the approval token that "
    "validate returned; a pending token needs a person to run `apex approve TOKEN`."
)


@dataclass(frozen=True)
class Fallback[T: BaseModel]:
    """The rule-based decision a task falls back on.

    Attributes
    ----------
    rule
        What the rule does, in words (recorded), e.g. "stop at the CV minimum".
    decide
        Returns the rule's decision.
    thresholds
        The rule's named thresholds, recorded with the decision.
    """

    rule: str
    decide: Callable[[], T]
    thresholds: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LocalTool:
    """A tool an agent defines itself, beside the registry's.

    The chat assistant offers APEX's agents this way: they live in the ``agents`` layer,
    which the tool registry (in ``services``) cannot import. A local tool reads only; it is
    never a way around approval, since spending stays in the registry's tools.

    Attributes
    ----------
    name, description
        As the model sees them.
    arguments
        The pydantic model the arguments are validated with.
    handler
        Returns the result (JSON-serialisable), given the validated arguments.
    """

    name: str
    description: str
    arguments: type[BaseModel]
    handler: Callable[[Any], Any]


#: Set inside a bounded tool's thread: an event the loop sets when it abandons the tool (its
#: time limit, or a stop), so that a tool that can be cut short, such as one running in a
#: worker process, stops instead of running on.
_CANCEL: contextvars.ContextVar[threading.Event | None] = contextvars.ContextVar(
    "tool_cancel", default=None
)


def tool_cancelled() -> threading.Event | None:
    """Return the running tool's cancel event (set once the loop abandons it), if bounded."""
    return _CANCEL.get()


#: Why a task stopped when its caller asked (the web UI's Stop).
STOPPED = "stopped by the caller"


class _StopRequested(Exception):
    """Raised inside a task when its caller asked it to stop; ends it as a fallback."""


#: Called with an event name and its fields while a task runs: ``model_call`` (turn),
#: ``tool_call`` (tool, arguments), ``tool_result`` (tool, ok, result: an excerpt, see
#: :func:`result_excerpt`). Progress only.
EventHook = Callable[[str, Mapping[str, Any]], None]

#: The most of a tool result a ``tool_result`` event carries (characters of JSON), and of
#: any one of its fields.
EXCERPT_LIMIT = 2000
EXCERPT_FIELD_LIMIT = 500
#: How many left-out fields an excerpt names (each to 40 characters); the rest are counted.
EXCERPT_NAMED = 10


def result_excerpt(
    result: Mapping[str, Any], limit: int = EXCERPT_LIMIT, field_limit: int = EXCERPT_FIELD_LIMIT
) -> dict[str, Any]:
    """Return a tool result's short fields, whole, and the names of the others.

    Whoever follows a turn from its events can then hold an answer to what the tools said,
    e.g. a plan's ``approval``, without the events carrying whole drafts or tables (a
    reported issue: a reader of the events saw only ``ok`` and called a statement taken
    from a tool result invented). A long mapping (a local tool's
    ``result``) is excerpted in turn, within what is left of ``limit``. Fields are kept
    whole or left out, never cut, so the excerpt is valid JSON that reads as complete:
    ``_omitted`` names the first :data:`EXCERPT_NAMED` left out and ``_omitted_count``
    counts them all, so that a mapping of many short entries stays within ``limit``.
    """
    kept: dict[str, Any] = {}
    omitted: list[str] = []
    size = 0
    for key, value in result.items():
        text = json.dumps(value, default=str)
        room = limit - size - len(str(key)) - 6
        if len(text) <= field_limit and len(text) <= room:
            kept[key] = value
            size += len(text) + len(str(key)) + 6
        elif isinstance(value, Mapping) and room > field_limit:
            inner = result_excerpt(value, room, field_limit)
            kept[key] = inner
            size += len(json.dumps(inner, default=str)) + len(str(key)) + 6
        else:
            omitted.append(str(key))
    if omitted:
        kept["_omitted"] = [name[:40] for name in omitted[:EXCERPT_NAMED]]
        kept["_omitted_count"] = len(omitted)
    return kept


@dataclass(frozen=True)
class LoopSettings:
    """Bounds of one task.

    Attributes
    ----------
    max_turns
        Model calls per task, tool rounds included.
    max_retries
        Invalid answers in a row that get their error back before the fallback decides
        (default 2).
    refusal_turns
        Model calls added to ``max_turns``, one per failed tool call (a planner's refusal,
        invalid arguments, a missing file), up to this many, so that looking things up and
        one refusal cannot use up a task before the model revises its call. 0: none.
    max_tokens, temperature, seed
        Passed to the provider with every call.
    """

    max_turns: int = 8
    max_retries: int = 2
    refusal_turns: int = 0
    max_tokens: int = 2048
    #: A tool running longer is abandoned (it runs on in its thread; its result is dropped)
    #: and reported as timed out. ``None``: no limit.
    tool_timeout_s: float | None = None
    temperature: float = 0.0
    seed: int | None = None


@dataclass(frozen=True)
class Outcome[T: BaseModel]:
    """What a task ended with.

    Attributes
    ----------
    decision
        The validated decision (the model's, or the fallback's).
    fell_back
        Whether the fallback decided.
    reason
        Why it did (``None`` when the model decided).
    model_calls
        Calls made to the provider.
    tool_calls
        The tools the model called: name, arguments and whether they succeeded.
    trace
        The decision's trace file.
    unavailable
        Whether it fell back because the provider was unavailable (``LLMUnavailable``: it
        could not be reached, a usage limit, a timeout), not for another failure.
    timed_out
        Whether that was a timeout (``LLMTimeout``): the model took too long to answer.
    """

    decision: T
    fell_back: bool
    reason: str | None
    model_calls: int
    tool_calls: list[dict[str, Any]]
    trace: Path
    unavailable: bool = False
    timed_out: bool = False


class _Invalid(Exception):
    """An answer that does not validate; its message goes back to the model."""


class AgentLoop:
    """Runs tasks on one provider (see the module docstring).

    Parameters
    ----------
    provider
        The model.
    recorder
        Where calls and decisions are traced.
    session
        The tool session (client, model, budget). ``None``: the task gets no tools.
    tools
        Names of the tools offered (default: all).
    settings
        Bounds of each task.
    """

    def __init__(
        self,
        provider: LLMProvider,
        recorder: CallRecorder,
        *,
        session: ToolSession | None = None,
        tools: Sequence[str] | None = None,
        settings: LoopSettings | None = None,
        local_tools: Sequence[LocalTool] = (),
        on_event: EventHook | None = None,
        stream_answers: bool = False,
        should_stop: Callable[[], bool] | None = None,
        abandoned: dict[str, threading.Thread] | None = None,
    ) -> None:
        self.provider = provider
        #: Asked before each model call, each tool call and each streamed piece of the
        #: answer; true ends the task as a fallback with :data:`STOPPED` (the web UI's Stop).
        #: A call already under way is not cut short (but a stream is, at its next piece).
        self.should_stop = should_stop
        self._calls = 0
        #: Tools abandoned while running (a time limit or a stop), by name; a caller may
        #: share one mapping over several tasks (the chat: over its turns).
        self._abandoned: dict[str, threading.Thread] = {} if abandoned is None else abandoned
        self.recorder = recorder
        self.session = session
        names = list(TOOLS) if tools is None else list(tools)
        unknown = [n for n in names if n not in TOOLS]
        if unknown:
            raise ValueError(f"unknown tools {unknown}; available: {', '.join(TOOLS)}")
        self.local = {t.name: t for t in local_tools}
        #: Hand the answer's text to ``on_event`` (``answer_delta``) as it arrives, with a
        #: provider that streams; others answer at once, as before.
        self.stream_answers = stream_answers
        clash = sorted(set(self.local) & set(TOOLS))
        if clash:
            raise ValueError(f"local tools {clash} would shadow registry tools")
        self.tools = (
            []
            if session is None
            else [ToolSpec(n, TOOLS[n].description, TOOLS[n].schema()) for n in names]
        ) + [ToolSpec(t.name, t.description, t.arguments.model_json_schema()) for t in local_tools]
        self.on_event = on_event
        self.settings = settings or LoopSettings()
        self._tools_size = sum(
            len(t.name) + len(t.description) + len(json.dumps(t.parameters)) for t in self.tools
        )

    # ------------------------------------------------------------------ the loop

    def run[T: BaseModel](
        self,
        task: str,
        response_model: type[T],
        fallback: Fallback[T],
        *,
        system: str = SYSTEM,
        history: Sequence[Message] = (),
    ) -> Outcome[T]:
        """Return the decision on ``task``: the model's if valid, else the fallback's.

        ``history`` holds earlier turns of a conversation (user and assistant messages),
        placed between the system prompt and ``task``.
        """
        native = self.provider.capabilities.tool_calls
        messages = [
            Message("system", self._system(system, response_model, native)),
            *history,
            Message("user", task),
        ]
        tool_log: list[dict[str, Any]] = []
        limit = self.settings.max_turns
        reason = f"no valid answer in {limit} turns"
        try:
            return self._loop(task, messages, response_model, fallback, native, tool_log, limit)
        except _StopRequested:
            reason = STOPPED
        logger.info("agent stopped", extra={"event": "agent_stopped"})
        return self._outcome(task, fallback.decide(), True, reason, fallback, self._calls,
                             tool_log)  # fmt: skip

    def _loop[T: BaseModel](
        self,
        task: str,
        messages: list[Message],
        response_model: type[T],
        fallback: Fallback[T],
        native: bool,
        tool_log: list[dict[str, Any]],
        limit: int,
    ) -> Outcome[T]:
        """Run :meth:`run`'s model and tool calls (raises ``_StopRequested`` on a stop)."""
        invalid, calls, granted = 0, 0, 0
        self._calls = 0
        repeats: dict[str, tuple[int, str]] = {}  # call -> unchanged repeats, last result

        def refused(result: Mapping[str, Any]) -> None:
            nonlocal limit, granted
            if not result.get("ok") and granted < self.settings.refusal_turns:
                limit, granted = limit + 1, granted + 1

        reason = f"no valid answer in {limit} turns"
        unavailable = timed_out = False
        turn = -1
        while (turn := turn + 1) < limit:
            self._check_stop()
            reason = f"no valid answer in {limit} turns"
            # The last turn offers no tools and asks for the answer, so that a model busy
            # looking things up still decides instead of running out of turns.
            last = turn == limit - 1
            if last and self.tools and turn > 0:
                messages.append(Message("user", _LAST_TURN))
            messages = self._fit(messages)
            calls += 1
            self._calls = calls
            self._event("model_call", turn=turn + 1)
            try:
                completion = self._complete(messages, response_model, native, offer_tools=not last)
            except LLMInvalidOutput as error:
                # The model answered in a form the provider could not hand over (claude -p:
                # a native tool call): ask again, as for an invalid answer.
                invalid += 1
                if invalid > self.settings.max_retries:
                    reason = f"replies stayed unreadable ({invalid} in a row): {error}"
                    break
                messages.append(Message("user", _UNREADABLE + (_AGAIN if native else _PROMPTED)))
                continue
            except LLMError as error:
                self._check_stop()  # a call cut short by a stop (claude -p killed) is a stop
                reason = f"the provider failed: {error}"
                unavailable = isinstance(error, LLMUnavailable)
                timed_out = isinstance(error, LLMTimeout)
                break
            try:
                if native and completion.tool_calls:
                    if last:
                        raise _Invalid("no tools are left on the last turn")
                    messages.append(Message("assistant", completion.text, completion.tool_calls))
                    for tool_call in completion.tool_calls:
                        self._check_stop()
                        result = self._call_repeated(tool_call, tool_log, repeats)
                        refused(result)
                        messages.append(
                            Message("tool", json.dumps(result, default=str), (), tool_call.id,
                                    tool_call.name)
                        )  # fmt: skip
                    invalid = 0
                    continue
                answer = (
                    completion.parsed
                    if completion.parsed is not None
                    else _json_or_text(completion.text, response_model)
                )
                if not native and isinstance(answer, dict) and "tool" in answer:
                    if last:
                        raise _Invalid("no tools are left on the last turn")
                    tool_call, malformed = self._prompted_call(answer)
                    messages.append(Message("assistant", completion.text))
                    self._check_stop()
                    result = self._call_repeated(tool_call, tool_log, repeats)
                    refused(result)
                    messages.append(
                        Message("user", f"Result of {tool_call.name}: "
                                f"{json.dumps(result, default=str)}", name=tool_call.name)
                    )  # fmt: skip
                    if malformed:
                        # Recorded as a call, but still an invalid answer: a model that keeps
                        # sending bad arguments falls back after max_retries, as before.
                        invalid += 1
                        if invalid > self.settings.max_retries:
                            reason = f"answers stayed invalid ({invalid} in a row): {malformed}"
                            break
                    else:
                        invalid = 0
                    continue
                if not native and isinstance(answer, dict) and "answer" in answer:
                    answer = _unwrapped(answer, response_model)
                decision = self._validate(answer, response_model)
            except _Invalid as error:
                invalid += 1
                if invalid > self.settings.max_retries:
                    reason = f"answers stayed invalid ({invalid} in a row): {error}"
                    break
                messages.append(Message("assistant", completion.text))
                messages.append(Message("user", f"That answer is invalid: {error}. " + _AGAIN))
                continue
            return self._outcome(task, decision, False, None, fallback, calls, tool_log)
        logger.warning(
            "agent falls back: %s", reason, extra={"event": "agent_fallback", "reason": reason}
        )
        return self._outcome(task, fallback.decide(), True, reason, fallback, calls, tool_log,
                             unavailable=unavailable, timed_out=timed_out)  # fmt: skip

    # ------------------------------------------------------------------ parts

    def _system(self, system: str, response_model: type[BaseModel], native: bool) -> str:
        schema = json.dumps(response_model.model_json_schema())
        if native:
            return (
                f"{system}\n\nWhen you have the answer, reply with only a JSON object that "
                f"matches this JSON Schema, and nothing else:\n{schema}"
            )
        tools = "\n".join(
            f"- {t.name}: {t.description}\n  arguments schema: {json.dumps(t.parameters)}"
            for t in self.tools
        )
        protocol = (
            "Reply with only one JSON object, and nothing else. To call a tool: "
            '{"tool": "<name>", "arguments": {...}}. To give the answer: '
            '{"answer": <an object matching the answer schema>}.'
        )
        offered = f"\n\nTools:\n{tools}" if tools else "\n\nThere are no tools for this task."
        return f"{system}\n\n{protocol}{offered}\n\nAnswer schema:\n{schema}"

    def _complete(
        self,
        messages: Sequence[Message],
        response_model: type[BaseModel],
        native: bool,
        *,
        offer_tools: bool = True,
    ) -> Completion:
        s = self.settings
        tools = self.tools if native and offer_tools else []
        request = request_document(
            self.provider, messages, tools, response_model, s.max_tokens, s.temperature, s.seed
        )
        began = time.monotonic()
        try:
            if self.stream_answers and self.provider.capabilities.streaming:
                answer = AnswerText()

                def on_text(piece: str) -> None:
                    self._check_stop()  # ends the stream at its next piece
                    text = answer.feed(piece)
                    if text:
                        self._event("answer_delta", text=text)

                streaming = cast("StreamingProvider", self.provider)
                completion = streaming.stream_complete(
                    messages, on_text=on_text, tools=tools, response_model=response_model,
                    max_tokens=s.max_tokens, temperature=s.temperature, seed=s.seed,
                )  # fmt: skip
            else:
                completion = self.provider.complete(
                    messages,
                    tools=tools,
                    response_model=response_model,
                    max_tokens=s.max_tokens,
                    temperature=s.temperature,
                    seed=s.seed,
                )
        except LLMError as error:
            self.recorder.record_call(
                self.provider, request, completion=None, error=error,
                duration_s=time.monotonic() - began,
            )  # fmt: skip
            raise
        except _StopRequested:
            # A stream cut short by a stop: recorded as the call it was.
            self.recorder.record_call(
                self.provider, request, completion=None, error=LLMError(STOPPED),
                duration_s=time.monotonic() - began,
            )  # fmt: skip
            raise
        self.recorder.record_call(
            self.provider, request, completion=completion, error=None,
            duration_s=time.monotonic() - began,
        )  # fmt: skip
        return completion

    def _prompted_call(self, answer: Mapping[str, Any]) -> tuple[ToolCall, str]:
        """Return a prompted tool call to a known tool, and why its arguments are invalid.

        Invalid arguments are still run (and refused with the reason)
        as a native call's are, so the call is a ``tool_call``/``tool_result`` pair in the
        events and the decision's record; the reason (``""`` if valid) also counts the
        answer as invalid.
        """
        name = answer.get("tool")
        if not isinstance(name, str) or name not in {t.name for t in self.tools}:
            offered = ", ".join(t.name for t in self.tools) or "none"
            raise _Invalid(f"unknown tool {name!r}; the tools are: {offered}")
        arguments = answer.get("arguments", {})
        if not isinstance(arguments, dict):
            raise _Invalid("arguments must be a JSON object")
        model = self.local[name].arguments if name in self.local else TOOLS[name].arguments
        try:
            model.model_validate(arguments)
        except ValidationError as error:
            return ToolCall(f"prompted_{name}", name, arguments), (
                f"arguments of {name}: {_errors(error)}"
            )
        return ToolCall(f"prompted_{name}", name, arguments), ""

    def _call_repeated(
        self,
        tool_call: ToolCall,
        log: list[dict[str, Any]],
        repeats: dict[str, tuple[int, str]],
    ) -> dict[str, Any]:
        """Run a tool call, and say so when it repeats an earlier one with the same result.

        A model called list_runs eight times in a row, each time with the same arguments
        and the same empty result, until its turns ran out (release tester C7,
        2026-10-24). A repeat is still run, since a result may change (a job's state), but
        an unchanged one says so, so that the model stops asking.
        """
        key = json.dumps([tool_call.name, tool_call.arguments], sort_keys=True, default=str)
        count, previous = repeats.get(key, (0, ""))
        result = self._call_tool(tool_call, log)
        text = json.dumps(result, sort_keys=True, default=str)
        if previous and text == previous:
            count += 1
            repeats[key] = (count, text)
            return {**result, "repeated": _REPEATED.format(n=count + 1)}
        repeats[key] = (0, text)
        return result

    def _call_tool(self, tool_call: ToolCall, log: list[dict[str, Any]]) -> dict[str, Any]:
        """Run a tool through the dispatcher (approval, budgets, bounded results, call log)."""
        result: dict[str, Any]
        self._event("tool_call", tool=tool_call.name, arguments=tool_call.arguments)
        if tool_call.name in self.local and tool_call.arguments is not None:
            local, arguments = self.local[tool_call.name], tool_call.arguments
            result = self._bounded(tool_call.name, lambda: self._call_local(local, arguments))
        elif self.session is None:
            result = {"ok": False, "error": "this task has no tools"}
        elif tool_call.arguments is None:
            result = {"ok": False, "error": tool_call.error or "no arguments"}
        else:
            name, arguments, session = tool_call.name, tool_call.arguments, self.session
            result = self._bounded(name, lambda: call(name, arguments, session))
        ok = bool(result.get("ok"))
        # Why a call failed, briefly, for a person following the turn (a planner refusing
        # a design is the usual reason; the model then revises it).
        error = {} if ok else {"error": str(result.get("error", ""))[:300]}
        self._event(
            "tool_result",
            tool=tool_call.name,
            ok=ok,
            # ok and error are the event's own fields; the excerpt is the rest.
            result=result_excerpt({k: v for k, v in result.items() if k not in ("ok", "error")}),
            **error,
        )
        log.append(
            {"tool": tool_call.name, "arguments": tool_call.arguments, "ok": bool(result.get("ok"))}
        )
        return result

    @staticmethod
    def _call_local(tool: LocalTool, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Run a local tool; errors come back as results the model can act on."""
        try:
            parsed = tool.arguments.model_validate(dict(arguments))
        except ValidationError as error:
            return {"ok": False, "error": f"arguments of {tool.name}: {_errors(error)}"}
        try:
            return {"ok": True, "result": bound(tool.handler(parsed))}
        except (ApexError, ValueError, KeyError, OSError) as error:
            return {"ok": False, "error": f"{type(error).__name__}: {error}"}

    def _bounded(self, name: str, run: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        """Run a tool within ``tool_timeout_s``; past it, abandon it and say so.

        Python cannot stop a thread: an abandoned tool runs on (a daemon thread) and what it
        gives is dropped, unless it watches :func:`tool_cancelled` (a tool in a worker
        process is then killed). A stop requested meanwhile is seen at the next step.
        """
        limit = self.settings.tool_timeout_s or math.inf
        if limit == math.inf and self.should_stop is None:
            return run()
        running = self._abandoned.get(name)
        if running is not None and running.is_alive():
            # Its earlier call is still running after its time limit (or a stop): another
            # would only pile up threads.
            return {"ok": False, "error": f"{name} is still running from an earlier call that "
                    "took too long: try another way, or ask the user"}  # fmt: skip
        box: list[dict[str, Any]] = []
        failure: list[BaseException] = []
        cancel = threading.Event()

        def target() -> None:
            _CANCEL.set(cancel)  # in the copied context: this tool's only
            try:
                box.append(run())
            except BaseException as error:  # noqa: BLE001 (handed back to the caller's thread)
                failure.append(error)

        # The caller's context goes with the tool (e.g. the chat's turn number, which keeps a
        # stopped turn's late writes out).
        context = contextvars.copy_context()
        worker = threading.Thread(target=context.run, args=(target,), name=f"tool-{name}",
                                  daemon=True)  # fmt: skip
        worker.start()
        while worker.is_alive():
            worker.join(min(limit, 0.25))
            if not worker.is_alive():
                break
            if self.should_stop is not None and self.should_stop():
                self._abandoned[name] = worker
                cancel.set()
                raise _StopRequested
            limit -= 0.25
            if limit <= 0:
                self._abandoned[name] = worker
                cancel.set()
                logger.warning("tool abandoned after its time limit", extra={
                    "event": "tool_timeout", "tool": name,
                    "limit_s": self.settings.tool_timeout_s,
                })  # fmt: skip
                return {"ok": False, "error": (
                    f"{name} did not finish within {self.settings.tool_timeout_s or 0:g} s and "
                    "was abandoned: say so to the user, and try another way or ask them"
                )}  # fmt: skip
        if failure:
            raise failure[0]
        return box[0]

    def _check_stop(self) -> None:
        if self.should_stop is not None and self.should_stop():
            raise _StopRequested

    def _event(self, name: str, **fields: Any) -> None:  # noqa: ANN401
        if self.on_event is not None:
            try:
                self.on_event(name, fields)
            except Exception:
                logger.exception("agent event hook failed", extra={"event": "hook_failed"})

    @staticmethod
    def _validate[T: BaseModel](answer: object, response_model: type[T]) -> T:
        if isinstance(answer, response_model):
            return answer
        if not isinstance(answer, dict):
            raise _Invalid("the answer must be one JSON object matching the answer schema")
        try:
            return response_model.model_validate(answer)
        except ValidationError as error:
            raise _Invalid(_errors(error)) from error

    def _fit(self, messages: list[Message]) -> list[Message]:
        """Drop the oldest tool results until the conversation fits the context."""
        budget = self.provider.capabilities.max_context - self.settings.max_tokens
        chars = sum(len(m.content) + len(str(m.tool_calls)) for m in messages)
        size = int(chars / CHARS_PER_TOKEN)
        if self.provider.capabilities.tool_calls:
            # Native tool definitions travel with every request and take context too.
            size += int(self._tools_size / CHARS_PER_TOKEN)
        fitted = list(messages)
        for index, message in enumerate(fitted):
            if size <= budget:
                break
            is_result = message.role == "tool" or (message.role == "user" and message.name)
            if is_result and message.content != TRIMMED:
                size -= int((len(message.content) - len(TRIMMED)) / CHARS_PER_TOKEN)
                fitted[index] = replace(message, content=TRIMMED)
        if size > budget:
            # Say what is lost and what to change (release tester C8, 2026-10-24).
            logger.warning(
                "the conversation (~%d tokens) exceeds the context (%d) with every older tool "
                "result trimmed; the model may not see its start; raise the provider's "
                "max_context if the model has more", size, budget,
                extra={"event": "context_exceeded"},
            )  # fmt: skip
        return fitted

    def _outcome[T: BaseModel](
        self,
        task: str,
        decision: T,
        fell_back: bool,
        reason: str | None,
        fallback: Fallback[T],
        calls: int,
        tool_log: list[dict[str, Any]],
        *,
        unavailable: bool = False,
        timed_out: bool = False,
    ) -> Outcome[T]:
        trace = self.recorder.record_decision(
            self.provider,
            task=task,
            decision=decision,
            fell_back=fell_back,
            reason=reason,
            rule=fallback.rule,
            thresholds=fallback.thresholds,
            tool_calls=tool_log,
        )
        return Outcome(decision, fell_back, reason, calls, tool_log, trace, unavailable, timed_out)


_AGAIN = "Reply again with only the JSON object."
#: After a reply the provider could not read (:class:`~apex.core.llm.LLMInvalidOutput`).
_UNREADABLE = "Your last reply could not be read. "
_PROMPTED = (
    "You have no native tools here: to call a tool, reply with only "
    '{"tool": "<name>", "arguments": {...}}; to answer, {"answer": {...}}.'
)
_LAST_TURN = (
    "This is your last turn and no tools are left: give your answer now, as the JSON "
    "object, from what you know."
)


_REPEATED = (
    "you made this call {n} times with the same arguments and got the same result each "
    "time: unless you are waiting for something to change, answer from what you have, or "
    "try another tool or other arguments"
)


def _errors(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in e['loc']) or 'value'}: {e['msg']}" for e in error.errors()[:5]
    )


def _json(text: str) -> object:
    """Return the JSON object in a model's text: bare, fenced, or amid other text.

    Raises
    ------
    _Invalid
        If there is none.
    """
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL)
    candidates = [stripped]
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    start, end = stripped.find("{"), stripped.rfind("}")
    if 0 <= start < end:
        candidates.append(stripped[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise _Invalid("the reply contains no JSON object" if stripped else "the reply is empty")


def _unwrapped(answer: Mapping[str, Any], response_model: type[BaseModel]) -> object:
    """Return a prompted answer's decision: ``{"answer": <decision>}``.

    When the answer schema itself has a field ``answer`` (the chat's ``ChatAnswer``),
    models reply ``{"answer": "text"}``, meaning the decision, not its wrapper: that is kept
    whole when the unwrapped value would not validate but the whole object does.
    """
    inner = answer["answer"]
    if isinstance(inner, dict) or "answer" not in response_model.model_fields:
        return inner
    try:
        response_model.model_validate(dict(answer))
    except ValidationError:
        return inner
    return dict(answer)


def _json_or_text(text: str, response_model: type[BaseModel]) -> object:
    """Return the JSON object of a reply; for a one-text-field answer, the plain text too.

    Some models (Claude through ``claude -p``) end a conversation in prose rather
    than JSON. When the answer schema is a single required text field (the chat's reply),
    such prose is the answer; anything that looks like a tool call is still refused.
    """
    try:
        return _json(text)
    except _Invalid:
        call = _xml_tool_call(text)
        if call is not None:
            return call
        fields = response_model.model_fields
        required = [n for n, f in fields.items() if f.is_required()]
        if (
            len(fields) == 1
            and len(required) == 1
            and fields[required[0]].annotation is str
            and text.strip()
            and "<invoke" not in text
            and "function_calls" not in text
        ):
            return {required[0]: text.strip()}
        raise


_INVOKE = re.compile(r'<invoke name="([^"]+)">(.*?)</invoke>', re.DOTALL)
_PARAMETER = re.compile(r'<parameter name="([^"]+)">(.*?)</parameter>', re.DOTALL)


def _xml_tool_call(text: str) -> dict[str, Any] | None:
    """Read a tool call written in Claude's own XML form, as ``{"tool", "arguments"}``.

    Claude through ``claude -p`` sometimes calls a tool the way it would natively
    (``<invoke name="list_plugins"><parameter name="registry">metrics</parameter>``) rather
    than in the prompted JSON. Parameter values that are JSON are decoded; others are text.
    """
    match = _INVOKE.search(text)
    if match is None:
        return None
    arguments: dict[str, Any] = {}
    for name, value in _PARAMETER.findall(match.group(2)):
        value = value.strip()
        try:
            arguments[name] = json.loads(value)
        except json.JSONDecodeError:
            arguments[name] = value
    return {"tool": match.group(1), "arguments": arguments}

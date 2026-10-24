"""APEX's agent loop on scripted models. No live model is used (runtime-agent rules).

Covered: native tool calls, prompted JSON for models without them, invalid answers with
retries, the rule-based fallback (invalid answers, a timeout, no turns left), context
trimming, approval that no model bypasses, and a trace of every call and decision.
"""

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from apex.agents.loop import TRIMMED, AgentLoop, Fallback, LoopSettings, result_excerpt
from apex.agents.providers.stub import StubProvider, StubRequest
from apex.agents.trace import CallRecorder, request_hash
from apex.core.llm import Completion, LLMInvalidOutput, LLMTimeout, ToolCall
from apex.services.api import ApexService
from apex.services.tools import TOOLS, ToolSession, call


class Choice(BaseModel):
    """A decision the tests ask for."""

    localization: str
    reason: str = Field(min_length=3)


#: Recommend no localization when the ensemble is at least this many members (a stand-in
#: rule with a named threshold, as the runtime-agent rules require of every fallback).
FALLBACK = Fallback(
    "recommend distance localization unless the ensemble is large",
    lambda: Choice(localization="distance", reason="fallback rule"),
    thresholds={"large_ensemble": 200},
)

ANSWER = '{"localization": "adaptive", "reason": "the default, prior-only"}'


def _loop(
    tmp_path: Path, provider: StubProvider, **settings: int
) -> tuple[AgentLoop, CallRecorder]:
    session = ToolSession(
        ApexService(tmp_path / "service"), client="agent-test", model=provider.model_id
    )
    recorder = CallRecorder(tmp_path / "run", "advisor")
    return AgentLoop(
        provider, recorder, session=session, settings=LoopSettings(**settings)
    ), recorder


def _files(tmp_path: Path, kind: str) -> list[dict[str, object]]:
    paths = sorted((tmp_path / "run" / "agents").glob(f"advisor-*-{kind}.json"))
    return [json.loads(p.read_text()) for p in paths]


def test_native_tool_calls_run_through_the_dispatcher_and_every_call_is_traced(
    tmp_path: Path,
) -> None:
    call = ToolCall("c1", "list_plugins", {"registry": "localizations"})
    stub = StubProvider([Completion(tool_calls=(call,)), ANSWER])
    loop, _ = _loop(tmp_path, stub)

    outcome = loop.run("Which localization should I use?", Choice, FALLBACK)

    assert outcome.decision == Choice(localization="adaptive", reason="the default, prior-only")
    assert not outcome.fell_back
    assert outcome.model_calls == 2
    assert outcome.tool_calls == [
        {"tool": "list_plugins", "arguments": {"registry": "localizations"}, "ok": True}
    ]
    first, second = stub.requests
    assert [t.name for t in first.tools] == list(TOOLS)  # the registry
    result = second.messages[-1]
    assert (result.role, result.tool_call_id, result.name) == ("tool", "c1", "list_plugins")
    assert "adaptive_per_update" in result.content
    # Each call's trace holds its request, whose hash it records, and the model's answer.
    calls = _files(tmp_path, "call")
    assert len(calls) == 2
    for trace in calls:
        assert trace["request_sha256"] == request_hash(trace["request"])  # type: ignore[arg-type]
        assert trace["model"] == "stub-model"
    [decision] = _files(tmp_path, "decision")
    assert decision["decided_by"] == "model"
    assert decision["decision"] == outcome.decision.model_dump()
    assert decision["thresholds"] == {"large_ensemble": 200}
    assert len(decision["llm_calls"]) == 2  # type: ignore[arg-type]


def test_models_without_tool_calling_get_prompted_json(tmp_path: Path) -> None:
    stub = StubProvider(
        [
            '{"tool": "list_plugins", "arguments": {"registry": "localizations"}}',
            f'Here it is:\n```json\n{{"answer": {ANSWER}}}\n```',
        ],
        tool_calls=False,
    )
    loop, _ = _loop(tmp_path, stub)

    outcome = loop.run("Which localization?", Choice, FALLBACK)

    assert outcome.decision.localization == "adaptive"
    assert not outcome.fell_back
    first, second = stub.requests
    assert first.tools == ()  # no native tools: they are in the prompt
    system = first.messages[0].content
    assert '"tool": "<name>"' in system
    assert "list_plugins" in system
    assert TOOLS["list_plugins"].schema()["properties"]["registry"]["enum"][0] in system
    assert second.messages[-1].content.startswith("Result of list_plugins: ")


def test_invalid_prompted_tool_calls_get_the_validation_error_back(tmp_path: Path) -> None:
    stub = StubProvider(
        [
            '{"tool": "list_plugins", "arguments": {"registry": "nope"}}',
            '{"tool": "no_such_tool", "arguments": {}}',
            f'{{"answer": {ANSWER}}}',
        ],
        tool_calls=False,
    )
    loop, _ = _loop(tmp_path, stub)

    outcome = loop.run("Which localization?", Choice, FALLBACK)

    assert not outcome.fell_back
    # A known tool's refused arguments are a recorded call with its reason (the 2026-10-04
    # chat: a refused plan_study left no trace); an unknown tool is an invalid answer.
    assert outcome.tool_calls == [
        {"tool": "list_plugins", "arguments": {"registry": "nope"}, "ok": False}
    ]
    feedback = [r.messages[-1].content for r in stub.requests[1:]]
    assert feedback[0].startswith("Result of list_plugins: ")
    assert "registry" in feedback[0]
    assert '"ok": false' in feedback[0]
    assert "unknown tool 'no_such_tool'" in feedback[1]


def test_invalid_answers_are_retried_then_accepted(tmp_path: Path) -> None:
    stub = StubProvider(["I think adaptive.", '{"localization": "adaptive"}', ANSWER])
    loop, _ = _loop(tmp_path, stub)

    outcome = loop.run("Which localization?", Choice, FALLBACK)

    assert not outcome.fell_back
    assert outcome.model_calls == 3
    assert "no JSON object" in stub.requests[1].messages[-1].content
    assert "reason: Field required" in stub.requests[2].messages[-1].content


def test_answers_that_stay_invalid_fall_back_to_the_rule(tmp_path: Path) -> None:
    stub = StubProvider(["no", "still no", '{"localization": 1}', ANSWER])
    loop, _ = _loop(tmp_path, stub)  # max_retries = 2: three invalid answers in a row

    outcome = loop.run("Which localization?", Choice, FALLBACK)

    assert outcome.fell_back
    assert outcome.decision == Choice(localization="distance", reason="fallback rule")
    assert outcome.reason is not None
    assert "stayed invalid (3 in a row)" in outcome.reason
    assert len(stub.requests) == 3  # the valid fourth reply was never asked for
    [decision] = _files(tmp_path, "decision")
    assert decision["decided_by"] == "fallback"
    assert decision["fallback_rule"] == FALLBACK.rule


def test_an_unreadable_reply_is_asked_again(tmp_path: Path) -> None:
    """claude -p failing to parse a native tool call is not an unavailable model."""
    unreadable = LLMInvalidOutput("claude_code: The model's tool call could not be parsed")
    stub = StubProvider([unreadable, f'{{"answer": {ANSWER}}}'], tool_calls=False)
    loop, _ = _loop(tmp_path, stub)

    outcome = loop.run("Which localization?", Choice, FALLBACK)

    assert not outcome.fell_back
    assert outcome.decision.localization == "adaptive"
    reminder = stub.requests[1].messages[-1].content
    assert reminder.startswith("Your last reply could not be read.")
    assert '{"tool": "<name>", "arguments": {...}}' in reminder


def test_replies_that_stay_unreadable_fall_back(tmp_path: Path) -> None:
    unreadable = LLMInvalidOutput("claude_code: The model's tool call could not be parsed")
    stub = StubProvider([unreadable] * 3 + [ANSWER], tool_calls=False)
    loop, _ = _loop(tmp_path, stub)  # max_retries = 2

    outcome = loop.run("Which localization?", Choice, FALLBACK)

    assert outcome.fell_back
    assert outcome.reason is not None
    assert "replies stayed unreadable (3 in a row)" in outcome.reason


def test_a_timeout_falls_back_and_is_traced(tmp_path: Path) -> None:
    stub = StubProvider([LLMTimeout("no answer in 120 s")])
    loop, _ = _loop(tmp_path, stub)

    outcome = loop.run("Which localization?", Choice, FALLBACK)

    assert outcome.fell_back
    assert outcome.reason == "the provider failed: no answer in 120 s"
    [call] = _files(tmp_path, "call")
    assert call["error"] == "LLMTimeout: no answer in 120 s"
    assert call["answer"] is None


def test_running_out_of_turns_falls_back(tmp_path: Path) -> None:
    call = ToolCall("c", "list_plugins", {})
    stub = StubProvider([Completion(tool_calls=(call,))] * 3)
    loop, _ = _loop(tmp_path, stub, max_turns=3)

    outcome = loop.run("Which localization?", Choice, FALLBACK)

    assert outcome.fell_back
    assert outcome.reason == "no valid answer in 3 turns"
    assert len(outcome.tool_calls) == 2  # the last turn offers no tools, and runs none
    last = stub.requests[-1]
    assert last.tools == ()
    assert "last turn" in last.messages[-1].content


def test_the_last_turn_asks_for_the_answer(tmp_path: Path) -> None:
    # Regression: on M-5Spots, Groq's qwen3.8-27b kept calling run_summary with guessed ids
    # until its 6 turns ran out; now its last turn has no tools, and it answers.
    call = ToolCall("c", "list_plugins", {})

    def answer_if_no_tools(request: StubRequest) -> Completion:
        return Completion(tool_calls=(call,)) if request.tools else Completion(text=ANSWER)

    stub = StubProvider([answer_if_no_tools] * 3)
    loop, _ = _loop(tmp_path, stub, max_turns=3)

    outcome = loop.run("Which localization?", Choice, FALLBACK)

    assert not outcome.fell_back
    assert outcome.decision.localization == "adaptive"
    assert [bool(r.tools) for r in stub.requests] == [True, True, False]


def test_the_oldest_tool_results_are_trimmed_to_fit_the_context(tmp_path: Path) -> None:
    # The context holds the tool definitions and one detailed list_plugins result, not two.
    # Both are measured (at the loop's 1.5 characters per token): they grow with the plugins,
    # and the dispatcher truncates a large result, so fixed numbers go stale.
    session = ToolSession(ApexService(tmp_path / "measure"), client="agent-test", model="m")
    result_chars = len(json.dumps(call("list_plugins", {"details": True}, session), default=str))
    probe, _ = _loop(tmp_path / "probe", StubProvider([ANSWER]))
    tools_chars = probe._tools_size
    one_result = int((tools_chars + 1.5 * result_chars) / 1.5)
    max_context = 1_000 + one_result + int(0.25 * result_chars / 1.5)  # room for 1.25 results
    details = ToolCall("c1", "list_plugins", {"details": True})
    again = ToolCall("c2", "list_plugins", {"details": True})
    seen: list[StubRequest] = []

    def answer(request: StubRequest) -> str:
        seen.append(request)
        return ANSWER

    stub = StubProvider(
        [Completion(tool_calls=(details,)), Completion(tool_calls=(again,)), answer],
        max_context=max_context,
    )
    loop, _ = _loop(tmp_path, stub, max_tokens=1_000)

    outcome = loop.run("Which localization?", Choice, FALLBACK)

    assert not outcome.fell_back
    results = [m for m in seen[0].messages if m.role == "tool"]
    assert results[0].content == TRIMMED  # the oldest went
    assert results[1].content != TRIMMED  # the newest stayed
    assert json.loads(results[1].content)["ok"]


def test_no_model_bypasses_approval(tmp_path: Path) -> None:
    # The model tries to submit without validating: the dispatcher refuses, nothing runs.
    submit = ToolCall("c1", "submit_run", {"config_file": str(tmp_path / "config.toml")})
    stub = StubProvider([Completion(tool_calls=(submit,)), ANSWER])
    loop, _ = _loop(tmp_path, stub)

    outcome = loop.run("Run the case.", Choice, FALLBACK)

    assert outcome.tool_calls[0] == {
        "tool": "submit_run",
        "arguments": {"config_file": str(tmp_path / "config.toml")},
        "ok": False,
    }
    refusal = json.loads(stub.requests[1].messages[-1].content)
    assert "approval_token" in refusal["error"]
    assert ApexService(tmp_path / "service").jobs.all_records() == []


def test_a_task_without_tools_offers_none(tmp_path: Path) -> None:
    stub = StubProvider([ANSWER])
    loop = AgentLoop(stub, CallRecorder(tmp_path / "run", "advisor"))

    outcome = loop.run("Which localization?", Choice, FALLBACK)

    assert not outcome.fell_back
    assert stub.requests[0].tools == ()


def test_unknown_tools_are_refused_at_construction(tmp_path: Path) -> None:
    session = ToolSession(ApexService(tmp_path / "service"))
    with pytest.raises(ValueError, match="unknown tools"):
        AgentLoop(StubProvider(), CallRecorder(tmp_path, "x"), session=session, tools=["nope"])


class Reply(BaseModel):
    """An answer with a field named like the prompted protocol's wrapper (the chat's)."""

    answer: str = Field(min_length=1)


def _prompted(tmp_path: Path, replies: list[str]) -> tuple[AgentLoop, StubProvider]:
    stub = StubProvider(replies, tool_calls=False)
    loop, _ = _loop(tmp_path, stub)
    return loop, stub


@pytest.mark.parametrize(
    "reply",
    [
        '{"answer": {"answer": "hello"}}',  # the protocol's form
        '{"answer": "hello"}',  # the answer itself, as models write it
        "hello",  # prose: a one-text-field answer
    ],
)
def test_prompted_answers_in_the_forms_models_use(tmp_path: Path, reply: str) -> None:
    loop, _ = _prompted(tmp_path, [reply])

    outcome = loop.run("Say hello", Reply, Fallback("x", lambda: Reply(answer="fallback")))

    assert not outcome.fell_back
    assert outcome.decision.answer == "hello"


def test_prose_is_not_an_answer_when_the_schema_has_more_fields(tmp_path: Path) -> None:
    loop, _ = _prompted(tmp_path, ["adaptive, the default"] * 3)

    outcome = loop.run("Which?", Choice, FALLBACK)

    assert outcome.fell_back


def test_a_tool_call_in_claudes_xml_form_is_run(tmp_path: Path) -> None:
    xml = (
        '<function_calls>\n<invoke name="list_plugins">\n'
        '<parameter name="registry">localizations</parameter>\n</invoke>\n</function_calls>'
    )
    loop, stub = _prompted(tmp_path, [xml, '{"answer": "adaptive"}'])

    outcome = loop.run("Which localizations?", Reply, Fallback("x", lambda: Reply(answer="f")))

    assert outcome.tool_calls == [
        {"tool": "list_plugins", "arguments": {"registry": "localizations"}, "ok": True}
    ]
    assert outcome.decision.answer == "adaptive"
    assert "adaptive_per_update" in stub.requests[1].messages[-1].content


def test_a_refused_call_adds_a_turn_to_revise_it(tmp_path: Path) -> None:
    """The 2026-10-04 chat: lookups and one refused plan used up the turns before the model
    could revise it. ``refusal_turns`` adds a model call per refused tool call."""
    replies = [
        '{"tool": "list_plugins", "arguments": {"registry": "nope"}}',  # refused
        '{"tool": "list_plugins", "arguments": {"registry": "localizations"}}',
        f'{{"answer": {ANSWER}}}',
    ]
    tight, _ = _loop(tmp_path / "a", StubProvider(list(replies), tool_calls=False), max_turns=2)
    assert tight.run("Which localization?", Choice, FALLBACK).fell_back  # no turn to answer

    stub = StubProvider(list(replies), tool_calls=False)
    loop, _ = _loop(tmp_path / "b", stub, max_turns=2, refusal_turns=1)
    outcome = loop.run("Which localization?", Choice, FALLBACK)

    assert not outcome.fell_back
    assert outcome.model_calls == 3
    assert [c["ok"] for c in outcome.tool_calls] == [False, True]


def test_bad_prompted_arguments_are_recorded_and_still_end_in_the_fallback(
    tmp_path: Path,
) -> None:
    """Recording refused prompted calls must not change when an agent falls back: bad
    arguments still count as invalid answers (max_retries), as before."""
    bad = '{"tool": "list_plugins", "arguments": {"registry": "nope"}}'
    stub = StubProvider([bad] * 4 + [f'{{"answer": {ANSWER}}}'], tool_calls=False)
    loop, _ = _loop(tmp_path, stub, max_retries=2)

    outcome = loop.run("Which localization?", Choice, FALLBACK)

    assert outcome.fell_back
    assert "answers stayed invalid (3 in a row)" in str(outcome.reason)
    assert [c["ok"] for c in outcome.tool_calls] == [False, False, False]


# ---------------------------------------------------------------------- the Stop button


def test_a_stopped_task_makes_no_model_call(tmp_path: Path) -> None:
    from apex.agents.loop import STOPPED

    provider = StubProvider([ANSWER])
    loop, _ = _loop(tmp_path, provider)
    loop.should_stop = lambda: True
    outcome = loop.run("Which localization?", Choice, FALLBACK)
    assert outcome.fell_back
    assert outcome.reason == STOPPED
    assert provider.requests == []  # no model call


def test_a_stop_during_a_tool_call_ends_the_task_at_its_next_step(tmp_path: Path) -> None:
    from apex.agents.loop import STOPPED, LocalTool

    class NoArguments(BaseModel):
        pass

    stop = {"now": False}

    def slow(_: NoArguments) -> dict[str, str]:
        stop["now"] = True  # the user presses Stop while this tool runs
        return {"done": "yes"}

    provider = StubProvider([
        Completion(tool_calls=(ToolCall("c1", "look", {}), ToolCall("c2", "look", {}))),
        ANSWER,
    ])  # fmt: skip
    recorder = CallRecorder(tmp_path / "run", "advisor")
    loop = AgentLoop(provider, recorder, local_tools=[LocalTool("look", "Look.", NoArguments,
                                                                 slow)],
                     should_stop=lambda: stop["now"])  # fmt: skip
    outcome = loop.run("Which localization?", Choice, FALLBACK)
    assert outcome.reason == STOPPED
    assert [c["tool"] for c in outcome.tool_calls] == ["look"]  # the second never ran
    assert len(provider.requests) == 1  # no model call after the stop


def test_a_stop_ends_a_streamed_answer_at_its_next_piece(tmp_path: Path) -> None:
    """The stop is asked before the model call and at every streamed piece: true at the
    second piece, the stream (13 pieces of 4 characters) is read no further."""
    from apex.agents.loop import STOPPED

    asked = {"n": 0}

    def should_stop() -> bool:
        asked["n"] += 1
        return asked["n"] >= 3  # 1: before the model call; 2, 3: the first two pieces

    provider = StubProvider([ANSWER], streaming=True, chunk=4)
    recorder = CallRecorder(tmp_path / "run", "advisor")
    loop = AgentLoop(provider, recorder, stream_answers=True, should_stop=should_stop)
    outcome = loop.run("Which localization?", Choice, FALLBACK)
    assert outcome.reason == STOPPED
    assert asked["n"] == 3  # not one ask per piece of the whole stream


def test_a_stopped_claude_cli_is_killed_at_once(tmp_path: Path) -> None:
    """claude -p as a subprocess: the default runner kills it when the stop turns true."""
    import sys
    import time

    from apex.agents.providers.claude_code import CallStopped, _stoppable

    began = time.monotonic()
    with pytest.raises(CallStopped):
        _stoppable([sys.executable, "-c", "import time; time.sleep(60)"], "", 120.0,
                   str(tmp_path), {}, lambda: time.monotonic() - began > 0.5)  # fmt: skip
    assert time.monotonic() - began < 10
    done = _stoppable([sys.executable, "-c", "import sys; print(sys.stdin.read().upper())"],
                      "hello", 60.0, str(tmp_path), {}, lambda: False)  # fmt: skip
    assert (done.returncode, done.stdout.strip()) == (0, "HELLO")


def test_a_tool_past_its_time_limit_is_abandoned_and_a_stop_is_seen_meanwhile(
    tmp_path: Path,
) -> None:
    """A tool longer than tool_timeout_s is reported to the model as
    timed out (it runs on in the background); while it runs, a stop ends the task."""
    import threading
    import time

    from apex.agents.loop import STOPPED, LocalTool

    class NoArguments(BaseModel):
        pass

    release = threading.Event()

    def slow(_: NoArguments) -> dict[str, str]:
        release.wait(30)
        return {"done": "late"}

    def make(provider: StubProvider, **kwargs: object) -> AgentLoop:
        return AgentLoop(provider, CallRecorder(tmp_path / "run", "advisor"),
                         local_tools=[LocalTool("slow", "Slow.", NoArguments, slow)],
                         **kwargs)  # type: ignore[arg-type]  # fmt: skip

    provider = StubProvider([Completion(tool_calls=(ToolCall("c1", "slow", {}),)), ANSWER])
    loop = make(provider, settings=LoopSettings(tool_timeout_s=0.5))
    began = time.monotonic()
    outcome = loop.run("Which localization?", Choice, FALLBACK)
    assert time.monotonic() - began < 10
    assert not outcome.fell_back  # the model answered after the time-out
    sent = provider.requests[1].messages[-1].content
    assert "did not finish within 0.5 s" in sent
    stopped = {"now": False}
    provider = StubProvider([Completion(tool_calls=(ToolCall("c1", "slow", {}),)), ANSWER])
    loop = make(
        provider, settings=LoopSettings(tool_timeout_s=60.0), should_stop=lambda: stopped["now"]
    )
    threading.Timer(0.5, lambda: stopped.update(now=True)).start()
    began = time.monotonic()
    outcome = loop.run("Which localization?", Choice, FALLBACK)
    assert outcome.reason == STOPPED
    assert time.monotonic() - began < 10  # not the tool's 30 s
    release.set()


def test_a_tool_result_excerpt_keeps_short_fields_whole_and_names_the_rest() -> None:
    result = {"ok": True, "table": list(range(1000)),
              "result": {"approval": {"needed": True}, "text": "y" * 600}}  # fmt: skip

    excerpt = result_excerpt(result, limit=400, field_limit=100)

    assert excerpt == {
        "ok": True,
        "result": {"approval": {"needed": True}, "_omitted": ["text"], "_omitted_count": 1},
        "_omitted": ["table"], "_omitted_count": 1,
    }  # fmt: skip


def test_a_tool_result_excerpt_of_many_short_entries_stays_within_its_limit() -> None:
    # e.g. a comparison keyed by run: naming every entry left out would not be bounded.
    result = {"result": {f"run_{i:05d}": {"rmse": i} for i in range(3000)}}

    excerpt = result_excerpt(result)

    assert len(json.dumps(excerpt)) <= 2000 + 600  # the names and count of what was left out
    assert excerpt["result"]["_omitted_count"] == 3000 - (len(excerpt["result"]) - 2)


def test_a_repeated_call_with_an_unchanged_result_says_so(
    tmp_path: Path,
) -> None:
    """Release tester C7 (2026-10-24): a model called list_runs eight times with the same
    arguments and the same empty result. Each repeat is run (results may change) and told
    so when its result has not."""
    calls = [ToolCall(f"c{i}", "list_runs", {"limit": 5}) for i in range(4)]
    stub = StubProvider([Completion(tool_calls=(c,)) for c in calls] + [ANSWER])
    loop, _ = _loop(tmp_path, stub, max_turns=6)

    outcome = loop.run("Which runs do I have?", Choice, FALLBACK)

    assert not outcome.fell_back
    assert len(outcome.tool_calls) == 4  # every call is run, and traced
    results = [json.loads(r.messages[-1].content) for r in stub.requests[1:]]
    assert "repeated" not in results[0]
    assert [r["repeated"].split(" times")[0][-1] for r in results[1:]] == ["2", "3", "4"]
    assert all(r["result"] == results[0]["result"] for r in results)

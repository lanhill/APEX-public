"""Streamed answers: the extractor, the loop, the chat and providers."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from apex.agents.loop import AgentLoop, Fallback
from apex.agents.providers.openai_compatible import OpenAICompatibleProvider
from apex.agents.providers.stub import StubProvider
from apex.agents.streaming import AnswerText
from apex.agents.trace import CallRecorder
from apex.core.llm import Completion, LLMError, Message, ToolCall, ToolSpec


class Answer(BaseModel):
    """A chat-like answer."""

    answer: str


@pytest.mark.parametrize(
    ("pieces", "expected"),
    [
        (['{"ans', 'wer": "Hel', "lo\\n wor", "ld \\u00e9\\", '"x"}'], 'Hello\n world é"x'),
        (['{"answer": {"answer": "inner"}}'], "inner"),  # the prompted wrapper
        (['{"answer"', ': "a', 'b"} trailing'], "ab"),
        (['{"tool": "list_runs", "arguments": {}}'], ""),  # a tool call: nothing to show
        (["not json at all"], ""),
    ],
)
def test_the_answer_is_picked_out_of_the_streamed_json(pieces: list[str], expected: str) -> None:
    text = AnswerText()
    assert "".join(text.feed(p) for p in pieces) == expected


def _loop(tmp_path: Path, provider: Any, events: list[tuple[str, dict[str, Any]]]) -> AgentLoop:  # noqa: ANN401
    return AgentLoop(
        provider, CallRecorder(tmp_path, "t"), on_event=lambda k, f: events.append((k, dict(f))),
        stream_answers=True,
    )  # fmt: skip


@pytest.mark.parametrize("tool_calls", [True, False])
def test_the_loop_hands_the_answer_over_as_it_arrives(tmp_path: Path, tool_calls: bool) -> None:
    answer = "The run converged after 6 iterations; its RMSE fell by 40 %."
    reply = json.dumps({"answer": answer} if tool_calls else {"answer": {"answer": answer}})
    events: list[tuple[str, dict[str, Any]]] = []
    stub = StubProvider([reply], tool_calls=tool_calls, streaming=True)

    fallback = Fallback("none", lambda: Answer(answer="-"))
    outcome = _loop(tmp_path, stub, events).run("How did it go?", Answer, fallback)

    assert outcome.decision.answer == answer
    deltas = [f["text"] for k, f in events if k == "answer_delta"]
    assert len(deltas) > 3  # in pieces
    assert "".join(deltas) == answer


def test_without_streaming_nothing_is_handed_over_early(tmp_path: Path) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    stub = StubProvider(['{"answer": "done"}'], streaming=False)
    _loop(tmp_path, stub, events).run("?", Answer, Fallback("none", lambda: Answer(answer="-")))
    assert not [k for k, _ in events if k == "answer_delta"]


def _sse(*events: dict[str, Any]) -> bytes:
    lines = [f"data: {json.dumps(e)}\n\n" for e in events] + ["data: [DONE]\n\n"]
    return "".join(lines).encode()


def _provider(handler: Any) -> OpenAICompatibleProvider:  # noqa: ANN401
    return OpenAICompatibleProvider(
        "qwen", "http://server/v1", max_retries=1, max_retry_wait_s=0.1,
        http=lambda: httpx.Client(transport=httpx.MockTransport(handler)),
    )  # fmt: skip


def test_a_chat_completions_stream_is_assembled() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = _sse(
            {"id": "c1", "choices": [{"delta": {"content": "Let me "}}]},
            {"id": "c1", "choices": [{"delta": {"content": "look."}}]},
            {"id": "c1", "choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1",
                "function": {"name": "list_runs", "arguments": '{"lim'}}]}}]},
            {"id": "c1", "choices": [{"delta": {"tool_calls": [{"index": 0,
                "function": {"arguments": 'it": 5}'}}]}, "finish_reason": "tool_calls"}],
             "usage": {"prompt_tokens": 50, "completion_tokens": 9}},
        )  # fmt: skip
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    pieces: list[str] = []
    completion = _provider(handler).stream_complete(
        [Message("user", "Which runs?")], on_text=pieces.append,
        tools=[ToolSpec("list_runs", "List runs.", {"type": "object"})],
        max_tokens=100, temperature=0.0,
    )  # fmt: skip

    assert json.loads(seen[0].content)["stream"] is True
    assert pieces == ["Let me ", "look."]
    assert completion.text == "Let me look."
    assert completion.tool_calls == (ToolCall("call_1", "list_runs", {"limit": 5}),)
    assert completion.stop_reason == "tool_calls"
    assert (completion.usage.input_tokens, completion.usage.output_tokens) == (50, 9)


def test_a_stream_is_retried_when_rate_limited_and_refused_otherwise() -> None:
    ok = _sse({"choices": [{"delta": {"content": "ok"}}]})
    answers = [
        httpx.Response(429, headers={"retry-after": "0.01"}, text="slow down"),
        httpx.Response(200, content=ok),
    ]
    completion = _provider(lambda r: answers.pop(0)).stream_complete(
        [Message("user", "hi")], on_text=lambda _: None, max_tokens=10, temperature=0.0
    )
    assert completion.text == "ok"
    with pytest.raises(LLMError, match="HTTP 400"):
        _provider(lambda r: httpx.Response(400, text="bad")).stream_complete(
            [Message("user", "hi")], on_text=lambda _: None, max_tokens=10, temperature=0.0
        )


def test_an_anthropic_stream_hands_over_text_and_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("anthropic")
    import httpx2

    from apex.agents.providers.anthropic import AnthropicProvider

    events = [
        ("message_start", {"type": "message_start", "message": {
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 12, "output_tokens": 1}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": '{"answer": "Al'}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": 'l good"}'}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta",
                           "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                           "usage": {"output_tokens": 7}}),
        ("message_stop", {"type": "message_stop"}),
    ]  # fmt: skip
    body = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)

    def handler(request: Any) -> Any:  # noqa: ANN401
        return httpx2.Response(200, content=body.encode(),
                               headers={"content-type": "text/event-stream"})  # fmt: skip

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    provider = AnthropicProvider("claude-opus-5-5", max_retries=0, http_client=client)
    assert provider.capabilities.streaming
    pieces: list[str] = []

    completion = provider.stream_complete(
        [Message("user", "How is it?")], on_text=pieces.append, max_tokens=100, temperature=0.0
    )

    assert pieces == ['{"answer": "Al', 'l good"}']
    assert completion.text == '{"answer": "All good"}'
    assert completion.stop_reason == "end_turn"
    assert isinstance(completion, Completion)

"""The LLM providers' requests and answers, on recorded API examples.

No network: each provider gets an HTTP client with a mock transport that records the
request and answers with an example of the API's documented format. Tool definitions
round-trip through the tool registry's schemas (``apex.services.tools.export``). The opt-in
smoke tests at the end use a real model and are never part of CI.
"""

import json
import os
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from apex.agents.providers import create_provider
from apex.agents.providers.ollama import OllamaProvider
from apex.agents.providers.openai_compatible import OpenAICompatibleProvider
from apex.core.errors import ConfigError
from apex.core.llm import LLMError, LLMTimeout, LLMUnavailable, Message, ToolCall, ToolSpec
from apex.services.settings import LLMSection
from apex.services.tools import TOOLS, export

SPECS = [ToolSpec(t.name, t.description, t.schema()) for t in TOOLS.values()]

#: A conversation with every kind of message, including a tool call and its result.
CONVERSATION = [
    Message("system", "You are an agent of APEX."),
    Message("user", "Which localizations are there?"),
    Message("assistant", "", (ToolCall("call_1", "list_plugins", {"registry": "localizations"}),)),
    Message("tool", '{"ok": true}', tool_call_id="call_1", name="list_plugins"),
]


class Verdict(BaseModel):
    """A structured answer."""

    stop: bool


def _mock(
    answer: Callable[[httpx.Request], httpx.Response], seen: list[httpx.Request]
) -> Callable[[], httpx.Client]:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return answer(request)

    return lambda: httpx.Client(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------- openai_compatible

#: A chat-completions answer with a tool call (the OpenAI API reference's format).
OPENAI_ANSWER = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_schema", "arguments": '{"kind": "study"}'},
                    },
                    {
                        "id": "call_bad",
                        "type": "function",
                        "function": {"name": "list_plugins", "arguments": "{not json"},
                    },
                ],
            },
        }
    ],
    "usage": {"prompt_tokens": 812, "completion_tokens": 25, "total_tokens": 837},
}


def test_openai_compatible_requests_and_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_LLM_KEY", "sk-secret")
    seen: list[httpx.Request] = []
    provider = OpenAICompatibleProvider(
        "glm-4.6",
        base_url="http://llm.example/v1/",
        api_key_env="TEST_LLM_KEY",
        http=_mock(lambda r: httpx.Response(200, json=OPENAI_ANSWER), seen),
    )

    completion = provider.complete(
        CONVERSATION, tools=SPECS, max_tokens=512, temperature=0.0, seed=7
    )

    [request] = seen
    assert str(request.url) == "http://llm.example/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-secret"
    body = json.loads(request.content)
    assert body["tools"] == export("openai")  # the registry's own definitions
    assert (body["model"], body["max_tokens"], body["temperature"], body["seed"]) == (
        "glm-4.6", 512, 0.0, 7,
    )  # fmt: skip
    assert body["messages"][2] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "list_plugins", "arguments": '{"registry": "localizations"}'},
            }
        ],
    }
    assert body["messages"][3] == {
        "role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}', "name": "list_plugins",
    }  # fmt: skip
    good, bad = completion.tool_calls
    assert good == ToolCall("call_abc", "get_schema", {"kind": "study"})
    assert bad.arguments is None
    assert bad.error is not None
    assert "not JSON" in bad.error
    assert (completion.usage.input_tokens, completion.usage.output_tokens) == (812, 25)
    assert completion.raw_id == "chatcmpl-123"
    assert "sk-secret" not in json.dumps(provider.describe())


#: A Gemini 3 tool call through its OpenAI-compatible API: a thought signature is attached
#: (Gemini's thought-signature documentation), and must come back with the conversation.
GEMINI_ANSWER = {
    "id": "gemini-1",
    "choices": [
        {
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "extra_content": {"google": {"thought_signature": "c2lnbmF0dXJl"}},
                        "function": {
                            "arguments": '{"registry": "localizations"}',
                            "name": "list_plugins",
                        },
                        "id": "function-call-1",
                        "type": "function",
                    }
                ],
            },
        }
    ],
}


def test_extra_fields_of_tool_calls_are_echoed_back() -> None:
    """Regression: Gemini refused the follow-up request (HTTP 400, "Function call is missing
    a thought_signature") because the provider dropped the call's extra_content.
    """
    seen: list[httpx.Request] = []
    provider = OpenAICompatibleProvider(
        "gemini-3.8-flash", http=_mock(lambda r: httpx.Response(200, json=GEMINI_ANSWER), seen)
    )
    [call] = provider.complete(
        CONVERSATION[:2], tools=SPECS, max_tokens=64, temperature=0
    ).tool_calls
    assert call.extra == {"extra_content": {"google": {"thought_signature": "c2lnbmF0dXJl"}}}

    follow_up = [*CONVERSATION[:2], Message("assistant", "", (call,)),
                 Message("tool", '{"ok": true}', tool_call_id=call.id, name=call.name)]  # fmt: skip
    provider.complete(follow_up, tools=SPECS, max_tokens=64, temperature=0)

    sent = json.loads(seen[1].content)["messages"][2]["tool_calls"][0]
    assert sent["extra_content"] == {"google": {"thought_signature": "c2lnbmF0dXJl"}}
    assert (sent["id"], sent["function"]["name"]) == ("function-call-1", "list_plugins")


def test_openai_compatible_json_mode_and_no_key() -> None:
    seen: list[httpx.Request] = []
    answer = {"id": "x", "choices": [{"message": {"content": '{"stop": true}'}}]}
    provider = OpenAICompatibleProvider(
        "qwen3:32b", json_mode=True, http=_mock(lambda r: httpx.Response(200, json=answer), seen)
    )

    completion = provider.complete(
        CONVERSATION[:2], response_model=Verdict, max_tokens=64, temperature=0.0
    )

    body = json.loads(seen[0].content)
    assert body["response_format"] == {"type": "json_object"}
    assert "tools" not in body
    assert "seed" not in body
    assert "authorization" not in seen[0].headers
    assert completion.text == '{"stop": true}'


@pytest.mark.parametrize(
    ("answer", "error"),
    [
        (lambda r: httpx.Response(503, text="overloaded"), LLMUnavailable),
        (lambda r: httpx.Response(429, text="slow down"), LLMUnavailable),
        (lambda r: httpx.Response(400, text="bad request"), LLMError),
        (lambda r: httpx.Response(200, text="<html>"), LLMError),
    ],
)
def test_openai_compatible_errors(
    answer: Callable[[httpx.Request], httpx.Response], error: type[LLMError]
) -> None:
    provider = OpenAICompatibleProvider("m", http=_mock(answer, []))

    with pytest.raises(error) as raised:
        provider.complete(CONVERSATION[:2], max_tokens=8, temperature=0.0)

    assert type(raised.value) is error


#: Groq's answer when a task exceeds the free tier's input tokens per minute (2026-09-29).
GROQ_429 = (
    '{"error":{"message":"Rate limit reached for model `qwen/qwen3.8-27b` on input tokens '
    "per minute (ITPM): Limit 7000, Used 4535, Requested 4740. Please try again in 0.02s. "
    'Need more tokens?","type":"tokens","code":"rate_limit_exceeded"}}'
)


def _sequence(*responses: httpx.Response) -> tuple[Callable[[], httpx.Client], list[httpx.Request]]:
    seen: list[httpx.Request] = []
    replies = iter(responses)
    return _mock(lambda r: next(replies), seen), seen


@pytest.mark.parametrize(
    "limited",
    [
        httpx.Response(429, headers={"retry-after": "0.01"}, text="slow down"),
        httpx.Response(429, text=GROQ_429),
        httpx.Response(503, text="overloaded, try again in 10ms"),
    ],
)
def test_rate_limits_are_retried_after_the_stated_wait(limited: httpx.Response) -> None:
    """Regression: one monitor task exceeded Groq's per-minute limit, and the first 429
    ("Please try again in 19.5s") ended it; now the provider waits and retries.
    """
    ok = httpx.Response(200, json={"id": "x", "choices": [{"message": {"content": "fine"}}]})
    http, seen = _sequence(limited, ok)

    completion = OpenAICompatibleProvider("m", http=http).complete(
        CONVERSATION[:2], max_tokens=8, temperature=0.0
    )

    assert completion.text == "fine"
    assert len(seen) == 2


@pytest.mark.parametrize(
    "limited",
    [
        httpx.Response(429, text="Request too large: reduce max_tokens"),  # no wait stated
        httpx.Response(429, headers={"retry-after": "3600"}, text="come back in an hour"),
    ],
)
def test_rate_limits_without_a_short_stated_wait_fail_at_once(limited: httpx.Response) -> None:
    http, seen = _sequence(limited)

    with pytest.raises(LLMUnavailable):
        OpenAICompatibleProvider("m", http=http).complete(
            CONVERSATION[:2], max_tokens=8, temperature=0.0
        )

    assert len(seen) == 1


def test_retries_are_bounded() -> None:
    limited = [httpx.Response(429, headers={"retry-after": "0"}, text="x") for _ in range(4)]
    http, seen = _sequence(*limited)

    with pytest.raises(LLMUnavailable):
        OpenAICompatibleProvider("m", max_retries=2, http=http).complete(
            CONVERSATION[:2], max_tokens=8, temperature=0.0
        )

    assert len(seen) == 3  # the request and two retries


@pytest.mark.parametrize(
    ("transport_error", "error"),
    [(httpx.ReadTimeout("slow"), LLMTimeout), (httpx.ConnectError("refused"), LLMUnavailable)],
)
def test_transport_failures_are_unavailable(
    transport_error: Exception, error: type[LLMError]
) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise transport_error

    provider = OpenAICompatibleProvider("m", http=_mock(fail, []))

    with pytest.raises(error):
        provider.complete(CONVERSATION[:2], max_tokens=8, temperature=0.0)


# ---------------------------------------------------------------------- ollama

#: An ``/api/chat`` answer with a tool call (Ollama's API documentation's format).
OLLAMA_ANSWER = {
    "model": "qwen3:32b",
    "created_at": "2026-09-29T07:00:00Z",
    "message": {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "get_schema", "arguments": {"kind": "prediction"}}}],
    },
    "done": True,
    "done_reason": "stop",
    "prompt_eval_count": 950,
    "eval_count": 18,
}


def test_ollama_requests_and_answers() -> None:
    seen: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"name": "qwen3:32b"}, {"name": "glm-4.6"}]}
            )
        return httpx.Response(200, json=OLLAMA_ANSWER)

    provider = OllamaProvider(
        "qwen3:32b",
        base_url="http://localhost:11434/v1",  # the OpenAI address of the same server
        max_context=65_536,
        keep_alive="30m",
        http=_mock(answer, seen),
    )

    completion = provider.complete(
        CONVERSATION, tools=SPECS, max_tokens=256, temperature=0.0, seed=3
    )

    assert [r.url.path for r in seen] == ["/api/show", "/api/chat"]  # the model's context
    body = json.loads(seen[1].content)
    assert str(seen[1].url) == "http://localhost:11434/api/chat"
    assert body["tools"] == export("openai")
    # A short conversation asks for the least context (release tester C8: it grows).
    assert body["options"] == {"temperature": 0.0, "num_predict": 256, "num_ctx": 32_768, "seed": 3}
    assert (body["stream"], body["keep_alive"]) == (False, "30m")
    assert body["messages"][2]["tool_calls"] == [
        {"function": {"name": "list_plugins", "arguments": {"registry": "localizations"}}}
    ]
    assert body["messages"][3]["tool_name"] == "list_plugins"
    assert completion.tool_calls == (ToolCall("call_0", "get_schema", {"kind": "prediction"}),)
    assert (completion.usage.input_tokens, completion.usage.output_tokens) == (950, 18)
    assert provider.models() == ["qwen3:32b", "glm-4.6"]


def test_ollama_constrains_structured_answers_to_their_schema() -> None:
    seen: list[httpx.Request] = []
    answer = {"message": {"content": '{"stop": false}'}, "done": True}
    provider = OllamaProvider(
        "glm-4.6", http=_mock(lambda r: httpx.Response(200, json=answer), seen)
    )

    completion = provider.complete(
        CONVERSATION[:2], response_model=Verdict, max_tokens=32, temperature=0
    )

    [chat] = [r for r in seen if r.url.path == "/api/chat"]  # after /api/show, once
    assert json.loads(chat.content)["format"] == Verdict.model_json_schema()
    assert completion.text == '{"stop": false}'


# ---------------------------------------------------------------------- anthropic

#: A Messages API answer with text and a tool use (Anthropic's API reference's format).
ANTHROPIC_ANSWER = {
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-5-5",
    "content": [
        {"type": "text", "text": "Let me look."},
        {"type": "tool_use", "id": "toolu_01", "name": "get_schema", "input": {"kind": "study"}},
    ],
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "usage": {"input_tokens": 1200, "output_tokens": 40},
}


def _anthropic(
    monkeypatch: pytest.MonkeyPatch, answer: Callable[[Any], Any], seen: list[Any]
) -> Any:  # noqa: ANN401 (an AnthropicProvider on an httpx2 mock transport)
    pytest.importorskip("anthropic")
    import httpx2

    from apex.agents.providers.anthropic import AnthropicProvider

    def handler(request: Any) -> Any:  # noqa: ANN401
        seen.append(request)
        return answer(request)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    return AnthropicProvider("claude-opus-5-5", max_retries=0, http_client=client)


def test_anthropic_requests_and_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx2

    seen: list[Any] = []
    provider = _anthropic(monkeypatch, lambda r: httpx2.Response(200, json=ANTHROPIC_ANSWER), seen)

    completion = provider.complete(CONVERSATION, tools=SPECS, max_tokens=1024, temperature=0.0)

    [request] = seen
    assert request.url.path == "/v1/messages"
    assert request.headers["x-api-key"] == "sk-ant-secret"
    body = json.loads(request.content)
    assert body["tools"] == export("anthropic")  # the registry's own definitions
    assert "temperature" not in body
    assert body["system"] == "You are an agent of APEX."
    assert body["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Which localizations are there?"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "list_plugins",
                    "input": {"registry": "localizations"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": '{"ok": true}'}
            ],
        },
    ]
    assert completion.text == "Let me look."
    assert completion.tool_calls == (ToolCall("toolu_01", "get_schema", {"kind": "study"}),)
    assert (completion.usage.input_tokens, completion.usage.output_tokens) == (1200, 40)
    assert completion.stop_reason == "tool_use"
    assert "sk-ant-secret" not in json.dumps(provider.describe())


@pytest.mark.parametrize(("status", "error"), [(529, LLMUnavailable), (400, LLMError)])
def test_anthropic_errors(
    monkeypatch: pytest.MonkeyPatch, status: int, error: type[LLMError]
) -> None:
    import httpx2

    body = {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}
    provider = _anthropic(monkeypatch, lambda r: httpx2.Response(status, json=body), [])

    with pytest.raises(error) as raised:
        provider.complete(CONVERSATION[:2], max_tokens=8, temperature=0.0)

    assert type(raised.value) is error


# ---------------------------------------------------------------------- configuration


def test_providers_are_created_from_agents_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    ollama = create_provider(
        LLMSection(
            provider="ollama", model="glm-4.6", max_context=65_536, options={"keep_alive": "5m"}
        )
    )
    assert ollama.describe()["keep_alive"] == "5m"
    assert ollama.describe()["max_context"] == 65_536  # capabilities would ask the server
    prompted = create_provider(LLMSection(model="kimi-k2", tool_calls=False))
    assert prompted.provider_name == "openai_compatible"
    assert not prompted.capabilities.tool_calls
    assert create_provider(LLMSection(provider="stub")).model_id == "stub-model"

    with pytest.raises(ConfigError, match="unknown llm_providers plugin 'gpt'"):
        create_provider(LLMSection(provider="gpt", model="m"))
    with pytest.raises(ConfigError, match="'openai_compatible' needs model"):
        create_provider(LLMSection(provider="openai_compatible"))
    with pytest.raises(ConfigError, match="keepalive"):
        create_provider(LLMSection(provider="ollama", model="m", options={"keepalive": "5m"}))
    monkeypatch.delenv("NO_SUCH_KEY_VARIABLE", raising=False)
    with pytest.raises(ConfigError, match=r"\$NO_SUCH_KEY_VARIABLE, which is not set"):
        create_provider(LLMSection(model="m", api_key_env="NO_SUCH_KEY_VARIABLE"))


# ---------------------------------------------------------------------- opt-in smoke tests


@pytest.mark.skipif(
    not os.environ.get("APEX_TEST_OLLAMA_MODEL"),
    reason="opt-in: set APEX_TEST_OLLAMA_MODEL (and APEX_LLM_BASE_URL) to use a local model",
)
def test_a_local_model_answers_through_the_loop(tmp_path: Any) -> None:  # noqa: ANN401
    from apex.agents.loop import AgentLoop, Fallback
    from apex.agents.trace import CallRecorder

    provider = OpenAICompatibleProvider(
        os.environ["APEX_TEST_OLLAMA_MODEL"],
        base_url=os.environ.get("APEX_LLM_BASE_URL", "http://localhost:11434/v1"),
    )
    outcome = AgentLoop(provider, CallRecorder(tmp_path, "smoke")).run(
        'Should a run stop? Answer {"stop": true}.',
        Verdict,
        Fallback("never", lambda: Verdict(stop=False)),
    )
    assert not outcome.fell_back, outcome.reason


@pytest.mark.skipif(
    not (os.environ.get("APEX_TEST_ANTHROPIC") and os.environ.get("ANTHROPIC_API_KEY")),
    reason="opt-in: set APEX_TEST_ANTHROPIC=1 and ANTHROPIC_API_KEY to use the Anthropic API",
)
def test_claude_answers_through_the_loop(tmp_path: Any) -> None:  # noqa: ANN401
    from apex.agents.loop import AgentLoop, Fallback
    from apex.agents.providers.anthropic import AnthropicProvider
    from apex.agents.trace import CallRecorder

    provider = AnthropicProvider(
        os.environ.get("APEX_TEST_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    )
    outcome = AgentLoop(provider, CallRecorder(tmp_path, "smoke")).run(
        'Should a run stop? Answer {"stop": true}.',
        Verdict,
        Fallback("never", lambda: Verdict(stop=False)),
    )
    assert not outcome.fell_back, outcome.reason


def test_ollama_asks_for_the_context_a_conversation_needs_up_to_its_maximum() -> None:
    """Release tester C8 (2026-10-24): a report of ~36k tokens was cut to Ollama's fixed
    32k. A request now asks for what it needs, in steps, between min and max context."""
    from apex.agents.providers.ollama import CHARS_PER_TOKEN

    def num_ctx(provider: OllamaProvider, tokens: int) -> int:
        long = [Message("user", "x" * int(tokens * CHARS_PER_TOKEN))]
        body = provider.request_body(long, (), None, 1_000, 0.0, None)
        return int(body["options"]["num_ctx"])

    unknown = _mock(lambda r: httpx.Response(404, json={"error": "not found"}), [])
    provider = OllamaProvider("gemma4:31b", http=unknown)
    assert provider.capabilities.max_context == 65_536  # the model's own context unknown
    assert num_ctx(provider, 100) == 32_768
    assert num_ctx(provider, 36_000) == 65_536  # powers of two: Ollama reloads rarely
    assert num_ctx(provider, 500_000) == 65_536
    small = OllamaProvider("gemma3:4b", max_context=16_384, http=unknown)
    assert num_ctx(small, 100) == 16_384  # never above max_context


def test_ollama_never_asks_for_more_context_than_the_model_has() -> None:
    """Review of the C8 fix: /api/show's context length bounds max_context, which the loop
    trims to, and every request; it is asked once."""
    seen: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"model_info": {"gemma3.context_length": 40_000}})
        return httpx.Response(200, json=OLLAMA_ANSWER)

    provider = OllamaProvider("gemma3:4b", http=_mock(answer, seen))
    assert provider.capabilities.max_context == 40_000  # what the loop trims to
    assert provider.min_context == 32_768
    long = [Message("user", "x" * 60_000)]
    provider.complete(long, max_tokens=256, temperature=0.0)
    provider.complete(long, max_tokens=256, temperature=0.0)
    assert [r.url.path for r in seen] == ["/api/show", "/api/chat", "/api/chat"]
    assert json.loads(seen[-1].content)["options"]["num_ctx"] == 40_000

    seen.clear()
    small = OllamaProvider("qwen", http=_mock(lambda r: httpx.Response(
        200, json={"model_info": {"qwen2.context_length": 16_384}}
        if r.url.path == "/api/show" else OLLAMA_ANSWER), seen))  # fmt: skip
    assert small.capabilities.max_context == 16_384
    small.complete([Message("user", "hi")], max_tokens=256, temperature=0.0)
    assert json.loads(seen[-1].content)["options"]["num_ctx"] == 16_384

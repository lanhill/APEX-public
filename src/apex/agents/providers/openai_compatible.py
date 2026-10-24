"""The ``openai_compatible`` provider: any server with the OpenAI chat-completions API.

It covers Ollama (``http://localhost:11434/v1``), vLLM, llama.cpp's server, LM Studio, and
hosted APIs such as Kimi (Moonshot), GLM (Zhipu), OpenRouter and OpenAI. A new
model or server needs no code, only ``base_url`` and ``model``. Tools go out in the
``tools`` format; servers or models without tool calling are declared with
``tool_calls = false``, and the agent loop then prompts for JSON. ``json_mode = true`` asks
for ``response_format = {"type": "json_object"}`` where the server supports it. A server
that rate-limits or is overloaded (HTTP 429, 503) and says when to try again is retried
after that wait, up to ``max_retries`` times and ``max_retry_wait_s`` seconds a wait; one
that does not say, or asks for longer, fails at once, and the agent falls back to its rule.
"""

import json
import logging
import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from pydantic import BaseModel

from apex.agents.providers._common import (
    HTTPFactory,
    api_key,
    http_errors,
    parse_arguments,
    response_json,
    retry_after,
    status_error,
)
from apex.core.errors import ConfigError
from apex.core.llm import Capabilities, Completion, LLMError, Message, ToolCall, ToolSpec, Usage
from apex.core.plugins import llm_providers

NAME = "openai_compatible"

logger = logging.getLogger(__name__)


def to_openai(message: Message) -> dict[str, Any]:
    """Return a message in the chat-completions format."""
    if message.role == "tool":
        body: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": message.tool_call_id or "",
            "content": message.content,
        }
        if message.name:
            body["name"] = message.name
        return body
    body = {"role": message.role, "content": message.content}
    if message.tool_calls:
        body["content"] = message.content or None
        body["tool_calls"] = [
            {
                **call.extra,  # e.g. Gemini's thought signature, which it requires back
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments or {})},
            }
            for call in message.tool_calls
        ]
    return body


def tool_definitions(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    """Return tools in the chat-completions ``tools`` format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": dict(t.parameters),
            },
        }
        for t in tools
    ]


class OpenAICompatibleProvider:
    """A chat-completions server (see the module docstring)."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        api_key_env: str | None = None,
        timeout_s: float = 120.0,
        tool_calls: bool = True,
        json_mode: bool = False,
        max_context: int = 32_768,
        max_retries: int = 2,
        max_retry_wait_s: float = 60.0,
        http: HTTPFactory | None = None,
    ) -> None:
        if not model:
            raise ConfigError("the openai_compatible provider needs a model")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s
        # Chat-completions servers stream with "stream": true (server-sent events).
        self._capabilities = Capabilities(tool_calls, json_mode, max_context, streaming=True)
        self._key = api_key(api_key_env, required=False, provider=NAME)
        self.max_retries = max_retries
        self.max_retry_wait_s = max_retry_wait_s
        self._http = http

    @property
    def model_id(self) -> str:
        """The model's name on the server."""
        return self.model

    @property
    def provider_name(self) -> str:
        """``openai_compatible``."""
        return NAME

    @property
    def capabilities(self) -> Capabilities:
        """As configured: servers do not say whether a model calls tools."""
        return self._capabilities

    def describe(self) -> dict[str, Any]:
        """Return the settings for traces (the key's variable, never the key)."""
        return {
            "provider": NAME,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "tool_calls": self._capabilities.tool_calls,
            "json_mode": self._capabilities.json_mode,
        }

    def request_body(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        response_model: type[BaseModel] | None,
        max_tokens: int,
        temperature: float,
        seed: int | None,
    ) -> dict[str, Any]:
        """Return the JSON body of a chat-completions request."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [to_openai(m) for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if seed is not None:
            body["seed"] = seed
        if tools:
            body["tools"] = tool_definitions(tools)
        if response_model is not None and self._capabilities.json_mode and not tools:
            body["response_format"] = {"type": "json_object"}
        return body

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
        """Return the model's answer (see :class:`apex.core.llm.LLMProvider`)."""
        import httpx

        body = self.request_body(messages, tools, response_model, max_tokens, temperature, seed)
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        for attempt in range(self.max_retries + 1):
            client = self._http() if self._http else httpx.Client(timeout=self.timeout_s)
            with http_errors(NAME), client:
                response = client.post(
                    f"{self.base_url}/chat/completions", json=body, headers=headers
                )
            # Rate limits of free tiers are per minute, and one agent task may exceed them
            # by itself; waiting as long as the server asks usually gets the answer.
            wait = retry_after(response)
            if wait is None or wait > self.max_retry_wait_s or attempt == self.max_retries:
                break
            logger.warning(
                "%s: HTTP %d, retrying in %.1f s (%d of %d)",
                NAME, response.status_code, wait, attempt + 1, self.max_retries,
                extra={"event": "llm_retry", "status": response.status_code, "wait_s": wait},
            )  # fmt: skip
            time.sleep(wait)
        return parse_response(response_json(response, NAME))

    def stream_complete(
        self,
        messages: Sequence[Message],
        *,
        on_text: Callable[[str], None],
        tools: Sequence[ToolSpec] = (),
        response_model: type[BaseModel] | None = None,
        max_tokens: int,
        temperature: float,
        seed: int | None = None,
    ) -> Completion:
        """As :meth:`complete`, handing the text over as it arrives (``"stream": true``)."""
        import httpx

        body = self.request_body(messages, tools, response_model, max_tokens, temperature, seed)
        body["stream"] = True
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        for attempt in range(self.max_retries + 1):
            client = self._http() if self._http else httpx.Client(timeout=self.timeout_s)
            with (
                http_errors(NAME),
                client,
                client.stream(
                    "POST", f"{self.base_url}/chat/completions", json=body, headers=headers
                ) as response,
            ):
                if response.status_code < 400:
                    return _read_stream(response.iter_lines(), on_text)
                response.read()
                wait = retry_after(response)
                if wait is None or wait > self.max_retry_wait_s or attempt == self.max_retries:
                    raise status_error(response.status_code, response.text, NAME)
            logger.warning(
                "%s: HTTP %d, retrying in %.1f s (%d of %d)",
                NAME, response.status_code, wait, attempt + 1, self.max_retries,
                extra={"event": "llm_retry", "status": response.status_code, "wait_s": wait},
            )  # fmt: skip
            time.sleep(wait)
        raise LLMError(f"{NAME}: no answer")  # not reached: the last attempt raises


def _read_stream(lines: Iterable[str], on_text: Callable[[str], None]) -> Completion:
    """Assemble a streamed chat completion from its server-sent events."""
    text: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    finish: str | None = None
    raw_id: str | None = None
    usage: dict[str, Any] = {}
    for line in lines:
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            event = json.loads(data)
        except json.JSONDecodeError as error:
            raise LLMError(f"{NAME}: a streamed event is not JSON: {data[:200]}") from error
        raw_id = event.get("id", raw_id)
        usage = event.get("usage") or usage
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                text.append(str(delta["content"]))
                on_text(str(delta["content"]))
            for part in delta.get("tool_calls") or []:
                call = calls.setdefault(int(part.get("index", 0)), {"function": {}})
                for key in ("id", "type"):
                    if part.get(key):
                        call[key] = part[key]
                function = part.get("function") or {}
                if function.get("name"):
                    call["function"]["name"] = function["name"]
                if function.get("arguments"):
                    call["function"]["arguments"] = (
                        call["function"].get("arguments", "") + function["arguments"]
                    )
            finish = choice.get("finish_reason") or finish
    message: dict[str, Any] = {"content": "".join(text)}
    if calls:
        message["tool_calls"] = [calls[i] for i in sorted(calls)]
    return parse_response(
        {"id": raw_id, "usage": usage, "choices": [{"message": message, "finish_reason": finish}]}
    )


def parse_response(body: dict[str, Any]) -> Completion:
    """Return the completion of a chat-completions response."""
    try:
        choice = body["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as error:
        raise LLMError(f"{NAME}: no choice in the answer: {str(body)[:200]}") from error
    calls: list[ToolCall] = []
    for index, raw in enumerate(message.get("tool_calls") or []):
        function = raw.get("function") or {}
        call_id = raw.get("id") or f"call_{index}"
        # Fields the server added to the call (e.g. extra_content) go back with it.
        extra = {k: v for k, v in raw.items() if k not in ("id", "type", "function", "index")}
        calls.append(
            parse_arguments(
                call_id, str(function.get("name", "")), function.get("arguments"), extra
            )
        )
    usage = body.get("usage") or {}
    return Completion(
        text=str(message.get("content") or ""),
        tool_calls=tuple(calls),
        usage=Usage(usage.get("prompt_tokens"), usage.get("completion_tokens")),
        raw_id=body.get("id"),
        stop_reason=choice.get("finish_reason"),
    )


@llm_providers.register(NAME)
def openai_compatible(
    model: str,
    base_url: str = "http://localhost:11434/v1",
    api_key_env: str | None = None,
    timeout_s: float = 120.0,
    tool_calls: bool = True,
    json_mode: bool = False,
    max_context: int = 32_768,
    max_retries: int = 2,
    max_retry_wait_s: float = 60.0,
) -> OpenAICompatibleProvider:
    """Return a provider for a chat-completions server."""
    return OpenAICompatibleProvider(
        model, base_url, api_key_env, timeout_s, tool_calls, json_mode, max_context,
        max_retries, max_retry_wait_s,
    )  # fmt: skip

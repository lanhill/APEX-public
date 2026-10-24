"""The ``ollama`` provider: Ollama's native ``/api/chat``.

``openai_compatible`` with ``base_url = "http://localhost:11434/v1"`` covers most uses of
Ollama. The native API adds what the OpenAI one lacks: the context length (``num_ctx``,
which Ollama otherwise keeps small), ``keep_alive``, a JSON Schema for structured answers
(``format``), and the list of local models (:meth:`OllamaProvider.models`).

Each request asks for the context it needs, at least ``min_context`` and at most
``max_context`` and the model's own context (Ollama's ``/api/show``, asked once, when the
loop first reads the capabilities, so that it trims a conversation to the same limit): a
fixed 32k cut long reports short (release tester C8, 2026-10-24), while a context as large
as the model's on every call would cost a local model its memory for nothing. Sizes are
powers of two, so that a growing conversation makes Ollama reload the model rarely.
"""

import json
import math
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from pydantic import BaseModel

from apex.agents.providers._common import (
    HTTPFactory,
    api_key,
    http_errors,
    parse_arguments,
    response_json,
)
from apex.agents.providers.openai_compatible import tool_definitions
from apex.core.errors import ConfigError
from apex.core.llm import Capabilities, Completion, LLMError, Message, ToolCall, ToolSpec, Usage
from apex.core.plugins import llm_providers

NAME = "ollama"
#: Seconds to wait for ``/api/show`` (the model's context): no answer leaves it unknown.
SHOW_TIMEOUT_S = 5.0
#: Characters per token when sizing a request's context: the agent loop's estimate, on the
#: safe side for JSON and code.
CHARS_PER_TOKEN = 1.5


def to_ollama(message: Message) -> dict[str, Any]:
    """Return a message in Ollama's chat format."""
    body: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.role == "tool" and message.name:
        body["tool_name"] = message.name
    if message.tool_calls:
        body["tool_calls"] = [
            {"function": {"name": call.name, "arguments": dict(call.arguments or {})}}
            for call in message.tool_calls
        ]
    return body


class OllamaProvider:
    """A model served by Ollama (see the module docstring)."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        api_key_env: str | None = None,
        timeout_s: float = 120.0,
        tool_calls: bool = True,
        json_mode: bool = True,
        max_context: int = 65_536,
        keep_alive: str | None = None,
        http: HTTPFactory | None = None,
        min_context: int = 32_768,
    ) -> None:
        if not model:
            raise ConfigError("the ollama provider needs a model")
        self.model = model
        # Also accept the OpenAI-compatible address of the same server.
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s
        self.keep_alive = keep_alive
        self.min_context = min(min_context, max_context)
        self._capabilities = Capabilities(tool_calls, json_mode, max_context)
        self._key = api_key(api_key_env, required=False, provider=NAME)
        self._http = http
        self._model_context: int | None = None
        self._asked_context = False

    @property
    def model_id(self) -> str:
        """The model's name in Ollama (``qwen3:32b``, ``glm-4.6``, ...)."""
        return self.model

    @property
    def provider_name(self) -> str:
        """``ollama``."""
        return NAME

    @property
    def capabilities(self) -> Capabilities:
        """As configured, with ``max_context`` no more than the model's own context.

        The loop trims a conversation to ``max_context``; a model with less context than
        configured would otherwise get a prompt Ollama silently cuts (review of the C8 fix).
        """
        own = self.model_context()
        if own is not None and own < self._capabilities.max_context:
            self._capabilities = replace(self._capabilities, max_context=own)
            self.min_context = min(self.min_context, own)
        return self._capabilities

    def describe(self) -> dict[str, Any]:
        """Return the settings for traces (the key's variable, never the key)."""
        return {
            "provider": NAME,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "max_context": self._capabilities.max_context,
            "min_context": self.min_context,
            "keep_alive": self.keep_alive,
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
        """Return the JSON body of an ``/api/chat`` request."""
        options: dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": 0,  # set below, from the whole body
        }
        if seed is not None:
            options["seed"] = seed
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [to_ollama(m) for m in messages],
            "stream": False,
            "options": options,
        }
        if tools:
            body["tools"] = tool_definitions(tools)
        elif response_model is not None and self._capabilities.json_mode:
            body["format"] = response_model.model_json_schema()
        if self.keep_alive is not None:
            body["keep_alive"] = self.keep_alive
        needed = len(json.dumps(body, default=str)) / CHARS_PER_TOKEN + max_tokens
        stepped = 2 ** math.ceil(math.log2(max(needed, 1)))
        options["num_ctx"] = min(self._capabilities.max_context, max(self.min_context, stepped))
        return body

    def _client(self) -> Any:  # noqa: ANN401 (httpx.Client)
        import httpx

        headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
        return self._http() if self._http else httpx.Client(timeout=self.timeout_s, headers=headers)

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
        self.capabilities  # noqa: B018 (bounds max_context by the model's own, once)
        body = self.request_body(messages, tools, response_model, max_tokens, temperature, seed)
        with http_errors(NAME), self._client() as client:
            response = client.post(f"{self.base_url}/api/chat", json=body)
        return parse_response(response_json(response, NAME))

    def model_context(self) -> int | None:
        """Return the model's own context length (``/api/show``), or ``None`` if unknown.

        Asked once, briefly: a cloud model, an older Ollama, no server or a failed request
        leave it unknown, and ``max_context`` alone then bounds the request.
        """
        if not self._asked_context:
            self._asked_context = True
            try:
                with http_errors(NAME), self._client() as client:
                    response = client.post(
                        f"{self.base_url}/api/show", json={"model": self.model},
                        timeout=SHOW_TIMEOUT_S,
                    )  # fmt: skip
                info = response_json(response, NAME).get("model_info") or {}
            except (LLMError, ValueError, TypeError, AttributeError):
                info = {}
            lengths = [v for k, v in info.items()
                       if str(k).endswith(".context_length") and isinstance(v, int)]  # fmt: skip
            self._model_context = min(lengths) if lengths else None
        return self._model_context

    def models(self) -> list[str]:
        """Return the names of the models Ollama has locally."""
        with http_errors(NAME), self._client() as client:
            response = client.get(f"{self.base_url}/api/tags")
        return [str(m["name"]) for m in response_json(response, NAME).get("models", [])]


def parse_response(body: dict[str, Any]) -> Completion:
    """Return the completion of an ``/api/chat`` response."""
    message = body.get("message")
    if not isinstance(message, dict):
        raise LLMError(f"{NAME}: no message in the answer: {str(body)[:200]}")
    calls: list[ToolCall] = []
    for index, raw in enumerate(message.get("tool_calls") or []):
        function = raw.get("function") or {}
        call_id = raw.get("id") or f"call_{index}"
        calls.append(
            parse_arguments(call_id, str(function.get("name", "")), function.get("arguments"))
        )
    return Completion(
        text=str(message.get("content") or ""),
        tool_calls=tuple(calls),
        usage=Usage(body.get("prompt_eval_count"), body.get("eval_count")),
        raw_id=body.get("created_at"),
        stop_reason=body.get("done_reason"),
    )


@llm_providers.register(NAME)
def ollama(
    model: str,
    base_url: str = "http://localhost:11434",
    api_key_env: str | None = None,
    timeout_s: float = 120.0,
    tool_calls: bool = True,
    json_mode: bool = True,
    max_context: int = 65_536,
    keep_alive: str | None = None,
    min_context: int = 32_768,
) -> OllamaProvider:
    """Return a provider for a model served by Ollama.

    ``max_context`` is the most context a request asks for (never more than the model's
    own); ``min_context`` the least.
    """
    return OllamaProvider(
        model,
        base_url,
        api_key_env,
        timeout_s,
        tool_calls,
        json_mode,
        max_context,
        keep_alive,
        min_context=min_context,
    )

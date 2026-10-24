"""The ``anthropic`` provider: Claude models through the Messages API.

It uses the ``anthropic`` SDK (the ``llm`` extra) with native tool use. The key comes from
the environment variable named by ``api_key_env`` (default ``ANTHROPIC_API_KEY``). The
Messages API of the SDK in use (1.9) takes neither a temperature nor a seed, so both are
ignored, and repeated decisions are only as repeatable as the model; the trace records
each answer. The SDK retries transient errors itself (``max_retries``); what still fails
reaches the agent loop as :class:`~apex.core.llm.LLMError`, and the loop falls back.
"""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from pydantic import BaseModel

from apex.agents.providers._common import api_key
from apex.core.errors import ConfigError
from apex.core.llm import (
    Capabilities,
    Completion,
    LLMError,
    LLMTimeout,
    LLMUnavailable,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
)
from apex.core.plugins import llm_providers

NAME = "anthropic"


def to_anthropic(messages: Sequence[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Return the system text and the Messages-API messages of a conversation.

    Tool results travel as ``tool_result`` blocks of a user message, and consecutive
    messages of one role are merged, as the API requires alternating roles.
    """
    system = "\n\n".join(m.content for m in messages if m.role == "system")
    out: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            continue
        blocks: list[dict[str, Any]]
        if message.role == "tool":
            role = "user"
            blocks = [
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id or "",
                    "content": message.content,
                }
            ]
        else:
            role = message.role
            blocks = [{"type": "text", "text": message.content}] if message.content else []
            blocks += [
                {"type": "tool_use", "id": c.id, "name": c.name, "input": dict(c.arguments or {})}
                for c in message.tool_calls
            ]
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": blocks})
    return system, out


class AnthropicProvider:
    """Claude through the Messages API (see the module docstring)."""

    def __init__(
        self,
        model: str,
        api_key_env: str = "ANTHROPIC_API_KEY",
        base_url: str | None = None,
        timeout_s: float = 120.0,
        max_context: int = 200_000,
        max_retries: int = 2,
        http_client: Any = None,  # noqa: ANN401 (an httpx client, for tests)
    ) -> None:
        if not model:
            raise ConfigError("the anthropic provider needs a model, e.g. claude-opus-5-5")
        try:
            import anthropic
        except ImportError as error:
            raise ConfigError(
                "the anthropic provider needs the llm extra: uv sync --inexact --extra llm"
            ) from error
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url
        self._capabilities = Capabilities(
            tool_calls=True, json_mode=False, max_context=max_context, streaming=True
        )
        self._client = anthropic.Anthropic(
            api_key=api_key(api_key_env, required=True, provider=NAME),
            base_url=base_url,
            timeout=timeout_s,
            max_retries=max_retries,
            http_client=http_client,
        )

    @property
    def model_id(self) -> str:
        """The Claude model id."""
        return self.model

    @property
    def provider_name(self) -> str:
        """``anthropic``."""
        return NAME

    @property
    def capabilities(self) -> Capabilities:
        """Native tool use; answers are parsed from text (no JSON mode)."""
        return self._capabilities

    def describe(self) -> dict[str, Any]:
        """Return the settings for traces (the key's variable, never the key)."""
        return {
            "provider": NAME,
            "model": self.model,
            "api_key_env": self.api_key_env,
            "base_url": self.base_url,
        }

    def request_arguments(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        max_tokens: int,
    ) -> dict[str, Any]:
        """Return the arguments of ``messages.create``."""
        system, converted = to_anthropic(messages)
        # No temperature: current Messages API models do not take one (see the module
        # docstring).
        arguments: dict[str, Any] = {
            "model": self.model,
            "messages": converted,
            "max_tokens": max_tokens,
        }
        if system:
            arguments["system"] = system
        if tools:
            arguments["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": dict(t.parameters)}
                for t in tools
            ]
        return arguments

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
        arguments = self.request_arguments(messages, tools, max_tokens)
        with _api_errors():
            response = self._client.messages.create(**arguments)
        return _completion(response)

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
        """As :meth:`complete`, handing the text over as it arrives (``messages.stream``)."""
        arguments = self.request_arguments(messages, tools, max_tokens)
        with _api_errors(), self._client.messages.stream(**arguments) as stream:
            for text in stream.text_stream:
                on_text(text)
            response = stream.get_final_message()
        return _completion(response)


@contextmanager
def _api_errors() -> Iterator[None]:
    """Turn the SDK's errors into :class:`LLMError` subclasses."""
    import anthropic

    try:
        yield
    except anthropic.APITimeoutError as error:
        raise LLMTimeout(f"{NAME}: no answer before the timeout") from error
    except anthropic.APIConnectionError as error:
        raise LLMUnavailable(f"{NAME}: cannot reach the API ({error})") from error
    except anthropic.APIStatusError as error:
        status = error.status_code
        detail = f"{NAME}: HTTP {status}: {error.message}"
        if status >= 500 or status in (408, 429, 529):
            raise LLMUnavailable(detail) from error
        raise LLMError(detail) from error


def _completion(response: Any) -> Completion:  # noqa: ANN401 (an anthropic Message)
    """Return the completion of a Messages API response."""
    text: list[str] = []
    calls: list[ToolCall] = []
    for block in response.content:
        if block.type == "text":
            text.append(block.text)
        elif block.type == "tool_use":
            arguments_ = block.input if isinstance(block.input, dict) else {}
            calls.append(ToolCall(block.id, block.name, arguments_))
    return Completion(
        text="".join(text),
        tool_calls=tuple(calls),
        usage=Usage(response.usage.input_tokens, response.usage.output_tokens),
        raw_id=response.id,
        stop_reason=response.stop_reason,
    )


@llm_providers.register(NAME)
def anthropic_provider(
    model: str,
    api_key_env: str = "ANTHROPIC_API_KEY",
    base_url: str | None = None,
    timeout_s: float = 120.0,
    max_context: int = 200_000,
    max_retries: int = 2,
) -> AnthropicProvider:
    """Return a provider for a Claude model."""
    return AnthropicProvider(model, api_key_env, base_url, timeout_s, max_context, max_retries)

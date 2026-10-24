"""The ``stub`` provider: scripted replies, for tests and dry runs.

Tests must never use a live model (runtime-agent rules), so every agent test drives the
loop with a :class:`StubProvider`. Each request takes the next scripted reply:
- a :class:`~apex.core.llm.Completion`, returned as it is;
- a string, returned as the completion's text;
- an :class:`~apex.core.llm.LLMError`, raised (e.g. ``LLMTimeout`` to test the fallback);
- a callable, called with the request and returning one of the above.

When the script runs out, the provider is unavailable. Every request is kept in
``requests`` for assertions.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from apex.core.llm import (
    Capabilities,
    Completion,
    LLMError,
    LLMUnavailable,
    Message,
    ToolSpec,
    Usage,
)
from apex.core.plugins import llm_providers


@dataclass(frozen=True)
class StubRequest:
    """One request the stub received."""

    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...]
    response_model: type[BaseModel] | None
    max_tokens: int
    temperature: float
    seed: int | None


Reply = Completion | str | LLMError | Callable[[StubRequest], "Completion | str"]


@dataclass
class StubProvider:
    """Scripted replies (see the module docstring)."""

    replies: Sequence[Reply] = ()
    model: str = "stub-model"
    tool_calls: bool = True
    json_mode: bool = False
    max_context: int = 32_768
    #: Stream each reply's text in pieces of ``chunk`` characters.
    streaming: bool = False
    chunk: int = 7
    requests: list[StubRequest] = field(default_factory=list, init=False)
    _next: int = field(default=0, init=False)

    @property
    def model_id(self) -> str:
        """The scripted model's name."""
        return self.model

    @property
    def provider_name(self) -> str:
        """``stub``."""
        return "stub"

    @property
    def capabilities(self) -> Capabilities:
        """As configured, to test both native and prompted tool calling."""
        return Capabilities(self.tool_calls, self.json_mode, self.max_context, self.streaming)

    def describe(self) -> dict[str, Any]:
        """Return the stub's settings."""
        return {"provider": "stub", "model": self.model, "replies": len(self.replies)}

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
        """Return the next scripted reply."""
        request = StubRequest(
            tuple(messages), tuple(tools), response_model, max_tokens, temperature, seed
        )
        self.requests.append(request)
        if self._next >= len(self.replies):
            raise LLMUnavailable(f"the stub has no reply left ({len(self.replies)} scripted)")
        reply = self.replies[self._next]
        self._next += 1
        if isinstance(reply, LLMError):
            raise reply
        if callable(reply) and not isinstance(reply, Completion):
            reply = reply(request)
        if isinstance(reply, str):
            return Completion(text=reply, usage=Usage(len(str(messages)) // 4, len(reply) // 4))
        return reply

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
        """Return the next scripted reply, handing its text over in pieces."""
        completion = self.complete(
            messages, tools=tools, response_model=response_model, max_tokens=max_tokens,
            temperature=temperature, seed=seed,
        )  # fmt: skip
        for start in range(0, len(completion.text), self.chunk):
            on_text(completion.text[start : start + self.chunk])
        return completion


@llm_providers.register("stub")
def stub(
    model: str = "stub-model",
    replies: Sequence[str] = (),
    tool_calls: bool = True,
    json_mode: bool = False,
    max_context: int = 32_768,
) -> StubProvider:
    """Return a stub from configuration: text replies only (``provider = "stub"``)."""
    return StubProvider(list(replies), model, tool_calls, json_mode, max_context)

"""Language-model providers for APEX's runtime agents.

An :class:`LLMProvider` turns a conversation (and optionally the tools and a
pydantic model the answer must match) into one :class:`Completion`. Providers are plugins
in the ``llm_providers`` registry (``apex.core.plugins``): ``anthropic`` (the Messages
API), ``openai_compatible`` (Ollama, vLLM, llama.cpp, LM Studio, Kimi, GLM, OpenRouter,
OpenAI), ``ollama`` (Ollama's native API) and ``stub`` (scripted replies, the only provider
tests may use). Implementations live in ``apex.agents.providers``.

The interface is deliberately small, so that weak and strong models are driven the same
way: a provider states what it can do (:class:`Capabilities`), and the agent loop
(``apex.agents.loop``) adapts: prompted JSON where there is no native tool calling,
validation of every answer, and a rule-based fallback when a provider fails
(:class:`LLMError`).
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from apex.core.errors import ApexError
from apex.domain.config import _Strict

Role = Literal["system", "user", "assistant", "tool"]


class LLMError(ApexError):
    """A provider could not produce a completion; the agent falls back to its rule."""


class LLMUnavailable(LLMError):
    """The provider cannot be reached (connection refused, server error, no such model)."""


class LLMTimeout(LLMUnavailable):
    """The provider did not answer within its timeout."""


class LLMInvalidOutput(LLMError):
    """The model answered, but the provider could not hand the answer over.

    For example, Claude through ``claude -p`` (which has no tools) attempting a native tool
    call. The agent loop asks again, as it does for an invalid answer.
    """


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One call of a tool that a model asked for.

    Attributes
    ----------
    id
        The provider's id for the call, echoed back with its result.
    name
        The tool's name.
    arguments
        The arguments, parsed from JSON. ``None`` with ``error`` set when the model's
        arguments were not a JSON object.
    error
        Why the arguments could not be parsed, if they could not.
    extra
        Further fields the server attached to the call, echoed back unchanged with the
        conversation. Gemini 3 models, for example, attach a thought signature
        (``extra_content``) and refuse a follow-up request without it.
    """

    id: str
    name: str
    arguments: Mapping[str, Any] | None
    error: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Message:
    """One message of a conversation.

    Attributes
    ----------
    role
        ``system``, ``user``, ``assistant`` or ``tool``.
    content
        The text (for a ``tool`` message, the tool's result as JSON).
    tool_calls
        For an ``assistant`` message, the tools it called.
    tool_call_id
        For a ``tool`` message, the id of the call it answers.
    name
        For a ``tool`` message, the tool's name.
    """

    role: Role
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool offered to a model: name, description and a JSON Schema of its arguments."""

    name: str
    description: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Usage:
    """Tokens a call used, as the provider reports them (``None`` when it does not)."""

    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class Completion:
    """A model's answer to one request.

    Attributes
    ----------
    text
        The answer's text (empty when the model only called tools).
    tool_calls
        The tools it called (native tool calling only).
    parsed
        The answer validated as the request's ``response_model``, when the provider
        parsed it (e.g. from a JSON mode); the agent loop parses ``text`` otherwise.
    usage
        Tokens used.
    raw_id
        The provider's id of the response, for tracing.
    stop_reason
        Why the model stopped, in the provider's words (``end_turn``, ``length``, ...).
    """

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    parsed: BaseModel | None = None
    usage: Usage = field(default_factory=Usage)
    raw_id: str | None = None
    stop_reason: str | None = None


@dataclass(frozen=True, slots=True)
class Capabilities:
    """What a provider's model can do.

    Attributes
    ----------
    tool_calls
        Native tool calling. Without it the agent loop prompts for JSON tool calls.
    json_mode
        The server can constrain the answer to JSON.
    max_context
        Context length in tokens; the loop trims the conversation to fit.
    streaming
        The provider has ``stream_complete`` (:class:`StreamingProvider`): it hands over
        the answer's text as it arrives.
    """

    tool_calls: bool = True
    json_mode: bool = False
    max_context: int = 32_768
    streaming: bool = False


class LLMProvider(Protocol):
    """A language model behind some API (see the module docstring)."""

    @property
    def model_id(self) -> str:
        """The model's name, as the provider knows it (recorded with every call)."""
        ...

    @property
    def provider_name(self) -> str:
        """The provider plugin's name (``anthropic``, ``openai_compatible``, ...)."""
        ...

    @property
    def capabilities(self) -> Capabilities:
        """What the model can do."""
        ...

    def describe(self) -> dict[str, Any]:
        """Return the provider's settings for traces: never keys or other secrets."""
        ...

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
        """Return the model's answer to ``messages``.

        Raises
        ------
        LLMError
            When no answer can be had (``LLMUnavailable``, ``LLMTimeout``, or a refused
            request).
        """
        ...


class StreamingProvider(LLMProvider, Protocol):
    """A provider that can also hand over the answer's text as it arrives.

    ``capabilities.streaming`` says it has :meth:`stream_complete`.
    """

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
        """As :meth:`LLMProvider.complete`, calling ``on_text`` with each piece of text.

        Raises
        ------
        LLMError
            As :meth:`LLMProvider.complete`.
        """
        ...


# In core so that the settings of agents other packages contribute may use it (core.agents);
# services.settings re-exports it. Its docstring is the schema's.
class LLMSection(_Strict):
    """``[agents.llm]``: the model runtime agents use.

    Keys come from the environment variable named by ``api_key_env``; they are never
    written to a file or a log.
    """

    provider: str = "openai_compatible"
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    # Deterministic answers where the server honours it; agents decide, they do not muse.
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_s: float = Field(default=120.0, gt=0)
    # Enough for a decision with its reasons; tool results are bounded separately.
    max_tokens: int = Field(default=2048, ge=16)
    max_context: int | None = Field(default=None, ge=1024)
    tool_calls: bool | None = Field(
        default=None, description="Override the provider's native tool calling (false: JSON)."
    )
    json_mode: bool | None = None
    options: dict[str, Any] = Field(
        default_factory=dict, description="Further provider options (e.g. Ollama keep_alive)."
    )

"""Models tried in order: the next when one is unavailable.

A web-UI guest's assistant answers with the first model of their list that works: free
models come and go, and one withdrawn, over its limit or unreachable hands its requests to
the next. A model is left for the rest of the turn once it fails; a model whose settings
cannot even make a provider (a key not set) is skipped. When every model has failed,
the request fails with each one's reason, and the agent falls back to its rule.

Only unavailability moves on (:class:`~apex.core.llm.LLMUnavailable`, timeouts included):
an answer the agent loop finds invalid is the model's own, and is asked again as usual.
"""

import logging
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel

from apex.core.errors import ApexError
from apex.core.llm import (
    Capabilities,
    Completion,
    LLMProvider,
    LLMUnavailable,
    Message,
    ToolSpec,
)

log = logging.getLogger(__name__)


class FallbackProvider:
    """The first of several models that answers (see the module docstring).

    Parameters
    ----------
    models
        ``(name, factory)`` pairs in the order they are tried; each factory makes its
        model's provider when it is first needed.
    """

    def __init__(self, models: Sequence[tuple[str, Callable[[], LLMProvider]]]) -> None:
        if not models:
            raise ValueError("a fallback needs at least one model")
        self._models = list(models)
        self._index = 0
        self._made: dict[int, LLMProvider] = {}
        self._failures: list[str] = []
        self._should_stop: Callable[[], bool] | None = None

    # ------------------------------------------------------------------ the active model

    @property
    def name(self) -> str:
        """The name of the model now answering (``[chat.models]``'s), or the last one tried."""
        return self._models[self._shown()[0]][0]

    @property
    def chat_model(self) -> str:
        """The answering model's name, as the chat records it (same as :attr:`name`)."""
        return self.name

    def _current(self) -> LLMProvider:
        while self._index < len(self._models):
            made = self._made.get(self._index)
            if made is not None:
                return made
            name, factory = self._models[self._index]
            try:
                made = factory()
            except ApexError as error:
                self._skip(f"{name}: {error}")
                continue
            if self._should_stop is not None and hasattr(made, "should_stop"):
                made.should_stop = self._should_stop
            self._made[self._index] = made
            return made
        raise LLMUnavailable("no model available: " + "; ".join(self._failures))

    def _shown(self) -> tuple[int, LLMProvider]:
        """Return the model to describe: the answering one, else the last that was made.

        After every model has failed, the turn still records which one it tried last (the
        agent falls back; it must not fail on reading the model's name).
        """
        try:
            return self._index, self._current()
        except LLMUnavailable:
            if not self._made:
                raise
            last = max(self._made)
            return last, self._made[last]

    def _skip(self, reason: str) -> None:
        log.warning("chat model unavailable, trying the next: %s", reason)
        self._failures.append(reason)
        self._index += 1

    @property
    def should_stop(self) -> Callable[[], bool] | None:
        """The Stop button's flag, passed to every model that takes one."""
        return self._should_stop

    @should_stop.setter
    def should_stop(self, value: Callable[[], bool] | None) -> None:
        self._should_stop = value
        for made in self._made.values():
            if hasattr(made, "should_stop"):
                made.should_stop = value

    # ------------------------------------------------------------------ LLMProvider

    @property
    def model_id(self) -> str:
        """The answering model's id."""
        return self._shown()[1].model_id

    @property
    def provider_name(self) -> str:
        """The answering model's provider."""
        return self._shown()[1].provider_name

    @property
    def capabilities(self) -> Capabilities:
        """The answering model's capabilities."""
        return self._shown()[1].capabilities

    def describe(self) -> dict[str, Any]:
        """Return the answering model's settings, its name and the models left behind."""
        return {**self._shown()[1].describe(), "chat_model": self.name,
                "unavailable": list(self._failures)}  # fmt: skip

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
        """Return the first available model's answer.

        Raises
        ------
        LLMUnavailable
            When every model is unavailable (each one's reason in the message).
        """
        while True:
            provider, name = self._current(), self.name
            try:
                return provider.complete(messages, tools=tools, response_model=response_model,
                                         max_tokens=max_tokens, temperature=temperature,
                                         seed=seed)  # fmt: skip
            except LLMUnavailable as error:
                self._skip(f"{name}: {error}")

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
        """As :meth:`complete`, streaming with a model that can (the whole text otherwise).

        Raises
        ------
        LLMUnavailable
            When every model is unavailable.
        """
        while True:
            provider, name = self._current(), self.name
            stream = getattr(provider, "stream_complete", None)
            try:
                if stream is not None and provider.capabilities.streaming:
                    done: Completion = stream(messages, on_text=on_text, tools=tools,
                                              response_model=response_model,
                                              max_tokens=max_tokens,
                                              temperature=temperature, seed=seed)  # fmt: skip
                    return done
                done = provider.complete(messages, tools=tools, response_model=response_model,
                                         max_tokens=max_tokens, temperature=temperature,
                                         seed=seed)  # fmt: skip
            except LLMUnavailable as error:
                self._skip(f"{name}: {error}")
                continue
            if done.text:
                on_text(done.text)
            return done

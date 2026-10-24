"""The LLM provider plugins, and the provider a configuration names.

Importing this package registers ``anthropic``, ``claude_code``, ``openai_compatible``,
``ollama`` and ``stub`` in ``apex.core.plugins.llm_providers``; other packages can add
providers through the ``apex.llm_providers`` entry-point group.
"""

import inspect
import logging
from typing import Any

from apex.agents.providers import (  # noqa: F401
    anthropic,
    claude_code,
    ollama,
    openai_compatible,
    stub,
)
from apex.core.errors import ApexError, ConfigError, PluginError
from apex.core.llm import LLMProvider
from apex.core.plugins import llm_providers
from apex.services.settings import LLMSection

#: The ``[agents.llm]`` keys every HTTP provider takes, passed on when they are set.
_COMMON = (
    "model",
    "base_url",
    "api_key_env",
    "timeout_s",
    "max_context",
    "tool_calls",
    "json_mode",
)


#: What each built-in provider is, for listings (``apex agent providers``); a provider a
#: package adds is described by its factory's docstring.
DESCRIPTIONS: dict[str, str] = {
    "anthropic": "Claude models through Anthropic's API (the llm extra; the key in "
    "api_key_env). Run facts go to Anthropic.",
    "claude_code": "Claude through the Claude Code CLI (claude -p), with no tools of its own. "
    "Run facts go to Anthropic.",
    "ollama": "A model served by Ollama (local or remote); with a local server nothing "
    "leaves the machine.",
    "openai_compatible": "Any chat-completions server: vLLM, llama.cpp, Groq, Gemini, ... "
    "Run facts go to that server.",
    "stub": "Fixed replies from the configuration, for tests (no model).",
}


def create_provider(settings: LLMSection) -> LLMProvider:
    """Return the provider ``settings`` (``[agents.llm]``) name.

    Raises
    ------
    ConfigError
        If the provider is unknown, lacks a setting (a model, a key), or does not take
        one of the options given.
    """
    try:
        factory = llm_providers.factory(settings.provider)
    except PluginError as error:
        raise ConfigError(str(error)) from error
    # The common keys go to the providers that take them (the stub takes no timeout);
    # `options` are the provider's own, and a misspelt one fails below.
    accepted = inspect.signature(factory).parameters
    options: dict[str, Any] = {
        key: getattr(settings, key)
        for key in _COMMON
        if key in accepted and getattr(settings, key) is not None
    }
    options.update(settings.options)
    missing = [
        name
        for name, parameter in accepted.items()
        if parameter.default is inspect.Parameter.empty and name not in options
    ]
    if missing:
        raise ConfigError(
            f"[agents.llm] provider {settings.provider!r} needs {', '.join(missing)} "
            f"(in service.toml, or e.g. --model)"
        )
    try:
        return llm_providers.create(settings.provider, **options)
    except PluginError as error:
        raise ConfigError(str(error)) from error
    except TypeError as error:
        raise ConfigError(f"[agents.llm] provider {settings.provider!r}: {error}") from error


def provider_or_rules(settings: LLMSection) -> LLMProvider | None:
    """Return the model of ``settings``, or ``None`` (rules only) if it cannot be set up."""
    try:
        return create_provider(settings)
    except ApexError as error:
        logging.getLogger(__name__).warning(
            "rules only: %s", error, extra={"event": "rules_only", "reason": str(error)}
        )
        return None

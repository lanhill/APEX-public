"""Helpers shared by the HTTP providers: keys from the environment, errors, JSON."""

import json
import os
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from apex.core.errors import ConfigError
from apex.core.llm import LLMError, LLMTimeout, LLMUnavailable, ToolCall


def api_key(variable: str | None, *, required: bool, provider: str) -> str | None:
    """Return the key held by the environment variable ``variable`` (never logged).

    Raises
    ------
    ConfigError
        If the variable is named (or required) but not set.
    """
    if variable is None:
        if required:
            raise ConfigError(
                f"the {provider} provider needs api_key_env, the variable holding the key"
            )
        return None
    value = os.environ.get(variable)
    if not value:
        raise ConfigError(
            f"the {provider} provider reads its key from ${variable}, which is not set"
        )
    return value


#: Statuses a server may ask to be retried after: rate limited, overloaded.
RETRYABLE = frozenset({429, 503})


def retry_after(response: Any) -> float | None:  # noqa: ANN401 (httpx.Response)
    """Return the seconds a rate-limited or overloaded server asks to wait, if it says.

    From the ``Retry-After`` header (seconds), else from the text "try again in 19.5s" that
    Groq and OpenAI put in the error message. ``None``: no stated wait, so no retry.
    """
    if response.status_code not in RETRYABLE:
        return None
    header = response.headers.get("retry-after")
    if header:
        try:
            return max(0.0, float(header))
        except ValueError:
            pass
    match = re.search(r"try again in (\d+(?:\.\d+)?)\s*(ms|s)\b", response.text)
    if match is None:
        return None
    value = float(match.group(1))
    return value / 1000 if match.group(2) == "ms" else value


def status_error(status: int, body: str, provider: str) -> LLMError:
    """Map an HTTP error status to the error the agent loop falls back on."""
    detail = body[:500]
    # Overloaded, rate limited or broken servers may answer later; say so to the loop.
    if status >= 500 or status in (408, 429, 529):
        return LLMUnavailable(f"{provider}: HTTP {status}: {detail}")
    return LLMError(f"{provider}: HTTP {status}: {detail}")


@contextmanager
def http_errors(provider: str) -> Iterator[None]:
    """Turn httpx's transport errors into :class:`LLMError` subclasses."""
    import httpx

    try:
        yield
    except httpx.TimeoutException as error:
        raise LLMTimeout(f"{provider}: no answer before the timeout ({error})") from error
    except httpx.TransportError as error:
        raise LLMUnavailable(f"{provider}: cannot reach the server ({error})") from error


def parse_arguments(
    call_id: str, name: str, raw: object, extra: dict[str, Any] | None = None
) -> ToolCall:
    """Return a tool call, parsing JSON arguments given as text (OpenAI style)."""
    extra = extra or {}
    if isinstance(raw, dict):
        return ToolCall(call_id, name, raw, extra=extra)
    if raw in (None, ""):
        return ToolCall(call_id, name, {}, extra=extra)
    try:
        value = json.loads(str(raw))
    except json.JSONDecodeError as error:
        return ToolCall(call_id, name, None, f"arguments are not JSON: {error}", extra)
    if not isinstance(value, dict):
        return ToolCall(call_id, name, None, "arguments must be a JSON object", extra)
    return ToolCall(call_id, name, value, extra=extra)


def response_json(response: Any, provider: str) -> dict[str, Any]:  # noqa: ANN401 (httpx.Response)
    """Return a JSON response body, or raise the matching :class:`LLMError`."""
    if response.status_code >= 400:
        raise status_error(response.status_code, response.text, provider)
    try:
        body = response.json()
    except ValueError as error:
        raise LLMError(f"{provider}: the answer is not JSON: {response.text[:200]}") from error
    if not isinstance(body, dict):
        raise LLMError(f"{provider}: unexpected answer {str(body)[:200]}")
    return body


#: How an HTTP provider gets its client: tests inject one with a mock transport.
HTTPFactory = Callable[[], Any]

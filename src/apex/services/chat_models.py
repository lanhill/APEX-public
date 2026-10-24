"""Which models a web-UI user's assistant answers with.

``service.toml``'s ``[chat.models]`` names the models. The owner chooses any of them in the
web UI, ``[chat] default_model`` first. A guest gets the models ``ui_access.toml``'s
``[chat] models`` lists, tried in that order (the next when one is unavailable) and not
theirs to choose; the web UI tells them so, plainly. Other kinds of web-UI user (ones an
installed package adds) answer with the owner's default. The agents' buttons follow the same
rule for guests; the owner's, and other kinds', keep ``[agents.llm]``.

Without ``[chat.models]`` nothing changes: every chat answers with ``[chat] llm``, else
``[agents.llm]``, and every agent with ``[agents.llm]``.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlparse

from apex.services.settings import ChatSection, LLMSection

#: Who asks: the owner, a guest, or another kind of web-UI user.
Asker = Literal["owner", "guest", "other"]


@dataclass(frozen=True)
class ModelChoice:
    """The models one web-UI user's assistant may answer with.

    Attributes
    ----------
    names
        The models, in the order they are tried (empty: no model, or none configured).
    default
        The model a chat answers with unless the user picks another (``None``: none).
    choosable
        Whether the user may pick among ``names`` (the owner only).
    note
        What the web UI tells a guest about their model, or ``None``.
    configured
        Whether the service defines ``[chat.models]`` at all.
    """

    names: tuple[str, ...]
    default: str | None
    choosable: bool
    note: str | None
    configured: bool = True

    def ordered(self, requested: str | None) -> tuple[str, ...]:
        """Return the models to try for a turn: ``requested`` first, then the user's others.

        The owner's choice is the one model asked for (an owner who picks a model means it);
        a guest's list keeps its fallbacks.

        Raises
        ------
        PermissionError
            If ``requested`` is not one of the user's models.
        """
        if requested is None:
            if self.choosable:
                return (self.default,) if self.default else ()
            return self.names
        if requested not in self.names:
            raise PermissionError(f"no model {requested!r} for this user")
        if self.choosable:
            return (requested,)
        return (requested, *(n for n in self.names if n != requested))


def choice(chat: ChatSection, asker: Asker, guest_models: list[str] | None = None) -> ModelChoice:
    """Return the models ``asker`` may use (see the module docstring)."""
    models = chat.models
    if not models:
        return ModelChoice((), None, False, None, configured=False)
    if asker == "owner":
        return ModelChoice(tuple(models), chat.default, True, None)
    if asker == "other":
        default = chat.default
        return ModelChoice((default,) if default else (), default, False, None)
    names = tuple(n for n in guest_models or [] if n in models)
    if not names:
        note = "No assistant model is available for your account at the moment."
    elif len(names) == 1:
        note = "This is the model available for your account."
    else:
        note = "These are the models available for your account, used in this order."
    return ModelChoice(names, names[0] if names else None, False, note)


def label(chat: ChatSection, name: str) -> str:
    """Return the name the web UI shows for model ``name``."""
    model = chat.models[name]
    return model.label or name


#: Providers whose server is local unless its address says otherwise (their defaults).
_LOCAL_SERVERS = {"ollama": "http://localhost:11434",
                  "openai_compatible": "http://localhost:11434/v1", "stub": "local"}  # fmt: skip
_LOOPBACK = {"localhost", "127.0.0.1", "::1"}


def on_machine(chat: ChatSection, name: str) -> bool:
    """Whether model ``name`` runs on this machine, so a chat's text stays on it.

    A local server (Ollama's, or an OpenAI-compatible one at a loopback address) that does
    not forward to a cloud model (Ollama's ``-cloud`` models); every other provider is
    hosted. Unsure is hosted, so that the web UI warns rather than reassures (release
    tester C2, 2026-10-24).
    """
    model = chat.models[name]
    if model.provider not in _LOCAL_SERVERS:
        return False
    if model.provider == "stub":
        return True
    address = model.base_url or _LOCAL_SERVERS[model.provider]
    host = urlparse(address if "//" in address else f"//{address}").hostname or ""
    if host not in _LOOPBACK and not host.startswith("127."):
        return False
    return not (model.model or "").endswith(("-cloud", ":cloud"))


def last_model(turns: Iterable[Any], names: Iterable[str]) -> str | None:
    """Return the model of a chat's last answer, if it is one of ``names`` (else ``None``).

    A reopened chat keeps answering with its model (release tester C2, 2026-10-24): one
    held with a local model went on with the default, a hosted one, unnoticed.
    """
    offered = set(names)
    for turn in reversed(list(turns)):
        data: Mapping[str, Any] = getattr(turn, "data", None) or {}
        model = data.get("chat_model")
        if isinstance(model, str):
            return model if model in offered else None
    return None


def section(chat: ChatSection, name: str) -> LLMSection:
    """Return model ``name``'s settings, as ``[agents.llm]`` takes them (no ``label``)."""
    return LLMSection.model_validate(chat.models[name].model_dump(exclude={"label"}))

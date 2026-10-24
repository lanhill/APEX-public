"""Advice to runtime agents from installed packages.

An agent asks for advice on its system prompt (:func:`advise_prompt`) and on its
``[agents.<name>]`` thresholds (:func:`advise_thresholds`). Each installed advisor (an
entry point of the group ``apex.agent_advisors`` naming a class or a factory of an
:class:`Advisor`) may change them and returns the ids of the advice it applied; the agent
records them with its decision as ``"<advisor>:<id>"`` (the entry point's name first), so
one advisor's ids never read as another's. With no advisor installed, both return their
input unchanged and no ids: an agent never depends on one.

An advisor may also advise an agent's skills (:func:`advise_skill`, its optional ``skill``
method) and hear the events the agent's code reports (:func:`report_event`, its optional
``observe`` method). With no advisor installed, an agent gets no advice and its events go
nowhere.

An advisor that fails to load is skipped until the process restarts (they are loaded
once), and one that fails on a call is skipped for that call: neither stops an agent.
"""

import functools
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

#: The entry-point group of advisors.
ADVISORS_GROUP = "apex.agent_advisors"

logger = logging.getLogger(__name__)


@runtime_checkable
class Advisor(Protocol):
    """What advises the agents: on their prompts and on their thresholds."""

    def prompt(self, agent: str, system: str, root: Path) -> tuple[str, list[int]]:
        """Return ``system`` with the advice for ``agent``, and the advice's ids."""
        ...

    def thresholds[S: BaseModel](
        self, section_name: str, section: S, root: Path
    ) -> tuple[S, list[int]]:
        """Return the ``[section_name]`` section with the advice on it, and the ids."""
        ...


@functools.cache
def advisors() -> tuple[tuple[str, Advisor], ...]:
    """Return the installed advisors with their names, in their entry points' order."""
    found: list[tuple[str, Advisor]] = []
    for entry_point in entry_points(group=ADVISORS_GROUP):
        try:
            factory: Callable[[], object] = entry_point.load()
            advisor = factory()
        except Exception as error:  # noqa: BLE001 (a package's advisor must not stop agents)
            logger.error("the advisor %s cannot be loaded: %s", entry_point.value, error,
                         extra={"event": "advisor_unloadable"})  # fmt: skip
            continue
        if not isinstance(advisor, Advisor):
            logger.error("%s is no advisor (prompt and thresholds)", entry_point.value,
                         extra={"event": "advisor_unloadable"})  # fmt: skip
            continue
        found.append((entry_point.name, advisor))
    return tuple(found)


def advise_prompt(system: str, agent: str, root: Path) -> tuple[str, list[str]]:
    """Return ``agent``'s system prompt with every advisor's advice, and the advice's ids."""
    ids: list[str] = []
    for name, advisor in advisors():
        try:
            system, applied = advisor.prompt(agent, system, root)
        except Exception:  # (advice must never stop an agent)
            logger.exception("the advisor %s failed on %s's prompt", name, agent,
                             extra={"event": "advisor_failed", "advisor": name})  # fmt: skip
            continue
        ids += [f"{name}:{i}" for i in applied]
    return system, sorted(set(ids))


def advise_thresholds[S: BaseModel](
    section_name: str, section: S, root: Path
) -> tuple[S, list[str]]:
    """Return a ``[section_name]`` section with every advisor's advice on it, and the ids."""
    ids: list[str] = []
    for name, advisor in advisors():
        try:
            section, applied = advisor.thresholds(section_name, section, root)
        except Exception:  # (advice must never stop an agent)
            logger.exception("the advisor %s failed on [%s]", name, section_name,
                             extra={"event": "advisor_failed", "advisor": name})  # fmt: skip
            continue
        ids += [f"{name}:{i}" for i in applied]
    return section, sorted(set(ids))


@dataclass(frozen=True)
class AgentEvent:
    """Something an agent's code saw in its work, reported to the installed advisors.

    ``key`` names its kind; ``summary`` says it in words; ``skill`` is the agent's skill it
    concerns (empty: the agent); ``occurrence`` the case it happened in; ``evidence`` data.
    """

    key: str
    summary: str
    skill: str = ""
    occurrence: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)


def advise_skill(agent: str, skill: str, root: Path) -> tuple[list[str], list[str]]:
    """Return every advisor's advice for one of ``agent``'s skills, and the advice's ids."""
    texts: list[str] = []
    ids: list[str] = []
    for name, advisor in advisors():
        method = getattr(advisor, "skill", None)
        if method is None:
            continue
        try:
            given, applied = method(agent, skill, root)
        except Exception:  # (advice must never stop an agent)
            logger.exception("the advisor %s failed on %s's skill %s", name, agent, skill,
                             extra={"event": "advisor_failed", "advisor": name})  # fmt: skip
            continue
        texts += list(given)
        ids += [f"{name}:{i}" for i in applied]
    return texts, sorted(set(ids))


def report_event(agent: str, event: AgentEvent, root: Path) -> None:
    """Give an event of ``agent``'s work to every advisor that takes events."""
    for name, advisor in advisors():
        method = getattr(advisor, "observe", None)
        if method is None:
            continue
        try:
            method(agent, event, root)
        except Exception:  # (an advisor must never stop an agent)
            logger.exception("the advisor %s failed on %s's event %s", name, agent, event.key,
                             extra={"event": "advisor_failed", "advisor": name})  # fmt: skip

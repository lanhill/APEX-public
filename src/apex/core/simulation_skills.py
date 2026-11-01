"""Simulation skills: what an agent that runs simulations knows of each simulator and technique.

A skill is a plugin (``core.plugins.simulation_skills``, the entry-point group
``apex.simulation_skills``): a package that adds a simulator adds its skill, and an agent that
uses skills needs no change. Two kinds:

- ``simulator``: names a simulator plugin (``simulator``), reads that simulator's input into
  :class:`InputFacts` and gives the command line that runs it;
- ``technique``: a way of working with any simulator (gridding, tuning, checks), with guidance
  and, optionally, the failures it recognises in a run's log.

Every skill has guidance (what a model is told) and may declare :class:`FailurePattern` s, by
which :meth:`SimulationSkill.diagnose` turns a log into :class:`Finding` s. A simulator that
needs licences names the features a run checks out (:meth:`SimulationSkill.licence_features`;
none by default). What a skill cannot
do, :class:`SimulationSkill`'s methods refuse with the reason, so an unsupported request fails
loudly, never silently.
"""

import re
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from apex.core.errors import ConfigError

Severity = Literal["info", "warning", "error"]
SkillKind = Literal["simulator", "technique"]
#: What a skill does beyond its guidance: read input facts, diagnose logs, give a command.
Capability = Literal["facts", "diagnose", "command"]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FailurePattern(_Frozen):
    """A failure a skill recognises in a log: a regular expression, what it means, the advice.

    ``pattern`` is matched case-insensitively, line by line (``re.MULTILINE``).
    """

    pattern: str
    meaning: str
    advice: str
    severity: Severity = "error"

    def matches(self, text: str) -> bool:
        """Whether ``text`` holds the pattern."""
        return re.search(self.pattern, text, re.IGNORECASE | re.MULTILINE) is not None


class Finding(_Frozen):
    """What a skill found: its severity, the finding in words and what to do about it."""

    skill: str
    severity: Severity
    message: str
    advice: str = ""


class InputFacts(_Frozen):
    """What a simulator skill read from a simulator's input: facts, nothing guessed.

    Files are named by name only, never by the server's full path (the facts may go to a
    model).
    """

    format: str
    skill: str
    source: str = Field(description="The input's file name.")
    grid: list[int] | None = Field(default=None, description="The grid's dimensions.")
    cells: int | None = None
    active_cells: int | None = None
    phases: list[str] = Field(default_factory=list)
    wells: int | None = None
    includes: list[str] = Field(default_factory=list, description="Included files, by name.")
    unread: list[str] = Field(
        default_factory=list, description="What code could not read, and why."
    )


class SimulationSkill:
    """A simulation skill's interface; subclasses override what ``capabilities`` lists."""

    name: ClassVar[str]
    kind: ClassVar[SkillKind]
    #: The simulator plugin a ``simulator`` skill runs (``None`` for a technique).
    simulator: ClassVar[str | None] = None
    #: The input formats it reads (``eclipse_deck``, ``grdecl``, ...).
    formats: ClassVar[tuple[str, ...]] = ()
    #: What a model is told about the simulator or technique.
    guidance: ClassVar[str] = ""
    #: The failures it recognises in a log (:meth:`diagnose`).
    failure_patterns: ClassVar[tuple[FailurePattern, ...]] = ()
    capabilities: ClassVar[frozenset[Capability]] = frozenset({"diagnose"})

    def accepts(self, source: Path) -> bool:
        """Whether ``source`` looks like this skill's input (by name or content)."""
        return False

    def facts(self, source: Path) -> InputFacts:
        """Return the facts of the input at ``source``."""
        raise self._refuse("facts")

    def diagnose(self, log_text: str) -> list[Finding]:
        """Return a finding for each of ``failure_patterns`` that ``log_text`` holds."""
        return [
            Finding(skill=self.name, severity=p.severity, message=p.meaning, advice=p.advice)
            for p in self.failure_patterns
            if p.matches(log_text)
        ]

    def command(self, source: Path, *, threads: int = 1, ranks: int = 1) -> list[str]:
        """Return the command line that runs ``source`` with ``threads`` per rank."""
        raise self._refuse("command")

    def licence_features(self, source: Path, *, ranks: int = 1) -> dict[str, int]:
        """Return the licence features one run of ``source`` checks out, with their counts.

        None by default: a simulator without licences (open-source ones) needs no check.
        """
        return {}

    def _refuse(self, capability: str) -> ConfigError:
        return ConfigError(f"the simulation skill {self.name!r} cannot give {capability}")

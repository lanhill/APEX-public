"""Knowledge of how a simulator computes: facts with their evidence, for agents that plan runs.

A simulator's skill (:mod:`apex.core.simulation_skills`) may keep a TOML file of facts: each
measured or read for one version of the simulator, with whether it is confirmed for the
installed version, its evidence, what it means for APEX and the deck keywords it concerns.
The facts inform a plan, never a run: :func:`knowledge_findings` gives the facts a deck's
keywords concern as ``info`` findings of the skill.

The model is shared so that every simulator's knowledge has the same shape and the same
findings; a simulator may extend :class:`KnowledgeFact` with fields of its own (e.g. how
other versions differ).
"""

import tomllib
from collections.abc import Callable, Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from apex.core.simulation_skills import Finding


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class KnowledgeFact(_Frozen):
    """One fact of how a simulator computes, with its evidence."""

    topic: str = Field(min_length=1)
    title: str = Field(min_length=1)
    keywords: list[str] = Field(
        default_factory=list, description="Deck keywords it concerns (none: every deck)."
    )
    read_from: str = Field(min_length=1, description="The simulator version it was read from.")
    confirmed_installed: bool = Field(description="Confirmed for the installed version.")
    statement: str = Field(min_length=1)
    source: str = Field(min_length=1, description="Where it is in the source, or the evidence.")
    verified: str = Field(min_length=1, description="How it was checked.")
    consequence: str = Field(min_length=1, description="What it means for APEX.")

    @field_validator("keywords")
    @classmethod
    def _upper(cls, keywords: list[str]) -> list[str]:
        """Refuse a keyword that is not upper case (decks' keywords are matched upper)."""
        lower = [k for k in keywords if k != k.upper()]
        if lower:
            raise ValueError(f"keywords must be upper case: {lower}")
        return keywords


class SimulatorKnowledge[FactT: KnowledgeFact](_Frozen):
    """A knowledge file: the installed version, the sources read, and the facts."""

    installed: str = Field(min_length=1)
    sources: list[str] = Field(min_length=1)
    facts: list[FactT] = Field(min_length=1)

    def applicable(self, keywords: Iterable[str] | None) -> list[FactT]:
        """Return the facts a deck with ``keywords`` concerns (keyword-free facts always).

        ``None``: the deck's keywords are unknown, and every fact is given.
        """
        if keywords is None:
            return list(self.facts)
        present = {k.upper() for k in keywords}
        return [f for f in self.facts if not f.keywords or present & set(f.keywords)]


def read_knowledge[ModelT: BaseModel](path: Path, model: type[ModelT]) -> ModelT:
    """Return the knowledge file at ``path`` as ``model``, validated.

    Raises
    ------
    ValueError
        If the file is not valid TOML or does not match ``model``.
    OSError
        If it cannot be read.
    """
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"{path.name} is not valid TOML: {error}") from error
    return model.model_validate(data)


def knowledge_findings[FactT: KnowledgeFact](
    load: Callable[[], SimulatorKnowledge[FactT]],
    keywords: Iterable[str] | None,
    skill: str,
    *,
    label: str,
    file_name: str,
    unconfirmed: str = ", not confirmed for the installed version",
    evidence: str = "source",
) -> list[Finding]:
    """Return the facts ``keywords`` concern as ``info`` findings of ``skill``.

    ``label`` names the knowledge in the messages (e.g. ``"OPM"``), ``unconfirmed`` is added
    to the version of a fact not confirmed for the installed version, and ``evidence`` names
    a fact's ``source`` in the advice. An unreadable knowledge file is itself an ``info``
    finding, never an error or a flag: the knowledge informs a plan and must neither stop
    nor flag one. Its message names the file by ``file_name`` only (these findings reach a
    model).
    """
    try:
        knowledge = load()
    except OSError as error:
        why = error.strerror or type(error).__name__
        message = f"the {label} knowledge is unreadable: {file_name}: {why}"
        return [Finding(skill=skill, severity="info", message=message)]
    except ValueError as error:
        return [Finding(skill=skill, severity="info",
                        message=f"the {label} knowledge is unreadable: {error}")]  # fmt: skip
    found = []
    for fact in knowledge.applicable(keywords):
        version = fact.read_from + ("" if fact.confirmed_installed else unconfirmed)
        found.append(
            Finding(
                skill=skill,
                severity="info",
                message=f"{label} knowledge ({version}): {fact.title}: {fact.statement}",
                advice=f"{fact.consequence} ({evidence}: {fact.source})",
            )
        )
    return found

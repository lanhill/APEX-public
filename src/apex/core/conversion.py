"""Converter skills: what the case-converter agent knows of each format.

A skill reads a case of its format into :class:`CaseFacts` (what code finds, nothing
guessed), and writes an APEX case from the facts and the agent's :class:`ConversionChoices`
as a :class:`ConversionDraft`. Skills are plugins (``core.plugins.converter_skills``, the
entry-point group ``apex.converter_skills``): a package that adds a simulator adds its skill,
and the agent needs no change.

A skill does what its ``capabilities`` name; :class:`ConverterSkill`'s other methods refuse
with the reason, so an unsupported request fails loudly, never silently.

A skill may take options: it declares them in ``options``
(:class:`SkillOption`: a description, a kind and, for text, its choices), each an attribute
of the skill with its default; :meth:`ConverterSkill.configure` sets them on one instance,
refusing a name it does not declare (with those it does) or a value of the wrong kind. The
command line gives them as ``--option NAME=VALUE`` (:func:`parse_option_pairs`).
"""

import datetime as dt
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from apex.core.errors import ConfigError
from apex.domain.deck_spec import DeckSpec

Capability = Literal["read", "write_config", "create", "convert_to", "convert_from"]


class _Fact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PropertyFact(_Fact):
    """A per-cell property of a deck (``PERMX``, ``PORO``, ...) as code read it.

    ``source`` says how the deck sets it: ``array`` (every cell's value, in the deck or a
    file it includes), ``derived`` (``COPY``, ``MULTIPLY``, ... from another property) or
    ``uniform`` (``EQUALS``). Only an ``array`` property can carry the state (``choosable``);
    ``reason`` says why another cannot. ``minimum``, ``maximum`` and ``mean`` are over the
    cells that are not 0.
    """

    name: str
    source: Literal["array", "derived", "uniform"]
    choosable: bool
    zero_cells: int = Field(
        default=0, description="Cells set to 0 (inactive, barriers): they stay 0, not matched."
    )
    reason: str | None = None
    file: str | None = Field(default=None, description="The file it is set in, by name.")
    derived_from: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None


class WellFact(_Fact):
    """A well: its kind (from its controls), its completions (1-based cells), its phase."""

    name: str
    kind: Literal["producer", "injector", "unknown"]
    cells: list[list[int]] = Field(default_factory=list)
    phase: str | None = None


class ObservationCandidate(_Fact):
    """A vector that could be observed (``WOPR:PROD1``), where its data are and their range."""

    vector: str
    source: Literal["csv", "summary", "deck_history"]
    times: int
    first: dt.date | None = None
    last: dt.date | None = None
    largest: float = 0.0


class CaseFacts(_Fact):
    """What a skill read from a case: the facts every choice is held to.

    Paths are given by name, never as the server's full path (the facts go to a model).
    """

    format: str
    skill: str
    source: str = Field(description="The case's file name.")
    grid: list[int] | None = None
    cells: int | None = None
    units: str | None = None
    phases: list[str] = Field(default_factory=list)
    start: dt.date | None = None
    report_dates: list[dt.date] = Field(default_factory=list)
    properties: dict[str, PropertyFact] = Field(default_factory=dict)
    wells: dict[str, WellFact] = Field(default_factory=dict)
    candidates: list[ObservationCandidate] = Field(default_factory=list)
    includes: list[str] = Field(default_factory=list, description="Included files, by name.")
    unread: list[str] = Field(
        default_factory=list, description="What code could not read, and why."
    )


class VariableChoice(BaseModel):
    """A property that becomes a state variable, and how it is represented."""

    model_config = ConfigDict(extra="forbid")

    property: str = Field(description="A choosable property of the facts, e.g. PERMX.")
    transform: Literal["none", "log"] = Field(
        default="none", description="'log': the state is ln(value) (permeability)."
    )
    lower: float | None = Field(default=None, description="The state's lower bound.")
    upper: float | None = Field(default=None, description="The state's upper bound.")


class ObservationChoice(BaseModel):
    """An observed vector and its data error: max(relative_error * |value|, floor)."""

    model_config = ConfigDict(extra="forbid")

    vector: str = Field(description="A candidate's vector from the facts, e.g. WOPR:PROD1.")
    relative_error: float = Field(gt=0, le=1)
    floor: float = Field(gt=0, description="The smallest standard deviation, in its units.")


class ConversionChoices(BaseModel):
    """What the agent chooses for a converted case."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    variables: list[VariableChoice] = Field(min_length=1)
    observations: list[ObservationChoice] = Field(min_length=1)
    method: Literal["gies", "esmda"] = "gies"
    iterations: int = Field(default=10, ge=1, le=50)
    ensemble_size: int = Field(default=100, ge=10, le=1000)
    localization: Literal["adaptive_prior", "none"] = "adaptive_prior"
    reason: str = Field(default="", max_length=800)


@dataclass
class ConversionDraft:
    """A converted case: its configuration, its files (by relative path) and notes.

    ``notes`` record every decision code made and every value assumed, for the user.
    """

    config: dict[str, object]
    files: dict[str, str | bytes] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class ConversionUsageError(ConfigError):
    """A conversion asked of the wrong input, or of a skill that cannot run here.

    Not a refusal of the case (the input is not what the skill takes, or an extra is
    missing), so it is not reported as an event of the skill's.
    """


@dataclass
class ConversionResult:
    """A translated case: its main file, and notes on what was left out or assumed."""

    path: Path
    notes: list[str] = field(default_factory=list)


#: An option's value, parsed (:meth:`SkillOption.parse`).
OptionValue = bool | int | float | str
_TRUE = ("true", "yes", "on", "1")
#: A date option's text: ISO's extended form only (``20300101`` reads as a number).
_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_FALSE = ("false", "no", "off", "0")


@dataclass(frozen=True)
class SkillOption:
    """One option a converter skill takes (its attribute of the same name holds the default).

    Attributes
    ----------
    description
        What it does, for listings and refusals.
    kind
        ``bool``, ``int``, ``float``, ``str`` or ``datetime.date`` (an ISO date,
        ``YYYY-MM-DD``, held as that text: a date is never free text either).
    choices
        A ``str`` option's allowed values (required: no option is free text, such as a
        path, which a web-UI user's folders would have to confine).
    """

    description: str
    kind: type[bool] | type[int] | type[float] | type[str] | type[dt.date] = bool
    choices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Refuse a text option without choices."""
        if self.kind is str and not self.choices:
            raise ValueError("a str option needs its choices")

    def parse(self, name: str, value: object) -> OptionValue:
        """Return ``value`` (text from a command line, or typed) as the option's kind.

        Raises
        ------
        ConfigError
            If it is not of the kind, or not one of the choices.
        """
        text = value.strip() if isinstance(value, str) else None
        found: OptionValue | None = None
        if self.kind is bool:
            if isinstance(value, bool):
                found = value
            elif text is not None and text.lower() in (*_TRUE, *_FALSE):
                found = text.lower() in _TRUE
        elif self.kind is int:
            if isinstance(value, int) and not isinstance(value, bool):
                found = value
            elif text is not None:
                try:
                    found = int(text)
                except ValueError:
                    found = None
        elif self.kind is float:
            if isinstance(value, int | float) and not isinstance(value, bool):
                found = float(value)
            elif text is not None:
                try:
                    found = float(text)
                except ValueError:
                    found = None
        elif self.kind is dt.date:
            if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
                found = value.isoformat()
            elif text is not None and _ISO_DATE.fullmatch(text):
                try:
                    found = dt.date.fromisoformat(text).isoformat()
                except ValueError:
                    found = None
        elif text is not None and text in self.choices:
            found = text
        if found is None:
            expected = (
                f"one of {', '.join(self.choices)}" if self.kind is str
                else "true or false" if self.kind is bool
                else "an ISO date (YYYY-MM-DD)" if self.kind is dt.date
                else f"a {self.kind.__name__}"
            )  # fmt: skip
            raise ConfigError(f"option {name}={value!r}: give {expected}")
        return found

    def describe(self) -> str:
        """Return the option's kind (or choices) and description, for listings."""
        kind = "|".join(self.choices) if self.kind is str else self.kind.__name__
        return f"{kind}: {self.description}"


def parse_option_pairs(pairs: Sequence[str]) -> dict[str, str]:
    """Return ``NAME=VALUE`` pairs (the command line's ``--option``) as a mapping.

    Raises
    ------
    ConfigError
        If a pair has no ``=`` or no name, or a name is given twice.
    """
    found: dict[str, str] = {}
    for pair in pairs:
        name, sep, value = pair.partition("=")
        name = name.strip()
        if not sep or not name:
            raise ConfigError(f"--option {pair!r}: give NAME=VALUE")
        if name in found:
            raise ConfigError(f"--option {name} is given twice")
        found[name] = value
    return found


class ConverterSkill:
    """A converter skill's interface; subclasses override what ``capabilities`` lists."""

    name: ClassVar[str]
    #: The formats it reads or writes (``eclipse_deck``, ``apex_config``, ``pet_config``).
    formats: ClassVar[tuple[str, ...]]
    #: The simulator plugin the converted cases run on, if any.
    simulator: ClassVar[str | None] = None
    capabilities: ClassVar[frozenset[Capability]] = frozenset()
    #: What the agent's model is told about this format.
    guidance: ClassVar[str] = ""
    #: What it refuses, and why (shown to the model and the user).
    unsupported: ClassVar[tuple[str, ...]] = ()
    #: The vector types (the part of a candidate's vector before ``:``) the case-converter's
    #: rules observe when no model chooses: by default well rates and bottom-hole pressures,
    #: a deck's usual data; a skill whose cases observe other data (temperatures at points)
    #: names its own.
    observed: ClassVar[tuple[str, ...]] = ("WOPR", "WWPR", "WGPR", "WWIR", "WGIR", "WBHP")
    #: Which files a case may include (a web-UI user's folders), set by the agent; ``None``:
    #: any file. A skill that follows includes must check each one with it.
    allowed: Callable[[Path], bool] | None = None
    #: The options it takes (:class:`SkillOption`), each an attribute of the same name
    #: holding its default; :meth:`configure` sets them.
    options: ClassVar[Mapping[str, SkillOption]] = MappingProxyType({})

    def configure(self, values: Mapping[str, object]) -> None:
        """Set options on this instance (text from a command line, or typed values).

        Raises
        ------
        ConfigError
            If a name is not one of ``options`` (naming those it takes), or a value is not
            of its kind.
        """
        unknown = sorted(set(values) - set(self.options))
        if unknown:
            allowed = ", ".join(sorted(self.options)) or "none"
            raise ConfigError(
                f"the converter skill {getattr(self, 'name', type(self).__name__)!r} takes "
                "no option "
                f"{', '.join(unknown)} (its options: {allowed})"
            )
        for name, value in values.items():
            setattr(self, name, self.options[name].parse(name, value))

    def accepts(self, source: Path) -> bool:
        """Whether ``source`` looks like this skill's input (by name or content)."""
        return False

    def read(self, source: Path, history: Path | None = None) -> CaseFacts:
        """Return the facts of the case at ``source`` (``history``: its observed data)."""
        raise self._refuse("read")

    def write_config(
        self,
        source: Path,
        facts: CaseFacts,
        choices: ConversionChoices,
        history: Path | None = None,
    ) -> ConversionDraft:
        """Return the APEX case ``choices`` make of the case at ``source``."""
        raise self._refuse("write_config")

    def create(self, spec: DeckSpec) -> ConversionDraft:
        """Return a synthetic case's deck and the configuration's simulator, state and case.

        The prior, the method, the truth and the observations are the agent's to add.
        """
        raise self._refuse("create")

    def convert_to(self, config: Path, output: Path) -> ConversionResult:
        """Translate ``config`` into this skill's format (the first of ``formats``), in ``output``.

        The source is the other format of ``formats``: an APEX configuration for most
        skills, a deck for a translation between simulators (``geos``).
        """
        raise self._refuse("convert_to")

    def convert_from(self, source: Path, output: Path) -> ConversionResult:
        """Translate ``source``, in this skill's format, into the other of ``formats``."""
        raise self._refuse("convert_from")

    def _refuse(self, capability: str) -> ConfigError:
        return ConfigError(f"the converter skill {self.name!r} cannot {capability}")

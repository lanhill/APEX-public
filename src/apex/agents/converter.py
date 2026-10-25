"""The case-converter agent: a simulator's case as an APEX case, ready to run.

1. **Facts from code.** A converter skill reads the case (``core.plugins.converter_skills``;
   a deck goes to ``[agents.case_converter] default_skill`` when several skills read it):
   its grid, properties, wells, report dates and observation candidates. A real case brings
   its observations, from a CSV, a summary file or the deck's history; nothing is
   simulated.
2. **The model chooses** :class:`~apex.core.conversion.ConversionChoices`: the state
   variables, the observations and their data errors, the method. They are held to the
   facts: a property that cannot carry the state, a vector with no data, or a log of
   values that are not positive is refused, with the reasons, and the model may revise.
3. **The skill writes** the draft (configuration, deck template, observations, files) into
   the output folder, with ``CONVERSION.md`` listing every note and assumed value.
4. **Code validates** it (:meth:`~apex.services.api.ApexService.validate`). Errors go back to
   the model, up to ``max_revisions`` times; then the rules convert.
5. **The rules** (:func:`rules`): the deck's PERMX as its logarithm and PORO (else its
   first property that can carry the state); every rate and pressure with data, with
   errors of ``relative_error`` and a floor of ``floor_fraction`` of the vector's largest
   value; GIES with ``rlmmac`` and ``adaptive_prior``.

The agent reads the case and writes only its output folder. It never simulates or submits.
"""

import json
import math
import os
import re
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tomli_w
from pydantic import model_validator

from apex.agents.base import AgentContext
from apex.agents.common import decide
from apex.agents.loop import LoopSettings
from apex.core.conversion import (
    CaseFacts,
    ConversionChoices,
    ConversionDraft,
    ConversionResult,
    ConversionUsageError,
    ConverterSkill,
    ObservationChoice,
    VariableChoice,
)
from apex.core.errors import ConfigError
from apex.core.llm import LLMProvider
from apex.core.plugins import converter_skills
from apex.domain.deck_spec import DeckSpec, five_spot
from apex.domain.services import Validation
from apex.services.api import ApexService
from apex.services.settings import CaseConverterSection
from apex.simulation.conversion.skills import method_section

NAME = "case-converter"
#: A created case's truth and description, beside its configuration.
TRUTH_FILE = "truth.npz"
SPEC_FILE = "case_spec.json"
#: Seeds of a created case's truth and of its observations' noise: fixed, recorded in its
#: notes, and apart from the run's own (the members' prior is drawn from the run's seed).
TRUTH_SEED = 20261024
NOISE_SEED = 20261025
CONFIG = "config.toml"
NOTES = "CONVERSION.md"
#: Vector types the rules observe: well rates and bottom-hole pressures.
OBSERVED = ("WOPR", "WWPR", "WGPR", "WWIR", "WGIR", "WBHP")
#: The largest |bound| of a 'log' variable: e^50 is beyond any property's value, so a larger
#: one is a bound of the value given by mistake.
LOG_LIMIT = 50.0
#: Properties the rules make the state, and their transform, in order of preference.
PREFERRED = (("PERMX", "log"), ("PORO", "none"))

SYSTEM = (
    "You are APEX's case-converter. You turn a simulator's case into an APEX history-matching "
    "case. From the facts only, choose: the state variables (properties whose 'choosable' is "
    "true; permeability usually as its logarithm, transform 'log', which needs positive "
    "values), the observations (vectors among the candidates, each with a relative error "
    "and a floor, its smallest standard deviation in the vector's units: zero rates need "
    "a floor), the method (gies by default, esmda when asked), its iterations and "
    "ensemble size, and the localization (adaptive_prior unless the user asks for none). "
    "Follow the user's request where it is given. Name only properties and vectors from the "
    "facts. Give a short reason."
)


CREATE_SYSTEM = (
    "You are APEX's case-converter. Turn the user's request for a synthetic reservoir case "
    "into a deck description following the schema: a box grid, wells (producers and "
    "injectors with their controls and limits, inside the grid, 1-based columns), a "
    "schedule of report steps, a fluid preset (oil_water: oil with water injection; "
    "co2_brine: CO2 storage in brine, on OPM Flow only, CO2 injectors on gas_rate in sm3/day "
    "of CO2, 1 Mt a year being about 1.46e6), and the properties' statistics (the "
    "geometric mean of PERMX in mD with the std of its logarithm, porosity's mean and std, "
    "the variogram range in cells). Keep what the request does not mention as in the "
    "example. Give only the description."
)


def choices_model(facts: CaseFacts) -> type[ConversionChoices]:
    """Return :class:`ConversionChoices` held to ``facts``: what the model's answer must pass."""

    class CheckedChoices(ConversionChoices):
        @model_validator(mode="after")
        def _against_the_facts(self) -> "CheckedChoices":
            problems = choice_problems(self, facts)
            if problems:
                raise ValueError("; ".join(problems))
            return self

    CheckedChoices.__name__ = "ConversionChoices"
    return CheckedChoices


def choice_problems(choices: ConversionChoices, facts: CaseFacts) -> list[str]:
    """Return why ``choices`` do not fit ``facts`` (empty: they do)."""
    problems: list[str] = []
    seen: set[str] = set()
    for variable in choices.variables:
        found = facts.properties.get(variable.property)
        if variable.property in seen:
            problems.append(f"{variable.property} is chosen twice")
        seen.add(variable.property)
        if found is None:
            problems.append(f"{variable.property} is not a property of the case")
        elif not found.choosable:
            problems.append(f"{variable.property} cannot carry the state: {found.reason}")
        elif variable.transform == "log" and (found.minimum is None or found.minimum <= 0):
            problems.append(f"{variable.property} has values <= 0: it cannot be 'log'")
        elif found.minimum is not None and found.maximum is not None:
            # Bounds are of the state: ln(value) for 'log'. They must hold the deck's values,
            # or every member would be clipped away from them.
            low, high = found.minimum, found.maximum
            if variable.transform == "log":
                low, high = math.log(low), math.log(high)
            given = [b for b in (variable.lower, variable.upper) if b is not None]
            if variable.transform == "log" and any(abs(b) > LOG_LIMIT for b in given):
                problems.append(
                    f"{variable.property}: the bounds of a 'log' variable are of ln(value) "
                    f"(|bound| <= {LOG_LIMIT:g}), not of the value"
                )
            if variable.lower is not None and variable.lower > low:
                problems.append(f"{variable.property}: lower bound {variable.lower:g} is above the "
                                f"deck's smallest value {low:.4g} (state space"
                                f"{', ln' if variable.transform == 'log' else ''})")  # fmt: skip
            if variable.upper is not None and variable.upper < high:
                problems.append(f"{variable.property}: upper bound {variable.upper:g} is below the "
                                f"deck's largest value {high:.4g} (state space"
                                f"{', ln' if variable.transform == 'log' else ''})")  # fmt: skip
    known = {c.vector for c in facts.candidates}
    vectors = [o.vector for o in choices.observations]
    problems += [f"{v} has no observed data" for v in vectors if v not in known]
    problems += [
        f"{v} is chosen twice" for v in sorted({v for v in vectors if vectors.count(v) > 1})
    ]
    return problems


def rules(facts: CaseFacts, settings: CaseConverterSection, name: str) -> ConversionChoices:
    """Return the rules' choices (see the module).

    Raises
    ------
    ConfigError
        If no property can carry the state, or no vector has non-zero data.
    """
    variables = [
        VariableChoice(property=p, transform=t)  # type: ignore[arg-type]
        for p, t in PREFERRED
        if (found := facts.properties.get(p)) is not None
        and found.choosable
        and (t != "log" or (found.minimum or 0) > 0)
    ]
    if not variables:
        choosable = [p for p in facts.properties.values() if p.choosable]
        if not choosable:
            raise ConfigError("no property of the case can carry the state (none is an array)")
        variables = [VariableChoice(property=choosable[0].name)]
    observations = [
        ObservationChoice(
            vector=c.vector,
            relative_error=settings.relative_error,
            floor=settings.floor_fraction * c.largest,
        )
        for c in facts.candidates
        if c.vector.split(":", 1)[0] in OBSERVED and c.largest > 0
    ]
    if not observations:
        raise ConfigError("no well rate or pressure of the case has data to match")
    return ConversionChoices(
        name=name, variables=variables, observations=observations, method="gies",
        iterations=settings.iterations, ensemble_size=settings.ensemble_size,
        reason="rules: PERMX (log) and PORO where the case sets them as arrays; every well "
        "rate and pressure with data",
    )  # fmt: skip


def case_name(source: Path) -> str:
    """Return a run name from a case's file name (``BRUGGE.DATA``: ``brugge``)."""
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", source.stem).strip("._-").lower()
    return name[:64] or "case"


@dataclass
class ConversionOutcome:
    """A converted case: where it is, whether it validates, and who chose it."""

    output: Path
    skill: str
    choices: ConversionChoices
    validation: Validation
    notes: list[str]
    decided_by: str
    reason: str | None
    traces: list[Path] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Whether the draft validates."""
        return self.validation.valid


class CaseConverter:
    """The case-converter agent.

    Parameters
    ----------
    service
        The service (to validate drafts and record decisions).
    provider
        The model, or ``None`` for the rules.
    settings
        ``[agents.case_converter]`` (default: the service's).
    loop_settings
        The agent loop's settings.
    """

    def __init__(
        self,
        service: ApexService,
        provider: LLMProvider | None,
        *,
        settings: CaseConverterSection | None = None,
        loop_settings: LoopSettings | None = None,
    ) -> None:
        self.service = service
        self.provider = provider
        self.settings = settings or service.jobs.settings.agents.case_converter
        self.loop_settings = loop_settings
        self._written: list[Path] = []
        #: The executor of a created case's truth run: ``process`` for the machine's pools
        #: and retries (tests set ``serial``).
        self.truth_executor = "process"
        #: What a deck may include, for a web-UI user (their folders); ``None``: any file.
        self.allowed: Callable[[Path], bool] | None = None
        #: The largest case (cells) a web-UI user may convert or create; ``None``: any.
        self.max_cells: int | None = None

    def skill_for(self, source: Path, name: str | None = None) -> ConverterSkill:
        """Return the skill ``name``, or the one that reads ``source``.

        Raises
        ------
        ConfigError
            If no skill reads it, or ``name`` cannot write APEX cases.
        """
        if name is not None:
            skill = converter_skills.create(name)
            if "read" not in skill.capabilities:
                raise ConfigError(f"the converter skill {name!r} cannot read a case")
            if not skill.accepts(source):
                raise ConfigError(f"the converter skill {name!r} does not read {source.name}")
            skill.allowed = self.allowed
            return skill
        accepting = [n for n in converter_skills.names()
                     if converter_skills.create(n).accepts(source)]  # fmt: skip
        readers = [n for n in accepting if "read" in converter_skills.create(n).capabilities]
        if accepting and not readers:
            raise ConfigError(f"{source.name} is translated, not imported: `apex agent convert "
                              f"from {accepting[0]} {source.name} OUT`")  # fmt: skip
        if not readers:
            raise ConfigError(f"no converter skill reads {source.name} (`apex agent convert "
                              "skills` lists them)")  # fmt: skip
        chosen = (
            self.settings.default_skill if self.settings.default_skill in readers else readers[0]
        )
        skill = converter_skills.create(chosen)
        skill.allowed = self.allowed  # every path: a new instance, the caller's predicate
        return skill

    def convert(
        self,
        source: Path,
        output: Path,
        *,
        history: Path | None = None,
        request: str | None = None,
        name: str | None = None,
        skill: str | None = None,
    ) -> ConversionOutcome:
        """Convert the case at ``source`` into ``output`` (see the module).

        Raises
        ------
        ConfigError
            If the case cannot be read, ``output`` holds other files, or neither the model's
            draft nor the rules' can be written.
        """
        given = Path(source)
        source = given.resolve()
        if not source.is_file():
            raise ConfigError(f"no case at {given}")
        # Absolute: the simulator runs in a folder of its own (a reported issue, 2026-10-24).
        output = Path(os.path.abspath(output))  # normalized: shown to the user
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ConfigError(f"{output} is not an empty folder: give a new one")
        self._written = []
        chosen = self.skill_for(source, skill)
        if "write_config" not in chosen.capabilities:
            raise ConfigError(f"the converter skill {chosen.name!r} cannot write APEX cases")
        facts = chosen.read(source, history)
        if self.max_cells is not None and (facts.cells or 0) > self.max_cells:
            raise ConfigError(f"{facts.cells} cells: the web UI converts cases of at most "
                              f"{self.max_cells}; use `apex agent convert import`")  # fmt: skip
        supervised = AgentContext(self.service.root, NAME)
        settings = supervised.thresholds(self.settings)
        fallback = rules(facts, settings, name or case_name(source))
        errors: list[str] = []
        first_errors: list[str] = []
        previous: ConversionChoices | None = None
        traces: list[Path] = []
        advice = supervised.skill_advice(chosen.name)
        for _ in range(settings.max_revisions + 1):
            decision = decide(
                self.service, self.provider, supervised.recorder,
                task=self._task(chosen, facts, request, name, previous, errors, advice),
                response_model=choices_model(facts), rules=fallback, rule="convert_rules",
                thresholds=settings.model_dump(mode="json"), client="agent:case-converter",
                settings=self.loop_settings, tools=(), system=supervised.system(SYSTEM),
            )  # fmt: skip
            traces.append(decision.trace)
            choices = ConversionChoices.model_validate(decision.value.model_dump())
            if name is not None:
                choices = choices.model_copy(update={"name": name})
            draft, validation = self._draft(chosen, source, facts, choices, history, output)
            decided_by, reason = decision.decided_by, decision.reason
            if validation.valid or decided_by != "model":
                break
            previous, errors = choices, validation.errors
            first_errors = first_errors or list(validation.errors)
        else:
            # The model could not repair its draft: the rules convert.
            choices = fallback
            draft, validation = self._draft(chosen, source, facts, choices, history, output)
            decided_by = "fallback"
            reason = f"the model's draft did not validate after {settings.max_revisions} revisions"
        self._report(supervised, chosen, source, facts, first_errors, decided_by, validation)
        notes = list(draft.notes) if draft is not None else []
        if request and decided_by != "model":
            notes.insert(0, f"{_not_applied(request, decided_by, reason)}; the rules' choices "
                         "are used")  # fmt: skip
        if draft is not None:
            _write_notes(output, chosen, choices, notes, validation, decided_by)
        supervised.recorder.record_action(
            "converted", skill=chosen.name, source=source.name, valid=validation.valid,
            errors=validation.errors[:20], decided_by=decided_by,
            decisions=[t.name for t in traces],
        )  # fmt: skip
        return ConversionOutcome(output, chosen.name, choices, validation, notes, decided_by,
                                 reason, traces)  # fmt: skip

    def create(
        self,
        output: Path,
        *,
        request: str | None = None,
        name: str | None = None,
        skill: str | None = None,
        simulate: bool = False,
        spec: DeckSpec | None = None,
    ) -> "CreateOutcome":
        """Create a synthetic case in ``output`` from ``request``.

        The model turns the request into a :class:`~apex.domain.deck_spec.DeckSpec` (held
        to its checks), or the rules give a five-spot. The skill writes the deck; the prior
        is the ``continuous`` geomodeling skill's, from the description's statistics; the
        truth is one draw of it from a seed of its own. With ``simulate`` the truth runs once
        and its noisy observations are written (the caller asked the user first); without,
        :meth:`observe` does it later. A ``spec`` given (e.g. by the assistant, which wrote
        it) is used as it is: no model is asked.

        Raises
        ------
        ConfigError
            If ``output`` holds other files, or the skill cannot create cases.
        """
        output = Path(os.path.abspath(output))
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ConfigError(f"{output} is not an empty folder: give a new one")
        chosen = converter_skills.create(skill or self.settings.default_skill)
        if "create" not in chosen.capabilities:
            raise ConfigError(f"the converter skill {chosen.name!r} cannot create cases")
        supervised = AgentContext(self.service.root, NAME)
        settings = supervised.thresholds(self.settings)
        fallback = five_spot(name or "synthetic")
        trace: Path | None = None
        if spec is not None:
            decided_by, reason = "request", None
        else:
            task = json.dumps({
                "request": request or "(none: a five-spot)",
                "name": name or "synthetic",
                "skill": {"name": chosen.name, "guidance": chosen.guidance,
                          "advice": supervised.skill_advice(chosen.name)},
                "schema": DeckSpec.model_json_schema(),
                "example": fallback.model_dump(mode="json"),
            }, default=str)  # fmt: skip
            decision = decide(
                self.service, self.provider, supervised.recorder, task=task,
                response_model=DeckSpec, rules=fallback, rule="create_rules",
                thresholds=settings.model_dump(mode="json"), client="agent:case-converter",
                settings=self.loop_settings, tools=(), system=supervised.system(CREATE_SYSTEM),
            )  # fmt: skip
            spec = DeckSpec.model_validate(decision.value.model_dump())
            decided_by, reason, trace = decision.decided_by, decision.reason, decision.trace
        if name is not None:  # validated as the description's own name is
            spec = DeckSpec.model_validate({**spec.model_dump(), "name": name})
        cells = spec.grid.nx * spec.grid.ny * spec.grid.nz
        if self.max_cells is not None and cells > self.max_cells:
            raise ConfigError(f"{cells} cells: the web UI creates cases of at most "
                              f"{self.max_cells}; use `apex agent convert create`")  # fmt: skip
        try:
            draft = self._assemble(chosen, spec, settings)
        except ConfigError as error:
            if decided_by != "model":
                raise
            # The model's description passed its schema but not the prior's checks (e.g. a
            # range too long for its grid): the rules' case, said so.
            spec = fallback if name is None else DeckSpec.model_validate(
                {**fallback.model_dump(), "name": name})  # fmt: skip
            draft = self._assemble(chosen, spec, settings)
            decided_by, reason = "fallback", f"the model's description failed: {error}"
        write_draft(output, draft)
        truth = _draw_truth(draft.config)
        np.savez(output / TRUTH_FILE, **truth)
        notes = [*draft.notes, f"the truth is one draw of the prior from seed {TRUTH_SEED}"]
        if request and decided_by not in ("model", "request"):
            # A reported issue (2026-10-24): the request was dropped without a word.
            notes.insert(0, f"{_not_applied(request, decided_by, reason)}; the rules' case "
                         f"is written instead: {_spec_summary(spec)} (its values are in "
                         f"{NOTES})")  # fmt: skip
        missing = f"no observations yet: run `apex agent convert observe {output}` (one truth run)"
        validation = Validation(valid=False, errors=[missing])
        failure: ConfigError | None = None
        if simulate:
            try:
                notes += self._observe(output, spec, truth, settings)
                validation = self.service.validate(output / CONFIG)
            except ConfigError as error:
                failure = error
                notes.append(f"the truth run failed: {error}")
        _write_create_notes(output, spec, notes, validation, decided_by)
        supervised.recorder.record_action(
            "created", skill=chosen.name, name=spec.name, valid=validation.valid,
            errors=validation.errors[:10], decided_by=decided_by,
            failed=None if failure is None else str(failure),
        )  # fmt: skip
        if failure is not None:
            raise ConfigError(f"{failure}; the case is written: once the cause is fixed, run "
                              f"`apex agent convert observe {output}`") from failure  # fmt: skip
        return CreateOutcome(output, spec, validation, notes, decided_by, reason, trace)

    def _assemble(
        self, skill: ConverterSkill, spec: DeckSpec, settings: CaseConverterSection
    ) -> ConversionDraft:
        """Return the skill's draft of ``spec`` with its prior and method.

        Raises
        ------
        ConfigError
            If the prior's checks refuse the description's statistics.
        """
        draft = skill.create(spec)
        props = spec.properties
        fields = [
            {"variable": "permx", "mean": props.log_permx, "std": props.permx_log_std,
             "major_range": props.range_cells},
            {"variable": "poro", "mean": props.poro, "std": props.poro_std,
             "major_range": props.range_cells},
        ]  # fmt: skip
        draft.config["prior"] = {"ensemble_size": settings.ensemble_size,
                                 "variables": _continuous_prior(fields, draft.config)}  # fmt: skip
        draft.config["method"] = method_section("gies", settings.iterations, "adaptive_prior")
        return draft

    def observe(self, output: Path) -> "CreateOutcome":
        """Run a created case's truth once and write its observations; validate it.

        Raises
        ------
        ConfigError
            If ``output`` is not a created case, or the run fails.
        """
        output = Path(os.path.abspath(output))
        try:
            spec = DeckSpec.model_validate_json((output / SPEC_FILE).read_text())
            with np.load(output / TRUTH_FILE) as data:
                truth = {key: np.asarray(data[key]) for key in data.files}
        except (OSError, ValueError) as error:
            raise ConfigError(f"{output} is not a case the converter created: {error}") from None
        supervised = AgentContext(self.service.root, NAME)
        notes = self._observe(output, spec, truth, supervised.thresholds(self.settings))
        validation = self.service.validate(output / CONFIG)
        _write_create_notes(output, spec, notes, validation, "code")
        supervised.recorder.record_action("observed", name=spec.name, valid=validation.valid,
                                          errors=validation.errors[:10])  # fmt: skip
        return CreateOutcome(output, spec, validation, notes, "code", None, None)

    def _observe(
        self,
        output: Path,
        spec: DeckSpec,
        truth: dict[str, Any],
        settings: CaseConverterSection,
    ) -> list[str]:
        """Run the truth (the case's simulator, one run) and write its observations."""
        from apex.services.synthetic import observe_truth
        from apex.simulation.conversion.deck_writer import well_vectors

        config = tomllib.loads((output / CONFIG).read_text())
        simulator = dict(config["simulator"])
        options = dict(simulator.get("options", {}))
        options["deck_template_file"] = str((output / str(options["deck_template_file"])).resolve())
        vectors = well_vectors(spec)
        observed = observe_truth(
            simulator=str(simulator["name"]), options=options, truth=truth, vectors=vectors,
            dates=spec.schedule.report_dates(), relative_error=settings.relative_error,
            floor_fraction=settings.floor_fraction, seed=NOISE_SEED,
            work_dir=output / "truth_run", output=output / "observations.csv",
            executor=self.truth_executor,
        )  # fmt: skip
        notes = [f"observed {len(observed.vectors)} vectors ({observed.rows} values) from one "
                 f"truth run, with noise from seed {NOISE_SEED}"]  # fmt: skip
        if observed.left_out:
            notes.append(
                f"not observed, 0 throughout the truth run: {', '.join(observed.left_out)}"
            )
        return notes

    def translate(
        self, skill: str, direction: Literal["to", "from"], source: Path, output: Path
    ) -> ConversionResult:
        """Translate a case with ``skill``: an APEX case ``to`` its format, or ``from`` it.

        Code translates (no model). What the format refuses is reported to the installed
        advisors (:meth:`AgentContext.report`).

        Raises
        ------
        ConfigError
            If the skill cannot translate that way, ``output`` holds other files, or the
            case uses what the other side cannot run.
        """
        chosen = converter_skills.create(skill)
        capability = "convert_to" if direction == "to" else "convert_from"
        if capability not in chosen.capabilities:
            raise ConfigError(f"the converter skill {skill!r} cannot translate {direction} its "
                              "format")  # fmt: skip
        output = Path(os.path.abspath(output))
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ConfigError(f"{output} is not an empty folder: give a new one")
        supervised = AgentContext(self.service.root, NAME)
        try:
            if direction == "to":
                result = chosen.convert_to(Path(source), output)
            else:
                result = chosen.convert_from(Path(source), output)
        except ConversionUsageError:
            raise  # the input, or a missing extra: no event of the skill's
        except ConfigError as error:
            reason = _no_paths(str(error))
            supervised.report(
                f"refused:{direction}:{_kind(reason)}",
                f"refused translating {direction} {skill}: {reason[:200]}",
                skill=skill, occurrence=str(Path(source).resolve()), evidence={"reason": reason},
            )  # fmt: skip
            supervised.recorder.record_action("translated", skill=skill, direction=direction,
                                              source=Path(source).name, valid=False,
                                              errors=[reason])  # fmt: skip
            raise
        supervised.recorder.record_action("translated", skill=skill, direction=direction,
                                          source=Path(source).name, valid=True,
                                          notes=result.notes[:20])  # fmt: skip
        return result

    def _report(
        self,
        supervised: AgentContext,
        skill: ConverterSkill,
        source: Path,
        facts: CaseFacts,
        first_errors: Sequence[str],
        decided_by: str,
        validation: Validation,
    ) -> None:
        """Report what this conversion saw to the installed advisors.

        - a draft the model had to revise: its first validation error, by kind;
        - what the skill could not read in the case, by keyword.

        One event per kind per conversion. Paths are left out of the texts.
        """
        events: dict[str, tuple[str, dict[str, Any]]] = {}
        if first_errors:
            error = _no_paths(first_errors[0])
            outcome = "a revision fixed it" if validation.valid and decided_by == "model" else (
                "the rules converted instead")  # fmt: skip
            events[f"validation:{_kind(error)}"] = (
                f"draft failed validation ({outcome}): {error[:200]}",
                {"errors": [_no_paths(e) for e in first_errors[:5]]},
            )
        for item in facts.unread:
            keyword = item.split(maxsplit=1)[0].rstrip(":")
            events.setdefault(f"unread:{keyword}", (
                f"unread: {_no_paths(item)[:200]}", {"unread": _no_paths(item)},
            ))  # fmt: skip
        for key, (summary, evidence) in events.items():
            supervised.report(key, summary, skill=skill.name, occurrence=str(source),
                              evidence=evidence)  # fmt: skip

    def _task(
        self,
        skill: ConverterSkill,
        facts: CaseFacts,
        request: str | None,
        name: str | None,
        previous: ConversionChoices | None,
        errors: Sequence[str],
        advice: Sequence[str] = (),
    ) -> str:
        shown = facts.model_dump(mode="json")
        # Completions as counts: a field's well list would fill the context.
        shown["wells"] = {w: {"kind": f.kind, "phase": f.phase, "completions": len(f.cells)}
                          for w, f in facts.wells.items()}  # fmt: skip
        limit = self.settings.max_candidates
        shown["candidates"] = shown["candidates"][:limit]
        task: dict[str, Any] = {
            "request": request or "(none: a history match with APEX's defaults)",
            "name": name or case_name(Path(facts.source)),
            "skill": {"name": skill.name, "simulator": skill.simulator,
                      "guidance": skill.guidance, "unsupported": list(skill.unsupported),
                      # The installed advisors' advice for this skill, if any.
                      "advice": list(advice)},
            "defaults": {"method": "gies", "iterations": self.settings.iterations,
                         "ensemble_size": self.settings.ensemble_size,
                         "relative_error": self.settings.relative_error,
                         "floor_fraction_of_largest": self.settings.floor_fraction},
            "facts": shown,
        }  # fmt: skip
        if len(facts.candidates) > limit:
            task["note"] = f"{len(facts.candidates)} candidates; the first {limit} are shown"
        if previous is not None:
            task["previous_choices"] = previous.model_dump(mode="json")
            task["validation_errors"] = list(errors)[:20]
        return json.dumps(task, default=str)

    def _draft(
        self,
        skill: ConverterSkill,
        source: Path,
        facts: CaseFacts,
        choices: ConversionChoices,
        history: Path | None,
        output: Path,
    ) -> tuple[ConversionDraft | None, Validation]:
        """Write the skill's draft into ``output`` and validate it.

        The previous attempt's files are removed first; nothing else in ``output`` is.
        """
        for path in self._written:
            path.unlink(missing_ok=True)
        self._written = []
        try:
            draft = skill.write_config(source, facts, choices, history)
            write_draft(output, draft, self._written)
        except ConfigError as error:
            return None, Validation(valid=False, errors=[str(error)])
        return draft, self.service.validate(output / CONFIG)


def _kind(error: str) -> str:
    """Return a validation error's kind: its words and names, with numbers left out."""
    words = re.sub(r"[0-9][0-9.eE+-]*", "_", _no_paths(error).lower())
    return re.sub(r"\s+", " ", words).strip()[:160]


def _no_paths(text: str) -> str:
    """Return ``text`` with file paths replaced by their names (notes go to prompts)."""
    return re.sub(r"(?:[A-Za-z]:)?(?:/[\w.@+-]+)+/([\w.@+-]+)", r"\1", text)


def write_draft(output: Path, draft: ConversionDraft, written: list[Path] | None = None) -> None:
    """Write ``draft``'s configuration and files into ``output``.

    Each file is added to ``written`` as soon as it is written, so a draft that fails part
    way can still be removed.

    Raises
    ------
    ConfigError
        If a file of the draft would lie outside ``output``.
    """
    written = [] if written is None else written
    output.mkdir(parents=True, exist_ok=True)
    root = output.resolve()
    for relative, content in draft.files.items():
        path = output / relative
        if root not in path.resolve().parents:
            raise ConfigError(f"a draft file outside its folder: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        written.append(path)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content)
    header = ("# Written by APEX's case-converter; see CONVERSION.md for its notes "
              "and assumed values.\n")  # fmt: skip
    written.append(output / CONFIG)
    (output / CONFIG).write_text(header + tomli_w.dumps(draft.config))


def _write_notes(
    output: Path,
    skill: ConverterSkill,
    choices: ConversionChoices,
    notes: Sequence[str],
    validation: Validation,
    decided_by: str,
) -> None:
    lines = [
        "# Conversion notes",
        "",
        f"Converted by the `{skill.name}` skill; choices by the {decided_by}.",
        "",
        f"- State: {', '.join(f'{v.property} ({v.transform})' for v in choices.variables)}",
        f"- Observations: {len(choices.observations)} vectors",
        f"- Method: {choices.method}, {choices.iterations} iterations, "
        f"{choices.ensemble_size} members, localization {choices.localization}",
    ]
    if choices.reason:
        lines += ["", f"Reason: {choices.reason}"]
    if notes:
        lines += ["", "## Notes and assumed values", ""] + [f"- {_no_paths(n)}" for n in notes]
    if not validation.valid:
        errors = [f"- {_no_paths(e)}" for e in validation.errors]
        lines += ["", "## Validation errors", "", *errors]
    (output / NOTES).write_text("\n".join(lines) + "\n")


@dataclass
class CreateOutcome:
    """A created synthetic case: where, its description, and whether it validates."""

    output: Path
    spec: DeckSpec
    validation: Validation
    notes: list[str]
    decided_by: str
    reason: str | None
    trace: Path | None

    @property
    def valid(self) -> bool:
        """Whether the case validates (it has its observations)."""
        return self.validation.valid


def _continuous_prior(fields: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Return the ``continuous`` geomodeling skill's generators for ``fields``."""
    from apex.core.geomodeling import GeomodelFacts, VariableFact
    from apex.core.plugins import geomodeling_skills

    variables = {
        str(v["name"]): VariableFact(name=str(v["name"]), size=int(v["size"]), grid=v.get("grid"),
                                     lower=v.get("lower"), upper=v.get("upper"))
        for v in config["state"]["variables"]
    }  # fmt: skip
    skill = geomodeling_skills.create("continuous")
    parameters = skill.parameters.model_validate({"fields": fields})
    facts = GeomodelFacts(variables=variables, kind="synthetic")
    problems = skill.check(parameters, facts)
    if problems:
        raise ConfigError("; ".join(problems))
    return skill.prior(parameters, facts).variables


def _draw_truth(config: dict[str, Any]) -> dict[str, Any]:
    from apex.services.synthetic import draw_truth

    return draw_truth(config, TRUTH_SEED)


def _write_create_notes(
    output: Path,
    spec: DeckSpec,
    notes: Sequence[str],
    validation: Validation,
    decided_by: str,
) -> None:
    shown = [f"- {_no_paths(n)}" for n in notes]
    values = [f"- {line}" for line in _spec_values(spec)]
    lines = ["# Created case", "", f"`{spec.name}`, described by the {decided_by} "
             "(case_spec.json).", "", "## Notes", "", *shown, "",
             "## Values", "", "What the request did not give is assumed.", "", *values]  # fmt: skip
    if not validation.valid:
        lines += ["", "## Not valid yet", "", *(f"- {_no_paths(e)}" for e in validation.errors)]
    (output / NOTES).write_text("\n".join(lines) + "\n")


def _not_applied(request: str, decided_by: str, reason: str | None) -> str:
    """Return a note that ``request`` was not applied, and why (the model was not used)."""
    why = "no model was asked" if decided_by == "rules" else (reason or "the model failed")
    return f"the request was not applied ({_no_paths(why)}): {request!r}"


def _spec_summary(spec: DeckSpec) -> str:
    """Return a created case's grid, wells and schedule in a few words."""
    g, s = spec.grid, spec.schedule
    return (f"{g.nx} x {g.ny} x {g.nz} cells, {len(spec.wells)} wells, {s.steps} report steps "
            f"of {s.step_days} days")  # fmt: skip


def _spec_values(spec: DeckSpec) -> list[str]:
    """Return every value of a created case's description, one line per part."""
    g, s, p = spec.grid, spec.schedule, spec.properties
    fields = type(spec.fluid).model_fields
    fluid = ", ".join(
        f"{key} {value}" + (f" ({fields[key].description})" if fields[key].description else "")
        for key, value in spec.fluid.model_dump().items()
    )
    lines = [
        f"grid: {g.nx} x {g.ny} x {g.nz} cells of {g.dx:g} x {g.dy:g} x {g.dz:g} m, its top "
        f"at {g.top:g} m",
        f"schedule: {s.steps} report steps of {s.step_days} days from {s.start}",
        f"permeability: PERMX's geometric mean {p.permx_md:g} mD, standard deviation of "
        f"ln PERMX {p.permx_log_std:g}, kv/kh {p.kv_kh:g}",
        f"porosity: mean {p.poro:g}, standard deviation {p.poro_std:g}",
        f"correlation range: {p.range_cells:g} cells",
        f"fluid: {fluid}",
    ]
    for well in spec.wells:
        unit = "bar" if well.control == "bhp" else "sm3/day"
        limit = "minimum" if well.kind == "producer" else "maximum"
        lines.append(
            f"well {well.name}: {well.kind} at ({well.i}, {well.j}), layers {well.k_top}-"
            f"{well.k_bottom or g.nz}, {well.control} {well.target:g} {unit}, {limit} "
            f"bottom-hole pressure {well.bhp_limit:g} bar"
        )
    return lines


def visible_skills(*, public_only: bool) -> list[str]:
    """Return the converter skills' names; ``public_only``: those of public packages only.

    A skill of a private package (``intersect``, ``geos``) is private (as its
    simulator): web-UI guests and the public never see it.
    """
    from apex.core.audience import audience

    names = converter_skills.names()
    if not public_only:
        return names
    found = []
    for name in names:
        origin = converter_skills.origin(name)
        if origin is None or audience(origin) == "public":
            found.append(name)
    return found

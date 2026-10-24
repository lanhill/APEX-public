"""The geomodeler agent: a prior for the state variables a case draws, not reads.

1. **Facts from code** (:func:`geomodel_facts`): the configuration's state variables, their
   grids, bounds and current generators, and the statistics of a reference field where
   one is given (``--reference``: an ``.npz`` of arrays in state space, e.g. the deck's
   values). The variables to model are those not read from a stored ensemble.
2. **The model chooses** a :class:`GeomodelPlan`: a geomodeling skill
   (``core.plugins.geomodeling_skills``) and its parameters. They are held to the skill's
   schema and its checks against the facts; refused, the reasons go back to the model.
3. **The skill writes** the prior: ``[prior]`` generators, groups and
   ``[[parameterizations]]``, into a new configuration beside the old one.
4. **Code validates** it and draws a small sample (``qc_members``): the skill's QC says
   whether it is the prior asked for. Problems go back to the model, up to
   ``max_revisions`` times; then the rules decide.
5. **The rules** (:func:`rules`): ``continuous`` Gaussian fields, the mean and std of the
   reference (else the bounds' midpoint and a sixth of their width, else 0 and 1), ranges
   of ``default_range_fraction`` of the grid's larger side, every such value marked
   assumed.

The agent writes only the new configuration and its files; a run never calls it: a prior
is a recorded section, so runs stay reproducible.
"""

import copy
import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tomli_w
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from apex.agents.base import AgentContext
from apex.agents.common import decide
from apex.agents.loop import LoopSettings
from apex.core.errors import ApexError, ConfigError
from apex.core.geomodeling import GeomodelFacts, GeomodelingSkill, PriorDraft, VariableFact
from apex.core.llm import LLMProvider
from apex.core.plugins import geomodeling_skills
from apex.domain.services import Validation
from apex.services.api import ApexService
from apex.services.settings import GeomodelerSection

NAME = "geomodeler"

SYSTEM = (
    "You are APEX's geomodeler. You give a history-matching case's state variables a prior: "
    "choose one geomodeling skill from the facts and its parameters, following its schema "
    "and guidance. Take the user's values as given; use the reference field's mean and std "
    "where there is one; otherwise propose values and list each such value in `assumed`. "
    "Name only state variables from the facts. Give a short reason."
)


class GeomodelPlan(BaseModel):
    """A skill and its parameters (the skill's own schema)."""

    model_config = ConfigDict(extra="forbid")

    skill: str = Field(description="A geomodeling skill's name from the facts.")
    parameters: dict[str, Any] = Field(description="The skill's parameters, by its schema.")
    reason: str = Field(default="", max_length=800)


def plan_model(facts: GeomodelFacts) -> type[GeomodelPlan]:
    """Return :class:`GeomodelPlan` held to the skills and ``facts``."""

    class CheckedPlan(GeomodelPlan):
        @model_validator(mode="after")
        def _against_the_facts(self) -> "CheckedPlan":
            problems = plan_problems(self, facts)
            if problems:
                raise ValueError("; ".join(problems))
            return self

    CheckedPlan.__name__ = "GeomodelPlan"
    return CheckedPlan


def plan_problems(plan: GeomodelPlan, facts: GeomodelFacts) -> list[str]:
    """Return why ``plan`` cannot be used on ``facts`` (empty: it can)."""
    if plan.skill not in geomodeling_skills.names():
        return [f"no geomodeling skill {plan.skill!r} (skills: "
                f"{', '.join(geomodeling_skills.names())})"]  # fmt: skip
    skill = geomodeling_skills.create(plan.skill)
    try:
        parameters = skill.parameters.model_validate(plan.parameters)
    except ValidationError as error:
        return [f"{plan.skill} parameters: {e['loc']}: {e['msg']}" for e in error.errors()]
    return skill.check(parameters, facts)


def geomodel_facts(raw: dict[str, Any], base: Path, reference: Path | None) -> GeomodelFacts:
    """Return the facts of a configuration (``raw``, read from ``base``).

    Raises
    ------
    ConfigError
        If the configuration has no state variables, or the reference cannot be read.
    """
    variables = list(dict(raw.get("state", {})).get("variables", []))
    if not variables:
        raise ConfigError("the configuration has no [[state.variables]]")
    prior = dict(raw.get("prior", {}))
    grouped = {v for g in dict(prior.get("groups", {})).values() for v in g.get("variables", [])}
    default = _named(prior.get("generator", {}))
    arrays: dict[str, np.ndarray] = {}
    if reference is not None:
        try:
            with np.load(reference) as data:
                arrays = {name: np.asarray(data[name], dtype=float) for name in data.files}
        except (OSError, ValueError) as error:
            raise ConfigError(f"cannot read the reference {reference.name}: {error}") from None
    found = {}
    for variable in variables:
        name = str(variable["name"])
        generator = _named(dict(prior.get("variables", {})).get(name, default))
        values = arrays.get(name)
        if values is not None:
            values = values.ravel()
            # Cells that are 0 (inactive, barriers; the converter keeps them so) are not
            # the field's, nor are ln 0's -inf.
            values = values[np.isfinite(values) & (values != 0)]
        found[name] = VariableFact(
            name=name, size=int(variable["size"]), grid=variable.get("grid"),
            lower=variable.get("lower"), upper=variable.get("upper"),
            reference_mean=float(values.mean()) if values is not None and values.size else None,
            reference_std=float(values.std()) if values is not None and values.size else None,
            prior="group" if name in grouped else str(generator.get("name", "gaussian")),
        )  # fmt: skip
    size = prior.get("ensemble_size")
    return GeomodelFacts(variables=found, ensemble_size=None if size is None else int(size),
                         kind=str(dict(raw.get("case", {})).get("kind", "synthetic")),
                         log_variables=_log_variables(raw, base))  # fmt: skip


def _named(section: object) -> dict[str, Any]:
    """Return a generator section as a table (a bare name is ``{"name": name}``)."""
    return {"name": section} if isinstance(section, str) else dict(section or {})  # type: ignore[call-overload]


#: A template's loop over a variable: ``% for value in permx:`` (or ``value, kept in``).
_LOOP = re.compile(r"^\s*%\s*for\s+value\b[^:]*\bin\s+(?:zip\()?\s*(\w+)", re.MULTILINE)


def _log_variables(raw: dict[str, Any], base: Path) -> list[str]:
    """Return the names the deck template writes as ``exp(value)`` (their values are logs).

    A loop over a name whose next line exponentiates the value, as the converter's
    templates and the examples' write ln PERMX.
    """
    options = dict(dict(raw.get("simulator", {})).get("options", {}))
    template = base / str(options.get("deck_template_file", ""))
    if not template.is_file():
        return []
    lines = template.read_text(errors="replace").splitlines()
    found = []
    for number, line in enumerate(lines[:-1]):
        loop = _LOOP.match(line)
        if loop and "exp(" in lines[number + 1]:
            found.append(loop.group(1))
    return sorted(set(found))


def targets(facts: GeomodelFacts) -> list[str]:
    """Return the variables to model: fields not read from a stored ensemble nor grouped."""
    return [n for n, v in facts.variables.items()
            if v.prior not in ("file", "group") and v.grid and len(v.grid) == 3]  # fmt: skip


def rules(facts: GeomodelFacts, settings: GeomodelerSection) -> GeomodelPlan:
    """Return the rules' plan (see the module).

    Raises
    ------
    ConfigError
        If no variable needs a prior, or one has no grid.
    """
    names = targets(facts)
    if not names:
        raise ConfigError(
            "no gridded state variable to model: each is read from a stored ensemble, drawn "
            "in a group (facies: ask the model for facies_pluri), or has no grid"
        )
    fields = []
    for name in names:
        fact = facts.variables[name]
        assert fact.grid is not None
        assumed = ["range", "variogram"]
        if fact.reference_mean is not None and fact.reference_std:
            mean, std = fact.reference_mean, fact.reference_std
        elif fact.lower is not None and fact.upper is not None and fact.upper > fact.lower:
            # The bounds' midpoint, and a sixth of their width: three stds on each side.
            mean, std = (fact.lower + fact.upper) / 2, (fact.upper - fact.lower) / 6
            assumed += ["mean", "std"]
        else:
            mean, std = 0.0, 1.0
            assumed += ["mean", "std"]
        extent = max(fact.grid[0], fact.grid[1])
        fields.append({"variable": name, "mean": mean, "std": std, "variogram": "spherical",
                       "major_range": max(1.0, round(settings.default_range_fraction * extent, 1)),
                       "assumed": assumed})  # fmt: skip
    return GeomodelPlan(skill="continuous", parameters={"fields": fields},
                        reason="rules: Gaussian fields around the reference (or the bounds), "
                        "ranges a fraction of the grid")  # fmt: skip


@dataclass
class GeomodelOutcome:
    """A prior written: where, whether it validates and passes QC, and who chose it."""

    config: Path
    plan: GeomodelPlan
    validation: Validation
    qc: list[str]
    notes: list[str]
    decided_by: str
    reason: str | None
    traces: list[Path] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Whether the new configuration validates and its prior passed QC."""
        return self.validation.valid and not self.qc


class Geomodeler:
    """The geomodeler agent.

    Parameters
    ----------
    service
        The service (to validate drafts and record decisions).
    provider
        The model, or ``None`` for the rules.
    settings
        ``[agents.geomodeler]`` (default: the service's).
    loop_settings
        The agent loop's settings.
    """

    def __init__(
        self,
        service: ApexService,
        provider: LLMProvider | None,
        *,
        settings: GeomodelerSection | None = None,
        loop_settings: LoopSettings | None = None,
    ) -> None:
        self.service = service
        self.provider = provider
        self.settings = settings or service.jobs.settings.agents.geomodeler
        self.loop_settings = loop_settings
        #: The largest prior (cells over every variable) a web-UI user may have modelled.
        self.max_cells: int | None = None

    def model(
        self,
        config: Path,
        output: Path | None = None,
        *,
        request: str | None = None,
        reference: Path | None = None,
    ) -> GeomodelOutcome:
        """Write ``config`` with a new prior to ``output`` (see the module).

        ``output`` defaults to ``<config>_prior.toml`` beside it, so relative paths still
        resolve; it must not exist.

        Raises
        ------
        ConfigError
            If the configuration cannot be read, nothing needs a prior, or ``output`` exists.
        """
        config = Path(config).resolve()
        try:
            raw = tomllib.loads(config.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise ConfigError(f"cannot read {config.name}: {error}") from None
        output = (
            Path(output) if output is not None else config.with_name(f"{config.stem}_prior.toml")
        )
        if output.exists():
            raise ConfigError(f"{output.name} exists: give another output")
        if output.resolve().parent != config.parent:
            raise ConfigError("write the new configuration beside the old one, so that its "
                              "relative paths still resolve")  # fmt: skip
        facts = geomodel_facts(raw, config.parent, reference)
        cells = sum(v.size for v in facts.variables.values())
        if self.max_cells is not None and cells > self.max_cells:
            raise ConfigError(f"{cells} cells: the web UI models priors of at most "
                              f"{self.max_cells}; use `apex agent geomodel`")  # fmt: skip
        supervised = AgentContext(self.service.root, NAME)
        settings = supervised.thresholds(self.settings)
        try:
            fallback: GeomodelPlan | None = rules(facts, settings)
        except ConfigError:
            if self.provider is None:
                raise
            fallback = None  # only the model can plan it (e.g. facies)
        placeholder = GeomodelPlan(skill="", parameters={}, reason="no rules' plan")
        problems: list[str] = []
        previous: GeomodelPlan | None = None
        traces: list[Path] = []
        written: list[Path] = []
        notes: list[str] = []
        try:
            for _ in range(settings.max_revisions + 1):
                decision = decide(
                    self.service, self.provider, supervised.recorder,
                    task=self._task(facts, request, previous, problems, supervised),
                    response_model=plan_model(facts), rules=fallback or placeholder,
                    rule="geomodel_rules", thresholds=settings.model_dump(mode="json"),
                    client="agent:geomodeler", settings=self.loop_settings, tools=(),
                    system=supervised.system(SYSTEM),
                )  # fmt: skip
                traces.append(decision.trace)
                plan = GeomodelPlan.model_validate(decision.value.model_dump())
                if not plan.skill:
                    raise ConfigError(f"the model gave no usable plan ({decision.reason}), and "
                                      "the rules have none for this case")  # fmt: skip
                draft, validation, qc = self._try(plan, raw, facts, output, settings, written)
                decided_by, reason = decision.decided_by, decision.reason
                if (validation.valid and not qc) or decided_by != "model":
                    break
                previous, problems = plan, [*validation.errors, *qc]
            else:
                if fallback is None:
                    raise ConfigError("the model's prior did not pass, and the rules have none "
                                      f"for this case: {'; '.join(problems[:5])}")  # fmt: skip
                plan = fallback
                draft, validation, qc = self._try(plan, raw, facts, output, settings, written)
                decided_by = "fallback"
                reason = f"the model's prior did not pass after {settings.max_revisions} revisions"
        except BaseException:
            for path in written:  # nothing half-written stays behind
                path.unlink(missing_ok=True)
            raise
        if decided_by == "fallback" and request:
            notes.append("the rules' prior replaced the model's: your request was not kept "
                         f"({reason})")  # fmt: skip
        supervised.recorder.record_action(
            "geomodeled", config=config.name, skill=plan.skill, valid=validation.valid,
            qc=qc[:10], errors=validation.errors[:10], decided_by=decided_by,
        )  # fmt: skip
        return GeomodelOutcome(output, plan, validation, qc, [*notes, *draft.notes], decided_by,
                               reason, traces)  # fmt: skip

    def _task(
        self,
        facts: GeomodelFacts,
        request: str | None,
        previous: GeomodelPlan | None,
        problems: list[str],
        supervised: AgentContext,
    ) -> str:
        skills = {}
        for name in geomodeling_skills.names():
            skill = geomodeling_skills.create(name)
            skills[name] = {"scenario": skill.scenario, "guidance": skill.guidance,
                            "parameters": skill.parameters.model_json_schema(),
                            # The installed advisors' advice for this skill, if any.
                            "advice": supervised.skill_advice(name)}  # fmt: skip
        task: dict[str, Any] = {
            "request": request or "(none: a prior for the variables that have none)",
            "variables_to_model": targets(facts),
            "facts": facts.model_dump(mode="json"),
            "skills": skills,
        }
        if previous is not None:
            task["previous_plan"] = previous.model_dump(mode="json")
            task["problems"] = problems[:20]
        return json.dumps(task, default=str)

    def _try(
        self,
        plan: GeomodelPlan,
        raw: dict[str, Any],
        facts: GeomodelFacts,
        output: Path,
        settings: GeomodelerSection,
        written: list[Path],
    ) -> tuple[PriorDraft, Validation, list[str]]:
        """Write the plan's configuration, validate it and check a drawn sample."""
        for path in written:
            path.unlink(missing_ok=True)
        written.clear()
        skill = geomodeling_skills.create(plan.skill)
        parameters = skill.parameters.model_validate(plan.parameters)
        draft = skill.prior(parameters, facts)
        updated = _with_prior(raw, draft, settings.ensemble_size)
        for relative, content in draft.files.items():
            path = output.parent / relative
            if path.exists():
                raise ConfigError(f"{relative} exists beside the configuration: not overwritten")
            written.append(path)
            path.write_bytes(content)
        written.append(output)
        header = (f"# Its prior by APEX's geomodeler (skill {plan.skill}); see the notes in "
                  "its decision record.\n")  # fmt: skip
        output.write_text(header + tomli_w.dumps(updated))
        validation = self.service.validate(output)
        if not validation.valid:
            return draft, validation, []
        return draft, validation, self._qc(skill, parameters, output, settings, facts, draft)

    def _qc(
        self,
        skill: GeomodelingSkill,
        parameters: BaseModel,
        config: Path,
        settings: GeomodelerSection,
        facts: GeomodelFacts,
        draft: PriorDraft,
    ) -> list[str]:
        """Draw ``qc_members`` members of the prior and ask the skill whether it is right.

        Not drawn, with a note, when it would be slow: SGSIM, or more than
        ``qc_max_values`` values.
        """
        from apex.services import composition  # the composition draws priors

        values = sum(v.size for v in facts.variables.values()) * settings.qc_members
        sgsim = any(v.get("name") == "geostatspy_sgsim" for v in draft.variables.values())
        if sgsim or values > settings.qc_max_values:
            draft.notes.append(
                "the prior was not drawn for its check (SGSIM, or more than "
                f"{settings.qc_max_values} values): check it after a run"
            )
            return []
        try:
            loaded = composition.load_config(config)
            prior = loaded.prior.model_copy(update={"ensemble_size": settings.qc_members})
            run = composition.build_run(loaded.model_copy(update={"prior": prior}))
            sample = run.workflow.layout.split(composition.draw_prior(run))
        except (ApexError, ImportError) as error:
            return [f"the prior cannot be drawn: {error}"]
        tolerance = settings.qc_facies_tolerance if skill.name == "facies_pluri" else (
            settings.qc_tolerance)  # fmt: skip
        return skill.qc(parameters, sample, tolerance, facts)


def _with_prior(raw: dict[str, Any], draft: PriorDraft, ensemble_size: int) -> dict[str, Any]:
    """Return ``raw`` with ``draft``'s prior in it.

    A variable the draft models leaves any group it was in (an emptied group goes); a
    parameterization of the draft replaces, in place, the one of the same name over the
    same fields, else it is added.
    """
    updated = copy.deepcopy(raw)
    prior = updated.setdefault("prior", {})
    prior.setdefault("ensemble_size", ensemble_size)
    modeled = set(draft.variables) | {v for g in draft.groups.values() for v in g["variables"]}
    groups = prior.get("groups", {})
    for name in list(groups):
        kept = [v for v in groups[name].get("variables", []) if v not in modeled]
        if kept:
            groups[name]["variables"] = kept
        else:
            del groups[name]
    variables = prior.setdefault("variables", {})
    for name in modeled:
        variables.pop(name, None)
    variables.update(draft.variables)
    if draft.groups:
        prior.setdefault("groups", {}).update(draft.groups)
    if not variables:
        del prior["variables"]
    if not prior.get("groups", True):
        del prior["groups"]
    existing = list(updated.get("parameterizations", []))
    for new in draft.parameterizations:
        fields = set(dict(new.get("options", {})).get("fields", []))
        for index, old in enumerate(existing):
            if old.get("name") == new.get("name") and fields & set(
                    dict(old.get("options", {})).get("fields", [])):  # fmt: skip
                existing[index] = new
                break
        else:
            existing.append(new)
    if existing:
        updated["parameterizations"] = existing
    return updated

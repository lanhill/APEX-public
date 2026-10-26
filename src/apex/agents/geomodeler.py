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
5. **The rules** (:func:`rules`): ``continuous`` Gaussian fields, with the statistics and
   variogram estimated from the reference field or the hard data where given
   (``--reference``, ``--data``, :mod:`apex.services.geomodeling_data`), conditioned to
   the hard data, else the bounds' midpoint and a sixth of their width (else 0 and 1) and
   ranges of ``default_range_fraction`` of the grid's larger side, every such value
   marked assumed.

Hard data (phase 2) and a reference field (revision 1) are estimated from by code, not the
model: the facts carry each property's mean, std and fitted variogram (the reference's
where both are given), and the facies' proportions, length scale and conditioning cells;
the model takes them unless the user gives other values. The continuous skills condition
their fields to the hard data (revision 1; code passes the data on, never the model); a
reference is never conditioned to, only its statistics are shared.

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
from apex.core.geomodeling import (
    GeomodelFacts,
    GeomodelingSkill,
    PriorDraft,
    PropertyEstimate,
    VariableFact,
)
from apex.core.llm import LLMProvider
from apex.core.plugins import geomodeling_skills
from apex.domain.services import Validation
from apex.services.api import ApexService
from apex.services.settings import GeomodelerSection

NAME = "geomodeler"

SYSTEM = (
    "You are APEX's geomodeler. You give a history-matching case's state variables a prior: "
    "choose one geomodeling skill from the facts and its parameters, following its schema "
    "and guidance. Take the user's values as given; use the estimates in the facts (from a "
    "reference field or hard data) where there are some, and list those values in "
    "`estimated`; otherwise propose values and list each such value in `assumed`. "
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
    log_variables = _log_variables(raw, base)
    found = {}
    for variable in variables:
        name = str(variable["name"])
        generator = _named(dict(prior.get("variables", {})).get(name, default))
        values = arrays.get(name)
        if values is not None:
            from apex.services.geomodeling_data import active_cells

            values = values.ravel()
            # Cells that are 0 (inactive, barriers; the converter keeps them so) are not
            # the field's, nor are ln 0's -inf; a log variable's 0 is ln 1, a value.
            values = values[active_cells(values, name in log_variables)]
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
                         log_variables=log_variables, folder=str(base))  # fmt: skip


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
        estimate = facts.estimates.get(name)
        if estimate is not None and estimate.std > 0:
            fields.append(_estimated_field(estimate, fact, fact.grid, settings))
            continue
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
    for chosen in fields:
        _keep_conditioning_valid(chosen, facts)
    reason = (
        "rules: Gaussian fields from the reference's or the hard data's estimates where "
        "given (conditioned to the hard data), else around the bounds with ranges a "
        "fraction of the grid"
    )
    return GeomodelPlan(skill="continuous", parameters={"fields": fields}, reason=reason)


def _keep_conditioning_valid(field: dict[str, Any], facts: GeomodelFacts) -> None:
    """Turn a rules' field's conditioning off when its data cannot be honoured, and say so.

    Data outside the bounds (the prior is clipped) or improbable under the field's
    variogram: the skill would refuse them, and the rules' plan must pass, so the field
    shares the data's statistics only. Every field, whether or not its statistics came from
    the data (one datum gives no std).
    """
    estimate = facts.estimates.get(field["variable"])
    if estimate is None or not estimate.values:
        return
    from apex.services.geomodeling_skills import FieldChoice, conditioning_problems

    problems = conditioning_problems(FieldChoice.model_validate(field), facts)
    if problems:
        field["condition"] = False
        field["assumed"].append(f"no conditioning ({problems[0]})")


def _estimated_field(
    estimate: PropertyEstimate, fact: VariableFact, grid: list[int], settings: GeomodelerSection
) -> dict[str, Any]:
    """Return the rules' field from a hard-data estimate; what it lacks is assumed.

    A mean outside the variable's bounds (data in other units than the deck's, say) is
    moved to the nearer bound and listed as assumed, so that the rules' plan still passes.
    """
    mean = estimate.mean
    if fact.lower is not None and fact.upper is not None:
        mean = min(max(mean, fact.lower), fact.upper)
    field: dict[str, Any] = {"variable": estimate.variable, "mean": mean,
                             "std": estimate.std, "estimated": ["std"], "assumed": []}  # fmt: skip

    if mean == estimate.mean:
        field["estimated"].insert(0, "mean")
    else:
        field["assumed"].append(f"mean (the data's {estimate.mean:.4g} is outside the bounds)")
    if estimate.major_range is not None and estimate.variogram is not None:
        # The skill's own limits: half a cell to four grid lengths.
        top = 4.0 * max(grid[0], grid[1])
        field.update(variogram=estimate.variogram,
                     major_range=min(max(estimate.major_range, 0.5), top),
                     azimuth=estimate.azimuth or 0.0)  # fmt: skip
        if estimate.minor_range is not None and estimate.minor_range < estimate.major_range:
            field["minor_range"] = min(max(estimate.minor_range, 0.5), top)
        field["estimated"] += ["variogram", "range", "azimuth"]
    else:
        field.update(
            variogram="spherical",
            major_range=max(1.0, round(settings.default_range_fraction * max(grid[0], grid[1]), 1)),
        )
        field["assumed"] += ["range", "variogram"]
    return field


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
        data: Path | None = None,
    ) -> GeomodelOutcome:
        """Write ``config`` with a new prior to ``output`` (see the module).

        ``data`` is a CSV of hard data (:mod:`apex.services.geomodeling_data`): the
        properties' statistics and variograms, and the facies, are estimated from it.

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
        if output.exists() or output.is_symlink():
            raise ConfigError(_taken(output))
        if output.resolve().parent != config.parent:
            raise ConfigError("write the new configuration beside the old one, so that its "
                              "relative paths still resolve")  # fmt: skip
        facts = geomodel_facts(raw, config.parent, reference)
        if facts.ensemble_size is None:  # the size the draft will have, for skills that draw
            facts = facts.model_copy(update={"ensemble_size": self.settings.ensemble_size})
        # A skill's stored ensemble is named after the new configuration: one per output.
        ensemble = output.with_name(f"{output.stem}_ensemble.npz")
        if ensemble.exists() or ensemble.is_symlink():
            raise ConfigError(_taken(ensemble))
        facts = facts.model_copy(update={"ensemble_file": ensemble.name})
        data_notes: list[str] = []
        if data is not None or reference is not None:
            from apex.services.geomodeling_data import (
                estimate_from_data,
                estimate_from_reference,
                load_hard_data,
                merge_estimates,
            )

            estimates: dict[str, PropertyEstimate] = {}
            facies = None
            if reference is not None:
                estimates, data_notes = estimate_from_reference(
                    Path(reference), facts, self.settings.estimator
                )
            if data is not None:
                found, facies, found_notes = estimate_from_data(
                    load_hard_data(Path(data)), facts, self.settings.estimator
                )
                estimates, data_notes = merge_estimates(estimates, found), data_notes + found_notes
            for estimate in estimates.values():
                data_notes += [f"{estimate.variable}: {n}" for n in estimate.notes]
            if facies is not None:
                data_notes += [f"facies: {n}" for n in facies.notes]
            facts = facts.model_copy(update={"estimates": estimates, "facies_data": facies})
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
        notes: list[str] = list(data_notes)
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
            if path.exists() or path.is_symlink():  # never through a link out
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
            members = settings.qc_members
            if draft.files:  # a stored ensemble: at most its own members
                members = min(members, loaded.prior.ensemble_size or members)
            prior = loaded.prior.model_copy(update={"ensemble_size": members})
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


def _taken(path: Path) -> str:
    """Return why ``path`` cannot be written, and what to do (release tester H12).

    The web UI and the assistant always write the default name, so the remedy must work
    there too: move the file away, or (on the command line) choose another output.
    """
    what = " (the new configuration's ensemble)" if path.name.endswith("_ensemble.npz") else ""
    return (f"{path.parent.name}/{path.name} exists{what}, and the geomodeler never writes over "
            "a file: delete or rename it, or choose another output file (--out on the "
            "command line)")  # fmt: skip

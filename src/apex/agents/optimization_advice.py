"""Diagnosis and planning for optimizations (2026-10-04).

The pattern: facts from code, named rules as the fallback, and the model's
answer held to the facts.

**Diagnosis** (:class:`OptimizationDiagnostician`): the monitor's facts of an optimization
(:mod:`apex.agents.optimization`) and its optimizer's settings become causes
(``failed_evaluations``, ``stall``, ``no_improvement``, ``iteration_limit``) and
recommendations with JSON merge patches over its configuration, each validated by the
service before it is shown (:func:`check_patch`). Advisory: nothing is changed.

**Planning** (:class:`OptimizationPlanner`): a study of optimizations for a case
and a goal, within a budget of simulator runs: optimizers to compare as complete
``[study.methods.<name>]`` entries (the rule), variants that patch anything but the
optimizer, and seeds. Code validates and costs every proposal (nothing is evaluated); an
invalid or over-budget one goes back to the model, then the rules plan. Submission goes
through :func:`apex.agents.planner.submit` (the tools, an approval token, a person's
confirmation), which takes optimization studies as any study.
"""

import inspect
import json
import logging
import math
import os
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import tomli_w
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic.errors import PydanticSchemaGenerationError, PydanticUserError

from apex.agents.common import Decision, decide
from apex.agents.diagnosis import Cause, Diagnosis, Recommendation, check_patches
from apex.agents.loop import LoopSettings
from apex.agents.optimization import (
    RULE_CODES,
    OptimizationFacts,
    OptimizationMonitor,
    optimization_rules,
)
from apex.agents.planner import _drop_none, null_paths
from apex.agents.trace import CallRecorder
from apex.core.advice import advise_prompt, advise_thresholds
from apex.core.errors import ApexError, ConfigError, PluginError
from apex.core.llm import LLMProvider
from apex.domain.optimization import OptimizationConfig
from apex.domain.optimization_study import (
    MULTI_OBJECTIVE_METRICS,
    OPTIMIZATION_METRICS,
    OptimizationStudySpec,
)
from apex.domain.services import Validation
from apex.domain.study import merge_patch
from apex.services.api import ApexService
from apex.services.plugins import option_model
from apex.services.settings import PlannerSection
from apex.storage import optimization_directory as layout

if TYPE_CHECKING:
    from apex.services.workspace import Workspace

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------- diagnosis

#: The diagnosis's records and advice are its own: advice for history-matching diagnosis
#: (localization, coverage, ...) does not apply to optimizations.
AGENT = "optimization_diagnosis"


class OptimizationDiagnosisFacts(BaseModel):
    """What the diagnosis knows of an optimization."""

    model_config = ConfigDict(extra="forbid")

    monitor: OptimizationFacts
    optimizer: dict[str, Any] = Field(description="Its [optimizer] section (name, options).")
    controls: int = Field(description="Number of control values.")
    realizations: int
    simulated: bool
    execution: dict[str, Any] = Field(description="Its [execution] section.")
    notes: list[str] = Field(default_factory=list)


def diagnosis_facts(monitor: OptimizationMonitor, target: str | Path) -> tuple[
    OptimizationDiagnosisFacts, dict[str, Any]
]:  # fmt: skip
    """Return the facts of an optimization and its configuration (as stored)."""
    facts = monitor.facts(target)
    config = layout.read_config(Path(facts.path))
    controls = sum(
        len(v["initial"]) if isinstance(v.get("initial"), list) else int(v.get("size", 1))
        for v in config.get("controls", {}).get("variables", [])
    )  # fmt: skip
    return OptimizationDiagnosisFacts(
        monitor=facts,
        optimizer=config["optimizer"],
        controls=controls,
        realizations=int(config.get("controls", {}).get("realizations", 1)),
        simulated=config.get("simulator") is not None,
        execution=config.get("execution", {"name": "serial"}),
    ), config


def diagnosis_rules(facts: OptimizationDiagnosisFacts, report_codes: set[str]) -> Diagnosis:
    """Return the rules' diagnosis from the monitor rules' findings ``report_codes``."""
    m = facts.monitor
    options = facts.optimizer.get("options", {})
    causes: list[Cause] = []
    recs: list[Recommendation] = []
    if "failed_evaluations" in report_codes:
        causes.append(Cause(
            code="failed_evaluations", severity="critical",
            evidence=f"{m.failed_evaluations} simulated evaluations failed",
        ))  # fmt: skip
        retries = facts.execution.get("options", {}).get("retries", 2)
        recs.append(Recommendation(
            change="look at the failed simulations' folders (simulations/eval_<id>, marked "
            "APEX_FAILED) and fix the cause; then resume, which simulates only those",
            rationale="the run stops at a failed evaluation", addresses=["failed_evaluations"],
            patch=None if retries else {"execution": {"options": {"retries": 2}}},
        ))  # fmt: skip
    if "stall" in report_codes or "no_improvement" in report_codes:
        code = "stall" if "stall" in report_codes else "no_improvement"
        gain = m.window_improvement if code == "stall" else m.relative_improvement
        causes.append(Cause(
            code=code, severity="warning",
            evidence=f"relative improvement {gain if gain is not None else 'none'} after "
            f"{m.accepted_steps} accepted steps",
        ))  # fmt: skip
        if "alpha" in options:
            recs.append(Recommendation(
                change=f"halve the step size alpha ({options['alpha']} -> {options['alpha'] / 2})",
                rationale="a step too long for the objective's curvature is rejected or "
                "overshoots", addresses=[code],
                patch={"optimizer": {"options": {"alpha": options["alpha"] / 2}}},
            ))  # fmt: skip
        from apex.algorithms.optimization.population import is_population_optimizer

        name = str(facts.optimizer.get("name"))
        if is_population_optimizer(name):
            import inspect

            from apex.algorithms.optimization import optimizers

            default = inspect.signature(optimizers.factory(name)).parameters["pop_size"].default
            size = int(options.get("pop_size", default))
            recs.append(Recommendation(
                change=f"double the population pop_size ({size} -> {2 * size})",
                rationale="a small population loses its diversity and settles early; a larger "
                "one evaluates more members per generation", addresses=[code],
                patch={"optimizer": {"options": {"pop_size": 2 * size}}},
            ))  # fmt: skip
        if "ne" in options and facts.simulated:
            recs.append(Recommendation(
                change=f"double the ensemble ne ({options['ne']} -> {2 * options['ne']})",
                rationale="a few perturbations give a noisy gradient for "
                f"{facts.controls} controls", addresses=[code],
                patch={"optimizer": {"options": {"ne": 2 * int(options["ne"])}}},
            ))  # fmt: skip
    if "iteration_limit" in report_codes:
        causes.append(Cause(
            code="iteration_limit", severity="info",
            evidence=f"stopped at maxiter after {m.accepted_steps} accepted steps",
        ))  # fmt: skip
        if "maxiter" in options and not report_codes & {"stall", "no_improvement", "front_stall"}:
            recs.append(Recommendation(
                change=f"resume with more iterations (maxiter {options['maxiter']} -> "
                f"{2 * int(options['maxiter'])}), or `apex optimize --resume`",
                rationale="it was still improving when it stopped", addresses=["iteration_limit"],
                patch={"optimizer": {"options": {"maxiter": 2 * int(options["maxiter"])}}},
            ))  # fmt: skip
    _front_diagnosis(facts, report_codes, causes, recs)
    codes = ", ".join(c.code for c in causes) or "nothing to fix"
    return Diagnosis(
        verdict=_verdict(causes),
        causes=causes,
        recommendations=recs,
        summary=f"{m.optimizer} optimization {m.name} ({m.state}, {m.accepted_steps} accepted "
        f"steps, {m.evaluations} evaluations): {codes}.",
    )


def _front_diagnosis(
    facts: OptimizationDiagnosisFacts,
    report_codes: set[str],
    causes: list[Cause],
    recs: list[Recommendation],
) -> None:
    """Add the causes and recommendations of a multi-objective run's front."""
    from apex.agents.optimization import option_default
    from apex.algorithms.optimization import optimizers

    m = facts.monitor
    name = str(facts.optimizer.get("name"))
    if not m.objective_names or name not in optimizers.names():
        return
    factory = optimizers.factory(name)
    options = facts.optimizer.get("options", {})
    # About twice the population, the optimizer's own way (nsga3, moead: more directions),
    # from the run's real population (the monitor's facts): a merge patch of what changes.
    larger, _, change = larger_setting(name, options, max(len(m.objective_names), 2),
                                       m.pop_size)  # fmt: skip
    grow: dict[str, Any] = {k: v for k, v in larger.items() if options.get(k) != v}
    grow.update({k: None for k in options if k not in larger})  # e.g. nsga3's pop_size
    grow_patch = {"optimizer": {"options": grow}}
    if "no_feasible_member" in report_codes:
        causes.append(Cause(
            code="no_feasible_member", severity="warning",
            evidence=f"no feasible member for {m.infeasible_generations} generations",
        ))  # fmt: skip
        if options.get("constraint_handling") == "epf":
            epf = dict(options.get("epf") or {})
            r = float(epf.get("r", 1.0))
            recs.append(Recommendation(
                change=f"raise the penalty factor epf.r ({r:g} -> {10 * r:g}); the evaluation "
                "bound is unchanged",
                rationale="with a small factor the objectives outweigh the violation, so the "
                "population settles outside the constraints", addresses=["no_feasible_member"],
                patch={"optimizer": {"options": {"epf": {**epf, "r": 10 * r}}}},
            ))  # fmt: skip
        else:
            recs.append(Recommendation(
                change="check that the constraints can be met within the bounds (e.g. start "
                f"from a known feasible point); if they can, give the search more room: {change}, "
                "which doubles the evaluations",
                rationale="Deb's feasibility rules already rank infeasible members by violation "
                "alone, the strongest push towards the constraints; if that does not reach "
                "them, the feasible region is small or empty", addresses=["no_feasible_member"],
                patch=grow_patch,
            ))  # fmt: skip
        tolerance = float(options.get("eq_tolerance", option_default(factory, "eq_tolerance")
                                      or 1e-4))  # fmt: skip
        recs.append(Recommendation(
            change=f"if the constraints include equalities, loosen eq_tolerance ({tolerance:g} "
            f"-> {10 * tolerance:g}): the variation rarely hits an equality exactly",
            rationale="an equality is met only within eq_tolerance; the facts do not say "
            "whether this case has equalities, so no patch is given",
            addresses=["no_feasible_member"], patch=None,
        ))  # fmt: skip
    if "small_front" in report_codes:
        causes.append(Cause(
            code="small_front", severity="warning",
            evidence=f"a front of {m.front_size} members after {m.accepted_steps} generations, "
            f"of {m.pop_size} members"
            + ("" if m.feasible_share is None else f", {m.feasible_share:.0%} of them feasible"),
        ))  # fmt: skip
        recs.append(Recommendation(
            change=f"a larger population: {change}, which doubles the evaluations",
            rationale="a larger population keeps more of the trade-off", addresses=["small_front"],
            patch=grow_patch,
        ))  # fmt: skip
        eta = float(options.get("eta_m", option_default(factory, "eta_m") or 20.0))
        recs.append(Recommendation(
            change=f"wider mutations: halve eta_m ({eta:g} -> {eta / 2:g})",
            rationale="a smaller distribution index spreads the children further, restoring "
            "diversity", addresses=["small_front"],
            patch={"optimizer": {"options": {"eta_m": eta / 2}}},
        ))  # fmt: skip
    if "front_stall" in report_codes:
        causes.append(Cause(
            code="front_stall", severity="warning",
            evidence=f"the front has not moved for {m.stalled_generations} generations",
        ))  # fmt: skip
        recs.append(Recommendation(
            change="accept the front: stop the run (apex stop, or the job's Stop) and read "
            "pareto.json; more generations will not move it",
            rationale="the front moved less than ftol for many generations",
            addresses=["front_stall"], patch=None,
        ))  # fmt: skip
        recs.append(Recommendation(
            change=f"for a better front, run again with a larger population ({change}; twice "
            "the evaluations)",
            rationale="a larger population explores more of the trade-off",
            addresses=["front_stall"], patch=grow_patch,
        ))  # fmt: skip

    if "hypervolume_drop" in report_codes:
        causes.append(Cause(
            code="hypervolume_drop", severity="info",
            evidence=f"the hypervolume fell from {m.best_hypervolume} to {m.hypervolume} under "
            "one penalty factor; look at the history for a noisy objective or a front cut by "
            "crowding (nothing to change otherwise)",
        ))  # fmt: skip


def _verdict(causes: list[Cause]) -> Literal["healthy", "problems", "failed"]:
    """Return the verdict the causes support: failed if critical, problems if a warning."""
    severities = {c.severity for c in causes}
    if "critical" in severities:
        return "failed"
    return "problems" if severities - {"info"} else "healthy"


def check_patch(service: ApexService, config: dict[str, Any], patch: dict[str, Any]) -> str | None:
    """Return why ``patch`` does not apply to an optimization's ``config`` (None: it does)."""
    document = merge_patch(config, patch)
    try:
        OptimizationConfig.model_validate(document)
    except ValidationError as error:
        return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
    validation = service.validate_optimization(document)
    return None if validation.valid else "; ".join(validation.errors)


def reconcile(
    model: Diagnosis, rules: Diagnosis, patch_error: Callable[[dict[str, Any]], str | None]
) -> tuple[Diagnosis, list[str]]:
    """Hold the model's diagnosis to the facts (as :func:`apex.agents.diagnosis.reconcile`)."""
    found = {c.code for c in rules.causes}
    changes = [f"dropped cause {c.code}: the rules did not find it in the facts"
               for c in model.causes if c.code in RULE_CODES and c.code not in found]  # fmt: skip
    causes = [c for c in model.causes if c.code not in RULE_CODES or c.code in found]
    recommendations, dropped = check_patches(model.recommendations, patch_error)
    changes += dropped
    if not changes:
        return model, []
    verdict = model.verdict
    if len(causes) < len(model.causes) and verdict != _verdict(causes):
        # Never a verdict worse (or better) than the causes kept support.
        changes.append(f"verdict {verdict} became {_verdict(causes)}: the causes kept")
        verdict = _verdict(causes)
    summary = model.summary + f" [Corrected: {len(changes)} change(s), see the corrections.]"
    return model.model_copy(
        update={"causes": causes, "recommendations": recommendations, "summary": summary,
                "verdict": verdict}
    ), changes  # fmt: skip


DIAGNOSIS_SYSTEM = (
    "You diagnose an APEX optimization (an ensemble optimizer such as EnOpt minimising an "
    "objective, e.g. -NPV): what went wrong and what to change. You get facts computed from "
    "its directory, its optimizer's settings, and the diagnosis of threshold rules. Answer "
    "from the facts; keep every cause of the rules unless the facts contradict it, and cite "
    "numbers. Give a patch (a JSON merge patch over the optimization's configuration, e.g. "
    '{"optimizer": {"options": {"alpha": 0.05}}}) only for a setting you are sure exists. '
    "A multi-objective run (nsga2, nsga3 or moead; several objective_names) is judged by "
    "its front: "
    "feasibility, the front's size against pop_size, and whether it still moves."
)


@dataclass(frozen=True)
class OptimizationDiagnosis:
    """One diagnosis of an optimization (see :class:`apex.agents.diagnosis.DiagnosisResult`)."""

    facts: OptimizationDiagnosisFacts
    rules_diagnosis: Diagnosis
    diagnosis: Diagnosis
    decided_by: str
    reason: str | None
    trace: Path
    corrections: tuple[str, ...] = ()


class OptimizationDiagnostician:
    """Diagnoses optimizations (see the module docstring)."""

    def __init__(
        self,
        service: ApexService,
        provider: LLMProvider | None,
        *,
        settings: LoopSettings | None = None,
    ) -> None:
        self.service = service
        self.provider = provider
        self.settings = settings or LoopSettings(max_turns=6)
        self._monitor = OptimizationMonitor(service, None)

    def diagnose(self, target: str | Path) -> OptimizationDiagnosis:
        """Diagnose an optimization (its directory or job) once."""
        facts, config = diagnosis_facts(self._monitor, target)
        report = optimization_rules(facts.monitor, self._monitor.thresholds)
        rules = diagnosis_rules(facts, {f.code for f in report.findings})

        def patch_error(patch: dict[str, Any]) -> str | None:
            return check_patch(self.service, config, patch)

        checked, dropped = check_patches(rules.recommendations, patch_error)
        rules = rules.model_copy(update={"recommendations": checked})
        facts = facts.model_copy(update={"notes": [*facts.notes, *dropped]})
        recorder = CallRecorder(Path(facts.monitor.path), AGENT)
        system, advice = advise_prompt(DIAGNOSIS_SYSTEM, AGENT, self.service.root)
        recorder.advice = sorted(advice)
        decision: Decision[Diagnosis] = decide(
            self.service, self.provider, recorder,
            task=(f"Facts of the optimization:\n{facts.model_dump_json(indent=1)}\n\n"
                  f"The threshold rules' diagnosis:\n{rules.model_dump_json(indent=1)}\n\n"
                  "Give your diagnosis."),
            response_model=Diagnosis, rules=rules,
            rule="the optimization diagnosis rules",
            thresholds={"monitor": self._monitor.thresholds.model_dump(mode="json")},
            client=AGENT, settings=self.settings, system=system,
        )  # fmt: skip
        diagnosis = decision.value
        corrections: list[str] = []
        if decision.decided_by == "model":
            diagnosis, corrections = reconcile(diagnosis, rules, patch_error)
            if corrections:
                recorder.record_action(
                    "corrected the model's diagnosis: " + "; ".join(corrections),
                    corrections=corrections, diagnosis=diagnosis.model_dump(),
                )  # fmt: skip
        return OptimizationDiagnosis(
            facts, rules, diagnosis, decision.decided_by, decision.reason, decision.trace,
            tuple(corrections),
        )  # fmt: skip


# ---------------------------------------------------------------------- planning


class OptimizationPlanFacts(BaseModel):
    """What the planner knows of an optimization case."""

    model_config = ConfigDict(extra="forbid")

    case: str
    config_file: str
    objective: str
    objective_names: list[str] = Field(
        default_factory=list,
        description="Several objectives: only a multi-objective optimizer (nsga2, "
        "nsga3, moead) runs this case, and studies compare fronts (hypervolume:final, "
        "front_size:final).",
    )
    simulated: bool
    optimizer: dict[str, Any] = Field(description="The case's own [optimizer] (name, options).")
    controls: int
    runs_per_cell: int | None = Field(
        description="Simulator runs one optimization may take (its plan's bound; 0 analytic)."
    )
    budget_simulator_runs: int
    max_cells: int | None = Field(description="Optimizations the budget allows at this cost.")
    max_variants: int = Field(description="At most this many methods x variants.")
    optimizers: list[str] = Field(description="Registered optimizers.")
    step_rules: list[str]
    configuration: dict[str, Any]


class OptimizationMethod(BaseModel):
    """A complete [optimizer] entry."""

    model_config = ConfigDict(extra="forbid")

    name: str
    options: dict[str, Any] = Field(default_factory=dict)


class OptimizationVariantProposal(BaseModel):
    """One variant: a patch of anything but the optimizer and the run's identity."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    patch: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(min_length=3)


class OptimizationStudyProposal(BaseModel):
    """A proposed study of optimizations."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    hypothesis: str = Field(min_length=3)
    methods: dict[str, OptimizationMethod] = Field(
        default_factory=dict,
        description="Optimizers to compare, name -> a complete [optimizer] entry; {} runs the "
        "case's own. Copy the case's optimizer only for that optimizer.",
    )
    variants: list[OptimizationVariantProposal] = Field(min_length=1)
    replicates: int = Field(ge=1, description="Seeds per method and variant (paired).")
    metrics: list[str] = Field(default_factory=lambda: list(OPTIMIZATION_METRICS))
    reasoning: str = Field(min_length=3)


def plan_facts(
    service: ApexService, config_file: Path, *, budget: int | None, thresholds: PlannerSection
) -> OptimizationPlanFacts:
    """Return the facts of an optimization case."""
    from apex.algorithms.optimization import optimizers, step_rules
    from apex.services import optimization as opt

    config = service.resolve_optimization(config_file)
    try:
        plan = opt.plan_optimization(config)
    except (ValueError, TypeError) as error:
        raise ConfigError(f"the optimization case does not build: {error}") from error
    budget_runs = budget or service.jobs.settings.approval.max_simulator_runs
    per = plan.max_simulator_runs
    # An analytic objective costs no simulator runs: the planner's own bounds limit it.
    analytic = thresholds.max_replicates * thresholds.max_variants
    cells = None if per is None else (budget_runs // per if per else analytic)
    return OptimizationPlanFacts(
        case=config.optimization.name,
        config_file=str(Path(config_file).resolve()),
        objective=config.objective_label,
        objective_names=_objective_names(config),
        simulated=config.simulator is not None,
        optimizer=config.optimizer.model_dump(mode="json", exclude_none=True),
        controls=plan.n_controls,
        runs_per_cell=per,
        budget_simulator_runs=budget_runs,
        max_cells=cells,
        max_variants=thresholds.max_variants,
        optimizers=sorted(optimizers.names()),
        step_rules=sorted(step_rules.names()),
        configuration=config.model_dump(mode="json", exclude={"optimization"}),
    )


def _objective_names(config: OptimizationConfig) -> list[str]:
    """Return a configuration's objectives when it has several, else ``[]``."""
    from apex.algorithms.optimization import objectives

    if config.objective is None:
        return [entry.shown for entry in config.objectives]
    # Built with its options: DTLZ's names depend on its n_obj, not its class.
    try:
        plugin = objectives.create(config.objective.name, **config.objective.options)
    except (PluginError, ValueError, TypeError):
        return []
    return [str(n) for n in getattr(plugin, "objective_names", ())]


def larger_setting(
    name: str, options: dict[str, Any], n_obj: int, pop_size: int | None = None
) -> tuple[dict[str, Any], str, str]:
    """Return a multi-objective optimizer's options at about twice its population.

    Twice ``pop_size`` (``nsga2``), or, for an optimizer with reference directions
    (``nsga3``, ``moead``: their population follows them), the partitions of both layers
    grown until there are at least twice the directions. Returns the options, a label and a
    description. ``pop_size`` is the run's real population when known (the monitor's facts).
    """
    from apex.agents.optimization import option_default
    from apex.algorithms.optimization import optimizers
    from apex.algorithms.optimization.reference_directions import (
        default_layers,
        layered_directions,
    )

    factory = optimizers.factory(name)
    if "n_partitions" in inspect.signature(factory).parameters:
        given_p, given_i = options.get("n_partitions"), options.get("inner_partitions")
        if given_p is None and given_i is None:
            p, i = default_layers(n_obj)
        else:
            p, i = int(given_p or default_layers(n_obj)[0]), int(given_i or 0)
        base = len(layered_directions(p, i, n_obj))
        q, j = p + 1, (i + 1 if i else 0)  # both layers grow, to keep an interior
        while len(layered_directions(q, j, n_obj)) < 2 * base:
            q, j = q + 1, (j + 1 if j else 0)
        larger: dict[str, Any] = {**options, "n_partitions": q}
        if j:
            larger["inner_partitions"] = j
        # nsga3's population follows the directions unless set (then doubled, so the larger
        # setting is never smaller); moead has none.
        if options.get("pop_size") is not None:
            larger["pop_size"] = 2 * int(options["pop_size"])
        count = len(layered_directions(q, j, n_obj))
        description = (f"{base} reference directions (partitions {p}"
                       + (f", inner {i}" if i else "") + f") -> {count} ({q}"
                       + (f", inner {j}" if j else "") + ")")  # fmt: skip
        return larger, f"{name}_p{q}" + (f"_{j}" if j else ""), description
    size = int(pop_size or options.get("pop_size", option_default(factory, "pop_size") or 40))
    return ({**options, "pop_size": 2 * size}, f"{name}_pop{2 * size}",
            f"pop_size {size} -> {2 * size}")  # fmt: skip


def _front_plan(
    facts: OptimizationPlanFacts, goal: str, t: PlannerSection
) -> OptimizationStudyProposal:
    """Return the rules' plan for several objectives: the case's optimizer, and a larger one.

    Larger means twice the population (``nsga2``), or the reference directions' partitions
    that give at least twice as many directions (``nsga3`` and ``moead``, whose population
    follows their directions). The fronts are compared (hypervolume, size), and the
    evaluations. Only a multi-objective optimizer can run the case, so no single-objective
    one is proposed; the larger one costs about twice the case's evaluations.
    """
    own = OptimizationMethod.model_validate(facts.optimizer)
    n_obj = max(len(facts.objective_names), 2)
    larger_options, label, change = larger_setting(own.name, own.options, n_obj)
    compared_with = f"{own.name}: {change}"
    larger = OptimizationMethod(name=own.name, options=larger_options)
    per = facts.runs_per_cell
    # Simulator runs of one replicate of both methods (the larger: about twice the case's).
    pair = 3 * per if per else 0
    fits = (facts.max_variants >= 2
            and (per is None or pair == 0 or pair <= facts.budget_simulator_runs))  # fmt: skip
    methods = {own.name: own, label: larger} if fits else {}
    if per:
        cost = pair if methods else per
        replicates = max(1, min(t.max_replicates, facts.budget_simulator_runs // cost))
    else:
        replicates = t.max_replicates
    compared = compared_with if methods else f"{own.name} as configured"
    return OptimizationStudyProposal(
        name=f"{facts.case}_plan",
        hypothesis=f"Which population size best finds the trade-off between "
        f"{', '.join(facts.objective_names)}: {goal}"[:300],
        methods=methods,
        variants=[OptimizationVariantProposal(name="base", rationale="the case as configured")],
        replicates=replicates,
        metrics=list(MULTI_OBJECTIVE_METRICS),
        reasoning=f"rules: {compared}, {replicates} paired seed(s) "
        + ("within" if not per or per <= facts.budget_simulator_runs else "over")
        + f" the budget of {facts.budget_simulator_runs} simulator runs ({per} per "
        "optimization of the case); fronts compared by hypervolume and size",
    )


def plan_rules(
    facts: OptimizationPlanFacts, goal: str, thresholds: PlannerSection | None = None
) -> OptimizationStudyProposal:
    """Return the rules' plan: the case's optimizer against another, with paired seeds.

    The other is EnOpt (PET's default) unless the case already runs it, then SmcOpt (a
    sampler of the same ensemble size). Replicates are as many as the budget allows,
    capped by ``max_replicates``; with no room for two optimizers, the case's alone.
    """
    t = thresholds or PlannerSection()
    if facts.objective_names:
        return _front_plan(facts, goal, t)
    from apex.algorithms.optimization.population import is_population_optimizer

    own = OptimizationMethod.model_validate(facts.optimizer)
    ne = own.options.get("ne", 10)
    # A population optimizer's maxiter counts generations, not EnOpt's iterations.
    maxiter = 10 if is_population_optimizer(own.name) else own.options.get("maxiter", 10)
    other = (
        OptimizationMethod(name="smcopt", options={"ne": ne, "maxiter": maxiter})
        if own.name == "enopt"
        else OptimizationMethod(name="enopt", options={"ne": ne, "maxiter": maxiter})
    )
    methods = {own.name: own, other.name: other} if other.name in facts.optimizers else {}
    groups = max(len(methods), 1)
    cells = facts.max_cells if facts.max_cells is not None else groups
    if cells < groups:
        methods, groups = {}, 1  # the case's own optimizer alone
    replicates = max(1, min(t.max_replicates, cells // groups))
    variants = [OptimizationVariantProposal(name="base", rationale="the case as configured")]
    compared = " against ".join(methods) if methods else own.name
    return OptimizationStudyProposal(
        name=f"{facts.case}_plan",
        hypothesis=f"Which optimizer best serves the goal: {goal}"[:300],
        methods=methods,
        variants=variants,
        replicates=replicates,
        reasoning=f"rules: {compared}, {replicates} paired seed(s) within the budget of "
        f"{facts.budget_simulator_runs} simulator runs ({facts.runs_per_cell} per optimization)",
    )


@dataclass(frozen=True)
class OptimizationPlanResult:
    """A planned study of optimizations (see :class:`apex.agents.planner.PlanResult`)."""

    facts: OptimizationPlanFacts
    proposal: OptimizationStudyProposal
    spec: OptimizationStudySpec
    validation: Validation
    cost: int | None
    within_budget: bool
    decided_by: str
    reason: str | None
    trace: Path
    revisions: tuple[str, ...] = ()
    #: Settings :func:`match_arms` gave an arm so that the arms run alike.
    matched: tuple[str, ...] = ()

    def toml(self, workspace: "Workspace | None" = None, study_dir: Path | None = None) -> str:
        """Return the study as TOML, headed by the hypothesis and reasoning.

        With ``workspace`` and ``study_dir`` (the file's folder), its case is named as the
        workspace shows it, or relative to the file: no server paths.
        """
        header = (
            f"# Planned by APEX's planning agent ({self.decided_by}).\n"
            f"# Hypothesis: {self.proposal.hypothesis}\n"
            f"# Reasoning: {self.proposal.reasoning}\n"
            + "".join(f"# {v.name}: {v.rationale}\n" for v in self.proposal.variants)
            + f"# Cost: at most {self.cost} simulator runs "
            f"(budget {self.facts.budget_simulator_runs}).\n\n"
        )
        spec = self.spec
        if workspace is not None and study_dir is not None:
            cases: dict[str, Path] = {}
            for name, path in spec.study.cases.items():
                try:
                    cases[name] = Path(workspace.relative(path))
                except ConfigError:  # outside the workspace
                    cases[name] = Path(os.path.relpath(path, study_dir))
            spec = spec.model_copy(update={"study": spec.study.model_copy(update={"cases": cases})})
        text = header + tomli_w.dumps(_drop_none(spec.model_dump(mode="json")))
        if OptimizationStudySpec.model_validate(tomllib.loads(text)) != spec:
            raise ConfigError("the study's TOML would differ from the validated study")
        return text


PLAN_SYSTEM = (
    "You plan a study of APEX optimizations: optimizers to compare on a case, "
    "within a budget of simulator runs. Each optimizer is a complete [study.methods.<name>] "
    "entry (name and options; list_plugins and get_schema show them); variants patch "
    "anything but [optimizer] and the run's name, seed and place; replicates are paired "
    "seeds. Keep the study within the budget: runs_per_cell x methods x variants x "
    "replicates simulator runs. A case with several objective_names runs only on a "
    "multi-objective optimizer (nsga2, nsga3, moead): compare its settings (pop_size, or "
    "n_partitions for nsga3 and moead) as methods, and "
    "its fronts by hypervolume:final, front_size:final and evaluations:all."
)


def objective_count(config: OptimizationConfig) -> int:
    """Return how many objectives a configuration optimizes (one unless several)."""
    return max(len(_objective_names(config)), 1)


#: Options that size a population optimizer, matched as one population (a reported issue).
POPULATION_OPTIONS = frozenset({"pop_size", "n_partitions", "inner_partitions"})
#: Optimizers whose populations :func:`match_arms` matches.
POPULATION_OPTIMIZERS = frozenset({"ga", "nsga2", "nsga3", "moead", "pso"})
#: Options with one meaning across a family of optimizers: the operators. Not termination
#: (its units differ: a generation is pop_size evaluations, an EnOpt iteration ne), nor
#: options that only share a name (line_search's Wolfe c1, c2 and pso's acceleration).
SHARED_OPTIONS = (
    (frozenset({"ga", "nsga2", "nsga3", "moead"}),
     frozenset({"crossover_prob", "crossover_prob_var", "eta_c", "mutation_prob",
                "mutation_prob_var", "eta_m", "eliminate_duplicates"})),
    (frozenset({"enopt", "genopt", "smcopt"}),
     frozenset({"ne", "alpha", "alpha_maxiter", "cov_factor", "resample"})),
)  # fmt: skip


def shared_options(one: str, other: str) -> frozenset[str]:
    """Return the options two different optimizers share in meaning (:data:`SHARED_OPTIONS`)."""
    if one == other:
        return frozenset()
    for family, options in SHARED_OPTIONS:
        if one in family and other in family:
            return options
    return frozenset()


def _native(value: Any) -> Any:  # noqa: ANN401 (an option's value)
    """Return a JSON-native scalar as is, anything else (None, objects) as ``None``."""
    return value if isinstance(value, bool | int | float | str) else None


_NO_OPTION_MODEL = (KeyError, ApexError, TypeError, PydanticUserError,
                    PydanticSchemaGenerationError)  # fmt: skip


#: The least settings of the options that bound an optimizer's simulator runs, by option name
#: (an optimizer without the option is left alone): one iteration (or batch), a two-member
#: ensemble, no backtracking trials, resampling, gradient recomputation or radius cuts. Two
#: members, not one: a one-member ensemble estimates no gradient, and genopt's CMA refuses it
#: (n_mu = 0). A population's size is not cut here (its least size is the method's own
#: question) and is reported as kept. A reported issue (#242): over a small budget the
#: assistant could not learn a method's least cost and guessed designs one refusal at a time.
CHEAPEST_OPTIONS: dict[str, int] = {
    "maxiter": 1, "max_batches": 1, "ne": 2, "alpha_maxiter": 0, "resample": 0,
    "lsmaxiter": 1, "recompute_jac": 0, "trust_radius_cuts": 0, "number_of_perturbations": 1,
}  # fmt: skip


@dataclass(frozen=True)
class LeastCost:
    """One method's least cost in simulator runs (``None``: not bounded, or ``error``)."""

    cost: int | None
    applied: dict[str, int]
    population: int | None = None
    error: str | None = None


def _option_fields(name: str) -> dict[str, Any] | None:
    try:
        return dict(option_model("optimizers", name).model_fields)
    except _NO_OPTION_MODEL:
        return None


def population(name: str, options: Mapping[str, Any], n_obj: int) -> int | None:
    """Return the population optimizer ``name`` runs with ``options``, or ``None``.

    ``None`` for an optimizer without a population (``enopt``) or one that cannot be told.
    ``nsga3`` and ``moead`` follow their reference directions: ``moead`` has one member
    per direction, ``nsga3`` its ``pop_size`` if given, else the directions rounded up to a
    multiple of four (its own rule, ``NSGA3.set_objectives``).
    """
    from apex.algorithms.optimization.reference_directions import directions_for

    fields = _option_fields(name)
    if fields is None:
        return None
    if "n_partitions" in fields:
        try:
            k = len(directions_for(n_obj, options.get("n_partitions"),
                                   options.get("inner_partitions")))  # fmt: skip
        except (ValueError, TypeError):  # fewer than two objectives, bad partitions
            return None
        if "pop_size" not in fields:
            return k
        given = options.get("pop_size")
        return int(given) if given is not None else 4 * math.ceil(k / 4)
    if "pop_size" in fields:
        value = options.get("pop_size", fields["pop_size"].get_default(call_default_factory=True))
        return int(value) if value is not None else None
    return None


def population_options(name: str, n_obj: int, size: int) -> dict[str, Any] | None:
    """Return options giving optimizer ``name`` a population of ``size``, or ``None``.

    ``nsga3`` takes ``pop_size`` and the most one-layer directions within it, so every
    direction keeps a member; ``moead`` (one member per direction) the direction count
    nearest ``size``, which :func:`population` then reports if it is not exact.
    """
    from apex.algorithms.optimization.reference_directions import number_of_directions

    fields = _option_fields(name)
    if fields is None:
        return None
    if "n_partitions" in fields:
        if n_obj < 2 or number_of_directions(1, n_obj) > size:
            return None
        q = 1
        if "pop_size" in fields:
            while number_of_directions(q + 1, n_obj) <= size:
                q += 1
            return {"n_partitions": q, "pop_size": size}
        while abs(number_of_directions(q + 1, n_obj) - size) < abs(
            number_of_directions(q, n_obj) - size
        ):
            q += 1
        return {"n_partitions": q}
    if "pop_size" in fields:
        return {"pop_size": size}
    return None


def match_arms(
    proposal: OptimizationStudyProposal, n_obj: int
) -> tuple[OptimizationStudyProposal, list[str]]:
    """Give every arm the first arm's settings it leaves to its plugin (a reported issue).

    Compared optimizers ran with their plugins' own defaults (nsga2's crossover_prob 0.9
    and eta_c 15 against nsga3's 1.0 and 30, a population of 8 against one of the
    directions), so a gap could be the settings, not the optimizer. An arm of another
    optimizer than the first's gets the first arm's value (given or default) of each
    option of :func:`shared_options` it does not set, and, between population optimizers,
    the first arm's population (:func:`population_options`). What an arm sets itself is
    never changed, a value its plugin refuses is left as it was, and arms of the first
    arm's own optimizer compare its settings, so they are left alone. Whatever still
    differs is said by the comparison's ``unmatched``. Returns the proposal and what was
    changed.
    """
    arms = list(proposal.methods.items())
    if len(arms) < 2:
        return proposal, []
    first_label, first = arms[0]
    first_fields = _option_fields(first.name)
    if first_fields is None:
        return proposal, []
    reference = {
        key: _native(first.options[key] if key in first.options
                     else info.get_default(call_default_factory=True))
        for key, info in first_fields.items()
        if key in first.options or not info.is_required()
    }  # fmt: skip
    size = population(first.name, first.options, n_obj)
    methods = {first_label: first}
    changes: list[str] = []
    for label, arm in arms[1:]:
        fields = _option_fields(arm.name)
        if fields is None or arm.name == first.name:
            methods[label] = arm
            continue
        shared = shared_options(first.name, arm.name)
        model = option_model("optimizers", arm.name)

        def valid(options: dict[str, Any], model: type[BaseModel] = model) -> bool:
            try:
                model.model_validate(options)
            except ValidationError:
                return False
            return True

        options = dict(arm.options)
        for key, value in reference.items():
            if key not in shared or key in arm.options or key not in fields or value is None:
                continue  # TOML holds no null: an unset (None) option stays the plugin's
            default = _native(fields[key].get_default(call_default_factory=True))
            if default != value and valid({**options, key: value}):
                options[key] = value
                changes.append(f"{label}: {key} {default!r} -> {value!r}, as {first_label}")
        if (size is not None and {first.name, arm.name} <= POPULATION_OPTIMIZERS
                and not POPULATION_OPTIONS & arm.options.keys()):  # fmt: skip
            before = population(arm.name, options, n_obj)
            sized = population_options(arm.name, n_obj, size)
            if before != size and sized is not None and valid({**options, **sized}):
                options.update(sized)
                after = population(arm.name, options, n_obj)
                given = ", ".join(f"{k} = {v}" for k, v in sized.items())
                changes.append(f"{label}: population {before} -> {after} ({given}), as "
                               f"{first_label}'s {size}")  # fmt: skip
        methods[label] = arm.model_copy(update={"options": options})
    return proposal.model_copy(update={"methods": methods}), changes


class OptimizationPlanner:
    """Plans studies of optimizations (see the module docstring)."""

    def __init__(
        self,
        service: ApexService,
        provider: LLMProvider | None,
        *,
        thresholds: PlannerSection | None = None,
        settings: LoopSettings | None = None,
    ) -> None:
        self.service = service
        self.provider = provider
        self.threshold_advice: list[str] = []
        if thresholds is None:
            thresholds, self.threshold_advice = advise_thresholds(
                "agents.planner", service.jobs.settings.agents.planner, service.root
            )
        self.thresholds = thresholds
        self.settings = settings or LoopSettings(max_turns=6)

    def spec(
        self, facts: OptimizationPlanFacts, proposal: OptimizationStudyProposal
    ) -> OptimizationStudySpec:
        """Return the study a proposal makes.

        Raises
        ------
        pydantic.ValidationError
            If the proposal does not make a valid study.
        """
        return OptimizationStudySpec.model_validate({
            "study": {
                "kind": "optimization",
                "name": proposal.name,
                "replicates": proposal.replicates,
                "metrics": proposal.metrics,
                "budget": {"max_simulator_runs": facts.budget_simulator_runs},
                "annotations": {"planned_by": "apex-planner"},
                "cases": {facts.case: facts.config_file},
                "methods": {k: v.model_dump(mode="json") for k, v in proposal.methods.items()},
                "variants": [{"name": v.name, "patch": v.patch} for v in proposal.variants],
            }
        })  # fmt: skip

    def check(
        self, facts: OptimizationPlanFacts, proposal: OptimizationStudyProposal
    ) -> tuple[OptimizationStudySpec | None, Validation | None, int | None, list[str]]:
        """Hold a proposal to the facts: return its study, validation, cost and errors."""
        errors: list[str] = []
        groups = max(len(proposal.methods), 1) * len(proposal.variants)
        if groups > facts.max_variants:
            errors.append(f"{groups} methods x variants; at most {facts.max_variants}")
        for v in proposal.variants:
            if null_paths(v.patch):
                errors.append(f"variant {v.name!r} uses null, which a TOML study cannot hold")
        if errors:
            return None, None, None, errors
        try:
            spec = self.spec(facts, proposal)
        except ValidationError as error:
            errors += [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors()]
            return None, None, None, errors
        from apex.services.optimization_study import plan_optimization_study

        # The study's own validation refuses a cost over its budget; the cost is kept either
        # way, so that an over-budget plan says by how much.
        validation = self.service.validate_study(spec)
        errors += validation.errors
        planned = plan_optimization_study(self.service, spec, Path(facts.config_file).parent)
        cost = planned.max_simulator_runs if planned.cells else None
        return spec, validation, cost, errors

    def plan(
        self, config_file: Path, goal: str, *, budget: int | None = None
    ) -> OptimizationPlanResult:
        """Plan a study for an optimization case and a goal."""
        facts = plan_facts(self.service, config_file, budget=budget, thresholds=self.thresholds)
        recorder = CallRecorder(self.service.root, "planner")
        system, advice = advise_prompt(PLAN_SYSTEM, "planner", self.service.root)
        recorder.advice = sorted([*advice, *self.threshold_advice])
        rule_plan = plan_rules(facts, goal, self.thresholds)
        thresholds = self.thresholds.model_dump(mode="json")
        base_task = (
            f"Goal: {goal}\n\nFacts of the case:\n{facts.model_dump_json(indent=1)}\n\n"
            f"The rules' plan, for reference:\n{rule_plan.model_dump_json(indent=1)}\n\n"
            "Propose the study."
        )
        revisions: list[str] = []
        decision: Decision[OptimizationStudyProposal] | None = None
        attempts = 1 + self.thresholds.max_revisions if self.provider is not None else 1
        for attempt in range(attempts):
            task = base_task if not revisions else (
                base_task + f"\n\nYour previous proposal was refused:\n{revisions[-1]}\n"
                "Correct it and propose again."
            )  # fmt: skip
            decision = decide(
                self.service, self.provider, recorder, task=task,
                response_model=OptimizationStudyProposal, rules=rule_plan,
                rule="the optimization planner's rules", thresholds=thresholds,
                client="planner", settings=self.settings, system=system,
            )  # fmt: skip
            if decision.decided_by != "model":
                break
            spec, validation, cost, errors = self.check(facts, decision.value)
            if not errors and spec is not None and validation is not None:
                return self._result(facts, decision, spec, validation, cost, revisions)
            revisions.append("; ".join(errors))
            recorder.record_action(
                f"refused proposal {attempt + 1}: " + revisions[-1], proposal=decision.value
            )
        assert decision is not None
        if decision.decided_by == "model":  # every revision refused: the rules plan
            reason = f"{len(revisions)} proposal(s) refused; the rules planned"
            trace = recorder.record_decision(
                self.provider, task=base_task, decision=rule_plan, fell_back=True,
                reason=reason, rule="the optimization planner's rules",
                thresholds=thresholds, tool_calls=[],
            )  # fmt: skip
            decision = Decision(rule_plan, "fallback", reason, trace)
        if decision.decided_by != "model":
            decision = Decision(self._fit(facts, decision.value), decision.decided_by,
                                decision.reason, decision.trace)  # fmt: skip
        spec, validation, cost, errors = self.check(facts, decision.value)
        if spec is None or validation is None:
            raise ConfigError("the rules' plan is not a valid study: " + "; ".join(errors))
        return self._result(facts, decision, spec, validation, cost, revisions)

    def plan_given(
        self,
        config_file: Path,
        goal: str,
        given: dict[str, Any],
        *,
        budget: int | None = None,
    ) -> OptimizationPlanResult:
        """Plan with a design the caller gives (the web UI's assistant).

        ``given`` holds any of the proposal's fields (``methods``, ``variants``,
        ``replicates``, ``metrics``, ``name``); the rules' plan fills the rest. The design is
        held to the facts as a model's would be, but never cut or replaced: a refused design
        raises, so its author can revise it, rather than silently becoming another study.
        Without ``replicates``, as many as the budget allows for this design (its methods,
        and its variants or else the rules' one "base"), each count costed in full.

        Raises
        ------
        ConfigError
            If the design is not a valid study within the budget.
        """
        facts = plan_facts(self.service, config_file, budget=budget, thresholds=self.thresholds)
        recorder = CallRecorder(self.service.root, "planner")
        rule_plan = plan_rules(facts, goal, self.thresholds)
        try:
            proposal = OptimizationStudyProposal.model_validate(
                {
                    **rule_plan.model_dump(),
                    **given,
                    "reasoning": "the design given by the web UI's assistant",
                }
            )
        except ValidationError as error:
            raise ConfigError(f"not a study of optimizations: {error}") from error
        # Before the replicates are sized: a matched population changes the cost.
        proposal, matched = match_arms(proposal, max(len(facts.objective_names), 1))
        if "replicates" not in given:
            # The rules sized replicates for their own methods; size them for this design,
            # as many as the budget allows (each method may cost differently, so cost them).
            groups = max(len(proposal.methods), 1) * len(proposal.variants)
            cells = facts.max_cells if facts.max_cells is not None else groups
            replicates = max(1, min(self.thresholds.max_replicates, cells // groups))
            proposal = proposal.model_copy(update={"replicates": replicates})
            while proposal.replicates > 1:
                _, _, cost, _ = self.check(facts, proposal)
                if cost is None or cost <= facts.budget_simulator_runs:
                    break
                proposal = proposal.model_copy(update={"replicates": proposal.replicates - 1})
        spec, validation, cost, errors = self.check(facts, proposal)
        if cost is not None and cost > facts.budget_simulator_runs:
            errors.append(f"it costs {cost} simulator runs, over the budget of "
                          f"{facts.budget_simulator_runs}")  # fmt: skip
            if (note := self.least_costs_note(facts, proposal)) is not None:
                errors.append(note)
        if errors or spec is None or validation is None:
            raise ConfigError("; ".join(dict.fromkeys(errors)) or "not a valid study")
        trace = recorder.record_decision(
            None, task=f"Goal: {goal}\nGiven: {json.dumps(given, default=str)[:2000]}",
            decision=proposal, fell_back=False, reason="the caller's design, checked",
            rule="the optimization planner's checks",
            thresholds=self.thresholds.model_dump(mode="json"), tool_calls=[],
        )  # fmt: skip
        return OptimizationPlanResult(facts, proposal, spec, validation, cost, True, "caller",
                                      "the caller's design, checked", trace,
                                      matched=tuple(matched))  # fmt: skip

    def least_costs(
        self, facts: OptimizationPlanFacts, proposal: OptimizationStudyProposal
    ) -> dict[str, LeastCost]:
        """Return each method's least cost: one seed of the case as configured.

        The options the optimizer has among :data:`CHEAPEST_OPTIONS` take their least
        values; a population's size is kept as proposed and reported.
        """
        optimizer = facts.optimizer
        methods = proposal.methods or {str(optimizer["name"]): OptimizationMethod(
            name=str(optimizer["name"]), options=dict(optimizer.get("options") or {}),
        )}  # fmt: skip
        base = OptimizationVariantProposal(name="base", patch={},
                                           rationale="the case as configured")  # fmt: skip
        n_obj = max(len(facts.objective_names), 1)
        costs: dict[str, LeastCost] = {}
        for label, method in methods.items():
            fields = _option_fields(method.name) or {}
            applied = {k: v for k, v in CHEAPEST_OPTIONS.items() if k in fields}
            options = {**method.options, **applied}
            arm = method.model_copy(update={"options": options})
            alone = proposal.model_copy(
                update={"methods": {label: arm}, "variants": [base], "replicates": 1}
            )
            _, _, cost, errors = self.check(facts, alone)
            # The budget's own refusal is not a fault of the design.
            faults = [e for e in errors if "budget" not in e]
            costs[label] = LeastCost(
                cost=cost,
                applied=applied,
                population=population(method.name, options, n_obj),
                error="; ".join(faults) if cost is None and faults else None,
            )
        return costs

    def least_costs_note(
        self, facts: OptimizationPlanFacts, proposal: OptimizationStudyProposal
    ) -> str | None:
        """Say what the cheapest design of these methods costs, against the budget (#242).

        ``None`` when it cannot be told: the refusal is then the budget's alone.
        """
        try:
            costs = self.least_costs(facts, proposal)
        except (ApexError, ValueError, TypeError) as error:
            logger.warning("least_costs_failed", extra={"error": str(error)})
            return None
        parts = []
        for label, least in costs.items():
            settings = ", ".join(f"{k} = {v}" for k, v in least.applied.items()) or "as given"
            if least.error is not None:
                parts.append(f"{label}: not costed ({least.error})")
                continue
            kept = (f", population {least.population} kept as proposed"
                    if least.population is not None else "")  # fmt: skip
            shown = least.cost if least.cost is not None else "not bounded"
            parts.append(f"{label} {shown} ({settings}{kept})")
        note = "computed by APEX, each method's least cost for one seed: " + "; ".join(parts)
        exact = all(c.cost is not None and c.error is None and c.population is None
                    for c in costs.values())  # fmt: skip
        least_total = sum(c.cost or 0 for c in costs.values())
        if exact and least_total > facts.budget_simulator_runs:
            note += (f"; together at least {least_total}, so no design of these methods fits "
                     f"the budget of {facts.budget_simulator_runs}: tell the user, and ask "
                     "for a larger budget or fewer methods")  # fmt: skip
        return note

    def _fit(
        self, facts: OptimizationPlanFacts, proposal: OptimizationStudyProposal
    ) -> OptimizationStudyProposal:
        """Cut the rules' plan to the budget: fewer replicates, then the case's optimizer.

        The rules size replicates by the case's own cost; another optimizer may cost more
        (its own bounds), so the plan is costed and cut until it fits.
        """
        while True:
            _, _, cost, _ = self.check(facts, proposal)
            if cost is None or cost <= facts.budget_simulator_runs:
                return proposal
            if proposal.replicates > 1:
                proposal = proposal.model_copy(update={"replicates": proposal.replicates - 1})
            elif len(proposal.methods) > 1:
                own = facts.optimizer["name"]
                kept = {k: v for k, v in proposal.methods.items() if k == own}
                proposal = proposal.model_copy(update={
                    "methods": kept,
                    "reasoning": proposal.reasoning + "; cut to the case's optimizer by the "
                    "budget",
                })  # fmt: skip
            else:
                return proposal  # the case alone, once: over budget, flagged

    def _result(
        self,
        facts: OptimizationPlanFacts,
        decision: Decision[OptimizationStudyProposal],
        spec: OptimizationStudySpec,
        validation: Validation,
        cost: int | None,
        revisions: list[str],
    ) -> OptimizationPlanResult:
        within = cost is not None and cost <= facts.budget_simulator_runs and validation.valid
        return OptimizationPlanResult(
            facts, decision.value, spec, validation, cost, within, decision.decided_by,
            decision.reason, decision.trace, tuple(revisions),
        )  # fmt: skip


def is_optimization_config(path: Path) -> bool:
    """Whether ``path`` is an optimization configuration, not a history match's.

    It has ``[optimization]`` and ``[optimizer]``.
    """
    try:
        document = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return False
    return "optimization" in document and "optimizer" in document


__all__ = [
    "OptimizationDiagnosis",
    "OptimizationDiagnostician",
    "OptimizationPlanResult",
    "OptimizationPlanner",
    "diagnosis_rules",
    "is_optimization_config",
    "plan_rules",
]


def optimizers_that_fit(
    n_objectives: int, allowed: Callable[[str], bool] | None = None
) -> list[str]:
    """Return the registered optimizers that can run a case with ``n_objectives``.

    Several objectives: the multi-objective ones (``nsga2``, ``nsga3``, ``moead``); one: the
    others. Validation refuses any other pairing. ``allowed`` keeps only a
    web-UI user's optimizers (their list).
    """
    from apex.algorithms.optimization import optimizers

    several = n_objectives > 1
    fit = []
    for name in optimizers.names():
        try:
            multi = bool(getattr(optimizers.factory(name), "multi_objective", False))
        except ApexError:  # a broken plugin is not offered
            continue
        if multi == several and (allowed is None or allowed(name)):
            fit.append(name)
    return sorted(fit)


#: How each kind of optimizer's simulator runs are bounded (for the assistant).
BOUNDS_NOTE = (
    "Each optimization's simulator runs are bounded by its options (validate shows the "
    "bound): a population optimizer (ga, nsga2) runs pop_size x realizations at the start "
    "and n_offsprings (default pop_size) x realizations per generation, pso pop_size x "
    "realizations per generation, for maxiter generations; nsga3's and moead's population, "
    "and moead's children per generation, are their reference directions' count "
    "(n_partitions, inner_partitions); an ensemble optimizer (enopt, genopt, smcopt) runs "
    "about ne x realizations per iteration; line_search and trust_region a few per "
    "iteration; ropt needs max_batches. Set these so the study fits the budget"
)


def optimization_case_info(
    service: ApexService,
    path: Path,
    allowed: Callable[[str], bool] | None = None,
    shown: Callable[[Any], Any] = lambda value: value,
) -> dict[str, Any]:
    """Return what an optimization case is: optimizer, objectives, controls, plan, and fit.

    Which optimizers fit it, and how runs are bounded (a reported issue: case_info read only
    history-matching cases). ``allowed`` keeps a web-UI user's optimizers; ``shown`` makes
    the sections' resolved (absolute) file options presentable (no server paths).
    """
    from apex.services.optimization import load_optimization_config, plan_optimization

    config = load_optimization_config(path)
    names = _objective_names(config)
    n_objectives = max(len(names), 1)
    info: dict[str, Any] = {
        "kind": "optimization",
        "optimizer": shown(config.optimizer.model_dump(mode="json", exclude_none=True)),
        "objectives": shown(
            [entry.model_dump(mode="json", exclude_none=True) for entry in config.objectives]
            if config.objectives
            else [config.objective.model_dump(mode="json", exclude_none=True)]
            if config.objective is not None else []
        ),
        "objective_names": names or [config.objective_label],
        "several_objectives": n_objectives > 1,
        "controls": [
            {"name": v.name, "values": 1 if isinstance(v.initial, float | int) else len(v.initial),
             "lower": v.lower, "upper": v.upper}
            for v in config.controls.variables
        ],
        "simulator": None if config.simulator is None else config.simulator.name,
        "optimizers_that_fit": optimizers_that_fit(n_objectives, allowed),
        "bounds": BOUNDS_NOTE,
    }  # fmt: skip
    try:
        plan = plan_optimization(config)
        info["plan"] = {
            "realizations": plan.n_realizations,
            "max_evaluations": plan.max_evaluations,
            "max_simulator_runs": plan.max_simulator_runs,
            "warnings": shown(plan.warnings),
        }
    except ConfigError as error:
        info["plan_error"] = shown(str(error))
    if n_objectives == 1:
        info["note"] = (
            "one objective: compare single-objective optimizers; a multi-objective one "
            "(nsga2, nsga3, moead) needs a second objective in [[objectives]] (ask the user "
            "which, e.g. the water injected: name = 'field_total' with options vector = "
            "'FWIT' and time; or the objective's spread over realizations: the same entry "
            "again with statistic = 'std')"
        )
    return info


def fit_hint(service: ApexService, path: Path, allowed: Callable[[str], bool] | None = None) -> str:
    """Return which optimizers fit a case, and how runs are bounded: for a refused plan."""
    try:
        from apex.services.optimization import load_optimization_config

        names = _objective_names(load_optimization_config(path))
    except ApexError:
        return BOUNDS_NOTE
    n = max(len(names), 1)
    fit = ", ".join(optimizers_that_fit(n, allowed)) or "none of those available to you"
    return (f"The case has {n} objective{'s' if n > 1 else ''}; optimizers that fit it: "
            f"{fit}. {BOUNDS_NOTE}.")  # fmt: skip

"""The run monitor and the report agent for optimizations (the pattern).

Code computes an optimization's facts from its directory (:func:`optimization_facts`) and
applies named threshold rules (``[agents.monitor] optimization_*``); a model, if
given, judges the facts with read-only tools, and its report is held to them: a finding
with a rule's code the rules did not find is dropped, and a recommendation to stop becomes
``investigate`` (this monitor never acts). The rules are the fallback. The report agent
writes Markdown from the same facts; its key numbers must match them
(:mod:`apex.agents.reporter`).

A multi-objective run (``nsga2``) is judged by its front, not by its
representative's first objective: whether any member is feasible, the front's size against
the population, and whether the front still moves (the optimizer's own measure, logged per
generation as ``front_movement`` and ``stalled_generations``). Its report describes the
front: its size, the best member of each objective and the representative.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from apex.agents.common import decide
from apex.agents.loop import LoopSettings
from apex.agents.monitor import Finding, MonitorReport
from apex.agents.reporter import NarrativeReport, ReportResult, Section
from apex.agents.trace import CallRecorder
from apex.core.advice import advise_prompt, advise_thresholds
from apex.core.errors import ConfigError, PluginError
from apex.core.llm import LLMProvider
from apex.domain.services import OptimizationSummary
from apex.services.api import ApexService
from apex.services.settings import MonitorSection
from apex.storage import optimization_directory as layout

#: The monitor's records and advice are its own: advice for the history-matching monitor
#: (overfitting, coverage, ...) does not apply to optimizations.
AGENT = "optimization_monitor"
#: The tools the model may use: reading only.
TOOLS = ("optimization_summary", "job_status", "job_events")
#: The codes the rules give (a model's finding with one of these must be found by them).
RULE_CODES = frozenset({
    "failed_evaluations", "stall", "no_improvement", "iteration_limit",
    "no_feasible_member", "small_front", "front_stall", "hypervolume_drop",
})  # fmt: skip


class OptimizationFacts(BaseModel):
    """What the agents know of an optimization, computed from its directory."""

    model_config = ConfigDict(extra="forbid")

    path: str
    name: str
    job_id: str | None = None
    state: Literal["new", "running", "stopped", "finished"]
    optimizer: str
    objective: str
    accepted_steps: int
    start_objective: float | None
    best_objective: float | None
    last_objective: float | None
    relative_improvement: float | None = Field(
        description="(start - last) / |start|: the share of the start's objective gained "
        "(with EPF, values include the penalty, under different factors)."
    )
    window_improvement: float | None = Field(
        description="Relative improvement over the last stall-window accepted steps; none "
        "for a multi-objective run, whose first objective may rightly worsen as "
        "its front spreads (its own stall watches the front's movement)."
    )
    stall_window: int = Field(
        default=3,
        description="Accepted steps the stall rule looks back over (for a population "
        "optimizer at least its own stall: every generation is an accepted step).",
    )
    evaluations: int = Field(description="Simulated evaluations in the journal (0: analytic).")
    failed_evaluations: int
    message: str | None = Field(description="Why it stopped (the optimizer's message).")
    # A multi-objective run's front; empty or none otherwise.
    objective_names: list[str] = Field(
        default_factory=list, description="The objectives (several: a multi-objective run)."
    )
    pop_size: int | None = Field(default=None, description="The population's size.")
    front_size: int | None = Field(default=None, description="Members of the current front.")
    feasible_share: float | None = Field(
        default=None, description="Share of the population that is feasible (constrained runs)."
    )
    infeasible_generations: int = Field(
        default=0, description="The latest generations in a row with no feasible member."
    )
    hypervolume: float | None = Field(
        default=None, description="The front's hypervolume against the run's reference point."
    )
    best_hypervolume: float | None = Field(
        default=None, description="The largest hypervolume under the current penalty factor."
    )
    front_movement: float | None = Field(
        default=None,
        description="How far the front moved in the last generation "
        "(normalised; the optimizer's stall compares it with ftol).",
    )
    stalled_generations: int | None = Field(
        default=None,
        description="Generations in a row without progress, as the optimizer "
        "counts them (it stops at its stall).",
    )
    front_stall_window: int | None = Field(
        default=None,
        description="Stalled generations at which the front_stall rule speaks: the "
        "optimizer's default stall, when the run's own is larger (it would keep a stalled run "
        "going); none otherwise, as the run then stops by itself.",
    )


@dataclass(frozen=True)
class FrontSettings:
    """A multi-objective run's settings the rules need.

    Attributes
    ----------
    objective_names
        The objectives, from the configuration (the front may not be written yet).
    pop_size
        The population.
    stall_window
        Stalled generations at which ``front_stall`` speaks, or ``None``: the optimizer's
        default stall when the run's own is larger (a run stops by itself at its stall, so
        below that the rule has nothing to add).
    """

    objective_names: tuple[str, ...]
    pop_size: int | None
    stall_window: int | None


def option_default(factory: Any, option: str) -> Any:  # noqa: ANN401 (a plugin's option)
    """Return a plugin's default for ``option``, or ``None`` if it has none."""
    import inspect

    parameter = inspect.signature(factory).parameters.get(option)
    if parameter is None or parameter.default is inspect.Parameter.empty:
        return None
    return parameter.default


def front_settings(path: Path) -> FrontSettings | None:
    """Return a multi-objective optimizer's settings from its configuration, else ``None``."""
    from apex.algorithms.optimization import optimizers

    try:
        config = layout.read_config(path)
    except (OSError, ValueError):
        return None
    section = config.get("optimizer") or {}
    name = str(section.get("name"))
    if name not in optimizers.names():
        return None
    factory = optimizers.factory(name)
    if not getattr(factory, "multi_objective", False):
        return None
    options = section.get("options") or {}
    names = _objective_names(config)
    try:
        default_stall = option_default(factory, "stall")
        stall = options.get("stall", default_stall)
        window = (int(default_stall) if default_stall is not None and stall is not None
                  and int(stall) > int(default_stall) else None)  # fmt: skip
        # The population as the optimizer sizes it: NSGA-III's and MOEA/D's follow their
        # reference directions, so it is built and told the objectives.
        optimizer = factory(**options)
        set_objectives = getattr(optimizer, "set_objectives", None)
        if callable(set_objectives) and names:
            set_objectives(len(names))
        size = getattr(optimizer, "pop_size", None)
        return FrontSettings(names, None if size is None else int(size), window)
    except (TypeError, ValueError):
        return FrontSettings(names, None, None)  # malformed options: the run refuses them


def _objective_names(config: dict[str, Any]) -> tuple[str, ...]:
    """Return a stored configuration's objectives' names, as the run shows them."""
    from apex.algorithms.optimization import objectives
    from apex.domain.optimization import ObjectiveEntry

    entries = config.get("objectives") or []
    try:
        if entries:
            return tuple(ObjectiveEntry.model_validate(e).shown for e in entries)
        single = config.get("objective") or {}
        plugin = objectives.create(str(single.get("name")), **(single.get("options") or {}))
        return tuple(str(n) for n in getattr(plugin, "objective_names", ()))
    except (PluginError, ValueError, TypeError, KeyError):  # unknown plugin or options
        return ()


def _trailing(rows: list[dict[str, Any]], test: Any) -> int:  # noqa: ANN401 (a predicate)
    count = 0
    for row in reversed(rows):
        if not test(row):
            break
        count += 1
    return count


def optimization_facts(
    summary: OptimizationSummary,
    window: int,
    job_id: str | None = None,
    front: FrontSettings | None = None,
) -> OptimizationFacts:
    """Return the facts of an optimization from its summary (``front``: several objectives)."""
    rows = [r for r in summary.history if "objective" in r and r.get("kind") != "epf_update"]
    objectives = [float(r["objective"]) for r in rows]
    steps = sum(1 for r in rows if r.get("kind") == "step")
    start = objectives[0] if objectives else None
    last = objectives[-1] if objectives else None

    def relative(before: float | None, after: float | None) -> float | None:
        if before is None or after is None or before == 0 or not math.isfinite(before):
            return None
        return (before - after) / abs(before)

    # The window compares accepted steps under one penalty factor: an EPF update doubles r,
    # so values across it are not comparable (the numerics review of 2026-10-03).
    def penalty(row: dict[str, Any]) -> int:  # the start row is logged before EPF's 1
        return max(1, int(row.get("epf_iteration") or 0))

    current = penalty(rows[-1]) if rows else None
    same_penalty = [float(r["objective"]) for r in rows if penalty(r) == current]
    window_start = same_penalty[-1 - window] if len(same_penalty) > window else None
    window_last = same_penalty[-1] if same_penalty else None
    if any("hypervolume" in r for r in rows):
        # The representative's first objective is no measure of a front's progress, nor is
        # a hypervolume against a fixed reference point: the front's rules judge it.
        window_start = window_last = None
    multi = _front_facts(summary, rows, front, current, penalty)
    return OptimizationFacts(
        path=summary.path,
        name=summary.name,
        job_id=job_id,
        state=summary.state,
        optimizer=summary.optimizer,
        objective=summary.objective,
        accepted_steps=steps,
        start_objective=start,
        best_objective=min(objectives) if objectives else None,
        last_objective=last,
        relative_improvement=relative(start, last),
        window_improvement=relative(window_start, window_last),
        stall_window=window,
        evaluations=summary.evaluations,
        failed_evaluations=summary.failed_evaluations,
        message=None if summary.result is None else str(summary.result.get("message")),
        **multi,
    )


def _front_facts(
    summary: OptimizationSummary,
    rows: list[dict[str, Any]],
    front: FrontSettings | None,
    current: int | None,
    penalty: Any,  # noqa: ANN401 (the row's penalty factor index)
) -> dict[str, Any]:
    """Return the facts of a multi-objective run's front, or nothing for one objective."""
    pareto = summary.pareto or {}
    names = [str(n) for n in pareto.get("objective_names") or []]
    if not names and front is not None:
        names = list(front.objective_names)  # before the first front is written
    if front is None and not names:
        return {}
    last = rows[-1] if rows else {}
    # Between penalty factors (EPF), the last generation's stall count belongs to the factor
    # that just ended (its inner loop stopped there): no stall for the run going on.
    between = bool(summary.history) and summary.history[-1].get("kind") == "epf_update"
    hv_rows = [float(r["hypervolume"]) for r in rows
               if "hypervolume" in r and penalty(r) == current]  # fmt: skip
    stalled = last.get("stalled_generations")
    return {
        "objective_names": names,
        "pop_size": None if front is None else front.pop_size,
        "front_size": int(last["front_size"]) if "front_size" in last else None,
        "feasible_share": float(last["feasible_share"]) if "feasible_share" in last else None,
        "infeasible_generations": _trailing(
            [r for r in rows if r.get("kind") == "step"],  # generations, not the start
            lambda r: "feasible_share" in r and float(r["feasible_share"]) <= 0.0,
        ),
        "hypervolume": hv_rows[-1] if hv_rows else None,
        "best_hypervolume": max(hv_rows) if hv_rows else None,
        "front_movement": float(last["front_movement"]) if "front_movement" in last else None,
        "stalled_generations": None if stalled is None or between else int(stalled),
        "front_stall_window": None if front is None else front.stall_window,
    }


def optimization_rules(facts: OptimizationFacts, thresholds: MonitorSection) -> MonitorReport:
    """Return the rules' report on an optimization."""
    findings: list[Finding] = []
    if facts.failed_evaluations:
        findings.append(Finding(
            code="failed_evaluations", severity="critical",
            evidence=f"{facts.failed_evaluations} simulated evaluations failed; the run stops "
            "at a failure and resumes once the cause is fixed",
        ))  # fmt: skip
    gain = facts.window_improvement
    if gain is not None and gain < thresholds.optimization_stall_tolerance:
        findings.append(Finding(
            code="stall", severity="warning",
            evidence=f"the objective improved by {gain:.3g} (relative) over the last "
            f"{facts.stall_window} accepted steps, below "
            f"{thresholds.optimization_stall_tolerance:g}",
        ))  # fmt: skip
    findings.extend(_front_rules(facts, thresholds))
    if facts.state == "finished" and facts.accepted_steps == 0:
        findings.append(Finding(
            code="no_improvement", severity="warning",
            evidence="the optimization ended without an accepted step: "
            f"{facts.message or 'no message'}",
        ))  # fmt: skip
    if facts.state == "finished" and (facts.message or "").startswith("Maximum number"):
        findings.append(Finding(
            code="iteration_limit", severity="info",
            evidence=f"it stopped at maxiter after {facts.accepted_steps} accepted steps, "
            "not by a convergence test: more iterations may still gain",
        ))  # fmt: skip
    severities = {f.severity for f in findings}
    status: Literal["healthy", "warning", "critical"] = (
        "critical" if "critical" in severities else "warning" if "warning" in severities
        else "healthy"
    )  # fmt: skip
    running = facts.state == "running"
    recommendation: Literal["continue", "stop", "investigate", "none"] = (
        "investigate" if severities - {"info"} else "continue" if running else "none"
    )
    trend = (
        "no objective yet"
        if facts.start_objective is None
        else f"objective {facts.start_objective:.6g} -> {facts.last_objective:.6g} over "
        f"{facts.accepted_steps} accepted steps, {facts.evaluations} simulated evaluations"
    )
    if facts.objective_names:
        hv = "n/a" if facts.hypervolume is None else f"{facts.hypervolume:.6g}"
        trend = (
            f"a front of {facts.front_size if facts.front_size is not None else 'no'} members "
            f"over {', '.join(facts.objective_names)} after {facts.accepted_steps} "
            f"generations (hypervolume {hv}), {facts.evaluations} simulated evaluations"
        )
    noticed = "; ".join(sorted(f.code for f in findings)) or "nothing unusual"
    return MonitorReport(
        status=status,
        recommendation=recommendation,
        findings=findings,
        summary=f"{facts.optimizer} optimization {facts.name} is {facts.state}: {trend}. "
        f"Noticed: {noticed}.",
    )


def _front_rules(facts: OptimizationFacts, thresholds: MonitorSection) -> list[Finding]:
    """Return the findings on a multi-objective run's front; none otherwise."""
    if not facts.objective_names:
        return []
    findings: list[Finding] = []
    if facts.infeasible_generations >= thresholds.optimization_infeasible_generations:
        findings.append(Finding(
            code="no_feasible_member", severity="warning",
            evidence=f"no member has been feasible for {facts.infeasible_generations} "
            "generations: the front is the least infeasible member only",
        ))  # fmt: skip
    share = thresholds.optimization_min_front_share
    # Only feasible members can be on the front: compare it with them, not the population.
    feasible = (None if facts.pop_size is None else facts.pop_size * (
        1.0 if facts.feasible_share is None else facts.feasible_share))  # fmt: skip
    if (feasible and facts.front_size is not None and facts.infeasible_generations == 0
            and facts.accepted_steps >= thresholds.optimization_min_front_generations
            and facts.front_size < share * feasible):  # fmt: skip
        findings.append(Finding(
            code="small_front", severity="warning",
            evidence=f"the front holds {facts.front_size} of {feasible:g} feasible members "
            f"after {facts.accepted_steps} generations, under {share:g} of them: the "
            "population may have lost its diversity, or the objectives hardly conflict (then "
            "a small front is the answer)",
        ))  # fmt: skip
    if (facts.state == "running" and facts.stalled_generations is not None
            and facts.front_stall_window is not None and facts.infeasible_generations == 0
            and facts.stalled_generations >= facts.front_stall_window):  # fmt: skip
        findings.append(Finding(
            code="front_stall", severity="warning",
            evidence=f"the front has not moved for {facts.stalled_generations} generations "
            "(beyond its tolerance) and the run goes on",
        ))  # fmt: skip
    # Crowding truncation drops front members once the front outgrows the population, so
    # small dips are normal; a large one means the front lost ground.
    drop = thresholds.optimization_hypervolume_drop
    if (facts.hypervolume is not None and facts.best_hypervolume
            and facts.hypervolume < (1.0 - drop) * facts.best_hypervolume):  # fmt: skip
        findings.append(Finding(
            code="hypervolume_drop", severity="info",
            evidence=f"the hypervolume fell from {facts.best_hypervolume:.6g} to "
            f"{facts.hypervolume:.6g} (over {drop:.0%}) under one penalty factor: "
            "the front lost ground, e.g. members cut by crowding or a noisy objective",
        ))  # fmt: skip
    return findings


def reconcile(model: MonitorReport, rules: MonitorReport) -> tuple[MonitorReport, list[str]]:
    """Hold the model's report to the facts (see the module docstring)."""
    found = {f.code for f in rules.findings}
    kept = [f for f in model.findings if f.code not in RULE_CODES or f.code in found]
    changes = [f"dropped {f.code}: the rules did not find it" for f in model.findings
               if f not in kept]  # fmt: skip
    recommendation = model.recommendation
    if recommendation == "stop":
        recommendation = "investigate"
        changes.append("stop became investigate: the optimization monitor does not act")
    if not changes:
        return model, []
    return model.model_copy(update={"findings": kept, "recommendation": recommendation}), changes


SYSTEM = (
    "You monitor an APEX optimization (an ensemble optimizer such as EnOpt minimising an "
    "objective, e.g. -NPV; or NSGA-II finding a Pareto front of several objectives, judged "
    "by its front: feasibility, size and movement). You get facts computed from its "
    "directory and the report of "
    "threshold rules. Decide its status. The facts are complete: answer from them, cite "
    "numbers as evidence, and keep every finding of the rules unless the facts contradict "
    "it. Recommend continue, investigate or none (this monitor never stops a run)."
)
_RULE = "the optimization monitor's threshold rules"


@dataclass(frozen=True)
class OptimizationCheck:
    """One check of an optimization (see :class:`OptimizationMonitor`)."""

    facts: OptimizationFacts
    rules_report: MonitorReport
    report: MonitorReport
    decided_by: str
    reason: str | None
    trace: Path
    corrections: tuple[str, ...] = ()


def stall_window(path: Path, window: int) -> int:
    """Return the stall rule's window: ``window``, or at least a population optimizer's stall.

    Every generation of ``ga`` and ``pso`` is an accepted step, and their best routinely
    stays put for several generations while the population explores; a window of a few steps
    would call a healthy run stalled (the review of 2026-10-06).
    """
    import inspect

    from apex.algorithms.optimization import optimizers
    from apex.algorithms.optimization.population import is_population_optimizer

    try:
        section = layout.read_config(path).get("optimizer") or {}
    except (OSError, ValueError):
        return window
    name = str(section.get("name"))
    if not is_population_optimizer(name):
        return window
    factory = optimizers.factory(name)
    options = section.get("options") or {}
    default = inspect.signature(factory).parameters["stall"].default
    return max(window, int(options.get("stall", default)))


def is_optimization_dir(path: Path) -> bool:
    """Return whether ``path`` holds an optimization run (not a history-matching run)."""
    if not (path / layout.CONFIG).exists() or not (path / layout.HISTORY).exists():
        return False
    try:
        return "optimizer" in layout.read_config(path)
    except (OSError, ValueError):
        return False


class OptimizationMonitor:
    """Checks optimizations (see the module docstring).

    Parameters
    ----------
    service
        The services.
    provider
        The model; ``None`` for rules only.
    thresholds
        The rules' thresholds (default: ``[agents.monitor]``, with the installed advice).
    settings
        Bounds of the model's task.
    """

    def __init__(
        self,
        service: ApexService,
        provider: LLMProvider | None,
        *,
        thresholds: MonitorSection | None = None,
        settings: LoopSettings | None = None,
    ) -> None:
        self.service = service
        self.provider = provider
        if thresholds is None:
            thresholds, _ = advise_thresholds(
                "agents.monitor", service.jobs.settings.agents.monitor, service.root
            )
        self.thresholds = thresholds
        self.settings = settings or LoopSettings(max_turns=6)

    def resolve(self, target: str | Path) -> tuple[Path, str | None]:
        """Return the optimization directory and its job, from a directory or a job id.

        Raises
        ------
        ConfigError
            If ``target`` names no optimization.
        """
        path = Path(target)
        records = self.service.jobs.all_records()
        if is_optimization_dir(path):
            resolved = path.resolve()
            for record in reversed(records):
                dirs = [Path(d).resolve() for d in record.get("run_dirs") or []]
                if record["kind"] in ("optimization", "optimization_resume") and resolved in dirs:
                    return resolved, str(record["id"])
            return resolved, None
        try:
            record = self.service.jobs.record(str(target))
        except KeyError:
            raise ConfigError(f"{target!r} is not an optimization directory or job") from None
        if record["kind"] not in ("optimization", "optimization_resume"):
            raise ConfigError(f"job {target} is not an optimization")
        return Path(record["run_dirs"][0]), str(target)

    def facts(self, target: str | Path) -> OptimizationFacts:
        """Return the facts of an optimization."""
        path, job = self.resolve(target)
        summary = self.service.optimization_summary(path)
        window = stall_window(path, self.thresholds.optimization_stall_window)
        return optimization_facts(summary, window, job, front_settings(path))

    def check(self, target: str | Path) -> OptimizationCheck:
        """Check an optimization once: facts, rules, and the model's judgement."""
        facts = self.facts(target)
        rules_report = optimization_rules(facts, self.thresholds)
        recorder = CallRecorder(Path(facts.path), AGENT)
        system, recorder.advice = advise_prompt(SYSTEM, AGENT, self.service.root)
        decision = decide(
            self.service,
            self.provider,
            recorder,
            task=f"Facts of the optimization:\n{facts.model_dump_json(indent=1)}\n\n"
            f"The threshold rules' report:\n{rules_report.model_dump_json(indent=1)}\n\n"
            "Give your report on this optimization.",
            response_model=MonitorReport,
            rules=rules_report,
            rule=_RULE,
            thresholds=self.thresholds.model_dump(mode="json"),
            client="monitor",
            settings=self.settings,
            tools=TOOLS,
            system=system,
        )
        report = decision.value
        corrections: list[str] = []
        if decision.decided_by == "model":
            report, corrections = reconcile(report, rules_report)
            if corrections:
                recorder.record_action(
                    "corrected the model's report: " + "; ".join(corrections),
                    corrections=corrections,
                )
        return OptimizationCheck(
            facts, rules_report, report, decision.decided_by, decision.reason, decision.trace,
            tuple(corrections),
        )  # fmt: skip


class FrontMember(BaseModel):
    """A member of a multi-objective run's front, as its report shows it."""

    model_config = ConfigDict(extra="forbid")

    roles: list[str] = Field(description="Why it is shown: best <objective>, representative.")
    objectives: dict[str, float] = Field(description="Its objectives (means), all minimised.")
    controls: dict[str, float]
    violation: float


class FrontReport(BaseModel):
    """A multi-objective run's front for its report: the extremes and the representative."""

    model_config = ConfigDict(extra="forbid")

    size: int
    hypervolume: float
    reference_point: list[float] | None
    members: list[FrontMember]


class OptimizationReportFacts(BaseModel):
    """What the report of an optimization is written from."""

    model_config = ConfigDict(extra="forbid")

    facts: OptimizationFacts
    monitor: MonitorReport
    controls: dict[str, float] = Field(description="The final controls, by name.")
    front: FrontReport | None = Field(
        default=None, description="A multi-objective run's front; none otherwise."
    )


def front_report(pareto: dict[str, Any] | None, controls: list[str]) -> FrontReport | None:
    """Return the front's extremes (the best member of each objective) and representative."""
    if not pareto or not pareto.get("objectives"):
        return None
    names = [str(n) for n in pareto["objective_names"]]
    values = [[float(v) for v in row] for row in pareto["objectives"]]
    roles: dict[int, list[str]] = {}
    for j, name in enumerate(names):
        best = min(range(len(values)), key=lambda i: values[i][j])
        roles.setdefault(best, []).append(f"best {name}")
    roles.setdefault(int(pareto["representative"]), []).append("representative")
    members = [
        FrontMember(
            roles=roles[i],
            objectives=dict(zip(names, values[i], strict=True)),
            controls={c: float(v) for c, v in zip(controls, pareto["x"][i], strict=False)},
            violation=float(pareto["violation"][i]),
        )
        for i in sorted(roles)
    ]
    reference = pareto.get("reference_point")
    return FrontReport(
        size=len(values),
        hypervolume=float(pareto.get("hypervolume") or 0.0),
        reference_point=None if reference is None else [float(v) for v in reference],
        members=members,
    )


def report_facts(monitor: OptimizationMonitor, target: str | Path) -> OptimizationReportFacts:
    """Return the facts of an optimization's report."""
    facts = monitor.facts(target)
    summary = monitor.service.optimization_summary(Path(facts.path))
    x = (summary.result or {}).get("x") or (summary.history[-1]["x"] if summary.history else [])
    return OptimizationReportFacts(
        facts=facts,
        monitor=optimization_rules(facts, monitor.thresholds),
        controls={name: float(v) for name, v in zip(summary.controls, x, strict=False)},
        front=front_report(summary.pareto, list(summary.controls)),
    )


def report_template(report: OptimizationReportFacts) -> NarrativeReport:
    """Return the template report of an optimization."""
    f = report.facts

    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.6g}"

    controls = "\n".join(f"| {k} | {v:.6g} |" for k, v in report.controls.items())
    if report.front is not None:
        return _front_template(report, fmt, controls)
    sections = [
        Section(
            heading="Result",
            text=f"{f.optimizer} minimised {f.objective} ({f.state}): the objective went "
            f"from {fmt(f.start_objective)} to {fmt(f.last_objective)} (best "
            f"{fmt(f.best_objective)}) in {f.accepted_steps} accepted steps, with "
            f"{f.evaluations} simulated evaluations"
            + (f" ({f.failed_evaluations} failed)" if f.failed_evaluations else "")
            + f". {f.message or ''}",
        ),
        Section(
            heading="Controls",
            text=f"| Control | Value |\n|---|---|\n{controls}"
            if controls
            else "No controls recorded yet.",
        ),
        Section(heading="Health", text=report.monitor.summary),
    ]
    return NarrativeReport(title=f"Optimization {f.name}", sections=sections)


def _front_template(
    report: OptimizationReportFacts, fmt: Callable[[float | None], str], controls: str
) -> NarrativeReport:
    """Return the template report of a multi-objective run: its front, not one objective."""
    f, front = report.facts, report.front
    assert front is not None
    names = list(front.members[0].objectives) if front.members else f.objective_names
    reference = (
        ""
        if front.reference_point is None
        else f" against the reference point ({', '.join(fmt(v) for v in front.reference_point)})"
    )
    from apex.algorithms.optimization.multiobjective import MAX_HYPERVOLUME_OBJECTIVES

    measured = (fmt(front.hypervolume) if len(names) <= MAX_HYPERVOLUME_OBJECTIVES
                else f"not computed (over {MAX_HYPERVOLUME_OBJECTIVES} objectives)")  # fmt: skip
    head = "| Member | " + " | ".join(names) + " | Controls |\n|---|" + "---|" * (len(names) + 1)
    rows = "\n".join(
        f"| {', '.join(m.roles)} | "
        + " | ".join(fmt(m.objectives[n]) for n in names)
        + " | "
        + ", ".join(f"{k} {v:.6g}" for k, v in m.controls.items())
        + " |"
        for m in front.members
    )
    sections = [
        Section(
            heading="Result",
            text=f"{f.optimizer} searched the trade-off between {', '.join(names)} (all "
            f"minimised; {f.state}): a Pareto front of {front.size} members after "
            f"{f.accepted_steps} generations, with hypervolume {measured}"
            f"{reference}, and {f.evaluations} simulated evaluations"
            + (f" ({f.failed_evaluations} failed)" if f.failed_evaluations else "")
            + f". {f.message or ''}",
        ),
        Section(
            heading="Front",
            text="The best member of each objective and the representative (the member "
            "closest to the ideal point in normalised objectives, the run's single answer); "
            f"pareto.json holds every member.\n\n{head}\n{rows}",
        ),
        Section(
            heading="Representative's controls",
            text=f"| Control | Value |\n|---|---|\n{controls}"
            if controls
            else "No controls recorded yet.",
        ),
        Section(heading="Health", text=report.monitor.summary),
    ]
    return NarrativeReport(title=f"Optimization {f.name}", sections=sections)


def optimization_report(
    service: ApexService, provider: LLMProvider | None, target: str | Path
) -> ReportResult:
    """Write the report of an optimization (a directory or job id)."""
    from apex.agents.reporter import Reporter

    monitor = OptimizationMonitor(service, None)
    facts = report_facts(monitor, target)
    recorder = CallRecorder(Path(facts.facts.path), "report")
    return Reporter(service, provider).write(facts, report_template(facts), recorder)

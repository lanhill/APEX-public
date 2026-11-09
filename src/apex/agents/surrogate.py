"""The run monitor and the report agent for surrogate builds.

Code computes a build's facts from its folder (:func:`surrogate_facts`: the design's
runs from its journal, the summary's gate, fit and timings, and the model's checks) and
applies named threshold rules (``[agents.monitor] surrogate_*``); a model, if given,
judges the facts with read-only tools, and its report is held to them as the
optimization monitor's is: a finding with a rule's code the rules did not find is
dropped, and a recommendation to stop becomes ``investigate`` (this monitor never acts:
a build's job is stopped by its owner). The rules are the fallback. The report agent
writes Markdown from the same facts; its key numbers must match them
(:mod:`apex.agents.reporter`).

What matters for a build is whether its model may be used: the gate on the held-out
runs, and the latest check on new runs (history matching refuses a model whose last
check failed). The rules say so, and how many design runs failed and how much faster
than a full run the model predicts.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from apex.agents.common import decide
from apex.agents.loop import LoopSettings
from apex.agents.monitor import Finding, MonitorReport
from apex.agents.reporter import NarrativeReport, ReportResult, Section
from apex.agents.trace import CallRecorder
from apex.core.advice import advise_prompt, advise_thresholds
from apex.core.errors import ConfigError
from apex.core.llm import LLMProvider
from apex.services.api import ApexService
from apex.services.settings import MonitorSection
from apex.storage.locks import in_use

#: The monitor's records and advice are its own (advice for runs does not apply to builds).
AGENT = "surrogate_monitor"
#: The tools the model may use: reading only.
TOOLS = ("surrogate_summary", "job_status", "job_events")
#: The codes the rules give (a model's finding with one of these must be found by them).
RULE_CODES = frozenset({
    "failed_runs", "gate_failed", "check_failed", "small_speedup", "stopped_pending",
})  # fmt: skip
#: The job kinds of a build.
BUILD_KINDS = ("surrogate_build", "surrogate_build_resume")


class DatatypeFact(BaseModel):
    """One datatype through the gate."""

    model_config = ConfigDict(extra="forbid")

    datatype: str
    measure: str
    error: float | None = Field(description="Its error in the measure's units; none: not finite.")
    tolerance: float
    passed: bool


class CheckFact(BaseModel):
    """The latest check of the model on new full runs."""

    model_config = ConfigDict(extra="forbid")

    id: str
    sequence: int
    against: str
    runs: int
    succeeded: int
    inside: int
    passed: bool
    failures: list[str] = Field(default_factory=list, description="What failed, one line each.")


class SurrogateFacts(BaseModel):
    """What the agents know of a surrogate build, computed from its folder."""

    model_config = ConfigDict(extra="forbid")

    path: str
    name: str
    job_id: str | None = None
    state: Literal["new", "queued", "running", "stopped", "failed", "complete"]
    plugin: str
    plan: str = Field(description="The design's plan: prior, lhs, file, run or an adaptive one.")
    planned_runs: int = Field(description="Training plus held-out runs the design asks for.")
    finished_runs: int = Field(description="Design runs finished so far, failed ones included.")
    failed_runs: int
    failed_share: float | None = Field(description="Failed over finished runs; none before any.")
    pending_runs: int = Field(
        description="Design runs not run yet (an adaptive design's undrawn ones included)."
    )
    coarse_runs: int | None = Field(
        default=None, description="Runs on the coarse stack so far; none without one."
    )
    coarse_pending: int = Field(
        default=0, description="Coarse runs left when a build stopped in its coarse stage."
    )
    gate_passed: bool | None = Field(description="The gate's verdict; none before the fit.")
    gate: list[DatatypeFact] = Field(default_factory=list)
    gate_failures: list[str] = Field(default_factory=list)
    rank_correlation: float | None = Field(
        default=None, description="The held-out objectives' Spearman rho (optimization cases)."
    )
    fit_seconds: float | None = None
    prediction_seconds: float | None = Field(
        default=None, description="One member's prediction, averaged over the held-out runs."
    )
    simulator_median_seconds: float | None = None
    member_simulation: Literal["none", "coarse", "early"] = Field(
        default="none",
        description="What the model also simulates per member: a coarse run (coarse-corrected) "
        "or an early run (early-start); none.",
    )
    member_seconds: float | None = Field(
        default=None,
        description="One member's whole prediction (with its coarse run's median time); none "
        "when unknown (an early-start model's early run is not timed).",
    )
    speedup: float | None = Field(
        default=None,
        description="The median full run over member_seconds; none if unknown.",
    )
    model: str | None = Field(default=None, description="The manifest, relative to the build.")
    checks: int = Field(default=0, description="Complete checks of the model.")
    last_check: CheckFact | None = None


def is_build_dir(path: Path) -> bool:
    """Return whether ``path`` holds a surrogate build (an unreadable state: no)."""
    from apex.storage.surrogate_directory import SurrogateDirectory

    try:
        return SurrogateDirectory.holds_build(path) and (path / "config.json").exists()
    except (OSError, ValueError):
        return False


def surrogate_facts(path: Path, job: dict[str, object] | None) -> SurrogateFacts:
    """Return a build's facts from its folder and its latest job's record (if any).

    Raises
    ------
    ConfigError
        If ``path`` holds no build.
    """
    from apex.domain.surrogate import (
        SurrogateBuildSummary,
        SurrogateCheckSummary,
        SurrogateConfig,
    )
    from apex.services import surrogates
    from apex.storage.surrogate_directory import CHECK, SUMMARY, SurrogateDirectory

    if not is_build_dir(path):
        raise ConfigError(f"{path} holds no surrogate build")
    directory = SurrogateDirectory(path)
    config = SurrogateConfig.model_validate_json((path / "config.json").read_text())
    raw = directory.read_json(SUMMARY)
    summary = None if raw is None else SurrogateBuildSummary.model_validate(raw)
    # The design's full runs from the build's journal, not the summary: the summary is the
    # last stop's (stale while a resume runs), and its pending count also holds an adaptive
    # design's undrawn runs and a coarse stage's runs (the review of 2026-11-09).
    planned = config.design.training + config.design.held_out
    lost = set() if summary is None else set(summary.failed)
    if config.design.plan == "run":  # a run's forecasts: nothing simulated, all there once read
        finished, failed = (0, 0) if summary is None else (planned, len(lost))
    else:
        journal = directory.journal()
        finished = len(journal)
        # The journal's failures and, after a fit, the runs its summary lost (a pair whose
        # coarse run failed is journalled ok here).
        failed = len({m for m, r in journal.items() if not r.ok} | lost)
    pending = max(planned - finished, 0)
    coarse_finished = None
    coarse = path / surrogates.COARSE
    if coarse.is_dir():
        coarse_finished = len(SurrogateDirectory(coarse).journal())
    # A build stopped in its coarse stage, every full run done: what is left is coarse.
    state = _state(summary, job, path)
    coarse_pending = (
        summary.pending if summary is not None and state == "stopped" and pending == 0 else 0
    )
    gate = summary.gate if summary is not None else None
    cost = None if summary is None else surrogates.prediction_cost(path, summary)
    last = None
    count = 0
    if summary is not None and summary.model is not None:
        model_path = path / summary.model
        checks_dir = surrogates.checks_folder(model_path)
        for found in sorted(checks_dir.glob(f"*/{CHECK}")) if checks_dir.is_dir() else []:
            try:
                check = SurrogateCheckSummary.model_validate_json(found.read_text())
            except (OSError, ValueError):  # a check being written: not complete yet
                continue
            if check.status == "complete" and check.identity == summary.identity:
                count += 1
        latest = surrogates.last_check(model_path, summary.identity)
        if latest is not None:
            last = CheckFact(
                id=latest.id, sequence=latest.sequence, against=latest.against,
                runs=latest.runs, succeeded=latest.succeeded, inside=latest.inside,
                passed=latest.passed,
                failures=[] if latest.gate is None else latest.gate.failures(),
            )  # fmt: skip
    return SurrogateFacts(
        path=str(path),
        name=config.surrogate.name,
        job_id=None if job is None else str(job["id"]),
        state=state,
        plugin=config.surrogate.model,
        plan=config.design.plan,
        planned_runs=planned,
        finished_runs=finished,
        failed_runs=failed,
        failed_share=None if finished == 0 else failed / finished,
        pending_runs=pending,
        coarse_runs=coarse_finished,
        coarse_pending=coarse_pending,
        gate_passed=None if gate is None else gate.passed,
        gate=[]
        if gate is None
        else [
            DatatypeFact(
                datatype=d.datatype,
                measure=str(d.measure),
                error=d.error,
                tolerance=d.tolerance,
                passed=d.passed,
            )
            for d in gate.datatypes
        ],
        gate_failures=[] if gate is None else gate.failures(),
        rank_correlation=None if gate is None or gate.rank is None else gate.rank.correlation,
        fit_seconds=None if summary is None else summary.fit_seconds,
        prediction_seconds=None if summary is None else summary.prediction_seconds,
        simulator_median_seconds=None if summary is None else summary.simulator_median_seconds,
        member_simulation="none" if cost is None else cost.simulation,
        member_seconds=None if cost is None else cost.seconds,
        speedup=None if cost is None else cost.speedup,
        model=None if summary is None else summary.model,
        checks=count,
        last_check=last,
    )


def _state(
    summary: object, job: dict[str, object] | None, path: Path | None = None
) -> Literal["new", "queued", "running", "stopped", "failed", "complete"]:
    state = None if job is None else job.get("state")
    # The job's own word while it is live; then the summary's.
    if state == "queued":
        return "queued"
    if state == "running":
        return "running"
    if state == "failed":
        return "failed"
    status = getattr(summary, "status", None)
    if status == "complete":
        return "complete"
    if path is not None and _held(path):  # a build its CLI runs, with no job (the review)
        return "running"
    if status == "stopped" or state in ("stopped", "cancelled"):
        return "stopped"
    return "new"


def _held(path: Path) -> bool:
    """Whether a live process holds the build's lock (probed only when one is named).

    ``in_use``, not ``locked``: a bare probe takes the lock for an instant, and a CLI build
    starting then would be refused (the coder's review). A folder removed meanwhile: no.
    """
    try:
        return in_use(path)
    except OSError:
        return False


def surrogate_rules(facts: SurrogateFacts, thresholds: MonitorSection) -> MonitorReport:
    """Return the rules' report on a surrogate build."""
    findings: list[Finding] = []
    share = facts.failed_share
    if share is not None and share > thresholds.surrogate_max_failed_share:
        findings.append(Finding(
            code="failed_runs", severity="warning",
            evidence=f"{facts.failed_runs} of {facts.finished_runs} design runs failed "
            f"({share:.0%}, over {thresholds.surrogate_max_failed_share:.0%}): the model is "
            "fitted and gated on fewer runs, and the failures may leave a region of the "
            "inputs unsampled",
        ))  # fmt: skip
    if facts.gate_passed is False:
        findings.append(Finding(
            code="gate_failed", severity="critical",
            evidence="the gate on the held-out runs failed, so runs refuse the model: "
            + "; ".join(facts.gate_failures),
        ))  # fmt: skip
    if facts.last_check is not None and not facts.last_check.passed:
        c = facts.last_check
        why = "; ".join(c.failures) or (
            f"{c.succeeded} of {c.runs} runs succeeded and {c.inside} lay inside the domain"
        )
        findings.append(Finding(
            code="check_failed", severity="critical",
            evidence=f"the latest check (#{c.sequence}, against {c.against}) failed, so "
            f"history matching refuses the model: {why}",
        ))  # fmt: skip
    if facts.speedup is not None and facts.speedup < thresholds.surrogate_min_speedup:
        findings.append(Finding(
            code="small_speedup", severity="info",
            evidence=f"one member's prediction takes {facts.member_seconds:.3g} s"
            + (" with its coarse run" if facts.member_simulation == "coarse" else "")
            + f" against a median full run of {facts.simulator_median_seconds:.3g} s "
            f"({facts.speedup:.3g} times faster, under {thresholds.surrogate_min_speedup:g}): "
            "a run gains little from it",
        ))  # fmt: skip
    if facts.state == "stopped" and (facts.pending_runs or facts.coarse_pending):
        left = (
            f"{facts.pending_runs} of {facts.planned_runs} design runs"
            if facts.pending_runs
            else f"{facts.coarse_pending} coarse runs"
        )
        findings.append(Finding(
            code="stopped_pending", severity="info",
            evidence=f"the build stopped with {left} pending; resuming it runs them and fits "
            "(the same model files as an uninterrupted build)",
        ))  # fmt: skip
    severities = {f.severity for f in findings}
    status: Literal["healthy", "warning", "critical"] = (
        "critical" if "critical" in severities else "warning" if "warning" in severities
        else "healthy"
    )  # fmt: skip
    live = facts.state in ("queued", "running")
    recommendation: Literal["continue", "stop", "investigate", "none"] = (
        "investigate" if severities - {"info"} else "continue" if live else "none"
    )
    verdict = (
        "not fitted yet" if facts.gate_passed is None
        else "gate passed" if facts.gate_passed else "gate failed"
    )  # fmt: skip
    noticed = "; ".join(sorted(f.code for f in findings)) or "nothing unusual"
    return MonitorReport(
        status=status,
        recommendation=recommendation,
        findings=findings,
        summary=f"{facts.plugin} build {facts.name} is {facts.state}: {facts.finished_runs} "
        f"of {facts.planned_runs} design runs finished ({facts.failed_runs} failed), "
        f"{verdict}. Noticed: {noticed}.",
    )


def reconcile(model: MonitorReport, rules: MonitorReport) -> tuple[MonitorReport, list[str]]:
    """Hold the model's report to the facts (see the module docstring)."""
    found = {f.code for f in rules.findings}
    kept = [f for f in model.findings if f.code not in RULE_CODES or f.code in found]
    changes = [f"dropped {f.code}: the rules did not find it" for f in model.findings
               if f not in kept]  # fmt: skip
    recommendation = model.recommendation
    if recommendation == "stop":
        recommendation = "investigate"
        changes.append("stop became investigate: the surrogate monitor does not act")
    if not changes:
        return model, []
    return model.model_copy(update={"findings": kept, "recommendation": recommendation}), changes


SYSTEM = (
    "You monitor an APEX surrogate build: full simulator runs of a design, a surrogate "
    "model fitted on the training runs and gated on held-out ones, and checks of the model "
    "on new runs. A model is used by runs only if its gate passed and its latest check did "
    "not fail. You get facts computed from the build's folder and the report of threshold "
    "rules. Decide its status. The facts are complete: answer from them, cite numbers as "
    "evidence, and keep every finding of the rules unless the facts contradict it. "
    "Recommend continue, investigate or none (this monitor never stops a build)."
)
_RULE = "the surrogate monitor's threshold rules"


@dataclass(frozen=True)
class SurrogateCheck:
    """One check of a build (see :class:`SurrogateMonitor`)."""

    facts: SurrogateFacts
    rules_report: MonitorReport
    report: MonitorReport
    decided_by: str
    reason: str | None
    trace: Path
    corrections: tuple[str, ...] = ()


class SurrogateMonitor:
    """Checks surrogate builds (see the module docstring).

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

    def resolve(self, target: str | Path) -> tuple[Path, dict[str, object] | None]:
        """Return the build's folder and its latest job, from a folder or a job id.

        Raises
        ------
        ConfigError
            If ``target`` names no surrogate build.
        """
        path = Path(target)
        records = self.service.jobs.all_records()
        if is_build_dir(path):
            resolved = path.resolve()
            for record in reversed(records):
                dirs = [Path(d).resolve() for d in record.get("run_dirs") or []]
                if record["kind"] in BUILD_KINDS and resolved in dirs:
                    return resolved, dict(record)
            return resolved, None
        try:
            record = self.service.jobs.record(str(target))
        except KeyError:
            raise ConfigError(f"{target!r} is not a surrogate build or its job") from None
        if record["kind"] not in BUILD_KINDS:
            raise ConfigError(f"job {target} is not a surrogate build")
        dirs = record.get("run_dirs") or []
        if not dirs:
            raise ConfigError(f"job {target} has no build folder yet")
        return Path(dirs[0]), dict(record)

    def facts(self, target: str | Path) -> SurrogateFacts:
        """Return the facts of a build."""
        from apex.services import policy

        path, job = self.resolve(target)
        policy.check_source(path, self.service.jobs.settings.policy, self.service.root)
        return surrogate_facts(path, job)

    def check(self, target: str | Path) -> SurrogateCheck:
        """Check a build once: facts, rules, and the model's judgement."""
        facts = self.facts(target)
        rules_report = surrogate_rules(facts, self.thresholds)
        recorder = CallRecorder(Path(facts.path), AGENT)
        system, recorder.advice = advise_prompt(SYSTEM, AGENT, self.service.root)
        decision = decide(
            self.service,
            self.provider,
            recorder,
            task=f"Facts of the surrogate build:\n{facts.model_dump_json(indent=1)}\n\n"
            f"The threshold rules' report:\n{rules_report.model_dump_json(indent=1)}\n\n"
            "Give your report on this build.",
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
        return SurrogateCheck(
            facts, rules_report, report, decision.decided_by, decision.reason, decision.trace,
            tuple(corrections),
        )  # fmt: skip


class SurrogateReportFacts(BaseModel):
    """What the report of a surrogate build is written from."""

    model_config = ConfigDict(extra="forbid")

    facts: SurrogateFacts
    monitor: MonitorReport


def report_facts(monitor: SurrogateMonitor, target: str | Path) -> SurrogateReportFacts:
    """Return the facts of a build's report."""
    facts = monitor.facts(target)
    return SurrogateReportFacts(facts=facts, monitor=surrogate_rules(facts, monitor.thresholds))


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4g}"


def report_template(report: SurrogateReportFacts) -> NarrativeReport:
    """Return the template report of a surrogate build."""
    f = report.facts
    verdict = (
        "has not been fitted yet" if f.gate_passed is None
        else "passed its gate" if f.gate_passed else "failed its gate"
    )  # fmt: skip
    sections = [
        Section(
            heading="Result",
            text=f"The {f.plugin} surrogate {f.name} ({f.state}) {verdict}: "
            f"{f.finished_runs} of {f.planned_runs} design runs ({f.plan}) finished, "
            f"{f.failed_runs} failed, {f.pending_runs} pending."
            + ("" if f.fit_seconds is None else f" The fit took {_fmt(f.fit_seconds)} s.")
            + (
                ""
                if f.speedup is None
                else f" One member's prediction takes {_fmt(f.member_seconds)} s"
                + (" with its coarse run" if f.member_simulation == "coarse" else "")
                + f" against a median full run of {_fmt(f.simulator_median_seconds)} s "
                f"({_fmt(f.speedup)} times faster)."
            )
            + (
                " Each member is also simulated up to the model's date (not timed here)."
                if f.member_simulation == "early"
                else ""
            ),
        ),
    ]
    if f.gate:
        rows = "\n".join(
            f"| {d.datatype} | {d.measure} | {_fmt(d.error)} | {_fmt(d.tolerance)} | "
            f"{'yes' if d.passed else 'no'} |"
            for d in f.gate
        )
        rank = (
            ""
            if f.rank_correlation is None
            else f"\n\nThe held-out objectives' rank correlation: {_fmt(f.rank_correlation)}."
        )
        sections.append(Section(
            heading="Gate",
            text="On the held-out runs, never fitted on:\n\n| Datatype | Measure | Error | "
            f"Tolerance | Passed |\n|---|---|---|---|---|\n{rows}{rank}",
        ))  # fmt: skip
    if f.last_check is not None:
        c = f.last_check
        sections.append(Section(
            heading="Checks",
            text=f"{f.checks} complete check(s) of the model. The latest (#{c.sequence}, "
            f"against {c.against}) {'passed' if c.passed else 'failed'}: {c.succeeded} of "
            f"{c.runs} runs succeeded, {c.inside} inside the domain."
            + ("" if not c.failures else " " + "; ".join(c.failures)),
        ))  # fmt: skip
    sections.append(Section(heading="Health", text=report.monitor.summary))
    return NarrativeReport(title=f"Surrogate build {f.name}", sections=sections)


def surrogate_report(
    service: ApexService, provider: LLMProvider | None, target: str | Path
) -> ReportResult:
    """Write the report of a surrogate build (a folder or job id)."""
    from apex.agents.reporter import Reporter

    monitor = SurrogateMonitor(service, None)
    facts = report_facts(monitor, target)
    recorder = CallRecorder(Path(facts.facts.path), "report")
    return Reporter(service, provider).write(facts, report_template(facts), recorder)

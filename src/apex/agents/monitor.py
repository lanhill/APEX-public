"""The run monitor agent: is a history-matching run healthy, and should it stop.

Facts come from code, judgement from the model:

1. :func:`observe` reads a run directory into a :class:`RunObservation`, deterministically:
   data mismatch trajectory, spread relative to the prior, failed members, rejected trials, the
   normalized data mismatch against the noise level, the CV curve.
2. :func:`rules` turns it into a :class:`MonitorReport` with named thresholds
   (``[agents.monitor]``, :class:`~apex.services.settings.MonitorSection`). This is also the
   fallback.
3. :class:`RunMonitor` gives the model the observation, the rules' report and read-only
   tools, and takes back its own validated report; without a model the rules decide.

The monitor is advisory. With ``act``, a ``stop`` recommendation stops the run's job at its
next checkpoint, but only for a running iterative method (stopping ES-MDA mid-schedule
would leave a posterior its theory does not describe), and only once a person confirms.
"""

import itertools
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from apex.agents.loop import AgentLoop, Fallback, LoopSettings
from apex.agents.trace import CallRecorder
from apex.algorithms.data_assimilation.registries import methods
from apex.core.advice import advise_prompt, advise_thresholds
from apex.core.errors import ConfigError
from apex.core.llm import LLMProvider
from apex.services import composition
from apex.services.api import ApexService
from apex.services.settings import MonitorSection
from apex.services.tools import ToolSession
from apex.storage.run_directory import RunDirectory

#: The tools the model may use: reading only. It cannot stop, resume or submit anything.
READ_ONLY_TOOLS = ("run_summary", "job_status", "job_events")

Severity = Literal["info", "warning", "critical"]
Recommendation = Literal["continue", "stop", "investigate", "none"]

#: Findings that justify stopping a running iterative method.
STOP_WORTHY = frozenset({"damping_runaway", "stagnation", "overfitting", "cv_past_minimum"})

#: The codes the rules compute from the facts. A model finding with one of these codes that
#: the rules did not find is contradicted by the facts, and is dropped (:func:`reconcile`).
RULE_CODES = frozenset({
    "ensemble_collapse", "member_failures", "damping_runaway", "stagnation", "overfitting",
    "poor_fit", "cv_past_minimum", "no_progress", "mismatch_increased",
})  # fmt: skip


class RunObservation(BaseModel):
    """What the monitor knows of a run, computed from its directory (see :func:`observe`)."""

    model_config = ConfigDict(extra="forbid")

    run_dir: str
    run_id: str | None = Field(default=None, description="The catalogue id (run_summary takes it).")
    name: str
    method: str
    iterative: bool = Field(description="Iterative (GIES) or a fixed schedule (ES-MDA).")
    state: Literal["running", "complete", "stopped", "failed", "not started", "incomplete"]
    job_id: str | None = None
    forecasts: int = Field(description="Scored forecasts, rejected trials included.")
    accepted: int
    rejected: int
    rejections_in_a_row: int = Field(description="Rejected trials since the last acceptance.")
    members_prior: int
    members_now: int
    failed_fraction: float
    data_mismatch_iterations: list[int] = Field(description="The accepted iterates.")
    data_mismatch_means: list[float] = Field(description="Their mean data mismatch.")
    last_relative_change: float | None = Field(
        description="|1 - m_k / m_(k-1)| of the last two accepted data mismatches."
    )
    normalized_data_mismatch: float | None = Field(
        description="Mean-square normalized data mismatch of the last accepted iterate "
        "(about 1 at the noise level)."
    )
    spread_ratio: dict[str, float] = Field(
        description="Per variable: mean ensemble std of the last accepted iterate / the prior's."
    )
    damping: float | None = Field(
        description="The last trial's damping (iterative methods; GN-EnRML: its step length, "
        "which falls on rejections)."
    )
    initial_damping: float | None = None
    converged: bool | None = None
    cv_mismatch: list[float] | None = Field(
        default=None, description="CV normalized data mismatch of each accepted iterate."
    )
    cv_minimum_iteration: int | None = None
    cv_minimum_step: int | None = Field(
        default=None, description="That iterate's step (0: the prior; rejected trials not counted)."
    )
    accepted_since_cv_minimum: int | None = None


class Finding(BaseModel):
    """One thing the monitor noticed."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(description="e.g. ensemble_collapse, member_failures, stagnation.")
    severity: Severity
    evidence: str = Field(min_length=3, description="The numbers that show it.")


class MonitorReport(BaseModel):
    """The monitor's verdict on a run."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["healthy", "warning", "critical"]
    recommendation: Recommendation = Field(
        description="continue, stop (a running iterative run only), investigate, or none "
        "(a finished run with nothing to look at)."
    )
    findings: list[Finding] = Field(default_factory=list)
    summary: str = Field(min_length=3, description="Two or three sentences for a person.")


# ---------------------------------------------------------------------- observation


def _state(run_dir: RunDirectory, job_state: str | None) -> str:
    if job_state in ("running", "failed"):
        return job_state
    history = run_dir.history_dicts()
    if not history:
        return "not started"
    summary = run_dir.path / "summary.json"
    if summary.exists() and json.loads(summary.read_text()).get("complete"):
        return "complete"
    return "stopped" if job_state in ("stopped", "cancelled", "complete") else "incomplete"


def observe(
    run_path: Path, *, job_id: str | None = None, job_state: str | None = None
) -> RunObservation:
    """Return what a run directory shows (deterministic; no model involved).

    Raises
    ------
    ConfigError
        If the directory holds no run, or no scored forecast yet.
    """
    run_dir = RunDirectory(run_path)
    config = composition.snapshot_config(run_dir)
    results = composition.load_results(run_path)
    history = results.history
    accepted = [r for r in history if r.accepted]
    trailing = 0
    for record in reversed(history):
        if record.accepted:
            break
        trailing += 1
    means = [r.data_mismatch_mean for r in accepted]
    change = abs(1.0 - means[-1] / means[-2]) if len(means) > 1 and means[-2] > 0 else None
    last = accepted[-1]
    mismatch = None
    if last.metrics and "normalized_data_mismatch" in last.metrics:
        mismatch = last.metrics["normalized_data_mismatch"].get("all")
    # Spread: the members' std per cell, averaged over cells, against the prior's.
    prior, latest = results.iterate(0), results.iterate(last.iteration)
    spread = {}
    for name, values in prior.state.items():
        before = float(np.mean(np.std(values, axis=-1, ddof=1)))
        after = float(np.mean(np.std(latest.state[name], axis=-1, ddof=1)))
        spread[name] = after / before if before > 0 else float("nan")
    dampings = [r.damping for r in history if r.damping is not None]
    cv = [
        r.metrics["cv_normalized_data_mismatch"]["all"]
        for r in accepted
        if r.metrics and "cv_normalized_data_mismatch" in r.metrics
    ]
    cv_minimum = since = None
    if cv:
        best = int(np.argmin(cv))
        cv_minimum, since = accepted[best].iteration, len(cv) - 1 - best
    # The configured size, not the prior forecast's survivors: failures there count too.
    # A stored prior of unstated size falls back to the prior's saved members.
    members_prior = config.prior.ensemble_size or results.iterate(0).member_ids.size
    members_now = last.n_members
    state_file = run_dir.path / "state.json"
    converged = None
    if state_file.exists():
        algorithm = json.loads(state_file.read_text()).get("algorithm") or {}
        converged = algorithm.get("converged")
    return RunObservation(
        run_dir=str(run_path),
        name=config.run.name,
        method=config.method.name,
        iterative=bool(getattr(methods.factory(config.method.name), "iterative", False)),
        state=_state(run_dir, job_state),  # type: ignore[arg-type]
        job_id=job_id,
        forecasts=len(history),
        accepted=len(accepted),
        rejected=len(history) - len(accepted),
        rejections_in_a_row=trailing,
        members_prior=members_prior,
        members_now=members_now,
        failed_fraction=1.0 - members_now / members_prior if members_prior else 0.0,
        data_mismatch_iterations=[r.iteration for r in accepted],
        data_mismatch_means=means,
        last_relative_change=change,
        normalized_data_mismatch=mismatch,
        spread_ratio=spread,
        damping=dampings[-1] if dampings else None,
        initial_damping=dampings[0] if dampings else None,
        converged=None if converged is None else bool(converged),
        cv_mismatch=cv or None,
        cv_minimum_iteration=cv_minimum,
        cv_minimum_step=None
        if cv_minimum is None
        else [r.iteration for r in accepted].index(cv_minimum),
        accepted_since_cv_minimum=since,
    )


# ---------------------------------------------------------------------- rules


def rules(observation: RunObservation, thresholds: MonitorSection | None = None) -> MonitorReport:
    """Return the threshold rules' report on ``observation``."""
    t = thresholds or MonitorSection()
    o = observation
    findings: list[Finding] = []
    collapsed = {k: v for k, v in o.spread_ratio.items() if v < t.collapse_spread_ratio}
    if collapsed:
        worst = ", ".join(f"{k} {v:.3g}" for k, v in sorted(collapsed.items(), key=lambda x: x[1]))
        findings.append(Finding(
            code="ensemble_collapse", severity="warning",
            evidence=f"spread relative to the prior's below {t.collapse_spread_ratio}: {worst}",
        ))  # fmt: skip
    if o.failed_fraction > t.max_failed_fraction:
        findings.append(Finding(
            code="member_failures", severity="warning",
            evidence=f"{o.members_prior - o.members_now} of {o.members_prior} members failed "
            f"({o.failed_fraction:.0%} > {t.max_failed_fraction:.0%})",
        ))  # fmt: skip
    if o.iterative and o.rejections_in_a_row >= t.max_consecutive_rejections:
        findings.append(Finding(
            code="damping_runaway", severity="warning",
            evidence=f"{o.rejections_in_a_row} rejected trials in a row; damping (GN-EnRML: "
            f"step length) {o.damping:.3g} (initially {o.initial_damping:.3g})",
        ))  # fmt: skip
    window = o.data_mismatch_means[-(t.stagnation_window + 1) :]
    if o.iterative and len(window) == t.stagnation_window + 1:
        changes = [abs(1 - b / a) for a, b in itertools.pairwise(window) if a > 0]
        if changes and max(changes) < t.stagnation_tolerance:
            findings.append(Finding(
                code="stagnation", severity="info",
                evidence="the accepted data mismatch changed by at most "
                f"{max(changes):.2%} over the "
                f"last {t.stagnation_window} accepted iterates (< {t.stagnation_tolerance:.0%})",
            ))  # fmt: skip
    if o.normalized_data_mismatch is not None and o.normalized_data_mismatch < t.overfit_mismatch:
        findings.append(Finding(
            code="overfitting", severity="warning",
            evidence=f"normalized data mismatch {o.normalized_data_mismatch:.3g} < "
            f"{t.overfit_mismatch} (1 is the "
            "noise level): the noise is being fitted",
        ))  # fmt: skip
    if (
        o.state == "complete"
        and o.normalized_data_mismatch is not None
        and o.normalized_data_mismatch > t.poor_fit_mismatch
    ):
        findings.append(Finding(
            code="poor_fit", severity="warning",
            evidence="final normalized data mismatch "
            f"{o.normalized_data_mismatch:.3g} > {t.poor_fit_mismatch} (1 is "
            "the noise level)",
        ))  # fmt: skip
    means = o.data_mismatch_means
    # A prior that already fits (normalized mismatch near 1, the noise) has nothing to gain.
    unfit = o.normalized_data_mismatch is None or o.normalized_data_mismatch > 2.0
    if o.state == "complete" and len(means) >= 2 and means[0] > 0:
        drop = 1.0 - means[-1] / means[0]
        spread = ", ".join(f"{k} {v:.2g}" for k, v in sorted(o.spread_ratio.items()))
        if -drop > t.stagnation_tolerance:
            # A rise is not "no progress": the update moved the ensemble away from the data
            # (a reported issue, 2026-10-25: ES-MDA without localization, 45 % higher). Within
            # the stagnation tolerance it is noise; where the data are fitted, a note only.
            findings.append(Finding(
                code="mismatch_increased", severity="warning" if unfit else "info",
                evidence=f"finished with the data mismatch {means[0]:.4g} -> {means[-1]:.4g} "
                f"({-drop:.1%} higher) after {len(means) - 1} accepted iterate(s); spread "
                f"relative to the prior's: {spread or 'n/a'}: the update moved the ensemble "
                "away from the data",
            ))  # fmt: skip
        elif unfit and drop < t.min_mismatch_reduction:
            findings.append(Finding(
                code="no_progress", severity="warning",
                evidence=f"finished with the data mismatch {means[0]:.4g} -> {means[-1]:.4g} "
                f"({abs(drop):.1%} {'lower' if drop >= 0 else 'higher'}, a change < "
                f"{t.min_mismatch_reduction:.0%}) after "
                f"{len(means) - 1} accepted iterate(s); spread relative to the prior's: "
                f"{spread or 'n/a'}: the posterior is still close to the prior",
            ))  # fmt: skip
    if o.accepted_since_cv_minimum is not None and o.accepted_since_cv_minimum >= t.cv_patience:
        findings.append(Finding(
            code="cv_past_minimum", severity="info",
            evidence="CV mismatch lowest at step "
            f"{o.cv_minimum_iteration if o.cv_minimum_step is None else o.cv_minimum_step}; "
            f"{o.accepted_since_cv_minimum} accepted iterates since without improvement",
        ))  # fmt: skip
    return _report(o, findings)


def _report(o: RunObservation, findings: list[Finding]) -> MonitorReport:
    running = o.state in ("running", "incomplete")
    codes = {f.code for f in findings}
    # overfitting justifies a stop only without CV, which would stop the run itself.
    stop_codes = STOP_WORTHY - ({"overfitting"} if o.cv_mismatch else set())
    recommendation: Recommendation
    if running and o.iterative and codes & stop_codes:
        recommendation = "stop"
    elif any(f.severity != "info" for f in findings):
        recommendation = "investigate"
    elif running:
        recommendation = "continue"
    else:
        recommendation = "none"
    severities = {f.severity for f in findings}
    status: Literal["healthy", "warning", "critical"] = (
        "critical"
        if "critical" in severities
        else "warning"
        if "warning" in severities
        else "healthy"
    )
    trend = (
        f"data mismatch {o.data_mismatch_means[0]:.4g} -> {o.data_mismatch_means[-1]:.4g} "
        f"over {o.accepted} accepted "
        f"iterates ({o.rejected} rejected)"
    )
    noticed = "; ".join(sorted(codes)) or "nothing unusual"
    summary = f"{o.method} run {o.name} is {o.state}: {trend}. Noticed: {noticed}."
    return MonitorReport(
        status=status, recommendation=recommendation, findings=findings, summary=summary
    )


def reconcile(model: MonitorReport, rules_report: MonitorReport) -> tuple[MonitorReport, list[str]]:
    """Hold the model's report to the facts; return it and what was changed.

    - A finding with a rule's code (:data:`RULE_CODES`) that the rules did not find is
      contradicted by the facts they were computed from: it is dropped. Findings with the
      model's own codes stay, as do rule findings it chose to leave out (it may judge them
      harmless, e.g. the spread reduction ES-MDA is expected to make).
    - A ``stop`` left without a stop-worthy finding becomes ``investigate``: acting relies
      on it.
    """
    found = {f.code for f in rules_report.findings}
    kept, changes = [], []
    for finding in model.findings:
        if finding.code in RULE_CODES and finding.code not in found:
            changes.append(
                f"dropped {finding.code} ({finding.evidence!r}): the rules did not find it in "
                "the facts"
            )
        else:
            kept.append(finding)
    recommendation = model.recommendation
    supported = {f.code for f in kept} & STOP_WORTHY or rules_report.recommendation == "stop"
    if recommendation == "stop" and not supported:
        recommendation = "investigate"
        changes.append("stop became investigate: no stop-worthy finding is left")
    if not changes:
        return model, []
    # The summary is the model's prose: say what was corrected rather than edit it.
    dropped = [f.code for f in model.findings if f not in kept]
    summary = model.summary
    if dropped:
        summary += f" [Corrected: {', '.join(dropped)} dropped, not supported by the facts.]"
    if recommendation != model.recommendation:
        summary += f" [Corrected: {model.recommendation} became {recommendation}.]"
    update = {"findings": kept, "recommendation": recommendation, "summary": summary}
    return model.model_copy(update=update), changes


# ---------------------------------------------------------------------- the agent

SYSTEM = (
    "You monitor an ensemble history-matching run of APEX. You get facts computed from the "
    "run directory and the report of threshold rules. Decide the run's status and what to "
    "do. The facts are complete: answer from them, and call a tool only for something they "
    "lack (run_summary takes the run_id given in the facts, job_status and job_events the "
    "job_id). Keep every finding of the rules unless the facts contradict it, and cite numbers "
    "as evidence. Recommend stop only for a running iterative method (never for a fixed "
    "schedule such as ES-MDA, whose theory needs every step). For a finished run, "
    "recommend none unless something needs a person's attention (investigate). Use the "
    "tools only to look things up."
)


@dataclass(frozen=True)
class MonitorResult:
    """One check of a run.

    Attributes
    ----------
    observation, rules_report
        The facts, and the rules' report on them.
    report
        The decision: the model's, or the rules' when it fell back or ran without a model.
    decided_by
        ``model``, ``fallback`` (the model failed) or ``rules`` (no model).
    reason
        Why the fallback decided.
    action
        What acting did: ``none``, ``stopped job ...``, or ``refused: ...``.
    trace
        The decision's trace file.
    corrections
        What :func:`reconcile` changed in the model's report, e.g. a dropped finding.
    """

    observation: RunObservation
    rules_report: MonitorReport
    report: MonitorReport
    decided_by: Literal["model", "fallback", "rules"]
    reason: str | None
    action: str
    trace: Path
    corrections: tuple[str, ...] = ()


class RunMonitor:
    """Checks runs (see the module docstring).

    Parameters
    ----------
    service
        The services (to find runs and jobs, and to stop a job when acting).
    provider
        The model; ``None`` for rules only.
    thresholds
        The rules' thresholds (default: ``[agents.monitor]`` of ``service.toml``).
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
        # The installed advice applies unless thresholds are given.
        self.threshold_advice: list[str] = []
        if thresholds is None:
            thresholds, self.threshold_advice = advise_thresholds(
                "agents.monitor", service.jobs.settings.agents.monitor, service.root
            )
        self.thresholds = thresholds
        self.settings = settings or LoopSettings(max_turns=6)

    def resolve(self, target: str | Path) -> tuple[Path, str | None]:
        """Return the run directory and job of a run directory, run id or job id.

        Raises
        ------
        ConfigError
            If ``target`` names no run.
        """
        path = Path(target)
        if (path / "config.json").exists():
            return path.resolve(), self._job_of(path.resolve())
        text = str(target)
        try:
            record = self.service.jobs.record(text)
        except KeyError:
            record = None
        if record is not None:
            if not record.get("run_dirs") or record["kind"] not in ("run", "resume"):
                raise ConfigError(f"job {text} is not a history-matching run")
            return Path(record["run_dirs"][0]), text
        matches = [e for e in self.service.store.entries() if e.run_id.startswith(text)]
        if len(matches) == 1:
            return matches[0].path, self._job_of(matches[0].path)
        if len(matches) > 1:
            raise ConfigError(f"run id {text!r} is ambiguous ({len(matches)} runs)")
        raise ConfigError(f"{text!r} is not a run directory, a job or a run id")

    def _job_of(self, run_path: Path) -> str | None:
        for record in reversed(self.service.jobs.all_records()):
            dirs = [Path(d).resolve() for d in record.get("run_dirs") or []]
            if record["kind"] in ("run", "resume") and run_path in dirs:
                return str(record["id"])
        return None

    def check(
        self,
        target: str | Path,
        *,
        act: bool = False,
        confirm: Callable[[str], bool] | None = None,
    ) -> MonitorResult:
        """Check a run once: observe, apply the rules, ask the model, and act if asked.

        With ``act``, a stop that is safe (a running iterative run) is still put to a
        person first: ``confirm`` gets the question and returns their answer. Without
        ``confirm`` nothing is stopped.
        """
        run_path, job = self.resolve(target)
        job_state = None if job is None else str(self.service.jobs.record(job)["state"])
        observation = observe(run_path, job_id=job, job_state=job_state)
        entry = next(
            (e for e in self.service.store.entries() if e.path.resolve() == run_path), None
        )
        if entry is not None:
            observation = observation.model_copy(update={"run_id": entry.run_id})
        rules_report = rules(observation, self.thresholds)
        recorder = CallRecorder(run_path, "monitor")
        system, prompt_advice = advise_prompt(SYSTEM, "monitor", self.service.root)
        recorder.advice = sorted([*prompt_advice, *self.threshold_advice])
        corrections: list[str] = []
        if self.provider is None:
            trace = recorder.record_decision(
                None,
                task="monitor the run (rules only)",
                decision=rules_report,
                fell_back=False,
                reason=None,
                rule=_RULE,
                thresholds=self.thresholds.model_dump(mode="json"),
                tool_calls=[],
            )
            report, decided_by, reason = rules_report, "rules", None
        else:
            loop = AgentLoop(
                self.provider,
                recorder,
                session=ToolSession(self.service, client="monitor", model=self.provider.model_id),
                tools=READ_ONLY_TOOLS,
                settings=self.settings,
            )
            outcome = loop.run(
                self._task(observation, rules_report),
                MonitorReport,
                Fallback(_RULE, lambda: rules_report, self.thresholds.model_dump(mode="json")),
                system=system,
            )
            report, trace, reason = outcome.decision, outcome.trace, outcome.reason
            decided_by = "fallback" if outcome.fell_back else "model"
            if not outcome.fell_back:
                report, corrections = reconcile(report, rules_report)
                if corrections:
                    recorder.record_action(
                        "corrected the model's report: " + "; ".join(corrections),
                        corrections=list(corrections),
                        report=report.model_dump(),
                    )
        action = self._act(observation, report, recorder, confirm) if act else "none"
        return MonitorResult(
            observation, rules_report, report, decided_by, reason, action, trace,  # type: ignore[arg-type]
            tuple(corrections),
        )  # fmt: skip

    def _task(self, observation: RunObservation, rules_report: MonitorReport) -> str:
        return (
            f"Facts of the run:\n{observation.model_dump_json(indent=1)}\n\n"
            f"The threshold rules' report:\n{rules_report.model_dump_json(indent=1)}\n\n"
            f"Thresholds:\n{self.thresholds.model_dump_json(indent=1)}\n\n"
            "Give your report on this run."
        )

    def _act(
        self,
        o: RunObservation,
        report: MonitorReport,
        recorder: CallRecorder,
        confirm: Callable[[str], bool] | None,
    ) -> str:
        """Stop the run's job if the report says so, it is safe, and a person agrees."""
        if report.recommendation != "stop":
            return "none"
        if not o.iterative:
            action = (
                f"refused: {o.method} is a fixed schedule; stopping it mid-schedule would "
                "break its theory"
            )
        elif o.job_id is None or o.state != "running":
            action = f"refused: the run is {o.state}, with no running job to stop"
        elif confirm is None:
            action = "refused: no one to confirm the stop"
        elif not confirm(_question(o, report)):
            action = f"declined: job {o.job_id} was not stopped"
        else:
            self.service.stop(o.job_id)
            action = f"stopped job {o.job_id} (at its next checkpoint; resumable)"
        recorder.record_action(action, run_dir=o.run_dir, job=o.job_id, report=report.model_dump())
        return action


_RULE = "the run monitor's threshold rules"


def _question(o: RunObservation, report: MonitorReport) -> str:
    """Return the question put to a person before a stop."""
    findings = "; ".join(f"{f.code}: {f.evidence}" for f in report.findings) or "none"
    return (
        f"The monitor recommends stopping job {o.job_id} ({o.method} run {o.name}, "
        f"{o.accepted} accepted iterates). {report.summary} Findings: {findings}. "
        "Stop it at its next checkpoint (it can be resumed)?"
    )

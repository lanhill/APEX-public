"""The diagnosis agent: what went wrong in a run, and what to change.

Facts come from code, judgement from the model, as for the run monitor:

1. :func:`diagnose_facts` reads a run into :class:`DiagnosisFacts`: the monitor's
   observation, the QA/QC of the posterior (coverage, Mahalanobis scores, gains and
   parameter movement; :mod:`apex.services.qaqc`), the failed members' reasons and the
   method's set-up.
2. :func:`rules` turns them into a :class:`Diagnosis` with named thresholds
   (``[agents.diagnosis]``, :class:`~apex.services.settings.DiagnosisSection`, and the
   monitor's). This is also the fallback.
3. :class:`Diagnostician` gives the model the facts and the rules' diagnosis, with read-only
   tools, and holds its answer to the facts (:func:`reconcile`): causes the facts do not
   show are dropped, and every configuration patch is validated against the run's
   configuration before it is shown.

The agent is advisory: it changes nothing. A patch is a JSON merge patch, as a study
variant's, so a recommendation can be tried as a variant.
"""

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from apex.agents.common import Decision, decide, default_localization
from apex.agents.loop import LoopSettings
from apex.agents.monitor import RunMonitor, RunObservation, observe
from apex.agents.monitor import rules as monitor_rules
from apex.agents.trace import CallRecorder
from apex.core.advice import advise_prompt, advise_thresholds
from apex.core.errors import ApexError
from apex.core.llm import LLMProvider
from apex.domain.config import HistoryMatchConfig
from apex.domain.study import merge_patch
from apex.services import composition
from apex.services.api import ApexService
from apex.services.qaqc import qaqc_report
from apex.services.settings import DiagnosisSection, MonitorSection
from apex.storage.run_directory import RunDirectory

Severity = Literal["info", "warning", "critical"]

#: Codes the diagnosis rules compute (with the monitor's): a model cause with one of these
#: codes that the rules did not find is contradicted by the facts, and is dropped.
DIAGNOSIS_CODES = frozenset({
    "data_outside_forecast", "model_deficiency", "overcorrection", "recurring_failure",
})  # fmt: skip
MONITOR_CODES = frozenset({
    "ensemble_collapse", "member_failures", "damping_runaway", "stagnation", "overfitting",
    "poor_fit", "cv_past_minimum", "no_progress", "mismatch_increased",
})  # fmt: skip
RULE_CODES = DIAGNOSIS_CODES | MONITOR_CODES

#: Failure reasons are grouped by their first characters, with numbers replaced by "#".
REASON_PREFIX = 80
NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


class TypeCoverageFact(BaseModel):
    """A data type's observations against the forecast range (QA/QC coverage)."""

    model_config = ConfigDict(extra="forbid")

    datatype: str
    observations: int
    outside: int
    outside_fraction: float
    outside_with_noise: int | None = Field(
        default=None,
        description="Observations outside the forecast range widened by two observation "
        "error deviations: those the ensemble cannot explain (the rule's count).",
    )


class MovementFact(BaseModel):
    """How far a parameter group moved from the prior (QA/QC movement)."""

    model_config = ConfigDict(extra="forbid")

    variable: str
    spread_ratio: float = Field(description="Mean member std now / the prior's.")
    percent_beyond_1_std: float
    percent_beyond_2_std: float
    percent_beyond_3_std: float


class FailureFact(BaseModel):
    """Members that failed with the same reason."""

    model_config = ConfigDict(extra="forbid")

    reason: str
    members: int
    transient: bool
    iterations: list[int]


class DiagnosisFacts(BaseModel):
    """What the diagnosis knows of a run (see :func:`diagnose_facts`)."""

    model_config = ConfigDict(extra="forbid")

    observation: RunObservation
    iteration: int = Field(description="The iterate diagnosed (the posterior).")
    strategy: str | None
    localization: str | None
    has_validation: bool = Field(description="Whether [validation] holds data out.")
    synthetic: bool
    coverage: list[TypeCoverageFact]
    data_scores: list[tuple[str, float]] = Field(
        description="The largest level-1 Mahalanobis scores per datum (chi-squared(1) if "
        "consistent)."
    )
    type_scores: list[tuple[str, float]] = Field(
        description="Level-1 scores of each data type's leading component."
    )
    gains: list[tuple[str, float]] = Field(
        description="Largest gain entries, 'datatype -> variable': which data move what."
    )
    movement: list[MovementFact]
    failures: list[FailureFact]
    notes: list[str] = Field(default_factory=list)


class Cause(BaseModel):
    """A cause of a problem, with the numbers that show it."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(description="e.g. model_deficiency, ensemble_collapse, overcorrection.")
    severity: Severity
    evidence: str = Field(min_length=3)


class Recommendation(BaseModel):
    """A change to try."""

    model_config = ConfigDict(extra="forbid")

    change: str = Field(min_length=3, description="What to change, for a person.")
    rationale: str = Field(min_length=3, description="Why, citing the causes.")
    addresses: list[str] = Field(default_factory=list, description="The causes' codes.")
    patch: dict[str, Any] | None = Field(
        default=None,
        description="A JSON merge patch (RFC 7386) over the run's configuration, e.g. "
        '{"method": {"localization": {"name": "adaptive"}}}; none for a change that is not '
        "a configuration setting (e.g. checking the data).",
    )


class Diagnosis(BaseModel):
    """The diagnosis of a run."""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["healthy", "problems", "failed"] = Field(
        description="healthy: nothing to fix; problems: causes found; failed: the run failed."
    )
    causes: list[Cause] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    summary: str = Field(min_length=3, description="Two to four sentences for a person.")


# ---------------------------------------------------------------------- facts


def _failures(run_path: Path) -> list[FailureFact]:
    events = run_path / "events.jsonl"
    if not events.exists():
        return []
    groups: dict[str, dict[str, Any]] = {}
    for line in events.read_text().splitlines():
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if raw.get("event") != "member_failed":
            continue
        text = str(raw.get("reason") or "").strip()
        # Member numbers, values and times differ between members failing for one reason.
        reason = NUMBER.sub("#", text.splitlines()[0])[:REASON_PREFIX] if text else "(no reason)"
        group = groups.setdefault(reason, {"members": 0, "transient": False, "iterations": set()})
        group["members"] += 1
        group["transient"] |= bool(raw.get("transient"))
        if raw.get("iteration") is not None:
            group["iterations"].add(int(raw["iteration"]))
    return [
        FailureFact(
            reason=reason,
            members=g["members"],
            transient=g["transient"],
            iterations=sorted(g["iterations"]),
        )
        for reason, g in sorted(groups.items(), key=lambda item: -item[1]["members"])
    ]


def diagnose_facts(
    run_path: Path, *, job_id: str | None = None, job_state: str | None = None, top: int = 10
) -> DiagnosisFacts:
    """Return what a run directory shows about its problems (deterministic; no model).

    Raises
    ------
    ConfigError
        If the directory holds no run, or no scored forecast yet.
    """
    observation = observe(run_path, job_id=job_id, job_state=job_state)
    config = composition.snapshot_config(RunDirectory(run_path))
    notes: list[str] = []
    coverage: list[TypeCoverageFact] = []
    movement: list[MovementFact] = []
    data_scores: list[tuple[str, float]] = []
    type_scores: list[tuple[str, float]] = []
    gains: list[tuple[str, float]] = []
    iteration = observation.data_mismatch_iterations[-1]
    try:
        report = qaqc_report(run_path, top=top)
    except (ApexError, ValueError, KeyError, OSError) as error:
        # QA/QC needs the observations and the saved iterates; a run without them is
        # still diagnosed from its history.
        notes.append(f"QA/QC not available: {error}")
    else:
        iteration = report.iteration
        for c in report.coverage:
            n = int(c.observations.size)
            coverage.append(TypeCoverageFact(
                datatype=c.datatype, observations=n, outside=c.outside,
                outside_fraction=c.outside / n if n else 0.0,
                outside_with_noise=c.outside_with_noise(),
            ))  # fmt: skip
        for name, m in report.movement.items():
            movement.append(MovementFact(
                variable=name,
                spread_ratio=m.std / m.prior_std if m.prior_std > 0 else float("nan"),
                percent_beyond_1_std=m.beyond[0],
                percent_beyond_2_std=m.beyond[1],
                percent_beyond_3_std=m.beyond[2],
            ))  # fmt: skip
        data_scores, type_scores = list(report.data_scores), list(report.type_scores)
        gains = list(report.gains_max)
        notes.extend(report.notes)
    return DiagnosisFacts(
        observation=observation,
        iteration=iteration,
        strategy=config.method.strategy.name if config.method.strategy else None,
        localization=config.method.localization.name if config.method.localization else None,
        has_validation=config.validation is not None,
        synthetic=config.case.kind == "synthetic",
        coverage=coverage,
        data_scores=data_scores,
        type_scores=type_scores,
        gains=gains,
        movement=movement,
        failures=_failures(run_path),
        notes=notes,
    )


# ---------------------------------------------------------------------- rules


#: The chance an observation the ensemble explains lies outside the widened range: an upper
#: bound (outside the members' whole range and two error deviations beyond it), so the
#: binomial test below errs towards silence.
OUTSIDE_RATE = 0.05


def _outside_count(c: TypeCoverageFact) -> int:
    return c.outside if c.outside_with_noise is None else c.outside_with_noise


def _outside_significant(c: TypeCoverageFact, t: DiagnosisSection) -> bool:
    """Whether a data type has too many observations outside to be chance (release-tester U1).

    Both the fraction (``coverage_outside_fraction``) and a binomial test against
    :data:`OUTSIDE_RATE` (``coverage_significance``) must say so: with few data a fraction
    alone is noise (2 of 8 is 25 %, and happens by chance about one run in twenty). The test
    takes the misses as independent; a data type's observations form a time series whose
    misses go together, so it errs towards flagging, which the fraction then limits.
    """
    k, n = _outside_count(c), c.observations
    if n == 0 or k / n <= t.coverage_outside_fraction:
        return False
    tail = sum(
        math.comb(n, i) * OUTSIDE_RATE**i * (1 - OUTSIDE_RATE) ** (n - i) for i in range(k, n + 1)
    )
    return tail < t.coverage_significance


def rules(
    facts: DiagnosisFacts,
    thresholds: DiagnosisSection | None = None,
    monitor: MonitorSection | None = None,
) -> Diagnosis:
    """Return the threshold rules' diagnosis of ``facts``."""
    t = thresholds or DiagnosisSection()
    o = facts.observation
    causes = [
        Cause(code=f.code, severity=f.severity, evidence=f.evidence)
        for f in monitor_rules(o, monitor).findings
    ]
    outside = [c for c in facts.coverage if _outside_significant(c, t)]
    if outside:
        causes.append(Cause(
            code="data_outside_forecast", severity="warning",
            evidence="observations outside the forecast range widened by their error (> "
            f"{t.coverage_outside_fraction:.0%}, binomial p < {t.coverage_significance:g}): "
            + ", ".join(f"{c.datatype} {_outside_count(c)}/{c.observations}" for c in outside),
        ))  # fmt: skip
    far = [(k, v) for k, v in facts.data_scores + facts.type_scores if v > t.mahalanobis_limit]
    if far:
        worst = ", ".join(f"{k} {v:.3g}" for k, v in sorted(far, key=lambda x: -x[1])[:5])
        causes.append(Cause(
            code="model_deficiency", severity="warning",
            evidence=f"Mahalanobis scores above {t.mahalanobis_limit} (chi-squared(1) 99.9%): "
            f"{worst}",
        ))  # fmt: skip
    over = [m for m in facts.movement if m.percent_beyond_3_std > t.overcorrection_percent]
    if over:
        causes.append(Cause(
            code="overcorrection", severity="warning",
            evidence="parameters moved beyond 3 prior std: "
            + ", ".join(f"{m.variable} {m.percent_beyond_3_std:.3g}%" for m in over)
            + f" (> {t.overcorrection_percent:g}%)",
        ))  # fmt: skip
    recurring = [f for f in facts.failures if f.members >= t.recurring_failure_members]
    if recurring:
        causes.append(Cause(
            code="recurring_failure", severity="warning",
            evidence="; ".join(f"{f.members} members: {f.reason!r}" for f in recurring[:3]),
        ))  # fmt: skip
    codes = {c.code for c in causes}
    # A rise on a run that fits its data is a note only: nothing to change for it.
    advised = codes - {c.code for c in causes if c.code == "mismatch_increased"
                       and c.severity == "info"}  # fmt: skip
    recommendations = _recommend(facts, advised)
    failed = o.state == "failed"
    serious = any(c.severity != "info" for c in causes)
    verdict: Literal["healthy", "problems", "failed"] = (
        "failed" if failed else "problems" if serious else "healthy"
    )
    found = ", ".join(sorted(codes)) or "nothing to fix"
    summary = f"{o.method} run {o.name} ({o.state}, iterate {facts.iteration} diagnosed): {found}."
    if recommendations:
        summary += f" {len(recommendations)} change(s) recommended."
    return Diagnosis(
        verdict=verdict, causes=causes, recommendations=recommendations, summary=summary
    )


def _recommend(facts: DiagnosisFacts, codes: set[str]) -> list[Recommendation]:
    """Return the rules' recommendations for the causes found."""
    out: list[Recommendation] = []
    spurious = codes & {"ensemble_collapse", "overcorrection", "mismatch_increased"}
    if spurious and facts.localization is None:
        name = default_localization(facts.observation.method)
        out.append(Recommendation(
            change=f"Localize the update with adaptive localization ({name})",
            rationale="the ensemble collapsed, over-corrected or moved away from the data, "
            "the signature of spurious correlations in a small ensemble; localization "
            "removes them",
            addresses=sorted(spurious),
            patch={"method": {"localization": {"name": name}}},
        ))  # fmt: skip
    elif "mismatch_increased" in codes:
        out.append(Recommendation(
            change="Use more members, or smaller steps (for ES-MDA, more steps with larger "
            "inflation factors), and check that the data errors are not too small",
            rationale="the localized update still moved the ensemble away from the data: "
            "the steps are too large for what the ensemble can estimate",
            addresses=["mismatch_increased"],
        ))  # fmt: skip
    if "overfitting" in codes and not facts.has_validation:
        out.append(Recommendation(
            change="Hold some wells out for cross validation ([validation]; apex data-split "
            "proposes a split)",
            rationale="the data are fitted below their noise level; a CV stopping rule picks "
            "the iterate before that",
            addresses=["overfitting"],
        ))  # fmt: skip
    if codes & {"model_deficiency", "data_outside_forecast", "poor_fit"}:
        out.append(Recommendation(
            change="Check the flagged data's errors and the prior's range for the parameters "
            "that control them; do not drop the data to improve the fit",
            rationale="the forecast cannot reach these observations: the prior is biased or "
            "too narrow, the data errors are too small, or the model misses a process",
            addresses=sorted(codes & {"model_deficiency", "data_outside_forecast", "poor_fit"}),
        ))  # fmt: skip
    if "no_progress" in codes:
        out.append(Recommendation(
            change="Run more, less damped iterations: lower the schedule's initial damping "
            "(e.g. initial = \"auto\" for LM-EnRML), raise max_iterations, and keep the LM "
            "scheme's confirm_convergence on; for ES-MDA, check that the inflation factors "
            "are not so large that the steps vanish",
            rationale="the run stopped with its data mismatch barely below the prior's, so "
            "the posterior is still essentially the prior; a heavily damped step is small "
            "without the data mismatch being near a minimum",
            addresses=["no_progress"],
        ))  # fmt: skip
    if "damping_runaway" in codes:
        out.append(Recommendation(
            change="Start with a larger damping (schedule initial damping) or fewer data "
            "per update",
            rationale="trials keep being rejected, so the damping keeps growing and the "
            "steps shrink to nothing",
            addresses=["damping_runaway"],
        ))  # fmt: skip
    if codes & {"member_failures", "recurring_failure"}:
        transient = any(f.transient for f in facts.failures)
        out.append(Recommendation(
            change="Look at the failed members' simulator logs (simulations/iter_k/member_i, "
            "kept with [output] keep_simulations = \"failed\")"
            + ("; transient failures are retried, consider more retries" if transient else ""),
            rationale="dropping many members biases the posterior toward the members that "
            "simulate",
            addresses=sorted(codes & {"member_failures", "recurring_failure"}),
            patch=None,
        ))  # fmt: skip
    return out


# ---------------------------------------------------------------------- reconcile


def check_patch(
    service: ApexService, config: HistoryMatchConfig, patch: dict[str, Any]
) -> str | None:
    """Return why ``patch`` does not apply to ``config`` (``None``: it validates).

    The patched configuration is validated as a model, then by the service (which reads
    its files and plans its cost, but simulates nothing).
    """
    document = merge_patch(config.model_dump(mode="json"), patch)
    try:
        HistoryMatchConfig.model_validate(document)
    except ValidationError as error:
        return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
    validation = service.validate(document)
    return None if validation.valid else "; ".join(validation.errors)


def check_patches(
    recommendations: list[Recommendation], patch_error: Callable[[dict[str, Any]], str | None]
) -> tuple[list[Recommendation], list[str]]:
    """Drop the patches that do not validate (the advice stays); return what was dropped."""
    kept, dropped = [], []
    for rec in recommendations:
        if rec.patch is not None:
            error = patch_error(rec.patch)
            if error is not None:
                dropped.append(f"dropped the patch of {rec.change!r}: {error[:300]}")
                rec = rec.model_copy(update={"patch": None})
        kept.append(rec)
    return kept, dropped


def reconcile(
    model: Diagnosis,
    rules_diagnosis: Diagnosis,
    patch_error: Callable[[dict[str, Any]], str | None],
) -> tuple[Diagnosis, list[str]]:
    """Hold the model's diagnosis to the facts; return it and what was changed.

    - A cause with a rule's code (:data:`RULE_CODES`) that the rules did not find is
      dropped: the facts contradict it. The model's own codes stay.
    - A recommendation's patch that does not validate is dropped, with its error; the
      recommendation stays, as advice without a patch.
    """
    found = {c.code for c in rules_diagnosis.causes}
    changes: list[str] = []
    causes = []
    for cause in model.causes:
        if cause.code in RULE_CODES and cause.code not in found:
            changes.append(f"dropped cause {cause.code}: the rules did not find it in the facts")
        else:
            causes.append(cause)
    recommendations, dropped = check_patches(model.recommendations, patch_error)
    changes.extend(dropped)
    if not changes:
        return model, []
    summary = model.summary + f" [Corrected: {len(changes)} change(s), see the corrections.]"
    return model.model_copy(
        update={"causes": causes, "recommendations": recommendations, "summary": summary}
    ), changes


# ---------------------------------------------------------------------- the agent

SYSTEM = (
    "You diagnose an ensemble history-matching run of APEX: what went wrong and what to "
    "change. You get facts computed from the run directory (the run's trajectory, QA/QC of "
    "the posterior: coverage, Mahalanobis scores, gains and parameter movement; failed "
    "members) and the diagnosis of threshold rules. Answer from the facts; call a tool only "
    "for something they lack. Keep every cause of the rules unless the facts contradict it, "
    "and cite numbers as evidence. Recommend changes that address the causes; give a patch "
    "(a JSON merge patch over the run's configuration) only for a configuration setting you "
    "are sure exists (get_schema and list_plugins show them). Never recommend removing data "
    "to improve the fit."
)
_RULE = "the diagnosis agent's threshold rules"


@dataclass(frozen=True)
class DiagnosisResult:
    """One diagnosis of a run.

    Attributes
    ----------
    facts, rules_diagnosis
        The facts, and the rules' diagnosis of them.
    diagnosis
        The decision: the model's (reconciled), or the rules'.
    decided_by, reason, trace
        See :class:`~apex.agents.common.Decision`.
    corrections
        What :func:`reconcile` changed in the model's diagnosis.
    """

    facts: DiagnosisFacts
    rules_diagnosis: Diagnosis
    diagnosis: Diagnosis
    decided_by: str
    reason: str | None
    trace: Path
    corrections: tuple[str, ...] = ()


class Diagnostician:
    """Diagnoses runs (see the module docstring).

    Parameters
    ----------
    service
        The services (to find runs and jobs, and to validate patches).
    provider
        The model; ``None`` for rules only.
    thresholds
        The rules' thresholds (default: ``[agents.diagnosis]`` of ``service.toml``).
    settings
        Bounds of the model's task.
    """

    def __init__(
        self,
        service: ApexService,
        provider: LLMProvider | None,
        *,
        thresholds: DiagnosisSection | None = None,
        settings: LoopSettings | None = None,
    ) -> None:
        self.service = service
        self.provider = provider
        # The installed advice applies unless thresholds are given.
        self.threshold_advice: list[str] = []
        if thresholds is None:
            thresholds, self.threshold_advice = advise_thresholds(
                "agents.diagnosis", service.jobs.settings.agents.diagnosis, service.root
            )
        self.thresholds = thresholds
        self.monitor_thresholds, monitor_advice = advise_thresholds(
            "agents.monitor", service.jobs.settings.agents.monitor, service.root
        )
        self.threshold_advice += monitor_advice
        self.settings = settings or LoopSettings(max_turns=6)
        self._monitor = RunMonitor(service, None)

    def resolve(self, target: str | Path) -> tuple[Path, str | None]:
        """Return the run directory and job of a run directory, run id or job id."""
        return self._monitor.resolve(target)

    def diagnose(self, target: str | Path) -> DiagnosisResult:
        """Diagnose a run once: facts, rules, the model's judgement held to the facts."""
        run_path, job = self.resolve(target)
        job_state = None if job is None else str(self.service.jobs.record(job)["state"])
        facts = diagnose_facts(run_path, job_id=job, job_state=job_state)
        config = composition.snapshot_config(RunDirectory(run_path))

        def patch_error(patch: dict[str, Any]) -> str | None:
            return check_patch(self.service, config, patch)

        rules_diagnosis = rules(facts, self.thresholds, self.monitor_thresholds)
        # The rules' patches are held to the configuration too (e.g. a method that takes
        # no localization).
        checked, dropped = check_patches(rules_diagnosis.recommendations, patch_error)
        rules_diagnosis = rules_diagnosis.model_copy(update={"recommendations": checked})
        facts = facts.model_copy(update={"notes": [*facts.notes, *dropped]})
        recorder = CallRecorder(run_path, "diagnosis")
        system, prompt_advice = advise_prompt(SYSTEM, "diagnosis", self.service.root)
        recorder.advice = sorted([*prompt_advice, *self.threshold_advice])
        thresholds = {
            "diagnosis": self.thresholds.model_dump(mode="json"),
            "monitor": self.monitor_thresholds.model_dump(mode="json"),
        }
        decision: Decision[Diagnosis] = decide(
            self.service,
            self.provider,
            recorder,
            task=(
                f"Facts of the run:\n{facts.model_dump_json(indent=1)}\n\n"
                f"The threshold rules' diagnosis:\n{rules_diagnosis.model_dump_json(indent=1)}"
                f"\n\nThresholds:\n{json.dumps(thresholds, indent=1)}\n\nGive your diagnosis."
            ),
            response_model=Diagnosis,
            rules=rules_diagnosis,
            rule=_RULE,
            thresholds=thresholds,
            client="diagnosis",
            settings=self.settings,
            system=system,
        )
        diagnosis = decision.value
        corrections: list[str] = []
        if decision.decided_by == "model":
            diagnosis, corrections = reconcile(diagnosis, rules_diagnosis, patch_error)
            if corrections:
                recorder.record_action(
                    "corrected the model's diagnosis: " + "; ".join(corrections),
                    corrections=corrections,
                    diagnosis=diagnosis.model_dump(),
                )
        return DiagnosisResult(
            facts, rules_diagnosis, diagnosis, decision.decided_by, decision.reason,
            decision.trace, tuple(corrections),
        )  # fmt: skip

"""The report agent: a narrative of a run's or a study's results.

1. Facts come from code: for a run, :class:`RunReportFacts` (its summary, the metrics step
   by step, the monitor's rules report and the diagnosis rules' causes); for a study,
   :class:`StudyReportFacts` (its comparison).
2. The model writes a :class:`NarrativeReport`: sections of prose and the ``key_numbers``
   the prose relies on, each with its source in the facts.
3. Code holds it to the facts (:func:`reconcile`): a key number that matches no fact
   (within :data:`NUMBER_RTOL`) is dropped and listed under the report's corrections, so a
   reader sees what the model got wrong. Without a model, :func:`run_rules` and
   :func:`study_rules` write the report from a template.

The report agent reads only; it writes Markdown (:func:`markdown`).
"""

import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from apex.agents.common import close, decide
from apex.agents.diagnosis import Diagnostician, diagnose_facts
from apex.agents.diagnosis import rules as diagnosis_rules
from apex.agents.loop import LoopSettings
from apex.agents.monitor import rules as monitor_rules
from apex.agents.trace import CallRecorder
from apex.core.advice import advise_prompt
from apex.core.llm import LLMProvider
from apex.domain.services import Comparison, RunSteps, RunSummary
from apex.services.api import ApexService, steps_document
from apex.services.summaries import summarize_one

#: Numbers in prose are rounded: a key number within 0.5% of a fact matches it.
NUMBER_RTOL = 0.005
#: A number written in a fact's text: "0.0338", "78.8", "-1.5e+04" (a sign only before digits).
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w])")
#: The longest setup value a report shows, in characters (the facts keep it whole).
SETUP_CHARS = 300
#: Paired differences below this p-value are called significant in the rules' report.
SIGNIFICANCE = 0.05


class RunReportFacts(BaseModel):
    """What the report of a run is written from."""

    model_config = ConfigDict(extra="forbid")

    summary: RunSummary
    steps: RunSteps
    monitor: dict[str, object] = Field(description="The monitor's rules report.")
    causes: list[dict[str, str]] = Field(description="The diagnosis rules' causes.")
    setup: dict[str, object] = Field(
        default_factory=dict,
        description="The run's setup from its configuration snapshot: method and options, "
        "strategy, localization, damping, prior, data and simulator.",
    )
    compared_with: "ComparedRun | None" = Field(
        default=None, description="Another run the user asked to compare with."
    )


class ComparedRun(BaseModel):
    """A run a report compares its run with: its summary and setup."""

    model_config = ConfigDict(extra="forbid")

    summary: RunSummary
    setup: dict[str, object]


RunReportFacts.model_rebuild()  # its compared_with names ComparedRun, defined after it


class StudyReportFacts(BaseModel):
    """What the report of a study is written from."""

    model_config = ConfigDict(extra="forbid")

    comparison: Comparison


class Section(BaseModel):
    """A section of the report."""

    model_config = ConfigDict(extra="forbid")

    heading: str = Field(min_length=1)
    text: str = Field(min_length=1, description="Markdown prose.")


class KeyNumber(BaseModel):
    """A number the prose relies on, and where in the facts it comes from."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="e.g. 'posterior rmse (mean)'.")
    value: float
    source: str = Field(min_length=1, description="e.g. 'steps.metrics rmse/all iteration 4'.")


class NarrativeReport(BaseModel):
    """A narrative report."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    sections: list[Section] = Field(min_length=1)
    key_numbers: list[KeyNumber] = Field(default_factory=list)


# ---------------------------------------------------------------------- facts


def run_facts(service: ApexService, run_path: Path) -> RunReportFacts:
    """Return the facts of a run's report (deterministic).

    Raises
    ------
    ConfigError
        If the directory holds no summarisable run.
    """
    from apex.services.api import summarise_run

    diagnosis = diagnose_facts(run_path)
    settings = service.jobs.settings.agents
    return RunReportFacts(
        setup=run_setup(run_path),
        summary=summarise_run(run_path),
        steps=steps_document(summarize_one(run_path)),
        monitor=monitor_rules(diagnosis.observation, settings.monitor).model_dump(),
        causes=[
            c.model_dump()
            for c in diagnosis_rules(diagnosis, settings.diagnosis, settings.monitor).causes
        ],
    )


def run_setup(run_path: Path) -> dict[str, object]:
    """Return a run's setup from its configuration snapshot (``config.json``).

    A reported issue (#178): a report asked for the setup and parameters had none. Read as plain
    JSON, so snapshots of every version describe themselves; paths are given by name only.
    """
    try:
        config = json.loads((run_path / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    def plugin(section: object) -> object:
        if not isinstance(section, dict):
            return _no_paths(section)
        shown = {"name": section.get("name")}
        if section.get("options"):
            shown["options"] = _no_paths(section["options"])
        return shown

    method = config.get("method") or {}
    prior = config.get("prior") or {}
    setup: dict[str, object] = {
        "method": plugin(method),
        **{
            key: plugin(method[key])
            for key in ("strategy", "localization", "damping")
            if isinstance(method, dict) and method.get(key) is not None
        },
        "ensemble_size": prior.get("ensemble_size"),
        "prior": {
            "generator": plugin(prior.get("generator")),
            **(
                {"variables": {k: plugin(v) for k, v in prior["variables"].items()}}
                if prior.get("variables")
                else {}
            ),
            # Grouped variables are drawn by their group's generator, not the default.
            **(
                {
                    "groups": {
                        k: {**_as_dict(plugin(v)), "variables": v.get("variables")}
                        for k, v in prior["groups"].items()
                    }
                }
                if prior.get("groups")
                else {}
            ),
        },
        "state_variables": [
            {k: v for k, v in variable.items() if k in ("name", "size", "lower", "upper")}
            for variable in (config.get("state") or {}).get("variables", [])
        ],
        "simulator": (config.get("simulator") or {}).get("name"),
        "case_kind": (config.get("case") or {}).get("kind"),
        # A comparison says whether two runs drew the same random numbers (#237).
        "seed": (config.get("run") or {}).get("seed"),
    }
    transforms = [t.get("name") for t in config.get("data_transforms") or []]
    if transforms:
        setup["data_transforms"] = transforms
    if config.get("validation"):
        setup["validation"] = True
    return setup


def _no_paths(value: object) -> object:
    """Return ``value`` with every absolute path, at any depth, reduced to its name.

    A run's snapshot holds resolved paths (``*_file`` options, nested in e.g. GMAC's
    constraints), which a web-UI guest must not see.
    """
    if isinstance(value, str):
        return Path(value).name if value.startswith(("/", "~")) else value
    if isinstance(value, dict):
        return {k: _no_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_no_paths(v) for v in value]
    return value


def _as_dict(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, dict) else {"name": value}


def fact_numbers(facts: BaseModel) -> list[float]:
    """Return every finite number in the facts (what a key number may match).

    Numbers written in the facts' text count too: the monitor's findings and the diagnosis'
    causes state theirs in their evidence ("permx 0.0338", "78.8%"), and a key number
    citing them is true (release-tester N30).
    """
    found: list[float] = []

    def walk(value: object) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, int | float):
            if math.isfinite(value):
                found.append(float(value))
        elif isinstance(value, str):
            for match in _NUMBER.finditer(value):
                number = float(match.group())
                if math.isfinite(number):
                    found.append(number)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list | tuple):
            for item in value:
                walk(item)

    walk(facts.model_dump(mode="json"))
    return found


# ---------------------------------------------------------------------- rules


def _fmt(value: float | None) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.4g}"


def run_rules(facts: RunReportFacts) -> NarrativeReport:
    """Return the template report of a run."""
    s, steps = facts.summary, facts.steps
    selected = steps.selected_iteration
    first = steps.steps[0].iteration if steps.steps else None
    # Said as step numbers (0: the prior; rejected trials not counted); the sources keep
    # the records (iteration), which key the metrics.
    step_of = {s.iteration: s.step for s in steps.steps}
    first_step = step_of.get(first) if first is not None else None
    result_step = step_of.get(selected) if selected is not None else None
    lines, numbers = [], []
    labels: dict[str, list[str]] = {}
    for row in steps.metrics:
        labels.setdefault(row.metric, [])
        if row.label not in labels[row.metric]:
            labels[row.metric].append(row.label)
    for metric, metric_labels in labels.items():
        for label in metric_labels:
            rows = {r.iteration: r for r in steps.metrics if (r.metric, r.label) == (metric, label)}
            start = rows.get(first) if first is not None else None
            end = rows.get(selected) if selected is not None else None
            if end is None or end.mean is None:
                continue
            name = metric if label == "all" else f"{metric} ({label})"
            if start is not None and start.mean is not None:
                lines.append(
                    f"- **{name}**: {_fmt(start.mean)} at step {first_step} to "
                    f"{_fmt(end.mean)} at step {result_step} (member mean; p10 to p90 "
                    f"{_fmt(end.p10)} to {_fmt(end.p90)})."
                )
                numbers.append(KeyNumber(
                    name=f"{name} at step {first_step}", value=start.mean,
                    source=f"steps.metrics {metric}/{label} iteration {first}",
                ))  # fmt: skip
            else:
                lines.append(f"- **{name}**: {_fmt(end.mean)} at step {result_step}.")
            numbers.append(KeyNumber(
                name=f"{name} at step {result_step}", value=end.mean,
                source=f"steps.metrics {metric}/{label} iteration {selected}",
            ))  # fmt: skip
    overview = (
        # The prior counted apart, as `apex summary` does ("4 steps after the prior").
        f"{s.method} run **{s.name}** is {steps.status}: {max(len(steps.steps) - 1, 0)} steps "
        "after the prior, "
        f"{steps.forecasts} forecasts, {s.simulator_runs} simulator runs, "
        f"{s.members} members at the end. The result is step {result_step}"
        + (" (selected by cross validation)." if s.selected_iteration is not None else ".")
    )
    causes = facts.causes
    health = (
        "\n".join(f"- `{c['code']}` ({c['severity']}): {c['evidence']}" for c in causes)
        if causes
        else "The monitor and diagnosis rules found nothing to fix."
    )
    sections = [
        Section(heading="Summary", text=_summary(facts, result_step, numbers)),
        Section(heading="Overview", text=overview),
    ]
    if facts.setup:
        sections.append(Section(heading="Setup", text=_setup_text(facts.setup)))
    sections += [
        Section(heading="Performance metrics", text="\n".join(lines) or "No metrics."),
        Section(heading="Health and diagnosis", text=health),
    ]
    if facts.compared_with is not None:
        sections.append(Section(heading="Comparison", text=_comparison_text(facts)))
    return NarrativeReport(
        title=f"History matching of {s.name}", sections=sections, key_numbers=numbers
    )


def _summary(facts: RunReportFacts, result_step: int | None, numbers: list[KeyNumber]) -> str:
    """Return the report's opening: the outcome and the result's key numbers, in brief."""
    s = facts.summary
    state = {True: "converged", False: "did not converge", None: "has no convergence test"}
    found = (
        f"{len(facts.causes)} issue(s) found by the monitor and diagnosis rules"
        if facts.causes
        else "no issues found by the monitor and diagnosis rules"
    )
    ends = [n for n in numbers if n.name.endswith(f"at step {result_step}")]
    shown = "; ".join(f"{n.name}: {_fmt(n.value)}" for n in ends[:4])
    return (
        f"{s.method} on {s.members} members, {s.state}; it {state[s.converged]}, with "
        f"{found}." + (f" Result: {shown}." if shown else "")
    )


def _setup_text(setup: dict[str, object]) -> str:
    """Return the setup as a list, one line per part."""
    labels = {
        "method": "Method", "strategy": "Update strategy", "localization": "Localization",
        "damping": "Damping", "ensemble_size": "Ensemble size", "prior": "Prior",
        "state_variables": "State variables", "data_transforms": "Data transforms",
        "simulator": "Simulator", "case_kind": "Case", "validation": "Cross validation",
        "seed": "Seed",
    }  # fmt: skip

    def shown(value: object) -> str:
        text = _readable(value)  # long options (positions, schedules) are cut short
        return text if len(text) <= SETUP_CHARS else text[: SETUP_CHARS - 1] + "…"

    return "\n".join(
        f"- **{labels.get(key, key)}**: {shown(value)}"
        for key, value in setup.items()
        if value is not None
    )


def _readable(value: object) -> str:
    """Return a setup value as prose, not JSON (release-tester U8).

    A plugin ``{"name": n, "options": {...}}`` reads ``n (key = value, ...)``; other tables
    ``key = value, ...``; lists ``a; b``. Scalars keep TOML's look (strings unquoted).
    """
    if isinstance(value, dict):
        if isinstance(value.get("name"), str) and set(value) <= {"name", "options"}:
            options = value.get("options")
            inner = _readable(options) if isinstance(options, dict) and options else ""
            return f"{value['name']} ({inner})" if inner else str(value["name"])
        return ", ".join(f"{k} = {_scalar(v)}" for k, v in value.items() if v is not None)
    if isinstance(value, list) and any(isinstance(v, dict) for v in value):
        return "; ".join(_readable(v) for v in value)
    return _scalar(value)


def _scalar(value: object) -> str:
    if isinstance(value, dict):
        return "{" + _readable(value) + "}"
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _comparison_text(facts: RunReportFacts) -> str:
    """Return how the run compares with the other run: setup differences and final metrics."""
    other = facts.compared_with
    assert other is not None
    # The seed has its own sentence: replicates differ only by it, and their setups agree.
    differs = sorted(k for k in {*facts.setup, *other.setup}
                     if k != "seed" and facts.setup.get(k) != other.setup.get(k))  # fmt: skip
    lines = [
        f"Compared with **{other.summary.name}** ({other.summary.method}, "
        f"{other.summary.members} members, {other.summary.simulator_runs} simulator runs)."
    ]
    lines.append(
        "Setup differences: " + ", ".join(differs) + "." if differs else "The setups agree."
    )
    # From the snapshots, so that no answer has to say it did not check (#237).
    ours, theirs = facts.setup.get("seed"), other.setup.get("seed")
    if ours is not None and theirs is not None and ours != theirs:
        lines.append(f"The seeds differ ({ours} here, {theirs} there).")
    elif ours is not None and theirs is not None:
        # The same root seed draws the same numbers only for the same draws (a review).
        alike = all(facts.setup.get(k) == other.setup.get(k)
                    for k in ("method", "ensemble_size", "prior"))  # fmt: skip
        lines.append(f"Both runs used seed {ours}" + (
            ", with the same method, ensemble size and prior: the same prior ensemble."
            if alike else "."))  # fmt: skip
    for metric, by_label in facts.summary.posterior_metrics.items():
        theirs = other.summary.posterior_metrics.get(metric, {})
        for label, value in by_label.items():
            if label in theirs:
                name = metric if label == "all" else f"{metric} ({label})"
                lines.append(f"- **{name}**: {_fmt(value)} here, {_fmt(theirs[label])} there.")
    return "\n".join(lines)


def study_rules(facts: StudyReportFacts) -> NarrativeReport:
    """Return the template report of a study."""
    c = facts.comparison
    sections = [
        Section(
            heading="Overview",
            text=f"Study **{c.study}**: {c.replicates} replicate(s) per variant, compared at "
            f"each run's posterior against **{c.reference}**"
            + (
                f" within each method, and each method against **{c.reference_method}** with "
                "the same variant"
                if c.reference_method
                else ""
            )
            + " (paired by replicate; Wilcoxon signed-rank p-values)."
            + (f" Without results yet: {', '.join(c.missing)}." if c.missing else ""),
        )
    ]
    numbers = []
    for m in c.metrics:
        lines = [
            f"- {v.label()}: mean {_fmt(v.mean)}"
            + (f" ± {_fmt(v.std)}" if v.std is not None else "")
            + f" (n = {v.n})"
            for v in m.variants
        ]
        for d in m.differences:
            significant = d.p_value is not None and d.p_value < SIGNIFICANCE
            lines.append(
                f"- {d.label()} minus {d.reference_label()}: {_fmt(d.mean_difference)}"
                + (f", p = {_fmt(d.p_value)}" if d.p_value is not None else "")
                + (" (significant)" if significant else "")
            )
            numbers.append(KeyNumber(
                name=f"{m.metric} {d.label()} - {d.reference_label()}",
                value=d.mean_difference,
                source=f"comparison {m.case}/{m.metric} differences {d.label()} - "
                f"{d.reference_label()}",
            ))  # fmt: skip
        for v in m.variants:
            numbers.append(KeyNumber(
                name=f"{m.metric} {v.label()} mean", value=v.mean,
                source=f"comparison {m.case}/{m.metric} variants {v.label()}",
            ))  # fmt: skip
        sections.append(Section(heading=f"{m.case}: {m.metric}", text="\n".join(lines)))
    if c.notes:
        sections.append(Section(heading="Notes", text="\n".join(f"- {n}" for n in c.notes)))
    return NarrativeReport(title=f"Study {c.study}", sections=sections, key_numbers=numbers)


# ---------------------------------------------------------------------- reconcile


def reconcile(
    report: NarrativeReport, numbers: Iterable[float], rtol: float = NUMBER_RTOL
) -> tuple[NarrativeReport, list[str]]:
    """Drop key numbers that match no fact within ``rtol``; return the report and what went."""
    facts = list(numbers)
    kept, dropped = [], []
    for number in report.key_numbers:
        if any(close(number.value, fact, rtol) for fact in facts):
            kept.append(number)
        else:
            dropped.append(
                f"{number.name} = {number.value:g} (said to be from {number.source}) matches "
                "no fact"
            )
    if not dropped:
        return report, []
    return report.model_copy(update={"key_numbers": kept}), dropped


def markdown(report: NarrativeReport, *, corrections: Iterable[str] = (), by: str = "") -> str:
    """Return the report as Markdown, with its key numbers and any corrections."""
    parts = [f"# {report.title}\n"]
    for section in report.sections:
        parts.append(f"## {section.heading}\n\n{section.text.strip()}\n")
    if report.key_numbers:
        rows = "\n".join(f"| {n.name} | {n.value:.6g} | {n.source} |" for n in report.key_numbers)
        parts.append(f"## Key numbers\n\n| Quantity | Value | Source |\n|---|---|---|\n{rows}\n")
    corrections = list(corrections)
    if corrections:
        parts.append(
            "## Corrections\n\nThese numbers of the model's report matched no fact and were "
            "removed; check the prose that uses them:\n\n"
            + "\n".join(f"- {c}" for c in corrections)
            + "\n"
        )
    if by:
        parts.append(f"---\n\n*Written by APEX's report agent ({by}).*\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------- the agent

SYSTEM = (
    "You write the results report of an APEX history-matching run or study for engineers "
    "and researchers. You get facts computed by APEX. Write from them only: every number "
    "in your prose must come from the facts, and list each one in key_numbers with its "
    "source. Be concise and precise; explain what the numbers mean (1 is the noise level "
    "of normalized_data_mismatch; rmse is error against the true model; a lower data "
    "mismatch is not "
    "better when it is below the noise). Do not invent causes; use the diagnosis given. "
    "Open with a short summary of the key results; for a run, describe its setup (method, "
    "strategy, localization, damping, prior, data, simulator) from the facts' setup, and "
    "when the facts have compared_with, add a comparison with that run."
)
_RULE = "the report agent's template"


@dataclass(frozen=True)
class ReportResult:
    """A written report.

    Attributes
    ----------
    report
        The report (the model's, reconciled, or the template's).
    text
        It as Markdown.
    decided_by, reason, trace
        See :class:`~apex.agents.common.Decision`.
    corrections
        The key numbers :func:`reconcile` dropped.
    """

    report: NarrativeReport
    text: str
    decided_by: str
    reason: str | None
    trace: Path
    corrections: tuple[str, ...] = ()


class Reporter:
    """Writes reports (see the module docstring).

    Parameters
    ----------
    service
        The services (to find runs and studies).
    provider
        The model; ``None`` for the template.
    settings
        Bounds of the model's task.
    """

    def __init__(
        self,
        service: ApexService,
        provider: LLMProvider | None,
        *,
        settings: LoopSettings | None = None,
    ) -> None:
        self.service = service
        self.provider = provider
        self.settings = settings or LoopSettings(max_turns=4, max_tokens=4096)

    def run_report(
        self, target: str | Path, compare_with: str | Path | None = None
    ) -> ReportResult:
        """Write the report of a run (a directory, run id or job id).

        ``compare_with`` names another run; the report then compares the two.
        """
        from apex.services.api import summarise_run

        resolve = Diagnostician(self.service, None).resolve
        run_path, _ = resolve(target)
        facts = run_facts(self.service, run_path)
        if compare_with is not None:
            other, _ = resolve(compare_with)
            facts.compared_with = ComparedRun(summary=summarise_run(other), setup=run_setup(other))
        return self._write(facts, run_rules(facts), CallRecorder(run_path, "report"))

    def study_report(self, name: str) -> ReportResult:
        """Write the report of a study."""
        facts = StudyReportFacts(comparison=self.service.compare(name))
        recorder = CallRecorder(self.service.root / "studies" / name, "report")
        return self._write(facts, study_rules(facts), recorder)

    def write(
        self, facts: BaseModel, template: NarrativeReport, recorder: CallRecorder
    ) -> ReportResult:
        """Write a report from ``facts``: the model's, or ``template``.

        The model's key numbers are held to the facts, as for runs and studies.
        """
        return self._write(facts, template, recorder)

    def _write(
        self, facts: BaseModel, template: NarrativeReport, recorder: CallRecorder
    ) -> ReportResult:
        system, recorder.advice = advise_prompt(SYSTEM, "report", self.service.root)
        decision = decide(
            self.service,
            self.provider,
            recorder,
            task=f"Facts:\n{facts.model_dump_json(indent=1)}\n\nWrite the report.",
            response_model=NarrativeReport,
            rules=template,
            rule=_RULE,
            thresholds={"number_rtol": NUMBER_RTOL, "significance": SIGNIFICANCE},
            client="report",
            settings=self.settings,
            system=system,
        )
        report = decision.value
        corrections: list[str] = []
        if decision.decided_by == "model":
            report, corrections = reconcile(report, fact_numbers(facts))
            if corrections:
                recorder.record_action(
                    "dropped key numbers that match no fact", corrections=corrections
                )
        by = {"model": "model", "fallback": "template, the model failed", "rules": "template"}[
            decision.decided_by
        ]
        text = markdown(report, corrections=corrections, by=by)
        return ReportResult(
            report, text, decision.decided_by, decision.reason, decision.trace, tuple(corrections)
        )

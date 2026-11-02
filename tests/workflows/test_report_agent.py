"""The report agent, on linear cases and scripted models (no live model).

Independent reference: the template's numbers are compared with the run's history
(``history.json``), which the report's facts do not read directly.
"""

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol

import pytest
from typer.testing import CliRunner

from apex.agents.providers.stub import StubProvider
from apex.agents.reporter import (
    KeyNumber,
    NarrativeReport,
    Reporter,
    RunReportFacts,
    Section,
    StudyReportFacts,
    fact_numbers,
    reconcile,
    run_facts,
    run_rules,
    run_setup,
    run_stop,
    run_warnings,
    study_rules,
)
from apex.cli.main import app
from apex.core.llm import LLMUnavailable
from apex.domain.services import (
    Comparison,
    MetricComparison,
    PairedDifference,
    RunSteps,
    RunSummary,
    VariantStats,
)
from apex.services import composition
from apex.services.api import ApexService
from apex.storage.run_directory import RunDirectory


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path


CaseFactory = Callable[..., Case]


def _run(linear_case: CaseFactory) -> Path:
    case = linear_case(ensemble_size=20)
    composition.start(case.config)
    return case.directory / "runs" / "case"


def test_the_template_reports_the_trajectory(tmp_path: Path, linear_case: CaseFactory) -> None:
    run_dir = _run(linear_case)

    result = Reporter(ApexService(tmp_path / "service"), None).run_report(run_dir)

    history = RunDirectory(run_dir).history()
    first, last = history[0].metrics, history[-1].metrics
    assert first is not None
    assert last is not None
    numbers = {n.name: n.value for n in result.report.key_numbers}
    assert numbers["rmse at step 0"] == pytest.approx(first["rmse"]["all"])
    assert numbers["rmse at step 4"] == pytest.approx(last["rmse"]["all"])
    assert numbers["data_mismatch at step 4"] == pytest.approx(history[-1].data_mismatch_mean)
    assert result.decided_by == "rules"
    assert "# History matching of case" in result.text
    assert "## Key numbers" in result.text
    assert "Corrections" not in result.text


def test_a_run_report_opens_with_a_summary_and_gives_the_setup(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """A reported issue (#178): asked for the setup and parameters, the report had none, and it
    could not compare with another run."""
    run_dir = _run(linear_case)
    config = json.loads((run_dir / "config.json").read_text())

    result = Reporter(ApexService(tmp_path / "service"), None).run_report(run_dir)

    headings = [s.heading for s in result.report.sections]
    assert headings[:3] == ["Summary", "Overview", "Setup"]
    setup = next(s.text for s in result.report.sections if s.heading == "Setup")
    # Prose, not JSON (release-tester U8).
    assert f"**Method**: {config['method']['name']} (inflation = [" in setup
    assert '{"' not in setup
    assert "**Ensemble size**: 20" in setup
    assert "**Prior**" in setup
    assert "**Simulator**" in setup
    assert "Comparison" not in headings

    compared = Reporter(ApexService(tmp_path / "service"), None).run_report(run_dir, run_dir)

    text = next(s.text for s in compared.report.sections if s.heading == "Comparison")
    assert "The setups agree." in text
    seed = config["run"]["seed"]
    assert (
        f"Both runs used seed {seed}, with the same method, ensemble size and prior: the "
        "same prior ensemble."
    ) in text  # #237
    assert "- **rmse**:" in text

    other = tmp_path / "other_run"
    shutil.copytree(run_dir, other)
    changed = json.loads((other / "config.json").read_text())
    changed["simulator"]["name"] = "another"
    changed["run"]["seed"] = seed + 1
    (other / "config.json").write_text(json.dumps(changed))
    against = Reporter(ApexService(tmp_path / "service"), None).run_report(run_dir, other)
    text = next(s.text for s in against.report.sections if s.heading == "Comparison")
    assert "Setup differences: simulator." in text
    assert f"The seeds differ ({seed} here, {seed + 1} there)." in text


def test_a_run_setup_shows_no_server_paths_and_names_prior_groups(tmp_path: Path) -> None:
    """The review of #178: paths nested in options (GMAC's constraints, lists of files)
    reached the report; grouped variables showed the default generator."""
    config = {
        "method": {"name": "gies", "strategy": {"name": "gmac", "options": {
            "constraints": [{"function": {"options": {"reference_file": "/srv/x/ref.npz"}}}],
            "files": ["/srv/x/a.npz", "relative/b.npz"]}}},
        "prior": {"ensemble_size": 10, "generator": {"name": "gaussian"},
                  "groups": {"latent": {"name": "gstools_pgs", "variables": ["z1", "z2"],
                                        "options": {"grid_file": "/srv/x/grid.npz"}}}},
        "simulator": {"name": "opm_flow"},
    }  # fmt: skip
    (tmp_path / "config.json").write_text(json.dumps(config))

    setup = run_setup(tmp_path)

    assert "/srv" not in json.dumps(setup)
    options = setup["strategy"]["options"]  # type: ignore[index]
    assert options["files"] == ["a.npz", "relative/b.npz"]  # only absolute paths reduced
    groups = setup["prior"]["groups"]  # type: ignore[index]
    assert groups["latent"]["name"] == "gstools_pgs"
    assert groups["latent"]["variables"] == ["z1", "z2"]


def test_numbers_the_facts_do_not_hold_are_dropped_and_shown(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    run_dir = _run(linear_case)
    history = RunDirectory(run_dir).history()
    rmse = history[-1].metrics["rmse"]["all"]  # type: ignore[index]
    answer = NarrativeReport(
        title="Case results",
        sections=[Section(heading="Result", text=f"The RMSE fell to {rmse:.3g}.")],
        key_numbers=[
            KeyNumber(name="posterior rmse", value=float(f"{rmse:.3g}"), source="steps"),
            KeyNumber(name="invented", value=123456.0, source="nowhere"),
        ],
    )
    stub = StubProvider([answer.model_dump_json()])

    result = Reporter(ApexService(tmp_path / "service"), stub).run_report(run_dir)

    assert result.decided_by == "model"
    assert [n.name for n in result.report.key_numbers] == ["posterior rmse"]  # rounded: kept
    assert "invented" in result.corrections[0]
    assert "## Corrections" in result.text
    assert "invented = 123456" in result.text
    assert result.trace.parent == run_dir / "agents"
    [request] = stub.requests
    assert "submit_run" not in [t.name for t in request.tools]


def test_reconcile_uses_a_relative_tolerance() -> None:
    report = NarrativeReport(
        title="t",
        sections=[Section(heading="h", text="x")],
        key_numbers=[
            KeyNumber(name="a", value=1.004, source="s"),
            KeyNumber(name="b", value=1.006, source="s"),
            KeyNumber(name="zero", value=0.0, source="s"),
        ],
    )

    kept, dropped = reconcile(report, [1.0, 0.0])

    assert [n.name for n in kept.key_numbers] == ["a", "zero"]
    assert len(dropped) == 1


def test_numbers_in_the_findings_evidence_are_facts() -> None:
    """Release-tester N30: the monitor's and the diagnosis' findings state their numbers in
    their evidence; a key number citing them is true and kept."""
    from pydantic import BaseModel

    class Facts(BaseModel):
        monitor: dict[str, object]
        causes: list[dict[str, str]]

    collapse = {"code": "ensemble_collapse", "evidence": "below 0.05: permx 0.0338"}
    facts = Facts(
        monitor={"findings": [collapse]},
        causes=[{"code": "overcorrection", "evidence": "permx 78.8% (> 10%)"}],
    )
    report = NarrativeReport(
        title="t",
        sections=[Section(heading="h", text="x")],
        key_numbers=[
            KeyNumber(name="spread", value=0.0338, source="monitor.findings"),
            KeyNumber(name="beyond 3 std (%)", value=78.8, source="causes"),
            KeyNumber(name="invented", value=0.5, source="nowhere"),
        ],
    )

    kept, dropped = reconcile(report, fact_numbers(facts))

    assert [n.name for n in kept.key_numbers] == ["spread", "beyond 3 std (%)"]
    assert len(dropped) == 1


def test_a_study_report_marks_significant_differences() -> None:
    comparison = Comparison(
        study="loc",
        replicates=5,
        reference="baseline",
        metrics=[
            MetricComparison(
                case="c",
                metric="rmse:all",
                variants=[
                    VariantStats(variant="baseline", n=5, mean=0.5, std=0.1, minimum=0.4,
                                 maximum=0.6, values=[0.4, 0.5, 0.5, 0.5, 0.6]),
                    VariantStats(variant="adaptive", n=5, mean=0.3, std=0.05, minimum=0.25,
                                 maximum=0.35, values=[0.25, 0.3, 0.3, 0.3, 0.35]),
                ],
                differences=[
                    PairedDifference(variant="adaptive", reference="baseline", n=5,
                                     mean_difference=-0.2, standard_error=0.02, p_value=0.031),
                ],
            )
        ],
    )  # fmt: skip

    report = study_rules(StudyReportFacts(comparison=comparison))

    text = report.sections[1].text
    assert "adaptive minus baseline: -0.2, p = 0.031 (significant)" in text
    assert {n.value for n in report.key_numbers} == {-0.2, 0.5, 0.3}


def test_an_unavailable_model_gets_the_template(tmp_path: Path, linear_case: CaseFactory) -> None:
    run_dir = _run(linear_case)
    service = ApexService(tmp_path / "service")

    result = Reporter(service, StubProvider([LLMUnavailable("down")])).run_report(run_dir)

    assert result.decided_by == "fallback"
    assert "template, the model failed" in result.text


def test_the_command_writes_the_narrative(tmp_path: Path, linear_case: CaseFactory) -> None:
    run_dir = _run(linear_case)

    result = CliRunner().invoke(
        app, ["agent", "report", str(run_dir), "--no-llm", "--root", str(tmp_path / "service")]
    )

    assert result.exit_code == 0, result.output
    narrative = run_dir / "report" / "narrative.md"
    assert narrative.read_text().startswith("# History matching of case")
    both = CliRunner().invoke(app, ["agent", "report", str(run_dir), "--study", "x", "--no-llm"])
    assert both.exit_code == 1
    decision = next((run_dir / "agents").glob("report-*-decision.json"))
    assert json.loads(decision.read_text())["decided_by"] == "rules"


def _gies_run(linear_case: CaseFactory, max_iterations: int) -> Path:
    """Run GIES on the linear case for ``max_iterations`` (1: it cannot converge)."""
    case = linear_case(
        ensemble_size=20,
        method=f'name = "gies"\nmax_iterations = {max_iterations}\n'
        'strategy = { name = "rlmmac" }',
    )
    composition.start(case.config)
    return case.directory / "runs" / "case"


def test_a_run_that_did_not_converge_says_why_and_warns(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """A reported issue (#259): a report said a run did not converge while its Warnings said
    "None were reported", and the assistant gave a cause no record held. The stop reason is
    a fact read from the run, and not converging is a warning in code."""
    run_dir = _gies_run(linear_case, max_iterations=1)
    service = ApexService(tmp_path / "service")

    facts = run_facts(service, run_dir)

    assert facts.summary.converged is False
    assert facts.stop is not None
    assert (facts.stop.code, facts.stop.recorded) == ("iteration_budget", True)
    assert "max_iterations = 1" in facts.stop.text
    assert facts.warnings[0].startswith("The run did not converge. It used its whole iteration")

    result = Reporter(service, None).run_report(run_dir)

    sections = {s.heading: s.text for s in result.report.sections}
    assert facts.stop.text in sections["Overview"]
    assert sections["Warnings"].startswith("- The run did not converge.")
    assert "None" not in sections["Warnings"]


def test_a_converged_or_scheduled_run_has_no_stop_warning(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """#259: ES-MDA has no convergence test; its end is its schedule, not a warning."""
    run_dir = _run(linear_case)

    facts = run_facts(ApexService(tmp_path / "service"), run_dir)

    assert facts.stop is not None
    assert facts.stop.code == "schedule_complete"
    assert "fixed schedule of 4 update step(s)" in facts.stop.text
    assert not any("did not" in w for w in facts.warnings)


def _write_run(
    path: Path,
    *,
    complete: bool,
    converged: bool | None,
    events: list[dict[str, object]],
    method: dict[str, object] | None = None,
    last_accepted: bool = True,
    damping: float | None = None,
) -> Path:
    """Write the records :func:`run_stop` reads: four records, the last ``last_accepted``."""
    path.mkdir()
    method = method or {"name": "gies", "max_iterations": 5, "damping": {
        "name": "rlmmac", "options": {"data_mismatch_tolerance": 0.02}}}  # fmt: skip
    (path / "config.json").write_text(json.dumps({"method": method}))
    (path / "summary.json").write_text(json.dumps({"complete": complete, "converged": converged}))
    (path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    history = [{"iteration": k, "data_mismatch_mean": 1.0, "data_mismatch_std": 0.1,
                "n_members": 10, "accepted": k < 3 or last_accepted} for k in range(4)]  # fmt: skip
    (path / "history.json").write_text(json.dumps(history))
    (path / "state.json").write_text(json.dumps({"algorithm": {"damping": damping}}))
    return path


def _summary(state: Literal["complete", "stopped"], converged: bool | None) -> RunSummary:
    return RunSummary(
        run_id=None, path="", name="r", method="gies", state=state, converged=converged,
        iterations=[0, 1, 2, 3], accepted_iterations=[0, 1, 3], posterior_iteration=3,
        selected_iteration=None, members=10, data_mismatch={}, metrics={},
        posterior_metrics={}, simulator_runs=40,
    )  # fmt: skip


def test_the_stop_reason_comes_from_the_records_or_is_not_recorded(tmp_path: Path) -> None:
    """#259: each way a run ends is read from its records; an unrecorded one says so."""
    stalled = _write_run(tmp_path / "stalled", complete=True, converged=True,
                         events=[{"event": "trial", "iteration": 3},
                                 {"event": "stalled", "iteration": 3, "stalls": 3}])  # fmt: skip
    assert run_stop(stalled, _summary("complete", True)).code == "stalled"

    stopped = _write_run(tmp_path / "stopped", complete=False, converged=False, events=[])
    stop = run_stop(stopped, _summary("stopped", False))
    assert (stop.code, stop.recorded) == ("stopped", False)
    assert "not recorded" in stop.text
    assert run_warnings(_summary("stopped", False), stop, [])[0].startswith(
        "The run did not finish."
    )

    asked = _write_run(tmp_path / "asked", complete=False, converged=False,
                       events=[{"event": "stopped_on_request", "iteration": 3}])  # fmt: skip
    assert run_stop(asked, _summary("stopped", False)).code == "stopped_on_request"
    resumed = _write_run(tmp_path / "resumed", complete=False, converged=False,
                         events=[{"event": "stopped_on_request"}, {"event": "resume"}])  # fmt: skip
    assert run_stop(resumed, _summary("stopped", False)).code == "stopped"
    limit = _write_run(tmp_path / "limit", complete=False, converged=False,
                       events=[{"event": "max_steps", "iteration": 3, "max_steps": 2}])  # fmt: skip
    stop = run_stop(limit, _summary("stopped", False))
    assert (stop.code, stop.recorded) == ("step_limit", True)
    assert "--max-steps" in stop.text


def test_a_converged_run_says_which_test_stopped_it(tmp_path: Path) -> None:
    """The review of #259: "converged" is not always an improving trial within the
    tolerance: a rejected one may be (converge_on_rejection, GN-EnRML's fixed step), and LM
    stops at its maximum damping. Each is told from the records."""
    improved = run_stop(_write_run(tmp_path / "improved", complete=True, converged=True,
                                   events=[]), _summary("complete", True))  # fmt: skip
    assert improved.text == ("It met its convergence test (relative data-mismatch tolerance 0.02).")
    twice = run_stop(_write_run(tmp_path / "twice", complete=True, converged=True,
                                events=[{"event": "converged", "iteration": 3}]),
                     _summary("complete", True))  # fmt: skip
    assert twice.text.endswith("two small improving trials in a row.")

    rejected = _write_run(tmp_path / "rejected", complete=True, converged=True, events=[],
                          last_accepted=False)  # fmt: skip
    stop = run_stop(rejected, _summary("complete", True))
    assert stop.text.startswith("A rejected trial changed the mean data mismatch")
    gn = _write_run(tmp_path / "gn", complete=True, converged=True, events=[],
                    method={"name": "gn_enrml", "max_iterations": 5},
                    last_accepted=False)  # fmt: skip
    assert run_stop(gn, _summary("complete", True)).text.startswith("A rejected trial")

    lm = {"name": "lm_enrml", "max_iterations": 5}
    at_max = _write_run(tmp_path / "lm_max", complete=True, converged=True, events=[],
                        method=lm, damping=1e10)  # fmt: skip
    stop = run_stop(at_max, _summary("complete", True))
    assert stop.code == "damping_maximum"
    assert "1e+10 >= 1e+10" in stop.text
    below = _write_run(tmp_path / "lm_below", complete=True, converged=True, events=[],
                       method=lm, damping=0.5)  # fmt: skip
    assert run_stop(below, _summary("complete", True)).code == "converged"

    # Events before the last resume belong to an earlier process (a crash).
    crashed: list[dict[str, object]] = [{"event": "stalled", "iteration": 3}, {"event": "resume"}]
    stale = _write_run(tmp_path / "stale", complete=True, converged=True, events=crashed)
    assert run_stop(stale, _summary("complete", True)).code == "converged"
    # cv_stop alone (before its checkpoint) is no stop; summary.json's flag decides.
    cv = _write_run(tmp_path / "cv", complete=True, converged=False,
                    events=[{"event": "cv_stop", "iteration": 3}])  # fmt: skip
    assert run_stop(cv, _summary("complete", False)).code == "iteration_budget"


def _steps() -> RunSteps:
    return RunSteps(workflow="history_matching", name="r", path="", method="gies",
                    status="stopped", forecasts=4, steps=[], metrics=[])  # fmt: skip


def test_a_running_run_is_not_reported_as_stopped(tmp_path: Path) -> None:
    """The review of #259: a run still running has complete = false, and was reported as
    stopped for an unrecorded reason, with a "did not finish" warning."""
    from apex.storage.locks import exclusive

    run = _write_run(tmp_path / "running", complete=False, converged=False, events=[])
    with exclusive(run):
        stop = run_stop(run, _summary("stopped", False))
        assert (stop.code, stop.recorded, stop.text) == (
            "running", True, "The run is still running.",
        )  # fmt: skip
        assert run_warnings(_summary("stopped", False), stop, []) == []
        # The report says running throughout (the re-review of #259).
        facts = RunReportFacts(summary=_summary("stopped", False), steps=_steps(), monitor={},
                               causes=[], stop=stop, warnings=[])  # fmt: skip
        text = {x.heading: x.text for x in run_rules(facts).sections}
        assert text["Summary"].startswith("gies on 10 members, still running")
        assert "stopped" not in text["Summary"]
        assert "did not converge" not in text["Summary"]
        assert text["Overview"].startswith("gies run **r** is running:")
        assert text["Overview"].endswith("The run is still running.")
        assert text["Warnings"].startswith("None so far: the run is still running")
    assert run_stop(run, _summary("stopped", False)).code == "stopped"


def test_a_step_limited_run_says_so(tmp_path: Path, linear_case: CaseFactory) -> None:
    """The review of #259: a run stopped by --max-steps read "not recorded"; the workflow
    logs the limit as an event."""
    case = linear_case(ensemble_size=20, method='name = "gies"\nmax_iterations = 5\n'
                       'strategy = { name = "rlmmac" }')  # fmt: skip
    composition.start(case.config, max_steps=1)
    run_dir = case.directory / "runs" / "case"

    facts = run_facts(ApexService(tmp_path / "service"), run_dir)

    assert facts.stop is not None
    assert (facts.stop.code, facts.stop.recorded) == ("step_limit", True)
    assert facts.warnings[0].startswith("The run did not finish. It stopped after its step")


def test_a_cross_validation_stop_is_no_convergence_warning(
    tmp_path: Path, well_case: Callable[..., Case]
) -> None:
    """The review of #259: a run stopped by cross validation (converged false) was warned
    of as not converging, and the model's report then got a false correction."""
    case = well_case(validation='[validation]\ncv = { wells = ["P7", "P8"] }\n'
                     'stopping = { name = "patience", options = { patience = 2 } }')  # fmt: skip
    composition.start(case.config)
    run_dir = case.directory / "runs" / "case"
    service = ApexService(tmp_path / "service")

    facts = run_facts(service, run_dir)

    assert facts.stop is not None
    assert facts.stop.code == "cross_validation"
    assert not any(w.startswith("The run did not") for w in facts.warnings)
    answer = NarrativeReport(title="Report", sections=[
        Section(heading="Overview", text="Cross validation stopped it."),
        Section(heading="Warnings", text="None."),
    ])  # fmt: skip
    stub = StubProvider([answer.model_dump_json()])
    result = Reporter(service, stub).run_report(run_dir)
    assert not any("did not converge" in c for c in result.corrections)


def test_an_adaptive_rows_run_reports_its_localization(tmp_path: Path) -> None:
    """The review of #261: the setup of an adaptive_rows run showed no localization."""
    run = tmp_path / "rows"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"method": {
        "name": "esmda", "strategy": {"name": "adaptive_rows", "options": {}},
        "localization": None}}))  # fmt: skip

    setup = run_setup(run)

    assert list(setup)[:3] == ["method", "strategy", "localization"]
    assert setup["localization"] == "within the strategy adaptive_rows (it localizes by itself)"


def test_a_model_report_saying_no_warnings_gets_the_recorded_ones(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """#259: the model's "Warnings: None were reported" of a run that did not converge is
    replaced by the facts' warnings, and listed as a correction."""
    run_dir = _gies_run(linear_case, max_iterations=1)
    answer = NarrativeReport(
        title="Report",
        sections=[
            Section(heading="Summary", text="The run did not converge."),
            Section(heading="Warnings", text="None were reported."),
        ],
    )
    stub = StubProvider([answer.model_dump_json()])

    result = Reporter(ApexService(tmp_path / "service"), stub).run_report(run_dir)

    assert result.decided_by == "model"
    sections = {s.heading: s.text for s in result.report.sections}
    assert sections["Warnings"].startswith("- The run did not converge. It used its whole")
    assert "None were reported" not in result.text
    assert any("left out recorded warnings" in c for c in result.corrections)
    [request] = stub.requests
    assert "stop.recorded" in request.messages[0].content


def test_a_model_report_keeps_the_stop_reason_and_accurate_warnings(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """The review of #259: the model's overview gets the facts' stop reason; a warnings
    section that mentions every warning keeps its prose; "converged" is not "did not
    converge"."""
    run_dir = _gies_run(linear_case, max_iterations=1)
    service = ApexService(tmp_path / "service")
    facts = run_facts(service, run_dir)
    assert facts.stop is not None
    others = [w for w in facts.warnings if not w.startswith("The run did not")]
    codes = " ".join(w.split(" (")[0] for w in others)

    def answer(warning: str) -> str:
        return NarrativeReport(title="Report", sections=[
            Section(heading="Overview", text="A GIES run."),
            Section(heading="Warnings", text=f"{warning} {codes}".strip()),
        ]).model_dump_json()  # fmt: skip

    stub = StubProvider([answer("The run did not converge; more iterations may help.")])
    result = Reporter(service, stub).run_report(run_dir)
    sections = {s.heading: s.text for s in result.report.sections}
    assert sections["Overview"] == f"A GIES run. {facts.stop.text}"
    assert sections["Warnings"].startswith("- The run did not converge. It used")
    # The model's warning prose is dropped, even when complete: it may carry an inferred
    # cause, the #259 failure (the re-review).
    assert "more iterations may help" not in result.text
    assert not any("left out" in c for c in result.corrections)

    stub = StubProvider([answer("It converged.")])
    result = Reporter(service, stub).run_report(run_dir)
    assert any("left out recorded warnings" in c for c in result.corrections)

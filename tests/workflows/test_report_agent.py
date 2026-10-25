"""The report agent, on linear cases and scripted models (no live model).

Independent reference: the template's numbers are compared with the run's history
(``history.json``), which the report's facts do not read directly.
"""

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import pytest
from typer.testing import CliRunner

from apex.agents.providers.stub import StubProvider
from apex.agents.reporter import (
    KeyNumber,
    NarrativeReport,
    Reporter,
    Section,
    StudyReportFacts,
    fact_numbers,
    reconcile,
    run_setup,
    study_rules,
)
from apex.cli.main import app
from apex.core.llm import LLMUnavailable
from apex.domain.services import Comparison, MetricComparison, PairedDifference, VariantStats
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

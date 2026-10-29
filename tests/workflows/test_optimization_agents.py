"""The run monitor and the report agent on optimizations, with the stub LLM."""

from pathlib import Path
from typing import Any

import pytest
import tomli_w

from apex.agents.monitor import Finding, MonitorReport
from apex.agents.optimization import (
    OptimizationFacts,
    OptimizationMonitor,
    is_optimization_dir,
    optimization_report,
    optimization_rules,
)
from apex.agents.optimization_advice import OptimizationPlanResult
from apex.agents.providers.stub import StubProvider
from apex.services import optimization
from apex.services.api import ApexService
from apex.services.settings import MonitorSection

TIMES = [f"{2001 + k}-01-01" for k in range(4)]


def _optimization(tmp_path: Path) -> Path:
    config = {
        "optimization": {"name": "toy", "seed": 2, "output_dir": str(tmp_path / "opt")},
        "controls": {"variables": [{"name": "inj", "initial": [300.0, 250.0], "lower": 10.0,
                                    "upper": 400.0, "variance": 100.0}]},
        "objective": {"name": "npv", "options": {"start": "2000-01-01", "times": TIMES,
                                                 "wop": 1.0, "wwi": 1.0, "obj_scaling": -1.0}},
        "optimizer": {"name": "enopt", "options": {"ne": 4, "maxiter": 3, "alpha": 0.1}},
        "simulator": {"name": "toy_reservoir"},
    }  # fmt: skip
    path = tmp_path / "toy.toml"
    path.write_text(tomli_w.dumps(config), encoding="utf-8")
    return optimization.optimize(path).run_dir


def _facts(**changes: object) -> OptimizationFacts:
    base: dict[str, object] = {
        "path": "/x", "name": "toy", "state": "running", "optimizer": "enopt",
        "objective": "npv", "accepted_steps": 5, "start_objective": -10.0,
        "best_objective": -12.0, "last_objective": -12.0, "relative_improvement": 0.2,
        "window_improvement": 0.05, "evaluations": 40, "failed_evaluations": 0,
        "message": None,
    }  # fmt: skip
    return OptimizationFacts.model_validate(base | changes)


def test_the_rules_find_stalls_failures_and_the_iteration_limit() -> None:
    thresholds = MonitorSection()
    assert optimization_rules(_facts(), thresholds).status == "healthy"
    stalled = optimization_rules(_facts(window_improvement=1e-5), thresholds)
    assert [f.code for f in stalled.findings] == ["stall"]
    assert stalled.recommendation == "investigate"
    failed = optimization_rules(_facts(failed_evaluations=2), thresholds)
    assert failed.status == "critical"
    limit = optimization_rules(
        _facts(state="finished", message="Maximum number of iterations reached"), thresholds
    )
    assert [f.code for f in limit.findings] == ["iteration_limit"]
    assert limit.status == "healthy"


def test_the_monitor_reports_on_a_real_optimization_and_holds_the_model_to_the_facts(
    tmp_path: Path,
) -> None:
    run_dir = _optimization(tmp_path)
    assert is_optimization_dir(run_dir)
    service = ApexService(tmp_path / "service")
    rules_only = OptimizationMonitor(service, None).check(run_dir)
    assert rules_only.decided_by == "rules"
    history = optimization.optimization_summary(run_dir).history
    assert rules_only.facts.start_objective == history[0]["objective"]
    assert rules_only.facts.evaluations > 0
    # The model claims failures the facts do not show, and wants a stop: both corrected.
    claim = MonitorReport(
        status="critical",
        recommendation="stop",
        findings=[Finding(code="failed_evaluations", severity="critical", evidence="3 failed")],
        summary="Something failed.",
    )
    stub = StubProvider([claim.model_dump_json()])
    checked = OptimizationMonitor(service, stub).check(run_dir)
    assert checked.decided_by == "model"
    assert checked.report.findings == []
    assert checked.report.recommendation == "investigate"
    assert any("failed_evaluations" in c for c in checked.corrections)


def test_the_report_of_an_optimization_quotes_its_facts(tmp_path: Path) -> None:
    run_dir = _optimization(tmp_path)
    service = ApexService(tmp_path / "service")
    result = optimization_report(service, None, run_dir)
    assert result.decided_by == "rules"
    summary = optimization.optimization_summary(run_dir)
    assert f"{summary.history[0]['objective']:.6g}" in result.text
    assert "| inj[0] |" in result.text
    with pytest.raises(Exception, match="not an optimization"):
        OptimizationMonitor(service, None).resolve("no-such-job")


def test_the_stall_window_stays_within_one_penalty_factor() -> None:
    """An EPF update doubles r: the window does not compare values across it (review)."""
    from apex.agents.optimization import optimization_facts
    from apex.domain.services import OptimizationSummary

    history = [
        {"kind": "start", "epf_iteration": 0, "objective": 10.0},
        {"kind": "step", "epf_iteration": 1, "objective": 8.0},
        {"kind": "step", "epf_iteration": 1, "objective": 6.0},
        {"kind": "epf_update", "epf_iteration": 2, "objective": 6.0},
        {"kind": "step", "epf_iteration": 2, "objective": 9.0},  # higher: r doubled
        {"kind": "step", "epf_iteration": 2, "objective": 8.5},
    ]
    summary = OptimizationSummary(
        path="/x", name="toy", state="running", optimizer="enopt", objective="quadratic",
        controls=["x"], history=history, result=None, evaluations=0,
    )  # fmt: skip
    facts = optimization_facts(summary, window=2)
    assert facts.window_improvement is None  # two steps under r_2 are not a window of 2 + 1
    facts = optimization_facts(summary, window=1)
    assert facts.window_improvement == pytest.approx((9.0 - 8.5) / 9.0)
    assert facts.accepted_steps == 4


def test_a_multi_objective_run_gets_no_window_stall_finding() -> None:
    """Review finding (2026-10-06): the rule read the representative's first objective,
    which may rightly worsen as an NSGA-II front spreads, so auto monitoring warned of
    stalls on healthy runs (a front-based rule is phase 2b's)."""
    from apex.agents.optimization import optimization_facts, optimization_rules
    from apex.domain.services import OptimizationSummary
    from apex.services.settings import MonitorSection

    history = [{"kind": "start" if k == 0 else "step", "iteration": k, "objective": 1.0 + k,
                "hypervolume": 2.0} for k in range(40)]  # fmt: skip
    summary = OptimizationSummary(
        path="/x", name="zdt1", state="running", optimizer="nsga2", objective="zdt1",
        controls=["x"], history=history, result=None, evaluations=0,
    )  # fmt: skip
    facts = optimization_facts(summary, window=30)
    assert facts.window_improvement is None
    assert "stall" not in {f.code for f in optimization_rules(facts, MonitorSection()).findings}


# ---------------------------------------------------------------------- diagnosis, planning
# (2026-10-04)


def test_the_diagnosis_of_an_optimization_at_its_iteration_limit(tmp_path: Path) -> None:
    """The rules: the limit is a cause, and resuming with more iterations a validated patch."""
    from apex.agents.optimization_advice import OptimizationDiagnostician

    run_dir = _optimization(tmp_path)
    service = ApexService(tmp_path / "service")
    result = OptimizationDiagnostician(service, None).diagnose(run_dir)
    assert result.decided_by == "rules"
    assert result.facts.monitor.message == "Maximum number of iterations reached"
    assert {c.code for c in result.diagnosis.causes} == {"iteration_limit"}
    assert result.diagnosis.verdict == "healthy"  # an info cause
    patches = [r.patch for r in result.diagnosis.recommendations if r.patch]
    assert patches == [{"optimizer": {"options": {"maxiter": 6}}}]
    assert result.facts.optimizer["name"] == "enopt"
    assert result.facts.controls == 2


def test_the_diagnosis_holds_the_model_to_the_facts(tmp_path: Path) -> None:
    """A cause the rules did not find is dropped, and a patch that does not validate loses
    its patch (the advice stays)."""
    from apex.agents.diagnosis import Cause, Diagnosis, Recommendation
    from apex.agents.optimization_advice import OptimizationDiagnostician

    run_dir = _optimization(tmp_path)
    service = ApexService(tmp_path / "service")
    claim = Diagnosis(
        verdict="failed",
        causes=[Cause(code="failed_evaluations", severity="critical", evidence="3 failed")],
        recommendations=[
            Recommendation(change="use a bigger ensemble", rationale="noise",
                           patch={"optimizer": {"options": {"ne": -5}}}),
            Recommendation(change="no such setting", rationale="none",
                           patch={"optimizer": {"options": {"bogus_option": 1}}}),
        ],
        summary="it failed badly",
    )  # fmt: skip
    result = OptimizationDiagnostician(service, StubProvider([claim.model_dump_json()])).diagnose(
        run_dir
    )
    assert result.decided_by == "model"
    assert not any(c.code == "failed_evaluations" for c in result.diagnosis.causes)
    assert all(r.patch is None for r in result.diagnosis.recommendations)
    assert result.diagnosis.verdict == "healthy"  # no cause kept to support "failed"
    assert len(result.corrections) == 4


def test_diagnosis_rules_cover_stalls_and_failures() -> None:
    from apex.agents.optimization_advice import OptimizationDiagnosisFacts, diagnosis_rules

    facts = OptimizationDiagnosisFacts(
        monitor=_facts(window_improvement=1e-5, failed_evaluations=2),
        optimizer={"name": "enopt", "options": {"ne": 4, "alpha": 0.2, "maxiter": 10}},
        controls=6, realizations=1, simulated=True,
        execution={"name": "process", "options": {"retries": 0}},
    )  # fmt: skip
    found = diagnosis_rules(facts, {"stall", "failed_evaluations"})
    assert found.verdict == "failed"
    patches = [r.patch for r in found.recommendations]
    assert {"execution": {"options": {"retries": 2}}} in patches
    assert {"optimizer": {"options": {"alpha": 0.1}}} in patches
    assert {"optimizer": {"options": {"ne": 8}}} in patches
    healthy = diagnosis_rules(facts.model_copy(update={"monitor": _facts()}), set())
    assert healthy.verdict == "healthy"
    assert healthy.recommendations == []


def _case(tmp_path: Path) -> Path:
    config = {
        "optimization": {"name": "toy", "seed": 2},
        "controls": {"variables": [{"name": "inj", "initial": [300.0, 250.0], "lower": 10.0,
                                    "upper": 400.0, "variance": 100.0}]},
        "objective": {"name": "npv", "options": {"start": "2000-01-01", "times": TIMES,
                                                 "wop": 1.0, "wwi": 1.0, "obj_scaling": -1.0}},
        "optimizer": {"name": "enopt", "options": {"ne": 4, "maxiter": 3, "alpha": 0.1}},
        "simulator": {"name": "toy_reservoir"},
    }  # fmt: skip
    path = tmp_path / "case.toml"
    path.write_text(tomli_w.dumps(config), encoding="utf-8")
    return path


def test_the_rules_plan_a_paired_optimizer_comparison_within_the_budget(tmp_path: Path) -> None:
    from apex.agents.optimization_advice import OptimizationPlanner, is_optimization_config
    from apex.services.optimization_study import load_optimization_study

    case = _case(tmp_path)
    assert is_optimization_config(case)
    service = ApexService(tmp_path / "service")
    result = OptimizationPlanner(service, None).plan(case, "which optimizer", budget=200)
    assert result.decided_by == "rules"
    assert set(result.proposal.methods) == {"enopt", "smcopt"}
    assert result.within_budget
    assert result.cost is not None
    assert result.cost <= 200
    # Paired seeds: as many as the budget allows for both optimizers (at most 5).
    per = result.facts.runs_per_cell
    assert per is not None
    assert result.proposal.replicates == max(1, min(5, (200 // per) // 2))
    study = tmp_path / "study.toml"
    study.write_text(result.toml(service.workspace(), tmp_path))
    assert load_optimization_study(study).study.methods.keys() == {"enopt", "smcopt"}


def test_an_over_budget_proposal_goes_back_and_the_rules_plan(tmp_path: Path) -> None:
    from apex.agents.optimization_advice import (
        OptimizationMethod,
        OptimizationPlanner,
        OptimizationStudyProposal,
        OptimizationVariantProposal,
    )
    from apex.services.settings import PlannerSection

    case = _case(tmp_path)
    service = ApexService(tmp_path / "service")
    greedy = OptimizationStudyProposal(
        name="greedy", hypothesis="everything", reasoning="more is better",
        methods={"big": OptimizationMethod(name="enopt", options={"ne": 50, "maxiter": 50})},
        variants=[OptimizationVariantProposal(name="base", rationale="as is")], replicates=5,
    )  # fmt: skip
    planner = OptimizationPlanner(
        service,
        StubProvider([greedy.model_dump_json()] * 3),
        thresholds=PlannerSection(max_revisions=1),
    )
    result = planner.plan(case, "compare", budget=100)
    assert result.decided_by == "fallback"
    assert len(result.revisions) == 2
    assert "budget" in result.revisions[0]
    assert result.within_budget


def test_the_commands_take_optimizations(tmp_path: Path) -> None:
    """``apex agent diagnose`` on an optimization's directory and ``apex agent plan`` on an
    optimization case use the optimization agents; the plan names its case without a
    server path."""
    import json
    import tomllib

    from typer.testing import CliRunner

    from apex.cli.main import app

    run_dir = _optimization(tmp_path)
    root = str(tmp_path / "service")
    diagnosed = CliRunner().invoke(
        app, ["agent", "diagnose", str(run_dir), "--no-llm", "--json", "--root", root]
    )
    assert diagnosed.exit_code == 0, diagnosed.output
    assert json.loads(diagnosed.stdout)["facts"]["optimizer"]["name"] == "enopt"
    out = tmp_path / "plans" / "planned.toml"
    out.parent.mkdir()
    planned = CliRunner().invoke(
        app,
        ["agent", "plan", str(_case(tmp_path)), "--goal", "which optimizer", "--budget", "200",
         "--no-llm", "--json", "--out", str(out), "--root", root],
    )  # fmt: skip
    assert planned.exit_code == 0, planned.output
    assert json.loads(planned.stdout)["within_budget"]
    study = tomllib.loads(out.read_text())["study"]
    assert study["kind"] == "optimization"
    assert study["cases"] == {"toy": "../case.toml"}


def test_a_valid_model_proposal_is_accepted(tmp_path: Path) -> None:
    from apex.agents.optimization_advice import (
        OptimizationMethod,
        OptimizationPlanner,
        OptimizationStudyProposal,
        OptimizationVariantProposal,
    )

    service = ApexService(tmp_path / "service")
    proposal = OptimizationStudyProposal(
        name="two_alphas", hypothesis="does a shorter step help", reasoning="paired seeds",
        methods={"enopt": OptimizationMethod(name="enopt", options={"ne": 4, "maxiter": 3,
                                                                    "alpha": 0.1})},
        variants=[OptimizationVariantProposal(name="base", rationale="as is"),
                  OptimizationVariantProposal(name="cheap_water", rationale="cost",
                                              patch={"objective": {"options": {"wwi": 0.5}}})],
        replicates=2,
    )  # fmt: skip
    result = OptimizationPlanner(service, StubProvider([proposal.model_dump_json()])).plan(
        _case(tmp_path), "does the water cost matter", budget=500
    )
    assert result.decided_by == "model"
    assert result.revisions == ()
    assert result.within_budget
    assert result.cost is not None
    assert result.spec.study.name == "two_alphas"


def test_the_rules_plan_is_cut_to_the_budget(tmp_path: Path) -> None:
    """Another optimizer may cost more than the case's: the rules' plan is costed and cut,
    first in replicates, then to the case's own optimizer (review of 2026-10-04)."""
    from apex.agents.optimization_advice import (
        OptimizationMethod,
        OptimizationPlanner,
        OptimizationStudyProposal,
        OptimizationVariantProposal,
        plan_facts,
    )
    from apex.services.settings import PlannerSection

    service = ApexService(tmp_path / "service")
    planner = OptimizationPlanner(service, None)
    facts = plan_facts(service, _case(tmp_path), budget=40, thresholds=PlannerSection())
    per = facts.runs_per_cell
    assert per is not None
    big = OptimizationStudyProposal(
        name="big", hypothesis="too much", reasoning="all of it", replicates=3,
        methods={"enopt": OptimizationMethod(name="enopt", options={"ne": 4, "maxiter": 3,
                                                                    "alpha": 0.1}),
                 "smcopt": OptimizationMethod(name="smcopt", options={"ne": 20, "maxiter": 9})},
        variants=[OptimizationVariantProposal(name="base", rationale="as is")],
    )  # fmt: skip
    fitted = planner._fit(facts.model_copy(update={"budget_simulator_runs": per}), big)
    assert set(fitted.methods) == {"enopt"}
    assert fitted.replicates == 1
    over = planner.check(facts.model_copy(update={"budget_simulator_runs": per}), big)
    assert over[2] is not None  # the cost is known even though it is over the budget
    assert over[2] > per


def test_the_stall_window_waits_for_a_population_optimizers_own_stall(tmp_path: Path) -> None:
    """Review of 2026-10-06: every generation of ga or pso is an accepted step and their best
    sits still for generations; the monitor's stall rule looks back at least their stall."""
    from apex.agents.optimization import stall_window
    from apex.storage import optimization_directory as layout

    for name, options, expected in (("enopt", {}, 3), ("ga", {}, 20), ("pso", {"stall": 30}, 30)):
        run = tmp_path / name
        layout.write_config(run, {"optimizer": {"name": name, "options": options}})
        assert stall_window(run, 3) == expected


# ---------------------------------------------------------------------- several objectives


def _multi_case(tmp_path: Path, **options: object) -> Path:
    """The toy reservoir's NPV against the water injected, by nsga2 (seconds)."""
    config = {
        "optimization": {"name": "front", "seed": 3, "output_dir": str(tmp_path / "front")},
        "controls": {"variables": [{"name": "inj", "initial": [100.0, 100.0], "lower": 1.0,
                                    "upper": 400.0}]},
        "objectives": [
            {"name": "npv", "label": "npv", "options": {
                "start": "2000-01-01", "times": TIMES, "wop": 1.0, "wwi": 0.02,
                "obj_scaling": -1000.0}},
            {"name": "field_total", "label": "water", "options": {
                "vector": "FWIT", "time": TIMES[-1], "scale": 0.001}},
        ],
        "optimizer": {"name": "nsga2", "options": {"pop_size": 8, "maxiter": 4,
                                                   "transform": True, **options}},
        "simulator": {"name": "toy_reservoir"},
    }  # fmt: skip
    path = tmp_path / "front.toml"
    path.write_text(tomli_w.dumps(config), encoding="utf-8")
    return path


def _front_facts(**changes: object) -> OptimizationFacts:
    return _facts(**{
        "optimizer": "nsga2", "objective": "npv+water", "objective_names": ["npv", "water"],
        "pop_size": 40, "front_size": 20, "accepted_steps": 20, "window_improvement": None,
        "feasible_share": 1.0, "hypervolume": 5.0, "best_hypervolume": 5.0,
        "stalled_generations": 0, "front_stall_window": 30, **changes,
    })  # fmt: skip


def test_a_front_is_judged_by_feasibility_size_movement_and_hypervolume() -> None:
    """The front's rules, each at its threshold."""
    t = MonitorSection()

    def codes(**changes: object) -> set[str]:
        return {f.code for f in optimization_rules(_front_facts(**changes), t).findings}

    assert codes() == set()  # a healthy front
    assert "no_feasible_member" in codes(infeasible_generations=10, feasible_share=0.0)
    assert "no_feasible_member" not in codes(infeasible_generations=9, feasible_share=0.0)
    assert "small_front" in codes(front_size=3)  # under 0.1 x 40
    assert "small_front" not in codes(front_size=3, accepted_steps=9)  # too early to judge
    assert "small_front" not in codes(front_size=4)
    # The front holds feasible members only: 2 of 2 feasible is not a lost diversity.
    assert "small_front" not in codes(front_size=2, feasible_share=0.05)
    assert "front_stall" in codes(stalled_generations=30)
    assert "front_stall" not in codes(stalled_generations=30, state="finished")
    assert "hypervolume_drop" in codes(hypervolume=4.7)  # 6 % below the best
    assert "hypervolume_drop" not in codes(hypervolume=4.9)  # 2 %: crowding's dips
    # One objective: none of the front's rules apply.
    assert {f.code for f in optimization_rules(_facts(), t).findings} <= {"stall"}


def test_the_monitor_and_report_read_a_real_front(tmp_path: Path) -> None:
    run_dir = optimization.optimize(_multi_case(tmp_path)).run_dir
    from apex.storage import optimization_directory as layout

    history = [row for row in layout.read_history(run_dir) if row.get("kind") == "step"]
    assert all("stalled_generations" in row for row in history)
    assert any("front_movement" in row for row in history)  # logged once two fronts exist
    service = ApexService(tmp_path / "service")
    check = OptimizationMonitor(service, None).check(run_dir)
    assert check.facts.objective_names == ["npv", "water"]
    assert check.facts.pop_size == 8
    assert check.facts.front_size is not None
    assert "front of" in check.report.summary
    report = optimization_report(service, None, run_dir)
    text = report.text
    assert "## Front" in text
    assert "best npv" in text
    assert "best water" in text
    assert "representative" in text


def test_the_diagnosis_recommends_for_a_front(tmp_path: Path) -> None:
    from apex.agents.optimization_advice import OptimizationDiagnosisFacts, diagnosis_rules

    facts = OptimizationDiagnosisFacts(
        monitor=_front_facts(infeasible_generations=12, feasible_share=0.0, front_size=1,
                             stalled_generations=31),
        optimizer={"name": "nsga2", "options": {"pop_size": 40}},
        controls=2, realizations=1, simulated=True, execution={"name": "serial"},
    )  # fmt: skip
    diagnosis = diagnosis_rules(facts, {"no_feasible_member", "small_front", "front_stall"})
    assert {c.code for c in diagnosis.causes} == {"no_feasible_member", "small_front",
                                                  "front_stall"}  # fmt: skip
    patches = [r.patch for r in diagnosis.recommendations if r.patch]
    # Deb's rules already push by violation alone: no switch to EPF (whose default 10 outer
    # iterations would multiply the evaluation bound), but more room, at its stated cost.
    assert not any("constraint_handling" in str(p) for p in patches)
    assert {"optimizer": {"options": {"pop_size": 80}}} in patches
    assert {"optimizer": {"options": {"eta_m": 10.0}}} in patches
    assert any("doubles the evaluations" in r.change for r in diagnosis.recommendations)
    # Under EPF, a larger penalty factor: the bound is unchanged.
    epf = facts.model_copy(update={"optimizer": {"name": "nsga2", "options": {
        "pop_size": 40, "constraint_handling": "epf", "epf": {"r": 2.0}}}})  # fmt: skip
    epf_patches = [r.patch for r in diagnosis_rules(epf, {"no_feasible_member"}).recommendations]
    assert {"optimizer": {"options": {"epf": {"r": 20.0}}}} in epf_patches


def test_the_planner_compares_populations_of_nsga2_by_their_fronts(tmp_path: Path) -> None:
    """A case with several objectives runs only on nsga2: the rules never propose EnOpt (the
    planner bug of 2026-10-06), and the study compares fronts."""
    from apex.agents.optimization_advice import OptimizationPlanner

    service = ApexService(tmp_path / "service")
    result = OptimizationPlanner(service, None).plan(_multi_case(tmp_path), "the trade-off",
                                                     budget=20_000)  # fmt: skip
    assert result.decided_by == "rules"
    assert result.facts.objective_names == ["npv", "water"]
    methods = result.proposal.methods
    assert {m.name for m in methods.values()} == {"nsga2"}
    assert sorted(m.options["pop_size"] for m in methods.values()) == [8, 16]
    assert result.proposal.metrics == ["hypervolume:final", "front_size:final", "evaluations:all"]
    assert result.within_budget
    assert result.validation is not None
    assert result.validation.valid, result.validation.errors


def test_an_optimizer_that_does_not_match_its_objectives_is_refused_at_validation(
    tmp_path: Path,
) -> None:
    """A reported issue (2026-10-06): an nsga2 cell on a single-objective case passed
    validation and failed when it ran; and the reverse."""
    import tomllib

    def variant(source: Path, name: str, optimizer: dict[str, object]) -> Path:
        config = tomllib.loads(source.read_text("utf-8"))
        config["optimizer"] = optimizer
        path = tmp_path / f"{name}.toml"
        path.write_text(tomli_w.dumps(config), encoding="utf-8")
        return path

    service = ApexService(tmp_path / "service")
    scalar = variant(
        _case(tmp_path), "scalar", {"name": "nsga2", "options": {"pop_size": 8, "maxiter": 2}}
    )
    refused = service.validate_optimization(scalar)
    assert not refused.valid
    assert "needs several objectives" in refused.errors[0]
    single = variant(
        _multi_case(tmp_path), "single", {"name": "ga", "options": {"pop_size": 8, "maxiter": 2}}
    )
    refused = service.validate_optimization(single)
    assert not refused.valid
    assert "optimizes one objective" in refused.errors[0]


def _summary(history: list[dict[str, object]], pareto: dict[str, object] | None = None) -> object:
    from apex.domain.services import OptimizationSummary

    return OptimizationSummary(
        path="/x", name="front", state="running", optimizer="nsga2", objective="a+b",
        controls=["x"], history=history, result=None, evaluations=0, pareto=pareto,
    )  # fmt: skip


def test_the_front_facts_count_generations_and_respect_penalty_factors() -> None:
    """Review findings (2026-10-06): the start is not a generation; between penalty factors
    the last stall count belongs to the factor that ended; before the first front is written
    the objectives come from the configuration."""
    from apex.agents.optimization import FrontSettings, optimization_facts

    front = FrontSettings(("a", "b"), 10, 30)
    rows: list[dict[str, object]] = [
        {"kind": "start", "iteration": 0, "objective": 1.0, "feasible_share": 0.0},
        *({"kind": "step", "iteration": k, "objective": 1.0, "feasible_share": 0.0,
           "stalled_generations": float(k)} for k in range(1, 4)),
    ]  # fmt: skip
    facts = optimization_facts(_summary(rows), 30, None, front)  # type: ignore[arg-type]
    assert facts.objective_names == ["a", "b"]  # no pareto.json yet
    assert facts.infeasible_generations == 3  # the start row is not counted
    assert facts.stalled_generations == 3
    ended = [*rows, {"kind": "epf_update", "iteration": 3, "epf_iteration": 2, "objective": 1.0}]
    facts = optimization_facts(_summary(ended), 30, None, front)  # type: ignore[arg-type]
    assert facts.stalled_generations is None  # no front_stall between penalty factors


def test_front_settings_read_the_configuration(tmp_path: Path) -> None:
    import json

    from apex.agents.optimization import FrontSettings, front_settings

    def settings(options: dict[str, object]) -> FrontSettings:
        (tmp_path / "config.json").write_text(json.dumps({
            "optimizer": {"name": "nsga2", "options": options},
            "objectives": [{"name": "npv", "label": "npv"}, {"name": "field_total"}],
        }))  # fmt: skip
        found = front_settings(tmp_path)
        assert found is not None
        return found

    big = settings({"stall": 50, "pop_size": 12})
    assert big.objective_names == ("npv", "field_total")  # a label, else the name
    assert (big.pop_size, big.stall_window) == (12, 30)  # the default stall speaks first
    assert settings({"stall": 20}).stall_window is None
    assert settings({}).stall_window is None
    assert settings({"pop_size": "many"}).pop_size is None


def test_the_planner_keeps_nsga2_alone_without_room_for_two(tmp_path: Path) -> None:
    from apex.agents.optimization_advice import OptimizationPlanFacts, plan_rules
    from apex.services.settings import PlannerSection

    base = {
        "case": "front", "config_file": "/x/front.toml", "objective": "a+b",
        "objective_names": ["a", "b"], "simulated": True,
        "optimizer": {"name": "nsga2", "options": {"pop_size": 10}}, "controls": 2,
        "runs_per_cell": 100, "budget_simulator_runs": 250, "max_cells": 2,
        "max_variants": 6, "optimizers": ["nsga2"], "step_rules": [], "configuration": {},
    }  # fmt: skip
    tight = plan_rules(OptimizationPlanFacts.model_validate(base), "trade-off")
    assert tight.methods == {}  # 3 x 100 runs do not fit 250: the case's own alone
    assert tight.replicates == 2
    capped = plan_rules(OptimizationPlanFacts.model_validate({**base, "max_variants": 1,
                                                              "budget_simulator_runs": 10_000}),
                        "trade-off")  # fmt: skip
    assert capped.methods == {}
    analytic = plan_rules(OptimizationPlanFacts.model_validate({**base, "runs_per_cell": 0}),
                          "trade-off", PlannerSection(max_replicates=3))  # fmt: skip
    assert len(analytic.methods) == 2
    assert analytic.replicates == 3


@pytest.mark.parametrize(("name", "pair"), [("nsga3", (19, 27)), ("moead", (19, 27))])
def test_the_planner_compares_directions_for_nsga3_and_moead(
    tmp_path: Path, name: str, pair: tuple[int, int]
) -> None:
    """MOEA/D has no pop_size and NSGA-III's follows its directions, so the
    rules compare n_partitions (19 -> 20 directions; 39 -> 40 at least doubles them)."""
    import tomllib

    from apex.agents.optimization_advice import OptimizationPlanner

    config = tomllib.loads(_multi_case(tmp_path).read_text("utf-8"))
    config["optimizer"] = {"name": name, "options": {"maxiter": 3, "transform": True}}
    path = tmp_path / f"{name}.toml"
    path.write_text(tomli_w.dumps(config), encoding="utf-8")
    service = ApexService(tmp_path / "service")
    result = OptimizationPlanner(service, None).plan(path, "the trade-off", budget=50_000)
    partitions = sorted(m.options.get("n_partitions", 19) for m in result.proposal.methods.values())
    assert partitions[0] == pair[0]
    assert partitions[1] >= 39  # at least twice the directions
    assert all("pop_size" not in m.options for m in result.proposal.methods.values())
    assert result.validation is not None
    assert result.validation.valid, result.validation.errors


def test_front_settings_size_nsga3_and_name_statistics(tmp_path: Path) -> None:
    import json

    from apex.agents.optimization import front_settings

    (tmp_path / "config.json").write_text(json.dumps({
        "optimizer": {"name": "nsga3", "options": {}},
        "objectives": [{"name": "npv"}, {"name": "npv", "statistic": "std"},
                       {"name": "npv", "label": "risk", "statistic": "p90"}],
    }))  # fmt: skip
    settings = front_settings(tmp_path)
    assert settings is not None
    assert settings.objective_names == ("npv", "npv_std", "risk")
    assert settings.pop_size == 28  # three objectives: 28 directions


@pytest.mark.parametrize(("optimizer", "n_obj"), [("nsga2", 3), ("nsga3", 5)])
def test_the_planner_sees_a_dtlz_cases_objectives(
    tmp_path: Path, optimizer: str, n_obj: int
) -> None:
    """Review: DTLZ's names depend on n_obj (per instance), so the planner builds
    the objective; reading the class left none, and the single-objective rules planned EnOpt."""
    from apex.agents.optimization_advice import OptimizationPlanner

    path = tmp_path / "dtlz.toml"
    path.write_text(tomli_w.dumps({
        "optimization": {"name": "dtlz", "seed": 1},
        "controls": {"variables": [{"name": "x", "initial": [0.5] * (n_obj + 4),
                                    "lower": 0.0, "upper": 1.0}]},
        "objective": {"name": "dtlz2", "options": {"n_obj": n_obj}},
        "optimizer": {"name": optimizer, "options": {"maxiter": 3, "transform": True}},
    }), encoding="utf-8")  # fmt: skip
    service = ApexService(tmp_path / "service")
    result = OptimizationPlanner(service, None).plan(path, "the trade-off", budget=50_000)
    assert len(result.facts.objective_names) == n_obj
    assert {m.name for m in result.proposal.methods.values()} == {optimizer}
    assert result.validation is not None
    assert result.validation.valid, result.validation.errors


def test_the_diagnosis_grows_moead_and_nsga3_by_their_directions() -> None:
    """Review: moead has no pop_size and nsga3's follows its directions; the
    patches grow the partitions, from the run's real population (an explicit pop_size is
    doubled)."""
    from apex.agents.optimization_advice import OptimizationDiagnosisFacts, diagnosis_rules

    for name, options in (("moead", {}), ("nsga3", {"pop_size": 24})):
        facts = OptimizationDiagnosisFacts(
            monitor=_front_facts(optimizer=name, pop_size=20, front_size=1),
            optimizer={"name": name, "options": options},
            controls=2, realizations=1, simulated=True, execution={"name": "serial"},
        )  # fmt: skip
        patches = [r.patch for r in diagnosis_rules(facts, {"small_front"}).recommendations
                   if r.patch]  # fmt: skip
        grow = next(p for p in patches if "eta_m" not in p["optimizer"]["options"])
        changed = grow["optimizer"]["options"]
        assert changed["n_partitions"] >= 39  # two objectives: 20 directions -> at least 40
        # moead has no pop_size; nsga3's explicit one is doubled, never made smaller.
        assert changed.get("pop_size") == (None if name == "moead" else 48)


def _dtlz_case(tmp_path: Path, n_obj: int) -> Path:
    path = tmp_path / "dtlz.toml"
    path.write_text(tomli_w.dumps({
        "optimization": {"name": "dtlz", "seed": 1},
        "controls": {"variables": [{"name": "x", "initial": [0.5] * (n_obj + 4),
                                    "lower": 0.0, "upper": 1.0}]},
        "objective": {"name": "dtlz2", "options": {"n_obj": n_obj}},
        "optimizer": {"name": "nsga2", "options": {"pop_size": 8, "maxiter": 2}},
    }), encoding="utf-8")  # fmt: skip
    return path


def _planned_comparison(
    tmp_path: Path, path: Path, methods: dict[str, object]
) -> tuple[OptimizationPlanResult, dict[str, Any]]:
    from apex.agents.chat import _comparison, _optimization_matching
    from apex.agents.optimization_advice import OptimizationPlanner
    from apex.services.optimization_study import expand

    service = ApexService(tmp_path / "service")
    result = OptimizationPlanner(service, None).plan_given(
        path, "compare", {"methods": methods, "replicates": 1}, budget=50_000
    )
    cells = expand(service, result.spec, path.parent)
    return result, _comparison(cells, _optimization_matching(result.spec, path.parent, cells))


def test_a_planned_comparison_of_optimizers_matches_what_the_arms_leave_unset(
    tmp_path: Path,
) -> None:
    """A reported issue (2026-10-24): nsga2 (pop_size 8) against nsga3 left at its defaults
    ran crossover_prob 0.9 against 1.0, eta_c 15 against 30, and a population of the
    directions. The planner gives nsga3 nsga2's values and population; what an arm sets
    itself is kept."""
    from apex.algorithms.optimization.nsga3 import NSGA3

    path = _dtlz_case(tmp_path, 2)
    nsga2 = {"name": "nsga2", "options": {"pop_size": 8, "maxiter": 2}}
    result, comparison = _planned_comparison(tmp_path, path, {
        "nsga2": nsga2, "nsga3": {"name": "nsga3", "options": {"maxiter": 2, "eta_m": 10.0}},
    })  # fmt: skip
    options = result.proposal.methods["nsga3"].options
    assert options["crossover_prob"] == 0.9
    assert options["eta_c"] == 15.0
    assert options["eta_m"] == 10.0  # set by the arm: kept
    assert (options["n_partitions"], options["pop_size"]) == (7, 8)
    assert result.proposal.methods["nsga2"].options == nsga2["options"]
    assert any("population" in m and "-> 8" in m for m in result.matched)
    # The population the optimizer itself runs with (its own rule, not the planner's copy).
    nsga3 = NSGA3(**options)
    nsga3.set_objectives(2)
    assert nsga3.pop_size == 8
    # eta_m stays a difference the arms chose: reported, nothing else is.
    unmatched = comparison.get("unmatched", [])
    assert [u for u in unmatched if "eta_m" in u]
    assert not [u for u in unmatched if "population" in u or "crossover_prob" in u
                or "eta_c" in u]  # fmt: skip


def test_a_given_comparison_of_optimizers_is_headed_as_one(tmp_path: Path) -> None:
    """A reported issue: MOEA/D against NSGA-II, given by the assistant, was headed
    "Which population size best finds the trade-off ...", the rules' wording for one
    optimizer at two sizes. The heading follows the design's own methods."""
    path = _dtlz_case(tmp_path, 2)
    nsga2 = {"name": "nsga2", "options": {"pop_size": 8, "maxiter": 2}}
    result, _ = _planned_comparison(tmp_path, path, {
        "nsga2": nsga2, "moead": {"name": "moead", "options": {"maxiter": 2}},
    })  # fmt: skip
    hypothesis = result.proposal.hypothesis
    assert hypothesis.startswith("Which optimizer (nsga2, moead) best finds the trade-off")
    assert "population size" not in hypothesis
    assert f"# Hypothesis: {hypothesis}" in result.toml(
        ApexService(tmp_path / "service").workspace(), tmp_path
    )


def test_the_rules_heading_follows_their_own_methods() -> None:
    from apex.agents.optimization_advice import OptimizationMethod, study_hypothesis

    one = {"a": OptimizationMethod(name="nsga2"), "b": OptimizationMethod(name="nsga2")}
    assert study_hypothesis(one, ["npv", "water"], "g", "nsga2").startswith(
        "Which setting of nsga2 best finds the trade-off between npv, water"
    )
    assert study_hypothesis({}, ["npv", "water"], "g", "nsga2").startswith(
        "How nsga2 as configured finds the trade-off"
    )
    two = {"a": OptimizationMethod(name="enopt"), "b": OptimizationMethod(name="smcopt")}
    assert study_hypothesis(two, [], "g", "enopt") == (
        "Which optimizer (enopt, smcopt) best serves the goal: g"
    )


def test_a_population_that_cannot_be_matched_is_reported(tmp_path: Path) -> None:
    """A reported issue (2026-10-24): moead has one member per direction, and three
    objectives have no 12 directions (10 or 15): the nearest is planned, and the difference
    is on the card."""
    path = _dtlz_case(tmp_path, 3)
    result, comparison = _planned_comparison(tmp_path, path, {
        "nsga2": {"name": "nsga2", "options": {"pop_size": 12, "maxiter": 2}},
        "moead": {"name": "moead", "options": {"maxiter": 2}},
    })  # fmt: skip
    assert result.proposal.methods["moead"].options["n_partitions"] == 3  # 10 directions
    [population] = [u for u in comparison["unmatched"] if u.startswith("population")]
    assert "12" in population
    assert "10" in population


def test_only_one_familys_operators_are_matched_and_one_optimizers_settings_never() -> None:
    """Review of the matching (a reported issue, 2026-10-24): two settings of one optimizer
    are what a study compares; options that only share a name (line_search's Wolfe c1, c2
    and pso's acceleration) and termination (a generation is not an EnOpt iteration) are
    another optimizer's own."""
    from apex.agents.optimization_advice import OptimizationStudyProposal, match_arms

    def matched(
        methods: dict[str, dict[str, Any]], n_obj: int = 1
    ) -> tuple[dict[str, Any], list[str]]:
        proposal = OptimizationStudyProposal.model_validate({
            "name": "s", "hypothesis": "h..", "reasoning": "r..", "replicates": 1,
            "methods": methods, "variants": [{"name": "base", "rationale": "r.."}],
        })  # fmt: skip
        result, changes = match_arms(proposal, n_obj)
        return {k: m.options for k, m in result.methods.items()}, changes

    same = {"tuned": {"name": "nsga2", "options": {"crossover_prob": 0.7, "pop_size": 20}},
            "base": {"name": "nsga2", "options": {}}}  # fmt: skip
    assert matched(same, 2) == ({"tuned": same["tuned"]["options"], "base": {}}, [])
    for first, other in (("line_search", "pso"), ("ga", "enopt"), ("trust_region", "enopt")):
        methods = {first: {"name": first, "options": {"maxiter": 5}},
                   other: {"name": other, "options": {}}}  # fmt: skip
        assert matched(methods)[0][other] == {}, (first, other)
    options, _ = matched({"ga": {"name": "ga", "options": {"maxiter": 5, "stall": 3}},
                                "nsga2": {"name": "nsga2", "options": {}}})  # fmt: skip
    assert options["nsga2"] == {"pop_size": 20}  # ga's population; not its termination
    options, _ = matched({"en": {"name": "enopt", "options": {"alpha": 0.3, "maxiter": 2}},
                          "smc": {"name": "smcopt", "options": {}}})  # fmt: skip
    assert options["smc"] == {"alpha": 0.3}


def test_a_design_over_the_budget_is_refused_with_each_methods_least_cost(
    tmp_path: Path,
) -> None:
    """A reported issue (#242): genopt against EnOpt within 4 simulator runs was refused four
    times, at 20, 10, 8 and 192 runs, with no word of what the cheapest design costs. The
    refusal now gives each method's least cost (one seed, one iteration, a two-member
    ensemble, no backtracking), and says when no design of the methods fits; a population's
    size is not cut, so then it says only that the size was kept (the review of the fix)."""
    from apex.agents.optimization_advice import OptimizationPlanner
    from apex.core.errors import ConfigError

    case = _case(tmp_path)
    planner = OptimizationPlanner(ApexService(tmp_path / "service"), None)

    def refusal(methods: dict[str, Any], budget: int) -> str:
        with pytest.raises(ConfigError) as refused:
            planner.plan_given(case, "compare", {"methods": methods, "replicates": 1},
                               budget=budget)  # fmt: skip
        return str(refused.value)

    genopt = {"name": "genopt", "options": {"ne": 4, "maxiter": 3, "cma": True}}
    methods: dict[str, Any] = {
        "enopt": {"name": "enopt", "options": {"ne": 4, "maxiter": 3}}, "genopt": genopt,
    }  # fmt: skip
    # One realization: the start, then one iteration's two members and one trial point.
    message = refusal(methods, 4)
    assert "computed by APEX, each method's least cost for one seed: enopt 5 (" in message
    assert "genopt 5 (" in message
    assert "together at least 10, so no design of these methods fits the budget of 4" in message
    assert "no design" not in refusal(methods, 10)
    # The least design itself plans within that budget, CMA included (ne = 1 would not run).
    cheapest = {k: {"name": k, "options": {**m["options"], "ne": 2, "maxiter": 1,
                                           "alpha_maxiter": 0}}
                for k, m in methods.items()}  # fmt: skip
    result = planner.plan_given(case, "compare", {"methods": cheapest, "replicates": 1},
                                budget=10)  # fmt: skip
    assert result.cost == 10
    # A population optimizer's size is kept, so nothing is said about every design.
    ga = refusal({"enopt": methods["enopt"],
                  "ga": {"name": "ga", "options": {"pop_size": 20, "maxiter": 3}}}, 4)  # fmt: skip
    assert "population 20 kept as proposed" in ga
    assert "no design" not in ga

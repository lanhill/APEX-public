"""The optimization agents on a surrogate optimization's final check.

Synthetic ``result.json`` records in the shape ``SurrogateObjective.finish`` writes them:
the monitor's facts read them, each rule speaks above its threshold and is silent at it, the
report shows the check's table with numbers from the facts, and the diagnosis recommends a
rebuild and refits whose patch the surrogate section accepts.
"""

import re
from typing import Any

from apex.agents.optimization import (
    OptimizationFacts,
    OptimizationReportFacts,
    _final_facts,
    optimization_rules,
    report_template,
)
from apex.agents.optimization_advice import OptimizationDiagnosisFacts, diagnosis_rules
from apex.agents.reporter import fact_numbers
from apex.domain.optimization import OptimizationSurrogateSection
from apex.services.settings import MonitorSection


def _stats(mean: float, std: float, n: int = 6) -> dict[str, float | int]:
    return {"n": n, "mean": mean, "std": std, "min": mean - 2 * std, "max": mean + 2 * std,
            "median": mean, "p10": mean - 1.28 * std, "p90": mean + 1.28 * std}  # fmt: skip


def _record(mean: float, std: float, source: str) -> dict[str, Any]:
    return {"source": source, "objective_scaling": -1.0, "value": _stats(mean, std),
            "minimised": _stats(-mean, std), "penalty": "excluded"}  # fmt: skip


def _result(
    *,
    sim_mean: float = 100.0,
    sur_mean: float = 101.0,
    sim_std: float = 10.0,
    sur_std: float = 10.0,
    opt_relative: float = 0.01,
    passed: bool | None = None,
    outside: int = 0,
    compared: int = 6,
    error: str | None = None,
) -> dict[str, Any]:
    gap = abs(sur_mean - sim_mean) / abs(sim_mean)
    ensemble: dict[str, Any] = {
        "members": 6, "realizations": [0], "member_realizations": [0] * 6,
        "new_simulator_runs": 6, "simulator": _record(sim_mean, sim_std, "simulator"),
        "members_compared": compared, "not_answered": {"outside": outside, "coarse_lost": 0},
        "surrogate": _record(sur_mean, sur_std, "surrogate"),
        "simulator_compared": _record(sim_mean, sim_std, "simulator"),
        "difference": {"mean": sur_mean - sim_mean, "std": 1.5, "worst": -3.0, "units": "value"},
        "relative_mean_difference": gap,
    }  # fmt: skip
    check: dict[str, Any] = {
        "simulator_name": "toy_reservoir",
        "objectives": [{"name": "objective", "surrogate": -0.99, "simulator": -1.0,
                        "difference": 0.01, "relative": opt_relative,
                        "value": {"surrogate": 0.99, "simulator": 1.0}}],
        "ensemble": ensemble,
    }  # fmt: skip
    if passed is not None:
        check |= {"tolerance": 0.02, "passed": passed}
    if error is not None:
        check = {"simulator_name": "toy_reservoir", "error": error}
    return {"objective": -1.0, "message": "Maximum number of iterations reached",
            "surrogate": {"optimum_check": check}}  # fmt: skip


def _facts(result: dict[str, Any]) -> OptimizationFacts:
    base: dict[str, object] = {
        "path": "/x", "name": "toy", "state": "finished", "optimizer": "enopt",
        "objective": "npv", "accepted_steps": 5, "start_objective": -0.5,
        "best_objective": -1.0, "last_objective": -1.0, "relative_improvement": 1.0,
        "window_improvement": 0.05, "evaluations": 40, "failed_evaluations": 0,
        "message": None,
    }  # fmt: skip
    return OptimizationFacts.model_validate(base | _final_facts(result))


def _codes(result: dict[str, Any], **thresholds: float) -> set[str]:
    report = optimization_rules(_facts(result), MonitorSection.model_validate(thresholds))
    return {f.code for f in report.findings}


FINAL = {"final_check_failed", "surrogate_bias", "surrogate_spread_mismatch",
         "final_check_incomplete"}  # fmt: skip


def test_the_final_check_facts_are_read_from_the_result() -> None:
    facts = _facts(_result(passed=True, outside=1, compared=5))
    c = facts.final_check
    assert c is not None
    assert c.simulator == "toy_reservoir"
    assert c.passed is True
    assert c.tolerance == 0.02
    assert c.members == 6
    assert c.members_compared == 5
    assert c.members_outside == 1
    assert c.new_simulator_runs == 6
    assert c.ensemble_simulator is not None
    assert c.ensemble_simulator.mean == 100.0
    assert c.ensemble_surrogate is not None
    assert c.ensemble_surrogate.mean == 101.0
    assert c.optimum_simulator == 1.0
    assert c.optimum_surrogate == 0.99
    assert c.difference_worst == -3.0
    # The final ensemble's values are the simulator's.
    assert facts.final_ensemble is not None
    assert facts.final_ensemble.mean == 100.0
    # A run with neither has no such facts; a plain run has its final ensemble only.
    assert _facts({"objective": -1.0}).final_check is None
    plain = _facts({"final_ensemble": _record(50.0, 5.0, "objective") | {"members": 6}})
    assert plain.final_check is None
    assert plain.final_ensemble is not None
    assert plain.final_ensemble.std == 5.0


def test_a_good_final_check_finds_nothing() -> None:
    assert not _codes(_result(passed=True)) & FINAL


def test_final_check_failed_speaks_when_the_tolerance_is_exceeded() -> None:
    assert "final_check_failed" in _codes(_result(passed=False))
    assert "final_check_failed" not in _codes(_result(passed=True))
    assert "final_check_failed" not in _codes(_result())  # no tolerance: reported only


def test_surrogate_bias_speaks_above_its_threshold_and_not_at_it() -> None:
    # Ensemble mean 5 % off at a 5 % threshold: silent (strictly above speaks).
    assert "surrogate_bias" not in _codes(_result(sur_mean=105.0), optimization_final_bias=0.05)
    assert "surrogate_bias" in _codes(_result(sur_mean=106.0), optimization_final_bias=0.05)
    # The optimum's difference alone also counts.
    assert "surrogate_bias" in _codes(_result(opt_relative=0.2))
    report = optimization_rules(_facts(_result(sur_mean=110.0)), MonitorSection())
    bias = next(f for f in report.findings if f.code == "surrogate_bias")
    assert bias.severity == "warning"
    assert "0.1" in bias.evidence
    assert report.status == "warning"
    assert report.recommendation == "investigate"


def test_surrogate_spread_mismatch_speaks_outside_its_ratio() -> None:
    assert "surrogate_spread_mismatch" not in _codes(_result(sur_std=20.0))  # ratio 2: at it
    assert "surrogate_spread_mismatch" in _codes(_result(sur_std=21.0))
    assert "surrogate_spread_mismatch" not in _codes(_result(sur_std=5.0))  # 1/2: at it
    assert "surrogate_spread_mismatch" in _codes(_result(sur_std=4.0))
    # A spread where the simulator has none.
    assert "surrogate_spread_mismatch" in _codes(_result(sim_std=0.0, sur_std=1.0))
    assert "surrogate_spread_mismatch" not in _codes(_result(sim_std=0.0, sur_std=0.0))


def test_final_check_incomplete_speaks_on_errors_and_members_outside() -> None:
    assert "final_check_incomplete" in _codes(_result(error="SurrogateDomainError: x"))
    assert "final_check_incomplete" in _codes(_result(outside=4, compared=2))  # 4/6 > 0.5
    assert "final_check_incomplete" not in _codes(_result(outside=3, compared=3))  # 0.5: at it
    assert "final_check_incomplete" in _codes(_result(outside=0, compared=0))


def test_the_report_shows_the_final_check_with_numbers_from_the_facts() -> None:
    facts = _facts(_result(passed=True, outside=1, compared=5))
    monitor = optimization_rules(facts, MonitorSection())
    report = report_template(OptimizationReportFacts(facts=facts, monitor=monitor, controls={}))
    section = next(s for s in report.sections if s.heading == "Final check")
    assert "toy_reservoir" in section.text
    assert "Passed" in section.text
    assert "| final ensemble mean | 100 | 101 |" in section.text
    assert "5 of 6 members compared (1 outside" in section.text
    known = fact_numbers(OptimizationReportFacts(facts=facts, monitor=monitor, controls={}))
    for token in re.findall(r"(?<![A-Za-z_\d.])-?\d+(?:\.\d+)?(?:e[-+]?\d+)?", section.text):
        value = float(token)
        assert any(abs(value - k) <= 1e-5 * max(1.0, abs(k)) for k in known), token


def test_a_plain_runs_report_shows_its_final_ensemble() -> None:
    facts = _facts({"final_ensemble": _record(50.0, 5.0, "objective") | {"members": 6}})
    monitor = optimization_rules(facts, MonitorSection())
    report = report_template(OptimizationReportFacts(facts=facts, monitor=monitor, controls={}))
    section = next(s for s in report.sections if s.heading == "Final ensemble")
    assert "mean 50, std 5" in section.text
    assert not {f.code for f in monitor.findings} & FINAL


def test_the_diagnosis_recommends_a_rebuild_and_refits_that_validate() -> None:
    facts = _facts(_result(sur_mean=110.0, passed=False))
    codes = {f.code for f in optimization_rules(facts, MonitorSection()).findings}
    surrogate = {"model_file": "m.json"}
    diagnosis = diagnosis_rules(OptimizationDiagnosisFacts(
        monitor=facts, optimizer={"name": "enopt", "options": {"ne": 6}}, controls=3,
        realizations=1, simulated=True, execution={"name": "serial"}, surrogate=surrogate,
    ), codes)  # fmt: skip
    assert {"final_check_failed", "surrogate_bias"} <= {c.code for c in diagnosis.causes}
    patches = [r.patch for r in diagnosis.recommendations if r.patch]
    assert len(diagnosis.recommendations) == 2
    assert len(patches) == 1
    # The refit patch is a valid [surrogate] section.
    OptimizationSurrogateSection.model_validate(surrogate | patches[0]["surrogate"])
    # A run already refitting gets the rebuild only; a healthy check gets nothing.
    refitting = diagnosis_rules(OptimizationDiagnosisFacts(
        monitor=facts, optimizer={"name": "enopt"}, controls=3, realizations=1,
        simulated=True, execution={"name": "serial"},
        surrogate=surrogate | patches[0]["surrogate"],
    ), codes)  # fmt: skip
    assert [r.patch for r in refitting.recommendations] == [None]
    healthy = _facts(_result(passed=True))
    quiet = diagnosis_rules(OptimizationDiagnosisFacts(
        monitor=healthy, optimizer={"name": "enopt"}, controls=3, realizations=1,
        simulated=True, execution={"name": "serial"}, surrogate=surrogate,
    ), {f.code for f in optimization_rules(healthy, MonitorSection()).findings})  # fmt: skip
    assert quiet.recommendations == []

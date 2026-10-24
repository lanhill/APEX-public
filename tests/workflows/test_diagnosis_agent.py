"""The diagnosis agent, on linear cases and scripted models (no live model).

Independent references: coverage is recomputed here from the saved forecast and the
observations file, parameter movement from the saved ensembles, and failures from the
run's events.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pytest
from typer.testing import CliRunner

from apex.agents.diagnosis import (
    Cause,
    Diagnosis,
    DiagnosisFacts,
    Diagnostician,
    MovementFact,
    Recommendation,
    TypeCoverageFact,
    diagnose_facts,
    reconcile,
    rules,
)
from apex.agents.monitor import RunObservation
from apex.agents.providers.stub import StubProvider
from apex.cli.main import app
from apex.core.llm import LLMUnavailable
from apex.services import composition
from apex.services.api import ApexService
from apex.services.settings import DiagnosisSection
from apex.storage.run_directory import RunDirectory

GIES = (
    'name = "gies"\nstrategy = { name = "rlmmac", options = { energy = 1.0 } }\n'
    "schedule = { data_mismatch_tolerance = 1e-9 }\nmax_iterations = 12"
)
FAILING = 'nonlinearity = "tanh"\nfail_above = 2.0'


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path


CaseFactory = Callable[..., Case]


def _observation(**fields: Any) -> RunObservation:  # noqa: ANN401
    base: dict[str, Any] = {
        "run_dir": "/runs/x", "name": "x", "method": "esmda", "iterative": False,
        "state": "complete", "forecasts": 5, "accepted": 5, "rejected": 0,
        "rejections_in_a_row": 0, "members_prior": 50, "members_now": 50,
        "failed_fraction": 0.0, "data_mismatch_iterations": [0, 1, 2, 3, 4],
        "data_mismatch_means": [1000.0, 400.0, 200.0, 120.0, 90.0], "last_relative_change": 0.25,
        "normalized_data_mismatch": 1.4, "spread_ratio": {"x": 0.4}, "damping": None,
        "initial_damping": None,
    }  # fmt: skip
    return RunObservation(**{**base, **fields})


def _facts(**fields: Any) -> DiagnosisFacts:  # noqa: ANN401
    base: dict[str, Any] = {
        "observation": _observation(), "iteration": 4, "strategy": "approx",
        "localization": None, "has_validation": False, "synthetic": True,
        "coverage": [TypeCoverageFact(datatype="y", observations=10, outside=1,
                                      outside_fraction=0.1)],
        "data_scores": [("y @ t0", 3.0)], "type_scores": [("y", 2.0)], "gains": [],
        "movement": [MovementFact(variable="x", spread_ratio=0.4, percent_beyond_1_std=20.0,
                                  percent_beyond_2_std=5.0, percent_beyond_3_std=1.0)],
        "failures": [],
    }  # fmt: skip
    return DiagnosisFacts(**{**base, **fields})


def _codes(diagnosis: Diagnosis) -> set[str]:
    return {c.code for c in diagnosis.causes}


# ---------------------------------------------------------------------- rules


def test_a_healthy_run_has_nothing_to_fix() -> None:
    d = rules(_facts())

    assert (d.verdict, d.causes, d.recommendations) == ("healthy", [], [])


@pytest.mark.parametrize(
    ("fields", "code"),
    [
        (
            {"coverage": [TypeCoverageFact(datatype="y", observations=10, outside=4,
                                           outside_fraction=0.4)]},
            "data_outside_forecast",
        ),
        ({"data_scores": [("y @ t0", 11.0)]}, "model_deficiency"),
        ({"type_scores": [("y", 25.0)]}, "model_deficiency"),
        (
            {"movement": [MovementFact(variable="x", spread_ratio=0.1, percent_beyond_1_std=60,
                                       percent_beyond_2_std=30, percent_beyond_3_std=12)]},
            "overcorrection",
        ),
    ],
)  # fmt: skip
def test_each_cause_beyond_its_threshold(fields: dict[str, Any], code: str) -> None:
    d = rules(_facts(**fields))

    assert code in _codes(d)
    assert d.verdict == "problems"
    assert any(code in r.addresses for r in d.recommendations)


def test_just_inside_the_thresholds_nothing_is_found() -> None:
    t = DiagnosisSection()
    d = rules(
        _facts(
            coverage=[
                TypeCoverageFact(
                    datatype="y", observations=10, outside=2,
                    outside_fraction=t.coverage_outside_fraction,
                )
            ],
            data_scores=[("y @ t0", t.mahalanobis_limit)],
            movement=[
                MovementFact(
                    variable="x", spread_ratio=0.3, percent_beyond_1_std=50,
                    percent_beyond_2_std=20, percent_beyond_3_std=t.overcorrection_percent,
                )
            ],
        )
    )  # fmt: skip

    assert d.causes == []


def test_a_few_data_outside_by_chance_are_no_cause() -> None:
    """Release-tester U1: an exact (Kalman) posterior was flagged for 2 of 8 data outside
    the bare forecast range; observations carry their own error, and 2 of 8 is chance."""
    two_of_eight = TypeCoverageFact(
        datatype="y", observations=8, outside=2, outside_fraction=0.25, outside_with_noise=2
    )
    assert "data_outside_forecast" not in _codes(rules(_facts(coverage=[two_of_eight])))
    # Outside the bare range but inside the error-widened one: explained by the noise.
    noisy = TypeCoverageFact(
        datatype="y", observations=40, outside=20, outside_fraction=0.5, outside_with_noise=1
    )
    assert "data_outside_forecast" not in _codes(rules(_facts(coverage=[noisy])))
    # Many beyond the widened range: a cause, counted by the widened range.
    many = TypeCoverageFact(
        datatype="y", observations=40, outside=20, outside_fraction=0.5, outside_with_noise=12
    )
    d = rules(_facts(coverage=[many]))
    assert "data_outside_forecast" in _codes(d)
    assert "y 12/40" in next(c.evidence for c in d.causes if c.code == "data_outside_forecast")


def test_the_monitors_findings_are_causes_with_their_advice() -> None:
    d = rules(
        _facts(observation=_observation(spread_ratio={"x": 0.01}, normalized_data_mismatch=0.3))
    )

    assert {"ensemble_collapse", "overfitting"} <= _codes(d)
    localize = next(r for r in d.recommendations if "ensemble_collapse" in r.addresses)
    assert localize.patch == {"method": {"localization": {"name": "adaptive"}}}
    assert any("overfitting" in r.addresses and "validation" in r.change for r in d.recommendations)
    already = rules(_facts(observation=_observation(spread_ratio={"x": 0.01}),
                           localization="adaptive"))  # fmt: skip
    assert all(r.patch is None for r in already.recommendations)
    # LM-EnRML and GN-EnRML take only upstream PET's projected-data localization.
    lm = rules(_facts(observation=_observation(spread_ratio={"x": 0.01}, method="lm_enrml")))
    localize = next(r for r in lm.recommendations if "ensemble_collapse" in r.addresses)
    assert localize.patch == {"method": {"localization": {"name": "adaptive_projected"}}}


def test_no_progress_is_a_cause_with_less_damping_as_its_advice() -> None:
    stuck = _observation(state="complete", method="lm_enrml", data_mismatch_means=[252.0, 251.8],
                         data_mismatch_iterations=[0, 1], normalized_data_mismatch=5.0)  # fmt: skip

    d = rules(_facts(observation=stuck))

    assert "no_progress" in _codes(d)
    advice = next(r for r in d.recommendations if "no_progress" in r.addresses)
    assert "initial damping" in advice.change


def test_the_model_is_held_to_the_facts() -> None:
    rules_d = rules(_facts(data_scores=[("y @ t0", 40.0)]))
    model = Diagnosis(
        verdict="problems",
        causes=[
            Cause(code="model_deficiency", severity="warning", evidence="y @ t0 scores 40"),
            Cause(code="ensemble_collapse", severity="warning", evidence="spread 0.001"),
            Cause(code="prior_bias", severity="info", evidence="the data sit above the prior"),
        ],
        recommendations=[
            Recommendation(change="widen the prior", rationale="bias", patch={"prior": 1}),
            Recommendation(change="check data errors", rationale="deficiency"),
        ],
        summary="The model cannot reach y at t0.",
    )

    fixed, changes = reconcile(model, rules_d, lambda p: "bad" if "prior" in p else None)

    assert _codes(fixed) == {"model_deficiency", "prior_bias"}  # invented collapse dropped
    assert fixed.recommendations[0].patch is None  # an invalid patch is dropped...
    assert fixed.recommendations[0].change == "widen the prior"  # ...the advice stays
    assert len(changes) == 2
    assert reconcile(rules_d, rules_d, lambda p: None) == (rules_d, [])


# ---------------------------------------------------------------------- facts


def test_the_facts_of_a_run_with_an_unreachable_datum(linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=40)
    csv = case.directory / "observations.csv"
    lines = csv.read_text().splitlines()
    time, datatype, value, variance = lines[1].split(",")
    # Fifty standard deviations off: no member can reach it.
    lines[1] = f"{time},{datatype},{float(value) + 50 * float(variance) ** 0.5!r},{variance}"
    csv.write_text("\n".join(lines) + "\n")
    composition.start(case.config)
    run_dir = case.directory / "runs" / "case"

    facts = diagnose_facts(run_dir)
    d = rules(facts)

    predicted = RunDirectory(run_dir).load_arrays("iter_4")["predicted"]
    observed = np.array([float(row.split(",")[2]) for row in lines[1:]])
    outside = (observed < predicted.min(axis=-1)) | (observed > predicted.max(axis=-1))
    assert facts.coverage[0].outside == int(outside.sum()) >= 1
    assert facts.data_scores[0][0] == "y @ t0"
    assert facts.data_scores[0][1] > DiagnosisSection().mahalanobis_limit
    assert "model_deficiency" in _codes(d)
    prior = RunDirectory(run_dir).load_arrays("iter_0")["state"]
    last = RunDirectory(run_dir).load_arrays("iter_4")["state"]
    assert facts.movement[0].spread_ratio == pytest.approx(
        last.std(axis=1).mean() / prior.std(axis=1).mean(), rel=1e-12
    )


def test_failures_are_grouped_by_reason(linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=24, method=GIES, simulator_extra=FAILING)
    composition.start(case.config)
    run_dir = case.directory / "runs" / "case"

    facts = diagnose_facts(run_dir)

    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    failed = [e for e in events if e.get("event") == "member_failed"]
    assert sum(f.members for f in facts.failures) == len(failed) > 0
    assert "recurring_failure" in _codes(rules(facts))


# ---------------------------------------------------------------------- the agent


def test_the_model_diagnoses_and_its_patches_are_validated(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    case = linear_case(ensemble_size=20)
    composition.start(case.config)
    run_dir = case.directory / "runs" / "case"
    answer = Diagnosis(
        verdict="problems",
        causes=[Cause(code="small_ensemble", severity="info", evidence="20 members")],
        recommendations=[
            Recommendation(
                change="localize", rationale="spurious correlations",
                patch={"method": {"localization": {"name": "adaptive"}}},
            ),
            Recommendation(
                change="use a localization that does not exist", rationale="none",
                patch={"method": {"localization": {"name": "no_such_localization"}}},
            ),
        ],
        summary="Twenty members are few for this case.",
    )  # fmt: skip
    stub = StubProvider([answer.model_dump_json()])

    result = Diagnostician(ApexService(tmp_path / "service"), stub).diagnose(run_dir)

    assert result.decided_by == "model"
    first, second = result.diagnosis.recommendations
    assert first.patch == {"method": {"localization": {"name": "adaptive"}}}
    assert second.patch is None
    assert "no_such_localization" in result.corrections[0]
    [request] = stub.requests
    assert "stop_job" not in [t.name for t in request.tools]
    assert "submit_run" not in [t.name for t in request.tools]
    assert '"data_scores"' in request.messages[1].content
    assert result.trace.parent == run_dir / "agents"
    [action] = (run_dir / "agents").glob("diagnosis-*-action.json")
    assert json.loads(action.read_text())["corrections"] == list(result.corrections)


def test_without_a_model_the_rules_diagnose(tmp_path: Path, linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=20)
    composition.start(case.config)
    run_dir = case.directory / "runs" / "case"
    service = ApexService(tmp_path / "service")

    unavailable = Diagnostician(service, StubProvider([LLMUnavailable("down")])).diagnose(run_dir)
    rules_only = Diagnostician(service, None).diagnose(run_dir)

    assert unavailable.decided_by == "fallback"
    assert unavailable.diagnosis == unavailable.rules_diagnosis
    assert rules_only.decided_by == "rules"
    decision = json.loads(rules_only.trace.read_text())
    assert decision["decided_by"] == "rules"
    assert decision["thresholds"]["diagnosis"]["mahalanobis_limit"] == 10.83


def test_the_command(tmp_path: Path, linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=20)
    composition.start(case.config)
    run_dir = case.directory / "runs" / "case"

    result = CliRunner().invoke(
        app,
        ["agent", "diagnose", str(run_dir), "--no-llm", "--json", "--root",
         str(tmp_path / "service")],
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    document = json.loads(result.stdout)
    assert document["decided_by"] == "rules"
    assert document["facts"]["observation"]["method"] == "esmda"

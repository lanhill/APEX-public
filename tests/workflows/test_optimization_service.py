"""Optimizations as jobs of the service: placement, reuse, stop, resume, approval.

The objective is NPV on the ``toy_reservoir`` simulator (analytic, in the worker process).
"""

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apex.services.api import ApexService
from apex.services.approval import ApprovalRequired
from apex.services.policy import violations
from apex.services.settings import PolicySection
from apex.storage import optimization_directory as layout

TIMEOUT = 120.0
TIMES = [f"{2001 + k}-01-01" for k in range(4)]

CONFIG: dict[str, Any] = {
    "optimization": {"name": "toy", "seed": 3},
    "controls": {
        "variables": [
            {
                "name": "inj",
                "initial": [300.0, 250.0],
                "lower": 10.0,
                "upper": 400.0,
                "variance": 100.0,
            }
        ]
    },
    "objective": {
        "name": "npv",
        "options": {
            "start": "2000-01-01",
            "times": TIMES,
            "wop": 1.0,
            "wwi": 1.0,
            "disc": 0.08,
            "obj_scaling": -1.0,
        },
    },
    "optimizer": {"name": "enopt", "options": {"ne": 4, "maxiter": 3, "alpha": 0.1}},
    "simulator": {"name": "toy_reservoir"},
    "execution": {"name": "serial", "options": {"retries": 0}},
}


def _config(**changes: Any) -> dict[str, Any]:  # noqa: ANN401
    config = copy.deepcopy(CONFIG)
    for section, values in changes.items():
        config[section].update(values)
    return config


@pytest.fixture
def service(tmp_path: Path) -> ApexService:
    return ApexService(tmp_path / "service", max_concurrent_jobs=2)


def test_an_optimization_job_runs_in_its_place_and_is_reused(
    service: ApexService, tmp_path: Path
) -> None:
    validation = service.validate_optimization(_config(), tmp_path)
    assert validation.valid, validation.errors
    plan = validation.optimization
    assert plan is not None
    assert plan.max_simulator_runs == plan.max_evaluations

    handle = service.submit_optimization(_config(), base_dir=tmp_path)
    assert service.wait(handle.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
    [row] = service.optimizations()
    path = Path(row["path"])
    assert path == service.root / "optimizations" / "toy" / plan.optimization_id[:12]
    summary = service.optimization_summary(path)
    assert summary.state == "finished"
    assert summary.simulated
    assert summary.objective_scaling == -1.0
    assert not summary.penalised
    assert summary.history[-1]["objective"] < summary.history[0]["objective"]
    # The plan's bound holds: every evaluation is one journaled simulation.
    assert plan.max_evaluations is not None
    assert 0 < summary.evaluations <= plan.max_evaluations

    again = service.submit_optimization(_config(), base_dir=tmp_path)
    assert service.wait(again.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
    assert service.optimization_summary(path).history == summary.history  # nothing re-run


def test_a_stopped_optimization_resumes_to_the_uninterrupted_result(
    service: ApexService, tmp_path: Path
) -> None:
    whole = service.submit_optimization(
        _config(optimization={"output_dir": str(tmp_path / "whole")}), base_dir=tmp_path
    )
    part = service.submit_optimization(
        _config(optimization={"output_dir": str(tmp_path / "part")}),
        base_dir=tmp_path,
        max_iterations=1,
    )
    assert service.wait(whole.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
    assert service.wait(part.id, timeout_s=TIMEOUT, poll_s=0.2).state == "stopped"
    assert service.optimization_summary(tmp_path / "part").state == "stopped"
    resumed = service.resume(part.id)
    assert resumed.kind == "optimization_resume"
    assert service.wait(resumed.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
    a = layout.read_result(tmp_path / "whole")
    b = layout.read_result(tmp_path / "part")
    np.testing.assert_array_equal(a["x"], b["x"])


def test_a_large_optimization_needs_approval(service: ApexService, tmp_path: Path) -> None:
    big = _config(optimizer={"options": {"ne": 100, "maxiter": 300}})
    with pytest.raises(ApprovalRequired, match="simulator runs"):
        service.submit_optimization(big, base_dir=tmp_path)


def test_the_policy_judges_an_optimizations_plugins(service: ApexService, tmp_path: Path) -> None:
    from apex.services.optimization import resolve_optimization_config

    allowed = tmp_path / "allowed"
    rules = PolicySection(
        input_roots=[str(allowed)], code_roots=[str(allowed)], output_roots=[str(allowed)]
    )
    template = tmp_path / "elsewhere" / "deck.mako"
    config = resolve_optimization_config(
        _config(
            simulator={"name": "opm_flow", "options": {"deck_template_file": str(template)}},
            optimization={"output_dir": str(allowed / "out")},
        ),
        tmp_path,
    )
    problems = violations(config, rules, service.root)
    assert any("deck_template_file" in p and "code_roots" in p for p in problems)


def test_the_tools_validate_submit_and_summarize_an_optimization(
    service: ApexService, tmp_path: Path
) -> None:
    import tomli_w

    from apex.services.tools import ToolSession, call

    # In the workspace: tool clients name files only there (release tester N3).
    path = service.workspace().roots[0] / "toy.toml"
    path.write_text(tomli_w.dumps(_config()), encoding="utf-8")
    session = ToolSession(service, client="llm-test")
    checked = call("validate", {"config_file": str(path)}, session)["result"]
    assert checked["valid"], checked
    assert checked["plan"]["objective"] == "npv"
    with_bad_token = call(
        "submit_optimization", {"optimization_file": str(path), "approval_token": "nope"}, session
    )
    assert "error" in with_bad_token
    submitted = call(
        "submit_optimization",
        {"optimization_file": str(path), "approval_token": checked["approval_token"]},
        session,
    )["result"]
    status = service.wait(submitted["job_id"], timeout_s=TIMEOUT, poll_s=0.2)
    assert status.state == "complete"
    [row] = service.optimizations()
    summary = call("optimization_summary", {"path": row["path"]}, session)["result"]
    assert summary["state"] == "finished"
    assert summary["result"]["objective"] < summary["history"][0]["objective"]


def test_a_study_of_optimizations_runs_and_compares(service: ApexService, tmp_path: Path) -> None:
    """Two optimizers x two variants x two seeds, compared by objective."""
    import tomli_w

    case = tmp_path / "toy.toml"
    case.write_text(tomli_w.dumps(_config()), encoding="utf-8")
    study = tmp_path / "study.toml"
    study.write_text(
        tomli_w.dumps(
            {
                "study": {
                    "kind": "optimization",
                    "name": "toy_optimizers",
                    "replicates": 2,
                    "cases": {"toy": "toy.toml"},
                    "methods": {
                        "enopt": {"name": "enopt", "options": {"ne": 4, "maxiter": 2}},
                        "smcopt": {"name": "smcopt", "options": {"ne": 4, "maxiter": 2}},
                    },
                    "variants": [
                        {"name": "base"},
                        {"name": "cheap_water", "patch": {"objective": {"options": {"wwi": 0.5}}}},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    validation = service.validate_study(study)
    assert validation.valid, validation.errors
    handle = service.submit_study(study)
    assert service.wait(handle.id, timeout_s=TIMEOUT * 2, poll_s=0.5).state == "complete"
    comparison = service.compare("toy_optimizers")
    assert comparison.missing == []
    final = [m for m in comparison.metrics if m.metric == "objective:final"]
    assert len(final) == 1
    labels = {v.label() for v in final[0].variants}
    assert labels == {"enopt/base", "enopt/cheap_water", "smcopt/base", "smcopt/cheap_water"}
    assert all(v.n == 2 for v in final[0].variants)
    # Cheaper water injection raises NPV: a lower objective (-NPV) for the same method.
    by = {v.label(): v.mean for v in final[0].variants}
    assert by["enopt/cheap_water"] < by["enopt/base"]
    from apex.domain.optimization_study import OptimizationVariant

    with pytest.raises(ValueError, match="patches \\[optimizer\\]"):
        OptimizationVariant(name="x", patch={"optimizer": {"name": "smcopt"}})


def test_a_study_compares_ga_and_pso_with_enopt(service: ApexService, tmp_path: Path) -> None:
    """The population optimizers run as cells of a study of optimizations, next to
    an ensemble optimizer, and are compared by objective and evaluations."""
    import tomli_w

    case = tmp_path / "toy.toml"
    case.write_text(tomli_w.dumps(_config()), encoding="utf-8")
    study = tmp_path / "study.toml"
    small = {"pop_size": 4, "maxiter": 2, "transform": True}
    study.write_text(
        tomli_w.dumps({"study": {
            "kind": "optimization", "name": "toy_global", "replicates": 1,
            "cases": {"toy": "toy.toml"},
            "methods": {
                "enopt": {"name": "enopt", "options": {"ne": 4, "maxiter": 2}},
                "ga": {"name": "ga", "options": small},
                "pso": {"name": "pso", "options": small},
            },
            "variants": [{"name": "base"}],
        }}),
        encoding="utf-8",
    )  # fmt: skip
    validation = service.validate_study(study)
    assert validation.valid, validation.errors
    handle = service.submit_study(study)
    assert service.wait(handle.id, timeout_s=TIMEOUT * 2, poll_s=0.5).state == "complete"
    comparison = service.compare("toy_global")
    assert comparison.missing == []
    final = next(m for m in comparison.metrics if m.metric == "objective:final")
    assert {v.label() for v in final.variants} == {"enopt/base", "ga/base", "pso/base"}


def test_a_study_compares_fronts_by_hypervolume(service: ApexService, tmp_path: Path) -> None:
    """Cells with several objectives are compared by the hypervolume of their fronts
    against one reference point for the study, and by their fronts' sizes."""
    import tomli_w

    case = tmp_path / "zdt1.toml"
    case.write_text(tomli_w.dumps({
        "optimization": {"name": "zdt1", "seed": 3},
        "controls": {"variables": [{"name": "x", "initial": [0.5] * 5, "lower": 0.0,
                                    "upper": 1.0}]},
        "objective": {"name": "zdt1"},
        "optimizer": {"name": "nsga2", "options": {"pop_size": 8, "maxiter": 3}},
    }), encoding="utf-8")  # fmt: skip
    study = tmp_path / "study.toml"
    study.write_text(tomli_w.dumps({"study": {
        "kind": "optimization", "name": "fronts", "replicates": 1,
        "cases": {"zdt1": "zdt1.toml"},
        "methods": {
            "small": {"name": "nsga2", "options": {"pop_size": 8, "maxiter": 3}},
            "large": {"name": "nsga2", "options": {"pop_size": 16, "maxiter": 6}},
        },
        "variants": [{"name": "base"}],
        "metrics": ["hypervolume:final", "front_size:final", "objective:final"],
        "reference_point": [1.1, 10.0],
    }}), encoding="utf-8")  # fmt: skip
    assert service.validate_study(study).valid
    handle = service.submit_study(study)
    assert service.wait(handle.id, timeout_s=TIMEOUT * 2, poll_s=0.5).state == "complete"
    comparison = service.compare("fronts")
    by_metric = {m.metric: {v.label(): v.mean for v in m.variants} for m in comparison.metrics}
    hv = by_metric["hypervolume:final"]
    assert set(hv) == {"small/base", "large/base"}
    assert all(0.0 < value < 1.1 * 10.0 for value in hv.values())
    assert by_metric["front_size:final"]["large/base"] >= 1


def test_a_studys_reference_point_is_refused_for_unlike_fronts(tmp_path: Path) -> None:
    """Review finding (2026-10-06): cells with different numbers of objectives, or a
    reference point of the wrong length, raised a bare ValueError (a 500 from the API)."""
    from apex.core.errors import ConfigError
    from apex.services.optimization_study import _study_reference

    two, three = np.ones((2, 4)), np.ones((3, 5))
    with pytest.raises(ConfigError, match="objectives"):
        _study_reference({tmp_path / "a": two, tmp_path / "b": three}, None)
    with pytest.raises(ConfigError, match="reference_point has 3 values for 2"):
        _study_reference({tmp_path / "a": two}, [1.0, 2.0, 3.0])
    infeasible = np.empty((2, 0))  # a front with no feasible member still has 2 objectives
    reference = _study_reference({tmp_path / "a": two, tmp_path / "b": infeasible}, None)
    assert reference is not None
    assert reference.shape == (2,)


def test_the_report_draws_the_front_and_the_hypervolume(
    service: ApexService, tmp_path: Path
) -> None:
    import tomli_w

    from apex.services import optimization as optimizations

    pytest.importorskip("matplotlib")
    case = tmp_path / "zdt1.toml"
    case.write_text(tomli_w.dumps({
        "optimization": {"name": "fig", "seed": 3},
        "controls": {"variables": [{"name": "x", "initial": [0.5] * 4, "lower": 0.0,
                                    "upper": 1.0}]},
        "objective": {"name": "zdt1"},
        "optimizer": {"name": "nsga2", "options": {"pop_size": 8, "maxiter": 3}},
    }), encoding="utf-8")  # fmt: skip
    run = optimizations.optimize(case)
    written = optimizations.write_optimization_figures(run.run_dir, tmp_path / "figures")
    assert sorted(p.name for p in written) == ["front.png", "hypervolume.png", "objective.png"]
    assert all(p.with_suffix(".pdf").exists() for p in written)


@pytest.mark.parametrize(
    ("metrics", "valid"),
    [
        (["objective:final", "objective:best", "evaluations", "iterations:all"], True),
        (["hypervolume:final", "front_size:final"], True),
        (["objective:mean"], False),  # a known value, but not reported so
        (["hypervolume"], False),  # bare: hypervolume:all, which nothing reports
        (["objective"], False),  # bare: objective:all
        (["cost:final"], False),
    ],
)
def test_a_studys_metrics_must_be_ones_the_comparison_reports(
    metrics: list[str], valid: bool
) -> None:
    """A reported issue's follow-up: a known name with an unreported label validated and
    then compared as an empty column."""
    from pydantic import ValidationError

    from apex.domain.optimization_study import OptimizationStudySpec

    spec = {"study": {"kind": "optimization", "name": "s", "cases": {"c": "c.toml"},
                      "methods": {"m": {"name": "enopt"}}, "variants": [{"name": "base"}],
                      "metrics": metrics}}  # fmt: skip
    if valid:
        OptimizationStudySpec.model_validate(spec)
    else:
        with pytest.raises(ValidationError, match="reports objective:final"):
            OptimizationStudySpec.model_validate(spec)


def test_comparing_by_an_unreported_metric_is_refused(tmp_path: Path) -> None:
    """Review finding: ``apex study compare NAME --metric objective:mean`` gave an empty column."""
    from apex.core.errors import ConfigError
    from apex.services.optimization_study import compare_optimization_study

    with pytest.raises(ConfigError, match="unknown metrics"):
        compare_optimization_study(ApexService(tmp_path), "any", ["objective:mean"])


def test_an_optimization_jobs_stop_asked_before_its_lock_is_kept(tmp_path: Path) -> None:
    """The review of the follow-ups: an optimization withdraws the last stop once
    it holds its lock; the job's own stop, asked for before that, is requested again, so it
    stops at its first checkpoint instead of running to the end."""
    from apex.domain.optimization import OptimizationConfig
    from apex.services import optimization

    config = OptimizationConfig.model_validate(
        _config(optimization={"output_dir": str(tmp_path / "opt")})
    )
    run = optimization.optimize_config(config, stopped=lambda: True)
    assert not run.finished
    assert len(layout.read_history(tmp_path / "opt")) == 1  # the start only


def test_the_summary_says_a_run_has_several_objectives_before_its_first_front(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Release tester U3 (review): the web UI tells a multi-objective run from the start."""
    from apex.services import optimization as optimizations

    controls = {"variables": [{"name": "x", "initial": [0.5] * 4, "lower": 0.0, "upper": 1.0}]}
    # A vector objective plugin with a multi-objective optimizer: no history, no front yet.
    vector = tmp_path / "vector"
    layout.write_config(vector, {
        "optimization": {"name": "v", "seed": 1}, "controls": controls,
        "objective": {"name": "zdt1"},
        "optimizer": {"name": "nsga2", "options": {"pop_size": 8, "maxiter": 3}},
    })  # fmt: skip
    summary = optimizations.optimization_summary(vector)
    assert (summary.state, summary.pareto, summary.history) == ("new", None, [])
    assert summary.multi_objective
    assert summary.objective_names == ["f1", "f2"]
    assert summary.objective_scalings is None

    # A plugin that refuses its options (an ApexError) leaves the names unknown, not a failure.
    from apex.algorithms.optimization import objectives
    from apex.core.errors import ConfigError

    def refuse(*args: object, **kwargs: object) -> object:
        raise ConfigError("refused")

    with monkeypatch.context() as patch:
        patch.setattr(objectives, "create", refuse)
        refused = optimizations.optimization_summary(vector)
    assert refused.multi_objective
    assert refused.objective_names is None

    # [[objectives]]: their names as results show them, and each one's obj_scaling.
    listed = tmp_path / "listed"
    layout.write_config(listed, {
        "optimization": {"name": "l", "seed": 1}, "controls": controls,
        "objectives": [
            {"name": "rosenbrock", "label": "a", "options": {"obj_scaling": -2.0}},
            {"name": "quadratic", "label": "b"},
        ],
        "optimizer": {"name": "nsga2", "options": {"pop_size": 8, "maxiter": 3}},
    })  # fmt: skip
    summary = optimizations.optimization_summary(listed)
    assert summary.multi_objective
    assert summary.objective_names == ["a", "b"]
    assert summary.objective_scalings == [-2.0, None]

    # One objective: none of it.
    single = tmp_path / "single"
    layout.write_config(single, {
        "optimization": {"name": "s", "seed": 1}, "controls": controls,
        "objective": {"name": "rosenbrock"},
        "optimizer": {"name": "enopt", "options": {"ne": 4, "maxiter": 2}},
    })  # fmt: skip

    def unexpected(*args: object, **kwargs: object) -> object:
        raise AssertionError("no objective plugin is created for a single-objective run")

    with monkeypatch.context() as patch:
        patch.setattr(objectives, "create", unexpected)
        summary = optimizations.optimization_summary(single)
    assert not summary.multi_objective
    assert summary.objective_names is None
    assert summary.objective_scalings is None

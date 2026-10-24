"""APEX's optimizers as a ropt backend: the path ropt drives is APEX's own.

ropt's ``BasicOptimizer`` runs ``apex/<optimizer>`` (found through the entry point
``ropt.plugins.backend``) on analytic functions; no simulator, no Everest server.
"""

from typing import Any

import numpy as np
import pytest

pytest.importorskip("ropt")

from ropt.enums import ExitCode
from ropt.evaluation import EvaluationBatchContext, EvaluationBatchResult
from ropt.workflow import BasicOptimizer
from scipy.optimize import rosen

from apex.algorithms.optimization import EnOpt, GeneratorDraws, Problem, Rosenbrock

SEED = 123
X0 = np.array([-0.8, 0.6, 0.2])
MAGNITUDE = 0.05


def _rosenbrock(variables: np.ndarray, context: EvaluationBatchContext) -> EvaluationBatchResult:
    return EvaluationBatchResult(objectives=np.array([[rosen(row)] for row in variables]))


def _config(method: str, options: dict[str, Any], **extra: Any) -> dict[str, Any]:  # noqa: ANN401
    return {
        "variables": {
            "variable_count": X0.size,
            "lower_bounds": [-2.0] * X0.size,
            "upper_bounds": [2.0] * X0.size,
            "perturbation_magnitudes": MAGNITUDE,
            "seed": (SEED,),
        },
        "backend": {"method": method, "options": options},
        **extra,
    }


def test_enopt_through_ropt_follows_apex_s_own_path() -> None:
    """Every point ropt evaluates for apex/enopt is the one APEX's EnOpt evaluates directly.

    (ropt's ``results`` is the best point it saw, which may be a perturbed member, so the
    whole sequence of evaluated points is compared instead.)
    """
    options: dict[str, Any] = {"ne": 8, "maxiter": 6, "alpha": 0.05}
    seen_by_ropt: list[np.ndarray] = []

    def recorded(variables: np.ndarray, context: EvaluationBatchContext) -> EvaluationBatchResult:
        seen_by_ropt.extend(np.array(row) for row in variables)
        return _rosenbrock(variables, context)

    runner = BasicOptimizer(_config("apex/enopt", options), recorded)
    assert runner.run(X0) == ExitCode.OPTIMIZER_FINISHED

    seen_by_apex: list[np.ndarray] = []

    class Recorded(Rosenbrock):
        def __call__(self, controls: np.ndarray, realizations: np.ndarray) -> np.ndarray:
            seen_by_apex.extend(np.array(column) for column in controls.T)
            return super().__call__(controls, realizations)

    (child,) = np.random.SeedSequence([SEED]).spawn(1)
    direct = EnOpt(**options).minimize(
        Problem(Recorded(), X0, lower=np.full(3, -2.0), upper=np.full(3, 2.0)),
        GeneratorDraws(np.random.default_rng(child)),
        cov=np.eye(3) * MAGNITUDE**2,
    )
    assert len(seen_by_ropt) == len(seen_by_apex)
    np.testing.assert_array_equal(np.array(seen_by_ropt), np.array(seen_by_apex))
    assert rosen(direct.x) < rosen(X0)


def test_line_search_uses_ropt_s_gradient() -> None:
    runner = BasicOptimizer(
        _config("apex/line_search", {"maxiter": 5, "method": "BFGS"}), _rosenbrock
    )
    runner.run(X0)
    assert runner.results is not None
    assert rosen(runner.results.evaluations.variables) < rosen(X0)


def test_nonlinear_constraints_become_the_external_penalty() -> None:
    """minimise ||x - 1||^2 / 2 subject to sum(x) <= 1.5: the optimum is near x = 0.5."""

    def evaluate(variables: np.ndarray, context: EvaluationBatchContext) -> EvaluationBatchResult:
        objective = 0.5 * np.sum((variables - 1.0) ** 2, axis=1, keepdims=True)
        constraint = np.sum(variables, axis=1, keepdims=True)
        return EvaluationBatchResult(objectives=objective, constraints=constraint)

    options = {"ne": 10, "maxiter": 30, "alpha": 0.2, "epf": {"r": 10.0, "max_epf_iter": 6}}
    runner = BasicOptimizer(
        _config(
            "apex/enopt",
            options,
            nonlinear_constraints={"lower_bounds": [-np.inf], "upper_bounds": [1.5]},
        ),
        evaluate,
    )
    runner.run(np.zeros(3))
    assert runner.results is not None
    x = runner.results.evaluations.variables
    assert np.sum(x) < 1.6
    np.testing.assert_allclose(x, 0.5, atol=0.15)


@pytest.mark.parametrize(
    ("method", "options", "extra", "message"),
    [
        ("apex/line_search", {"method": "Newton-CG"}, {}, "Hessian"),
        ("apex/enopt", {"ne": 4}, {"nonlinear_constraints": {"lower_bounds": [0.0],
                                                             "upper_bounds": [1.0]}}, "epf"),
        ("apex/trust_region", {}, {"nonlinear_constraints": {"lower_bounds": [0.0],
                                                             "upper_bounds": [1.0]}}, "ensemble"),
        ("apex/enopt", {"ne": 4, "nonsense": 1}, {}, "nonsense"),
    ],
)  # fmt: skip
def test_what_apex_cannot_honour_is_refused(
    method: str,
    options: dict[str, Any],
    extra: dict[str, Any],
    message: str,
) -> None:
    def evaluate(variables: np.ndarray, context: EvaluationBatchContext) -> EvaluationBatchResult:
        objective = np.array([[rosen(row)] for row in variables])
        has = "nonlinear_constraints" in extra
        return EvaluationBatchResult(
            objectives=objective,
            constraints=np.sum(variables, axis=1, keepdims=True) if has else None,
        )

    with pytest.raises(Exception, match=message):
        BasicOptimizer(_config(method, options, **extra), evaluate).run(X0)


def test_relative_perturbations_scale_with_the_range() -> None:
    """A RELATIVE magnitude m is m (upper - lower), as ropt's own gradient (review 2026-10-03)."""
    seen: dict[str, list[np.ndarray]] = {"absolute": [], "relative": []}

    def recording(label: str) -> Any:  # noqa: ANN401
        def evaluate(
            variables: np.ndarray, context: EvaluationBatchContext
        ) -> EvaluationBatchResult:
            seen[label].extend(np.array(row) for row in variables)
            return _rosenbrock(variables, context)

        return evaluate

    options = {"ne": 6, "maxiter": 3, "alpha": 0.05}
    absolute = _config("apex/enopt", options)
    absolute["variables"]["perturbation_magnitudes"] = 0.1
    relative = _config("apex/enopt", options)
    relative["variables"]["perturbation_magnitudes"] = 0.025  # x the range 4 = 0.1
    relative["variables"]["perturbation_types"] = 2  # ropt's PerturbationType.RELATIVE
    BasicOptimizer(absolute, recording("absolute")).run(X0)
    BasicOptimizer(relative, recording("relative")).run(X0)
    np.testing.assert_allclose(np.array(seen["relative"]), np.array(seen["absolute"]), rtol=1e-12)


def test_an_equality_constraint_is_approached_by_the_penalty() -> None:
    """x1 + x2 = 1 with x3 fixed (ropt's mask): APEX's iterates end near the constraint.

    The penalty approaches it from outside, within conv_crit: ropt's own "best" result
    needs feasibility within its tolerance (1e-10), so what APEX evaluated is checked.
    """
    seen: list[np.ndarray] = []

    def evaluate(variables: np.ndarray, context: EvaluationBatchContext) -> EvaluationBatchResult:
        seen.extend(np.array(row) for row in variables)
        objective = 0.5 * np.sum((variables[:, :2] - 1.0) ** 2, axis=1, keepdims=True)
        return EvaluationBatchResult(
            objectives=objective, constraints=np.sum(variables[:, :2], axis=1, keepdims=True)
        )

    config = _config(
        "apex/enopt",
        {"ne": 10, "maxiter": 30, "alpha": 0.2, "epf": {"r": 100.0, "max_epf_iter": 8}},
        nonlinear_constraints={"lower_bounds": [1.0], "upper_bounds": [1.0]},
    )
    config["variables"]["mask"] = [True, True, False]
    BasicOptimizer(config, evaluate).run(np.array([0.0, 0.0, 0.3]))
    last = seen[-1]
    assert last[2] == 0.3  # the fixed variable never moves
    assert abs(last[0] + last[1] - 1.0) < 0.02
    np.testing.assert_allclose(last[:2], 0.5, atol=0.15)  # still moving along the constraint


def test_the_penalty_is_refused_without_constraints() -> None:
    with pytest.raises(Exception, match="needs nonlinear constraints"):
        BasicOptimizer(_config("apex/enopt", {"ne": 4, "epf": {"r": 1.0}}), _rosenbrock).run(X0)

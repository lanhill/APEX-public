"""ropt's optimizers as an APEX optimizer, through ``apex optimize``'s composition.

Independent references: the known optima of APEX's analytic objectives, Rosenbrock (all
ones, value 0) and PET's quadratic ((x - 1)^T (I/2) (x - 1): all ones, value 0), reached by
SciPy's SLSQP and L-BFGS-B through ropt with its ensemble gradient; the toy reservoir's
known optimum (each injector at (p_o oil_rate / (2 p_w water_rate))^2 = 100).
"""

import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("ropt")

from apex.algorithms.optimization import optimizers
from apex.core.errors import ConfigError
from apex.services import optimization
from apex.storage import optimization_directory as layout

CONFIG = """
[optimization]
name = "{name}"
seed = 7

[controls]
realizations = {realizations}

[[controls.variables]]
name = "x"
initial = [-0.5, 0.5, 0.2]
lower = -2.0
upper = 2.0

[objective]
name = "{objective}"

[optimizer]
name = "ropt"
options = {options}
"""


def _config(
    tmp_path: Path,
    name: str,
    options: str,
    objective: str = "quadratic",
    realizations: int = 1,
) -> Path:
    path = tmp_path / f"{name}.toml"
    path.write_text(
        CONFIG.format(name=name, options=options, objective=objective,
                      realizations=realizations),
        encoding="utf-8",
    )  # fmt: skip
    return path


@pytest.mark.parametrize("method", ["scipy/slsqp", "scipy/l-bfgs-b"])
def test_ropt_reaches_the_quadratics_optimum(tmp_path: Path, method: str) -> None:
    options = (
        f'{{ method = "{method}", maxiter = 60, number_of_perturbations = 8, '
        "perturbation_magnitude = 0.001, max_batches = 400 }"
    )
    run = optimization.optimize(_config(tmp_path, "quad", options))
    result = layout.read_result(run.run_dir)
    np.testing.assert_allclose(result["x"], np.ones(3), atol=0.02)
    assert result["objective"] < 1e-3
    history = layout.read_history(run.run_dir)
    assert history[0]["kind"] == "start"
    objectives = [row["objective"] for row in history]
    assert all(b < a for a, b in pairwise(objectives))  # each step improves


def test_ropt_on_rosenbrock_and_its_evaluation_bound(tmp_path: Path) -> None:
    options = (
        '{ method = "scipy/slsqp", maxiter = 200, number_of_perturbations = 10, '
        "perturbation_magnitude = 0.0005, max_batches = 600 }"
    )
    config = optimization.load_optimization_config(
        _config(tmp_path, "rosen", options, objective="rosenbrock")
    )
    plan = optimization.plan_optimization(config)
    # The start, then at most 600 batches of 1 x (10 + 1) evaluations.
    assert plan.max_evaluations == 1 + 600 * 11
    # Its terms, for a planner to state (a reported issue, #235).
    assert plan.evaluations_basis == (
        "1 for the start + 600 batches x 1 realization x (10 perturbations + 1) = 6601"
    )
    run = optimization.optimize(_config(tmp_path, "rosen", options, objective="rosenbrock"))
    result = layout.read_result(run.run_dir)
    assert result["objective"] < 0.05  # from 1.5 ... at (-0.5, 0.5, 0.2)
    assert result["member_evaluations"] <= plan.max_evaluations
    no_bound = optimizers.create("ropt", method="scipy/slsqp")
    assert no_bound.max_evaluations(3) is None  # without max_batches: unknown
    assert no_bound.max_evaluations_basis(3) is None


def test_unknown_methods_and_epf_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no ropt backend"):
        optimizers.create("ropt", method="scipy/no_such_method")
    with pytest.raises(ValueError, match="<backend>/<method>"):
        optimizers.create("ropt", method="slsqp")
    with pytest.raises(ConfigError):
        optimization.optimize(_config(tmp_path, "bad", '{ method = "nope/x" }'))
    with pytest.raises(ValueError, match="takes no epf"):
        optimizers.create("ropt", epf={"r": 1.0})


def test_a_stopped_run_resumes_as_the_uninterrupted_one(tmp_path: Path) -> None:
    """ropt cannot continue mid-run, so a resume replays its run from the start with the
    same seed (deterministic) and continues: history and result are the uninterrupted run's
    exactly (2026-10-04)."""
    options = (
        '{ method = "scipy/slsqp", maxiter = 100, number_of_perturbations = 6, '
        "perturbation_magnitude = 0.001, max_batches = 400 }"
    )
    whole = optimization.optimize(_config(tmp_path, "whole", options, "rosenbrock"))
    full = layout.read_history(whole.run_dir)
    assert len([r for r in full if r["kind"] == "step"]) > 4
    part = optimization.optimize(_config(tmp_path, "part", options, "rosenbrock"), max_iterations=3)
    assert len([r for r in layout.read_history(part.run_dir) if r["kind"] == "step"]) == 3
    resumed = optimization.resume(part.run_dir)
    assert layout.read_history(resumed.run_dir) == full
    assert layout.read_result(resumed.run_dir)["x"] == layout.read_result(whole.run_dir)["x"]


def test_a_stop_is_seen_before_the_next_batch() -> None:
    """The workflow's stop request is asked before every batch, not only at accepted
    steps; the run stops with its last step standing."""
    from apex.algorithms.optimization import Problem, Rosenbrock

    problem = Problem(objective=Rosenbrock(), x0=np.full(3, 1.5), lower=np.full(3, -2.0),
                      upper=np.full(3, 2.0))  # fmt: skip
    optimizer = optimizers.create(
        "ropt", method="scipy/slsqp", maxiter=100, number_of_perturbations=4,
        perturbation_magnitude=0.001, seed=3,
    )  # fmt: skip
    asked = {"n": 0}

    def stop() -> bool:
        asked["n"] += 1
        return asked["n"] > 3  # the fourth batch is refused

    optimizer.stop_check = stop
    result = optimizer.minimize(problem)
    assert optimizer.stopped
    assert result.member_evaluations <= 1 + 3 * 5  # the start, then three batches at most


def test_robust_realizations_are_weighed_equally() -> None:
    """Realization r's objective is 0.5 |x - s_r|^2: the mean over equally weighted
    realizations is minimal at the mean of the shifts (analytic reference)."""
    from apex.algorithms.optimization import Problem

    shifts = np.array([[0.2, -0.4], [1.0, 0.6], [-0.3, 0.1]])  # (n_real, n_x)

    def shifted(controls: np.ndarray, realizations: np.ndarray) -> np.ndarray:
        u = controls - shifts[realizations].T
        return 0.5 * np.sum(u * u, axis=0)

    problem = Problem(
        objective=shifted, x0=np.array([1.5, -1.5]), lower=np.full(2, -2.0),
        upper=np.full(2, 2.0), n_realizations=3,
    )  # fmt: skip
    optimizer = optimizers.create(
        "ropt", method="scipy/slsqp", maxiter=60, number_of_perturbations=6,
        perturbation_magnitude=0.001, seed=5,
    )  # fmt: skip
    result = optimizer.minimize(problem)
    np.testing.assert_allclose(result.x, shifts.mean(axis=0), atol=0.02)


def test_nonlinear_constraints_reach_ropt() -> None:
    """A constrained objective's c_iq >= 0 is ropt's constraint: min |x - 1|^2 / 2 with
    x_0 <= 0.5 ends on the constraint (independent reference: the projection)."""
    from apex.algorithms.optimization import Problem, Quadratic

    class Capped(Quadratic):
        def constraints(self, controls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return np.zeros((0, controls.shape[1])), 0.5 - controls[:1]

    problem = Problem(
        objective=Capped(), x0=np.array([0.0, 0.0]), lower=np.full(2, -2.0),
        upper=np.full(2, 2.0),
    )  # fmt: skip
    optimizer = optimizers.create(
        "ropt", method="scipy/slsqp", maxiter=60, number_of_perturbations=8,
        perturbation_magnitude=0.001, seed=3, constraints=True,
    )  # fmt: skip
    result = optimizer.minimize(problem)
    np.testing.assert_allclose(result.x, [0.5, 1.0], atol=0.02)


@pytest.mark.parametrize("method", ["scipy/cobyla", "scipy/slsqp"])
def test_only_feasible_points_are_accepted_and_an_infeasible_start_is_left(method: str) -> None:
    """From an infeasible start (x_0 = 1.8 > 0.5) the accepted steps reduce the violation,
    then improve the objective among feasible points; the result is feasible and on the
    constraint, never the unconstrained optimum (1, 1) (review of 2026-10-04)."""
    from apex.algorithms.optimization import Problem, Quadratic

    class Capped(Quadratic):
        def constraints(self, controls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return np.zeros((0, controls.shape[1])), 0.5 - controls[:1]

    problem = Problem(
        objective=Capped(), x0=np.array([1.8, 0.0]), lower=np.full(2, -2.0),
        upper=np.full(2, 2.0),
    )  # fmt: skip
    optimizer = optimizers.create(
        "ropt", method=method, maxiter=200, number_of_perturbations=8,
        perturbation_magnitude=0.001, seed=3, constraints=True, constraint_tolerance=1e-4,
    )  # fmt: skip
    result = optimizer.minimize(problem)
    assert result.x[0] <= 0.5 + 1e-4
    np.testing.assert_allclose(result.x, [0.5, 1.0], atol=0.03)
    assert result.extra["feasible"] == 1.0
    assert result.extra["violation"] <= 1e-4


def test_an_infeasible_result_is_marked_so() -> None:
    """With a constraint no point meets (x_0^2 + 1 <= 0), the result is the least violation
    and says so in its record, not only its message (review of 2026-10-04)."""
    from apex.algorithms.optimization import Problem, Quadratic

    class Impossible(Quadratic):
        def constraints(self, controls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return np.zeros((0, controls.shape[1])), -1.0 - controls[:1] ** 2

    problem = Problem(
        objective=Impossible(), x0=np.array([0.5, 0.0]), lower=np.full(2, -2.0),
        upper=np.full(2, 2.0),
    )  # fmt: skip
    optimizer = optimizers.create(
        "ropt", method="scipy/cobyla", maxiter=50, number_of_perturbations=4,
        perturbation_magnitude=0.001, seed=3, constraints=True,
    )  # fmt: skip
    result = optimizer.minimize(problem)
    assert result.extra["feasible"] == 0.0
    assert result.extra["violation"] >= 1.0 - 1e-6  # at best x_0 = 0: violation 1
    assert "no feasible point" in result.message


def test_a_simulated_objective_runs_on_the_toy_reservoir(tmp_path: Path) -> None:
    times = [f"{2001 + k}-01-01" for k in range(4)]
    npv = (
        f'{{ start = "2000-01-01", times = {json.dumps(times)}, wop = 1.0, wwi = 1.0, '
        "obj_scaling = -1.0 }"
    )
    ropt = (
        '{ method = "scipy/slsqp", maxiter = 30, number_of_perturbations = 6, '
        "perturbation_magnitude = 0.01, max_batches = 120, transform = true }"
    )
    text = f"""
[optimization]
name = "toy"
seed = 2

[[controls.variables]]
name = "inj"
initial = [300.0, 250.0]
lower = 10.0
upper = 400.0

[objective]
name = "npv"
options = {npv}

[optimizer]
name = "ropt"
options = {ropt}

[simulator]
name = "toy_reservoir"
"""
    path = tmp_path / "toy.toml"
    path.write_text(text, encoding="utf-8")
    run = optimization.optimize(path)
    result = layout.read_result(run.run_dir)
    np.testing.assert_allclose(result["x"], [100.0, 100.0], rtol=0.1)
    journal = layout.read_evaluations(run.run_dir)
    assert len({row["key"] for row in journal}) == len(journal)  # each point simulated once


def test_a_crashed_run_resumes_without_simulating_again(tmp_path: Path) -> None:
    """A simulation fails mid-run (after the last accepted step); the resume replays ropt's
    run from the journal and simulates only what the uninterrupted run would have: the
    approved bound is not re-spent (2026-10-04)."""
    from collections.abc import Sequence

    from apex.core.errors import SimulationError, WorkflowError
    from apex.core.plugins import simulators
    from apex.core.simulation import MemberInput
    from apex.domain.observations import DataKey
    from apex.simulation.analytic import ToyReservoir

    class Counting(ToyReservoir):
        runs = 0
        fail_at: int | None = None

        def run(self, member: MemberInput, keys: Sequence[DataKey]) -> np.ndarray:
            type(self).runs += 1
            if type(self).fail_at == type(self).runs:
                type(self).fail_at = None
                raise SimulationError("failed on purpose")
            return super().run(member, keys)

    simulators.register("test_counting_toy")(Counting)
    try:
        times = [f"{2001 + k}-01-01" for k in range(4)]
        text = f"""
[optimization]
name = "{{name}}"
seed = 2

[[controls.variables]]
name = "inj"
initial = [300.0, 250.0]
lower = 10.0
upper = 400.0

[objective]
name = "npv"
options = {{{{ start = "2000-01-01", times = {json.dumps(times)}, wop = 1.0, wwi = 1.0, \
obj_scaling = -1.0 }}}}

[optimizer]
name = "ropt"
options = {{{{ method = "scipy/slsqp", maxiter = 30, number_of_perturbations = 6, \
perturbation_magnitude = 0.01, max_batches = 120, transform = true }}}}

[simulator]
name = "test_counting_toy"

[execution]
name = "serial"
options = {{{{ retries = 0 }}}}
"""
        whole_path = tmp_path / "whole.toml"
        whole_path.write_text(text.format(name="whole"))
        optimization.optimize(whole_path)
        total = Counting.runs
        Counting.runs, Counting.fail_at = 0, total // 2
        crash_path = tmp_path / "crash.toml"
        crash_path.write_text(text.format(name="crash"))
        with pytest.raises(WorkflowError, match=r"failed|simulation"):
            optimization.optimize(crash_path)
        run_dir = tmp_path / "runs" / "crash"
        whole_dir = tmp_path / "runs" / "whole"
        checkpoint = layout.load_checkpoint(run_dir)
        assert checkpoint is not None
        assert isinstance(checkpoint["iteration"], int)
        assert checkpoint["iteration"] > 0  # the replay path, past accepted steps
        optimization.resume(run_dir)
        # The failed member is simulated again; nothing else is.
        assert Counting.runs == total + 1
        # And the run is the uninterrupted one (from the journal's values).
        assert layout.read_history(run_dir) == layout.read_history(whole_dir)
        assert layout.read_result(run_dir)["x"] == layout.read_result(whole_dir)["x"]
    finally:
        simulators._factories.pop("test_counting_toy")


#: Dakota's methods measured against the bound (its genetic algorithms abort the process).
DAKOTA_MEASURED = (
    "dakota/optpp_q_newton", "dakota/conmin_frcg", "dakota/conmin_mfd",
    "dakota/asynch_pattern_search", "dakota/coliny_ea", "dakota/mesh_adaptive_search",
)  # fmt: skip


def _dakota() -> bool:
    import importlib.util

    return importlib.util.find_spec("ropt_dakota") is not None


@pytest.mark.parametrize(
    "method",
    [
        "scipy/slsqp", "scipy/l-bfgs-b", "scipy/cobyla", "scipy/nelder-mead", "scipy/powell",
        "scipy/trust-constr", "scipy/differential_evolution",
        *(pytest.param(m, marks=pytest.mark.skipif(
            not _dakota(), reason="ropt-dakota is not installed"))
          for m in DAKOTA_MEASURED),
    ],
)  # fmt: skip
def test_every_batch_fits_the_evaluation_bound(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Measured 2026-10-05: the approval bound assumes no ropt batch asks for more than
    ``n_realizations (number_of_perturbations + 1)`` rows (a function value and its
    perturbations per realization), for gradient, derivative-free and population methods
    of SciPy and Dakota alike; measured, every one asks for at most that."""
    from apex.algorithms.optimization import Problem

    shifts = np.array([[0.2, -0.4, 0.1], [1.0, 0.6, -0.2], [-0.3, 0.1, 0.4]])

    def shifted(controls: np.ndarray, realizations: np.ndarray) -> np.ndarray:
        u = controls - shifts[realizations].T
        return 0.5 * np.sum(u * u, axis=0)

    rows: list[int] = []
    problem = Problem(
        objective=shifted, x0=np.array([1.5, -1.5, 0.5]), lower=np.full(3, -2.0),
        upper=np.full(3, 2.0), n_realizations=3,
    )  # fmt: skip
    optimizer = optimizers.create(
        "ropt", method=method, maxiter=200, number_of_perturbations=4,
        perturbation_magnitude=0.001, seed=5, max_batches=12,
    )  # fmt: skip
    # The instance's method: ropt is handed it when it runs. (Not the class's: a test that
    # reloads the module leaves the registry holding an older class object.)
    evaluate = optimizer._evaluate  # type: ignore[attr-defined]

    def counted(variables: np.ndarray, context: Any) -> Any:  # noqa: ANN401
        rows.append(int(np.count_nonzero(context.active)))
        return evaluate(variables, context)

    monkeypatch.setattr(optimizer, "_evaluate", counted)
    optimizer.minimize(problem)
    assert 0 < len(rows) <= 12
    assert max(rows) <= 3 * (4 + 1)
    bound = optimizer.max_evaluations(3)
    assert bound is not None
    assert optimizer.member_evaluations <= bound


@pytest.mark.parametrize("method", ["dakota/soga", "dakota/moga", "DAKOTA/SOGA"])
def test_dakotas_genetic_algorithms_are_refused(method: str) -> None:
    """Measured 2026-10-05: Dakota's soga and moga abort the whole process inside Dakota,
    which would kill an optimization's worker; they are refused when the optimizer is built."""
    with pytest.raises(ValueError, match="refused"):
        optimizers.create("ropt", method=method)

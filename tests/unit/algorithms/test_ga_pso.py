"""The population optimizers ``ga`` and ``pso``.

Independent references, as every method PET lacks needs:
- analytic optima: the sphere (0 at 0), Rosenbrock (0 at 1), PET's constrained quadratic
  (its KKT point, worked out in the test), and the multimodal Rastrigin and Ackley functions
  (0 at 0, every other local minimum at 0.99 or more);
- pymoo 0.6.2's operators (tests/data/reference/pymoo/operators.npz, written by
  tools/pymoo_reference/generate.py in a separate environment): given a generator with the
  same seed, APEX's SBX, polynomial mutation, tournament and PSO update return pymoo's
  arrays exactly (pymoo's rows are APEX's columns: transposed here, at the boundary).
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from apex.algorithms.optimization import (
    Ackley,
    GeneratorDraws,
    OptimizationResult,
    OptimizerBase,
    Problem,
    Quadratic,
    Rastrigin,
    Rosenbrock,
    optimizers,
)
from apex.algorithms.optimization.ga import polynomial_mutation, sbx, tournament
from apex.algorithms.optimization.population import PopulationOptimizer, violation
from apex.algorithms.optimization.protocols import FloatArray, IntArray
from apex.algorithms.optimization.pso import keep_in_bounds, velocity_update

REFERENCE = Path(__file__).parents[2] / "data" / "reference" / "pymoo" / "operators.npz"
NAMES = ("ga", "pso")
#: SBX and polynomial mutation call ``pow``, whose last bit depends on the platform's libm:
#: the same draws give pymoo's children exactly here, and within a few ulps on GitHub's
#: runners (CI, 2026-10-06: 2 of 96 and 2 of 120 values off by 1.1e-16, relative 2.6e-16). The
#: PSO update only multiplies, adds and clips, each a separate ufunc (correctly rounded, no
#: FMA across them), and stays exact. The trajectory tests below (seeds reaching the basin,
#: the comparison with pymoo's statistics) depend on libm too, through pow, cos and exp; they
#: are statistical or within tolerances.
POW_RTOL, POW_ATOL = 1e-13, 1e-15


def _sphere(controls: FloatArray, realizations: IntArray) -> FloatArray:
    return np.asarray(np.sum(controls * controls, axis=0), dtype=float)


def _problem(objective: object, x0: list[float], bound: float, n_realizations: int = 1) -> Problem:
    n = len(x0)
    return Problem(
        objective,  # type: ignore[arg-type]
        np.array(x0, dtype=float),
        lower=np.full(n, -bound),
        upper=np.full(n, bound),
        n_realizations=n_realizations,
    )


def _run(
    name: str, problem: Problem, seed: int, **options: object
) -> tuple[OptimizationResult, OptimizerBase]:
    optimizer = optimizers.create(name, **{"transform": True, **options})
    return optimizer.minimize(problem, GeneratorDraws(np.random.default_rng(seed))), optimizer


# ---------------------------------------------------------------------------- pymoo operators


@pytest.fixture(scope="module")
def pymoo() -> np.lib.npyio.NpzFile:
    if not REFERENCE.exists():
        pytest.fail(f"missing pymoo reference {REFERENCE}: run tools/pymoo_reference/generate.py")
    reference: np.lib.npyio.NpzFile = np.load(REFERENCE)
    return reference


@pytest.mark.parametrize("case", ["sbx", "sbx_eta5"])
def test_sbx_is_pymoos(pymoo: np.lib.npyio.NpzFile, case: str) -> None:
    parents = pymoo[f"{case}_in_parents"]
    c1, c2 = sbx(
        np.random.default_rng(int(pymoo[f"{case}_seed"])), parents[0::2].T, parents[1::2].T,
        pymoo[f"{case}_in_xl"], pymoo[f"{case}_in_xu"], eta=float(pymoo[f"{case}_eta"]),
        prob=float(pymoo[f"{case}_prob"]),
    )  # fmt: skip
    np.testing.assert_allclose(
        np.vstack([c1.T, c2.T]), pymoo[f"{case}_out_children"], rtol=POW_RTOL, atol=POW_ATOL
    )


@pytest.mark.parametrize("case", ["pm", "pm_eta5"])
def test_polynomial_mutation_is_pymoos(pymoo: np.lib.npyio.NpzFile, case: str) -> None:
    out = polynomial_mutation(
        np.random.default_rng(int(pymoo[f"{case}_seed"])), pymoo[f"{case}_in_members"].T,
        pymoo["sbx_in_xl"], pymoo["sbx_in_xu"], eta=float(pymoo[f"{case}_eta"]),
        prob=float(pymoo[f"{case}_prob"]),
    )  # fmt: skip
    np.testing.assert_allclose(out.T, pymoo[f"{case}_out_members"], rtol=POW_RTOL, atol=POW_ATOL)


def test_tournament_is_pymoos(pymoo: np.lib.npyio.NpzFile) -> None:
    winners = tournament(
        np.random.default_rng(int(pymoo["tournament_seed"])), pymoo["tournament_in_value"],
        pymoo["tournament_in_cv"], pymoo["tournament_out_winners"].size,
    )  # fmt: skip
    np.testing.assert_array_equal(winners, pymoo["tournament_out_winners"])


def test_pso_update_is_pymoos(pymoo: np.lib.npyio.NpzFile) -> None:
    X, V = velocity_update(
        np.random.default_rng(int(pymoo["pso_seed"])), pymoo["pso_in_X"].T, pymoo["pso_in_P_X"].T,
        pymoo["pso_in_S_X"].T, pymoo["pso_in_V"].T, pymoo["pso_in_V_max"], 0.7298, 1.49618,
        1.49618,
    )  # fmt: skip
    np.testing.assert_array_equal(X.T, pymoo["pso_out_X"])
    np.testing.assert_array_equal(V.T, pymoo["pso_out_V"])


@pytest.mark.parametrize(("name", "init"), [("ga", "uniform"), ("pso", "lhs")])
def test_the_best_after_50_generations_agrees_with_pymoo(
    pymoo: np.lib.npyio.NpzFile, name: str, init: str
) -> None:
    """Validation 4: Rastrigin (n = 5), population 20, 50 generations (pymoo counts
    the initial one), 20 seeds. pymoo's bests are stored (pymoo's GA with its default
    operators; its PSO with APEX's fixed weights, no adaptation, no perturbation of the best,
    LHS sampling as pymoo's). The two samples agree (Mann-Whitney U, p > 0.05; measured
    2026-10-06: p = 0.41 for GA, 0.44 for PSO, medians 1.08 against 0.94 and 5.6 against 4.8)
    and the medians are within a factor of 1.5."""
    from scipy.stats import mannwhitneyu

    best = []
    for seed in range(20):
        x0 = np.random.default_rng(1000 + seed).uniform(-5.12, 5.12, 5)
        problem = Problem(Rastrigin(), x0, lower=np.full(5, -5.12), upper=np.full(5, 5.12))
        optimizer = optimizers.create(name, pop_size=20, maxiter=49, stall=1000, xtol=0.0,
                                      init=init)  # fmt: skip
        result = optimizer.minimize(problem, GeneratorDraws(np.random.default_rng(seed)))
        best.append(float(np.mean(result.fun)))
    theirs = pymoo[f"stats_{name}_best"]
    assert mannwhitneyu(best, theirs).pvalue > 0.05
    assert np.median(best) <= 1.5 * np.median(theirs)


# ---------------------------------------------------------------------------- analytic optima


@pytest.mark.parametrize("name", NAMES)
def test_the_sphere_and_rosenbrock_are_solved(name: str) -> None:
    result, _ = _run(name, _problem(_sphere, [1.5, -0.8, 0.6], 2.0), seed=1, maxiter=200)
    assert float(np.mean(result.fun)) < 1e-4
    result, _ = _run(name, _problem(Rosenbrock(), [-1.2, 1.0], 2.0), seed=1, maxiter=300,
                     pop_size=40)  # fmt: skip
    np.testing.assert_allclose(result.x, [1.0, 1.0], atol=0.1)
    assert float(np.mean(result.fun)) < 1e-2


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize(("objective", "bound"), [(Rastrigin(), 5.12), (Ackley(), 32.768)])
def test_multimodal_functions_reach_the_global_basin(
    name: str, objective: object, bound: float
) -> None:
    """At least 18 of 20 seeds end in the global optimum's basin: below 0.5,
    where every other local minimum of Rastrigin and Ackley lies at 0.99 or above."""
    found = 0
    for seed in range(20):
        result, _ = _run(name, _problem(objective, [0.6 * bound, -0.5 * bound], bound), seed,
                         maxiter=200)  # fmt: skip
        found += float(np.mean(result.fun)) < 0.5
    assert found >= 18


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("handling", ["feasibility", "epf"])
def test_the_constrained_quadratic_reaches_its_kkt_point(name: str, handling: str) -> None:
    """PET's quadratic ``0.5 ||x - 1||^2`` with ``x_0 + x_1 = 3`` and ``x_0 <= 0`` (n = 2):
    on the equality, ``0.5 ((x_0 - 1)^2 + (2 - x_0)^2)`` is least at ``x_0 = 1.5``, which the
    inequality forbids, so the optimum is ``(0, 3)``; the start is infeasible.

    With the feasibility rules an equality needs a tolerance the variation can hit:
    measured 2026-10-06, at pymoo's 1e-4 both GA and PSO stall on the equality far from the
    optimum (children rarely land within 1e-4 of the line), at 1e-2 they reach it, offset by
    up to the tolerance. The penalty reaches it from ``r = 100``.
    """
    options: dict[str, object] = {"maxiter": 300, "pop_size": 40, "constraint_handling": handling}
    if handling == "epf":
        options["epf"] = {"r": 100.0, "conv_crit": 1e-8, "max_epf_iter": 8}
        tolerance, atol = 0.0, 0.01
    else:
        options["eq_tolerance"] = tolerance = 1e-2
        atol = 0.02
    result, optimizer = _run(name, _problem(Quadratic(), [1.0, 1.0], 4.0), seed=3, **options)
    x = result.x
    np.testing.assert_allclose(x, [0.0, 3.0], atol=atol)
    c_eq, c_iq = Quadratic().constraints(x[:, None])
    assert float(violation(c_eq, c_iq, tolerance)[0]) <= (0.0 if tolerance else 0.01)
    bound = optimizer.max_evaluations(1)
    assert bound is not None
    assert optimizer.member_evaluations <= bound


# ---------------------------------------------------------------------------- the loop


FACTORIES: dict[str, Callable[[], OptimizerBase]] = {
    "ga": lambda: optimizers.create("ga", maxiter=8, pop_size=8, transform=True),
    "ga_lhs": lambda: optimizers.create("ga", maxiter=8, pop_size=8, init="lhs", n_offsprings=5),
    "pso": lambda: optimizers.create("pso", maxiter=8, pop_size=8, transform=True),
    "pso_ring_clip": lambda: optimizers.create(
        "pso", maxiter=8, pop_size=8, topology="ring", bounds_handling="clip"
    ),
    "ga_epf": lambda: optimizers.create(
        "ga",
        maxiter=4,
        pop_size=8,
        constraint_handling="epf",
        epf={"r": 1.0, "conv_crit": 1e-9, "max_epf_iter": 3},
    ),
    "pso_epf": lambda: optimizers.create(
        "pso",
        maxiter=4,
        pop_size=8,
        constraint_handling="epf",
        epf={"r": 1.0, "conv_crit": 1e-9, "max_epf_iter": 3},
    ),
}


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_resume_is_bit_identical_at_every_generation(name: str) -> None:
    """Stopping after every generation and resuming gives the uninterrupted run."""
    objective = Quadratic() if name.endswith("epf") else Rastrigin()

    def problem() -> Problem:
        return _problem(objective, [1.0, -1.0, 0.5], 2.0, n_realizations=2)

    full = FACTORIES[name]().minimize(problem(), GeneratorDraws(np.random.default_rng(7)))
    state = None
    resumed = None
    for _ in range(100):
        optimizer = FACTORIES[name]()
        resumed = optimizer.minimize(
            problem(), GeneratorDraws(np.random.default_rng(7)),
            on_iteration=lambda _: False, state=state,
        )  # fmt: skip
        if resumed.message != "stopped by the caller":
            break
        state = optimizer.export_state()
    assert resumed is not None
    np.testing.assert_array_equal(resumed.x, full.x)
    np.testing.assert_array_equal(resumed.fun, full.fun)
    assert (resumed.nfev, resumed.member_evaluations, resumed.nit, resumed.message) == (
        full.nfev, full.member_evaluations, full.nit, full.message,
    )  # fmt: skip


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_the_evaluation_bound_holds_and_generations_are_one_batch(name: str) -> None:
    """``max_evaluations`` bounds the members evaluated; each generation is one batch of
    ``n_new n_r`` columns (EPF updates evaluate nothing)."""
    batches: list[int] = []
    objective = Quadratic() if name.endswith("epf") else Rastrigin()

    class Counted:
        def __call__(self, controls: FloatArray, realizations: IntArray) -> FloatArray:
            batches.append(controls.shape[1])
            return objective(controls, realizations)

        def constraints(self, controls: FloatArray) -> tuple[FloatArray, FloatArray]:
            return Quadratic().constraints(controls)

    def plain(controls: FloatArray, realizations: IntArray) -> FloatArray:
        batches.append(controls.shape[1])
        return objective(controls, realizations)

    counted: object = Counted() if name.endswith("epf") else plain
    optimizer = FACTORIES[name]()
    optimizer.minimize(_problem(counted, [1.0, -1.0, 0.5], 2.0, n_realizations=2),
                       GeneratorDraws(np.random.default_rng(5)))  # fmt: skip
    bound = optimizer.max_evaluations(2)
    assert bound is not None
    assert optimizer.member_evaluations == sum(batches) <= bound
    per_generation = optimizer.evaluations_per_iteration(2)
    assert per_generation is not None
    assert max(batches[1:]) <= per_generation  # the start is the whole population
    # Its terms, for a planner to state (a reported issue, #235).
    basis = optimizer.max_evaluations_basis(2)
    assert basis is not None
    assert basis.endswith(f"= {bound}")
    assert f"{optimizer.maxiter} generations x {per_generation // 2} new members" in basis


def test_a_stalled_best_ends_the_run_not_one_generation() -> None:
    """The base's |Δf| test would stop after one generation without improvement; the stall
    counter waits ``stall`` generations."""
    result, _ = _run("ga", _problem(_sphere, [0.0, 0.0], 1.0), seed=2, maxiter=100,
                             stall=7)  # fmt: skip
    assert "for 7 generations" in result.message
    assert result.nit >= 7


def test_reflected_particles_stay_in_the_box_and_turn_back() -> None:
    X = np.array([[1.3, -0.2, 0.5]])
    V = np.array([[0.4, -0.3, 0.1]])
    lower, upper = np.array([0.0]), np.array([1.0])
    Xr, Vr = keep_in_bounds(X, V, lower, upper, "reflect")
    np.testing.assert_allclose(Xr, [[0.7, 0.2, 0.5]])
    np.testing.assert_allclose(Vr, [[-0.4, 0.3, 0.1]])
    Xc, Vc = keep_in_bounds(X, V, lower, upper, "clip")
    np.testing.assert_allclose(Xc, [[1.0, 0.0, 0.5]])
    np.testing.assert_allclose(Vc, [[0.0, 0.0, 0.1]])


@pytest.mark.parametrize(
    ("name", "options", "match"),
    [
        ("ga", {"epf": {"r": 1.0}}, "constraint_handling"),
        ("pso", {"constraint_handling": "epf"}, "needs the epf options"),
        ("ga", {"pop_size": 1}, "pop_size"),
        ("ga", {"crossover_prob": 1.5}, "crossover_prob"),
        ("pso", {"topology": "star"}, "topology"),
        ("pso", {"v_max": 0.0}, "v_max"),
    ],
)
def test_bad_options_are_refused(name: str, options: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        optimizers.create(name, **options)


@pytest.mark.parametrize("name", NAMES)
def test_unbounded_controls_are_refused(name: str) -> None:
    problem = Problem(_sphere, np.zeros(2), lower=None, upper=None)
    with pytest.raises(ValueError, match="finite lower and upper bounds"):
        optimizers.create(name).minimize(problem, GeneratorDraws(np.random.default_rng(0)))


def test_survival_is_pymoos(pymoo: np.lib.npyio.NpzFile) -> None:
    """pymoo's ``FitnessSurvival``: feasible first, then the value; ties keep their order."""
    from apex.algorithms.optimization.population import rank_order

    kept = rank_order(pymoo["survival_in_value"], pymoo["survival_in_cv"])[:7]
    np.testing.assert_array_equal(kept, pymoo["survival_out_kept"])


def test_larger_tournaments_pick_the_best_of_each_group() -> None:
    """``tournament > 2`` (an option pymoo lacks): the best of each group under the ranking."""
    value = np.array([5.0, 1.0, 3.0, 0.5, 4.0, 2.0])
    cv = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0])  # member 3: best value, but infeasible
    winners = tournament(np.random.default_rng(4), value, cv, 50, size=6)
    np.testing.assert_array_equal(winners, np.full(50, 1))  # the whole population: member 1


def test_the_feasible_share_counts_the_constraints_under_epf() -> None:
    """Review of 2026-10-06: with EPF the ranking has no violation (the penalty carries it),
    and the logged feasible share read 1.0 for an infeasible population."""
    optimizer = optimizers.create(
        "ga", pop_size=8, maxiter=1, constraint_handling="epf", epf={"r": 1.0, "max_epf_iter": 1}
    )
    # A box where x_0 > 0.5 everywhere: the inequality x_0 <= 0 never holds.
    problem = Problem(Quadratic(), np.array([1.0, 2.0]), lower=np.array([0.5, 0.0]),
                      upper=np.array([2.0, 3.0]))  # fmt: skip
    optimizer.minimize(problem, GeneratorDraws(np.random.default_rng(0)))
    assert optimizer.log_columns()["feasible_share"] == 0.0


def test_non_finite_objective_values_fail_loudly() -> None:
    def broken(controls: FloatArray, realizations: IntArray) -> FloatArray:
        values = np.sum(controls * controls, axis=0)
        values[0] = np.nan
        return np.asarray(values, dtype=float)

    with pytest.raises(ValueError, match="non-finite"):
        _run("pso", _problem(broken, [0.5, 0.5], 1.0), seed=0, maxiter=2)


@pytest.mark.parametrize("n_realizations", [1, 3])
def test_a_population_bound_states_its_terms_with_epf(n_realizations: int) -> None:
    """The review of #235: the population basis's start and EPF terms, on one realization
    (no "x realizations") and on several; the terms multiply out to the bound."""
    optimizer = optimizers.create("ga", maxiter=4, pop_size=8, constraint_handling="epf",
                                  epf={"r": 1.0, "conv_crit": 1e-9, "max_epf_iter": 3})  # fmt: skip
    assert isinstance(optimizer, PopulationOptimizer)
    bound = optimizer.max_evaluations(n_realizations)
    basis = optimizer.max_evaluations_basis(n_realizations)
    assert bound == 8 * n_realizations + 3 * 4 * optimizer.n_new * n_realizations
    assert basis is not None
    assert basis.endswith(f"= {bound}")
    assert "8 members for the start" in basis
    assert f"3 EPF rounds x 4 generations x {optimizer.n_new} new members" in basis
    assert ("realizations" in basis) == (n_realizations > 1)

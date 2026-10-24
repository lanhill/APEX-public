"""NSGA-II (``nsga2``) and the multi-objective kernels.

Independent references (a method PET lacks):
- pymoo 0.6.2 (tests/data/reference/pymoo/multiobjective.npz, written by
  tools/pymoo_reference/generate.py in a separate environment): non-dominated sorting,
  crowding distance, RankAndCrowding survival and NSGA-II's binary tournament give pymoo's
  results exactly, given the same random numbers; the hypervolume equals pymoo's ``HV``; and
  NSGA-II's IGD over 20 seeds agrees with pymoo's statistically (measured 2026-10-06: medians
  0.0196 against 0.0197 on ZDT1, p = 0.58; 0.1157 against 0.1153 on DTLZ2, p = 0.60);
- the analytic Pareto fronts of the test problems (ten, from the papers; TNK's and KUR's as
  pymoo stores them): the IGD to the front, relative to its range, stays below 2.5 % in the
  median of three seeds (measured: 1-1.5 %; DTLZ2's three objectives: below 15 %, measured
  10 %, as pymoo's), and every member of the constrained fronts is feasible;
- hand-computed hypervolumes (rectangles and boxes).
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apex.algorithms.optimization import (
    GeneratorDraws,
    OptimizerBase,
    Problem,
    Quadratic,
    Rosenbrock,
    Zdt1,
    objectives,
    optimizers,
)
from apex.algorithms.optimization.combined import StackedResponses, stack_objectives
from apex.algorithms.optimization.multiobjective import (
    crowding_distance,
    front_movement,
    hypervolume,
    igd,
    pareto_ranks,
    rank_and_crowding,
    representative,
    tournament,
)
from apex.algorithms.optimization.protocols import FloatArray, IntArray, VectorObjective
from apex.domain.observations import DataKey

REFERENCE = Path(__file__).parents[2] / "data" / "reference" / "pymoo" / "multiobjective.npz"

#: The test problems' variables: the analytic problems' papers (and pymoo's) dimensions.
DIMENSIONS = {"sch": 1, "fon": 3, "kur": 3, "zdt1": 10, "zdt2": 10, "zdt3": 10, "dtlz2": 12,
              "bnh": 2, "tnk": 2, "srn": 2}  # fmt: skip


@pytest.fixture(scope="module")
def pymoo() -> np.lib.npyio.NpzFile:
    if not REFERENCE.exists():
        pytest.fail(f"missing pymoo reference {REFERENCE}: run tools/pymoo_reference/generate.py")
    reference: np.lib.npyio.NpzFile = np.load(REFERENCE)
    return reference


def _box(name: str) -> tuple[FloatArray, FloatArray]:
    problem: Any = objectives.create(name)
    n_x = DIMENSIONS[name]
    lower = np.array(getattr(problem, "LOWER", [problem.bounds[0]] * n_x), dtype=float)
    upper = np.array(getattr(problem, "UPPER", [problem.bounds[1]] * n_x), dtype=float)
    return lower, upper


def _solve(name: str, seed: int, **options: object) -> OptimizerBase:
    lower, upper = _box(name)
    problem = Problem(objectives.create(name), (lower + upper) / 2, lower=lower, upper=upper)
    optimizer = optimizers.create("nsga2", **{"transform": True, **options})
    optimizer.minimize(problem, GeneratorDraws(np.random.default_rng(seed)))
    return optimizer


# ---------------------------------------------------------------------------- pymoo kernels


def test_non_dominated_sorting_is_pymoos(pymoo: np.lib.npyio.NpzFile) -> None:
    np.testing.assert_array_equal(pareto_ranks(pymoo["nds_in_F"].T), pymoo["nds_out_rank"])
    np.testing.assert_array_equal(pareto_ranks(pymoo["nds3_in_F"].T), pymoo["nds3_out_rank"])


def test_crowding_distance_is_pymoos(pymoo: np.lib.npyio.NpzFile) -> None:
    np.testing.assert_allclose(
        crowding_distance(pymoo["crowding_in_F"].T), pymoo["crowding_out"], rtol=1e-13
    )


def test_rank_and_crowding_survival_is_pymoos(pymoo: np.lib.npyio.NpzFile) -> None:
    kept, _, _ = rank_and_crowding(
        np.random.default_rng(int(pymoo["survival_seed"])), pymoo["survival_in_F"].T, 12
    )
    np.testing.assert_array_equal(kept, pymoo["survival_out_kept"])


def test_the_binary_tournament_is_pymoos(pymoo: np.lib.npyio.NpzFile) -> None:
    winners = tournament(
        np.random.default_rng(int(pymoo["tournament_seed"])), pymoo["tournament_in_F"].T,
        pymoo["tournament_in_cv"], pymoo["tournament_in_crowding"],
        pymoo["tournament_out_winners"].size,
    )  # fmt: skip
    np.testing.assert_array_equal(winners, pymoo["tournament_out_winners"])


@pytest.mark.parametrize("n_obj", [2, 3, 4])
def test_the_hypervolume_is_pymoos(pymoo: np.lib.npyio.NpzFile, n_obj: int) -> None:
    value = hypervolume(pymoo[f"hv{n_obj}_in_F"].T, pymoo[f"hv{n_obj}_in_ref"])
    assert value == pytest.approx(float(pymoo[f"hv{n_obj}_out"]), rel=1e-12)


@pytest.mark.parametrize("tag", ["2", "3", "flat"])
def test_the_front_movement_is_pymoos(pymoo: np.lib.npyio.NpzFile, tag: str) -> None:
    """The stall's measure: pymoo's MultiObjectiveSpaceTermination._delta."""
    value = front_movement(pymoo[f"move{tag}_in_prev"].T, pymoo[f"move{tag}_in_cur"].T)
    assert value == pytest.approx(float(pymoo[f"move{tag}_out"]), rel=1e-12)


@pytest.mark.parametrize(("name", "n_x"), [("zdt1", 10), ("dtlz2", 12)])
def test_the_igd_after_60_generations_agrees_with_pymoos(
    pymoo: np.lib.npyio.NpzFile, name: str, n_x: int
) -> None:
    """Population 40, 60 generations (pymoo counts the first), 60 seeds; the stall stop off.

    60 seeds, not 20: with 20, one fixed set of seeds gave p 0.03 by chance (another 40
    gave 0.20, medians within 5 %), so the sample is larger on both sides.
    """
    from scipy.stats import mannwhitneyu

    problem_object: Any = objectives.create(name)
    front = problem_object.pareto_front(100 if name == "zdt1" else 400)
    values = []
    for seed in range(60):
        x0 = np.random.default_rng(2000 + seed).random(n_x)
        problem = Problem(problem_object, x0, lower=np.zeros(n_x), upper=np.ones(n_x))
        optimizer = optimizers.create("nsga2", pop_size=40, maxiter=59, stall=10**6, xtol=0.0)
        optimizer.minimize(problem, GeneratorDraws(np.random.default_rng(seed)))
        values.append(igd(np.array(optimizer.front()["objectives"]).T, front))  # type: ignore[attr-defined]
    theirs = pymoo[f"stats_{name}_igd"]
    assert mannwhitneyu(values, theirs).pvalue > 0.05
    assert np.median(values) <= 1.25 * np.median(theirs)


# ---------------------------------------------------------------------------- the problems


@pytest.mark.parametrize("name", sorted(DIMENSIONS))
def test_every_test_problem_reaches_its_known_front(pymoo: np.lib.npyio.NpzFile, name: str) -> None:
    problem: Any = objectives.create(name)
    front = (pymoo["front_tnk"].T if name == "tnk" else pymoo["front_kur"].T if name == "kur"
             else problem.pareto_front(200))  # fmt: skip
    scale = float(np.ptp(front, axis=1).max())
    distances, violations = [], []
    for seed in range(3):
        optimizer = _solve(name, seed, pop_size=40, maxiter=150)
        result = optimizer.front()  # type: ignore[attr-defined]
        distances.append(igd(np.array(result["objectives"]).T, front) / scale)
        violations.append(max(result["violation"]))
    # 40 members cover a line at about 1 % of its range, a three-objective octant at about
    # 10 % (pymoo's NSGA-II: 0.115 on DTLZ2).
    assert np.median(distances) < (0.15 if front.shape[0] == 3 else 0.025)
    assert max(violations) == 0.0  # constrained fronts: every member feasible


def test_the_analytic_fronts_are_non_dominated() -> None:
    for name in ("sch", "fon", "zdt1", "zdt2", "zdt3", "dtlz2", "bnh", "srn"):
        front = objectives.create(name).pareto_front(60)  # type: ignore[attr-defined]
        assert np.all(pareto_ranks(front) == 0), name


# ---------------------------------------------------------------------------- hypervolume


def test_hypervolumes_by_hand() -> None:
    # Two objectives: staircase (1,3), (2,2), (3,1) against (4,4): 1x1 + 1x2 + 1x3 = 6.
    assert hypervolume(np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]), [4, 4]) == 6.0
    # Three: one box (0,0,0)-(1,2,3) = 6; two overlapping half boxes = 0.25 + 0.25 - 0.125.
    assert hypervolume(np.zeros((3, 1)), [1, 2, 3]) == 6.0
    two = np.array([[0.0, 0.5], [0.5, 0.0], [0.5, 0.5]])
    assert hypervolume(two, [1, 1, 1]) == pytest.approx(0.375)
    # Points not better than the reference add nothing; dominated points add nothing.
    assert hypervolume(np.array([[5.0], [0.0]]), [4, 4]) == 0.0
    assert hypervolume(np.array([[1.0, 2.0], [1.0, 2.0]]), [3, 3]) == 4.0
    with pytest.raises(ValueError, match="at most 4"):
        hypervolume(np.zeros((5, 2)), np.ones(5))


def test_the_representative_is_the_compromise() -> None:
    F = np.array([[0.0, 0.5, 1.0], [1.0, 0.45, 0.0]])
    assert representative(F) == 1


# ---------------------------------------------------------------------------- the loop


def _zdt1(n_realizations: int = 1) -> Problem:
    return Problem(Zdt1(), np.full(5, 0.5), lower=np.zeros(5), upper=np.ones(5),
                   n_realizations=n_realizations)  # fmt: skip


def test_resume_is_bit_identical_at_every_generation() -> None:
    def make() -> OptimizerBase:
        return optimizers.create("nsga2", pop_size=10, maxiter=6, transform=True)

    full_optimizer = make()
    full = full_optimizer.minimize(_zdt1(2), GeneratorDraws(np.random.default_rng(4)))
    state, resumed, optimizer = None, None, make()
    for _ in range(50):
        optimizer = make()
        resumed = optimizer.minimize(_zdt1(2), GeneratorDraws(np.random.default_rng(4)),
                                     on_iteration=lambda _: False, state=state)  # fmt: skip
        if resumed.message != "stopped by the caller":
            break
        state = optimizer.export_state()
    assert resumed is not None
    np.testing.assert_array_equal(resumed.x, full.x)
    np.testing.assert_array_equal(resumed.fun, full.fun)
    assert optimizer.front() == full_optimizer.front()  # type: ignore[attr-defined]


def test_the_bound_holds_and_a_generation_is_one_batch() -> None:
    batches: list[int] = []
    zdt = Zdt1()

    class Counted:
        objective_names = zdt.objective_names

        def __call__(self, controls: FloatArray, realizations: np.ndarray) -> FloatArray:
            batches.append(controls.shape[1])
            return zdt(controls, realizations)

    problem = Problem(Counted(), np.full(5, 0.5), lower=np.zeros(5), upper=np.ones(5),
                      n_realizations=3)  # fmt: skip
    optimizer = optimizers.create("nsga2", pop_size=8, maxiter=5)
    optimizer.minimize(problem, GeneratorDraws(np.random.default_rng(1)))
    bound = optimizer.max_evaluations(3)
    assert bound is not None
    assert optimizer.member_evaluations == sum(batches) <= bound
    assert max(batches) <= 8 * 3


def test_one_objective_and_several_are_refused_where_they_do_not_belong() -> None:
    with pytest.raises(ValueError, match="optimizes one objective"):
        optimizers.create("ga").minimize(_zdt1(), GeneratorDraws(np.random.default_rng(0)))
    scalar = Problem(Rosenbrock(), np.zeros(2), lower=-np.ones(2), upper=np.ones(2))
    with pytest.raises(ValueError, match="needs several objectives"):
        optimizers.create("nsga2").minimize(scalar, GeneratorDraws(np.random.default_rng(0)))
    with pytest.raises(ValueError, match="ref_point"):
        optimizers.create("nsga2", ref_point=[1.0]).minimize(
            _zdt1(), GeneratorDraws(np.random.default_rng(0))
        )


def test_the_hypervolume_stall_ends_the_run() -> None:
    optimizer = _solve("sch", 0, pop_size=20, maxiter=500, stall=8)
    assert "moved less than" in optimizer.conv_msg
    assert "for 8 generations" in optimizer.conv_msg
    assert optimizer.iteration < 500


def test_epf_penalises_every_objective_and_reaches_a_feasible_front() -> None:
    optimizer = _solve("bnh", 2, pop_size=30, maxiter=60, constraint_handling="epf",
                       epf={"r": 100.0, "max_epf_iter": 3})  # fmt: skip
    result = optimizer.front()  # type: ignore[attr-defined]
    assert max(result["violation"]) <= 1e-3
    assert len(result["x"]) > 5


class _Corner:
    """Two objectives (SCH's on x_0), feasible only where x_0 + x_1 >= 1.99 on [0, 1]^2."""

    objective_names = ("f1", "f2")

    def __call__(self, controls: FloatArray, realizations: IntArray) -> FloatArray:
        x = controls[0]
        return np.vstack([x**2, (x - 2.0) ** 2])

    def constraints(self, controls: FloatArray) -> tuple[FloatArray, FloatArray]:
        return np.empty((0, controls.shape[1])), (controls[0] + controls[1] - 1.99)[None, :]


def test_no_stall_while_the_violation_falls() -> None:
    """Review finding (numerics, 2026-10-06): with no feasible member the hypervolume is 0,
    and the stall used to count every generation, ending the run as converged infeasible.
    Progress before feasibility is the least violation, as phase 1's."""
    problem = Problem(_Corner(), np.array([0.1, 0.1]), lower=np.zeros(2), upper=np.ones(2))
    optimizer = optimizers.create("nsga2", pop_size=10, maxiter=60, stall=5, transform=True)
    optimizer.minimize(problem, GeneratorDraws(np.random.default_rng(3)))
    result = optimizer.front()  # type: ignore[attr-defined]
    assert max(result["violation"]) == 0.0, optimizer.conv_msg


@pytest.mark.parametrize("r", [1.0, 10.0])
def test_an_epf_front_holds_only_feasible_members(r: float) -> None:
    """Review finding: on SRN with a small penalty factor the front used to include members
    that violate the constraints (penalised, not feasible)."""
    optimizer = _solve("srn", 0, pop_size=30, maxiter=60, constraint_handling="epf",
                       epf={"r": r, "max_epf_iter": 3})  # fmt: skip
    result = optimizer.front()  # type: ignore[attr-defined]
    assert max(result["violation"]) == 0.0
    assert len(result["x"]) > 1


# ---------------------------------------------------------------------------- stacking


def test_stacked_objectives_keep_their_names_and_constraints() -> None:
    stacked = stack_objectives(["q", "r"], [Quadratic(), Rosenbrock()])
    assert isinstance(stacked, VectorObjective)
    controls = np.array([[0.0, 1.0], [2.0, 1.0]])
    zeros = np.zeros(2, dtype=np.int64)
    expected = np.vstack([Quadratic()(controls, zeros), Rosenbrock()(controls, zeros)])
    np.testing.assert_allclose(np.asarray(stacked(controls, zeros)), expected)
    c_eq, c_iq = stacked.constraints(controls)  # type: ignore[attr-defined]
    assert c_eq.shape == (1, 2)
    assert c_iq.shape == (1, 2)


def test_stacked_responses_simulate_their_keys_once() -> None:
    from apex.algorithms.optimization import response_objectives

    npv = response_objectives.create("npv", start="2000-01-01", times=["2001-01-01"], wop=1.0,
                                     wwi=1.0)  # fmt: skip
    water = response_objectives.create("field_total", vector="FWIT", time="2001-01-01")
    stacked = StackedResponses(["npv", "water"], [npv, water])
    keys = list(stacked.response_keys())
    assert keys.count(DataKey("2001-01-01", "FWIT")) == 1  # shared by both
    responses = np.array([[100.0], [10.0], [366.0]]) if len(keys) == 3 else None
    assert responses is not None, keys
    values = stacked.evaluate(responses)
    assert values.shape == (2, 1)
    assert values[1, 0] == 10.0  # the water injected, scale 1

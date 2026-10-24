"""NSGA-III and MOEA/D against pymoo 0.6.2.

Kernels exactly (tests/data/reference/pymoo/manyobjective.npz, written by
tools/pymoo_reference/generate.py in the pymoo environment): Das-Dennis directions, NSGA-III's
association, niching, hyperplane normalisation, survival and tournament, MOEA/D's
neighbourhoods, selection and decompositions. The whole methods statistically: the IGD over
60 seeds against pymoo's NSGA3 and MOEAD (Mann-Whitney U, medians within 1.25x). pymoo's
layout is rows; these tests transpose at their boundary.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apex.algorithms.optimization import (
    GeneratorDraws,
    OptimizerBase,
    Problem,
    objectives,
    optimizers,
)
from apex.algorithms.optimization.moead import (
    neighbourhood_parents,
    neighbourhoods,
    pbi,
    tchebicheff,
)
from apex.algorithms.optimization.multiobjective import igd
from apex.algorithms.optimization.nsga3 import (
    HyperplaneNormalization,
    associate_to_niches,
    niching,
    survive_by_references,
    tournament_by_violation,
)
from apex.algorithms.optimization.reference_directions import (
    das_dennis,
    directions_for,
    number_of_directions,
)

REFERENCE = Path(__file__).parents[2] / "data" / "reference" / "pymoo" / "manyobjective.npz"


@pytest.fixture(scope="module")
def pymoo() -> np.lib.npyio.NpzFile:
    if not REFERENCE.exists():
        pytest.fail(f"missing pymoo reference {REFERENCE}: run tools/pymoo_reference/generate.py")
    reference: np.lib.npyio.NpzFile = np.load(REFERENCE)
    return reference


# ---------------------------------------------------------------------------- directions


@pytest.mark.parametrize(("p", "m"), [(19, 2), (6, 3), (3, 5), (12, 3), (0, 3)])
def test_das_dennis_is_pymoos(pymoo: np.lib.npyio.NpzFile, p: int, m: int) -> None:
    directions = das_dennis(p, m)
    np.testing.assert_array_equal(directions, pymoo[f"das_{p}_{m}"])
    if p:
        assert directions.shape[0] == number_of_directions(p, m)


def test_the_default_directions_are_smaller_than_the_papers() -> None:
    """About 20-35 directions where the papers take 91 for three objectives; from 5
    objectives two layers (the user's choice, 2026-10-06)."""
    sizes = {m: len(directions_for(m)) for m in (2, 3, 4, 5, 6, 8)}
    assert sizes == {2: 20, 3: 28, 4: 35, 5: 20, 6: 27, 8: 44}
    assert number_of_directions(12, 3) == 91


@pytest.mark.parametrize("m", [5, 6, 8])
def test_two_layers_are_pymoos_and_reach_the_interior(pymoo: np.lib.npyio.NpzFile, m: int) -> None:
    """Numerics review: one layer with fewer partitions than objectives has no interior
    direction. The inner layer's are interior (every coordinate positive); the set is
    pymoo's multi-layer directions (the order within a layer is its own)."""
    layers = directions_for(m)
    theirs = pymoo[f"layers_{m}"]
    np.testing.assert_allclose(np.sort(layers, axis=0), np.sort(theirs, axis=0), rtol=1e-15)
    assert {tuple(np.round(r, 12)) for r in layers} == {tuple(np.round(r, 12)) for r in theirs}
    np.testing.assert_allclose(layers.sum(axis=1), 1.0)
    assert np.all(layers[-m:] > 0.0)


# ---------------------------------------------------------------------------- NSGA-III


def test_the_association_is_pymoos(pymoo: np.lib.npyio.NpzFile) -> None:
    niche, dist, matrix = associate_to_niches(
        pymoo["assoc_in_F"], das_dennis(6, 3), pymoo["assoc_in_ideal"], pymoo["assoc_in_nadir"]
    )
    np.testing.assert_array_equal(niche, pymoo["assoc_out_niche"])
    np.testing.assert_allclose(dist, pymoo["assoc_out_dist"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(matrix, pymoo["assoc_out_matrix"], rtol=1e-12, atol=1e-14)


def test_niching_is_pymoos(pymoo: np.lib.npyio.NpzFile) -> None:
    picks = niching(np.random.default_rng(int(pymoo["niching_seed"])), 9,
                    pymoo["niching_in_counts"].copy(), pymoo["niching_in_niche"],
                    pymoo["niching_in_dist"])  # fmt: skip
    np.testing.assert_array_equal(picks, pymoo["niching_out"])


def test_the_hyperplane_normalisation_is_pymoos(pymoo: np.lib.npyio.NpzFile) -> None:
    norm = HyperplaneNormalization(3)
    for case in ("1", "2"):  # its state carries over from the first update to the second
        norm.update(pymoo[f"norm{case}_in_F"], pymoo[f"norm{case}_in_nds"])
        for name, value in (("ideal", norm.ideal), ("worst", norm.worst),
                            ("extremes", norm.extremes), ("nadir", norm.nadir)):  # fmt: skip
            assert value is not None
            np.testing.assert_allclose(value, pymoo[f"norm{case}_out_{name}"], rtol=1e-12)


def test_the_survival_is_pymoos(pymoo: np.lib.npyio.NpzFile) -> None:
    norm = HyperplaneNormalization(3)
    for case in ("a", "b"):  # one survival object twice: the normalisation persists
        F = pymoo[f"survival_{case}_in_F"].T
        rng = np.random.default_rng(int(pymoo[f"survival_{case}_seed"]))
        kept, _ = survive_by_references(rng, F, np.zeros(F.shape[1]), 28, das_dennis(6, 3), norm)
        np.testing.assert_array_equal(kept, pymoo[f"survival_{case}_out_kept"])
    F = pymoo["survival_c_in_F"].T
    kept, _ = survive_by_references(np.random.default_rng(int(pymoo["survival_c_seed"])), F,
                                    pymoo["survival_c_in_cv"], 28, das_dennis(6, 3),
                                    HyperplaneNormalization(3))  # fmt: skip
    np.testing.assert_array_equal(kept, pymoo["survival_c_out_kept"])


def test_the_tournament_is_pymoos(pymoo: np.lib.npyio.NpzFile) -> None:
    rng = np.random.default_rng(int(pymoo["tournament3_seed"]))
    winners = tournament_by_violation(rng, pymoo["tournament3_in_cv"], 40)  # 20 matings x 2
    np.testing.assert_array_equal(winners, pymoo["tournament3_out"])


# ---------------------------------------------------------------------------- MOEA/D


def test_the_neighbourhoods_and_selection_are_pymoos(pymoo: np.lib.npyio.NpzFile) -> None:
    near = neighbourhoods(das_dennis(6, 3), 6)
    np.testing.assert_array_equal(near, pymoo["neighbors_out"])
    rng = np.random.default_rng(int(pymoo["selection_seed"]))
    parents = neighbourhood_parents(rng, near[pymoo["selection_in_order"]], near.shape[0], 0.9)
    np.testing.assert_array_equal(parents, pymoo["selection_out"])


def test_the_decompositions_are_pymoos(pymoo: np.lib.npyio.NpzFile) -> None:
    F, w, z = pymoo["decomp_in_F"], pymoo["decomp_in_weights"], pymoo["decomp_in_ideal"]
    child = np.repeat(pymoo["decomp_in_child"][None, :], w.shape[0], axis=0)
    np.testing.assert_allclose(tchebicheff(F, w, z), pymoo["tcheb_out"], rtol=1e-13)
    np.testing.assert_allclose(tchebicheff(child, w, z), pymoo["tcheb_child_out"], rtol=1e-13)
    np.testing.assert_allclose(pbi(F, w, z, 5.0), pymoo["pbi_out"], rtol=1e-12)
    np.testing.assert_allclose(pbi(child, w, z, 5.0), pymoo["pbi_child_out"], rtol=1e-12)


# ---------------------------------------------------------------------------- statistics


def _run(name: str, problem: str, n_x: int, seed: int, maxiter: int, **kwargs: int) -> float:
    objective: Any = objectives.create(problem, **kwargs)
    x0 = np.random.default_rng(3000 + seed).random(n_x)
    optimizer = optimizers.create(name, maxiter=maxiter, stall=10**6, xtol=0.0)
    optimizer.minimize(Problem(objective, x0, lower=np.zeros(n_x), upper=np.ones(n_x)),
                       GeneratorDraws(np.random.default_rng(seed)))  # fmt: skip
    front = objective.pareto_front(100 if problem == "zdt1" else 400)
    return igd(np.array(optimizer.front()["objectives"]).T, front)  # type: ignore[attr-defined]


@pytest.mark.parametrize(("label", "n_obj", "n_x"), [("3", 3, 12), ("5", 5, 14)])
def test_nsga3_agrees_with_pymoo_on_dtlz2(
    pymoo: np.lib.npyio.NpzFile, label: str, n_obj: int, n_x: int
) -> None:
    """APEX's default directions given to pymoo too (3 objectives: 28; 5: two layers, 20),
    the same populations, 60 generations, 60 seeds."""
    from scipy.stats import mannwhitneyu

    values = [_run("nsga3", "dtlz2", n_x, s, 59, n_obj=n_obj) for s in range(60)]
    theirs = pymoo[f"stats_nsga3_dtlz2_{label}_igd"]
    assert mannwhitneyu(values, theirs).pvalue > 0.05
    assert np.median(values) <= 1.25 * np.median(theirs)


@pytest.mark.parametrize(("problem", "n_x"), [("zdt1", 10), ("dtlz2", 12)])
def test_moead_is_no_worse_than_pymoos(pymoo: np.lib.npyio.NpzFile, problem: str, n_x: int) -> None:
    """Against pymoo's loop-wise MOEAD (the method as published), 100 generations, 60 seeds.

    One-sided: APEX's batched generation should not be worse (Mann-Whitney ``less`` than
    pymoo's is not required; ``greater`` must not be significant), medians within 1.25x.
    pymoo's ParallelMOEAD is recorded too; its children cross subproblems (see moead.py).
    """
    from scipy.stats import mannwhitneyu

    values = [_run("moead", problem, n_x, s, 99) for s in range(60)]
    theirs = pymoo[f"stats_moead_{problem}_igd"]
    assert mannwhitneyu(values, theirs, alternative="greater").pvalue > 0.05
    assert np.median(values) <= 1.25 * np.median(theirs)


# ---------------------------------------------------------------------------- the workflow


@pytest.mark.parametrize("name", ["nsga3", "moead"])
def test_resume_is_bit_identical_at_every_generation_and_the_bound_holds(name: str) -> None:
    """Stopped after every generation and resumed from its exported state (NSGA-III's
    normalisation, MOEA/D's ideal point): the same front as one run, within the bound."""

    def problem() -> Problem:
        objective: Any = objectives.create("dtlz2")
        return Problem(objective, np.full(12, 0.5), lower=np.zeros(12), upper=np.ones(12))

    def make() -> OptimizerBase:
        return optimizers.create(name, maxiter=5, transform=True)

    whole = make()
    full = whole.minimize(problem(), GeneratorDraws(np.random.default_rng(4)))
    bound = whole.max_evaluations(1)
    assert bound is not None
    assert whole.member_evaluations <= bound
    state, resumed, optimizer = None, None, make()
    for _ in range(50):
        optimizer = make()
        resumed = optimizer.minimize(problem(), GeneratorDraws(np.random.default_rng(4)),
                                     on_iteration=lambda _: False, state=state)  # fmt: skip
        if resumed.message != "stopped by the caller":
            break
        state = optimizer.export_state()
    assert resumed is not None
    np.testing.assert_array_equal(resumed.fun, full.fun)
    assert optimizer.front() == whole.front()  # type: ignore[attr-defined]


def test_moead_refuses_constraints_without_epf() -> None:
    objective: Any = objectives.create("bnh")
    problem = Problem(
        objective, np.array([2.5, 1.5]), lower=np.zeros(2), upper=np.array([5.0, 3.0])
    )
    with pytest.raises(ValueError, match="external penalty only"):
        optimizers.create("moead", maxiter=2).minimize(
            problem, GeneratorDraws(np.random.default_rng(0))
        )
    optimizer = optimizers.create("moead", maxiter=3, constraint_handling="epf",
                                  epf={"r": 100.0, "max_epf_iter": 2})  # fmt: skip
    optimizer.minimize(problem, GeneratorDraws(np.random.default_rng(0)))
    assert optimizer.front()["x"]  # type: ignore[attr-defined]


def test_dtlz_scales_and_its_fronts_are_on_the_simplex_and_sphere() -> None:
    for m in (2, 3, 5):
        d1: Any = objectives.create("dtlz1", n_obj=m)
        d2: Any = objectives.create("dtlz2", n_obj=m)
        np.testing.assert_allclose(d1.pareto_front(100).sum(axis=0), 0.5)
        np.testing.assert_allclose(np.linalg.norm(d2.pareto_front(100), axis=0), 1.0)
        # On the front: the distance variables at 0.5 (g = 0).
        x = np.full((m + 4, 6), 0.5)
        x[: m - 1] = np.random.default_rng(m).random((m - 1, 6))
        np.testing.assert_allclose(d1(x, np.zeros(6, dtype=np.int64)).sum(axis=0), 0.5)
        np.testing.assert_allclose(np.linalg.norm(d2(x, np.zeros(6, dtype=np.int64)), axis=0), 1.0)


def test_moead_replaces_the_neighbours_a_child_improves_and_keeps_the_ideal() -> None:
    """By hand: a child at the ideal point beats every neighbour (Tchebicheff 0), a child far
    off replaces none; after a generation the ideal is at or below every member."""
    objective: Any = objectives.create("zdt1")
    problem = Problem(objective, np.full(3, 0.5), lower=np.zeros(3), upper=np.ones(3))
    optimizer: Any = optimizers.create("moead", n_partitions=3, n_neighbors=2, maxiter=1)
    optimizer.minimize(problem, GeneratorDraws(np.random.default_rng(2)))
    mean, _ = optimizer.means(optimizer.F, optimizer.c_eq, optimizer.c_iq)
    assert np.all(optimizer.ideal[:, None] <= mean + 1e-15)
    near = optimizer.neighbors[0]
    others = np.setdiff1d(np.arange(optimizer.X.shape[1]), near)
    before = optimizer.X.copy()
    empty = np.empty(0)
    far = np.full((2, 1), 1e6)
    optimizer._replace(0, np.full(3, 0.9), far, empty, empty)
    np.testing.assert_array_equal(optimizer.X, before)  # worse everywhere: nothing replaced
    best = optimizer.ideal[:, None].copy()
    optimizer._replace(0, np.full(3, 0.1), best, empty, empty)
    np.testing.assert_array_equal(optimizer.X[:, near], np.full((3, near.size), 0.1))
    np.testing.assert_array_equal(optimizer.X[:, others], before[:, others])
    assert optimizer.X.flags.c_contiguous
    assert optimizer.F.flags.c_contiguous


def test_the_statistic_patterns_agree() -> None:
    """The domain's pattern and the kernel's are written twice (the layers): kept equal."""
    from apex.algorithms.optimization.statistics import STATISTIC_PATTERN
    from apex.domain.optimization import ObjectiveEntry

    field = ObjectiveEntry.model_fields["statistic"]
    patterns = [m.pattern for m in field.metadata if hasattr(m, "pattern")]
    assert patterns == [STATISTIC_PATTERN]


def test_nsga3_starts_its_normalisation_afresh_at_each_penalty_factor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Numerics review (2026-10-06): the normalisation came from the old factor's penalised
    values. A run through two EPF stages makes a new one at the second."""
    from apex.algorithms.optimization import nsga3

    made: list[int] = []

    class Counting(nsga3.HyperplaneNormalization):
        def __init__(self, n_obj: int) -> None:
            made.append(n_obj)
            super().__init__(n_obj)

    monkeypatch.setattr(nsga3, "HyperplaneNormalization", Counting)
    # Feasible only where x0 + x1 >= 1.999 on [0, 1]^2: the first stage ends infeasible.
    problem = Problem(_Corner(), np.array([0.1, 0.1]), lower=np.zeros(2), upper=np.ones(2))
    optimizer: Any = optimizers.create("nsga3", maxiter=2, transform=True,
                                       constraint_handling="epf",
                                       epf={"r": 1e-9, "max_epf_iter": 2,
                                            "conv_crit": 1e-12})  # fmt: skip
    optimizer.minimize(problem, GeneratorDraws(np.random.default_rng(1)))
    assert optimizer.epf_iteration == 2
    assert len(made) == 2  # set_objectives, then the second stage


def test_one_moead_generation_follows_its_steps() -> None:
    """A transcription of one generation from the tested kernels, with a copy of the
    generator: each subproblem's own children, the ideal updated from the batch before any
    replacement, then each child replacing the neighbours it improves, in order."""
    from apex.algorithms.optimization.ga import polynomial_mutation, sbx

    objective: Any = objectives.create("dtlz2")
    problem = Problem(objective, np.full(5, 0.5), lower=np.zeros(5), upper=np.ones(5))
    probe: Any = optimizers.create("moead", maxiter=1, n_partitions=3)
    stop = {"n": 0}

    def once(_: object) -> bool:  # stop after the initial population
        stop["n"] += 1
        return False

    probe.minimize(problem, GeneratorDraws(np.random.default_rng(6)), on_iteration=once)
    assert type(probe).__name__ == "MOEAD"
    X0, F0 = probe.X.copy(), probe.F.copy()
    ideal0 = probe.ideal.copy()
    rng = np.random.default_rng()
    rng.bit_generator.state = probe.rng.bit_generator.state
    # The transcription.
    n = X0.shape[1]
    order = rng.permutation(n)[: probe.n_new]
    parents = neighbourhood_parents(rng, probe.neighbors[order], n, probe.prob_neighbor_mating)
    c1, c2 = sbx(rng, X0[:, parents[:, 0]], X0[:, parents[:, 1]], probe.xl, probe.xu,
                 eta=probe.eta_c, prob=probe.crossover_prob,
                 prob_var=probe.crossover_prob_var)  # fmt: skip
    children = polynomial_mutation(rng, np.column_stack([c1, c2]), probe.xl, probe.xu,
                                   eta=probe.eta_m, prob=probe.mutation_prob,
                                   prob_var=probe.mutation_prob_var)  # fmt: skip
    m = order.size
    picks = [int(rng.choice(np.array([k, m + k]))) for k in range(m)]
    X = children[:, picks]
    F = objective(X, np.zeros(m, dtype=np.int64))  # (n_obj, m); no transform: X is physical
    ideal = np.minimum(ideal0, F.min(axis=1))
    expected_X, expected_F = X0.copy(), F0[:, 0, :].copy()
    weights = probe.weights
    for j, k in enumerate(order):
        near = probe.neighbors[k]
        child = np.repeat(F[:, j][None, :], near.size, axis=0)
        better = near[pbi(child, weights[near], ideal, 5.0)
                      < pbi(expected_F[:, near].T, weights[near], ideal, 5.0)]  # fmt: skip
        expected_X[:, better] = X[:, j][:, None]
        expected_F[:, better] = F[:, j][:, None]
    # The optimizer's own generation.
    probe.generation()
    np.testing.assert_allclose(probe.ideal, ideal, rtol=1e-14)
    np.testing.assert_array_equal(probe.X, expected_X)
    np.testing.assert_allclose(probe.F[:, 0, :], expected_F, rtol=1e-14)


class _Corner:
    """Two objectives on x0, feasible only in the corner x0 + x1 >= 1.999 of [0, 1]^2."""

    objective_names = ("f1", "f2")

    def __call__(self, controls: np.ndarray, realizations: np.ndarray) -> np.ndarray:
        return np.vstack([controls[0] ** 2, (controls[0] - 2.0) ** 2])

    def constraints(self, controls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.empty((0, controls.shape[1])), (controls[0] + controls[1] - 1.999)[None, :]


class _Shifted:
    """Two scalar objectives that depend on the realization r: x0^2 + r, (x1 - 1)^2 (1 + r)."""

    def __init__(self, which: int) -> None:
        self.which = which

    def __call__(self, controls: np.ndarray, realizations: np.ndarray) -> np.ndarray:
        r = realizations.astype(float)
        if self.which == 0:
            return np.asarray(controls[0] ** 2 + r, dtype=float)
        return np.asarray((controls[1] - 1.0) ** 2 * (1.0 + r), dtype=float)


@pytest.mark.parametrize("name", ["nsga3", "moead"])
def test_resume_with_statistics_over_realizations(name: str) -> None:
    """Three realizations, the mean and the 90th percentile: resumed after every generation,
    the same front as one run."""
    from apex.algorithms.optimization.combined import stack_objectives

    def problem() -> Problem:
        stacked = stack_objectives(["mean", "p90"], [_Shifted(0), _Shifted(1)], ["mean", "p90"])
        return Problem(stacked, np.full(2, 0.5), lower=np.full(2, -2.0), upper=np.full(2, 2.0),
                       n_realizations=3)  # fmt: skip

    def make() -> OptimizerBase:
        return optimizers.create(name, maxiter=4, transform=True)

    whole = make()
    whole.minimize(problem(), GeneratorDraws(np.random.default_rng(8)))
    state, optimizer = None, make()
    for _ in range(50):
        optimizer = make()
        result = optimizer.minimize(problem(), GeneratorDraws(np.random.default_rng(8)),
                                    on_iteration=lambda _: False, state=state)  # fmt: skip
        if result.message != "stopped by the caller":
            break
        state = optimizer.export_state()
    assert optimizer.front() == whole.front()  # type: ignore[attr-defined]
    assert whole.front()["objective_statistics"] == ["mean", "p90"]  # type: ignore[attr-defined]

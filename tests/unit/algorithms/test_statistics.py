"""Statistics over realizations as objectives, against NumPy on given arrays."""

import numpy as np
import pytest

from apex.algorithms.optimization import GeneratorDraws, Problem, optimizers
from apex.algorithms.optimization.combined import stack_objectives
from apex.algorithms.optimization.protocols import FloatArray, IntArray
from apex.algorithms.optimization.statistics import check_statistic, reduce_realizations


def test_each_statistic_is_numpys() -> None:
    values = np.random.default_rng(3).normal(size=(6, 7, 5))  # objectives, realizations, members
    stats = ["mean", "std", "worst", "p10", "p50", "p90"]
    got = reduce_realizations(values, stats)
    expected = [
        values[0].mean(axis=0), values[1].std(axis=0, ddof=1), values[2].max(axis=0),
        np.percentile(values[3], 10, axis=0), np.percentile(values[4], 50, axis=0),
        np.percentile(values[5], 90, axis=0),
    ]  # fmt: skip
    np.testing.assert_array_equal(got, np.vstack(expected))
    assert got.flags.c_contiguous


def test_statistics_are_checked() -> None:
    for name in ("mean", "std", "worst", "p1", "p10", "p99"):
        assert check_statistic(name) == name
    for bad in ("p0", "p100", "median", "P10", "p05", ""):
        with pytest.raises(ValueError, match="unknown statistic"):
            check_statistic(bad)
    with pytest.raises(ValueError, match="two realizations"):
        reduce_realizations(np.zeros((1, 1, 3)), ["std"])
    with pytest.raises(ValueError, match="statistics for"):
        reduce_realizations(np.zeros((2, 3, 3)), ["mean"])


class _Shift:
    """A scalar objective with a realization-dependent shift: x^2 + r."""

    def __call__(self, controls: FloatArray, realizations: IntArray) -> FloatArray:
        return np.asarray(controls[0] ** 2 + realizations, dtype=float)


def test_nsga2_ranks_each_objectives_statistic() -> None:
    stacked = stack_objectives(["mean", "worst"], [_Shift(), _Shift()], ["mean", "worst"])
    problem = Problem(stacked, np.array([0.5]), lower=np.array([-1.0]), upper=np.array([1.0]),
                      n_realizations=3)  # fmt: skip
    optimizer = optimizers.create("nsga2", pop_size=6, maxiter=2)
    optimizer.minimize(problem, GeneratorDraws(np.random.default_rng(0)))
    nsga2 = optimizer  # the front's values are the statistics ranked
    front = nsga2.front()  # type: ignore[attr-defined]
    assert front["objective_statistics"] == ["mean", "worst"]
    for x, (mean, worst) in zip(front["x"], front["objectives"], strict=True):
        assert mean == pytest.approx(x[0] ** 2 + 1.0)  # realizations 0, 1, 2
        assert worst == pytest.approx(x[0] ** 2 + 2.0)


def test_std_needs_two_realizations() -> None:
    stacked = stack_objectives(["mean", "std"], [_Shift(), _Shift()], ["mean", "std"])
    problem = Problem(stacked, np.array([0.5]), lower=np.array([-1.0]), upper=np.array([1.0]))
    with pytest.raises(ValueError, match="std needs at least two realizations"):
        optimizers.create("nsga2", pop_size=6, maxiter=1).minimize(
            problem, GeneratorDraws(np.random.default_rng(0))
        )

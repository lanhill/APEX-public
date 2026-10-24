"""Forecast scores, each against an independent computation."""

from itertools import pairwise

import numpy as np
import pytest

from apex.algorithms.metrics.forecast import coverage, crps, crps_per_time, mean_error, spread
from apex.core.plugins import forecast_scores


def _crps_by_integration(members: np.ndarray, truth: float) -> float:
    """CRPS as the integral of (F(x) - 1{x >= t})^2 over x, for the empirical CDF F."""
    points = np.sort(np.append(members, truth))
    total = 0.0
    for left, right in pairwise(points):
        middle = 0.5 * (left + right)
        cdf = np.mean(members <= middle)
        total += (cdf - float(middle >= truth)) ** 2 * (right - left)
    return total


def test_crps_of_two_members_has_its_closed_form() -> None:
    # Members a < b, truth t between them: CRPS = (|a-t| + |b-t|)/2 - (b-a)/4.
    a, b, t = 1.0, 5.0, 2.0

    value = crps(np.array([[a, b]]), np.array([t]))

    assert value == pytest.approx((abs(a - t) + abs(b - t)) / 2 - (b - a) / 4)


def test_crps_equals_the_integral_of_the_squared_cdf_difference() -> None:
    rng = np.random.default_rng(3)
    ensemble, truth = rng.standard_normal((6, 25)), rng.standard_normal(6)

    per_time = crps_per_time(ensemble, truth)

    expected = [_crps_by_integration(ensemble[i], truth[i]) for i in range(6)]
    np.testing.assert_allclose(per_time, expected, rtol=1e-12)
    # The sorted O(n log n) form equals the direct pairwise mean.
    pairwise = np.abs(ensemble[:, :, None] - ensemble[:, None, :]).mean(axis=(1, 2))
    direct = np.abs(ensemble - truth[:, None]).mean(axis=1) - 0.5 * pairwise
    np.testing.assert_allclose(per_time, direct, rtol=1e-12)


def test_crps_is_zero_for_a_collapsed_ensemble_on_the_truth() -> None:
    assert crps(np.full((3, 4), 2.5), np.full(3, 2.5)) == pytest.approx(0.0)


def test_coverage_counts_the_times_inside_the_band() -> None:
    ensemble = np.tile(np.arange(11.0), (4, 1))  # p10 = 1, p90 = 9 at every time
    truth = np.array([0.5, 1.0, 9.0, 9.5])

    assert coverage(ensemble, truth) == 0.5


def test_mean_error_and_spread() -> None:
    ensemble = np.array([[1.0, 3.0], [2.0, 6.0]])  # means 2 and 4; std (ddof=1) sqrt(2), sqrt(8)
    truth = np.array([2.0, 1.0])

    assert mean_error(ensemble, truth) == pytest.approx(np.sqrt((0 + 9) / 2))
    assert spread(ensemble, truth) == pytest.approx((np.sqrt(2) + np.sqrt(8)) / 2)


def test_scores_validate_shapes_and_are_registered() -> None:
    with pytest.raises(ValueError, match="n_times, n_members"):
        crps(np.ones(3), np.ones(3))
    with pytest.raises(ValueError, match="2 member"):
        spread(np.ones((3, 1)), np.ones(3))
    assert {"coverage_p10_p90", "crps", "mean_error", "spread"} <= set(forecast_scores.names())
    score = forecast_scores.create("crps").score(np.array([[1.0, 5.0]]), np.array([2.0]))
    assert score == pytest.approx(1.0)  # the closed form above


@pytest.mark.parametrize("n", [10, 20, 100])
def test_nominal_coverage_matches_a_calibrated_ensemble(n: int) -> None:
    from apex.algorithms.metrics.forecast import nominal_coverage

    # Truth and members drawn from one distribution: the expected coverage by simulation.
    rng = np.random.default_rng(n)
    draws = rng.standard_normal((20_000, n + 1))
    covered = coverage(draws[:, :n], draws[:, n])

    # The formula is exact in rank; within a gap between members the truth's position
    # depends on the distribution: for Gaussians the approximation is 0.015 low at 10
    # members and within the Monte Carlo error (about 0.003) from 20 on.
    assert covered == pytest.approx(nominal_coverage(n), abs=0.02 if n == 10 else 0.01)
